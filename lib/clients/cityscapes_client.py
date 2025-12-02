import logging

import flwr as fl
import torch

import torch.amp as amp

from lib.models import BiSeNetV2
from lib.ohem_ce_loss import OhemCELoss
from lib.lr_scheduler import WarmupPolyLrScheduler
from lib.meters import TimeMeter, AvgMeter
from lib.logger import log_msg


class CityScapesClient(fl.client.NumPyClient):
    """Flower client for CityScapes dataset.

    Inherits from Flower's NumPyClient to handle training on CityScapes data.
    """

    def __init__(self, cfg, train_loader):
        """Initialize the CityScapes client.

        Args:
            cfg: Configuration object.
            train_loader (DataLoader): DataLoader for the training data.
        """
        self.cfg = cfg
        self.train_loader = train_loader
        self.model, self.criteria_pre, self.criteria_aux = self._init_local_model()
        self.optimizer = self._init_local_optimizer(self.model)
        self.time_meter, self.loss_meter, self.loss_pre_meter, self.loss_aux_meters = (
            self._init_meters()
        )

    def fit(self, parameters, config):
        """Fit the model to the provided parameters.

        Args:
            parameters : Model parameters to set.
            config : Configuration for the training process.

        Returns:
            tuple: Updated model parameters, number of training examples, and an empty dictionary.
        """
        logger = logging.getLogger()

        logger.info("Setting Model Parameters")
        self._set_parameters(parameters)

        scaler = amp.GradScaler()

        lr_scheduler = WarmupPolyLrScheduler(
            self.optimizer,
            power=0.9,
            max_iter=self.cfg.max_iter,
            warmup_iter=self.cfg.warmup_iters,
            warmup_ratio=0.1,
            warmup="exp",
            last_epoch=-1,
        )

        for _ in range(1):  # TODO: Use config for number of epochs
            for it, (im, lb) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                with amp.autocast(device_type="cuda", enabled=self.cfg.use_fp16):
                    logits, *logits_aux = self.model(im)
                    loss_pre = self.criteria_pre(logits, lb)
                    loss_aux = [
                        crit(lgt, lb)
                        for crit, lgt in zip(self.criteria_aux, logits_aux)
                    ]
                    loss = loss_pre + sum(loss_aux)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                lr_scheduler.step()

                self.time_meter.update()
                self.loss_meter.update(loss.item())
                self.loss_pre_meter.update(loss_pre.item())
                _ = [
                    mter.update(lss.item())
                    for mter, lss in zip(self.loss_aux_meters, loss_aux)
                ]

                if (it + 1) % 100 == 0:
                    lr = lr_scheduler.get_lr()
                    lr = sum(lr) / len(lr)
                    msg = log_msg(
                        it,
                        self.cfg.max_iter,  # TODO: Remove Max_iter
                        lr,
                        self.time_meter,
                        self.loss_meter,
                        self.loss_pre_meter,
                        self.loss_aux_meters,
                    )
                    logger.info(msg)

        logger.info("\nFinished Training Client Model")

        return self._get_parameters(), len(self.train_loader.dataset), {}

    def _init_local_model(self, lb_ignore=255):
        """Initialize the local model.

        Args:
            lb_ignore (int, optional): Label to ignore during training. Defaults to 255.

        Returns:
            tuple: Model, primary loss criterion, and auxiliary loss criteria.
        """
        model = BiSeNetV2(self.cfg.n_cats)

        criteria_pre = OhemCELoss(0.7, lb_ignore)
        criteria_aux = [
            OhemCELoss(0.7, lb_ignore) for _ in range(self.cfg.num_aux_heads)
        ]

        return model, criteria_pre, criteria_aux

    def _init_local_optimizer(self, model):
        """Set the optimizer for the local model.

        Args:
            model: The model to optimize.

        Returns:
            torch.optim.Optimizer: The optimizer for the local model.
        """
        if hasattr(model, "get_params"):
            wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = (
                model.get_params()
            )
            wd_val = 0
            params_list = [
                {
                    "params": wd_params,
                },
                {"params": nowd_params, "weight_decay": wd_val},
                {"params": lr_mul_wd_params, "lr": self.cfg.lr_start * 10},
                {
                    "params": lr_mul_nowd_params,
                    "weight_decay": wd_val,
                    "lr": self.cfg.lr_start * 10,
                },
            ]
        else:
            wd_params, non_wd_params = [], []
            for name, param in model.named_parameters():
                if param.dim() == 1:
                    non_wd_params.append(param)
                elif param.dim() == 2 or param.dim() == 4:
                    wd_params.append(param)
            params_list = [
                {
                    "params": wd_params,
                },
                {"params": non_wd_params, "weight_decay": 0},
            ]
        optim = torch.optim.SGD(
            params_list,
            lr=self.cfg.lr_start,
            momentum=0.9,
            weight_decay=self.cfg.weight_decay,
        )
        return optim

    def _init_meters(self):
        """Initialize the meters for tracking training progress.

        Returns:
            tuple: Time meter, loss meter, primary loss meter, and auxiliary loss meters.
        """
        time_meter = TimeMeter(self.cfg.max_iter)
        loss_meter = AvgMeter("loss")
        loss_pre_meter = AvgMeter("loss_prem")
        loss_aux_meters = [
            AvgMeter("loss_aux{}".format(i)) for i in range(self.cfg.num_aux_heads)
        ]
        return time_meter, loss_meter, loss_pre_meter, loss_aux_meters

    def _set_parameters(self, parameters):
        """Set the parameters of the model.

        Args:
            parameters (list): The parameters to set.
        """
        state_dict = self.model.state_dict()
        for (k, v), param in zip(state_dict.items(), parameters):
            state_dict[k] = torch.tensor(param)
        self.model.load_state_dict(state_dict)

    def _get_parameters(self, config={}):
        """Gets the parameters of the model.

        Args:
            config (dict, optional): Configuration options. Defaults to {}.

        Returns:
            list: The parameters of the model.
        """
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
