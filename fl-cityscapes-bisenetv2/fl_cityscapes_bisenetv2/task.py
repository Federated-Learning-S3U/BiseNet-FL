"""fl-cityscapes-bisenetv2: A Flower / PyTorch app."""

import json
import numpy as np

import torch
import torch.amp as amp

from flwr.app import ArrayRecord, MetricRecord, Context


from lib.models import BiSeNetV2
from lib.ohem_ce_loss import OhemCELoss
from tools.eval_metrics import compute_eval_metrics

from fl_cityscapes_bisenetv2.data_preparation.datasets import load_server_eval_data


def train(net, trainloader, epochs, lr, device, num_aux_heads):
    """Train the model on the training set."""
    net.to(device)
    criterion_pre = OhemCELoss(0.7, device=device)
    criterion_aux = [OhemCELoss(0.7, device=device) for _ in range(num_aux_heads)]

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # TODO

    scaler = amp.GradScaler()

    # lr_scheduler = WarmupPolyLrScheduler( # TODO: Check if LR scheduler is needed, either client or server
    #     self.optimizer,
    #     power=0.9,
    #     max_iter=self.cfg.max_iter,
    #     warmup_iter=self.cfg.warmup_iters,
    #     warmup_ratio=0.1,
    #     warmup="exp",
    #     last_epoch=-1,
    # )

    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for it, (im, lb) in enumerate(trainloader):
            images = im.to(device)
            labels = lb.to(device)

            optimizer.zero_grad()

            with amp.autocast(device_type="cuda", enabled=False):
                logits, *logits_aux = net(images)
                loss_pre = criterion_pre(logits, labels)
                loss_aux = [
                    crit(lgt, labels) for crit, lgt in zip(criterion_aux, logits_aux)
                ]
                loss = loss_pre + sum(loss_aux)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # lr_scheduler.step()

            # self.time_meter.update() # TODO: Add Local Model Loggers
            # self.loss_meter.update(loss.item())
            # self.loss_pre_meter.update(loss_pre.item())
            # _ = [
            #     mter.update(lss.item())
            #     for mter, lss in zip(self.loss_aux_meters, loss_aux)
            # ]

            # if (it + 1) % 100 == 0:
            #     lr = lr_scheduler.get_lr()
            #     lr = sum(lr) / len(lr)
            #     msg = log_msg(
            #         it,
            #         self.cfg.max_iter,
            #         lr,
            #         self.time_meter,
            #         self.loss_meter,
            #         self.loss_pre_meter,
            #         self.loss_aux_meters,
            #     )
            #     logger.info(msg)
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)

    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""

    org_aux = net.aux_mode
    net.aux_mode = "eval"
    net.eval()

    lb_ignore = 255  # TODO

    all_y_true, all_y_pred = [], []
    with torch.no_grad():
        for batch_idx, (im, lb) in enumerate(testloader):
            images = im.to(device)
            labels = lb

            outputs = net(images)[0]

            all_y_true.append(labels.flatten())

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_y_pred.append(preds.flatten())

    final_y_true = np.concatenate(all_y_true)
    final_y_pred = np.concatenate(all_y_pred)

    metrics = compute_eval_metrics(
        y_true=final_y_true,
        y_pred=final_y_pred,
        num_classes=net.num_classes,
        ignore_index=lb_ignore,
    )

    net.aux_mode = org_aux

    return metrics  # TODO: Add Loss


def make_central_evaluate(context: Context):
    def central_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """Evaluate the global model on the server side (optional)."""

        if server_round == 0:
            return MetricRecord({})

        # Read run config
        num_rounds: int = context.run_config["num-server-rounds"]
        fraction_train: float = context.run_config["fraction-train"]

        server_device: str = context.run_config["server-device"]

        eval_batch_size: int = context.run_config["eval-batch-size"]
        eval_interval: int = context.run_config["eval-interval"]

        local_epochs: int = context.run_config["local-epochs"]
        batch_size: int = context.run_config["batch_size"]

        im_root: str = context.run_config["im_root"]
        client_data_partition: str = context.run_config["client_data_partition"]
        server_data_partition: str = context.run_config["server_data_partition"]

        num_classes: int = context.run_config["num_classes"]

        lr: float = context.run_config["lr"]
        weight_decay: float = context.run_config["weight_decay"]

        num_aux_heads: int = context.run_config["num_aux_heads"]

        scales: list = json.loads(context.run_config["scales"])
        cropsize: list = json.loads(context.run_config["cropsize"])

        save_path: str = context.run_config["respth"]

        # Load Global Model
        model = BiSeNetV2(num_classes)
        model.load_state_dict(arrays.to_torch_state_dict())

        device = torch.device(server_device)

        print("=" * 50)
        print(f"Server evaluation using device: {device}")
        print("Server round {} / {}".format(server_round, num_rounds))
        print("=" * 50)

        model.to(device)

        # Load the entire Cityscapes val dataset
        eval_loader = load_server_eval_data(
            data_root=im_root,
            data_file=server_data_partition,
            batch_size=eval_batch_size,
        )

        # Evaluate the model on the test set
        metrics = test(model, eval_loader, device)

        # Return the evaluation metrics
        return MetricRecord(metrics)

    return central_evaluate
