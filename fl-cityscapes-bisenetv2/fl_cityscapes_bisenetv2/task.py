"""fl-cityscapes-bisenetv2: A Flower / PyTorch app."""

import json
import torch
from torch.utils.data import DataLoader
import torch.amp as amp

from lib.clients.cityscapes_client_dataset import CityScapesClientDataset
import lib.data.transform_cv2 as T
from lib.ohem_ce_loss import OhemCELoss


def load_data(
    data_root: str,
    partitions: str,
    partition_id: int,
    batch_size: int,
    scales: list,
    cropsize: list,
):
    """Load partition CityScapes data."""
    with open(partitions, "r") as f:
        data_partitions = json.load(f)

    partition = data_partitions[str(partition_id)]

    ds = CityScapesClientDataset(
        data_root,
        partition["data"],
        T.TransformationTrain(scales, cropsize),
    )

    # Construct dataloaders
    trainloader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    return trainloader


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


# def test(net, testloader, device): # TODO: Add evaluation function either per client or per server
#     """Validate the model on the test set."""
#     net.to(device)
#     criterion = torch.nn.CrossEntropyLoss()
#     correct, loss = 0, 0.0
#     with torch.no_grad():
#         for batch in testloader:
#             images = batch["img"].to(device)
#             labels = batch["label"].to(device)
#             outputs = net(images)
#             loss += criterion(outputs, labels).item()
#             correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
#     accuracy = correct / len(testloader.dataset)
#     loss = loss / len(testloader)
#     return loss, accuracy
