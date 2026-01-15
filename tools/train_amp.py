#!/usr/bin/python
# -*- encoding: utf-8 -*-

import sys

sys.path.insert(0, ".")
import os
import os.path as osp
import random
import logging
import time
import json
import argparse
import numpy as np
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.cuda.amp as amp

from lib.models import model_factory
from configs import set_cfg_from_file
from lib.data import get_data_loader
from evaluate import eval_model, Metrics, SizePreprocessor
from lib.ohem_ce_loss import OhemCELoss
from lib.lr_scheduler import WarmupPolyLrScheduler
from lib.meters import TimeMeter, AvgMeter
from lib.logger import setup_logger, log_msg


## fix all random seeds
#  torch.manual_seed(123)
#  torch.cuda.manual_seed(123)
#  np.random.seed(123)
#  random.seed(123)
#  torch.backends.cudnn.deterministic = True
#  torch.backends.cudnn.benchmark = True
#  torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--config",
        dest="config",
        type=str,
        default="configs/bisenetv2.py",
    )
    parse.add_argument(
        "--finetune-from",
        type=str,
        default=None,
    )
    parse.add_argument(
        "--resume-from",
        type=str,
        default=None,
    )
    parse.add_argument(
        "--eval-interval",
        type=int,
        default=1000,
        help="Evaluate every N iterations during training",
    )
    return parse.parse_args()


args = parse_args()
cfg = set_cfg_from_file(args.config)


@torch.no_grad()
def eval_single_scale(net, dl, n_classes, lb_ignore=255):
    """
    Single-scale evaluation without cropping or flipping.
    Returns a dict with mIoU and F1 scores.
    Used during training for quick evaluation.
    """
    net.eval()
    metric_observer = Metrics(n_classes, lb_ignore)

    for imgs, label in dl:
        imgs = imgs.cuda()
        label = label.squeeze(1).cuda()

        with torch.no_grad():
            logits = net(imgs)[0]

        preds = torch.argmax(logits, dim=1)
        metric_observer.update(preds, label)

    metrics = metric_observer.compute_metrics()
    if dist.is_initialized():
        dist.barrier()  # Synchronize all processes
    net.train()
    return metrics


@torch.no_grad()
def compute_val_loss(net, dl, criteria_pre):
    """
    Compute validation loss on the validation dataset (primary loss only, no aux heads).
    """
    net.eval()
    total_loss = 0.0
    num_batches = 0

    for imgs, label in dl:
        imgs = imgs.cuda()
        label = label.squeeze(1).cuda()

        with torch.no_grad():
            logits = net(imgs)[0]
            loss = criteria_pre(logits, label)

        total_loss += loss.item()
        num_batches += 1

    net.train()

    if num_batches == 0:
        return 0.0

    return total_loss / num_batches


def save_checkpoint(
    net, optim, lr_schdr, it, best_miou, latest_metrics, is_best=False, respth=None
):
    """
    Save model checkpoint with optimizer state and training metadata.
    """
    if respth is None or dist.get_rank() != 0:
        return

    model_state = net.module.state_dict()

    checkpoint = {
        "iteration": it,
        "model_state_dict": model_state,
        "optimizer_state_dict": optim.state_dict(),
        "lr_scheduler_state_dict": lr_schdr.state_dict(),
        "best_miou": best_miou,
        "latest_metrics": latest_metrics,
    }

    # Save latest model
    latest_path = osp.join(respth, "model_latest.pth")
    torch.save(checkpoint, latest_path)

    # Save best model
    if is_best:
        best_path = osp.join(respth, "model_best.pth")
        torch.save(checkpoint, best_path)


def load_checkpoint(net, optim, lr_schdr, checkpoint_path):
    """
    Load model checkpoint and restore training state.
    Returns: (start_iteration, best_miou, latest_metrics)
    """
    logger = logging.getLogger()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    net.module.load_state_dict(checkpoint["model_state_dict"])
    optim.load_state_dict(checkpoint["optimizer_state_dict"])
    lr_schdr.load_state_dict(checkpoint["lr_scheduler_state_dict"])

    start_it = checkpoint["iteration"] + 1
    best_miou = checkpoint["best_miou"]
    latest_metrics = checkpoint["latest_metrics"]

    logger.info(f"Resumed from checkpoint at iteration {start_it}")
    logger.info(f"Best mIoU so far: {best_miou:.6f}")

    return start_it, best_miou, latest_metrics


def set_model(lb_ignore=255):
    logger = logging.getLogger()
    net = model_factory[cfg.model_type](cfg.n_cats)
    if not args.finetune_from is None:
        logger.info(f"load pretrained weights from {args.finetune_from}")
        msg = net.load_state_dict(
            torch.load(args.finetune_from, map_location="cpu"), strict=False
        )
        logger.info("\tmissing keys: " + json.dumps(msg.missing_keys))
        logger.info("\tunexpected keys: " + json.dumps(msg.unexpected_keys))
    if cfg.use_sync_bn:
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net.cuda()
    net.train()
    criteria_pre = OhemCELoss(0.7, lb_ignore)
    criteria_aux = [OhemCELoss(0.7, lb_ignore) for _ in range(cfg.num_aux_heads)]
    return net, criteria_pre, criteria_aux


def set_optimizer(model):
    if hasattr(model, "get_params"):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = (
            model.get_params()
        )
        #  wd_val = cfg.weight_decay
        wd_val = 0
        params_list = [
            {
                "params": wd_params,
            },
            {"params": nowd_params, "weight_decay": wd_val},
            {"params": lr_mul_wd_params, "lr": cfg.lr_start * 10},
            {
                "params": lr_mul_nowd_params,
                "weight_decay": wd_val,
                "lr": cfg.lr_start * 10,
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
        lr=cfg.lr_start,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
    )
    return optim


def set_model_dist(net):
    local_rank = int(os.environ["LOCAL_RANK"])
    net = nn.parallel.DistributedDataParallel(
        net,
        device_ids=[
            local_rank,
        ],
        #  find_unused_parameters=True,
        output_device=local_rank,
    )
    return net


def set_meters():
    time_meter = TimeMeter(cfg.max_iter)
    loss_meter = AvgMeter("loss")
    loss_pre_meter = AvgMeter("loss_prem")
    loss_aux_meters = [
        AvgMeter("loss_aux{}".format(i)) for i in range(cfg.num_aux_heads)
    ]
    return time_meter, loss_meter, loss_pre_meter, loss_aux_meters


def train():
    logger = logging.getLogger()

    ## dataset
    dl_train = get_data_loader(cfg, mode="train")
    dl_val = get_data_loader(cfg, mode="val")

    ## model
    net, criteria_pre, criteria_aux = set_model(dl_train.dataset.lb_ignore)

    ## optimizer
    optim = set_optimizer(net)

    ## mixed precision training
    scaler = amp.GradScaler()

    ## ddp training
    net = set_model_dist(net)

    ## meters
    time_meter, loss_meter, loss_pre_meter, loss_aux_meters = set_meters()

    ## lr scheduler
    lr_schdr = WarmupPolyLrScheduler(
        optim,
        power=0.9,
        max_iter=cfg.max_iter,
        warmup_iter=cfg.warmup_iters,
        warmup_ratio=0.1,
        warmup="exp",
        last_epoch=-1,
    )

    ## Resume from checkpoint if specified
    start_it = 0
    best_miou = 0.0
    latest_metrics = {
        "train_miou": 0.0,
        "train_f1": 0.0,
        "val_miou": 0.0,
        "val_f1": 0.0,
    }

    if args.resume_from is not None:
        start_it, best_miou, latest_metrics = load_checkpoint(
            net, optim, lr_schdr, args.resume_from
        )

    ## training loop
    for it, (im, lb) in enumerate(dl_train, start=start_it):
        im = im.cuda()
        lb = lb.cuda()
        lb = torch.squeeze(lb, 1)

        optim.zero_grad()
        with amp.autocast(enabled=cfg.use_fp16):
            logits, *logits_aux = net(im)
            loss_pre = criteria_pre(logits, lb)
            loss_aux = [crit(lgt, lb) for crit, lgt in zip(criteria_aux, logits_aux)]
            loss = loss_pre + sum(loss_aux)

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        torch.cuda.synchronize()

        time_meter.update()
        loss_meter.update(loss.item())
        loss_pre_meter.update(loss_pre.item())
        _ = [mter.update(lss.item()) for mter, lss in zip(loss_aux_meters, loss_aux)]

        ## print training log message
        if (it + 1) % 100 == 0:
            lr = lr_schdr.get_lr()
            lr = sum(lr) / len(lr)
            msg = log_msg(
                it,
                cfg.max_iter,
                lr,
                time_meter,
                loss_meter,
                loss_pre_meter,
                loss_aux_meters,
            )
            logger.info(msg)

        ## evaluation during training
        if (it + 1) % args.eval_interval == 0:
            logger.info(f"\n--- Evaluation at iteration {it + 1} ---")
            torch.cuda.empty_cache()

            # Single-scale evaluation on training set
            train_metrics = eval_single_scale(
                net.module, dl_train, cfg.n_cats, dl_train.dataset.lb_ignore
            )
            train_miou = train_metrics["miou"]
            train_f1 = train_metrics["macro_f1"]

            # Single-scale evaluation on validation set
            val_metrics = eval_single_scale(
                net.module, dl_val, cfg.n_cats, dl_val.dataset.lb_ignore
            )
            val_miou = val_metrics["miou"]
            val_f1 = val_metrics["macro_f1"]

            # Compute validation loss (primary loss only, no aux heads)
            val_loss = compute_val_loss(net.module, dl_val, criteria_pre)

            logger.info(f"Train - mIoU: {train_miou:.6f}, F1: {train_f1:.6f}")
            logger.info(
                f"Val   - mIoU: {val_miou:.6f}, F1: {val_f1:.6f}, Loss: {val_loss:.6f}"
            )

            # Update metrics
            latest_metrics = {
                "train_miou": train_miou,
                "train_f1": train_f1,
                "val_miou": val_miou,
                "val_f1": val_f1,
                "val_loss": val_loss,
            }

            # Save checkpoint
            is_best = val_miou > best_miou
            if is_best:
                best_miou = val_miou
                logger.info(f"New best mIoU: {best_miou:.6f}")

            save_checkpoint(
                net,
                optim,
                lr_schdr,
                it,
                best_miou,
                latest_metrics,
                is_best=is_best,
                respth=cfg.respth,
            )

            net.train()

        lr_schdr.step()

        if it >= cfg.max_iter:
            break

    ## dump the final model and evaluate the result
    save_pth = osp.join(cfg.respth, "model_final.pth")
    logger.info("\nsave models to {}".format(save_pth))
    state = net.module.state_dict()
    if dist.get_rank() == 0:
        torch.save(state, save_pth)

    logger.info("\nevaluating the final model with multi-scale evaluation")
    torch.cuda.empty_cache()
    iou_heads, iou_content, f1_heads, f1_content = eval_model(cfg, net.module)
    logger.info("\neval results of f1 score metric:")
    logger.info("\n" + tabulate(f1_content, headers=f1_heads, tablefmt="orgtbl"))
    logger.info("\neval results of miou metric:")
    logger.info("\n" + tabulate(iou_content, headers=iou_heads, tablefmt="orgtbl"))

    return


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    if not osp.exists(cfg.respth):
        os.makedirs(cfg.respth)
    setup_logger(f"{cfg.model_type}-{cfg.dataset.lower()}-train", cfg.respth)
    train()


if __name__ == "__main__":
    main()
