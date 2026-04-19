"""fl-cityscapes-bisenetv2: A Flower / PyTorch app."""

import numpy as np
import copy

import torch

from lib.ohem_ce_loss import OhemCELoss
from tools.eval_metrics import compute_metrics_from_cm


def set_optimizer(model, lr_start, weight_decay):
    """Optimizer with parameter groups for different learning rates and weight decay."""
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
            {"params": lr_mul_wd_params, "lr": lr_start * 10},
            {
                "params": lr_mul_nowd_params,
                "weight_decay": wd_val,
                "lr": lr_start * 10,
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
        lr=lr_start,
        momentum=0.9,
        weight_decay=weight_decay,
    )
    return optim


def train_bisenetv2(
    net,
    trainloader,
    epochs,
    lr,
    wd,
    device,
    strategy,
    prox_mu,
    neg_entropy_weight: float = 0.0,
    num_aux_heads=4,
):
    """Train the model on the training set."""
    global_weights = None
    if strategy == "FedProx":
        global_params = copy.deepcopy(net)
        global_weights = list(global_params.parameters())

    net.to(device)
    criterion_pre = OhemCELoss(0.7, device=device)
    criterion_aux = [OhemCELoss(0.7, device=device) for _ in range(num_aux_heads)]

    optimizer = set_optimizer(net, lr_start=lr, weight_decay=wd)

    net.train()
    running_loss = 0.0
    train_loss_pre = 0.0

    for _ in range(epochs):
        for im, lb in trainloader:
            images = im.to(device)
            labels = lb.to(device)

            optimizer.zero_grad()

            logits, *logits_aux = net(images)

            loss_pre = criterion_pre(logits, labels)
            loss_aux = [
                crit(lgt, labels) for crit, lgt in zip(criterion_aux, logits_aux)
            ]
            loss = loss_pre + sum(loss_aux)

            # FedProx: add proximal term to the loss
            if strategy == "FedProx" and prox_mu > 0.0:
                proximal_term = 0.0
                local_weights = list(net.parameters())
                for w, w_t in zip(local_weights, global_weights):
                    proximal_term += (w - w_t).norm(2)
                loss += (prox_mu / 2) * proximal_term

            # Negative-entropy regularization (encourages confident predictions)
            if strategy == "FedEMA" and neg_entropy_weight > 0.0:
                probs = torch.softmax(logits, dim=1)
                log_probs = torch.log_softmax(logits, dim=1)
                # Average negative entropy over batch and pixels
                neg_entropy = (probs * log_probs).sum(dim=1).mean()
                loss = loss + neg_entropy_weight * neg_entropy

            # Standard backward pass and optimization step
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_loss_pre += loss_pre.item()

    avg_trainloss = running_loss / (len(trainloader) * epochs)
    train_loss_pre = train_loss_pre / (len(trainloader) * epochs)

    return avg_trainloss, train_loss_pre


def test_bisenetv2(net, testloader, device, num_classes, lb_ignore=255):
    """Validate the model on the test set."""

    net.to(device)
    org_aux = net.aux_mode
    net.aux_mode = "eval"
    net.eval()

    criterion_pre = OhemCELoss(0.7, device=device)

    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_loss = 0.0

    with torch.no_grad():
        for im, lb in testloader:
            images = im.to(device)
            labels = lb.to(device)

            logits = net(images)[0]
            loss = criterion_pre(logits, labels)
            total_loss += loss.item()

            preds = torch.softmax(logits, dim=1).argmax(dim=1)

            preds_np = preds.cpu().numpy().reshape(-1)
            labels_np = labels.cpu().numpy().reshape(-1)

            mask = labels_np != lb_ignore
            preds_np = preds_np[mask]
            labels_np = labels_np[mask]

            conf_matrix += np.bincount(
                labels_np * num_classes + preds_np,
                minlength=num_classes**2,
            ).reshape(num_classes, num_classes)

    final_loss = total_loss / len(testloader)
    metrics = compute_metrics_from_cm(conf_matrix)

    net.aux_mode = org_aux

    print("Finished evaluation on test set.")

    return {**metrics, "val_loss": final_loss}
