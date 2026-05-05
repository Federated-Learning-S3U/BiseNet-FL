import numpy as np
import copy

import torch

from lib.ohem_ce_loss import OhemCELoss
from tools.eval_metrics import compute_metrics_from_cm

import torch


import torch


def set_optimizer_smp(model, lr_start, weight_decay, is_pretrained):
    """
    Optimizer tailored for SMP models.
    If pretrained: Uses 10x LR for decoder/head.
    If from scratch: Uses uniform LR across the whole model.
    Always disables weight decay for 1D params (biases/BatchNorms).
    """

    def get_params_by_dim(module):
        """Helper to separate weights (requires weight decay) from biases/BNs."""
        wd_params, non_wd_params = [], []
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() == 1:
                # Biases and BatchNorm weights
                non_wd_params.append(param)
            else:
                # Conv and Linear weights
                wd_params.append(param)
        return wd_params, non_wd_params

    # Scenario 1: Training from Scratch (Uniform Learning Rate)
    if not is_pretrained:
        wd_params, non_wd_params = get_params_by_dim(model)

        params_list = [
            {"params": wd_params, "lr": lr_start, "weight_decay": weight_decay},
            {"params": non_wd_params, "lr": lr_start, "weight_decay": 0.0},
        ]

    # Scenario 2: Fine-Tuning a Pretrained Model (Differential Learning Rate)
    else:
        encoder_wd, encoder_nowd = get_params_by_dim(model.encoder)
        decoder_wd, decoder_nowd = get_params_by_dim(model.decoder)
        head_wd, head_nowd = get_params_by_dim(model.segmentation_head)

        dec_head_wd = decoder_wd + head_wd
        dec_head_nowd = decoder_nowd + head_nowd

        params_list = [
            # Encoder: Base Learning Rate
            {"params": encoder_wd, "lr": lr_start, "weight_decay": weight_decay},
            {"params": encoder_nowd, "lr": lr_start, "weight_decay": 0.0},
            # Decoder & Head: 10x Learning Rate
            {"params": dec_head_wd, "lr": lr_start * 10, "weight_decay": weight_decay},
            {"params": dec_head_nowd, "lr": lr_start * 10, "weight_decay": 0.0},
        ]

    # Initialize Optimizer
    optim = torch.optim.SGD(
        params_list,
        lr=lr_start,
        momentum=0.9,
        weight_decay=weight_decay,
    )

    return optim


def train_deeplabv3p(
    net,
    is_pretrained,
    trainloader,
    epochs,
    lr,
    wd,
    device,
    strategy,
    prox_mu,
    neg_entropy_weight: float = 0.0,
    num_aux_heads: int = 0,
):
    """Train the model on the training set."""
    global_weights = None
    if strategy == "FedProx":
        global_params = copy.deepcopy(net)
        global_weights = list(global_params.parameters())

    net.to(device)

    criterion = OhemCELoss(0.7, device=device)

    optimizer = set_optimizer_smp(
        net, lr_start=lr, weight_decay=wd, is_pretrained=is_pretrained
    )

    net.train()

    running_loss = 0.0

    for _ in range(epochs):
        for im, lb in trainloader:
            images = im.to(device)
            labels = lb.to(device)

            optimizer.zero_grad()

            logits = net(images)

            loss = criterion(logits, labels)

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

    avg_trainloss = running_loss / (len(trainloader) * epochs)

    return avg_trainloss, 0.0


def test_deeplabv3p(net, testloader, device, num_classes, lb_ignore=255):
    """Validate the model on the test set."""

    net.to(device)

    net.eval()

    criterion = OhemCELoss(0.7, device=device)

    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_loss = 0.0

    with torch.no_grad():
        for im, lb in testloader:
            images = im.to(device)
            labels = lb.to(device)

            logits = net(images)
            loss = criterion(logits, labels)
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

    print("Finished evaluation on test set.")

    return {**metrics, "val_loss": final_loss}
