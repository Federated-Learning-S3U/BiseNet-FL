"""fl-cityscapes-bisenetv2: A Flower / PyTorch app."""

import json
import numpy as np
import copy

import torch
import torch.amp as amp

from flwr.app import ArrayRecord, MetricRecord, Context


from lib.models import BiSeNetV2
from lib.ohem_ce_loss import OhemCELoss
from tools.eval_metrics import compute_metrics_from_cm

from fl_cityscapes_bisenetv2.data_preparation.datasets import load_server_eval_data
from fl_cityscapes_bisenetv2.utils.model_utils import set_optimizer
from fl_cityscapes_bisenetv2.data_preparation.utils import aggregate_client_metrics


def train(net, trainloader, epochs, lr, wd, device, num_aux_heads, strategy, prox_mu):
    """Train the model on the training set."""
    global_weights = None
    if strategy == "FedProx":
        global_params = copy.deepcopy(net)
        global_weights = list(global_params.parameters())

    net.to(device)
    criterion_pre = OhemCELoss(0.7, device=device)
    criterion_aux = [OhemCELoss(0.7, device=device) for _ in range(num_aux_heads)]

    optimizer = set_optimizer(net, lr_start=lr, weight_decay=wd)

    scaler = amp.GradScaler()

    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for im, lb in trainloader:
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

                # FedProx: add proximal term to the loss
                if strategy == "FedProx":
                    proximal_term = 0.0
                    local_weights = list(net.parameters())
                    for w, w_t in zip(local_weights, global_weights):
                        proximal_term += (w - w_t).norm(2)
                    loss += (prox_mu / 2) * proximal_term

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

    avg_trainloss = running_loss / (len(trainloader) * epochs)

    return avg_trainloss


def test(net, testloader, device, num_classes, lb_ignore=255):
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

    print("Finished evaluation on test set.")

    final_loss = total_loss / len(testloader)

    metrics = compute_metrics_from_cm(conf_matrix)

    net.aux_mode = org_aux

    return {**metrics, "val_loss": final_loss}


def make_central_evaluate(context: Context):
    """Create a central evaluation function that accepts context as an argument."""

    # This best_miou is only set once and retains its value across multiple calls to central_evaluate
    best_miou = context.run_config["best-miou"]
    best_miou = {"value": best_miou}

    save_latest = context.run_config["save-latest"]
    save_best = context.run_config["save-best"]
    best_metric_file = context.run_config["best-metric"]
    latest_metric_file = context.run_config["latest-metric"]
    client_data_partition = context.run_config["client-data-partition"]
    strategy_name = context.run_config["strategy-name"]

    data_mean, data_std = aggregate_client_metrics(client_data_partition)
    data_metrics = {"mean": data_mean, "std": data_std}

    def central_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """Evaluate the global model on the server side (optional).
        
        For SiloBN: Server evaluation is skipped as BN statistics are siloed on clients.
        Instead, evaluation is performed on clients and results are aggregated by the strategy.
        """

        # Read run config
        server_device: str = context.run_config["server-device"]

        eval_batch_size: int = context.run_config["eval-batch-size"]
        eval_interval: int = context.run_config["eval-interval"]

        im_root: str = context.run_config["im-root"]
        server_data_partition: str = context.run_config["server-data-partition"]

        num_classes: int = context.run_config["num-classes"]
        lb_ignore: int = context.run_config["lb-ignore"]

        rounds_trained = context.run_config["rounds-trained"]

        if server_round == 0 or server_round % eval_interval != 0:
            return MetricRecord({})

        # For SiloBN: Skip server-side evaluation
        # Evaluation happens on clients and is aggregated by the strategy
        if strategy_name == "FedSiloBN":
            print(f"[SiloBN] Round {rounds_trained + server_round}: Server-side evaluation skipped.")
            print(f"[SiloBN] Evaluation is performed on clients with their local BN statistics.")
            # Return empty metrics - the strategy will provide aggregated client metrics
            return MetricRecord({})

        device = torch.device(server_device)

        # Load Global Model
        model = BiSeNetV2(num_classes).cpu()
        sd = arrays.to_torch_state_dict()
        model.load_state_dict(sd, strict=True)

        # Load the entire Cityscapes val dataset
        eval_loader = load_server_eval_data(
            data_root=im_root,
            data_file=server_data_partition,
            normalization_metrics=data_metrics,
            batch_size=eval_batch_size,
        )

        metrics = {}

        # Evaluate the model on the test set
        try:
            model.to(device)
            metrics = test(model, eval_loader, device, num_classes, lb_ignore)

        finally:
            try:
                model.cpu()
            except Exception:
                pass

            del model
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        state_dict = arrays.to_torch_state_dict()

        torch.save(state_dict, save_latest)
        with open(latest_metric_file, "w") as f:
            json.dump(
                {"mIoU": metrics["mIoU"], "round": rounds_trained + server_round},
                f,
                indent=4,
            )
        print(
            f"[Server] Eval Round {rounds_trained + server_round}: Saved latest model and updated {latest_metric_file} "
            f"(best {best_miou['value']:.4f})"
        )

        miou = metrics.get("mIoU", 0.0)

        if miou > best_miou["value"]:
            torch.save(state_dict, save_best)
            print(f"[Server] 🎉 New best mIoU {miou:.4f} ")
            best_miou["value"] = miou

            with open(best_metric_file, "w") as f:
                json.dump(
                    {"best_miou": miou, "round": rounds_trained + server_round},
                    f,
                    indent=4,
                )

            print(
                f"[Server] New best mIoU {miou:.4f}, saved model and updated {best_metric_file}"
            )

        return MetricRecord(metrics)

    return central_evaluate


def make_silobn_evaluate_aggregator(context: Context):
    """Create an evaluation aggregator for SiloBN that handles client evaluation results.
    
    In SiloBN, since BN statistics are siloed on clients, evaluation must be performed
    on clients with their local BN statistics. This function creates an aggregator that:
    1. Receives evaluation metrics from all clients
    2. Averages the metrics (weighted by number of examples)
    3. Saves models based on aggregated mIoU
    """

    best_miou = context.run_config["best-miou"]
    best_miou = {"value": best_miou}

    save_latest = context.run_config["save-latest"]
    save_best = context.run_config["save-best"]
    best_metric_file = context.run_config["best-metric"]
    latest_metric_file = context.run_config["latest-metric"]
    rounds_trained = context.run_config["rounds-trained"]

    def aggregate_client_eval_results(
        server_round: int,
        arrays: ArrayRecord,
        client_metrics: list[dict],
    ) -> MetricRecord:
        """Aggregate evaluation results from all clients.
        
        Args:
            server_round: Current server round
            arrays: Server model arrays (for saving)
            client_metrics: List of dicts containing metrics from each client
            
        Returns:
            Aggregated metrics as MetricRecord
        """
        if not client_metrics:
            print(f"[SiloBN] Round {rounds_trained + server_round}: No client evaluation results received.")
            return MetricRecord({})

        # Aggregate metrics weighted by number of examples
        total_examples = 0
        weighted_miou = 0.0
        weighted_loss = 0.0
        
        # Per-class IoU aggregation
        all_class_ious = []
        
        for metrics in client_metrics:
            num_examples = metrics.get("num-examples", 1)
            miou = metrics.get("mIoU", 0.0)
            val_loss = metrics.get("val_loss", 0.0)
            class_ious = metrics.get("class_ious", None)
            
            total_examples += num_examples
            weighted_miou += miou * num_examples
            weighted_loss += val_loss * num_examples
            
            if class_ious is not None:
                all_class_ious.append((class_ious, num_examples))

        if total_examples == 0:
            return MetricRecord({})

        avg_miou = weighted_miou / total_examples
        avg_loss = weighted_loss / total_examples
        
        # Average class IoUs if available
        avg_class_ious = None
        if all_class_ious:
            # Simple average of class IoUs (could also do weighted)
            avg_class_ious = np.zeros_like(all_class_ious[0][0])
            total_weight = sum(w for _, w in all_class_ious)
            for ious, weight in all_class_ious:
                avg_class_ious += ious * weight
            avg_class_ious /= total_weight

        print(f"[SiloBN] Round {rounds_trained + server_round}: Aggregated client evaluation results")
        print(f"[SiloBN] Aggregated mIoU: {avg_miou:.4f} from {len(client_metrics)} clients ({total_examples} total examples)")

        # Save model checkpoints
        state_dict = arrays.to_torch_state_dict()

        torch.save(state_dict, save_latest)
        with open(latest_metric_file, "w") as f:
            json.dump(
                {"mIoU": avg_miou, "round": rounds_trained + server_round},
                f,
                indent=4,
            )
        print(
            f"[SiloBN] Saved latest model and updated {latest_metric_file} "
            f"(best {best_miou['value']:.4f})"
        )

        if avg_miou > best_miou["value"]:
            torch.save(state_dict, save_best)
            print(f"[SiloBN] 🎉 New best mIoU {avg_miou:.4f}")
            best_miou["value"] = avg_miou

            with open(best_metric_file, "w") as f:
                json.dump(
                    {"best_miou": avg_miou, "round": rounds_trained + server_round},
                    f,
                    indent=4,
                )

            print(
                f"[SiloBN] New best mIoU {avg_miou:.4f}, saved model and updated {best_metric_file}"
            )

        aggregated_metrics = {
            "mIoU": avg_miou,
            "val_loss": avg_loss,
            "num-examples": total_examples,
            "num-clients": len(client_metrics),
        }
        
        if avg_class_ious is not None:
            aggregated_metrics["class_ious"] = avg_class_ious.tolist()

        return MetricRecord(aggregated_metrics)

    return aggregate_client_eval_results
