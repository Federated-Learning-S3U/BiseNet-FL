"""fl-cityscapes-bisenetv2: A Flower / PyTorch app."""

import json

import torch

from flwr.app import ArrayRecord, MetricRecord, Context

from fl_cityscapes_bisenetv2.models.model_utils import get_model
from fl_cityscapes_bisenetv2.data_preparation.datasets import load_server_eval_data

from fl_cityscapes_bisenetv2.models.bisenetv2_train import (
    train_bisenetv2,
    test_bisenetv2,
)
from fl_cityscapes_bisenetv2.models.deeplabv3p_train import (
    train_deeplabv3p,
    test_deeplabv3p,
)


def train(
    net,
    model_name,
    trainloader,
    epochs,
    lr,
    wd,
    device,
    num_aux_heads,
    strategy,
    prox_mu,
    neg_entropy_weight: float = 0.0,
):
    """Train the model on the training set."""
    train_fn = train_bisenetv2 if model_name == "BiSeNetV2" else train_deeplabv3p

    return train_fn(
        net=net,
        trainloader=trainloader,
        epochs=epochs,
        lr=lr,
        wd=wd,
        device=device,
        strategy=strategy,
        prox_mu=prox_mu,
        neg_entropy_weight=neg_entropy_weight,
        num_aux_heads=num_aux_heads,
    )


def test(net, model_name, testloader, device, num_classes, lb_ignore=255):
    """Validate the model on the test set."""
    test_fn = test_bisenetv2 if model_name == "BiSeNetV2" else test_deeplabv3p

    return test_fn(
        net=net,
        testloader=testloader,
        device=device,
        num_classes=num_classes,
        lb_ignore=lb_ignore,
    )


def make_central_evaluate(context: Context):
    """Create a central evaluation function that accepts context as an argument."""

    # This best_miou is only set once and retains its value across multiple calls to central_evaluate
    best_miou = context.run_config["best-miou"]
    best_miou = {"value": best_miou}

    save_latest = context.run_config["save-latest"]
    save_best = context.run_config["save-best"]
    best_metric_file = context.run_config["best-metric"]
    latest_metric_file = context.run_config["latest-metric"]

    def central_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """Evaluate the global model on the server side (optional)."""

        # Read run config
        model_name: str = context.run_config["model-name"]
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

        device = torch.device(server_device)

        # Load Global Model
        model = get_model(num_classes, model_name).cpu()
        sd = arrays.to_torch_state_dict()
        model.load_state_dict(sd, strict=True)

        # Load the entire Cityscapes val dataset
        eval_loader = load_server_eval_data(
            data_root=im_root,
            data_file=server_data_partition,
            batch_size=eval_batch_size,
        )

        metrics = {}

        # Evaluate the model on the test set
        try:
            model.to(device)
            metrics = test(
                model, model_name, eval_loader, device, num_classes, lb_ignore
            )

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
