"""fl-cityscapes-bisenetv2: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from lib.models import BiSeNetV2

from fl_cityscapes_bisenetv2.task import make_central_evaluate

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    num_rounds: int = context.run_config["num-server-rounds"]
    fraction_train: float = context.run_config["fraction-train"]

    num_classes: int = context.run_config["num_classes"]

    resume = context.run_config["resume"]
    pretrained_path = context.run_config["pretrained_path"]

    lr: float = context.run_config["lr"]
    weight_decay: float = context.run_config["weight_decay"]

    # Load global model
    global_model = BiSeNetV2(num_classes)

    if resume:
        print(f"[Server] Resuming from pretrained model at {pretrained_path}")
        sd = torch.load(pretrained_path, map_location="cpu")
        global_model.load_state_dict(sd, strict=True)
    else:
        print(f"[Server] Starting from random initialized model.")

    arrays = ArrayRecord(global_model.state_dict())

    # Create central evaluation function that accepts context as an argument
    evaluate_fn = make_central_evaluate(context)

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_train=fraction_train, fraction_evaluate=0.0)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord(
            {
                "lr": lr,
                "weight_decay": weight_decay,
            }
        ),
        num_rounds=num_rounds,
        evaluate_fn=evaluate_fn,
    )
