"""fl-cityscapes-bisenetv2: A Flower / PyTorch app."""

import json

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from lib.models import BiSeNetV2

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    num_rounds: int = context.run_config["num-server-rounds"]
    fraction_train: float = context.run_config["fraction-train"]

    local_epochs: int = context.run_config["local-epochs"]
    batch_size: int = context.run_config["batch_size"]

    im_root: str = context.run_config["im_root"]
    client_data_partition: str = context.run_config["client_data_partition"]

    num_classes: int = context.run_config["num_classes"]

    lr: float = context.run_config["lr"]
    weight_decay: float = context.run_config["weight_decay"]

    num_aux_heads: int = context.run_config["num_aux_heads"]

    scales: list = json.loads(context.run_config["scales"])
    cropsize: list = json.loads(context.run_config["cropsize"])

    save_path: str = context.run_config["respth"]

    # Load global model
    global_model = BiSeNetV2(num_classes)
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_train=fraction_train, fraction_evaluate=0.0)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, save_path)
