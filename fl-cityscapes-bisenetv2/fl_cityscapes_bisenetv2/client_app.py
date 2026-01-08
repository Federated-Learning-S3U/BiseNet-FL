"""fl-cityscapes-bisenetv2: A Flower / PyTorch app."""

import json
import gc

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fl_cityscapes_bisenetv2.data_preparation.datasets import load_client_train_data
from fl_cityscapes_bisenetv2.task import train as train_fn
from fl_cityscapes_bisenetv2.utils.bn_utils import (
    filter_bn_statistics,
    extract_bn_statistics,
    merge_with_local_bn_stats,
)

from lib.models import BiSeNetV2

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Read run config
    local_epochs: int = context.run_config["local-epochs"]
    batch_size: int = context.run_config["batch-size"]

    weight_decay: float = context.run_config["weight-decay"]

    im_root: str = context.run_config["im-root"]
    client_data_partition: str = context.run_config["client-data-partition"]

    num_classes: int = context.run_config["num-classes"]

    num_aux_heads: int = context.run_config["num-aux-heads"]

    scales: list = json.loads(context.run_config["scales"])
    cropsize: list = json.loads(context.run_config["cropsize"])
    
    strategy_name: str = context.run_config["strategy-name"]

    # Load the model and initialize it with the received weights
    model = BiSeNetV2(num_classes)
    
    # For SiloBN: merge server params with local BN statistics
    server_state_dict = msg.content["arrays"].to_torch_state_dict()
    
    if strategy_name == "FedSiloBN":
        # Check if we have local BN statistics stored from previous round
        local_bn_record = context.state.get("local_bn_statistics", None)
        if local_bn_record is not None:
            # Convert ArrayRecord back to dict for merging
            local_bn_stats = local_bn_record.to_torch_state_dict()
            # Merge server learnable params with our local BN statistics
            merged_state_dict = merge_with_local_bn_stats(
                server_state_dict, local_bn_stats
            )
            model.load_state_dict(merged_state_dict)
        else:
            # First participation for this client - no local BN stats yet
            # Server may send full params (round 1) or only learnable params (round 2+)
            # Use model's default BN stats and load server params with strict=False
            # This handles both cases: full state dict or learnable-only state dict
            model.load_state_dict(server_state_dict, strict=False)
            print(f"[SiloBN] First participation - using model's default BN statistics")
    else:
        model.load_state_dict(server_state_dict)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]

    trainloader = load_client_train_data(
        im_root, client_data_partition, partition_id, batch_size, scales, cropsize
    )

    # Call the training function
    prox_mu = msg.content["config"].get("proximal-mu", 0.0)
    print(f"[Client {partition_id}] Starting training.")
    train_loss = train_fn(
        net=model,
        trainloader=trainloader,
        epochs=local_epochs,
        lr=msg.content["config"]["lr"],
        wd=weight_decay,
        device=device,
        num_aux_heads=num_aux_heads,
        strategy=strategy_name,
        prox_mu=prox_mu,
    )

    model.cpu()
    cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    # For SiloBN: filter out BN statistics before sending to server
    # and store local BN statistics for next round
    if strategy_name == "FedSiloBN":
        # Store local BN statistics in context state for next round
        # Must wrap in ArrayRecord since context.state only accepts Record types
        local_bn_stats = extract_bn_statistics(cpu_state_dict)
        context.state["local_bn_statistics"] = ArrayRecord(local_bn_stats)
        print(f"[Client {partition_id}] SiloBN: Stored {len(local_bn_stats)} local BN statistics")
        
        # Filter out BN statistics before sending to server
        filtered_state_dict = filter_bn_statistics(cpu_state_dict)
        print(f"[Client {partition_id}] SiloBN: Sending {len(filtered_state_dict)} learnable params to server")
        model_record = ArrayRecord(filtered_state_dict)
    else:
        model_record = ArrayRecord(cpu_state_dict)

    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    print(f"[Client {partition_id}] Training completed.")

    try:
        del model
    except Exception:
        pass
    gc.collect()

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass

    return Message(content=content, reply_to=msg)


# @app.evaluate() # TODO: Add evaluation function per client
# def evaluate(msg: Message, context: Context):
#     """Evaluate the model on local data."""

#     # Read run config
#     num_rounds: int = context.run_config["num-server-rounds"]
#     fraction_train: float = context.run_config["fraction-train"]

#     local_epochs: int = context.run_config["local-epochs"]
#     batch_size: int = context.run_config["batch-size"]

#     im_root: str = context.run_config["im-root"]
#     client_data_partition: str = context.run_config["client-data-partition"]

#     num_classes: int = context.run_config["num-classes"]

#     lr: float = context.run_config["lr"]
#     weight_decay: float = context.run_config["weight-decay"]

#     num_aux_heads: int = context.run_config["num-aux-heads"]

#     scales: list = context.run_config["scales"]
#     cropsize: list = context.run_config["cropsize"]

#     save_path: str = context.run_config["respth"]

#     # Load the model and initialize it with the received weights
#     model = BiSeNetV2(num_classes)
#     model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     # Load the data
#     partition_id = context.node_config["partition-id"]
#     num_partitions = context.node_config["num-partitions"]
#     _, valloader = load_data(partition_id, num_partitions)

#     # Call the evaluation function
#     eval_loss, eval_acc = test_fn(
#         model,
#         valloader,
#         device,
#     )

#     # Construct and return reply Message
#     metrics = {
#         "eval_loss": eval_loss,
#         "eval_acc": eval_acc,
#         "num-examples": len(valloader.dataset),
#     }
#     metric_record = MetricRecord(metrics)
#     content = RecordDict({"metrics": metric_record})
#     return Message(content=content, reply_to=msg)
