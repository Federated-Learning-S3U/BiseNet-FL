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
    save_client_bn_stats,
    load_client_bn_stats,
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
    
    # SiloBN resume training configs
    rounds_trained: int = context.run_config.get("rounds-trained", 0)
    client_bn_stats_dir: str = context.run_config.get("client-bn-stats-dir", "./res/client_bn_stats")

    # Load the model and initialize it with the received weights
    model = BiSeNetV2(num_classes)
    
    # For SiloBN: merge server params with local BN statistics
    server_state_dict = msg.content["arrays"].to_torch_state_dict()
    
    # Get partition_id early for SiloBN operations
    partition_id = context.node_config["partition-id"]
    
    if strategy_name == "FedSiloBN":
        # Check if we have local BN statistics stored from previous round (in-memory)
        local_bn_record = context.state.get("local_bn_statistics", None)
        
        if local_bn_record is not None:
            # Convert ArrayRecord back to dict for merging
            local_bn_stats = local_bn_record.to_torch_state_dict()
            # Merge server learnable params with our local BN statistics
            merged_state_dict = merge_with_local_bn_stats(
                server_state_dict, local_bn_stats
            )
            model.load_state_dict(merged_state_dict)
            print(f"[Client {partition_id}] SiloBN: Using in-memory local BN statistics")
        elif rounds_trained > 0:
            # Resume training: try to load BN stats from disk
            disk_bn_stats = load_client_bn_stats(client_bn_stats_dir, partition_id)
            if disk_bn_stats is not None:
                # Merge server learnable params with loaded local BN statistics
                merged_state_dict = merge_with_local_bn_stats(
                    server_state_dict, disk_bn_stats
                )
                model.load_state_dict(merged_state_dict)
                # Also store in context.state for subsequent rounds
                context.state["local_bn_statistics"] = ArrayRecord(disk_bn_stats)
                print(f"[Client {partition_id}] SiloBN Resume: Loaded {len(disk_bn_stats)} BN stats from disk")
            else:
                # Resuming but no saved BN stats found - use defaults
                model.load_state_dict(server_state_dict, strict=False)
                print(f"[Client {partition_id}] SiloBN Resume: No saved BN stats found, using defaults")
        else:
            # First participation for this client - no local BN stats yet
            # Server may send full params (round 1) or only learnable params (round 2+)
            # Use model's default BN stats and load server params with strict=False
            # This handles both cases: full state dict or learnable-only state dict
            model.load_state_dict(server_state_dict, strict=False)
            print(f"[Client {partition_id}] SiloBN: First participation - using model's default BN statistics")
    else:
        model.load_state_dict(server_state_dict)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
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
    # and store local BN statistics for next round (both in-memory and on disk)
    if strategy_name == "FedSiloBN":
        # Extract and store local BN statistics
        local_bn_stats = extract_bn_statistics(cpu_state_dict)
        
        # Store in context.state for next round (in-memory, within session)
        context.state["local_bn_statistics"] = ArrayRecord(local_bn_stats)
        
        # Save to disk for resume training support (persists across sessions)
        save_path = save_client_bn_stats(local_bn_stats, client_bn_stats_dir, partition_id)
        print(f"[Client {partition_id}] SiloBN: Stored {len(local_bn_stats)} local BN statistics")
        print(f"[Client {partition_id}] SiloBN: Saved BN stats to {save_path}")
        
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


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data (used for SiloBN client-side evaluation)."""
    from fl_cityscapes_bisenetv2.task import test as test_fn
    from fl_cityscapes_bisenetv2.data_preparation.datasets import load_server_eval_data
    from fl_cityscapes_bisenetv2.data_preparation.utils import aggregate_client_metrics

    # Read run config
    eval_batch_size: int = context.run_config["eval-batch-size"]
    im_root: str = context.run_config["im-root"]
    server_data_partition: str = context.run_config["server-data-partition"]
    client_data_partition: str = context.run_config["client-data-partition"]
    num_classes: int = context.run_config["num-classes"]
    lb_ignore: int = context.run_config["lb-ignore"]
    strategy_name: str = context.run_config["strategy-name"]
    
    # SiloBN resume training configs
    rounds_trained: int = context.run_config.get("rounds-trained", 0)
    client_bn_stats_dir: str = context.run_config.get("client-bn-stats-dir", "./res/client_bn_stats")

    partition_id = context.node_config["partition-id"]

    # Load the model
    model = BiSeNetV2(num_classes)
    server_state_dict = msg.content["arrays"].to_torch_state_dict()

    # For SiloBN: merge server params with local BN statistics
    if strategy_name == "FedSiloBN":
        local_bn_record = context.state.get("local_bn_statistics", None)
        
        if local_bn_record is not None:
            local_bn_stats = local_bn_record.to_torch_state_dict()
            merged_state_dict = merge_with_local_bn_stats(
                server_state_dict, local_bn_stats
            )
            model.load_state_dict(merged_state_dict)
            print(f"[Client {partition_id}] SiloBN Eval: Using in-memory local BN statistics")
        elif rounds_trained > 0:
            # Resume training: try to load BN stats from disk
            disk_bn_stats = load_client_bn_stats(client_bn_stats_dir, partition_id)
            if disk_bn_stats is not None:
                merged_state_dict = merge_with_local_bn_stats(
                    server_state_dict, disk_bn_stats
                )
                model.load_state_dict(merged_state_dict)
                # Also store in context.state for subsequent operations
                context.state["local_bn_statistics"] = ArrayRecord(disk_bn_stats)
                print(f"[Client {partition_id}] SiloBN Eval Resume: Loaded BN stats from disk")
            else:
                model.load_state_dict(server_state_dict, strict=False)
                print(f"[Client {partition_id}] SiloBN Eval Resume: No saved BN stats, using defaults")
        else:
            # First eval - no local BN stats, use default
            model.load_state_dict(server_state_dict, strict=False)
            print(f"[Client {partition_id}] SiloBN Eval: Using default BN statistics (first eval)")
    else:
        model.load_state_dict(server_state_dict)
    else:
        model.load_state_dict(server_state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the server eval data (full validation set)
    data_mean, data_std = aggregate_client_metrics(client_data_partition)
    data_metrics = {"mean": data_mean, "std": data_std}

    eval_loader = load_server_eval_data(
        data_root=im_root,
        data_file=server_data_partition,
        normalization_metrics=data_metrics,
        batch_size=eval_batch_size,
    )

    # Evaluate the model
    metrics = test_fn(model, eval_loader, device, num_classes, lb_ignore)
    print(f"[Client {partition_id}] Evaluation completed. mIoU: {metrics.get('mIoU', 0.0):.4f}")

    # Clean up
    model.cpu()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Construct and return reply Message
    metrics["num-examples"] = len(eval_loader.dataset)
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
