"""fl-cityscapes-bisenetv2: A Flower / PyTorch app."""

import json

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fl_cityscapes_bisenetv2.task import load_data

# from fl_cityscapes_bisenetv2.task import test as test_fn
from fl_cityscapes_bisenetv2.task import train as train_fn

from lib.models import BiSeNetV2

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

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

    # Load the model and initialize it with the received weights
    model = BiSeNetV2(num_classes)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    trainloader = load_data(
        im_root, client_data_partition, partition_id, batch_size, scales, cropsize
    )

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        local_epochs,
        msg.content["config"]["lr"],
        device,
        num_aux_heads,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


# @app.evaluate() # TODO: Add evaluation function either per client or per server
# def evaluate(msg: Message, context: Context):
#     """Evaluate the model on local data."""

#     # Read run config
#     num_rounds: int = context.run_config["num-server-rounds"]
#     fraction_train: float = context.run_config["fraction-train"]

#     local_epochs: int = context.run_config["local-epochs"]
#     batch_size: int = context.run_config["batch_size"]

#     im_root: str = context.run_config["im_root"]
#     client_data_partition: str = context.run_config["client_data_partition"]

#     num_classes: int = context.run_config["num_classes"]

#     lr: float = context.run_config["lr"]
#     weight_decay: float = context.run_config["weight_decay"]

#     num_aux_heads: int = context.run_config["num_aux_heads"]

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
