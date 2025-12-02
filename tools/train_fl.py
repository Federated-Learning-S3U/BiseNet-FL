from functools import partial
import logging
import json

import flwr as fl
from torch.utils.data import DataLoader
import lib.data.transform_cv2 as T

from configs import set_cfg_from_file
from tools.utils import parse_args

from lib.logger import setup_logger, log_msg

from lib.clients.cityscapes_client import CityScapesClient
from lib.clients.cityscapes_client_dataset import CityScapesClientDataset


def get_client_data(cfg, partition_id) -> DataLoader:
    """Get the data loader for the client.

    Args:
        cfg: Configuration object.
        partition_id: ID of the client partition.

    Returns:
        DataLoader: PyTorch Train DataLoader for the client dataset.
    """
    with open(cfg.client_data_partition, "r") as f:
        data_partitions = json.load(f)

    client_data = data_partitions[partition_id]

    ds = CityScapesClientDataset(
        cfg.im_root,
        client_data["data"],
        T.TransformationTrain(cfg.scales, cfg.cropsize),
    )
    dl = DataLoader(
        ds,
        batch_size=cfg.ims_per_gpu,
        shuffle=True,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )
    return dl


def client_fn_impl(context: fl.common.Context, cfg) -> fl.client.NumPyClient:
    """Create a Flower client representing a single organization.

    Args:
        context (fl.common.Context): The context for the client.

    Returns:
        CityScapesClient: The Flower client instance.
    """
    partition_id = context.node_config["partition-id"]

    loader = get_client_data(cfg, partition_id)

    return CityScapesClient(cfg, loader)


def main(cfg):
    """Main function to start the federated learning simulation.

    Args:
        cfg: Configuration object.
    """

    client_fn_with_cfg = partial(client_fn_impl, cfg=cfg)

    client_app = fl.client.ClientApp(client_fn=client_fn_with_cfg)

    server_app = fl.server.ServerApp(
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=fl.server.strategy.FedAvg(
            fraction_fit=0.5,
            min_fit_clients=7,
        ),
    )

    # "backend_config" lets you specify resources.
    fl.simulation.run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=100,  # Simulate 100 total clients
        backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 1.0}},
    )


if __name__ == "__main__":
    args = parse_args()
    cfg = set_cfg_from_file(args.config)

    setup_logger(f"federated-{cfg.model_type}-{cfg.dataset.lower()}-train", cfg.respth)

    main(cfg)
