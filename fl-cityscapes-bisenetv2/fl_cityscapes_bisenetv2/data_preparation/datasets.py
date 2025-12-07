"""fl-cityscapes-bisenetv2: A Flower / PyTorch app."""

import json

from torch.utils.data import DataLoader

import lib.data.transform_cv2 as T
from fl_cityscapes_bisenetv2.data_preparation.client_dataset import (
    CityScapesClientDataset,
)


def load_client_train_data(
    data_root: str,
    partitions: str,
    partition_id: int,
    batch_size: int,
    scales: list,
    cropsize: list,
):
    """Load client partition CityScapes data."""
    with open(partitions, "r", encoding="utf-8") as f:
        data_partitions = json.load(f)

    partition = data_partitions[str(partition_id)]

    ds = CityScapesClientDataset(
        data_root,
        partition["data"],
        partition["data_metrics"],
        T.TransformationTrain(scales, cropsize),
    )

    # Construct dataloader
    trainloader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    return trainloader


def load_server_eval_data(
    data_root: str,
    data_file: str,
    normalization_metrics: dict,
    batch_size: int,
):
    """Load full CityScapes val data for server evaluation."""
    data = []
    with open(
        data_file,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            cleaned_line = line.strip()

            if not cleaned_line:
                continue

            split_line = cleaned_line.split(",")

            final_list = [item.strip() for item in split_line]

            data.append(final_list)

    ds = CityScapesClientDataset(
        data_root,
        data,
        normalization_metrics,
        T.TransformationVal(),
    )

    # Construct dataloader
    evalloader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=False,
    )
    return evalloader
