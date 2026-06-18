"""fl-cityscapes-bisenetv2: A Flower / PyTorch app."""

import json

from torch.utils.data import DataLoader

import lib.data.transform_cv2 as T
from fl_cityscapes_bisenetv2.data_preparation import (
    CityScapesClientDataset,
    BDD100KClientDataset,
    IDDAClientDataset,
    RandomPadSampler,
)

# Maps dataset-name → Dataset class
_DATASET_CLASSES = {
    "cityscapes": CityScapesClientDataset,
    "bdd100k": BDD100KClientDataset,
    "iddav3": IDDAClientDataset,
}

# Per-dataset (num_classes, lb_ignore) used when auto-deriving these values
DATASET_CONFIG = {
    "cityscapes": {"num_classes": 19, "lb_ignore": 255},
    "bdd100k":    {"num_classes": 19, "lb_ignore": 255},
    "iddav3":     {"num_classes": 26, "lb_ignore": 255},
}


def _get_dataset_class(dataset_name: str):
    name = dataset_name.lower()
    if name not in _DATASET_CLASSES:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Supported datasets: {list(_DATASET_CLASSES.keys())}"
        )
    return _DATASET_CLASSES[name]


def load_client_train_data(
    data_root: str,
    partitions: str,
    partition_id: int,
    batch_size: int,
    scales: list,
    cropsize: list,
    dataset_name: str = "cityscapes",
):
    """Load client partition training data for the specified dataset."""
    DatasetClass = _get_dataset_class(dataset_name)

    with open(partitions, "r", encoding="utf-8") as f:
        data_partitions = json.load(f)

    partition = data_partitions[str(partition_id)]

    ds = DatasetClass(
        data_root,
        partition["data"],
        T.TransformationTrain(scales, cropsize),
    )

    sampler = RandomPadSampler(ds, batch_size)

    trainloader = DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    return trainloader


def load_server_eval_data(
    data_root: str,
    data_file: str,
    batch_size: int,
    dataset_name: str = "cityscapes",
):
    """Load full validation data for server-side evaluation."""
    DatasetClass = _get_dataset_class(dataset_name)

    data = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            cleaned_line = line.strip()

            if not cleaned_line:
                continue

            split_line = cleaned_line.split(",")
            data.append([item.strip() for item in split_line])

    ds = DatasetClass(
        data_root,
        data,
        T.TransformationVal(),
    )

    evalloader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=False,
    )
    return evalloader
