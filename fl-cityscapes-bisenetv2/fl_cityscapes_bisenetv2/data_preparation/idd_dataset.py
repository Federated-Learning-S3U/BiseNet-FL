"""IDDAV3 (India Driving Dataset v3) Client Dataset Module"""

import os

import numpy as np
import cv2
from torch.utils.data import Dataset
import lib.data.transform_cv2 as T

# IDDAV3 level-2 taxonomy — 26 training classes:
#  trainId |  name
# ---------+-------------------------------
#     0    | road
#     1    | drivable fallback
#     2    | sidewalk
#     3    | non-drivable fallback
#     4    | person
#     5    | animal
#     6    | rider
#     7    | motorcycle
#     8    | bicycle
#     9    | autorickshaw
#    10    | car
#    11    | truck
#    12    | bus
#    13    | caravan
#    14    | vehicle fallback
#    15    | curb
#    16    | wall
#    17    | fence
#    18    | guardrail
#    19    | billboard
#    20    | traffic sign
#    21    | traffic light
#    22    | pole
#    23    | obs-str-bar-fallback
#    24    | building
#    25    | vegetation
#   255    | (ignore: bridge, tunnel, sky, fallback background, unlabeled)
#
# The raw IDD level-2 label PNG files use sequential IDs starting from 1 for
# "road".  ID 0 and IDs 27–255 are mapped to the ignore value (255).
# Update `IDD_LABELS_INFO` below if your dataset uses a different ID scheme
# (e.g., level-3 raw annotation IDs that require a two-step mapping).

IDD_LABELS_INFO = [
    # (raw_id, trainId, name)
    (0,   255, "unlabeled"),
    (1,   0,   "road"),
    (2,   1,   "drivable fallback"),
    (3,   2,   "sidewalk"),
    (4,   3,   "non-drivable fallback"),
    (5,   4,   "person"),
    (6,   5,   "animal"),
    (7,   6,   "rider"),
    (8,   7,   "motorcycle"),
    (9,   8,   "bicycle"),
    (10,  9,   "autorickshaw"),
    (11,  10,  "car"),
    (12,  11,  "truck"),
    (13,  12,  "bus"),
    (14,  13,  "caravan"),
    (15,  14,  "vehicle fallback"),
    (16,  15,  "curb"),
    (17,  16,  "wall"),
    (18,  17,  "fence"),
    (19,  18,  "guardrail"),
    (20,  19,  "billboard"),
    (21,  20,  "traffic sign"),
    (22,  21,  "traffic light"),
    (23,  22,  "pole"),
    (24,  23,  "obs-str-bar-fallback"),
    (25,  24,  "building"),
    (26,  25,  "vegetation"),
    # The following categories are ignored during training/evaluation:
    (27,  255, "bridge"),
    (28,  255, "tunnel"),
    (29,  255, "sky"),
    (30,  255, "fallback background"),
]

# 26 training class names indexed by trainId (useful for logging / confusion matrix labels)
IDD_CLASS_NAMES = [name for _, tid, name in IDD_LABELS_INFO if tid != 255]


def _build_lb_map(lb_ignore: int = 255) -> np.ndarray:
    """Build a 256-element uint8 lookup table: raw_id → trainId."""
    lb_map = np.full(256, lb_ignore, dtype=np.uint8)
    for raw_id, train_id, _ in IDD_LABELS_INFO:
        lb_map[raw_id] = train_id
    return lb_map


class IDDAClientDataset(Dataset):
    """IDDAV3 Client Dataset for federated semantic segmentation.

    Applies a label remapping from raw IDD level-2 category IDs to compact
    training IDs (0–25) before returning the label tensor.  Pixels with
    unmapped or unknown IDs are assigned the ignore value (255).

    If your pre-processing already outputs trainId masks (values 0–25 and 255),
    the default `IDD_LABELS_INFO` mapping is effectively an identity operation
    for the 26 training classes, so no changes are needed.
    """

    N_CLASSES = 26
    LB_IGNORE = 255

    def __init__(self, data_root, data, transform=None):
        """IDDAV3 Client Dataset Initialization

        Args:
            data_root (str): Root directory of the dataset.
            data (list): List of (image_path, label_path) tuples (relative to data_root).
            transform (callable, optional): Spatial / colour transform applied
                to the (image, label) pair before tensor conversion.
        """
        self.data_root = data_root
        self.data = data
        self.transform = transform
        self.lb_map = _build_lb_map(self.LB_IGNORE)
        self.to_tensor = T.ToTensor(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rel_img_path, rel_lb_path = self.data[idx]

        img_path = os.path.join(self.data_root, rel_img_path)
        lb_path = os.path.join(self.data_root, rel_lb_path)

        try:
            image = cv2.imread(img_path)[:, :, ::-1].copy()  # BGR → RGB
            label = cv2.imread(lb_path, 0)
        except (TypeError, FileNotFoundError):
            raise FileNotFoundError(f"File not found: {img_path} or {lb_path}")

        # Remap raw IDD level-2 IDs to compact trainIds
        label = self.lb_map[label]

        im_lb = dict(im=image, lb=label)

        if self.transform:
            im_lb = self.transform(im_lb)

        out = self.to_tensor(im_lb)
        image, label = out["im"], out["lb"]
        return image.detach(), label.unsqueeze(0).detach()
