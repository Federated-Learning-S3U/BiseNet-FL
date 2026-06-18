"""BDD100K Client Dataset Module"""

import os

import cv2
from torch.utils.data import Dataset
import lib.data.transform_cv2 as T

# BDD100K semantic segmentation uses the same 19 Cityscapes-compatible classes:
#   0: road,  1: sidewalk,     2: building,      3: wall,         4: fence,
#   5: pole,  6: traffic light, 7: traffic sign,  8: vegetation,   9: terrain,
#  10: sky,  11: person,       12: rider,        13: car,         14: truck,
#  15: bus,  16: train,        17: motorcycle,   18: bicycle
# Unlabeled / ignore pixels are encoded as 255.
# BDD100K masks are provided in trainId format — no label remapping is needed.


class BDD100KClientDataset(Dataset):
    """BDD100K Client Dataset for federated semantic segmentation.

    Expects label masks in trainId format (values 0–18 for the 19 classes and
    255 for unlabeled pixels).  This is the standard format shipped with the
    official BDD100K semantic segmentation benchmark.
    """

    N_CLASSES = 19
    LB_IGNORE = 255

    def __init__(self, data_root, data, transform=None):
        """BDD100K Client Dataset Initialization

        Args:
            data_root (str): Root directory of the dataset.
            data (list): List of (image_path, label_path) tuples (relative to data_root).
            transform (callable, optional): Spatial / colour transform applied
                to the (image, label) pair before tensor conversion.
        """
        self.data_root = data_root
        self.data = data
        self.transform = transform
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

        im_lb = dict(im=image, lb=label)

        if self.transform:
            im_lb = self.transform(im_lb)

        out = self.to_tensor(im_lb)
        image, label = out["im"], out["lb"]
        return image.detach(), label.unsqueeze(0).detach()
