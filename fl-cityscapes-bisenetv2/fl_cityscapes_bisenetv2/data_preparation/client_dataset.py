"""Fl CityScapes BiSeNetV2 Client Dataset Module"""

import logging
import os

import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset
import lib.data.transform_cv2 as T


class CityScapesClientDataset(Dataset):
    """CityScapes Client Dataset

    Inherits from PyTorch Dataset class to handle CityScapes data for a specific client.
    """

    def __init__(self, data_root, data, transform=None):
        """CityScapes Client Dataset Initialization

        Args:
            data_root (str): Root directory of the dataset.
            data (list): List of image and label paths.
            transform (callable, optional): Transform to be applied to the images and labels. Defaults to None.
        """
        self.data_root = data_root
        self.data = data
        self.transform = transform
        self.to_tensor = T.ToTensor(*self._compute_mean_std())

    def _compute_mean_std(self):
        """Compute the mean and standard deviation of the dataset, for normalization.

        Returns:
            tuple: Mean and standard deviation for each channel (R, G, B).
        """
        print("=" * 50)
        print("Computing mean and std for dataset...")
        print("=" * 50)
        pixel_sum = np.zeros(3, dtype=np.float64)
        pixel_sq_sum = np.zeros(3, dtype=np.float64)
        n_pixels = 0

        logger = logging.getLogger()

        logger.info("Computing mean and std for client dataset")
        for path, _ in self.data:
            im = cv2.imread(os.path.join(self.data_root, path))
            if im is None:
                continue

            im = im[:, :, ::-1].astype(np.float32) / 255.0

            n_pixels += im.shape[0] * im.shape[1]
            pixel_sum += im.sum(axis=(0, 1))
            pixel_sq_sum += (im**2).sum(axis=(0, 1))

        rgb_mean = pixel_sum / n_pixels

        rgb_std = np.sqrt((pixel_sq_sum / n_pixels) - (rgb_mean**2))

        return rgb_mean.tolist(), rgb_std.tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Get a single item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Raises:
            FileNotFoundError: _description_

        Returns:
            _type_: _description_
        """
        rel_img_path, rel_lb_path = self.data[idx]

        # Load Image and Label
        img_path = os.path.join(self.data_root, rel_img_path)
        lb_path = os.path.join(self.data_root, rel_lb_path)

        try:
            image = cv2.imread(img_path)[:, :, ::-1].copy()
            label = cv2.imread(lb_path, 0)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {img_path} or {lb_path}")

        # Apply Tranformations
        im_lb = dict(im=image, lb=label)

        if self.transform:
            im_lb = self.transform(im_lb)

        # Convert to Tensor With Normalization
        out = self.to_tensor(im_lb)

        image, label = out["im"], out["lb"]
        return image.detach(), label.unsqueeze(0).detach()
