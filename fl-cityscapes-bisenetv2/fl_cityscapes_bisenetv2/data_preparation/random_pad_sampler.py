import math
import random
from torch.utils.data import Sampler


class RandomPadSampler(Sampler):
    """
    Randomly samples elements and pads the dataset
    so that total number of samples is divisible by batch_size.
    """

    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

        self.num_samples = len(data_source)
        self.target_size = math.ceil(self.num_samples / batch_size) * batch_size
        self.pad_size = self.target_size - self.num_samples

    def __iter__(self):
        # Shuffle original indices
        indices = list(range(self.num_samples))
        random.shuffle(indices)

        # Randomly resample extra indices (with replacement)
        if self.pad_size > 0:
            extra_indices = random.choices(indices, k=self.pad_size)
            indices.extend(extra_indices)

        return iter(indices)

    def __len__(self):
        return self.target_size
