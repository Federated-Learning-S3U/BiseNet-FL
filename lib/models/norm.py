#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch.nn as nn


def _resolve_num_groups(num_channels: int, max_groups: int = 32) -> int:
    """Pick the largest valid group count not larger than max_groups."""
    num_groups = min(max_groups, num_channels)
    while num_groups > 1:
        if num_channels % num_groups == 0:
            return num_groups
        num_groups -= 1
    return 1


def make_group_norm(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    return nn.GroupNorm(_resolve_num_groups(num_channels, max_groups), num_channels)
