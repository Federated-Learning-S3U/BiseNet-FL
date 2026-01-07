# my_classes_directory/__init__.py
from .CustomFedAvg import CustomFedAvg
from .CustomFedAvgM import CustomFedAvgM
from .CustomFedProx import CustomFedProx
from .CustomFedEMA import CustomFedEMA
from .CustomFedSiloBN import CustomFedSiloBN

# Re-export BN utilities for backward compatibility
# Prefer importing directly from fl_cityscapes_bisenetv2.utils.bn_utils
from fl_cityscapes_bisenetv2.utils.bn_utils import (
    is_bn_statistic,
    filter_bn_statistics,
    extract_bn_statistics,
    merge_with_local_bn_stats,
)
