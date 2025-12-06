import numpy as np
from typing import Dict


def compute_metrics_from_cm(cm: np.ndarray) -> Dict[str, float]:
    """
    Computes semantic segmentation metrics from a confusion matrix directly.
    cm: num_classes x num_classes confusion matrix
    """
    num_classes = cm.shape[0]

    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP

    total_pixels = np.sum(cm)

    pixel_accuracy = np.sum(TP) / total_pixels if total_pixels > 0 else 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        precision_per_class = TP / (TP + FP)
        recall_per_class = TP / (TP + FN)
        f1_per_class = (
            2
            * precision_per_class
            * recall_per_class
            / (precision_per_class + recall_per_class)
        )
        iou_per_class = TP / (TP + FP + FN)

    precision_per_class[np.isnan(precision_per_class)] = 0.0
    recall_per_class[np.isnan(recall_per_class)] = 0.0
    f1_per_class[np.isnan(f1_per_class)] = 0.0
    iou_per_class[np.isnan(iou_per_class)] = 0.0

    mean_precision = np.mean(precision_per_class)
    mean_recall = np.mean(recall_per_class)
    mean_f1 = np.mean(f1_per_class)
    m_iou = np.mean(iou_per_class)

    class_frequency = np.sum(cm, axis=1) / total_pixels
    fw_m_iou = np.sum(class_frequency * iou_per_class)

    metrics = {
        "Accuracy": pixel_accuracy,
        "Precision": mean_precision,
        "Recall": mean_recall,
        "F1_Score": mean_f1,
        "mIoU": m_iou,
        "fw_mIoU": fw_m_iou,
    }

    metrics = {k: float(v) for k, v in metrics.items()}

    return metrics
