import numpy as np
from typing import Dict


def compute_eval_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 19,
    ignore_index: int = 255,
) -> Dict[str, float]:
    """
    Computes semantic segmentation metrics using a confusion matrix.

    Args:
        y_true: Ground truth labels (e.g., shape (H, W) or (N, H, W)).
        y_pred: Predicted labels (same shape as y_true).
        num_classes: Number of valid classes (e.g., 19 for Cityscapes training set).
        ignore_index: Index to ignore in evaluation (e.g., 255 for Cityscapes).

    Returns:
        A dictionary containing the computed metrics.
    """

    # 1. Flatten the arrays and filter out ignored pixels
    valid_mask = y_true != ignore_index
    y_true_flat = y_true[valid_mask].flatten()
    y_pred_flat = y_pred[valid_mask].flatten()

    # The labels in y_true_flat are from 0 to 18 (the 19 valid classes).

    # 2. Compute the Confusion Matrix (CM)
    # CM[i, j] is the number of pixels of class i (true) classified as class j (pred)
    # The size will be num_classes x num_classes
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    # Create a joint 1D index: index = (true_class * num_classes) + pred_class
    # For Cityscapes: 19 * 0 + pred_0, 19 * 0 + pred_1, ..., 19 * 18 + pred_18
    combined_index = y_true_flat * num_classes + y_pred_flat

    # Use bincount to get counts for each combination
    counts = np.bincount(combined_index, minlength=num_classes**2)
    cm = counts.reshape(num_classes, num_classes)

    # 3. Compute necessary components from the Confusion Matrix
    # TP (True Positives) for class i is cm[i, i] (diagonal elements)
    # FN (False Negatives) for class i is sum(cm[i, :]) - cm[i, i] (sum of row i, excluding TP)
    # FP (False Positives) for class i is sum(cm[:, i]) - cm[i, i] (sum of col i, excluding TP)

    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP

    # Total pixels for all valid classes
    total_pixels = np.sum(cm)

    # --- Compute Individual Metrics ---

    ## Overall Pixel Accuracy
    # The ratio of correctly classified pixels to the total number of valid pixels.
    pixel_accuracy = np.sum(TP) / total_pixels if total_pixels > 0 else 0.0

    ## Class-wise Metrics (Precision, Recall, F1, IoU)

    # Suppress division by zero warnings for individual class metrics
    with np.errstate(divide="ignore", invalid="ignore"):
        # Precision (PPV - Positive Predictive Value) for each class: TP / (TP + FP)
        precision_per_class = TP / (TP + FP)

        # Recall (Sensitivity/True Positive Rate) for each class: TP / (TP + FN)
        recall_per_class = TP / (TP + FN)

        # F1 Score for each class: 2 * (Precision * Recall) / (Precision + Recall)
        f1_per_class = (
            2
            * (precision_per_class * recall_per_class)
            / (precision_per_class + recall_per_class)
        )

        # Jaccard Index / IoU (Intersection over Union) for each class: TP / (TP + FP + FN)
        iou_per_class = TP / (TP + FP + FN)

    # Handle NaNs (where TP+FP=0 or TP+FN=0 or sum=0) by setting them to 0.0
    precision_per_class[np.isnan(precision_per_class)] = 0.0
    recall_per_class[np.isnan(recall_per_class)] = 0.0
    f1_per_class[np.isnan(f1_per_class)] = 0.0
    iou_per_class[np.isnan(iou_per_class)] = 0.0

    # --- Compute Mean Metrics (Macro-averaging) ---

    ## Mean Precision, Mean Recall, Mean F1
    # Average the per-class scores across all valid classes.
    mean_precision = np.mean(precision_per_class)
    mean_recall = np.mean(recall_per_class)
    mean_f1 = np.mean(f1_per_class)

    ## Mean Intersection over Union (mIoU)
    # Average of the IoU for all valid classes.
    m_iou = np.mean(iou_per_class)

    ## Frequency Weighted Intersection over Union (fw-mIoU)
    # Weighted average where weights are based on the proportion of pixels
    # of each class relative to the total number of valid pixels.
    class_frequency = np.sum(cm, axis=1) / total_pixels
    fw_m_iou = np.sum(class_frequency * iou_per_class)

    return {
        "Accuracy": pixel_accuracy,
        "Precision": mean_precision,
        "Recall": mean_recall,
        "F1_Score": mean_f1,
        "mIoU": m_iou,
        "fw_mIoU": fw_m_iou,
        # "IoU_Per_Class": iou_per_class.tolist() # Can be included for detailed results
    }
