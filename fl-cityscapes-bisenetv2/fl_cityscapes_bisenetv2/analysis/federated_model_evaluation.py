#!/usr/bin/env python3
"""
Federated Learning Model Evaluation Script for BiSeNetV2
Evaluates models across data partitions, aggregators, and communication rounds.

This is the main evaluation script. It can be run directly or imported as a module.
To use as Jupyter notebook: Copy cells into a notebook or use: jupyter nbconvert --to notebook this_script.py
"""

# ============================================================================
# SECTION 1: IMPORT REQUIRED LIBRARIES
# ============================================================================

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, "/home/moustafa/Me/Projects/Grad/Code/BiseNet-FL")

# Import from the project
from lib.models import BiSeNetV2
from lib.data import get_data_loader
import lib.data.transform_cv2 as T
from fl_cityscapes_bisenetv2.data_preparation.client_dataset import (
    CityScapesClientDataset,
)
from fl_cityscapes_bisenetv2.data_preparation.utils import get_normalization_metrics
from tools.eval_metrics import compute_metrics_from_cm

print("✓ Libraries imported successfully!")


# ============================================================================
# SECTION 2: DEFINE MODEL LOADING AND EVALUATION FUNCTIONS
# ============================================================================


class MetricsCalculator:
    """Calculate semantic segmentation metrics from predictions and labels."""

    def __init__(self, n_classes, ignore_label=255):
        self.n_classes = n_classes
        self.ignore_label = ignore_label
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, predictions, labels):
        """Update confusion matrix with new predictions and labels."""
        predictions = (
            predictions.cpu().numpy()
            if isinstance(predictions, torch.Tensor)
            else predictions
        )
        labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels

        predictions = predictions.flatten()
        labels = labels.flatten()

        mask = labels != self.ignore_label
        predictions = predictions[mask]
        labels = labels[mask]

        np.add.at(self.confusion_matrix, (labels, predictions), 1)

    def compute_metrics(self) -> Dict[str, float]:
        """Compute mIoU and F1-score from confusion matrix."""
        return compute_metrics_from_cm(self.confusion_matrix)

    def reset(self):
        """Reset confusion matrix."""
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


def load_model(
    model_path: str, n_classes: int = 19, device: str = "cuda"
) -> Optional[BiSeNetV2]:
    """Load BiSeNetV2 model from checkpoint."""
    model = BiSeNetV2(n_classes, aux_mode="eval")

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        print(f"  ✓ Model loaded from {model_path}")
    else:
        print(f"  ✗ Model path not found: {model_path}")
        return None

    model = model.to(device)
    model.eval()
    return model


def load_client_dataset(
    im_root: str,
    partition_file: str,
    partition_id: int,
    n_classes: int = 19,
    batch_size: int = 4,
):
    """Load client dataset for evaluation."""
    try:
        with open(partition_file, "r", encoding="utf-8") as f:
            data_partitions = json.load(f)

        partition = data_partitions[str(partition_id)]
        normalization_metrics = partition.get("data_metrics", {})

        ds = CityScapesClientDataset(
            im_root, partition["data"], normalization_metrics, T.TransformationVal()
        )

        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
        )

        return dataloader, len(ds)
    except Exception as e:
        print(f"  Error loading client {partition_id} data: {e}")
        return None, 0


@torch.no_grad()
def evaluate_model_on_client(
    model: BiSeNetV2, dataloader, n_classes: int = 19, device: str = "cuda"
) -> Optional[Dict[str, float]]:
    """Evaluate model on client dataset."""
    if model is None or dataloader is None:
        return None

    metrics_calc = MetricsCalculator(n_classes, ignore_label=255)
    model.eval()

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            logits = outputs[0]

            predictions = torch.argmax(logits, dim=1)
            metrics_calc.update(predictions, labels.squeeze(1))

    return metrics_calc.compute_metrics()


print("✓ Model loading and evaluation functions defined!")


# ============================================================================
# SECTION 3: SET UP DIRECTORY PATHS AND CONFIGURATION
# ============================================================================

# CONFIGURATION - Modify these paths according to your directory structure
PROJECT_ROOT = Path("/home/moustafa/Me/Projects/Grad/Code/BiseNet-FL")
RESULTS_ROOT = PROJECT_ROOT / "fl-cityscapes-bisenetv2" / "results"
DATA_ROOT = PROJECT_ROOT / "datasets" / "cityscapes"
DATA_PARTITION_FILE = DATA_ROOT / "iid_partitions.json"

NUM_PARTITIONS = 4
NUM_CLASSES = 19
AGGREGATORS = ["FedAvg", "FedProx"]
COMMUNICATION_ROUND_BATCHES = [60, 120, 180, 240]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_BATCH_SIZE = 4

print(f"\n{'='*70}")
print("CONFIGURATION")
print(f"{'='*70}")
print(f"Project Root: {PROJECT_ROOT}")
print(f"Results Root: {RESULTS_ROOT}")
print(f"Data Root: {DATA_ROOT}")
print(f"Aggregators: {AGGREGATORS}")
print(f"Communication Rounds: {COMMUNICATION_ROUND_BATCHES}")
print(f"Device: {DEVICE}")

# Discover available models
print(f"\n{'='*70}")
print("DISCOVERING AVAILABLE MODELS")
print(f"{'='*70}")

available_models = {}
for partition_id in range(NUM_PARTITIONS):
    partition_key = f"partition_{partition_id}"
    available_models[partition_key] = {}

    for aggregator in AGGREGATORS:
        available_models[partition_key][aggregator] = []

        possible_paths = [
            RESULTS_ROOT / f"data_partition_{partition_id}" / aggregator,
            RESULTS_ROOT / f"partition_{partition_id}" / aggregator,
            PROJECT_ROOT / f"data_partition_{partition_id}" / aggregator,
        ]

        for base_path in possible_paths:
            if base_path.exists():
                print(f"\n✓ Found models at: {base_path}")
                for item in sorted(base_path.iterdir()):
                    if item.is_dir() and item.name.isdigit():
                        round_num = int(item.name)
                        latest_model = item / "latest_model.pth"
                        if latest_model.exists():
                            available_models[partition_key][aggregator].append(
                                round_num
                            )
                            print(f"  └─ Round {round_num}: ✓ latest_model.pth found")
                break


# ============================================================================
# SECTION 4: LOAD MODELS AND EVALUATE PER CLIENT
# ============================================================================

evaluation_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))


def find_model_base_path(partition_id: int, aggregator: str) -> Optional[Path]:
    """Find the base path for models of a specific partition and aggregator."""
    possible_paths = [
        RESULTS_ROOT / f"data_partition_{partition_id}" / aggregator,
        RESULTS_ROOT / f"partition_{partition_id}" / aggregator,
        PROJECT_ROOT / f"data_partition_{partition_id}" / aggregator,
    ]

    for path in possible_paths:
        if path.exists():
            return path
    return None


print(f"\n{'='*70}")
print("STARTING MODEL EVALUATION")
print(f"{'='*70}")

evaluated_configs = []

for partition_id in range(NUM_PARTITIONS):
    for aggregator in AGGREGATORS:
        base_path = find_model_base_path(partition_id, aggregator)

        if base_path is None:
            print(
                f"\n[SKIP] Partition {partition_id}, {aggregator}: No model directory found"
            )
            continue

        print(f"\n{'-'*70}")
        print(f"Evaluating: Partition {partition_id} | Aggregator: {aggregator}")
        print(f"Base Path: {base_path}")
        print(f"{'-'*70}")

        round_dirs = sorted(
            [
                int(d.name)
                for d in base_path.iterdir()
                if d.is_dir() and d.name.isdigit()
            ]
        )

        if not round_dirs:
            print(f"✗ No communication round directories found")
            continue

        print(f"Found rounds: {round_dirs}")

        for comm_round in round_dirs:
            round_path = base_path / str(comm_round)
            latest_model_path = round_path / "latest_model.pth"

            if not latest_model_path.exists():
                print(f"  ✗ Round {comm_round}: latest_model.pth not found")
                continue

            print(f"\n  Loading model for round {comm_round}...")
            model = load_model(
                str(latest_model_path), n_classes=NUM_CLASSES, device=DEVICE
            )

            if model is None:
                continue

            for client_id in range(NUM_PARTITIONS):
                try:
                    dataloader, num_samples = load_client_dataset(
                        str(DATA_ROOT),
                        str(DATA_PARTITION_FILE),
                        client_id,
                        n_classes=NUM_CLASSES,
                        batch_size=EVAL_BATCH_SIZE,
                    )

                    if dataloader is None:
                        continue

                    metrics = evaluate_model_on_client(
                        model, dataloader, n_classes=NUM_CLASSES, device=DEVICE
                    )

                    if metrics is not None:
                        evaluation_results[partition_id][aggregator][client_id][
                            comm_round
                        ] = metrics

                        miou = metrics.get("mIoU", 0.0)
                        f1 = metrics.get("F1_Score", 0.0)

                        print(
                            f"    Client {client_id}: mIoU={miou:.4f}, F1={f1:.4f} ({num_samples} samples)"
                        )

                except Exception as e:
                    print(f"    ✗ Error evaluating client {client_id}: {str(e)}")

            del model
            torch.cuda.empty_cache()

        evaluated_configs.append((partition_id, aggregator))

print(f"\n{'='*70}")
print("EVALUATION COMPLETE")
print(f"Configurations evaluated: {len(evaluated_configs)}")
print(f"{'='*70}")


# ============================================================================
# SECTION 5: AGGREGATE METRICS ACROSS COMMUNICATION ROUNDS
# ============================================================================


def aggregate_and_prepare_plot_data(
    evaluation_results, aggregator: str, metric: str = "mIoU"
) -> Tuple[Dict, Dict]:
    """Aggregate evaluation results for plotting."""
    plot_data = defaultdict(dict)
    best_models = {}

    for partition_id in evaluation_results:
        if aggregator not in evaluation_results[partition_id]:
            continue

        agg_results = evaluation_results[partition_id][aggregator]

        for client_id in agg_results:
            client_data = agg_results[client_id]

            for comm_round in sorted(client_data.keys()):
                metrics = client_data[comm_round]
                if metric in metrics:
                    plot_data[client_id][comm_round] = metrics[metric]

            all_values = [
                client_data[r].get(metric, 0)
                for r in client_data
                if metric in client_data[r]
            ]
            if all_values:
                best_models[client_id] = max(all_values)

    return dict(plot_data), best_models


processed_results = {}

for partition_id in evaluation_results:
    processed_results[partition_id] = {}

    for aggregator in evaluation_results[partition_id]:
        miou_data, best_miou = aggregate_and_prepare_plot_data(
            evaluation_results, aggregator, metric="mIoU"
        )
        f1_data, best_f1 = aggregate_and_prepare_plot_data(
            evaluation_results, aggregator, metric="F1_Score"
        )

        processed_results[partition_id][aggregator] = {
            "mIoU": {"data": miou_data, "best": best_miou},
            "F1_Score": {"data": f1_data, "best": best_f1},
        }

print("✓ Metrics aggregated and prepared for visualization!")
print(f"\nProcessed results for {len(processed_results)} partitions")


# ============================================================================
# SECTION 6: VISUALIZE CLIENT PERFORMANCE WITH BAR PLOTS
# ============================================================================


def plot_client_metrics(
    plot_data: Dict,
    best_models: Dict,
    metric_name: str = "mIoU",
    aggregator: str = "FedAvg",
    partition_id: int = 0,
    figsize: Tuple[int, int] = (14, 6),
):
    """Create a grouped bar plot for client performance."""
    if not plot_data:
        print(f"No data to plot for {aggregator}, Partition {partition_id}")
        return None

    client_ids = sorted(plot_data.keys())
    all_rounds = sorted(
        set(r for client_rounds in plot_data.values() for r in client_rounds.keys())
    )

    plot_df_list = []
    for client_id in client_ids:
        client_rounds = plot_data[client_id]
        for round_num in all_rounds:
            value = client_rounds.get(round_num, None)
            if value is not None:
                plot_df_list.append(
                    {
                        "Client": f"Client {client_id}",
                        "Communication Round": round_num,
                        metric_name: value,
                        "Model Type": f"Round {round_num}",
                    }
                )

        best_val = best_models.get(client_id, None)
        if best_val is not None:
            plot_df_list.append(
                {
                    "Client": f"Client {client_id}",
                    "Communication Round": "Best",
                    metric_name: best_val,
                    "Model Type": "Best",
                }
            )

    if not plot_df_list:
        return None

    plot_df = pd.DataFrame(plot_df_list)

    fig, ax = plt.subplots(figsize=figsize)

    rounds_and_best = sorted(set(plot_df["Model Type"]))
    colors = plt.cm.tab20(np.linspace(0, 1, len(rounds_and_best)))
    color_map = {r: colors[i] for i, r in enumerate(rounds_and_best)}

    x = np.arange(len(client_ids))
    width = 0.8 / (len(rounds_and_best) + 0.5)

    for idx, model_type in enumerate(rounds_and_best):
        data_subset = plot_df[plot_df["Model Type"] == model_type]
        values = [
            (
                data_subset[data_subset["Client"] == f"Client {cid}"][
                    metric_name
                ].values[0]
                if len(data_subset[data_subset["Client"] == f"Client {cid}"]) > 0
                else 0
            )
            for cid in client_ids
        ]

        offset = (idx - len(rounds_and_best) / 2 + 0.5) * width

        if model_type == "Best":
            ax.bar(
                x + offset,
                values,
                width,
                label=model_type,
                color=color_map[model_type],
                edgecolor="black",
                linewidth=2,
                alpha=0.9,
            )
        else:
            ax.bar(
                x + offset,
                values,
                width,
                label=model_type,
                color=color_map[model_type],
                alpha=0.7,
            )

    ax.set_xlabel("Clients", fontsize=12, fontweight="bold")
    ax.set_ylabel(metric_name, fontsize=12, fontweight="bold")
    ax.set_title(
        f"{metric_name} per Client\nPartition {partition_id} | Aggregator: {aggregator}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{cid}" for cid in client_ids])
    ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left", ncol=1)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0, max(plot_df[metric_name]) * 1.1])

    plt.tight_layout()
    return fig, ax


plots_generated = 0

for partition_id in sorted(processed_results.keys()):
    print(f"\n{'='*70}")
    print(f"Generating plots for Partition {partition_id}")
    print(f"{'='*70}")

    for aggregator in sorted(processed_results[partition_id].keys()):
        miou_data = processed_results[partition_id][aggregator]["mIoU"]["data"]
        miou_best = processed_results[partition_id][aggregator]["mIoU"]["best"]

        if miou_data:
            fig, ax = plot_client_metrics(
                miou_data,
                miou_best,
                metric_name="mIoU",
                aggregator=aggregator,
                partition_id=partition_id,
            )
            if fig:
                plt.savefig(
                    f"miou_partition_{partition_id}_{aggregator}.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.show()
                plots_generated += 1
                print(f"  ✓ Saved: miou_partition_{partition_id}_{aggregator}.png")

        f1_data = processed_results[partition_id][aggregator]["F1_Score"]["data"]
        f1_best = processed_results[partition_id][aggregator]["F1_Score"]["best"]

        if f1_data:
            fig, ax = plot_client_metrics(
                f1_data,
                f1_best,
                metric_name="F1-Score",
                aggregator=aggregator,
                partition_id=partition_id,
            )
            if fig:
                plt.savefig(
                    f"f1score_partition_{partition_id}_{aggregator}.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.show()
                plots_generated += 1
                print(f"  ✓ Saved: f1score_partition_{partition_id}_{aggregator}.png")

print(f"\n{'='*70}")
print(f"VISUALIZATION COMPLETE: {plots_generated} plots generated")
print(f"{'='*70}")


# ============================================================================
# SECTION 7: EXPORT RESULTS TO CSV
# ============================================================================


def export_results_to_csv(evaluation_results, output_dir: str = "."):
    """Export evaluation results to CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    all_results = []

    for partition_id in evaluation_results:
        for aggregator in evaluation_results[partition_id]:
            for client_id in evaluation_results[partition_id][aggregator]:
                for comm_round in evaluation_results[partition_id][aggregator][
                    client_id
                ]:
                    metrics = evaluation_results[partition_id][aggregator][client_id][
                        comm_round
                    ]

                    all_results.append(
                        {
                            "Partition": partition_id,
                            "Aggregator": aggregator,
                            "Client": client_id,
                            "Communication_Round": comm_round,
                            "mIoU": metrics.get("mIoU", None),
                            "F1_Score": metrics.get("F1_Score", None),
                            "Accuracy": metrics.get("Accuracy", None),
                            "Precision": metrics.get("Precision", None),
                            "Recall": metrics.get("Recall", None),
                            "fw_mIoU": metrics.get("fw_mIoU", None),
                        }
                    )

    results_df = pd.DataFrame(all_results)
    csv_path = output_path / "evaluation_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"✓ Results exported to: {csv_path}")

    summary_df = (
        results_df.groupby(["Partition", "Aggregator", "Client"])
        .agg({"mIoU": ["mean", "std", "max"], "F1_Score": ["mean", "std", "max"]})
        .round(4)
    )

    summary_path = output_path / "evaluation_summary.csv"
    summary_df.to_csv(summary_path)
    print(f"✓ Summary exported to: {summary_path}")

    return results_df, summary_df


if evaluation_results:
    results_df, summary_df = export_results_to_csv(evaluation_results, output_dir=".")
    print("\nResults DataFrame (first 10 rows):")
    print(results_df.head(10))
else:
    print("No evaluation results to export.")


print("\n" + "=" * 70)
print("EVALUATION COMPLETE!")
print("=" * 70)
print("\nGenerated files:")
print("  - evaluation_results.csv")
print("  - evaluation_summary.csv")
print("  - PNG plot files (miou_partition_*.png, f1score_partition_*.png)")
