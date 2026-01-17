# Configuration Template for Federated Learning Model Evaluation
# This file provides a template for configuring the evaluation notebook

import json
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Root directory of the BiseNet-FL project
PROJECT_ROOT = Path("/home/moustafa/Me/Projects/Grad/Code/BiseNet-FL")

# Directory where model checkpoints are stored
# This should be the parent directory containing data_partition_X directories
RESULTS_ROOT = PROJECT_ROOT / "fl-cityscapes-bisenetv2" / "results"

# Cityscapes dataset root
DATA_ROOT = PROJECT_ROOT / "datasets" / "cityscapes"

# ============================================================================
# DATA PARTITION CONFIGURATION
# ============================================================================

# Path to the partition JSON file
# Choose one:
#   - 'iid_partitions.json' for IID data distribution
#   - 'non_iid_partitions.json' for non-IID data distribution
DATA_PARTITION_FILE = DATA_ROOT / "iid_partitions.json"

# Number of data partitions (typically 4 for Cityscapes FL)
NUM_PARTITIONS = 4

# Number of semantic classes (19 for Cityscapes)
NUM_CLASSES = 19

# ============================================================================
# FEDERATED LEARNING CONFIGURATION
# ============================================================================

# List of aggregators to evaluate
# Examples: ['FedAvg', 'FedProx', 'FedEMA', 'FedOptimizer']
AGGREGATORS = ["FedAvg", "FedProx"]

# Communication round batches
# These should correspond to the directories in your RESULTS_ROOT
# Example: [60, 120, 180, 240] if you have checkpoints at these rounds
COMMUNICATION_ROUND_BATCHES = [60, 120, 180, 240]

# ============================================================================
# EVALUATION PARAMETERS
# ============================================================================

# Device for evaluation
# 'cuda' for GPU, 'cpu' for CPU (auto-detected if 'auto')
DEVICE = "auto"

# Batch size for evaluation (reduce if out of memory)
EVAL_BATCH_SIZE = 4

# Enable multi-scale evaluation (slows down evaluation significantly)
MULTI_SCALE_EVAL = False

# Evaluation scales if multi-scale is enabled
EVAL_SCALES = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

# Enable flip augmentation during evaluation
EVAL_FLIP = False

# Label to ignore during evaluation (255 for Cityscapes)
IGNORE_LABEL = 255

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

# Directory to save output plots and CSV files
OUTPUT_DIR = Path("./evaluation_outputs")

# DPI for saved plots
PLOT_DPI = 150

# Figure size for plots (width, height)
PLOT_FIGSIZE = (14, 6)

# ============================================================================
# ADVANCED CONFIGURATION
# ============================================================================

# Enable verbose logging
VERBOSE = True

# Save intermediate results after each evaluation step
SAVE_INTERMEDIATE = False

# Number of workers for data loading
NUM_WORKERS = 2

# Pin memory for data loading (faster on GPU systems)
PIN_MEMORY = True

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def verify_configuration():
    """Verify that all paths exist and configuration is valid."""
    errors = []
    warnings = []

    # Check paths
    if not PROJECT_ROOT.exists():
        errors.append(f"PROJECT_ROOT does not exist: {PROJECT_ROOT}")

    if not DATA_ROOT.exists():
        errors.append(f"DATA_ROOT does not exist: {DATA_ROOT}")

    if not DATA_PARTITION_FILE.exists():
        errors.append(f"DATA_PARTITION_FILE does not exist: {DATA_PARTITION_FILE}")

    # Check if RESULTS_ROOT exists
    if not RESULTS_ROOT.exists():
        warnings.append(f"RESULTS_ROOT does not exist yet: {RESULTS_ROOT}")

    # Check aggregators
    if not AGGREGATORS:
        errors.append("No aggregators specified in AGGREGATORS list")

    # Check communication rounds
    if not COMMUNICATION_ROUND_BATCHES:
        errors.append("No communication round batches specified")

    # Check numeric parameters
    if NUM_PARTITIONS < 1:
        errors.append("NUM_PARTITIONS must be >= 1")

    if NUM_CLASSES < 1:
        errors.append("NUM_CLASSES must be >= 1")

    if EVAL_BATCH_SIZE < 1:
        errors.append("EVAL_BATCH_SIZE must be >= 1")

    # Report results
    if errors:
        print("❌ Configuration Errors:")
        for error in errors:
            print(f"  - {error}")
        return False

    if warnings:
        print("⚠️  Configuration Warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    if not errors:
        print("✓ Configuration is valid!")
        return True


def get_expected_model_structure():
    """Print the expected directory structure for models."""
    print("\nExpected Model Directory Structure:")
    print(
        f"""
{RESULTS_ROOT}
├── data_partition_0/
│   ├── FedAvg/
│   │   ├── 60/
│   │   │   ├── latest_model.pth      ← Used for evaluation
│   │   │   └── best_model.pth        ← Not used in this evaluation
│   │   ├── 120/
│   │   ├── 180/
│   │   └── ...
│   ├── FedProx/
│   │   ├── 60/
│   │   ├── 120/
│   │   └── ...
│   └── [Other aggregators...]
├── data_partition_1/
│   ├── FedAvg/
│   ├── FedProx/
│   └── ...
├── data_partition_2/
│   └── ...
└── data_partition_3/
    └── ...

Note: The notebook loads "latest_model.pth" from each checkpoint directory.
For each data partition and aggregator, it evaluates the model on each client's training data.
"""
    )


def save_configuration_to_json(filepath=None):
    """Save current configuration to a JSON file."""
    if filepath is None:
        filepath = OUTPUT_DIR / "configuration.json"

    config = {
        "project_root": str(PROJECT_ROOT),
        "results_root": str(RESULTS_ROOT),
        "data_root": str(DATA_ROOT),
        "data_partition_file": str(DATA_PARTITION_FILE),
        "num_partitions": NUM_PARTITIONS,
        "num_classes": NUM_CLASSES,
        "aggregators": AGGREGATORS,
        "communication_round_batches": COMMUNICATION_ROUND_BATCHES,
        "device": DEVICE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "multi_scale_eval": MULTI_SCALE_EVAL,
        "eval_scales": EVAL_SCALES,
        "eval_flip": EVAL_FLIP,
        "ignore_label": IGNORE_LABEL,
    }

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Configuration saved to {filepath}")


# ============================================================================
# VALIDATION (Run when module is imported)
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FEDERATED LEARNING EVALUATION CONFIGURATION")
    print("=" * 70)
    print()

    print("Current Configuration:")
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"  Results Root: {RESULTS_ROOT}")
    print(f"  Data Root: {DATA_ROOT}")
    print(f"  Aggregators: {AGGREGATORS}")
    print(f"  Communication Rounds: {COMMUNICATION_ROUND_BATCHES}")
    print(f"  Device: {DEVICE}")
    print()

    verify_configuration()
    print()
    get_expected_model_structure()
    print()
    save_configuration_to_json()
