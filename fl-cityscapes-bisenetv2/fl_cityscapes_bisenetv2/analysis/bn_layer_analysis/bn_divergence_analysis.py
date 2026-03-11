# # Batch Normalization Analysis for Federated Learning
# 
# This notebook analyzes BatchNorm layer statistics across local and global models
# to validate the hypothesis that BN aggregation artifacts cause the Non-IID > IID performance gap.

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
from lib.models import BiSeNetV2
from fl_cityscapes_bisenetv2.utils.checkpoint_utils import (
    load_local_model, load_global_model, load_client_metadata, get_available_rounds, get_clients_in_round
)
from fl_cityscapes_bisenetv2.analysis.bn_layer_analysis.bn_divergence_utils import (
    extract_bn_statistics_from_state_dict, compute_bn_divergence, compute_bn_stability,
    compute_bn_variance_per_client, get_bn_layer_names
)

# Configuration
PARTITIONS = ['iid_partitions', 'non_iid_partitions']
BASE_PATH = 'res'
NUM_CLASSES = 19

# %% [markdown]
# ## 1. Load Saved Models and Extract BN Statistics

print("Loading models and extracting BN statistics...")

bn_stats_by_partition = {}
for partition_name in PARTITIONS:
    print(f"\n=== Processing {partition_name} ===")
    
    # Get available rounds
    available_rounds = get_available_rounds(BASE_PATH, partition_name)
    print(f"Available rounds: {available_rounds}")
    
    partition_data = {
        'rounds': available_rounds,
        'global_bn_stats': {},  # round -> bn_stats
        'client_bn_stats': {},   # round -> client_id -> bn_stats
    }
    
    for round_num in available_rounds:
        # Load global model
        try:
            global_state = load_global_model(BASE_PATH, partition_name, round_num)
            global_bn = extract_bn_statistics_from_state_dict(global_state)
            partition_data['global_bn_stats'][round_num] = global_bn
            print(f"  Round {round_num}: Loaded global model, found {len(global_bn)} BN layers")
        except Exception as e:
            print(f"  Round {round_num}: Failed to load global model: {e}")
            continue
        
        # Load client models
        client_ids = get_clients_in_round(BASE_PATH, partition_name, round_num)
        partition_data['client_bn_stats'][round_num] = {}
        
        for client_id in client_ids:
            try:
                local_state = load_local_model(BASE_PATH, partition_name, round_num, client_id)
                local_bn = extract_bn_statistics_from_state_dict(local_state)
                partition_data['client_bn_stats'][round_num][client_id] = local_bn
            except Exception as e:
                print(f"  Round {round_num}, Client {client_id}: Failed to load model")
        
        print(f"  Round {round_num}: Loaded {len(partition_data['client_bn_stats'][round_num])} client models")
    
    bn_stats_by_partition[partition_name] = partition_data

# %% [markdown]
# ## 2. Compute BN Divergence Between Local and Global Models

print("\n\nComputing BN divergence (local vs global)...")

divergence_analysis = {}

for partition_name, partition_data in bn_stats_by_partition.items():
    print(f"\n=== {partition_name} ===")
    
    divergence_analysis[partition_name] = {
        'per_round': {},  # round -> {layer -> per-client divergences}
        'per_client': {},  # client_id -> [divergence over rounds]
    }
    
    for round_num in partition_data['rounds']:
        if round_num not in partition_data['client_bn_stats']:
            continue
        
        global_bn = partition_data['global_bn_stats'].get(round_num, {})
        client_bns = partition_data['client_bn_stats'][round_num]
        
        round_divergences = {}
        
        for client_id, local_bn in client_bns.items():
            client_div = compute_bn_divergence(local_bn, global_bn)
            
            # Store per-layer divergences
            for layer_name, layer_div in client_div.items():
                if layer_name not in round_divergences:
                    round_divergences[layer_name] = []
                round_divergences[layer_name].append(layer_div['mean_l2'])
        
        divergence_analysis[partition_name]['per_round'][round_num] = round_divergences
        print(f"  Round {round_num}: Analyzed {len(client_bns)} clients, {len(round_divergences)} BN layers")

# %% [markdown]
# ## 3. Compute BN Stability (Changes Across Rounds)

print("\n\nComputing BN stability across rounds...")

stability_analysis = {}

for partition_name, partition_data in bn_stats_by_partition.items():
    print(f"\n=== {partition_name} ===")
    
    # Collect global BN stats per layer over all rounds
    layer_stats_over_rounds = {}
    
    for round_num in partition_data['rounds']:
        global_bn = partition_data['global_bn_stats'].get(round_num, {})
        
        for layer_name, layer_stats in global_bn.items():
            if layer_name not in layer_stats_over_rounds:
                layer_stats_over_rounds[layer_name] = {
                    'running_mean': [],
                    'running_var': []
                }
            
            if 'running_mean' in layer_stats:
                layer_stats_over_rounds[layer_name]['running_mean'].append(
                    layer_stats['running_mean'].cpu().numpy()
                )
            if 'running_var' in layer_stats:
                layer_stats_over_rounds[layer_name]['running_var'].append(
                    layer_stats['running_var'].cpu().numpy()
                )
    
    # Compute stability for each layer
    layer_stability = {}
    for layer_name, stats in layer_stats_over_rounds.items():
        stability_metrics = {}
        
        if stats['running_mean']:
            running_means = np.stack(stats['running_mean'])
            stability_metrics['running_mean_std'] = float(np.std(running_means, axis=0).mean())
        
        if stats['running_var']:
            running_vars = np.stack(stats['running_var'])
            stability_metrics['running_var_std'] = float(np.std(running_vars, axis=0).mean())
        
        layer_stability[layer_name] = stability_metrics
    
    stability_analysis[partition_name] = layer_stability
    print(f"  Computed stability for {len(layer_stability)} layers")

# %% [markdown]
# ## 4. Compare IID vs Non-IID BN Divergence

print("\n\nComparing IID vs Non-IID BN divergence...")

# Average divergence per round for each partition
iid_avg_div_per_round = {}
non_iid_avg_div_per_round = {}

for partition_name in PARTITIONS:
    output_dict = iid_avg_div_per_round if 'iid_partitions' in partition_name else non_iid_avg_div_per_round
    
    for round_num, layer_divs in divergence_analysis[partition_name]['per_round'].items():
        avg_divs = []
        for layer_name, divergences in layer_divs.items():
            avg_divs.extend(divergences)
        
        if avg_divs:
            output_dict[round_num] = float(np.mean(avg_divs))

print("\nAverage BN divergence per round:")
print("IID:", iid_avg_div_per_round)
print("Non-IID:", non_iid_avg_div_per_round)

# %% [markdown]
# ## 5. Visualizations

print("\n\nGenerating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: BN Divergence over rounds
ax = axes[0, 0]
if iid_avg_div_per_round:
    rounds = sorted(iid_avg_div_per_round.keys())
    iid_divs = [iid_avg_div_per_round[r] for r in rounds]
    ax.plot(rounds, iid_divs, marker='o', label='IID', linewidth=2)

if non_iid_avg_div_per_round:
    rounds = sorted(non_iid_avg_div_per_round.keys())
    non_iid_divs = [non_iid_avg_div_per_round[r] for r in rounds]
    ax.plot(rounds, non_iid_divs, marker='s', label='Non-IID', linewidth=2)

ax.set_xlabel('Communication Round')
ax.set_ylabel('Average BN Divergence (L2 distance)')
ax.set_title('BN Divergence Over Communication Rounds')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: BN Stability across layers (IID)
ax = axes[0, 1]
if 'iid_partitions' in stability_analysis:
    layer_names = list(stability_analysis['iid_partitions'].keys())[:10]  # First 10 layers
    stability_values = [
        stability_analysis['iid_partitions'][layer]['running_mean_std']
        for layer in layer_names
    ]
    ax.barh(range(len(layer_names)), stability_values, color='steelblue')
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels([l.split('.')[-1] for l in layer_names], fontsize=8)
    ax.set_xlabel('Running Mean Std (Stability)')
    ax.set_title('BN Running Mean Stability - IID Partition')
    ax.grid(True, alpha=0.3, axis='x')

# Plot 3: BN Stability across layers (Non-IID)
ax = axes[1, 0]
if 'non_iid_partitions' in stability_analysis:
    layer_names = list(stability_analysis['non_iid_partitions'].keys())[:10]
    stability_values = [
        stability_analysis['non_iid_partitions'][layer]['running_mean_std']
        for layer in layer_names
    ]
    ax.barh(range(len(layer_names)), stability_values, color='coral')
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels([l.split('.')[-1] for l in layer_names], fontsize=8)
    ax.set_xlabel('Running Mean Std (Stability)')
    ax.set_title('BN Running Mean Stability - Non-IID Partition')
    ax.grid(True, alpha=0.3, axis='x')

# Plot 4: Divergence distribution comparison
ax = axes[1, 1]
all_iid_divs = []
all_non_iid_divs = []

for round_num, layer_divs in divergence_analysis['iid_partitions']['per_round'].items():
    for divergences in layer_divs.values():
        all_iid_divs.extend(divergences)

for round_num, layer_divs in divergence_analysis['non_iid_partitions']['per_round'].items():
    for divergences in layer_divs.values():
        all_non_iid_divs.extend(divergences)

data_to_plot = [all_iid_divs, all_non_iid_divs]
bp = ax.boxplot(data_to_plot, labels=['IID', 'Non-IID'], patch_artist=True)
bp['boxes'][0].set_facecolor('steelblue')
bp['boxes'][1].set_facecolor('coral')
ax.set_ylabel('BN Divergence')
ax.set_title('BN Divergence Distribution')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('res/bn_divergence_analysis.png', dpi=300, bbox_inches='tight')
print("Saved visualization to res/bn_divergence_analysis.png")
plt.show()

# %% [markdown]
# ## 6. Key Findings

print("\n\n=== KEY FINDINGS ===")
print(f"\nIID Partition:")
print(f"  Average BN divergence: {np.mean(list(iid_avg_div_per_round.values())) if iid_avg_div_per_round else 'N/A':.6f}")
print(f"  Number of BN layers analyzed: {len(stability_analysis.get('iid_partitions', {}))}")

print(f"\nNon-IID Partition:")
print(f"  Average BN divergence: {np.mean(list(non_iid_avg_div_per_round.values())) if non_iid_avg_div_per_round else 'N/A':.6f}")
print(f"  Number of BN layers analyzed: {len(stability_analysis.get('non_iid_partitions', {}))}")

if iid_avg_div_per_round and non_iid_avg_div_per_round:
    iid_avg = np.mean(list(iid_avg_div_per_round.values()))
    non_iid_avg = np.mean(list(non_iid_avg_div_per_round.values()))
    print(f"\nBN Divergence Gap (Non-IID - IID): {non_iid_avg - iid_avg:.6f}")
    if non_iid_avg > iid_avg:
        print("✓ Hypothesis validated: Non-IID BN divergence is HIGHER than IID")
    else:
        print("✗ Hypothesis NOT validated: Non-IID BN divergence is LOWER than or equal to IID")

print("\n\nAnalysis complete! Review plots and metrics above.")
