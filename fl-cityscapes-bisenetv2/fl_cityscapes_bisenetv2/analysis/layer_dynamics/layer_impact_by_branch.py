# # Branch Learning Dynamics Analysis for Federated Learning
# 
# This notebook provides deep insights into how detail and semantic branches
# learn and converge differently under IID vs Non-IID data distributions.

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

import torch
from lib.models import BiSeNetV2
from fl_cityscapes_bisenetv2.utils.checkpoint_utils import (
    load_local_model, load_global_model, load_client_metadata, get_available_rounds, get_clients_in_round
)
from fl_cityscapes_bisenetv2.analysis.weight_divergence.branch_dynamics import (
    compare_branch_learning_speed, branch_aggregation_impact, branch_client_variance,
    branch_weight_magnitude, branch_convergence_rate, branch_layer_wise_convergence,
    get_branch_parameter_count
)
from fl_cityscapes_bisenetv2.utils.weight_utils import (
    get_branch_state_dict, compute_layer_wise_cosine_similarity, compute_layer_wise_l2_distance
)

# Configuration
PARTITIONS = ['iid_partitions', 'non_iid_partitions']
BASE_PATH = 'res'
NUM_CLASSES = 19

# %% [markdown]
# ## 1. Load Saved Models

print("Loading saved local and global models for branch analysis...")

models_by_partition = {}

for partition_name in PARTITIONS:
    print(f"\n=== Loading {partition_name} ===")
    
    partition_data = {
        'local_models': {},  # round -> {client_id -> state_dict}
        'global_models': {}  # round -> state_dict
    }
    
    available_rounds = get_available_rounds(BASE_PATH, partition_name)
    
    for round_num in available_rounds:
        client_ids = get_clients_in_round(BASE_PATH, partition_name, round_num)
        partition_data['local_models'][round_num] = {}
        
        for client_id in client_ids:
            try:
                state_dict = load_local_model(BASE_PATH, partition_name, round_num, client_id)
                partition_data['local_models'][round_num][client_id] = state_dict
            except Exception as e:
                pass  # Skip on error
        
        try:
            state_dict = load_global_model(BASE_PATH, partition_name, round_num)
            partition_data['global_models'][round_num] = state_dict
        except Exception as e:
            pass  # Skip on error
        
        print(f"  Round {round_num}: {len(partition_data['local_models'][round_num])} local models")
    
    models_by_partition[partition_name] = partition_data

# Initialize a model for layer mapping
model = BiSeNetV2(NUM_CLASSES)
branch_names = ['detail_branch', 'semantic_branch', 'decoder']

# %% [markdown]
# ## 2. Analyze Branch Learning Speed

print("\n\nAnalyzing branch learning speed (weight change magnitude per round)...")

learning_speed_analysis = {}

for partition_name, partition_data in models_by_partition.items():
    print(f"\n=== {partition_name} ===")
    
    learning_speed_analysis[partition_name] = {}
    
    for branch_name in branch_names:
        speed = compare_branch_learning_speed(partition_data['local_models'], branch_name, model)
        learning_speed_analysis[partition_name][branch_name] = speed
        
        if speed:
            avg_speed = np.mean(list(speed.values()))
            print(f"  {branch_name}: avg_speed={avg_speed:.6f}")

# %% [markdown]
# ## 3. Analyze Aggregation Impact per Branch

print("\n\nAnalyzing branch-specific aggregation impact...")

aggregation_impact_analysis = {}

for partition_name, partition_data in models_by_partition.items():
    print(f"\n=== {partition_name} ===")
    
    aggregation_impact_analysis[partition_name] = {
        'before': {},  # branch -> round -> {client_id -> distance}
        'after': {}    # branch -> round -> {client_id -> distance}
    }
    
    for round_num in sorted(partition_data['local_models'].keys())[:-1]:  # Skip last round
        next_round = round_num + 1
        
        if (round_num not in partition_data['local_models'] or
            next_round not in partition_data['local_models'] or
            round_num not in partition_data['global_models']):
            continue
        
        # Before aggregation: distance from current local models to next global model
        current_local = partition_data['local_models'][round_num]
        next_global = partition_data['global_models'][next_round]
        
        for branch_name in branch_names:
            impact = branch_aggregation_impact(current_local, next_global, branch_name, model)
            
            if branch_name not in aggregation_impact_analysis[partition_name]['before']:
                aggregation_impact_analysis[partition_name]['before'][branch_name] = {}
            
            if impact:
                avg_distance = np.mean([imp['mean_distance'] for imp in impact.values()])
                aggregation_impact_analysis[partition_name]['before'][branch_name][round_num] = avg_distance
            
            print(f"  Round {round_num} -> {next_round}, {branch_name}: "
                  f"avg_distance={aggregation_impact_analysis[partition_name]['before'][branch_name].get(round_num, 0):.6f}")

# %% [markdown]
# ## 4. Branch Convergence Rate Analysis

print("\n\nAnalyzing branch convergence rates...")

convergence_analysis = {}

for partition_name, partition_data in models_by_partition.items():
    print(f"\n=== {partition_name} ===")
    
    convergence_analysis[partition_name] = {}
    
    for branch_name in branch_names:
        convergence = branch_convergence_rate(
            partition_data['local_models'],
            partition_data['global_models'],
            branch_name,
            model
        )
        
        convergence_analysis[partition_name][branch_name] = convergence
        
        if convergence:
            print(f"  {branch_name}:")
            rates = list(convergence.values())
            print(f"    Avg distance to global: {np.mean(rates):.6f}")
            print(f"    Convergence trend: {('decreasing' if rates[-1] < rates[0] else 'increasing')}")

# %% [markdown]
# ## 5. Layer-wise Convergence Within Each Branch

print("\n\nAnalyzing layer-wise convergence...")

layer_convergence_analysis = {}

for partition_name, partition_data in models_by_partition.items():
    print(f"\n=== {partition_name} ===")
    
    layer_convergence_analysis[partition_name] = {}
    
    for branch_name in branch_names:
        layer_conv = branch_layer_wise_convergence(
            partition_data['local_models'],
            partition_data['global_models'],
            branch_name,
            model
        )
        
        layer_convergence_analysis[partition_name][branch_name] = layer_conv
        
        if layer_conv:
            # Get final round
            final_round = max(layer_conv.keys())
            final_layers = layer_conv[final_round]
            
            if final_layers:
                # Find layers with highest and lowest convergence
                sorted_layers = sorted(final_layers.items(), key=lambda x: x[1], reverse=True)
                print(f"  {branch_name} (round {final_round}):")
                if sorted_layers:
                    most_conv = sorted_layers[-1]
                    least_conv = sorted_layers[0]
                    print(f"    Most converged: {most_conv[0].split('.')[-1]}")
                    print(f"    Least converged: {least_conv[0].split('.')[-1]}")

# %% [markdown]
# ## 6. Branch Parameter Count

print("\n\nBranch parameter statistics:")

for partition_name in PARTITIONS:
    print(f"\n{partition_name}:")
    
    partition_data = models_by_partition[partition_name]
    first_round = min(partition_data['local_models'].keys())
    first_local = next(iter(partition_data['local_models'][first_round].values()))
    
    for branch_name in branch_names:
        param_count = get_branch_parameter_count(first_local, branch_name, model)
        print(f"  {branch_name}: {param_count:,} parameters")

# %% [markdown]
# ## 7. Visualizations

print("\n\nGenerating comprehensive visualizations...")

fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

# Row 1: Learning speed per branch
for col_idx, branch_name in enumerate(branch_names):
    ax = fig.add_subplot(gs[0, col_idx])
    
    for partition_name in PARTITIONS:
        speed = learning_speed_analysis[partition_name].get(branch_name, {})
        
        if speed:
            rounds = sorted(speed.keys())
            speeds = [speed[r] for r in rounds]
            
            label = 'IID' if 'iid' in partition_name else 'Non-IID'
            ax.plot(rounds, speeds, marker='o', label=label, linewidth=2)
    
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Weight Change Magnitude')
    ax.set_title(f'{branch_name.replace("_", " ")} Learning Speed')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Row 2: Branch variance per partition
for col_idx, partition_name in enumerate(PARTITIONS):
    ax = fig.add_subplot(gs[1, col_idx])
    
    partition_data = models_by_partition[partition_name]
    
    for branch_name in branch_names:
        variances = []
        rounds_list = sorted(partition_data['local_models'].keys())
        
        for round_num in rounds_list:
            local_models = partition_data['local_models'][round_num]
            variance = branch_client_variance(local_models, branch_name, model)
            if variance:
                variances.append(variance.get('std', 0))
            else:
                variances.append(0)
        
        if variances:
            ax.plot(rounds_list, variances, marker='s', label=branch_name.replace('_', ' '), linewidth=2)
    
    label = 'IID' if 'iid' in partition_name else 'Non-IID'
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Weight Variance (Std)')
    ax.set_title(f'{label} Partition - Branch Variance')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Axis for overall branch variance comparison
ax = fig.add_subplot(gs[1, 2])
final_variances = {}

for partition_name in PARTITIONS:
    partition_data = models_by_partition[partition_name]
    final_round = max(partition_data['local_models'].keys())
    local_models = partition_data['local_models'][final_round]
    
    partition_label = 'IID' if 'iid' in partition_name else 'Non-IID'
    
    for branch_name in branch_names:
        variance = branch_client_variance(local_models, branch_name, model)
        key = f"{partition_label}\n{branch_name.replace('_branch', '')}"
        final_variances[key] = variance.get('std', 0)

if final_variances:
    keys = list(final_variances.keys())
    values = list(final_variances.values())
    colors = ['steelblue'] * 3 + ['coral'] * 3
    
    bars = ax.bar(range(len(keys)), values, color=colors)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, fontsize=8, rotation=45, ha='right')
    ax.set_ylabel('Weight Variance')
    ax.set_title('Final Round Branch Variance')
    ax.grid(True, alpha=0.3, axis='y')

# Row 3: Convergence rate per branch
for col_idx, branch_name in enumerate(branch_names):
    ax = fig.add_subplot(gs[2, col_idx])
    
    for partition_name in PARTITIONS:
        convergence = convergence_analysis[partition_name].get(branch_name, {})
        
        if convergence:
            rounds = sorted(convergence.keys())
            distances = [convergence[r] for r in rounds]
            
            label = 'IID' if 'iid' in partition_name else 'Non-IID'
            ax.plot(rounds, distances, marker='o', label=label, linewidth=2)
    
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Avg Distance to Global')
    ax.set_title(f'{branch_name.replace("_", " ")} Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Row 4: Aggregation impact comparison
ax = fig.add_subplot(gs[3, :2])

for partition_name in PARTITIONS:
    partition_label = 'IID' if 'iid' in partition_name else 'Non-IID'
    
    for branch_idx, branch_name in enumerate(branch_names):
        impact_dict = aggregation_impact_analysis[partition_name]['before'].get(branch_name, {})
        
        if impact_dict:
            rounds = sorted(impact_dict.keys())
            distances = [impact_dict[r] for r in rounds]
            
            linestyle = '-' if 'iid' in partition_name else '--'
            ax.plot(rounds, distances, marker='o', label=f'{partition_label} - {branch_name.replace("_branch", "")}',
                    linestyle=linestyle, linewidth=2)

ax.set_xlabel('Communication Round')
ax.set_ylabel('Distance to Next Global')
ax.set_title('Aggregation Impact - Client to Next Global Model Distance')
ax.legend(fontsize=8, loc='best')
ax.grid(True, alpha=0.3)

# Summary statistics
ax = fig.add_subplot(gs[3, 2])
ax.axis('off')

summary_text = "SUMMARY - FINAL ROUND:\n\n"

for partition_name in PARTITIONS:
    partition_label = 'IID' if 'iid' in partition_name else 'Non-IID'
    partition_data = models_by_partition[partition_name]
    final_round = max(partition_data['local_models'].keys())
    local_models = partition_data['local_models'][final_round]
    
    summary_text += f"{partition_label}:\n"
    
    for branch_name in branch_names:
        variance = branch_client_variance(local_models, branch_name, model)
        convergence = convergence_analysis[partition_name].get(branch_name, {})
        
        var = variance.get('std', 0)
        conv = convergence.get(final_round, 0)
        
        summary_text += f"  {branch_name.replace('_branch', '')}:\n"
        summary_text += f"    Variance: {var:.4f}\n"
        summary_text += f"    Convergence: {conv:.4f}\n"
    
    summary_text += "\n"

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.savefig('res/branch_learning_dynamics.png', dpi=300, bbox_inches='tight')
print("Saved visualization to res/branch_learning_dynamics.png")
plt.show()

# %% [markdown]
# ## 8. Key Insights

print("\n\n=== KEY INSIGHTS ===")

print("\n1. LEARNING SPEED:")
for branch_name in branch_names:
    print(f"\n  {branch_name.replace('_', ' ')}:")
    
    for partition_name in PARTITIONS:
        speed = learning_speed_analysis[partition_name].get(branch_name, {})
        if speed:
            partition_label = 'IID' if 'iid' in partition_name else 'Non-IID'
            avg_speed = np.mean(list(speed.values()))
            print(f"    {partition_label}: {avg_speed:.6f}")

print("\n2. CONVERGENCE:")
for branch_name in branch_names:
    print(f"\n  {branch_name.replace('_', ' ')}:")
    
    for partition_name in PARTITIONS:
        convergence = convergence_analysis[partition_name].get(branch_name, {})
        if convergence:
            partition_label = 'IID' if 'iid' in partition_name else 'Non-IID'
            
            initial = list(convergence.values())[0] if convergence else 0
            final = list(convergence.values())[-1] if convergence else 0
            improvement = ((initial - final) / initial * 100) if initial > 0 else 0
            
            print(f"    {partition_label}: {final:.6f} (improved {improvement:.1f}%)")

print("\n3. BRANCH DIVERGENCE (Final Round):")
for partition_name in PARTITIONS:
    partition_label = 'IID' if 'iid' in partition_name else 'Non-IID'
    partition_data = models_by_partition[partition_name]
    final_round = max(partition_data['local_models'].keys())
    local_models = partition_data['local_models'][final_round]
    
    print(f"\n  {partition_label}:")
    for branch_name in branch_names:
        variance = branch_client_variance(local_models, branch_name, model)
        var = variance.get('std', 0)
        print(f"    {branch_name.replace('_', ' ')}: {var:.6f}")

print("\n\nAnalysis complete! Review visualizations for more detailed insights.")
