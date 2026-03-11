# # Weight Divergence Analysis for Federated Learning
# 
# This notebook analyzes how model weights diverge across clients and how different
# architectural branches (detail vs semantic) respond to data heterogeneity.

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
from fl_cityscapes_bisenetv2.analysis.weight_divergence.model_similarity import (
    cosine_similarity_matrices, branch_similarity_matrices, l2_distance_matrix,
    layer_wise_divergence, model_to_global_distance, identify_outlier_clients,
    cluster_clients_by_similarity, compute_model_consensus_distance
)
from fl_cityscapes_bisenetv2.analysis.weight_divergence.branch_dynamics import (
    branch_client_variance, branch_weight_magnitude, branch_convergence_rate,
    branch_layer_wise_convergence
)

# Configuration
PARTITIONS = ['iid_partitions', 'non_iid_partitions']
BASE_PATH = 'res'
NUM_CLASSES = 19

# %% [markdown]
# ## 1. Load Saved Models

print("Loading saved local and global models...")

models_by_partition = {}

for partition_name in PARTITIONS:
    print(f"\n=== Loading {partition_name} ===")
    
    partition_data = {
        'local_models': {},  # round -> {client_id -> state_dict}
        'global_models': {}  # round -> state_dict
    }
    
    available_rounds = get_available_rounds(BASE_PATH, partition_name)
    print(f"Available rounds: {available_rounds}")
    
    for round_num in available_rounds:
        # Load local models
        client_ids = get_clients_in_round(BASE_PATH, partition_name, round_num)
        partition_data['local_models'][round_num] = {}
        
        for client_id in client_ids:
            try:
                state_dict = load_local_model(BASE_PATH, partition_name, round_num, client_id)
                partition_data['local_models'][round_num][client_id] = state_dict
            except Exception as e:
                print(f"  Failed to load local model R{round_num}C{client_id}: {e}")
        
        # Load global model
        try:
            state_dict = load_global_model(BASE_PATH, partition_name, round_num)
            partition_data['global_models'][round_num] = state_dict
        except Exception as e:
            print(f"  Failed to load global model R{round_num}: {e}")
        
        print(f"  Round {round_num}: Loaded {len(partition_data['local_models'][round_num])} local models, global model")
    
    models_by_partition[partition_name] = partition_data

# %% [markdown]
# ## 2. Compute Model Similarity Matrices - Full Model

print("\n\nComputing cosine similarity matrices (full model)...")

similarity_analysis = {}

for partition_name, partition_data in models_by_partition.items():
    print(f"\n=== {partition_name} ===")
    
    similarity_analysis[partition_name] = {
        'client_similarity': {},  # round -> similarity_matrix
        'distance_to_global': {},  # round -> {client_id -> distance}
        'consensus_distance': {}  # round -> avg_distance_to_centroid
    }
    
    for round_num in sorted(partition_data['local_models'].keys()):
        local_models = partition_data['local_models'][round_num]
        global_model = partition_data['global_models'].get(round_num)
        
        if not local_models or not global_model:
            continue
        
        # Cosine similarity matrix
        sim_matrix, client_ids = cosine_similarity_matrices(local_models)
        similarity_analysis[partition_name]['client_similarity'][round_num] = {
            'matrix': sim_matrix,
            'client_ids': client_ids
        }
        
        # Distance to global model
        distances = model_to_global_distance(local_models, global_model)
        similarity_analysis[partition_name]['distance_to_global'][round_num] = distances
        
        # Consensus distance
        consensus_dist = compute_model_consensus_distance(local_models)
        similarity_analysis[partition_name]['consensus_distance'][round_num] = consensus_dist
        
        print(f"  Round {round_num}: Similarity matrix {sim_matrix.shape}, "
              f"consensus distance: {consensus_dist:.6f}")

# %% [markdown]
# ## 3. Branch-Specific Weight Analysis

print("\n\nAnalyzing detail vs semantic branch weights...")

# Initialize a model for layer mapping
model = BiSeNetV2(NUM_CLASSES)
branch_names = ['detail_branch', 'semantic_branch', 'decoder']

branch_analysis = {}

for partition_name, partition_data in models_by_partition.items():
    print(f"\n=== {partition_name} ===")
    
    branch_analysis[partition_name] = {
        'variance': {},  # branch -> round -> variance_metrics
        'magnitude': {},  # branch -> round -> {client_id -> magnitude}
        'similarity': {},  # branch -> round -> similarity_matrix
    }
    
    for branch_name in branch_names:
        print(f"\n  Branch: {branch_name}")
        
        branch_analysis[partition_name]['variance'][branch_name] = {}
        branch_analysis[partition_name]['magnitude'][branch_name] = {}
        branch_analysis[partition_name]['similarity'][branch_name] = {}
        
        for round_num in sorted(partition_data['local_models'].keys()):
            local_models = partition_data['local_models'][round_num]
            
            if not local_models:
                continue
            
            # Branch weight variance across clients
            variance = branch_client_variance(local_models, branch_name, model)
            branch_analysis[partition_name]['variance'][branch_name][round_num] = variance
            
            # Branch weight magnitude per client
            magnitude = branch_weight_magnitude(local_models, branch_name, model)
            branch_analysis[partition_name]['magnitude'][branch_name][round_num] = magnitude
            
            # Branch similarity matrix
            sim_matrices = branch_similarity_matrices(local_models, [branch_name], model)
            if branch_name in sim_matrices:
                sim_matrix, client_ids = sim_matrices[branch_name]
                branch_analysis[partition_name]['similarity'][branch_name][round_num] = {
                    'matrix': sim_matrix,
                    'client_ids': client_ids
                }
            
            print(f"    Round {round_num}: variance_std={variance.get('std', 0):.6f}")

# %% [markdown]
# ## 4. Compare IID vs Non-IID

print("\n\nComparing IID vs Non-IID metrics...")

comparison_results = {}

# Full-model consensus distance
print("\nFull-model consensus distance (lower = more agreement):")
for partition_name in PARTITIONS:
    consensus_dist = similarity_analysis[partition_name]['consensus_distance']
    if consensus_dist:
        avg_consensus = np.mean(list(consensus_dist.values()))
        print(f"  {partition_name}: {avg_consensus:.6f}")
        comparison_results[f"{partition_name}_consensus_distance"] = avg_consensus

# Branch variance comparison
print("\nBranch weight variance (standard deviation):")
for branch_name in branch_names:
    print(f"\n  {branch_name}:")
    for partition_name in PARTITIONS:
        variance_dict = branch_analysis[partition_name]['variance'].get(branch_name, {})
        if variance_dict:
            final_round = max(variance_dict.keys())
            final_variance = variance_dict[final_round].get('std', 0)
            print(f"    {partition_name}: {final_variance:.6f}")

# %% [markdown]
# ## 5. Identify Outlier Clients

print("\n\nIdentifying outlier clients (weight-wise divergent)...")

outlier_analysis = {}

for partition_name, partition_data in models_by_partition.items():
    print(f"\n=== {partition_name} ===")
    
    outlier_analysis[partition_name] = {}
    
    for round_num in sorted(partition_data['local_models'].keys()):
        local_models = partition_data['local_models'][round_num]
        
        if not local_models:
            continue
        
        outliers = identify_outlier_clients(local_models, threshold_std=2.0)
        
        outlier_analysis[partition_name][round_num] = outliers
        
        if outliers:
            print(f"  Round {round_num}: Found {len(outliers)} outlier clients")
            for client_id, distance in sorted(outliers.items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"    Client {client_id}: divergence={distance:.6f}")

# %% [markdown]
# ## 6. Visualizations

print("\n\nGenerating visualizations...")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Row 1: Consensus distance and similarity heatmaps
ax = fig.add_subplot(gs[0, 0])
for partition_name in PARTITIONS:
    consensus_dist = similarity_analysis[partition_name]['consensus_distance']
    if consensus_dist:
        rounds = sorted(consensus_dist.keys())
        distances = [consensus_dist[r] for r in rounds]
        label = 'IID' if 'iid_partitions' in partition_name else 'Non-IID'
        ax.plot(rounds, distances, marker='o', label=label, linewidth=2)
ax.set_xlabel('Communication Round')
ax.set_ylabel('Consensus Distance')
ax.set_title('Model Consensus (Centroid Distance)')
ax.legend()
ax.grid(True, alpha=0.3)

# Distance to global model evolution
ax = fig.add_subplot(gs[0, 1])
for partition_name in PARTITIONS:
    dist_to_global = similarity_analysis[partition_name]['distance_to_global']
    round_means = {}
    for round_num, client_dists in dist_to_global.items():
        round_means[round_num] = np.mean(list(client_dists.values()))
    
    if round_means:
        rounds = sorted(round_means.keys())
        means = [round_means[r] for r in rounds]
        label = 'IID' if 'iid_partitions' in partition_name else 'Non-IID'
        ax.plot(rounds, means, marker='s', label=label, linewidth=2)

ax.set_xlabel('Communication Round')
ax.set_ylabel('Avg Distance to Global')
ax.set_title('Client Convergence to Global Model')
ax.legend()
ax.grid(True, alpha=0.3)

# Branch variance comparison
ax = fig.add_subplot(gs[0, 2])
branches_var_iid = {}
branches_var_non_iid = {}

for branch_name in branch_names:
    var_iid = branch_analysis['iid_partitions']['variance'].get(branch_name, {})
    var_non_iid = branch_analysis['non_iid_partitions']['variance'].get(branch_name, {})
    
    if var_iid:
        final_round = max(var_iid.keys())
        branches_var_iid[branch_name] = var_iid[final_round].get('std', 0)
    
    if var_non_iid:
        final_round = max(var_non_iid.keys())
        branches_var_non_iid[branch_name] = var_non_iid[final_round].get('std', 0)

x = np.arange(len(branch_names))
width = 0.35

if branches_var_iid and branches_var_non_iid:
    iid_values = [branches_var_iid.get(b, 0) for b in branch_names]
    non_iid_values = [branches_var_non_iid.get(b, 0) for b in branch_names]
    
    ax.bar(x - width/2, iid_values, width, label='IID', color='steelblue')
    ax.bar(x + width/2, non_iid_values, width, label='Non-IID', color='coral')

ax.set_ylabel('Weight Variance (Std)')
ax.set_title('Branch Weight Variance Comparison')
ax.set_xticks(x)
ax.set_xticklabels([b.replace('_branch', '').replace('_', '') for b in branch_names])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Row 2: Similarity heatmaps
# IID similarity heatmap
ax = fig.add_subplot(gs[1, :2])
iid_similarity = similarity_analysis['iid_partitions']['client_similarity']
if iid_similarity:
    final_round = max(iid_similarity.keys())
    sim_matrix = iid_similarity[final_round]['matrix']
    client_ids = iid_similarity[final_round]['client_ids']
    
    im = ax.imshow(sim_matrix, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    ax.set_title(f'IID Client Similarity - Round {final_round}')
    ax.set_xlabel('Client ID')
    ax.set_ylabel('Client ID')
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    
    # Set tick labels
    if len(client_ids) <= 20:
        ax.set_xticks(range(len(client_ids)))
        ax.set_yticks(range(len(client_ids)))
        ax.set_xticklabels(client_ids, fontsize=8)
        ax.set_yticklabels(client_ids, fontsize=8)

# Non-IID similarity heatmap
ax = fig.add_subplot(gs[1, 2])
non_iid_similarity = similarity_analysis['non_iid_partitions']['client_similarity']
if non_iid_similarity:
    final_round = max(non_iid_similarity.keys())
    sim_matrix = non_iid_similarity[final_round]['matrix']
    client_ids = non_iid_similarity[final_round]['client_ids']
    
    im = ax.imshow(sim_matrix, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    ax.set_title(f'Non-IID Client Similarity - Round {final_round}')
    ax.set_xlabel('Client ID')
    ax.set_ylabel('Client ID')
    plt.colorbar(im, ax=ax, label='Cosine Similarity')

# Row 3: Branch convergence
ax = fig.add_subplot(gs[2, :])

for idx, branch_name in enumerate(branch_names):
    for partition_name in PARTITIONS:
        conv_rate = branch_analysis[partition_name].get('magnitude', {}).get(branch_name, {})
        
        if conv_rate:
            client_magnitudes_per_round = {}
            for round_num, client_mags in conv_rate.items():
                if client_mags:
                    client_magnitudes_per_round[round_num] = np.mean(list(client_mags.values()))
            
            if client_magnitudes_per_round:
                rounds = sorted(client_magnitudes_per_round.keys())
                mags = [client_magnitudes_per_round[r] for r in rounds]
                
                partition_label = 'IID' if 'iid_partitions' in partition_name else 'Non-IID'
                linestyle = '-' if 'iid' in partition_name else '--'
                ax.plot(rounds, mags, marker='o', label=f'{branch_name.replace("_", " ")} ({partition_label})',
                        linestyle=linestyle, linewidth=2)

ax.set_xlabel('Communication Round')
ax.set_ylabel('Average Weight Magnitude')
ax.set_title('Branch-wise Weight Magnitude Evolution')
ax.legend(fontsize=8, loc='best')
ax.grid(True, alpha=0.3)

plt.savefig('res/weight_divergence_analysis.png', dpi=300, bbox_inches='tight')
print("Saved visualization to res/weight_divergence_analysis.png")
plt.show()

# %% [markdown]
# ## 7. Key Findings

print("\n\n=== KEY FINDINGS ===")

# Consensus distance comparison
print("\nModel Consensus (lower is better - more agreement):")
for partition_name in PARTITIONS:
    consensus_dist = similarity_analysis[partition_name]['consensus_distance']
    if consensus_dist:
        avg_consensus = np.mean(list(consensus_dist.values()))
        label = 'IID' if 'iid' in partition_name else 'Non-IID'
        print(f"  {label}: {avg_consensus:.6f}")

# Branch variance comparison
print("\nBranch Weight Variance (final round, higher = more divergent):")
for branch_name in branch_names:
    print(f"  {branch_name.replace('_', ' ')}:")
    
    var_iid = branch_analysis['iid_partitions']['variance'].get(branch_name, {})
    var_non_iid = branch_analysis['non_iid_partitions']['variance'].get(branch_name, {})
    
    if var_iid and var_non_iid:
        final_round_iid = max(var_iid.keys())
        final_round_non_iid = max(var_non_iid.keys())
        
        var_iid_std = var_iid[final_round_iid].get('std', 0)
        var_non_iid_std = var_non_iid[final_round_non_iid].get('std', 0)
        
        print(f"    IID: {var_iid_std:.6f}")
        print(f"    Non-IID: {var_non_iid_std:.6f}")
        
        if var_non_iid_std > var_iid_std:
            print(f"    ✓ Non-IID diverges MORE (as expected)")
        else:
            print(f"    ✗ Non-IID diverges LESS (unexpected)")

print("\n\nAnalysis complete! Review visualizations above for detailed insights.")
