"""Utilities for correlating model properties (weights, BN stats) with performance metrics."""

import numpy as np
from typing import Dict, Optional, Tuple
from scipy import stats as scipy_stats


def correlate_weight_distance_with_accuracy(
    weight_distances: Dict[int, float],
    client_accuracies: Dict[int, float]
) -> Dict[str, float]:
    """
    Compute correlation between weight distances and client accuracy metrics.
    
    Args:
        weight_distances: Dictionary mapping client_id -> distance metric
        client_accuracies: Dictionary mapping client_id -> accuracy/mIoU score
    
    Returns:
        Dictionary with correlation results
    """
    common_clients = set(weight_distances.keys()) & set(client_accuracies.keys())
    
    if len(common_clients) < 3:
        return {}
    
    distances = np.array([weight_distances[cid] for cid in sorted(common_clients)])
    accuracies = np.array([client_accuracies[cid] for cid in sorted(common_clients)])
    
    # Pearson correlation
    pearson_r, pearson_p = scipy_stats.pearsonr(distances, accuracies)
    
    # Spearman correlation
    spearman_r, spearman_p = scipy_stats.spearmanr(distances, accuracies)
    
    return {
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'n_samples': len(common_clients)
    }


def correlate_bn_divergence_with_accuracy(
    bn_divergences: Dict[int, Dict[str, float]],
    client_accuracies: Dict[int, float]
) -> Dict[str, float]:
    """
    Compute correlation between BN divergence and client accuracy.
    
    Args:
        bn_divergences: Dictionary mapping client_id -> {layer: divergence_score}
        client_accuracies: Dictionary mapping client_id -> accuracy score
    
    Returns:
        Dictionary with correlation results for each layer/metric
    """
    common_clients = set(bn_divergences.keys()) & set(client_accuracies.keys())
    
    if len(common_clients) < 3:
        return {}
    
    results = {}
    
    # Average BN divergence per client
    avg_divergences = []
    accuracies = []
    
    for client_id in sorted(common_clients):
        layer_divs = list(bn_divergences[client_id].values())
        if layer_divs:
            avg_divergences.append(np.mean(layer_divs))
            accuracies.append(client_accuracies[client_id])
    
    if len(avg_divergences) >= 3:
        pearson_r, pearson_p = scipy_stats.pearsonr(avg_divergences, accuracies)
        results['avg_bn_divergence_pearson_r'] = float(pearson_r)
        results['avg_bn_divergence_pearson_p'] = float(pearson_p)
    
    return results


def correlate_weight_magnitude_with_accuracy(
    weight_magnitudes: Dict[int, float],
    client_accuracies: Dict[int, float]
) -> Dict[str, float]:
    """
    Compute correlation between weight magnitude and accuracy.
    
    Args:
        weight_magnitudes: Dictionary mapping client_id -> magnitude
        client_accuracies: Dictionary mapping client_id -> accuracy
    
    Returns:
        Dictionary with correlation results
    """
    return correlate_weight_distance_with_accuracy(weight_magnitudes, client_accuracies)


def analyze_performance_vs_property(
    client_property: Dict[int, float],
    client_accuracies: Dict[int, float],
    property_name: str = "metric"
) -> Dict[str, any]:
    """
    Comprehensive analysis of correlation between a client property and accuracy.
    
    Args:
        client_property: Dictionary mapping client_id -> property value
        client_accuracies: Dictionary mapping client_id -> accuracy
        property_name: Name of the property for reporting
    
    Returns:
        Dictionary with comprehensive analysis results
    """
    common_clients = set(client_property.keys()) & set(client_accuracies.keys())
    
    if len(common_clients) < 3:
        return {'n_samples': 0}
    
    props = np.array([client_property[cid] for cid in sorted(common_clients)])
    accs = np.array([client_accuracies[cid] for cid in sorted(common_clients)])
    
    results = {
        'property_name': property_name,
        'n_samples': len(common_clients),
        
        # Property statistics
        'property_mean': float(np.mean(props)),
        'property_std': float(np.std(props)),
        'property_min': float(np.min(props)),
        'property_max': float(np.max(props)),
        
        # Accuracy statistics
        'accuracy_mean': float(np.mean(accs)),
        'accuracy_std': float(np.std(accs)),
        'accuracy_min': float(np.min(accs)),
        'accuracy_max': float(np.max(accs)),
        
        # Correlation metrics
    }
    
    if len(common_clients) >= 3:
        # Pearson
        pearson_r, pearson_p = scipy_stats.pearsonr(props, accs)
        results['pearson_r'] = float(pearson_r)
        results['pearson_p'] = float(pearson_p)
        
        # Spearman
        spearman_r, spearman_p = scipy_stats.spearmanr(props, accs)
        results['spearman_r'] = float(spearman_r)
        results['spearman_p'] = float(spearman_p)
        
        # Linear regression slope
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(props, accs)
        results['linregress_slope'] = float(slope)
        results['linregress_intercept'] = float(intercept)
        results['linregress_r_squared'] = float(r_value**2)
        results['linregress_p_value'] = float(p_value)
    
    return results


def identify_performance_outliers(
    client_property: Dict[int, float],
    client_accuracies: Dict[int, float],
    threshold_std: float = 2.0
) -> Dict[int, Dict[str, float]]:
    """
    Identify clients that are outliers in terms of accuracy but normal in property, 
    or vice versa.
    
    Args:
        client_property: Dictionary mapping client_id -> property value
        client_accuracies: Dictionary mapping client_id -> accuracy
        threshold_std: Number of standard deviations for outlier detection
    
    Returns:
        Dictionary mapping client_id -> {property, accuracy, outlier_dimension}
    """
    common_clients = set(client_property.keys()) & set(client_accuracies.keys())
    
    if len(common_clients) < 3:
        return {}
    
    props = np.array([client_property[cid] for cid in sorted(common_clients)])
    accs = np.array([client_accuracies[cid] for cid in sorted(common_clients)])
    
    prop_mean, prop_std = np.mean(props), np.std(props)
    acc_mean, acc_std = np.mean(accs), np.std(accs)
    
    outliers = {}
    
    for client_id in sorted(common_clients):
        prop = client_property[client_id]
        acc = client_accuracies[client_id]
        
        prop_zscore = (prop - prop_mean) / (prop_std + 1e-8)
        acc_zscore = (acc - acc_mean) / (acc_std + 1e-8)
        
        is_prop_outlier = abs(prop_zscore) > threshold_std
        is_acc_outlier = abs(acc_zscore) > threshold_std
        
        if is_prop_outlier or is_acc_outlier:
            outliers[client_id] = {
                'property': float(prop),
                'accuracy': float(acc),
                'property_zscore': float(prop_zscore),
                'accuracy_zscore': float(acc_zscore),
                'property_outlier': bool(is_prop_outlier),
                'accuracy_outlier': bool(is_acc_outlier)
            }
    
    return outliers


def compare_metrics_between_partitions(
    iid_property: Dict[int, float],
    iid_accuracy: Dict[int, float],
    non_iid_property: Dict[int, float],
    non_iid_accuracy: Dict[int, float],
    property_name: str = "metric"
) -> Dict[str, any]:
    """
    Compare correlation patterns between IID and Non-IID partitions.
    
    Args:
        iid_property: Dictionary mapping client_id -> property value (IID partition)
        iid_accuracy: Dictionary mapping client_id -> accuracy (IID partition)
        non_iid_property: Dictionary mapping client_id -> property value (Non-IID)
        non_iid_accuracy: Dictionary mapping client_id -> accuracy (Non-IID)
        property_name: Name of the property
    
    Returns:
        Dictionary comparing both partitions
    """
    iid_analysis = analyze_performance_vs_property(iid_property, iid_accuracy, property_name)
    non_iid_analysis = analyze_performance_vs_property(non_iid_property, non_iid_accuracy, property_name)
    
    results = {
        'property_name': property_name,
        'iid': iid_analysis,
        'non_iid': non_iid_analysis,
        'differences': {}
    }
    
    # Compare key metrics
    for metric in ['pearson_r', 'spearman_r']:
        if metric in iid_analysis and metric in non_iid_analysis:
            results['differences'][f'{metric}_diff'] = \
                non_iid_analysis[metric] - iid_analysis[metric]
    
    # Property and accuracy differences
    if 'property_mean' in iid_analysis and 'property_mean' in non_iid_analysis:
        results['differences']['property_mean_diff'] = \
            non_iid_analysis['property_mean'] - iid_analysis['property_mean']
    
    if 'accuracy_mean' in iid_analysis and 'accuracy_mean' in non_iid_analysis:
        results['differences']['accuracy_mean_diff'] = \
            non_iid_analysis['accuracy_mean'] - iid_analysis['accuracy_mean']
    
    return results
