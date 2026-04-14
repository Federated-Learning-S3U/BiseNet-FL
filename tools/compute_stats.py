#!/usr/bin/env python3
"""
Compute mean/std statistics for client partitions in JSON files.
Usage: python compute_stats.py <input_json> <output_json> <data_root>
"""

import json
import os
import sys
import numpy as np
import cv2
from datetime import datetime

def log_message(message):
    """Print message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def compute_mean_std_for_client(data_list, data_root=""):
    """
    Computes the mean and standard deviation for a client's image list.
    
    Args:
        data_list (list): List of [img_path, lbl_path] pairs.
        data_root (str): The base directory to resolve the image paths.
    
    Returns:
        tuple: (rgb_mean, rgb_std) as lists of floats [R, G, B].
    """
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    n_pixels = 0
    errors = 0
    
    for img_path, _ in data_list:
        full_path = os.path.join(data_root, img_path)
        im = cv2.imread(full_path)
        
        if im is None:
            log_message(f"Warning: Could not read image at {full_path}")
            errors += 1
            continue
        
        # BGR to RGB and normalize to [0, 1]
        im = im[:, :, ::-1].astype(np.float32) / 255.0
        
        n_pixels += im.shape[0] * im.shape[1]
        pixel_sum += im.sum(axis=(0, 1))
        pixel_sq_sum += (im**2).sum(axis=(0, 1))
    
    if n_pixels == 0:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], errors
    
    rgb_mean = pixel_sum / n_pixels
    rgb_std = np.sqrt(np.maximum(0, (pixel_sq_sum / n_pixels) - (rgb_mean**2)))
    
    return rgb_mean.tolist(), rgb_std.tolist(), errors

def add_stats_to_json(input_json_path, output_json_path, data_root=""):
    """
    Reads a JSON file with client partitions, computes mean/std for each client,
    and saves a new JSON with the statistics added with checkpoint support.
    
    Args:
        input_json_path (str): Path to the input JSON file.
        output_json_path (str): Path to save the output JSON file.
        data_root (str): Base directory for resolving image paths.
    """
    log_message(f"Starting processing of {input_json_path}")
    log_message(f"Data root: {data_root}")
    log_message(f"Output will be saved to: {output_json_path}")
    
    # Load the input JSON
    log_message(f"Loading {input_json_path}...")
    with open(input_json_path, 'r') as f:
        clients_data = json.load(f)
    
    total_clients = len(clients_data)
    log_message(f"Found {total_clients} clients to process")
    
    processed = 0
    skipped = 0
    
    # Process each client
    for client_id, client_info in clients_data.items():
        # Skip if already computed (checkpoint support)
        if "data_metrics" in client_info:
            skipped += 1
            log_message(f"[{skipped}/{total_clients}] Skipping Client {client_id} (already processed)")
            continue
        
        client_name = client_info.get("client_name", client_id)
        data_list = client_info.get("data", [])
        
        log_message(f"[{processed+1}/{total_clients}] Computing stats for Client {client_id} ({client_name}) with {len(data_list)} samples...")
        
        # Compute mean and std
        mean, std, errors = compute_mean_std_for_client(data_list, data_root=data_root)
        
        # Rebuild client_info with data_metrics BEFORE data
        # Extract existing fields
        client_name_val = client_info.get("client_name", client_id)
        num_samples_val = client_info.get("num_samples", len(data_list))
        
        # Clear and rebuild in desired order
        client_info.clear()
        client_info["client_name"] = client_name_val
        client_info["num_samples"] = num_samples_val
        client_info["data_metrics"] = {
            "mean": mean,
            "std": std
        }
        client_info["data"] = data_list
        
        processed += 1
        log_message(f"  → Mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
        log_message(f"  → Std:  [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
        if errors > 0:
            log_message(f"  → Errors: {errors} images could not be read")
        
        # Save checkpoint after each client
        with open(output_json_path, 'w') as f:
            json.dump(clients_data, f, indent=4)
        log_message(f"  ✓ Checkpoint saved ({processed} clients processed)")
    
    log_message(f"\n{'='*60}")
    log_message(f"Processing complete!")
    log_message(f"Total clients: {total_clients}")
    log_message(f"Newly processed: {processed}")
    log_message(f"Already done (skipped): {skipped}")
    log_message(f"Output saved to: {output_json_path}")
    log_message(f"{'='*60}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python compute_stats.py <input_json> <output_json> <data_root>")
        print("\nExample:")
        print("  python compute_stats.py setting1_iid.json setting1_iid_with_stats.json /path/to/data")
        sys.exit(1)
    
    input_json = sys.argv[1]
    output_json = sys.argv[2]
    data_root = sys.argv[3]
    
    # Validate input file exists
    if not os.path.exists(input_json):
        log_message(f"ERROR: Input file '{input_json}' not found!")
        sys.exit(1)
    
    # Validate data root exists
    if not os.path.exists(data_root):
        log_message(f"ERROR: Data root '{data_root}' not found!")
        sys.exit(1)
    
    try:
        add_stats_to_json(input_json, output_json, data_root)
    except Exception as e:
        log_message(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
