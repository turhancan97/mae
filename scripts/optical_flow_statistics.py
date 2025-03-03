import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Optical Flow Statistics Calculator', add_help=False)
    parser.add_argument('--flow_dir', default='/shared/sets/datasets/vision/videos/walking_tour/Frames/Venice/step_1/train/OpticalFlow',
                        type=str, help='Directory containing optical flow .npy files')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for processing flow files')
    return parser

def calculate_online_stats(flow_dir, batch_size):
    """
    Calculate mean and std of flow tensors using online/streaming algorithm
    to avoid loading all tensors into memory at once.
    """
    # Get list of all flow files
    flow_files = sorted([f for f in os.listdir(flow_dir) if f.endswith('.npy')])
    total_files = len(flow_files)
    
    if total_files == 0:
        raise ValueError(f"No .npy files found in {flow_dir}")
    
    # Initialize variables for online mean and variance calculation
    n = 0  # count
    mean_x = 0
    mean_y = 0
    M2_x = 0  # sum of squared differences from mean for x channel
    M2_y = 0  # sum of squared differences from mean for y channel
    
    # Process files in batches
    for i in tqdm(range(0, total_files, batch_size), desc="Processing flow files"):
        batch_files = flow_files[i:min(i + batch_size, total_files)]
        
        # Load and process batch
        for flow_file in batch_files:
            flow_path = os.path.join(flow_dir, flow_file)
            flow = np.load(flow_path)  # shape: [2, H, W]
            
            # Flatten spatial dimensions
            flow_x = flow[0].flatten()
            flow_y = flow[1].flatten()
            
            # Update count
            n_old = n
            n += flow_x.size
            
            # Update mean and M2 for x channel
            delta_x = flow_x - mean_x
            mean_x += np.sum(delta_x) / n
            delta2_x = flow_x - mean_x
            M2_x += np.sum(delta_x * delta2_x)
            
            # Update mean and M2 for y channel
            delta_y = flow_y - mean_y
            mean_y += np.sum(delta_y) / n
            delta2_y = flow_y - mean_y
            M2_y += np.sum(delta_y * delta2_y)
    
    # Calculate final statistics
    mean = np.array([mean_x, mean_y])
    std = np.sqrt(np.array([M2_x, M2_y]) / (n - 1))
    
    return mean, std, n

def main(args):
    print(f"Calculating flow statistics from: {args.flow_dir}")
    
    # Calculate statistics
    mean, std, total_values = calculate_online_stats(args.flow_dir, args.batch_size)
    
    # Print results
    print("\nOptical Flow Statistics:")
    print(f"Total number of values processed: {total_values}")
    print(f"Mean: [{mean[0]:.4f}, {mean[1]:.4f}]")
    print(f"Std:  [{std[0]:.4f}, {std[1]:.4f}]")
    
    # Format for easy copying into code
    print("\nFor use in code:")
    print(f"self.flow_mean = torch.tensor([{mean[0]:.4f}, {mean[1]:.4f}])")
    print(f"self.flow_std = torch.tensor([{std[0]:.4f}, {std[1]:.4f}])")

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
