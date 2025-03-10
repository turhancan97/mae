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
    parser.add_argument('--magnitude', action='store_true',
                        help='Calculate statistics for flow magnitude instead of separate x,y components')
    return parser

def calculate_online_stats(flow_dir, batch_size, use_magnitude=False):
    """
    Calculate mean and std of flow tensors using online/streaming algorithm
    to avoid loading all tensors into memory at once.
    
    If use_magnitude is True, calculate statistics for the magnitude of flow vectors.
    Otherwise, calculate statistics for x and y components separately.
    """
    # Get list of all flow files
    flow_files = sorted([f for f in os.listdir(flow_dir) if f.endswith('.npy')])
    total_files = len(flow_files)
    
    if total_files == 0:
        raise ValueError(f"No .npy files found in {flow_dir}")
    
    # Initialize variables for online mean and variance calculation
    n = 0  # count
    
    if use_magnitude:
        # For magnitude mode
        mean_mag = 0
        M2_mag = 0  # sum of squared differences from mean for magnitude
    else:
        # For separate x,y components
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
            flow_np = np.load(flow_path)  # shape: [2, H, W]
            
            if use_magnitude:
                # Convert to torch tensor and calculate magnitude using torch.norm
                flow = torch.from_numpy(flow_np).to('cuda')
                flow_mag = torch.norm(flow, p=2, dim=0, keepdim=True)  # Shape: (1, H, W)
                
                # Flatten spatial dimensions and convert back to numpy
                flow_mag = flow_mag.cpu().numpy().flatten()
                
                # Update count
                n_old = n
                n += flow_mag.size
                
                # Update mean and M2 for magnitude
                delta_mag = flow_mag - mean_mag
                mean_mag += np.sum(delta_mag) / n
                delta2_mag = flow_mag - mean_mag
                M2_mag += np.sum(delta_mag * delta2_mag)
            else:
                # Flatten spatial dimensions
                flow_x = flow_np[0].flatten()
                flow_y = flow_np[1].flatten()
                
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
    if use_magnitude:
        mean = np.array([mean_mag])
        std = np.sqrt(np.array([M2_mag]) / (n - 1))
    else:
        mean = np.array([mean_x, mean_y])
        std = np.sqrt(np.array([M2_x, M2_y]) / (n - 1))
    
    return mean, std, n

def main(args):
    print(f"Calculating flow statistics from: {args.flow_dir}")
    print(f"Mode: {'Magnitude' if args.magnitude else 'X,Y Components'}")
    
    # Calculate statistics
    mean, std, total_values = calculate_online_stats(args.flow_dir, args.batch_size, args.magnitude)
    
    # Print results
    print("\nOptical Flow Statistics:")
    print(f"Total number of values processed: {total_values}")
    
    if args.magnitude:
        print(f"Mean magnitude: {mean[0]:.4f}")
        print(f"Std magnitude:  {std[0]:.4f}")
        
        # Format for easy copying into code
        print("\nFor use in code:")
        print(f"self.flow_mag_mean = torch.tensor([{mean[0]:.4f}])")
        print(f"self.flow_mag_std = torch.tensor([{std[0]:.4f}])")
    else:
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
