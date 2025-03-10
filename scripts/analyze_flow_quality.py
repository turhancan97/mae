import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torchvision.utils import flow_to_image
from PIL import Image

from util.misc import analyze_flow_quality, visualize_flow_samples

def main():
    parser = argparse.ArgumentParser('Flow Quality Analysis Tool')
    parser.add_argument('--data_path', default='/shared/sets/datasets/vision/videos/walking_tour/Frames/Venice/step_30', 
                        type=str, help='dataset path')
    parser.add_argument('--num_samples', default=200, type=int, help='Number of samples to analyze')
    parser.add_argument('--output_dir', default='flow_analysis', type=str, help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Visualize flow samples')
    parser.add_argument('--split', default='train', type=str, help='Dataset split to analyze')
    args = parser.parse_args()
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Get directories
    image_dir = os.path.join(args.data_path, args.split, 'Image')
    flow_dir = os.path.join(args.data_path, args.split, 'OpticalFlow')
    
    # Analyze flow quality
    print(f"Analyzing flow quality in {flow_dir}...")
    stats = analyze_flow_quality(flow_dir, args.num_samples, args.output_dir, plot=True)
    
    if stats:
        # Print statistics
        print("\nFlow Quality Statistics:")
        print(f"Magnitude - Mean: {stats['mag_mean']:.2f}, Median: {stats['mag_median']:.2f}, Std: {stats['mag_std']:.2f}")
        print(f"Variance - Mean: {stats['var_mean']:.2f}, Median: {stats['var_median']:.2f}, Std: {stats['var_std']:.2f}")
        print(f"Coverage - Mean: {stats['coverage_mean']:.2f}, Median: {stats['coverage_median']:.2f}, Std: {stats['coverage_std']:.2f}")
        
        # Suggest thresholds
        print("\nSuggested Filtering Thresholds:")
        print(f"Magnitude Threshold: {stats['suggested_mag_threshold']:.2f}")
        print(f"Variance Threshold: {stats['suggested_var_threshold']:.2f}")
        
        # Save statistics to file
        with open(os.path.join(args.output_dir, 'flow_stats.txt'), 'w') as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
    
    # Visualize samples if requested
    if args.visualize:
        print(f"\nVisualizing {args.num_samples} flow samples...")
        visualize_flow_samples(image_dir, flow_dir, os.path.join(args.output_dir, 'samples'), args.num_samples)
    
    print(f"\nAnalysis complete. Results saved to {args.output_dir}")
    
    # Instructions for using the thresholds
    print("\nTo use these thresholds, run your training with:")
    print(f"  --filter_flow --flow_mag_threshold {stats.get('suggested_mag_threshold', 1.5):.2f} --flow_var_threshold {stats.get('suggested_var_threshold', 1.0):.2f}")

if __name__ == '__main__':
    main() 