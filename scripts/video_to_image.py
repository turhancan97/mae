import os
import torch
import torchvision.transforms as transforms
import torchvision
from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np

def get_args_parser():
    parser = argparse.ArgumentParser('Video to frames converter', add_help=False)
    parser.add_argument('--step_between_frames', default=1, type=int,
                        help='Number of frames to skip between saved frames')
    parser.add_argument('--input_size', default=224, type=int,
                        help='Size of output images')
    parser.add_argument('--video_path', default='/shared/sets/datasets/vision/videos/walking_tour/Venice.mp4',
                        type=str, help='Path to input video file')
    parser.add_argument('--base_dir', default='/shared/sets/datasets/vision/videos/walking_tour/Frames/Venice/step_1',
                        type=str, help='Base directory for all outputs')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the first frame pair and optical flow')
    parser.add_argument('--raft_model', default='large', type=str, choices=['small', 'large'],
                        help='Choose RAFT model size (small or large)')
    return parser

def create_transform(input_size=224):
    """Create the same transformation pipeline as in main_pretrain.py"""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform

def save_frame(frame_tensor, save_path):
    """Save the transformed frame tensor as an image"""
    # Denormalize
    frame = frame_tensor * torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    frame = frame + torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    # Convert to PIL and save
    frame_pil = transforms.ToPILImage()(frame)
    frame_pil.save(save_path)

def save_flow(flow_tensor, save_path):
    """Save the optical flow tensor"""
    # Save flow as numpy array
    flow_np = flow_tensor.cpu().numpy()
    np.save(save_path, flow_np)

def visualize_flow_and_frames(output_dir, flow_dir, frame_num=0):
    """Visualize optical flow and corresponding frames"""
    import matplotlib.pyplot as plt
    from PIL import Image
    from torchvision.utils import flow_to_image
    
    # Load data
    flow = np.load(os.path.join(flow_dir, f'Venice_flow_{frame_num:06d}.npy'))
    flow = torch.from_numpy(flow)
    flow_img = flow_to_image(flow)
    frame1 = Image.open(os.path.join(output_dir, f'Venice_frame_{frame_num:06d}_1.jpg'))
    frame2 = Image.open(os.path.join(output_dir, f'Venice_frame_{frame_num:06d}_2.jpg'))
    
    # Create subplot
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(frame1)
    axs[0].set_title('Frame 1')
    axs[1].imshow(frame2)
    axs[1].set_title('Frame 2')
    axs[2].imshow(flow_img.permute(1, 2, 0))
    axs[2].set_title('Optical Flow')
    
    # Save plot
    plt.savefig('frame_flow_visualization.jpg')
    plt.close()

def process_video_segment(video_reader, frame_indices, transform, raft_model, device, split_name, args):
    """Process a segment of video frames"""
    # Create output directories for this split
    output_dir = os.path.join(args.base_dir, split_name, 'Image')
    flow_dir = os.path.join(args.base_dir, split_name, 'OpticalFlow')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(flow_dir).mkdir(parents=True, exist_ok=True)
    
    # Process frames in batches
    batch_size = 128  # Adjust based on your GPU memory
    for i in tqdm(range(0, len(frame_indices), batch_size), desc=f"Processing {split_name} split"):
        batch_indices = frame_indices[i:i + batch_size]
        
        # Read current and next frames in batch
        frames = video_reader.get_batch(batch_indices).asnumpy()
        next_indices = [idx + args.step_between_frames for idx in batch_indices]
        frames_next = video_reader.get_batch(next_indices).asnumpy()
        
        # Process each frame pair in the batch
        frames_transformed = []
        frames_next_transformed = []
        
        for j in range(len(batch_indices)):
            # Generate random seed for consistent transforms
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            
            # Apply same transformation to both frames
            torch.manual_seed(seed)
            frame_transformed = transform(frames[j])
            torch.manual_seed(seed)
            frame_next_transformed = transform(frames_next[j])
            
            frames_transformed.append(frame_transformed)
            frames_next_transformed.append(frame_next_transformed)
            
            # Save transformed frames
            frame_num = i + j
            frame_path = os.path.join(output_dir, f'Venice_frame_{frame_num:06d}_1.jpg')
            frame_next_path = os.path.join(output_dir, f'Venice_frame_{frame_num:06d}_2.jpg')
            save_frame(frame_transformed, frame_path)
            save_frame(frame_next_transformed, frame_next_path)
        
        # Stack frames for batch processing
        frames_batch = torch.stack(frames_transformed).to(device)
        frames_next_batch = torch.stack(frames_next_transformed).to(device)
        
        # Calculate optical flow for the batch
        with torch.no_grad():
            flow_predictions = raft_model(frames_batch, frames_next_batch)
            flows = flow_predictions[-1]  # Get last prediction
            
            # Save flows
            for j, flow in enumerate(flows):
                frame_num = i + j
                flow_path = os.path.join(flow_dir, f'Venice_flow_{frame_num:06d}.npy')
                save_flow(flow, flow_path)
    
    return len(frame_indices)

def main(args):
    if not args.visualize:
        # Create transform and model
        transform = create_transform(args.input_size)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize RAFT model based on argument
        if args.raft_model == 'large':
            print("Loading RAFT large model...")
            raft_model = torchvision.models.optical_flow.raft_large(
                weights=torchvision.models.optical_flow.Raft_Large_Weights.DEFAULT
            )
        else:
            print("Loading RAFT small model...")
            raft_model = torchvision.models.optical_flow.raft_small(
                weights=torchvision.models.optical_flow.Raft_Small_Weights.DEFAULT
            )
            
        raft_model = raft_model.to(device)
        raft_model.eval()

        import decord
        # Initialize video reader
        video_reader = decord.VideoReader(args.video_path, ctx=decord.cpu(0))
        total_frames = len(video_reader)
        
        # Calculate split points
        train_end = int(0.9 * total_frames)
        val_end = int(0.95 * total_frames)
        
        # Create frame indices for each split
        frame_indices = range(0, total_frames - args.step_between_frames)
        # frame_indices = range(0, total_frames - args.step_between_frames, args.step_between_frames)
        frame_indices = list(frame_indices)
        
        train_indices = [idx for idx in frame_indices if idx < train_end]
        val_indices = [idx for idx in frame_indices if train_end <= idx < val_end]
        test_indices = [idx for idx in frame_indices if idx >= val_end]
        
        print(f"Total frames: {total_frames}")
        print(f"Train frames: {len(train_indices)}")
        print(f"Val frames: {len(val_indices)}")
        print(f"Test frames: {len(test_indices)}")
        
        # Process each split
        train_processed = process_video_segment(video_reader, train_indices, transform, raft_model, device, 'train', args)
        val_processed = process_video_segment(video_reader, val_indices, transform, raft_model, device, 'val', args)
        test_processed = process_video_segment(video_reader, test_indices, transform, raft_model, device, 'test', args)
        
        print(f"\nProcessed frames:")
        print(f"Train: {train_processed} pairs")
        print(f"Val: {val_processed} pairs")
        print(f"Test: {test_processed} pairs")
        print(f"\nSaved to {args.base_dir}")
    else:
        # For visualization, use train split by default
        output_dir = os.path.join(args.base_dir, 'train', 'Image')
        flow_dir = os.path.join(args.base_dir, 'train', 'OpticalFlow')
        visualize_flow_and_frames(output_dir, flow_dir)

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
