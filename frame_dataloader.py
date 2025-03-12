import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class FrameDataset(Dataset):
    """Dataset for loading frame pairs"""
    
    def __init__(self, root_dir, split='train', transform=None, frame_step=1):
        """
        Args:
            root_dir (str): Root directory containing 'Image' and 'OpticalFlow' folders
            split (str): Dataset split ('train' or 'val')
            transform (callable, optional): Optional transform to be applied on the frames
        """
        self.image_dir = os.path.join(root_dir, split, 'Image')
        self.transform = transform
        # Get all frame pairs (assuming they're numbered sequentially)
        self.frame_pairs = []
        for filename in sorted(os.listdir(self.image_dir)):
            if filename.endswith('_1.jpg'):  # First frame of each pair
                frame_num = int(filename.split('_')[2])  # Extract frame number
                self.frame_pairs.append(frame_num)
    
    def __len__(self):
        return len(self.frame_pairs)
    
    def __getitem__(self, idx):
        frame_num = self.frame_pairs[idx]
        
        # Load frame pair
        frame_1_path = os.path.join(self.image_dir, f'Venice_frame_{frame_num:06d}_1.jpg')
        frame_2_path = os.path.join(self.image_dir, f'Venice_frame_{frame_num:06d}_2.jpg')
        
        # Read images
        frame_1 = Image.open(frame_1_path).convert('RGB')
        frame_2 = Image.open(frame_2_path).convert('RGB')
        
        # Apply transforms if specified
        if self.transform is not None:
            frame_1 = self.transform(frame_1)    
            frame_2 = self.transform(frame_2)
        else:
            # Convert to tensor if no transform specified
            to_tensor = transforms.ToTensor()
            frame_1 = to_tensor(frame_1)
            frame_2 = to_tensor(frame_2)
        
        return frame_1, frame_2
