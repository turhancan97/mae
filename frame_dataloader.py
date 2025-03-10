import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class FrameDataset(Dataset):
    """Dataset for loading frame pairs and their optical flow"""
    
    def __init__(self, root_dir, split='train', transform=None, frame_step=1):
        """
        Args:
            root_dir (str): Root directory containing 'Image' and 'OpticalFlow' folders
            split (str): Dataset split ('train' or 'val')
            transform (callable, optional): Optional transform to be applied on the frames
        """
        self.image_dir = os.path.join(root_dir, split, 'Image')
        self.flow_dir = os.path.join(root_dir, split, 'OpticalFlow')
        self.transform = transform
        if frame_step == 1:
            # Venice step_1 values (mean and std) [x, y]
            self.flow_mean = torch.tensor([-0.0025, -0.0107])
            self.flow_std = torch.tensor([2.0609, 0.5573])
            # Venice step_1 values (mean and std) [magnitude]
            self.flow_mag_mean = torch.tensor([1.1076])
            self.flow_mag_std = torch.tensor([1.8252])

        elif frame_step == 10:
            # Venice step_10 values (mean and std) [x, y]
            self.flow_mean = torch.tensor([-0.0189, -0.0450])
            self.flow_std = torch.tensor([21.0320, 5.2326])
            # Venice step_10 values (mean and std) [magnitude]
            self.flow_mag_mean = torch.tensor([11.9170])
            self.flow_mag_std = torch.tensor([18.1029])

        elif frame_step == 30:
            # Venice step_30 values (mean and std) [x, y]
            self.flow_mean = torch.tensor([-1.0678, 0.1349])
            self.flow_std = torch.tensor([49.8568, 12.3739])
            # Venice step_30 values (mean and std) [magnitude]
            self.flow_mag_mean = torch.tensor([32.7873])
            self.flow_mag_std = torch.tensor([39.5597])

        elif frame_step == 60:
            # Venice step_60 values (mean and std) [x, y]
            self.flow_mean = torch.tensor([-3.6112, 0.1128])
            self.flow_std = torch.tensor([73.9313, 19.0465])
            # Venice step_60 values (mean and std) [magnitude]
            self.flow_mag_mean = torch.tensor([54.1642])
            self.flow_mag_std = torch.tensor([53.9249])
            
        else:
            raise ValueError(f"Invalid frame step: {frame_step}")
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
        frame_path = os.path.join(self.image_dir, f'Venice_frame_{frame_num:06d}_1.jpg')
        flow_path = os.path.join(self.flow_dir, f'Venice_flow_{frame_num:06d}.npy')
        
        # Read images
        frame = Image.open(frame_path).convert('RGB')
        
        # Apply transforms if specified
        if self.transform is not None:
            frame = self.transform(frame)    
        else:
            # Convert to tensor if no transform specified
            to_tensor = transforms.ToTensor()
            frame = to_tensor(frame)
        
        # Load optical flow
        flow = torch.from_numpy(np.load(flow_path))

        if True:
            # calculate magnitude of flow vectors
            flow = torch.norm(flow, p=2, dim=0, keepdim=True)  # Shape: (1, 224, 224)
            # normalize magnitude with mean and std
            flow = (flow - self.flow_mag_mean.view(1, 1, 1)) / self.flow_mag_std.view(1, 1, 1)
        else:
            # normalize optical flow
            flow = (flow - self.flow_mean.view(2, 1, 1)) / self.flow_std.view(2, 1, 1) 
            # Calculate magnitude of flow vectors
            # torch.norm calculates the L2 norm along dimension 0 (channel dimension)
            flow = torch.norm(flow, p=2, dim=0, keepdim=True)  # Shape: (1, 224, 224)
        
        return frame, flow
