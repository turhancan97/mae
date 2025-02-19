from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets import VisionDataset
import torchvision
import os
from PIL import Image
import torch
import decord
import numpy as np
import random


class DecordInit(object):

    def __init__(self, num_threads=1, **kwargs):
        self.num_threads = num_threads
        self.ctx = decord.cpu(0)
        self.kwargs = kwargs
        
    def __call__(self, filename):
        
        reader = decord.VideoReader(filename,
                                    ctx=self.ctx,
                                    num_threads=self.num_threads)
        return reader

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'num_threads={self.num_threads})')
        return repr_str

class WTDatasetOneVideo(torch.utils.data.Dataset):
    

    def __init__(self,
                 video_path,
                 step_between_clips,
                 transform=None):
        
        self.path = video_path

        self.transform = transform
        self.step_between_clips = step_between_clips
        self.v_decoder = DecordInit()
        v_reader = self.v_decoder(self.path)
        total_frames = len(v_reader)
        
        self.total_frames = total_frames 

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        while True:
            try:
               
                v_reader = self.v_decoder(self.path)
                total_frames = len(v_reader)
                
                # Sampling video frames
                frame_indice = np.array([index * self.step_between_clips], dtype=int)

                # Get video frames
                video = v_reader.get_batch(frame_indice).asnumpy()
                del v_reader
                break
            except Exception as e:
                print(e)
                
        with torch.no_grad():
            video = video.squeeze(0)
            if self.transform is not None:
                video = self.transform(video)
                
        return video, 0


    def __len__(self):
        return self.total_frames // self.step_between_clips
    
    def get_video_length(self):
        return self.total_frames
