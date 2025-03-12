# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.distributed as dist
from torch._six import inf


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('SiamMAE-checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.output_dir, tag="SiamMAE-checkpoint-%s" % epoch_name, client_state=client_state)


def convert_qkv_weights(state_dict):
    """Convert separate Q, K, V weights into combined QKV weights.
    
    Args:
        state_dict (dict): State dict from pretrained model
        
    Returns:
        dict: Converted state dict with combined QKV weights
    """
    new_state_dict = {}
    
    # Copy non-QKV weights directly
    for k, v in state_dict.items():
        if not any(x in k for x in ['.q.', '.k.', '.v.']):
            new_state_dict[k] = v
            
    # Combine Q, K, V weights and biases for each block
    for block_idx in range(12):  # Assuming 12 blocks
        prefix = f'blocks.{block_idx}.attn.'
        
        # Get Q, K, V weights
        q_weight = state_dict[prefix + 'q.weight']
        k_weight = state_dict[prefix + 'k.weight']
        v_weight = state_dict[prefix + 'v.weight']
        
        # Get Q, K, V biases
        q_bias = state_dict[prefix + 'q.bias']
        k_bias = state_dict[prefix + 'k.bias']
        v_bias = state_dict[prefix + 'v.bias']
        
        # Concatenate weights and biases
        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
        
        # Store combined weights
        new_state_dict[prefix + 'qkv.weight'] = qkv_weight
        new_state_dict[prefix + 'qkv.bias'] = qkv_bias
    
    return new_state_dict


def convert_mmcv_state_dict(state_dict):
    """Convert MMCV model state dict keys to match the target model structure.
    
    Args:
        state_dict (dict): State dict from MMCV model
        
    Returns:
        dict: Converted state dict with matching keys
    """
    new_state_dict = {}
    
    # Define keys to skip
    skip_keys = [
        'backbone.mask_token',
        'target_generator.weight_x',
        'target_generator.weight_y', 
        'target_generator.gaussian_kernel',
        'neck.fc.weight',
        'neck.fc.bias'
    ]
    
    key_mapping = {
        'backbone.cls_token': 'cls_token',
        'backbone.pos_embed': 'pos_embed',
        'backbone.patch_embed.projection': 'patch_embed.proj',
        'backbone.ln1': 'norm',
    }
    
    # Handle transformer blocks mapping
    block_mapping = {
        'backbone.layers': 'blocks',
        'ln1': 'norm1',
        'ln2': 'norm2',
        'ffn.layers.0.0': 'mlp.fc1',
        'ffn.layers.1': 'mlp.fc2'
    }
    
    for old_key, param in state_dict.items():
        # Skip unwanted keys
        if any(skip_key in old_key for skip_key in skip_keys):
            continue
            
        new_key = old_key
        
        # Apply direct key mappings
        for old_pattern, new_pattern in key_mapping.items():
            if old_key.startswith(old_pattern):
                new_key = old_key.replace(old_pattern, new_pattern)
                break
                
        # Handle transformer blocks
        if 'backbone.layers' in old_key:
            new_key = old_key
            for old_pattern, new_pattern in block_mapping.items():
                new_key = new_key.replace(old_pattern, new_pattern)
            new_key = new_key.replace('backbone.', '')
            
        new_state_dict[new_key] = param
        
    return new_state_dict


def load_model(args, model_without_ddp, optimizer, loss_scaler, mmcv=True):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
            
        if mmcv:
            # Convert MMCV state dict keys
            model_state = convert_mmcv_state_dict(checkpoint['state_dict'])
        else:
            # Convert QKV weights if needed
            if 'blocks.0.attn.q.weight' in checkpoint['model_state']:
                checkpoint['model_state'] = convert_qkv_weights(checkpoint['model_state'])
                
            # Filter out hog-related keys and mask_token
            model_state = {k: v for k, v in checkpoint['model_state'].items() 
                        if 'hog' not in k.lower() and 'mask_token' not in k}
            
        msg = model_without_ddp.load_state_dict(model_state, strict=False)
        print("Resume checkpoint %s" % args.resume)
        print("Missing keys:", msg.missing_keys)
        print("Unexpected keys:", msg.unexpected_keys)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


def analyze_flow_quality(flow_dir, num_samples=100, output_dir=None, plot=False):
    """
    Analyze the quality of optical flow samples to determine appropriate filtering thresholds
    
    Args:
        flow_dir (str): Directory containing optical flow files
        num_samples (int): Number of samples to analyze
        plot (bool): Whether to create visualization plots
    
    Returns:
        dict: Dictionary with flow statistics
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from tqdm import tqdm
    
    if not os.path.exists(flow_dir):
        print(f"Flow directory {flow_dir} does not exist")
        return {}
    
    flow_files = sorted([f for f in os.listdir(flow_dir) if f.endswith('.npy')])
    if len(flow_files) == 0:
        print(f"No flow files found in {flow_dir}")
        return {}
    
    # Select a subset of samples if requested
    if num_samples < len(flow_files):
        import random
        flow_files = random.sample(flow_files, num_samples)
    
    # Collect statistics
    magnitudes = []
    variances = []
    flow_coverage = []
    
    for file in tqdm(flow_files, desc="Analyzing flow quality"):
        try:
            flow = np.load(os.path.join(flow_dir, file))
            
            # Calculate flow magnitude
            flow_magnitude = np.sqrt(flow[0]**2 + flow[1]**2)
            avg_magnitude = np.mean(flow_magnitude)
            magnitudes.append(avg_magnitude)
            
            # Calculate flow variance
            flow_variance = np.var(flow)
            variances.append(flow_variance)
            
            # Calculate percentage of pixels with significant flow
            significant_flow = (flow_magnitude > 1.0).sum() / flow_magnitude.size
            flow_coverage.append(significant_flow)
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    if plot:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.hist(magnitudes, bins=30)
        plt.axvline(np.median(magnitudes), color='r', linestyle='--')
        plt.title(f'Flow Magnitude Distribution\nMedian: {np.median(magnitudes):.2f}')
        plt.xlabel('Average Magnitude')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 3, 2)
        plt.hist(variances, bins=30)
        plt.axvline(np.median(variances), color='r', linestyle='--')
        plt.title(f'Flow Variance Distribution\nMedian: {np.median(variances):.2f}')
        plt.xlabel('Flow Variance')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 3, 3)
        plt.hist(flow_coverage, bins=30)
        plt.axvline(np.median(flow_coverage), color='r', linestyle='--')
        plt.title(f'Flow Coverage Distribution\nMedian: {np.median(flow_coverage):.2f}')
        plt.xlabel('Percentage of Significant Flow')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'flow_quality_analysis.png'))
        plt.close()
    
    stats = {
        'mag_mean': np.mean(magnitudes),
        'mag_median': np.median(magnitudes),
        'mag_std': np.std(magnitudes),
        'var_mean': np.mean(variances),
        'var_median': np.median(variances),
        'var_std': np.std(variances),
        'coverage_mean': np.mean(flow_coverage),
        'coverage_median': np.median(flow_coverage),
        'coverage_std': np.std(flow_coverage),
    }
    
    # Suggest thresholds
    stats['suggested_mag_threshold'] = max(np.mean(magnitudes) - 0.5 * np.std(magnitudes), 0.5)
    stats['suggested_var_threshold'] = max(np.mean(variances) - 0.5 * np.std(variances), 0.5)
    
    return stats


def visualize_flow_samples(image_dir, flow_dir, output_dir='flow_samples', num_samples=10):
    """
    Visualize image and corresponding optical flow samples to help assess quality
    
    Args:
        image_dir (str): Directory containing image files
        flow_dir (str): Directory containing optical flow files
        output_dir (str): Directory to save visualizations
        num_samples (int): Number of samples to visualize
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from PIL import Image
    from torchvision.utils import flow_to_image
    import torch
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all flow files
    flow_files = sorted([f for f in os.listdir(flow_dir) if f.endswith('.npy')])
    
    if len(flow_files) == 0:
        print(f"No flow files found in {flow_dir}")
        return
    
    # Select a subset of samples
    if num_samples < len(flow_files):
        import random
        selected_samples = random.sample(range(len(flow_files)), num_samples)
        flow_files = [flow_files[i] for i in selected_samples]
    
    for flow_file in flow_files:
        try:
            # Extract frame number from flow filename
            frame_num = int(flow_file.split('_')[2].split('.')[0])
            
            # Find corresponding image file
            image_file = f'Venice_frame_{frame_num:06d}_1.jpg'
            image_path = os.path.join(image_dir, image_file)
            
            if not os.path.exists(image_path):
                print(f"Image file {image_path} not found")
                continue
            
            # Load image and flow
            image = np.array(Image.open(image_path))
            flow = np.load(os.path.join(flow_dir, flow_file))
            
            # Convert flow to tensor and then to image
            flow_tensor = torch.from_numpy(flow)
            flow_img = flow_to_image(flow_tensor)
            flow_img = flow_img.permute(1, 2, 0).numpy()
            
            # Calculate flow statistics
            flow_magnitude = np.sqrt(flow[0]**2 + flow[1]**2)
            avg_magnitude = np.mean(flow_magnitude)
            flow_variance = np.var(flow)
            significant_flow = (flow_magnitude > 1.0).sum() / flow_magnitude.size
            
            # Create visualization
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(flow_img)
            plt.title('Optical Flow')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(flow_magnitude, cmap='hot')
            plt.colorbar()
            plt.title(f'Flow Magnitude\nAvg: {avg_magnitude:.2f}, Var: {flow_variance:.2f}')
            plt.axis('off')
            
            plt.suptitle(f"Frame {frame_num} - Significant Flow Coverage: {significant_flow*100:.1f}%")
            plt.tight_layout()
            
            # Save visualization
            plt.savefig(os.path.join(output_dir, f'flow_sample_{frame_num:06d}.png'))
            plt.close()
            
        except Exception as e:
            print(f"Error processing {flow_file}: {e}")