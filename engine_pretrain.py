# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
from einops import rearrange
from torchvision.utils import flow_to_image

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    # if log_writer is not None:
    #     print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples_1, samples_2, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples_1 = samples_1.to(device, non_blocking=True)
        samples_2 = samples_2.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, pred, mask = model(samples_1, samples_2, targets, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        # if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
        #     """ We use epoch_1000x as the x-axis in tensorboard.
        #     This calibrates different curves when batch size changes.
        #     """
        #     epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
        #     log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
        #     log_writer.add_scalar('lr', lr, epoch_1000x)
        
        if log_writer is not None and args.log_wandb and (data_iter_step + 1) % accum_iter == 0 and misc.is_main_process():
            log_writer.update(
                {
                    'loss': loss_value_reduce,
                    'lr': lr,
                }
            )
            log_writer.set_step()

    # log the last batch
    if log_writer is not None and args.log_wandb and misc.is_main_process() and epoch % 5 == 0:
        # log_writer.log_image(samples, 'samples')
        samples_1 = samples_1[0:8]
        samples_2 = samples_2[0:8]
        samples_1 = samples_1.detach().cpu()
        samples_2 = samples_2.detach().cpu()

        targets = targets[0:8]
        targets = targets.detach().cpu()

        pred = model.module.flow_unpatchify(pred[0:8])
        pred = pred.detach().cpu()
        mask_input = mask[0:8].detach()
        mask_output = mask[0:8].detach()

        mask_input = mask_input.unsqueeze(-1).repeat(1, 1, model.module.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
        mask_input = model.module.unpatchify(mask_input)  # 1 is removing, 0 is keeping
        mask_input = mask_input.detach().cpu()

        mask_output = mask_output.unsqueeze(-1).repeat(1, 1, model.module.patch_embed.patch_size[0]**2 * model.module.flow_in_chans)  # (N, H*W, p*p*flow_in_chans)
        mask_output = model.module.flow_unpatchify(mask_output)  # 1 is removing, 0 is keeping
        mask_output = mask_output.detach().cpu()

        im_masked = samples_2 * (1 - mask_input)
        target_paste = targets * (1 - mask_output) + pred * mask_output

        # concat dummy channel to match the shape of the target
        target_paste = torch.cat([target_paste, torch.zeros_like(target_paste)], dim=1)
        targets = torch.cat([targets, torch.zeros_like(targets)], dim=1)

        target_paste = flow_to_image(target_paste)
        target = flow_to_image(targets)

        log_writer.log_image(samples_1, f"original input Past Frame")
        log_writer.log_image(samples_2, f"original input Future Frame")
        log_writer.log_image(target, f"original target Optical Flow")
        log_writer.log_image(im_masked, f"masked input Future Frame")
        log_writer.log_image(target_paste, f"reconstructed output Optical Flow")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}