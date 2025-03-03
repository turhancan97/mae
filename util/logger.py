import wandb
import torch

class WandbLogger(object):
    def __init__(self, args):
        wandb.init(
            dir=args.log_dir,
            config=args,
            entity=args.wandb_entity,
            project=args.wandb_project,
            group=getattr(args, 'wandb_group', None),
            name=getattr(args, 'wandb_run_name', None)
        )

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, metrics):
        log_dict = dict()
        for k, v in metrics.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            log_dict[k] = v

        wandb.log(log_dict, step=self.step)
    
    def log_image(self, image, name):
        wandb.log({name: wandb.Image(image, caption=name)}, step=self.step)

    def flush(self):
        pass