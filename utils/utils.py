from collections import OrderedDict
from typing import Any, Tuple
import os, torch

class Dict(OrderedDict):
    """Base ModelOutput class fixing the output type from the models. This class is inspired from
    the ``ModelOutput`` class from hugginface transformers library"""

    def __getitem__(self, k):
        if isinstance(k, str):
            self_dict = {k: v for (k, v) in self.items()}
            return self_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, Dict):
            value = Dict(value)
        elif isinstance(value, list):
            value = [Dict(v) if isinstance(v, dict) and not isinstance(v, Dict) else v for v in value]
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())
    
    def to_dict(self) -> dict:
        result = {}
        for k, v in self.items():
            if isinstance(v, Dict):
                result[k] = v.to_dict()
            elif isinstance(v, list):
                result[k] = [i.to_dict() if isinstance(i, Dict) else i for i in v]
            else:
                result[k] = v
        return result

    def __repr__(self):
        return repr(self.to_dict())

    def __str__(self):
        return str(self.to_dict())
    

def get_ckpt_path(config, ckpt_steps=None):
    parts = [config.ckpt_dir, config.task_name, config.phase]
    if config.phase == 'novel':
        parts.append(f'run-{config.run}' if config.run is not None else "run-default")
        parts.append(f'{config.dataset.novel_settings.k_shot}-shot')
        parts.append(f'shuffle_seed-{config.dataset.novel_settings.shuffle_seed}')
    if ckpt_steps is not None:
        parts.append(f'checkpoint-{ckpt_steps}')
    os.makedirs(os.path.join(*parts), exist_ok=True)
    return os.path.join(*parts)


def get_output_path(config, ckpt_steps=None):
    parts = [config.inference.output_dir, config.task_name, config.phase]
    if config.phase == 'novel':
        parts.append(f'run-{config.run}' if config.run is not None else "run-default")
        parts.append(f'{config.dataset.novel_settings.k_shot}-shot')
        parts.append(f'shuffle_seed-{config.dataset.novel_settings.shuffle_seed}')
    if ckpt_steps is not None:
        parts.append(f'checkpoint-{ckpt_steps}')
    os.makedirs(os.path.join(*parts), exist_ok=True)
    return os.path.join(*parts)


def manual_average_gradients(model, accelerator):
    if accelerator.num_processes > 1:
        for param in model.parameters():
            if param.grad is not None and param.requires_grad:
                torch.distributed.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
                param.grad.data /= accelerator.num_processes