import torch
import itertools

def get_optimizer(config, model):
    if config.training.optimizer.name == "AdamW":
        return torch.optim.AdamW(
            itertools.chain(model.unet.parameters(), model.image_proj_model.parameters()),
            lr=config.training.optimizer.learning_rate,
            betas=config.training.optimizer.adam_beta,
            weight_decay=config.training.optimizer.weight_decay,
            eps=config.training.optimizer.adam_epsilon,
        )
    if config.training.optimizer.name == "AdamWGating":
        base_lr = config.training.optimizer.learning_rate
        gating_multiplier = 100.0 

        gating_params, regular_params = [], []

        for module in (model.unet, model.image_proj_model):
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                if "gating_param" in name:
                    gating_params.append(param)
                else:
                    regular_params.append(param)

        param_groups = [
            {
                'params': regular_params, 
                'lr': base_lr
            },
            {
                'params': gating_params, 
                'lr': base_lr * gating_multiplier
            }
        ]
        
        # if len(gating_params) > 0:
        #     print(f"[Optimizer] Found {len(gating_params)} gating parameters. Setting LR to {base_lr * gating_multiplier} (x{gating_multiplier})")
        # else:
        #     print("[Optimizer] Warning: No 'gating_param' found in trainable parameters!")

        return torch.optim.AdamW(
            param_groups,
            lr=base_lr,
            betas=config.training.optimizer.adam_beta,
            weight_decay=config.training.optimizer.weight_decay,
            eps=config.training.optimizer.adam_epsilon,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.training.optimizer.name}. Supported optimizers: AdamW.")