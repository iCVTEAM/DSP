import diffusers

def get_scheduler(config, optimizer, accelerator, max_train_steps):
    return diffusers.optimization.get_scheduler(
        config.training.scheduler.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.training.scheduler.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )