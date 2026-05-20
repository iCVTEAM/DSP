import accelerate, datasets, diffusers, transformers, safetensors
from packaging import version
from itertools import chain
from tqdm.auto import tqdm
import torch, math, yaml, os

from utils import load_config, set_logger, get_optimizer, get_scheduler, get_ckpt_path, Dict
from datamodules import Loader
import models


def set_accelerator(config):
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=config.accelerator.gradient_accumulation_steps,
        mixed_precision=config.accelerator.mixed_precision,
        log_with=config.accelerator.report_to,
        project_config=accelerate.utils.ProjectConfiguration(project_dir=get_ckpt_path(config)),
    )
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if config.seed is not None:
        accelerate.utils.set_seed(config.seed)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))
                    weights.pop() # make sure to pop weight so that corresponding model is not saved again

        def load_model_hook(models, input_dir):
            for i in range(len(models)):
                model = models.pop() # pop models so that they are not loaded again
                state_dict = safetensors.torch.load_file(os.path.join(input_dir, 'unet/diffusion_pytorch_model.safetensors'))
                model.load_state_dict(state_dict)

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    torch.backends.cuda.matmul.allow_tf32 = True
    return accelerator


def create_model(config):
    core_module = getattr(models, config.model.name)
    noise_scheduler = diffusers.DDPMScheduler.from_pretrained(config.model.sd15_weight_path, subfolder="scheduler")
    tokenizer = transformers.CLIPTokenizer.from_pretrained(config.model.sd15_weight_path, subfolder="tokenizer")

    # returns either a context list that includes one that will disable zero.Init or an empty context list
    def deepspeed_zero_init_disabled_context_manager():
        deepspeed_plugin = accelerate.state.AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        return [] if deepspeed_plugin is None else [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    with transformers.utils.ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = transformers.CLIPTextModel.from_pretrained(config.model.sd15_weight_path, subfolder="text_encoder")
        vae = diffusers.AutoencoderKL.from_pretrained(config.model.sd15_weight_path, subfolder="vae")
        image_encoder = core_module.FrozenDinoV2Encoder(config.model.dinov2_vitl14_path)
        text_projector = transformers.CLIPTextModelWithProjection.from_pretrained(config.model.clip_weight_path)
        image_processor = transformers.AutoProcessor.from_pretrained(config.model.clip_weight_path)

    unet = diffusers.UNet2DConditionModel.from_pretrained(config.model.sd15_weight_path, subfolder="unet")
    core_module.set_processors(unet, phase=config.phase)
    custom_layers = diffusers.loaders.AttnProcsLayers(unet.attn_processors)   # collect training layers

    image_proj_model = core_module.SerialSampler(config, image_processor, image_encoder, dim=config.model.image_proj_model.dim, depth=config.model.image_proj_model.depth, 
                                                 dim_head=config.model.image_proj_model.dim_head, num_queries=config.model.image_proj_model.num_queries, 
                                                 embedding_dim=image_encoder.model.embed_dim, output_dim=unet.config.cross_attention_dim, ff_mult=config.model.image_proj_model.ff_mult)
    exemplar_pool = core_module.ExemplarPool(config.model.exemplar_pool.data_embeds_dict_path, config.model.exemplar_pool.exemplar_pool_path, image_processor)
    cam_generator = core_module.CAMGenerator(categories=config.dataset.categories.all, clip_path=config.model.clip_vit_b16_path)

    if config.phase == 'novel':
        base_weight_path = os.path.join(config.ckpt_dir, config.task_name, 'base', f'checkpoint-{config.training.base_ckpt_steps}')
        try:
            custom_layers.load_state_dict({k: v for k, v in safetensors.torch.load_file(os.path.join(base_weight_path, 'unet/diffusion_pytorch_model.safetensors')).items() if '.processor' in k or '.self_attn' in k}, strict=False)
        except:
            custom_layers.load_state_dict(torch.load(os.path.join(base_weight_path, 'CustomLayers.pth')), strict=False)
        image_proj_model.load_state_dict(torch.load(os.path.join(base_weight_path, 'ImageProjModel.pth')), strict=False)
        for name, param in image_proj_model.named_parameters():
            param.requires_grad = True if 'sample_aggregator' in name else False

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_projector.requires_grad_(False)
    image_encoder.requires_grad_(False)
    # for name, param in unet.named_parameters():        
    #     param.requires_grad = True if 'attn2.processor' in name else False
    unet.requires_grad_(False)
    custom_layers.requires_grad_(True)

    return Dict(unet=unet, text_encoder=text_encoder, tokenizer=tokenizer, vae=vae, noise_scheduler=noise_scheduler, 
                image_encoder=image_encoder, image_processor=image_processor, text_projector=text_projector, 
                image_proj_model=image_proj_model, custom_layers=custom_layers, exemplar_pool=exemplar_pool, 
                cam_generator=cam_generator, core_module=core_module)


def create_dataloader(config, accelerator, image_processor, logger=None):
    dataloader_builder = Loader(config, image_processor, split='train', logger=logger)
    if config.phase == 'novel' and accelerator.is_local_main_process:
        dataloader_builder.dump_novel_sample_dict()
    return dataloader_builder()


def calculate_training_schedule(config, dataloader):
    max_train_steps, num_train_epochs = config.training.max_train_steps, config.training.num_train_epochs
    if max_train_steps is None and num_train_epochs is None:
        raise ValueError("Invalid configuration: You must provide either `max_train_steps` or `num_train_epochs`.")
    steps_per_epoch = math.ceil(len(dataloader) / config.accelerator.gradient_accumulation_steps)
    if max_train_steps is None:
        max_train_steps = num_train_epochs * steps_per_epoch
    num_train_epochs = math.ceil(max_train_steps / steps_per_epoch)
    return steps_per_epoch, max_train_steps, num_train_epochs


def prepare_models(accelerator, model, optimizer, dataloader, scheduler):
    model.unet, optimizer, dataloader, scheduler = accelerator.prepare(model.unet, optimizer, dataloader, scheduler)
    weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(accelerator.mixed_precision, torch.float32)
    for attr in ["text_encoder", "text_projector", "vae", "image_encoder", "image_proj_model", "exemplar_pool", "cam_generator"]:
        getattr(model, attr).to(accelerator.device, dtype=weight_dtype)
    return model, optimizer, dataloader, scheduler


class Trainer:
    def __init__(self, config, accelerator, logger, model, dataloader, optimizer, scheduler, steps_per_epoch, max_train_steps, num_train_epochs):
        self.config, self.accelerator, self.logger, self.model, self.dataloader, self.optimizer, self.scheduler = \
            config, accelerator, logger, model, dataloader, optimizer, scheduler
        self.steps_per_epoch, self.max_train_steps, self.num_train_epochs = steps_per_epoch, max_train_steps, num_train_epochs
        self.image_column, self.caption_column, self.bbox_column, self.obbox_column = config.dataset.column_names
        self.dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(accelerator.mixed_precision, torch.float32)
        self.batch_size_per_device = self.config.training.batch_size
        self.total_batch_size = self.config.training.batch_size * self.accelerator.num_processes * self.config.accelerator.gradient_accumulation_steps
        self.log_before_training()
        self._trainable_params()

    def _trainable_params(self):
        self.trainable_params = []
        for key, submodule in self.model.items():
            if hasattr(submodule, 'parameters'):
                self.trainable_params.extend(filter(lambda p: p.requires_grad, submodule.parameters()))
        total_trainable = sum(p.numel() for p in self.trainable_params)
        self.logger.info(f"====== Total Trainable Params: {total_trainable / 1e6:.2f} M ======")

    def log_before_training(self):
        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {len(self.dataloader.dataset)}")
        self.logger.info(f"  Num Epochs = {self.num_train_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {self.batch_size_per_device}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.total_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {self.config.accelerator.gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps = {self.max_train_steps}")
        yaml.dump(self.config.to_dict(), open(os.path.join(get_ckpt_path(self.config), 'config.yaml'), 'w', encoding="utf-8"))

    def _latents(self, image):
        # vae.encode(): batch[self.image_column]:{.shape: [B, 3, 512, 512]} -> latent_dist.mean:{.shape: [B, 4, 64, 64]}, latent_dist.logvar:{.shape: [B, 4, 64, 64]}
        # type(latent_dist): diffusers.models.autoencoders.vae.DiagonalGaussianDistribution; latent_dist.sample(): -> mean + std:{torch.exp(0.5 * logvar)} * randn_tensor(mean.shape)
        latents = self.model.vae.encode(image.to(self.dtype)).latent_dist.sample()
        latents = latents * self.model.vae.config.scaling_factor  # scaling_factor: 0.18215
        return latents

    def _noise(self, shape):
        noise = torch.randn(shape, device=self.accelerator.device, dtype=self.dtype)
        if self.config.training.noise_offset:
            noise += self.config.training.noise_offset * torch.randn((shape[0], shape[1], 1, 1), device=self.accelerator.device, dtype=self.dtype)
        if self.config.training.input_perturbation:
            perturbed_noise = noise + self.config.training.input_perturbation * torch.randn_like(noise)
            return noise, perturbed_noise
        return noise, noise

    def _timesteps(self):
        timesteps = torch.randint(0, self.model.noise_scheduler.config.num_train_timesteps, (self.batch_size_per_device,), device=self.accelerator.device)
        timesteps = timesteps.long()
        return timesteps

    def _target(self, latents, noise, timesteps):
        if self.config.training.prediction_type is not None:
            self.model.noise_scheduler.register_to_config(prediction_type=self.config.training.prediction_type)

        if self.model.noise_scheduler.config.prediction_type == "epsilon":
            return noise
        elif self.model.noise_scheduler.config.prediction_type == "v_prediction":
            return self.model.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.model.noise_scheduler.config.prediction_type}")

    def _encode_prompt(self, prompts):
        group_size = len(prompts[0]) # 16
        flattened_prompts = list(chain.from_iterable(prompts))
        # len(flattened_prompts): B * 16 -> text_input_ids.shape [B * 16, 77], untruncated_ids.shapeB * 16, 36(longest text)]
        text_inputs = self.model.tokenizer(flattened_prompts, padding="max_length", max_length=self.model.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_input_ids, attention_mask = text_inputs.input_ids.to(self.accelerator.device), text_inputs.attention_mask.to(self.accelerator.device) # Original Version as None?
        # untruncated_ids = self.model.tokenizer(flattened_prompts, padding="longest", return_tensors="pt").input_ids
        # if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids.to(self.accelerator.device)):
        #     removed_text = self.model.tokenizer.batch_decode(untruncated_ids[:, self.model.tokenizer.model_max_length - 1: -1])
        #     self.logger.warning("The following part of your input was truncated because CLIP can only handle sequences up to"f" {self.model.tokenizer.model_max_length} tokens: {removed_text}")
        text_encoder_output = self.model.text_encoder(text_input_ids, attention_mask=None)
        per_token_embeds, pooler_embeds = text_encoder_output.last_hidden_state, text_encoder_output.pooler_output # prompt_embeds.shape [B * 16, 77, 768], embeds_pooler.shape [B * 16, 768]
        text_embeds = self.model.text_projector(text_input_ids, attention_mask=None).text_embeds # .shape [B * 16, 768]
        per_token_embeds = per_token_embeds.view(self.batch_size_per_device, group_size, self.model.tokenizer.model_max_length, -1) # per_token_embeds.shape [B, 16, 77, 768]
        pooler_embeds = pooler_embeds.view(self.batch_size_per_device, group_size, -1).unsqueeze(2) # embeds_pooler.shape [B, 16, 1, 768]
        text_embeds = text_embeds.view(self.batch_size_per_device, group_size, -1) # text_embeds.shape [B, 16, 768]
        return per_token_embeds, pooler_embeds, text_embeds

    def _cross_attention_kwargs(self, batch, pooler_embeds, text_embeds):
        captions = batch[self.caption_column]
        bboxes, obboxes, height, width = batch[self.bbox_column], batch[self.obbox_column], self.config.dataset.resolution, self.config.dataset.resolution
        instances_imgs, exemplar_imgs = batch["instances"], self.model.exemplar_pool.get_similar_exemplars(text_embeds[:, 0], topk=5) # [B, num_inst, k, 3, 224, 224], [B, topk, 3, 224, 224]
        num_instances, instance_group_size, exemplar_group_size = instances_imgs.shape[1], instances_imgs.shape[2], exemplar_imgs.shape[1] # num_inst = 15, topk = 1
        instance_imgs, exemplar_imgs = instances_imgs.flatten(0, 2), exemplar_imgs.flatten(0, 1) # [B * num_inst, 3, 224, 224], [B * topk, 3, 224, 224]
        with torch.no_grad():
            instance_features = self.model.image_encoder(instance_imgs) # [B * num_inst, 257, 1024]
            instance_features = instance_features.view(self.batch_size_per_device, num_instances, instance_group_size, instance_features.shape[-2], instance_features.shape[-1])
            exemplar_features = self.model.image_encoder(exemplar_imgs) # [B * topk, 257, 1024]
            exemplar_features = exemplar_features.view(self.batch_size_per_device, 1, exemplar_group_size, exemplar_features.shape[-2], exemplar_features.shape[-1])
        resampled_instance_features_list, resampled_exemplar_features_list = [], []
        for i in range(self.batch_size_per_device):
            resampled_instance_features, resampled_exemplar_features = self.model.image_proj_model(instance_features[i], [obboxes[i]], exemplar_features[i], [captions[i][1:]])
            resampled_instance_features_list.append(resampled_instance_features), resampled_exemplar_features_list.append(resampled_exemplar_features)
        resampled_instance_features, resampled_exemplar_features = torch.stack(resampled_instance_features_list), torch.stack(resampled_exemplar_features_list) # [B, num_inst, 16, 768], [B, topk, 16, 768]
        guidance_masks, supplement_mask, in_box = self.model.core_module.get_masks(obboxes, height, width, self.accelerator.device)
        sigmoid_values = self.model.core_module.get_sigmoid(bboxes, height, width, self.accelerator.device)
        if self.config.phase == 'novel':
            prototypes = torch.stack([self.model.image_proj_model.prototype_bank.get_prototypes(captions[i][1:]) for i in range(len(captions))])
        else:
            prototypes = None
        return {'bboxes': bboxes, 'obboxes': obboxes, 'height': height, 'width': width, 'embeds_pooler': pooler_embeds, 'ref_features': (resampled_instance_features, resampled_exemplar_features), 'prototypes': prototypes,
                'guidance_masks': guidance_masks, 'supplement_mask': supplement_mask, 'sigmoid_values': sigmoid_values, 'in_box': in_box}

    def _split_kwargs(self, cross_attention_kwargs, i):
        return {'bboxes': [cross_attention_kwargs['bboxes'][i]], 'obboxes': [cross_attention_kwargs['obboxes'][i]], 
                'height': cross_attention_kwargs['height'], 'width': cross_attention_kwargs['width'], 
                'embeds_pooler': cross_attention_kwargs['embeds_pooler'][i], 
                'prototypes': cross_attention_kwargs['prototypes'][i] if cross_attention_kwargs['prototypes'] is not None else None,
                'ref_features': (cross_attention_kwargs['ref_features'][0][i], cross_attention_kwargs['ref_features'][1][i]),
                'guidance_masks': cross_attention_kwargs['guidance_masks'], 'supplement_mask': cross_attention_kwargs['supplement_mask'], 
                'sigmoid_values': cross_attention_kwargs['sigmoid_values'], 'in_box': cross_attention_kwargs['in_box']}
    
    def _get_images(self, batch, model_pred, timesteps, noisy_latents):
        image_in = (batch[self.image_column] / 2) + 0.5
        denoised_latents = self.model.noise_scheduler.step(model_pred, timesteps, noisy_latents).pred_original_sample
        image_out = (self.model.vae.decode(1 / self.model.vae.config.scaling_factor * denoised_latents)[0] / 2) + 0.5
        return image_in, image_out

    def _mse_loss(self, batch, model_pred, target, image_in, image_out):
        cam_in, _ = self.model.cam_generator(image_in.detach(), batch[self.caption_column], batch[self.bbox_column], gt_bboxes_only=True)
        cam_out, _ = self.model.cam_generator(image_out.detach(), batch[self.caption_column], batch[self.bbox_column], gt_bboxes_only=False)
        weight = torch.nn.functional.l1_loss(cam_in.detach(), cam_out.detach(), reduction="none").unsqueeze(0)
        weight = torch.nn.functional.interpolate(weight, size=(64, 64), mode='bilinear', align_corners=False)
        weight = (weight / (torch.quantile(weight, 0.95) + 1e-6)).clamp(0, 1)
        mse_loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="none")
        weighted_mse_loss = ((weight + 1) * mse_loss).mean() # run-1
        return weighted_mse_loss

    def _get_timestep_weight(self, timesteps, t_start_decay=900, t_end_decay=600):
        t = timesteps.float()
        weight = (t - t_end_decay) / (t_start_decay - t_end_decay)
        weight = weight.clamp(0.0, 1.0)
        return weight.item()

    def _dino_perceptual_loss_fg(self, batch, image_in, image_out, timesteps):
        image_out.clamp_(0.0, 1.0)
        bboxes = batch[self.bbox_column]
        batch_crops = []
        H, W = image_out.shape[-2:]
        device = image_out.device

        for b, boxes in enumerate(bboxes):
            for box in boxes:
                t_box = torch.tensor(box, device=device) if not isinstance(box, torch.Tensor) else box.to(device)
                if t_box.abs().sum() < 1e-6: continue
                coords = (t_box * torch.tensor([H, W, H, W], device=device)).round().long()
                y1, x1, y2, x2 = coords.tolist()                
                y1, x1 = max(0, y1), max(0, x1)
                y2, x2 = min(H, y2), min(W, x2)
                if y2 <= y1 + 8 or x2 <= x1 + 8: continue
                pair = torch.cat([image_out[b:b+1, :, y1:y2, x1:x2], 
                                image_in[b:b+1, :, y1:y2, x1:x2]], dim=0)
                batch_crops.append(torch.nn.functional.interpolate(pair, size=(224, 224), mode='bilinear', align_corners=False))
        if not batch_crops: 
            return torch.tensor(0.0, device=device, requires_grad=True)
        features = self.model.image_encoder(torch.cat(batch_crops, dim=0), mode='x_norm_patchtokens')
        f_pred, f_target = features[0::2], features[1::2]
        mu = self._get_timestep_weight(timesteps)
        return torch.nn.functional.mse_loss(f_pred, f_target.detach()) * 0.0005 * mu

    def train(self):
        global_step, first_epoch = 0, 0
        ## TODO: Maybe need the resume logic here, well maybe not.
        progress_bar = tqdm(range(global_step, self.max_train_steps), disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        for epoch in range(first_epoch, self.num_train_epochs):
            self.model.unet.train()
            train_loss = 0.0
            for step, batch in enumerate(self.dataloader):
                ## TODO: resume logics
                with self.accelerator.accumulate(self.model.unet):
                    '''Prepare Conditions'''
                    latents = self._latents(batch[self.image_column])
                    noise, perturbed_noise = self._noise(latents.shape)
                    timesteps = self._timesteps()
                    noisy_latents = self.model.noise_scheduler.add_noise(latents, perturbed_noise, timesteps)
                    target = self._target(latents, noise, timesteps)
                    per_token_embeds, pooler_embeds, text_embeds = self._encode_prompt(prompts=batch[self.caption_column])
                    cross_attention_kwargs = self._cross_attention_kwargs(batch, pooler_embeds, text_embeds)

                    '''Predict Noise'''
                    # noisy_latents [2, 4, 64, 64]; timesteps [2]; per_token_embeds [2, 16, 77, 768]; embeds_poolers [2, 16, 1, 768] 
                    # cross_attention_kwargs['ref_features'][0] [2, 15, 16, 768]; cross_attention_kwargs['ref_features'][1] [2, 1, 16, 768]
                    model_pred = torch.cat([
                        self.model.unet(noisy_latents[i:i+1], timesteps[i:i+1], per_token_embeds[i], cross_attention_kwargs=self._split_kwargs(cross_attention_kwargs, i)).sample 
                        for i in range(self.batch_size_per_device)
                    ], dim=0)

                    '''Calculate Losses'''
                    if self.config.phase == 'novel':
                        image_in, image_out = self._get_images(batch, model_pred, timesteps, noisy_latents)
                        weighted_mse_loss = self._mse_loss(batch, model_pred, target, image_in, image_out)
                        # dino_perceptual_loss = self._dino_perceptual_loss_fg(batch, image_in, image_out, timesteps)
                        loss = weighted_mse_loss #+ dino_perceptual_loss
                    else:
                        loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    avg_loss = self.accelerator.gather(loss.detach().repeat(self.batch_size_per_device)).mean()
                    train_loss += avg_loss.item() / self.config.accelerator.gradient_accumulation_steps

                    '''Update Parameters'''
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.unet.parameters(), self.config.training.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    '''Finish Batch'''
                    progress_bar.update(1)
                    global_step += 1
                    self.accelerator.log({"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0

                    '''Save Models'''
                    if global_step % self.config.training.ckpt_interval_steps == 0 and self.accelerator.is_main_process:
                        ckpt_path = get_ckpt_path(self.config, ckpt_steps=global_step)
                        # self.accelerator.save_state(ckpt_path)
                        torch.save(self.model.custom_layers.state_dict(), os.path.join(ckpt_path, 'CustomLayers.pth'))
                        torch.save(self.model.image_proj_model.state_dict(), os.path.join(ckpt_path, 'ImageProjModel.pth'))
                        self.logger.info(f"Saved state to {ckpt_path}.")

                '''Update Bar'''
                logs = {"step_loss": f"{loss.detach().item():.6f}", 
                        "lr": self.scheduler.get_last_lr()[0]}
                if self.config.phase == 'novel':
                    logs["mse_loss"] = f"{weighted_mse_loss.detach().item():.6f}"
                progress_bar.set_postfix(**logs)

                '''Finish Training'''
                if global_step >= self.max_train_steps:
                    break

        self.accelerator.end_training()


def main(config):
    accelerator = set_accelerator(config)
    logger = set_logger(config, accelerator, prefix=f'Train-{config.phase}')
    model = create_model(config)
    dataloader = create_dataloader(config, accelerator, model.image_processor, logger)
    _, max_train_steps, _ = calculate_training_schedule(config, dataloader)
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer, accelerator, max_train_steps)
    model, optimizer, dataloader, scheduler = prepare_models(accelerator, model, optimizer, dataloader, scheduler)
    steps_per_epoch, max_train_steps, num_train_epochs = calculate_training_schedule(config, dataloader)

    trainer = Trainer(config, accelerator, logger, model, dataloader, optimizer, scheduler, steps_per_epoch, max_train_steps, num_train_epochs)
    trainer.train()


if __name__ == "__main__":
    config = load_config()
    main(config)