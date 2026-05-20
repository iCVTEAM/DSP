import accelerate, datasets, diffusers, transformers, safetensors
from typing import Any, Callable, Dict, List, Optional, Union
from PIL import Image, ImageDraw, ImageFont
from packaging import version
from itertools import chain
from tqdm.auto import tqdm
import torch
import math
import time
import os

from utils import load_config, set_logger, get_ckpt_path, get_output_path, Dict
from datamodules import Loader
import models

import numpy as np
import cv2

config, core_module = None, None

class StableDiffusionCCDiffPipeline(diffusers.pipelines.stable_diffusion.StableDiffusionPipeline):
    def __init__(
        self,
        vae: diffusers.AutoencoderKL,
        text_encoder: transformers.CLIPTextModel,
        tokenizer: transformers.CLIPTokenizer,
        unet: diffusers.UNet2DConditionModel,
        scheduler: diffusers.schedulers.KarrasDiffusionSchedulers,
        safety_checker: diffusers.pipelines.stable_diffusion.StableDiffusionSafetyChecker,
        feature_extractor: transformers.CLIPImageProcessor,
        image_encoder: transformers.CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        self.text_projector = transformers.CLIPTextModelWithProjection.from_pretrained(config.model.clip_weight_path)
        self.image_encoder = core_module.FrozenDinoV2Encoder(config.model.dinov2_vitl14_path)
        self.image_proj_model = core_module.SerialSampler(config, feature_extractor, self.image_encoder, dim=config.model.image_proj_model.dim, depth=config.model.image_proj_model.depth, 
                                                     dim_head=config.model.image_proj_model.dim_head, num_queries=config.model.image_proj_model.num_queries, 
                                                     embedding_dim=self.image_encoder.model.embed_dim, output_dim=unet.config.cross_attention_dim, ff_mult=config.model.image_proj_model.ff_mult)
        self.exemplar_pool = core_module.ExemplarPool(config.model.exemplar_pool.data_embeds_dict_path, config.model.exemplar_pool.exemplar_pool_path, feature_extractor)
        self.cam_generator = core_module.CAMGenerator(categories=config.dataset.categories.all, clip_path=config.model.clip_vit_b16_path)

        init_kwargs = {
            "image_encoder": self.image_encoder,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "unet": unet,
            "scheduler": scheduler,
            "safety_checker": safety_checker,
            "feature_extractor": feature_extractor,
            "requires_safety_checker": requires_safety_checker
        }
        super().__init__(**init_kwargs)

    def _encode_prompt(self, prompts, device, do_classifier_free_guidance, negative_prompt):
        batch_size, group_size = len(prompts), len(prompts[0]) # 16
        flattened_prompts = list(chain.from_iterable(prompts))
        # len(flattened_prompts): B * 16 -> text_input_ids.shape [B * 16, 77], untruncated_ids.shapeB * 16, 36(longest text)]
        text_inputs = self.tokenizer(flattened_prompts, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_input_ids, attention_mask = text_inputs.input_ids.to(device), text_inputs.attention_mask.to(device) # Original Version as None?
        # untruncated_ids = self.model.tokenizer(flattened_prompts, padding="longest", return_tensors="pt").input_ids
        # if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids.to(self.accelerator.device)):
        #     removed_text = self.model.tokenizer.batch_decode(untruncated_ids[:, self.model.tokenizer.model_max_length - 1: -1])
        #     self.logger.warning("The following part of your input was truncated because CLIP can only handle sequences up to"f" {self.model.tokenizer.model_max_length} tokens: {removed_text}")
        text_encoder_output = self.text_encoder(text_input_ids, attention_mask=None)
        per_token_embeds, pooler_embeds = text_encoder_output.last_hidden_state, text_encoder_output.pooler_output # prompt_embeds.shape [B * 16, 77, 768], embeds_pooler.shape [B * 16, 768]
        text_embeds = self.text_projector(text_input_ids, attention_mask=None).text_embeds # .shape [B * 16, 768]
        per_token_embeds = per_token_embeds.view(batch_size, group_size, self.tokenizer.model_max_length, -1) # per_token_embeds.shape [B, 16, 77, 768]
        pooler_embeds = pooler_embeds.view(batch_size, group_size, -1).unsqueeze(2) # embeds_pooler.shape [B, 16, 1, 768]
        text_embeds = text_embeds.view(batch_size, group_size, -1) # text_embeds.shape [B, 16, 768]

        if do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = "worst quality, low quality, bad anatomy"
            uncond_tokens = [negative_prompt] * batch_size # len(uncond_tokens) = B as 1
            uncond_input = self.tokenizer(uncond_tokens, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
            uncond_input_ids, uncond_attention_mask = uncond_input.input_ids.to(device), uncond_input.attention_mask.to(device)
            negative_prompt_embeds = self.text_encoder(uncond_input_ids, attention_mask=None)
            negative_prompt_embeds = negative_prompt_embeds.last_hidden_state # [B * 1, 77, 768]
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size, 1, self.tokenizer.model_max_length, -1)
            final_per_token_embeds = torch.cat([negative_prompt_embeds, per_token_embeds], dim=1) # [B, 17, 77, 768]

        return final_per_token_embeds, per_token_embeds, pooler_embeds, text_embeds

    def __call__(
            self,
            prompt: List[List[str]] = None,
            obboxes: List[List[List[float]]] = None,
            bboxes: List[List[List[float]]] = None,
            instances: Optional[torch.FloatTensor] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            GUI_progress=None,
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        batch_size, group_size = len(prompt), len(prompt[0])
        device = self.unet.device
        self.image_encoder.to(device)
        self.text_projector.to(device)
        self.image_proj_model.to(device)
        self.exemplar_pool.to(device)
        do_classifier_free_guidance = guidance_scale > 1.0

        final_per_token_embeds, per_token_embeds, pooler_embeds, text_embeds = self._encode_prompt(prompt, device, do_classifier_free_guidance, negative_prompt)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(batch_size * num_images_per_prompt, num_channels_latents, height, width, final_per_token_embeds.dtype, device, generator, latents) # [1, 4, 64, 64]
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta) # {'generator': None}

        instances_imgs, exemplar_imgs = instances, self.exemplar_pool.get_similar_exemplars(text_embeds[:, 0], topk=5) # [B, 15, 3, 224, 224], [B, topk, 3, 224, 224]
        num_instances, instance_group_size, exemplar_group_size = instances_imgs.shape[1], instances_imgs.shape[2], exemplar_imgs.shape[1] # 15, topk
        instance_imgs, exemplar_imgs = instances_imgs.flatten(0, 2), exemplar_imgs.flatten(0, 1) # [B * 15, 3, 224, 224], [B * topk, 3, 224, 224]
        with torch.no_grad():
            instance_features = self.image_encoder(instance_imgs) # [B * 15, 257, 1024]
            instance_features = instance_features.view(batch_size, num_instances, instance_group_size, instance_features.shape[-2], instance_features.shape[-1])
            exemplar_features = self.image_encoder(exemplar_imgs) # [B * topk, 257, 1024]
            exemplar_features = exemplar_features.view(batch_size, 1, exemplar_group_size, exemplar_features.shape[-2], exemplar_features.shape[-1])
        resampled_instance_features_list, resampled_exemplar_features_list = [], []
        for i in range(batch_size):
            resampled_instance_features, resampled_exemplar_features = self.image_proj_model(instance_features[i], [obboxes[i]], exemplar_features[i], [prompt[i][1:]])
            resampled_instance_features_list.append(resampled_instance_features), resampled_exemplar_features_list.append(resampled_exemplar_features)
        resampled_instance_features, resampled_exemplar_features = torch.stack(resampled_instance_features_list), torch.stack(resampled_exemplar_features_list)
        prototypes = torch.stack([self.image_proj_model.prototype_bank.get_prototypes(prompt[i][1:]) for i in range(len([prompt]))])
        guidance_masks, supplement_mask, in_box = core_module.get_masks(obboxes, height, width, device)
        sigmoid_values = core_module.get_sigmoid(bboxes, height, width, device)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order # 0
        self.unet.eval()
        # with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            latent_model_input = (torch.cat([latents] * 2) if do_classifier_free_guidance else latents)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            cross_attention_kwargs = {
                'bboxes': [bboxes[0]], 'obboxes': [obboxes[0]], 'embeds_pooler': pooler_embeds[0], 'height': height, 'width': width, 'prototypes': prototypes[0],
                'ref_features': (resampled_instance_features[0], resampled_exemplar_features[0]), 'do_classifier_free_guidance': do_classifier_free_guidance,
                'guidance_masks': guidance_masks, 'supplement_mask': supplement_mask, 'in_box': in_box, 'sigmoid_values': sigmoid_values
            }
            # latent_model_input.shape [2, 4, 64, 64], final_per_token_embeds.shape [17, 77, 768]]
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=final_per_token_embeds[0], cross_attention_kwargs=cross_attention_kwargs).sample # [2, 4, 64, 64]
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            step_output = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)
            latents = step_output.prev_sample # another key is pred_original_sample

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                # progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

        if output_type == "latent":
            image = latents
        elif output_type == "pil":
            image = self.decode_latents(latents)
            image = self.numpy_to_pil(image)
        else:
            image = self.decode_latents(latents)
            
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, None)

        return diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)

def load_pipe(config, device):
    ckpt_path = get_ckpt_path(config, ckpt_steps=config.inference.ckpt_steps)
    pipe = StableDiffusionCCDiffPipeline.from_pretrained(config.model.sd15_weight_path)
    core_module.set_processors(pipe.unet, phase=config.phase)
    custom_layers = diffusers.loaders.AttnProcsLayers(pipe.unet.attn_processors)
    # state_dict = {k: v for k, v in safetensors.torch.load_file(os.path.join(ckpt_path, 'unet/diffusion_pytorch_model.safetensors')).items() if '.processor' in k or '.self_attn' in k}
    # custom_layers.load_state_dict(state_dict)
    custom_layers.load_state_dict(torch.load(os.path.join(ckpt_path, 'CustomLayers.pth')))
    pipe.image_proj_model.load_state_dict(torch.load(os.path.join(ckpt_path, 'ImageProjModel.pth')), strict=False)
    pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    return pipe


def set_seed(config):
    seed = config.inference.get("seed", 42)
    accelerate.utils.set_seed(seed)


def draw_box_desc(pil_img: Image, bboxes: List[List[float]], prompt: List[str]) -> Image:
    """Draw bounding boxes and object descriptions on an image."""
    color_list = ['red', 'blue', 'yellow', 'purple', 'green', 'black', 'brown', 'orange', 'white', 'gray']
    width, height = pil_img.size
    draw = ImageDraw.Draw(pil_img)

    # Load font
    font_path = './fonts/Rainbow-Party-2.ttf'
    font = ImageFont.truetype(font_path, 30)

    for obj_box, text_desc in zip(bboxes, prompt):
        # Determine fill color based on color keywords in the description
        fill_color = next((color for color in text_desc.split(' ') if color in color_list), 'white')
        # Only take the first part before a comma for text
        text = text_desc.split(',')[0]

        # Draw polygon for the bounding box
        polygon_points = [(x * width, y * height) for x, y in zip(obj_box[::2], obj_box[1::2])]
        draw.polygon(polygon_points, outline=fill_color, width=4)

        # Draw the text at the top-left corner of the box
        x_min, y_min = obj_box[0] * width, obj_box[1] * height
        draw.text((int(x_min), int(y_min)), text, fill=fill_color, font=font)

    return pil_img


tensor_to_pil = lambda t: Image.fromarray((t.permute(1,2,0) * 255).byte().cpu().numpy())


def save_image(output_dir, batch, image):
    image.save(os.path.join(output_dir, 'image', f'{batch["dataid"][0]}.jpg'))

    gt_img = tensor_to_pil(batch['image'][0])
    image_with_gt = Image.new('RGB', (gt_img.width + image.width, image.height))
    image_with_gt.paste(gt_img, (0, 0))
    image_with_gt.paste(image, (gt_img.width, 0))
    image_with_gt.save(os.path.join(output_dir, 'image_with_gt', f'{batch["dataid"][0]}.jpg'))

    image_with_box = draw_box_desc(image, batch['obboxes'][0], batch['captions'][0][1:])
    image_with_box.save(os.path.join(output_dir, 'image_with_box', f'{batch["dataid"][0]}.jpg'))


def main(config):
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    globals()['config'] = config
    global core_module
    core_module = getattr(models, config.model.name)
    logger = set_logger(config, accelerator, prefix=f'Infer-{config.phase}')
    pipe = load_pipe(config, device)
    dataloader = Loader(config, pipe.feature_extractor, 'infer', logger)()
    pipe, dataloader = accelerator.prepare(pipe, dataloader)
    set_seed(config)
    negative_prompt = 'worst quality, low quality, bad anatomy, watermark, text, blurry'

    output_dir = get_output_path(config, ckpt_steps=config.inference.ckpt_steps)
    if accelerator.is_local_main_process:
        for subdir in ["image", "image_with_gt", "image_with_box"]:
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    accelerator.wait_for_everyone()

    # total_inference_time = 0.0
    # img_count = 0
    # max_count = 5

    # if torch.cuda.is_available(): 
    #     torch.cuda.reset_peak_memory_stats()

    progress_bar = tqdm(range(len(dataloader.dataset)), disable=not accelerator.is_local_main_process)
    with torch.no_grad():
        for batch in dataloader:
            if not os.path.isfile(os.path.join(output_dir, 'image', f'{batch["dataid"][0]}.jpg')):
                # if torch.cuda.is_available(): torch.cuda.synchronize()
                # t_start = time.time()
                image = pipe(prompt=batch['captions'], obboxes=batch['obboxes'], bboxes=batch['bndboxes'], instances=batch['instances'], height=512, 
                            num_inference_steps=50, guidance_scale=7.5, negative_prompt=negative_prompt).images[0]
                # if torch.cuda.is_available(): torch.cuda.synchronize()
                # total_inference_time += (time.time() - t_start)
                save_image(output_dir, batch, image)

                # img_count += 1
                # if img_count >= max_count:
                #     peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
                #     print(f"GPU Memory Peak: {peak_gb:.2f} GB")
                #     print(f"\n inference {max_count} images cost: {total_inference_time:.2f}s | inference speed: {total_inference_time/max_count:.4f} s/image")
                #     break

            local_completed = torch.tensor(len(batch['dataid']), device=device)
            total_completed = accelerator.gather(local_completed)
            progress_bar.update(total_completed.sum().item())

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("All results saved.")


if __name__ == '__main__':
    config = load_config()
    main(config)