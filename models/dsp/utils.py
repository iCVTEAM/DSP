import functools
import os
import random
import numpy as np
import imagesize
import torch
import torch.nn.functional as F
from PIL import Image
import cv2

def seed_everything(seed):
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array/value - 1)).argmin()
        return idx

def get_sup_mask(mask_list):
    or_mask = np.zeros_like(mask_list[0])
    for mask in mask_list:
        or_mask += mask
    or_mask[or_mask >= 1] = 1
    sup_mask = 1 - or_mask
    return sup_mask

def get_masks(obboxes, height, width, device):
    # Construct Instance Guidance Mask
    guidance_masks, in_box = [], []
    for obbox in obboxes[0]:  
        guidance_mask = np.zeros((height, width))
        if np.count_nonzero(obbox):
            pts = np.array(obbox).reshape(-1, 1, 2)
            pts[..., 0] = pts[..., 0] * width
            pts[..., 1] = pts[..., 1] * height
            pts = np.int32(pts)
            guidance_masks.append(cv2.fillPoly(guidance_mask, [pts], 1)[None, ...])
        else:
            guidance_masks.append(guidance_mask[None, ...])
        in_box.append([obbox[0], obbox[2], obbox[4], obbox[6], obbox[1], obbox[3], obbox[5], obbox[7]])
    
    # Construct Background Guidance Mask
    sup_mask = get_sup_mask(guidance_masks)
    supplement_mask = torch.from_numpy(sup_mask[None, ...])
    supplement_mask = F.interpolate(supplement_mask, (height//8, width//8), mode='bilinear').float()
    supplement_mask = supplement_mask.to(device)  # (1, 1, H, W)

    guidance_masks = np.concatenate(guidance_masks, axis=0)
    guidance_masks = guidance_masks[None, ...]
    guidance_masks = torch.from_numpy(guidance_masks).float().to(device)
    guidance_masks = F.interpolate(guidance_masks, (height//8, width//8), mode='bilinear')  # (1, instance_num, H, W)
    # guidance_masks.shape [1, 15, 64, 64] ; supplement_mask.shape [1, 1, 64, 64]
    in_box = torch.from_numpy(np.array(in_box))[None, ...].float().to(device)  # (1, instance_num, 4)
    return guidance_masks, supplement_mask, in_box


def get_sigmoid(bboxes, height, width, device):
    sigmoid_values = []
    for w_min, h_min, w_max, h_max in bboxes[0]:
        H, W = height//8, width // 8
        x = torch.linspace(0, W - 1, W)
        y = torch.linspace(0, H - 1, H)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        xx, yy = xx / H, yy / W
        mu1 = (w_min + w_max) / 2
        mu2 = (h_min + h_max) / 2
        sigma1 = ((w_max - w_min) ** 2) / 4
        sigma2 = ((h_max - h_min) ** 2) / 4
        if sigma1 == 0 or sigma2 == 0:
            sigmoid_values.append(torch.zeros_like(xx))
            continue
        exponent = -10 * (1 - ((xx - mu1) ** 2) / sigma1 - ((yy - mu2) ** 2) / sigma2)
        sigmoid_value = 1 / (1 + torch.exp(exponent))
        sigmoid_values.append(sigmoid_value)
    sigmoid_values = torch.stack(sigmoid_values, dim=0)[None, ...].to(device)
    # sigmoid_values.shape [1, 15, 64, 64] ; in_box.shape [1, 15, 8]
    return sigmoid_values


class ExemplarPool:
    def __init__(self, data_embeds_dict_path, exemplar_pool_path, image_processor):
        self.data_embeds_dict = torch.load(data_embeds_dict_path, map_location='cpu')
        self.all_img_names = np.array(list(self.data_embeds_dict.keys()))
        self.all_txt_embs = torch.cat([self.data_embeds_dict[name]['txt_emb'] for name in self.all_img_names], dim=0)
        self.all_img_embs = torch.cat([self.data_embeds_dict[name]['img_emb'] for name in self.all_img_names], dim=0)
        
        self.exemplar_pool_path = exemplar_pool_path
        self.image_processor = image_processor

    def to(self, device, dtype=None):
        self.device = device
        self.all_txt_embs = self.all_txt_embs.to(device=device, dtype=dtype)
        self.all_img_embs = self.all_img_embs.to(device=device, dtype=dtype)

    def get_similar_exemplars_names(self, prompt_emb, topk, sim_mode):
        prompt_emb = F.normalize(prompt_emb, dim=-1).detach()

        if sim_mode == 'text2text':
            sim_vals = torch.matmul(prompt_emb, self.all_txt_embs.T)
        elif sim_mode == 'text2img':
            sim_vals = torch.matmul(prompt_emb, self.all_img_embs.T)
        elif sim_mode == 'both':
            txt_sim_vals = torch.matmul(prompt_emb, self.all_txt_embs.T)
            img_sim_vals = torch.matmul(prompt_emb, self.all_img_embs.T)
            sim_vals = (txt_sim_vals + img_sim_vals) * 0.5
        else:
            raise ValueError('Invalid mode for similarity computation!')
        
        _, topk_indices = torch.topk(sim_vals, k=topk, dim=1)
        topk_img_names = self.all_img_names[topk_indices.cpu().numpy()].tolist()

        return topk_img_names
    
    def names_to_tensors(self, topk_img_names):
        topk_img_tensors = []
        for names_for_one_prompt in topk_img_names:
            topk_tensors_per_prompt = []
            for name in names_for_one_prompt:
                img = Image.open(os.path.join(self.exemplar_pool_path, name)).convert('RGB')
                # tensor = self.image_processor(images=img, return_tensors="pt", do_normalize=False)['pixel_values'].squeeze(0)
                tensor = self.image_processor(images=img, return_tensors="pt")['pixel_values'].squeeze(0)
                topk_tensors_per_prompt.append(tensor)
            stacked_k_tensors = torch.stack(topk_tensors_per_prompt)
            topk_img_tensors.append(stacked_k_tensors)
        topk_img_tensors = torch.stack(topk_img_tensors)
        return topk_img_tensors

    def get_similar_exemplars(self, prompt_emb, topk=1, sim_mode='text2img'):
        topk_img_names = self.get_similar_exemplars_names(prompt_emb, topk, sim_mode)
        topk_img_tensors = self.names_to_tensors(topk_img_names)
        return topk_img_tensors.to(self.device)