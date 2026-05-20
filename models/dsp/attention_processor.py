import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from typing import Optional

from .layers import MIFusion, MIFusionPrototype
from .utils import get_masks, get_sigmoid


class AttnProcessor2_0(nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, hidden_size=None, cross_attention_dim=None):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        super().__init__()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        # Useless
        bboxes=[],
        obboxes=[],
        embeds_pooler=None,
        height=512,
        width=512,
        prototypes=None,
        ref_features=None,
        guidance_masks=None,
        supplement_mask=None,
        sigmoid_values=None,
        in_box=None,
        do_classifier_free_guidance=False,
        # End Useless
        *args,
        **kwargs,
    ) -> torch.Tensor:

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class MaskedProcessor2_0(nn.Module):
    def __init__(self, hidden_size, cross_attention_dim=None,
                 use_ea_attn=False, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.use_ea_attn = use_ea_attn
        self.prototype_mode = use_ea_attn and (kwargs['phase'] == 'novel') and (kwargs.get('attn_prototype_switch') is True)
        if self.prototype_mode:
            self.fusion = MIFusionPrototype(hidden_size, context_dim=cross_attention_dim)
        elif use_ea_attn:
            self.fusion = MIFusion(hidden_size, context_dim=cross_attention_dim)
    
    # Train ; Test (do_classifier_free_guidance)
    # Train // Train2(self.use_ea_attn) ; Test // Test2(self.use_ea_attn)
    def __call__(
            self,
            attn: Attention,
            # Output the same size as hidden_states, encoder_hidden_states as Key and Value to inject information to hidden_states
            # shape[-2] 4096 as 64x64, 64 as 8x8; shape[-1] as hidden_size
            hidden_states, # [1, 4096, 320] // [1, 64, 1280] ; [2, 4096, 320] // [2, 64, 1280] && torch.all(hidden_states[0] == hidden_states[1]) is Ture
            encoder_hidden_states=None, # [16, 77, 768] ; [17, 77, 768]
            attention_mask=None,
            bboxes=[],
            obboxes=[],
            embeds_pooler=None,
            height=512,
            width=512,
            prototypes=None,
            ref_features=None,
            guidance_masks=None,
            supplement_mask=None,
            sigmoid_values=None,
            in_box=None,
            do_classifier_free_guidance=False,
    ):

        instance_num = len(obboxes[0]) # 15

        if not self.use_ea_attn:
            # [1, 77, 768]; [2, 77, 768]
            encoder_hidden_states = encoder_hidden_states[:2, ...] if do_classifier_free_guidance else encoder_hidden_states[:1, ...]

        if self.use_ea_attn:
            if do_classifier_free_guidance:
                hidden_states = torch.cat([hidden_states[0:1], hidden_states[1:2].repeat(instance_num + 1, 1, 1)]) # ;//[17, 64, 1280]
                image_token = hidden_states[1:]
            else:
                hidden_states = hidden_states.repeat(instance_num + 1, 1, 1) # //[16, 64, 1280]
                image_token = hidden_states

        batch_size, sequence_length, _ = hidden_states.shape # _ is hidden_size

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)            # [1, 4096, 320] // [16, 64, 1280] ; [2, 4096, 320]->[2, 4096, 320] // [17, 64, 1280]->[17, 64, 1280]
        key = attn.to_k(encoder_hidden_states)      # [1, 77, 320]   // [16, 77, 1280] ; [2, 77, 768]->[2, 77, 320]     // [17, 77, 768 (cross_attention_dim)] -> [17, 77, 1280 (self.inner_kv_dim)]
        value = attn.to_v(encoder_hidden_states)    # Same with key

        inner_dim = key.shape[-1] # 320 // 1280
        head_dim = inner_dim // attn.heads # 40 // 160

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2) # [1, 8, 4096, 40] // [16, 8, 64, 160]

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2) # [1, 8, 77, 40] // [16, 8, 77, 160]
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2) # [1, 8, 77, 40] // [16, 8, 77, 160]

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        ) # [1, 8, 4096, 40] // [16, 8, 64, 1280]

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim) # [1, 4096, 320] // [16, 64, 1280]
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)           # [1, 4096, 320] // [16, 64, 1280] ; [2, 4096, 320] // [17, 64, 1280]  # Linear
        hidden_states = attn.to_out[1](hidden_states)           # [1, 4096, 320] // [16, 64, 1280] ; [2, 4096, 320] // [17, 64, 1280]  # Dropout

        if not self.use_ea_attn:
            return hidden_states

        assert self.use_ea_attn

        if do_classifier_free_guidance:
            hidden_states_uncond, hidden_states = hidden_states[0:1], hidden_states[1:]  # torch.Size([1, HW, C])

        other_info = {}
        other_info['image_token'] = image_token.unsqueeze(0) # [1, 16, 64, 1280]
        other_info['context'] = encoder_hidden_states[1:, ...] if do_classifier_free_guidance else encoder_hidden_states # [16, 77, 768]
        other_info['box'] = in_box # [1, 15, 8]
        other_info['context_pooler'] = embeds_pooler  # [16, 1, 768]
        other_info['supplement_mask'] = supplement_mask # [1, 1, 64, 64]
        other_info['height'] = height # 512
        other_info['width'] = width # 512
        other_info['ref_features'] = ref_features # [15, 16, 768], [1, 16, 768]
        other_info['sigmoid_values'] = sigmoid_values # [1, 16, 768]
        other_info['guidance_masks'] = guidance_masks  # [1, 15, 64, 64]
        other_info['instance_num'] = instance_num
        other_info['prototypes'] = prototypes

        hidden_states = self.fusion(hidden_states.unsqueeze(0),  # [1, 16, 64, 1280]
                                    other_info=other_info)
        # hidden_states_cond.shape [1, 64, 1280]

        if do_classifier_free_guidance:
            hidden_states = torch.cat([hidden_states_uncond, hidden_states])
        return hidden_states

    
def set_processors(unet, **kwargs):
    attn_processors = {}
    for name, _ in unet.attn_processors.items():
        use_ea_attn = False
        kwargs['attn_prototype_switch'] = False
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1] # unet.config.block_out_channels [320, 640, 1280, 1280]
            use_ea_attn = True         
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            attention_id = int(name[len("up_blocks.2.attentions.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            if block_id == 1:
                use_ea_attn = True
            elif (block_id != 1) and (kwargs['phase'] == 'novel'): # run-4
                use_ea_attn = True
                kwargs['attn_prototype_switch'] = True
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]                
        if cross_attention_dim is not None:
            attn_processors[name] = MaskedProcessor2_0(hidden_size=hidden_size,
                                                    cross_attention_dim=cross_attention_dim,
                                                    use_ea_attn=use_ea_attn,
                                                    **kwargs)
        else:
            attn_processors[name] = AttnProcessor2_0()
    unet.set_attn_processor(attn_processors)