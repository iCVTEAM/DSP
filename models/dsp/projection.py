import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from .layers import PositionNet, GatedSelfAttentionDense, CrossAttention
from .prototype_bank import PrototypeBank

from datamodules import RefTable
from PIL import Image
import os

from utils import Dict

class FourierEmbedder(nn.Module):
    def __init__(self, num_freqs=64, temperature=100):
        super().__init__()

        self.num_freqs = num_freqs
        self.temperature = temperature

        freq_bands = temperature ** (torch.arange(num_freqs) / num_freqs)
        freq_bands = freq_bands[None, None]
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    def __call__(self, x):
        x = self.freq_bands * x.unsqueeze(-1)
        return torch.stack((x.sin(), x.cos()), dim=-1).permute(0, 2, 3, 1).reshape(x.shape[0], -1)

# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class SelfAttentionLayer(nn.Module):
    def __init__(self, channels, nhead, dropout=0.0):
        super().__init__() 
        self.norm1 = nn.LayerNorm(channels)
        self.self_attn = nn.MultiheadAttention(channels, nhead, dropout=dropout)

        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                input,
                mask = None,):
        h = self.norm1(input)
        h1 = self.self_attn(query=h, key=h, value=h, attn_mask=mask)[0]
        h = h + self.dropout(h1)
        h = self.norm2(h)
        return h
        

class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x) # [15, 257, 1280]
        latents = self.norm2(latents) # [15, 16, 1280]

        b, l, _ = latents.shape

        q = self.to_q(latents) # [15, 16, 1280]
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1) # [15, 257 + 16, 1280] 

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)

class CrossAttentionLayer(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.perceiver_fg = PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads)
        self.perceiver_bg = PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads)
    def forward(self, x, latents):
        x_fg, x_bg = x
        latents_fg, latents_bg = latents
        out_fg = self.perceiver_fg(x_fg, latents_fg)
        out_bg = self.perceiver_bg(x_bg, latents_bg)
        return out_fg, out_bg

class Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
        max_seq_len: int = 257,  # CLIP tokens + CLS token
        apply_pos_emb: bool = False,
        num_latents_mean_pooled: int = 0,  # number of latents derived from mean pooled representation of the sequence
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim) if apply_pos_emb else None

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.to_latents_from_mean_pooled_seq = (
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange("b (n d) -> b n d", n=num_latents_mean_pooled),
            )
            if num_latents_mean_pooled > 0
            else None
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x, coherent_queries=None):
        if self.pos_emb is not None:
            n, device = x.shape[1], x.device
            pos_emb = self.pos_emb(torch.arange(n, device=device))
            x = x + pos_emb

        latents = self.latents.repeat(x.size(0), 1, 1) if coherent_queries is None else \
                  torch.cat([self.latents, coherent_queries], dim=1).repeat(x.size(0), 1, 1) # fg [15, 16, 1280], bg [1, 8 + 8, 1280]

        x = self.proj_in(x) # fg [15, 257, 1280]
        
        if self.to_latents_from_mean_pooled_seq:
            meanpooled_seq = masked_mean(x, dim=1, mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.bool))
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim=-2)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents # fg [15, 16, 1280]
            latents = ff(latents) + latents # fg [15, 16, 1280]
        
        latents = self.proj_out(latents) # fg [15, 16, 768]
        return self.norm_out(latents)

class SerialSampler(nn.Module):
    def __init__(
        self,
        config,
        image_processor,
        image_encoder,
        dim=1024,
        depth=8,
        dim_head=64,
        num_queries=[8, 8, 8],
        embedding_dim=768,
        output_dim=1024,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.fg_resampler = Resampler(dim=dim, depth=depth, heads=dim // dim_head, dim_head=dim_head,
                                      num_queries=num_queries[0], embedding_dim=embedding_dim,
                                      output_dim=output_dim, **kwargs)
        self.bg_resampler = Resampler(dim=dim, depth=depth, heads=dim // dim_head, dim_head=dim_head,
                                      num_queries=num_queries[1], embedding_dim=embedding_dim,
                                      output_dim=output_dim, **kwargs)
        self.point_net = PositionNet(in_dim=output_dim, out_dim=output_dim)
        self.coherent_bridge = GatedSelfAttentionDense(query_dim=dim, context_dim=output_dim, 
                                                       n_heads=dim // dim_head, d_head=dim_head)
        self.coherent_queries = nn.Parameter(torch.randn(1, num_queries[2], dim) / dim**0.5) # [1, 8, 1280]

        # For Novel Phase
        self.config = config
        if self.config.phase == 'novel':
            self.sample_aggregator_fg = SampleAggregator()
            self.categories = config.dataset.categories[config.phase]
            self.aux = Dict(image_processor=image_processor, image_encoder=image_encoder)
            self.image_patch_path = config.dataset.image_patch_path
            self.backups = None
            num_prototypes = config.model.get('num_prototypes') or 128
            print(f'[SerialSampler]: num_prototypes is {num_prototypes}')
            self.prototype_bank = PrototypeBank(config, image_encoder, image_processor, num_prototypes=num_prototypes)

    @property
    def image_processor(self):
        return self.aux.image_processor

    @property
    def image_encoder(self):
        return self.aux.image_encoder

    def build_backup_pool(self):
        self.ref_table = RefTable()()
        self.backups, self.backup_masks = {}, {}
        max_length, max_len_category = 0, None
        for category in self.categories:
            files = list(self.ref_table[category].keys())
            if len(files) > max_length:
                max_length = len(files)
                max_len_category = category
            backups_each_categories = []
            for file in files:
                img = Image.open(os.path.join(self.image_patch_path, category, file)).convert('RGB')
                img = self.image_processor(images=img, return_tensors="pt")['pixel_values'].squeeze(0)
                backups_each_categories.append(img)
            self.backups[category] = self.fg_resampler(self.image_encoder(torch.stack(backups_each_categories).to(next(self.parameters()).device)))
            self.backup_masks[category] = torch.ones(self.backups[category].shape[0], dtype=torch.bool, device=self.backups[category].device)
        for category, item in self.backups.items():
            cur_length = item.shape[0]
            if cur_length < max_length:
                pad_length = max_length - cur_length
                pad = torch.zeros(pad_length, *item.shape[1:], device=item.device, dtype=item.dtype)
                self.backups[category] = torch.cat([item, pad], dim=0)
                pad_mask = torch.zeros(pad_length, dtype=torch.bool, device=item.device)
                self.backup_masks[category] = torch.cat([self.backup_masks[category], pad_mask], dim=0)
        self.backups[''] = torch.zeros_like(self.backups[max_len_category])   
        self.backup_masks[''] = torch.zeros(max_length, dtype=torch.bool, device=self.backups[max_len_category].device)

    # x_objs.shape [15, 257, 1024] ; x_bg.shape [1, 257, 1024]
    def forward(self, x_objs, obboxes, x_bg, captions):
        if self.config.phase == 'novel' and self.backups is None:
            self.build_backup_pool()
            self.prototype_bank.build_prototypes()

        B = x_bg.shape[0] # 1
        obboxes = torch.from_numpy(np.array([obbox[::2] + obbox[1::2] for obbox in obboxes[0]])).float().to(x_objs.device) # [15, 8]
        embed_obboxes = self.point_net(obboxes).unsqueeze(1) # [15, 8] -> [15, 1, 768]
        embed_objs = self.fg_resampler(x_objs[:, 0]) # [15, 257, 1024] -> [15, 16, 768]
        conherent_queries = self.coherent_bridge(self.coherent_queries, (embed_obboxes + embed_objs.detach()).view(B, -1, self.output_dim)) # [1, 8, 1280]
        embed_context = self.bg_resampler(x_bg[:, 0], conherent_queries) # [1, 16, 768]

        if self.config.phase == 'novel':
            backup_embeddings = torch.stack(list(map(lambda caption: self.backups[caption], captions[0])), dim=0)
            backup_masks = torch.stack(list(map(lambda caption: self.backup_masks[caption], captions[0])), dim=0)
            embed_objs = self.sample_aggregator_fg(embed_objs, backup_embeddings, backup_masks)
        return embed_objs, embed_context # [15, 16, 768], [1, 16, 768]


class SampleAggregator(nn.Module):
    def __init__(self, dim=768, heads=8, dim_head=160, dropout=0.0, alpha_init=0.0):
        super().__init__()

        # backup self-attention (use the same cross-attn but context=x)
        self.backup_self_attn = CrossAttention(
            query_dim=dim,
            context_dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout
        )
        self.backup_ln = nn.LayerNorm(dim)

        # primary cross-attention (primary Q, backup K/V)
        self.primary_cross_attn = CrossAttention(
            query_dim=dim,
            context_dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout
        )
        self.primary_ln = nn.LayerNorm(dim)

        # FFN
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        self.ff_ln = nn.LayerNorm(dim)

        self.gating_param_ca = nn.Parameter(torch.tensor(alpha_init))      # cross-attn scale
        self.gating_param_ff = nn.Parameter(torch.tensor(alpha_init))  # FFN scale

    def forward(self, primary, backup, backup_mask):
        """
        primary: [B, T, C] = [B, 16, 768]
        backup:  [B, N, T, C] = [B, 4, 16, 768] (B bboxes, N refs, T features, C dim)
        backup_mask:  [B, N]   # True = valid
        """
        B, N, T, C = backup.shape

        # --------- 1) backup self-attention ---------
        backup_groups = backup.reshape(B * N, T, C)                 # (B*N, T, C)
        bk = self.backup_ln(backup_groups)
        backup_sa = self.backup_self_attn(bk, context=bk)
        backup_groups = backup_groups + backup_sa
        backup_groups = backup_groups.reshape(B, N, T, C)

        # --------- 1b) pooled backup memory (small, controlled)
        backup_mem = backup_groups.mean(dim=2)                      # (B, N, C)  # torch.Size([15, 4, 768])

        # --------- 2) primary cross-attention (Q=primary, K/V=backup) ---------
        p = self.primary_ln(primary)
        primary_ca = self.primary_cross_attn(p, context=backup_mem, mask=backup_mask)
        fused = primary + self.gating_param_ca * primary_ca

        # --------- 3) FFN refinement ---------
        f = self.ff_ln(fused)
        fused = fused + self.gating_param_ff * self.ff(f)

        return fused


def masked_mean(t, *, dim, mask=None):
    if mask is None:
        return t.mean(dim=dim)

    denom = mask.sum(dim=dim, keepdim=True)
    mask = rearrange(mask, "b n -> b n 1")
    masked_t = t.masked_fill(~mask, 0.0)

    return masked_t.sum(dim=dim) / denom.clamp(min=1e-5)