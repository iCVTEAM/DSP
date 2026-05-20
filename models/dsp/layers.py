import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from inspect import isfunction
from einops import rearrange, repeat
from torch import einsum

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None, return_attn=False, need_softmax=True):
        h = self.heads
        b = x.shape[0]

        q = self.to_q(x) # [15, 64, 1280]
        context = default(context, x)
        k = self.to_k(context) # [15, 18, 1280]
        v = self.to_v(context) # [15, 18, 1280]

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        if need_softmax:
            attn = sim.softmax(dim=-1)
        else:
            attn = sim

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        if return_attn:
            attn = attn.view(b, h, attn.shape[-2], attn.shape[-1])
            return self.to_out(out), attn
        else:
            return self.to_out(out)

class FourierEmbedder():
    def __init__(self, num_freqs=64, temperature=100):
        self.num_freqs = num_freqs
        self.temperature = temperature
        self.freq_bands = temperature ** ( torch.arange(num_freqs) / num_freqs )

    @ torch.no_grad()
    def __call__(self, x, cat_dim=-1):
        out = []
        for freq in self.freq_bands:
            out.append( torch.sin( freq*x ) )
            out.append( torch.cos( freq*x ) )
        return torch.cat(out, cat_dim)  # torch.Size([5, 30, 64])

class PositionNet(nn.Module):
    def __init__(self, in_dim, out_dim, fourier_freqs=8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs * 2 * 8  # 2 is sin&cos, 8 is xyxyxyxy

        # -------------------------------------------------------------- #
        self.linears_position = nn.Sequential(
            nn.Linear(self.position_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

    def forward(self, boxes):

        # embedding position (it may includes padding as placeholder)
        xyxy_embedding = self.fourier_embedder(boxes)  # B*1*4 --> B*1*C torch.Size([5, 1, 64])
        xyxy_embedding = self.linears_position(xyxy_embedding)  # B*1*C --> B*1*768 torch.Size([5, 1, 768])

        return xyxy_embedding

class LayoutAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., use_lora=False):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.use_lora = use_lora
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None, return_attn=False, need_softmax=True, guidance_mask=None):
        h = self.heads
        b = x.shape[0]

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        _, phase_num, H, W = guidance_mask.shape
        HW = H * W
        guidance_mask_o = guidance_mask.view(b * phase_num, HW, 1)
        guidance_mask_t = guidance_mask.view(b * phase_num, 1, HW)
        guidance_mask_sim = torch.bmm(guidance_mask_o, guidance_mask_t)  # (B * phase_num, HW, HW)
        guidance_mask_sim = guidance_mask_sim.view(b, phase_num, HW, HW).sum(dim=1)
        guidance_mask_sim[guidance_mask_sim > 1] = 1  # (B, HW, HW)
        guidance_mask_sim = guidance_mask_sim.view(b, 1, HW, HW)
        guidance_mask_sim = guidance_mask_sim.repeat(1, self.heads, 1, 1)
        guidance_mask_sim = guidance_mask_sim.view(b * self.heads, HW, HW)  # (B * head, HW, HW)

        sim[:, :, :HW][guidance_mask_sim == 0] = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of

        if need_softmax:
            attn = sim.softmax(dim=-1)
        else:
            attn = sim
            
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        if return_attn:
            attn = attn.view(b, h, attn.shape[-2], attn.shape[-1])
            return self.to_out(out), attn
        else:
            return self.to_out(out)

# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

class SelfAttention(nn.Module):
    def __init__(self, query_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout) )

    def forward(self, x):
        q = self.to_q(x) # B*N*(H*C)
        k = self.to_k(x) # B*N*(H*C)
        v = self.to_v(x) # B*N*(H*C)

        B, N, HC = q.shape 
        H = self.heads
        C = HC // H 

        q = q.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C
        k = k.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C
        v = v.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C

        sim = torch.einsum('b i c, b j c -> b i j', q, k) * self.scale  # (B*H)*N*N
        attn = sim.softmax(dim=-1) # (B*H)*N*N

        out = torch.einsum('b i j, b j c -> b i c', attn, v) # (B*H)*N*C
        out = out.view(B,H,N,C).permute(0,2,1,3).reshape(B,N,(H*C)) # B*N*(H*C)

        return self.to_out(out)

class GatedSelfAttentionDense(nn.Module):
    def __init__(self, query_dim, context_dim,  n_heads, d_head):
        super().__init__()
        
        # we need a linear projection since we need cat visual feature and obj feature
        self.linear = nn.Linear(context_dim, query_dim)

        self.attn = SelfAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)) )
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)) )

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as original one 
        self.scale = 1  


    def forward(self, x, objs):

        N_visual = x.shape[1]
        objs = self.linear(objs)

        x = x + self.scale*torch.tanh(self.alpha_attn) * self.attn(  self.norm1(torch.cat([x,objs],dim=1))  )[:,0:N_visual,:]
        x = x + self.scale*torch.tanh(self.alpha_dense) * self.ff( self.norm2(x) )  
        
        return x 

class MIFusion(nn.Module):
    def __init__(self, C, attn_type='base', context_dim=768, heads=8):
        # context_dim: SD1.4 768  SD2.1 1024
        super().__init__()
        self.ea_obj = CrossAttention(query_dim=C, context_dim=context_dim,
                                 heads=heads, dim_head=C // heads,
                                 dropout=0.0)
        self.norm_obj = nn.LayerNorm(C)
        self.ea2 = CrossAttention(query_dim=C, context_dim=context_dim,
                                 heads=heads, dim_head=C // heads,
                                 dropout=0.0)
        self.norm2 = nn.LayerNorm(C)
        self.pos_net = PositionNet(in_dim=context_dim, out_dim=context_dim)
        self.la = LayoutAttention(query_dim=C, heads=heads, 
                                  dim_head=C // heads, dropout=0.0)

    def forward(self, ca_x, other_info):
        # x: (B, instance_num+1, HW, C)
        # guidance_mask: (B, instance_num, H, W)
        # box: (instance_num, 4)
        # image_token: (B, instance_num+1, HW, C)

        # Reminder for shapes
        # ca_x                          # [1, 16, 64, 1280]                 # After Cross-Attn with encoder_hidden_states
        # guidance_mask                 # [1, 15, 64, 64]
        # other_info['image_token']     # [1, 16, 64, 1280]                 # Original hidden_states repeated 1+instance_num times
        # other_info['context']         # [16, 77, 768]
        # other_info['box']             # [1, 15, 8]
        # other_info['context_pooler']  # [16, 1, 768]
        # other_info['supplement_mask'] # [1, 1, 64, 64]
        # other_info['height'] = height # 512
        # other_info['width'] = width   # 512
        # other_info['ref_features']    # [15, 16, 768], [1, 16, 768]
        # other_info['sigmoid_values']  # [1, 16, 768]

        height, width = other_info['height'], other_info['width'] # 512, 512
        instance_num = other_info['instance_num'] # 15
        B, _, HW, C = ca_x.shape      # [1, 16, 64, 1280]
        down_scale = int(math.sqrt(height * width // ca_x.shape[2])) # 64
        H = height // down_scale # 8
        W = width // down_scale # 8

        guidance_masks = other_info['guidance_masks']
        guidance_masks = F.interpolate(guidance_masks, size=(H, W), mode='bilinear')   # [1, 15, 8, 8]

        supplement_mask = other_info['supplement_mask']  # (1, 1, 64, 64)
        supplement_mask = F.interpolate(supplement_mask, size=(H, W), mode='bilinear')  # (1, 1, 8, 8)

        image_token = other_info['image_token'] # [1, 16, 64, 1280]
        assert image_token.shape == ca_x.shape

        context_pooler = other_info['context_pooler'] # [16, 1, 768]
        box = other_info['box'] # [1, 15, 8]
        box = box.view(B * instance_num, 1, -1) # [15, 1, 8]
        box_token = self.pos_net(box) # [15, 1, 768]
        
        # add reference image feature as condition
        img_features, bg_features = other_info['ref_features'] # [15, 16, 768], [1, 16, 768]
        
        context_fg = torch.cat([context_pooler[1:, ...], img_features, box_token], dim=1)      # [15, 1 (text) + 16 (ref) + 1 (box), 768]
        ea_x, anchor_attn = self.ea_obj(self.norm_obj(image_token[:, 1:, ...].view(B * instance_num, HW, C)),    # [15, 64, 1280]
                                context=context_fg, return_attn=True)                   # ea_x.shape [15, 64, 1280]
        ea_x = ea_x.view(B, instance_num, HW, C)                                        # ea_x.shape [1, 15, 64, 1280]
        sigmoid_values = other_info['sigmoid_values']                                   # [1, 15, 64, 64]
        sigmoid_values = F.interpolate(sigmoid_values, size=(H, W), mode='bilinear')    # [1, 15, 8, 8]
        ea_x = ea_x * sigmoid_values.view(B, instance_num, HW, 1)                       # [1, 15, 64, 1280] * [1, 15, 64, 1]
        ca_x[:, 1:, ...] = ca_x[:, 1:, ...] * sigmoid_values.view(B, instance_num, HW, 1)  # (B, phase_num, HW, C)
        ca_x[:, 1:, ...] = ca_x[:, 1:, ...] + ea_x                                      # ca_x.shape [1, 16, 64, 1280]
        
        context_bg = torch.cat([context_pooler[[0], ...], bg_features], dim=1)                 # [1, 1 (text) + 16 (ref), 768]
        ea_x_bg, _ = self.ea2(self.norm2(ca_x[:, 1:, ...].sum(dim=1, keepdim=True).view(B * 1, HW, C)),
                             context=context_bg, return_attn=True)                      # [1, 64, 1280]
        ca_x[:, 0, ...] = ca_x[:, 0, ...] + ea_x_bg                                     # [1, 64, 1280]
        
        # image_token[:, 0, ...].shape [1, 64, 1280] ; torch.cat([guidance_mask[:, :, ...], supplement_mask], dim=1).shape [1, 1 + 15, 8, 8]
        fusion_template = self.la(x=image_token[:, 0, ...], guidance_mask=torch.cat([guidance_masks[:, :, ...], supplement_mask], dim=1))  # [1, 64, 1280]
        fusion_template = fusion_template.view(B, 1, HW, C)     # [1, 1, 64, 1280]
        ca_x = torch.cat([ca_x, fusion_template], dim = 1)      # [1, 17, 64, 1280]
        out = torch.sum(ca_x, dim=1)                            # [1, 64, 1280]
        return out

class MIFusionPrototype(nn.Module):
    def __init__(self, C, attn_type='base', context_dim=768, heads=8, prototype_dim=1024):
        super().__init__()
        self.prototype_attn = CrossAttention(query_dim=C, context_dim=prototype_dim,
                                            heads=heads, dim_head=C // heads,
                                            dropout=0.0)
        self.norm_prototype = nn.LayerNorm(C)
        self.la = LayoutAttention(query_dim=C, heads=heads, 
                                  dim_head=C // heads, dropout=0.0)
        self.gating_param = nn.Parameter(torch.zeros(1))
        self._init_novel_weights()

    def _init_novel_weights(self):
        if hasattr(self.prototype_attn, 'to_out'):
            output_layer = self.prototype_attn.to_out[0] if isinstance(self.prototype_attn.to_out, nn.Sequential) or isinstance(self.prototype_attn.to_out, nn.ModuleList) else self.prototype_attn.to_out
            nn.init.zeros_(output_layer.weight)
            nn.init.zeros_(output_layer.bias)
            if hasattr(self.prototype_attn, 'to_q'):
                nn.init.xavier_uniform_(self.prototype_attn.to_q.weight)
            if hasattr(self.prototype_attn, 'to_k'):
                nn.init.xavier_uniform_(self.prototype_attn.to_k.weight)
            if hasattr(self.prototype_attn, 'to_v'):
                nn.init.xavier_uniform_(self.prototype_attn.to_v.weight)
        if hasattr(self.la, 'to_out'):
            output_layer = self.la.to_out[0] if isinstance(self.la.to_out, nn.Sequential) or isinstance(self.la.to_out, nn.ModuleList) else self.la.to_out
            nn.init.zeros_(output_layer.weight)
            nn.init.zeros_(output_layer.bias)

    def forward(self, ca_x, other_info):
        height, width = other_info['height'], other_info['width'] # 512, 512
        instance_num = other_info['instance_num'] # 15
        B, _, HW, C = ca_x.shape      # [1, 16, 64, 1280]
        down_scale = int(math.sqrt(height * width // ca_x.shape[2])) # 64
        H = height // down_scale # 8
        W = width // down_scale # 8

        guidance_masks = other_info['guidance_masks']
        guidance_masks = F.interpolate(guidance_masks, size=(H, W), mode='bilinear')   # [1, 15, 8, 8]

        supplement_mask = other_info['supplement_mask']  # (1, 1, 64, 64)
        supplement_mask = F.interpolate(supplement_mask, size=(H, W), mode='bilinear')  # (1, 1, 8, 8)

        sigmoid_values = other_info['sigmoid_values']                                   # [1, 15, 64, 64]
        sigmoid_values = F.interpolate(sigmoid_values, size=(H, W), mode='bilinear')    # [1, 15, 8, 8]

        image_token = other_info['image_token'] # [1, 16, 64, 1280]
        assert image_token.shape == ca_x.shape
        
        prototypes = other_info['prototypes'] # [15, 64, 768]

        base, side_feats_input = ca_x[:, 0, ...].clone(), ca_x[:, 1:, ...].clone()

        mask_flat = sigmoid_values.view(B * instance_num, -1)
        x_flat = image_token[:, 1:, ...].reshape(B * instance_num, HW, C)

        keep_k = 256
        _, topk_indices = torch.topk(mask_flat, k=keep_k, dim=1)
        gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, C)
        x_selected = torch.gather(x_flat, 1, gather_indices)
        ea_selected, primitive_attn = self.prototype_attn(self.norm_prototype(x_selected), context=prototypes, return_attn=True)
        ea_full = torch.zeros_like(x_flat) 
        ea_full.scatter_(1, gather_indices, ea_selected)
        ea_x = ea_full.view(B, instance_num, HW, C)

        ea_x = ea_x * sigmoid_values.view(B, instance_num, HW, 1)                       # [1, 15, 64, 1280] * [1, 15, 64, 1]
        side_feats_gated = side_feats_input * sigmoid_values.view(B, instance_num, HW, 1)  # (B, phase_num, HW, C)
        total_side_residual = side_feats_gated + ea_x                                      # ca_x.shape [1, 16, 64, 1280]
        
        # image_token[:, 0, ...].shape [1, 64, 1280] ; torch.cat([guidance_mask[:, :, ...], supplement_mask], dim=1).shape [1, 1 + 15, 8, 8]
        fusion_template = self.la(x=image_token[:, 0, ...], guidance_mask=torch.cat([guidance_masks[:, :, ...], supplement_mask], dim=1))  # [1, 64, 1280]
        fusion_template = fusion_template.view(B, 1, HW, C)     # [1, 1, 64, 1280]
        
        final_residual = torch.sum(total_side_residual, dim=1) + fusion_template.squeeze(1)
        out = base + self.gating_param * final_residual
        return out