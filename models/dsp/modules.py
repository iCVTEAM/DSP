import sys

import torch
import torch.nn as nn

from .dinov2 import hubconf

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

class FrozenDinoV2Encoder(AbstractEncoder):
    """
    Uses the DINOv2 encoder for image
    """
    def __init__(self, weight_path, device="cpu", freeze=True):
        super().__init__()
        dinov2 = hubconf.dinov2_vitl14(pretrained=False) 
        state_dict = torch.load(weight_path)
        dinov2.load_state_dict(state_dict, strict=False)
        self.model = dinov2.to(device)
        # self.device = device
        if freeze:
            self.freeze() 
        self.register_buffer('image_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('image_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))       
        # self.projector = nn.Linear(1536, 768)

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    def freeze(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    # image.shape [15, 3, 224, 224]
    def forward(self, image, mode=None):
        if isinstance(image,list):
            image = torch.cat(image,0)

        image = (image  - self.image_mean) / self.image_std
        features = self.model.forward_features(image)   # dict_keys(['x_norm_clstoken', 'x_norm_regtokens', 'x_norm_patchtokens', 'x_prenorm', 'masks'])
        
        if mode is not None:
            return features[mode]
        
        tokens = features["x_norm_patchtokens"]         # [15, 256, 1024]
        image_features  = features["x_norm_clstoken"]   # [15, 1024]
        image_features = image_features.unsqueeze(1)    # [15, 1, 1024]
        hint = torch.cat([image_features,tokens],1)     # [15, 257, 1024]
        # hint = self.projector(hint)
        return hint

    def encode(self, image):
        return self(image)