import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from .kmeans_pytorch import kmeans

from datamodules import RefTable
from PIL import Image
import os

from utils import Dict

class PrototypeLearner(nn.Module):
    def __init__(self, num_prototypes=64, prototype_dim=1024, lambda_reg=0.1, lr=0.01, iter_steps=50, verbose=True):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.prototype_dim = prototype_dim
        self.lambda_reg = lambda_reg
        self.lr = lr
        self.iter_steps = iter_steps
        self.verbose = verbose
        
    def get_initial_prototypes(self, features, device):
        # features: [N_total, D]
        feats_norm = F.normalize(features, p=2, dim=-1)
        _, centers = kmeans(
            X=feats_norm,
            num_clusters=self.num_prototypes,
            distance='cosine',
            device=device,
            tqdm_flag=False
        )
        return centers.to(device) # [K, D]
    
    def calculate_reconstruction_loss(self, targets, protos):
        protos_norm = F.normalize(protos, p=2, dim=-1)
        
        # Formula: W = T * P^T * (P * P^T + lambda * I)^(-1)
        # Ref: Eq. (2) in paper 

        # P * P^T: Gram Matrix [K, K], (P * P^T + lambda * I)^(-1)
        p_gram = torch.matmul(protos_norm, protos_norm.t())
        identity = torch.eye(self.num_prototypes, device=targets.device)
        inverse_term = torch.inverse(p_gram + self.lambda_reg * identity)
        
        # Mapping weights [N, K]
        # W = T * P^T * Inverse
        mapping_weights = torch.matmul(targets, protos_norm.t())
        mapping_weights = torch.matmul(mapping_weights, inverse_term)
        
        # Reconstruct: T_hat = W * P
        reconstructed = torch.matmul(mapping_weights, protos_norm)
        
        return F.mse_loss(reconstructed, targets)

    def forward(self, features):
        """
        features: [N_total, D]
        Return: Optimized Prototypes [K, D]
        """
        device = features.device
        N, D = features.shape
        
        targets = F.normalize(features, p=2, dim=-1).detach()
        
        initial_protos = self.get_initial_prototypes(features, device)
        initial_protos_static = initial_protos.clone().detach()

        with torch.no_grad():
            baseline_loss = self.calculate_reconstruction_loss(targets, initial_protos_static)

        if self.verbose:
            print(f"\n[ProtoLearner] Start. Baseline (K-Means) Reconstruction MSE: {baseline_loss.item():.6f}")
        
        prototypes = nn.Parameter(initial_protos.clone())
        optimizer = optim.Adam([prototypes], lr=self.lr)
        
        for i in range(self.iter_steps):
            optimizer.zero_grad()
            loss = self.calculate_reconstruction_loss(targets, prototypes)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                prototypes.data.copy_(F.normalize(prototypes.data, p=2, dim=-1))
            
            if self.verbose and (i == 0 or (i + 1) % 10 == 0 or i == self.iter_steps - 1):
                with torch.no_grad():
                    curr_protos_norm = F.normalize(prototypes, p=2, dim=-1)
                    init_protos_norm = F.normalize(initial_protos_static, p=2, dim=-1)
                    
                    shift_dist = torch.norm(curr_protos_norm - init_protos_norm, dim=-1).mean().item()
                    
                    cosine_sim = F.cosine_similarity(curr_protos_norm, init_protos_norm, dim=-1).mean().item()
                    
                    gain_pct = (baseline_loss.item() - loss.item()) / baseline_loss.item() * 100
                    
                print(f"Iter {i+1:02d}/{self.iter_steps} | "
                      f"Loss: {loss.item():.6f} | "
                      f"Gain: {gain_pct:.2f}% | "
                      f"Shift(L2): {shift_dist:.4f} | "
                      f"Sim(Cos): {cosine_sim:.4f}")

        return F.normalize(prototypes, p=2, dim=-1).detach()


class PrototypeBank(nn.Module):
    def __init__(self, config, image_encoder, image_processor, num_prototypes=128, prototype_dim=1024):
        super().__init__()

        self.config = config
        self.num_prototypes = num_prototypes
        self.prototype_dim = prototype_dim
        self.image_patch_path = config.dataset.image_patch_path
        self.categories = config.dataset.categories[config.phase]
        self.aux = Dict(image_processor=image_processor, image_encoder=image_encoder)

        shape = (len(self.categories) + 1, num_prototypes, prototype_dim)
        self.register_buffer('prototypes', torch.zeros(shape))
        self.register_buffer('prototype_flag', torch.tensor(False))

        self.learner = PrototypeLearner(
            num_prototypes=num_prototypes,
            prototype_dim=prototype_dim,
            lambda_reg=0.1,
            lr=0.05,
            iter_steps=50
        )

    @property
    def image_processor(self):
        return self.aux.image_processor

    @property
    def image_encoder(self):
        return self.aux.image_encoder

    def build_prototypes(self):
        self.ref_table = RefTable()()
        self.cate_to_id = {cate: idx for (idx, cate) in enumerate(self.categories)}
        self.cate_to_id[''] = -1

        if self.prototype_flag.item():
            if (dist.is_initialized() and dist.get_rank() == 0) or (not dist.is_initialized()):
                print(f"[PrototypeBank] Prototypes loaded from checkpoint (Frozen). Skip calculation.")
            return

        is_dist = dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0
        device = next(iter(self.image_encoder.parameters())).device
        temp_prototypes = None
        
        if rank == 0:
            print("[PrototypeBank] Calculating prototypes online...")
            self.dense_features = {}
            for category in self.categories:
                files = list(self.ref_table[category].keys())
                ref_images = []
                for file in files:
                    img = Image.open(os.path.join(self.image_patch_path, category, file)).convert('RGB')
                    img = self.image_processor(images=img, return_tensors="pt", do_normalize=False)['pixel_values'].squeeze(0)
                    ref_images.append(img)
                self.dense_features[category] = self.image_encoder(torch.stack(ref_images).to(device), mode='x_norm_patchtokens')

            prototype_list = []
            for category in self.categories:
                flat_feats = self.dense_features[category].reshape(-1, self.prototype_dim)                
                optimized_protos = self.learner(flat_feats)
                prototype_list.append(optimized_protos)
            
            prototype_list.append(torch.zeros_like(prototype_list[-1]))
            temp_prototypes = torch.stack(prototype_list)
            
            self.prototypes.copy_(temp_prototypes)
            self.prototype_flag.fill_(True)

        if is_dist:
            dist.broadcast(self.prototypes, src=0)
            dist.broadcast(self.prototype_flag, src=0)

    def get_prototypes(self, captions):
        return torch.stack(list(map(lambda caption: self.prototypes[self.cate_to_id[caption]], captions)))