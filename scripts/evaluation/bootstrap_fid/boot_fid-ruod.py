import os
os.environ["OMP_NUM_THREADS"] = "4" 
os.environ["MKL_NUM_THREADS"] = "4" 
os.environ["NUMEXPR_NUM_THREADS"] = "4" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
import glob
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import hashlib
from scipy import linalg
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json

class InceptionV3FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        inception.fc = nn.Identity()
        inception.eval()
        self.inception = inception

    def forward(self, x):
        return self.inception(x)

def get_transforms(resize_size=299):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class SimpleImageDataset(Dataset):
    def __init__(self, file_paths, transform):
        self.files = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img)
    
def get_cache_path(file_paths, cache_dir, prefix="feat"):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    
    sorted_paths = sorted(file_paths)
    path_str = "".join(sorted_paths).encode('utf-8')
    path_hash = hashlib.md5(path_str).hexdigest()
    
    filename = f"{prefix}_{path_hash}.npy"
    return os.path.join(cache_dir, filename)

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.iscomplexobj(covmean.diagonal()):
            covmean = covmean.real
        else:
            covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def extract_features(file_paths, batch_size=64, device='cuda', dims=2048, cache_path=None):
    if cache_path is not None and os.path.exists(cache_path):
        print(f"Found cache: {cache_path}")
        print("Loading features from file (skipping inference)...")
        try:
            features = np.load(cache_path)
            if features.shape[0] == len(file_paths):
                return features
            else:
                print("Cache size mismatch (files changed?), recalculating...")
        except Exception as e:
            print(f"Error loading cache: {e}, recalculating...")

    transform = get_transforms()
    dataset = SimpleImageDataset(file_paths, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = InceptionV3FeatureExtractor().to(device)
    
    pred_arr = np.empty((len(file_paths), dims))
    start_idx = 0
    
    print(f"Processing {len(file_paths)} images...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            features = model(batch) 
            
            features = features.cpu().numpy()
            
            pred_arr[start_idx:start_idx + features.shape[0]] = features
            start_idx = start_idx + features.shape[0]
    
    if cache_path is not None:
        print(f"Saving features to {cache_path}...")
        np.save(cache_path, pred_arr)
            
    return pred_arr

def bootstrap_fid_analysis(real_paths, gen_paths, cache_dir, num_bootstraps=100, sample_size=None, device='cuda', seed=64):

    rng = np.random.RandomState(seed)

    print("--- Extracting Real Features ---")
    real_cache_file = get_cache_path(real_paths, cache_dir, prefix="real_feats")
    real_feats = extract_features(real_paths, cache_path=real_cache_file, device=device)
    
    print("--- Extracting Fake Features (All Seeds) ---")
    gen_cache_file = get_cache_path(gen_paths, cache_dir, prefix="gen_feats")
    gen_feats = extract_features(gen_paths, cache_path=gen_cache_file, device=device)
    
    if sample_size is None:
        sample_size = min(len(real_paths), len(gen_paths))
    print(f"--- Starting Bootstrap (K={num_bootstraps}, Sample Size={sample_size}) ---")

    fids = []

    mu_real = np.mean(real_feats, axis=0)
    sigma_real = np.cov(real_feats, rowvar=False)
    
    for k in (pbar := tqdm(range(num_bootstraps), desc="Bootstrapping FID")):
        idx_gen = rng.choice(gen_feats.shape[0], sample_size, replace=True)
        feat_gen_subset = gen_feats[idx_gen]
        
        mu_gen = np.mean(feat_gen_subset, axis=0)
        sigma_gen = np.cov(feat_gen_subset, rowvar=False)
        
        fid_value = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
        fids.append(fid_value)

        current_mean = np.mean(fids)
        pbar.set_postfix({
            "cur": f"{fid_value:.2f}",
            "avg": f"{current_mean:.2f}"
        })
        
    fids = np.array(fids)
    return fids.mean(), fids.std()

if __name__ == "__main__":
    PROJECT_DIR = os.getenv('DSP_PROJECT_DIR', '/path/to/DSP_PROJECT_DIR') # Set this manually if the environment variable is unavailable

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_dir', type=str, default=os.path.join(PROJECT_DIR, 'data/RUOD/metadatas/data_setting1'), help='Ref data directory')
    parser.add_argument('--gen_root', type=str, default=os.path.join(PROJECT_DIR, 'outputs'), help='Generated root directory')
    parser.add_argument('--metric_dir', type=str, default=os.path.join(PROJECT_DIR, 'metrics/BootstrapFID/ruod'))
    parser.add_argument('--cache_dir', type=str, default='./cache', help='Cache directory')
    parser.add_argument('--sample_size', type=int, default=2000, help='FID sample size')
    parser.add_argument('--iter', type=int, default=50, help='Bootstrap repeated iterations')
    parser.add_argument('--config', type=str, default='dsp-ruod')
    parser.add_argument('-r', '--run_id', type=int, default=1)
    parser.add_argument('-k', '--k_shot', type=int, default=5)
    parser.add_argument('-n', '--num_seeds', type=int, default=50)
    parser.add_argument('-c', '--ckpt', type=int, default=100)
    args = parser.parse_args()

    real_files = []
    jsonl_files = ['test_novel_corals.jsonl', 'test_novel_cuttlefish.jsonl', 'test_novel_jellyfish.jsonl', 'test_novel_turtle.jsonl']
    for jf in jsonl_files:
        with open(os.path.join(args.ref_dir, jf), 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                data = json.loads(line)
                real_files.append(os.path.normpath(os.path.join(args.ref_dir, data['file_name'])))
    
    gen_files = []
    seeds = list(map(lambda s: s.strip(), open('seeds-aaa.txt', 'r').readlines()))
    for seed in seeds[:args.num_seeds]:
        gen_files.extend(glob.glob(os.path.join(args.gen_root, args.config, 'novel', f'run-{args.run_id}', f'{args.k_shot}-shot', f'shuffle_seed-{seed}', f'checkpoint-{args.ckpt}', 'image', '*.jpg'), recursive=True))


    print(f"Found {len(real_files)} Real images.")
    print(f"Found {len(gen_files)} Gen images (across all seeds).")

    if len(gen_files) < args.sample_size:
        print(f"Warning: Total gen images ({len(gen_files)}) < sample size ({args.sample_size}). Using full set size.")
        args.sample_size = len(gen_files)

    mean_fid, std_fid = bootstrap_fid_analysis(
        real_files, 
        gen_files, 
        cache_dir=args.cache_dir,
        num_bootstraps=args.iter, 
        sample_size=args.sample_size
    )

    print(f"\nFinal Result: FID = {mean_fid:.4f} ± {std_fid:.4f}")
    
    os.makedirs(args.metric_dir, exist_ok=True)
    output_filename = os.path.join(args.metric_dir, f'{args.config}-{args.k_shot}shot-run{args.run_id}-ckpt{args.ckpt}-Bootstrap_FID-{args.num_seeds}.txt')
    with open(output_filename, 'w') as f:
        f.write(f"Mean: {mean_fid}\nStd: {std_fid}\nSample_Size: {args.sample_size}\nIters: {args.iter}")