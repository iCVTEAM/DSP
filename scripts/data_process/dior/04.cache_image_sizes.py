import os
import json
import imagesize
from tqdm import tqdm

PROJECT_DIR = os.getenv('DSP_PROJECT_DIR', '/path/to/DSP_PROJECT_DIR') # Set this manually if the environment variable is unavailable
image_patch_path = os.path.join(PROJECT_DIR, "data", "DIOR", "patches")
output_path = os.path.join(image_patch_path, "image_sizes.json")

def generate_size_cache(image_patch_path, output_path):
    print(f"Scanning: {image_patch_path}")
    size_cache = {}
    
    categories = [d for d in os.listdir(image_patch_path) if os.path.isdir(os.path.join(image_patch_path, d))]
    
    for category in tqdm(categories, desc="Processing"):
        category_dir = os.path.join(image_patch_path, category)
        
        for item in os.listdir(category_dir):
            if item.endswith('.jpg'):
                img_path = os.path.join(category_dir, item)
                try:
                    size_cache[img_path] = imagesize.get(img_path)
                except Exception as e:
                    print(f"Failed {img_path}: {e}")
                    
        aug_dir = os.path.join(category_dir, 'augmented')
        if os.path.exists(aug_dir):
            for item in os.listdir(aug_dir):
                if item.endswith('.jpg'):
                    img_path = os.path.join(aug_dir, item)
                    try:
                        size_cache[img_path] = imagesize.get(img_path)
                    except Exception as e:
                        print(f"Failed {img_path}: {e}")

    with open(output_path, 'w') as f:
        json.dump(size_cache, f)
    print(f"\nSaved: {output_path} (Containing {len(size_cache)} images)")

if __name__ == "__main__":
    generate_size_cache(image_patch_path, output_path)