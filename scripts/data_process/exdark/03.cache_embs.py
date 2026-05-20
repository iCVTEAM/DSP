import os, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPModel, AutoTokenizer, AutoProcessor
from PIL import Image

PROJECT_DIR = os.getenv('DSP_PROJECT_DIR', '/path/to/DSP_PROJECT_DIR') # Set this manually if the environment variable is unavailable
data_setting_path = os.path.join(PROJECT_DIR, 'data', 'EXDARK', 'metadatas', 'data_setting1')
cache_name = os.path.join(PROJECT_DIR, 'data', 'EXDARK', 'exdark_emb.pt')

class myCLIPEnc(nn.Module):
    def __init__(self, model_config='openai/clip-vit-large-patch14', device='cuda'):
        super().__init__()
        self.device = device
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_config)
        self.processor = AutoProcessor.from_pretrained(model_config)
        self.model = CLIPModel.from_pretrained(model_config).to(device)
        self.model.eval()

    def forward(self, caption=None, img=None):
        if caption is not None:
            txt_inp  = self.tokenizer(caption, padding=True, truncation=True, return_tensors="pt").to(self.device) # pad and truncate to the max_length
            txt_feat = self.model.get_text_features(**txt_inp)
            txt_feat = F.normalize(txt_feat, dim=-1).detach().cpu()
        else:
            txt_feat = None

        if img is not None:
            img_inp  = self.processor(images=img, return_tensors="pt").to(self.device)
            img_feat = self.model.get_image_features(**img_inp)
            img_feat = F.normalize(img_feat, dim=-1).detach().cpu()
        else:
            img_feat = None
        
        return txt_feat, img_feat
    

if __name__ == '__main__':
    if not os.path.exists(cache_name):
        myCLIP = myCLIPEnc()

        data = []
        with open(os.path.join(data_setting_path, 'train_base.jsonl'), 'r') as f:
            for line in f:
                data.append(json.loads(line))
        emb_dict = {}
        sample_data = random.sample(data, 1600)
        for sample in tqdm(sample_data):
            img_name = sample['file_name']
            emb_dict[os.path.basename(img_name)] = {}

            caption = sample['captions'][0]
            img = Image.open(os.path.join(data_setting_path, img_name)).convert('RGB')
            txt_emb, img_emb = myCLIP(caption=caption, img=img)
            
            emb_dict[os.path.basename(img_name)]['txt_emb'] = txt_emb.detach().cpu()
            emb_dict[os.path.basename(img_name)]['img_emb'] = img_emb.detach().cpu()
            # torch.cuda.empty_cache()

        torch.save(emb_dict, cache_name)