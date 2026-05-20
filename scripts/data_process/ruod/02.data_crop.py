import os
import json
import cv2
import torch
from torchvision import transforms
from PIL import Image
from collections import defaultdict
import xml.etree.ElementTree as ET
from collections import Counter

PROJECT_DIR = os.getenv('DSP_PROJECT_DIR', '/path/to/DSP_PROJECT_DIR') # Set this manually if the environment variable is unavailable
image_dir = '/path/to/RUOD/RUOD_pic/train' # Replace with your actual path
label_dir = '/path/to/RUOD/RUOD_ANN' # Replace with your actual path

output_dir = os.path.join(PROJECT_DIR, "data", "RUOD", "patches")

os.makedirs(output_dir, exist_ok=True)

annos = json.load(open(os.path.join(label_dir, "instances_train.json"), "r"))


images_items = annos["images"]
annos_items = annos["annotations"]
cates_items = annos["categories"]
catemap = {}
for cate in cates_items:
    catemap[cate["id"]] = cate["name"]


files = [i["file_name"] for i in images_items]
labels = defaultdict(list)
for item in annos_items:
    image_id = item["image_id"]
    filename = files[image_id-1]
    labels[filename].append([catemap[item["category_id"]]] + item["bbox"])
            
print(len(files))
counter = Counter()
for image_name in files:
    if not image_name.endswith(".jpg"):
        continue
        
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape
    lines = labels[image_name]
    # if image_name == '008431.jpg':
    #     import pdb; pdb.set_trace()
        
    for i,line in enumerate(lines):
        parts = line
       
        class_name = parts[0]
        xmin, ymin, w, h = parts[1:]
        bbox_width = w
        bbox_height = h
        xmax = xmin + w
        ymax = ymin + h
        
        
        bbox_area = bbox_width * bbox_height
        image_area = image_width * image_height
        bbox_ratio = bbox_area / image_area
        
        if bbox_ratio < 0.001:
            continue
            
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        counter[class_dir] += 1
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        cropped_image = image[ymin:ymax, xmin:xmax]
        
        output_image_name = f"{image_name[:-4]}_{i}.jpg"
        output_image_path = os.path.join(class_dir, output_image_name)
        try:
            cv2.imwrite(output_image_path, cropped_image)
        except:
            import pdb; pdb.set_trace()
            print(bbox_ratio, xmin, ymin, xmax, ymax, image_name)
        
print(counter)        
