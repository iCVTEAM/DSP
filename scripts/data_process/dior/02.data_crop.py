import os
import json
import cv2
import torch
import numpy as np
from collections import defaultdict
from torchvision import transforms
from PIL import Image
import xml.etree.ElementTree as ET
from collections import Counter

PROJECT_DIR = os.getenv('DSP_PROJECT_DIR', '/path/to/DSP_PROJECT_DIR') # Set this manually if the environment variable is unavailable
base_dir = '/path/to/DIOR-VOC/Annotations/Horizontal_Bounding_Boxes' # Replace with your actual path
image_dir = '/path/to/DIOR-VOC/VOC2007/JPEGImages' # Replace with your actual path

category_list = [
    'vehicle', 'baseballfield', 'groundtrackfield', 'windmill', 'bridge',
    'overpass', 'ship', 'airplane', 'tenniscourt', 'airport',
    'expressway-service-area', 'basketballcourt', 'stadium', 'storagetank', 'chimney',
    'dam', 'expressway-toll-station', 'golffield', 'trainstation', 'harbor'
]
category_dict_rev = {v: i for i, v in enumerate(category_list)}
width_height = 800

output_dir = os.path.join(PROJECT_DIR, "data", "DIOR", "patches")

if __name__ == '__main__':
    os.makedirs(output_dir, exist_ok=True)
    counter = Counter()
    filenames = sorted(os.listdir(base_dir))[:5862]
    for filename in filenames:
        dictin = {}
        image = cv2.imread(os.path.join(image_dir, f'{os.path.splitext(filename)[0]}.jpg'))
        root = ET.parse(os.path.join(base_dir, filename)).getroot()
        categories, bndboxes, obndboxes= [], [], []
        for i, object in enumerate(root.findall('object')):
            category = object.find('name').text.lower()
            category_id = category_dict_rev[category]
            xmin, ymin, xmax, ymax = [int(child.text) for child in object.find('bndbox')]

            bbox_width = xmax - xmin
            bbox_height = ymax - ymin
            
            bbox_area = bbox_width * bbox_height
            image_area = width_height * width_height
            bbox_ratio = bbox_area / image_area
            
            if bbox_ratio < 0.0005:
                continue
            class_dir = os.path.join(output_dir, category)
            os.makedirs(class_dir, exist_ok=True)

            counter[class_dir] += 1
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            cropped_image = image[ymin:ymax, xmin:xmax]
            
            output_image_name = f"{os.path.splitext(filename)[0]}_{i}.jpg"
            output_image_path = os.path.join(class_dir, output_image_name)
            
            cv2.imwrite(output_image_path, cropped_image)
            
            
    print(counter)