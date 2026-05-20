import os
import xml.etree.ElementTree as ET
import json

PROJECT_DIR = os.getenv('DSP_PROJECT_DIR', '/path/to/DSP_PROJECT_DIR') # Set this manually if the environment variable is unavailable
base_dir = '/path/to/DIOR-VOC/Annotations/Horizontal_Bounding_Boxes' # Replace with your actual path

category_list = [
    'vehicle', 'baseballfield', 'groundtrackfield', 'windmill', 'bridge',
    'overpass', 'ship', 'airplane', 'tenniscourt', 'airport',
    'expressway-service-area', 'basketballcourt', 'stadium', 'storagetank', 'chimney',
    'dam', 'expressway-toll-station', 'golffield', 'trainstation', 'harbor'
]
category_dict_rev = {v: i for i, v in enumerate(category_list)}

novel_categories = [
    'windmill', 'airport', 'chimney', 'dam', 'trainstation'
]
novel_ids = set([category_dict_rev[cate] for cate in novel_categories])
novel_dict_rev = {v: i for i, v in enumerate(novel_categories)}

num_classes = 20
width_height = 800
caption_prefix = "The aerial image features a city with "
thr = 15

os.makedirs(os.path.join(PROJECT_DIR, 'data', 'DIOR', 'metadatas', 'data_setting1'), exist_ok=True)

if __name__ == '__main__':
    base_list, novel_list = [], [[], [], [], [], [], []]
    filenames = sorted(os.listdir(base_dir))[:5862]
    for filename in filenames:
        dictin = {}
        root = ET.parse(os.path.join(base_dir, filename)).getroot()
        categories_in_this_image = set()
        categories, bndboxes, obndboxes= [], [], []
        for object in root.findall('object'):
            category = object.find('name').text.lower()
            category_id = category_dict_rev[category]
            xmin, ymin, xmax, ymax = [int(child.text) / width_height for child in object.find('bndbox')]
            categories.append(category)
            bndboxes.append([xmin, ymin, xmax, ymax])
            obndboxes.append([xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax])
            categories_in_this_image.add(category_id)

        # caption put to line 58
        caption = [caption_prefix + ", ".join(categories)]
        
        is_novel = False
        if categories_in_this_image & set(novel_ids):
            is_novel = True
            tmp_categories, tmp_bndboxes, tmp_obndboxes= [], [], []
            for i in range(len(categories)):
                if categories[i] in novel_categories:
                    tmp_categories.append(categories[i])
                    tmp_bndboxes.append(bndboxes[i])
                    tmp_obndboxes.append(obndboxes[i])
            categories, bndboxes, obndboxes = tmp_categories, tmp_bndboxes, tmp_obndboxes
            
        if len(categories) > thr:
            categories = categories[:thr]
            bndboxes = bndboxes[:thr]
            obndboxes = obndboxes[:thr]
        while len(categories) < thr:
            categories.append("")
            bndboxes.append([0,0,0,0])
            obndboxes.append([0,0,0,0,0,0,0,0])

        dictin["file_name"] = f"../../images/{filename[:5]}.jpg"
        caplist = caption + categories
        dictin["captions"] = caplist
        dictin["bndboxes"] = bndboxes
        dictin["obboxes"] = obndboxes

        if is_novel:
            if len(categories_in_this_image) > 1:
                novel_list[5].append(dictin.copy())
            else:
                novel_list[novel_dict_rev[category_list[next(iter(categories_in_this_image))]]].append(dictin.copy())
        else:
            base_list.append(dictin.copy())

    with open(os.path.join(PROJECT_DIR, "data/DIOR/metadatas/data_setting1/train_base.jsonl"), "w", encoding="utf-8") as f:
        for item in base_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    for i in range(5):
        with open(os.path.join(PROJECT_DIR, f"data/DIOR/metadatas/data_setting1/train_novel_{novel_categories[i]}.jsonl"), "w", encoding="utf-8") as f:
            for item in novel_list[i]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(os.path.join(PROJECT_DIR, "data/DIOR/metadatas/data_setting1/train_novel_mixed.jsonl"), "w", encoding="utf-8") as f:
        for item in novel_list[5]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        