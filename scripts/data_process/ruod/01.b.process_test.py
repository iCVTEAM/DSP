import os
import xml.etree.ElementTree as ET
import json
from collections import defaultdict

PROJECT_DIR = os.getenv('DSP_PROJECT_DIR', '/path/to/DSP_PROJECT_DIR') # Set this manually if the environment variable is unavailable
base_dir = '/path/to/RUOD' # Replace with your actual path
anno_path = os.path.join(base_dir, 'RUOD_ANN', 'instances_test.json')

annotations = json.load(open(anno_path,"r"))
images_items = annotations["images"]
annos_items = annotations["annotations"]
cates_items = annotations["categories"]
category_dict = {}
for cate in cates_items:
    category_dict[cate["id"]] = cate["name"]

category_dict_rev = {v: i for i, v in category_dict.items()}

novel_categories = ['corals', 'cuttlefish', 'turtle', 'jellyfish',]
novel_ids = set([category_dict_rev[cate] for cate in novel_categories])
novel_dict_rev = {v: i for i, v in enumerate(novel_categories)}

num_classes = 10
caption_prefix = "An underwater image of "
thr = 15

os.makedirs(os.path.join(PROJECT_DIR, 'data', 'RUOD', 'metadatas', 'data_setting1'), exist_ok=True)

annos_dict = defaultdict(list)
for item in annos_items:
    image_id = item["image_id"]
    filename = images_items[image_id-1]['file_name']
    annos_dict[filename].append(item["bbox"] + [item["category_id"]])

if __name__ == '__main__':
    base_list, novel_list = [], [[] for i in range(len(novel_categories) + 1)]
    for image_item in images_items:
        dictin = {}
        dictin['file_name'] = image_item['file_name']
        width, height = image_item['width'], image_item['height']
        categories_in_this_image = set()
        categories, bndboxes, obndboxes= [], [], []
        annos = annos_dict[image_item['file_name']]
        for anno in annos:
            xmin, ymin, w, h, category_id = anno
            xmin, ymin, w, h = int(xmin), int(ymin), int(w), int(h)
            xmin, ymin, xmax, ymax = xmin, ymin, xmin + w, ymin + h
            xmin = xmin / width
            ymin = ymin / height
            xmax = xmax / width
            ymax = ymax / height
            categories.append(category_dict[category_id])
            bndboxes.append([xmin, ymin, xmax, ymax])
            obndboxes.append([xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax])
            categories_in_this_image.add(category_id)
        
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

        caption = [caption_prefix + ", ".join(categories)]
            
        if len(categories) > thr:
            categories = categories[:thr]
            bndboxes = bndboxes[:thr]
            obndboxes = obndboxes[:thr]
        while len(categories) < thr:
            categories.append("")
            bndboxes.append([0,0,0,0])
            obndboxes.append([0,0,0,0,0,0,0,0])

        dictin["file_name"] = f"../../images/test/{image_item['file_name']}"
        caplist = caption + categories
        dictin["captions"] = caplist
        dictin["bndboxes"] = bndboxes
        dictin["obboxes"] = obndboxes
        
        if is_novel:
            if len(categories_in_this_image) > 1:
                novel_list[-1].append(dictin.copy())
            else:
                novel_list[novel_dict_rev[category_dict[next(iter(categories_in_this_image))]]].append(dictin.copy())
        else:
            base_list.append(dictin.copy())

    with open(os.path.join(PROJECT_DIR, "data/RUOD/metadatas/data_setting1/test_base.jsonl"), "w", encoding="utf-8") as f:
        for item in base_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    for i in range(len(novel_categories)):
        with open(os.path.join(PROJECT_DIR, f"data/RUOD/metadatas/data_setting1/test_novel_{novel_categories[i]}.jsonl"), "w", encoding="utf-8") as f:
            for item in novel_list[i]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(os.path.join(PROJECT_DIR, "data/RUOD/metadatas/data_setting1/test_novel_mixed.jsonl"), "w", encoding="utf-8") as f:
        for item in novel_list[-1]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        