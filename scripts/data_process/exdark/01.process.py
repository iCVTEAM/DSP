from pathlib import Path
from PIL import Image
import shutil
import os
import imagesize
import json

PROJECT_DIR = os.getenv('DSP_PROJECT_DIR', '/path/to/DSP_PROJECT_DIR') # Set this manually if the environment variable is unavailable
base_dir = '/path/to/ExDark' # Replace with your actual path
output_dir = os.path.join(PROJECT_DIR, 'data', 'EXDARK', 'images')

image_dir = os.path.join(base_dir, 'images')
anno_dir = os.path.join(base_dir, 'annos')

category_dict = {
    1: 'Bicycle', 2: 'Boat', 3: 'Bottle', 4: 'Bus', 5: 'Car', 6: 'Cat',
    7: 'Chair', 8: 'Cup', 9: 'Dog', 10: 'Motorbike', 11: 'People', 12: 'Table'
}
category_dict_rev = {v: i for i, v in category_dict.items()}

novel_categories = ['Bus', 'Dog', 'Motorbike', 'Table']
novel_ids = set([category_dict_rev[cate] for cate in novel_categories])
novel_dict_rev = {v: i for i, v in enumerate(novel_categories)}

split_dict = {
    1: 'train', 2: 'val', 3: 'test'
}

caption_prefix = "A dark image of "
thr = 15

os.makedirs(os.path.join(PROJECT_DIR, 'data', 'EXDARK', 'metadatas', 'data_setting1'), exist_ok=True)

def save_as_jpg(src_path, dst_dir):
    src = Path(src_path)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    if src.suffix == ".jpg":
        shutil.copy(src, dst_dir / src.name)
    else:
        img = Image.open(src).convert("RGB")
        img.save(dst_dir / (src.stem + ".jpg"), "JPEG")

if __name__ == '__main__':
    train_base_list, train_novel_list = [], [[] for i in range(len(novel_categories) + 1)]
    val_base_list, val_novel_list = [], [[] for i in range(len(novel_categories) + 1)]
    test_base_list, test_novel_list = [], [[] for i in range(len(novel_categories) + 1)]

    meta_base_list = [train_base_list, val_base_list, test_base_list]
    meta_novel_list = [train_novel_list, val_novel_list, test_novel_list]

    with open(os.path.join(base_dir, 'imageclasslist.txt'), 'r') as f:
        metadata = list(map(lambda line: line.strip().split(), f.readlines()[1:]))
        metadata = list(map(lambda line: [line[0]] + list(map(int, line[1:])), metadata))

    for data in metadata:
        image_file = os.path.join(base_dir, 'images', category_dict[data[1]], data[0])
        save_as_jpg(image_file, os.path.join(output_dir, split_dict[data[-1]]))
        width, height = imagesize.get(image_file)

        anno_file = os.path.join(base_dir, 'annos', category_dict[data[1]], f'{data[0]}.txt')
        with open(anno_file, 'r') as f:
            anno = list(map(lambda line: line.strip().split(), f.readlines()[1:]))
            anno = list(map(lambda line: [line[0]] + list(map(int, line[1:5])), anno))

        dictin = {}
        categories, bndboxes, obndboxes= [], [], []
        categories_in_this_image = set()
        for object in anno:
            category, xmin, ymin, w, h = object
            xmin, ymin, w, h = int(xmin), int(ymin), int(w), int(h)
            xmin, ymin, xmax, ymax = xmin, ymin, xmin + w, ymin + h
            xmin = xmin / width
            ymin = ymin / height
            xmax = xmax / width
            ymax = ymax / height
            categories.append(category)
            bndboxes.append([xmin, ymin, xmax, ymax])
            obndboxes.append([xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax])
            categories_in_this_image.add(category_dict_rev[category])
        
        is_novel = False
        if categories_in_this_image & set(novel_ids):
            is_novel = True
            tmp_categories, tmp_bndboxes, tmp_obndboxes = [], [], []
            for i in range(len(categories)):
                if categories[i] in novel_categories:
                    tmp_categories.append(categories[i])
                    tmp_bndboxes.append(bndboxes[i])
                    tmp_obndboxes.append(obndboxes[i])
            categories, bndboxes, obndboxes = tmp_categories, tmp_bndboxes, tmp_obndboxes

        categories = [cate.lower() for cate in categories]
        caption = [caption_prefix + ", ".join(categories)]
            
        if len(categories) > thr:
            categories = categories[:thr]
            bndboxes = bndboxes[:thr]
            obndboxes = obndboxes[:thr]
        while len(categories) < thr:
            categories.append("")
            bndboxes.append([0,0,0,0])
            obndboxes.append([0,0,0,0,0,0,0,0])

        dictin['file_name'] = f'../../images/{split_dict[data[-1]]}/{os.path.splitext(os.path.basename(image_file))[0]}.jpg'
        caplist = caption + categories
        dictin["captions"] = caplist
        dictin["bndboxes"] = bndboxes
        dictin["obboxes"] = obndboxes
 
        if is_novel:
            if len(categories_in_this_image) > 1:
                meta_novel_list[data[-1] - 1][-1].append(dictin.copy())
            else:
                meta_novel_list[data[-1] - 1][novel_dict_rev[category_dict[next(iter(categories_in_this_image))]]].append(dictin.copy())
        else:
            meta_base_list[data[-1] - 1].append(dictin.copy())
        # json_list[data[-1]].append(dictin)

    for i in range(1, 4):
        with open(os.path.join(PROJECT_DIR, f"data/EXDARK/metadatas/data_setting1/{split_dict[i]}_base.jsonl"), "w", encoding="utf-8") as f:
            for item in meta_base_list[i - 1]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
        for j in range(len(novel_categories)):
            with open(os.path.join(PROJECT_DIR, f"data/EXDARK/metadatas/data_setting1/{split_dict[i]}_novel_{novel_categories[j].lower()}.jsonl"), "w", encoding="utf-8") as f:
                for item in meta_novel_list[i - 1][j]:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        with open(os.path.join(PROJECT_DIR, f"data/EXDARK/metadatas/data_setting1/{split_dict[i]}_novel_mixed.jsonl"), "w", encoding="utf-8") as f:
            for item in meta_novel_list[i - 1][-1]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")