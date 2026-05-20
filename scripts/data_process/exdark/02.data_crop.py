from pathlib import Path
from PIL import Image
import shutil
import os
import imagesize
import json
import cv2
from collections import Counter


PROJECT_DIR = os.getenv('DSP_PROJECT_DIR', '/path/to/DSP_PROJECT_DIR') # Set this manually if the environment variable is unavailable
base_dir = '/path/to/ExDark' # Replace with your actual path
output_dir = os.path.join(PROJECT_DIR, "data", "EXDARK", "patches")

image_dir = os.path.join(base_dir, 'images')
anno_dir = os.path.join(base_dir, 'annos')

category_dict = {
    1: 'Bicycle', 2: 'Boat', 3: 'Bottle', 4: 'Bus', 5: 'Car', 6: 'Cat',
    7: 'Chair', 8: 'Cup', 9: 'Dog', 10: 'Motorbike', 11: 'People', 12: 'Table'
}

if __name__ == '__main__':
    with open(os.path.join(base_dir, 'imageclasslist.txt'), 'r') as f:
        metadata = list(map(lambda line: line.strip().split(), f.readlines()[1:]))
        metadata = list(map(lambda line: [line[0]] + list(map(int, line[1:])), metadata))

    counter = Counter()
    for data in metadata:
        if data[-1] != 1:
            continue
        assert data[-1] == 1

        image_file = os.path.join(base_dir, 'images', category_dict[data[1]], data[0])
        # width, height = imagesize.get(image_file)
        image = cv2.imread(image_file)
        image_height, image_width, _ = image.shape

        anno_file = os.path.join(base_dir, 'annos', category_dict[data[1]], f'{data[0]}.txt')
        with open(anno_file, 'r') as f:
            anno = list(map(lambda line: line.strip().split(), f.readlines()[1:]))
            anno = list(map(lambda line: [line[0].lower()] + list(map(int, line[1:5])), anno))

        categories, bndboxes, obndboxes= [], [], []
        for i, object in enumerate(anno):
            category, xmin, ymin, w, h = object
            xmin, ymin, w, h = int(xmin), int(ymin), int(w), int(h)
            xmin = max(xmin, 0)
            xmin, ymin, xmax, ymax = xmin, ymin, xmin + w, ymin + h
            bbox_width, bbox_height = w, h
            
            bbox_area = bbox_width * bbox_height
            image_area = image_width * image_height
            bbox_ratio = bbox_area / image_area
            
            if bbox_ratio < 0.001:
                continue

            class_dir = os.path.join(output_dir, category)
            os.makedirs(class_dir, exist_ok=True)
            counter[class_dir] += 1

            cropped_image = image[ymin:ymax, xmin:xmax]
        
            output_image_name = f"{os.path.splitext(data[0])[0]}_{i}.jpg"
            output_image_path = os.path.join(class_dir, output_image_name)
            try:
                cv2.imwrite(output_image_path, cropped_image)
            except:
                import pdb; pdb.set_trace()
    print(counter)