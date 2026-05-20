import albumentations as A
from torchvision import transforms
from PIL import Image
import numpy as np
import functools
import imagesize
import torch
import cv2
import os


class LayoutTransform:
    def __init__(self, config, image_processor, split, ref_table=None, filter_dict=None):
        self.split, self.phase = split, config.phase
        if split == 'train' and self.phase == 'novel':
            self.image_transforms = A.Compose([
                A.OneOf([
                    A.RandomSizedBBoxSafeCrop(height=config.dataset.resolution, width=config.dataset.resolution, erosion_rate=0.0, interpolation=cv2.INTER_CUBIC, p=0.3),
                    A.Resize(height=config.dataset.resolution, width=config.dataset.resolution, interpolation=cv2.INTER_CUBIC, p=0.7),
                ], p=1.0),
                A.Normalize(mean=[0.5], std=[0.5]),
                A.pytorch.ToTensorV2(),
            ], bbox_params=A.BboxParams(format='albumentations', label_fields=['labels'], min_area=0, min_visibility=0.0))
        elif split == 'infer' or self.phase == 'base':
            self.image_transforms = A.Compose([
                A.Resize(config.dataset.resolution, config.dataset.resolution),
                A.Normalize(mean=[0], std=[1]),
                A.pytorch.ToTensorV2(),
            ])
        else:
            raise ValueError("Invalid mode for Transform.")
        self.image_patch_path = config.dataset.image_patch_path
        self.ref_resolution = config.dataset.ref_resolution
        self.image_column, self.caption_column, self.bbox_column, self.obbox_column = config.dataset.column_names
        self.image_processor = image_processor
        if ref_table is not None:
            self.ref_table = ref_table
        else:
            self.categories = config.dataset.categories[self.phase]
            self.filter_dict = filter_dict
            self.ref_table = self.build_ref_table()
        self.k_shot = config.dataset.novel_settings.k_shot
        self.top_k = 1

    def build_ref_table(self):
        ref_table = {}
        for category in self.categories:
            category_dir = os.path.join(self.image_patch_path, category)
            patch_list = os.listdir(category_dir)
            # Filter the patch list to avoid data leakage of few-shot learning
            if self.phase == 'novel':
                assert self.filter_dict is not None
                patch_list = list(filter(lambda patch_name: patch_name.split('_')[0] in self.filter_dict[category], patch_list))
            patch_list = sorted(patch_list, key = lambda img: functools.reduce(lambda x, y: x*y, imagesize.get(os.path.join(category_dir, img))), reverse=True)
            ref_table[category] = {img: functools.reduce(lambda x, y: x/y, imagesize.get(os.path.join(category_dir, img))) for img in patch_list[:200]}
        return ref_table

    @staticmethod
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array/value - 1)).argmin()
        return idx
    
    @staticmethod
    def find_k_nearest(array, value, k=1):
        array = np.asarray(array)
        dist = np.abs(array / value - 1)
        idxs = np.argsort(dist)[:k]
        return idxs
    
    def get_instances(self, examples):
        instances = []
        for index, (caption, bboxes) in enumerate(zip(examples[self.caption_column], examples[self.bbox_column])):
            categories = caption[1:]
            instances_per_example = []
            for name, bbox in zip(categories, bboxes):
                if name == '':
                    instances_per_example.append(torch.zeros([self.top_k, 3, self.ref_resolution, self.ref_resolution]))
                else:
                    value = (bbox[2] - bbox[0]) / max(bbox[3] - bbox[1], 1e-8)
                    chosen_idxs = self.find_k_nearest(list(self.ref_table[name].values()), value, k=self.top_k)
                    instances_per_bbox = []
                    for idx in chosen_idxs:
                        chosen_file = list(self.ref_table[name].keys())[idx]
                        img = Image.open(os.path.join(self.image_patch_path, name, chosen_file)).convert('RGB')
                        img = self.image_processor(images=img, return_tensors="pt")['pixel_values'].squeeze(0)
                        instances_per_bbox.append(img)
                    instances_per_example.append(torch.stack(instances_per_bbox))
            instances.append(torch.stack(instances_per_example))
        return instances
    
    def train_transform(self, examples):
        images, bboxes, obboxes = examples[self.image_column], examples[self.bbox_column], examples[self.obbox_column]
        global_prompt = [caption[:1] for caption in examples[self.caption_column]]
        captions = [caption[1:] for caption in examples[self.caption_column]]

        new_images, new_bboxes, new_obboxes, new_captions = [], [], [], []
        for i in range(len(images)):
            num_instances = sum(bool(s) for s in captions[i])
            bboxes_i = bboxes[i][:num_instances]
            captions_i = captions[i][:num_instances]
            transformed = self.image_transforms(image=images[i], bboxes=bboxes_i, labels=captions_i)
            transformed["obboxes"] = []
            for xmin, ymin, xmax, ymax in transformed["bboxes"]:
                transformed["obboxes"].append([xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax])
            for _ in range(len(captions[i]) - num_instances):
                transformed["bboxes"].append([0,0,0,0])
                transformed["obboxes"].append([0,0,0,0,0,0,0,0])
                transformed["labels"].append("")
            new_images.append(transformed["image"])
            new_bboxes.append(transformed["bboxes"])
            new_obboxes.append(transformed["obboxes"])
            new_captions.append(global_prompt[i] + transformed["labels"])
        examples[self.image_column] = new_images
        examples[self.bbox_column] = new_bboxes
        examples[self.obbox_column] = new_obboxes
        examples[self.caption_column] = new_captions
        return examples

    def infer_transform(self, examples):
        examples[self.image_column] = [self.image_transforms(image=image)['image'] for image in examples[self.image_column]]
        return examples

    def __call__(self, examples):
        # print("Received keys:", examples.keys())
        examples[self.image_column] = [np.array(image.convert("RGB")) for image in examples[self.image_column]]
        if self.split == "train":
            examples = self.train_transform(examples)
        elif self.split == "infer":
            examples = self.infer_transform(examples)
        examples["instances"] = self.get_instances(examples)
        return examples