import functools
import imagesize
import torch
import os
import json


def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance


@singleton
class RefTable:
    def __init__(self, config, filter_dict=None):
        self.phase = config.phase
        self.categories_base, self.categories_novel = config.dataset.categories.base, config.dataset.categories.novel
        self.categories = config.dataset.categories.get('all', None) or (self.categories_novel + self.categories_base)
        self.image_patch_path = config.dataset.image_patch_path
        self.filter_dict = filter_dict
        self.augment = config.dataset.get('ref_augment', False)

        cache_path = os.path.join(self.image_patch_path, "image_sizes.json")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"File Not Found: {cache_path}")
        with open(cache_path, 'r') as f:
            self.img_size_cache = json.load(f)
        self.ref_table, self.base_ref_table, self.novel_ref_table = self.build_ref_table()

    def build_ref_table(self):
        ref_table, base_ref_table, novel_ref_table = {}, {}, {}
        for category in self.categories:
            category_dir = os.path.join(self.image_patch_path, category)
            patch_list = os.listdir(category_dir)
            patch_list = list(filter(lambda s: s.endswith('.jpg'), patch_list))

            # Filter the patch list to avoid data leakage of few-shot learning
            if self.phase == 'novel' and category in self.categories_novel:
                assert self.filter_dict is not None
                patch_list = list(filter(lambda patch_name: patch_name.rsplit('_', 1)[0] in self.filter_dict[category], patch_list))
                if self.augment:
                    aug_category_dir = os.path.join(self.image_patch_path, category, 'augmented')
                    aug_patch_list = os.listdir(aug_category_dir)
                    aug_patch_list = list(filter(lambda patch_name: patch_name.startswith(tuple(self.filter_dict[category])), aug_patch_list))
                    aug_patch_list = list(map(lambda patch_name: f"augmented/{patch_name}", aug_patch_list))
                    patch_list += aug_patch_list

            def get_size(img_name):
                return self.img_size_cache.get(img_name, [0, 1])

            patch_list = sorted(patch_list, key = lambda img: get_size(img)[0] * get_size(img)[1], reverse=True)

            # Only build novel ref table at novel phase
            if self.phase == 'novel' and category in self.categories_novel:
                novel_ref_table[category] = {
                    img: get_size(img)[0] / get_size(img)[1] for img in patch_list[:200]
                }
            elif category in self.categories_base:
                base_ref_table[category] = {
                    img: get_size(img)[0] / get_size(img)[1] for img in patch_list[:200]
                }

        ref_table = novel_ref_table | base_ref_table
        return ref_table, base_ref_table, novel_ref_table

    def __call__(self, phase=None):
        if phase is None:
            return self.ref_table
        if phase == 'base':
            return self.base_ref_table
        if phase == 'novel':
            return self.novel_ref_table