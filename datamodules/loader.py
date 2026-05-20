from . import transforms
from .ref_table import RefTable
from utils import get_ckpt_path
import databuilders
import datasets
import torch
import json
import os


class Loader:
    def __init__(self, config, image_processor, split='train', logger=None):
        self.logger = logger
        self.split, self.phase = split, config.phase
        self.data_name = config.dataset.name
        # self.data_files = config.dataset.data_files
        self.data_files = config.dataset.data_files[self.split][self.phase]
        self.categories = config.dataset.categories[self.phase]
        self.image_column, self.caption_column, self.bbox_column, self.obbox_column = config.dataset.column_names
        self.batch_size = config.training.batch_size
        self.num_workers = config.training.num_workers
        self.max_inference_size = config.inference.get('max_inference_size', None)
        self.novel_sample_dict = None

        self.set_dataset()
        if self.phase == 'novel':
            self.k_shot = config.dataset.novel_settings.k_shot
            self.shuffle_seed = config.dataset.novel_settings.get('shuffle_seed', 42)
            self.dump_file = os.path.join(get_ckpt_path(config), 'novel_sample_dict.json')
            if self.split == 'train':
                self.sample_dataset()
            elif self.split == 'infer':
                self.load_novel_sample_dict()
                self.sample_dataset_infer_phase()

        self.dataset = datasets.concatenate_datasets(self.dataset.values())
        self.ref_table = RefTable(config, filter_dict=self.novel_sample_dict)
        self.transform = getattr(transforms, config.dataset.get('transform', 'DefaultTransform'))(config, image_processor, self.split, ref_table=self.ref_table())
        self.dataset = self.dataset.with_transform(self.transform)

    def set_dataset(self):
        builder = getattr(databuilders, self.data_name.lower(), None)
        if builder is None:
            raise ValueError(f"Unknown dataset: {self.data_name}")
        builder = builder(data_files=self.data_files)
        builder.download_and_prepare()
        self.dataset = builder.as_dataset()

    def sample_dataset(self):
        ''' The data sample logics for few-shot learning. '''
        self.novel_sample_dict = {}
        for category in self.categories:
            self.dataset[category] = self.dataset[category].shuffle(self.shuffle_seed).select(range(min(self.k_shot, len(self.dataset[category]))))
            self.novel_sample_dict[category] = list(self.dataset[category]['dataid'])

    def sample_dataset_infer_phase(self):
        if self.max_inference_size is not None:
            # self.dataset['default'] = self.dataset['default'].shuffle(self.shuffle_seed).select(range(self.max_inference_size))
            for category in self.categories:
                self.dataset[category] = self.dataset[category].shuffle(self.shuffle_seed).select(range(min(self.max_inference_size, len(self.dataset[category]))))

    def collate_fn(self, examples):
        images = torch.stack([example[self.image_column] for example in examples])
        images = images.to(memory_format=torch.contiguous_format).float()
        captions = [example[self.caption_column] for example in examples]
        bboxes = [example[self.bbox_column] for example in examples]
        obboxes = [example[self.obbox_column] for example in examples]
        if isinstance(examples[0]["instances"], list):
            instances = [example["instances"] for example in examples]
        else:   
            instances = torch.stack([example["instances"] for example in examples])
        dataid = [example["dataid"] for example in examples]
        # Custom Keys
        if 'masks' in examples[0].keys():
            masks = torch.stack([example["masks"] for example in examples])
            return {self.image_column: images, self.caption_column: captions, self.bbox_column: bboxes, self.obbox_column: obboxes, "instances": instances, "dataid": dataid, "masks": masks}
        if 'parallels' in examples[0].keys():
            # parallels = torch.cat([example["parallels"] for example in examples])
            # images = torch.cat([images, parallels], dim=0)
            parallels = [example["parallels"] for example in examples]
            return {self.image_column: images, self.caption_column: captions, self.bbox_column: bboxes, self.obbox_column: obboxes, "instances": instances, "dataid": dataid, "parallels": parallels}
        return {self.image_column: images, self.caption_column: captions, self.bbox_column: bboxes, self.obbox_column: obboxes, "instances": instances, "dataid": dataid}

    def __call__(self):
        # for i in range(6):
        #     _ = self.dataset[i]
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            shuffle=True,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return dataloader

    def dump_novel_sample_dict(self):
        if self.phase == 'novel' and self.logger is not None:
            self.logger.info(f'Novel Sample Dict: \n{json.dumps(self.novel_sample_dict, indent=2)}', main_process_only=False)
            self.logger.info(f'Novel Ref Table: \n{json.dumps(self.ref_table("novel"), indent=2)}', main_process_only=False)
            self.logger.info(f'Dump novel_sample_dict at {self.dump_file}')
        with open(self.dump_file, 'w') as f:
            json.dump(self.novel_sample_dict, f, indent=2)

    def load_novel_sample_dict(self):
        if self.phase == 'novel' and self.logger is not None:
            self.logger.info(f'Load novel_sample_dict at {self.dump_file}')
            with open(self.dump_file, 'r') as f:
                self.novel_sample_dict = json.load(f)
            self.logger.info(f'Novel Sample Dict: \n{json.dumps(self.novel_sample_dict, indent=2)}', main_process_only=False)