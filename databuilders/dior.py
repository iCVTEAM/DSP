import os
import json
import datasets
from PIL import Image
from dataclasses import dataclass

@dataclass
class DiorConfig(datasets.BuilderConfig):
    """BuilderConfig for Dior dataset."""
    pass


class Dior(datasets.GeneratorBasedBuilder):
    """DIOR Dataset."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIG_CLASS = DiorConfig

    BUILDER_CONFIGS = [
        DiorConfig(
            name="default",
            description="Default configuration for the DIOR dataset.",
        ),
    ]
    
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):

        return datasets.DatasetInfo(
            description="The DIOR Dataset.",
            features=datasets.Features({
                "image": datasets.Image(),
                "captions": datasets.Sequence(datasets.Value("string")),
                "bndboxes": datasets.Array2D(shape=(None, 4), dtype="float32"),
                "obboxes": datasets.Array2D(shape=(None, 8), dtype="float32"),
                "dataid": datasets.Value("string")
            }),
            homepage="http://www.example.com/",
            citation="",
        )

    def _split_generators(self, dl_manager):

        data_files = dl_manager.download_and_extract(self.config.data_files)

        if not data_files or not isinstance(data_files, dict):
            raise ValueError(
                "This builder requires you to pass data_files as a dictionary."
                "for example: data_files={'train': 'path/to/train.jsonl', 'test': 'path/to/test.jsonl'}"
            )
        split_generators = []

        for split_name, metadata_list in data_files.items():
            split_generators.append(
                datasets.SplitGenerator(
                    name=split_name,
                    gen_kwargs={"metadata_list": metadata_list},
                )
            )
        
        return split_generators

    def _generate_examples(self, metadata_list):

        idx = 0

        for metadata_path in metadata_list:
            # print(f"--> [Generator] Processing file: {metadata_path}")
            base_dir = os.path.dirname(os.path.abspath(metadata_path))

            with open(metadata_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    try:
                        data = json.loads(line)

                        absolute_image_path = os.path.join(base_dir, data["file_name"])
                        # image = Image.open(absolute_image_path).convert("RGB")
                        
                        example = {
                            "image": absolute_image_path,
                            "captions": data.get("captions", []),
                            "bndboxes": data.get("bndboxes", []),
                            "obboxes": data.get("obboxes", []),
                            "dataid": os.path.splitext(os.path.basename(data["file_name"]))[0]
                        }
                        
                        yield idx, example

                        idx += 1

                    except Exception as e:
                        print(f"  - Skip Invalid Data ({metadata_path} Line {i}): {e}")


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "DIOR")
    # data_files = {
    #     "train": os.path.join(data_dir, "train_meta.jsonl"),
    #     "test": os.path.join(data_dir, "test_meta.jsonl"),
    # }
    data_files = {
        "train": [os.path.join(data_dir, "train_meta_sample1.jsonl"), 
                  os.path.join(data_dir, "train_meta_sample2.jsonl")],
        "test": os.path.join(data_dir, "infer_meta_sample1.jsonl"),
    }

    builder = Dior(data_files=data_files, config_name="default")
    builder.download_and_prepare()
    dataset = builder.as_dataset()

    print(dataset)