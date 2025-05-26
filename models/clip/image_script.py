from typing import Dict, OrderedDict
from functools import lru_cache
from pathlib import Path
from random import Random

import torch
from transformers import AutoImageProcessor

from olive.data.component.dataset import BaseDataset
from olive.data.registry import Registry


@lru_cache(maxsize=1)
def get_imagenet_label_map():
    file_path = Path(__file__).parent / "imagenet_class_index.json"
    if not file_path.exists():
        import requests

        imagenet_class_index_url = "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/imagenet_class_index.json"
        response = requests.get(imagenet_class_index_url)
        response.raise_for_status()  # Ensure the request was successful
        content = response.json()
    else:
        import json

        with open(file_path, "r") as f:
            content = json.loads(f.read())

    # Convert {0: ["n01440764", "tench"], ...} to {synset: index}
    return {v[0]: int(k) for k, v in content.items()}


@Registry.register_pre_process()
def image_pre_process(
    dataset,
    model_name,
    input_col,
    label_col,
    max_samples=None,
    shuffle=False,
    seed=42,
    **kwargs,
):
    if max_samples is not None:
        max_samples = min(max_samples, len(dataset))
        dataset = dataset.select(
            Random(seed).sample(range(len(dataset)), max_samples)
            if shuffle
            else range(max_samples)
        )

    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    label_names = dataset.features[label_col].names
    label_map = get_imagenet_label_map()
    tensor_ds = dataset.map(
        lambda example: {
            **processor([img.convert("RGB") for img in example[input_col]]),
            "class": [label_map[label_names[x]] for x in example[label_col]],
        },
        batched=True,
        remove_columns=dataset.column_names,
    )
    tensor_ds.set_format("torch", output_all_columns=True)

    return BaseDataset(tensor_ds, label_col="class")


@Registry.register_post_process()
def image_post_process(output):
    if isinstance(output, (Dict, OrderedDict)):
        return output["logits"].argmax(dim=-1)
    elif isinstance(output, torch.Tensor):
        return output.argmax(dim=-1)

    raise ValueError(f"Unsupported output type: {type(output)}")
