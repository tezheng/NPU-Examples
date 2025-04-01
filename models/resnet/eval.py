from pathlib import Path
from functools import lru_cache
from random import Random
from pprint import pprint

from evaluate import load
from datasets import load_dataset

from transformers import AutoImageProcessor
from utils.npumodel import QNPUImageModel


@lru_cache(maxsize=1)
def get_imagenet_label_map():
    import requests

    imagenet_class_index_url = "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/imagenet_class_index.json"
    response = requests.get(imagenet_class_index_url)
    response.raise_for_status()  # Ensure the request was successful

    # Convert {0: ["n01440764", "tench"], ...} to {synset: index}
    return {v[0]: int(k) for k, v in response.json().items()}


def image_pre_process(
    dataset,
    data_processor,
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

    label_names = dataset.features[label_col].names
    label_map = get_imagenet_label_map()

    tensor_ds = dataset.map(
        lambda example: {
            **data_processor([img.convert("RGB") for img in example[input_col]]),
            "class": [label_map[label_names[x]] for x in example[label_col]],
        },
        batched=True,
        remove_columns=dataset.column_names,
    )
    tensor_ds.set_format("torch", output_all_columns=True)

    return tensor_ds


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate an NPU model")
    parser.add_argument(
        "--device",
        type=str,
        default="npu",
        choices=["cpu", "npu"],
        help="Device to run the model",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="microsoft/resnet-50",
        help="Orignal model name to load data processor",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="NPU model path",
    )
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)

    args = parser.parse_args()
    if args.model_path is None:
        qdq_model = "microsoft/resnet_50"
        root = Path(__file__).resolve().parent
        args.model_path = root / "outputs" / qdq_model / "model.onnx"
        args.model_path = root / "adaquant/resnet50_adaround.onnx"

    return args


if __name__ == "__main__":
    args = parse_args()

    model = QNPUImageModel(
        args.model_path.resolve(),
        device=args.device,
        qnpu_config={
            "disable_cpu_fallback": "1",
        },
    )
    image_processor = AutoImageProcessor.from_pretrained(args.model_name)

    hf_dataset = load_dataset("timm/mini-imagenet", split="test")
    dataset = image_pre_process(
        hf_dataset,
        image_processor,
        "image",
        "label",
        max_samples=args.max_samples,
    )

    output = model.inference(dataset, batch_size=args.batch_size)
    preds = output[0].argmax(-1)
    accu = (
        load("accuracy").compute(
            predictions=preds,
            references=dataset["class"],
        )
        or {}
    )
    f1 = (
        load("f1").compute(
            predictions=preds,
            references=dataset["class"],
            average="micro",
        )
        or {}
    )
    print("Accuracy:")
    pprint({**accu, **f1})

    avg, p90 = model.npu_session.latency
    print("Latency:")
    pprint({"avg": avg, "p90": p90})
