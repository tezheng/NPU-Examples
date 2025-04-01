from itertools import chain
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    CLIPProcessor,
    CLIPTextModelWithProjection as TextEncoder,
    CLIPVisionModelWithProjection as ImagenEncoder,
)

from utils.npumodel import QNPUBertModel, QNPUImageModel


def calculate_score(npu_model, torch_model, dataloader):
    scores = []
    ellapsed_time_npu = 0
    ellapsed_time_cpu = 0
    for batch in dataloader:
        start = perf_counter()
        outputs = npu_model(**batch)
        embeds_1 = F.normalize(outputs[0])
        ellapsed_time_npu = ellapsed_time_npu + (perf_counter() - start) * 1000

        with torch.no_grad():
            start = perf_counter()
            model_output = torch_model(**batch)
            embeds_2 = F.normalize(model_output[0])
            ellapsed_time_cpu = ellapsed_time_cpu + (perf_counter() - start) * 1000

        scores.append(F.cosine_similarity(embeds_1, embeds_2))

    scores = torch.concat(scores)

    print(f"Number of samples: {len(scores)}")
    print(f"NPU inference: {round(ellapsed_time_npu / len(scores), 2)}ms")
    print(f"CPU inference: {round(ellapsed_time_cpu / len(scores), 2)}ms")

    return scores


def eval_text_encoder(model_name, model_path, dataset, batch_size):
    processor = AutoTokenizer.from_pretrained(model_name)
    npu_model = QNPUBertModel(
        qnpu_model_path=model_path,
        device="npu",
        qnpu_config={
            "disable_cpu_fallback": "1",
        },
    )
    torch_model = TextEncoder.from_pretrained(model_name).eval()

    dataloader = DataLoader(
        dataset.map(
            lambda batch: processor(
                text=list(chain(*batch["caption"])),
                padding="max_length",
                max_length=npu_model.sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            ),
            batched=True,
            remove_columns=dataset.column_names,
        ).with_format("torch"),
        batch_size=batch_size,
    )

    return calculate_score(npu_model, torch_model, dataloader)


def eval_vision_encoder(model_name, model_path, dataset, batch_size):
    processor = CLIPProcessor.from_pretrained(model_name)
    npu_model = QNPUImageModel(
        model_path=model_path,
        device="npu",
        qnpu_config={
            "disable_cpu_fallback": "1",
        },
    )
    torch_model = ImagenEncoder.from_pretrained(model_name).eval()

    dataloader = DataLoader(
        dataset.map(
            lambda batch: processor(
                images=batch["image"],
                return_tensors="pt",
            ),
            batched=True,
            remove_columns=dataset.column_names,
        ).with_format("torch"),
        batch_size=batch_size,
    )

    return calculate_score(npu_model, torch_model, dataloader)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a NPU model")
    parser.add_argument(
        "--model-name",
        type=str,
        default="openai/clip-vit-base-patch16",
        help="Orignal model name to load the tokenizer from",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Converted model path to load the model from",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="text",
        choices=["text", "image"],
        help="Encoder type: 'text' or 'image'",
    )
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)

    args = parser.parse_args()
    if args.model_path is None:
        qdq_model = "openai/clip_b16/"
        output_dir = Path(__file__).resolve().parent / "outputs"
        args.model_path = output_dir / qdq_model / args.encoder / "model.onnx"

    return args


if __name__ == "__main__":
    args = parse_args()

    model_name = args.model_name
    model_path = args.model_path.resolve()
    batch_size = args.batch_size

    dataset = load_dataset(
        "nlphuji/flickr30k",
        split="test",
        streaming=True,
    ).take(args.max_samples)

    if args.encoder == "text":
        scores = eval_text_encoder(model_name, model_path, dataset, batch_size)
    else:
        scores = eval_vision_encoder(model_name, model_path, dataset, batch_size)

    average_score = torch.mean(scores)
    p90_score = torch.quantile(scores, 0.90)
    print(f"Average score: {round(average_score.item(), 4)}")
    print(f"P90 score: {round(p90_score.item(), 4)}")
