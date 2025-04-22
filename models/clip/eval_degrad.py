from itertools import chain
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import (
    AutoProcessor,
    CLIPTextModelWithProjection as TextEncoder,
    CLIPVisionModelWithProjection as ImagenEncoder,
)

from utils.npumodel import QNPUBertModel, QNPUImageModel


def load_hf_image_encoder(model_name):
    if model_name == "sentence-transformers/clip-ViT-B-32":
        from sbert_clip_script import load_sbert_image_encoder

        return load_sbert_image_encoder(model_name)

    return ImagenEncoder.from_pretrained(model_name).eval()


def load_hf_text_encoder(model_name):
    if model_name == "sentence-transformers/clip-ViT-B-32-multilingual-v1":
        from transformers import DistilBertModel
        from transformers.modeling_outputs import ModelOutput
        from transformers.utils import cached_file

        class DistillBertTextEncoder(torch.nn.Module):
            def __init__(self, model_name: str, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.distilbert = DistilBertModel.from_pretrained(model_name).eval()
                self.dense_weights = torch.load(
                    cached_file(model_name, "2_Dense/pytorch_model.bin")
                )["linear.weight"]

            @torch.inference_mode()
            def forward(self, input_ids, attention_mask):
                model_output = self.distilbert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                # Mean Pooling - Take attention mask into account for correct averaging
                last_hidden_state = model_output[0]
                input_mask_expanded = (
                    attention_mask.float()
                    .unsqueeze(dim=-1)
                    .expand(last_hidden_state.size())
                )
                sum_state = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
                pooled_output = F.normalize(sum_state, p=2, dim=1)

                # Calculate embedding
                text_embeds = torch.matmul(pooled_output, self.dense_weights.T)

                return ModelOutput(
                    embeds=text_embeds,
                    last_hidden_state=last_hidden_state,
                )

        return DistillBertTextEncoder(model_name)

    return TextEncoder.from_pretrained(model_name).eval()


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
    processor = AutoProcessor.from_pretrained(model_name)
    npu_model = QNPUBertModel(qnpu_model_path=model_path)
    torch_model = load_hf_text_encoder(model_name)

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
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    npu_model = QNPUImageModel(model_path=model_path)
    torch_model = load_hf_image_encoder(model_name)

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
        "--model",
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
    parser.add_argument(
        "--dataset",
        "--dataset-name",
        type=str,
        default="nlphuji/flickr_1k_test_image_text_retrieval",
        help="Dataset for evaluation",
    )
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)

    args = parser.parse_args()
    if args.model_path is None:
        qdq_model = "openai/clip_b16/"
        output_dir = Path(__file__).resolve().parent / "models"
        args.model_path = output_dir / qdq_model / args.encoder / "model.onnx"

    return args


if __name__ == "__main__":
    args = parse_args()

    model_name = args.model
    model_path = args.model_path.resolve()
    batch_size = args.batch_size

    print(
        f"Evaluating degradation of CLIP {args.encoder} encoder with "
        f"dataset {args.dataset}..."
    )
    print(f"Model name: {model_name}")
    print(f"Model path: {model_path}")

    print(f"Loading dataset: {args.dataset}...")
    dataset = load_dataset(
        args.dataset,
        split="test",
        streaming=True,
    ).take(args.max_samples)

    print("Evaluating: calculating similarity scores...")
    if args.encoder == "text":
        scores = eval_text_encoder(model_name, model_path, dataset, batch_size)
    else:
        scores = eval_vision_encoder(model_name, model_path, dataset, batch_size)

    average_score = torch.mean(scores)
    p90_score = torch.quantile(scores, 0.90)
    print(f"Average score: {round(average_score.item(), 4)}")
    print(f"P90 score: {round(p90_score.item(), 4)}")
