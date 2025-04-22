from typing import Optional
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    CLIPProcessor,
    CLIPTextModelWithProjection as TextEncoder,
    CLIPVisionModelWithProjection as ImagenEncoder,
)
from datasets import load_dataset

from utils.npumodel import QNPUModule


class QNPUCLIPModel:
    def __init__(
        self,
        model_name: str,
        text_model_path: Path,
        vision_model_path: Path,
        tokenizer_name: Optional[str] = None,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
        self.text_model = QNPUModule(text_model_path)
        self.vision_model = QNPUModule(vision_model_path)

    def get_image_features(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        output = self.vision_model.run(inputs)
        return output["embeds"]

    def get_text_features(self, text):
        max_length = self.text_model.session.get_inputs()[0].shape[1]
        inputs = self.tokenizer(
            text=text,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        output = self.text_model.run(
            {
                "input_ids": inputs["input_ids"].int(),
                "attention_mask": self._create_4d_mask(
                    inputs["attention_mask"],
                    inputs["input_ids"].shape,
                ),
            }
        )
        return output["embeds"]

    def _create_4d_mask(self, mask, input_shape, masked_value=-50.0):
        # (batch_size, num_heads, seq_len, head_dim)
        batch_sz, seq_len = input_shape
        expanded_mask = mask[:, None, None, :].expand(batch_sz, 1, seq_len, seq_len)
        inverted_mask = 1.0 - expanded_mask.float()
        return inverted_mask.masked_fill(inverted_mask.bool(), masked_value)


class HfCLIPModel:
    def __init__(self, model_name: str) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.text_model = TextEncoder.from_pretrained(model_name).eval().to(self.device)
        self.vision_model = (
            ImagenEncoder.from_pretrained(model_name).eval().to(self.device)
        )

    @torch.inference_mode()
    def get_image_features(self, images, batch_size=128):
        from tqdm.auto import tqdm

        all_embeds = [
            self.vision_model(
                **self.processor(
                    images=images[i : i + batch_size], return_tensors="pt"
                ).to(self.device)
            ).image_embeds
            for i in tqdm(range(0, len(images), batch_size), desc="Encoding images")
        ]

        return torch.cat(all_embeds, dim=0)

    @torch.inference_mode()
    def get_text_features(self, texts, batch_size=128):
        from tqdm.auto import tqdm

        all_embeds = [
            self.text_model(
                **self.processor(
                    text=texts[i : i + batch_size],
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                ).to(self.device)
            ).text_embeds
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts")
        ]

        return torch.cat(all_embeds, dim=0)


def compute_logits(text_embeds, image_embeds, logit_scale=100.0):
    text_embeds = np.asarray(text_embeds)
    image_embeds = np.asarray(image_embeds)

    text_norm = text_embeds / (
        np.linalg.norm(text_embeds, axis=1, keepdims=True) + 1e-8
    )
    image_norm = image_embeds / (
        np.linalg.norm(image_embeds, axis=1, keepdims=True) + 1e-8
    )
    return np.dot(text_norm, image_norm.T) * logit_scale


def compute_topk_accuracy(logits, labels, k=1):
    topk_preds = np.argsort(logits, axis=-1)[:, -k:]  # Get top-k indices
    labels = np.array(labels).reshape(-1, 1)
    correct = (topk_preds == labels).any(axis=1)
    return {"top{}_accuracy".format(k): correct.mean()}


def eval(
    model_name,
    text_model_path,
    vision_model_path,
    dataset_name,
    split,
    tokenizer_name=None,
):
    model = QNPUCLIPModel(
        model_name,
        text_model_path=text_model_path,
        vision_model_path=vision_model_path,
        tokenizer_name=tokenizer_name,
    )
    # model = HfCLIPModel(model_name)

    dataset = load_dataset(dataset_name, split=split)
    stacked_captions = dataset["caption"]
    captions = list(chain(*stacked_captions))
    images = dataset["image"]

    text_embeds = model.get_text_features(captions)
    image_embeds = model.get_image_features(images)
    logits = compute_logits(text_embeds, image_embeds)

    labels = list(
        chain(
            *[
                [i] * len(c)
                for i, c in zip(
                    range(len(stacked_captions)),
                    stacked_captions,
                )
            ]
        )
    )

    print("Text encoding latency")
    print(model.text_model.latency)
    print("Image encoding latency")
    print(model.vision_model.latency)

    return {
        **compute_topk_accuracy(logits, labels, k=1),
        **compute_topk_accuracy(logits, labels, k=5),
    }


def format_output(data, ratio=100.0):
    import json

    formatted = json.loads(
        json.dumps(data), parse_float=lambda x: round(float(x) * ratio, 2)
    )
    return json.dumps(formatted, indent=2)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a NPU model")
    parser.add_argument(
        "--model",
        "--model-name",
        type=str,
        default="openai/clip-vit-base-patch32",
    )
    parser.add_argument(
        "--text-encoder",
        "--text-encoder-path",
        type=Path,
        default="models/openai/clip_b32/text/model.onnx",
    )
    parser.add_argument(
        "--image-encoder",
        "--image-encoder-path",
        type=Path,
        default="models/openai/clip_b32/image/model.onnx",
    )
    parser.add_argument(
        "--tokenizer",
        "--tokenizer-name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dataset",
        "--dataset-name",
        type=str,
        default="nlphuji/flickr_1k_test_image_text_retrieval",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-samples", type=int, default=100)

    return parser.parse_args()


if __name__ == "__main__":
    from time import perf_counter

    args = parse_args()

    start_time = perf_counter()
    result = eval(
        args.model,
        args.text_encoder,
        args.image_encoder,
        args.dataset,
        args.split,
        args.tokenizer,
    )
    end_time = perf_counter()

    print(f"Evaluation result: {format_output(result)}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
