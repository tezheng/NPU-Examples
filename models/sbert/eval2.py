from itertools import chain
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from transformers.utils import cached_file

from qnpumodel import QNPUBertModel


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate a QNPU model for sentence embeddings"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/clip-ViT-B-32-multilingual-v1",
        help="Orignal model name to load the tokenizer from",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Converted model path to load the model from",
    )

    args = parser.parse_args()
    if args.model_path is None:
        qdq_model = "sbert/clip_distilbert_multilingual_v1"
        model_dir = Path(__file__).resolve().parent / "outputs"
        args.model_path = model_dir / qdq_model / "model.onnx"

    return args


if __name__ == "__main__":
    args = parse_args()

    dense_weights = torch.load(
        cached_file(args.model_name, "2_Dense/pytorch_model.bin")
    )["linear.weight"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = QNPUBertModel(
        qnpu_model_path=args.model_path,
        device="npu",
        qnpu_config={
            "disable_cpu_fallback": "1",
        },
    )
    torch_model = AutoModel.from_pretrained(args.model_name).eval()

    # dataset = load_dataset("glue", "mrpc", split="test[:100]")
    # sentences = dataset["sentence1"] + dataset["sentence2"]
    dataset = load_dataset(
        "nlphuji/flickr_1k_test_image_text_retrieval",
        split="test[:20]",
    )
    sentences = list(chain(*dataset["caption"]))
    encoded_input = tokenizer(
        sentences,
        padding="max_length",
        max_length=model.sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )

    start = perf_counter()
    outputs = model(**encoded_input)
    print(f"NPU inference: {round((perf_counter() - start) * 1000)}ms")

    # embeds_1 = outputs[0]
    embeds_1 = mean_pooling(outputs[1], encoded_input["attention_mask"])
    embeds_1 = torch.matmul(embeds_1, dense_weights.T)
    embeds_1 = F.normalize(embeds_1, p=2, dim=1)

    with torch.no_grad():
        start = perf_counter()
        model_output = torch_model(**encoded_input)
        print(f"CPU inference: {round((perf_counter() - start) * 1000)}ms")
        embeds_2 = mean_pooling(model_output[0], encoded_input["attention_mask"])
        embeds_2 = torch.matmul(embeds_2, dense_weights.T)
        embeds_2 = F.normalize(embeds_2, p=2, dim=1)

    scores = F.cosine_similarity(embeds_1, embeds_2)
    average_score = torch.mean(scores)
    print(f"Average score: {round(average_score.item(), 2)}")
    p90_score = torch.quantile(scores, 0.90)
    print(f"P90 score: {round(p90_score.item(), 2)}")
