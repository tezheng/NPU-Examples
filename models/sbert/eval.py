from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

from qnpumodel import QNPUBertModel


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate a QNPU model for sentence embeddings"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
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
        qdq_model = "sbert/all_minilm_l6_v2"
        root = Path(__file__).resolve().parent
        args.model_path = root / "outputs" / qdq_model / "model.onnx"

    return args


if __name__ == "__main__":
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = QNPUBertModel(
        qnpu_model_path=args.model_path,
        device="npu",
        qnpu_config={
            "disable_cpu_fallback": "0",
        },
    )
    torch_model = AutoModel.from_pretrained(args.model_name).eval()

    dataset = load_dataset("glue", "mrpc", split="test[:1000]")
    sentences = dataset["sentence1"] + dataset["sentence2"]
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
    print(f"NPU inference: {round((perf_counter() - start)*1000)}ms")

    embeds_1 = mean_pooling(outputs, encoded_input['attention_mask'])
    embeds_1 = F.normalize(embeds_1, p=2, dim=1)

    with torch.no_grad():
        start = perf_counter()
        model_output = torch_model(**encoded_input)
        print(f"CPU inference: {round((perf_counter() - start)*1000)}ms")
        embeds_2 = mean_pooling(
            model_output, encoded_input['attention_mask'])
        embeds_2 = F.normalize(embeds_2, p=2, dim=1)

    scores = F.cosine_similarity(embeds_1, embeds_2)
    average_score = torch.mean(scores)
    print(f"Average score: {round(average_score.item(), 3)}")
    p90_score = torch.quantile(scores, 0.90)
    print(f"P90 score: {round(p90_score.item(), 3)}")
