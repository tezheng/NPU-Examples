from pathlib import Path
from pprint import pprint

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from datasets import load_dataset
from evaluate import load

from bert_common import tokenize_hfdataset
from qnpumodel import QNPUBertModel
from utils.npumodel import ModelOutput


def inference(
    tokenizer,
    dataset,
    qnpu_model_path,
    input_cols,
    label_col=None,
    device="npu",
    batch_size=128,
):
    model = QNPUBertModel(
        qnpu_model_path,
        device=device,
        qnpu_config={
            "disable_cpu_fallback": "0",
        },
    )
    tokenized_dataset = tokenize_hfdataset(
        dataset,
        tokenizer,
        input_cols=input_cols,
        label_col=label_col,
        seq_length=model.sequence_length,
    )

    all_outputs = []
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size)
    for batch in dataloader:
        all_outputs.append(model(**batch))

    avg, p90 = model.qnpu_session.latency
    print("Latency:")
    pprint(
        {
            "avg": float(avg),
            "p90": float(p90),
        }
    )

    return ModelOutput(
        **{
            k: torch.cat([o[k] for o in all_outputs], dim=0)
            for k in all_outputs[0].keys()
        }
    )


def eval_squad(
    start_logits,
    end_logits,
    tokenizer,
    dataset,
    seq_length,
):
    predictions = []
    references = []
    for idx, logits in enumerate(zip(start_logits, end_logits)):
        sample = dataset[idx]
        offset_mapping = tokenizer(
            sample["question"],
            sample["context"],
            padding="max_length",
            max_length=seq_length,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        ).offset_mapping

        start_index = logits[0].argmax(dim=-1)
        end_index = logits[1].argmax(dim=-1)
        answer_start = offset_mapping[:, start_index, 0].squeeze()
        answer_end = offset_mapping[:, end_index, 1].squeeze()
        pred_answer = sample["context"][answer_start:answer_end]

        references.append(
            {
                "id": sample["id"],
                "answers": {
                    "answer_start": sample["answers"]["answer_start"],
                    "text": sample["answers"]["text"],
                },
            }
        )
        predictions.append(
            {
                "id": sample["id"],
                "prediction_text": pred_answer,
            }
        )

    return load("squad").compute(
        predictions=predictions,
        references=references,
    )


def format_output(data, ratio=100.0):
    import json

    return json.loads(
        json.dumps(data), parse_float=lambda x: round(float(x) * ratio, 2)
    )


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
        default="google-bert/bert-base-multilingual-cased",
        help="Orignal model name to load the tokenizer from",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Converted model path to load the model from",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name to load the dataset from",
    )
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)

    args = parser.parse_args()
    if args.model_path is None:
        qdq_model = "google/bert_base_multilingual_cased"
        output_dir = Path(__file__).resolve().parent / "outputs"
        args.model_path = output_dir / qdq_model / "model.onnx"

    return args


if __name__ == "__main__":
    args = parse_args()

    batch_size = args.batch_size
    dataset = args.dataset
    max_samples = args.max_samples

    device = args.device
    model_name = args.model_name
    model_path = args.model_path.resolve()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if dataset == "glue/qnli":
        ds = load_dataset(
            "glue",
            "qnli",
            split="validation",
        )
        input_cols = ["question", "sentence"]
    elif dataset == "glue/mrpc":
        ds = load_dataset(
            "glue",
            "mrpc",
            split="test",
        )
        input_cols = ["sentence1", "sentence2"]
    else:
        ds = load_dataset(
            path="csv",
            data_files="nsp_wikitext_pairs.csv",
            split="train",
        )
        input_cols = ["sentence1", "sentence2"]
    ds = ds.select(range(min(max_samples, len(ds))))

    outputs = inference(
        tokenizer=tokenizer,
        dataset=ds,
        qnpu_model_path=model_path,
        input_cols=input_cols,
        device=device,
        batch_size=batch_size,
    )
    preds = outputs[0].argmax(dim=-1)

    accu = (
        load("accuracy").compute(
            predictions=preds,
            references=ds["label"],
        )
        or {}
    )
    f1 = (
        load("f1").compute(
            predictions=preds,
            references=list(ds["label"]),
        )
        or {}
    )
    pprint(format_output({**accu, **f1}))
