from pathlib import Path
from pprint import pprint

import numpy as np
import torch
from evaluate import load
from datasets import load_dataset
from transformers import AutoTokenizer

from bert_common import npz_to_hfdataset, tokenize_hfdataset
from qnpumodel import (
    QNPUBertForTokenClassification as QNPUBertForTCL,
)


def inference(
    tokenizer,
    dataset,
    qnpu_model_path,
    input_cols,
):
    model = QNPUBertForTCL(qnpu_model_path, qnpu_config={
        "disable_cpu_fallback": "0",
    })
    tokenized_dataset = tokenize_hfdataset(
        dataset,
        tokenizer,
        input_cols=input_cols,
        seq_length=model.qnpu_session.sequence_length,
    )

    inputs = {col: tokenized_dataset[col]
              for col in tokenized_dataset.column_names}
    return model(**inputs)


def eval_llmlingua2_tcl(logits, targets):
    f1 = load("f1")
    accu = load("accuracy")
    metrics = {
      "r00": 0.00,
      "r50": 0.50,
      "r30": 0.30,
    }

    def logits_to_label(rate, logits) -> torch.Tensor:
        probs = logits.softmax(dim=-1)[:, 1]
        threshold = np.percentile(probs, int(100 * rate + 1))
        return (probs > threshold).int()

    results = {}
    for key, rate in metrics.items():
        predictions = torch.cat([logits_to_label(rate, p)
                                 for p in logits])
        references = torch.cat([logits_to_label(rate, torch.from_numpy(t))
                                for t in targets])
        accu_results = accu.compute(
          predictions=predictions,
          references=references,
        )
        f1_results = f1.compute(
          predictions=predictions,
          references=references,
        )
        results[key] = {**(f1_results or {}), **(accu_results or {})}

    return results


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
            return_tensors="pt"
        ).offset_mapping

        start_index = logits[0].argmax(dim=-1)
        end_index = logits[1].argmax(dim=-1)
        answer_start = offset_mapping[:, start_index, 0].squeeze()
        answer_end = offset_mapping[:, end_index, 1].squeeze()
        pred_answer = sample["context"][answer_start:answer_end]

        references.append({
            "id": sample["id"],
            "answers": {
                "answer_start": sample["answers"]["answer_start"],
                "text": sample["answers"]["text"],
            },
        })
        predictions.append({
            "id": sample["id"],
            "prediction_text": pred_answer,
        })

    return load("squad").compute(
        predictions=predictions,
        references=references,
    )


def format_output(data, ratio=100.0):
    import json
    return json.loads(
        json.dumps(data),
        # Format floats to 2 decimal places
        parse_float=lambda x: round(float(x) * ratio, 2)
    )


if __name__ == "__main__":
    import argparse

    root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="scl-glue-mrpc")
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()

    if args.task == "scl-glue-mrpc":
        tokenizer = AutoTokenizer.from_pretrained(
            "intel/bert-base-uncased-mrpc")
        dataset = load_dataset("glue", "mrpc", split="test").select(
            range(args.max_samples))
        qdq_model = "intel/bert_base_uncased_scl"
        qnpu_model_path = root / "outputs" / qdq_model / "model/model.onnx"

        logits = inference(
            tokenizer=tokenizer,
            dataset=dataset,
            qnpu_model_path=qnpu_model_path,
            input_cols=["sentence1", "sentence2"],
        )[0]

        accu = load("accuracy").compute(
            predictions=logits.argmax(dim=-1),
            references=dataset["label"],
        ) or {}
        f1 = load("f1").compute(
            predictions=logits.argmax(dim=-1),
            references=dataset["label"],
        ) or {}
        pprint(format_output({**accu, **f1}))
    elif args.task == "qa-squad":
        model_name = "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        dataset = load_dataset("squad", split="validation").select(
            range(args.max_samples))
        qdq_model = "google/bert_large_uncased_qa"
        qnpu_model_path = root / "outputs" / qdq_model / "model/model.onnx"

        outputs = inference(
            tokenizer=tokenizer,
            dataset=dataset,
            qnpu_model_path=qnpu_model_path,
            input_cols=["question", "context"],
        )
        pprint(format_output(
            eval_squad(
                start_logits=outputs.start_logits,
                end_logits=outputs.end_logits,
                tokenizer=tokenizer,
                dataset=dataset,
                seq_length=512,
            ),
            ratio=1.0,
        ))
    elif args.task == "tcl-llmlingua2-meetingbank":
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank")
        added_tokens = [f"[NEW{i}]" for i in range(100)]
        tokenizer.add_special_tokens(
            {"additional_special_tokens": added_tokens}  # type: ignore
        )

        dataset = npz_to_hfdataset(
            root / "data/llmlingua2_bert_base_multilingual_cased_meetingbank.npz",
            max_samples=args.max_samples,
        )
        qdq_model = "microsoft/llmlingua2_bert_base_multilingual_cased"
        qnpu_model_path = root / "outputs" / qdq_model / "model/model.onnx"

        outputs = inference(
            tokenizer=tokenizer,
            dataset=dataset,
            qnpu_model_path=qnpu_model_path,
            input_cols=["prompts"],
        )
        pprint(format_output(eval_llmlingua2_tcl(
            logits=outputs.logits,
            targets=np.array(dataset["logits"]),
        )))
    else:
        raise NotImplementedError(f"Unknown task: {args.task}")
