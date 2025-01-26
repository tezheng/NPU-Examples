from typing import Dict, List, Union
from pathlib import Path

import numpy as np
import torch
from evaluate import load
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification as AutoModelTCL,
)

from olive.data.component.dataset import BaseDataset
from olive.data.registry import Registry

from bert_script import bert_tcl_post_process  # noqa: F401
from bert_common import (
    SimpleBert,
    npz_to_hfdataset,
    tokenize_hfdataset,
)


def create_llmlingua2_tokenizer(model_name: str, max_force_token: int = 100):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    added_tokens = [f"[NEW{i}]" for i in range(max_force_token)]
    tokenizer.add_special_tokens(
      {"additional_special_tokens": added_tokens}  # type: ignore
    )
    return tokenizer


def load_llmlingua2_bert_model(model_name: str) -> torch.nn.Module:
    model = AutoModelTCL.from_pretrained(model_name)
    model.eval()
    model.bert = SimpleBert(model.bert)
    return model


@Registry.register_dataset()
def load_meetingbank_data(
    data_path: str,
    model_name: str,
    max_samples: int,
    input_cols: List[str],
    label_col: str,
    seq_length: int = 512,
    max_force_token: int = 100,
    **kwargs
):
    dataset = tokenize_hfdataset(
        dataset=npz_to_hfdataset(Path(data_path).resolve(), max_samples),
        tokenizer=create_llmlingua2_tokenizer(model_name, max_force_token),
        input_cols=input_cols,
        label_col=label_col,
        seq_length=seq_length,
    )
    return BaseDataset(list(dataset), label_col)


def eval_token_classification(
    outputs,
    targets: torch.Tensor,
    **kwargs,
) -> Dict[str, Union[float, int]]:
    f1 = load("f1")
    accu = load("accuracy")
    metrics = {
      "r00": 0.00,
      "r50": 0.50,
      "r33": 0.33,
    }

    @torch.inference_mode()
    def logits_to_label(rate, logits: torch.Tensor) -> torch.Tensor:
        probs = logits.softmax(dim=-1)[:, 1]
        threshold = np.percentile(probs, int(100 * rate + 1))
        return (probs > threshold).int()

    results = {}
    for key, rate in metrics.items():
        predictions = torch.cat([logits_to_label(rate, p)
                                for p in outputs.preds])
        references = torch.cat([logits_to_label(rate, t) for t in targets])
        accu_results = accu.compute(
          predictions=predictions,
          references=references,
        )
        f1_results = f1.compute(
          predictions=predictions,
          references=references,
        )
        metrics = {**(f1_results or {}), **(accu_results or {})}
        results.update({f"{n}-{key}": v for n, v in metrics.items()})

    return results
