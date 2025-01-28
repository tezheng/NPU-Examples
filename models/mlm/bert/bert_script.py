from typing import List, Optional
from collections import OrderedDict
from pathlib import Path

import torch
from transformers import AutoTokenizer

from olive.data.component.dataset import BaseDataset
from olive.data.registry import Registry

from bert_common import npz_to_hfdataset, tokenize_hfdataset


@Registry.register_dataset()
def load_npz_dataset(
  data_path: Path,
  max_samples: int,
):
    return npz_to_hfdataset(data_path, max_samples)


@Registry.register_pre_process()
def tokenize_dataset(
    dataset,
    model_name: str,
    input_cols: List[str],
    label_col: str,
    max_samples: Optional[int],
    seq_length=512,
    **kwargs
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = tokenize_hfdataset(
        dataset, tokenizer, input_cols, label_col=label_col,
        seq_length=seq_length, max_samples=max_samples,
    )
    return BaseDataset(list(dataset), label_col)


@Registry.register_post_process()
def bert_scl_post_process(outputs) -> torch.Tensor:
    """Post-processing for Sequence Classification tasks.
    """
    if isinstance(outputs, torch.Tensor):
        return outputs.argmax(dim=-1)
    if isinstance(outputs, (OrderedDict, dict)):
        if "logits" in outputs:
            return outputs['logits'].argmax(dim=-1)
        if "last_hidden_state" in outputs:
            return outputs["last_hidden_state"]
    raise ValueError(f"Unsupported output type: {type(outputs)}")


@Registry.register_post_process()
def bert_tcl_post_process(outputs) -> torch.Tensor:
    """Post-processing for Token Classification tasks.
    """
    if isinstance(outputs, torch.Tensor):
        return outputs
    if isinstance(outputs, (OrderedDict, dict)):
        if "logits" in outputs:
            return outputs["logits"]
        if "last_hidden_state" in outputs:
            return outputs["last_hidden_state"]
    raise ValueError(f"Unsupported output type: {type(outputs)}")


@Registry.register_post_process()
def bert_qa_post_process(outputs) -> torch.Tensor:
    """Post-processing for Question Answering tasks.
    """
    if isinstance(outputs, (OrderedDict, dict)):
        if "start_logits" in outputs and "end_logits" in outputs:
            logits = [outputs["start_logits"], outputs["end_logits"]]
            return torch.stack(logits, dim=1)
    raise ValueError(f"Unsupported output type: {type(outputs)}")
