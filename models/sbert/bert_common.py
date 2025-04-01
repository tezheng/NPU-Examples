from typing import List, Optional, Tuple, Union, TYPE_CHECKING
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import BertModel, DistilBertModel
from transformers.modeling_outputs import ModelOutput as _ModelOutput

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


@dataclass
class ModelOutput(_ModelOutput):
    """Wrapper for ModelOutput class from transformers.modeling_outputs.
    Always returns None for missing keys when accessed with __getitem__
    or __getattr__.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, k):
        if isinstance(k, str) and k not in self.keys():
            return None
        return super().__getitem__(k)

    def __getattr__(self, k):
        if k in self.keys():
            return self[k]
        return None

    def __repr__(self):
        return super().__repr__()


class SimpleBert(torch.nn.Module):
    def __init__(self, model: BertModel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embeddings = model.embeddings
        self.encoder = model.encoder
        self.pooler = model.pooler

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> ModelOutput:
        embedding_output = self.embeddings(input_ids)
        sequence_output = self.encoder(
            hidden_states=embedding_output,
            attention_mask=attention_mask,
        )[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )
        return ModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )


class SimpleDistilBert(torch.nn.Module):
    def __init__(self, model: DistilBertModel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = model.config
        self.embeddings = model.embeddings
        self.transformer = model.transformer

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> ModelOutput:
        embedding_output = self.embeddings(input_ids)
        head_mask = [None] * self.config.num_hidden_layers
        sequence_output = self.transformer(
            x=embedding_output,
            attn_mask=attention_mask,
            head_mask=head_mask,
        )
        return ModelOutput(
            last_hidden_state=sequence_output[0],
        )


def create_4d_mask(
    mask: torch.Tensor,
    input_shape: Union[torch.Size, Tuple[int, int]],
    masked_value: float = -50.0,
) -> torch.Tensor:
    # (batch_size, num_heads, seq_len, head_dim)
    batch_sz, seq_len = input_shape
    expanded_mask = mask[:, None, None, :].expand(batch_sz, 1, seq_len, seq_len)
    inverted_mask = 1.0 - expanded_mask.float()
    return inverted_mask.masked_fill(inverted_mask.bool(), masked_value)


def npz_to_hfdataset(npz_path: Path, max_samples: int):
    from datasets import Dataset

    data_source = np.load(npz_path)
    data = {key: value.tolist()[:max_samples] for key, value in data_source.items()}
    return Dataset.from_dict(data)


def tokenize_hfdataset(
    dataset: "Dataset",
    tokenizer: "Union[PreTrainedTokenizer, PreTrainedTokenizerFast]",
    input_cols: List[str],
    label_col: Optional[str] = None,
    seq_length: int = 512,
    max_samples: Optional[int] = None,
):
    def generate_inputs(sample, indices):
        encoded_input = tokenizer(
            *[sample[input_col] for input_col in input_cols],
            padding="max_length",
            max_length=seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        batch_sz = encoded_input.input_ids.shape[0]
        input_ids = encoded_input.input_ids
        attention_mask = create_4d_mask(
            encoded_input.attention_mask,
            (batch_sz, seq_length),
        )

        return {
            **{
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            **(
                {label_col: sample.get(label_col, indices)}
                if label_col is not None
                else {}
            ),
        }

    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    tokenized_datasets = dataset.map(
        generate_inputs,
        batched=True,
        with_indices=True,
        remove_columns=dataset.column_names,
    )

    def enforce_dtype(batch):
        batch = {k: torch.Tensor(v) for k, v in batch.items()}
        batch["input_ids"] = batch["input_ids"].int()
        return batch

    tokenized_datasets.set_transform(enforce_dtype)

    return tokenized_datasets
