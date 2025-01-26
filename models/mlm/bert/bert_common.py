from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from datasets import Dataset as HFDataset

from transformers import BertModel
from transformers.modeling_outputs import ModelOutput as _ModelOutput


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
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
        **kwargs,
    ) -> ModelOutput:
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        sequence_output = self.encoder(
            hidden_states=embedding_output,
            attention_mask=attention_mask,
        )[0]
        pooled_output = (
            self.pooler(sequence_output)
            if self.pooler is not None
            else None
        )
        return ModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )


def create_4d_mask(
    mask: torch.Tensor,
    input_shape: Tuple[int, int],
    masked_value: float = -50.0
) -> torch.Tensor:
    # (batch_size, num_heads, seq_len, head_dim)
    batch_sz, seq_len = input_shape
    expanded_mask = mask[:, None, None, :].expand(batch_sz, 1, seq_len, seq_len)
    inverted_mask = 1.0 - expanded_mask.float()
    return inverted_mask.masked_fill(inverted_mask.bool(), masked_value)


def npz_to_hfdataset(npz_path: Path, max_samples: int) -> HFDataset:
    data_source = np.load(npz_path)
    data = {key: value.tolist()[:max_samples]
            for key, value in data_source.items()}
    return HFDataset.from_dict(data)


def tokenize_hfdataset(
    dataset: HFDataset,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    input_cols: List[str],
    label_col: Optional[str] = None,
    seq_length: int = 512,
    max_samples: Optional[int] = None,
) -> HFDataset:
    def generate_inputs(sample, indices):
        encoded_input = tokenizer(
            *[sample[input_col] for input_col in input_cols],
            padding="max_length",
            max_length=seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        batch_sz = encoded_input.input_ids.shape[0]
        input_ids = encoded_input.input_ids
        attention_mask = create_4d_mask(
            encoded_input.attention_mask,
            (batch_sz, seq_length),
        )
        position_ids = torch.arange(seq_length).expand(batch_sz, -1)
        token_type_ids = (
            encoded_input.token_type_ids
            if "token_type_ids" in encoded_input
            else torch.zeros(seq_length).expand(batch_sz, -1)
        )

        return {
            **{
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "token_type_ids": token_type_ids,
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
    tokenized_datasets.set_format("torch", output_all_columns=True)

    return tokenized_datasets
