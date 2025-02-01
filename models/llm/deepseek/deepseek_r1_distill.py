from typing import Dict, List
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import Qwen2ForCausalLM

from olive.data.registry import Registry

from model import Qwen2Block


def load_model(model_name: str):
  model = Qwen2ForCausalLM.from_pretrained(model_name)
  model.eval()
  return Qwen2Block(model)


class DictDataset(Dataset):
  def __init__(self, data: Dict, labels: List):
    self._data = [dict(zip(data.keys(), v)) for v in zip(*data.values())]
    self._labels = torch.tensor(labels)

  def __len__(self):
    return len(self._data)

  def __getitem__(self, idx):
    return self._data[idx], self._labels[idx]


@Registry.register_dataset()
def load_npz_dataset(data_path: str, input_cols: List[str],
                     max_samples: int) -> Dataset:
  npz_path = Path(data_path).resolve()
  if not npz_path.exists():
    raise FileNotFoundError(f"Data file not found at: {npz_path}")

  data_source = dict(np.load(npz_path))
  data = {k: [x for x in v[:max_samples]]
          for k, v in data_source.items()
          if k in input_cols}
  return DictDataset(data, list(range(max_samples)))


# load_npz_dataset(Path(__file__).parent / "outputs/data/decode.npz", 10)

# "dynamic_axes": {
#     "input_ids": {"0": "batch_size", "1": "sequence_length"},
#     "attention_mask": {
#         "0": "batch_size",
#         "2": "sequence_length"
#     },
#     "position_ids": {"0": "batch_size", "1": "sequence_length"},
#     "past_keys": {
#         "1": "batch_size",
#         "2": "num_heads",
#         "3": "cache_length"
#     },
#     "past_values": {
#         "1": "batch_size",
#         "2": "num_heads",
#         "3": "cache_length"
#     },
#     "new_past_keys": {
#         "1": "batch_size",
#         "2": "num_heads",
#         "3": "context_length"
#     },
#     "new_past_values": {
#         "1": "batch_size",
#         "2": "num_heads",
#         "3": "context_length"
#     }
# }
# "to_fixed_shape": {
#   "type": "DynamicToFixedShape",
#   "dim_param": [
#     "batch_size",
#     "sequence_length",
#     "num_heads",
#     "cache_length",
#     "context_length"
#   ],
#   "dim_value": [1, 1, 2, 511, 512],
#   "save_as_external_data": true,
#   "all_tensors_to_one_file": true
# }
