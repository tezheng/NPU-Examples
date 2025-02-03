from typing import List, Dict, Tuple
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .model import (
  ModelOutput,
  Qwen2WithKVCache,
  TwoStagesMixin,
)
from .util import logger


@dataclass
class CalibConfig:
  max_samples: int = 16
  skip_prefill: bool = False
  skip_decode: bool = False

  @classmethod
  def from_kwargs(cls, kwargs) -> 'CalibConfig':
    return cls(
      max_samples=kwargs.pop('max_samples', 16),
      skip_prefill=kwargs.pop('skip_prefill', False),
      skip_decode=kwargs.pop('skip_decode', False),
    )


class CalibDataMixin(TwoStagesMixin):
  _prefill_data: List[Dict[str, np.ndarray]] = []
  _decode_data: List[Dict[str, np.ndarray]] = []

  def prefill(self, inputs: Dict[str, torch.Tensor]) -> Tuple[int, ModelOutput]:
    token, outputs = self._generate(inputs)
    self._prefill_data.append({
      **{k: v.numpy(force=True) for k, v in inputs.items()},
      **{k: v.numpy(force=True) for k, v in outputs.items()},
    })
    self._prefill_len = inputs['position_ids'].max()
    return token, outputs

  def decode(self, inputs: Dict[str, torch.Tensor]) -> Tuple[int, ModelOutput]:
    token, outputs = self._generate(inputs)
    self._decode_data.append({
      **{k: v.numpy(force=True) for k, v in inputs.items()},
      **{k: v.numpy(force=True) for k, v in outputs.items()},
    })

    generated_len = inputs['position_ids'].max() - self._prefill_len
    max_samples = self.calib_cfg.max_samples
    if self.is_full or generated_len >= max_samples:
      token = self.eos_token_id

    return token, outputs

  @property
  def prefill_data(self):
    return self._prefill_data

  @property
  def decode_data(self):
    return self._decode_data


class CalibDataGenerator(Qwen2WithKVCache, CalibDataMixin):
  def __init__(self, model_name: str, **kwargs):
    self.calib_cfg = CalibConfig.from_kwargs(kwargs)
    super().__init__(model_name, **kwargs)

  def save_data(self, data_dir: Path) -> None:
    logger.info(f"Persisting prefill data: {len(self.prefill_data)} items")
    self._save_data(self.prefill_data, data_dir / 'prefill')
    logger.info(f"Persisting decode data: {len(self.decode_data)} items")
    self._save_data(self.decode_data, data_dir / 'decode')
    logger.info('Calibration data persisted successfully!')

  def _save_data(self, data: List[Dict[str, np.ndarray]],
                 file_path: Path) -> None:
    if len(data) == 0:
      raise RuntimeError('No data to save!')

    file_path = file_path.with_suffix('.npz')
    if (file_path.exists()):
      logger.warning(f"Remove existing calibration data: {file_path}")
      file_path.unlink()

    np.savez_compressed(
      file=file_path,
      allow_pickle=False,
      **{k: np.stack([i[k] for i in data]) for k in data[0]}
    )
    logger.info(f"Calibration data saved to {file_path}")


if __name__ == '__main__':
  from .util import parse_args

  args = parse_args()

  model_name = args.model_name
  logger.info(f'Generate calibration data for model: {model_name}')

  prompt = (
    args.prompt
    if args.prompt is not None
    else 'Who is the first president of United States?'
  )
  logger.info(f"Prompt:\n{prompt}")

  config = CalibConfig(
    skip_prefill=args.skip_prefill,
    skip_decode=args.skip_decode,
    max_samples=args.max_samples,
  )

  calib_data_gen = CalibDataGenerator(model_name, **asdict(config))
  calib_data_gen.run(prompt)

  data_dir = args.data_dir
  data_dir.mkdir(parents=True, exist_ok=True)
  calib_data_gen.save_data(data_dir=data_dir)
