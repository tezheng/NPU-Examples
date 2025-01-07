
from typing import TypedDict, NotRequired

import numpy as np


class InferInput(TypedDict):
  input_ids: np.ndarray
  position_ids: np.ndarray
  attention_mask: np.ndarray
  cache_position: NotRequired[np.ndarray]
  past_keys: NotRequired[np.ndarray]
  past_values: NotRequired[np.ndarray]


class ForwardSequence:
  batch_size = 1
  seq_len = 128
  ctx_len = 512
  num_layers = 16
  num_heads = 8
  head_dim = 64
  mask_val = -50  # magic number

  # shape: (batch_size, 1, seq_len, ctx_len)
  attn_mask: np.ndarray
  # shape: (num_layers, batch_size, num_heads, ctx_len, head_dim)
  kvcache: tuple[np.ndarray, np.ndarray]

  _input_tokens: list[int] = []
  _output_tokens: list[int] = []

  def __init__(self, batch_size=1, seq_len=128, ctx_len=512,
               num_layers=16, num_heads=8, head_dim=64) -> None:
    self.batch_size = batch_size
    self.seq_len = seq_len
    self.ctx_len = ctx_len
    self.num_layers = num_layers
    self.num_heads = num_heads
    self.head_dim = head_dim

    mask = np.full((ctx_len + 1, ctx_len),
                   fill_value=self.mask_val, dtype=np.float32)
    mask = np.triu(mask, k=1)
    mask[-1, :] = self.mask_val
    self.attn_mask = mask

    cache_shape = (num_layers, batch_size, num_heads, self.ctx_len, head_dim)
    self.kvcache = (
      np.zeros(cache_shape, dtype=np.float32),
      np.zeros(cache_shape, dtype=np.float32),
    )

  def prompt(self, input_tokens: list[int]) -> InferInput:
    self._input_tokens = input_tokens
    return self.tensors(self.seq_len)

  def append(self, new_token: int) -> None:
    self._output_tokens.append(new_token)

  def tensors(self, seq_len=1) -> InferInput:
    tokens = self.all_tokens
    tok_len = len(tokens)
    min_len = min(seq_len, tok_len)

    input_ids = np.zeros((self.batch_size, seq_len), dtype=np.int32)
    input_ids[:, :min_len] = tokens[-min_len:]

    position_ids = np.full((self.batch_size, seq_len), -1, dtype=np.int32)
    position_ids[:, :min_len] = np.arange(tok_len)[-min_len:]

    cache_position = (
      np.arange(seq_len, dtype=np.int32)
      if seq_len > tok_len
      else np.arange(tok_len - seq_len, tok_len, dtype=np.int32)
    )

    # causal_mask = np.full((seq_len, self.ctx_len),
    #                       fill_value=self.mask_val, dtype=np.float32)
    # causal_mask *= np.arange(self.ctx_len) > cache_position[:, None]
    # attention_mask = np.broadcast_to(causal_mask,
    #                                  (self.batch_size, 1, *causal_mask.shape))
    attention_mask = self.attn_mask[position_ids][:, None, :, :]

    return {
      'input_ids': input_ids,
      'attention_mask': attention_mask,
      'position_ids': position_ids,
      'cache_position': cache_position,
      'past_keys': self.kvcache[0],
      'past_values': self.kvcache[1],
    }

  @property
  def generated_tokens(self) -> list[int]:
    return self._output_tokens

  @property
  def all_tokens(self) -> list[int]:
    return self._input_tokens + self._output_tokens

  @property
  def full(self) -> bool:
    return len(self._input_tokens) + len(self._output_tokens) >= self.ctx_len


class SinkSequence:
  batch_size = 1
  seq_len = 128
  ctx_len = 512
  num_layers = 16
  num_heads = 8
  head_dim = 64

  mask_val = -50.0  # magic number

  _input_tokens: list[int] = []
  _output_tokens: list[int] = []

  def __init__(self, batch_size=1, seq_len=128, ctx_len=512,
               num_layers=16, num_heads=8, head_dim=64) -> None:
    self.batch_size = batch_size
    self.seq_len = seq_len
    self.ctx_len = ctx_len
    self.num_layers = num_layers
    self.num_heads = num_heads
    self.head_dim = head_dim

  def prompt(self, input_tokens: list[int]) -> InferInput:
    # TODO: warning if input_tokens exceed seq_len
    if not input_tokens:
      raise RuntimeError('Please prompt with at least one token')

    self._input_tokens = input_tokens
    self._output_tokens = []
    return self._input_tensors(self.seq_len)

  def token(self, new_token: int) -> InferInput:
    if not self._input_tokens:
      raise RuntimeError('Please prompt first before token')
    if self.full:
      raise RuntimeError('Context is full, cannot append more tokens')

    self._output_tokens.append(new_token)
    return self._input_tensors(1)

  def _input_tensors(self, seq_len=1) -> InferInput:
    if seq_len > self.ctx_len:
      raise RuntimeError('Sequence length exceeds context length')

    tok_len = len(self)
    min_len = min(seq_len, tok_len)

    input_ids = np.zeros((self.batch_size, seq_len), dtype=np.int32)
    input_ids[:, -min_len:] = self.all_tokens[-min_len:]

    position_ids = np.full((self.batch_size, seq_len), -1, dtype=np.int32)
    position_ids[:, -min_len:] = np.arange(tok_len)[-min_len:]

    mask = np.full((seq_len, self.ctx_len),
                   fill_value=self.mask_val, dtype=np.float32)
    mask = np.triu(mask, k=self.ctx_len - tok_len + 1)
    mask[:, :self.ctx_len - seq_len] = self.mask_val

    positions = np.arange(self.ctx_len - seq_len, self.ctx_len)[:, None]
    mask = (np.arange(self.ctx_len) > positions) * self.mask_val
    mask[:, :self.ctx_len - len(self) - 1] = self.mask_val
    # shape: (batch_size, 1, seq_len, ctx_len)
    attn_mask = np.broadcast_to(mask.astype(np.float32),
                                (self.batch_size, 1, *mask.shape))
    return {
      'input_ids': input_ids,
      'attention_mask': attn_mask,
      'position_ids': position_ids,
    }

  def __len__(self) -> int:
    return len(self._input_tokens) + len(self._output_tokens)

  @property
  def generated_tokens(self) -> list[int]:
    return self._output_tokens

  @property
  def all_tokens(self) -> list[int]:
    return self._input_tokens + self._output_tokens

  @property
  def full(self) -> bool:
    return len(self._input_tokens) + len(self._output_tokens) >= self.ctx_len
