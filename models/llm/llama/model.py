
from abc import ABC, abstractmethod
from typing import cast, Any, Optional, Self, TypedDict

import torch

from transformers import AutoTokenizer
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama import (
  LlamaConfig,
  LlamaForCausalLM,
  LlamaTokenizerFast,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from .sequence import SinkSequence


class StaticCache(Cache):
  """
  StaticCache: A cache implementation with static entries.
  """

  def __init__(
    self,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor
  ) -> None:
    super().__init__()

    self.key_cache = key_cache
    self.value_cache = value_cache
    self.max_cache_len = value_cache[0].shape[-2]

  def update(
    self,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
    cache_kwargs: Optional[dict[str, Any]] = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO
    # 1. Figure out why cache update doesnt work if key_cache/value_cache
    #    are instantiated with tensor.from_numpy().
    # 2. Do we need `torch._dynamo.mark_static_address()`?
    # 3. `tensor.index_copy_` vs `tensor[:, :, index]` in ONNX/QNN HVX?

    # Notes (from the `transformers.cache_utils.StaticCache`):
    # 1. `mark_static_address` is used to tag the cache as an fixed data
    #    pointer, preventing cuda graph breaks when updating the cache. It
    #    can't be used if the cache code is being compiled (but in that case
    #    it is not needed anyway)
    # 2. `torch.export()` requires mutations to be registered as buffers.

    assert cache_kwargs is not None
    cache_position = cache_kwargs.get('cache_position')
    assert cache_position is not None

    k_out = self.key_cache[layer_idx]
    v_out = self.value_cache[layer_idx]

    # idx = cache_position.long()
    # k_out.index_copy_(2, idx, key_states)
    # v_out.index_copy_(2, idx, value_states)
    k_out[:, :, cache_position] = key_states
    v_out[:, :, cache_position] = value_states

    return k_out, v_out

  def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
    # FIXME: This method is deprecating, and wont affect the result when
    # cache_position is provided in the `model.forward` method.
    return self.max_cache_len

  def get_max_cache_shape(self) -> Optional[int]:
    return self.max_cache_len

  def to_legacy_cache(self) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    return self.key_cache, self.value_cache


class StaticShapeCache(DynamicCache):
  """
  StaticShapeCache: A cache implementation for static shapes.
  """

  def to_legacy_cache(self) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.stack(self.key_cache), torch.stack(self.value_cache)

  @classmethod
  def from_legacy_cache(
    cls,
    past_key_values: tuple[torch.Tensor, torch.Tensor]
  ) -> Self:
    cache = cls()
    for idx, (key_states, value_states) in enumerate(zip(*past_key_values)):
      cache.update(key_states, value_states, idx)
    return cache


class FIFOCache(Cache):
  """
  FIFOCache: A cache implementation with FIFO eviction policy.
  """

  def __init__(
    self,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor
  ) -> None:
    super().__init__()

    self.key_cache = key_cache
    self.value_cache = value_cache

  def update(
    self,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
    cache_kwargs: Optional[dict[str, Any]] = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    if key_states is not None and value_states is not None:
      if self.key_cache.shape[0] <= layer_idx:
        # Fill missing layers
        for _ in range(self.key_cache.shape[0], layer_idx):
          self.key_cache = torch.cat(
            [self.key_cache, torch.zeros(self.key_cache.shape[1:])], dim=0)
          self.value_cache = torch.cat(
            [self.value_cache, torch.zeros(self.value_cache.shape[1:])], dim=0)
        self.key_cache = torch.cat([self.key_cache, key_states], dim=0)
        self.value_cache = torch.cat([self.value_cache, value_states], dim=0)
      else:
        self.key_cache[layer_idx] = torch.cat(
            [self.key_cache[layer_idx], key_states], dim=-2)
        self.value_cache[layer_idx] = torch.cat(
            [self.value_cache[layer_idx], value_states], dim=-2)

    return self.key_cache[layer_idx], self.value_cache[layer_idx]

  def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
    return self.key_cache[layer_idx].shape[-2]

  def get_max_cache_shape(self) -> Optional[int]:
    return None

  def to_legacy_cache(self) -> tuple[torch.Tensor, torch.Tensor]:
    return self.key_cache, self.value_cache


class LogitsWithPast(TypedDict):
  logits: torch.Tensor
  past_keys: torch.Tensor
  past_values: torch.Tensor


class LlamaBlock(torch.nn.Module):
  def __init__(self, model: LlamaForCausalLM) -> None:
    super().__init__()
    self._model = model

  @torch.inference_mode()
  def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    past_keys: torch.Tensor,
    past_values: torch.Tensor,
  ) -> LogitsWithPast:
    outputs: CausalLMOutputWithPast = self._model(
      input_ids=input_ids,
      attention_mask=attention_mask,
      position_ids=position_ids,
      past_key_values=StaticShapeCache.from_legacy_cache(
        (past_keys, past_values)),
      use_cache=True,
    )
    # assert type(outputs.past_key_values) is StaticShapeCache
    past_keys, past_values = outputs.past_key_values.to_legacy_cache()

    # return outputs.logits[:, -1, :], past_keys, past_values
    return LogitsWithPast(
      logits=outputs.logits[:, -1, :],
      past_keys=past_keys,
      past_values=past_values,
    )


class SequenceMixin:
  _tokenizer: LlamaTokenizerFast

  def __init__(self, model_name: str) -> None:
    self._tokenizer = cast(
      LlamaTokenizerFast,
      AutoTokenizer.from_pretrained(model_name),
    )
    self._tokenizer.add_bos_token = False

    model_config = LlamaConfig.from_pretrained(model_name)
    self._sequence = SinkSequence(
      num_heads=model_config.num_key_value_heads,
      num_layers=model_config.num_hidden_layers,
      head_dim=model_config.head_dim,
    )

  def prompt(self, text: str) -> dict[str, torch.Tensor]:
    tokens = self._tokenizer(
      text, max_length=self._sequence.seq_len, truncation=True)
    inputs = self._sequence.prompt(tokens.input_ids)
    tensors = {k: torch.tensor(v) for k, v in inputs.items()}

    cache_shape = (
      self._sequence.num_layers,
      self._sequence.batch_size,
      self._sequence.num_heads,
      self._sequence.ctx_len - tensors['input_ids'].shape[-1],
      self._sequence.head_dim,
    )
    self._kvcache = dict(
      past_keys=torch.zeros(cache_shape, dtype=torch.float32),
      past_values=torch.zeros(cache_shape, dtype=torch.float32),
    )

    return tensors | self._kvcache

  def token(self, new_token: int) -> dict[str, torch.Tensor]:
    inputs = self._sequence.token(new_token)
    tensors = {k: torch.tensor(v) for k, v in inputs.items()}
    return tensors | self._kvcache

  def update_kvcache(self, past_keys: torch.Tensor, past_values: torch.Tensor):
    self._kvcache = dict(
      past_keys=past_keys[:, :, :, 1:, :],
      past_values=past_values[:, :, :, 1:, :]
    )

  @property
  def eos_token_id(self) -> int:
    return cast(int, self._tokenizer.eos_token_id)

  @property
  def is_full(self) -> bool:
    return self._sequence.full

  @property
  def generated_tokens(self) -> list[int]:
    return self._sequence.generated_tokens


class TwoStagesMixin(ABC):
  def prefill(self,
              inputs: dict[str, torch.Tensor]) -> tuple[int, LogitsWithPast]:
    return self._generate(inputs)

  def decode(self,
             inputs: dict[str, torch.Tensor]) -> tuple[int, LogitsWithPast]:
    return self._generate(inputs)

  @abstractmethod
  def _generate(self,
                inputs: dict[str, torch.Tensor]) -> tuple[int, LogitsWithPast]:
    pass

  @property
  @abstractmethod
  def eos_token_id(self) -> int:
    pass


class LlamaWithKVCache(SequenceMixin, TwoStagesMixin):
  _streaming: bool = True

  def __init__(self, model_name: str,
               use_streaming: bool = True, **kwargs) -> None:
    SequenceMixin.__init__(self, model_name)
    TwoStagesMixin.__init__(self)

    model = LlamaForCausalLM.from_pretrained(model_name)
    self._model = LlamaBlock(model)
    self._streaming = use_streaming

  def run(self, prompt: str) -> str:
    new_token, _ = self.prefill(self.prompt(prompt))
    while (new_token != self.eos_token_id and not self.is_full):
      new_token, _ = self.decode(self.token(new_token))
    if self._streaming:
      print('\n')
    return ''.join(self._tokenizer.batch_decode(self.generated_tokens))

  def _generate(self,
                inputs: dict[str, torch.Tensor]) -> tuple[int, LogitsWithPast]:
    output = self._model(**inputs)
    logits, past_keys, past_values = output.values()
    new_token = int(torch.argmax(logits, dim=-1).item())
    self.update_kvcache(past_keys, past_values)

    if self._streaming:
      print(self._tokenizer.decode(new_token), end='', flush=True)

    return new_token, output
