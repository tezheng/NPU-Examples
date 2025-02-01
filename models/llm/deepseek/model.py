
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from dataclasses import dataclass

import torch

from transformers import (
  AutoTokenizer,
  Qwen2Config,
  Qwen2ForCausalLM,
)
from transformers.cache_utils import DynamicCache
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
  ) -> 'StaticShapeCache':
    cache = cls()
    for idx, (key_states, value_states) in enumerate(zip(*past_key_values)):
      cache.update(key_states, value_states, idx)
    return cache


class Qwen2Block(torch.nn.Module):
  def __init__(self, model: Qwen2ForCausalLM) -> None:
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
    cache_position: torch.Tensor,
  ) -> ModelOutput:
    outputs = self._model(
      input_ids=input_ids,
      attention_mask=attention_mask,
      position_ids=position_ids,
      past_key_values=StaticShapeCache.from_legacy_cache(
        (past_keys, past_values)),
      cache_position=cache_position,
      use_cache=True,
    )
    new_keys, new_values = outputs.past_key_values.to_legacy_cache()

    return ModelOutput(
      logits=outputs.logits[:, -1, :],
      new_keys=new_keys,
      new_values=new_values,
    )


class CausalLMWithKVCache:
  def __init__(self, model_name: str) -> None:
    from sequence import SinkSequence, SinkCache

    self._tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_config = Qwen2Config.from_pretrained(model_name)
    self._sequence = SinkSequence(
      num_heads=model_config.num_key_value_heads,
      num_layers=model_config.num_hidden_layers,
      head_dim=model_config.hidden_size // model_config.num_attention_heads,
    )
    self._kvcache = SinkCache(self._sequence)

  def prompt(self, prompt: str) -> Dict[str, torch.Tensor]:
    from util import logger

    # CoT
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    # # TIR
    # messages = [
    #     {"role": "system", "content": "Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."},
    #     {"role": "user", "content": prompt}
    # ]
    input_ids: List[int] = self._tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        max_length=self._sequence.ctx_len,
        truncation=True,
    )  # type: ignore
    logger.info(
      f"Prompt(tokens={len(input_ids)}):\n{prompt}")

    inputs = self._sequence.prompt(input_ids)
    caches = self._kvcache.slice(inputs["input_ids"].shape[-1])

    return {k: torch.from_numpy(v) for k, v in (inputs | caches).items()}

  def token(self, new_token: int) -> Dict[str, torch.Tensor]:
    inputs = self._sequence.token(new_token)
    caches = self._kvcache.slice(inputs["input_ids"].shape[-1])
    return {k: torch.from_numpy(v) for k, v in (inputs | caches).items()}

  def update_kvcache(self, new_keys: torch.Tensor, new_values: torch.Tensor):
    self._kvcache.update(
      new_keys=new_keys.numpy(force=True),
      new_values=new_values.numpy(force=True),
    )

  @property
  def eos_token_id(self) -> int:
    return self._tokenizer.eos_token_id  # type: ignore

  @property
  def is_full(self) -> bool:
    return self._sequence.full

  @property
  def generated_tokens(self) -> List[int]:
    return self._sequence.generated_tokens


class TwoStagesMixin(ABC):
  def prefill(self,
              inputs: Dict[str, torch.Tensor]) -> Tuple[int, ModelOutput]:
    return self._generate(inputs)

  def decode(self,
             inputs: Dict[str, torch.Tensor]) -> Tuple[int, ModelOutput]:
    return self._generate(inputs)

  @abstractmethod
  def _generate(self,
                inputs: Dict[str, torch.Tensor]) -> Tuple[int, ModelOutput]:
    pass

  @property
  @abstractmethod
  def eos_token_id(self) -> int:
    pass


class Qwen2WithKVCache(CausalLMWithKVCache, TwoStagesMixin):
  _streaming: bool = True

  def __init__(self, model_name: str,
               use_streaming: bool = True, **kwargs) -> None:
    super().__init__(model_name)
    self._model = Qwen2Block(Qwen2ForCausalLM.from_pretrained(model_name))
    self._streaming = use_streaming

  def run(self, prompt: str) -> Tuple[str, int]:
    new_token, _ = self.prefill(self.prompt(prompt))
    while (new_token != self.eos_token_id and not self.is_full):
      new_token, _ = self.decode(self.token(new_token))
    if self._streaming:
      print('\n')
    return (
      ''.join(self._tokenizer.batch_decode(self.generated_tokens)),
      len(self.generated_tokens),
    )

  def _generate(
    self,
    inputs: Dict[str, torch.Tensor],
  ) -> Tuple[int, ModelOutput]:
    output = self._model(**inputs)
    logits, new_keys, new_values = output.values()
    new_token = int(torch.argmax(logits, dim=-1).item())
    self.update_kvcache(new_keys, new_values)

    if self._streaming:
      print(self._tokenizer.decode(new_token), end='', flush=True)

    return new_token, output
