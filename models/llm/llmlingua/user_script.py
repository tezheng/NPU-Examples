from typing import Dict, List, Tuple, Union

import numpy as np

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    XLMRobertaForTokenClassification as XLMRobertaTCL,
    XLMRobertaModel,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions as ModelOutput,
    BaseModelOutputWithPastAndCrossAttentions as EncoderOutput,
    TokenClassifierOutput,
)

from evaluate import load

from olive.data.registry import Registry


class XLMRobertaWithoutEmbedding(torch.nn.Module):
  def __init__(self, model: XLMRobertaModel):
    super().__init__()
    self.encoder = model.encoder
    self.pooler = model.pooler

  def forward(self, *args, **kwargs):
    sequence_output = self.encoder(
      hidden_states=kwargs.get("inputs_embeds"),
      attention_mask=kwargs.get("attention_mask"),
    )[0]
    pooled_output = (
      self.pooler(sequence_output)
      if self.pooler is not None
      else None
    )
    return ModelOutput(
        last_hidden_state=sequence_output,
        pooler_output=pooled_output,  # type: ignore
    )


def load_xlmroberta_tcl_model(model_name: str) -> torch.nn.Module:
  model = XLMRobertaTCL.from_pretrained(model_name)
  model.eval()
  model.roberta = XLMRobertaWithoutEmbedding(model.roberta)  # type: ignore
  return model


def load_xlmroberta_tcl_encoder(model_name: str) -> torch.nn.Module:
  model = XLMRobertaTCL.from_pretrained(model_name)
  model.eval()
  return model.roberta.encoder


def create_4d_mask(mask, input_shape):
  batch_sz, seq_len = input_shape
  expanded_mask = mask[:, None, None, :].expand(batch_sz, 1, seq_len, seq_len)
  inverted_mask = 1.0 - expanded_mask.float()
  return inverted_mask.masked_fill(inverted_mask.bool(), -50.0)


class SimpleDataset(Dataset):
  def __init__(
    self,
    data: List[Dict[str, torch.Tensor]],
    labels: List[torch.Tensor]
  ) -> None:
    self.data = data
    self.labels = labels

  def __len__(self) -> int:
    return min(len(self.data), len(self.labels))

  def __getitem__(self, idx) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    return self.data[idx], self.labels[idx]


@Registry.register_dataset()
def meetingbank_dataset(
  data_path: str,
  model_name: str,
  max_samples: int,
  seq_length: int = 512,
  max_force_token: int = 100,
  **kwargs
) -> Dataset:
  added_tokens = [f"[NEW{i}]" for i in range(max_force_token)]

  tokenizer = AutoTokenizer.from_pretrained(model_name)
  tokenizer.add_special_tokens(
    {"additional_special_tokens": added_tokens}  # type: ignore
  )

  model = XLMRobertaTCL.from_pretrained(model_name)
  model.resize_token_embeddings(len(tokenizer))
  model.eval()

  with np.load(data_path) as data:
    texts = data['prompts'].tolist()[:max_samples]
    logits = data['logits'].tolist()[:max_samples]

  with torch.inference_mode():
    encoded_input = tokenizer(
      texts,
      padding="max_length",
      max_length=seq_length,
      truncation=True,
      add_special_tokens=True,
      return_tensors="pt",
    )
    inputs_embeds = model.roberta.embeddings(
      encoded_input.input_ids,
      token_type_ids=encoded_input.get("token_type_ids", None)
    )
    attention_mask = create_4d_mask(
      encoded_input.attention_mask,
      encoded_input.input_ids.shape,
    )

  embeds_field_name = kwargs.get("embeds_field_name", "inputs_embeds")
  return SimpleDataset(
    [{
      embeds_field_name: inputs_embeds[i],
      "attention_mask": attention_mask[i],
    } for i in range(len(texts))],
    [torch.tensor(x) for x in logits],
  )


# @Registry.register_dataset()
# def meetingbank_dataset2(
#   data_path: str,
#   model_name: str,
#   max_samples: int,
#   seq_length: int = 512,
#   max_force_token: int = 100,
#   **kwargs
# ) -> Dataset:
#   added_tokens = [f"[NEW{i}]" for i in range(max_force_token)]

#   tokenizer = AutoTokenizer.from_pretrained(model_name)
#   tokenizer.add_special_tokens(
#     {"additional_special_tokens": added_tokens}
#   )

#   model = XLMRobertaTCL.from_pretrained(model_name)
#   model.resize_token_embeddings(len(tokenizer))
#   model.eval()

#   with np.load(data_path) as data:
#     texts = list(data['prompts'])[:max_samples]
#     logits = list(data['logits'])[:max_samples]

#   from common import TokenClfDataset
#   dataset = TokenClfDataset(
#     texts=texts,
#     tokenizer=tokenizer,
#     max_len=seq_length,
#     model_name="xlm-roberta-large",
#   )

#   @torch.inference_mode()
#   def inference(samples):
#     input_ids = samples["ids"].unsqueeze(0)
#     mask = samples["mask"].unsqueeze(0)
#     inputs_embeds = model.roberta.embeddings(input_ids)
#     attention_mask = create_4d_mask(mask, input_ids.shape)
#     return {
#       "inputs_embeds": inputs_embeds.squeeze(0),
#       "attention_mask": attention_mask.squeeze(0),
#     }

#   return SimpleDataset(
#     [inference(sample) for sample in dataset],
#     [torch.tensor(x) for x in logits]
#   )


@Registry.register_post_process()
def xlmroberta_tcl_post_process(outputs) -> torch.Tensor:
  if isinstance(outputs, TokenClassifierOutput):
    return outputs.logits
  elif isinstance(outputs, EncoderOutput):
    return outputs.last_hidden_state
  return outputs


def eval_tok_cls_accu_f1(logits, targets) -> Dict[str, Union[float, int]]:
  f1 = load("f1")
  accu = load("accuracy")
  metrics = {
    "r00": 0.00,
    "r50": 0.50,
    "r30": 0.30,
  }

  @torch.inference_mode()
  def logits_to_label(rate, logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=-1)[:, 1]
    threshold = np.percentile(probs, int(100 * rate + 1))
    return (probs > threshold).int()

  results = {}
  for key, rate in metrics.items():
    predictions = torch.cat([logits_to_label(rate, p) for p in logits])
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


def eval_token_classification(outputs, targets,
                              **kwargs) -> Dict[str, Union[float, int]]:
  return eval_tok_cls_accu_f1(outputs.preds, targets)


def eval_token_cls_with_encoder_output(
  outputs,
  targets,
  **kwargs,
) -> Dict[str, Union[float, int]]:
  model_name = kwargs.get("model_name")
  if model_name is None:
    raise ValueError(
      "model_name is required for eval_token_cls_with_encoder_output")

  model = XLMRobertaTCL.from_pretrained(model_name)
  model.eval()
  logits = model.classifier(outputs.preds)

  return eval_tok_cls_accu_f1(logits, targets)
