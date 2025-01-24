
from pathlib import Path

import numpy as np
import torch
from transformers import (
  AutoModel,
  AutoTokenizer,
  XLMRobertaForTokenClassification as XLMRobertaForTCL,
)
from transformers.modeling_outputs import (
  ModelOutput,
  TokenClassifierOutput,
)

import onnxruntime as ort


def load_hf_model(model_name: str) -> torch.nn.Module:
  model = AutoModel.from_pretrained(model_name)
  return model.encoder


class QNPUModule():
  def __init__(self, model_path: Path, **kwargs) -> None:
    disable_cpu_fallback = kwargs.get('disable_cpu_fallback', '1')
    ep_context_enable = kwargs.get('ep_context_enable', '1')
    ep_context_embed = kwargs.get('ep_context_embed', '0')
    htp_performance_mode = kwargs.get('htp_performance_mode', 'burst')
    htp_graph_opt_mode = kwargs.get('htp_graph_optimization_mode', '3')

    options = ort.SessionOptions()
    options.add_session_config_entry(
      'session.disable_cpu_ep_fallback', disable_cpu_fallback)

    if not str(model_path.name).endswith('.onnx_ctx.onnx'):
      epctx_model_path = model_path.with_suffix('.onnx_ctx.onnx')
      if epctx_model_path.exists():
        model_path = epctx_model_path
      else:
        options.add_session_config_entry(
          'ep.context_enable', ep_context_enable)
        options.add_session_config_entry(
          'ep.context_embed_mode', ep_context_embed)

    provider_options = [{
      'backend_path': 'QnnHtp.dll',
      'htp_performance_mode': htp_performance_mode,
      'htp_graph_finalization_optimization_mode': htp_graph_opt_mode,
    }]

    if kwargs.get('qnpu_prof', False):
      prof_path = Path(kwargs.get('qnpu_prof_file_path', 'qnpu_profile.csv'))
      prof_level = kwargs.get('qnpu_prof_level', 'detailed')

      if prof_path.exists():
        prof_path.unlink()

      provider_options[0].update({
        'profiling_level': prof_level,
        'profiling_file_path': str(prof_path),
      })

    self.session = ort.InferenceSession(
      str(model_path),
      sess_options=options,
      providers=['QNNExecutionProvider'],
      provider_options=provider_options,
    )
    self._input_names = [i.name for i in self.session.get_inputs()]
    self._outputs_names = [o.name for o in self.session.get_outputs()]
    self._batch_size = self.session.get_inputs()[0].shape[0]

  def run(self, **tensors):
    if self.session is None:
      raise RuntimeError(
        "ONNX session not initialized. Please call _init_session() first.")

    inputs = {name: tensor.split(split_size=self._batch_size, dim=0)
              for name, tensor in tensors.items() if name in self._input_names}
    missing_inputs = self._input_names - inputs.keys()
    if missing_inputs:
      raise RuntimeError(f"Missing inputs for ONNX model: {missing_inputs}")

    # Split batches and convert torch tensors to numpy arrays
    batches = [dict(zip(inputs.keys(), [v.numpy() for v in values]))
               for values in zip(*inputs.values())]
    # Run the ONNX model
    outputs = [self.session.run(None, batch) for batch in batches]

    # Map outputs back to a dictionary with output names
    return ModelOutput(
      **dict(zip(
        self._outputs_names,
        [torch.from_numpy(np.concatenate(a)) for a in zip(*outputs)],
      ))  # type: ignore
    )


# (batch_size, seq_len) -> (batch_size, num_heads, seq_len, head_dim)
def _create_4d_mask(mask, input_shape):
  batch_sz, seq_len = input_shape
  expanded_mask = mask[:, None, None, :].expand(batch_sz, 1, seq_len, seq_len)
  inverted_mask = 1.0 - expanded_mask.float()
  return inverted_mask.masked_fill(inverted_mask.to(torch.bool), -50.0)


# Copied from modeling_roberta.create_position_ids_from_input_ids
def _create_position_ids(input_ids, padding_idx, past_key_values_len=0):
  # The series of casts and type-conversions here are carefully balanced
  # to both work with ONNX export and XLA.
  mask = input_ids.ne(padding_idx).int()
  indices = (torch.cumsum(mask, dim=1) + past_key_values_len) * mask
  return indices.long() + padding_idx


class QNPUXLMRobertaForTCLEncoder(torch.nn.Module):
  def __init__(self, model_name: str, qnpu_model_path: Path,
               /, *args, **kwargs) -> None:
    config = kwargs.pop('model_config', {})
    vocab_size = kwargs.pop('vocab_size', None)
    qnpu_config = kwargs.pop('qnpu_config', {})

    super().__init__(*args, **kwargs)

    # Load the Hugging Face model
    hf_model = XLMRobertaForTCL.from_pretrained(model_name, **config)
    hf_model.eval()

    if vocab_size is not None:
      hf_model.resize_token_embeddings(vocab_size)

    self.embeddings = hf_model.roberta.embeddings
    self.classifier = hf_model.classifier

    # Load the rest parts from the quantized ONNX model
    self.qnpu_session = QNPUModule(qnpu_model_path, **qnpu_config)

  def forward(self, input_ids, attention_mask,
              **kwargs) -> TokenClassifierOutput:
    input_embeds = self.embeddings(
      input_ids=input_ids,
      token_type_ids=kwargs.get('model_config', None),
    )
    attention_mask = _create_4d_mask(attention_mask, input_ids.shape)

    inputs = {
      'hidden_states': input_embeds,
      'attention_mask': attention_mask
    }
    outputs = self.qnpu_session.run(**inputs)
    logits = self.classifier(outputs[0])

    return TokenClassifierOutput(logits=logits)


class QNPUXLMRobertaForTCL(torch.nn.Module):
  def __init__(self, model_name: str, qnpu_model_path: Path,
               /, *args, **kwargs) -> None:
    config = kwargs.pop('model_config', {})
    vocab_size = kwargs.pop('vocab_size', None)
    qnpu_config = kwargs.pop('qnpu_config', {})

    super().__init__(*args, **kwargs)

    # Load the Hugging Face model
    hf_model = XLMRobertaForTCL.from_pretrained(model_name, **config)
    hf_model.eval()

    if vocab_size is not None:
      hf_model.resize_token_embeddings(vocab_size)

    self.embeddings = hf_model.roberta.embeddings

    # Load the rest parts from the quantized ONNX model
    self.qnpu_session = QNPUModule(qnpu_model_path, **qnpu_config)

  def forward(self, input_ids, attention_mask,
              **kwargs) -> TokenClassifierOutput:
    input_embeds = self.embeddings(
      input_ids=input_ids,
      token_type_ids=kwargs.get('model_config', None),
    )
    attention_mask = _create_4d_mask(attention_mask, input_ids.shape)

    inputs = {
      'inputs_embeds': input_embeds,
      'attention_mask': attention_mask
    }
    logits = self.qnpu_session.run(**inputs)[0]

    return TokenClassifierOutput(logits=logits)


def create_llmlingua2_tokenizer(model_name: str, max_force_token: int = 100):
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  added_tokens = [f"[NEW{i}]" for i in range(max_force_token)]
  tokenizer.add_special_tokens(
    {"additional_special_tokens": added_tokens}  # type: ignore
  )

  return tokenizer
