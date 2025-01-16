
from typing import Dict, List, Union

from pathlib import Path

import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset

from transformers import (
  AutoConfig,
  AutoModelForTokenClassification,
  AutoTokenizer,
)
from transformers.modeling_outputs import (
  BaseModelOutputWithPastAndCrossAttentions as EncoderOutput,
  TokenClassifierOutput,
)
from transformers.models.xlm_roberta.modeling_xlm_roberta import (
  XLMRobertaForTokenClassification,
)
import onnxruntime as ort


def load_hf_model(model_name: str) -> nn.Module:
  model = AutoModelForTokenClassification.from_pretrained(model_name)
  return model.roberta.encoder


class QNPUXLMRobertaEncoder(nn.Module):
  _model_path: Path
  _session: ort.InferenceSession
  _input_names: List[str]
  _output_names: List[str]

  def __init__(self, model_path: Union[str, Path]):
    """Initializes the ONNXModelWrapper with the given model path.

    Args:
        model_path (str): Path to the ONNX model file.
    """
    super().__init__()
    self._model_path = Path(model_path)
    self._session = self._init_session(self._model_path)
    self._input_names = [i.name for i in self._session.get_inputs()]
    self._output_names = [o.name for o in self._session.get_outputs()]

  def forward(self, **kwargs) -> EncoderOutput:
    """
    Matches the kwargs with the model's inputs and runs inference if all required inputs are present.

    Args:
        kwargs (dict[str, torch.Tensor]): A dictionary of input names to torch.Tensor.

    Returns:
        dict[str, torch.Tensor]: A dictionary of output names to torch.Tensor.

    Raises:
        RuntimeError: If any required inputs are missing.
    """
    # Check if all required inputs are provided
    inputs = {name: tensor.detach().cpu()
              for name, tensor in kwargs.items() if name in self._input_names}
    missing_inputs = self._input_names - inputs.keys()
    if missing_inputs:
      raise RuntimeError(f"Missing inputs for ONNX model: {missing_inputs}")

    # Split batches and convert torch tensors to numpy arrays
    batches = [{key: value.numpy().astype(np.float32)
                for key, value in zip(inputs.keys(), values)}
               for values in zip(*[torch.split(tensor, 1, dim=0)
                                   for _, tensor in inputs.items()])]
    # Run the ONNX model
    logits = np.concatenate([self._session.run(None, batch)[0]
                             for batch in batches])

    # Map outputs back to a dictionary with output names
    return EncoderOutput(
      last_hidden_state=torch.FloatTensor(logits),
    )

  def _init_session(self, model_path: Path,
                    profile=False) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')

    onnx_ctx_path = self._model_path.with_suffix('.onnx_ctx.onnx')
    if onnx_ctx_path.exists():
      model_path = onnx_ctx_path
    else:
      options.add_session_config_entry('ep.context_enable', '1')
      options.add_session_config_entry('ep.context_embed_mode', '0')

    provider_options = [{
      'backend_path': 'QnnHtp.dll',
      'htp_performance_mode': 'burst',
      'htp_graph_finalization_optimization_mode': '3',
    }]

    if profile:
      prof_path = Path('qnpu_profile.csv')
      if prof_path.exists():
        prof_path.unlink()

      provider_options[0].update({
        'profiling_level': 'detailed',
        'profiling_file_path': str(prof_path),
      })

    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=['QNNExecutionProvider'],
        provider_options=provider_options,
    )


class QNPUXLMRobertaTCL(nn.Module):
  def __init__(self, model_name: str, encoder_path: Path, **kwargs) -> None:
    super().__init__()

    config: Dict = kwargs.get('model_config') or {}

    # Load the Hugging Face model
    hf_model = XLMRobertaForTokenClassification.from_pretrained(
      model_name,
      **config
    )

    # Load the Hugging Face model config and tokenizer
    self.config = AutoConfig.from_pretrained(model_name, **config)
    self.tokenizer = AutoTokenizer.from_pretrained(model_name, **config)

    # Load the encoder from the quantized ONNX model
    self.encoder = QNPUXLMRobertaEncoder(encoder_path)

    # Load the rest of the model from the Hugging Face model
    self.embeddings = hf_model.roberta.embeddings
    self.dropout = hf_model.dropout
    self.classifier = hf_model.classifier

  def forward(self, input_ids, attention_mask) -> TokenClassifierOutput:
    input_shape = input_ids.shape

    # Create the embeddings
    position_ids = self._create_position_ids(
      input_ids, self.config.pad_token_id)
    token_type_ids = self.embeddings.token_type_ids[:, :input_shape[-1]].expand(
      input_shape)
    embeds = self.embeddings(
      input_ids=input_ids,
      position_ids=position_ids,
      token_type_ids=token_type_ids,
    )

    # Create the attention mask
    attention_mask = self._create_4d_mask(attention_mask, input_shape)

    # Run the encoder
    inputs = {
      'hidden_states': embeds,
      'attention_mask': attention_mask
    }
    outputs = self.encoder(**inputs)

    # Run the classifier
    logits = self.classifier(self.dropout(outputs[0]))

    return TokenClassifierOutput(logits=logits)

  # (batch_size, seq_len) -> (batch_size, num_heads, seq_len, head_dim)
  def _create_4d_mask(self, mask, input_shape):
    batch_sz, seq_len = input_shape
    expanded_mask = mask[:, None, None, :].expand(batch_sz, 1, seq_len, seq_len)
    inverted_mask = 1.0 - expanded_mask.float()
    return inverted_mask.masked_fill(
      inverted_mask.to(torch.bool),
      torch.finfo(torch.float).min
    )

  # Copied from modeling_roberta.create_position_ids_from_input_ids
  def _create_position_ids(self, input_ids, padding_idx, past_key_values_len=0):
    """Replace non-padding symbols with their position numbers. Position
    numbers begin at padding_idx+1. Padding symbols are ignored. This is
    modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced
    # to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    indices = (torch.cumsum(mask, dim=1) + past_key_values_len) * mask
    return indices.long() + padding_idx


def DatasetForXLMRobetaEncoder(Dataset):
  def __init__(self, ):
    super().__init__(dataset)
