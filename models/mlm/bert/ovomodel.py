from typing import Dict, Optional
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from tqdm import tqdm

from bert_common import ModelOutput, create_4d_mask

import openvino as ov
print(f"OpenVINO avaialable devices: {ov.Core().available_devices}")


class OpenVINOModule():
    def __init__(self, model_path: Path, device='npu', **kwargs) -> None:
        self.model_path = model_path

        if device == 'npu':
            self._init_npu_session(**kwargs)
        elif device == 'cpu':
            self._init_cpu_session(**kwargs)
        else:
            raise ValueError(f"OpenVINOModule does not support device: {device}")

        self._input_names = [i.name for i in self.session.get_inputs()]
        self._outputs_names = [o.name for o in self.session.get_outputs()]
        self._batch_size = self.session.get_inputs()[0].shape[0]
        self._seqence_length = self.session.get_inputs()[0].shape[1]

    def _init_cpu_session(self, **kwargs):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        options = ort.SessionOptions()
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=['CPUExecutionProvider'],
        )

    def _init_npu_session(self, **kwargs):
        disable_cpu_fallback = kwargs.get('disable_cpu_fallback', '1')
        ctx_enable = kwargs.get('ep_context_enable', '1')
        ctx_embed = kwargs.get('ep_context_embed', '0')
        ctx_model_suffix = kwargs.get('ep_context_model_suffix', '.onnx_ctx.onnx')

        options = ort.SessionOptions()
        options.add_session_config_entry(
            'session.disable_cpu_ep_fallback', disable_cpu_fallback)

        # if not self.model_path.name.endswith(ctx_model_suffix):
        #     epctx_model_path = self.model_path.with_suffix(ctx_model_suffix)
        #     if epctx_model_path.exists():
        #         self.model_path = epctx_model_path
        #     else:
        #         options.add_session_config_entry(
        #             'ep.context_enable', ctx_enable)
        #         options.add_session_config_entry(
        #             'ep.context_embed_mode', ctx_embed)
        #         # options.add_session_config_entry(
        #         #     'ep.context_file_path', str(epctx_model_path))

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        else:
            print(f"Loading model with OpenVINO EP: {self.model_path}")

        provider_options = [{
            'device_type': 'NPU',
            'enable_qdq_optimizer': True,
            'cache_dir': str(self.model_path.parent),
        }]

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=options,
            providers=['OpenVINOExecutionProvider'],
            provider_options=provider_options,
        )

    def run(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        inputs = {name: tensor.split(self._batch_size, dim=0)
                  for name, tensor in tensors.items()
                  if name in self._input_names}
        missing_inputs = self._input_names - inputs.keys()
        if missing_inputs:
            raise RuntimeError(
                f"Missing inputs for ONNX model: {missing_inputs}")

        # Split batches and convert torch tensors to numpy arrays
        batches = [dict(zip(inputs.keys(), [v.numpy() for v in values]))
                   for values in zip(*inputs.values())]
        # Run the ONNX model
        outputs = [self.session.run(None, batch) for batch in tqdm(batches)]

        return dict(zip(
            self._outputs_names,
            [torch.from_numpy(np.concatenate(a)) for a in zip(*outputs)],
        ))

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sequence_length(self):
        return self._seqence_length


class OVOBertModel(torch.nn.Module):
    def __init__(self, ovo_model_path: Path, device="npu",
                 *args, **kwargs) -> None:
        npu_config = kwargs.pop("npu_config", {})

        super().__init__(*args, **kwargs)

        self.model = OpenVINOModule(ovo_model_path, device, **npu_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        batch_sz, seq_length = input_ids.shape
        assert seq_length == self.model.sequence_length

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_ids.shape).long()
        if position_ids is None:
            position_ids = torch.arange(seq_length).long().expand(batch_sz, -1)
        if attention_mask.dim() == 2:
            attention_mask = create_4d_mask(attention_mask, input_ids.shape)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'position_ids': position_ids,
        }
        outputs = self.model.run(inputs)
        return ModelOutput(**outputs)
