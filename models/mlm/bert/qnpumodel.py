from typing import Dict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from tqdm import tqdm

from bert_common import ModelOutput


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
            prof_path = Path(kwargs.get(
                'qnpu_prof_file_path', 'qnpu_profile.csv'))
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
        self._seqence_length = self.session.get_inputs()[0].shape[1]

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


class QNPUBertForTokenClassification(torch.nn.Module):
    def __init__(self, qnpu_model_path: Path,
                 /, *args, **kwargs) -> None:
        qnpu_config = kwargs.pop('qnpu_config', {})

        super().__init__(*args, **kwargs)

        self.qnpu_session = QNPUModule(qnpu_model_path, **qnpu_config)

    def forward(self, input_ids, attention_mask, token_type_ids, position_ids):
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'position_ids': position_ids,
        }
        outputs = self.qnpu_session.run(inputs)
        return ModelOutput(**outputs)
