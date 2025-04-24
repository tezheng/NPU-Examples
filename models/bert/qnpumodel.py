from typing import Dict, Optional
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
import torch
from tqdm import tqdm

from bert_common import ModelOutput, create_4d_mask


class QNPUModule:
    def __init__(self, model_path: Path, device="npu", **kwargs) -> None:
        self.model_path = model_path

        if device == "npu":
            self._init_npu_session(**kwargs)
        elif device == "cpu":
            self._init_cpu_session(**kwargs)
        else:
            raise ValueError(f"QNPUModule does not support device: {device}")

        self._input_names = [i.name for i in self.session.get_inputs()]
        self._outputs_names = [o.name for o in self.session.get_outputs()]
        self._batch_size = self.session.get_inputs()[0].shape[0]
        self._seqence_length = self.session.get_inputs()[0].shape[1]
        self._latency_trace = []

    def _init_cpu_session(self, **kwargs):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        options = ort.SessionOptions()
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    def _init_npu_session(self, **kwargs):
        disable_cpu_fallback = kwargs.get("disable_cpu_fallback", "0")
        ep_context_enable = kwargs.get("ep_context_enable", "1")
        ep_context_embed = kwargs.get("ep_context_embed", "0")
        htp_performance_mode = kwargs.get("htp_performance_mode", "burst")
        htp_graph_opt_mode = kwargs.get("htp_graph_optimization_mode", "3")

        options = ort.SessionOptions()
        options.add_session_config_entry(
            "session.disable_cpu_ep_fallback", disable_cpu_fallback
        )

        if not str(self.model_path.name).endswith(".onnx_ctx.onnx"):
            epctx_model_path = self.model_path.with_suffix(".onnx_ctx.onnx")
            if epctx_model_path.exists():
                self.model_path = epctx_model_path
            else:
                options.add_session_config_entry("ep.context_enable", ep_context_enable)
                options.add_session_config_entry(
                    "ep.context_embed_mode", ep_context_embed
                )

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        provider_options = [
            {
                "backend_path": "QnnHtp.dll",
                "htp_performance_mode": htp_performance_mode,
                "htp_graph_finalization_optimization_mode": htp_graph_opt_mode,
            },
            {},
        ]

        if kwargs.get("qnpu_prof", False):
            prof_path = Path(kwargs.get("qnpu_prof_file_path", "qnpu_profile.csv"))
            prof_level = kwargs.get("qnpu_prof_level", "detailed")

            if prof_path.exists():
                prof_path.unlink()

            provider_options[0].update(
                {
                    "profiling_level": prof_level,
                    "profiling_file_path": str(prof_path),
                }
            )

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["QNNExecutionProvider", "CPUExecutionProvider"],
            provider_options=provider_options,
        )

    def run(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        inputs = {
            name: tensor.split(self._batch_size, dim=0)
            for name, tensor in tensors.items()
            if name in self._input_names
        }
        missing_inputs = self._input_names - inputs.keys()
        if missing_inputs:
            raise RuntimeError(f"Missing inputs for ONNX model: {missing_inputs}")

        # Split batches and convert torch tensors to numpy arrays
        batches = [
            dict(zip(inputs.keys(), [v.numpy() for v in values]))
            for values in zip(*inputs.values())
        ]
        # Run the ONNX model
        start = perf_counter()
        outputs = [self.session.run(None, batch) for batch in tqdm(batches)]
        self._latency_trace.append((len(batches), perf_counter() - start))

        return dict(
            zip(
                self._outputs_names,
                [torch.from_numpy(np.concatenate(a)) for a in zip(*outputs)],
            )
        )

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sequence_length(self):
        return self._seqence_length

    @property
    def latency(self):
        latencies = np.concatenate(
            [
                np.full(num_batches, total_time / num_batches * 1000)
                for num_batches, total_time in self._latency_trace
            ]
        )
        return (
            round(np.mean(latencies), 2),
            round(np.percentile(latencies, 90), 2),
        )


class QNPUBertModel(torch.nn.Module):
    def __init__(self, qnpu_model_path: Path, device="npu", *args, **kwargs) -> None:
        qnpu_config = kwargs.pop("qnpu_config", {})

        super().__init__(*args, **kwargs)

        self.qnpu_session = QNPUModule(qnpu_model_path, device, **qnpu_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        batch_sz, seq_length = input_ids.shape
        assert seq_length == self.qnpu_session.sequence_length

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_ids.shape).long()
        if attention_mask.dim() == 2:
            attention_mask = create_4d_mask(attention_mask, input_ids.shape)
        inputs = {
            "input_ids": input_ids.int(),
            "attention_mask": attention_mask.float(),
            "token_type_ids": token_type_ids.int(),
        }
        outputs = self.qnpu_session.run(inputs)
        return ModelOutput(**outputs)

    @property
    def sequence_length(self):
        return self.qnpu_session.session.get_inputs()[0].shape[1]
