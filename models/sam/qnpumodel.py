from typing import Dict, Tuple, Union
from pathlib import Path

import numpy as np
import onnxruntime as ort


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
            }
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
            providers=["QNNExecutionProvider"],
            provider_options=provider_options,
        )

    def run(
        self,
        tensors: Dict[str, np.ndarray],
        return_dict=False,
    ) -> Union[Tuple, Dict[str, np.ndarray]]:
        inputs = {k: v for k, v in tensors.items() if k in self._input_names}

        missing_inputs = self._input_names - inputs.keys()
        if missing_inputs:
            raise RuntimeError(f"Missing inputs for ONNX model: {missing_inputs}")

        outputs = self.session.run(None, inputs)

        return dict(zip(self._outputs_names, outputs)) if return_dict else outputs

    def run_batch(
        self,
        data: np.ndarray,
        batch_size: int,
        return_dict=False,
    ) -> Union[Tuple, Dict[str, np.ndarray]]:
        raise NotImplementedError("Batch execution is not supported yet")

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sequence_length(self):
        return self._seqence_length
