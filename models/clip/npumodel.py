from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader


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

        providers = ["QNNExecutionProvider"]
        provider_options = [
            {
                "backend_path": "QnnHtp.dll",
                "htp_performance_mode": htp_performance_mode,
                "htp_graph_finalization_optimization_mode": htp_graph_opt_mode,
            },
        ]

        if disable_cpu_fallback == "0":
            providers.append("CPUExecutionProvider")
            provider_options.append({})

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
            providers=providers,
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
    def latency(self):
        latencies = np.concatenate(
            [
                np.full(num_batches, total_time / num_batches * 1000)
                for num_batches, total_time in self._latency_trace
            ]
        )
        return (
            round(np.mean(latencies).item(), 2),
            round(np.percentile(latencies, 90).item(), 2),
        )


from transformers.modeling_outputs import ModelOutput as _ModelOutput


def create_4d_mask(
    mask: torch.Tensor,
    input_shape: Union[torch.Size, Tuple[int, int]],
    masked_value: float = -50.0,
) -> torch.Tensor:
    # (batch_size, num_heads, seq_len, head_dim)
    batch_sz, seq_len = input_shape
    expanded_mask = mask[:, None, None, :].expand(batch_sz, 1, seq_len, seq_len)
    inverted_mask = 1.0 - expanded_mask.float()
    return inverted_mask.masked_fill(inverted_mask.bool(), masked_value)


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

    def __repr__(self):
        return super().__repr__()


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
        position_ids: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        # batch_sz, seq_length = input_ids.shape
        # assert seq_length == self.qnpu_session.sequence_length
        # if token_type_ids is None:
        #     token_type_ids = torch.zeros(input_ids.shape).long()
        if attention_mask.dim() == 2:
            attention_mask = create_4d_mask(attention_mask, input_ids.shape)

        inputs = {
            "input_ids": input_ids.int(),
            "attention_mask": attention_mask.float(),
        }
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids.int()
        if position_ids is not None:
            inputs["position_ids"] = position_ids.int()

        return ModelOutput(**self.qnpu_session.run(inputs))

    @property
    def sequence_length(self):
        return self.qnpu_session.session.get_inputs()[0].shape[1]


class QNPUImageModel(torch.nn.Module):
    def __init__(self, model_path: Path, device="npu", *args, **kwargs) -> None:
        qnpu_config = kwargs.pop("qnpu_config", {})

        super().__init__(*args, **kwargs)

        self.npu_session = QNPUModule(model_path, device, **qnpu_config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        **kwargs,
    ) -> ModelOutput:
        inputs = {"pixel_values": pixel_values}
        outputs = self.npu_session.run(inputs)
        return ModelOutput(**outputs)

    def inference(self, dataset, batch_size=128):
        outputs = [
            self.forward(**batch)
            for batch in DataLoader(dataset, batch_size=batch_size)
        ]

        return ModelOutput(
            **{k: torch.cat([o[k] for o in outputs], dim=0) for k in outputs[0].keys()}
        )


class QNPUCLIPModel:
    def __init__(
        self,
        model_name: str,
        text_model_path: Path,
        vision_model_path: Path,
    ) -> None:
        from transformers import CLIPProcessor

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.text_model = QNPUModule(text_model_path)
        self.vision_model = QNPUModule(vision_model_path)

    def get_image_embedding(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        output = self.vision_model.run(inputs)
        return output["embeds"]

    def get_text_embedding(self, text):
        inputs = self.processor(
            text=text,
            padding="max_length",
            max_length=self.text_model.sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        output = self.text_model.run(
            {
                "input_ids": inputs["input_ids"].int(),
                "attention_mask": self._create_4d_mask(
                    inputs["attention_mask"],
                    inputs["input_ids"].shape,
                ),
            }
        )
        return output["embeds"]

    def _create_4d_mask(self, mask, input_shape, masked_value=-50.0):
        # (batch_size, num_heads, seq_len, head_dim)
        batch_sz, seq_len = input_shape
        expanded_mask = mask[:, None, None, :].expand(batch_sz, 1, seq_len, seq_len)
        inverted_mask = 1.0 - expanded_mask.float()
        return inverted_mask.masked_fill(inverted_mask.bool(), masked_value)
