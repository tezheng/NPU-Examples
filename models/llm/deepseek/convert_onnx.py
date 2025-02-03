from typing import List, Dict, Tuple
from dataclasses import asdict, dataclass

from tempfile import TemporaryDirectory
from pathlib import Path

import onnx
import torch

from .model import (
  Qwen2WithKVCache2 as Qwen2WithKVCache,
  ModelOutput,
  TwoStagesMixin,
)
from .util import logger


@dataclass
class ConversionConfig:
  skip_prefill: bool = False
  skip_decode: bool = False
  to_fp16: bool = False

  @classmethod
  def from_kwargs(cls, kwargs) -> 'ConversionConfig':
    return cls(
      skip_prefill=kwargs.pop('skip_prefill', False),
      skip_decode=kwargs.pop('skip_decode', False),
      to_fp16=kwargs.pop('to_fp16', False),
    )


class ConvertONNXMixin(TwoStagesMixin):
  @dataclass
  class ModuleIO:
    input_values: Dict[str, torch.Tensor]
    input_names: List[str]
    output_names: List[str]

  def prefill(self, inputs: Dict[str, torch.Tensor]) -> Tuple[int, ModelOutput]:
    token, outputs = self._generate(inputs)

    # To avoid naming conflicts with the model's input/output names
    self.prefill_input = ConvertONNXMixin.ModuleIO(
      input_values=inputs,
      input_names=list(inputs.keys()),
      output_names=list(outputs.keys()),
    )

    return token, outputs

  def decode(self, inputs: Dict[str, torch.Tensor]) -> Tuple[int, ModelOutput]:
    _, outputs = self._generate(inputs)

    # To avoid naming conflicts with the model's input/output names
    self.decode_input = ConvertONNXMixin.ModuleIO(
      input_values=inputs,
      input_names=list(inputs.keys()),
      output_names=list(outputs.keys()),
    )

    return self.eos_token_id, outputs


class ConvertONNX(Qwen2WithKVCache, ConvertONNXMixin):
  def __init__(self, model_name: str, **kwargs) -> None:
    self.convert_cfg = ConversionConfig.from_kwargs(kwargs)
    super().__init__(model_name, **kwargs, use_streaming=False)

  def export(self, model_dir: Path) -> None:
    if not self.convert_cfg.skip_prefill:
      self._convert_onnx(self.model, model_dir /
                         'prefill.onnx', self.prefill_input)

    if not self.convert_cfg.skip_decode:
      self._convert_onnx(self.model, model_dir /
                         'decode.onnx', self.decode_input)

  def _convert_onnx(self, model: torch.nn.Module, model_path: Path,
                    sample: ConvertONNXMixin.ModuleIO):
    with TemporaryDirectory(prefix='qwen2.to.onnx.') as tmp_path:
      tmp_model = Path(tmp_path) / 'model.onnx'

      logger.info(f"Exporting pytorch model to {tmp_model}")
      torch.onnx.export(
        model,
        f=str(tmp_model),
        kwargs=sample.input_values,
        input_names=sample.input_names,
        output_names=sample.output_names,
        verbose=False,
      )

      onnx_data_path = model_path.with_suffix('.onnx.data')
      if model_path.exists():
        logger.warning(f"Remove existing model: {model_path}")
        model_path.unlink()
      if onnx_data_path.exists():
        logger.warning(f"Remove existing model data: {onnx_data_path}")
        onnx_data_path.unlink()

      onnx_model = onnx.load_model(tmp_model)
      if self.convert_cfg.to_fp16:
        from onnxruntime.transformers.onnx_model import OnnxModel
        om = OnnxModel(onnx_model)
        om.convert_float_to_float16(
          use_symbolic_shape_infer=True,
          **{
            "min_positive_val": 1e-5,
            "max_finite_val": 1e4,
            "keep_io_types": True,
            "op_block_list": None,
            "node_block_list": None,
          },
        )
        onnx_model = om.model

      logger.info(f"Saving onnx model to {model_path}")
      onnx.save_model(
        onnx_model,
        model_path,
        save_as_external_data=True,
        location=onnx_data_path.name,
      )
      logger.info("ONNX model exported successfully!")

  def _validate_onnx(self, model_path: Path):
    input = self.prompt('Hello world!')
    input = {k: v.numpy(force=True) for k, v in input.items()}

    import onnxruntime as ort
    ort.set_default_logger_severity(0)
    ort.set_default_logger_verbosity(1)

    session = ort.InferenceSession(
      str(model_path),
      providers=['CPUExecutionProvider'],
    )
    logits, past_keys, _ = session.run(None, input_feed=input)
    logger.debug(logits)
    logger.debug(past_keys.shape())


if __name__ == '__main__':
  from .util import parse_args

  args = parse_args()

  model_name = args.model_name
  logger.info(f'Convert to onnx model: {model_name}')

  model_dir = args.model_dir
  model_dir.mkdir(parents=True, exist_ok=True)
  data_dir = args.data_dir
  data_dir.mkdir(parents=True, exist_ok=True)

  config = ConversionConfig(
    to_fp16=args.to_fp16,
    skip_prefill=args.skip_prefill,
    skip_decode=args.skip_decode,
  )

  converter = ConvertONNX(model_name, **asdict(config))
  converter.run('Hello world!')
  converter.export(model_dir)
