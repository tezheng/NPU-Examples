from dataclasses import dataclass

from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import onnx
import torch

from .model import (
  LlamaBlock,
  LlamaWithKVCache,
  LogitsWithPast,
  TwoStagesMixin,
)
from .util import logger, parse_args


@dataclass
class ConversionConfig:
  skip_prefill: bool = False
  skip_decode: bool = False
  max_samples: int = 32


class CalibDataMixin(TwoStagesMixin):
  _prefill_data: list[dict[str, np.ndarray]] = []
  _decode_data: list[dict[str, np.ndarray]] = []
  _max_samples: int = 16

  def __init__(self, max_samples: int = 16, **kwargs):
    super().__init__()
    self._max_samples = max_samples

  def prefill(self,
              inputs: dict[str, torch.Tensor]) -> tuple[int, LogitsWithPast]:
    token, outputs = self._generate(inputs)
    self._prefill_data.append({
      **{f'i_{k}': v.numpy(force=True) for k, v in inputs.items()},
      **{f'o_{k}': v.numpy(force=True) for k, v in outputs.items()},
    })
    self._prefill_len = inputs['position_ids'].max()
    return token, outputs

  def decode(self,
             inputs: dict[str, torch.Tensor]) -> tuple[int, LogitsWithPast]:
    token, outputs = self._generate(inputs)
    self._decode_data.append({
      **{f'i_{k}': v.numpy(force=True) for k, v in inputs.items()},
      **{f'o_{k}': v.numpy(force=True) for k, v in outputs.items()},
    })

    if inputs['position_ids'].max() - self._prefill_len >= self._max_samples:
      token = self.eos_token_id

    return token, outputs

  @property
  def prefill_data(self):
    return self._prefill_data

  @property
  def decode_data(self):
    return self._decode_data


class CalibDataGenerator(LlamaWithKVCache, CalibDataMixin):
  def __init__(self, **kwargs):
    LlamaWithKVCache.__init__(self, **kwargs)
    CalibDataMixin.__init__(self, **kwargs)

  def save_data(self, data_dir: Path) -> None:
    logger.info(f"Persisting prefill data: {len(self.prefill_data)} items")
    self._save_data(self.prefill_data, data_dir / 'prefill')
    logger.info(f"Persisting decode data: {len(self.decode_data)} items")
    self._save_data(self.decode_data, data_dir / 'decode')
    logger.info('Calibration data persisted successfully!')

  def _save_data(self, data: list[dict[str, np.ndarray]],
                 file_path: Path) -> None:
    if len(data) == 0:
      raise RuntimeError('No data to save!')

    file_path = file_path.with_suffix('.npz')
    if (file_path.exists()):
      logger.warning(f"Remove existing calibration data: {file_path}")
      file_path.unlink()

    np.savez_compressed(
      file_path,
      allow_pickle=False,
      **{k: np.stack([i[k] for i in data]) for k in data[0]}
    )
    logger.info(f"Calibration data saved to {file_path}")


class ConvertONNXMixin(TwoStagesMixin):
  @dataclass
  class ModuleIO:
    input_values: dict[str, torch.Tensor]
    input_names: list[str]
    output_names: list[str]

  def prefill(self,
              inputs: dict[str, torch.Tensor]) -> tuple[int, LogitsWithPast]:
    token, outputs = self._generate(inputs)

    # To avoid naming conflicts with the model's input/output names
    self.prefill_input = ConvertONNXMixin.ModuleIO(
      input_values=inputs,
      input_names=['i_' + k for k in inputs.keys()],
      output_names=['o_' + k for k in outputs.keys()],
    )

    return token, outputs

  def decode(self,
             inputs: dict[str, torch.Tensor]) -> tuple[int, LogitsWithPast]:
    _, outputs = self._generate(inputs)

    # To avoid naming conflicts with the model's input/output names
    self.decode_input = ConvertONNXMixin.ModuleIO(
      input_values=inputs,
      input_names=['i_' + k for k in inputs.keys()],
      output_names=['o_' + k for k in outputs.keys()],
    )

    return self.eos_token_id, outputs


class ConvertONNX(LlamaWithKVCache, ConvertONNXMixin):
  def __init__(self, skip_prefill: bool = False,
               skip_decode: bool = False, **kwargs):
    LlamaWithKVCache.__init__(self, **kwargs, use_streaming=False)
    ConvertONNXMixin.__init__(self)
    self._skip_prefill = skip_prefill
    self._skip_decode = skip_decode

  def export(self, model_dir: Path) -> None:
    if not self._skip_prefill:
      self._convert_onnx(self._model, model_dir /
                         'prefill.onnx', self.prefill_input)

    if not self._skip_decode:
      self._convert_onnx(self._model, model_dir /
                         'decode.onnx', self.decode_input)

  def _convert_onnx(self, model: LlamaBlock, model_path: Path,
                    sample: ConvertONNXMixin.ModuleIO):
    with TemporaryDirectory(prefix='llama.to.onnx.') as tmp_path:
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

      logger.info(f"Saving onnx model to {model_path}")
      onnx.save_model(
        onnx.load_model(tmp_model),
        model_path,
        save_as_external_data=True,
        location=onnx_data_path.name,
      )
      logger.info("ONNX model exported successfully!")

  def _validate_onnx(self, model_path: Path):
    input = self.prompt('Hello world!')
    input = {'i_' + k: v.numpy(force=True) for k, v in input.items()}

    import onnxruntime as ort
    ort.set_default_logger_severity(0)
    ort.set_default_logger_verbosity(1)

    session = ort.InferenceSession(
      str(model_path),
      providers=['CPUExecutionProvider'],
    )
    logits, past_keys, past_values = session.run(None, input_feed=input)
    logger.debug(logits)
    logger.debug(past_keys.shape())


if __name__ == '__main__':
  import sys

  args = parse_args()

  model_name = args.model_name
  logger.info(f'Exporting model {model_name}')

  model_dir = args.model_dir
  model_dir.mkdir(parents=True, exist_ok=True)
  data_dir = args.data_dir
  data_dir.mkdir(parents=True, exist_ok=True)

  ground_truth = [{
    'input': '''<|start_header_id|>system<|end_header_id|>
You are a space exploration history expert.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Who is the first female astronaut to walk on the moon?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>''',
    'output': 'The first female astronaut to walk on the moon was Valentina \
               Tereshkova, a Soviet cosmonaut. She flew aboard the Vostok 6 \
               spacecraft on June 16, 1963, and spent almost three days in \
               space, becoming the first woman to journey into outer space.',
  }, {
    'input': '''<|start_header_id|>system<|end_header_id|>
You are a American history expert.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Who is the first presedent of United States?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>''',
    'output': 'The first President of the United States was George Washington. \
               He was inaugurated on April 30, 1789, and served two terms in \
               office until March 4, 1797.',
  }]

  config = ConversionConfig(
    skip_prefill=args.skip_prefill,
    skip_decode=args.skip_decode,
    max_samples=args.max_samples,
  )

  if args.inference:
    model = LlamaWithKVCache(model_name, use_streaming=args.use_streaming)
    if args.prompt:
      logger.info(f"Prompt:\n{args.prompt}")
      generated_text = model.run(args.prompt)
    else:
      for item in ground_truth:
        logger.info(f"Prompt:\n{item['input']}")
        generated_text = model.run(item['input'])
    sys.exit(0)

  if args.gen_calib_data:
    calib_data_gen = CalibDataGenerator(model_name=model_name, config=config)
    for item in ground_truth:
      logger.info(f"Prompt:\n{item['input']}")
      generated_text = calib_data_gen.run(item['input'])
    calib_data_gen.save_data(data_dir=data_dir)

  if args.convert_onnx:
    converter = ConvertONNX(model_name=model_name, config=config)
    converter.run('Hello world!')
    converter.export(model_dir)
