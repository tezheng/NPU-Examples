
from pathlib import Path
from dataclasses import dataclass
from tempfile import TemporaryDirectory

import numpy as np

from onnxruntime.quantization import (
  CalibrationDataReader,
  QuantType,
  quantize,
)
from onnxruntime.quantization.execution_providers import qnn
from onnxruntime.quantization.shape_inference import quant_pre_process

# import onnxruntime as ort
# ort.set_default_logger_severity(0)
# ort.set_default_logger_verbosity(1)

from .util import logger, parse_args


class DataReader(CalibrationDataReader):
  _index = 0
  _datasize = 0

  def __init__(self, data_path) -> None:
    self._data = np.load(data_path)
    self._datasize = min(len(i) for i in self._data.values())

  def get_next(self) -> dict[str, np.ndarray] | None:
    if self._index < self._datasize:
      self._index += 1
      return {k: v[self._index - 1] for k, v in self._data.items()
              if k.startswith('i_')}
    return None

  def rewind(self) -> None:
    self._index = 0

  def __len__(self):
    return self._datasize


@dataclass
class QuantizationConfig:
  node_optimization: bool = False
  skip_pre_process: bool = False


def quant(model_path: Path, data_path: Path, output_dir: Path,
          config=QuantizationConfig()):
  model_output = output_dir / model_path.with_suffix('.wu8au16.onnx').name

  if not model_path.exists():
    raise FileNotFoundError(
      f"Model not found at: {model_path}, generate with --convert-onnx first")
  if not data_path.exists():
    raise FileNotFoundError(
      f"Calib data not found at: {data_path}, generate with --gen-calib-data first")

  for suffix in ['.wu8au16.onnx', '.wu8au16.onnx.data']:
    file_path = Path(output_dir / model_path.with_suffix(suffix).name)
    if file_path.exists():
      logger.warning(f"Remove existing moodel: {file_path}")
      file_path.unlink()

  with TemporaryDirectory(prefix='quant.qdq.') as tmp_dir:
    if not config.skip_pre_process:
      temp_path = Path(tmp_dir) / 'model.onnx'
      logger.info(f"Pre-processing {model_path} to {temp_path}")
      quant_pre_process(
        input_model=model_path,
        output_model_path=temp_path,
        save_as_external_data=True,
        external_data_location='model.onnx.data',
      )
      model_path = temp_path
    else:
      logger.info(f"Skip pre-processing {model_path}")

    quant_config = qnn.get_qnn_qdq_config(
      model_input=model_path,
      calibration_data_reader=DataReader(data_path),
      activation_type=QuantType.QUInt16,
      weight_type=QuantType.QUInt8,
    )
    if config.node_optimization:
      quant_config.nodes_to_exclude = (
        ['/model/model/rotary_emb/MatMul', '/Concat', '/Concat_1']
        + [f'/model/model/layers.{i}/self_attn/Expand' for i in range(16)]
        + [f'/model/model/layers.{i}/self_attn/Expand_1' for i in range(16)]
        + [f'/model/model/layers.{i}/self_attn/Unsqueeze_2' for i in range(16)]
        + [f'/model/model/layers.{i}/self_attn/Unsqueeze_3' for i in range(16)]
      )

    logger.info(f"QDQ to {model_output} with calib data from {data_path}")
    quantize(
      model_input=model_path,
      model_output=model_output,
      quant_config=quant_config,
    )

  logger.info('Model quantized successfully!')


if __name__ == '__main__':
  args = parse_args()

  model_dir = args.model_dir
  data_dir = args.data_dir

  config = QuantizationConfig(
    node_optimization=args.node_optimization,
    skip_pre_process=args.skip_pre_process,
  )

  if not args.skip_prefill:
    quant(
      model_path=model_dir / 'prefill.onnx',
      data_path=data_dir / 'prefill.npz',
      output_dir=model_dir,
      config=config,
    )

  if not args.skip_decode:
    quant(
      model_path=model_dir / 'decode.onnx',
      data_path=data_dir / 'decode.npz',
      output_dir=model_dir,
      config=config,
    )
