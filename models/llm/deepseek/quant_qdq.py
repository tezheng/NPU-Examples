
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

from .util import logger, parse_args


class DataReader(CalibrationDataReader):
  def __init__(self, data_path, input_names) -> None:
    data = {k: v for k, v in np.load(data_path).items() if k in input_names}
    self._data = [dict(zip(data.keys(), v)) for v in zip(*data.values())]
    self._index = 0

  def get_next(self) -> dict[str, np.ndarray] | None:
    if self._index < len(self):
      self._index += 1
      return self._data[self._index - 1]
    return None

  def rewind(self) -> None:
    self._index = 0

  def __len__(self):
    return len(self._data)


@dataclass
class QuantizationConfig:
  node_optimization: bool = False
  skip_pre_process: bool = False


def quant(model_path: Path, data_path: Path, output_dir: Path,
          config=QuantizationConfig()):
  model_suffix = '.wu8au16.onnx'
  model_output = output_dir / model_path.with_suffix(model_suffix).name

  if not model_path.exists():
    raise FileNotFoundError(
      f"Model not found at: {model_path}, generate with --convert-onnx first")
  if not data_path.exists():
    raise FileNotFoundError(
      f"Calib data not found at: {data_path}, generate with --gen-calib-data first")

  with TemporaryDirectory(prefix='quant.qdq.') as tmp_dir:
    if not config.skip_pre_process:
      temp_path = Path(tmp_dir) / 'fused.onnx'
      logger.info(f"QNN pre-processing {model_path} to {temp_path}")
      modified = qnn.qnn_preprocess_model(
          model_path,
          temp_path,
          fuse_layernorm=True,
          save_as_external_data=True,
          all_tensors_to_one_file=True,
      )
      if modified:
        model_path = temp_path

      temp_path = Path(tmp_dir) / 'model.onnx'
      logger.info(f"Quant pre-processing {model_path} to {temp_path}")
      quant_pre_process(
        input_model=model_path,
        output_model_path=temp_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        external_data_location='model.onnx.data',
      )
      model_path = temp_path
    else:
      logger.info(f"Skip pre-processing {model_path}")

    input_names = ['inputs_embeds', 'attention_mask',
                   'past_keys', 'past_values', 'position_sin', 'position_cos']
    quant_config = qnn.get_qnn_qdq_config(
      model_input=model_path,
      calibration_data_reader=DataReader(data_path, input_names),
      activation_type=QuantType.QUInt16,
      weight_type=QuantType.QUInt8,
    )
    if config.node_optimization:
      quant_config.nodes_to_exclude = (
        ['/_model/model/rotary_emb/MatMul', '/Concat', '/Concat_1']
        + [f'/_model/model/layers.{i}/self_attn/{name}'
           for i in range(28)
           for name in ["Expand", "Expand_1", "Unsqueeze_2", "Unsqueeze_3"]]
      )

    # quant_config.op_types_to_quantize = ['MatMul']
    # quant_config.quant_format = QuantFormat.QOperator

    for suffix in [model_suffix, model_suffix + '.data']:
      file_path = Path(output_dir / model_path.with_suffix(suffix).name)
      if file_path.exists():
        logger.warning(f"Remove existing model: {file_path}")
        file_path.unlink()

    # model_path = r"D:\NPU\NPU-Examples\models\llm\deepseek\outputs\deepseek-ai\decoder_pre_processed\model.onnx"
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
