
from pathlib import Path

from datasets import load_dataset

from .quant_qdq import QuantizationConfig, quant
from .util import logger, parse_args

from .model import QNPUXLMRobertaTCL
from .prompt_compressor import PromptCompressor

from time import time


def export():
  args = parse_args()

  model_dir = Path(
    "outputs/microsoft/llmlingua-2-xlm-roberta-large-meetingbank/model/").resolve()
  data_dir = Path("data/").resolve()

  if args.quantize_qdq:
    config = QuantizationConfig(
      node_optimization=args.node_optimization,
      skip_pre_process=args.skip_pre_process,
    )

    quant(
      model_path=model_dir / 'model.onnx',
      data_path=data_dir / 'data.npz',
      output_dir=model_dir,
      config=config,
    )

  model_name = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
  llm_lingua = PromptCompressor(
    model_name=model_name,
    use_llmlingua2=True,
    device_map="cpu"
  )

  qnpu_model = QNPUXLMRobertaTCL(
    model_name=model_name,
    encoder_path=model_dir / 'model.onnx'
  )
  llm_lingua.model = qnpu_model

  # Download the original prompt and dataset
  dataset = load_dataset("huuuyeah/meetingbank", split="test")
  context = dataset[0]["transcript"]

  time_start = time()
  # 2000 Compression
  compressed_prompt = llm_lingua.compress_prompt(
      context,
      rate=0.33,
      force_tokens=["!", ".", "?", "\n"],
      drop_consecutive=True,
  )
  print(f"Time taken: {time() - time_start})")

  print(f"Original Prompt:\n{context}")
  print(f"Compressed Prompt:\n{compressed_prompt}")
