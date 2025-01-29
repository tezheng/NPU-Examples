import argparse
from pathlib import Path

from utils.logger import get_logger
logger = get_logger('llamaqdq')


def parse_args():
  default_model_dir = Path.cwd()
  default_data_dir = Path.cwd()

  parser = argparse.ArgumentParser()
  parser.add_argument('--model-name', type=str,
                      default='deepseek-ai/deepseek-r1-distill-qwen-1.5b',
                      help='HF hub model to initialize tokenizer')
  parser.add_argument('--model-dir', type=Path, default=default_model_dir)
  parser.add_argument('--data-dir', type=Path, default=default_data_dir,
                      help='Directory to store calibration data')
  parser.add_argument('--inference', action='store_true', default=False,
                      help='Inference using pytorch with cpu, will ignore \
                        other operation flags, --convert-onnx, --quantize-qdq, \
                        etc.')
  parser.add_argument('--prompt', type=str,
                      help='Override default prompts, use with --inference')
  parser.add_argument('--all', action='store_true', default=False,
                      help='Run all steps')
  parser.add_argument('--convert-onnx', action='store_true', default=False)
  parser.add_argument('--gen-calib-data', action='store_true', default=False)
  parser.add_argument('--quantize-qdq', action='store_true', default=False)
  parser.add_argument('--use-streaming', action='store_true', default=True)
  parser.add_argument('--max-calib-len', type=int, default=32,
                      help='Number of tokens to persist for calibration')
  parser.add_argument('--skip-prefill', action='store_true', default=False)
  parser.add_argument('--skip-decode', action='store_true', default=False)
  parser.add_argument('--skip-pre-process', action='store_true', default=False)
  parser.add_argument('--node-optimization', action='store_true', default=False)
  parser.add_argument('--verbose', action='store_true', default=False)

  args, _ = parser.parse_known_args()

  level = 'DEBUG' if args.verbose else 'INFO'
  logger.setLevel(level)
  for handler in logger.handlers:
    handler.setLevel(level)

  args.model_dir = args.model_dir / 'outputs' / args.model_name
  args.data_dir = args.data_dir / 'outputs' / 'data'

  logger.debug(f"Arguments:\n {args}")

  return args
