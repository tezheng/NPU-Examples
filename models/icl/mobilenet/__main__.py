import argparse
from pathlib import Path

from olive.workflows import run


def parse_args():
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("--config", type=str, required=True)
  return arg_parser.parse_args()


if __name__ == "__main__":
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("--config", type=str, required=True)
  args = arg_parser.parse_args()

  workflow_path = Path(args.config).resolve()
  if not workflow_path.exists():
    cwd = Path(__file__).resolve().parent
    workflow_path = cwd / args.config
    if not workflow_path.exists():
      raise FileNotFoundError(
        f"Workflow configuration file not found: {workflow_path}")

  print(f"QDQ MobileNetV2 with config: {workflow_path}")

  run(run_config=workflow_path)
