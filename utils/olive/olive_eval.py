from typing import cast, Dict, Optional
from pathlib import Path
from time import time
import json

from pydantic import ValidationError

from olive.evaluator.metric_result import MetricResult
from olive.model.config import ModelConfig
from olive.package_config import OlivePackageConfig
from olive.resource_path import create_resource_path, LocalFile
from olive.systems.accelerator_creator import create_accelerators
from olive.systems.olive_system import OliveSystem
from olive.systems.system_config import SystemConfig
from olive.workflows.run.config import RunConfig

from utils.logger import get_logger

logger = get_logger("Evaluate")


def load_systems(config_file: Path) -> Dict[str, SystemConfig]:
    try:
        with open(config_file, "r") as f:
            data = json.load(f)

        result = {}
        for key, value in data.items():
            try:
                result[key] = SystemConfig(**value)
            except ValidationError as e:
                print(f"Validation error for key {key}: {e}")
                raise
        return result
    except FileNotFoundError:
        print(f"File {config_file} not found")
        raise
    except json.JSONDecodeError:
        print(f"Invalid JSON in {config_file}")
        raise


def evaluate(
    config_file: Path,
    evaluator: Optional[str] = None,
    target: Optional[str] = None,
    extra_systems: Optional[Dict[str, SystemConfig] | Path] = None,
):
    import os

    logger.info(f"Set working directory to {config_file.parent}")
    os.chdir(config_file.parent)

    logger.info(f"Parsing Olive config file {config_file} ...")
    run_config = cast(RunConfig, RunConfig.parse_file_or_obj(config_file))
    package_config = OlivePackageConfig.parse_file_or_obj(
        OlivePackageConfig.get_default_config_path()
    )

    logger.info("Creating Olive engine ...")
    engine = run_config.engine.create_engine(
        olive_config=package_config,
        azureml_client_config=None,
        workflow_id=run_config.workflow_id,
    )
    engine.initialize()

    logger.info("Creating target system ...")
    start_time = time()
    if target is not None:
        systems = (
            load_systems(extra_systems)
            if isinstance(extra_systems, Path)
            else extra_systems or run_config.systems
        )
        target_config = systems.get(target, engine.target_config)
    else:
        target_config = engine.target_config

    accelerator_specs = create_accelerators(
        target_config,
        skip_supported_eps_check=False,
        is_ep_required=True,
    )

    target_system: OliveSystem = target_config.create_system()
    logger.info("Target system created in %f seconds", time() - start_time)

    # load converted model from output dir
    p = Path(run_config.engine.output_dir) / "model_config.json"
    if not p.exists():
        raise FileNotFoundError(f"Model config file {p} does not exist.")

    logger.info("Parsing model config ...")
    model_config_file: LocalFile = cast(LocalFile, create_resource_path(p))
    model_config = cast(
        ModelConfig,
        ModelConfig.parse_file_or_obj(model_config_file.get_path()),
    )

    logger.info("Parsing evaluator config ...")
    evaluator_config = (
        run_config.evaluators.get(
            evaluator,
            engine.evaluator_config,
        )
        if evaluator is not None
        else engine.evaluator_config
    )
    if evaluator_config is None:
        raise ValueError(
            "Evaluator is either not specified or doesn't exist. Available "
            "evaluators are: {}".format(list(run_config.evaluators.keys()))
        )

    logger.info("Evaluating model ...")
    result: MetricResult = target_system.evaluate_model(
        model_config=model_config,
        evaluator_config=evaluator_config,
        accelerator=accelerator_specs[0],
    )
    logger.info("Evaluation result: %s", result.to_json())


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ONNX model that is converted with Olive using ONNX Runtime."
        )
    )
    parser.add_argument(
        "--run-config",
        "--config",
        type=str,
        help="Path to json config file",
        required=True,
    )
    parser.add_argument(
        "--tempdir",
        type=str,
        help="Root directory for tempfile directories and files",
        required=False,
    )
    parser.add_argument(
        "--evaluator",
        type=str,
        help="Evaluator name to use for evaluation",
        required=False,
    )
    parser.add_argument(
        "--target",
        type=str,
        help="Target system to use for evaluation",
        required=False,
    )
    parser.add_argument(
        "--system-config",
        type=Path,
        help="Path to json config file for extra systems",
        required=False,
    )

    return parser.parse_args()


if __name__ == "__main__":
    from utils.logger import set_logger_level

    set_logger_level(logger, "INFO")

    args = parse_args()

    config_file = Path(args.run_config).resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"Config file {config_file} does not exist.")

    evaluate(
        config_file,
        evaluator=args.evaluator,
        target=args.target,
        extra_systems=args.system_config,
    )
