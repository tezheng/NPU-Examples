
from __future__ import annotations

import argparse
from pathlib import Path
import timeit

import numpy as np
import onnxruntime

from qai_hub_models.models.whisper_base_en import (
  App as WhisperApp,
  Model as WhisperModel,
)


class QNPUModel:
  def __init__(self, model_path: Path) -> None:
    options = onnxruntime.SessionOptions()
    self.session = onnxruntime.InferenceSession(
      model_path,
      sess_options=options,
      providers=["QNNExecutionProvider"],
      provider_options=[{
        "backend_path": "QnnHtp.dll",
        "htp_performance_mode": "burst",
        "high_power_saver": "sustained_high_performance",
        "enable_htp_fp16_precision": "1",
        "htp_graph_finalization_optimization_mode": "3",
      }],
    )

  def to(self, *args):
    return self


class Encoder(QNPUModel):
  def __call__(self, audio):
    return self.session.run(None, {"audio": audio})


class Decoder(QNPUModel):
  def __call__(
      self, x, index, k_cache_cross, v_cache_cross, k_cache_self, v_cache_self
  ):
    return self.session.run(
      None,
      {
        "x": x.astype(np.int32),
        "index": np.array(index),
        "k_cache_cross": k_cache_cross,
        "v_cache_cross": v_cache_cross,
        "k_cache_self": k_cache_self,
        "v_cache_self": v_cache_self,
      },
    )


def load_demo_audio() -> tuple[np.ndarray, int]:
  from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

  demo_audio = CachedWebModelAsset.from_asset_store(
    "whisper_asr_shared", 1, "audio/jfk.npz"
  )
  demo_audio.fetch()
  with np.load(demo_audio.path()) as f:
    return f["audio"], 16000


def parse_args() -> argparse.ArgumentParser:
  from qai_hub_models.utils.args import get_model_cli_parser

  parser = get_model_cli_parser(WhisperModel)
  parser.add_argument(
      "--audio_file",
      type=str,
      default=None,
      help="Audio file path or URL",
  )
  return parser.parse_args()


def load_aihub_model() -> WhisperApp:
  # Model files
  root = Path(__file__).parent
  encoder_path = root / "build/whisper_base_en/WhisperEncoder.onnx"
  decoder_path = root / "build/whisper_base_en/WhisperDecoder.onnx"

  # Load whisper model
  model = WhisperModel(
      Encoder(encoder_path),
      Decoder(decoder_path),
      num_decoder_blocks=6,
      num_heads=8,
      attention_dim=512,
  )
  return WhisperApp(model)


def main():
  # Model files
  root = Path(__file__).parent
  encoder_path = root / "build/whisper_base_en/WhisperEncoder.onnx"
  decoder_path = root / "build/whisper_base_en/WhisperDecoder.onnx"

  # Parse arguments
  args = parse_args()
  audio = args.audio_file
  audio_sample_rate = None
  if not audio:
    audio, audio_sample_rate = load_demo_audio()

  # Load whisper model
  start_time = timeit.default_timer()
  app = load_aihub_model()

  # Execute Whisper Model
  infer_start_time = timeit.default_timer()
  transcription = app.transcribe(audio, audio_sample_rate)
  infer_done_time = timeit.default_timer()

  # Log execution time
  print("*" * 80)
  print("Model loading took:", round(
    infer_start_time - start_time, 2), "seconds")
  print("Transcribing took:", round(
    infer_done_time - infer_start_time, 2), "seconds")
  print("*" * 80)

  # Save transcription to file
  with open("transcript.txt", "w") as file:
    file.write(transcription)
    print("Transcription saved to transcript.txt")


if __name__ == "__main__":
  main()
