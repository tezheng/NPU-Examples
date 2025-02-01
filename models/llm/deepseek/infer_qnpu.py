from typing import List
from pathlib import Path

from time import perf_counter_ns

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from sequence import SinkSequence, SinkCache


def flatten(container):
  for i in container:
    if isinstance(i, (list, tuple)):
      yield from flatten(i)
    else:
      yield i


def init_ort_session(
    model_path: Path, profile=False, device="npu"
) -> ort.InferenceSession:
  if device == "cpu":
    if not model_path.exists():
      raise FileNotFoundError(f"Model file not found: {model_path}")
    return ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )

  options = ort.SessionOptions()
  options.add_session_config_entry("session.disable_cpu_ep_fallback", "0")

  onnx_ctx_path = model_path.with_suffix(".onnx_ctx.onnx")
  if onnx_ctx_path.exists():
    model_path = onnx_ctx_path
  else:
    options.add_session_config_entry("ep.context_enable", "1")
    options.add_session_config_entry("ep.context_embed_mode", "0")

  provider_options = [
      {
          "backend_path": "QnnHtp.dll",
          "enable_htp_fp16_precision": "1",
          "htp_performance_mode": "burst",
          "htp_graph_finalization_optimization_mode": "3"
      }
  ]

  if profile:
    prof_path = Path("qnpu_profile.csv")
    if prof_path.exists():
      prof_path.unlink()

    provider_options[0].update(
        {
            "profiling_level": "detailed",
            "profiling_file_path": "qnpu_profile.csv",
        }
    )

  if not model_path.exists():
    raise FileNotFoundError(f"Model file not found: {model_path}")

  return ort.InferenceSession(
      model_path,
      sess_options=options,
      providers=["QNNExecutionProvider"],
      provider_options=provider_options,
  )


class InferenceSession:
  _profile: bool = False

  def __init__(self, model_name: str, model_path: Path, profile: bool) -> None:
    self._profile = profile
    self._tokenizer = AutoTokenizer.from_pretrained(model_name)
    # self._decode = self._initiate_session(model_path / "decode.onnx")
    # self._prefill = self._initiate_session(model_path / "prefill.onnx")
    self._decode = self._initiate_session(model_path / "decode.wu8au16.onnx")
    self._prefill = self._initiate_session(model_path / "prefill.wu8au16.onnx")
    self._seq = SinkSequence(num_layers=28, num_heads=2, head_dim=128)

  def run(self, prompt: str) -> str:
    new_token, prefill_ts = self.prefill(prompt)
    print(self._tokenizer.decode(new_token), end="", flush=True)

    decode_ellapse = []
    while new_token != self._tokenizer.eos_token_id and not self._seq.full:
      new_token, decode_ts = self.decode(new_token)
      decode_ellapse.append(decode_ts)
      print(self._tokenizer.decode(new_token), end="", flush=True)

    print("\n")
    print(f"Prefill Latency: {prefill_ts / 1e6:.2f}ms")
    print(f"Decode Throughput: {1e9 / np.mean(decode_ellapse):.2f} tok/s")
    return "".join(self._tokenizer.batch_decode(self._seq.generated_tokens))

  def prefill(self, prompt: str) -> tuple[int, int]:
    # CoT
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    input_ids: List[int] = self._tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        max_length=self._seq.ctx_len,
        truncation=True,
    )  # type: ignore
    inputs = self._seq.prompt(input_ids)
    cache_shape = (
        self._seq.num_layers,
        self._seq.batch_size,
        self._seq.num_heads,
        self._seq.ctx_len - inputs["input_ids"].shape[-1],
        self._seq.head_dim,
    )
    self._kvcache = dict(
        i_past_keys=np.zeros(cache_shape, dtype=np.float32),
        i_past_values=np.zeros(cache_shape, dtype=np.float32),
    )

    log = f"""
Prompt(tokens={np.count_nonzero(inputs["input_ids"])}):
  {args.prompt}

Generation:"""
    print(log)

    start_ts = perf_counter_ns()
    outputs = self._prefill.run(
        None,
        input_feed={**{"i_" + k: v for k, v in inputs.items()}, **
                    self._kvcache},
    )
    end_ts = perf_counter_ns()

    return self._process_output(outputs), end_ts - start_ts

  def decode(self, new_token: int) -> tuple[int, int]:
    inputs = self._seq.token(new_token)

    start_ts = perf_counter_ns()
    outputs = self._decode.run(
        None,
        input_feed={**{"i_" + k: v for k, v in inputs.items()}, **
                    self._kvcache},
    )
    end_ts = perf_counter_ns()

    return self._process_output(outputs), end_ts - start_ts

  def _process_output(self, output: tuple[np.ndarray, ...]) -> int:
    logits, past_keys, past_values = output
    self._kvcache = dict(
        i_past_keys=past_keys[:, :, :, 1:, :],
        i_past_values=past_values[:, :, :, 1:, :],
    )
    return np.argmax(logits, axis=-1)[0]

  def _initiate_session(self, model_path: Path) -> ort.InferenceSession:
    return init_ort_session(model_path)


class Decoder:
  def __init__(
      self, model_name: str, model_path: Path, profile: bool, device="npu"
  ) -> None:
    self._tokenizer = AutoTokenizer.from_pretrained(model_name)
    self._decoder = init_ort_session(model_path, profile=profile, device=device)
    self._seq = SinkSequence(num_layers=28, num_heads=2, head_dim=128)
    self._kvcache = SinkCache(self._seq)
    self._input_names = [i.name for i in self._decoder.get_inputs()]

  def generate(self, prompt: str) -> str:
    # CoT
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    input_ids: List[int] = self._tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        max_length=self._seq.ctx_len,
        truncation=True,
    )  # type: ignore

    prefill_ellapse = []
    for token in input_ids:
      new_token, decode_ts = self.predict(token)
      prefill_ellapse.append(decode_ts)

    decode_ellapse = []
    while new_token != self._tokenizer.eos_token_id and not self._seq.full:
      print(self._tokenizer.decode(new_token), end="", flush=True)
      new_token, decode_ts = self.predict(new_token)
      decode_ellapse.append(decode_ts)

    print(self._tokenizer.decode(new_token), end="", flush=True)
    print("\n")
    print(f"Prefill Latency: {np.sum(prefill_ellapse) / 1e6: .2f}ms")
    print(
        f"Decode Throughput(Tokens={len(decode_ellapse)}): {1e9 / np.mean(decode_ellapse):.2f} tok/s"
    )
    return "".join(self._tokenizer.batch_decode(self._seq.generated_tokens))

  def predict(self, new_token: int) -> tuple[int, int]:
    inputs = self._seq.token(new_token)
    caches = self._kvcache.slice(inputs["input_ids"].shape[-1])
    input_feed = {k: v for k, v in (inputs | caches).items()
                  if k in self._input_names}

    start_ts = perf_counter_ns()
    logits, new_keys, new_values = self._decoder.run(None, input_feed)
    end_ts = perf_counter_ns()

    token = np.argmax(logits, axis=-1)[0]
    self._kvcache.update(new_keys=new_keys, new_values=new_values)

    return token, end_ts - start_ts


def parse_args():
  import argparse

  default_model_dir = Path.cwd()

  parser = argparse.ArgumentParser(
    description="Inference DeepSeek model with ONNX Runtime")
  parser.add_argument(
      "--model-name",
      type=str,
      default="deepseek-ai/deepseek-r1-distill-qwen-1.5b",
      help="HF hub model to initialize tokenizer",
  )
  parser.add_argument("--model-dir", type=Path, default=default_model_dir)
  parser.add_argument("--prompt", type=str,
                      default="Who is the first astronaut walking on the moon?")
  parser.add_argument("--profile", action="store_true", default=False)
  parser.add_argument("--device", type=str, default="npu")

  return parser.parse_known_args()[0]


if __name__ == "__main__":
  # ort.set_default_logger_severity(0)
  # ort.set_default_logger_verbosity(1)

  args = parse_args()

  model_path = args.model_dir / \
      "outputs/deepseek/deepseek_r1_distill_qwen_1.5b/model/model.onnx"
  model_path = args.model_dir / "outputs" / args.model_name / "decode.onnx"
  session = Decoder(
      model_name=args.model_name,
      model_path=model_path,
      profile=args.profile,
      device=args.device,
  )
  output = session.generate(args.prompt)

  # session = InferenceSession(
  #   model_name=args.model_name,
  #   model_path=args.model_dir / "outputs" / args.model_name,
  #   profile=args.profile
  # )
  # output = session.run(args.prompt)
