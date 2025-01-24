from pathlib import Path
from pprint import pprint

import numpy as np
import torch

from qnpumodel import (
  QNPUXLMRobertaForTCL,
  QNPUXLMRobertaForTCLEncoder,
  create_llmlingua2_tokenizer,
)

from evaluate import load


@torch.inference_mode()
def eval_meetingbank(
  data_path: Path,
  model_name: str,
  qnpu_model_path: Path,
  max_samples: int = 100,
  seq_length: int = 512,
  max_force_token: int = 100,
  **kwargs
):
  tokenizer = create_llmlingua2_tokenizer(
    model_name, max_force_token=max_force_token)
  model = QNPUXLMRobertaForTCL(
    model_name, qnpu_model_path, vocab_size=len(tokenizer))

  with np.load(data_path) as data:
    texts = list(data['prompts'])[:max_samples]
    targets = list(data['logits'])[:max_samples]

  encoded_input = tokenizer(
    texts,
    padding="max_length",
    max_length=seq_length,
    truncation=True,
    add_special_tokens=True,
    return_tensors="pt",
  )

  logits = model(**encoded_input)[0]

  f1 = load("f1")
  accu = load("accuracy")
  metrics = {
    "r00": 0.00,
    "r50": 0.50,
    "r30": 0.30,
  }

  def logits_to_label(rate, logits) -> torch.Tensor:
    probs = logits.softmax(dim=-1)[:, 1]
    threshold = np.percentile(probs, int(100 * rate + 1))
    return (probs > threshold).int()

  results = {}
  for key, rate in metrics.items():
    predictions = torch.cat([logits_to_label(rate, p)
                             for p in logits])
    references = torch.cat([logits_to_label(rate, torch.from_numpy(t))
                            for t in targets])
    accu_results = accu.compute(
      predictions=predictions,
      references=references,
    )
    f1_results = f1.compute(
      predictions=predictions,
      references=references,
    )
    results[key] = {**(f1_results or {}), **(accu_results or {})}

  return results


if __name__ == "__main__":
  model_name = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
  qdq_model = "microsoft/llmlingua-2-xlm-roberta-large"

  root = Path(__file__).resolve().parent
  data_path = root / "data/contexts.npz"
  qnpu_model_path = root / "outputs" / qdq_model / "model/model.onnx"

  results = eval_meetingbank(
    data_path=data_path,
    model_name=model_name,
    qnpu_model_path=qnpu_model_path,
  )

  pprint(results)
