from transformers import (
    AutoModelForSequenceClassification as AutoModelSCL,
)

from bert_common import SimpleBert
import bert_script  # noqa: F401

import openvino as ov
print(f"Available devices: {ov.Core().available_devices}")

def load_model(model_name: str):
    model = AutoModelSCL.from_pretrained(model_name)
    model.eval()
    model.bert = SimpleBert(model.bert)
    return model
