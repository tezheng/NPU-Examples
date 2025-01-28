from transformers import (
    AutoModelForSequenceClassification as AutoModelSCL,
)

from bert_common import SimpleBert
import bert_script  # noqa: F401


def load_model(model_name: str):
    model = AutoModelSCL.from_pretrained(model_name)
    model.eval()
    model.bert = SimpleBert(model.bert)
    return model
