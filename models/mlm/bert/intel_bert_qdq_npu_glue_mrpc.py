from transformers import (
    AutoModelForSequenceClassification as AutoModelSCL,
)

from bert_common import SimpleBert
from bert_script import bert_scl_post_process  # noqa: F401


def load_model(model_name: str):
    model = AutoModelSCL.from_pretrained(model_name)
    model.eval()
    model.bert = SimpleBert(model.bert)
    return model
