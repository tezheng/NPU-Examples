from transformers import (
    BertModel,
    AutoModelForNextSentencePrediction as AutoModelNSP,
)

from bert_common import SimpleBert


def load_bert_model(model_name: str):
    model = BertModel.from_pretrained(model_name).eval()
    return SimpleBert(model)


def load_bert_nsp_model(model_name: str):
    model = AutoModelNSP.from_pretrained(model_name).eval()
    model.bert = SimpleBert(model.bert)
    return model
