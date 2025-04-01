from transformers import AutoModel

from bert_common import SimpleBert, SimpleDistilBert


def load_bert_model(model_name: str):
    model = AutoModel.from_pretrained(model_name).eval()
    return SimpleBert(model)


def load_distilbert_model(model_name: str):
    model = AutoModel.from_pretrained(model_name).eval()
    return SimpleDistilBert(model)
