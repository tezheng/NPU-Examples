from transformers import (
    AutoModelForCausalLM,
)

def load_model(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return model
