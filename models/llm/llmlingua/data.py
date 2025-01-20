
from torch.utils.data import DataLoader, Dataset

from transformers import (AutoTokenizer, AutoModel)

from olive.data.component.dataset import BaseDataset
from olive.data.registry import Registry


@Registry.register_pre_process()
def llm_pre_process(dataset, model_name, input_cols, label_col="label", **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    target = model.encoder

    from inspect import signature
    params = [n for (n, _) in signature(target.forward).parameters.items()]

    inputs = {}

    def capture_inputs(_, args, kwargs):
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                inputs[params[i]] = arg[0,].clone()
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value[0,].clone()
        return None

    def inference(examples):
        inputs.clear()
        encoded_input = tokenizer(
            *[examples[input_col] for input_col in input_cols],
            padding='max_length',
            max_length=512,
            truncation=True,
            return_tensors='pt'
        )
        with torch.inference_mode():
            model(**encoded_input)
        inputs[label_col] = examples[label_col]
        return inputs.copy()

    target.register_forward_pre_hook(capture_inputs, with_kwargs=True)

    return BaseDataset(
        [inference(i) for i in dataset.select(range(4))],
        label_col
    )