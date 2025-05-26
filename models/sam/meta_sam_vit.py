from typing import List
from transformers import SamModel, SamProcessor

from olive.data.component.dataset import BaseDataset
from olive.data.registry import Registry


def load_vision_encoder(model_name: str):
    model = SamModel.from_pretrained(model_name).eval()
    return model.vision_encoder.eval()


def load_vision_neck(model_name: str):
    model = SamModel.from_pretrained(model_name).eval()
    return model.vision_encoder.neck


def load_prompt_encoder(model_name: str):
    model = SamModel.from_pretrained(model_name).eval()
    return model.prompt_encoder.eval()


@Registry.register_pre_process()
def process_dataset(
    dataset, model_name: str, input_cols: List[str], label_col: str, **kwargs
):
    processor = SamProcessor.from_pretrained(model_name)

    def process_fn(x):
        inputs = processor(images=x[input_cols[0]], return_tensors="pt")  # type: ignore
        labels = x[label_col]
        return {"pixel_values": inputs.pixel_values, label_col: labels}

    processed_ds = dataset.map(
        process_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )
    processed_ds.set_format("torch", output_all_columns=True)

    return BaseDataset(processed_ds, label_col=label_col)
