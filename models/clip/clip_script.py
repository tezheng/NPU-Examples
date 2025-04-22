from typing import Dict, OrderedDict, Optional

import torch
import torch.nn.functional as F
from transformers import (
    AutoProcessor,
    CLIPModel,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)
from transformers.modeling_outputs import ModelOutput
from datasets import load_dataset
from datasets.utils.logging import enable_progress_bar

from olive.data.component.dataset import BaseDataset
from olive.data.registry import Registry
from olive.model import OliveModelHandler


class CLIPTextEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.embeddings = model.text_model.embeddings
        self.encoder = model.text_model.encoder
        self.final_layer_norm = model.text_model.final_layer_norm
        self.text_projection = model.text_projection

    def forward(
        self,
        input_ids,
        attention_mask,
    ):
        inputs_embeds = self.embeddings(input_ids)
        encoder_outputs = self.encoder(inputs_embeds, attention_mask)
        last_hidden_state = self.final_layer_norm(encoder_outputs[0])

        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0]),
            input_ids.argmax(dim=-1),
        ]
        text_embeds = self.text_projection(pooled_output)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)

        return ModelOutput(
            text_embeds=text_embeds,
            last_hidden_state=last_hidden_state,
        )


class CLIPImageEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vision_model = model.vision_model
        self.visual_projection = model.visual_projection

    def forward(self, pixel_values):
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        image_embeds = self.visual_projection(vision_outputs.pooler_output)
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)

        return ModelOutput(
            image_embeds=image_embeds,
            last_hidden_state=vision_outputs[0],
        )


class SimpleCLIPModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.text_encoder = CLIPTextEncoder(model)
        self.image_encoder = CLIPImageEncoder(model)
        self.logit_scale = model.logit_scale.exp().detach()

    @torch.inference_mode()
    def forward(self, input_ids, attention_mask, pixel_values):
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        image_outputs = self.image_encoder(
            pixel_values=pixel_values,
        )

        text_embeds = text_outputs.text_embeds
        image_embeds = image_outputs.image_embeds
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * self.logit_scale

        return ModelOutput(
            logits=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
        )


def load_clip_model(model_name):
    model = CLIPModel.from_pretrained(model_name).eval()
    return SimpleCLIPModel(model)


def load_text_encoder(model_name):
    model = CLIPTextModelWithProjection.from_pretrained(model_name).eval()
    return CLIPTextEncoder(model)


def load_image_encoder(model_name):
    return CLIPVisionModelWithProjection.from_pretrained(model_name).eval()


def load_torch_text_encoder(model_name):
    if model_name == "sentence-transformers/clip-ViT-B-32-multilingual-v1":
        from sbert_clip_script import SDistilBertTextEncoder

        return SDistilBertTextEncoder(model_name).eval()
    elif model_name == "sentence-transformers/clip-ViT-B-32":
        from sbert_clip_script import load_sbert_text_encoder

        return load_sbert_text_encoder(model_name)

    return CLIPTextModelWithProjection.from_pretrained(model_name).eval()


def load_torch_image_encoder(model_name):
    if model_name == "sentence-transformers/clip-ViT-B-32":
        from sbert_clip_script import load_sbert_image_encoder

        return load_sbert_image_encoder(model_name)

    return CLIPVisionModelWithProjection.from_pretrained(model_name).eval()


def hfdataset_pre_process_for_clip(
    dataset,
    processor,
    torch_model=None,
    image_col: Optional[str] = None,
    caption_col: Optional[str] = None,
    label_col: str = "label",
    max_samples: Optional[int] = None,
    max_length: int = 77,
    batch_size: int = 32,
):
    def create_4d_mask(mask, input_shape, masked_value: float = -50.0):
        # (batch_size, num_heads, seq_len, head_dim)
        batch_sz, seq_len = input_shape
        expanded_mask = mask[:, None, None, :].expand(batch_sz, 1, seq_len, seq_len)
        inverted_mask = 1.0 - expanded_mask.float()
        return inverted_mask.masked_fill(inverted_mask.bool(), masked_value)

    def generate_inputs(sample, indices):
        captions = sample.get(caption_col, None)
        images = sample.get(image_col, None)

        kwargs = {
            "padding": "max_length",
            "max_length": max_length,
            "truncation": True,
            "add_special_tokens": True,
            "return_tensors": "pt",
        }
        if images:
            kwargs["images"] = [img.convert("RGB") for img in images]
        if captions:
            kwargs["text"] = [x[0] for x in captions]

        encoded_input = processor(**kwargs)

        output = {
            label_col: torch_model(**encoded_input)[0]
            if torch_model
            else sample.get(label_col, indices)
        }
        if images:
            output.update({"pixel_values": encoded_input.pixel_values})
        if captions:
            output.update(
                {
                    "input_ids": encoded_input.input_ids,
                    "attention_mask": create_4d_mask(
                        encoded_input.attention_mask,
                        (encoded_input.input_ids.shape[0], max_length),
                    ),
                }
            )

        return output

    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    enable_progress_bar()
    tokenized_datasets = dataset.map(
        generate_inputs,
        batched=True,
        batch_size=batch_size,
        with_indices=True,
        remove_columns=dataset.column_names,
        desc="Processing dataset",
    )
    tokenized_datasets.set_format("torch", output_all_columns=True)

    return tokenized_datasets


@Registry.register_pre_process()
def pre_process_dataset(
    dataset,
    model_name: str,
    generate_ground_truth: bool = False,
    image_col: Optional[str] = None,
    caption_col: Optional[str] = None,
    label_col: str = "label",
    max_samples: Optional[int] = None,
    max_length: int = 77,
    **kwargs,
):
    if image_col is None and caption_col is None:
        raise ValueError("Either image_col or caption_col must be provided.")

    if generate_ground_truth:
        if image_col and caption_col:
            raise ValueError(
                "Can not generate two types of embedding at the same time."
            )

        torch_model = (
            load_torch_image_encoder(model_name)
            if image_col
            else load_torch_text_encoder(model_name)
        )

    processor = AutoProcessor.from_pretrained(model_name)
    dataset = hfdataset_pre_process_for_clip(
        dataset,
        processor,
        torch_model=torch_model,
        image_col=image_col,
        caption_col=caption_col,
        label_col=label_col,
        max_length=max_length,
        max_samples=max_samples,
    )
    return BaseDataset(dataset, label_col)


@Registry.register_post_process()
def embed_post_process(output):
    if isinstance(output, (Dict, OrderedDict)):
        if "embeds" in output:
            return output["embeds"]
        elif "text_embeds" in output:
            return output["text_embeds"]
        elif "image_embeds" in output:
            return output["image_embeds"]
    elif isinstance(output, torch.Tensor):
        return output.argmax(dim=-1)

    raise ValueError(f"Unsupported output type: {type(output)}")


@Registry.register_dataset()
def load_caption_dataset(data_name: str, split: str, input_col: str):
    ds = load_dataset(data_name, split=split)
    dataset = ds.map(
        lambda batch, indices: {
            input_col: [c[0] for c in batch[input_col]],
            "label": indices,
        },
        batched=True,
        with_indices=True,
        remove_columns=ds.column_names,
    )
    return dataset


def eval_similarity_degrad(output, targets, batch_size=32):
    from torch.utils.data import TensorDataset, DataLoader

    dataset = TensorDataset(output.preds, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    scores = [F.cosine_similarity(x, y) for x, y in dataloader]

    return {"percentage": f"{100.0 - torch.mean(torch.cat(scores)) * 100.0:.2f}"}


def eval_retieval(
    model: OliveModelHandler, device, execution_providers, batch_size=1, **kwargs
):
    pass
