import torch
import torch.nn.functional as F

from transformers import (
    CLIPConfig,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    DistilBertModel,
)
from transformers.modeling_outputs import ModelOutput
from transformers.utils import cached_file


class SDistilBertTextEncoder(torch.nn.Module):
    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distilbert = DistilBertModel.from_pretrained(model_name).eval()
        self.dense_weights = torch.load(
            cached_file(model_name, "2_Dense/pytorch_model.bin")
        )["linear.weight"]

    def _get_boolean_mask(self, attn_mask):
        return attn_mask > -1.0

    @torch.inference_mode()
    def forward(self, input_ids, attention_mask):
        model_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Mean Pooling - Take attention mask into account for correct averaging
        last_hidden_state = model_output[0]
        input_mask_expanded = (
            self._get_boolean_mask(attention_mask)
            .float()
            .unsqueeze(dim=-1)
            .expand(last_hidden_state.size())
        )
        sum_state = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        pooled_output = F.normalize(sum_state, p=2, dim=1)

        # Calculate embedding
        text_embeds = torch.matmul(pooled_output, self.dense_weights.T)

        return ModelOutput(
            embeds=text_embeds,
            last_hidden_state=last_hidden_state,
        )


class SimpleDistilBert(torch.nn.Module):
    def __init__(self, model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = model.config
        self.embeddings = model.embeddings
        self.transformer = model.transformer

    @torch.inference_mode()
    def forward(self, input_ids, attention_mask):
        embedding_output = self.embeddings(input_ids)
        head_mask = [None] * self.config.num_hidden_layers
        sequence_output = self.transformer(
            x=embedding_output,
            attn_mask=attention_mask,
            head_mask=head_mask,
        )
        return ModelOutput(
            last_hidden_state=sequence_output[0],
        )


class SimpleSDistilBertTextEncoder(SDistilBertTextEncoder):
    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(model_name, *args, **kwargs)
        self.distilbert = SimpleDistilBert(self.distilbert)

    def _get_boolean_mask(self, attn_mask):
        return attn_mask[:, 0, 0, :] > -1.0


def load_sdistilbert_text_encoder(model_name: str):
    return SimpleSDistilBertTextEncoder(model_name).eval()


MODEL_MAPPING = {
    "sentence-transformers/clip-ViT-B-32": "openai/clip-vit-base-patch32",
}


def load_sbert_image_encoder(model_name):
    if model_name not in MODEL_MAPPING:
        raise ValueError(f"Model {model_name} not supported for SBERT image encoder.")

    config = CLIPConfig.from_pretrained(MODEL_MAPPING[model_name])
    model = CLIPVisionModelWithProjection(config.vision_config)

    state_dict = torch.load(cached_file(model_name, "0_CLIPModel/pytorch_model.bin"))
    missing, _ = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", missing)

    return model.eval()


def load_sbert_text_encoder(model_name):
    from clip_script import CLIPTextEncoder

    if model_name not in MODEL_MAPPING:
        raise ValueError(f"Model {model_name} not supported for SBERT image encoder.")

    config = CLIPConfig.from_pretrained(MODEL_MAPPING[model_name])
    model = CLIPTextModelWithProjection(config.text_config)

    state_dict = torch.load(cached_file(model_name, "0_CLIPModel/pytorch_model.bin"))
    missing, _ = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", missing)

    return CLIPTextEncoder(model.eval())
