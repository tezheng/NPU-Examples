from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor
from utils.npumodel import QNPUModule


class QNPUCLIPModel:
    def __init__(
        self,
        model_name: str,
        text_model_path: Path,
        vision_model_path: Path,
    ) -> None:
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.text_model = QNPUModule(text_model_path)
        self.vision_model = QNPUModule(vision_model_path)

    def get_image_embedding(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        output = self.vision_model.run(inputs)
        return output["embeds"]

    def get_text_embedding(self, text):
        inputs = self.processor(
            text=text,
            padding="max_length",
            max_length=self.text_model.sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        output = self.text_model.run(
            {
                "input_ids": inputs["input_ids"].int(),
                "attention_mask": self._create_4d_mask(
                    inputs["attention_mask"],
                    inputs["input_ids"].shape,
                ),
            }
        )
        return output["embeds"]

    def _create_4d_mask(self, mask, input_shape, masked_value=-50.0):
        # (batch_size, num_heads, seq_len, head_dim)
        batch_sz, seq_len = input_shape
        expanded_mask = mask[:, None, None, :].expand(batch_sz, 1, seq_len, seq_len)
        inverted_mask = 1.0 - expanded_mask.float()
        return inverted_mask.masked_fill(inverted_mask.bool(), masked_value)


def calculate_score(text_emb, image_emb):
    import torch

    image_emb /= torch.norm(image_emb, dim=-1, keepdim=True)
    text_emb /= torch.norm(text_emb, dim=-1, keepdim=True)
    return torch.softmax(torch.matmul(text_emb, image_emb.T) * 100.0, dim=0)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image-file", type=Path, required=True)
    parser.add_argument("--caption", type=str, required=True)
    parser.add_argument(
        "--model-name",
        type=str,
        default="openai/clip-vit-base-patch16",
    )
    parser.add_argument(
        "--text-model-path",
        type=Path,
        default="outputs/openai/clip/text/model/model.onnx_ctx.onnx",
    )
    parser.add_argument(
        "--vision-model-path",
        type=Path,
        default="outputs/openai/clip/vision/model/model.onnx_ctx.onnx",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model_name = args.model_name

    text_model_path = args.text_model_path.resolve()
    if not text_model_path.exists():
        raise FileNotFoundError(f"Text model file not found: {text_model_path}")

    vision_model_path = args.vision_model_path.resolve()
    if not vision_model_path.exists():
        raise FileNotFoundError(f"Vision model file not found: {vision_model_path}")

    image_path = args.image_file.resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    model = QNPUCLIPModel(
        model_name,
        text_model_path=text_model_path,
        vision_model_path=vision_model_path,
    )

    image_emb = model.get_image_embedding(Image.open(image_path))
    text_emb = model.get_text_embedding([args.caption, "a photo of a thing"])
    score = calculate_score(text_emb, image_emb)
    print(f"Score: {round(score[0].item(), 2)}")


if __name__ == "__main__":
    main()
