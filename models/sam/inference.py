from pathlib import Path
from PIL import Image
from transformers import SamProcessor


def load_processor(model_name) -> SamProcessor:
    return SamProcessor.from_pretrained(model_name)  # type: ignore


def encode_image(image_path, model_name, npu_model):
    processor = load_processor(model_name)
    image = Image.open(image_path)
    pixel_values = processor(images=image, return_tensors="np").pixel_values

    inputs = {
        "pixel_values": pixel_values,
    }
    outputs = npu_model.run(tensors=inputs)

    return outputs[0]


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image-file", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default="facebook/sam-vit-base")
    parser.add_argument(
        "--model-path",
        type=Path,
        default="outputs/meta/sam-vit-base/vision/model/model.onnx",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model_name = args.model_name

    image_path = args.image_file.resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    model_path = args.model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    from utils.npumodel import QNPUModule

    npu_model = QNPUModule(model_path)

    output = encode_image(image_path, model_name, npu_model)
    print(output)


if __name__ == "__main__":
    main()
