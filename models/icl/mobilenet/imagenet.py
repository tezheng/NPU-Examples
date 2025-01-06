from pathlib import Path
from logging import getLogger

from datasets import load_dataset
import polars as pl

import numpy as np
from PIL import Image

from torch import from_numpy
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForImageClassification

from olive.data.registry import Registry

logger = getLogger(__name__)


class MobileNetDataset(Dataset):
  def __init__(self, data_dir: str):
    with np.load(Path(data_dir) / 'data.npz') as data:
      self.images = from_numpy(data['images'])
      self.labels = from_numpy(data['labels'])

  def __len__(self):
    return min(len(self.images), len(self.labels))

  def __getitem__(self, idx):
    # data = torch.unsqueeze(self.data[idx], dim=0)
    label = self.labels[idx] if self.labels is not None else -1
    return {'input_image': self.images[idx]}, label


@Registry.register_dataset()
def qnn_evaluation_dataset(data_dir, **kwargs):
  return MobileNetDataset(data_dir)


@Registry.register_post_process()
def qnn_post_process(output):
  return output.argmax(axis=-1)


@Registry.register_dataloader()
def mobilenet_calibration_reader(dataset, batch_size, data_dir, **kwargs):
  dataset = MobileNetDataset(data_dir)
  return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def preprocess_image(image):
  src_img = Image.open(image).convert('RGB')
  transformations = transforms.Compose(
    [
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
  )
  return transformations(src_img).numpy().astype(np.float32)


def download_dataset(data_dir: Path, split: str, size: int):
  data_dir = data_dir / str(split)
  data_dir.mkdir(parents=True, exist_ok=True)

  # Load the dataset in streaming mode to avoid downloading the full dataset
  dataset = load_dataset(
    'imagenet-1k',
    split=split,
    streaming=True,
    trust_remote_code=True,
  )

  model = AutoModelForImageClassification.from_pretrained(
    'google/mobilenet_v2_1.4_224')
  model.eval()

  # Iterate over the dataset to extract a subset of the specified size
  records = []
  images = []
  for i, sample in enumerate(dataset):
    if i >= size:
      break

    # Save the image file and its metadata
    img_path = data_dir / f"img_{i}.jpg"
    sample['image'].save(img_path)
    logger.debug(f"Saved image to {img_path}")

    # Process the image and store it as a tensor
    image = preprocess_image(img_path)
    images.append(image)

    # Run the model on the image to get the predicted label
    input = dict(pixel_values=from_numpy(image).unsqueeze(dim=0))
    outputs = model(**input)
    label = outputs['logits'][0, :].argmax(axis=-1).item()

    # Save the image and label as a record
    records.append({'filename': str(img_path), 'label': label})

  # Save the images and labels as a NumPy array
  np.savez_compressed(
      data_dir / 'data.npz',
      allow_pickle=False,
      **dict(
          images=np.array(images),
          labels=np.array([record['label'] for record in records])
      ),
  )

  # Save the ground truth labels as CSV and Parquet files
  df = pl.DataFrame(records, schema=['filename', 'label'])
  df.write_csv(data_dir / 'ground_truth.csv')
  df.write_parquet(data_dir / 'ground_truth.parquet')


if __name__ == '__mp_main__':
  import sys
  sys.exit(0)
  # from multiprocessing import freeze_support
  # freeze_support()

data_dir = Path('./data/imagenet_subset').resolve()
data_dir.mkdir(parents=True, exist_ok=True)

completion_marker = data_dir / '.complete'
if completion_marker.exists():
  logger.debug('Dataset already downloaded.')
else:
  download_dataset(data_dir, 'train', size=64)
  download_dataset(data_dir, 'test', size=128)
  completion_marker.touch()
