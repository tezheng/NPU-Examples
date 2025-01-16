

from torch.utils.data import DataLoader, Dataset

from olive.data.registry import Registry


@Registry.register_dataloader()
def llmlingua_calibration_reader(dataset, batch_size, data_dir, **kwargs):
    class MobileNetDataset(Dataset):
        def __init__(self, data_dir):
        self.data = np.load(data_dir)

        def __len__(self):
        return len(self.data)

        def __getitem__(self, idx):
        return self.data[idx
  dataset= MobileNetDataset(data_dir)
  return DataLoader(dataset, batch_size=batch_size, shuffle=False)
