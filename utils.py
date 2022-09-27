import numpy as np
from torch.utils.data import DataLoader, Dataset


class CriolloDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, x):
        ret = self.data[x]
        return ret
