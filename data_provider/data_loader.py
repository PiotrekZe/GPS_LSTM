from torch.utils.data import Dataset
import torch
import numpy as np


class Dataset_GPS(Dataset):
    def __init__(self, data, labels, normalize=True):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.mean = self.data.mean()
        self.std = self.data.std()
        self.normalize = normalize

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        # data = data.dtype(torch.float32)
        # label = label.dtype(torch.float32)
        if self.normalize:
            data = (data - self.mean) / self.std
            label = (label - self.mean) / self.std
        return data, label
