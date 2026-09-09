import os
import torch
import random
import numpy as np
import numpy.fft as fft
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def Power_Normalization(x):
    for i in range(x.shape[0]):
        max_power = (np.power(x[i, 0, :], 2) + np.power(x[i, 1, :], 2)).max()
        x[i] = x[i] / np.power(max_power, 1 / 2)
    return x


def get_data(dataset_path, S, ft, k=None):
    x_train = np.load(os.path.join(dataset_path, 'WiFi-S', f"S{S}", f"x_train_{ft}ft.npy"))
    x_test = np.load(os.path.join(dataset_path, 'WiFi-S', f"S{S}", f"x_test_{ft}ft.npy"))
    y_train = np.load(os.path.join(dataset_path, 'WiFi-S', f"S{S}", f"y_train_{ft}ft.npy"))
    y_test = np.load(os.path.join(dataset_path, 'WiFi-S', f"S{S}", f"y_test_{ft}ft.npy"))

    if k is not None:
        train_valid_index = random.sample(range(len(y_train)), int(k * len(y_train)))
        x_train, x_valid, y_train, y_valid = train_test_split(x_train[train_valid_index], y_train[train_valid_index],
                                                              test_size=0.2, random_state=2025)

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.3, random_state=2025)

    return x_train, x_valid, x_test, y_train, y_valid, y_test


class GetDataSet(Dataset):
    def __init__(self, mode, data, target):
        super(GetDataSet, self).__init__()
        self.mode = mode
        self.data = torch.tensor(data, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.int8)
        self.len = len(self.target)

    def __getitem__(self, item):
        data = self.data[item, :, :]
        target = self.target[item]
        return data, target

    def __len__(self):
        return self.len


def get_dataloader(mode, data, target, batch_size):
    data_set = GetDataSet(mode, data, target)
    data_loader = DataLoader(data_set, batch_size, shuffle=True)
    return data_loader
