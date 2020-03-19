# coding: UTF-8
"""This module defines an example Torch dataset
Example
-------
$ dataloader = DataLoader(sampleDataset(DATSET_PATH),
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=NUM_WORKERS)
"""

import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class sampleDataset(Dataset):
    """
    Load dataset from two train.csv and test.csv file.
        Attributes
        ----------
        x: np.array
            time series input of shape (time , column)
        m: np.array
            Normalization constant.
        M: np.array
            Normalization constant.
    """

    def __init__(self, dataset_path, target_num=1, window_size=50, \
                 test_ratio=0.1, val_ratio=0.1,**kwargs):
        """Load dataset from csv.
        Parameters
        ---------
        dataset_path: str or Path
            Path to the dataset inputs as csv.
        target_num: str or Path, optional
            chooose target column number
        window_size: int
            time window for data
        test_ratio,val_ratio: float.optional
            ratio of test,val data
        """
        super().__init__(**kwargs)
        self._load_from_csv(dataset_path, target_num, window_size, test_ratio, val_ratio)

    def _load_from_path(self, dataset_path, target_num, \
                        window_size=50, test_ratio=0.1, val_ratio=0.1):
        x = pd.read_csv(dataset_path)

        data = x.values
        table = []
        for i in range(len(data) - window_size):
            table.append(data[i:i + window_size])
        table = np.array(table)

        col_index = np.ones(table.shape[2], dtype=bool)
        col_index[target_num] = False

        self._x = table[:, :, col_index]
        self._y = table[:, :, target_num]

        # Normalize
        self._M = np.max(self._x, axis=(0, 1))
        self._m = np.min(self._x, axis=(0, 1))
        self._x = (self._x - self.m) / (self.M - self.m + np.finfo(float).eps)
        # Convert to float32
        self._x = self._x.astype(np.float32)

        # Normalize
        self.M = np.max(self._y, axis=(0, 1))
        self.m = np.min(self._y, axis=(0, 1))
        self._y = (self._y - self.m) / (self.M - self.m + np.finfo(float).eps)
        # Convert to float32
        self._y = self._y.astype(np.float32)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return (self._x[idx], self._y[idx])

    def __len__(self):
        return self._x.shape[0]
