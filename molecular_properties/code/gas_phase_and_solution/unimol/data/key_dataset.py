# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache
from unicore.data import BaseWrapperDataset
import torch
import numpy as np


class KeyDataset(BaseWrapperDataset):
    def __init__(self, dataset, key):
        self.dataset = dataset
        self.key = key

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        # return self.dataset[idx][self.key]
        return  self.dataset[idx].get(self.key, 'cc')


class FeatureDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.dataset = dataset

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        cm = self.dataset[idx]
        return torch.from_numpy(cm)
    


class CMDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.dataset = dataset

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        cm = self.dataset[idx]
        if isinstance(cm, (np.int64, int, float)):
            cm = np.array([cm])  
        return torch.from_numpy(cm)