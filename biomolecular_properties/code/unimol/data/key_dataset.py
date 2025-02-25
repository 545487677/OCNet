# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache
from unicore.data import BaseWrapperDataset
import numpy as np 
import torch 


class KeyDataset(BaseWrapperDataset):
    def __init__(self, dataset, key):
        self.dataset = dataset
        self.key = key

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        return self.dataset[idx][self.key]

class ToTorchDataset(BaseWrapperDataset):
    def __init__(self, dataset, dtype='float32'):
        super().__init__(dataset)
        self.dtype = dtype

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        d = np.array(self.dataset[idx], dtype=self.dtype)
        return torch.from_numpy(d)
