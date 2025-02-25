# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import lmdb
import os
import numpy as np
import collections
import pickle
from functools import lru_cache
import logging
from . import data_utils

logger = logging.getLogger(__name__)

class LMDBASCIIDataset:
    def __init__(self, db_path):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(
            self.db_path
        )
        env = self.connect_db(self.db_path)
        with env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))

    def connect_db(self, lmdb_path, save_to_self=False):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        if not save_to_self:
            return env
        else:
            self.env = env

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if not hasattr(self, 'env'):
            self.connect_db(self.db_path, save_to_self=True)
        datapoint_pickled = self.env.begin().get(f'{idx}'.encode("ascii"))
        data = pickle.loads(datapoint_pickled)
        return data

class LMDBINTDataset:
    def __init__(self, db_path):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(self.db_path)
        env = self.connect_db(self.db_path)
        with env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))

    def connect_db(self, lmdb_path, save_to_self=False):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        if not save_to_self:
            return env
        else:
            self.env = env

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if not hasattr(self, "env"):
            self.connect_db(self.db_path, save_to_self=True)
        key = self._keys[idx]
        datapoint_pickled = self.env.begin().get(key)
        data = pickle.loads(datapoint_pickled)
        data["id"] = int.from_bytes(key, "big")
        return data

class LMDBDataset:
    def __init__(self, db_path):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(self.db_path)
        env = self.connect_db(self.db_path)
        with env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))

    def connect_db(self, lmdb_path, save_to_self=False):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        if not save_to_self:
            return env
        else:
            self.env = env

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if not hasattr(self, "env"):
            self.connect_db(self.db_path, save_to_self=True)
        datapoint_pickled = self.env.begin().get(f"{idx}".encode("ascii"))
        data = pickle.loads(datapoint_pickled)
        return data

class FoldLMDBDataset:
    def __init__(self, dataset, seed, cur_fold, nfolds=10, split_mode='scaffold', cache_fold_info=None):
        super().__init__()
        self.dataset = dataset
        self.split_mode = split_mode
        if cache_fold_info is None:
            self.keys = []
            self.fold_start = []
            self.fold_end = []

            if split_mode == 'random':
                self.init_random_split(dataset, seed, nfolds)
            elif split_mode == 'scaffold':
                self.init_scaffold_split(dataset, seed, nfolds)
            else:
                raise NotImplementedError
        else:
            # use cache fold info
            self.keys, self.fold_start, self.fold_end = cache_fold_info
        self.cur_fold = cur_fold
        self._len = self.fold_end[cur_fold] - self.fold_start[cur_fold]
        assert len(self.fold_end) == len(self.fold_start) == nfolds

    def init_random_split(self, dataset, seed, nfolds):
        with data_utils.numpy_seed(seed):
            self.keys = np.random.permutation(len(dataset))
        average_size = (len(dataset) + nfolds - 1) // nfolds
        cur_size = 0
        for i in range(nfolds):
            self.fold_start.append(cur_size)
            cur_size = min(cur_size + average_size, len(dataset))
            self.fold_end.append(cur_size)
            
    def init_scaffold_split(self, dataset, seed, nfolds):
        scaffold2idx = collections.defaultdict(list)
        for i in range(len(dataset)):
            _scaffold = dataset[i]['scaffold']
            scaffold2idx[_scaffold].append(i)
        
        scaffold_list = list(scaffold2idx.keys())
        assert len(scaffold_list) > nfolds, "unique scaffolds is smaller than {}".format(nfolds)
        with data_utils.numpy_seed(seed):
            np.random.shuffle(scaffold_list)
                 
        average_size = (len(scaffold_list) + nfolds - 1) // nfolds
        cur_size = 0
        for i in range(nfolds):
            start, end = i*average_size, min((i+1)*average_size, len(scaffold_list))
            scaffold_fold = scaffold_list[start:end]
            fold = []
            for _scaffold in scaffold_fold:
                fold.extend(scaffold2idx[_scaffold])
            with data_utils.numpy_seed(seed):
                np.random.shuffle(fold)

            self.keys.extend(fold)
            self.fold_start.append(cur_size)
            self.fold_end.append(cur_size + len(fold))
            cur_size = cur_size + len(fold)
    
    def get_fold_info(self):
        return self.keys, self.fold_start, self.fold_end

    def __len__(self):
        return self._len

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        global_idx = idx + self.fold_start[self.cur_fold]
        return self.dataset[self.keys[global_idx]]

class StackedLMDBDataset:
    def __init__(self, datasets):
        self._len = 0
        self.datasets = []
        self.idx_to_file = {}
        self.idx_offset = []
        for dataset in datasets:
            self.datasets.append(dataset)
            for i in range(len(dataset)):
                self.idx_to_file[i + self._len] = len(self.datasets) - 1
            self.idx_offset.append(self._len)
            self._len += len(dataset)

    def __len__(self):
        return self._len

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        file_idx = self.idx_to_file[idx]
        sub_idx = idx - self.idx_offset[file_idx]
        return self.datasets[file_idx][sub_idx]