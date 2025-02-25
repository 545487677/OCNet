# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np
from unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    LMDBDataset,
    RawLabelDataset,
    RawArrayDataset,
    RawNumpyDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    EpochShuffleDataset,
    TokenizeDataset,
    RightPadDataset2D,
    RawLabelDataset,
)
from unimol.data import (
    KeyDataset,
    LMDBASCIIDataset,
    LMDBINTDataset,
    MaskPointsDataset,
    DistanceDataset,
    EdgeTypeDataset,
    PrependAndAppend2DDataset,
    RightPadDatasetCoord,
    RemoveHydrogenDataset,
    CroppingDataset,
    NormalizeDataset,
)
from unicore.tasks import UnicoreTask, register_task

logger = logging.getLogger(__name__)

TASK_ATTR_REGISTER = {
    "H3": [4.01,1.42]
}

@register_task("unimol_finetune")
class UniMofAbsorbTask(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="downstream data path"
        )
        parser.add_argument(
            "--classification-head-name",
            default="finetune_head",
            help="finetune downstream task name"
        )
        parser.add_argument(
            "--num-classes",
            default=1,
            type=int,
            help="finetune downstream task classes numbers"
        )
        parser.add_argument(
            "--max-atoms",
            type=int,
            default=1024,
            help="selected maximum number of atoms in a molecule",
        )
        parser.add_argument(
            "--dict-name",
            default="dict.txt",
            help="dictionary file",
        )
        parser.add_argument(
            "--remove-hydrogen",
            action="store_true",
            help="remove hydrogen atoms",
        )
    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        ### mean std for normalization
        self.mean, self.std = TASK_ATTR_REGISTER['H3']

    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = Dictionary.load(os.path.join(args.data, args.dict_name))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the data scoure (e.g., train)
        """
        split_path = os.path.join(self.args.data, split + ".lmdb")
        dataset = LMDBINTDataset(split_path)
        tgt_dataset = KeyDataset(dataset, "target")
        tgt_dataset = RawLabelDataset(tgt_dataset)
        id_dataset = KeyDataset(dataset, "id")
        if self.args.remove_hydrogen:
            dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates")
        dataset = CroppingDataset(dataset, self.seed, "atoms", "coordinates", self.args.max_atoms)
        dataset = NormalizeDataset(dataset, "coordinates")
        src_dataset = KeyDataset(dataset, "atoms")
        src_dataset = TokenizeDataset(src_dataset, self.dictionary, max_seq_len=self.args.max_atoms+2)
        coord_dataset = KeyDataset(dataset, "coordinates")
        coord_dataset = RawNumpyDataset(coord_dataset)

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        src_dataset = PrependAndAppend(src_dataset, self.dictionary.bos(), self.dictionary.eos())
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)

        nest_dataset = NestedDictionaryDataset(
                {
                    "net_input": {
                        "src_tokens": RightPadDataset(
                            src_dataset,
                            pad_idx=self.dictionary.pad(),
                        ),
                        "src_coord": RightPadDatasetCoord(
                            coord_dataset,
                            pad_idx=0,
                        ),
                        "src_distance": RightPadDataset2D(
                            distance_dataset,
                            pad_idx=0,
                        ),
                        "src_edge_type": RightPadDataset2D(
                            edge_type,
                            pad_idx=0,
                        ),
                    },
                    "target": {
                        "finetune_target": tgt_dataset,
                    },
                    "ID": RawArrayDataset(id_dataset),
                },
            )
        if split in ["train", "train.small"]:
            nest_dataset = EpochShuffleDataset(nest_dataset, len(nest_dataset), self.args.seed)
        self.datasets[split] = nest_dataset

    def build_model(self, args):
        from unicore import models
        model = models.build_model(args, self)
        model.register_classification_head(
            self.args.classification_head_name,
            num_classes=self.args.num_classes,
        )
        return model
