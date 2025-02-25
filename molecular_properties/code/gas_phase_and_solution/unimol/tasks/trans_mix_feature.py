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
    # RawNumpyDataset,
    FromNumpyDataset,
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
    MaskPointsDataset,
    CMDataset,
    ConformerSampleDataset,
    DistanceDataset,
    EdgeTypeDataset,
    PrependAndAppend2DDataset,
    RightPadDatasetCoord,
    ConformerSamplePairDataset,
    RemoveHydrogenDataset,
    CroppingDataset,
    NormalizeDataset,
)
from unicore.tasks import UnicoreTask, register_task
from collections.abc import Iterable
from unicore import checkpoint_utils
logger = logging.getLogger(__name__)

task_metainfo = {
    'e_abs': {'mean': [426.5648193359375], 'std': [106.31590270996094],'target_name': ['e_abs']},
    'emi': {'mean': [498.8520202636719], 'std': [93.49156188964844],'target_name': ['emi']},
    'fwhm':  {'mean': [74.30380249023438], 'std': [28.53831672668457],'target_name': ['fwhm']},
    'plqy': {'mean': [0.344806432723999], 'std': [0.3082983195781708], 'target_name': ['plqy']},

}

@register_task("trans_mix_feature")
class UniMolTransMixFeatureFinetuneTask(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="downstream data path"
        )
        parser.add_argument("--task-name", type=str, help="downstream task name")
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
        parser.add_argument(
            "--finetune-mol-model",
            default=None,
            type=str,
            help="pretrained molecular model path",
        )
        parser.add_argument(
            "--finetune-pocket-model",
            default=None,
            type=str,
            help="pretrained pocket model path",
        )

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.seed = args.seed
        self.dictionary = dictionary
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        if self.args.task_name in task_metainfo:
            # for regression task, pre-compute mean and std
            self.mean = task_metainfo[self.args.task_name]["mean"]
            self.std = task_metainfo[self.args.task_name]["std"]

    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = Dictionary.load(os.path.join(args.data, args.task_name, args.dict_name))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the data scoure (e.g., train)
        """
        split_path = os.path.join(self.args.data, self.args.task_name, split + ".lmdb")
        dataset = LMDBDataset(split_path)
        tgt_dataset = KeyDataset(dataset, "target")
        smi_dataset = KeyDataset(dataset, "smi")
        tgt_dataset = RawLabelDataset(tgt_dataset)
        feature_dataset = KeyDataset(dataset, "features")

        dataset = ConformerSampleDataset(
                dataset, self.args.seed, "atoms", "coordinates"
            )
        dataset = RemoveHydrogenDataset(dataset, 'atoms', 'coordinates', False, False)
        dataset_a = CroppingDataset(dataset, self.seed, 'atoms', 'coordinates', self.args.max_atoms)

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)
        src_dataset = KeyDataset(dataset_a, 'atoms')
        src_dataset = TokenizeDataset(src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len)
        src_dataset = PrependAndAppend(src_dataset, self.dictionary.bos(), self.dictionary.eos())
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = KeyDataset(dataset_a, 'coordinates')
        coord_dataset = FromNumpyDataset(coord_dataset)
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
                        "mol_feature": 
                            CMDataset(feature_dataset),
                            
                    },
                    "target": {
                        "finetune_target": RawLabelDataset(tgt_dataset), 
                    },
                    "smi_name": RawArrayDataset(smi_dataset),
                },
            )
        if split.startswith('train'):
            nest_dataset = EpochShuffleDataset(nest_dataset, len(nest_dataset), self.args.seed)
        self.datasets[split] = nest_dataset

    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)
        if args.finetune_mol_model is not None:
            print("load pretrain model weight from...", args.finetune_mol_model)
            state = checkpoint_utils.load_checkpoint_to_cpu(
                args.finetune_mol_model,
            )
            model.mol_model.load_state_dict(state["model"], strict=False)
            missing_keys, unexpected_keys = model.mol_model.load_state_dict(
                state["model"], strict=False
            )
            
            # Print feedback after loading weights
            if not missing_keys and not unexpected_keys:
                print("All weights successfully loaded into the model!")
            else:
                print("Model weights loaded with warnings:")
                if missing_keys:
                    print(f"Missing keys: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys: {unexpected_keys}")
        return model

def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze
            
def freeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, True)

def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)

def set_freeze_by_idxs(model, idxs, freeze=True):
    if not isinstance(idxs, Iterable):
        idxs = [idxs]
    num_child = len(list(model.children()))
    idxs = tuple(map(lambda idx: num_child + idx if idx < 0 else idx, idxs))
    for idx, child in enumerate(model.children()):
        if idx not in idxs:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze
            
def freeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, True)

def unfreeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, False)
