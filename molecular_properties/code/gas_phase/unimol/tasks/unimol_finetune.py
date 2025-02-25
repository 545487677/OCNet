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
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    SortDataset,
    TokenizeDataset,
    RightPadDataset2D,
    RawLabelDataset,
    RawArrayDataset,
    FromNumpyDataset,
)
from unimol.data import (
    KeyDataset,
    ConformerSampleDataset,
    DistanceDataset,
    EdgeTypeDataset,
    RemoveHydrogenDataset,
    AtomTypeDataset,
    NormalizeDataset,
    CroppingDataset,
    RightPadDatasetCoord,
    data_utils,
)

from unimol.data.tta_dataset import TTADataset
from unicore.tasks import UnicoreTask, register_task
from unicore import checkpoint_utils

logger = logging.getLogger(__name__)

task_metainfo = {
    'e_abs':  {'mean': [426.404052734375], 'std': [106.55741882324219], 'target_name': ['e_abs']},
    'emi': {'mean': [498.9420471191406], 'std': [93.44234466552734], 'target_name': ['emi']},
    's0s1': {'mean': [3.5718233585357666], 'std': [0.64279705286026], 'target_name': ['s0s1']},
    'gap': {'mean': [3.187112808227539], 'std': [0.39865079522132874], 'target_name': ['gap']},
    'plqy': {'mean': [0.3437875509262085], 'std': [0.30846232175827026], 'target_name': ['plqy']},
    'er': {'mean': [0.48549774289131165], 'std': [0.17436909675598145], 'target_name': ['er']},
    'hr': {'mean': [0.4215644896030426], 'std': [0.1842222511768341], 'target_name': ['hr']},
    'pretrain': {'mean': [0.0043, 0.0048], 'std': [0.0086, 0.0097], 'target_name': ['homo' 'lumo']},
    '500w_200_pretrain': {'mean': [0.0057, 0.0098], 'std': [0.0062, 0.0108], 'target_name': ['homo' 'lumo']},
    'dimer_vh': {'mean': [16.7305], 'std': [24.44301], 'target_name': ['dimer_vh']},
    'dimer_vl': {'mean': [19.5309], 'std': [28.2036], 'target_name': ['dimer_vl']},
    'dimer_vh_delta': {'mean': [11.0573], 'std': [17.5165], 'target_name': ['dimer_vh']},
    'dimer_vl_delta': {'mean': [13.2752], 'std': [20.0366], 'target_name': ['dimer_vh']},
    'own_data_feature0_vh': {'mean': [16.6450], 'std': [24.1262], 'target_name': ['own_data_feature0_vh']},
    'own_data_feature0_vh_delta': {'mean': [10.9817], 'std': [17.1642], 'target_name': ['own_data_feature0_vh_delta']},
    'own_data_feature1_vh': {'mean': [16.6450], 'std': [24.1262], 'target_name': ['own_data_feature1_vh']},
    # single tower
    'deep4chem_abs_single_tower': {'mean': [426.5648193359375], 'std': [106.31590270996094], 'target_name': ['deep4chem_abs_single_tower']},
    'deep4chem_emi_single_tower': {'mean': [498.8520202636719], 'std': [93.49156188964844], 'target_name': ['deep4chem_emi_single_tower']},
    'deep4chem_fwhm_single_tower': {'mean': [74.30380249023438], 'std': [28.53831672668457], 'target_name': ['deep4chem_fwhm_single_tower']},
    'deep4chem_plqy_single_tower': {'mean': [0.344806432723999], 'std': [0.3082983195781708], 'target_name': ['deep4chem_plqy_single_tower']},
    'finetune_cz_large': {'mean': [1.2259587049484253, 2.0359466075897217, 0.34831202030181885, 3.7154295444488525, 0.6957359313964844], 'std': [0.6325188279151917, 1.0334678888320923, 0.44382670521736145, 0.9217318892478943, 0.372785359621048], 'target_name': ['gap', 'e_abs', 'edme', 'e_abs_max', 'edme_max']},
}


@register_task("mol_finetune")
class UniMolFinetuneTask(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", help="downstream data path")
        parser.add_argument("--task-name", type=str, help="downstream task name")
        parser.add_argument(
            "--classification-head-name",
            default="classification",
            help="finetune downstream task name",
        )
        parser.add_argument(
            "--num-classes",
            default=1,
            type=int,
            help="finetune downstream task classes numbers",
        )
        parser.add_argument("--no-shuffle", action="store_true", help="shuffle data")
        parser.add_argument(
            "--conf-size",
            default=10,
            type=int,
            help="number of conformers generated with each molecule",
        )
        parser.add_argument(
            "--remove-hydrogen",
            action="store_true",
            help="remove hydrogen atoms",
        )
        parser.add_argument(
            "--remove-polar-hydrogen",
            action="store_true",
            help="remove polar hydrogen atoms",
        )
        parser.add_argument(
            "--max-atoms",
            type=int,
            default=256,
            help="selected maximum number of atoms in a molecule",
        )
        parser.add_argument(
            "--dict-name",
            default="dict.txt",
            help="dictionary file",
        )
        # parser.add_argument(
        #     "--finetune-from-model",
        #     default="",
        #     help="pretrain weight",
        # )
        parser.add_argument(
            "--only-polar",
            default=1,
            type=int,
            help="1: only reserve polar hydrogen; 0: no hydrogen; -1: all hydrogen ",
        )
        parser.add_argument(
            "--consistent-loss",
            default=0.0,
            type=float,
            help="add consistent loss",
        )
        parser.add_argument(
            "--weight-constant",
            default=20.0,
            type=float,
            help="add consistent loss",
        )
        parser.add_argument(
            "--finetune-encoder-model",
            type=str,
            default=None, 
            help="pretrain encoder model path",
        ),

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        if self.args.only_polar > 0:
            self.args.remove_polar_hydrogen = True
        elif self.args.only_polar < 0:
            self.args.remove_polar_hydrogen = False
        else:
            self.args.remove_hydrogen = True
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
        if split == "train":
            tgt_dataset = KeyDataset(dataset, "target")
            smi_dataset = KeyDataset(dataset, "smi")
            sample_dataset = ConformerSampleDataset(
                dataset, self.args.seed, "atoms", "coordinates"
            )
            dataset = AtomTypeDataset(dataset, sample_dataset)
        else:
            dataset = TTADataset(
                dataset, self.args.seed, "atoms", "coordinates", self.args.conf_size
            )
            dataset = AtomTypeDataset(dataset, dataset)
            tgt_dataset = KeyDataset(dataset, "target")
            smi_dataset = KeyDataset(dataset, "smi")

        dataset = RemoveHydrogenDataset(
            dataset,
            "atoms",
            "coordinates",
            self.args.remove_hydrogen,
            self.args.remove_polar_hydrogen,
        )
        dataset = CroppingDataset(
            dataset, self.seed, "atoms", "coordinates", self.args.max_atoms
        )
        src_dataset = KeyDataset(dataset, "atoms")
        ori_atoms_dataset =  KeyDataset(dataset, "atoms")
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(dataset, "coordinates")
        ori_coord_dataset = KeyDataset(dataset, "coordinates")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = DistanceDataset(coord_dataset)

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
                    "finetune_target": RawLabelDataset(tgt_dataset),
                },
                "smi_name": RawArrayDataset(smi_dataset),
            },
        )
        if not self.args.no_shuffle and split == "train":
            with data_utils.numpy_seed(self.args.seed):
                shuffle = np.random.permutation(len(src_dataset))

            self.datasets[split] = SortDataset(
                nest_dataset,
                sort_order=[shuffle],
            )
        else:
            self.datasets[split] = nest_dataset
    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)
        model.register_classification_head(
            self.args.classification_head_name,
            num_classes=self.args.num_classes,
        )
        if args.finetune_from_model is not None:
                print("load pretrain model weight from...", args.finetune_from_model)
                state = checkpoint_utils.load_checkpoint_to_cpu(
                    args.finetune_from_model,
                )
                model.load_state_dict(state["model"], strict=False)
        return model