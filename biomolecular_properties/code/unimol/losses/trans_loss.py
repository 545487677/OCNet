# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch.nn.functional as F
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss
from sklearn.metrics import roc_auc_score
from collections.abc import Iterable
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings(action='ignore')

@register_loss("trans_loss")
class TransLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        reg_output = model(**sample["net_input"])
        
        loss = self.compute_loss(model, reg_output, sample, reduce=reduce)
        
        sample_size = sample["target"]["all_target"].size(0)

        if not self.training:
            if self.task.mean and self.task.std:
                targets_mean = torch.tensor(self.task.mean, device=reg_output.device)
                targets_std = torch.tensor(self.task.std, device=reg_output.device)
                reg_output = reg_output * targets_std + targets_mean
            logging_output = {
                "loss": loss.data,
                "predict": reg_output.view(-1, self.args.num_classes).data,
                "target": sample["target"]["all_target"]
                .view(-1, self.args.num_classes)
                .data,
                "sample_size": sample_size,
                "num_task": self.args.num_classes,
                "bsz": sample["target"]["all_target"].size(0),
            }
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "bsz": sample["target"]["all_target"].size(0),
            }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        predicts = net_output.view(-1, self.args.num_classes).float()
        targets = (
            sample["target"]["all_target"].view(-1, self.args.num_classes).float()
        )
        if self.task.mean and self.task.std:
            targets_mean = torch.tensor(self.task.mean, device=targets.device)
            targets_std = torch.tensor(self.task.std, device=targets.device)
            targets = (targets - targets_mean) / targets_std
        loss = F.mse_loss(
            predicts,
            targets,
            reduction="sum" if reduce else "none",
        )
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "{}_loss".format(split), loss_sum / sample_size, sample_size, round=5
        )
        if split in ['valid', 'test']:
            predicts = torch.cat([log.get("predict") for log in logging_outputs], dim=0).view(-1).cpu().numpy()
            targets = torch.cat([log.get("target", 0) for log in logging_outputs], dim=0).view(-1).cpu().numpy()
            mae = np.abs(predicts - targets).mean()
            mse = ((predicts - targets) ** 2).mean()

            metrics.log_scalar(f"{split}_mae", mae, sample_size, round=10)
            metrics.log_scalar(f"{split}_mse", mse, sample_size, round=10)


            r2 = r2_score(targets, predicts)
            metrics.log_scalar(f"{split}_total_r2_score", r2, sample_size, round=4)        

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train
