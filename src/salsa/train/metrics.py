""""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn.functional as F
import torchmetrics


def new_classification_metrics(vocab_size, prefix):
    task = "multiclass"
    acc1 = torchmetrics.Accuracy(task=task, top_k=1, num_classes=vocab_size)
    acc5 = torchmetrics.Accuracy(task=task, top_k=5, num_classes=vocab_size)
    return {f"{prefix}/acc1": acc1, f"{prefix}/acc5": acc5}


def get_metrics(params):
    train_metrics = {}
    recover_metrics = {}
    # metrics and loss
    if params.angular_emb:
        train_metrics["train/angle"] = AngularDistanceMetric()
        train_metrics["train/norm"] = AngularNormMetric()

        recover_metrics["recover/angle"] = AngularDistanceMetric()
        recover_metrics["recover/loss"] = MSELossMetric()
        recover_metrics["recover/norm"] = AngularNormMetric()

    else:
        train_metrics = new_classification_metrics(params.vocab_size, "train")
        recover_metrics = new_classification_metrics(params.vocab_size, "recover")
        recover_metrics["recover/loss"] = CrossEntropyMetric()

    train_metrics = torchmetrics.MetricCollection(train_metrics).to(params.device)
    recover_metrics = torchmetrics.MetricCollection(recover_metrics).to(params.device)

    train_metrics.reset()
    recover_metrics.reset()
    return train_metrics, recover_metrics


class CrossEntropyMetric(torchmetrics.Metric):
    full_state_update = False

    def __init__(self, ignore_index: int = -100, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.ignore_index = ignore_index
        self.add_state("sum_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_batches", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, outputs, targets) -> None:
        self.sum_loss += F.cross_entropy(
            outputs, targets, ignore_index=self.ignore_index, reduction="mean"
        )
        self.total_batches += 1

    def compute(self):
        assert isinstance(self.total_batches, torch.Tensor)
        assert isinstance(self.sum_loss, torch.Tensor)
        return self.sum_loss / self.total_batches


class MSELossMetric(torchmetrics.Metric):
    full_state_update = False

    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_batches", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, outputs, targets) -> None:
        self.sum_loss += F.mse_loss(outputs, targets, reduction="mean")
        self.total_batches += 1

    def compute(self):
        assert isinstance(self.total_batches, torch.Tensor)
        assert isinstance(self.sum_loss, torch.Tensor)
        return self.sum_loss / self.total_batches


class AngularDistanceMetric(torchmetrics.Metric):
    full_state_update = False

    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum_dist", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_batches", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, outputs, targets) -> None:
        _, cols = outputs.shape
        assert cols == 2, f"outputs should have 2 columns, not {cols}"
        pred_angles = torch.atan(outputs[:, 0] / outputs[:, 1])
        tgt_angles = torch.atan(targets[:, 0] / targets[:, 1])
        diff = torch.abs(pred_angles - tgt_angles)
        self.sum_dist += torch.mean(torch.minimum(diff, torch.pi * 2 - diff))
        self.total_batches += 1

    def compute(self):
        assert isinstance(self.total_batches, torch.Tensor)
        assert isinstance(self.sum_dist, torch.Tensor)
        return self.sum_dist / self.total_batches


class AngularNormMetric(torchmetrics.Metric):
    full_state_update = False

    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum_norm", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_batches", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, outputs, targets) -> None:
        _, cols = outputs.shape
        assert cols == 2, f"outputs should have 2 columns, not {cols}"
        norm = torch.linalg.norm(outputs, ord=2, dim=1).mean()
        self.sum_norm += norm
        self.total_batches += 1

    def compute(self):
        assert isinstance(self.total_batches, torch.Tensor)
        assert isinstance(self.sum_norm, torch.Tensor)
        return self.sum_norm / self.total_batches


class ModularMSEMetric(torchmetrics.Metric):
    full_state_update = False

    def __init__(self, q, tokenizer, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.q = q
        self.tokenizer = tokenizer
        self.add_state("sum_mse", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_batches", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, logits, targets) -> None:
        preds = logits.max(dim=-1)[1]
        pred_val = self.tokenizer.decode_z(preds.reshape(-1, 2))
        target_val = self.tokenizer.decode_z(targets.reshape(-1, 2))

        # This calculates the distance over the ring Z_Q.
        # See https://math.stackexchange.com/questions/1148911/compute-the-distance-between-two-elements-in-a-ring
        diff = torch.abs(pred_val - target_val)
        diff = torch.minimum(diff, self.q - diff)

        self.sum_mse += diff.pow(2).float().mean()
        self.total_batches += 1

    def compute(self):
        assert isinstance(self.total_batches, torch.Tensor)
        assert isinstance(self.sum_mse, torch.Tensor)
        return self.sum_mse / self.total_batches
