""""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from src.salsa.train.envs.datasets import LWEDataset, MLWEiDataset, RLWEDataset
from src.salsa.train.metrics import get_metrics
from src.salsa.train.model.encoder import DigitEncoder, Encoder


DATASET_CLS = {
    'lwe': LWEDataset,
    'rlwe': RLWEDataset,
    'mlwe': RLWEDataset,
    'mlwe-i': MLWEiDataset
}
MODELS_CLS = [DigitEncoder, Encoder]


def get_dataset(params):
    assert params.task in DATASET_CLS.keys(), f"Task {params.task} not supported! available tasks {DATASET_CLS.keys()}"

    dataset = DATASET_CLS[params.task](params)

    return dataset


def get_model(params):
    # Map models to tasks when there is more than one

    model = MODELS_CLS[params.angular_emb](params)
    return model
