""""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np
from logging import getLogger

from src.salsa.train.envs.lattice import AngularEncoder, DigitEncoder

logger = getLogger()


ENCODER_CLS = [DigitEncoder, AngularEncoder]


class LWEDataset(Dataset):
    def __init__(self, params):
        self.params = params
        self.io_encoder = ENCODER_CLS[params.angular_emb](params)
        self.test_dataset = self.load_A_b(params, "test", params.distinguisher_size)
        self.orig_dataset = self.load_A_b(params, "orig")

        if params.recover_only:
            return

        A, b = self.load_A_b(params, "train", params.max_samples)

        _, N = A.shape
        assert N == params.N, f"expected {params.N}; A had {N}"


        self.A = A
        self.b = b

    def __getitem__(self, index):
        return self.A[index], self.b[index]

    def build_train_dataloader(
        self,
    ):
        sampler = None

        if self.params.multi_gpu:
            sampler = torch.utils.data.distributed.DistributedSampler(
                self,
                drop_last=False,
                shuffle=False,  # Dataset will shuffle it
                num_replicas=self.params.world_size,
                rank=self.params.global_rank,
            )

        return DataLoader(
            self,
            batch_size=self.params.train_batch_size,
            sampler=sampler,
            num_workers=self.params.workers,
            drop_last=False,
            shuffle=self.params.shuffle,
            pin_memory=False,
            persistent_workers=(self.params.workers > 0),
            collate_fn=self.collate_fn,
        )

    @classmethod
    def load_A_b(cls, params, split, max_samples=None):
        assert split in ("train", "test", "orig"), split

        h, s = params.hamming, params.secret_seed

        A_path = os.path.join(os.path.dirname(params.data_path), f"{split}_A.npy")
        b_path = os.path.join(params.data_path, f"{split}_b_{h}_{s}.npy")

        A = np.load(A_path, mmap_mode="c")
        b = np.load(b_path, mmap_mode="c")

        A, b = cls.transform(A, b, params)

        assert len(A) > 0 and len(b) > 0, "Empty dataset!"
        if len(A) != len(b):
            logger.warning(f"A has {len(A)} elements but b has {len(b)}! dataset might be corrupt")

        n_samples = min(len(A), len(b))
        if max_samples:
            n_samples = min(max_samples, n_samples)

        A = A[:n_samples]
        b = b[:n_samples]

        cls.check_data_quality(params, A, b)

        logger.info("Loaded A and b. [split: %s]", split)

        assert len(A) > 0
        assert len(A) == len(b), f"A has {len(A)} elements but b has {len(b)}!"

        logger.info("Loaded data. [root: %s, count: %d]", params.data_path, len(A))
        return A, b

    @classmethod
    def check_data_quality(cls, params, A, b):
        def centered(arr):
            arr = torch.clone(arr)
            arr[arr > params.Q // 2] -= params.Q
            return arr

        # Gut check that data is loaded correctly.
        good_err_std = (
            torch.std(
                centered(
                    (A[0:5000] @ params.secret - torch.squeeze(b[0:5000])) % params.Q
                ).type(torch.FloatTensor)
            )
            / params.Q
        )
        idx = torch.randperm(params.secret.shape[0])
        logger.info("Good error std: %f", good_err_std)
        bad_err_std = (
            torch.std(
                centered(
                    (A[0:5000] @ params.secret[idx] - torch.squeeze(b[0:5000]))
                    % params.Q
                ).type(torch.FloatTensor)
            )
            / params.Q
        )

        logger.info("Bad err std: %f", bad_err_std)

    @classmethod
    def transform(cls, A, b, params):
        A = torch.from_numpy(A.astype(np.int64))
        b = torch.from_numpy(b.astype(np.int64)).unsqueeze(-1)
        return A, b

    def collate_fn(self, elements):
        """
        Collate samples into a batch. Because there are multiple y values for a single x
        value, we split it up into many elements. So a single batch is actually
            batch_size (cli arg) * params.n_unique_secrets
        """
        A, b = zip(*elements)

        A = torch.stack(A)
        b = torch.stack(b)

        A = self.io_encoder(A)
        b = self.io_encoder(b)
        return A, b

    def __len__(self):
        return len(self.A)

    def init_rng(self, seed):
        # Make random number generator
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rng = np.random.RandomState([worker_id, seed])
        logger.info(
            f"Initialized rng. [worker: %d, seed: %s].", worker_id, [worker_id, seed]
        )
        return rng


class RLWEDataset(LWEDataset):
    def __init__(self, params):
        super().__init__(params)

        assert params.rlwe > 0, "rlwe (k) argument should be an int > 0"
        self.k = params.rlwe
        self.n = params.N // self.k

    @classmethod
    def transform(cls, A, b, params):
        """Will 'decompress' RLWE a,b data by column swapping/negating RA matrices to reconstruct original circulant matrices."""

        k = params.rlwe
        n = params.N // k
        Q = params.Q

        new_mat = np.flip(A.reshape((len(A), k, n)), axis=2)
        circulated_mat = np.zeros((n, len(A), k, n), dtype=np.int32)
        for i in range(n):
            circulated_mat[~i] = new_mat
            new_mat = cls.shift_negate(new_mat)

        circulated_mat = circulated_mat % Q

        if params.stacked_circulants:
            circulated_mat = circulated_mat.swapaxes(0, 1)
        else:
            b = b.T

        A = circulated_mat.reshape((-1, k * n))
        b = b.flatten()

        b = b[: len(A)]

        return super().transform(A, b, params)

    @classmethod
    def shift_negate(cls, tnsor, shift=1):
        """Shift to the left and negate the wrapped element"""
        return np.concatenate((tnsor[..., shift:], -tnsor[..., :shift]), axis=-1)

class MLWEiDataset(RLWEDataset):
    def __init__(self, params):
        super().__init__(params)

        assert params.cruel_bits > 0, "Must specify the number of cruel bits!"

    @classmethod
    def transform(cls, A, b, params):
        """Will 'decompress' RLWE a,b data by column swapping/negating RA matrices to reconstruct original circulant matrices."""

        k = params.rlwe
        n = params.N // k
        Q = params.Q
        nu = params.cruel_bits

        assert params.cruel_bits > 0, f"# Cruel bits nu={nu} not provided"
        minhi, argminhi = cls.compute_minhi_mlwe(params.secret, n, k, nu)

        logger.info(f"min h(i, {nu}) = {minhi}, argmin = {argminhi}")
        A = np.flip(A.reshape((len(A), k, n)), axis=2)

        shift  = (n-nu//k+n-argminhi)%n
        
        A = cls.shift_negate(A, shift) % Q
        
        A = A.reshape((len(A), k*n))
        b = b[:,~shift]
        return super(RLWEDataset, cls).transform(A, b, params)

    @classmethod
    def compute_minhi_mlwe(cls, s, n, k, cruel_bits, step=1):
        assert k*n == len(s)
        s = (s!=0).astype(int)
        s = s.reshape((k,n)).T
        u = cruel_bits//k
        hi = s[:u].sum(axis=0).sum()
        minhi = hi
        argmin = 0

        for i in range(0, n, step):
            hi -= s[i].sum()
            hi += s[(i+u)%n].sum()
            if hi < minhi:
                minhi = hi
                argmin = i+1
        return minhi, argmin


class VRLWEDataset(LWEDataset):
    @classmethod
    def check_data_quality(cls, params, A, b):
        return True
