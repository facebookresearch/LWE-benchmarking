""""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from logging import getLogger
import numpy as np
from dataclasses import dataclass
import argparse
import pickle
import os

from reduction import make_n_reduced_samples

logger = getLogger()


@dataclass
class Data:
    """Class for keeping track of an item in inventory."""

    params: argparse.Namespace
    RA: np.ndarray
    RB: np.ndarray
    origA: np.ndarray
    origB: np.ndarray

    @classmethod
    def from_files(cls, path, hamming_weight=None, seed=0):
        path = path.rstrip("/")
        prefix = os.path.dirname(path)
        secret_prefix = os.path.basename(path)
        params = pickle.load(open(f"{prefix}/{secret_prefix}/params.pkl", "rb"))
        if isinstance(params, dict):
            params = argparse.Namespace(**params)

        hamming_weight = hamming_weight or params.min_hamming
        params.hamming = hamming_weight
        try:
            Bs = np.load(f"{prefix}/{secret_prefix}/b_{hamming_weight}_{seed}.npy")
        except:
            Bs = np.load(
                f"{prefix}/{secret_prefix}/train_b_{hamming_weight}_{seed}.npy"
            )

        try:
            As = np.load(f"{prefix}/reduced_A.npy")
        except:
            As = np.load(f"{prefix}/train_A.npy")

        origAs = np.load(f"{prefix}/orig_A.npy")
        origBs = np.load(f"{prefix}/{secret_prefix}/orig_b_{hamming_weight}_{seed}.npy")
        return cls(
            params=params,
            RA=As,
            RB=Bs,
            origA=origAs,
            origB=origBs,
        )

    @staticmethod
    def create_new_A(  # this is meant for experimentation
        secret_dim,
        q,
        n_sniffs,
        savepath="./data",
        lll_penalty=10,
        sample_target=100000,
    ):
        save_prefix = f"{savepath}/{secret_dim}_{q}_{n_sniffs}_{lll_penalty}/"
        if not os.path.exists(save_prefix + "origA.npy"):
            origA = np.random.randint(0, q, size=(n_sniffs, secret_dim)) - q // 2
            Rs, subsets = make_n_reduced_samples(sample_target)
            os.makedirs(save_prefix, exist_ok=True)
            np.save(save_prefix + "origA.npy", origA)
            np.save(save_prefix + "Rs_default.npy", Rs)
            np.save(save_prefix + "subsets_default.npy", subsets)
        else:
            origA = np.load(save_prefix + "origA.npy")
            Rs = np.load(save_prefix + "Rs_default.npy")
            subsets = np.load(save_prefix + "subsets_default.npy")

        return origA, Rs, subsets

    @classmethod
    def create_data_from_A(
        cls,
        origA,
        Rs,
        subsets,
        hamming_weight,
        Q,
        noise_variance=3.2,
        n_brute_force=None,
        hamming_weight_in_brute_force=None,
    ):
        secret_dim = origA.shape[1]
        secret = cls._make_secret_with(
            secret_dim, hamming_weight, n_brute_force, hamming_weight_in_brute_force
        )
        origB, _ = cls._make_B_from_A(origA, noise_variance, secret, Q)
        RAs, RBs = cls._make_RAs_RBs(origA, origB, Rs, subsets, Q)

        params = argparse.Namespace(
            Q=Q,
            n_sniffs=origA.shape[0],
            secret_dim=secret_dim,
            hamming_weight=hamming_weight,
            sigma=noise_variance,
            n_brute_force=n_brute_force,
            hamming_in_brute_force=hamming_weight_in_brute_force,
        )

        return cls(
            params=params,
            RA=RAs,
            RB=RBs,
            origA=origA,
            origB=origB,
        )

    @staticmethod
    def _make_secret_with(
        secret_dim,
        hamming_weight,
        n_brute_force=None,
        hamming_weight_in_brute_force=None,
    ):
        if hamming_weight_in_brute_force is None:
            return np.random.permutation(
                [1] * hamming_weight + [0] * (secret_dim - hamming_weight)
            )
        else:
            secret_part1 = np.random.permutation(
                [1] * hamming_weight_in_brute_force
                + [0] * (n_brute_force - hamming_weight_in_brute_force)
            )
            secret_part2 = np.random.permutation(
                [1] * (hamming_weight - hamming_weight_in_brute_force)
                + [0]
                * (
                    secret_dim
                    - n_brute_force
                    - hamming_weight
                    + hamming_weight_in_brute_force
                )
            )
            return np.concatenate([secret_part1, secret_part2])

    @staticmethod
    def _make_B_from_A(origA: np.array, noise_var, secret, Q):
        noise = np.random.normal(0, noise_var, size=origA.shape[0]).round(0)
        origB = (origA @ secret + noise) % Q
        return origB, noise

    @staticmethod
    def _make_RAs_RBs(origA, origB, Rs, subsets, Q):
        RAs = np.vstack([R_i @ origA[subset] for R_i, subset in zip(Rs, subsets)])
        RBs = np.hstack([R_i @ origB[subset] for R_i, subset in zip(Rs, subsets)])
        sel = (RAs != 0).any(1)
        logger.info(sel.mean())
        RAs = (RAs[sel] + Q // 2) % Q - Q // 2
        RBs = RBs[sel]
        return RAs, RBs


@dataclass
class MLWEData(Data):
    secret_window: int

    @classmethod
    def shift_negate(cls, mat, k=1):
        """Shift to the left and negate the wrapped element"""
        return np.concatenate((mat[..., k:], -mat[..., :k]), axis=-1)

    @classmethod
    def from_files(
        cls, path, hamming_weight=None, seed=0, mlwe_k=1, secret_window=0, bf_dim=0
    ):
        path = path.rstrip("/")
        prefix = os.path.dirname(path)
        secret_prefix = os.path.basename(path)
        params = pickle.load(open(f"{prefix}/{secret_prefix}/params.pkl", "rb"))
        if isinstance(params, dict):
            params = argparse.Namespace(**params)

        Q = params.Q
        N = params.N
        n = N // mlwe_k
        assert N % mlwe_k == 0, f"Invalid value for k = {mlwe_k}"
        hamming_weight = hamming_weight or params.min_hamming
        params.hamming = hamming_weight

        try:
            Bs = np.load(
                f"{prefix}/{secret_prefix}/train_b_{hamming_weight}_{seed}.npy"
            )
        except:
            Bs = np.load(f"{prefix}/{secret_prefix}/b_{hamming_weight}_{seed}.npy")

        assert Bs.shape[1] == params.N // mlwe_k
        A_shift = (-bf_dim // mlwe_k - secret_window) % n
        Bs = Bs[:, ~A_shift]

        try:
            As = np.load(f"{prefix}/train_A.npy")
        except:
            As = np.load(f"{prefix}/reduced_A.npy")

        assert As.shape[0] == len(Bs)

        def get_shift(A, mlwe_k, n, A_shift):
            A = np.flip(A.reshape((len(A), mlwe_k, n)), axis=2)

            A = cls.shift_negate(A, k=A_shift) % Q
            return A.reshape((len(A), mlwe_k * n))
        
        As = get_shift(As, mlwe_k, n, A_shift)
        
        origAs = np.load(f"{prefix}/orig_A.npy")
        origBs = np.load(f"{prefix}/{secret_prefix}/orig_b_{hamming_weight}_{seed}.npy")
        origAs = get_shift(origAs, mlwe_k, n, A_shift)
        origBs = origBs[:, ~A_shift]
        return cls(
            params=params,
            RA=As,
            RB=Bs,
            origA=origAs,
            origB=origBs,
            secret_window=secret_window,
        )
