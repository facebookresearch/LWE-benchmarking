""""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from logging import getLogger
import os
import numpy as np
from subprocess import Popen, PIPE
import sys

sys.path.append("./")
from src.generate.lllbkz import encode_intmat, decode_intmat

logger = getLogger()


def reduce_with_BKZ(mat):
    from fpylll import BKZ, GSO, IntegerMatrix
    from fpylll.algorithms.bkz2 import BKZReduction as BKZ2

    mat = mat.copy()
    fplll_Ap = IntegerMatrix.from_matrix(mat.tolist())
    M = GSO.Mat(fplll_Ap, update=True)
    bkz_params = BKZ.Param(block_size=2)
    BKZ_Obj = BKZ2(M)
    BKZ_Obj(bkz_params)
    return fplll_Ap.to_matrix(mat)


def reduce_with_flatter(Ap, alpha=0.025):
    from fpylll import BKZ, GSO, IntegerMatrix

    """
    Runs a single loop of flatter.
    """
    fplll_Ap = IntegerMatrix.from_matrix(Ap.tolist())
    fplll_Ap_encoded = encode_intmat(fplll_Ap)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    try:
        p = Popen(["flatter", "-alpha", str(alpha)], stdin=PIPE, stdout=PIPE, env=env)
    except Exception as e:
        logger.error(f"flatter failed with error {e}")
    out, _ = p.communicate(input=fplll_Ap_encoded)  # output from the flatter run.
    Ap = decode_intmat(out)
    return Ap


def setup_qary_matrix(mat: np.ndarray, q: np.int64, omega: int):
    # ours:
    # [ 0_nm qI_n ]
    # [ wI_m A_mn ]
    m = mat.shape[0]
    n = mat.shape[1]
    qary = np.zeros((m + n, m + n), dtype=mat.dtype)
    qary[n:, :m] = omega * np.eye(m)
    qary[n:, m:] = mat
    qary[:n, m:] = q * np.eye(n)
    return qary


def get_R_from_qary(qary, omega, secret_dim):
    R = qary[:, :-secret_dim].copy()
    return R // omega


def get_RA_from_qary(qary, secret_dim):
    return qary[:, -secret_dim:].copy()


def run_one_reduction(
    origA: np.array,
    seed,
    Q,
    lll_penalty,
    reduction_chunk_size=None,
    reduction_fn=reduce_with_flatter,
):
    SECRET_DIM = origA.shape[1]
    if reduction_chunk_size is None:
        reduction_chunk_size = SECRET_DIM
    np.random.seed(seed)
    subset = np.random.permutation(len(origA))[:reduction_chunk_size]
    A_i = origA[subset]
    Ap_i_qary = setup_qary_matrix(A_i, Q, lll_penalty)
    Ap_i_qary_reduced = reduction_fn(Ap_i_qary)
    R_i = get_R_from_qary(Ap_i_qary_reduced, lll_penalty, SECRET_DIM)
    RA_i = get_RA_from_qary(Ap_i_qary_reduced, SECRET_DIM)
    assert (RA_i % Q == (R_i @ A_i) % Q).all()
    return R_i, subset


def make_n_reduced_samples(
    origA,
    how_many,
    Q,
    reduction_chunk_size=None,
    lll_penalty=10,
    n_workers=50,
    reduction_fn=reduce_with_flatter,
):
    from multiprocessing import Pool
    from tqdm import tqdm
    import functools

    Rs = []
    subsets = []
    pool = Pool(n_workers)
    run_one = functools.partial(
        run_one_reduction,
        origA,
        reduction_chunk_size=reduction_chunk_size,
        Q=Q,
        lll_penalty=lll_penalty,
        reduction_fn=reduction_fn,
    )
    for R_i, subset in tqdm(
        pool.imap_unordered(run_one, range(how_many // reduction_chunk_size // 2))
    ):
        Rs.append(R_i)
        subsets.append(subset)

    return np.stack(Rs), np.stack(subsets)
