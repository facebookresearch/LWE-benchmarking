""""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import argparse
import pickle
import numpy as np
from glob import glob
import shutil
import sys
sys.path.append("./")

from src import utils
from src.slurm import init_signal_handler, init_distributed_mode
from src.utils import initialize_exp, create_this_logger
from usvp_benchmark import (
    BenchmarkUSVPInterleave,
    BenchmarkUSVPFlatter,
    BenchmarkUSVPBKZ,
)
from joblib import Parallel, delayed, cpu_count

np.seterr(all="raise")


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer")

    # main parameters
    parser.add_argument(
        "--dump_path", type=str, default="", help="Experiment dump path"
    )
    parser.add_argument(
        "--resume_path", type=str, default="", help="Path to load the checkpoints"
    )
    parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")

    # iteration
    parser.add_argument(
        "--env_base_seed",
        type=int,
        default=-1,
        help="Base seed for environments (-1 to use timestamp seed)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Number of CPU workers for DataLoader",
    )

    # Load data
    parser.add_argument(
        "--secret_path",
        type=str,
        default="",
        help="Directory to load the secrets file. Secrets have format secret.npy, created by running generate_A_b.py",
    )
    # CPU
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Multi-GPU - Local rank"
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=-1,
        help="Master port (for multi-node SLURM jobs)",
    )

    # LWE
    parser.add_argument("--N", type=int, default=-1, help="dimension of matrix")
    parser.add_argument("--Q", type=int, default=-1, help="modulo")
    parser.add_argument(
        "--hamming", type=int, default=-1, help="hamming weight of secret"
    )
    parser.add_argument("--secret_type", type=str, required=True, help="what secret distribution? Should match that in secret_path.")

    # Reduction parameters
    parser.add_argument(
        "--float_type",
        type=str,
        default="dd",
        help="double, long double, dpe, dd, qd, or mpfr_<precision>",
    )
    parser.add_argument(
        "--lll_penalty", type=int, default=1, help="penalty on norm of LLL Reduced A"
    )
    parser.add_argument(
        "--m",
        type=int,
        default=-1,
        help="number of samples used in BKZ reduction, defaults to 0.875*N",
    )

    # Interleave methods
    parser.add_argument(
        "--lookback",
        type=int,
        default=3,
        help="How many steps to go with one algorithm before you switch to other?",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="flatter",
        help="Phase 1",
        choices=["LLL", "BKZ", "BKZ2.0", "flatter"],
    )
    parser.add_argument(
        "--algo2",
        type=str,
        default="flatter",
        help="Phase 2, if you dont want to interleave, set algo2 = algo",
        choices=["LLL", "BKZ", "BKZ2.0", "flatter"],
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.4,0.41,0.5",
        help="thresholds to terminate reduction, start writing data, and switch to phase 2",
    )

    # BKZ/LLL specific params
    parser.add_argument(
        "--lll_delta", type=float, default=0.96, help="Phase 1, hermite factor for LLL"
    )
    parser.add_argument(
        "--bkz_block_size",
        type=int,
        default=30,
        help="Phase 1, block size of the BKZ reduction",
    )
    parser.add_argument(
        "--lll_delta2", type=float, default=0.99, help="Phase 2, hermite factor for LLL"
    )
    parser.add_argument(
        "--bkz_block_size2",
        type=int,
        default=40,
        help="Phase 2, block size of the BKZ reduction",
    )

    # Flatter params.
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.04,
        help="Phase 1, alpha param for flatter reduction",
    )
    parser.add_argument(
        "--alpha2",
        type=float,
        default=0.025,
        help="Phase 2, alpha param for flatter reduction",
    )

    return parser


def get_data_one_worker(i, params):
    logger = create_this_logger(params)
    if not os.path.isfile(os.path.join(params.dump_path, "results.pkl")):
        keys = []
        for expNum in range(10):  # fixed 5 experiments per hamming weight for now
            keys += [(expNum, params.hamming), (expNum, params.hamming - 1)]
        pickle.dump(
            dict([(key, []) for key in keys]),
            open(os.path.join(params.dump_path, "results.pkl"), "wb"),
        )
    if params.algo != params.algo2:
        sampleGen = BenchmarkUSVPInterleave(params, i, logger)
    else:
        if params.algo == "flatter":
            sampleGen = BenchmarkUSVPFlatter(params, i, logger)
        elif params.algo == "BKZ2.0":
            sampleGen = BenchmarkUSVPBKZ(params, i, logger)

    gen_more = True
    while gen_more:
        gen_more = sampleGen.generate()


def main(params):
    # initialize the multi-GPU / multi-node training
    # initialize experiment / SLURM signal handler for time limit / pre-emption
    init_distributed_mode(params)
    logger = initialize_exp(params)
    if params.is_slurm_job:
        init_signal_handler()

    assert not params.multi_gpu
    utils.CUDA = False

    if params.env_base_seed < 0:
        params.env_base_seed = np.random.randint(1_000_000_000)

    n_cpu = cpu_count()
    n_jobs = min(n_cpu, params.num_workers)
    logger.info(f" Nb CPU: {n_cpu} and Nb worker: {params.num_workers}")
    Parallel(n_jobs=n_jobs)(
        delayed(get_data_one_worker)(n, params) for n in range(n_jobs)
    )


if __name__ == "__main__":
    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    params.cpu, params.debug_slurm = True, False

    assert params.N > 0 and params.Q > 0 and params.hamming > 0

    params.threshold, params.threshold1, params.threshold2 = [
        float(th) for th in params.thresholds.split(",")
    ]
    params.sigma = 3  # dummy
    params.secret_type = params.secret_type

    # run experiment
    main(params)
