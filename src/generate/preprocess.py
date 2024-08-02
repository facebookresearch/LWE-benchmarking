""""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import sys
import numpy as np

sys.path.append(".")

from src import utils
from src.slurm import init_signal_handler, init_distributed_mode
from src.utils import bool_flag, initialize_exp, create_this_logger
from src.generate.genSamples import InterleavedReduction
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
        "--reload_data",
        type=str,
        default="",
        help="Directory to load the tiny A dataset from the disk.",
    )
    parser.add_argument(
        "--alternate_tiny_A_path",
        type=str,
        default="",
        help="load config from reload_data, but tiny A from somewhere else?",
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
    parser.add_argument(
        "--rlwe",
        type=bool_flag,
        default=False,
        help="Run preprocessing using RLWE matrix format rather than LWE",
    ) 
    parser.add_argument("--N", type=int, default=-1, help="dimension of matrix")
    parser.add_argument("--Q", type=int, default=-1, help="modulo")

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
        "--rand_rows",
        type=bool_flag,
        default=False,
        help="use row randomization during preprocessing?",
    )
    parser.add_argument(
        "--permute_cols",
        type=bool_flag,
        default=False,
        help="randomize cols during preprocessing? This is useful if you want a flat reduction profile over all columns.",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=-1,
        help="number of samples used in BKZ reduction, defaults to 0.875*N",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.4,0.41,0.5",
        help="thresholds to terminate reduction, start writing data, and switch to phase 2",
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
        default="BKZ2.0",
        help="Phase 2, if you dont want to interleave, set algo2 = algo",
        choices=["LLL", "BKZ", "BKZ2.0", "flatter"],
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
    parser.add_argument(
        "--k",
        type=int,
        default=0,
        help="Hybrid Dual exhaustive search dimension -> reduce n-k"
    )

    return parser


def get_data_one_worker(i, params):
    logger = create_this_logger(params)
    sampleGen = InterleavedReduction(params, i, logger)
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
    params.threshold, params.threshold1, params.threshold2 = [
        float(th) for th in params.thresholds.split(",")
    ]
    if params.m == -1:
        params.m = int(params.N * 7 // 8)

    # run experiment
    main(params)
