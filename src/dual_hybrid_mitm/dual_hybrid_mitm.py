"""""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""""

import os
import glob
import argparse
import pickle as pkl
import numpy as np
import sys

sys.path.append('.')
from src import utils
from src.slurm import init_signal_handler, init_distributed_mode
from src.utils import initialize_exp, create_this_logger, bool_flag
from run_attack import DualHybrid, MITM
from joblib import Parallel, delayed, cpu_count

np.seterr(all='raise')

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Run a dual hybrid MITM attack.")

    # main parameters
    parser.add_argument("--debug", type=bool_flag, default=False)
    parser.add_argument("--dump_path", type=str, default="",
                        help="Experiment dump path")
    parser.add_argument("--resume_path", type=str, default="",
                        help="Path to load the checkpoints")
    parser.add_argument("--exp_name", type=str, default="debug",
                        help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="",
                        help="Experiment ID")
    parser.add_argument("--slurm", type=bool, default=True,
                        help="If false, then will change how we load in data.prefix files (e.g. not looking for subdirectories)")

    # MITM parameters
    parser.add_argument("--step", type=str, choices=["reduce", "mitm"], default="reduce", help="Run reduction or MITM part of attack?")
    parser.add_argument("--k", type=int, default=0,
                        help="Splitting dimension for dim/error tradeoff.")
    parser.add_argument("--tau", type=int, default=-1, help="How many short vectors to get?")
    parser.add_argument("--bound", type=int, default=-1, help="Bound to use for guessing.")


    # iteration
    parser.add_argument("--env_base_seed", type=int, default=-1,
                        help="Base seed for environments (-1 to use timestamp seed)")
    parser.add_argument("--num_workers", type=int, default=10,
                        help="Number of CPU workers for DataLoader")

    # Load data
    parser.add_argument("--short_vectors_path", type=str, default="",
                        help="Path to data.prefix file saved off ")
    parser.add_argument("--secret_path", type=str, default="",
                        help="Directory to load the secrets file. Secrets have format secret.npy, created by running generate_A_b.py")
    parser.add_argument("--secret_seed", type=int, default=0, help="which index of secret to load.")
    
    # CPU
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Multi-GPU - Local rank")
    parser.add_argument("--master_port", type=int, default=-1,
                        help="Master port (for multi-node SLURM jobs)")

    # LWE
    parser.add_argument("--N", type=int, default=-1,
                        help="dimension of matrix, for MLWE should be n*k (e.g. if aiming for total dim 512 with 2 n=256 RLWE modules, give n=512)")
    parser.add_argument("--Q", type=int, default=-1,
                        help="modulo")
    parser.add_argument("--sigma", type=float, default=3, help="sigma for error")
    parser.add_argument("--gamma", type=float, default=2, help="gamma for binomial error")
    parser.add_argument("--hamming", type=int, default=-1, 
                        help="hamming weight of secret")
    parser.add_argument("--num_bits_in_table", type=int, default=-1, help="Number of bits we assume are in table. Default is hamming // 2.")
    parser.add_argument("--secret_type", type=str, default='binary', help="what secret distribution")
    parser.add_argument('--mlwe_k', type=int, default=0) # LWE = 0, RLWE=1, MLWE = k > 1 (k is the number of modules). 

    # Reduction parameters
    parser.add_argument("--float_type", type=str, default="dd",
                        help="double, long double, dpe, dd, qd, or mpfr_<precision>")
    parser.add_argument("--lll_penalty", type=int, default=1,
                        help="penalty on norm of LLL Reduced A")
    parser.add_argument("--m", type=int, default=-1,
                        help="number of samples used in BKZ reduction, defaults to 0.875*N")

    # Interleave methods
    parser.add_argument('--lookback', type=int, default=3,
                        help="How many steps to go with one algorithm before you switch to other?")
    parser.add_argument("--algo", type=str, default='flatter',
                        help='Phase 1', choices=["LLL", "BKZ", "BKZ2.0", "flatter"])
    parser.add_argument("--algo2", type=str, default='BKZ2.0',
                        help='Phase 2, if you dont want to interleave, set algo2 = algo', choices=["LLL", "BKZ", "BKZ2.0", "flatter"])
    parser.add_argument("--thresholds", type=str, default="0.4,0.41,0.5",
                        help="dummy for uSVP, ad we don't upgrade block size.")



    # BKZ/LLL specific params
    parser.add_argument("--lll_delta", type=float, default=0.96,
                        help="Phase 1, hermite factor for LLL")
    parser.add_argument("--bkz_block_size", type=int, default=30,
                        help="Phase 1, block size of the BKZ reduction")
    parser.add_argument("--lll_delta2", type=float, default=0.99,
                        help="Phase 2, hermite factor for LLL")
    parser.add_argument("--bkz_block_size2", type=int, default=40,
                        help="Phase 2, block size of the BKZ reduction")

    # Flatter params.
    parser.add_argument("--alpha", type=float, default=0.04,
                        help="Phase 1, alpha param for flatter reduction")
    parser.add_argument("--alpha2", type=float, default=0.025,
                        help="Phase 2, alpha param for flatter reduction")

    return parser

def get_data_one_worker(i, params):
    logger = create_this_logger(params)

    if params.step == "reduce":
        sampleGen = DualHybrid(params, i, logger)

        gen_more = True
        while gen_more:
            gen_more = sampleGen.generate()
    else:
        mitm = MITM(params, logger, i)
        mitm.run()

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

    if params.step == 'mitm':
        params.num_workers = 1 # only one worker for MITM.
        if params.short_vectors_path == "":
            assert False == True, "Must provide path to data.prefix file containing short vectors to run guessing part."

    if params.num_workers == 0:
        params.num_workers = 1
        get_data_one_worker(0, params) 
    else:
        n_cpu = cpu_count()
        n_jobs = min(n_cpu, params.num_workers)
        logger.info(f" Nb CPU: {n_cpu} and Nb worker: {params.num_workers}")
        Parallel(n_jobs=n_jobs)(delayed(get_data_one_worker)(n, params) for n in range(n_jobs))

if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    params.cpu, params.debug_slurm = True, False
    params.alternate_tiny_A_path = "" # dummy

    if params.step == 'reduce':
        assert params.N > 0 and params.Q > 0 and params.hamming > 0
    else: # MITM
        try:
            dual_params = pkl.load(open(os.path.join(params.short_vectors_path, "params.pkl"), "rb"))
        except: # it was a slurm job
            possible = glob.glob(os.path.join(params.short_vectors_path, "*/params.pkl"))
            if len(possible) == 0:
                assert False == True, "Could not find params.pkl file in short_vectors_path."
            dual_params = pkl.load(open(possible[0], "rb"))
        dual_params.short_vectors_path = params.short_vectors_path

        dual_params.debug = params.debug
        dual_params.mitm_hamming = params.hamming
        if dual_params.mitm_hamming != dual_params.hamming:
            print(f"Using hamming weight {dual_params.mitm_hamming} in experiment ({dual_params.hamming} used in reduction). Be careful!")
        
        if params.secret_path != dual_params.secret_path:
            # Prefer the secret path provided as script argument. 
            dual_params.secret_path = params.secret_path

        if params.secret_seed != dual_params.secret_seed:
            # Prefer the secret seed provided as script argument. 
            dual_params.secret_seed = params.secret_seed

        if params.tau > 0:
            # Prefer the tau provided as script argument.
            dual_params.tau = params.tau

        # Use the provided gamma and sigma
        dual_params.gamma = params.gamma
        dual_params.sigma = params.sigma

        # Set bound if provided. 
        if params.bound > 0: 
            dual_params.bound = params.bound

        params = dual_params # just use the same params as before.
        params.local_rank = -1
        params.step = 'mitm' # Set the right flag.
        params.num_workers = 1 # Only one fellow can look at the table.
        params.dump_path = os.path.dirname(params.dump_path)
        params.exp_name = params.exp_id
        idx=0
        while os.path.exists(os.path.join(params.dump_path, params.exp_name, f'mitm_{idx}')):
            idx += 1
        params.exp_id = f"mitm_{idx}"
        try:
            params.mlwe_k = dual_params.mlwe_k
        except:
            params.mlwe_k = 0 # If it doesn't have this, then it is 0. 

    params.threshold, params.threshold1, params.threshold2 = 0.1, 0.11,0.111 # dummy
    params.rand_rows = False # Do not shuffle.
    params.m = params.N # number of samples used in BKZ reduction.
    main(params)
