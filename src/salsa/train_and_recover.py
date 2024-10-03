""""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import getpass
import os
import sys
import time
sys.path.append("./")

from src.slurm import init_distributed_mode, init_signal_handler
from src.salsa.train.evaluator import SecretRecovery
from src.salsa.train import get_dataset, get_model, get_metrics
from src.salsa.train.trainer import Trainer
from src.utils import bool_flag, initialize_exp, load_params


def get_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("--seed", type=int, default=-1, help="-1 uses time() as seed")

    # Logging
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--val_every", type=int, default=60_000)
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument(
        "--save_periodic", type=int, default=0, help="Save every n epochs"
    )
    parser.add_argument("--check_secret_every", type=int, default=2000)
    user = getpass.getuser()
    parser.add_argument("--dump_path", default=f"/checkpoint/{user}/dumped")
    parser.add_argument("--exp_name", default="debug_recover")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    parser.add_argument("--run_id", default="")

    # Model args
    parser.add_argument("--enc_emb_dim", type=int, default=512)
    parser.add_argument("--n_enc_layers", type=int, default=4)
    parser.add_argument("--n_enc_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--attention_dropout", type=float, default=0)
    parser.add_argument(
        "--angular_emb",
        type=bool_flag,
        default=False,
        help="Whether to use xy coordinate embeddings",
    )
    parser.add_argument(
        "--matrix_emb",
        type=bool_flag,
        default=False,
        help="Use if training data are matrices to get matrix-specific embedding, currently only suported with RLWE",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=None,
        help="Size of patches to use for matrix embedding, currently only suported with RLWE",
    )
    parser.add_argument(
        "--compile", type=bool_flag, default=True, help="Use torch.compile?"
    )

    # Optimizer args
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam_warmup,lr=0.00001,warmup_updates=8000,weight_decay=0.001",
        help="Optimizer (SGD / RMSprop / Adam, etc.)",
    )
    parser.add_argument(
        "--timescale", type=int, default=40, help="How fast to decay the inv sqrt lr."
    )
    parser.add_argument(
        "--dtype", default="float16", choices=["float32", "float16", "bfloat16"]
    )

    # Training args
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--clip_grad_norm", type=float, default=5.0)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--val_batch_size", type=int, default=512)

    # Data args
    parser.add_argument("--data_path", required=True)
    parser.add_argument(
        "--base", type=int, help="base for the digit encoding.", default=81
    )
    parser.add_argument(
        "--bucket_size",
        type=int,
        help='how big the low digit "buckets" are.',
        default=1,
    )
    parser.add_argument(
        "--rlwe", type=int, default=0, help="#Blocks in MLWE/RLWE, 0 is for LWE"
    )
    parser.add_argument(
        "--stacked_circulants",
        type=bool_flag,
        default=True,
        help="If RLWE data, either stack rotated datasets or stack circulants",
    )
    parser.add_argument("--shuffle", type=bool_flag, default=True)
    parser.add_argument("--workers", type=int, default=8, help="CPU workers for data")

    # Slurm args
    parser.add_argument(
        "--master_port", type=int, default=int(os.environ.get("MASTER_PORT", 10035))
    )
    parser.add_argument(
        "--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1))
    )
    parser.add_argument(
        "--debug_slurm",
        type=bool_flag,
        default=False,
        help="Debug multi-GPU / multi-node within a SLURM job",
    )
    parser.add_argument("--cpu", type=bool_flag, default=False, help="Run on CPU")

    # Experiment args
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--secret_seed", type=int, required=True)
    parser.add_argument("--hamming", required=True, type=int, help="Hamming weights")
    parser.add_argument("--max_hours", type=float, default=70, help="Max time allowed")

    # Special args
    parser.add_argument(
        "--recover_only",
        type=bool_flag,
        default=False,
        help="Only do secret recovery, not training.",
    )
    parser.add_argument(
        "--dxdistinguisher",
        type=bool_flag,
        default=False,
        help="Run Derivative Distinguisher",
    )
    parser.add_argument(
        "--distinguisher_size",
        type=int,
        default=128,
        help="Sample count for distinguishing. Must fit in one inference-only batch.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default='lwe',
        help="Training Task, possible values: lwe, rlwe-all, mlwe-i"
    )
    parser.add_argument(
        "--A_shift",
        type=int,
        default=0,
        help="Shift of rows of A to train on if task = mlwe-i,"
    )
    #  Use compute_optimal_mlwe_shift.py script to get optimal shift (multiply cost by n), 
    # otherwise run attack on all/random shifts in [0,n-1]

    return parser

def main(params):
    # initialize the multi-GPU / multi-node training
    # initialize experiment / SLURM signal handler for time limit / pre-emption
    init_distributed_mode(params)
    if params.is_slurm_job:
        init_signal_handler()

    logger = initialize_exp(params)

    dataset = get_dataset(params)
    model = get_model(params)
    train_metrics, recover_metrics = get_metrics(params)

    secret_recovery = SecretRecovery(params, dataset, model, recover_metrics)
    trainer = Trainer(params, dataset, model, train_metrics, secret_recovery)

    if params.recover_only:
        recovered = secret_recovery.recover(trainer.epoch)
        sys.exit()

    while not trainer.should_stop_training:
        logger.info("============ Starting epoch %i ... ============" % trainer.epoch)
        trainer.train()
        logger.info("============ End of epoch %i ============" % trainer.epoch)

        trainer.end_epoch()

        if trainer.check_time_limit():
            logger.warning("Quitting because over time limit.")
            break


if __name__ == "__main__":
    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    params = load_params(params)

    # update seed to unix time (seconds)
    if params.seed < 0:
        params.seed = int(time.time())

    # run experiment
    main(params)
