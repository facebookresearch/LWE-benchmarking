"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Very simple script for generating secret datasets. It only supports LWE (no ring LWE) and does not support Cathy's "storage efficient" option.

The main arguments:

--processed_dump_path: the directory with data_*.prefix files, or the parent of that directory. See the NOTE in the comments.
--secret_path: path to an existing secret.npy file. If not provided, the script will generate a new secret.
--secret_type: binary, ternary, gaussian, binomial. Look at the SecretFactory classes.
--actions: one or more of "secrets", "only_secrets", "plot", "describe"
    * secrets: given A data, generate new secrets using --min_hamming, --max_hamming, etc., and associated b data.
    * only_secrets: given N, min_hamming, max_hamming, and secret type, generate only the secret.npy file (useful for uSVP and MiTM attacks).
    * plot: plot reduction levels for RA and save to the output directory.
    * describe: calculate a lot of statistics about the generated data.

python generate_A_b.py --processed_dump_path /path/to/preprocessed/data/ --exp_name debug --dump_path /path/to/dump/data --secret_type binary --num_secret_seeds 10 --min_hamming 1 --max_hamming 20 --actions secrets plot describe

This will make some binary secrets with hamming weights from 1 to 20 and describe the outputs.

Number of samples: 1350977
Regular std:  1203.9386
Centered std: 737.9868
Regular reduction:  1.2530
Centered reduction: 0.7677
Cruel bits:  143
Recommended max hamming: 10, max brute force bits: 6 , choices: 10.7B
BKZ block sizes: 35, 40
Std deviation of (RA*s - Rb mod Q) / Q: 0.322 (0.002)


You want to make sure that 

* the number of samples is what you expect (~4M)
* Centered reduction is similar to the write threshold of the preprocessing job
* BKZ block sizes don't surprise you (read from the original params file)

While reading from the data.prefix file, if there are issues in it (likely from concurrent writing to data.prefix), the logs will complain:

It might complain that the err std is too close to random. This means that (RA @ secret - Rb) is very close to a uniform error distribution, which is bad news.

It might also complain that the reduction is not below the original write threshold. This indicates a bug because the whole point of preprocessing is to get std(centered(RA)) under that threshold.
"""

import argparse
import getpass
import glob
import itertools
import json
import logging
import math
import os
import pickle
import shutil
import sys
import time
import numpy as np
from tqdm import tqdm
sys.path.append(".")
from src.generate.lllbkz import get_mlwe_circ, centered
from src.utils import create_this_logger, human, init_rng, mod_mult_torch, read, remove_redundant_rows, shift_negate, shuffled


def get_params():
    """
    Generate a parameters parser.
    """
    parser = argparse.ArgumentParser(
        description="Generate secrets from reduced A matrices."
    )

    user = getpass.getuser()
    parser.add_argument("--dump_path", default=f"/checkpoint/{user}/dumped")
    parser.add_argument(
        "--exp_name", default="secrets", help="Experiment name"
    )
    parser.add_argument("--exp_id", default="", help="Experiment ID")
    parser.add_argument(
        "--tag", default="", help="Extra info to include in folder name."
    )
    parser.add_argument(
        "--log_every", default=500, type=int, help="How often to log progress."
    )

    parser.add_argument("--seed", type=int, default=42, help="-1 to use timestamp seed")
    parser.add_argument(
        "--actions",
        nargs="+",
        choices=["secrets", "plot", "describe", "only_secrets"],
        default=["secrets"],
        help="Which functions to run.",
    )

    # Load data
    parser.add_argument(
        "--processed_dump_path",
        required=True,
        help="Directory to load the preprocessed data from.",
    )
    parser.add_argument("--secret_path", default="", help="Path to predefined secret.npy file.")

    # LWE
    parser.add_argument("--num_secret_seeds", type=int, default=10)
    parser.add_argument("--N", type=int, default=512, help="LWE dimension for only_secrets")
    parser.add_argument("--min_hamming", type=int, default=3, help="Min hamming weight")
    parser.add_argument(
        "--max_hamming", type=int, default=20, help="Max hamming weight."
    )
    parser.add_argument(
        "--secret_type",
        default="binary",
        choices=["binary", "ternary", "gaussian", "binomial"],
    )
    parser.add_argument(
        "--sigma", type=float, default=3, help="sigma for gaussian error"
    )
    parser.add_argument("--gamma", type=int, default=2, help="for binomial error")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=4_010_000,
        help="Maximum number of training samples. The number of training samples might be less than this number.",
    )
    parser.add_argument(
        "--std_threshold",
        default=1.0,
        type=float,
        help="Minimum std deviation of reduced A to use.",
    )

    # RLWE
    parser.add_argument("--rlwe", type=int, default=0, help="LWE (0) or RLWE (1), MLWE (k>1)")
    parser.add_argument("--nu", type=int, default=0, help="Number of cruel bits, used to split the cliff, if provided.")

    params = parser.parse_args()

    # Save command used
    params.command = (" ".join(["python"] + sys.argv),)

    return params


# Secret types
class BinaryFactory:
    type_ = "binary"

    def __init__(self, *, sigma):
        self.sigma = sigma

    def new(self, N, min_h, max_h, num_secret_seeds):
        num_secrets = num_secret_seeds * (max_h - min_h + 1)
        full = self._fill(shape=(N, num_secrets))
        secret = self._sparsify(full, min_h, max_h, num_secret_seeds)
        return secret

    def _fill(self, shape):
        return np.ones(shape, dtype=np.int64)

    def _sparsify(self, full, min_h, max_h, num_secret_seeds):
        for h in range(min_h, max_h + 1):
            for seed_id in range(num_secret_seeds):
                column = (h - min_h) * num_secret_seeds + seed_id
                nonzeros = np.flatnonzero(full[:, column])

                if len(nonzeros) > h:
                    # Too many nonzero coordinates, randomly zero some out.
                    extra = len(nonzeros) - h
                    idxs = rng.choice(nonzeros, size=extra, replace=False)
                    full[idxs, column] = 0

                # Required Hamming weight is higher than nonzeros in secret.
                assert h >= np.count_nonzero(full[:, column])

        return full

    def error(self, shape):
        return rng.normal(0, self.sigma, size=shape).round().astype(np.int64)

    def __repr__(self):
        variables = ", ".join([f"{k}={v}" for k, v in vars(self).items()])
        return f"{self.__class__.__name__}({variables})"


class TernaryFactory(BinaryFactory):
    type_ = "ternary"

    def _fill(self, shape):
        return rng.choice([-1, 1], size=shape)


class GaussianFactory(BinaryFactory):
    type_ = "gaussian"

    def _fill(self, shape):
        decimals = rng.normal(0, self.sigma, size=shape)
        return decimals.round().astype(np.int64)


class BinomialFactory(BinaryFactory):
    type_ = "binomial"

    def __init__(self, *, gamma):
        self.gamma = gamma

    def _fill(self, shape):
        half1 = rng.binomial(self.gamma, 0.5, size=shape)
        half2 = rng.binomial(self.gamma, 0.5, size=shape)
        return half1 - half2

    def error(self, shape):
        err1 = rng.binomial(self.gamma, 0.5, shape)
        err2 = rng.binomial(self.gamma, 0.5, shape)
        return err1 - err2


def circulate_mlwe_A(A, n, k):
    new_mat = np.flip(A.reshape((len(A), k, n)), axis=2)
    circulated_mat = np.zeros((n, len(A), k, n), dtype=np.int64)
    for i in range(n):
        circulated_mat[~i] = new_mat
        new_mat = shift_negate(new_mat)

    circulated_mat = circulated_mat.swapaxes(0, 1)
    A = circulated_mat.reshape((-1, k * n))
    return A

def data_check(batch_RA, batch_Rb, secret, rlwe=0):
    """Will return False if it finds that stdev is bad"""
    # Skip file if (RA @ s - Rb) % Q is not significantly better.
    if (
        rlwe
    ):  # because indexing is different for RLWE, must generate circulant matrices to check.
        N, k = batch_RA.shape[1], rlwe
        n = N // k
        batch_RA = circulate_mlwe_A(batch_RA, n, k) % params.Q
        batch_Rb = batch_Rb.reshape((-1, batch_Rb.shape[-1]))

    bad_err_std = (
        centered((batch_RA @ secret[:, 1:] - batch_Rb[:, :-1]) % params.Q, params.Q).std()
        / params.Q
    )
    good_err_std = (
        centered((batch_RA @ secret - batch_Rb) % params.Q, params.Q).std() / params.Q
    )
    logger.info("Good err std: %f \t Bad err std: %f", good_err_std, bad_err_std)

    if good_err_std + 0.005 >= bad_err_std:
        logger.warning(
            "Err std too close to random. [err: %.3f, random: %.3f]",
            good_err_std,
            bad_err_std,
        )
        return False

    # skip file if stddev is above write threshold.
    std = np.sqrt(12) * centered(batch_RA, params.Q).std() / params.Q
    # logger.warning('RA std is %.3f', std)
    if std > params.write_threshold:
        logger.warning("RA above reduction threshold. [std: %.3f]", std)
        return False
    return True


def init_tiny_A(nu=0, k=0):
    logger.info(f"Loading from {params.orig_A_path}")
    A = np.load(params.orig_A_path, allow_pickle=True)
    N = A.shape[1]
    idxs = np.arange(N, dtype=int)
    if nu>0 and k>1:
        nu_parts = np.array_split(np.arange(nu), k)  # Split nu into k parts
        nr_parts = np.array_split(nu+np.arange(N-nu), k)  # Split nr into k parts

        idxs = []
        for i in range(k):
            idxs.extend(nu_parts[i])
            idxs.extend(nr_parts[i])

    A = A[:,idxs]
    return A, idxs


def generate(test_size=10000):
    if params.secret_type == "binary":
        secret_factory = BinaryFactory(sigma=params.sigma)
    elif params.secret_type == "ternary":
        secret_factory = TernaryFactory(sigma=params.sigma)
    elif params.secret_type == "gaussian":
        secret_factory = GaussianFactory(sigma=params.sigma)
    elif params.secret_type == "binomial":
        secret_factory = BinomialFactory(gamma=params.gamma)
    else:
        raise ValueError(params.secret_type)

    if not os.path.isfile(params.secret_path):
        logger.info("Creating %s secret", secret_factory.type_)
        secret = secret_factory.new(
            params.N, params.min_hamming, params.max_hamming, params.num_secret_seeds
        )
    else: 
        logger.info("Loading secret from %s", params.secret_path)
        secret = np.load(params.secret_path)

    h_range = range(params.min_hamming, params.max_hamming + 1)
    num_secrets = len(h_range) * params.num_secret_seeds
   
    tiny_A, col_permutation = init_tiny_A(nu = params.nu, k=params.rlwe)

    if params.rlwe:
        assert params.N%params.rlwe == 0, f"Total MLWE dimension {params.N} should be a multiple of k={params.rlwe}"
        k = params.rlwe
        n = params.N//k

        params.max_samples //= n
        A_dot_s = np.array([get_mlwe_circ(ak, n, k)@secret for ak in tiny_A]) % params.Q
    else:
        A_dot_s = tiny_A @ secret % params.Q

    # generate the e in the original b=A*s+e
    tiny_e = secret_factory.error(A_dot_s.shape)

    # Will be used by Rb in so that the distribution and dependence of vars are correct
    tiny_b = (A_dot_s + tiny_e) % params.Q

    # full_RA (params.max_samples, N) is the entire set (~4M) of inputs to the transformer
    full_RA = np.zeros((params.max_samples, params.N), dtype=np.int64)

    # full_Rb (NUM_SAMPLES, num_secrets) is a matrix of outputs to the transformer.
    # There are num_secrets columns in Rb; one for each secret.
    # Shape is different for RLWE setting because of the circulant.
    b_shape = (
        (params.max_samples, params.N//params.rlwe, num_secrets)
        if params.rlwe
        else (params.max_samples, num_secrets)
    )
    full_Rb = np.zeros(b_shape, dtype=np.int64)

    # Find the data.prefix file
    data_prefix_path = os.path.join(params.processed_dump_path, "data.prefix")
    if not os.path.isfile(data_prefix_path):
        logger.warning("Copying all data_*.prefix to %s", data_prefix_path)
        # NOTE: you might have to change this glob if you're using threadsafe or not.
        # If you're going after some slurm jobs, then you probably want:
        #
        #    f"{params.processed_dump_path}/*/data_*.prefix"
        #
        # Note the /*/ to match the slurm jobs
        paths = glob.glob(f"{params.processed_dump_path}/*/data_*.prefix")
        with open(data_prefix_path, "w") as outfile:
            for path in tqdm(paths, desc="Copying data.prefix"):
                for line in remove_redundant_rows(path, params.m):
                    outfile.write(line)

    logger.info("Loading data from %s", data_prefix_path)

    n_matrices, n_pairs = 0, 0

    for A, R in read(data_prefix_path, params.m):
        # Remove any rows of R that are all zeros
        # https://stackoverflow.com/a/11188955
        R = R[~np.all(R == 0, axis=1)]

        idxs = list(A.flatten())
        A = tiny_A[idxs]
        assert A.shape == (params.m, params.N)

        b = tiny_b[idxs]

        if params.rlwe:
            assert b.shape == (params.m, params.N//params.rlwe, num_secrets)
        else:
            assert b.shape == (params.m, num_secrets)

        # Compute RA and check if it's too big.
        batch_RA = mod_mult_torch(R, A, params.Q)
        assert batch_RA.dtype == np.int64, (batch_RA.dtype, batch_RA[0][0])
        new_pairs, _ = batch_RA.shape
        end = n_pairs + new_pairs
        if (n_pairs > 0) and (end > params.max_samples):
            logger.info("About to exceed %d. Terminating.", params.max_samples)
            break

        batch_Rb = mod_mult_torch(R, b, params.Q)
        assert batch_Rb.dtype == np.int64, (batch_Rb.dtype, batch_Rb[0][0])
    
        if True: #data_check(batch_RA, batch_Rb, secret, params.rlwe):
            full_RA[n_pairs:end] = batch_RA
            full_Rb[n_pairs:end] = batch_Rb
        else:
            logger.debug("Error too large, ignoring this R")
            continue
        
        # Increment and log if needed.
        n_pairs += new_pairs
        n_matrices += 1
        if n_matrices % params.log_every == 0:
            logger.info(
                "Processed %s matrices from the disk, %s samples.",
                human(n_matrices),
                human(n_pairs),
            )
    # Truncate arrays to n_pairs
    full_RA = full_RA[:n_pairs]
    full_Rb = full_Rb[:n_pairs]

    if params.rlwe:
        # Need at least N samples for RLWE in order to have a full circulant.
        test_size = max(test_size//n, params.N + 1)

    test_RA, train_RA = full_RA[:test_size], full_RA[test_size:]
    test_Rb, train_Rb = full_Rb[:test_size], full_Rb[test_size:]

    def save_and_log(path, arr):
        np.save(path, arr)
        logging.info("Saved %s at %s", os.path.basename(path), path)

    # Write to disk.
    # Save the entire secret and orig_b if we want experiments on the same secrets.
    # We assume this is a unix system. I don't use os.path.join because of this.
    save_and_log(f"{params.dump_path}/orig_b.npy", tiny_b)
    save_and_log(f"{params.dump_path}/orig_A.npy", tiny_A)
    save_and_log(f"{params.dump_path}/test_A.npy", test_RA)
    save_and_log(f"{params.dump_path}/train_A.npy", train_RA)

    hamming_secret_pairs = itertools.product(h_range, range(params.num_secret_seeds))
    # Since we dump the secrets down a directory.
    save_and_log(f"{params.secret_dir}/secret.npy", secret)
    for h, seed_i in tqdm(list(hamming_secret_pairs)):
        secret_i = (h - params.min_hamming) * params.num_secret_seeds + seed_i
        np.save(f"{params.secret_dir}/secret_{h}_{seed_i}.npy", secret[..., secret_i])
        np.save(f"{params.secret_dir}/orig_b_{h}_{seed_i}.npy", tiny_b[..., secret_i])
        np.save(f"{params.secret_dir}/test_b_{h}_{seed_i}.npy", test_Rb[..., secret_i])
        np.save(f"{params.secret_dir}/train_b_{h}_{seed_i}.npy", train_Rb[..., secret_i])
        np.save(f"{params.secret_dir}/reduced_b_{h}_{seed_i}.npy", full_Rb[..., secret_i])

    logger.info(
        "Saved secret, orig_b, train_b, test_b, and diff for each h/seed at %s",
        params.dump_path,
    )


# Plotting
def plot(reduced_A, orig_A):
    """Plots A's column-wise standard deviation."""
    import matplotlib.pyplot as plt

    x = np.arange(params.N)
    fig, ax = plt.subplots()

    reduced_std = centered(reduced_A, params.Q).std()

    y = centered(reduced_A, params.Q).std(0)
    ax.plot(x, y, label="Reduced")

    # Original data
    orig_std = centered(orig_A, params.Q).std()

    y = centered(orig_A, params.Q).std(0)
    ax.plot(x, y, label="Original")

    ax.set_ylim(ymin=0)
    ax.set_xlabel("A[i]")
    ax.set_ylabel("Std. Dev.")
    ax.set_title(
        f"{os.path.basename(params.processed_dump_path)} (Reduction {human(reduced_std)}/{human(orig_std)} = {reduced_std/orig_std:.2f})"
    )
    ax.legend()

    fig.savefig(
        f"{params.dump_path}/R_A_{params.N}_{params.logq}_omega{params.omega}.pdf"
    )
    fig.savefig(
        f"{params.dump_path}/R_A_{params.N}_{params.logq}_omega{params.omega}.png"
    )

def generate_usvp_secret():
    if params.secret_type == "binary":
        secret_factory = BinaryFactory(sigma=params.sigma)
    elif params.secret_type == "ternary":
        secret_factory = TernaryFactory(sigma=params.sigma)
    elif params.secret_type == "gaussian":
        secret_factory = GaussianFactory(sigma=params.sigma)
    elif params.secret_type == "binomial":
        secret_factory = BinomialFactory(gamma=params.gamma)
    else:
        raise ValueError(params.secret_type)

    logger.info("Creating %s secret", secret_factory.type_)
    secret = secret_factory.new(
        params.N, params.min_hamming, params.max_hamming, params.num_secret_seeds
    )
    def save_and_log(path, arr):
        np.save(path, arr)
        logging.info("Saved %s at %s", os.path.basename(path), path)

    # Since we dump the secrets down a directory.
    save_and_log(f"{params.secret_dir}/secret.npy", secret)


def describe(reduced_A, orig_A):
    """
    Calculates and reports:
    * The number of samples in reduced_A
    * the std deviation
    * the reduction to 4 decimal places
    * the number of cruel bits
    * recommended hamming weights for secrets that will take 100B or fewer guesses.
    * the block sizes used during BKZ
    * the variance and std of the error distribution after reduction.
    """
    # Number of samples
    logger.info("Number of samples: %d", reduced_A.shape[0])

    # Std deviation
    logger.info("Regular std:  %.4f", reduced_A.std())
    logger.info("Centered std: %.4f", centered(reduced_A, params.Q).std())

    # Reduction
    logger.info("Regular reduction:  %.4f", reduced_A.std() / orig_A.std())
    logger.info(
        "Centered reduction: %.4f", centered(reduced_A, params.Q).std() / centered(orig_A, params.Q).std()
    )

    # Cruel bits
    orig_std = centered(orig_A, params.Q).std()
    # bits above this threshold are cruel.
    cruel_threshold = 0.5 * orig_std
    bitwise_std = centered(reduced_A, params.Q).std(axis=0)
    n_cruel_bits = (bitwise_std > cruel_threshold).sum()
    logger.info("Cruel bits:  %d", n_cruel_bits)

    # Recommended hamming weights
    max_brute_force_bits = 0
    while math.comb(n_cruel_bits, max_brute_force_bits + 1) < 100_000_000_000:
        max_brute_force_bits += 1

    max_h = int(max_brute_force_bits / n_cruel_bits * params.N)

    logger.info(
        "Recommended max hamming: %d, max brute force bits: %d , choices: %s",
        max_h,
        max_brute_force_bits,
        human(math.comb(n_cruel_bits, max_brute_force_bits)),
    )

    # BKZ block sizes
    # logger.info(
    #     "BKZ block sizes: %d, %d", params.bkz_block_size1, params.bkz_block_size2
    # )

    # Std dev of (RA @ s - RB) % Q
    # Do this for 10 secrets. Report all of the values, the mean and the std.
    diff_stds = []
    try:
        diff_files = [file for file in os.listdir(params.secret_dir) if "diff" in file]
        for diff_file in shuffled(diff_files, rng)[:50]:
            diff = np.load(os.path.join(params.secret_dir, diff_file))
            diff_stds.append((diff % params.Q).std() / params.Q)

        if diff_stds:
            logger.info(
                "Std deviation of (RA*s - Rb mod Q) / Q: %.3f (%.3f)",
                np.mean(diff_stds),
                np.std(diff_stds),
            )
    except Exception as e:
        logger.warning("Error calculating diff std: %s", e)


def get_loaded_params():
    # Copy the params file up from an experiment subdirectory, if it doesn't exist in
    # reload_data.
    params_pkl_path = os.path.join(params.processed_dump_path, "params.pkl")
    if not os.path.exists(params_pkl_path):
        for dirpath, dirnames, filenames in os.walk(params.processed_dump_path):
            if "params.pkl" not in filenames:
                continue
            src = os.path.join(dirpath, "params.pkl")
            dst = os.path.join(params.processed_dump_path, "params.pkl")
            try:
                shutil.copy(src, dst)
                params_pkl_path = dst
            except PermissionError:
                params_pkl_path = src

    # Grab parameters from the processed data
    with open(params_pkl_path, "rb") as fd:
        loaded_params = pickle.load(fd)
    if not isinstance(loaded_params, dict):
        loaded_params = vars(loaded_params)

    return loaded_params


if __name__ == "__main__":
    params = get_params()
    if params.seed < 0:
        params.seed = int(time.time())
    os.makedirs(params.dump_path, exist_ok=True)
    logger = create_this_logger(params)
    rng = init_rng(params.seed, logger)

    # If a secret path is provided, make sure params are updated accordingly. 
    if os.path.isfile(params.secret_path):
        # Infer characteristics from dirname (min_hamming, max_hamming, secret_type) 
        dirname = os.path.dirname(params.secret_path).split("/")[-1]
        try:
            s = np.load(params.secret_path)
            params.secret_type = dirname.split('_')[0]
            params.min_hamming = int(dirname.split('_')[-2][1:])
            params.max_hamming = int(dirname.split('_')[-1])
            params.num_secret_seeds = s.shape[-1] // (params.max_hamming - params.min_hamming + 1)
            print(params.min_hamming, params.max_hamming)
        except Exception as e:
            print(f"Failed with error {e}")
            assert False == True, 'Could not infer secret characteristics from dirname'

     # If you're running only_secrets, you only want secret.npy file. 
    if "only_secrets" in params.actions: 
        # Just make the secret file and exit.
        params.dump_path = os.path.dirname(params.dump_path) # We have custom experiment ID for only secrets.

        # Make a path to dump the secrets
        params.secret_dir = os.path.join(params.dump_path,
        f"secret_N{params.N}_{params.secret_type}_{params.min_hamming}_{params.max_hamming}",
        )
        os.makedirs(params.secret_dir, exist_ok=True)

        # Make the secret
        generate_usvp_secret()
        sys.exit(0) # Then you're done


    loaded_params = get_loaded_params()

    params.N, params.Q = loaded_params["N"], loaded_params["Q"]
    params.logq = int(np.ceil(np.log2(params.Q)))
    if "lll_penalty" in loaded_params:
        params.omega = loaded_params["lll_penalty"]
    else:
        params.omega = loaded_params["omega"]
    if "bkz_block_size" in loaded_params:
        params.bkz_block_size1 = loaded_params["bkz_block_size"]
    else:
        params.bkz_block_size1 = loaded_params["bkz_block_size1"]
    #params.bkz_block_size2 = loaded_params["bkz_block_size2"]

    # std. dev. threshold for when matrices were written to data.prefix.
    if "write_threshold" in loaded_params:
        params.write_threshold = loaded_params["write_threshold"]
    else:
        params.write_threshold = float("inf")
        for threshold_key in ("threshold1", "threshold2", "threshold"):
            if threshold_key in loaded_params:
                params.write_threshold = loaded_params[threshold_key]
                break

    dirname = f"A_b_{params.N}_{params.logq}_omega{params.omega}"
    if params.tag:
        dirname += f"_{params.tag}"

    params.secret_dir = os.path.join(
        params.dump_path,
        f"{params.secret_type}_secrets_h{params.min_hamming}_{params.max_hamming}",
    )
    if "secrets" in params.actions:
        os.makedirs(params.secret_dir, exist_ok=True)

    if "m" not in loaded_params or loaded_params["m"] == -1:
        params.m = loaded_params["N"]
    else:
        params.m = loaded_params["m"]

    if "reload_data" in loaded_params:
        params.orig_A_path = loaded_params["reload_data"]
    else:
        params.orig_A_path = loaded_params["orig_A_path"]


    if "secrets" in params.actions:
        with open(os.path.join(params.secret_dir, "params.pkl"), "wb") as fd:
            pickle.dump(vars(params), fd)
        with open(os.path.join(params.secret_dir, "params.json"), "w") as fd:
            json.dump(vars(params), fd)

        generate()

    # Check if these file exists
    # Reduced data
    reduced_A = np.load(f"{params.dump_path}/train_A.npy")  # Changed from reduced_A.npy
    orig_A = np.load(f"{params.dump_path}/orig_A.npy")

    if "plot" in params.actions:
        plot(reduced_A, orig_A)

    if "describe" in params.actions:
        describe(reduced_A, orig_A)
