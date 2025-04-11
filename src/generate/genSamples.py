""""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import io
import os
import numpy as np
from time import time
from glob import glob
from fpylll import FPLLL, LLL, BKZ, GSO, IntegerMatrix
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2
from src.generate.lllbkz import calc_std, polish, encode_intmat, decode_intmat
from subprocess import Popen, PIPE
from scipy.linalg import circulant


FLOAT_UPGRADE = {
    "double": "long double",
    "long double": "dd",
    "dd": "qd",
    "qd": "mpfr_250",
}
MAX_TIME_BKZ = 60


class Generator(object):
    def __init__(self, params, thread, logger):
        self.logger = logger
        # Some basic checks
        assert params.threshold < params.threshold1 < params.threshold2

        # Now set everything up
        self.params = params
        # self.step = params.step
        self.N, self.Q, self.m = params.N, params.Q, params.m
        self.d = params.N - params.k
        self.thread = thread
        self.longtype = np.log2(params.Q) > 30
        tiny_A_path = (
            params.reload_data
            if params.alternate_tiny_A_path == ""
            else params.alternate_tiny_A_path
        )
        self.tiny_A = np.load(tiny_A_path, allow_pickle=True)
        self.tiny_A[self.tiny_A > self.Q // 2] -= self.Q
        self.logger.info(f"Generating R,A with tiny samples at {tiny_A_path}.")
        assert self.tiny_A.shape[1] == self.N
        self.seed = [params.global_rank, params.env_base_seed, thread]
        # env_base_seed: different jobs will generate different data
        # thread: different workers will not generate same data
        self.export_path_prefix = os.path.join(
            params.dump_path, f"data_{thread}.prefix"
        )

        self.prev_std = 10000  # Condition to only save off matrix if things improve.

        # If interleaving, set up interleaving params:
        self.stdev_tracker = []
        self.lookback = (
            self.params.lookback
        )  # number of steps over which to calculate (avg) decrease, must run given algo at least this many times before switching.
        self.min_decrease = (
            -0.001
        )  # min decrease we have to see over self.lookback steps to consider it "working".

        # Filenames for saving/loading
        self.matrix_filename = os.path.join(params.dump_path, f"matrix_{thread}.npy")
        self.resume_filename = os.path.join(params.resume_path, f"matrix_{thread}.npy")
        self.temp_ap_filename = os.path.join(params.dump_path, f"ap_temp_{thread}.out")
        if os.path.isfile(self.resume_filename):
            mat_to_save = np.load(self.resume_filename)
            np.save(self.matrix_filename, mat_to_save)
            self.logger.info(f"Resuming from {self.resume_filename}.")
        self.logger.info(f"Random generator seed: {self.seed}.")

    def set_float_type(self, float_type):
        self.float_type = float_type
        parsed_float_type = float_type.split("_")
        if len(parsed_float_type) == 2:
            self.float_type, precision = parsed_float_type
            assert self.float_type == "mpfr"
            FPLLL.set_precision(int(precision))

    def write(self, X, Y):
        assert X.shape[0] == Y.shape[0]
        file_handler_prefix = io.open(
            self.export_path_prefix, mode="a", encoding="utf-8"
        )
        for i in range(X.shape[0]):
            prefix1_str = " ".join(X[i].astype(str))
            prefix2_str = " ".join(Y[i].astype(str))
            file_handler_prefix.write(f"{prefix1_str} ; {prefix2_str}\n")
        file_handler_prefix.flush()

    def save_mat(self, X, Y):
        mat_to_save = np.zeros((len(Y), len(Y) + self.m)).astype(int)
        mat_to_save[: len(X), : self.m] = X
        mat_to_save[:, self.m :] = Y
        np.save(self.matrix_filename, mat_to_save)

    def rlwe_circ(self, a):
        A = circulant(a)
        tri = np.triu_indices(self.N, 1)
        A[tri] *= -1
        return A

    def get_A_Ap(self):
        m, d, Q = self.m, self.d, self.Q
        rng = np.random.RandomState(self.seed + [int(time())])
        idxs = rng.choice(len(self.tiny_A), size=m, replace=False)
        U = idxs.reshape((m, 1))
        if not self.params.rlwe: 
            A = self.tiny_A[idxs, -d:]
        else: # Typically, we preprocess as LWE, not RLWE, but the RLWE option exists. 
            A = self.rlwe_circ(self.tiny_A[idxs[0], -d:])  # just take the first vector

        assert np.max(A) - np.min(A) < Q

        # Arrange the matrix as [0 q*Id; w*Id A]
        Ap = np.zeros((m + d, m + d), dtype=int)
        Ap[d:, :m] = np.identity(m, dtype=np.int64) * self.params.lll_penalty
        Ap[d:, m:] = A
        Ap[:d, m:] = np.identity(d, dtype=int) * Q

        return U, Ap  # U.shape = mx1, Ap.shape = (m+N)*(m+N)

    def run_flatter_once(self, Ap):
        """
        Runs a single loop of flatter.
        """
        self.logger.info(f"Worker {self.thread} starting new flatter run.")
        fplll_Ap = IntegerMatrix.from_matrix(Ap.tolist())
        fplll_Ap_encoded = encode_intmat(fplll_Ap)
        try:
            env = {**os.environ, "OMP_NUM_THREADS": "1"}
            p = Popen(
                ["/private/home/ewenger/usr/bin/flatter", "-alpha", str(self.alpha)], stdin=PIPE, stdout=PIPE, env=env
            )
        except Exception as e:
            self.logger.info(f"flatter failed with error {e}")
        out, _ = p.communicate(input=fplll_Ap_encoded)  # output from the flatter run.
        Ap = decode_intmat(out)
        if self.params.rand_rows:
            Ap = np.random.permutation(Ap)  # permute the rows.
        return Ap

    def run_bkz_once(self, Ap):
        """
        Runs a single round of BKZ.
        """
        self.logger.info(f"Worker {self.thread} starting new BKZ run.")
        fplll_Ap = IntegerMatrix.from_matrix(Ap.tolist())
        M = GSO.Mat(fplll_Ap, float_type=self.float_type, update=True)
        bkz_params = BKZ.Param(self.block_size, delta=self.delta, max_time=MAX_TIME_BKZ)
        if (self.params.algo == "BKZ") or (self.params.algo2 == "BKZ"):
            L = LLL.Reduction(M, delta=self.delta)
            BKZ_Obj = BKZ.Reduction(M, L, bkz_params)
        else:
            BKZ_Obj = BKZ2(M)
        # Run once.
        try:
            BKZ_Obj() if (
                (self.params.algo == "BKZ") or (self.params.algo2 == "BKZ")
            ) else BKZ_Obj(bkz_params)
        except Exception as e:
            self.logger.info(e)
            # for bkz2.0, this would catch the case where it needs more precision for floating point arithmetic
            # for bkz, the package does not throw the error properly. Make sure to start with enough precision
            self.set_float_type(FLOAT_UPGRADE[self.float_type])
            self.logger.info(
                f"Error running bkz. Upgrading to float type {self.float_type}."
            )
            M = GSO.Mat(fplll_Ap, float_type=self.float_type, update=True)
            BKZ_Obj = BKZ2(M)
            BKZ_Obj(bkz_params)
        Ap = np.zeros((Ap.shape[0], Ap.shape[1]), dtype=np.int64)
        fplll_Ap.to_matrix(Ap)
        if self.params.rand_rows:
            Ap = np.random.permutation(Ap)
        return Ap

    def run_lll_once(self, Ap):
        # TODO this should run the LLL algo from fpylll
        raise NotImplementedError("This is not implemented yet.")

    def compute_stdev(self, Ap, UT, use_polish=True, save=True, algo="flatter"):
        if use_polish:
            Ap = polish(Ap)
        newstddev = calc_std(Ap, self.Q, self.m)
        if save:
            self.stdev_tracker.append(newstddev)
            self.save_mat(UT, Ap)
            self.logger.info(
                f"stddev = {newstddev}. Saved progress at {self.matrix_filename} after {algo} run."
            )
        return newstddev

    def check_for_stall(self):
        if len(self.stdev_tracker) >= (self.lookback + 1):
            decreases = [
                self.stdev_tracker[i - 1] - self.stdev_tracker[i]
                for i in range(-self.lookback, 0)
            ]
            if np.mean(decreases) > self.min_decrease:
                return True  # Your mean decrease is higher than the mandated minimum decrease over the last self.lookback rounds - you've stalled.
        return False

    def check_for_param_upgrade(self, Ap, UT, newstddev, oldstddev=None):
        if (
            newstddev < self.params.threshold1
        ):  # start writing matrix when reaching threshold1
            self.logger.info(f"stddev = {newstddev}. Exporting {self.matrix_filename}")
            X = UT[:1].T
            R = (Ap[:, : self.m] / self.params.lll_penalty).astype(int)
            self.write(X, R.T)  # R.T: (m, m+N)
            if newstddev < self.params.threshold:  # terminate when reaching threshold
                self.logger.info(f"Starting new matrix at {self.matrix_filename}")
                return False  # You've reached the threshold, stop!
        # Upgrades for flatter/BKZ
        if not self.upgraded and newstddev < self.params.threshold2:
            # Go into Phase 2
            self.upgraded = True
            self.block_size, self.delta, self.alpha = (
                self.params.bkz_block_size2,
                self.params.lll_delta2,
                self.params.alpha2,
            )
            self.logger.info(
                f"Upgrading to delta = {self.delta}, block size = {self.block_size}, alpha = {self.alpha}"
            )
            return True

        # See if polishing helped at all -- BKZ only.
        if (oldstddev is not None) and (oldstddev - newstddev > 0):
            self.logger.info(
                f"stddev reduction: {oldstddev - newstddev} from polishing. "
            )
            return True
        return (
            None  # if you return None, it means you didn't meet any of these criteria.
        )


### INTERLEAVED REDUCTION ###
class InterleavedReduction(Generator):
    """Will run two reductions interleaved"""

    def __init__(self, params, thread, logger):
        super().__init__(params, thread, logger)
        self.set_float_type(params.float_type)

        # Set up function calls.
        if params.algo in ["BKZ", "BKZ2.0"]:
            self.algo1 = self.run_bkz_once
        elif params.algo == "flatter":
            self.algo1 = self.run_flatter_once
        else:
            self.algo1 = self.run_lll_once

        if params.algo2 in ["BKZ", "BKZ2.0"]:
            self.algo2 = self.run_bkz_once
        elif params.algo2 == "flatter":
            self.algo2 = self.run_flatter_once
        else:
            self.algo2 = self.run_lll_once

    def check_for_switch(self, Ap, UT):
        # Run checks.
        algo = self.params.algo if self.a1 else self.params.algo2
        algo2 = self.params.algo if not self.a1 else self.params.algo2
        newstddev = self.compute_stdev(Ap, UT, use_polish=True, save=True, algo=algo)
        check = self.check_for_param_upgrade(Ap, UT, newstddev)
        if (
            self.num_times_run >= self.lookback
        ) and self.check_for_stall():  # Repeat the above step.
            self.logger.info(f"Stalled, switching from {algo} to {algo2}.")
            self.stall_count += 1
            self.a1 = not self.a1
            self.a2 = not self.a2
            self.num_times_run = 0
        return (
            Ap,
            check,
        )  # Either you've upgraded params OR you've gone below the desired threshold.

    def run(self, UT, Ap):
        """Assumes no encoding of Ap, each individual function call will do this."""
        self.stdev_tracker = []
        self.num_times_run = 0  # number of times we have run a particular algorithm -- to make sure we've rough enough to implement our check.
        self.stall_count = 0  # number of times an algo (BKZ or flatter) has run without producing an avg.

        self.a1 = True
        self.a2 = False
        while True:
            # First do algo1.
            while self.a1:
                self.num_times_run += 1
                try:
                    Ap = self.algo1(Ap)
                except Exception as e:
                    self.logger.info(
                        f"Exception {e} encountered by worker {self.thread}, aborting."
                    )
                    return None, -1
                # Run checks.
                Ap, check = self.check_for_switch(Ap, UT)
                if check is not None:
                    return Ap, check

            # Then flip to algo2.
            while self.a2:
                self.num_times_run += 1
                try:
                    Ap = self.algo2(Ap)
                except Exception as e:
                    self.logger.info(
                        f"Exception {e} encountered by worker {self.thread}, aborting."
                    )
                    return None, -1
                # Run checks.
                Ap, check = self.check_for_switch(Ap, UT)
                if check is not None:
                    return Ap, check

    def generate(self):
        if os.path.isfile(self.matrix_filename):
            A_Ap = np.load(self.matrix_filename)
            UT, Ap = A_Ap[:, : self.m], A_Ap[:, self.m :]
        else:
            U, Ap = self.get_A_Ap()
            UT = U.T  # To have num_cols = m, U and A are always transposed

        # Params/upgrades for BKZ/flatter.
        self.upgraded = False
        self.block_size, self.delta, self.alpha = (
            self.params.bkz_block_size,
            self.params.lll_delta,
            self.params.alpha,
        )
        if calc_std(Ap, self.Q, self.m) < self.params.threshold2:  # use Phase 2 params
            self.upgraded = True
            self.block_size, self.delta, self.alpha = (
                self.params.bkz_block_size2,
                self.params.lll_delta2,
                self.params.alpha2,
            )

        param_change = True
        while param_change:
            Ap, param_change = self.run(UT, Ap)
            if param_change == -1:
                return False  # Worker encountered error, end now.

        # Rewrite checkpoint with new data and return the bkz reduced result
        newA, newAp = self.get_A_Ap()
        self.save_mat(newA.T, newAp)
        return True  # Worker finished, end now.
