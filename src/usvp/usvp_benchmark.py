""""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import pickle
import numpy as np
from time import time
from fpylll import FPLLL, BKZ, GSO, IntegerMatrix
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2
import sys
sys.path.append("./")
from src.generate.lllbkz import encode_intmat, decode_intmat, calc_std_usvp, usvp_params
from subprocess import Popen, PIPE
import io
from src.generate.genSamples import MAX_TIME_BKZ, FLOAT_UPGRADE


### Runs USVP benchmark: define generic class and then subclasses based on setup.
class BenchmarkUSVP(object):
    """Base class for the Benchmark USVP class"""

    def __init__(self, params, thread, logger):
        super().__init__()
        self.logger = logger
        # Some basic checks
        assert params.threshold < params.threshold1 < params.threshold2

        # Now set everything up
        self.params = params
        self.N, self.Q, self.m = params.N, params.Q, params.m
        self.sigma = params.sigma
        self.thread = thread
        self.longtype = np.log2(params.Q) > 30
        self.block_size, self.alpha = params.bkz_block_size, params.alpha
        self.upgraded = False
        self.expNum, self.hamming = thread // 2, params.hamming
        self.weight, self.m, self.delta = usvp_params(
            params.secret_type, self.N, self.Q, self.sigma, self.hamming
        )
        self.logger.info(
            f"w = {self.weight}, delta = {self.delta}, d = {self.m+self.N+1}"
        )
        self.set_float_type(params.float_type)
        secrets = np.load(os.path.join(params.secret_path, "secret.npy"))
        cols = np.where(np.sum(secrets != 0, axis=0) == self.hamming)[0]
        assert len(cols) > self.expNum
        self.s = (secrets[:, cols[self.expNum]]).reshape((self.N, 1))
        assert sum(self.s != 0) == self.hamming

        self.results_path = os.path.join(params.dump_path, "results.pkl")
        self.matrix_filename = os.path.join(params.dump_path, f"matrix_{thread}.npy")
        self.export_path_prefix = os.path.join(
            params.dump_path, f"data_{thread}.prefix"
        )

        self.seed = [params.global_rank, params.env_base_seed, thread]
        self.logger.info(
            f"secret seed = {self.expNum}, h = {self.hamming}, random generator seed: {self.seed}"
        )

        # Params for run
        self.prev_std = 10000  # Condition to only save off matrix if things improve.

        # If interleaving, set up interleaving params:
        self.stdev_tracker = []
        self.lookback = (
            self.params.lookback
        )  # number of steps over which to calculate (avg) decrease, must run given algo at least this many times before switching.
        self.min_decrease = (
            -0.001
        )  # min decrease we have to see over self.lookback steps to consider it "working".

        # Record num successes vs. num attempts
        self.num_success = 0
        self.num_attempts = 0

    def set_float_type(self, float_type):
        self.float_type = float_type
        parsed_float_type = float_type.split("_")
        if len(parsed_float_type) == 2:
            self.float_type, precision = parsed_float_type
            assert self.float_type == "mpfr"
            FPLLL.set_precision(int(precision))

    def get_secret_Ap(self):
        if os.path.isfile(self.matrix_filename):
            secret_Ap = np.load(self.matrix_filename)
            secret, Ap = secret_Ap[:1, : self.N], secret_Ap[:, self.N :]
            self.start = secret_Ap[-1, 0]
        else:
            secret, Ap = self.get_Kannans_embedding()
        return secret, Ap

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
        """
        Saves a matrix like below:
        Left (N+m+1)xN: secret          shape: 1xN
                        0               shape: (N+m-1)xN
                        starttime, 0    shape: 1xN
        Right (N+m+1)x(N+m+1): RAp
        """
        mat_to_save = np.zeros((len(Y), len(Y) + self.N)).astype(int)
        mat_to_save[: len(X), : self.N] = X
        mat_to_save[-1, 0] = int(np.round(self.start))
        mat_to_save[:, self.N :] = Y
        np.save(self.matrix_filename, mat_to_save)

    def check_for_upgrade(self, Ap, orig_std):
        # Run checks.
        newstddev = calc_std_usvp(Ap, orig_std, self.Q, self.m, self.N)
        self.logger.info(f"Stddev reduction = {newstddev}")
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

    def check_usvp_success(self, RAp, secret):
        """Check if uSVP succeeded, save and return True (ending loop) if it did"""
        self.save_mat(secret, RAp)
        guessed_secret = (RAp[0, : self.N] / self.weight).astype(int)
        self.logger.info(
            f"real secret = {secret.flatten()}. solved secret = {guessed_secret}"
        )
        self.logger.info(f"Saved progress at {self.matrix_filename}")
        success = np.all(secret.flatten().astype(bool) == guessed_secret.astype(bool))
        if success:
            self.logger.info(f"Found secret for {self.matrix_filename}")
            results = pickle.load(open(self.results_path, "rb"))
            if type(results) != dict:
                results = results.__dict__
            results[(self.expNum, self.hamming)] += (time() - self.start, success)
            pickle.dump(results, open(self.results_path, "wb"))
            return True  # just end here, don't restart
        return False

    def calc_Ap_stdev(self, Ap):
        orig_mat = Ap.copy()[:-1, self.N : (self.N + self.m)]
        orig_mat[orig_mat > self.Q // 2] -= self.Q
        orig_std = np.std(orig_mat[np.any(orig_mat != 0, axis=1)])
        return orig_std

    def get_Kannans_embedding(self):
        N, m, Q = self.N, self.m, self.Q
        rng = np.random.RandomState(self.seed + [int(time())])
        A = rng.randint(0, Q, size=(m, N), dtype=np.int64)
        assert (np.min(A) >= 0) and (np.max(A) < Q)
        e = rng.normal(0, self.sigma, size=m).round()
        b = ((A @ self.s).flatten() + e).astype(int) % Q
        self.start = time()

        Ap = np.zeros((m + N + 1, m + N + 1), dtype=int)
        Ap[m : m + N, :N] = np.identity(N, dtype=int) * self.weight
        Ap[m : m + N, N : m + N] = A.T
        Ap[:m, N : m + N] = np.identity(m, dtype=int) * Q
        Ap[m + N, N : m + N] = b
        Ap[m + N, m + N] = 1
        return self.s.reshape((1, N)), Ap


class BenchmarkUSVPInterleave(BenchmarkUSVP):
    """
    Note: We observe that if flatter does not succeed on first try, 
    it won't succeed at all. Running BKZ after failed flatter does not 
    increase chance of success. Running BKZ after successful flatter 
    will just return flatter's result.
    """
    def __init__(self, params, thread, logger):
        super().__init__(params, thread, logger)

    def generate(self):
        secret, Ap = self.get_secret_Ap()
        orig_std = self.calc_Ap_stdev(Ap)

        while True:
            ## Flatter first
            self.logger.info(f"Worker {self.thread} starting new flatter run.")
            fplll_Ap = IntegerMatrix.from_matrix(Ap.tolist())
            fplll_Ap_encoded = encode_intmat(fplll_Ap)
            try:
                p = Popen(
                    ["flatter", "-alpha", str(self.alpha)], stdin=PIPE, stdout=PIPE
                )
                out, _ = p.communicate(
                    input=fplll_Ap_encoded
                )  # output from the flatter run.
            except Exception as e:
                self.logger.info(f"flatter failed with error {e}")
                return False
            Ap = decode_intmat(out)
            self.check_for_upgrade(Ap, orig_std)
            if self.check_usvp_success(Ap, secret):
                return False  # Attack succeeded, end the experiment

            ## Now BKZ
            self.logger.info(f"Worker {self.thread} starting new BKZ2.0 run.")
            fplll_Ap = IntegerMatrix.from_matrix(Ap.tolist())
            M = GSO.Mat(fplll_Ap, float_type=self.float_type, update=True)
            BKZ_Obj = BKZ2(M)
            bkz_params = BKZ.EasyParam(
                self.block_size, delta=self.delta, max_time=MAX_TIME_BKZ
            )
            try:
                BKZ_Obj(bkz_params)
            except:
                self.set_float_type(FLOAT_UPGRADE[self.float_type])
                self.logger.info(
                    f"Error running bkz. Upgrading to float type {self.float_type}."
                )
                M = GSO.Mat(fplll_Ap, float_type=self.float_type, update=True)
                BKZ_Obj = BKZ2(M)
                BKZ_Obj(bkz_params)
            Ap = np.zeros((self.m + self.N + 1, self.m + self.N + 1), dtype=np.int64)
            fplll_Ap.to_matrix(Ap)
            # Check for upgrade, then restart if it fails.
            self.check_for_upgrade(Ap, orig_std)
            if self.check_usvp_success(Ap, secret):
                return False


### Run USVP benchmark on flatter WITHOUT interleaving
class BenchmarkUSVPFlatter(BenchmarkUSVP):
    def __init__(self, params, thread, logger):
        super().__init__(params, thread, logger)

    def generate(self):
        secret, Ap = self.get_secret_Ap()
        orig_std = self.calc_Ap_stdev(Ap)

        # THIS VERSION RUNS FLATTER ONCE ON MATRIX
        self.logger.info(f"Worker {self.thread} starting new flatter run.")
        fplll_Ap = IntegerMatrix.from_matrix(Ap.tolist())
        fplll_Ap_encoded = encode_intmat(fplll_Ap)
        try:
            p = Popen(["flatter", "-alpha", str(self.alpha)], stdin=PIPE, stdout=PIPE)
            out, _ = p.communicate(
                input=fplll_Ap_encoded
            )  # output from the flatter run.
        except Exception as e:
            self.logger.info(f"flatter failed with error {e}")
            return False
        Ap = decode_intmat(out)
        self.check_for_upgrade(Ap, orig_std)
        if self.check_usvp_success(
            Ap, secret
        ):  # If it succeeds, up the number of successes.
            self.num_success += 1
        self.num_attempts += 1
        self.logger.info(
            f"{self.num_success} successes out of {self.num_attempts} attempts. Starting new matrix."
        )

        # Now prep for a new run.
        newSecret, newAp = self.get_Kannans_embedding()
        self.save_mat(newSecret, newAp)
        ret_secret = np.zeros((1, self.m + self.N + 1), dtype=int)
        ret_secret[0, : self.N] = secret
        self.write(Ap, ret_secret.T)
        return True


# ### Benchmark uSVP on BKZ without interleaving
class BenchmarkUSVPBKZ(BenchmarkUSVP):
    def __init__(self, params, thread, logger):
        super().__init__(params, thread, logger)

    def generate(self):
        secret, Ap = self.get_secret_Ap()
        orig_std = self.calc_Ap_stdev(Ap)

        fplll_Ap = IntegerMatrix.from_matrix(Ap.tolist())
        M = GSO.Mat(fplll_Ap, float_type=self.float_type, update=True)
        BKZ_Obj = BKZ2(M)
        bkz_params = BKZ.EasyParam(
            self.block_size, delta=self.delta, max_time=MAX_TIME_BKZ
        )
        start_time = None
        while start_time is None or time() - start_time > MAX_TIME_BKZ:
            start_time = time()
            self.logger.info(f"Worker {self.thread} starting new BKZ run.")
            try:
                BKZ_Obj(bkz_params)
            except:
                self.set_float_type(FLOAT_UPGRADE[self.float_type])
                self.logger.info(
                    f"Error running bkz. Upgrading to float type {self.float_type}."
                )
                M = GSO.Mat(fplll_Ap, float_type=self.float_type, update=True)
                BKZ_Obj = BKZ2(M)
                BKZ_Obj(bkz_params)

            RAp = np.zeros((self.m + self.N + 1, self.m + self.N + 1), dtype=np.int64)
            fplll_Ap.to_matrix(RAp)
            self.check_for_upgrade(
                RAp, orig_std
            )  # putting this first so that it logs stddev reduction
            if self.check_usvp_success(RAp, secret):
                return False
