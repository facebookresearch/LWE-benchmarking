""""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from abc import abstractmethod
import json
import os
import pickle
from logging import getLogger
from functools import partial
import numpy as np
import torch
import itertools

from scipy import stats
from tqdm.auto import trange

from src.utils import to_json, mod_diff


logger = getLogger()


class SecretRecovery:
    def __init__(self, params, dataset, model, recover_metrics):
        self.params = params
        self.Q = params.Q
        self.device = params.device
        self.io_encoder = dataset.io_encoder
        self.test_dataset = dataset.test_dataset
        self.orig_dataset = dataset.orig_dataset
        self.model = model
        self.recover_metrics = recover_metrics
        self.secret_type = params.secret_type

        self.secret_log = SecretLog(epoch=0)

        secret_check = SecretCheck(self.params, self.orig_dataset)

        if self.params.dxdistinguisher:
            self.dist = SlopeDistinguisher(self.params, secret_check, self.secret_log)
        elif self.secret_type == "binary":
            self.dist = BinaryDistinguisher(self.params, secret_check, self.secret_log)
        elif self.secret_type in ("binomial", "ternary"):
            self.dist = TwoBitDistinguisher(
                self.params,
                secret_check,
                self.secret_log,
                mod_diff.__name__,
                (mod_diff.__name__, partial(mod_diff, Q=params.Q)),
            )
        else:
            raise ValueError(self.secret_type)

        self.amp_ctx = torch.amp.autocast(
            device_type="cuda", dtype=getattr(torch, self.params.dtype)
        )

    @torch.no_grad()
    def recover(self, epoch):
        logger.info("Starting secret recovery.")
        # Recover secret. Higher difference => bit more likely nonzero

        self.secret_log["epoch"] = epoch

        logger.info("Distinguishing non-zero bits using modular difference.")

        A, b = self.test_dataset
        assert len(A) == self.params.distinguisher_size

        f_a, f_ai, dx = self.compute_outputs(A, b)

        matched = self.dist.run(f_a, f_ai, dx)

        if matched:
            logger.info("Predicted secret.")
        else:
            logger.info("Failed predicting secret.")

        if self.params.is_master:
            metrics = {
                **{k: v.item() for k, v in self.recover_metrics.compute().items()},
                "recover/matched": matched,
                "recover/epoch": epoch,
            }
            logger.info("%s", json.dumps(metrics))
            self.secret_log.dump(self.params.dump_path, epoch)

        return matched

    def compute_outputs(self, A, b):
        self.recover_metrics.reset()

        f_a, logits = self.inference(A, True)
        self.recover_metrics(logits, self.io_encoder(b).to(self.params.device))
        self.recover_metrics.compute()

        f_ai = []
        dx = []
        for Ai, dxi in self.dist.get_inputs(A):
            preds = self.inference(Ai)
            f_ai.append(preds)
            dx.append(dxi)

        return f_a, f_ai, dx

    def inference(self, A, return_logits=False):
        self.model.eval()
        A_enc = self.io_encoder(A)
        A_enc = A_enc.to(self.device)
        with self.amp_ctx:
            logits = self.model(A_enc)

        preds = self.io_encoder.decode(logits)
        if return_logits:
            return preds, logits
        return preds


class SecretLog:
    success_key = "success"

    def __init__(self, epoch=0):
        self._log = {self.success_key: []}
        self.logger = getLogger("recovery-log")
        self["epoch"] = epoch

    def __getitem__(self, key):
        return self._log[key]

    def __setitem__(self, key, value):
        self._log[key] = value

    def __contains__(self, key):
        return key in self._log

    def add_success(self, method_name):
        if method_name not in self[self.success_key]:
            self[self.success_key].append(method_name)

    def dump(self, path, epoch):
        filepath = os.path.join(path, f"secret_recovery_{epoch}.pkl")
        try:
            with open(filepath, "wb") as fd:
                pickle.dump(self._log, fd)
        except Exception as err:
            self.logger.warning("Log: %s", self._log)
            self.logger.warning("Exception saving log: %s", err)


@to_json.register
def _(obj: SecretLog):
    return obj._log


class SecretCheck:
    def __init__(self, params, orig_dataset):
        self.N = params.N
        self.Q = params.Q
        self.sigma = params.sigma

        A, b = orig_dataset
        self.A = A.squeeze().numpy()
        self.b = b.squeeze().numpy()

        # Only need this for gaussian secret; eventually we won't need it.
        self.secret_type = params.secret_type

    def match_secret(self, guess):
        """Takes an int or bool (binary) list or array as secret guess and check against
        the original tiny dataset.
        """
        guess = np.array(guess).astype(int)

        err_pred = (self.A @ guess - self.b) % self.Q
        err_pred[err_pred > self.Q // 2] -= self.Q
        return (np.std(err_pred) < 2 * self.sigma).item()

    def match_secret_iter(self, idx_list, sorted_idx_with_scores, method_name):
        """
        Takes a list of indices sorted by scores (descending, high score means more likely to be 1)
        and iteratively matches the secret.
        """
        guess = np.zeros(self.N)
        for i in range(min(self.N // 5, len(idx_list))):  # sparse assumption
            guess[idx_list[i]] = 1
            if self.match_secret(guess):
                return True
        logger.info(f"{method_name}: secret not predicted.")
        return False


class BaseDistinguisher(object):
    def __init__(self, params, secret_check, secret_log) -> None:
        self.params = params
        self.Q = params.Q
        self.distinguisher_size = params.distinguisher_size
        self.secret_check = secret_check
        self.secret_log = secret_log

    def get_inputs(self, A):
        # Prepare the random values to add to each coordinate of A. The first half in
        # (0.3q, 0.4q), the second half in (0.6q, 0.7q)
        add_rand = torch.randint(
            low=3 * self.Q // 10,
            high=2 * self.Q // 5,
            size=(self.distinguisher_size // 2,),
            dtype=torch.int64,
        )
        add_rand = torch.concat((add_rand, -add_rand))

        for i in trange(self.params.N):
            # Modify ith "bit" of A
            Ai = A.clone()
            Ai[:, i] += add_rand
            Ai[:, i] %= self.Q
            yield Ai, add_rand

    def compute_scores(self, y0, y1s, dx=None):
        scores = [mod_diff(y0, mod_pred, Q=self.Q) for mod_pred in y1s]
        self.secret_log[mod_diff.__name__] = scores
        return scores

    @abstractmethod
    def run(self, base_pred, preds, dx):
        raise NotImplementedError()


class TwoBitDistinguisher(BaseDistinguisher):
    """Two bit distinguisher for binomial secrets"""

    def __init__(self, params, secret_check, secret_log, func_name, func1):
        super().__init__(params, secret_check, secret_log)
        self.func0 = func_name
        self.func1 = func1

    def check_cliques(self, cliques, nonzeros=None):
        """
        nonzeros: an array of secret indices
        cliques: partition of these indices. Their union should be integers in [0, len(nonzeros)).
        """
        # Case: we want to define a custom nonzero set.
        if nonzeros is None:
            nonzeros = self.nonzeros

        if self.params.secret_type == "binomial":
            s_i = np.arange(
                -self.params.gamma, self.params.gamma + 1
            )  # possible secret bits.
            s_i = sorted(
                list(np.delete(s_i, np.where(s_i == 0)[0])), key=lambda x: abs(x)
            )  # Start with 1 bits since they're more likely.
        elif self.params.secret_type == "ternary":  # TODO test if this works.
            s_i = np.array([-1, 1])

        # Get all possible combos of secret bits, cutting list in half since we can multiply by -1.
        secret_elements = [
            e
            for e in itertools.product(s_i, repeat=len(cliques))
            if len(np.unique(e)) > 1 and e[0] not in range(min(s_i), 0)
        ]
        for values in secret_elements:  # All combos of clique length.
            guess = np.zeros(self.params.N)
            for i, c in enumerate(cliques):
                guess[nonzeros[list(c)]] = values[i]

            method_name = "Distinguisher Method"

            if self.secret_check.match_secret(guess) or self.secret_check.match_secret(
                guess * -1
            ):
                logger.info("%s: all bits in secret recovered!", method_name)
                self.secret_log.add_success(method_name)
                return True

        return False

    def run(self, base_pred, lwe_preds, dx):
        """
        - First runs cliques on all bits.
        - Test for ternary secret.
        - Split first clique, test with three cliques.
        - Split second clique, test with three cliques.
        - Test with all four splits.
        """
        func1, diff_func = self.func1
        self.secret_log_name = f"{self.params.secret_type}_{self.func0}_{func1}"

        diffs = self.compute_scores(base_pred, lwe_preds)

        logger.info(f"Distinguishing secret bits using the {func1}.")
        idx_sorted = np.argsort(
            diffs
        )  # sorted secret indices. Closer to the end means more likely nonzero.


        # eliminate the cases of h=1 and 2
        if self.check_cliques([[0], []], nonzeros=idx_sorted[-1:]):
            return True
        if self.check_cliques(
            [[0, 1], []], nonzeros=idx_sorted[-2:]
        ) or self.check_cliques([[0], [1]], nonzeros=idx_sorted[-2:]):
            return True

        # compute the dist matrices and run.
        dists = np.zeros((self.params.N, self.params.N))
        half = self.params.distinguisher_size // 2  # TODO is 128 samples enough?
        for i in range(self.params.N):
            for j in range(i):
                dists[i, j] = diff_func(lwe_preds[i][half:], lwe_preds[j][half:])
                dists[j, i] = diff_func(lwe_preds[i][:half], lwe_preds[j][:half])

        # for each hamming weight, make secret guesses by forming partitions of the nonzero bits.
        for h in [
            self.params.hamming
        ]:  ##range(5, self.params.N // 5): # sparse assumption
            self.nonzeros = idx_sorted[-h:]
            dist_mats = [np.zeros((h, h)), np.zeros((h, h))]
            for i in range(h):
                for j in range(i):
                    idx1, idx2 = self.nonzeros[i], self.nonzeros[j]
                    dist_mats[0][i, j], dist_mats[1][i, j] = (
                        dists[idx1, idx2],
                        dists[idx2, idx1],
                    )

            dist_mats.append(dist_mats[0] + dist_mats[1])
            for i, dist_mat in enumerate(dist_mats):
                # First, bipartition WHOLE dist_mat (using all nonzeros). Go through all indices.
                x1, y1 = np.unravel_index(
                    np.argsort(dist_mat, axis=None), dist_mat.shape
                )
                c0, c1, success = self.bipartition_set(list(zip(x1, y1)), dist_mat)
                if success:
                    return True
                if self.params.secret_type == "binomial":  # Base case: gamma = 2.
                    # First separate c0, and test against c1 (assumes all c1 are same)
                    c00, c01, c10, c11 = (
                        [],
                        [],
                        [],
                        [],
                    )  # assumes four unique bit sets (e.g. gamma=2)
                    if len(list(c0)) > 1:
                        # CASE 1: assumes 1-2 bit values in each clique.
                        try:
                            # Try/except deals with the idx_to_test being generated in a funny way with el[1] > el[0] -- TODO find a way to know a priori which way to generate indices.
                            c00, c01, success = self.bipartition_set(
                                list(itertools.combinations(list(c0)[::-1], 2)),
                                dist_mat,
                                other_cliques=[c1],
                            )
                        except:
                            c00, c01, success = self.bipartition_set(
                                list(itertools.combinations(list(c0), 2)),
                                dist_mat,
                                other_cliques=[c1],
                            )
                        if success:
                            return True

                        # CASE 2: one clique ends up with > 2 bit values.
                        if len(c00) >= 3:
                            c000, c001, success = self.bipartition_set(
                                list(itertools.combinations(list(c00)[::-1], 2)),
                                dist_mat,
                                other_cliques=[c01, c1],
                            )
                            if success:
                                return True
                        if len(c01) >= 3:
                            c010, c011, success = self.bipartition_set(
                                list(itertools.combinations(list(c01)[::-1], 2)),
                                dist_mat,
                                other_cliques=[c00, c1],
                            )
                            if success:
                                return True

                    # Then bipartition c1, and test against c0 (assumes all c0 are same)
                    if len(list(c1)) > 1:
                        try:
                            c10, c11, success = self.bipartition_set(
                                list(itertools.combinations(list(c1)[::-1], 2)),
                                dist_mat,
                                other_cliques=[c0],
                            )
                        except:
                            c10, c11, success = self.bipartition_set(
                                list(itertools.combinations(list(c1), 2)),
                                dist_mat,
                                other_cliques=[c0],
                            )
                        if success:
                            return True

                        # CASE 2: one clique ends up with > 2 bit values. Need to partition again.
                        if len(c10) >= 3:
                            c100, c101, success = self.bipartition_set(
                                list(itertools.combinations(list(c10)[::-1], 2)),
                                dist_mat,
                                other_cliques=[c11, c0],
                            )
                            if success:
                                return True
                        if len(c11) >= 3:
                            c110, c111, success = self.bipartition_set(
                                list(itertools.combinations(list(c11)[::-1], 2)),
                                dist_mat,
                                other_cliques=[c10, c0],
                            )
                            if success:
                                return True

                    # Then test all four sets.
                    success = self.check_cliques([c00, c01, c10, c11])
                    if success:
                        return True
        return False

    def bipartition_set(self, idx_to_test, dist_mat, other_cliques=None):
        """
        Given a distance matrix and an array of nonzero bits,
        form bipartitions of the nonzero bits s.t. the distance within the partitions are low,
        and the distance across the partitions are high.

        curr_nonzeros are the bits we're testing currently, may not be the same as all_nonzeros.
        """

        cliques = []  # a list of disjoint sets
        for i, j in idx_to_test:
            if j >= i:
                continue
            if len(cliques) == 0:
                cliques.append(set([i, j]))
            else:
                membership = [None, None]
                for clique_id in range(len(cliques)):
                    if i in cliques[clique_id]:
                        membership[0] = clique_id
                    if j in cliques[clique_id]:
                        membership[1] = clique_id
                # if neither elements found, create a new clique
                if membership[0] is None and membership[1] is None:
                    cliques.append(set([i, j]))
                # if one element belongs to a clique, add the other element to that clique
                elif membership[0] is not None and membership[1] is None:
                    cliques[membership[0]].add(j)
                elif membership[0] is None and membership[1] is not None:
                    cliques[membership[1]].add(i)
                # if both elements are found in cliques, merge the cliques
                elif membership[0] != membership[1]:
                    cliques[membership[0]] = cliques[membership[0]].union(
                        cliques[membership[1]]
                    )
                    del cliques[membership[1]]

            # Special case: we just have two elements in one clique, and other_cliques is not None.
            if (len(idx_to_test) == 1) and (other_cliques is not None):
                cliques[0] = set([idx_to_test[0][0]])
                cliques.append(set([idx_to_test[0][1]]))

            # We have finished processing the pair. Check if we are done
            all_cliques = cliques.copy()
            if other_cliques is not None:
                all_cliques.extend(other_cliques)
            # Make sure you don't reuse bits from other cliques.
            used_bits = (
                []
                if other_cliques is None
                else list(itertools.chain.from_iterable(other_cliques))
            )
            unused_bits = set(range(len(self.nonzeros))) - set(used_bits)

            if len(cliques) == 1 and (
                np.sum([len(clique) for clique in all_cliques])
                >= len(self.nonzeros) - 1
            ):
                # CASE ONE: all but one bit accounted for, only one clique generated.
                # the algorithm is suggesting that all indices have the same sign, or only one index has a different sign
                if (
                    np.sum([len(clique) for clique in all_cliques])
                    == len(self.nonzeros) - 1
                ) and self.check_cliques(
                    [range(len(self.nonzeros)), []]
                    if other_cliques is None
                    else [
                        [el for el in range(len(self.nonzeros)) if el not in used_bits],
                        [],
                        *other_cliques,
                    ]
                ):
                    return None, None, True

                # CASE TWO: all bits accounted for but only one clique, meaning that all other bits are in other_cliques.
                # Edge case: need to update the cliques we test on. Either use the other_cliques or use the first element of other cliques.
                second_clique = unused_bits - cliques[0]
                all_els = list(cliques[0])
                all_els.extend(list(second_clique))
                idx_to_test = list(itertools.combinations(sorted(all_els)[::-1], 2))
                return self.fm(
                    idx_to_test,
                    dist_mat,
                    cliques[0],
                    second_clique,
                    other_cliques=other_cliques,
                )

            if len(cliques) == 2 and np.sum(
                [len(clique) for clique in all_cliques]
            ) == len(self.nonzeros):
                return self.fm(
                    idx_to_test,
                    dist_mat,
                    cliques[0],
                    cliques[1],
                    other_cliques=other_cliques,
                )

            all_cliques = []

        # Return statement if you reach here (you shouldn't)
        if len(cliques) == 1:
            return cliques[0], [], False
        elif len(cliques) == 2:
            return cliques[0], cliques[1], False
        else:
            return [], [], False

    def fm(self, idx_to_test, dist_mat, clique0, clique1, other_cliques=None):
        """
        This algorithm is inspired by the Fiduccia-Mattheyses Partitioning Algorithm.
        Relaxed the condition to allow more randomness.

        other_cliques: should be a list of sets or lists containing bit indicies of other, previously identified cliques.
        """

        def add_log_entry(c0, c1, other):
            log = [self.nonzeros[list(c0)], self.nonzeros[list(c1)]]
            if other is not None:
                for c in other:
                    log.append(self.nonzeros[list(c)])
            return log

        logs = [add_log_entry(clique0, clique1, other_cliques)]
        # check the correctness of the initial partition
        all_cliques = (
            [clique0, clique1]
            if other_cliques is None
            else [clique0, clique1, *other_cliques]
        )
        if self.check_cliques(all_cliques):
            self.secret_log[self.secret_log_name] = logs
            return clique0, clique1, True

        # initialize the metric: the average dist across cliques over the average dist within cliques
        in_sum, cross_sum, in_num, cross_num = 0, 0, 0, 0
        for ii, jj in idx_to_test:
            if jj >= ii:
                continue
            if (ii in clique0 and jj in clique0) or (ii in clique1 and jj in clique1):
                in_sum += dist_mat[ii][jj]
                in_num += 1
            else:
                cross_sum += dist_mat[ii][jj]
                cross_num += 1
        if cross_sum == 0 or in_sum == 0:
            return clique0, clique1, False
        before_move = cross_sum * in_num / cross_num / in_sum

        # try swapping for a finite number of times
        # Get the bits we care about
        bits_under_consideration = list(
            itertools.chain.from_iterable([clique0, clique1])
        )
        for _ in range(10):
            for ii in bits_under_consideration:
                clique0scores = [
                    dist_mat[max(ii, jj)][min(ii, jj)] for jj in clique0 if ii != jj
                ]
                clique1scores = [
                    dist_mat[max(ii, jj)][min(ii, jj)] for jj in clique1 if ii != jj
                ]
                # keep the move if it increases
                move = sum(clique0scores) - sum(clique1scores), len(
                    clique0scores
                ) - len(clique1scores)
                if ii in clique0 and len(clique0) != 1:  # move from clique0 to clique1
                    try:
                        after_move = (
                            (cross_sum + move[0])
                            * (in_num - move[1])
                            / (cross_num + move[1])
                            / (in_sum - move[0])
                        )
                    except:  # divide by zero
                        logger.info("exception in after_move")
                        after_move = before_move - 1
                    clique0.remove(ii)
                    clique1.add(ii)

                    all_cliques = (
                        [clique0, clique1]
                        if other_cliques is None
                        else [clique0, clique1, *other_cliques]
                    )
                    if self.check_cliques(all_cliques):
                        logs.append(add_log_entry(clique0, clique1, other_cliques))
                        self.secret_log[self.secret_log_name] = logs
                        return clique0, clique1, True
                    if after_move < before_move:  # revert
                        clique1.remove(ii)
                        clique0.add(ii)
                    else:
                        self.secret_log[self.secret_log_name] = logs
                elif (
                    ii in clique1 and len(clique1) != 1
                ):  # move from clique1 to clique0
                    try:
                        after_move = (
                            (cross_sum - move[0])
                            * (in_num + move[1])
                            / (cross_num - move[1])
                            / (in_sum + move[0])
                        )
                    except:
                        # error, force after_move to be smaller.
                        after_move = before_move - 1
                    clique1.remove(ii)
                    clique0.add(ii)

                    all_cliques = (
                        [clique0, clique1]
                        if other_cliques is None
                        else [clique0, clique1, *other_cliques]
                    )
                    if self.check_cliques(all_cliques):
                        logs.append(add_log_entry(clique0, clique1, other_cliques))
                        self.secret_log[self.secret_log_name] = logs
                        return clique0, clique1, True
                    if after_move < before_move:
                        clique0.remove(ii)
                        clique1.add(ii)
                    else:
                        self.secret_log[self.secret_log_name] = logs

        if len(self.nonzeros) == self.params.hamming:
            self.secret_log[self.secret_log_name] = logs

        all_cliques = (
            [clique0, clique1]
            if other_cliques is None
            else [clique0, clique1, *other_cliques]
        )
        return clique0, clique1, False


class BinaryDistinguisher(BaseDistinguisher):
    def __init__(self, params, secret_check, secret_log):
        super().__init__(params, secret_check, secret_log)
        self.N = params.N
        self.device = params.device

    @torch.no_grad()
    def run(self, base_pred, modified_preds, dx):
        scores = self.compute_scores(base_pred, modified_preds, dx)
        sorted_i_score = sorted(enumerate(scores), key=lambda i: i[1], reverse=True)
        # Only need to flip at most h = N/2 bits to 1.
        sorted_i_score = sorted_i_score[: self.N // 2 + 1]

        guess = np.zeros(self.N)
        for i, score in sorted_i_score:
            # Set another bit in the guess and check if it's the secret
            guess[i] = 1
            if self.secret_check.match_secret(guess):
                return True

        return False


class SlopeDistinguisher(BaseDistinguisher):
    def __init__(self, params, secret_check, secret_log):
        super().__init__(params, secret_check, secret_log)

        self.N = params.N
        self.Q = params.Q
        self.distinguisher_size = params.distinguisher_size
        self.device = params.device

    def run(self, base_pred, modified_preds, dx):
        derivative_samples = self.compute_scores(base_pred, modified_preds, dx)

        guesses = np.zeros((4, self.N), dtype=int)
        for i, sample in enumerate(derivative_samples):
            guesses[0, i] = int(stats.mode(sample).mode.round())
            guesses[1, i] = int(stats.mode(sample.round()).mode)
            guesses[2, i] = int(np.mean(sample).round())
            guesses[3, i] = int(np.median(sample).round())

        return bool(
            np.any([self.secret_check.match_secret(guess) for guess in guesses])
        )

    def get_inputs(self, A):
        for i in range(self.N):
            Ai = A.clone()
            dxi = Ai[:, i].clone()
            dxi[dxi > self.Q // 2] -= self.Q
            dxi //= 2
            Ai[:, i] -= dxi
            yield Ai, dxi

    def compute_scores(self, fa, fai, dx):
        # Compute the slopes: (f(x1)-f(x0))/dx where the diff is modQ.
        derivatives = [
            self.mod_derivative(fai[i], fa, dx[i].to(self.device), self.Q)
            for i in range(self.N)
        ]
        return derivatives

    def mod_derivative(self, Y0, Y1, dx, modulus):
        mask = dx != 0
        Y0 = Y0[mask]
        Y1 = Y1[mask]
        dx = dx[mask]
        assert Y0.shape == Y1.shape
        diff = torch.abs(Y1 - Y0)
        diff = torch.minimum(diff, modulus - diff)
        dfdx = torch.sign(Y1 - Y0).to(float) * diff / dx
        return dfdx.cpu().numpy()
