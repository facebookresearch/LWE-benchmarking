""""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from logging import getLogger
import torch
import tqdm
import numpy as np
import math
import time
import itertools
from sklearn.linear_model import LinearRegression
import sys

logger = getLogger()


def hamming_distance(a, b):
    return (a != b).sum()


def center(x, q):
    return (x + q // 2) % q - q // 2


def brute_force_one_batch(
    secret_combs: torch.Tensor,
    RAs: torch.Tensor,
    RBs: torch.Tensor,
    top_n,
    Q,
    brute_force_dim,
    possible_values=(1,),  # (-1, 1) for ternary
):
    keep_n_tops = top_n[0].shape[0]
    hw = secret_combs.shape[1]
    n_multipliers = len(possible_values) ** hw
    batch_size = secret_combs.shape[0]
    total_batch_size = batch_size * n_multipliers
    secret_cands = torch.zeros(
        total_batch_size, brute_force_dim, device=RAs.device, dtype=torch.float16
    )
    # secret_combs tell us where non-zero entries are.
    # multipliers tell us what the possible entries are, (-1, 1) for ternary and so on
    multipliers = torch.tensor(
        list(itertools.product(possible_values, repeat=hw)),
        device=RAs.device,
        dtype=torch.float16,
    )

    # expand them to use for scatter
    multipliers = multipliers.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, hw)
    secret_combs = (
        secret_combs.unsqueeze(1).expand(-1, n_multipliers, -1).reshape(-1, hw)
    )

    secret_cands.scatter_(1, secret_combs, multipliers)
    # logger.info(torch.cuda.memory_summary(abbreviated=False))
    dot = secret_cands @ RAs.T
    dot -= RBs
    dot %= Q
    test_stat = dot.std(1)
    if len(test_stat) > keep_n_tops:
        biggest_stds = test_stat.topk(keep_n_tops)
    else:
        biggest_stds = test_stat.topk(len(test_stat))
    top_2n = torch.cat([top_n[0], biggest_stds.values])
    top_2n_secrets = torch.cat([top_n[1], secret_cands[biggest_stds.indices]])
    new_topn = top_2n.topk(keep_n_tops)
    top_n = [new_topn.values, new_topn.indices]
    top_n[1] = top_2n_secrets[top_n[1]]

    return top_n


class Annealer:
    def __init__(
        self,
        RA,
        Rb,
        secret_cand,
        total_hw,
        Q,
        brute_force_dim,
        max_steps=None,
        accept_scaling=1.0,
    ):
        # cooling: go down from 1 -> 1e-2 exponentially
        self.cooling = (1e-2) ** (1.0 / max_steps)
        self.device = RA.device
        self.q = Q
        self.RA = RA
        self.RB = Rb
        self.dim = secret_cand.shape[0]
        self.bf_dim = brute_force_dim
        self.bf_hw = int(secret_cand[: self.bf_dim].sum().item())
        self.other_hw = total_hw - self.bf_hw
        self.hamming_weight = total_hw
        self.secret_cand = secret_cand
        # assume i know the target hamming weight
        self.secret_cand[self.bf_dim : self.bf_dim + self.other_hw] = 1
        self.secret_cand[self.bf_dim + self.other_hw :] = 0
        self.temp = 1.0
        self.loss = 1e9
        self.accept_scaling = accept_scaling
        self.step_up_prob = 0
        self.best_secret = self.secret_cand
        self.best_loss = 1e9

    def generate_new_secret_with_hw(self):
        n_changes = 1
        # now we move `n_changes` bits to something new
        # first, we choose which bits to change
        secret_cand_new = self.secret_cand[self.bf_dim :].clone()
        # we choose `n_changes` bits to change
        # random choice from the currently active bits
        make_this_0 = self.secret_cand[self.bf_dim :].multinomial(n_changes)
        # take another one from the inactive bits
        make_this_1 = (1 - self.secret_cand[self.bf_dim :]).multinomial(n_changes)
        secret_cand_new[make_this_0] = 0
        secret_cand_new[make_this_1] = 1
        return torch.cat([self.secret_cand[: self.bf_dim], secret_cand_new])

    def accept_step(self, loss_new):
        # self.temp goes from 1 -> 0.
        # at self.temp = 0, step_up_prob should be 0
        # in between it should be dependent on the loss difference
        loss_diff = self.loss - loss_new
        if loss_diff > 0:
            return True
        else:
            step_up_prob = torch.exp(loss_diff / (self.temp * self.accept_scaling))
            self.step_up_prob = step_up_prob
            return torch.rand(1, device=self.device) < step_up_prob

    def step(self):
        # 1. generate new binary secret (hamming dist depends on temp)
        secret_new = self.generate_new_secret_with_hw()
        # 2. compute loss
        loss_new = -((self.RA @ secret_new - self.RB) % self.q).std()
        # accept with probability
        if self.accept_step(loss_new):
            self.loss = loss_new
            self.secret_cand = secret_new
        self.temp *= self.cooling  # cooling

        if self.loss < self.best_loss:
            self.best_loss = self.loss
            self.best_secret = self.secret_cand


class Attacker:
    MAX_GPU_MEM = 16 * 1024**3

    def __init__(
        self,
        data,
        brute_force_dim,
        n_data_for_brute_force,
        n_data_for_greedy,
        keep_n_tops=100,
        check_every_n_batches=10000,
        batch_size=5000,
        secret_type="binary",
        use_tqdm=True,
        compile_bf=True,
        mlwe_k=False,
        secret_window=0,
    ):
        RAs = data.RA
        RBs = data.RB
        self.origA = data.origA
        self.origB = data.origB
        self.origQ = data.params.Q
        self.sigma = data.params.sigma
      
        self.Q = 10
        RAs = RAs / self.origQ * self.Q
        RBs = RBs / self.origQ * self.Q
   

        self.RAs = torch.tensor(RAs, dtype=torch.float16)
        self.RBs = torch.tensor(RBs, dtype=torch.float16)

        selection_for_bf = torch.randperm(len(RAs))[:n_data_for_brute_force]
        selection_for_G = torch.randperm(len(RAs))[:n_data_for_greedy]

        window_start = 0
        self.secret_dim = data.params.N

        if mlwe_k:
            window_start = secret_window
            logger.info(
                f"Window applied {window_start}: {(window_start+brute_force_dim//mlwe_k) %(self.secret_dim//mlwe_k)} on each component {mlwe_k}"
            )

        self.dim_selection_for_bf, self.reduced_dims = self.get_partitions(
            self.secret_dim, mlwe_k, brute_force_dim, window_start
        )

        self.RAs_BF = torch.tensor(
            RAs[selection_for_bf][:, self.dim_selection_for_bf], dtype=torch.float16
        )
        self.RBs_BF = torch.tensor(RBs[selection_for_bf], dtype=torch.float16)

        self.RAs_G = torch.tensor(RAs[selection_for_G], dtype=torch.float16)
        self.RBs_G = torch.tensor(RBs[selection_for_G], dtype=torch.float16)

        self.brute_force_dim = brute_force_dim
    
        self.keep_n_tops = keep_n_tops
        self.check_every_n_batches = check_every_n_batches
        self.batch_size = batch_size
        self.use_tqdm = use_tqdm
        self.compile_bf = compile_bf
        self.secret_type = secret_type

    def get_partitions(self, N, k, u, secret_window_start):
        if k == 0:
            # LWE case
            bf_selection = (secret_window_start + np.arange(u, dtype=int)) % N
            comp_selection = (secret_window_start + np.arange(u, N, dtype=int)) % N
            return bf_selection, comp_selection
        uk = u // k
        n = N // k
        indices = np.tile(secret_window_start + np.arange(uk, dtype=int), (k, 1)) % n
        indices = indices + n * np.arange(k, dtype=int)[:, np.newaxis]
        dim_selection_for_bf = indices.flatten()

        neg_mask = np.ones(n * k, dtype=bool)
        neg_mask[dim_selection_for_bf] = False
        complementary_partition = np.arange(n * k)[neg_mask]
        return dim_selection_for_bf, complementary_partition

    def secret_found(self, cand):
        cand = cand.cpu().numpy()
        err_pred = (self.origA @ cand - self.origB) % self.origQ
        err_pred[err_pred > self.origQ // 2] -= self.origQ
        return (np.std(err_pred) < 2 * self.sigma).item()
    
    def check_partial_candidates(
        self, cands, RAs_G, RBs_G, which="linear", possible_values=[1, 2, 3]
    ):
        for cand in cands:
            if which == "greedy":
                full_cand = self.greedy_secret_completion(cand, RAs_G, RBs_G)
            elif which == "linear":
                full_cand = self.linear_secret_completion(
                    cand, RAs_G, RBs_G, possible_values
                )
            else:
                raise ValueError(f'unknown method for secret completion "{which}"')
            found = self.secret_found(full_cand)
            
            if found:
                logger.info(f"SUCCESS! secret non zeros {full_cand.cpu().numpy().nonzero()}")
                return True
           
        return False

    @torch.inference_mode()
    def linear_secret_completion(self, cand, RAs_G, RBs_G, possible_values):
        model = LinearRegression()
        partial_cand = torch.zeros(
            size=(self.secret_dim,), dtype=RAs_G.dtype, device=cand.device
        )
        partial_cand[self.dim_selection_for_bf] = cand
        y_ = (
            center((RBs_G - RAs_G @ partial_cand) % self.Q, self.Q)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        x_ = (
            center(RAs_G[:, self.reduced_dims], self.Q).cpu().numpy().astype(np.float32)
        )
        sigma = x_.flatten().std()
        x = x_ / sigma
        y = y_ / sigma
        model.fit(x, y)

        coefs = model.coef_
        max_coef = np.abs(model.coef_).max()

        if max_coef == 0:
            return partial_cand

        coefs /= max_coef

        stds = []
        candidates = []
        scales = list(set(np.abs(possible_values)))

        coefs = torch.from_numpy(coefs).to(cand.dtype).to(cand.device)
        for scale in scales:
            full_cand = partial_cand.clone()
            full_cand[self.reduced_dims] = (coefs * scale).round()
            candidates.append(full_cand)
            stds.append(
                center((RBs_G - RAs_G @ full_cand.to(RAs_G.device)) % self.Q, self.Q)
                .to(torch.float32)
                .std()
                .cpu()
                .numpy()
            )

        return candidates[np.argmin(stds)].to(int)

    @torch.inference_mode()
    def greedy_secret_completion(self, secret_cand, RAs_G, RBs_G):
        secret = torch.zeros(self.secret_dim, dtype=torch.float16)
        secret[self.dim_selection_for_bf] = secret_cand.cpu()
        secret_cand = secret.to(secret_cand.device)

        current_std = ((RAs_G @ secret_cand - RBs_G) % self.Q).std()
        for current_idx in self.reduced_dims:
            secret_cand[current_idx] = 1
            new_std = ((RAs_G @ secret_cand - RBs_G) % self.Q).std()
            if new_std > current_std:
                current_std = new_std
            else:
                secret_cand[current_idx] = 0
        return secret_cand

    @staticmethod
    def _generate_in_batches(generator, batch_size):
        while True:
            batch = tuple(itertools.islice(generator, batch_size))
            if not batch:
                break
            yield batch

    def generate_from_to_in_batches(self, n, k, start, end, batch_size):
        combinations = itertools.islice(itertools.combinations(range(n), k), start, end)
        yield from self._generate_in_batches(combinations, batch_size)

    def num_secrets_with_hw(self, hw):
        return math.comb(self.brute_force_dim, hw)

    def calculate_idxs_for_each_hw(self, min_HW, max_HW, start_idx, stop_idx):
        """
        calculate which indices between start_idx and stop_idx span which HW
        """
        if stop_idx == -1:
            stop_idx = sum(
                self.num_secrets_with_hw(hw) for hw in range(min_HW, max_HW + 1)
            )
        hw_idxs = {}
        n_total_combs = 0
        for hw in range(min_HW, max_HW + 1):
            n_secrets_with_hw = self.num_secrets_with_hw(hw)
            if start_idx < n_total_combs + n_secrets_with_hw:
                if stop_idx <= n_total_combs + n_secrets_with_hw:
                    # all in this HW, start from 0
                    hw_idxs[hw] = (start_idx - n_total_combs, stop_idx - n_total_combs)
                    break
                else:
                    # start from 0, end at stop_idx - n_total_combs
                    hw_idxs[hw] = (start_idx - n_total_combs, n_secrets_with_hw)
                    start_idx = n_total_combs + n_secrets_with_hw
            n_total_combs += n_secrets_with_hw

        return hw_idxs

    @torch.no_grad()
    @torch.inference_mode()
    def brute_force_worker(
        self,
        min_HW,
        max_HW,
        start_idx,
        stop_idx,
        device="cpu",
    ):
        hw_idxs = self.calculate_idxs_for_each_hw(min_HW, max_HW, start_idx, stop_idx)
        logger.info(hw_idxs)

        top_n = [
            torch.zeros(self.keep_n_tops, device=device, dtype=torch.float16),
            torch.zeros(
                (self.keep_n_tops, self.brute_force_dim),
                device=device,
                dtype=torch.float16,
            ),
        ]

        RAs = self.RAs_BF.to(device)
        RBs = self.RBs_BF.to(device)

        RAs_G = self.RAs_G.to(device)
        RBs_G = self.RBs_G.to(device)

        if self.secret_type == "binary":
            possible_values = (1,)
        elif self.secret_type == "ternary":
            possible_values = (-1, 1)
        elif self.secret_type == "binomial":
            possible_values = (-2, -1, 1, 2)
        elif self.secret_type == "gaussian":
            possible_values = (-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6)

        if self.compile_bf:
            logger.info("compiling brute force function")
            brute_force_fn = torch.compile(brute_force_one_batch)
        else:
            brute_force_fn = brute_force_one_batch

        for hamming_weight, (start, stop) in hw_idxs.items():
            logger.info(
                f"hamming weight: {hamming_weight}, start: {start}, stop: {stop}"
            )
            optimal_batch_size = self.get_batch_size(
                self.batch_size, hamming_weight, len(possible_values), len(RBs)
            )
            generator = self.generate_from_to_in_batches(
                self.brute_force_dim, hamming_weight, start, stop, optimal_batch_size
            )
            length = math.ceil((stop - start) / optimal_batch_size)

            if self.use_tqdm:
                bar = tqdm.tqdm(generator, mininterval=1, total=length)
            else:
                bar = generator

            batch_counter = 0
            for secret_combs in bar:
                batch_counter += 1
                secret_combs = torch.tensor(
                    secret_combs, device=device, dtype=torch.int64
                )
                top_n = brute_force_fn(
                    secret_combs,
                    RAs,
                    RBs,
                    top_n,
                    self.Q,
                    self.brute_force_dim,
                    possible_values=possible_values,
                )
                if (
                    batch_counter > 0
                    and batch_counter % self.check_every_n_batches == 0
                ):
                    if self.check_partial_candidates(
                        top_n[1], RAs_G, RBs_G, possible_values=possible_values
                    ):
                        return True
                    top_n = [
                        torch.zeros(
                            self.keep_n_tops, device=device, dtype=torch.float16
                        ),
                        torch.zeros(
                            (self.keep_n_tops, self.brute_force_dim),
                            device=device,
                            dtype=torch.float16,
                        ),
                    ]

            logger.info(
                f"finalizing HW {hamming_weight}, last check here, ran through {batch_counter} batches"
            )
            if self.check_partial_candidates(
                top_n[1], RAs_G, RBs_G, possible_values=possible_values
            ):
                return True
        logger.info("done, secret not found")
        return False

    def get_batch_size(self, batch_size, hw, possible_values, len_data):
        """Get max batch size given the bottleneck (secret_cands @ RAs.T) of
        size (len_data x batch_size*possible_values^hw)"""
        fp16_size = 2
        max_batch_size = self.MAX_GPU_MEM / (
            fp16_size * possible_values**hw * len_data
        )
        max_batch_size = int(max_batch_size)
        logger.info(f"Max batch size: {max_batch_size}, batch size: {batch_size}")
        return min(batch_size, max_batch_size)
