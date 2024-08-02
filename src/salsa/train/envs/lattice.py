""""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from logging import getLogger
import numpy as np
import torch
import math


SPECIAL_WORDS = ["<eos>", "<pad>", "<mask>"]
logger = getLogger()


class DigitEncoder(object):
    """
    Tokenizes and de-tokenizes inputs and outputs
    """

    def __init__(self, params):
        self.int_base = params.base
        self.bucket_size = params.bucket_size
        self.N = params.N
        self.Q = params.Q

        self.int_len = math.ceil(math.log(self.Q, self.int_base))

        self.symbols = [
            str(i) for i in range(math.ceil(self.int_base / self.bucket_size))
        ]

        self.words = self.symbols  # SPECIAL_WORDS +

        params.vocab_size = len(self.words)

        self.id2word = {i: s for i, s in enumerate(self.words)}
        self.word2id = {s: i for i, s in self.id2word.items()}
        assert len(self.words) == len(set(self.words))

        # number of words / indices
        self.n_words = params.n_words = len(self.words)

        self.eos_index = params.eos_index = 0
        self.pad_index = params.pad_index = 1
        logger.info(f"vocabulary: {len(self.word2id)} words")
        if len(self.word2id) < 1000:
            logger.info(f"words: {self.word2id}")

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        return torch.LongTensor(np.apply_along_axis(self.encode, axis=-1, arr=x))

    def encode(self, row):
        digits = self.encode_base(row)
        ids = [self.word2id[d] for d in digits]
        return ids

    def encode_base(self, vector):
        lst = []
        for val in vector:
            res = [
                (val // self.int_base**i) % self.int_base
                for i in range(self.int_len - 1, -1, -1)
            ]
            lst += [str(res[0])] + [str(digit // self.bucket_size) for digit in res[1:]]
        return lst

    def decode(self, logits):
        device = logits.device
        ids = logits.max(dim=1)[1].cpu().numpy()
        assert ids.ndim == 2

        words = [[self.id2word[_id] for _id in seq] for seq in ids]

        b = (
            torch.LongTensor([self.decode_base(seq) for seq in words])
            .squeeze()
            .to(device)
        )
        return b

    def decode_base(self, lst):
        dim = len(lst) // self.int_len

        m = [0 for _ in range(dim)]
        for idx in range(dim):  # For each number in the sequence
            for bit in range(self.int_len):  # From high bit to low bit
                digit = lst[idx * self.int_len + bit]
                if not (digit.isdigit() or digit[0] == "-" and digit[1:].isdigit()):
                    logger.warning("Non digit tokens are not handled!")
                    continue
                m[idx] = m[idx] * self.int_base + int(digit)
        return m


class AngularEncoder(object):
    def __init__(self, params):
        self.params = params
        self.Q = params.Q

    def __call__(self, x):
        return self.encode(x)

    def encode(self, x):
        rad = x.squeeze() / self.Q * 2 * torch.pi  # convert to radians
        return torch.stack((torch.cos(rad), torch.sin(rad)), dim=-1)

    def decode(self, x):
        _, cols = x.shape
        assert cols == 2, f"xy should have 2 columns, not {cols}"
        x = x.to(torch.float64)
        # atan2 expects the y-coordinate first. atan2 also returns the range
        # (-pi, pi) so we add 2pi then mod by 2pi to the range (0, 2pi).
        angles = (torch.atan2(x[..., 1], x[..., 0]) + 2 * torch.pi) % (2 * torch.pi)

        return (angles / (2 * np.pi) * self.Q).round().to(torch.int64)
