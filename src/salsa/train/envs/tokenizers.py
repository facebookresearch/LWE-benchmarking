""""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import logging

import torch
import numpy as np

logger = logging.getLogger("tokenizer")


class Tokenizer:
    # Unknown secret. Can be fine-tuned.
    unk_secret = "<|unk-secret|>"

    def __init__(self, *, base, q, bucket_size, unique_secrets, matrix_emb=False):
        assert base >= 2, f"base must be 2 or more, not {base}"

        self.base = base
        self.q = q
        self.bucket_size = bucket_size

        # Param controlling whether data is of form
        #    (a1_tok1, a1_tok2, a2_tok1, a2_tok2, ...)
        # OR
        #    ((a1_tok1, a2_tok1, ...), (a1_tok2, a2_tok2, ...))
        self.matrix_emb = matrix_emb

        # Number of digits (normally 1 or 2)
        self.n_digits = int(np.ceil(np.emath.logn(self.base, self.q)))

        # Number of different digit symbols (normally 1K-50K)
        if self.base > self.q:
            num_digit_symbols = q // bucket_size + 1
        else:
            num_digit_symbols = max(q // base, base // bucket_size) + 1

        logger.info(
            "Initializing vocab. [digits: %d, symbols: %d]",
            self.n_digits,
            num_digit_symbols,
        )

        # This way all digits are their own symbol
        digit_symbols = list(range(num_digit_symbols))

        secret_symbols = [f"<|secret-{i}|>" for i in range(unique_secrets)]
        secret_symbols.append(self.unk_secret)

        self.secret2id = {symbol: i for i, symbol in enumerate(secret_symbols)}
        self.digit2id = {symbol: i for i, symbol in enumerate(digit_symbols)}

    def encode_z(self, batch):
        b, n = batch.shape

        if self.base <= self.q:
            highs = batch // self.base
            lows = (batch % self.base) // self.bucket_size
            toks = torch.stack((highs, lows), dim=-1)
        else:
            toks = batch // self.bucket_size

        if self.matrix_emb:
            return toks
        else:
            return toks.view(b, n * self.n_digits)

    def encode_secret_keys(self, keys):
        symbols = [f"<|secret-{key}|>" for key in keys]
        symbols = [sym if sym in self.secret2id else self.unk_secret for sym in symbols]
        toks = [self.secret2id[sym] for sym in symbols]
        return torch.tensor(toks)

    def __repr__(self):
        return f"<Tokenizer ({len(self.secret2id)} secrets)>"

    def __len__(self):
        return len(self.digit2id) + len(self.secret2id)
