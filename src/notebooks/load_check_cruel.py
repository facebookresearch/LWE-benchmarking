""""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import pickle
import numpy as np
import os
import pandas as pd


def centered(arr, Q):
    arr = arr.copy()
    arr[arr > Q // 2] -= Q
    return arr


def read_data(p, max_samples=20000):
    """
    Read the data stored in data.prefix (format: A, R.T).
    Input: p: str, the directory that contains data.prefix
    Output: paramsDict, dataList
    """
    localparams = pickle.load(open(os.path.join(p, "params.pkl"), "rb"))
    if type(localparams) is not dict:
        localparams = localparams.__dict__
    newparams = {
        "N": localparams["N"],
        "Q": localparams["Q"],
        "delta": localparams["lll_delta"],
        "pen": localparams["lll_penalty"],
        "block_size": localparams["bkz_block_size"],
    }

    m = localparams["N"]
    if "m" in localparams and localparams["m"] != -1:
        m = localparams["m"]
    print(p)
    print(str(newparams))
    print("m: ", m)
    tiny = None
    if "reload_data" in localparams and localparams["reload_data"] != "":
        tiny = np.load(localparams["reload_data"])

    N, Q, rows = localparams["N"], localparams["Q"], [[], []]
    longtype = np.log2(Q) > 30
    try:
        f = open(os.path.join(p, "data.prefix"), "r")
    except:
        f = open(os.path.join(p, "data_0.prefix"), "r")
    AA, RR = [], []
    count, all_count = 0, 0
    while True:
        row = f.readline()
        if not row:
            break
        a, r = row.split(" ; ")
        AA.append(a.split())
        RR.append(r.split())
        if len(AA) == m:
            A, R = (np.array(AA).astype(int), np.array(RR).astype(int).T)
            if tiny is not None:
                A = tiny[A.flatten()]
            nonzero_ind = [
                i
                for i in range(len(R))
                if np.any(R[i] != 0) and np.linalg.norm(R[i]) < Q
            ]
            R = R[nonzero_ind, :]
            count += len(nonzero_ind)
            all_count += N + m
            if longtype:
                R = R.astype(np.longdouble)
                RA = ((R // 10000) @ (A * 10000 % Q) + (R % 10000) @ A) % Q
            else:
                RA = (R @ A) % Q
            RA[RA > Q // 2] -= Q
            for k in range(len(RA)):
                rows[0].append(RA[k])
                rows[1].append(R[k])
                if len(rows[0]) == 20000:
                    print("Yield rate:", count / all_count)
                    return newparams, rows
            AA, RR = [], []
            if count > 500000:
                print("Yield rate:", count / all_count)
                return newparams, rows
            if count > max_samples:
                break
    print("Yield rate:", count / all_count)
    return newparams, rows, tiny


def print_RA_stats(all_dirs, max_samples=20000):
    df = []
    for p in all_dirs:
        params, data, origA = read_data(p, max_samples)
        RA, R = np.array(data[0]), np.array(data[1])
        std_ratio = np.mean(np.std(RA, axis=1)) / np.sqrt((params["Q"] ** 2 - 1) / 12)
        normOfUniform = np.mean(
            np.linalg.norm(
                np.random.randint(
                    -params["Q"] // 2, params["Q"] // 2, size=(10000, params["N"])
                ),
                axis=1,
            )
        )
        norm_ratio = np.mean(np.linalg.norm(RA, axis=1)) / normOfUniform
        orig_std = centered(origA, params["Q"]).std()
        # bits above this threshold are cruel.
        cruel_threshold = 0.5 * orig_std
        bitwise_std = centered(RA, params["Q"]).std(axis=0)
        n_cruel_bits = (bitwise_std > cruel_threshold).sum()
        print(f"Cruel bits: {n_cruel_bits}")
        print("LLL penalty:", params["pen"])
        print("Stddev ratio:", std_ratio)
        print("Norm ratio:", norm_ratio)
        print("R norm:", np.mean(np.linalg.norm(R, axis=1)) / params["Q"])
        print("\n")
        df.append(list(params.values()) + [std_ratio, norm_ratio])
    param_cols = ["N", "Q", "delta", "pen", "blk_sz"]
    df = pd.DataFrame(
        df, columns=param_cols + ["standard deviation reduction", "norm reduction"]
    )
    return df


dirs = [
    "/path/to/data.prefix/"
]

df = print_RA_stats(dirs, max_samples=3000)
print(df)
