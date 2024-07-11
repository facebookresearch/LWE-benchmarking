""""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import polars as pl
import pickle
import re
import torch

rng = np.random.default_rng()

# INFO - 05/15/24 16:58:54 - 19:21:59 {}
pattern = re.compile(r"(INFO) - .* - (\{.*\})")


def parse_recover_log(logpath, *, verbose=False):
    recover_rows = []
    train_rows = []

    with open(logpath, "rb") as fd:
        for line in fd:
            line = line.decode("utf8", errors="ignore")

            match = pattern.match(line)
            if not match:
                continue

            row = json.loads("{" + line.split("{")[1])
            if "recover/epoch" in row:
                recover_rows.append(
                    {key.removeprefix("recover/"): value for key, value in row.items()}
                )
            elif "train/epoch" in row:
                train_rows.append(
                    {key.removeprefix("train/"): value for key, value in row.items()}
                )
            else:
                print(row)

    epochs = [row["epoch"] for row in recover_rows]
    start = 0
    for i, (ep, next_ep) in enumerate(zip(epochs, epochs[1:])):
        if next_ep not in (ep, ep + 1):
            start = i + 1

    recover_rows = recover_rows[start:]

    if verbose:
        print(len(recover_rows), len(train_rows))

    return pl.DataFrame(recover_rows), pl.DataFrame(train_rows)


def plot_train_loss(config, train_df):
    loss_type = "MSE" if config["angular_emb"] else "Cross Entropy"

    fig, ax = plt.subplots()
    ax.plot(
        train_df.select("loss"),
        label=f"Train ({loss_type})",
        linestyle="",
        marker=".",
        ms=1,
    )
    ax.set_xlabel("Batch (x100)")
    ax.set_ylabel(f"{loss_type} Loss")

    run_id = config.get("exp_id") or config.get("run_id")
    ax.set_title(f"Exp {run_id} Train Loss")
    title = f"{run_id}-train-loss.png"
    return ax, fig, title


def plot_epoch_loss(config, recover_df, train_df):
    loss_type = "MSE" if config["angular_emb"] else "Cross Entropy"

    fig, ax = plt.subplots()
    ax.plot(
        train_df.group_by("epoch")
        .agg(pl.col("loss").mean())
        .sort("epoch")
        .select("loss"),
        label="Train",
        linestyle=":",
        marker=".",
        color="tab:blue",
    )
    ax.plot(
        recover_df.group_by("epoch").agg(pl.col("loss").mean()).select("loss"),
        label="Validation",
        linestyle=":",
        marker=".",
        color="tab:red",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"{loss_type} Loss")
    ax.legend()
    run_id = config.get("exp_id") or config.get("run_id")
    ax.set_title(f"Exp {run_id} Train vs Val Loss")
    title = f"{run_id}-epoch-loss.png"
    return ax, fig, title


def plot_secret_success(config):
    print(config)
    secret = config["secret"]
    n = config["N"]
    h = config["hamming"]
    run_dir = config.get("dump_path")

    true = set(np.nonzero(secret)[0])

    logs = []

    for filename in os.listdir(run_dir):
        if not filename.endswith(".pkl") or "params" in filename:
            continue
        path = os.path.join(run_dir, filename)
        with open(path, "rb") as fd:
            logs.append(pickle.load(fd))

    logs = sorted(logs, key=lambda d: d["epoch"])

    fig, ax = plt.subplots()

    xs, ys = [], []
    for log in logs:
        func = next(key for key in log if key not in ("success", "epoch"))
        preds = set(np.argsort(log[func])[-h:])
        correct = len(preds & true)
        xs.append(log["epoch"])
        ys.append(correct)

    ax.plot(xs, ys, label="Model", color="tab:blue")

    # Add a baseline of random bit guessing
    sample = []
    for _ in range(1000):
        guess = set(rng.choice(n, h, replace=False))
        correct = len(guess & true)
        sample.append(correct)
    mean, std = np.mean(sample), np.std(sample)
    ax.plot((0, max(xs)), (mean, mean), label="Guessing", color="tab:red")
    ax.plot((0, max(xs)), (mean + std, mean + std), color="tab:red", linestyle=":")
    ax.plot((0, max(xs)), (mean - std, mean - std), color="tab:red", linestyle=":")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Correct Bits")
    run_id = config.get("exp_id") or config.get("run_id")
    ax.set_title(f"Exp {run_id} Secret Success")
    fig.legend()
    title = f"{run_id}-secret-success.png"
    return ax, fig, title


def main(params):
    ckpt = torch.load(params.ckpt, map_location="cpu")

    logpath = os.path.join(ckpt["params"]["dump_path"], "train.log")
    recover_df, train_df = parse_recover_log(logpath, verbose=False)
    config = ckpt["params"]
    ax, fig, filename = plot_epoch_loss(config, recover_df, train_df)
    fig.savefig(filename)

    ax, fig, filename = plot_secret_success(config)
    fig.savefig(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to ckpt.json file")
    params = parser.parse_args()

    # plot and save plots
    main(params)
