""""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import datetime
import functools
import logging
import os
import re
import sys
import math
import time
import pickle
import random
import getpass
import argparse
import subprocess
import numpy as np
import errno
import signal
from functools import wraps, partial

from tqdm import tqdm


from .logger import create_logger


FALSY_STRINGS = {"off", "false", "0"}
TRUTHY_STRINGS = {"on", "true", "1"}

DUMP_PATH = "/checkpoint/%s/dumped" % getpass.getuser()
CUDA = True


def create_this_logger(params):
    return create_logger(
        os.path.join(params.dump_path, "train.log"),
        rank=getattr(params, "global_rank", 0),
    )


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")


def initialize_exp(params):
    """
    Initialize the experience:
    - dump parameters
    - create a logger
    """
    # dump parameters
    get_dump_path(params)
    try:
        pickle.dump(params, open(os.path.join(params.dump_path, "params.pkl"), "wb"))
    except Exception as e:
        print(e)

    # get running command
    command = ["python", sys.argv[0]]
    for x in sys.argv[1:]:
        if x.startswith("--"):
            assert '"' not in x and "'" not in x
            command.append(x)
        else:
            assert "'" not in x
            if re.match("^[a-zA-Z0-9_]+$", x):
                command.append("%s" % x)
            else:
                command.append("'%s'" % x)
    command = " ".join(command)
    params.command = command + ' --exp_id "%s"' % params.exp_id

    # check experiment name
    assert len(params.exp_name.strip()) > 0

    # create a logger
    logger = create_this_logger(params)
    logger.info("============ Initialized logger ============")
    logger.info(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(params)).items()))
    )
    logger.info("The experiment will be stored in %s\n" % params.dump_path)
    logger.info("Running command: %s" % command)
    logger.info("")
    return logger


def get_dump_path(params):
    """
    Create a directory to store the experiment.
    """
    params.dump_path = DUMP_PATH if params.dump_path == "" else params.dump_path
    assert len(params.exp_name) > 0

    # create the sweep path if it does not exist
    sweep_path = os.path.join(params.dump_path, params.exp_name)
    if not os.path.exists(sweep_path):
        subprocess.Popen("mkdir -p %s" % sweep_path, shell=True).wait()

    # create an ID for the job if it is not given in the parameters.
    # if we run on the cluster, the job ID is the one of Chronos.
    # otherwise, it is randomly generated
    if params.exp_id == "":
        chronos_job_id = os.environ.get("CHRONOS_JOB_ID")
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        assert chronos_job_id is None or slurm_job_id is None
        exp_id = chronos_job_id if chronos_job_id is not None else slurm_job_id
        if exp_id is None:
            chars = "abcdefghijklmnopqrstuvwxyz0123456789"
            while True:
                exp_id = "".join(random.choice(chars) for _ in range(10))
                if not os.path.isdir(os.path.join(sweep_path, exp_id)):
                    break
        else:
            assert exp_id.isdigit()
        params.exp_id = exp_id
        params.slurm_id = slurm_job_id

    # create the dump folder / update parameters
    params.dump_path = os.path.join(sweep_path, params.exp_id)
    params.ckpt_path = os.path.join(params.dump_path, "ckpt.json")
    if not os.path.isdir(params.dump_path):
        subprocess.Popen("mkdir -p %s" % params.dump_path, shell=True).wait()


def to_cuda(*args):
    """
    Move tensors to CUDA.
    """
    if not CUDA:
        return args
    return [None if x is None else x.cuda() for x in args]


class TimeoutError(BaseException):
    pass


def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(repeat_id, signum, frame):
            # Caught the signal ({repeat_id}) Setting signal handler {repeat_id + 1}
            signal.signal(signal.SIGALRM, partial(_handle_timeout, repeat_id + 1))
            signal.alarm(seconds)
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            old_signal = signal.signal(signal.SIGALRM, partial(_handle_timeout, 0))
            old_time_left = signal.alarm(seconds)
            assert type(old_time_left) is int and old_time_left >= 0
            if 0 < old_time_left < seconds:  # do not exceed previous timer
                signal.alarm(old_time_left)
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
            finally:
                if old_time_left == 0:
                    signal.alarm(0)
                else:
                    sub = time.time() - start_time
                    signal.signal(signal.SIGALRM, old_signal)
                    signal.alarm(max(0, math.ceil(old_time_left - sub)))
            return result

        return wraps(func)(wrapper)

    return decorator


def token2num(s, base):
    token = np.array([int(x) for x in s.split()])
    return np.array([base, 1]) @ np.transpose(token.reshape((-1, 2)))


class SimpleMovingAverage:
    def __init__(self, k):
        self.k = k
        self.reset()

    def step(self, value):
        self.history[self.i] = value
        self.i = (self.i + 1) % self.k

    def reset(self):
        self.history = [np.nan] * self.k
        self.i = 0

    @property
    def mean(self):
        return np.nanmean(self.history)


# def mod_mult(mat1, mat2, Q, longtype=False):
#     if not longtype:
#         return (mat1 @ mat2) % Q
#     mat1 = mat1.astype(np.longdouble)
#     return ((mat1 // 10000) @ (mat2 * 10000 % Q) + (mat1 % 10000) @ mat2) % Q

def mod_mult(mat1, mat2, Q):
    if np.log2(Q) <= 30:
        return np.tensordot(mat1, mat2, 1) % Q

    # Use 128-bit floats and scale the matrix slightly
    frac = 10_000
    mat1 = mat1.astype(np.float128)
    out = np.tensordot((mat1 // frac), (mat2 * frac % Q), 1) 
    out += np.tensordot((mat1 % frac), mat2, 1)
    out %= Q
    return out.astype(np.int64)


def mod_mult_torch(mat1, mat2, Q):
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mat1 = torch.from_numpy(mat1).double().to(device)
    mat2 = torch.from_numpy(mat2).double().to(device)
    if np.log2(Q) <= 30:
        out = torch.tensordot(mat1, mat2, dims=1) % Q
    
    else:
        # Use 64-bit floats and scale the matrix slightly
        frac = 10_000
        out = torch.tensordot((mat1 // frac), (mat2 * frac % Q), dims=1)
        out += torch.tensordot((mat1 % frac), mat2, dims=1)
        out %= Q
    return out.to(torch.int64).cpu().numpy()


def mod_diff(base_pred, mod_pred, Q):
    import torch
    assert base_pred.shape == mod_pred.shape
    diff = torch.abs(base_pred - mod_pred)
    diff = torch.minimum(diff, Q - diff)
    return diff.sum().cpu()


# Encode various things in this project to JSON
@functools.singledispatch
def to_json(obj):
    return str(obj)


@to_json.register
def _(obj: np.float16):
    return float(obj)


@to_json.register
def _(obj: np.ndarray):
    return obj.tolist()


# unit of time
hour = datetime.timedelta(hours=1)


def load_params(params):
    # Normalize paths
    params.data_path = os.path.realpath(params.data_path)
    params.dump_path = os.path.realpath(params.dump_path)

    if os.path.exists(os.path.join(params.data_path, "params.pkl")):
        with open(os.path.join(params.data_path, "params.pkl"), "rb") as fd:
            data = pickle.load(fd)
    elif os.path.exists(os.path.join(params.data_path, "secrets", "params.pkl")):
        # It's down in the 'secrets' folder
        with open(os.path.join(params.data_path, "secrets", "params.pkl"), "rb") as fd:
            data = pickle.load(fd)
    else:
        with open(os.path.join(params.data_path, "../", "params.pkl"), "rb") as fd:
            data = pickle.load(fd)

    if not isinstance(data, dict):
        data = vars(data)

    params.N = data["N"]
    params.Q = data["Q"]
    params.sigma = data["sigma"]
    params.gamma = data["gamma"]
    params.secret_type = data["secret_type"]

    return params

def init_rng(seed, logger):
    rng = np.random.default_rng(seed)
    logger.info("Initialized rng. [seed: %s].", seed)
    return rng


def human(i):
    # Makes a number human readable by appending B, M, or K to it.
    for limit, suffix in [(1e12, "G"), (1e9, "B"), (1e6, "M"), (1e3, "K")]:
        if i > limit:
            return f"{i / limit:.1f}{suffix}"

    if i < 1:
        return f"{i:.2g}"

    return f"{i:.1f}"


def shuffled(lst, rng):
    lst2 = lst[:]
    rng.shuffle(lst2)
    return lst2


def shift_negate(tnsor):
    """Shift to the left and negate the wrapped element"""
    return np.concatenate((tnsor[..., 1:], -tnsor[..., :1]), axis=-1)


def read(data_prefix_path, m):
    with open(data_prefix_path) as fd:
        A, RT = [], []
        for i, line in enumerate(tqdm(fd)):
            if not line:
                breakpoint()
                continue

            a, r = line.strip().split(";")
            A.append(np.array(a.split(), dtype=np.int64))
            RT.append(np.array(r.split(), dtype=np.int64))

            if len(A) == m:
                # assert (i + 1) % m == 0
                yield np.array(A), np.array(RT).T
                A, RT = [], []


def remove_redundant_rows(path, m):
    # Create a dictionary to store unique A vectors and their corresponding R matrices
    data_dict = {}
    for a, R in read(path, m):
        a = np.squeeze(a)
        assert a.ndim == 1, f"a of shape {a.shape} should be a 1d array of indices!"
        a = tuple(a)
        if a in data_dict:
            data_dict[a].append(R)
        else:
            data_dict[a] = [R]

    # Remove redundant rows in R for each A
    for a, rs in data_dict.items():
        r = np.vstack(rs)
        _, index = np.unique(r, axis=0, return_index=True)
        data_dict[a] = r[index]
    # return the unique A vectors and their corresponding R matrices to the outfile
    lines = []
    for a, r in data_dict.items():
        for k in range(len(a)):
            r_str = " ".join(r.T[k].astype(str))
            lines.append(f"{a[k]};{r_str}\n")
    return lines

