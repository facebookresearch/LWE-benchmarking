""""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import sys
import os
import numpy as np
import pickle as pkl
from sage.all import GF, matrix
from scipy.linalg import circulant

def rlwe_circ(a, n):
    A = circulant(a)
    tri = np.triu_indices(n, 1)
    A[tri] *= -1
    return A


def get_mlwe_circ(X, n, k):
    if np.ndim(X) == 1:
        X = X.reshape((k,n))
    
    A = np.zeros(shape=(n,k*n), dtype=np.int64)
    for i in range(k):
        a = X[i]
        A[:,n*i:n*(i+1)] = rlwe_circ(a, n)
    return A

def main(path, thread):
    # Load in params (they are saved)
    exp_params = pkl.load(open(os.path.join(path, 'params.pkl'), 'rb'))
    # Load in matrix
    reduced_mat = np.load(os.path.join(path, f"matrix_{thread}.npy"))

    # Pull out reduced dual matrix and indices of tinyA from which it was constructed.
    num_tinyA_per_lattice = exp_params.N if exp_params.mlwe_k == 0 else exp_params.mlwe_k
    idx = reduced_mat[0,:num_tinyA_per_lattice]
    qaryA = reduced_mat[:,exp_params.m:]
    # Load and remake original tinyA so we can compute lattice volume. 
    K = GF(exp_params.Q, proof=False)
    origA = np.load(os.path.join(exp_params.dump_path, f"Avecs_{thread}.npy"))
    tinyA = origA[idx[0:num_tinyA_per_lattice]]
    if exp_params.mlwe_k > 0:
        a = []
        for _a in tinyA:
            a.append([get_mlwe_circ(_a, exp_params.N // exp_params.mlwe_k, exp_params.mlwe_k)])
        tinyA = (np.squeeze(np.hstack(a)) % exp_params.Q)
    A1_cols = (exp_params.N-exp_params.k)
    A1 = tinyA[(exp_params.m-A1_cols):exp_params.m,:A1_cols].copy() # This is original matrix we reduced.
    # Now, get volume and expected delta. 
    A = matrix(K, A1)
    volL = int(A.determinant())

    # This is required length of short vector to apply to A2. 
    m = tinyA.shape[0] # From Cheon code
    shortvec = qaryA[0, -m:] 

    # Now get delta, B, tau
    delta = (np.linalg.norm(shortvec) / (volL**(1/m))) ** (1/m) # Inverting formula from pg 5 of Cheon. 

    c = 10 # Circular param definition: scale in their implementation is calculated using m, but optimal equation from end of paper requires m to compute c. 
    # We choose roughly 10 for now, since this is roughly their scale. Equation from Cheon implementation is scale = (alpha * q * sqrt(m) / sqrt(2 * pi * h)).round() 
    m = np.sqrt((exp_params.N * np.log2(exp_params.Q)/c)/(np.log2(delta))) 

    # Compute B
    alpha = 8 / exp_params.Q # Alpha is sigma / q, we always use sigma = 3.2.
    B = (2 + (1/np.sqrt(2*np.pi))) * alpha * exp_params.Q 
    B = B * np.sqrt(m/(m+exp_params.N)) 
    B = B * 2**(2*np.sqrt(exp_params.N * np.log2(delta) * np.log2(exp_params.Q) / c))

    # Compute B, delta
    mitm_params = np.array([B, delta])
    np.save(os.path.join(path, f"current_mitm_params_{thread}.npy"), mitm_params) # Save them in dump path. 
    return 

if __name__ == '__main__':
    dump_path = sys.argv[1] # First argument is always the path to the .npy file. 
    thread = sys.argv[2] # Second argument is the thread # so we know what matrix to look at. 
    main(dump_path, thread)

