""""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import os
import subprocess


##### ATTACK UTILS ##### 
def polish(X, longtype=False):
    if longtype:
        X = X.astype(np.longdouble)
    g, old = np.inner(X, X), np.inf # Initialize the Gram matrix
    while np.std(X) < old:
        old = np.std(X)
        # Calculate the projection coefficients
        c = np.round(g / np.diag(g)).T.astype(int)
        c[np.diag_indices(len(X))] = 0
        c[np.diag(g)==0] = 0

        sq = np.diag(g) + c*((c.T*np.diag(g)).T - 2*g) # doing l2 norm here
        s = np.sum(sq, axis=1) # Sum the squares. Can do other powers of norms
        it = np.argmin(s) # Determine which index minimizes the sum
        X -= np.outer(c[it], X[it]) # Project off the it-th vector
        g += np.outer(c[it], g[it][it] * c[it] - g[it]) - np.outer(g[:,it], c[it]) # Update the Gram matrix
    return X

def compute_curr_mitm_params(params, threadnum, logger):
    ''' Runs sage subprocess to determine if B (based on short vector norm) is small enough to be useful. '''
    try:
        subprocess.call(['sage', os.path.join(os.getcwd(), 'dual_hybrid_mitm/sage_scripts/compute_mitm_params.py'), params.dump_path, str(threadnum)])
    except Exception as e:
        print(e)
        subprocess.call(['sage', os.path.join(os.getcwd(), 'sage_scripts/compute_mitm_params.py'), params.dump_path, str(threadnum)])
    mitm_params = np.load(os.path.join(params.dump_path, f'current_mitm_params_{threadnum}.npy'))
    logger.info(f"For thread {threadnum}, current reduction delta={mitm_params[1]} and B={mitm_params[0]}.")
    return mitm_params[0] # This is the B parameter in Cheon attack. 
   

def mitm_params(sigma, Q, m, hamming):
    ''' 
    MITM parameters as defined in https://eprint.iacr.org/2019/1114 
    m = # vectors in lattice.  
    '''
    alpha = sigma / Q
    scale = np.round(alpha * Q * np.sqrt(m) / np.sqrt(2 * np.pi * hamming)) # from Cheon code, in the paper this is calculated wrt a short vector in an orthogonal lattice. 
    return alpha, scale

def calc_std_mitm(X, Q, m):
    ''' Will compute std as in AI attack AND vector norm as in dual attack. ''' 
    std = calc_std(X,Q,m)
    y = X[0,:]
    return std, np.linalg.norm(y)

def calc_std(X, Q, m):
    mat = X[:, m:] % Q # A is in right half of matrix, mat is now a new matrix with entries copied from X
    mat[mat > Q//2] -= Q
    return np.sqrt(12) * np.std(mat[np.any(mat!=0, axis=1)]) / Q
