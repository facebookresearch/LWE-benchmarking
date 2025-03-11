""""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
from scipy.linalg import circulant


def rlwe_circ(a, n):
    A = circulant(a)
    tri = np.triu_indices(n, 1)
    A[tri] *= -1
    return A


def usvp_params(secret_type, n, Q, sigma, h):
    q = 2 ** np.ceil(np.log2(Q))
    w = 1  # weight for gaussian secret
    if (secret_type == "binary") or (secret_type == "ternary"):
        # w = np.round(np.sqrt(2) * sigma) # DOES NOT ACCOUNT FOR h, see eq 3.10 in https://eprint.iacr.org/2020/539.pdf
        w = np.round(np.sqrt(n / h) * sigma)
    d = 2 * n * np.log(q / w) / np.log(q / (np.sqrt(2 * np.pi * np.e) * sigma))
    logdelta = np.log(q / (np.sqrt(2 * np.pi * np.e) * sigma)) ** 2 / (
        4 * n * np.log(q / w)
    )
    return (
        int(w),
        int(np.ceil(d)) - n - 1,
        min(np.round(np.e**logdelta, 4), 1),
    )  # w, m, delta


def orthogonalize(mat, i):
    count = 0  # Reorthogonalize several times to combat roundoff errors
    dot_prods = np.linalg.norm(mat[:i], axis=1) ** 2
    while count < 2:
        changed = False
        for k in range(i):
            i_k = mat[i] @ mat[k]
            mat[i] -= i_k / dot_prods[k] * mat[k]
            i_i = mat[i] @ mat[i]
            if i_k * i_k > 1e-28 * dot_prods[k] * i_i:
                changed = True
        if not changed:
            count += 1


def orthogonalize2(mat, i):
    count = 0  # Reorthogonalize several times to combat roundoff errors
    norm_mat = mat[:i] / np.expand_dims(np.linalg.norm(mat[:i], axis=1), 1)
    while count < 2:
        changed = False
        proj = norm_mat @ mat[i]
        mat[i] -= proj @ norm_mat
        if np.max(np.abs(proj)) < 1e-10:
            count += 1


def lll(a, delta):
    b = a.astype(float)
    # Run Gram-Schmidt (without normalization) on b
    for j in range(1, b.shape[0]):
        orthogonalize2(b, j)
    j = 1
    dot_prods = np.linalg.norm(b, axis=1) ** 2
    while j < a.shape[0]:
        # Reduce the basis
        for k in range(j - 1, -1, -1):
            g = a[j] @ b[k]
            a[j] -= a[k] * int(np.round(g / dot_prods[k]))
        f = dot_prods[j - 1]
        g = a[j] @ b[j - 1]
        g /= f
        if dot_prods[j] / f < delta - g * g:
            a[[j - 1, j]] = a[[j, j - 1]]
            # Recompute b[j] and b[j-1] from scratch for numerical stability
            b[j - 1] = a[j - 1].astype(float)
            if j > 1:
                orthogonalize2(b, j - 1)
            dot_prods[j - 1] = b[j - 1] @ b[j - 1]
            b[j] = a[j].astype(float)
            orthogonalize2(b, j)
            dot_prods[j] = b[j] @ b[j]
            if j > 1:
                j -= 1
        else:
            j += 1
    return a


def polish(X, longtype=False):
    if longtype:
        X = X.astype(np.longdouble)
    g, old = np.inner(X, X), np.inf  # Initialize the Gram matrix
    while np.std(X) < old:
        old = np.std(X)
        # Calculate the projection coefficients
        c = np.round(g / np.diag(g)).T.astype(int)
        c[np.diag_indices(len(X))] = 0
        c[np.diag(g) == 0] = 0

        sq = np.diag(g) + c * ((c.T * np.diag(g)).T - 2 * g)  # doing l2 norm here
        s = np.sum(sq, axis=1)  # Sum the squares. Can do other powers of norms
        it = np.argmin(s)  # Determine which index minimizes the sum
        X -= np.outer(c[it], X[it])  # Project off the it-th vector
        g += np.outer(c[it], g[it][it] * c[it] - g[it]) - np.outer(
            g[:, it], c[it]
        )  # Update the Gram matrix
    return X


def calc_std_usvp(X, orig_std, Q, m, n):
    """For now, just assume A is in right half of matrix.
    Shape is either base shape = [I*w, A.T 0; 0, q*I 0; 0 b 1] or Verde shape = [0, q*I 0; I*w, A.T 0; 0 b 1]
    """
    mat = X[
        :-1, n : (m + n)
    ].copy() # A is in right half of matrix, avoid bottom row and last col.
    mat[mat > Q // 2] -= Q
    return np.std(mat[np.any(mat != 0, axis=1)]) / orig_std


def calc_std(X, Q, m):
    mat = (
        X[:, m:] % Q
    ).copy()  # A is in right half of matrix, mat is now a new matrix with entries copied from X
    mat[mat > Q // 2] -= Q
    return np.sqrt(12) * np.std(mat[np.any(mat != 0, axis=1)]) / Q


# flatter functions
def encode_intmat(intmat):
    """will put in expected format for flatter input."""
    fplll_Ap_encode = "[" + intmat.__str__() + "]"
    fplll_Ap_encode = fplll_Ap_encode.encode()
    return fplll_Ap_encode


def decode_intmat(out):
    """Decodes output intmat from flatter and puts it back in np.array form."""
    t_str = out.rstrip().decode()[1:-1]
    Ap = np.array(
        [np.array(line[1:-1].split(" ")).astype(int) for line in t_str.split("\n")[:-1]]
    )
    return Ap


def centered(arr, q):
    try:
        return centered_arr(arr, q)
    except Exception as e:
        print("exception ", e)
        return centered_int(arr, q)

def centered_arr(arr, q):
    arr = arr.copy()
    arr[arr >  q // 2] -= q
    return arr


def centered_int(el, q):
    if el > q // 2:
        return el - q
    else:
        return el


def get_mlwe_circ(X, n, k):
    if np.ndim(X) == 1:
        X = X.reshape((k,n))
    
    A = np.zeros(shape=(n,k*n), dtype=np.int64)
    for i in range(k):
        a = X[i]
        A[:,n*i:n*(i+1)] = rlwe_circ(a, n)
    return A

    