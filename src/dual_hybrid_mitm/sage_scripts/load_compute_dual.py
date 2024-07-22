""""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import sys
from sage.all import matrix, identity_matrix, GF, ZZ

def get_dual_instance(A, scale=1):
    # Adapted from https://github.com/swanhong/HybridLWEAttack/blob/master/Implement/Mitm.py
    q = A.base_ring().order()
    n = A.ncols()
    m = A.nrows()
    try:
        B = A.matrix_from_rows(range(m-n, m)).inverse().change_ring(ZZ) # last m-n rows of T1, which is the bottom left quadrant of original A. 
        L = identity_matrix(ZZ, n).augment(matrix(ZZ,n,m-n))
        L = L.augment(B)
        L = L.stack(matrix(ZZ, m-n, n).augment(A.left_kernel().basis_matrix().change_ring(ZZ)))
        L = L.stack(matrix(ZZ, n, m).augment(q*identity_matrix(ZZ, n)))

        for i in range(0, m + n):
            for j in range(n, m + n):
                L[i, j] = scale*L[i, j] # left cols multipled by scaling factor. 
        return L
    except Exception as e:
        print(e)
        return -1

def main(filename):
    # Load file, compute dual, save back to same filename. 
    A_qvec = np.load(filename)
    A = A_qvec[:-1,:] # Last row contains q
    q = A_qvec[-1,0] # Get q from last row.
    scale = A_qvec[-1,1] # Get scale from last row.

    # Now compute dual.
    K = GF(q, proof=False)
    A = matrix(K, A) # Now A is a matrix in Z mod q. 
    A_dual = get_dual_instance(A, scale)

    if A_dual == -1:
        np.save(filename, np.zeros((2,2))) # Dummy matrix to trigger error in main script. 
        return

    # Now put it in numpy format. 
    _A_dual = np.array([[int(el) for el in ",".join(_el[1:-1].split()).split(",")] for _el in A_dual.str()[1:-1].split("\n")[:-1]])
    # last row is special
    last_row = [int(el) for el in ",".join(A_dual.str()[1:-1].split("\n")[-1][1:].split()).split(",")] 
    A_dual = np.vstack((_A_dual, last_row))

    # And save it off
    np.save(filename, A_dual)


if __name__ == '__main__':
    filename = sys.argv[1] # First argument is always the path to the .npy file. 
    main(filename)

