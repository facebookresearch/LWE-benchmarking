""""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import sys
from sage.all import matrix, vector, GF, ZZ


def balanced_lift(e):
    """
    Lift e mod q to integer such that result is between -q/2 and q/2

    :param e: a value or vector mod q

    """
    from sage.rings.finite_rings.integer_mod import is_IntegerMod

    q = e.base_ring().order()
    if is_IntegerMod(e):
        e = ZZ(e)
        if e > q//2:
            e -= q
        return e
    else:
        return vector(balanced_lift(ee) for ee in e)

def main(filename):
    # Load file, compute dual, save back to same filename. 
    Abqvec = np.load(filename)
    A = Abqvec[:-1,:-1].copy() # Last row contains q, last column contains B
    b = Abqvec[:-1,-1] # Last column contains b
    q = Abqvec[-1,0] # Get q from last row.

    # Now put A and b in the field
    K = GF(q, proof=False)
    A = matrix(K, A) # Now A is a matrix in Z mod q. 
    b = vector(K, b) # Now b is a vector in Z mod q.

    # Do LA operation
    n = A.rank()
    A_ = A.matrix_from_rows([i for i in range(n)])
    if n != A_.rank():
        print("Matrix is not full rank. Exiting.")
        return
    else:
        b = b[:n]
        print(f'Recovered secret from LA recovery: {balanced_lift(A_.inverse() * b)}')


if __name__ == '__main__':
    filename = sys.argv[1] # First argument is always the path to the .npy file. 
    main(filename)

