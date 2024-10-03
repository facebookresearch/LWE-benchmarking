""""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import numpy as np
import io
import glob
import sys
import pickle as pkl
from time import time
from tqdm import tqdm
sys.path.append('.')
from src.generate.genSamples import InterleavedReduction
from utils import calc_std_mitm, mitm_params, polish
from src.generate.lllbkz import get_mlwe_circ, centered, centered_int
import subprocess
import itertools


FLOAT_UPGRADE = {
    'double': 'long double',
    'long double': 'dd',
    'dd': 'qd',
    'qd': 'mpfr_250'
}

class DualHybrid(InterleavedReduction):
    ''' 
    Base class for the DualHybrid reduction
    Everything except the get_A_Ap function and compute_stdev is same as Interleave class. 
    '''
    def __init__(self, params, thread, logger):
        self.logger = logger
        self.params = params
        self.N, self.Q= params.N, params.Q
        self.thread = thread
        self.longtype = np.log2(params.Q) > 30
        self.seed = [params.global_rank, params.env_base_seed, thread]
        # env_base_seed: different jobs will generate different data
        # thread: different workers will not generate same data
        self.export_path_prefix = os.path.join(params.dump_path, f"data_{thread}.prefix")

        # If interleaving, set up interleaving params:
        self.stdev_tracker = []
        self.prev_std = 10000 # Condition to only save off matrix if things improve.
        self.num_times_run = 0 
        self.lookback = self.params.lookback # number of steps over which to calculate (avg) decrease, must run given algo at least this many times before switching.
        self.min_decrease = -0.001 # min decrease we have to see over self.lookback steps to consider it "working".

        self.set_float_type(params.float_type)
        self.setup_algos()
        
        # Filenames for saving/loading
        self.matrix_filename = os.path.join(params.dump_path, f"matrix_{thread}.npy")
        self.resume_filename = os.path.join(params.resume_path, f"matrix_{thread}.npy")
        if os.path.isfile(self.resume_filename):
            mat_to_save = np.load(self.resume_filename)
            np.save(self.matrix_filename, mat_to_save)
            self.logger.info(f"Resuming from {self.resume_filename}.")
        self.logger.info(f"Random generator seed: {self.seed}.")

        # Determine if you're using MLWE vs. LWE
        self.mlwe_k = params.mlwe_k # if 0, then LWE, else RLWE with k number of modules. 
        self.k = params.k
        self.sigma = params.sigma
        self.gamma = params.gamma
        self.hamming = params.hamming
        self.tau = params.tau
        self.m = params.N # rows of actual lattice, computed in dual_hybrid_mitm so it saves correctly. 
        self.mitm_alpha, self.scale = mitm_params(self.sigma, self.Q, self.N, self.hamming) # assume m == N
        self.num_tinyA_per_lattice = self.m if self.mlwe_k == 0 else self.mlwe_k # Number of tinyA used to construct lattice: If RLWE/MLWE, we stack k circulated vectors, 


        # Create A matrices and b vectors for this worker. 
        self.rng = np.random.RandomState(self.seed + [int(time())])
        self.tiny_A = self.rng.randint(0, self.Q, size=(max(10+self.num_tinyA_per_lattice*self.tau // params.num_workers, self.N), self.N)) #np.load(params.reload_data)
        while not np.linalg.matrix_rank(self.tiny_A) == self.N:
            print("not linearly independent, trying again")
            self.tiny_A = np.random.randint(0, self.Q, size=(max(10+self.num_tinyA_per_lattice*self.tau // params.num_workers, self.N), self.N))

        # Load in the secret
        secrets = np.load(os.path.join(params.secret_path, 'secret.npy'))
        cols = np.where(np.sum(secrets != 0, axis=0) == self.hamming)[0]
        assert len(cols) > params.secret_seed 
        self.s = np.squeeze(secrets[:, cols[params.secret_seed]])
        if params.secret_path[-1] == '/':
            params.secret_path = params.secret_path[:-1] # Get rid of extra /
        self.secret_type = params.secret_path.split("/")[-1].split("_")[-3]

        if self.mlwe_k == 0:
            e = self.get_error(len(self.tiny_A))
            self.tiny_B = (self.tiny_A @ self.s + e) % self.Q
        else:
            tiny_B = []
            for a in self.tiny_A:
                circA = get_mlwe_circ(a, self.N // self.mlwe_k, self.mlwe_k) % self.Q
                e = self.get_error(len(circA))
                tiny_B.append((circA @ self.s + e) % self.Q)
            self.tiny_B = np.array(tiny_B)

        np.save(os.path.join(self.params.dump_path, f"Avecs_{self.thread}.npy"), self.tiny_A)
        np.save(os.path.join(self.params.dump_path, f"Bvecs_{self.thread}_h_{self.hamming}_seed_{self.params.secret_seed}.npy"), self.tiny_B)
        self.curr_A_start_index = 0 # Starting at 0, will update this by m every time we move through array. 
        self.num_short = 0
        self.idxs = []
        logger.info(f"DualHybrid reduction initialized: k = {self.k}, alpha = {self.mitm_alpha}, scale = {self.scale}")

    def setup_algos(self):
        # Set up function calls.
        if self.params.algo in ["BKZ", "BKZ2.0"]:
            self.algo1 = self.run_bkz_once
        elif self.params.algo == "flatter":
            self.algo1 = self.run_flatter_once
        else:
            self.algo1 = self.run_lll_once

        if self.params.algo2 in ["BKZ", "BKZ2.0"]:
            self.algo2 = self.run_bkz_once
        elif self.params.algo2 == "flatter":
            self.algo2 = self.run_flatter_once
        else:
            self.algo2 = self.run_lll_once

    def get_error(self, shape):
        if self.secret_type != 'binomial':
            return self.rng.normal(0, self.sigma, size=shape).round().astype(np.int64)
        else: # Binomial error
            err1 = self.rng.binomial(self.gamma, 0.5, shape)
            err2 = self.rng.binomial(self.gamma, 0.5, shape)
            return err1 - err2
        
    def get_A_Ap(self):
        if self.curr_A_start_index + self.num_tinyA_per_lattice > len(self.tiny_A):
            assert False == True, "Ran out of vectors to sample from."
        self.idxs = np.arange(self.curr_A_start_index, self.curr_A_start_index+self.num_tinyA_per_lattice) 
        self.curr_A_start_index += self.num_tinyA_per_lattice # increment. 
        U = self.idxs.reshape((self.num_tinyA_per_lattice, 1))
        A = self.tiny_A[self.idxs]
        assert np.max(A) - np.min(A) < self.Q

        # If MLWE, circulate accordingly. 
        if self.mlwe_k > 0:
            a = []
            for _a in A:
                a.append([get_mlwe_circ(_a, self.N // self.mlwe_k, self.mlwe_k)])
            A = np.squeeze(np.hstack(a)) % self.Q

        # Cut along the splitting dimension: 
        _A1 = A[:, :(self.N - self.k)] # only the first N-k columns. 

        # Do the hacky trick to get the dual. 
        # Save off, including q, for sage computation
        randname = str(self.thread)
        qvec = np.zeros((self.N-self.k))
        qvec[0] = self.Q
        qvec[1] = self.scale
        _A_save = np.vstack((_A1, qvec))
        np.save(os.path.join(self.params.dump_path, f"temp_{randname}.npy"), _A_save) 

        # Subprocess runs sage
        from sage_scripts.load_compute_dual import main
        main(os.path.join(self.params.dump_path, f"temp_{randname}.npy"))

        # Reload saved. 
        Ap = np.load(os.path.join(self.params.dump_path, f"temp_{randname}.npy"))
        if np.all(Ap == 0):
            self.logger.info("Error in sage computation - submatrix was not full rank. Trying again.")
            return self.get_A_Ap()
        
        return U, Ap.astype(int) # U.shape = mx1, Ap.shape = (m+N)*(m+N)

    def apply_short_vectors(self, shortvec, A2, b):
        '''
        Applies short vector output by BKZ to reduce A2, b, and u.
        '''
        m = A2.shape[0] # How many rows in A2? 
        shortvec = shortvec[-m:] // self.scale
        Ra2 = centered((shortvec @ A2) % self.Q, self.Q)
        Rb = centered_int((shortvec @ b) % self.Q, self.Q) # Now this becomes error
        return Ra2, Rb

    def compute_stdev(self, Ap, UT, use_polish=True, save=True, algo='flatter'):
        if use_polish:
            Ap = polish(Ap)
        newstddev, norm = calc_std_mitm(Ap, self.Q, self.N)
        if save:
            self.stdev_tracker.append(newstddev)
            self.save_mat(UT, Ap)
            self.logger.info(f'stddev = {newstddev} and short vector norm={norm}. Saved progress at {self.matrix_filename} after {algo} run.')
            self.logger.info(f'Short vector is: {Ap[0,:]}')
        return newstddev

    def save_mat(self, X, Y, newmat=False):
        mat_to_save = np.zeros((len(Y), len(Y)+self.m)).astype(int)
        mat_to_save[:len(X), :self.num_tinyA_per_lattice] = X
        # Put in current algorithm to use. 1 = BZK, -1 = Flatter, 0 = No preference.
        # Also include information about lookback.  
        if not newmat:
            alg = self.params.algo if self.a1 else self.params.algo2
        else:
            alg = self.params.algo # If you're saving off a new matrix to run again, set it with the first algo.
            self.num_times_run  = 0 # Reset the number of times run.
            self.stdev_tracker = [] # Reset the tracker.
        mat_to_save[-1, 0] = 1 if alg in ['BKZ', 'BKZ2.0'] else -1 if alg == 'flatter' else 0
        mat_to_save[-1, 1] = self.num_times_run # How many times we have run this algorithm
        if len(self.stdev_tracker) > 0:
            el = self.stdev_tracker if len(self.stdev_tracker) < self.lookback else self.stdev_tracker[-self.lookback:]
            int_el = [int(1000*x) for x in el] # Turn into ints so you can store in numpy array.
            mat_to_save[-1, 2:min(2+len(self.stdev_tracker), 2+self.lookback)] = int_el if len(int_el) < self.lookback else int_el[-self.lookback:]
        mat_to_save[:, self.m:] = Y
        np.save(self.matrix_filename, mat_to_save)

    def write(self, idxs, shortY):
        # Write the length of the short vector because we need it to compute b. 
        #assert X.shape[0] == Y.shape[0]
        file_handler_prefix = io.open(self.export_path_prefix, mode="a", encoding="utf-8")
        prefix0_str = str(self.thread)
        prefix1_str = " ".join(idxs.astype(str)) # indexes of tinyA
        prefix2_str = " ".join(shortY.astype(str)) # short vector from lattice reduction
        file_handler_prefix.write(f"{prefix0_str} ; {prefix1_str} ; {prefix2_str}\n")#; {prefix3_str}\n")
        file_handler_prefix.flush()
        self.num_short += 1

    def compute_bound_from_cheon_code(self, shortvec):
        ''' Uses Cheon code bound ''' 
        length =  np.linalg.norm(shortvec)
        B = float(2 + 1/np.sqrt(2*np.pi)) * (self.mitm_alpha * self.Q)
        B = B * B * self.N / (self.N + self.N) # in our code m == N
        B = np.sqrt(B) * length / self.scale
        return B / self.Q

    def check_for_param_upgrade(self, Ap, UT, newstddev, oldstddev=None):
        '''
        Estimates delta and B based on norm of shortest vector. 
        '''
        _, norm = calc_std_mitm(Ap, self.Q, self.N)

        # Numpy can't do determinants in modular fields, so we pass to sage. This is gross. 
        B = self.compute_bound_from_cheon_code(Ap[0])
        self.logger.info(f"Bound from Cheon code={B} * Q")

        # We want B < 0.25Q to ensure that error size is reasonable.
        # But in practice, we only recover recrets when B < 0.12Q, so use this. 
        # Also want to ensure we have at least a few nonzero elements: deals with weird case where initial reduction gives "short" vector that is all 0s except for one. 
        non_zeros = len(np.where(Ap[0] != 0)[0])

        if (B < 0.12) and (non_zeros >= 2): # Future work: set this as a parameter.
            self.logger.info(f'stddev = {newstddev}, norm={norm}. Exporting {self.matrix_filename}')
            R = Ap[0] # Write the whole thing
            self.write(self.idxs, R) # Writes only the short vec, not A,b -- more extensible. 
            print(f"Written {self.num_short} vectors so far")
            if self.num_short >= self.tau // self.params.num_workers:
                self.logger.info(f"Computed sufficient number of short vectors ({self.tau}, {self.tau // self.params.num_workers} per worker), exiting.")
                return -1
            self.logger.info(f'Starting new matrix at {self.matrix_filename}')
            return False # You've reached the threshold, stop!
    
        if not self.upgraded and newstddev < self.params.threshold2:
            # Go into Phase 2
            self.upgraded = True
            self.block_size, self.delta, self.alpha = self.params.bkz_block_size2, self.params.lll_delta2, self.params.alpha2
            self.logger.info(f'Upgrading to delta = {self.delta}, block size = {self.block_size}, alpha = {self.alpha}')
            return True

        # See if polishing helped at all -- BKZ only.
        if (oldstddev is not None) and (oldstddev - newstddev > 0):
            self.logger.info(f'stddev reduction: {oldstddev - newstddev} from polishing. ')
            return True
        return None # if you return None, you didn't meet criteria; keep going 

    def generate(self):
        if os.path.isfile(self.matrix_filename):
            A_Ap = np.load(self.matrix_filename)
            UT, Ap = A_Ap[:, :self.num_tinyA_per_lattice], A_Ap[:, self.m:]
            algo_indicator = A_Ap[-1,0]
            if algo_indicator != 0:
                self.num_times_run = A_Ap[-1,1]
                self.stdev_tracker = np.array(A_Ap[-1,2:self.lookback+2])
                # make sure all stdev elements are nonzero
                self.stdev_tracker = list(self.stdev_tracker[self.stdev_tracker != 0] / 1000) # Divide by 1000 to remove the int conversion.
                orig_algo = self.params.algo
                if algo_indicator == 1:
                    self.params.algo = 'BKZ2.0' # BKZ is first
                    self.params.algo2 = 'flatter' if orig_algo != self.params.algo2 else 'BKZ2.0'
                else: 
                    self.params.algo = 'flatter'
                    self.params.algo2 = 'BKZ2.0' if orig_algo != self.params.algo2 else 'flatter'
                self.setup_algos() # Redo algorithms now that you have changed things. 
                self.logger.info('Different algorithm detected in saved-off matrix: starting with algorithm={}'.format(self.params.algo))
        else:
            U, Ap = self.get_A_Ap()
            UT = U.T # To have num_cols = m, U and A are always transposed

        # Params/upgrades for BKZ/flatter.
        self.upgraded = False
        self.block_size, self.delta, self.alpha = self.params.bkz_block_size, self.params.lll_delta, self.params.alpha

        param_change = True
        while param_change:
            Ap, param_change = self.run(UT, Ap)
            if param_change == -1:
                return False # Worker encountered error, end now.

        # Rewrite checkpoint with new data and return the bkz reduced result
        newA, newAp = self.get_A_Ap()
        self.save_mat(newA.T, newAp, newmat=True)
        return True # Worker finished, end now.

### Run MITM attack:
class MITM(object):
    def __init__(self, params, logger, thread):
        self.params = params
        self.logger = logger
        self.thread = thread
        self.short_vectors_path = params.short_vectors_path

        try:
            self.mlwe_k = params.mlwe_k
        except:
            params.mlwe_k = 0
            self.mlwe_k = params.mlwe_k

        self.k = params.k
        self.N = params.N
        self.Q = params.Q
        self.hamming = params.mitm_hamming # NOTE: need to use Hamming from reduction to get right scale, see below. 
        self.tau = params.tau
        self.sigma = params.sigma
        self.gamma = params.gamma
        if self.tau < 0: # Didn't set a limit. 
            self.tau = self.k

        assert self.k > 0 and self.tau > 0, "Must have k and tau > 0."

        self.m = self.N #if params.mlwe_k == 0 else params.mlwe_k 
        self.num_tinyA_per_lattice = self.N if params.mlwe_k == 0 else params.mlwe_k # Number of tinyA used to construct lattice: If RLWE/MLWE, we are just going to circulate a single line.

        secrets = np.load(os.path.join(params.secret_path, 'secret.npy'))
        print(secrets.shape,params.secret_path, self.hamming, params.secret_seed)
        cols = np.where(np.sum(secrets != 0, axis=0) == self.hamming)[0]
        assert len(cols) > params.secret_seed 
        self.s = np.squeeze(secrets[:, cols[params.secret_seed]])
        if params.secret_path[-1] == '/':
            params.secret_path = params.secret_path[:-1] # Get rid of extra /
        self.secret_type = params.secret_path.split("/")[-1].split("_")[-3]
        self.rng = np.random.RandomState(int(time())) 

        self.mitm_alpha, self.scale = mitm_params(self.sigma, self.Q, self.N, self.params.hamming) # Use the REDUCTION hamming for scale parameter. 
        self.logger.info("alpha={}, scale={}".format(self.mitm_alpha, self.scale))
        if self.params.debug == True:
            self.logger.info(f"Secret is: {self.s}")
            self.logger.info(f"Secret to guess is: {self.s[-self.k:]}, nonzero elements: {np.where(self.s[-self.k:] != 0)}")

    
    def get_error(self, shape):
        if self.secret_type != 'binomial':
            return self.rng.normal(0, self.sigma, size=shape).round().astype(np.int64)
        else: # Binomial error
            err1 = self.rng.binomial(self.gamma, 0.5, shape)
            err2 = self.rng.binomial(self.gamma, 0.5, shape)
            return err1 - err2
        

    def create_Bs(self,A_path, origA, seed):
        if self.mlwe_k == 0:
            e = self.get_error(len(origA))
            self.tiny_B = (origA @ self.s + e) % self.Q
        else:
            tiny_B = []
            for a in origA:
                circA = get_mlwe_circ(a, self.N // self.mlwe_k, self.mlwe_k) % self.Q
                e = self.get_error(len(circA)) 
                tiny_B.append((circA @ self.s + e) % self.Q)
            self.tiny_B = np.array(tiny_B)
        np.save(os.path.join(A_path, f"Bvecs_{int(seed)}_{self.secret_type}_h_{self.hamming}_seed_{self.params.secret_seed}.npy"), self.tiny_B)

    def read(self, path):
        with open(path) as fd:
            for _, line in enumerate(tqdm(fd)):
                if not line:
                    breakpoint()
                    continue
                splitit = line.strip().split(";")
                if len(splitit) == 3:
                    seed, idx, sv = line.strip().split(";")
                    seed = np.float64(seed.lstrip()) 
                    idx = np.array(idx.split(), dtype=np.int64)
                    sv = np.array(sv.split(), dtype=np.float64) 
                    yield seed, idx, sv
                elif len(splitit) == 4:
                    seed, idx, sv, A_path = line.strip().split(";")
                    seed = np.float64(seed.lstrip()) 
                    idx = np.array(idx.split(), dtype=np.int64)
                    sv = np.array(sv.split(), dtype=np.float64) 
                    A_path = ''.join(A_path)
                    yield seed, idx, sv, A_path


    def remove_redundant_rows(self, path):
        seed, shortvec, idx = [], [], []
        for s, i, sv in self.read(path):
            seed.append(s)
            idx.append(i)
            shortvec.append(sv)

        seed = np.array(seed, dtype=np.int64) # which dataworker produced it?
        idx = np.array(idx) # length of short vector
        shortvec = np.array(shortvec, dtype=np.int64) # what is short vec?

        lines = []
        for s, sv, i in zip(seed, shortvec, idx):
            s_str = str(s)
            i_str = " ".join(i.astype(str))
            sv_str = " ".join(sv.astype(str))
            path_str = os.path.dirname(path)
            lines.append(f"{s_str};{i_str};{sv_str};{path_str}\n") # Adds path to A vectors, since we need these. 
        return lines

    def load_short_vectors_and_computeAb(self):
        # Find the data.prefix file
        data_prefix_path = os.path.join(self.short_vectors_path, "data.prefix")
        if not os.path.isfile(data_prefix_path):
            # If you're not using slurm, don't need the extra *.
            if self.short_vectors_path[-1] == "/":
                self.short_vectors_path = self.short_vectors_path[:-1]
            paths = glob.glob(f"{self.short_vectors_path}/*/data_*.prefix")
            if len(paths) == 0:
                paths = glob.glob(f"{self.short_vectors_path}/data_*.prefix")

            with open(data_prefix_path, "w") as outfile:
                for path in tqdm(paths):
                    for line in self.remove_redundant_rows(path):
                        outfile.write(line)
        
        # Now load in the data
        self.logger.info("Loading data from %s", data_prefix_path)

        length = 0
        redA, redB, lens = [], [], []

        def apply_short_vectors(shortvec, A2, b):
            '''
            Applies short vector output by BKZ to reduce A2, b, and u.
            '''
            m = A2.shape[0] # How many rows in A2? 
            shortvec = shortvec[-m:] // self.scale
            Ra2 = centered((shortvec @ A2) % self.Q, self.Q)
            Rb = centered_int((shortvec @ b) % self.Q, self.Q) # Now this becomes error
            return Ra2, Rb

        # Compute short vectors from results in data.prefix
        for seed, i, sv, A_path in self.read(data_prefix_path):

            origA = np.load(os.path.join(A_path, f"Avecs_{int(seed)}.npy"))
            if not os.path.exists(os.path.join(A_path, f"Bvecs_{int(seed)}_{self.secret_type}_h_{self.hamming}_seed_{self.params.secret_seed}.npy")):
                self.logger.info(f"Could not find Bvecs_{int(seed)}_{self.secret_type}_h_{self.hamming}_seed_{self.params.secret_seed}.npy, creating data.")
                self.create_Bs(A_path, origA, seed)
                
            origB = np.load(os.path.join(A_path, f"Bvecs_{int(seed)}_{self.secret_type}_h_{self.hamming}_seed_{self.params.secret_seed}.npy")) #, mmap_mode='r')
            A = origA[i]
            bvecs = origB[i]

            # If MLWE, circulate accordingly. 
            if self.mlwe_k > 0:
                a = []
                for _a in A:
                    a.append([get_mlwe_circ(_a, self.N // self.mlwe_k, self.mlwe_k)])
                A2 = (np.squeeze(np.hstack(a)) % self.Q)[:,-self.k:]
                bvecs = np.hstack(bvecs) # Just stack them. 
                a = np.squeeze(np.hstack(a))
            else:
                A2 = A[:,-self.params.k:]
            bvecs = np.squeeze(bvecs) # Remove extra dim
            shortA, shortB = apply_short_vectors(sv, A2, bvecs)
            redA.append(shortA)
            redB.append(shortB)
            lens.append(np.linalg.norm(sv))
            if len(redA) > self.tau:
                break

        # Make sure they're in a nice format
        redA = np.vstack(redA)
        redB = np.array(redB) 

        # Put them back mod q
        redA = redA % self.Q
        redB = redB % self.Q

        # Now that you have length, compute error bound
        length = np.sum(lens) / len(redA) # Make sure to take the average length
    
        # B definition from Cheon code.
        # NOTE: this matches Cheon code but not definition given in Cheon paper on pg. 21
        B = (2 + 1 / np.sqrt(2 * np.pi)) * (self.mitm_alpha * self.Q)
        B = B * B * self.k / (self.k + self.N) # TODO using self.k instead of self.m since we need a square matrix, so guessed matrix is really k x k. 
        B = np.sqrt(B) * length / self.scale 
 
        if self.params.bound < 0:
            self.bound = B # This is the bound we will use for the MITM attack.
        else:
            self.bound = self.params.bound
        self.logger.info(f"Bound is {self.bound}")

        return redA, redB
    
    def lsh(self, x):
        m = len(x)
        pow2 = np.array([2**i for i in range(m)])
        return ((x % self.Q)  < self.Q//2).dot(pow2)

    def get_boundary_elements(self, query):
        # Given a vector a, list of a elements that are close to 0 or q.
        # Query is in (-q/2, q/2)
        absq = np.abs(query)
        sgnvec = np.zeros(len(query))
        sgnvec[query > 0] = 1
        sgnvec[np.where(absq < self.bound)[0]] = -1
        sgnvec[np.where(absq > self.Q//2 - self.bound)[0]] = -1
        return sgnvec, np.where((absq < self.bound) | (absq > self.Q//2 - self.bound))[0]

    def check_collision(self, query, A, sgnvec, Table, boundary_idx, index_num, orig_s):
        '''
        Recursive function to flip -1 bits in sgnvec and see if element represented by this in table. 

        :param sgnvec: ternary vector to check

        '''
        if index_num == len(boundary_idx):
            lsh_vec = sum(el*2**i for i,el in enumerate(sgnvec))
            if lsh_vec in Table:
                self.logger.info("found possible match")
                for s in Table[lsh_vec]:
                    sgnbits = list(itertools.product(self.get_possible_bit_values(), repeat=len(s)))
                    for sgn in sgnbits:
                        vec = centered(sum(A[i]*_sgn for i, _sgn in zip(s, sgn)) % self.Q, self.Q)
                        diff = centered((query - vec) % self.Q, self.Q)
                        self.logger.info(f"diff: {diff}")
                        self.logger.info(f"bound = {self.bound}, metrics on diff - norm: {np.linalg.norm(diff, np.inf)}, mean: {np.mean(np.abs(diff))}, std: {np.std(np.abs(diff))}, median: {np.median(np.abs(diff))}")
                        if self.params.debug == True:
                            self.logger.info(f"guess = {s}, {sgn}; other half = {orig_s}; true = {np.nonzero(self.s[-self.k:])}, {self.s[-self.k:][np.nonzero(self.s[-self.k:])[0]]}")
                        # NOTE: using the linalg norm lets outliers dominate, which is bad, so we use median value instead. 
                        if np.median(np.abs(diff)) < self.bound:
                            return (s, sgn)
        else:
            for i in [0,1]:
                sgnvec[boundary_idx[index_num]] = i
                res = self.check_collision(query, A, sgnvec, Table, boundary_idx, index_num + 1, orig_s)
                if res is not None:
                    return res
            
    def noisy_search(self, query, A, Table, orig_s=None):
        '''
        Using the bound B, search for all possible matches in Table.
        '''
        # Find v in S within distance 'bound' from query
        # Input T is a hash table from S
        # If no such v, return None
        sgnvec, boundary_idx = self.get_boundary_elements(query)
    
        if len(boundary_idx) == len(sgnvec):
            return [0] * len(query) # Everything is on the boundary, bad news :(

        v = self.check_collision(query, A, sgnvec, Table, boundary_idx, index_num = 0, orig_s=orig_s)
        return v

    def get_possible_bit_values(self):
        if self.secret_type == 'binary':
            return [0, 1]
        elif self.secret_type == 'ternary':
            return [-1, 0, 1]
        elif self.secret_type == 'binomial':
            return list(np.range(-self.gamma, self.gamma+1))
        elif self.secret_type == 'gaussian':
            return list(np.range(-self.sigma*2, self.sigma*2)) # Guesstimate
        else:
            raise ValueError(f"Unknown secret type: {self.secret_type}")
        

    def build_and_search(self, shortA, shortb, half):
        '''
        Faster way to do this: build the table and search in one go. 
        '''
        T = {}
        AT = shortA.T
        count = 0
        bits = self.get_possible_bit_values()
   

        if self.params.debug==True:
            # Just get a sense of where secret bits are. We don't use this info to recover secret. 
            sidx = np.where(self.s[-self.k:] != 0)[0]
            svals = self.s[-self.k:][sidx]
            half_sk = len(sidx) // 2
            firsthalf = centered(sum(AT[i]*s for i, s in zip(sidx[:half_sk], svals[:half_sk])) % self.Q, self.Q)
            secondhalf = centered(sum(AT[i]*s for i, s in zip(sidx[half_sk:],svals[half_sk:])) % self.Q, self.Q)
            bad = centered(sum(AT[i]*s for i, s in zip((13, 14), (1,1))) % self.Q, self.Q)
            good1 = centered(((shortb - secondhalf) % self.Q -firsthalf) % self.Q, self.Q).astype(int)
            good2 = centered(((shortb - firsthalf) % self.Q - secondhalf) % self.Q, self.Q).astype(int)
            bad1 = centered((shortb - bad) % self.Q, self.Q).astype(int)
            self.logger.info(f'Stats on good half1: max={np.max(good1)}, median={np.median(np.abs(good1))}, mean={ np.mean(np.abs(good1))}')
            self.logger.info(good1)
            self.logger.info(f'Stats on good half2: max={np.max(good2)}, median={np.median(np.abs(good2))}, mean={ np.mean(np.abs(good2))}')
            self.logger.info(good2)
            self.logger.info(f'Stats on bad: max={np.max(bad1)}, median={np.median(np.abs(bad1))}, mean={ np.mean(np.abs(bad1))}')
            self.logger.info(bad1)
            input()

        for h in range(1, half):
            signbits = list(itertools.product(bits, repeat=h)) # Get the different permutations of secret bit values for this h guess. 
            secret_combs = list(itertools.combinations(range(self.k), h))
            for s1 in secret_combs:  # Loop over possible combs of hamming weight h
                for sgn in signbits: # Loop over possible combs of different sign bits
                    b1 = centered(sum(AT[i]*s for i, s in zip(s1, sgn)) % self.Q, self.Q)
                    sgnvec = self.lsh(b1)
                    if sgnvec in T:
                        if s1 not in T[sgnvec]:
                            T[sgnvec].append(s1) # Not storing sign values to save space, trades off for more compute at last step. 
                    else:
                        T[sgnvec] = [s1]

                    # Now check and see if the inverse is in this. 
                    query = centered((shortb - b1) % self.Q, self.Q)
                    if self.params.debug == True:
                        sys.stdout.write("\033[F")
                        sys.stdout.write("\033[K")
                        print('Number of noisy searches = %d' % count)
                    count += 1

                    if count > (len(signbits)): # Skip querying on first element. 
                        guess_s2 = self.noisy_search(query, AT, T, orig_s=s1)
                    else:
                        # First element put in, don't do a noisy search
                        guess_s2 = None
                    
                    if guess_s2 is not None and guess_s2[0] != s1: 
                        # Check full secret, returned value from noisy search has 2 parts.
                        s2 = guess_s2[0]
                        _s1 = list(s1).copy()
                        # Concat secret vectors
                        _s1.extend(list(s2))

                        # Concat sign vectors if needed
                        if len(bits) > 1:
                            s2_sign = guess_s2[1]
                            s1_sign = list(sgn)
                            s1_sign.extend(list(s2_sign)) 
                        else: 
                            s1_sign = [1]*len(_s1)

                        # Check for correctness
                        A_s = sum(AT[i]*s for i, s in zip(_s1, s1_sign)) % self.Q
                        resid  = np.abs(centered((shortb - A_s) % self.Q, self.Q))
                        self.logger.info(f"residuals:{resid}")
                        self.logger.info(f"metrics on resid - norm: {np.linalg.norm(resid, np.inf)}, mean: {np.mean(resid)}, std: {np.std(resid)}, median: {np.median(resid)}")
                        if np.median(resid) < self.bound: 
                            self.logger.info("Found a secret match!")
                            s = np.zeros(self.k) 
                            for i, sval in zip(_s1, s1_sign):
                                s[i] = sval
                            self.logger.info(f"Guessed secret is: {s.astype(int)}")

                            if self.params.debug == True: # If debugging, use the real secret to guess.
                                self.logger.info(f"Real secret is: {self.s[-self.k:]}")
                                if np.all(s == self.s[-self.k:]):
                                    return
                                else:
                                    self.logger.info("Close but not quite.")
                            else: #  use LA method from Cheon to confirm guess. 
                                # A, c, query, A_s == vec from Cheon, q
                                error = query - A_s
                                _shortb = (shortb - error) % self.Q
                                # Save off, including q, for sage computation
                                qvec = np.zeros((shortA.shape[1]+1))
                                qvec[0] = self.Q
                                _Ab_save = np.hstack((shortA, _shortb[:,np.newaxis])) #, qvec))
                                _Abq_save = np.vstack((_Ab_save, qvec))
                                np.save(os.path.join(self.params.dump_path, f"tempAbq.npy"), _Abq_save) 

                                # Subprocess runs sage
                                from sage_scripts.recover_secret import main
                                main(os.path.join(self.params.dump_path, f"tempAbq.npy"))
                                return
                                                            
                                    
                                    

    def run_mitm(self, shortA, shortb):
        # First, generate the table.
        if self.params.num_bits_in_table < 0: # Maximum number of bits in table secret guesses.
            if self.hamming % 2 == 0:
                half = self.hamming // 2
            else:
                half = self.hamming // 2 + 1
        else:
            half = self.params.num_bits_in_table # The above should work but you can change it if you want. 

        if self.params.debug == True:  # Make sure that sufficient bits are in table: 
            s_half_true = self.s[-self.k:].sum()
            if half < s_half_true:
                print(f"Half of h is {half}, true bits to guess is {s_half_true}")
                half = s_half_true

        # Faster way to do this
        self.build_and_search(shortA, shortb, int(half))

    def run(self):
        # First, load in the short vector data.
        shortA, shortB = self.load_short_vectors_and_computeAb()

        # Now, run MITM
        start = time()
        self.run_mitm(shortA, shortB)
        self.logger.info(f"Done in {time() - start} seconds")


