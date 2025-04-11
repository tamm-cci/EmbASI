import numpy as np
import os
from mpi4py import MPI
from scalapack4py import ScaLAPACK4py
from numpy.random import rand
from ctypes import CDLL, RTLD_GLOBAL, POINTER, c_int, c_double
from embasi.parallel_utils import root_print
from embasi.roothan_hall_eigensolver_scalapack import pdsyevx_from_numpy_array

libpath = os.environ['ASI_LIB_PATH']

sl = ScaLAPACK4py(CDLL(libpath, mode=RTLD_GLOBAL))

n = 5
a = np.arange(n*n).reshape((n,n)) * (MPI.COMM_WORLD.rank+1) if MPI.COMM_WORLD.rank==0 else None

eigvals, eigvecs = pdsyevx_from_numpy_array(a)
root_print(f'EVALS: {eigvals}')
root_print(f'EVECS: {eigvecs}')
