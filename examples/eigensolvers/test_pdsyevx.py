import numpy as np, os
from mpi4py import MPI
from scalapack4py import ScaLAPACK4py
from numpy.random import rand
from ctypes import CDLL, RTLD_GLOBAL, POINTER, c_int, c_double

default_scalapack_path = '/usr/lib/x86_64-linux-gnu/libscalapack-openmpi.so'

if os.path.isfile(default_scalapack_path):
    # Try the default path used by most Linux systems
    libpath = '/usr/lib/x86_64-linux-gnu/libscalapack-openmpi.so'
elif 'SCALAPACK_ROOT' in os.environ:
    # ONEAPI Intel MKL case
    libpath = os.environ['SCALAPACK_ROOT']+'libmkl_scalapack_lp64.so'
elif 'SCALAPACK_LIB_PATH' in os.environ:
    # Finally, let user set their own SCALAPACK path
    libpath = os.environ['SCALAPACK_LIB_PATH']
else:
    OSError("Cannot find SCALAPACK library, try installing in the default \n \
    system path with apt-get, installing ONEAPI, or manually setting the \n \
    environmental variable SCALAPACK_LIB_PATH with location of the SCALAPACK \n \
    library.")

sl = ScaLAPACK4py(CDLL(libpath, mode=RTLD_GLOBAL))

n = 5
dtype=np.float64
a = np.arange(n*n, dtype=dtype).reshape((n,n), order='F') * (MPI.COMM_WORLD.rank+1) if MPI.COMM_WORLD.rank==0 else None
a = (a + a.T)/2.0 if MPI.COMM_WORLD.rank==0 else None
a = a.astype(dtype=dtype, order='F') if MPI.COMM_WORLD.rank==0 else None
print(a)

MP, NP = 2,1
ctx = sl.make_blacs_context(sl.get_default_system_context(), MP, NP)
descr = sl.make_blacs_desc(ctx, n, n)
print("descr", descr, descr.locrow, descr.loccol)

b = np.zeros((descr.locrow, descr.loccol), dtype=dtype, order='F')
sl.scatter_numpy(a, POINTER(c_int)(descr), b.ctypes.data_as(POINTER(c_double)), b.dtype)

d = np.zeros((n), dtype=dtype, order='F')

e = np.zeros((n), dtype=dtype, order='F')

tau = np.zeros((n), dtype=dtype, order='F')

taup = np.zeros((n), dtype=dtype, order='F')

tauq = np.zeros((n), dtype=dtype, order='F')

work = np.zeros((descr.locrow), dtype=dtype, order='F')

test_print = sl.gather_numpy(POINTER(c_int)(descr), b.ctypes.data_as(POINTER(c_double)), (n, n))
print(test_print)

# Workspace query for PDSYTRD
lwork = -1
info = -1
sl.pdsytrd("U", 5, b, 1, 1, descr,
           d, e, tau, work, lwork, info)

# Execute PDSYTRD with optimal workspace
lwork = int(work[0])
print(lwork)
work = np.zeros((lwork), dtype=dtype, order='F')

sl.pdsytrd("U", 5, b, 1, 1, descr,
           d, e, tau, work, lwork, info)

test_print = sl.gather_numpy(POINTER(c_int)(descr), b.ctypes.data_as(POINTER(c_double)), (n, n))

# Workspace query for PDGEBRD
lwork=-1
sl.pdgebrd(5, 5, b, 1, 1, descr,
           d, e, taup, tauq, work,
           lwork, info)

# Execute PDGEBRD with optimal workspace
lwork = int(work[0])
work = np.zeros((lwork), dtype=dtype, order='F')
sl.pdgebrd(5, 5, b, 1, 1, descr,
           d, e, taup, tauq, work,
           lwork, info)

test_print = sl.gather_numpy(POINTER(c_int)(descr), b.ctypes.data_as(POINTER(c_double)), (n, n))

# Workspace query for PDGEHRD
lwork=-1
sl.pdgehrd(5, 1, 5, b, 1, 1, descr,
           tau, work, lwork, info)

# Execute PDGEHRD with optimal workspace
lwork = int(work[0])
work = np.zeros((lwork), dtype=dtype, order='F')
sl.pdgehrd(5, 1, 5, b, 1, 1, descr,
           tau, work, lwork, info)

test_print = sl.gather_numpy(POINTER(c_int)(descr), b.ctypes.data_as(POINTER(c_double)), (n, n))
print(test_print)

# Execution of PDSYEVX
w = np.zeros(n, dtype=np.float64, order='F') # Eigenvalues
z = np.zeros((n, n), dtype=np.float64, order='F') # Eigenvectors

# Parameters for PDSYEVX
jobz = "V"  
range_type = "A"  
uplo = "U"  
ia = 1
ja = 1 
iz = 1
jz = 1 
vl = 0.0  
vu = 0.0  
il = 1  
iu = n  
abstol = 2.0 * 1e-15
orfac = -1.0

# Output parameters
m = 0  
nz = 0 

# Workspace query for PDSYEVX
lwork = -1
liwork = -1
work = np.zeros(1, dtype=np.float64, order='F')
iwork = np.zeros(1, dtype=np.int32, order='F')
info = 0

ifail = np.zeros(n, dtype=np.int32, order='F')
iclustr = np.zeros(1, dtype=np.int32, order='F')
gap = np.zeros(1, dtype=np.float64, order='F')
sl.pdsyevx("V", "A", "U", n, b, ia, ja, descr, 
             vl, vu, il, iu, abstol, m, nz, w, orfac, z, iz, jz, descr, 
             work, lwork, iwork, liwork, ifail, iclustr, gap, info)

# Execute PDSYEVX with optimal workspace
lwork = int(work[0])
liwork = int(iwork[0])
work = np.zeros(lwork, dtype=np.float64, order='F')
iwork = np.zeros(liwork, dtype=np.int32, order='F')

# Find nlocal rows and columns 
nprow, npcol, myrow, mycol = sl.blacs_gridinfo(ctx)
ifail = np.zeros(n, dtype=np.int32, order='F')
iclustr = np.zeros(2 * nprow * npcol, dtype=np.int32, order='F')
gap = np.zeros(nprow * npcol, dtype=np.float64, order='F')

# Call PDSYEVX with optimal workspace
sl.pdsyevx("V", "A", "U", n, b, ia, ja, descr, 
           vl, vu, il, iu, abstol, m, nz, w, orfac, z, iz, jz, descr, 
           work, lwork, iwork, liwork, ifail, iclustr, gap, info)

print(f'EIGVALS: {z}') if MPI.COMM_WORLD.rank==0 else None
test_print = sl.gather_numpy(POINTER(c_int)(descr), z.ctypes.data_as(POINTER(c_double)), (n, n))
print(f'EIGVEC: {test_print}') if MPI.COMM_WORLD.rank==0 else None
print(f'EIGVEC.T @ EIGVEC: {test_print.T @ test_print}') if MPI.COMM_WORLD.rank==0 else None

