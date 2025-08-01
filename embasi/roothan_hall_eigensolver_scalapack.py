import numpy as np
from ctypes import RTLD_GLOBAL, CDLL, POINTER, byref, c_int, c_int64, c_int32, c_bool, c_double
from embasi.parallel_utils import root_print, mpi_bcast_matrix
import os

def invsqr_overlap_calc(overlap):

    # @TODONPSCAL: REPLACE NP DIRECTIVE
    sigma, U = np.linalg.eigh(overlap)
    sigma_sqrt = np.diag(sigma**(-0.5))

    return U @ sigma_sqrt @ U.T

def xform_hamiltonian(hamiltonian, xform_mat):

    from embasi.parallel_utils import root_print

    return xform_mat.T @ hamiltonian @ xform_mat

def back_xform_evecs(eigenvectors, xform_mat):

    return xform_mat @ eigenvectors

def sort_eigvals_and_evecs(eigenvalues, eigenvectors):

    idx = np.argsort(eigenvalues)
    
    return eigenvalues[idx], eigenvectors[:,idx]

def calculate_occ_mat(eigenvalues, nelec):

    occ_mat = np.zeros(np.size(eigenvalues))
    occ_mat[:int(nelec/2)] = 2.0

    return occ_mat

def calculate_densmat(eigenvectors, occ_mat):

    import copy

    occ_evecs = copy.deepcopy(eigenvectors)
    # @TODONPSCAL: FIGURE OUT HOW THIS IS DONE LOCALLY
    for idx in range(np.size(occ_mat)):
        occ_evecs[:,idx] = occ_evecs[:,idx] * np.sqrt(occ_mat[idx])

    return occ_evecs @ occ_evecs.T

def overlap_illcondition_check_parallel(overlap, thresh, inv=True, return_mask=False):

    from scipy.linalg import eig_banded, eigh
    from embasi.parallel_utils import root_print

    # @TODONPSCAL: REPLACE WITH GLOBAL SHAPE
    n_basis = np.shape(overlap)[0]

    # @TODONPSCAL: REPLACE ARGUMENT WITH NPSCAL
    ovlp_evals, ovlp_evecs = pdsyevx_from_numpy_array(overlap, n_basis, vl=thresh, vu=100000)

    # Count non-singular values
    n_bad = (ovlp_evals < thresh).sum()
    n_good = np.shape(overlap)[0] - n_bad

    good_val_mask = (ovlp_evals > thresh)

    if n_bad > 0:
        # Transform overlap matrix
        # @TODONPSCAL: FIGURE OUT HOW THIS IS DONE LOCALLY
        ovlp_filtered = ovlp_evecs[:, good_val_mask]
        evals_filtered = ovlp_evals[good_val_mask]

        for idx in range(np.size(evals_filtered)):
            sqrt_ev = np.sqrt(evals_filtered[idx])

            if inv:
                # @TODONPSCAL: FIGURE OUT HOW THIS IS DONE LOCALLY
                ovlp_filtered[:, idx] = ovlp_filtered[:, idx]/sqrt_ev
            else:
                # @TODONPSCAL: FIGURE OUT HOW THIS IS DONE LOCALLY
                ovlp_filtered[:, idx] = ovlp_filtered[:, idx]*sqrt_ev

    else:
        if inv:
            # @TODONPSCAL: REPLACE NUMPY DIRECTIVE
            sigma_sqrt = np.diag(ovlp_evals**(-0.5))
        else:
            # @TODONPSCAL: REPLACE NUMPY DIRECTIVE
            sigma_sqrt = np.diag(ovlp_evals**(0.5))

        ovlp_filtered = ovlp_evecs @ sigma_sqrt @ ovlp_evecs.T

    if return_mask:
        return ovlp_filtered, n_bad, good_val_mask
    else:
        return ovlp_filtered, n_bad

def hamiltonian_eigensolv_parallel(hamiltonian, overlap, nelec, return_orthog=False):

    from embasi.parallel_utils import root_print

    thresh = 1e-5
    # @TODONPSCAL: REPLACE WITH GLOBAL ARRAY INDEXING
    n_basis = np.shape(overlap)[0]
    xform_mat, n_bad = overlap_illcondition_check_parallel(overlap, thresh)
    n_good = n_basis - n_bad

    evals, evecs = pdsyevx_from_numpy_array(xform_hamiltonian(hamiltonian, xform_mat), n_good)

    if return_orthog:
        evals, evecs = sort_eigvals_and_evecs(evals, evecs)
        occ_mat = calculate_occ_mat(evals, nelec)
        
        return evals, evecs, occ_mat, xform_mat

    else:
        evecs = back_xform_evecs(evecs, xform_mat)

        evals, evecs = sort_eigvals_and_evecs(evals, evecs)
        occ_mat = calculate_occ_mat(evals, nelec)

        return evals, evecs, occ_mat

def pdsyevx_from_numpy_array(array, vl=None, vu=None):

    from scalapack4py import ScaLAPACK4py
    from ctypes import CDLL, POINTER, c_int
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    ntasks = comm.Get_size()
    rank = comm.Get_rank()
    sl = ScaLAPACK4py(CDLL(os.environ['ASI_LIB_PATH'], mode=RTLD_GLOBAL))

    n = global_array_size
    dtype = np.float64
    a = array if MPI.COMM_WORLD.rank==0 else None
    a = a.astype(dtype=dtype, order='F') if MPI.COMM_WORLD.rank==0 else None

    MP, NP = find_squarest_grid(ntasks)
    ctx = sl.make_blacs_context(sl.get_default_system_context(), MP, NP)
    descr = sl.make_blacs_desc(ctx, n, n)

    b = np.zeros((descr.locrow, descr.loccol), dtype=dtype, order='F')
    sl.scatter_numpy(a, POINTER(c_int)(descr), b.ctypes.data_as(POINTER(c_double)), b.dtype)
    
    #d = np.zeros((n), dtype=dtype, order='F')
    #e = np.zeros((n), dtype=dtype, order='F')
    #tau = np.zeros((n), dtype=dtype, order='F')
    #taup = np.zeros((n), dtype=dtype, order='F')
    #tauq = np.zeros((n), dtype=dtype, order='F')
    #work = np.zeros((descr.locrow), dtype=dtype, order='F')
    #
    ## Workspace query for PDSYTRD
    #lwork = -1
    #info = -1
    #sl.pdsytrd("U", n, b, 1, 1, descr,
    #           d, e, tau, work, lwork, info)
    #
    ## Execute PDSYTRD with optimal workspace
    #lwork = int(work[0])
    #work = np.zeros((lwork), dtype=dtype, order='F')
    #
    #sl.pdsytrd("U", n, b, 1, 1, descr,
    #       d, e, tau, work, lwork, info)
    #
    ## Workspace query for PDGEBRD
    #lwork=-1
    #sl.pdgebrd(n, n, b, 1, 1, descr,
    #       d, e, taup, tauq, work,
    #       lwork, info)
    #
    ## Execute PDGEBRD with optimal workspace
    #lwork = int(work[0])
    #work = np.zeros((lwork), dtype=dtype, order='F')
    #sl.pdgebrd(n, n, b, 1, 1, descr,
    #       d, e, taup, tauq, work,
    #       lwork, info)
    #
    ## Workspace query for PDGEHRD
    #lwork=-1
    #sl.pdgehrd(n, 1, n, b, 1, 1, descr,
    #       tau, work, lwork, info)
    #
    ## Execute PDGEHRD with optimal workspace
    #lwork = int(work[0])
    #work = np.zeros((lwork), dtype=dtype, order='F')
    #sl.pdgehrd(n, 1, n, b, 1, 1, descr,
    #           tau, work, lwork, info)

    # Execution of PDSYEVX
    w = np.zeros(n, dtype=np.float64, order='F') # Eigenvalues
    z = np.zeros((n, n), dtype=np.float64, order='F') # Eigenvectors

    # Parameters for PDSYEVX
    jobz = "V"  
    uplo = "U"  
    ia = 1
    ja = 1 
    iz = 1
    jz = 1
    if vl is None:
        vl = 0.
        vu = 0.
        range_type = "A"
    else:
        vl = vl
        vu = vu
        range_type = "V"
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
    sl.pdsyevx("V", range_type, "U", n, b, ia, ja, descr, 
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
    sl.pdsyevx("V", range_type, "U", n, b, ia, ja, descr, 
               vl, vu, il, iu, abstol, m, nz, w, orfac, z, iz, jz, descr, 
               work, lwork, iwork, liwork, ifail, iclustr, gap, info)

    eigvals = w
    eigvecs = mpi_bcast_matrix(sl.gather_numpy(POINTER(c_int)(descr), z.ctypes.data_as(POINTER(c_double)), (n, n)))

    return eigvals, eigvecs

def pdgesvd_from_numpy_array(array):

    from scalapack4py import ScaLAPACK4py
    from ctypes import CDLL, POINTER, c_int
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    ntasks = comm.Get_size()
    rank = comm.Get_rank()
    
    sl = ScaLAPACK4py(CDLL(os.environ['ASI_LIB_PATH'], mode=RTLD_GLOBAL))

    m, n = np.shape(array)[0], np.shape(array)[1]
    dtype=np.float64

    array = array.astype(dtype=dtype, order='F') if MPI.COMM_WORLD.rank==0 else None

    size = min(m, n)

    MP, NP = find_squarest_grid(ntasks)
    ctx = sl.make_blacs_context(sl.get_default_system_context(), MP, NP)
    descr_a = sl.make_blacs_desc(ctx, m, n)
    descr_u = sl.make_blacs_desc(ctx, m, m)
    descr_vt = sl.make_blacs_desc(ctx, n, n)

    b = np.zeros((descr_a.locrow, descr_a.loccol), dtype=dtype, order='F')
    sl.scatter_numpy(array, POINTER(c_int)(descr_a), b.ctypes.data_as(POINTER(c_double)), b.dtype)

    s = np.zeros(size, dtype=dtype, order='F')
    u = np.zeros((descr_u.locrow, descr_u.loccol), dtype=dtype, order='F')
    vt = np.zeros((descr_vt.locrow, descr_vt.loccol), dtype=dtype, order='F')
    work = np.zeros(1, dtype=np.float64, order='F')

    # Workspace query for PDGESVD
    lwork = -1
    rwork = -1
    info = -1
    sl.pdgesvd("V", "V", m, n, b, 1, 1, descr_a,
               s, u, 1, 1, descr_u, vt, 1, 1, descr_vt,
               work, lwork, info) 

    # Execute PDGESVD with optimal workspace
    lwork = int(work[0])
    work = np.zeros((lwork), dtype=dtype, order='F')
    sl.pdgesvd("V", "V", m, n, b, 1, 1, descr_a,
               s, u, 1, 1, descr_u, vt, 1, 1, descr_vt,
               work, lwork, info)

    u_gather = mpi_bcast_matrix(sl.gather_numpy(POINTER(c_int)(descr_u), u.ctypes.data_as(POINTER(c_double)), (m, m)))
    s_vals = s
    vt_gather = mpi_bcast_matrix(sl.gather_numpy(POINTER(c_int)(descr_vt), vt.ctypes.data_as(POINTER(c_double)), (n, n)))

    return u_gather, s_vals, vt_gather
