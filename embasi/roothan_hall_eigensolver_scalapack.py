import numpy as np
from ctypes import RTLD_GLOBAL, CDLL, POINTER, byref, c_int, c_int64, c_int32, c_bool, c_double
from embasi.parallel_utils import root_print, mpi_bcast_matrix
import scalapack4py.npscal.math_utils.operations as op
import os

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

    occ_evecs = copy.copy(eigenvectors)
    # @TODONPSCAL: FIGURE OUT HOW THIS IS DONE LOCALLY
    for idx in range(np.size(occ_mat)):
        occ_evecs[:,idx] = occ_evecs[:,idx] * np.sqrt(occ_mat[idx])

    return occ_evecs @ occ_evecs.T

def overlap_illcondition_check_parallel(overlap, thresh, inv=True, return_mask=False):

    from scipy.linalg import eig_banded, eigh
    from embasi.parallel_utils import root_print
    from scalapack4py.npscal.math_utils.npscal2npscal import eig

    # @TODONPSCAL: REPLACE WITH GLOBAL SHAPE
    n_basis = overlap.gl_m

    # @TODONPSCAL: REPLACE ARGUMENT WITH NPSCAL
    ovlp_evals, ovlp_evecs = eig(overlap, vl=thresh, vu=100000)
    # Count non-singular values
    n_bad = (ovlp_evals < thresh).sum()
    n_good = overlap.gl_m - n_bad
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
                ovlp_filtered[:, idx] = ovlp_filtered[:, idx]*(1/sqrt_ev)
            else:
                # @TODONPSCAL: FIGURE OUT HOW THIS IS DONE LOCALLY
                ovlp_filtered[:, idx] = ovlp_filtered[:, idx]*sqrt_ev

    else:
        if inv:
            # @TODONPSCAL: REPLACE NUMPY DIRECTIVE
            ovlp_filtered = ovlp_evecs @ op.diag(ovlp_evals**(-0.5), ctxt_tag=overlap.ctxt_tag, descr_tag="main", lib=overlap.sl) @ ovlp_evecs.T
        else:
            ovlp_filtered = ovlp_evecs @ op.diag(ovlp_evals**(0.5), ctxt_tag=overlap.ctxt_tag, descr_tag="main", lib=overlap.sl) @ ovlp_evecs.T

    if return_mask:
        return ovlp_filtered, n_bad, good_val_mask
    else:
        return ovlp_filtered, n_bad

def hamiltonian_eigensolv_parallel(hamiltonian, overlap, nelec, return_orthog=False):

    from embasi.parallel_utils import root_print
    from scalapack4py.npscal.math_utils.npscal2npscal import eig    

    thresh = 1e-5

    n_basis = overlap.gl_m
    xform_mat, n_bad = overlap_illcondition_check_parallel(overlap, thresh)
    n_good = n_basis - n_bad
    evals, evecs = eig(xform_hamiltonian(hamiltonian, xform_mat))
    if return_orthog:
        #evals, evecs = sort_eigvals_and_evecs(evals, evecs)
        occ_mat = calculate_occ_mat(evals, nelec)
        
        return evals, evecs, occ_mat, xform_mat

    else:
        evecs = back_xform_evecs(evecs, xform_mat)

        evals, evecs = sort_eigvals_and_evecs(evals, evecs)
        occ_mat = calculate_occ_mat(evals, nelec)

        return evals, evecs, occ_mat
