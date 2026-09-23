import numpy as np
from ctypes import RTLD_GLOBAL, CDLL, POINTER, byref, c_int, c_int64, c_int32, c_bool, c_double
from embasi.parallel_utils import root_print, mpi_bcast_matrix
from scalapack4py.npscal import rechunk
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
    for idx in range(np.size(occ_mat)):
        occ_evecs[:,idx] = occ_evecs[:,idx] * np.sqrt(occ_mat[idx])

    return occ_evecs @ occ_evecs.T

def overlap_illcondition_check_parallel(overlap, thresh, inv=True, return_mask=False):

    from scipy.linalg import eig_banded, eigh
    from embasi.parallel_utils import root_print
    from scalapack4py.npscal.math_utils.npscal2npscal import eig

    n_basis = overlap.gl_m
    ovlp_evals, ovlp_evecs = eig(overlap, vl=thresh, vu=100000)

    # Count non-singular values
    n_bad = (ovlp_evals < thresh).sum()
    n_good = overlap.gl_m - n_bad
    good_val_mask = (ovlp_evals > thresh)
    if n_bad > 0:
        # Transform overlap matrix
        #
        # ovlp_evecs[:, good_val_mask] goes through select_slice, which picks
        # its own independently-"optimal" block size for the (n_good-wide)
        # sub-matrix -- there is no implicit mechanism reconciling that back
        # to overlap's. The explicit rechunk() below at the end of this
        # function is what actually guarantees the returned xform_mat is
        # consistent with overlap.
        ovlp_filtered = ovlp_evecs[:, good_val_mask]
        evals_filtered = ovlp_evals[good_val_mask]

        if inv:
            evals_diag = op.diag(evals_filtered**(-0.5), ctxt_tag=overlap.ctxt_tag, descr_tag="rank_reduced_eval", lib=overlap.sl,
                                 dmb=overlap.descr.mb, dnb=overlap.descr.nb)
            ovlp_filtered = ovlp_filtered.copy() @ evals_diag
        else:
            evals_diag = op.diag(evals_filtered**(0.5), ctxt_tag=overlap.ctxt_tag, descr_tag="rank_reduced_eval", lib=overlap.sl,
                                 dmb=overlap.descr.mb, dnb=overlap.descr.nb)
            ovlp_filtered = evals_diag @ ovlp_filtered.copy().T

    else:
        if inv:
            ovlp_filtered = ovlp_evecs @ op.diag(ovlp_evals**(-0.5), ctxt_tag=overlap.ctxt_tag, descr_tag=f"main_{overlap.gl_m}", lib=overlap.sl,
                                                 dmb=overlap.descr.mb, dnb=overlap.descr.nb) @ ovlp_evecs.T
        else:
            ovlp_filtered = ovlp_evecs @ op.diag(ovlp_evals**(0.5), ctxt_tag=overlap.ctxt_tag, descr_tag=f"main_{overlap.gl_m}", lib=overlap.sl,
                                                 dmb=overlap.descr.mb, dnb=overlap.descr.nb) @ ovlp_evecs.T

    # matmul() always inherits its LEFT operand's block size (ovlp_filtered's,
    # or ovlp_evecs's own slice-recomputed one in the n_bad==0 branch), never
    # evals_diag's -- so passing overlap's block size into evals_diag above
    # does not, by itself, fix the block size actually carried out of here.
    # This function returns xform_mat as a value meant to interoperate with
    # overlap (and, through the eigensolve, with arrays read straight off an
    # ASI callback under the same context) -- rechunk explicitly onto it
    # rather than leaving that to whoever calls this.
    ovlp_filtered = rechunk(ovlp_filtered, like=overlap)

    if return_mask:
        return ovlp_filtered, n_bad, good_val_mask
    else:
        return ovlp_filtered, n_bad

def hamiltonian_eigensolv_parallel(hamiltonian, overlap, nelec, nspins=1, nkpts=1, return_orthog=False, basis_illcond_thresh=1e-5, spin=None):

    from embasi.parallel_utils import root_print
    from scalapack4py.npscal.math_utils.npscal2npscal import eig
    from .ks_array import SpinKpointArray
    from .roothan_hall_eigensolver import fill_occupations

    n_basis = overlap[0,0].gl_m

    evals = {}
    evecs = {}
    if return_orthog:
        evecs_orthog = {}

    for ispin in range(nspins):
        for ikpt in range(nkpts):
            xform_mat, n_bad = overlap_illcondition_check_parallel(overlap[ispin,ikpt], basis_illcond_thresh)
            n_good = n_basis - n_bad

            evals[(ispin,ikpt)], evecs[(ispin,ikpt)] = eig(xform_hamiltonian(hamiltonian[ispin,ikpt], xform_mat))

            if (not return_orthog):
                idx = np.argsort(evals[(ispin,ikpt)])
                evals[(ispin,ikpt)] = evals[(ispin,ikpt)][idx]
                evecs[(ispin,ikpt)] = back_xform_evecs(evecs[(ispin,ikpt)], xform_mat)[:,idx]
            else:
                evecs_orthog[(ispin,ikpt)] = evecs[(ispin,ikpt)].copy()
                idx = np.argsort(evals[(ispin,ikpt)])
                evals[(ispin,ikpt)] = evals[(ispin,ikpt)][idx]
                evecs[(ispin,ikpt)] = back_xform_evecs(evecs[(ispin,ikpt)], xform_mat)[:,idx]
                evecs_orthog[(ispin,ikpt)] = evecs_orthog[(ispin,ikpt)][:,idx]

    # Just assume we're dealing with simple insulators for now
    # - fill from the bottom up

    # Only deal with spins for now - kpoints will need some way
    # to communicate k-indexed evals between nodes and also intelligently
    # compare eigenvalues
    occ_mat = fill_occupations(evals, nelec, nspins, spin)

    evecs = SpinKpointArray(evecs, nspins, nkpts)
    evals = SpinKpointArray(evals, nspins, nkpts)
    occ_mat = SpinKpointArray(occ_mat, nspins, nkpts)

    if return_orthog:
        evecs_orthog = SpinKpointArray(evecs_orthog, nspins, nkpts)
        return evals, evecs, evecs_orthog, occ_mat
    else:
        return evals, evecs, occ_mat
