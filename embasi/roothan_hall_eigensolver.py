import numpy as np

def invsqr_overlap_calc(overlap):

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

def calculate_occ_mat(eigenvalues, nelec, nspin):
    # This obviously won't work for smeared occupancies
    # Only valid for insulators
    occ_mat = np.zeros(np.size(eigenvalues))
    if nspin == 1:
        occ_mat[:int(nelec/2)] = 2.0
    if nspin == 2:
        occ_mat[:int(nelec/2)] = 1.0

    return occ_mat

def calculate_densmat(eigenvectors, occ_mat):

    import copy

    occ_evecs = copy.deepcopy(eigenvectors)
    for idx in range(np.size(occ_mat)):
        occ_evecs[:,idx] = occ_evecs[:,idx] * np.sqrt(occ_mat[idx])

    return occ_evecs @ occ_evecs.T

def overlap_illcondition_check(overlap, thresh, inv=True, return_mask=False):

    from scipy.linalg import eig_banded, eigh

    n_basis = np.shape(overlap)[0]

    ovlp_evals, ovlp_evecs = eigh(overlap)

    n_bad = (ovlp_evals < thresh).sum()
    n_good = np.shape(overlap)[0] - n_bad

    good_val_mask = (ovlp_evals > thresh)

    if n_bad > 0:
        # Transform overlap matrix
        ovlp_filtered = ovlp_evecs[:, good_val_mask]
        evals_filtered = ovlp_evals[good_val_mask]

        for idx in range(np.size(evals_filtered)):
            sqrt_ev = np.sqrt(evals_filtered[idx])

            if inv:
                ovlp_filtered[:, idx] = ovlp_filtered[:, idx]/sqrt_ev
            else:
                ovlp_filtered[:, idx] = ovlp_filtered[:, idx]*sqrt_ev

    else:
        if inv:
            sigma_sqrt = np.diag(ovlp_evals**(-0.5))
        else:
            sigma_sqrt = np.diag(ovlp_evals**(0.5))

        ovlp_filtered = ovlp_evecs @ sigma_sqrt @ ovlp_evecs.T

    if return_mask:
        return ovlp_filtered, n_bad, good_val_mask
    else:
        return ovlp_filtered, n_bad

def hamiltonian_eigensolv(hamiltonian, overlap, nelec, nspins=1, nkpts=1, basis_illcond_thresh=1e-5):

    from embasi.parallel_utils import root_print
    from .ks_array import SpinKpointArray

    thresh = basis_illcond_thresh
    n_basis = np.shape(overlap[0,0])[0]

    evals = {}
    evecs = {}
    for ispin in range(nspins):
        for ikpt in range(nkpts):
            xform_mat, n_bad = overlap_illcondition_check(overlap[ispin,ikpt], thresh)
            n_good = n_basis - n_bad

            evals[(ispin,ikpt)], evecs[(ispin,ikpt)] = np.linalg.eig(xform_hamiltonian(hamiltonian[ispin,ikpt], xform_mat))

            evecs[(ispin,ikpt)] = back_xform_evecs(evecs[(ispin,ikpt)], xform_mat)
            evals[(ispin,ikpt)], evecs[(ispin,ikpt)] = sort_eigvals_and_evecs(evals[(ispin,ikpt)], evecs[(ispin,ikpt)])

    # Just assume we're dealing with simple insulators for now
    # - fill from the bottom up

    # Only deal with spins for now - kpoints will need some way
    # to communicate k-indexed evals between nodes and also intelligently
    # compare eigenvalues
    occ_mat = {}
    if nspins > 1:
        remaining_electrons = int(round(nelec))
        alpha_nelecs = 0
        beta_nelecs = 0
        occ_mat[(0,0)] = np.zeros(np.size(evals[(0,0)]))
        occ_mat[(1,0)] = np.zeros(np.size(evals[(0,0)]))

        while remaining_electrons > 0:
            if evals[(0,0)][alpha_nelecs] < evals[(1,0)][beta_nelecs]:
                occ_mat[(0,0)][alpha_nelecs] = 1.0
                alpha_nelecs += 1
            else:
                occ_mat[(1,0)][beta_nelecs] = 1.0
                beta_nelecs += 1

            remaining_electrons += -1
    else:
        occ_mat[(0,0)] = np.zeros(np.size(evals[(0,0)]))
        occ_mat[(0,0)][:int(round(nelec/2))] = 2.0

    evecs = SpinKpointArray(evecs, nspins, nkpts)
    evals = SpinKpointArray(evals, nspins, nkpts)
    occ_mat = SpinKpointArray(occ_mat, nspins, nkpts)

    return evals, evecs, occ_mat
