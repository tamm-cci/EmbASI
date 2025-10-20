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

def calculate_occ_mat(eigenvalues, nelec):

    occ_mat = np.zeros(np.size(eigenvalues))
    occ_mat[:int(nelec/2)] = 2.0

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

def hamiltonian_eigensolv(hamiltonian, overlap, nelec, return_orthog=False):

    from embasi.parallel_utils import root_print

    thresh = 1e-8
    n_basis = np.shape(overlap)[0]
    xform_mat, n_bad = overlap_illcondition_check(overlap, thresh)
    n_good = n_basis - n_bad

    evals, evecs = np.linalg.eig(xform_hamiltonian(hamiltonian, xform_mat))

    if return_orthog:
        evals, evecs = sort_eigvals_and_evecs(evals, evecs)
        occ_mat = calculate_occ_mat(evals, nelec)
        
        return evals, evecs, occ_mat, xform_mat

    else:
        evecs = back_xform_evecs(evecs, xform_mat)

        evals, evecs = sort_eigvals_and_evecs(evals, evecs)
        occ_mat = calculate_occ_mat(evals, nelec)

        return evals, evecs, occ_mat

