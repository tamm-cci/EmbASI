import numpy as np

def invsqr_overlap_calc(overlap):

    sigma, U = np.linalg.eigh(overlap)
    sigma_sqrt = np.diag(sigma**(-0.5))

    return U @ sigma_sqrt @ U.T

def xform_hamiltonian(hamiltonian, xform_mat):

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

def hamiltonian_eigensolv(hamiltonian, overlap, nelec):

    xform_mat = invsqr_overlap_calc(overlap)

    evals, evecs = np.linalg.eig(xform_hamiltonian(hamiltonian, xform_mat))
    evecs = back_xform_evecs(evecs, xform_mat)

    evals, evecs = sort_eigvals_and_evecs(evals, evecs)
    occ_mat = calculate_occ_mat(evals, nelec)

    return evals, evecs, occ_mat

