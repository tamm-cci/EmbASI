import numpy as np
from embasi.ks_array import dagger


def invsqr_overlap_calc(overlap):

    sigma, U = np.linalg.eigh(overlap)
    sigma_sqrt = np.diag(sigma**(-0.5))

    return U @ sigma_sqrt @ dagger(U)

def xform_hamiltonian(hamiltonian, xform_mat):

    from embasi.parallel_utils import root_print

    return dagger(xform_mat) @ hamiltonian @ xform_mat

def back_xform_evecs(eigenvectors, xform_mat):

    return xform_mat @ eigenvectors

#def sort_eigvals_and_evecs(eigenvalues, eigenvectors):
#    idx = np.argsort(eigenvalues)
#    return eigenvalues[idx], eigenvectors[:,idx]

def calculate_occ_mat(eigenvalues, nelec, nspin):
    # This obviously won't work for smeared occupancies
    # Only valid for insulators
    occ_mat = np.zeros(np.size(eigenvalues))
    if nspin == 1:
        occ_mat[:round(nelec/2)] = 2.0
    if nspin == 2:
        occ_mat[:round(nelec/2)] = 1.0

    return occ_mat

def calculate_densmat(eigenvectors, occ_mat):

    import copy

    occ_evecs = copy.deepcopy(eigenvectors)
    for idx in range(np.size(occ_mat)):
        occ_evecs[:,idx] = occ_evecs[:,idx] * np.sqrt(occ_mat[idx])

    return occ_evecs @ dagger(occ_evecs)

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

        ovlp_filtered = ovlp_evecs @ sigma_sqrt @ dagger(ovlp_evecs)

    if return_mask:
        return ovlp_filtered, n_bad, good_val_mask
    else:
        return ovlp_filtered, n_bad

def hamiltonian_eigensolv(hamiltonian, overlap, nelec, nspins=1, nkpts=1, basis_illcond_thresh=1e-5, return_orthog=False, spin=None):

    from embasi.parallel_utils import root_print
    from .ks_array import SpinKpointArray

    thresh = basis_illcond_thresh
    n_basis = np.shape(overlap[0,0])[0]

    evals = {}
    evecs = {}
    if return_orthog:
        evecs_orthog = {}

    for ispin in range(nspins):
        for ikpt in range(nkpts):
            xform_mat, n_bad = overlap_illcondition_check(overlap[ispin,ikpt], thresh)
            n_good = n_basis - n_bad

            # X^dagger H X is Hermitian: eigh gives real eigenvalues and
            # orthonormal (complex, at k != Gamma) eigenvectors, including
            # within degenerate levels, which eig does not guarantee
            evals[(ispin,ikpt)], evecs[(ispin,ikpt)] = np.linalg.eigh(xform_hamiltonian(hamiltonian[ispin,ikpt], xform_mat))

            if (not return_orthog):
                evecs[(ispin,ikpt)] = back_xform_evecs(evecs[(ispin,ikpt)], xform_mat)
                idx = np.argsort(evals[(ispin,ikpt)])
                evals[(ispin,ikpt)] = evals[(ispin,ikpt)][idx]
                evecs[(ispin,ikpt)] = evecs[(ispin,ikpt)][:,idx]
            else:
                evecs_orthog[(ispin,ikpt)] = evecs[(ispin,ikpt)]
                evecs[(ispin,ikpt)] = back_xform_evecs(evecs[(ispin,ikpt)], xform_mat)
                idx = np.argsort(evals[(ispin,ikpt)])
                evals[(ispin,ikpt)] = evals[(ispin,ikpt)][idx]
                evecs[(ispin,ikpt)] = evecs[(ispin,ikpt)][:,idx]
                evecs_orthog[(ispin,ikpt)] = evecs_orthog[(ispin,ikpt)][:,idx]

    # Insulators only: the same number of states is filled at every k-point
    occ_mat = fill_occupations(evals, nelec, nspins, spin)

    evecs = SpinKpointArray(evecs, nspins, nkpts)
    evals = SpinKpointArray(evals, nspins, nkpts)
    occ_mat = SpinKpointArray(occ_mat, nspins, nkpts)

    if return_orthog:
        evecs_orthog = SpinKpointArray(evecs_orthog, nspins, nkpts)
        return evals, evecs, evecs_orthog, occ_mat
    else:
        return evals, evecs, occ_mat


def fill_occupations(evals, nelec, nspins, spin=None):
    """Occupation matrix for the (sorted) eigenvalues of a re-diagonalised Fock.

    With ``spin`` (Nalpha - Nbeta) known, each channel is filled on its own:
    Nalpha = (N + S)/2 and Nbeta = (N - S)/2 lowest states - the fixed-spin
    aufbau the SCF itself used, so a converged Fock reproduces the SCF's
    occupations exactly.

    Without it, falls back to a cross-channel aufbau on the total count alone.
    That cannot represent a state whose occupied levels in one channel lie
    above empty levels in the other - e.g. any excited-configuration triplet
    of a closed-shell molecule, where it silently returns the singlet's
    Nalpha == Nbeta filling. Pass ``spin`` whenever the caller knows it.

    With several k-points, N is the number of electrons per cell and the same
    number of states is filled in each channel at every k-point: an insulator,
    where every band is either full or empty across the Brillouin zone. The
    cross-channel aufbau is done on the first k-point. A warning is printed if
    a filled level at one k-point lies above an empty level at another (a
    metal, which fixed per-k filling cannot describe).
    """
    from embasi.parallel_utils import root_print

    kpts = sorted({ikpt for (_, ikpt) in evals.keys()})
    n_states = np.size(evals[(0,0)])
    if nspins > 1:
        n_total = int(round(nelec))
        if spin is not None:
            s = int(round(spin))
            if abs(s) > n_total or (n_total - s) % 2 != 0:
                raise ValueError(f"spin {s} is incompatible with {n_total} electrons")
            n_alpha, n_beta = (n_total + s) // 2, (n_total - s) // 2
            if max(n_alpha, n_beta) > n_states:
                raise ValueError(f"({n_alpha}, {n_beta}) electrons do not fit {n_states} states")
        else:
            n_alpha = 0
            n_beta = 0
            for _ in range(n_total):
                if evals[(0,0)][n_alpha] < evals[(1,0)][n_beta]:
                    n_alpha += 1
                else:
                    n_beta += 1
        n_filled = {0: n_alpha, 1: n_beta}
        occ_value = 1.0
    else:
        n_filled = {0: round(nelec/2)}
        occ_value = 2.0

    occ_mat = {}
    for ispin, n_occ in n_filled.items():
        for ikpt in kpts:
            occ_mat[(ispin, ikpt)] = np.zeros(n_states)
            occ_mat[(ispin, ikpt)][:n_occ] = occ_value

    if len(kpts) > 1:
        homo = max(evals[(ispin, ikpt)][n_occ - 1] for ispin, n_occ in n_filled.items()
                   if n_occ > 0 for ikpt in kpts)
        lumo = min(evals[(ispin, ikpt)][n_occ] for ispin, n_occ in n_filled.items()
                   if n_occ < n_states for ikpt in kpts)
        if homo >= lumo:
            root_print(f"WARNING: highest filled level ({homo:.6f}) is not below the lowest "
                       f"empty level ({lumo:.6f}) across the k-points: no band gap, so "
                       "filling the same number of states at every k-point is not valid")
    return occ_mat
