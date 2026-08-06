"""Constructing Embedding Projectors

Constructs the two supported embedding projection
matrices in EmbASI:

  1) The Manby level-shift operator
  2) The Huzinaga projection operator

The Huzinaga projection operator also contains
functionality for calculating the operator in a
monomolecular basis, as is supported in the
absolute truncation workflow.

"""
import numpy as np
from embasi.ks_array import SpinKpointArray

def levelshift_projector(self, densmat, overlap, mu_val):
    """Calculates level-shift projection operator

    Calculate the level-shift based projection operator from
    Manby et al.[1]:
                P^{B} = /mu S^{AB} D^{B} S^{AB}
    where S^{AB} is the overlap matrix for the supermolecular system, and
    the density matrix for subsystem B.

    [1] Manby, F. R.; Stella, M.; Goodpaster, J. D.; Miller, T. F. I.
    A Simple, Exact Density-Functional-Theory Embedding Scheme.
    J. Chem. Theory Comput. 2012, 8 (8), 2564–2568.
    """
    return mu_val * (overlap @ densmat @ overlap)

def huzinaga_fock_operator(hamiltonian_hl, vemb, overlap, densmat, n_spins):
    """Calculates the full Huzinaga equation

    Parameters
    ----------
    hamiltonian_hl : SpinKpointArray or np.ndarray
        Fock/Hamiltonian matrix of the high-level (subsystem A). May be
        a full SpinKpointArray, or a single spin/k-point's bare matrix
        (e.g. as passed per-callback-invocation from an ASI callback) -
        in the latter case, pass n_spins explicitly, since a bare matrix
        carries no n_spins of its own.
    vemb : SpinKpointArray or np.ndarray
        The embedding potential calculated as H^AB - H^A
    overlap : SpinKpointArray or np.ndarray
        Supersystem overlap matrix (same shape convention as hamiltonian).
    densmat : SpinKpointArray or np.ndarray
        Density matrix of the environment (subsystem B) (same shape
        convention as hamiltonian).
    n_spins : int or None
        Number of spin channels, used to pick the -1.0 (unrestricted) vs
        -0.5 (restricted) prefactor. Defaults to hamiltonian.n_spins,
        which requires hamiltonian to be a SpinKpointArray; pass this
        explicitly when hamiltonian is a bare per-spin/k-point matrix.

    Returns
    -------
    SpinKpointArray or np.ndarray
        The full Huzinaga embedded Fock/Hamiltonian matrix
        (F_emb - (F_emb . \gamma^{B} . S.T + S . \gamma^{B} . F_emb.T))
    """

    if n_spins is None:
        n_spins = hamiltonian.n_spins

    f_emb = hamiltonian_hl + vemb

    huz_projector = huzinaga_projector(f_emb, overlap, densmat, n_spins=n_spins)

    return f_emb + huz_projector

def huzinaga_projector(hamiltonian, overlap, densmat, n_spins=None):
    """Calculates the Huzinaga projection operator

    Parameters
    ----------
    hamiltonian : SpinKpointArray or np.ndarray
        Fock/Hamiltonian matrix of the environment (subsystem B). May be
        a full SpinKpointArray, or a single spin/k-point's bare matrix
        (e.g. as passed per-callback-invocation from an ASI callback) -
        in the latter case, pass n_spins explicitly, since a bare matrix
        carries no n_spins of its own.
    overlap : SpinKpointArray or np.ndarray
        Supersystem overlap matrix (same shape convention as hamiltonian).
    densmat : SpinKpointArray or np.ndarray
        Density matrix of the environment (subsystem B) (same shape
        convention as hamiltonian).
    n_spins : int or None
        Number of spin channels, used to pick the -1.0 (unrestricted) vs
        -0.5 (restricted) prefactor. Defaults to hamiltonian.n_spins,
        which requires hamiltonian to be a SpinKpointArray; pass this
        explicitly when hamiltonian is a bare per-spin/k-point matrix.

    Returns
    -------
    SpinKpointArray or np.ndarray
        The Huzinaga projector, P^{B}, of the same type as the inputs.
    """

    if n_spins is None:
        n_spins = hamiltonian.n_spins

    if n_spins > 1:
        return -1.0 * ((hamiltonian @ densmat @ overlap.T) + (overlap @ densmat @ hamiltonian.T))
    else:
        return -0.5 * ((hamiltonian @ densmat @ overlap.T) + (overlap @ densmat @ hamiltonian.T))


def get_abs_trunc_indices(atomsembed):
    active_atoms = np.array(atomsembed.basis_info.active_atoms_mask)

    # First and last truncated atom
    trunc_at_first = np.argmax(active_atoms == True)
    trunc_at_last = len(active_atoms) - 1 - np.argmax((active_atoms == True)[::-1])
    # Find first and last active atom
    full_at_first = np.argmax(active_atoms == False)
    full_at_last = len(active_atoms) - 1 - np.argmax((active_atoms == False)[::-1])

    full_basis_min_idx = atomsembed.basis_info.full_basis_min_idx
    full_basis_max_idx = atomsembed.basis_info.full_basis_max_idx
    A_block_min = full_basis_min_idx[trunc_at_first]
    A_block_max = full_basis_max_idx[trunc_at_last]

    B_block_min = full_basis_min_idx[full_at_first]
    B_block_max = full_basis_max_idx[full_at_last]

    return A_block_min, A_block_max, B_block_min, B_block_max


def huzinaga_projector_abs_trunc(atomsembed):
    """Calculates the Huzinaga projector for the absolute-truncation/
    freeze-and-thaw workflow

    Used by AtomsEmbed.run_embasi_diag_emb_pot. Reads
    atomsembed.huzinaga_ovlp_in, atomsembed.huzinaga_dm_in and
    atomsembed.embedding_ham_in (the supersystem overlap, environment
    density and embedding Hamiltonian set up by run_emb_scf/
    run_embasi_diag_emb_pot), and - when atomsembed.truncate is set -
    first slices out the A-B coupling block via atomsembed.basis_info
    before forming the projector.

    Parameters
    ----------
    atomsembed : AtomsEmbed
        The subsystem AtomsEmbed instance (A_LL or B_LL) computing its
        own projector against the (implicit) other fragment.

    Returns
    -------
    SpinKpointArray
        The Huzinaga projector, P^{B}.
    """

    projector = {}

    if atomsembed.truncate:
        for PiS in range(atomsembed.n_spins):
            for PiK in range(atomsembed.n_kpoints):
                ovlp_supermol = atomsembed.huzinaga_ovlp_in[PiS, PiK]
                dm_supermol = atomsembed.huzinaga_dm_in[PiS, PiK]
                fock_supermol = atomsembed.embedding_ham_in[PiS, PiK]

                A_block_min, A_block_max, B_block_min, B_block_max = get_abs_trunc_indices(atomsembed)

                fmat_supermol = fock_supermol[A_block_min:A_block_max,B_block_min:B_block_max]
                dm_supermol = dm_supermol[B_block_min:B_block_max,B_block_min:B_block_max]
                ovlp_supermol = ovlp_supermol[A_block_min:A_block_max,B_block_min:B_block_max]

                projector[(PiS,PiK)] = huzinaga_projector(fmat_supermol, ovlp_supermol, dm_supermol,
                                                          n_spins=atomsembed.huzinaga_dm_in.n_spins)

    else:
        for PiS in range(atomsembed.n_spins):
            for PiK in range(atomsembed.n_kpoints):
                fock_supermol = atomsembed.embedding_ham_in[PiS, PiK]
                ovlp = atomsembed.huzinaga_ovlp_in[PiS, PiK]
                dm = atomsembed.huzinaga_dm_in[PiS, PiK]

                projector[(PiS,PiK)] = huzinaga_projector(fock_supermol, ovlp, dm,
                                                          n_spins=atomsembed.embedding_ham_in.n_spins)

    return SpinKpointArray(projector, atomsembed.n_spins, atomsembed.n_kpoints)
