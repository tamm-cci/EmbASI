import numpy as np
from embasi.parallel_utils import root_print, mpi_bcast_matrix
from embasi.roothan_hall_eigensolver import hamiltonian_eigensolv


def uno_spade_localisation(atomsembed, hamiltonian, overlap, parallel=False,
                           spade_ncores=0, spade_manual_state=0,
                           basis_illcond_thresh=1e-5, return_mo_coeffs=False,
                           a_nspade_mos=None, occ_window=(0.02, 1.98), n_env=None):
    """SPADE partition of the doubly occupied unrestricted natural orbitals (UNOs).

    For a spin-unrestricted reference, plain SPADE partitions the alpha and
    beta occupied orbitals separately. On a spin-polarised system the two
    environments then span different spaces (rotated by several degrees on a
    triplet or broken-symmetry singlet), so no single spatial orbital set is
    orthogonal to both, and a spin-adapted treatment of subsystem A cannot be
    embedded without leaking into one channel's environment.

    Here the environment is instead built from the doubly occupied block of the
    supersystem UNOs, making it one closed-shell set of spatial orbitals
    shared by both spins:

    1. The supersystem Fock matrix is diagonalised per spin and filled with
       each channel's own electron count (the SCF's spin).
    2. UNOs: natural orbitals of the spin-summed density. Occupations above
       ``occ_window[1]`` form the doubly occupied block; those at or below it
       (fractional ones, and singly occupied ones at ~1) carry all the spin
       and static correlation.
    3. SPADE on the doubly occupied block only: the SVD of its fragment rows
       (symmetric Loewdin basis, as in ``spade_localisation``) ranks the
       orbitals by fragment character; the top ``k`` go to subsystem A, the
       rest form the environment B. ``k`` is set by ``n_env`` if given (B gets
       exactly ``n_env`` orbitals, A the remaining doubly occupied ones), else by
       ``a_nspade_mos`` (the number of DOUBLY OCCUPIED orbitals assigned to A),
       else by the largest gap in the squared singular values, offset by
       ``spade_manual_state``. Fixing ``n_env`` keeps the environment the same
       size along a scan: the doubly occupied count itself changes with
       geometry as orbitals become fractional, so a fixed ``a_nspade_mos`` does
       not.
    4. B's density is ``C_B C_B^T`` in both spin channels. A's density is the
       remainder ``D_sigma - C_B C_B^T``, so ``D_A + D_B`` reproduces the
       supersystem density exactly, and A holds all fractional and singly
       occupied density: A's spin is the supersystem spin and B's is zero.
    5. A's occupied orbitals per spin (the returned MO coefficients) are the
       part of that channel's occupied space S-orthogonal to B.

    The closed-shell (n_spins == 1) case has only doubly occupied UNOs, where
    this reduces to plain SPADE; it is delegated to ``spade_localisation``
    unchanged.

    Parameters
    ----------
    atomsembed : AtomsEmbed
        The supersystem AtomsEmbed instance (AB_LL).
    hamiltonian, overlap : SpinKpointArray
        Supersystem Fock and overlap matrices.
    occ_window : tuple of float
        UNO occupations strictly above ``occ_window[1]`` are doubly occupied;
        everything else stays in subsystem A (``occ_window[0]`` is reported
        only, for the fractional count).
    n_env : int or None
        Number of doubly occupied UNOs assigned to the environment B. Mutually
        exclusive with ``a_nspade_mos``.

    Returns
    -------
    density_matrix_subsys_a, density_matrix_subsys_b : SpinKpointArray
    rot_evecs_occ_a, rot_evecs_occ_b : SpinKpointArray
        Returned with ``return_mo_coeffs``: per-spin occupied orbitals of A,
        and the environment orbitals (identical in both channels).
    """
    from embasi.ks_array import SpinKpointArray

    if atomsembed.n_spins == 1:
        from embasi.spade_localisation import spade_localisation
        root_print('UNO-SPADE: closed-shell reference, all UNOs doubly occupied -> plain SPADE')
        return spade_localisation(atomsembed, hamiltonian, overlap, parallel=parallel,
                                  spade_ncores=spade_ncores,
                                  spade_manual_state=spade_manual_state,
                                  basis_illcond_thresh=basis_illcond_thresh,
                                  return_mo_coeffs=return_mo_coeffs,
                                  a_nspade_mos=a_nspade_mos)
    if parallel:
        raise NotImplementedError("UNO-SPADE is implemented for the serial path only")
    if spade_ncores > 0:
        raise NotImplementedError("UNO-SPADE does not support separate core localisation")
    if atomsembed.n_kpoints != 1:
        raise NotImplementedError("UNO-SPADE is implemented for a single k-point only")

    root_print('Starting UNO-SPADE localisation...')
    nelecs = atomsembed.free_atom_nelectrons - atomsembed.input_total_charge
    target_spin = atomsembed.fragment_spin
    if target_spin is None:
        raise ValueError("UNO-SPADE needs the supersystem spin; the QM adapter reports none")

    s = np.asarray(overlap[0, 0])
    s_evals, s_evecs = np.linalg.eigh(s)
    if s_evals.min() < basis_illcond_thresh:
        raise NotImplementedError(
            f"UNO-SPADE needs a well-conditioned overlap (min eigenvalue {s_evals.min():.2e} "
            f"< {basis_illcond_thresh})")
    s_half = s_evecs @ np.diag(np.sqrt(s_evals)) @ s_evecs.T
    s_mhalf = s_evecs @ np.diag(1.0 / np.sqrt(s_evals)) @ s_evecs.T

    # 1. Per-spin occupied orbitals of the supersystem, filled from the SCF's spin.
    _evals, evecs, occ_mat = hamiltonian_eigensolv(hamiltonian, overlap, nelecs,
                                                   nspins=2, nkpts=1,
                                                   basis_illcond_thresh=basis_illcond_thresh,
                                                   spin=target_spin)
    c_occ = [np.asarray(evecs[k, 0])[:, np.asarray(occ_mat[k, 0]) > 0] for k in (0, 1)]
    n_occ = [c.shape[1] for c in c_occ]
    dm = [c @ c.T for c in c_occ]

    # 2. UNOs (Loewdin representation u, AO coefficients c_no), descending occupation.
    occ, u = np.linalg.eigh(s_half @ (dm[0] + dm[1]) @ s_half)
    order = np.argsort(occ)[::-1]
    occ, u = occ[order], u[:, order]
    dbl = occ > occ_window[1]
    n_dbl = int(dbl.sum())
    n_frac = int(((occ > occ_window[0]) & ~dbl).sum())
    if n_dbl > min(n_occ):
        raise ValueError(
            f"{n_dbl} UNOs above {occ_window[1]} exceed the beta occupied count {min(n_occ)}")

    # 3. SPADE on the doubly occupied block's fragment rows.
    mask_val = np.array([atomsembed.embed_mask[b] == 1
                         for b in atomsembed.basis_info.full_basis_atoms])
    u_dbl = u[:, dbl]
    _u, svals, vt = np.linalg.svd(u_dbl[mask_val, :], full_matrices=True)
    if n_env is not None and a_nspade_mos is not None:
        raise ValueError("UNO-SPADE: give n_env or a_nspade_mos, not both")
    if n_env is not None:
        if not 0 <= int(n_env) <= n_dbl:
            raise ValueError(
                f"UNO-SPADE: n_env={n_env} outside [0, {n_dbl}] doubly occupied UNOs "
                f"(occupations above {occ_window[1]}); the environment cannot be that size "
                "at this geometry")
        k = n_dbl - int(n_env)
    elif a_nspade_mos is not None:
        k = int(a_nspade_mos)
    else:
        k = int(np.argmax(np.abs(np.ediff1d(svals**2))) + spade_manual_state + 1)
    if not 0 <= k <= n_dbl:
        raise ValueError(f"UNO-SPADE cut {k} outside [0, {n_dbl}] doubly occupied UNOs")
    c_dbl = s_mhalf @ u_dbl
    c_env = c_dbl @ vt[k:, :].T
    n_env = c_env.shape[1]

    # 4. Densities: closed-shell B shared by both spins; A is the remainder.
    dm_env = c_env @ c_env.T
    dm_a = [d - dm_env for d in dm]

    # 5. A's occupied orbitals per spin: the channel's occupied space with B projected out.
    proj = np.eye(s.shape[0]) - dm_env @ s
    c_a, kept = [], []
    for k_spin in (0, 1):
        y = proj @ c_occ[k_spin]
        w, v = np.linalg.eigh(y.T @ s @ y)
        n_a = n_occ[k_spin] - n_env
        top = np.argsort(w)[::-1][:n_a]
        c_a.append(y @ v[:, top] @ np.diag(1.0 / np.sqrt(w[top])))
        kept.append(float(w[top].min()) if n_a else 1.0)

    root_print(f'UNO-SPADE: {n_dbl} doubly occupied, {n_frac} fractional UNOs '
               f'(window {occ_window}); fractional occupations: '
               + ' '.join(f'{x:.3f}' for x in occ[(occ > occ_window[0]) & ~dbl]))
    root_print(f'UNO-SPADE: {k} doubly occupied UNOs to subsystem A, {n_env} to B; '
               f'A occupied per spin {n_occ[0] - n_env}/{n_occ[1] - n_env}; '
               f'smallest occupied-space weight outside B {min(kept):.6f}')

    data_a = {(k_spin, 0): mpi_bcast_matrix(dm_a[k_spin]) for k_spin in (0, 1)}
    data_b = {(k_spin, 0): mpi_bcast_matrix(dm_env.copy()) for k_spin in (0, 1)}
    density_matrix_subsys_a = SpinKpointArray(data_a, 2, 1).rechunk(like=overlap)
    density_matrix_subsys_b = SpinKpointArray(data_b, 2, 1).rechunk(like=overlap)

    root_print('Exiting UNO-SPADE localisation...')
    if not return_mo_coeffs:
        return density_matrix_subsys_a, density_matrix_subsys_b
    rot_evecs_occ_a = SpinKpointArray({(k_spin, 0): c_a[k_spin] for k_spin in (0, 1)}, 2, 1)
    rot_evecs_occ_b = SpinKpointArray({(k_spin, 0): c_env.copy() for k_spin in (0, 1)}, 2, 1)
    return density_matrix_subsys_a, density_matrix_subsys_b, rot_evecs_occ_a, rot_evecs_occ_b
