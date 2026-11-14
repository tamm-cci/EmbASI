import numpy as np
from embasi.ks_array import dagger
from embasi.kpoint_parallel import gather_by_key
from embasi.parallel_utils import root_print, mpi_bcast_matrix
from embasi.roothan_hall_eigensolver import hamiltonian_eigensolv
from embasi.roothan_hall_eigensolver_scalapack import hamiltonian_eigensolv_parallel
from scalapack4py.npscal.linalg import svd


def spade_localisation(atomsembed, hamiltonian, overlap, parallel=False,
                       spade_ncores=0, spade_manual_state=0,
                       basis_illcond_thresh=1e-5, return_mo_coeffs=False,
                       a_nspade_mos=None):
    """Calculate the localised density matrices with the SPADE method

    As the eigenvectors (MO coefficient matrix) is not a part of the
    ASI specification, we solve the Roothan-Hall eigenvalue problem
    and construct the density matrix at the wrapper level.

    Parameters
    ----------
    atomsembed : AtomsEmbed
        The supersystem AtomsEmbed instance (e.g. AB_LL) providing the
        electron count, spin/k-point structure, embedding mask and basis
        info needed to run and partition the eigensolve.
    hamiltonian : SpinKpointArray
        Supersystem Hamiltonian to diagonalise.
    overlap : SpinKpointArray
        Supersystem overlap matrix.
    parallel : bool
        Whether to use the ScaLAPACK-distributed eigensolver/SVD path.
    spade_ncores : int
        Number of core states localised separately from the valence
        states. 0 to skip separate core-valence localisation.
    spade_manual_state : int
        Manual offset applied to the automatically detected SPADE state
        (the singular value gap used to split occupied MOs between
        subsystems A and B).
    basis_illcond_thresh : float
        Threshold below which near-linear-dependent basis functions are
        discarded during the eigensolve.
    output_coeffs : bool
        Determines whether the final MO coefficients should be output
    Returns
    -------
    density_matrix_subsys_a, density_matrix_subsys_b : SpinKpointArray
        Localised density matrices for subsystems A and B.
    rot_evecs_occ_a, rot_evecs_occ_a : SpinKpointArray
        Rotated eigenvectors for if ou

    """

    # TODO: @SPIN AND K-POINT LOOP - and needs syncing?? - SHOULD WE JUST PLACE THE LOOP AROUND THIS ROUTINE?
    root_print('Starting SPADE localisation...')

    nelecs = atomsembed.free_atom_nelectrons - atomsembed.input_total_charge

    # For an open-shell (n_spins==2) reference, the supersystem spin
    # (Nalpha - Nbeta) is the spin the converged calculation was run at
    # (atomsembed.fragment_spin == the user's mol.spin for AB_LL). It is
    # passed into the eigensolve so each channel is filled with its own
    # electron count, and the same value sets the beta-channel SPADE cut
    # below, so subsystem A carries all of the spin and subsystem B none.
    # Filling each channel separately also represents states whose occupied
    # levels in one channel lie above empty levels in the other (e.g. the
    # pi->pi* triplet of a nitrile: alpha pi* at +0.20 Ha occupied, beta pi
    # at -0.24 Ha empty), which a cross-channel aufbau on the total count
    # would fill as the singlet.
    target_spin = atomsembed.fragment_spin if atomsembed.n_spins == 2 else None
    if parallel:
        evals, evecs, evecs_orthog, occ_mat = hamiltonian_eigensolv_parallel(hamiltonian, \
                                                                             overlap, \
                                                                             nelecs, \
                                                                             nspins=atomsembed.n_spins, \
                                                                             nkpts=atomsembed.n_kpoints, \
                                                                             basis_illcond_thresh=basis_illcond_thresh,
                                                                             return_orthog=True,
                                                                             spin=target_spin)
    else:
        evals, evecs, evecs_orthog, occ_mat = hamiltonian_eigensolv(hamiltonian, \
                                                      overlap, \
                                                      nelecs, \
                                                      nspins=atomsembed.n_spins,
                                                      nkpts=atomsembed.n_kpoints,
                                                      basis_illcond_thresh=basis_illcond_thresh,
                                                      return_orthog=True,
                                                      spin=target_spin)


    mask_val = []

    for idx, basis2atom in enumerate(atomsembed.basis_info.full_basis_atoms):
        if atomsembed.embed_mask[basis2atom]==1:
            mask_val.append(True)
        else:
            mask_val.append(False)

    mask_val = np.array(mask_val)

    evecs_occ_ab = evecs.copy()
    rot_evecs_occ_a = evecs.copy()
    rot_evecs_occ_b = evecs.copy()

    kpt_weights = overlap.kpt_weights

    def spade_svds(state_min, state_max):
        """SVD of the subsystem-A rows of the orthogonalised occupied MOs
        state_min..state_max (per spin channel; the upper limit may be None
        for each channel's own occupied count), at every (spin, k-point)"""
        out = {}
        for ispin, ikpt in occ_mat.keys():
            top = np.count_nonzero(occ_mat[ispin, ikpt]) if state_max is None else state_max
            evecs_occ_a_orthog = evecs_orthog[ispin, ikpt, :, state_min:top][mask_val, :]
            if parallel:
                u, svals, vh = svd(evecs_occ_a_orthog)
            else:
                u, svals, vh = np.linalg.svd(evecs_occ_a_orthog, full_matrices=True)
            out[(ispin, ikpt)] = (np.asarray(svals), vh, top)
        return out

    def common_cut(svds, ispin, manual_state):
        """One SPADE cut for every k-point of a spin channel, at the largest
        gap of the k-point-weighted mean of the squared singular values: the
        same number of subsystem-A orbitals at every k-point, so A has an
        integer number of electrons per cell. With one k-point this is the
        usual largest-gap cut."""
        local = {(s, k): kpt_weights[k] * svds[(s, k)][0]**2.0 for (s, k) in svds if s == ispin}
        if atomsembed.n_kpoints > 1:
            # k-points of other process groups too (collective: every process
            # calls this for every spin channel)
            local = gather_by_key(local)
        sv2 = sum(local[key] for key in sorted(local))
        return int(np.argmax(np.abs(np.ediff1d(sv2)))) + manual_state + 1

    def n_occupied(ispin):
        """Occupied states of a spin channel: the same at every k-point
        (fill_occupations), so read from any k-point, wherever it is held"""
        local = {(s, k): int(np.count_nonzero(occ_mat[s, k])) for (s, k) in occ_mat.keys() if s == ispin}
        if atomsembed.n_kpoints > 1:
            local = gather_by_key(local)
        return next(iter(local.values()))

    def rotate(svds, ispin, cut, state_min):
        for ikpt in sorted(k for (s, k) in svds if s == ispin):
            _, vh, top = svds[(ispin, ikpt)]
            evecs_occ = evecs[ispin, ikpt, :, state_min:top]
            # vh holds the right singular vectors as conjugated rows
            rot_evecs_occ_a[ispin, ikpt] = evecs_occ @ dagger(vh[:cut, :])
            rot_evecs_occ_b[ispin, ikpt] = evecs_occ @ dagger(vh[cut:, :])
            evecs_occ_ab[ispin, ikpt] = evecs_occ.copy()

    def densmats(scale):
        return (scale * (evecs_occ_ab @ evecs_occ_ab.H),
                scale * (rot_evecs_occ_a @ rot_evecs_occ_a.H),
                scale * (rot_evecs_occ_b @ rot_evecs_occ_b.H))

    occ_scale = 2.0 if atomsembed.n_spins == 1 else 1.0

    # TODO: Clean this up and add automatic detection based on eigenvalues.
    if spade_ncores > 0:
        root_print("Performing separate core-valence SPADE localisation.")
        core_svds = spade_svds(0, spade_ncores)
        for ispin in range(atomsembed.n_spins):
            cut = common_cut(core_svds, ispin, 0)
            root_print(f'MAX OCC CORE STATE {spade_ncores} for Spin Channel {ispin}')
            root_print(f'Maximum SPADE core state for subsystem A, Spin Channel {ispin}, '
                       f'all {atomsembed.n_kpoints} k-point(s): {cut}')
            rotate(core_svds, ispin, cut, 0)

        density_matrix_supersystem, density_matrix_subsys_a, density_matrix_subsys_b = \
            densmats(occ_scale)

        root_print(f'SPADE CORE localised subsystem A charge: {(overlap @ density_matrix_subsys_a).trace()}')
        root_print(f'SPADE CORE localised subsystem B charge: {(overlap @ density_matrix_subsys_b).trace()}')

    supersystem_spin = None
    if atomsembed.n_spins == 2:
        n_occ_alpha_total = n_occupied(0)
        n_occ_beta_total = n_occupied(1)
        supersystem_spin = n_occ_alpha_total - n_occ_beta_total
        if target_spin is None:
            # The QM adapter does not expose the spin it ran at (e.g. FHI-aims,
            # whose adapter does not override get_qm_input_spin), so the
            # occupations come from the cross-channel aufbau. Best effort; see
            # fill_occupations.
            root_print('WARNING: supersystem spin unknown to the QM adapter; '
                       'using the cross-channel aufbau split, which cannot '
                       'represent excited-configuration open shells.')
        elif supersystem_spin != int(round(target_spin)):
            raise Exception(
                f"SPADE occupations give Nalpha-Nbeta={supersystem_spin} but the "
                f"calculation ran at spin {target_spin}")
        root_print(f'Supersystem spin (Nalpha-Nbeta), to be carried entirely '
                   f'by subsystem A: {supersystem_spin} '
                   f'(Nalpha={n_occ_alpha_total}, Nbeta={n_occ_beta_total})')

    valence_svds = spade_svds(spade_ncores, None)
    alpha_cut = None
    for ispin in range(atomsembed.n_spins):
        max_occ_state = n_occupied(ispin)
        beta_channel = (ispin == 1) and (atomsembed.n_spins == 2)

        if beta_channel:
            # Do not independently gap-search the beta channel: two
            # unrelated SVD gap searches (one per channel) have no
            # reason to agree on a partition that nets subsystem B
            # to zero spin. Instead force the cutoff so subsystem A
            # carries the whole supersystem spin exactly, and
            # subsystem B nets to zero by construction.
            cut = alpha_cut - supersystem_spin
            n_occ_beta_channel = max_occ_state - spade_ncores

            if not (0 <= cut <= n_occ_beta_channel):
                raise Exception(
                    f"SPADE beta-channel cutoff ({cut}), forced from "
                    f"the alpha-channel partition ({alpha_cut}) "
                    f"and the supersystem spin ({supersystem_spin}), falls outside the "
                    f"valid range [0, {n_occ_beta_channel}]. Subsystem A cannot be "
                    f"assigned this much spin from the beta-occupied MOs available."
                )
        elif a_nspade_mos is not None:
            cut = a_nspade_mos - spade_ncores
        else:
            cut = common_cut(valence_svds, ispin, spade_manual_state)

        if ispin == 0:
            alpha_cut = cut

        root_print(f'MAX OCC STATE {max_occ_state} for Spin Channel {ispin}')
        root_print(f'Maximum SPADE state for subsystem A, Spin Channel {ispin}, '
                   f'all {atomsembed.n_kpoints} k-point(s): {cut}')
        rotate(valence_svds, ispin, cut, spade_ncores)

    valence = densmats(occ_scale)
    if spade_ncores > 0:
        density_matrix_supersystem = density_matrix_supersystem + valence[0]
        density_matrix_subsys_a = density_matrix_subsys_a + valence[1]
        density_matrix_subsys_b = density_matrix_subsys_b + valence[2]
    else:
        density_matrix_supersystem, density_matrix_subsys_a, density_matrix_subsys_b = valence

    # I don't think this is needed anymore - density matrices should be synched before this
    # point.
    if not parallel:
        for ispin, ikpt in density_matrix_supersystem.keys():
            density_matrix_supersystem[ispin,ikpt] = mpi_bcast_matrix(density_matrix_supersystem[ispin,ikpt])
            density_matrix_subsys_a[ispin,ikpt] = mpi_bcast_matrix(density_matrix_subsys_a[ispin,ikpt])
            density_matrix_subsys_b[ispin,ikpt] = mpi_bcast_matrix(density_matrix_subsys_b[ispin,ikpt])

    # density_matrix_subsys_a/b were each built via rot_evecs_occ_a/b @ .T --
    # matmul chains starting from ovlp_evecs slices of DIFFERENT column
    # counts (n_occ_A != n_occ_B), whose own block sizes are independently
    # "optimal" for their own (different) shapes with no relation to each
    # other. Both end up (n_basis, n_basis) here, but nothing makes them
    # agree with each other or with overlap's -- and this codebase's callers
    # rely on being able to add/subtract these against overlap-space and
    # A_HL-space quantities (embedding.py's densmat_A_LL + densmat_B_LL, and
    # densmat_A_HL - densmat_A_LL). Rechunk explicitly onto overlap's block
    # size, the one reference both this function's caller and A_HL's own
    # ASI-native arrays already interoperate against.
    density_matrix_supersystem = density_matrix_supersystem.rechunk(like=overlap)
    density_matrix_subsys_a = density_matrix_subsys_a.rechunk(like=overlap)
    density_matrix_subsys_b = density_matrix_subsys_b.rechunk(like=overlap)

    root_print(f'SPADE total supersystem A+B charge: {(overlap @ density_matrix_supersystem).trace()}')
    root_print(f'SPADE localised subsystem A charge: {(overlap @ density_matrix_subsys_a).trace()}')
    root_print(f'SPADE localised subsystem B charge: {(overlap @ density_matrix_subsys_b).trace()}')

    root_print('Exiting SPADE localisation...')

    if return_mo_coeffs:
        return density_matrix_subsys_a, density_matrix_subsys_b, rot_evecs_occ_a, rot_evecs_occ_b
    else:
        return density_matrix_subsys_a, density_matrix_subsys_b
