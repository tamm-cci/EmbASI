import numpy as np
from embasi.parallel_utils import root_print, mpi_bcast_matrix
from embasi.roothan_hall_eigensolver import hamiltonian_eigensolv
from embasi.roothan_hall_eigensolver_scalapack import hamiltonian_eigensolv_parallel
from scalapack4py.npscal.math_utils.npscal2npscal import svd


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
    if parallel:
        evals, evecs, evecs_orthog, occ_mat = hamiltonian_eigensolv_parallel(hamiltonian, \
                                                                             overlap, \
                                                                             nelecs, \
                                                                             nspins=atomsembed.n_spins, \
                                                                             nkpts=atomsembed.n_kpoints, \
                                                                             basis_illcond_thresh=basis_illcond_thresh,
                                                                             return_orthog=True)
    else:
        evals, evecs, evecs_orthog, occ_mat = hamiltonian_eigensolv(hamiltonian, \
                                                      overlap, \
                                                      nelecs, \
                                                      nspins=atomsembed.n_spins,
                                                      nkpts=atomsembed.n_kpoints,
                                                      basis_illcond_thresh=basis_illcond_thresh,
                                                      return_orthog=True)


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

    # TODO: Clean this up and add automatic detection based on eigenvalues.
    # This should be done in one routine - this is very FORTRAN-like.
    if spade_ncores > 0:
        root_print("Performing separate core-valence SPADE localisation.")
        for ispin in range(atomsembed.n_spins):
            for ikpt in range(atomsembed.n_kpoints):
                max_occ_state = np.count_nonzero(occ_mat[ispin,ikpt])

                evecs_occ_orthog = evecs_orthog[ispin, ikpt, :, :spade_ncores]
                evecs_occ_a_orthog = evecs_occ_orthog[mask_val, :]

                if parallel:
                    u, svals, v = svd(evecs_occ_a_orthog)
                else:
                    u, svals, v = np.linalg.svd(evecs_occ_a_orthog, full_matrices=True, dtype=np.float64)

                svals_diff = np.ediff1d(svals**2.0)
                max_sval_change_idx = np.argmax(np.abs(svals_diff)) + 1

                root_print(f'MAX OCC CORE STATE {spade_ncores} for Spin Channel {ispin}')
                root_print(f'SPADE CORE STATE FOR: Spin Channel {ispin}, K-point {ikpt}')
                root_print(f'Maximum SPADE core state for subsystem A: {max_sval_change_idx}')

                evecs_occ = evecs[ispin, ikpt, :, :spade_ncores]

                rot_evecs_occ_a[ispin, ikpt] = evecs_occ @ v[:max_sval_change_idx, :].T
                rot_evecs_occ_b[ispin, ikpt] = evecs_occ @ v[max_sval_change_idx:, :].T
                evecs_occ_ab[ispin, ikpt] = evecs_occ.copy()

        # @TODOSPIN: Need to redefine occupancies - this obviously won't work for k-points
        if atomsembed.n_spins == 1:
            density_matrix_supersystem = 2.0 * (evecs_occ_ab @ evecs_occ_ab.copy().T)
            density_matrix_subsys_a = 2.0 * (rot_evecs_occ_a @ rot_evecs_occ_a.copy().T)
            density_matrix_subsys_b = 2.0 * (rot_evecs_occ_b @ rot_evecs_occ_b.copy().T)
        else:
            density_matrix_supersystem = (evecs_occ_ab @ evecs_occ_ab.copy().T)
            density_matrix_subsys_a = (rot_evecs_occ_a @ rot_evecs_occ_a.copy().T)
            density_matrix_subsys_b = (rot_evecs_occ_b @ rot_evecs_occ_b.copy().T)

        root_print(f'SPADE CORE localised subsystem A charge: {(overlap @ density_matrix_subsys_a).trace()}')
        root_print(f'SPADE CORE localised subsystem B charge: {(overlap @ density_matrix_subsys_b).trace()}')

    for ispin in range(atomsembed.n_spins):
        for ikpt in range(atomsembed.n_kpoints):
            max_occ_state = np.count_nonzero(occ_mat[ispin,ikpt])

            evecs_occ_orthog = evecs_orthog[ispin, ikpt, :, spade_ncores:max_occ_state]
            evecs_occ_a_orthog = evecs_occ_orthog[mask_val, :]

            if parallel:
                u, svals, v = svd(evecs_occ_a_orthog)
            else:
                u, svals, v = np.linalg.svd(evecs_occ_a_orthog, full_matrices=True)

            if a_nelecs is not None:
                max_sval_change_idx = a_nspade_mos - spade_ncores
            else:
                svals_diff = np.ediff1d(svals**2.0)
                max_sval_change_idx = np.argmax(np.abs(svals_diff)) + spade_manual_state + 1

            root_print(f'MAX OCC STATE {max_occ_state} for Spin Channel {ispin}')
            root_print(f'SPADE STATE FOR: Spin Channel {ispin}, K-point {ikpt}')
            root_print(f'Maximum SPADE state for subsystem A: {max_sval_change_idx}')

            evecs_occ = evecs[ispin, ikpt, :, spade_ncores:max_occ_state]

            rot_evecs_occ_a[ispin, ikpt] = evecs_occ @ v[:max_sval_change_idx, :].T
            rot_evecs_occ_b[ispin, ikpt] = evecs_occ @ v[max_sval_change_idx:, :].T
            evecs_occ_ab[ispin, ikpt] = evecs_occ.copy()

    # @TODOSPIN: Need to redefine occupancies - this obviously won't work for k-points
    if spade_ncores > 0:
        if atomsembed.n_spins == 1:
            density_matrix_supersystem = density_matrix_supersystem + 2.0 * (evecs_occ_ab @ evecs_occ_ab.copy().T)
            density_matrix_subsys_a = density_matrix_subsys_a + 2.0 * (rot_evecs_occ_a @ rot_evecs_occ_a.copy().T)
            density_matrix_subsys_b = density_matrix_subsys_b + 2.0 * (rot_evecs_occ_b @ rot_evecs_occ_b.copy().T)
        else:
            density_matrix_supersystem = density_matrix_supersystem + (evecs_occ_ab @ evecs_occ_ab.copy().T)
            density_matrix_subsys_a = density_matrix_subsys_a + (rot_evecs_occ_a @ rot_evecs_occ_a.copy().T)
            density_matrix_subsys_b = density_matrix_subsys_b + (rot_evecs_occ_b @ rot_evecs_occ_b.copy().T)
    else:
        if atomsembed.n_spins == 1:
            density_matrix_supersystem = 2.0 * (evecs_occ_ab @ evecs_occ_ab.copy().T)
            density_matrix_subsys_a = 2.0 * (rot_evecs_occ_a @ rot_evecs_occ_a.copy().T)
            density_matrix_subsys_b =  2.0 * (rot_evecs_occ_b @ rot_evecs_occ_b.copy().T)
        else:
            density_matrix_supersystem = (evecs_occ_ab @ evecs_occ_ab.copy().T)
            density_matrix_subsys_a = (rot_evecs_occ_a @ rot_evecs_occ_a.copy().T)
            density_matrix_subsys_b = (rot_evecs_occ_b @ rot_evecs_occ_b.copy().T)

    # I don't think this is needed anymore - density matrices should be synched before this
    # point.
    if not parallel:
        for ispin in range(atomsembed.n_spins):
            for ikpt in range(atomsembed.n_kpoints):
                density_matrix_supersystem[ispin,ikpt] = mpi_bcast_matrix(density_matrix_supersystem[ispin,ikpt])
                density_matrix_subsys_a[ispin,ikpt] = mpi_bcast_matrix(density_matrix_subsys_a[ispin,ikpt])
                density_matrix_subsys_b[ispin,ikpt] = mpi_bcast_matrix(density_matrix_subsys_b[ispin,ikpt])

    root_print(f'SPADE total supersystem A+B charge: {(overlap @ density_matrix_supersystem).trace()}')
    root_print(f'SPADE localised subsystem A charge: {(overlap @ density_matrix_subsys_a).trace()}')
    root_print(f'SPADE localised subsystem B charge: {(overlap @ density_matrix_subsys_b).trace()}')

    root_print('Exiting SPADE localisation...')

    if return_mo_coeffs:
        return density_matrix_subsys_a, density_matrix_subsys_b, rot_evecs_occ_a, rot_evecs_occ_b
    else:
        return density_matrix_subsys_a, density_matrix_subsys_b
