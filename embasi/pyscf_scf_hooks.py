"""PySCF per-iteration Fock construction hooks for embedded fragments.

Unlike the post-SCF correction in pyscf_postscf.py (which runs once, after
convergence), the hook here needs to run on every SCF iteration - it
injects the static embedding potential (level-shift, or the 'emb_pot'
term of Huzinaga projection) and/or the dynamic, self-consistent Huzinaga
projector (which depends on the current, per-iteration Fock, not just a
fixed matrix) into the Fock matrix PySCF's own SCF loop builds and
diagonalises each cycle.
"""


def embedded_get_fock_factory(mf, fock_func, mat_in=None, gamma_B=None, S=None, n_spins=1,
                              fock_full=None, to_full=None, to_trunc=None):
    """Builds a get_fock replacement that injects embedding physics

    PySCF's SCF.get_fock is called exactly once per SCF cycle, right
    before diagonalisation, with the raw h1e/vhf ingredients needed to
    reconstruct the current Fock matrix. Wrapping it here means PySCF's
    own SCF loop, and its damping/DIIS/level-shift machinery, need no
    reimplementation - the returned function only needs to augment vhf
    before delegating to the original get_fock.

    Parameters
    ----------
    mf : pyscf.scf.hf.SCF
        The mean-field object this hook will be installed on (as
        mf.get_fock = embedded_get_fock_factory(mf, mf.get_fock, ...)).
        Used only as a fallback source for h1e/vhf when the SCF loop
        calls the returned function without them (e.g., cc.CCSD/mp.MP2
        rebuilding their reference Fock via mf.get_fock(vhf=vhf, dm=dm)).
    fock_func : bound method
        The original (pre-override) mf.get_fock, captured before
        patching, so damping/DIIS/level-shift still apply to the
        augmented Fock exactly as PySCF would apply them to the bare one.
    mat_in : np.ndarray or None
        Static embedding potential to add to vhf every iteration (the
        level-shift potential, or the 'emb_pot' term of Huzinaga
        projection). None to skip.
    gamma_B : np.ndarray or None
        Environment (subsystem B) density matrix, for the self-consistent
        Huzinaga projector. None to skip the projector term entirely.
        Shape (2, nao, nao) for n_spins=2 (UHF/UKS/ROHF/ROKS), else
        (nao, nao).
    S : np.ndarray or None
        Supersystem overlap matrix. Required (non-None) whenever gamma_B
        is given. Always plain (nao, nao) - overlap is spin-independent,
        and numpy's batched @ broadcasts it against a (2, nao, nao)
        gamma_B/hamiltonian without needing an explicit spin axis here.
    n_spins : int
        Number of spin channels mat_in/gamma_B carry (1 or 2) - selects
        the -0.5 (restricted) vs -1.0 (unrestricted) Huzinaga prefactor.
        Defaults to 1.
    fock_full : np.ndarray or None
        Basis truncation only: the full-basis embedding potential, whose
        elements outside the truncated block stand in for the Fock
        matrix there. gamma_B and S are then full-size, the projector is
        built in the full basis from to_full(h1e + vhf) + fock_full and
        truncated with to_trunc. None without truncation.
    to_full, to_trunc : callable or None
        Pad a truncated matrix to the full basis, and cut a full one to
        the truncated basis (AtomsEmbed.truncated_mat_to_full and
        full_mat_to_truncated). Required with fock_full.

    Returns
    -------
    callable
        A get_fock-compatible function (same signature as
        pyscf.scf.hf.SCF.get_fock, minus the leading mf/self parameter)
        suitable for assigning directly as mf.get_fock.
    """

    def embedded_get_fock(h1e=None, s1e=None, vhf=None, dm=None, cycle=-1,
                          diis=None, diis_start_cycle=None,
                          level_shift_factor=None, damp_factor=None,
                          fock_last=None):
        from embasi.embedding_projectors import huzinaga_projector

        if h1e is None: h1e = mf.get_hcore()
        if vhf is None: vhf = mf.get_veff(mf.mol, dm)

        vhf_aug = vhf
        if mat_in is not None:
            vhf_aug = vhf_aug + mat_in
        if gamma_B is not None:
            import numpy as np

            f_emb = h1e + vhf_aug
            if fock_full is not None:
                # Truncated basis: build the projector in the full basis
                # from the current fragment Fock padded with the full
                # embedding potential, then truncate it.
                f_bare = h1e + vhf

                def proj_one(s):
                    f = f_bare[s] if n_spins == 2 else f_bare
                    ff = fock_full[s] if n_spins == 2 else fock_full
                    g = gamma_B[s] if n_spins == 2 else gamma_B
                    return to_trunc(huzinaga_projector(to_full(f) + ff, S, g, n_spins=n_spins))
            else:
                def proj_one(s):
                    f = f_emb[s] if n_spins == 2 else f_emb
                    g = gamma_B[s] if n_spins == 2 else gamma_B
                    return huzinaga_projector(f, S, g, n_spins=n_spins)

            # huzinaga_projector uses plain .T, which for a bare ndarray
            # transposes ALL axes, not just the trailing matrix ones -
            # fine for the 2D (nao, nao) case, but wrong on a stacked
            # (2, nao, nao) array (it would swap the spin axis into a
            # matrix axis instead of leaving it alone). Call it once per
            # spin channel on proper 2D slices instead, exactly the "bare
            # per-spin/k-point matrix" usage its docstring anticipates.
            if n_spins == 2:
                proj = np.stack([proj_one(s) for s in range(n_spins)])
            else:
                proj = proj_one(0)
            vhf_aug = vhf_aug + proj

        return fock_func(h1e=h1e, s1e=s1e, vhf=vhf_aug, dm=dm, cycle=cycle,
                         diis=diis, diis_start_cycle=diis_start_cycle,
                         level_shift_factor=level_shift_factor,
                         damp_factor=damp_factor, fock_last=fock_last)

    return embedded_get_fock
