"""PySCF per-iteration Fock construction hooks for embedded fragments.

Unlike the post-SCF correction in pyscf_postscf.py (which runs once, after
convergence), the hook here needs to run on every SCF iteration - it
injects the static embedding potential (level-shift, or the 'emb_pot'
term of Huzinaga projection) and/or the dynamic, self-consistent Huzinaga
projector (which depends on the current, per-iteration Fock, not just a
fixed matrix) into the Fock matrix PySCF's own SCF loop builds and
diagonalises each cycle.
"""


def embedded_get_fock_factory(mf, fock_func, mat_in=None, gamma_B=None, S=None):
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
    S : np.ndarray or None
        Supersystem overlap matrix. Required (non-None) whenever gamma_B
        is given.

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
        if h1e is None: h1e = mf.get_hcore()
        if vhf is None: vhf = mf.get_veff(mf.mol, dm)

        vhf_aug = vhf
        if mat_in is not None:
            vhf_aug = vhf_aug + mat_in
        if gamma_B is not None:
            f_emb = h1e + vhf_aug
            vhf_aug = vhf_aug - 0.5 * (f_emb @ gamma_B @ S.T + S @ gamma_B @ f_emb.T)

        return fock_func(h1e=h1e, s1e=s1e, vhf=vhf_aug, dm=dm, cycle=cycle,
                         diis=diis, diis_start_cycle=diis_start_cycle,
                         level_shift_factor=level_shift_factor,
                         damp_factor=damp_factor, fock_last=fock_last)

    return embedded_get_fock
