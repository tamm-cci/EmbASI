"""PySCF post-SCF (correlated wavefunction) dispatch for embedded fragments.

The correction is evaluated non-self-consistently: the mean-field
reference (and, for a projection-embedded fragment, the embedding
potential and/or Huzinaga projector baked into its per-iteration Fock via
get_fock) is fully converged first; the correlated method is then built
directly on top of it and never feeds back into the outer embedding SCF.
"""

import copy


def run_postscf(mf, post_scf):
    """Runs a post-SCF correlated wavefunction calculation on a converged mf

    Parameters
    ----------
    mf : pyscf.scf.hf.SCF
        A converged (possibly embedded/projected) mean-field object. If it
        was obtained under a projection-embedding potential or a Huzinaga
        projector, that must still be active in mf.get_fock() at the
        point this is called, as CCSD/MP2 rebuild their reference Fock
        via mf.get_fock(vhf=vhf, dm=dm) (see PySCFAdapter.run_scf).
    post_scf : str or object
        Either a bare method name - 'mp2', 'ccsd', 'ccsd(t)'
        (case-insensitive) - to build a post-HF object with PySCF
        defaults, or a pre-configured, unconverged post-HF object (e.g.
        pyscf.cc.CCSD(dummy_mf, frozen=2), pyscf.mp.MP2(dummy_mf)) whose
        solver options should be reused. In the latter case its _scf,
        mol, mo_coeff, mo_occ and mo_energy are overwritten to point at
        mf before running - only its solver options are actually taken
        from it. If frozen is set as an explicit list of orbital indices
        (rather than a plain integer count), those indices must already
        correspond to mf's orbital ordering. To also run the
        perturbative triples correction on a pre-configured CCSD object
        (pyscf.cc.CCSD is used for both CCSD and CCSD(T)), set
        ``.run_ccsd_t = True`` on it before passing it in.

    Returns
    -------
    float
        Correlation energy in Hartree (including any perturbative triples
        contribution), i.e., the amount to add to mf.e_tot.
    """
    from pyscf import cc, dft, mp

    if isinstance(mf, dft.KohnShamDFT):
        # mo_coeff/mo_energy are unchanged by this conversion - only the
        # class (and hence get_veff/get_fock) changes from DFT XC-based
        # to exact-exchange-based. Post-HF methods rebuild their Fock/
        # denominators from get_veff, not from mf.mo_energy directly, so
        # skipping this step would silently correlate on top of a
        # DFT-flavoured (not HF-like) Fock operator instead of treating
        # the converged KS orbitals as an HF-type reference - which is
        # what "MP2/CCSD(T)-in-<xc>" is actually supposed to mean.
        #
        # to_hf() builds a fresh object and does not carry over an
        # instance-level get_fock override, which is exactly how the
        # embedding potential and/or Huzinaga projector are applied here
        # (see PySCFAdapter.run_scf) - re-apply it so the correlated
        # method still sees the embedded/projected Fock, not the bare one.
        get_fock_override = mf.__dict__.get('get_fock')
        mf = mf.to_hf()
        if get_fock_override is not None:
            mf.get_fock = get_fock_override

    if isinstance(post_scf, str):
        method = post_scf.lower()

        if method == "mp2":
            wf = mp.MP2(mf)
        elif method == "ccsd":
            wf = cc.CCSD(mf)
        elif method in ("ccsd(t)", "ccsdt"):
            wf = cc.CCSD(mf)
            wf.kernel()
            return wf.e_corr + wf.ccsd_t()
        else:
            raise ValueError(f"Unsupported post_scf method for PySCF: {post_scf!r}")

        wf.kernel()
        return wf.e_corr

    # Pre-configured post-HF object: reuse its solver settings (frozen,
    # conv_tol, max_cycle, DIIS, ...), rebind it to the real reference.
    wf = copy.copy(post_scf)
    wf._scf = mf
    wf.mol = mf.mol
    wf.mo_coeff = mf.mo_coeff
    wf.mo_occ = mf.mo_occ
    wf.mo_energy = mf.mo_energy

    wf.kernel()

    if getattr(wf, "run_ccsd_t", False):
        return wf.e_corr + wf.ccsd_t()

    return wf.e_corr
