"""Internal stability analysis for PySCF SCF solutions.

An SCF can converge to a saddle point rather than a minimum -- for a stretched bond, the
closed-shell (RHF-like) solution of a UHF often survives as a stationary point below which a
lower, broken-symmetry solution exists.  Which one DIIS lands on depends on the starting
guess, so a scan can jump between them.  Following the lowest internal instability to a
stable solution removes that dependence.
"""
from embasi.parallel_utils import root_print


def follow_internal_instabilities(mf, max_rounds=20, plain_rounds=2):
    """Re-converge ``mf`` (in place) until it is internally stable.

    Each unstable analysis returns orbitals rotated downhill along the instability.  The
    first ``plain_rounds`` restarts use ordinary DIIS from that density; DIIS can fall back
    into the saddle it just left (or cycle between two), so later rounds use second-order
    SCF (``mf.newton()``) started from the rotated orbitals, and copy its solution back onto
    ``mf`` so callers keep using the same object.

    Only internal instabilities are followed (a UHF stays a UHF, an RHF an RHF).  For a
    spin-symmetric UHF solution the spin-symmetry-breaking direction is checked separately
    (``_triplet_instability``), since the internal analysis can miss it.  If no stable
    solution is reached in ``max_rounds``, the last solution is kept with a warning.

    Returns
    -------
    int
        The number of instabilities followed.
    """
    for rnd in range(max_rounds):
        mo_i, _mo_e, stable_i, _stable_e = mf.stability(return_status=True)
        if stable_i and mf.converged:
            mo_bs = _triplet_instability(mf)
            if mo_bs is None:
                root_print(f"SCF stability: internally stable after following {rnd} "
                           f"instabilit{'y' if rnd == 1 else 'ies'} (E = {mf.e_tot:.10f} Ha)")
                return rnd
            # A spin-symmetric solution unstable towards breaking the spin symmetry.
            mf.kernel(dm0=mf.make_rdm1(mo_bs, mf.mo_occ))
            continue
        if rnd < plain_rounds:
            mf.kernel(dm0=mf.make_rdm1(mo_i, mf.mo_occ))
        else:
            mf_n = mf.newton()
            mf_n.kernel(mo_i, mf.mo_occ)
            for attr in ("mo_coeff", "mo_occ", "mo_energy", "e_tot", "converged"):
                setattr(mf, attr, getattr(mf_n, attr))
    root_print(f"WARNING: SCF stability: no internally stable solution after {max_rounds} "
               f"rounds; keeping the last solution (E = {mf.e_tot:.10f} Ha, "
               f"converged={mf.converged})")
    return max_rounds


def _triplet_instability(mf, tol=1e-6):
    """Rotated UHF orbitals if a spin-SYMMETRIC unrestricted solution can lower its energy
    by breaking the spin symmetry, else ``None``.

    PySCF's internal UHF stability analysis can miss this direction when the alpha and beta
    orbitals are identical (it converged to a closed-shell local minimum on stretched N2,
    158 mHa above the broken-symmetry solution).  The same question, asked of the restricted
    wave function, is PySCF's external (RHF -> UHF) stability analysis, which returns
    unrestricted orbitals rotated along the instability.
    """
    import numpy as np
    from pyscf.scf import addons, uhf

    if not isinstance(mf, uhf.UHF):
        return None
    n_a, n_b = (int(np.count_nonzero(o)) for o in mf.mo_occ)
    dm_a, dm_b = mf.make_rdm1()
    if n_a != n_b or np.abs(dm_a - dm_b).max() > tol:
        return None
    rmf = addons.convert_to_rhf(mf)
    _mo_i, mo_e, _stable_i, stable_e = rmf.stability(internal=False, external=True,
                                                     return_status=True)
    return None if stable_e else mo_e
