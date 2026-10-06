"""UNO-SPADE partition (embasi.uno_spade_localisation), HF-in-HF with PySCF.

An OH radical 4 A from a water molecule, sto-3g, spin 1 (the unpaired electron on OH).
Plain SPADE partitions alpha and beta separately; UNO-SPADE must give one closed-shell
environment shared by both spins, with all the spin in subsystem A, at the cost of freezing
that environment closed-shell (variational: the HF-in-HF energy may only rise, slightly).
"""
import numpy as np
import pytest
import pyscf
from ase import Atoms
from pyscf.pbc.tools.pyscf_ase import PySCF, ase_atoms_to_pyscf

from embasi.embedding import ProjectionEmbedding

HA2EV = 27.211384500


def _run(localisation, spin=1, restricted=False, **options):
    atoms = Atoms("OHOHH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.97],
                                     [4.0, 0.0, 0.0], [4.0, 0.0, 0.96], [4.9, 0.0, -0.3]])
    if restricted:  # closed-shell variant: OH^- with water
        spin, charge = 0, -1
    else:
        charge = 0
    mol = pyscf.M(atom=ase_atoms_to_pyscf(atoms), basis="sto-3g", spin=spin, charge=charge,
                  verbose=0)

    def mf():
        m = mol.KS(xc="hf") if restricted else mol.UKS(xc="hf")
        m.conv_tol = 1e-10
        return m

    emb = ProjectionEmbedding(atoms, embed_mask=[1, 1, 2, 2, 2],
                              calc_base_ll=PySCF(method=mf()), calc_base_hl=PySCF(method=mf()),
                              projection="level-shift", localisation=localisation,
                              total_charge=charge, **options)
    emb.run()
    return emb


@pytest.fixture(scope="module")
def uno():
    return _run("UNO-SPADE")


@pytest.fixture(scope="module")
def spade():
    return _run("SPADE")


def test_environment_is_shared_by_both_spins(uno):
    cb_a, cb_b = (np.asarray(uno.mo_coeffs_B_LL[k, 0]) for k in (0, 1))
    np.testing.assert_allclose(cb_a, cb_b, atol=1e-12)
    s = np.asarray(uno.AB_LL.overlap[0, 0])
    for k in (0, 1):
        ca = np.asarray(uno.mo_coeffs_A_LL[k, 0])
        assert np.abs(ca.T @ s @ cb_a).max() < 1e-10, "A's occupied orbitals must be orthogonal to B"


def test_all_spin_is_in_subsystem_a(uno):
    assert uno.A_spin == 1 and uno.B_spin == 0


def test_hf_in_hf_energy_is_variational_and_close(uno, spade):
    e_super = uno.subsys_AB_lowlvl_scftotalen / HA2EV
    e_uno = uno.DFT_AinB_total_energy / HA2EV
    e_spade = spade.DFT_AinB_total_energy / HA2EV
    # Per-spin SPADE is exact for HF-in-HF; a frozen closed-shell environment may only raise it.
    assert e_spade == pytest.approx(e_super, abs=1e-7)
    assert -1e-7 < e_uno - e_super < 1e-3


def test_closed_shell_reference_is_plain_spade():
    e_uno = _run("UNO-SPADE", restricted=True).DFT_AinB_total_energy
    e_spade = _run("SPADE", restricted=True).DFT_AinB_total_energy
    assert e_uno == pytest.approx(e_spade, abs=1e-8)


def test_fixed_environment_count(uno):
    """uno_n_env pins how many doubly occupied UNOs form B: equal to the gap's choice it
    reproduces the gap result exactly, and a different count is honoured."""
    n_gap = np.asarray(uno.mo_coeffs_B_LL[0, 0]).shape[1]
    same = _run("UNO-SPADE", uno_n_env=n_gap)
    assert same.DFT_AinB_total_energy == pytest.approx(uno.DFT_AinB_total_energy, abs=1e-8)
    fewer = _run("UNO-SPADE", uno_n_env=n_gap - 1)
    assert np.asarray(fewer.mo_coeffs_B_LL[0, 0]).shape[1] == n_gap - 1
    assert fewer.A_spin == 1 and fewer.B_spin == 0
    # A larger A frozen less: still variational against the supersystem, and no worse.
    e_super = fewer.subsys_AB_lowlvl_scftotalen / HA2EV
    assert -1e-7 < fewer.DFT_AinB_total_energy / HA2EV - e_super <= (
        uno.DFT_AinB_total_energy / HA2EV - e_super + 1e-7
    )


def test_fixed_environment_count_excludes_a_nspade_mos():
    """Both options fix the same cut; giving both must fail loudly, not pick one."""
    atoms = Atoms("OHOHH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.97],
                                     [4.0, 0.0, 0.0], [4.0, 0.0, 0.96], [4.9, 0.0, -0.3]])
    mol = pyscf.M(atom=ase_atoms_to_pyscf(atoms), basis="sto-3g", spin=1, verbose=0)
    emb = ProjectionEmbedding(atoms, embed_mask=[1, 1, 2, 2, 2],
                              calc_base_ll=PySCF(method=mol.UKS(xc="hf")),
                              calc_base_hl=PySCF(method=mol.UKS(xc="hf")),
                              projection="level-shift", localisation="UNO-SPADE", uno_n_env=4)
    with pytest.raises(ValueError, match="not both"):
        emb.construct_embedding_potential(a_nspade_mos=2)
