"""Supersystem SCF stability analysis (``scf_stability``), PySCF adapter.

Stretched N2 with the symmetry-breaking guess switched off converges to the closed-shell,
RHF-like UHF solution -- a saddle point below which a broken-symmetry minimum exists.
``scf_stability=True`` must follow the instability down to it.
"""
import numpy as np
import pytest
import pyscf
from ase import Atoms
from pyscf.pbc.tools.pyscf_ase import PySCF, ase_atoms_to_pyscf

from embasi.embedding import ProjectionEmbedding

HA2EV = 27.211384500


def _supersystem(r, stability, symbols="NN"):
    atoms = Atoms(symbols, positions=[[0.0, 0.0, 0.0], [0.0, 0.0, r]])
    mol = pyscf.M(atom=ase_atoms_to_pyscf(atoms), basis="sto-3g", spin=0, verbose=0)

    def mf():
        m = mol.UKS(xc="hf")
        m.init_guess_breaksym = 0  # spin-symmetric guess: lands on the closed-shell solution
        m.conv_tol = 1e-10
        return m

    emb = ProjectionEmbedding(atoms, embed_mask=[1, 2], calc_base_ll=PySCF(method=mf()),
                              calc_base_hl=PySCF(method=mf()), projection="level-shift",
                              scf_stability=stability)
    emb.construct_embedding_potential()
    return emb.AB_LL.total_energy / HA2EV, emb.AB_LL.atoms.calc.method


def test_follows_the_instability_to_the_broken_symmetry_minimum():
    e_plain, mf_plain = _supersystem(2.0, stability=False)
    e_stab, mf_stab = _supersystem(2.0, stability=True)
    assert mf_plain.spin_square()[0] == pytest.approx(0.0, abs=1e-6)  # the closed-shell saddle
    assert e_stab < e_plain - 1e-3
    assert mf_stab.spin_square()[0] > 0.5  # broken symmetry
    assert mf_stab.stability(return_status=True)[2]  # and internally stable


def test_leaves_a_stable_solution_unchanged():
    e_plain, _ = _supersystem(1.1, stability=False)
    e_stab, _ = _supersystem(1.1, stability=True)
    assert e_stab == pytest.approx(e_plain, abs=1e-8)
