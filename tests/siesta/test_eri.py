"""embasi.eri.AOERIs: layout conversions and J/K/MO contractions, against PySCF

No QM code needed: PySCF integrals stand in for an exported ERI matrix.
"""
import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf import ao2mo, gto, scf  # noqa: E402

from embasi.eri import AOERIs, ASI_ERI_NOTATION_CHEMIST, ASI_ERI_IDX_4, \
    ASI_ERI_BASIS_AO, ASI_ERI_SYM_1, ASI_ERI_SYM_4, ASI_ERI_SYM_8  # noqa: E402


@pytest.fixture(scope="module")
def water():
    mol = gto.M(atom="O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59", basis="6-31g",
                verbose=0)
    mf = scf.RHF(mol).run()
    return mol, mf


def descr(sym):
    return {"sum_notation": ASI_ERI_NOTATION_CHEMIST, "n_index": ASI_ERI_IDX_4,
            "basis_type": ASI_ERI_BASIS_AO, "eri_symmetry": sym}


def test_layouts_agree(water):
    mol, _ = water
    n = mol.nao
    full = mol.intor("int2e")
    s4 = ao2mo.restore(4, full, n)
    sym1 = full.reshape(n * n, n * n, order="F")
    sym8 = np.tril(s4)  # ASI_ERI_SYM_8: one triangle of the pair matrix
    for eri, sym in ((s4, ASI_ERI_SYM_4), (sym1, ASI_ERI_SYM_1), (sym8, ASI_ERI_SYM_8)):
        e = AOERIs.from_asi(eri, descr(sym), n)
        assert np.allclose(e.eri_s4, s4, atol=1e-12)
        assert np.allclose(e.tensor(), full, atol=1e-12)


def test_jk_and_energies(water):
    mol, mf = water
    e = AOERIs(ao2mo.restore(4, mol.intor("int2e"), mol.nao), mol.nao)
    dm = mf.make_rdm1()
    vj, vk = mf.get_jk(mol, dm)
    j, k = e.get_jk(dm)
    assert np.allclose(j, vj, atol=1e-10) and np.allclose(k, vk, atol=1e-10)
    assert np.allclose(e.veff_hf(dm), mf.get_veff(mol, dm), atol=1e-10)
    e_coul = mf.energy_elec(dm)[1]
    assert abs(e.hartree_energy(dm) + e.exchange_energy(dm) - e_coul) < 1e-10


def test_veff_uhf(water):
    mol, mf = water
    e = AOERIs(ao2mo.restore(4, mol.intor("int2e"), mol.nao), mol.nao)
    c = mf.mo_coeff
    da = c[:, :5] @ c[:, :5].T
    db = c[:, :4] @ c[:, :4].T
    uhf = scf.UHF(mol)
    va_ref, vb_ref = uhf.get_veff(mol, np.array([da, db]))
    va, vb, e2 = e.veff_uhf(da, db)
    assert np.allclose(va, va_ref, atol=1e-10) and np.allclose(vb, vb_ref, atol=1e-10)
    e2_ref = 0.5 * (np.einsum("ij,ji", da, va_ref) + np.einsum("ij,ji", db, vb_ref))
    assert abs(e2 - e2_ref) < 1e-10


def test_eri_mo(water):
    mol, mf = water
    e = AOERIs(ao2mo.restore(4, mol.intor("int2e"), mol.nao), mol.nao)
    c = mf.mo_coeff[:, 2:7]
    ref = ao2mo.restore(1, ao2mo.kernel(mol, c), c.shape[1])
    assert np.allclose(e.eri_mo(c), ref, atol=1e-10)
    assert np.allclose(e.eri_mo_mixed(c, c), ref, atol=1e-10)
    c2 = mf.mo_coeff[:, 3:8]
    ref2 = ao2mo.general(mol, (c, c, c2, c2), compact=False).reshape((5,) * 4)
    assert np.allclose(e.eri_mo_mixed(c, c2), ref2, atol=1e-10)


def test_rejects_wrong_shape():
    with pytest.raises(ValueError):
        AOERIs(np.zeros((3, 3)), 4)
