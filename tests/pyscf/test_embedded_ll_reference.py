"""total_energy_corr='embedded_ll_reference' with basis truncation, PySCF.

Ethanol in 6-31G with the OH group as subsystem A; a Mulliken threshold of 0.05 |e|
keeps O, H and the alpha carbon and drops the other six atoms from the A layers.
The truncated-block embedding potential must be the full-basis one of Bennie et al.,
    v_emb = trunc(veff[gamma^AB] - veff[gamma^A] + V_nuc[dropped atoms]),
with no trace of the two-electron potential of the truncated density of A (which
does not hold A's electron count).
"""
import numpy as np
import pytest
import pyscf
from ase.build import molecule
from pyscf.pbc.tools.pyscf_ase import PySCF, ase_atoms_to_pyscf

from embasi.embedding import ProjectionEmbedding

THRESH = 0.05
PROJECTIONS = ["huzinaga", "huzinaga-sc"]


def _system():
    atoms = molecule("CH3CH2OH")
    o = list(atoms.symbols).index("O")
    dist = atoms.get_distances(o, range(len(atoms)))
    mask = [1 if (i == o or (s == "H" and dist[i] < 1.1)) else 2
            for i, s in enumerate(atoms.symbols)]
    # PySCF's Mole is fixed up front, so order it as ProjectionEmbedding will.
    idx = np.argsort(mask, kind="stable")
    atoms = atoms[idx]
    mask = [int(m) for m in np.sort(mask)]
    mol = pyscf.M(atom=ase_atoms_to_pyscf(atoms), basis="6-31g", verbose=0)
    return atoms, mask, mol


def _run(tmp_path_factory, projection, hl_xc, total_energy_corr, thresh):
    atoms, mask, mol = _system()

    def mf(xc):
        m = mol.KS(xc=xc)
        m.conv_tol = 1e-11
        return m

    emb = ProjectionEmbedding(atoms, embed_mask=mask,
                              calc_base_ll=PySCF(method=mf("PBE")),
                              calc_base_hl=PySCF(method=mf(hl_xc)),
                              projection=projection, localisation="SPADE",
                              truncate_basis_thresh=thresh,
                              total_energy_corr=total_energy_corr,
                              run_dir=str(tmp_path_factory.mktemp("emb")))
    emb.run()
    return emb, mol


@pytest.fixture(scope="module", params=PROJECTIONS)
def projection(request):
    return request.param


@pytest.fixture(scope="module")
def pbe_in_pbe_trunc(tmp_path_factory, projection):
    return _run(tmp_path_factory, projection, "PBE", "embedded_ll_reference", THRESH)


@pytest.fixture(scope="module")
def pbe0_in_pbe(tmp_path_factory, projection):
    return {(corr, thresh): _run(tmp_path_factory, projection, "PBE0", corr, thresh)[0].DFT_AinB_total_energy
            for corr in ("1storder", "embedded_ll_reference")
            for thresh in (None, THRESH)}


def test_atoms_are_truncated(pbe_in_pbe_trunc):
    emb, _ = pbe_in_pbe_trunc
    assert 0 < emb.basis_info.trunc_natoms < len(emb.AB_LL.atoms)


def test_vemb_is_full_basis_embedding_potential(pbe_in_pbe_trunc):
    emb, mol = pbe_in_pbe_trunc
    keep = np.asarray(emb.basis_mask, dtype=bool)
    ca = np.asarray(emb.mo_coeffs_A_LL[0, 0])
    cb = np.asarray(emb.mo_coeffs_B_LL[0, 0])
    gamma_a, gamma_b = 2 * ca @ ca.T, 2 * cb @ cb.T

    mf = mol.KS(xc="PBE")
    v_nuc_dropped = np.zeros((mol.nao, mol.nao))
    for atm in np.where(~keep)[0]:
        with mol.with_rinv_at_nucleus(atm):
            v_nuc_dropped -= mol.atom_charge(atm) * mol.intor("int1e_rinv")
    ref = mf.get_veff(mol, gamma_a + gamma_b) - mf.get_veff(mol, gamma_a) + v_nuc_dropped
    ao = np.concatenate([np.arange(*mol.aoslice_by_atom()[atm][2:]) for atm in np.where(keep)[0]])
    ref = ref[np.ix_(ao, ao)]

    vemb = np.asarray(emb.A_LL.full_mat_to_truncated(emb.vemb)[0, 0])
    corr = np.asarray(emb.A_LL.full_mat_to_truncated(emb.vemb_trunc_corr)[0, 0])
    # Residual is the XC integration on the truncated layer's (smaller) grid.
    assert np.abs(vemb - ref).max() < 1e-3
    # Without the correction the truncated density's two-electron potential remains.
    assert np.abs(vemb - corr - ref).max() > 1e-2


def test_same_functional_is_exact_with_truncation(pbe_in_pbe_trunc):
    emb, _ = pbe_in_pbe_trunc
    assert emb.DFT_AinB_total_energy == pytest.approx(emb.subsys_AB_lowlvl_scftotalen, abs=1e-6)


def test_untruncated_matches_1storder(pbe0_in_pbe):
    assert pbe0_in_pbe[("embedded_ll_reference", None)] == \
        pytest.approx(pbe0_in_pbe[("1storder", None)], abs=1e-6)


def test_truncation_error_is_reduced(pbe0_in_pbe):
    err = {corr: pbe0_in_pbe[(corr, THRESH)] - pbe0_in_pbe[(corr, None)]
           for corr in ("1storder", "embedded_ll_reference")}
    assert abs(err["embedded_ll_reference"]) < 0.1 * abs(err["1storder"])
