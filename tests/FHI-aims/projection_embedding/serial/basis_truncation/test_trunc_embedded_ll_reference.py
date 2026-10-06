"""total_energy_corr='embedded_ll_reference' with basis truncation, FHI-aims.

Methanol (light defaults) with the OH group as subsystem A; a Mulliken threshold
of 0.1 |e| keeps O, H and C and drops the methyl hydrogens from the A layers.
"""
import os

import numpy as np
import pytest
from ase.calculators.aims import Aims, AimsProfile
from ase.data.s22 import s26, create_s22_system

from embasi.embedding import ProjectionEmbedding

THRESH = 0.1
PROJECTIONS = ["huzinaga", "huzinaga-sc"]


def _calc(xc):
    return Aims(xc=xc, profile=AimsProfile(command="asi-doesnt-need-command"),
                KS_method="parallel",
                RI_method="LVL",
                collect_eigenvectors=True,
                density_update_method='density_matrix',
                atomic_solver_xc="PBE",
                compute_kinetic=True,
                override_initial_charge_check=True,
                )


def _run(tmp_path_factory, projection, hl_xc, thresh):
    os.environ["AIMS_SPECIES_DIR"] = os.environ["AIMS_ROOT_DIR"] + "/species_defaults/defaults_2020/light"
    methanol = create_s22_system(s26[22])[:6]
    emb = ProjectionEmbedding(methanol, embed_mask=[2, 1, 2, 2, 2, 1],
                              calc_base_ll=_calc("PBE"), calc_base_hl=_calc(hl_xc),
                              mu_val=1.e+6, projection=projection, localisation="SPADE",
                              truncate_basis_thresh=thresh,
                              total_energy_corr="embedded_ll_reference",
                              run_dir=str(tmp_path_factory.mktemp("MeOH_monomer")))
    emb.run()
    return emb


@pytest.fixture(scope="module", params=PROJECTIONS)
def projection(request):
    return request.param


@pytest.fixture(scope="module")
def pbe0_in_pbe_trunc(tmp_path_factory, projection):
    return _run(tmp_path_factory, projection, "PBE0", THRESH)


def test_same_functional_is_exact_with_truncation(tmp_path_factory, projection):
    emb = _run(tmp_path_factory, projection, "PBE", THRESH)
    assert 0 < emb.basis_info.trunc_natoms < len(emb.AB_LL.atoms)
    assert emb.DFT_AinB_total_energy == pytest.approx(emb.subsys_AB_lowlvl_scftotalen, abs=1e-6)


def test_truncated_two_electron_potential_cancels(pbe0_in_pbe_trunc):
    """v_emb relies on trunc(H^A_full[pad(gamma)]) - H^A_trunc[gamma] holding only the
    one-electron terms of the dropped atoms, i.e. the two layers building the same
    two-electron potential for the same density. Check it is density independent."""
    emb = pbe0_in_pbe_trunc
    cut = lambda m: emb.A_LL.truncated_mat_to_full(emb.A_LL.full_mat_to_truncated(m))
    block = lambda m: np.asarray(emb.A_LL.full_mat_to_truncated(m)[0, 0])

    diffs, ham_full = [], []
    for dm in (emb.A_LL.density_matrices_out.copy(), emb.A_HL.density_matrices_out.copy()):
        dm = cut(dm)
        emb.A_LL.run_noscf(dm_in=dm)
        ham_trunc = block(emb.A_LL.hamiltonian_total)
        emb.A_LL_full.run_noscf(dm_in=dm)
        ham_full.append(block(emb.A_LL_full.hamiltonian_total))
        diffs.append(ham_full[-1] - ham_trunc)

    # Residual is integration grid / Hartree multipole differences between layers.
    residual = np.abs(diffs[0] - diffs[1]).max()
    assert residual < 1e-4
    assert np.abs(ham_full[0] - ham_full[1]).max() > 10 * residual


def test_truncation_error_is_small(pbe0_in_pbe_trunc, tmp_path_factory, projection):
    full = _run(tmp_path_factory, projection, "PBE0", None)
    # 1storder gives ~4.8 eV here.
    assert abs(pbe0_in_pbe_trunc.DFT_AinB_total_energy - full.DFT_AinB_total_energy) < 0.1
