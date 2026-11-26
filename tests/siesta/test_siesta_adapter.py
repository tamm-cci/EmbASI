"""Tests for the SIESTA QM code adapter.

The unit tests need only ASE: they check the FDF keywords the adapter sets,
the siesta.fdf it writes (including ghost/floating-orbital species), and the
selection of the converged matrices from SIESTA's per-SCF-step exports.

The integration tests run SIESTA through ASI (including a PBE-in-PBE
projection embedding) and need ASI_LIB_PATH to point to a SIESTA library built
with ASI and ScaLAPACK, plus SIESTA_PP_PATH holding C, H and N pseudopotentials
(e.g. SIESTA's Tests/20.QuantumComputing). They are skipped otherwise.
"""
import os
import re
from types import SimpleNamespace

import numpy as np
import pytest
from ase.build import molecule
from ase.calculators.siesta import Siesta

from embasi.qmcode_adapters import SiestaAdapter, qm_code_adapter


def make_calc(**fdf):
    return Siesta(label="h2o", xc="PBE", basis_set="DZP", mesh_cutoff=200 * 13.605693,
                  fdf_arguments=dict(fdf))


def fdf_lines(path):
    return [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]


def fdf_value(lines, label):
    """Values of all lines whose (normalised) label is `label`"""
    norm = lambda s: s.lower().replace('.', '').replace('_', '').replace('-', '')
    return [l.split(None, 1)[1] if len(l.split(None, 1)) > 1 else ''
            for l in lines if norm(l.split()[0]) == norm(label)]


def test_factory_selects_siesta_adapter():
    adapter = qm_code_adapter(make_calc())
    assert isinstance(adapter, SiestaAdapter)
    assert adapter.asi_flavour == 3
    assert adapter.uses_asi_callbacks
    # Energy-only runs mix the DM, so E_KS is already that of the input DM
    assert not adapter.needs_nonscf_ene_corr


def test_charge_replaces_spelling_variants():
    adapter = SiestaAdapter()
    calc = adapter.set_qm_total_charge(None, make_calc(net_charge=2), -1)
    assert calc.parameters["fdf_arguments"] == {"NetCharge": -1.0}


def test_noscf_parameters():
    adapter = SiestaAdapter()
    for calc in (adapter.set_noscf_calc_params(make_calc()),
                 adapter.set_no_scf_total_energy_calc(make_calc())):
        args = calc.parameters["fdf_arguments"]
        assert args["MaxSCFIterations"] == 1
        assert "SCF.Mixer.Weight" not in args  # SIESTA rejects a zero weight
        assert args["SCF.MustConverge"] is False
        assert args["SCF.Mix"] == "density"  # E_KS evaluated at the input DM


def test_max_scf_cycles_floor():
    adapter = SiestaAdapter()
    assert adapter.set_max_scf_cycles(make_calc(), 0).parameters["fdf_arguments"]["MaxSCFIterations"] == 1
    assert adapter.set_max_scf_cycles(make_calc(), 40).parameters["fdf_arguments"]["MaxSCFIterations"] == 40


def test_full_scf_is_unchanged_and_unsupported_modes_raise():
    adapter = SiestaAdapter()
    calc = make_calc(MeshCutoff="200 Ry")
    assert adapter.set_full_scf_calc(calc).parameters["fdf_arguments"] == {"MeshCutoff": "200 Ry"}
    # The embedding potential enters through the modify-Hamiltonian callback
    calc = make_calc(MeshCutoff="200 Ry")
    assert adapter.set_full_scf_embed_calc(calc).parameters["fdf_arguments"] == {"MeshCutoff": "200 Ry"}
    with pytest.raises(NotImplementedError):
        adapter.set_embasi_calculation_type(make_calc(), "frozendensity-write")
    with pytest.raises(NotImplementedError):
        adapter.set_postscf_keyword(make_calc(), "MP2")
    assert adapter.set_postscf_keyword(calc, None) is calc
    with pytest.raises(NotImplementedError):
        adapter.set_qm_localise(make_calc())


@pytest.fixture
def pseudo_dir(tmp_path, monkeypatch):
    pp = tmp_path / "pseudos"
    pp.mkdir()
    for sym in ("H", "O"):
        (pp / f"{sym}.psml").write_text("dummy\n")
    monkeypatch.setenv("SIESTA_PP_PATH", str(pp))
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return work


def test_write_input_ghosts(pseudo_dir):
    adapter = SiestaAdapter()
    atoms = molecule("H2O")  # O, H, H
    calc = make_calc(**{"DM.UseSaveDM": True})
    calc = adapter.set_ghost_atoms(None, calc, [False, False, True])
    calc = adapter.set_qm_total_charge(None, calc, 0)
    adapter.write_input(None, calc, atoms)

    assert (pseudo_dir / "siesta.fdf").exists(), "ASI reads siesta.fdf"
    lines = fdf_lines(pseudo_dir / "siesta.fdf")
    text = "\n".join(lines)

    # Species: O, H and a floating-orbital (negative Z) ghost H
    labels = re.search(r"%block ChemicalSpecieslabel\n(.*?)\n%endblock", text, re.S).group(1)
    z = sorted(int(row.split()[1]) for row in labels.splitlines())
    assert z == [-1, 1, 8]
    ghost_row = [row.split() for row in labels.splitlines() if int(row.split()[1]) == -1][0]
    assert "ghost" in ghost_row[2]

    # The ghost atom (last) carries the ghost species; the real H the H species
    coords = re.search(r"%block AtomicCoordinatesAndAtomicSpecies\n(.*?)\n%endblock", text, re.S).group(1)
    species_of_atom = [int(row.split()[3]) for row in coords.splitlines()]
    ghost_species = int(ghost_row[0])
    assert species_of_atom[2] == ghost_species
    assert species_of_atom[1] != ghost_species

    # Ghosts keep the basis of the real species
    sizes = re.search(r"%block PAO.BasisSizes\n(.*?)\n%endblock", text, re.S).group(1)
    assert all(row.split()[1] == "DZP" for row in sizes.splitlines())

    assert fdf_value(lines, "SolutionMethod") == ["diagon"]
    assert fdf_value(lines, "NetCharge") == ["0.0"]
    # The user's DM.UseSaveDM is kept, and ASE does not add a second one
    assert len(fdf_value(lines, "DM.UseSaveDM")) == 1

    # The calculator itself is left untouched by the ghost bookkeeping
    assert list(calc["species"]) == []
    assert np.all(atoms.get_tags() == 0)


def test_write_input_defaults_without_ghosts(pseudo_dir):
    adapter = SiestaAdapter()
    calc = adapter.set_ghost_atoms(None, make_calc(), [False] * 3)
    adapter.write_input(None, calc, molecule("H2O"))
    lines = fdf_lines(pseudo_dir / "siesta.fdf")
    assert fdf_value(lines, "DM.UseSaveDM") == ["F"] or fdf_value(lines, "DM.UseSaveDM") == ["False"] \
        or fdf_value(lines, "DM.UseSaveDM") == [".false."]
    assert "-1" not in "".join(fdf_value(lines, "NumberOfSpecies"))
    assert fdf_value(lines, "NumberOfSpecies") == ["2"]


def fake_run(n_basis=3, n_scf=5):
    """Mimics asi storage after an n_scf-step SIESTA run

    With ASI.ExportHamComps SIESTA exports T, V_KB and H on every SCF step;
    ham_saving_callback keeps a rolling window of the last three exports.
    """
    rng = np.random.default_rng(0)
    kin = rng.standard_normal((n_basis, n_basis))
    vkb = rng.standard_normal((n_basis, n_basis))
    hams = [rng.standard_normal((n_basis, n_basis)) for _ in range(n_scf)]
    dms = [rng.standard_normal((n_basis, n_basis)) for _ in range(n_scf)]
    exports = [m for h in hams for m in (kin, vkb, h)]
    ham_storage = {i: {(0, 0): m} for i, m in enumerate(exports[-3:])}
    asi = SimpleNamespace(
        n_kpts=1, n_spin=1, n_basis=n_basis, basis_atoms=[0] * n_basis,
        ham_count=3, ham_storage=ham_storage,
        dm_count=n_scf, dm_storage={i: {(0, 0): d} for i, d in enumerate(dms)},
        overlap_storage={(0, 0): np.eye(n_basis)})
    atomsembed = SimpleNamespace(atoms=SimpleNamespace(calc=SimpleNamespace(asi=asi)),
                                 truncate=False)
    return atomsembed, dict(kin=kin, vkb=vkb, ham=hams[-1], dm=dms[-1])


@pytest.mark.parametrize("n_scf", [1, 5])
def test_extract_matrices_takes_converged_exports(n_scf):
    atomsembed, ref = fake_run(n_scf=n_scf)
    res = SiestaAdapter().extract_matrices(atomsembed)
    assert np.array_equal(res["ham_tot"][0, 0], ref["ham"])
    assert np.array_equal(res["ham_kin"][0, 0], ref["kin"])
    assert np.array_equal(res["ham_vkb"][0, 0], ref["vkb"])
    # As for FHI-aims, ham_kin + ham_estat_xc is the total Hamiltonian
    assert np.allclose(res["ham_kin"][0, 0] + res["ham_estat_xc"][0, 0], ref["ham"])
    assert np.array_equal(res["dm"][0, 0], ref["dm"])
    assert np.array_equal(res["ovlp"][0, 0], np.eye(3))
    assert res["dm_localised"] is None


def test_extract_matrices_needs_component_exports():
    atomsembed, _ = fake_run(n_scf=1)
    atomsembed.atoms.calc.asi.ham_count = 1  # a library without ExportHamComps
    with pytest.raises(RuntimeError, match="ExportHamComps"):
        SiestaAdapter().extract_matrices(atomsembed)


def test_extract_matrices_no_exports():
    atomsembed, _ = fake_run()
    atomsembed.atoms.calc.asi.ham_count = 0
    with pytest.raises(RuntimeError, match="MaxSCFIterations"):
        SiestaAdapter().extract_matrices(atomsembed)


def test_valence_electrons_from_pseudopotentials(tmp_path, monkeypatch):
    """Pseudopotential valence charges, found as ASE's Siesta finds them"""
    pp = tmp_path / "pseudos"
    pp.mkdir()
    (pp / "N.psml").write_text('<psml>\n<pseudo-atom-spec atomic-label="N" '
                               'z-pseudo="5" relativity="no">\n</psml>\n')
    (pp / "H.gga.psf").write_text(" H  ca nrl nc\n ATM3\n 1s 1.00\n"
                                  "   3  0 1006  0.4E-03  0.125E-01   1.00000000000\n")
    monkeypatch.setenv("SIESTA_PP_PATH", str(pp))
    nelec = SiestaAdapter().atom_nelectrons(molecule("NH3"), make_calc())
    assert list(nelec) == [5.0, 1.0, 1.0, 1.0]


def test_write_input_requests_hamiltonian_components(pseudo_dir):
    adapter = SiestaAdapter()
    calc = adapter.set_ghost_atoms(None, make_calc(), [False] * 3)
    adapter.write_input(None, calc, molecule("H2O"))
    lines = fdf_lines(pseudo_dir / "siesta.fdf")
    assert [v.lower() for v in fdf_value(lines, "ASI.ExportHamComps")] in (["true"], ["t"], [".true."])
    assert fdf_value(lines, "QComp.Workflow") == []  # no ERIs unless requested


def test_write_input_requests_coulomb_eris(pseudo_dir):
    adapter = SiestaAdapter()
    assert adapter.supports_eri_export
    calc = adapter.set_ghost_atoms(None, make_calc(), [False] * 3)
    adapter.write_input(SimpleNamespace(export_eri=True), calc, molecule("H2O"))
    lines = fdf_lines(pseudo_dir / "siesta.fdf")
    assert fdf_value(lines, "QComp.Workflow") == ["DROP"]
    assert fdf_value(lines, "QComp.Protocol") == ["FILE"]
    assert fdf_value(lines, "QComp.DataAddress") == [SiestaAdapter.ERI_DATA_FILE]
    assert fdf_value(lines, "QComp.ERIPotential") == ["coulomb"]


def test_extract_matrices_returns_eris():
    from embasi.eri import ASI_ERI_NOTATION_CHEMIST, ASI_ERI_SYM_4
    atomsembed, _ = fake_run(n_basis=3)
    asi = atomsembed.atoms.calc.asi
    rng = np.random.default_rng(1)
    eri = rng.standard_normal((6, 6))
    eri = eri + eri.T
    key = (1, 1, 1, 1, 1, 1)
    asi.eri_storage = {key: eri}
    asi.eri_descr_storage = {key: {"sum_notation": ASI_ERI_NOTATION_CHEMIST, "n_index": 0,
                                   "basis_type": 0, "eri_symmetry": ASI_ERI_SYM_4}}
    atomsembed.export_eri = False
    assert SiestaAdapter().extract_matrices(atomsembed)["eri"] is None
    atomsembed.export_eri = True
    res = SiestaAdapter().extract_matrices(atomsembed)
    assert res["eri"].nao == 3 and np.array_equal(res["eri"].eri_s4, eri)


def test_extract_matrices_missing_eris():
    atomsembed, _ = fake_run(n_basis=3)
    asi = atomsembed.atoms.calc.asi
    asi.eri_storage, asi.eri_descr_storage = {}, {}
    atomsembed.export_eri = True
    with pytest.raises(RuntimeError, match="SIESTA_WITH_QCOMP"):
        SiestaAdapter().extract_matrices(atomsembed)


def _siesta_lib():
    lib = os.environ.get("ASI_LIB_PATH")
    if not lib or not os.path.exists(lib):
        return None
    from ctypes import CDLL, RTLD_GLOBAL
    try:
        return lib if CDLL(lib, mode=RTLD_GLOBAL).ASI_flavour() == 3 else None
    except (OSError, AttributeError):
        return None


# Each stage runs in its own interpreter, so a SIESTA failure (error stop,
# segfault) fails the test rather than killing pytest.
_STAGE = r"""
import json, os, re, sys
import numpy as np
from ase.build import molecule
from ase.calculators.siesta import Siesta
from embasi.atoms_embedding_asi import AtomsEmbed
from embasi.ks_array import SpinKpointArray

HA = 27.211384500
stage, outdir, dm_file = sys.argv[1:4]


def make_calc(**fdf):
    args = {"MaxSCFIterations": 100, "SCF.DM.Tolerance": 1e-6}
    args.update(fdf)
    return Siesta(label="siesta", xc="PBE", basis_set="DZ", mesh_cutoff=200 * 13.605693,
                  fdf_arguments=args)


def scf_eks(outdir):
    return [float(l.split()[3]) for l in open(os.path.join(outdir, "asi.log"))
            if re.match(r"\s*scf:\s+\d", l)]


atoms = molecule("NH3", vacuum=4.0)
if stage == "scf":
    emb = AtomsEmbed(atoms, make_calc(), embed_mask=None, outdir=outdir)
    emb.run_scf()
    np.save(dm_file, emb.density_matrices_out[0, 0])
    S, D = emb.overlap[0, 0], emb.density_matrices_out[0, 0]
    T, H = emb.hamiltonian_kinetic[0, 0], emb.hamiltonian_total[0, 0]
    out = dict(energy=emb.total_energy, dft_energy=emb.dft_energy,
               nelec=float(np.trace(D @ S)), n_basis=emb.n_basis,
               ham_shape=list(H.shape),
               kin_plus_rest=bool(np.allclose(T + emb.hamiltonian_estat_plus_xc[0, 0], H)),
               kin_sym=float(abs(T - T.T).max()))
elif stage == "onestep":
    # One SCF step from the atomic guess: Tr(D T) and Tr(D V_KB) are SIESTA's
    # Ekin and Enl of that step, and its E_KS is the exact energy of D_out.
    emb = AtomsEmbed(atoms, make_calc(**{"MaxSCFIterations": 1, "SCF.MustConverge": False}),
                     embed_mask=None, outdir=outdir)
    emb.run_scf()
    np.save(dm_file, emb.density_matrices_out[0, 0])
    log = open(os.path.join(outdir, "asi.log")).read()
    D = emb.density_matrices_out[0, 0]
    vkb = emb.atoms.calc.asi.ham_storage[1][(0, 0)]
    out = dict(eks=scf_eks(outdir)[-1],
               tr_dt=float(np.trace(D @ emb.hamiltonian_kinetic[0, 0])) * HA,
               tr_dv=float(np.trace(D @ vkb)) * HA,
               ekin=float(re.search(r"siesta: Ekin\s*=\s*(\S+)", log).group(1)),
               enl=float(re.search(r"siesta: Enl\s*=\s*(\S+)", log).group(1)))
elif stage == "noscf":
    emb = AtomsEmbed(atoms, make_calc(), embed_mask=None, outdir=outdir)
    emb.run_noscf(dm_in=SpinKpointArray({(0, 0): np.load(dm_file)}, 1, 1))
    out = dict(energy=emb.total_energy, ev_corr_energy=emb.ev_corr_total_energy)
elif stage == "embed":
    # PBE-in-PBE projection embedding of methylamine (NH2 | CH3): with the
    # same functional at both levels it reproduces the supersystem energy.
    from embasi.embedding import ProjectionEmbedding
    atoms = molecule("H3CNH2", vacuum=4.0)
    iN = atoms.get_chemical_symbols().index("N")
    mask = [1 if (a.symbol == "N" or (a.symbol == "H" and
                  np.linalg.norm(a.position - atoms[iN].position) < 1.2)) else 2
            for a in atoms]
    emb = ProjectionEmbedding(atoms, embed_mask=mask, calc_base_ll=make_calc(),
                              calc_base_hl=make_calc(), mu_val=1.e+6,
                              run_dir=outdir, parallel=False)
    emb.run()
    out = dict(ab=emb.subsys_AB_lowlvl_scftotalen, a_ll=emb.subsys_A_lowlvl_totalen,
               a_hl=emb.subsys_A_highlvl_totalen, embedded=emb.DFT_AinB_total_energy,
               pop=[emb.AB_pop, emb.A_pop, emb.B_pop])
elif stage == "eri":
    # H2 with ERI export (needs a SIESTA_WITH_QCOMP library). The ERIs are
    # computed on SIESTA's Gaussian fits of the NAOs, printed as the NAO2GTO
    # block: PySCF on that basis gives the reference S and (ij|kl).
    from pyscf import gto
    atoms = molecule("H2", vacuum=4.0)
    emb = AtomsEmbed(atoms, make_calc(), embed_mask=None, outdir=outdir)
    emb.export_eri = True
    emb.run_scf()
    eri = emb.eri
    log = open(os.path.join(outdir, "asi.log")).read()
    block = log.split("%block NAO2GTO")[1].split("%endblock NAO2GTO")[0]
    shells = []
    for line in block.splitlines()[2:]:
        p = line.split()
        if len(p) == 4:
            shells.append((int(p[1]), []))
        elif len(p) == 2:
            shells[-1][1].append((float(p[0]), float(p[1])))
    basis = [[l] + [[z, c / gto.gto_norm(l, z)] for z, c in prims] for l, prims in shells]
    mol = gto.M(atom=[("H", a.position) for a in emb.atoms], unit="Angstrom",
                basis={"H": basis}, verbose=0)
    D = emb.density_matrices_out[0, 0]
    ref = mol.intor("int2e", aosym="s4")
    full = mol.intor("int2e")
    out = dict(nao=eri.nao, n_basis=emb.n_basis,
               ovlp_err=float(abs(emb.overlap[0, 0] - mol.intor("int1e_ovlp")).max()),
               eri_err=float(abs(eri.eri_s4 - ref).max()),
               j_err=float(abs(eri.get_j(D) - np.einsum("ijkl,kl->ij", full, D)).max()),
               k_err=float(abs(eri.get_k(D) - np.einsum("ikjl,kl->ij", full, D)).max()))
print("RESULT " + json.dumps(out))
"""


def _run_stage(stage, tmp_path, dm_name="dm.npy"):
    import json
    import subprocess
    import sys
    proc = subprocess.run([sys.executable, "-c", _STAGE, stage, str(tmp_path / stage),
                           str(tmp_path / dm_name)],
                          capture_output=True, text=True, timeout=1800)
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, \
        f"{stage} run failed ({proc.returncode}):\n{proc.stderr[-3000:]}"
    return json.loads(lines[-1][len("RESULT "):])


needs_siesta = pytest.mark.skipif(
    _siesta_lib() is None or not os.environ.get("SIESTA_PP_PATH"),
    reason="needs a SIESTA ASI library in ASI_LIB_PATH and SIESTA_PP_PATH")


@needs_siesta
def test_siesta_scf_and_noscf(tmp_path):
    """Full SCF, then a total-energy-only run from the converged DM"""
    scf = _run_stage("scf", tmp_path)
    assert scf["ham_shape"] == [scf["n_basis"]] * 2
    assert abs(scf["nelec"] - 8.0) < 1e-6
    assert abs(scf["energy"] - scf["dft_energy"]) < 1e-3
    assert scf["kin_plus_rest"] and scf["kin_sym"] < 1e-10

    noscf = _run_stage("noscf", tmp_path)
    assert abs(noscf["ev_corr_energy"] - scf["energy"]) < 1e-3


@needs_siesta
def test_siesta_hamiltonian_components_and_input_dm_energy(tmp_path):
    """T and V_KB match SIESTA's Ekin and Enl, and the total-energy-only run
    returns the KS energy of a non-converged input density matrix"""
    one = _run_stage("onestep", tmp_path, "d1.npy")
    assert abs(one["tr_dt"] - one["ekin"]) < 1e-3
    assert abs(one["tr_dv"] - one["enl"]) < 1e-3

    noscf = _run_stage("noscf", tmp_path, "d1.npy")
    assert abs(noscf["ev_corr_energy"] - one["eks"]) < 1e-4


@needs_siesta
def test_siesta_pbe_in_pbe_projection_embedding(tmp_path):
    res = _run_stage("embed", tmp_path)
    assert np.allclose(res["pop"], [14.0, 8.0, 6.0], atol=1e-6)  # valence electrons
    assert abs(res["a_hl"] - res["a_ll"]) < 1e-2
    assert abs(res["embedded"] - res["ab"]) < 1e-4


def _has_qcomp(lib):
    """Whether `lib` was built with SIESTA_WITH_QCOMP (which computes the ERIs)"""
    from ctypes import CDLL, RTLD_GLOBAL
    if lib is None:
        return False
    so = CDLL(lib, mode=RTLD_GLOBAL)
    return hasattr(so, "ASI_register_eri_callback") and \
        hasattr(so, "__qcomp_eris_MOD_qcomp_setup_hfx")


@needs_siesta
@pytest.mark.skipif(not _has_qcomp(_siesta_lib()),
                    reason="needs a SIESTA_WITH_QCOMP library with the ASI ERI export")
def test_siesta_eri_export(tmp_path):
    """Exported AO ERIs are the bare-Coulomb integrals over SIESTA's Gaussian
    fits of the NAOs, and their J/K contractions match"""
    res = _run_stage("eri", tmp_path)
    assert res["nao"] == res["n_basis"]
    assert res["ovlp_err"] < 1e-4      # the fitted basis is the NAO basis
    assert res["eri_err"] < 1e-6
    assert res["j_err"] < 1e-6 and res["k_err"] < 1e-6
