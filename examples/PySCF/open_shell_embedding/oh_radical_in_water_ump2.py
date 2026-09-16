import numpy as np
import pyscf
from pyscf.pbc.tools.pyscf_ase import PySCF, ase_atoms_to_pyscf
from ase import Atoms
from embasi.embedding import ProjectionEmbedding
from embasi.parallel_utils import root_print

'''
UMP2-in-PBE0 on the same open-shell OH-radical-in-water system as
oh_radical_in_water_uks.py, using EmbASI's post_scf functionality to
add a non-self-consistent MP2 correlation correction on top of the
converged, embedded UKS reference.

PySCFAdapter.run_scf converts the converged UKS reference to UHF (via
mf.to_hf(), re-attaching the embedding potential's get_fock override -
see pyscf_postscf.run_postscf) before building the post-HF object;
pyscf.mp.MP2() then dispatches on the *class* of that reference, so an
unrestricted (UHF) reference here automatically gives UMP2 rather than
plain (restricted) MP2 - no separate 'UMP2' keyword is needed.

See oh_radical_in_water_uks.py for the physical picture (the one
unpaired electron is carried entirely by the embedded OH fragment; the
water environment nets to zero spin but is still solved unrestricted).

Note: EmbASI currently requires the environmental variable
'ASI_LIB_PATH' to be set even for a pure-PySCF run (it is read
unconditionally in EmbeddingBase.__init__, though PySCF itself never
uses it) - point it at any FHI-aims shared library on your system.
'''

atoms = Atoms(
    'OHOHH',
    positions=[
        [0.0, 0.0, 0.0],    # O (OH radical)
        [0.0, 0.0, 0.97],   # H (OH radical)
        [4.0, 0.0, 0.0],    # O (H2O)
        [4.0, 0.0, 0.96],   # H (H2O)
        [4.9, 0.0, -0.3],   # H (H2O)
    ],
)
embed_mask = [1, 1, 2, 2, 2]

mol = pyscf.M(atom=ase_atoms_to_pyscf(atoms), basis='sto-3g', spin=1)

mf_ll = mol.UKS(xc='PBE')
mf_hl = mol.UKS(xc='PBE0')

calc_ll = PySCF(method=mf_ll)
calc_hl = PySCF(method=mf_hl)

# post_scf="MP2": a non-self-consistent MP2 correlation correction is
# added on top of the converged, embedded A_HL (UKS/PBE0) reference
# only - the reference itself is still found self-consistently at the
# UKS/PBE0 level.
Projection = ProjectionEmbedding(atoms,
                                 embed_mask=embed_mask,
                                 calc_base_ll=calc_ll,
                                 calc_base_hl=calc_hl,
                                 post_scf="MP2",
                                 projection="level-shift",
                                 mu_val=1.e+6,
                                 parallel=False)

root_print('\nRunning OH radical-in-water embedding (UMP2-in-PBE0-in-PBE)\n')
Projection.run()
root_print('Finished\n')

root_print(f"Subsystem A (embedded OH radical) spin, Nalpha-Nbeta: {Projection.A_spin}")
root_print(f"Subsystem B (water environment) spin, Nalpha-Nbeta: {Projection.B_spin}")

# Total energy for the embedded fragment, including the UMP2 correction:
ump2_in_pbe0_energy = Projection.DFT_AinB_total_energy

# The bare UMP2 correlation energy added on top of the embedded PBE0
# reference may be accessed separately:
ump2_correction = Projection.A_HL.post_scf_corr_energy - Projection.A_HL.dft_energy

root_print(f"Final UMP2-in-PBE0 total energy: {ump2_in_pbe0_energy} eV")
root_print(f"UMP2 correlation correction (A_HL only): {ump2_correction} eV")
