import os
import numpy as np
import pyscf
from pyscf import mp
from pyscf.pbc.tools.pyscf_ase import PySCF, ase_atoms_to_pyscf
from ase.data.s22 import s26, create_s22_system
from embasi.embedding import ProjectionEmbedding
from embasi.parallel_utils import root_print

'''
A minimal working example for running an MP2-in-PBE0 QM/QM embedding
simulation of a methanol monomer with PySCF, using EmbASI's post_scf_calc
option to pass a custom, pre-configured MP2 solver object.

post_scf accepts either a bare method name (see the sibling
meoh_mp2_in_pbe0_serial.py example) or a pre-configured PySCF post-HF
object directly, letting you set solver-level options (frozen core,
convergence thresholds, ...) that the string form has no way to express.
EmbASI reuses those settings but replaces the solver's reference
(mean-field object, mol, MO coefficients/occupations) with the actual
embedded A_HL reference at run time, so the object you build it from
does not need to relate to the real system at all.

Note: EmbASI currently requires the environmental variable
'ASI_LIB_PATH' to be set even for a pure-PySCF run (it is read
unconditionally in EmbeddingBase.__init__, though PySCF itself never
uses it) - point it at any FHI-aims shared library on your system.
'''

# Import a methanol monomer (first 6 atoms of the s26 methanol dimer:
# C, O, H, H, H, H)
methanol_dimer_idx = s26[22]
atoms = create_s22_system(methanol_dimer_idx)[:6]

# Embedding mask: 1 = high-level (PBE0 + MP2), 2 = low-level (PBE).
# Atom 1 is O, atom 5 is the hydroxyl H bonded to it - together they
# make up the embedded OH fragment.
embed_mask = len(atoms) * [2]
embed_mask[1] = 1
embed_mask[5] = 1

# ProjectionEmbedding requires embed_mask sorted so that region 1 atoms
# come first; reorder atoms to match before building the PySCF Mole so
# the two stay in sync.
idx_list = np.argsort(embed_mask)
sort_embed_mask = np.sort(embed_mask)
atoms = atoms[idx_list]

mol = pyscf.M(atom=ase_atoms_to_pyscf(atoms), basis='ccpvtz')

mf_ll = mol.KS(xc='PBE')
mf_hl = mol.KS(xc='PBE0')

calc_ll = PySCF(method=mf_ll)
calc_hl = PySCF(method=mf_hl)

# Build a custom, pre-configured MP2 solver template. It is built on a
# throwaway H2/STO-3G reference purely to get an object of the right
# type/shape to configure - EmbASI overwrites its _scf/mol/mo_coeff/
# mo_occ/mo_energy with the real, embedded reference before running it,
# so only the solver settings below (frozen, verbose) are actually used.
dummy_mol = pyscf.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')
dummy_mf = dummy_mol.RHF().run()

mp2_template = mp.MP2(dummy_mf)
mp2_template.frozen = 1     # freeze the O 1s core orbital
mp2_template.verbose = 4

# Set up ProjectionEmbedding, with:
# - Embedding mask (1=High-level (PBE0+MP2), 2=Low-level (PBE))
# - Assigned higher and lower level calculators
# - post_scf: correlated wavefunction correction applied to the
#   converged, embedded A_HL reference only (non-self-consistent).
#   Passing the custom MP2 solver object directly (rather than the bare
#   "MP2" string) is what makes its frozen/verbose settings take effect.
Projection = ProjectionEmbedding(atoms,
                                 embed_mask=sort_embed_mask,
                                 calc_base_ll=calc_ll,
                                 calc_base_hl=calc_hl,
                                 post_scf=mp2_template,
                                 projection="level-shift",
                                 mu_val=1.e+6,
                                 parallel=False)

# Now run the simulation!
root_print('\nRunning MeOH monomer (frozen-core MP2-in-PBE0)\n')
Projection.run()
root_print('Finished running MeOH monomer\n')

# Total energy for the embedded fragment, including the frozen-core MP2
# correction:
meoh_mp2inpbe0_energy = Projection.DFT_AinB_total_energy

# The bare MP2 correlation energy added on top of the embedded PBE0
# reference may be accessed separately:
mp2_correction = Projection.A_HL.post_scf_corr_energy - Projection.A_HL.dft_energy

root_print(f"Final frozen-core MP2-in-PBE0 total energy: {meoh_mp2inpbe0_energy} eV")
root_print(f"Frozen-core MP2 correlation correction (A_HL only): {mp2_correction} eV")
