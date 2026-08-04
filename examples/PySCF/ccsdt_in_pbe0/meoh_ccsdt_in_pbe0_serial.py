import os
import numpy as np
import pyscf
from pyscf.pbc.tools.pyscf_ase import PySCF, ase_atoms_to_pyscf
from ase.data.s22 import s26, create_s22_system
from embasi.embedding import ProjectionEmbedding
from embasi.parallel_utils import root_print

'''
A minimal working example for running a CCSD(T)-in-PBE0 QM/QM embedding
simulation of a methanol monomer with PySCF, using EmbASI's post_scf
functionality. The embedded OH fragment is treated with PBE0 self-
consistently, with a CCSD(T) correlation correction (CCSD plus the
perturbative triples correction) added non-self-consistently on top of
the converged, embedded PBE0 reference. The remainder of the molecule
(the CH3 fragment) is treated with PBE.

Note: CCSD(T) is noticeably more expensive than CCSD or MP2 (the
perturbative triples step scales as O(N^7)) - this example still uses
the same small OH-fragment/cc-pVTZ test system as the MP2/CCSD examples
for consistency, but expect it to take longer to run.

Note: EmbASI currently requires the environmental variable
'ASI_LIB_PATH' to be set even for a pure-PySCF run (it is read
unconditionally in EmbeddingBase.__init__, though PySCF itself never
uses it) - point it at any FHI-aims shared library on your system.
'''

try:
    root_print(f"ASI_LIB_PATH: {os.environ['ASI_LIB_PATH']}")
except KeyError:
    raise RuntimeError("Please set the environmental variable: ASI_LIB_PATH")

# Import a methanol monomer (first 6 atoms of the s26 methanol dimer:
# C, O, H, H, H, H)
methanol_dimer_idx = s26[22]
atoms = create_s22_system(methanol_dimer_idx)[:6]

# Embedding mask: 1 = high-level (PBE0 + CCSD(T)), 2 = low-level (PBE).
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

# Set up ProjectionEmbedding, with:
# - Embedding mask (1=High-level (PBE0+CCSD(T)), 2=Low-level (PBE))
# - Assigned higher and lower level calculators
# - post_scf: correlated wavefunction correction applied to the
#   converged, embedded A_HL reference only (non-self-consistent).
#   The "CCSD(T)" string form runs CCSD followed automatically by the
#   perturbative triples correction.
Projection = ProjectionEmbedding(atoms,
                                 embed_mask=sort_embed_mask,
                                 calc_base_ll=calc_ll,
                                 calc_base_hl=calc_hl,
                                 post_scf="CCSD(T)",
                                 projection="level-shift",
                                 mu_val=1.e+6,
                                 parallel=False)

# Now run the simulation!
root_print('\nRunning MeOH monomer (CCSD(T)-in-PBE0)\n')
Projection.run()
root_print('Finished running MeOH monomer\n')

# Total energy for the embedded fragment, including the CCSD(T)
# correction:
meoh_ccsdtinpbe0_energy = Projection.DFT_AinB_total_energy

# The bare CCSD(T) correlation energy (CCSD + perturbative triples)
# added on top of the embedded PBE0 reference may be accessed separately:
ccsdt_correction = Projection.A_HL.post_scf_corr_energy - Projection.A_HL.dft_energy

root_print(f"Final CCSD(T)-in-PBE0 total energy: {meoh_ccsdtinpbe0_energy} eV")
root_print(f"CCSD(T) correlation correction (A_HL only): {ccsdt_correction} eV")
