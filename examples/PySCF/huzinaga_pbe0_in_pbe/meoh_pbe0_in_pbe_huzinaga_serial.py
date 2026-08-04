import os
import numpy as np
import pyscf
from pyscf.pbc.tools.pyscf_ase import PySCF, ase_atoms_to_pyscf
from ase.data.s22 import s26, create_s22_system
from embasi.embedding import ProjectionEmbedding
from embasi.parallel_utils import root_print

'''
A minimal working example for running a PBE0-in-PBE QM/QM embedding
simulation of a methanol monomer with PySCF, using the self-consistent
Huzinaga projector (projection="huzinaga-sc") instead of the level-shift
projector used in the other PySCF examples. The embedded OH fragment is
treated with PBE0, the remainder of the molecule (the CH3 fragment) with
PBE. Unlike level-shift, which enforces A/B orthogonality only softly
(via a large but finite energy penalty, mu_val), the Huzinaga projector
enforces it as a hard constraint solved self-consistently as part of the
SCF cycle - a good check of this is that the projection-operator energy
correction (PB_corr) should come out essentially zero, unlike the small
but non-zero value level-shift gives.

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

# Embedding mask: 1 = high-level (PBE0), 2 = low-level (PBE).
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
# - Embedding mask (1=High-level (PBE0), 2=Low-level (PBE))
# - Assigned higher and lower level calculators
# - projection="huzinaga-sc": self-consistent Huzinaga projection instead
#   of level-shift - no mu_val needed, since there is no level-shift
#   penalty factor for this projection scheme
Projection = ProjectionEmbedding(atoms,
                                 embed_mask=sort_embed_mask,
                                 calc_base_ll=calc_ll,
                                 calc_base_hl=calc_hl,
                                 projection="huzinaga-sc",
                                 parallel=False)

# Now run the simulation!
root_print('\nRunning MeOH monomer (PBE0-in-PBE, Huzinaga projection)\n')
Projection.run()
root_print('Finished running MeOH monomer\n')

# Total energy for the embedded fragment:
meoh_pbe0inpbe_huz_energy = Projection.DFT_AinB_total_energy

root_print(f"Final PBE0-in-PBE (Huzinaga) total energy: {meoh_pbe0inpbe_huz_energy} eV")
root_print(f"Projection operator energy correction (PB_corr, expect ~0 for Huzinaga): {Projection.PB_corr} eV")
