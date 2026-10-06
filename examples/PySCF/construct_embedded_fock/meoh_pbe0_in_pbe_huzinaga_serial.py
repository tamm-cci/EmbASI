import os
import numpy as np
import pyscf
from pyscf.pbc.tools.pyscf_ase import PySCF, ase_atoms_to_pyscf
from ase.data.s22 import s26, create_s22_system
from embasi.embedding import ProjectionEmbedding
from embasi.parallel_utils import root_print

'''
A minimal working example for creating an embedded Fock matrix with
the localised density matrix for use in workflows outside of the
ProjectionEmbedding workflow.

Presently, only the level-shift operator is recommended for external
workflows. The self-consistent Huzinaga method requires the
construction of the embedding potential from the 'high-level' Hamiltonian/
Fock matrix within the SCF workflow. This means the Huzinaga
embedding projection cannot be exported as a constant offset
of the Hamiltonian
'''

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
mf_hl = mol.KS(xc='PBE')

calc_ll = PySCF(method=mf_ll)
calc_hl = PySCF(method=mf_hl)

Projection = ProjectionEmbedding(atoms,
                                 embed_mask=sort_embed_mask,
                                 calc_base_ll=calc_ll,
                                 calc_base_hl=calc_hl,
                                 projection="level-shift",
                                 parallel=False)

# Now run the simulation!
dm_a, dm_b, fock_matrix = Projection.construct_embedded_fock()

# Access localised occupied molecular orbital coefficients
mo_loc_a_ll = Projection.mo_coeffs_A_LL
mo_loc_b_ll = Projection.mo_coeffs_B_LL

# Example for constructing a new Fock matrix from the old density matrix
dm_a, dm_b, new_fock_matrix = Projection.construct_embedded_fock(dmab_in = dm_a + dm_b)
