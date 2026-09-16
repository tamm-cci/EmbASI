import numpy as np
import pyscf
from pyscf.pbc.tools.pyscf_ase import PySCF, ase_atoms_to_pyscf
from ase import Atoms
from embasi.embedding import ProjectionEmbedding
from embasi.parallel_utils import root_print

'''
A minimal open-shell (UKS-in-UKS) QM/QM embedding example: an OH
radical (a doublet, one unpaired electron) embedded in a water
molecule acting as a closed-shell environment.

This exercises EmbASI's open-shell/UHF-UKS support:
- The supersystem reference (AB_LL) runs unrestricted (mol.spin=1, the
  whole system's Nalpha - Nbeta), since it must be run open-shell to
  expose two independent (alpha and beta) sets of occupied MOs to
  localise.
- SPADE localisation partitions the alpha and beta occupied spaces
  *independently* per atom-fragment, but the beta-channel cutoff is
  deliberately not re-derived from its own singular-value gap: it is
  forced from the alpha-channel cutoff so that the whole supersystem's
  spin is carried entirely by the embedded fragment (A) and the
  environment (B) nets to exactly zero spin - matching the physical
  picture of a single localised radical in an otherwise closed-shell
  environment.
- Each fragment's own Mole is then run at the resulting spin
  (mol.spin=1 for the OH radical, mol.spin=0 for the water
  environment), while still using the same UKS class throughout - so
  the environment is solved at the unrestricted level even though its
  net spin is zero, allowing it to locally spin-polarise in response
  to the embedding potential from the radical.

Note: EmbASI currently requires the environmental variable
'ASI_LIB_PATH' to be set even for a pure-PySCF run (it is read
unconditionally in EmbeddingBase.__init__, though PySCF itself never
uses it) - point it at any FHI-aims shared library on your system.
'''

# OH radical (region 1 / high-level, doublet) well separated from a
# water molecule (region 2 / low-level environment).
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

# spin=1: Nalpha - Nbeta for the whole (OH radical + H2O) supersystem -
# the one unpaired electron on the OH radical.
mol = pyscf.M(atom=ase_atoms_to_pyscf(atoms), basis='sto-3g', spin=1)

mf_ll = mol.UKS(xc='PBE')
mf_hl = mol.UKS(xc='PBE0')

calc_ll = PySCF(method=mf_ll)
calc_hl = PySCF(method=mf_hl)

Projection = ProjectionEmbedding(atoms,
                                 embed_mask=embed_mask,
                                 calc_base_ll=calc_ll,
                                 calc_base_hl=calc_hl,
                                 projection="level-shift",
                                 mu_val=1.e+6,
                                 parallel=False)

root_print('\nRunning OH radical-in-water embedding (UKS-in-UKS)\n')
Projection.run()
root_print('Finished\n')

root_print(f"Subsystem A (embedded OH radical) spin, Nalpha-Nbeta: {Projection.A_spin}")
root_print(f"Subsystem B (water environment) spin, Nalpha-Nbeta: {Projection.B_spin}")

pbe0_in_pbe_energy = Projection.DFT_AinB_total_energy
root_print(f"Final PBE0-in-PBE total energy: {pbe0_in_pbe_energy} eV")
