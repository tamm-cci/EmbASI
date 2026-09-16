import numpy as np
import pyscf
from pyscf.pbc.tools.pyscf_ase import PySCF, ase_atoms_to_pyscf
from ase import Atoms
from embasi.embedding import ProjectionEmbedding
from embasi.parallel_utils import root_print

'''
The same open-shell (UKS-in-UKS) OH-radical-in-water system as
oh_radical_in_water_uks.py, but run with the freeze-and-thaw
self-consistent Huzinaga scheme (projection="huzinaga-sc",
freeze_and_thaw=True) instead of a single-shot level-shift embedding.

This exercises the self-consistent Huzinaga Fock-override injected
into PySCF's own SCF loop (embasi.pyscf_scf_hooks) under an
unrestricted (n_spins=2) reference, and confirms the environment
fragment (B_LL), which only actually runs within the freeze-and-thaw
cycle, correctly picks up its derived zero spin rather than the whole
supersystem's.

See oh_radical_in_water_uks.py for the physical picture (spin carried
entirely by the embedded radical, environment nets to zero spin but
is still solved unrestricted).

Note: EmbASI currently requires the environmental variable
'ASI_LIB_PATH' to be set even for a pure-PySCF run (it is read
unconditionally in EmbeddingBase.__init__, though PySCF itself never
uses it) - point it at any FHI-aims shared library on your system.
'''

atoms = Atoms(
    'OHOHH',
    positions=[
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.97],
        [4.0, 0.0, 0.0],
        [4.0, 0.0, 0.96],
        [4.9, 0.0, -0.3],
    ],
)
embed_mask = [1, 1, 2, 2, 2]

mol = pyscf.M(atom=ase_atoms_to_pyscf(atoms), basis='sto-3g', spin=1)

mf_ll = mol.UKS(xc='PBE')
mf_hl = mol.UKS(xc='PBE0')

calc_ll = PySCF(method=mf_ll)
calc_hl = PySCF(method=mf_hl)

Projection = ProjectionEmbedding(atoms,
                                 embed_mask=embed_mask,
                                 calc_base_ll=calc_ll,
                                 calc_base_hl=calc_hl,
                                 projection="huzinaga-sc",
                                 freeze_and_thaw=True,
                                 fat_mixing=0.3,
                                 parallel=False)

root_print('\nRunning OH radical-in-water freeze-and-thaw embedding (UKS-in-UKS)\n')
Projection.run()
root_print('Finished\n')

root_print(f"Subsystem A (embedded OH radical) spin, Nalpha-Nbeta: {Projection.A_spin}")
root_print(f"Subsystem B (water environment) spin, Nalpha-Nbeta: {Projection.B_spin}")
root_print(f"B_LL fragment spin actually used (via freeze-and-thaw): {Projection.B_LL.fragment_spin}")

pbe0_in_pbe_energy = Projection.DFT_AinB_total_energy
root_print(f"Final PBE0-in-PBE total energy: {pbe0_in_pbe_energy} eV")
