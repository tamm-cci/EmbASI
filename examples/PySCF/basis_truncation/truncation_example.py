import os
import time
import numpy as np
import pyscf
from pyscf.pbc.tools.pyscf_ase import PySCF, ase_atoms_to_pyscf
from ase.io import read
from embasi.embedding import ProjectionEmbedding
from embasi.parallel_utils import root_print

'''
A PySCF port of the FHI-aims basis truncation example
(examples/FHIaims/basis_truncation/truncation_example.py), for a decanol
molecule. The OH fragment (O and its bonded H) is embedded as the
high-level region, the hydrocarbon chain as the low-level region. Both
regions are run at PBE here (i.e., this is a PBE-in-PBE split) so any
change between the truncated and full-basis runs below is due solely to
basis truncation, not to a difference in level of theory.

Basis truncation removes basis centers which contribute a negligible
amount of charge to the embedded subsystem, thereby reducing the
computational cost. If any real time savings are to be realised with
QM/QM embedding, basis truncation is vital to reduce the cost incurred
by including extra basis functions (either through integration, or
evaluation of nonlocal matrix quantities).

Note: a small basis (STO-3G) is used here purely so the example runs
quickly; swap in a larger basis (e.g. 'def2-svp' or 'ccpvtz') for
production-quality results.

Note: EmbASI currently requires the environmental variable
'ASI_LIB_PATH' to be set even for a pure-PySCF run (it is read
unconditionally in EmbeddingBase.__init__, though PySCF itself never
uses it) - point it at any FHI-aims shared library on your system.
'''

try:
    root_print(f"ASI_LIB_PATH: {os.environ['ASI_LIB_PATH']}")
except KeyError:
    raise RuntimeError("Please set the environmental variable: ASI_LIB_PATH")

# Read decanol geometry and set up embedding mask: 1 = high-level (embedded
# OH fragment: the O and its bonded hydroxyl H), 2 = low-level (the
# hydrocarbon chain).
decanol = read("decanol.xyz")
embed_mask = len(decanol) * [2]
embed_mask[0], embed_mask[32] = 1, 1

# Unlike FHI-aims (whose calculator rebuilds geometry.in from the ASE
# atoms object on every call, so ProjectionEmbedding's internal per-layer
# reordering to match embed_mask is transparent), PySCF's Mole is built
# once, up front, from whatever atom order it's given here. Reorder atoms
# and embed_mask to match before building it, so that internal reordering
# is a no-op and stays in sync with this fixed Mole.
idx_list = np.argsort(embed_mask)
sort_embed_mask = np.sort(embed_mask)
decanol = decanol[idx_list]

mol = pyscf.M(atom=ase_atoms_to_pyscf(decanol), basis='sto-3g')

mf_ll = mol.KS(xc='PBE')
mf_hl = mol.KS(xc='PBE')

calc_ll = PySCF(method=mf_ll)
calc_hl = PySCF(method=mf_hl)

# Set up ProjectionEmbedding, with:
# - Embedding mask (1=High-level, 2=Low-level)
# - Assigned higher and lower level calculators
# - truncate_basis_thresh: Mulliken charge threshold (in |e|) below which
#   an environment atom's basis functions are dropped entirely
Projection = ProjectionEmbedding(decanol,
                                 embed_mask=sort_embed_mask,
                                 calc_base_ll=calc_ll,
                                 calc_base_hl=calc_hl,
                                 projection="huzinaga-sc",
                                 localisation="SPADE",
                                 parallel=False)

root_print('\nRunning truncated basis decanol \n')
start = time.time()

# Now run the simulation!
Projection.run()

end = time.time()
truncated_walltime = end - start
truncated_energy = Projection.DFT_AinB_total_energy
root_print('Finished running truncated basis decanol  \n')

# Wonderful! Now let's run without truncation to compare timings. It is
# advised to check whether basis truncation introduces significant
# changes in total energy.
#
# calc_ll/calc_hl are reused as-is: ProjectionEmbedding only ever works
# on deepcopies of the calculators passed in (see EmbeddingBase/
# ProjectionEmbedding.__init__), so the originals - and the underlying
# PySCF Mole they share - are never mutated by the run above.

Projection = ProjectionEmbedding(decanol,
                                 embed_mask=sort_embed_mask,
                                 calc_base_ll=calc_ll,
                                 calc_base_hl=calc_hl,
                                 truncate_basis_thresh=0.0001,
                                 projection="huzinaga-sc",
                                 localisation="SPADE",
                                 parallel=False)

root_print('\nRunning full basis decanol \n')
start = time.time()

# Now run the simulation!
Projection.run()

end = time.time()
fullbasis_walltime = end - start
fullbasis_energy = Projection.DFT_AinB_total_energy
root_print('Finished running full basis decanol  \n')


root_print('---------------------------------')
root_print('--------  FINAL ENERGY  ---------')
root_print('---------------------------------')
root_print(f'Truncated basis calc time: {truncated_walltime} s')
root_print(f'Full basis calc time: {fullbasis_walltime} s')
root_print(f'Total time savings: {fullbasis_walltime - truncated_walltime} s')
root_print(f'Total energy (truncated basis): {truncated_energy} eV')
root_print(f'Total energy (full basis): {fullbasis_energy} eV')
root_print(f'Total energy difference: {fullbasis_energy - truncated_energy} eV')
root_print('---------------------------------')
