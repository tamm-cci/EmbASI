import os
from asi_embedding.asi_embedding.embedding import ProjectionEmbedding
from ase.calculators.aims import Aims
from ase.build import molecule

os.environ['ASI_LIB_PATH'] = "/home/gabriellbramley/FHIaims_GAB/builds/mkl_scalapack_oneapi_embedding/libaims.240501.scalapack.mpi.so"
H2O_1 = molecule("H2O")
H2O_2 = molecule("SH2")
H2O_2.translate((0,0,2.5))
#H2O_1 += H2O_2

calc = Aims(xc='PBE',
    occupation_type="gaussian 0.01",
    mixer="pulay",
    n_max_pulay=10,
    charge_mix_param=0.5,
    sc_accuracy_rho=1E-05,
    sc_accuracy_eev=1E-03,
    sc_accuracy_etot=1E-06,
    override_error_charge_integration=True,
#    sc_accuracy_forces=1E-04,
    sc_iter_limit=100,
    collect_eigenvectors=True,
#    compute_forces=True,
    postprocess_anyway = True,
    density_update_method='density_matrix', # for DM export
  )

Projection = ProjectionEmbedding(H2O_1, embed_mask=3, calc_base=calc, scf_methods=["pbe","rpa"], mu_val=3.e+6)

Projection.AB_LL.run()
#Projection.A_LL.run(load_dm=Projection.AB_LL.atoms.calc.asi.dm_storage.get((1, 1), None))

import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 2)
ax1 = axs[0, 0]
ax2 = axs[0, 1]
ax3 = axs[1, 0]
ax4 = axs[1, 1]

ax1.matshow(Projection.AB_LL.atoms.calc.asi.dm_storage.get((1,1)))
#ax2.matshow(Projection.A_LL.atoms.calc.asi.dm_storage.get((1,1)))
plt.show()
