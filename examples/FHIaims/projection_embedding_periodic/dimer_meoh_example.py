from embasi.embedding import ProjectionEmbedding
from embasi.parallel_utils import root_print
from ase.data.s22 import s22, s26, create_s22_system
from ase.calculators.aims import Aims, AimsProfile
import os

'''
A minimal working example for running an FHIaims  QM/QM embedding simulation
of a methanol dimer. The interacting OH fragments calculated with
the PBE0 XC functional, and the CH3 fragment with the PBE XC functional.

To run this example you will need to:
- Set environmental variable 'ASI_LIB_PATH' to the location of
  the FHIaims shared object library
- Set environmental variable 'AIM_SPECIES_DIR' to the location of
  your desired basis set

The script should calculate the dimer dissociation energy of MeOH.
'''

# One may also use an environmental variable to achieve this
#os.environ['ASI_LIB_PATH'] = "/home/gabrielbramley/Software/FHIaims/_build_embasi/libaims.250320.scalapack.mpi.so"
#os.environ['AIMS_SPECIES_DIR'] = "/home/gabrielbramley/Software/FHIaims/species_defaults/defaults_2020/light/"

# root_print ensures only head node prints
try:
    root_print(f"FHIaims library: {os.environ['ASI_LIB_PATH']}")
except:
    raise "Please set the environmental variable: ASI_LIB_PATH"

try:
    root_print(f"Basis directory: {os.environ['AIMS_SPECIES_DIR']}")
except:
    raise "Please set the environmental variable: AIMS_SPECIES_DIR"


# Set-up calculator parameters (similar to FHIaims Calculator for
# ASE) for low-level and high-level calculations. Below are the
# absolute minimum parameters required for normal operation.

calc_ll = Aims(xc='PBE', profile=AimsProfile(command="asi-doesnt-need-command"),
    KS_method="parallel",
    RI_method="LVL",
    collect_eigenvectors=True,
    density_update_method='density_matrix', # for DM export
    atomic_solver_xc="PBE",
#    lmo_init_guess="random",
    compute_kinetic=True,
    override_initial_charge_check=True,
    k_grid=(1,1,1)
  )

calc_hl = Aims(xc='PBE', profile=AimsProfile(command="asi-doesnt-need-command"),
    KS_method="parallel",
    RI_method="LVL",
    collect_eigenvectors=True,
    density_update_method='density_matrix', # for DM export
    atomic_solver_xc="PBE",
#    lmo_init_guess="random",
    compute_kinetic=True,
    override_initial_charge_check=True,
    k_grid=(1,1,1)
  )

# Set-up directories
from ase.build import nanotube

cnt1 = nanotube(3, 3, vacuum=10, length=2)
embed_mask = len(cnt1) * [1]

# Set up ProjectionEmbedding, with:
# - Embedding mask (1=Highlevel (PBE0), 2=Low-level (PBE))
# - Assigned higher and lower level calculators
# - Fragment charge (Usually -1 per split covalent bond)
# - TODO: ADD AUTOMATIC DETECTION OF FRAGMENT CHARGE
Projection = ProjectionEmbedding(cnt1,
                                 embed_mask=embed_mask,
                                 calc_base_ll=calc_ll,
                                 calc_base_hl=calc_hl,
                                 mu_val=1.e+6)

# Now run the simulation!
root_print('\nRunning MeOH dimer \n')
Projection.run()
root_print('Finished running MeOH dimer \n')

# Total energy for the embedded fragment may be accessed:
meoh_dimer_pbe0inpbe_energy = Projection.DFT_AinB_total_energy
