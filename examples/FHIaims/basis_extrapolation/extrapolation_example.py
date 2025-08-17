import os,sys
from embasi.workflows.extrapolation import Extrapolation

from ase.data.s22 import s26,create_s22_system
from ase.calculators.aims import Aims, AimsProfile
'''
A example of running an FHIaims QM/MM embedding simulation of the extrapolation
of a methanol and a methanol dimer to calculate the end dissociation energy.
'''


# os.environ["ASI_LIB_PATH"] = "/home/dchen/Software/FHIaims/_build_lib/libaims.250711.scalapack.mpi.so"
# os.environ['AIMS_SPECIES_DIR'] = "/home/dchen/Software/FHIaims/species_defaults/NAO-VCC-nZ/NAO-VCC-
# The 'AIMS_SPECIES_DIR' and 'ASI_LIB_PATH' are passed into the class to allow for easier retrieval.

calc_ll = Aims(
    xc='PBE',
    profile=AimsProfile(command="asi-doesnt-need-command"),
    KS_method="parallel",
    RI_method="LVL",
    collect_eigenvectors=True,
    density_update_method='density_matrix', # for DM export
    atomic_solver_xc="PBE",
    compute_kinetic=True,
    override_initial_charge_check=True,
    override_illconditioning=True
)

calc_hl = Aims(
    xc='PBE',
    profile=AimsProfile(command="asi-doesnt-need-command"),
    KS_method="parallel",
    RI_method="LVL",
    collect_eigenvectors=True,
    density_update_method='density_matrix',  # for DM export
    atomic_solver_xc="PBE",
    compute_kinetic=True,
    override_initial_charge_check=True,
    override_illconditioning=True
)

# Import dimer from s26 test set
methanol_dimer_idx = s26[22]
methanol_dimer = create_s22_system(methanol_dimer_idx)

# Set-up calculator parameters (similar to FHIaims Calculator for
# ASE) for low-level and high-level calculations. Below are the
# absolute minimum parameters required for normal operation.

test = Extrapolation(
    file1= "3",
    file2 = "2",
    path = "/home/dchen/Software/FHIaims/species_defaults/NAO-VCC-nZ/NAO-VCC-",
    asi_path = "/home/dchen/Software/FHIaims/_build_lib/libaims.250711.scalapack.mpi.so",
    atom = methanol_dimer,
    embed_mask = [2,1,2,2,2,1,2,1,2,2,2,1],
    calc_ll = calc_ll,
    calc_hl = calc_hl,
)

test2 = Extrapolation(
    file1= "3",
    file2 = "2",
    path = "/home/dchen/Software/FHIaims/species_defaults/NAO-VCC-nZ/NAO-VCC-",
    asi_path = "/home/dchen/Software/FHIaims/_build_lib/libaims.250711.scalapack.mpi.so",
    atom = methanol_dimer[:6],
    embed_mask = [2,1,2,2,2,1],
    calc_ll = calc_ll,
    calc_hl = calc_hl
)

energy1 = test.extrapolate  # Returns the extrapolated energy for methanol
energy2 = test2.extrapolate  # Returns the extrapolated energy for the methanol dimer

# Calculate dimer dissociation energy in the usual way
print(energy1 - 2 * energy2)
