'''
A basic working example for running an EmbASI/GULP QM/QM/MM embedding single point
calculation for a silicate cluster. The high level QM calculator uses the PBE0
XC functional, and the low level QM calculator uses the PBE XC functional.

To run this script, you will require the silicate.pun and silicate.ff input file.
Additionally, you will need to modify the following arguments within the EmbASI object:
- asi_lib_path = location of the FHIaims shared object library
- aims_species_dir = location of desired basis set

This script should calclulate the single point energy of the silicate cluster.
'''

from chemsh import *

# As customary with ChemShell calculations, the fragment object is set up using the silicate input file.
silicate = Fragment(coords='silicate.pun', connect_mode='covalent')
silicate.save('silicate.xyz', 'xyz')
# Manual setting of qm_region, this is optional but allows for easy access of the variable.
qm_region = silicate.getRegion(1)

# Next, the EmbASI QM calculator can be constructed, which intakes the following variables:
# - qm_region: Defines the region on which the QM calculation is run. This can be defined either through
# the ChemShell getRegion(1) argument, or a list[int] of the QM indices.
# - embed_mask: A list[int] accounting for region assignment for the low level QM (2), and high level
# QM (1). In this example, 1 refers to PBE0, and 2 refers to PBE.
# - mu_val: Default to 1.e+6.
# - ll_calc_config: Allows customisation of the default calculator configurations, most importantly,
# for setting of the XC functional. This intakes a dictionary entry to update the internal settings.
# We will set the low level XC as PBE. Optional argument.
# - hl_calc_config: See above. We will set the high level XC as PBE0. Optional argument.
# - asi_lib_path: Customisation of environmental variable, should be set to location of the FHIaims
# shared object library.
# - aims_species_dir: Customisation of environmental variable, should be set to the location of your
# desired basis set.
# - charge: Allows manual customisation of the QM region charge, should the user require.
# Optional argument.
# - link_hl: Allows manual region assignment of any QM link atoms. Default to QM region 2, but this
# can be set to False in order to set as QM region 1. Optional argument.
qm_theory = EmbASI(qm_region=qm_region, embed_mask=[2,1,1,1,1], mu_val=1.0e+6,
            ll_calc_config={'xc':'PBE', 'load_balancing': False, 'use_local_index':False},
	    hl_calc_config={'xc':'PBE', 'load_balancing': False, 'use_local_index':False},
	    asi_lib_path="/home/eva/Software/FHI-aims_new/build_lib_debug/libaims.250806.scalapack.mpi.so",
            aims_species_dir="/home/eva/Software/FHI-aims_new/species_defaults/defaults_2020/light/", links_hl=False)

# Next, setting up of the GULP MM calculator.
ff = 'silicate.ff'
mm_theory = GULP(ff=ff, conjugate=True)

# Bond modifiers defined, accounting for dipole-adjustments.
silicate_modifiers = {('Si3', 'O1'):0.3, ('Si1', 'O3'):0.3}

# QMMM object is set up, combining the QM and MM objects.
qmmm = QMMM(qm=qm_theory, mm=mm_theory, qm_region=qm_region, frag=silicate, coupling='covalent',
       bond_modifiers=silicate_modifiers, embedding='electrostatic')

# Finally, single point calculation can be run as follows.
sp = SP(theory=qmmm, gradients=False)
sp.run()

