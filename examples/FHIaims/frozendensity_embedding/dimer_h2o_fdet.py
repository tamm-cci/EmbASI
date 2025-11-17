from embasi.embedding import FrozenDensityEmbedding
from embasi.parallel_utils import root_print
from ase.data.s22 import s22, s26, create_s22_system
from ase.calculators.aims import Aims, AimsProfile
import os

try:
    root_print(f"FHIaims library: {os.environ['ASI_LIB_PATH']}")
except:
    raise "Please set the environmental variable: ASI_LIB_PATH"

try:
    root_print(f"Basis directory: {os.environ['AIMS_SPECIES_DIR']}")
except:
    raise "Please set the environmental variable: AIMS_SPECIES_DIR"

calc = Aims(profile=AimsProfile(command="asi-doesnt-need-command"),
            override_warning_libxc=True,
            override_default_empty_basis_order=True,
            xc='libxc LDA_X+LDA_C_VWN', 
            relativistic="atomic_zora scalar",
            spin="none",
            xc_nakp="libxc  LDA_K_TF",
            xc_naxcp="libxc LDA_X+LDA_C_VWN",
            )
            

# Set-up directories
os.makedirs('H2O_dimer', exist_ok=True)

# Import dimer from s26 test set
os.chdir('H2O_dimer')
water_dimer_idx = s26[1]

water_dimer = create_s22_system(water_dimer_idx)

water_monomer1 = water_dimer[:3]
water_monomer2 = water_dimer[3:]

Projection = FrozenDensityEmbedding(water_dimer,
                                    embed_mask=[1,1,1,2,2,2],
                                    calc_base_ll=calc,
                                    calc_base_hl=calc)

# Now run the simulation!
root_print('\nRunning ... \n')
Projection.run()
