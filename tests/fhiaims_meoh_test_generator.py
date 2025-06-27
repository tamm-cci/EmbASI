from embasi.embedding import ProjectionEmbedding
from embasi.parallel_utils import root_print
from ase.data.s22 import s22, s26, create_s22_system
from ase.calculators.aims import Aims, AimsProfile
import numpy as np
import os

class FHIaims_projection_embedding_test():

    def __init__(self, localisation="SPADE", projection="huzinaga", parallel=False, gc=False,
                 calc_ll_param_dict={}, calc_hl_param_dict={}, test_dir=None, basis_dir="defaults_2020/light"):
        # Set-up calculator parameters (similar to FHIaims Calculator for
        # ASE) for low-level and high-level calculations. Below are the
        # absolute minimum parameters required for normal operation.

        self.calc_ll = Aims(xc='PBE', profile=AimsProfile(command="asi-doesnt-need-command"),
                       KS_method="parallel",
                       RI_method="LVL",
                       collect_eigenvectors=True,
                       density_update_method='density_matrix', # for DM export
                       atomic_solver_xc="PBE",
                       compute_kinetic=True,
                       override_initial_charge_check=True,
                       )
        
        for key, val in calc_ll_param_dict.items():
            self.calculator_ll.parameters[key]=val

        self.calc_hl = Aims(xc='PBE', profile=AimsProfile(command="asi-doesnt-need-command"),
                       KS_method="parallel",
                       RI_method="LVL",
                       collect_eigenvectors=True,
                       density_update_method='density_matrix', # for DM export
                       atomic_solver_xc="PBE",
                       compute_kinetic=True,
                       override_initial_charge_check=True,
                       )

        for key, val in calc_hl_param_dict.items():
            self.calculator_hl.parameters[key]=val

        self.localisation = localisation
        self.projection = projection
        self.parallel = False
        self.gc = False

        self.test_dir=test_dir

        os.environ["AIMS_SPECIES_DIR"] = os.environ["AIMS_ROOT_DIR"] + "/species_defaults/" + basis_dir

    def run_test(self):

        # Import dimer from s26 test set
        methanol_dimer_idx = s26[22]
        methanol_dimer = create_s22_system(methanol_dimer_idx)

        # Great! Now let's calculate the monomer total energy
        methanol = methanol_dimer[:6]

        Projection = ProjectionEmbedding(methanol,
                                 embed_mask=[2,1,2,2,2,1],
                                 calc_base_ll=self.calc_ll,
                                 calc_base_hl=self.calc_hl,
                                 mu_val=1.e+6,
                                 localisation=self.localisation,
                                 projection=self.projection,
                                 run_dir=os.path.join(self.test_dir, "MeOH_monomer"),
                                 #gc=self.gc
                                 #parallel=self.parallel
                                 )

        Projection.run()

        self.monomer_energy_total_energy = Projection.DFT_AinB_total_energy
        self.PB_corr = Projection.PB_corr

    def output_ref_values(self):

        return  {
                "MeOH_energy"       : self.monomer_energy_total_energy,
                "PB_corr_energy"    : self.PB_corr,
                }



