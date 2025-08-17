from ase.data.s22 import s22, s26, create_s22_system
import os,sys
from ase.calculators.aims import Aims, AimsProfile
from embasi.workflows.extrapolation import Extrapolation


class FHIaims_projection_embedding_extrpolate_test():
    def __init__(self, localisation="SPADE", projection="huzinaga", parallel=False, gc=False,
                 calc_ll_param_dict={}, calc_hl_param_dict={}, test_dir=None, basis_dir="defaults_2020/light"):
        # Set-up calculator parameters (similar to FHIaims Calculator for
        # ASE) for low-level and high-level calculations. Below are the
        # absolute minimum parameters required for normal operation.

        self.calc_ll = Aims(
            xc='PBE',
            profile=AimsProfile(command="asi-doesnt-need-command"),
            KS_method="parallel",
            RI_method="LVL",
            collect_eigenvectors=True,
            density_update_method='density_matrix',  # for DM export
            atomic_solver_xc="PBE",
            compute_kinetic=True,
            override_initial_charge_check=True,
        )

        for key, val in calc_ll_param_dict.items():
            self.calculator_ll.parameters[key] = val

        self.calc_hl = Aims(
            xc='PBE',
            profile=AimsProfile(command="asi-doesnt-need-command"),
            KS_method="parallel",
            RI_method="LVL",
            collect_eigenvectors=True,
            density_update_method='density_matrix',  # for DM export
            atomic_solver_xc="PBE",
            compute_kinetic=True,
            override_initial_charge_check=True,
        )
        for key, val in calc_hl_param_dict.items():
            self.calculator_hl.parameters[key] = val
        self.localisation = localisation
        self.projection = projection
        self.parallel = False
        self.gc = False
        self.test_dir = test_dir
        #os.environ["AIMS_SPECIES_DIR"] = os.environ["AIMS_ROOT_DIR"] + "/species_defaults/" + basis_dir
        self.energy = None
        self.energy2 = None
        self.diff_energy = None

    def run_test(self):

        # Creates a S22 system s26[22], with 22 being the ID from ASE
        methanol_dimer = create_s22_system(s26[22])
        energy = Extrapolation(
            file1="3",
            file2="2",
            path="/home/dchen/Software/FHIaims/species_defaults/NAO-VCC-nZ/NAO-VCC-",
            asi_path="/home/dchen/Software/FHIaims/_build_lib/libaims.250711.scalapack.mpi.so",
            atom= methanol_dimer,
            embed_mask=[2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 2, 1],
            calc_ll=self.calc_ll,
            calc_hl=self.calc_hl,
        )

        energy2 = Extrapolation(
            file1="3",
            file2="2",
            path="/home/dchen/Software/FHIaims/species_defaults/NAO-VCC-nZ/NAO-VCC-",
            asi_path="/home/dchen/Software/FHIaims/_build_lib/libaims.250711.scalapack.mpi.so",
            atom=methanol_dimer[:6],
            embed_mask=[2,1,2,2,2,1],
            calc_ll=self.calc_ll,
            calc_hl=self.calc_hl,
        )

        energy = energy.extrapolate  # Finds the monomer energy of the methanol
        self.energy = energy  # Stores the energy

        energy2 = energy2.extrapolate  # FInds the monomer energy of the methanol dimer
        self.energy2 = energy2

        self.diff_energy = energy - 2 * energy2 # Calculates the dissociation energy: diss_energy = energy2 - 2(energy1)

    def output_ref_values(self):
        return {
            "Methanol Dimer": self.energy2,
            "Methanol": self.energy,
            "Dissociation Energy": self.diff_energy,
        }




