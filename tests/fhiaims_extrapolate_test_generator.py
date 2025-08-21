from ase.data.s22 import s22, s26, create_s22_system
from ase.calculators.aims import Aims, AimsProfile
from embasi.workflows.extrapolation import Extrapolation
import os


class FHIaims_projection_embedding_extrapolate_test():
    def __init__(self, localisation="SPADE", projection="level-shift", total_energy_corr="1storder", parallel=False, gc=False,
                 calc_ll_param_dict=None, calc_hl_param_dict=None, test_dir=None, basis_dir1="3", basis_dir2="2"):
        # Set-up calculator parameters (similar to FHIaims Calculator for
        # ASE) for low-level and high-level calculations. Below are the
        # absolute minimum parameters required for normal operation.

        if calc_hl_param_dict is None:
            calc_hl_param_dict = {}
        if calc_ll_param_dict is None:
            calc_ll_param_dict = {}

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
            self.calc_ll.parameters[key] = val

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
            self.calc_hl.parameters[key] = val

        self.localisation = localisation
        self.projection = projection
        self.total_energy_corr = total_energy_corr
        self.parallel = False
        self.gc = False
        self.test_dir = test_dir
        self.basis_dir1 = basis_dir1
        self.basis_dir2 = basis_dir2
        #os.environ["AIMS_SPECIES_DIR"] = os.environ["AIMS_ROOT_DIR"] + "/species_defaults/" + basis_dir
        self.energy = None
        self.energy2 = None
        self.diff_energy = None

    def run_test(self):

        # Creates a S22 system s26[22], with 22 being the ID from ASE
        methanol_dimer = create_s22_system(s26[22])
        energy = Extrapolation(
            file1=self.basis_dir1,
            file2=self.basis_dir2,
            path=os.environ["AIMS_ROOT_DIR"] + "/species_defaults/NAO-VCC-nZ/NAO-VCC-",
            asi_path=os.environ["AIMS_LIB_PATH"],
            atom=methanol_dimer,
            embed_mask=[2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 2, 1],
            calc_ll=self.calc_ll,
            calc_hl=self.calc_hl,
            projection1_param={
                "localisation": self.localisation,
                "projection": self.projection,
                "total_energy_corr": self.total_energy_corr,
                "parallel" : self.parallel,
                "gc" : self.gc,
            },
            projection2_param={
                "localisation": self.localisation,
                "projection": self.projection,
                "total_energy_corr": self.total_energy_corr,
                "parallel": self.parallel,
                "gc": self.gc,
            }
        )

        energy2 = Extrapolation(
            file1=self.basis_dir1,
            file2=self.basis_dir2,
            path=os.environ["AIMS_ROOT_DIR"] + "/species_defaults/NAO-VCC-nZ/NAO-VCC-",
            asi_path=os.environ["AIMS_LIB_PATH"],
            atom=methanol_dimer[:6],
            embed_mask=[2,1,2,2,2,1],
            calc_ll=self.calc_ll,
            calc_hl=self.calc_hl,
            projection1_param={
                "localisation": self.localisation,
                "projection": self.projection,
                "total_energy_corr": self.total_energy_corr,
                "parallel": self.parallel,
                "gc": self.gc,
            },
            projection2_param={
                "localisation": self.localisation,
                "projection": self.projection,
                "total_energy_corr": self.total_energy_corr,
                "parallel": self.parallel,
                "gc": self.gc,
            }
        )

        energy = energy.extrapolate  # Finds the monomer energy of the methanol
        self.energy = energy  # Stores the energy

        energy2 = energy2.extrapolate  # Finds the monomer energy of the methanol dimer
        self.energy2 = energy2

        self.diff_energy = energy - 2 * energy2 # Calculates the dissociation energy: diss_energy = energy2 - 2(energy1)

    def output_ref_values(self):
        return {
            "Methanol Dimer": self.energy2,
            "Methanol": self.energy,
            "Dissociation Energy": self.diff_energy,
        }




