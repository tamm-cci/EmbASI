# ~ Overall Embedding object
from abc import ABC, abstractmethod
from ASI_embedding.parallel_utils import root_print
import numpy as np

class EmbeddingBase(ABC): 

    def __init__(self, atoms, embed_mask, calc_base_ll=None, calc_base_hl=None):
        import os

        self.asi_lib_path = os.environ['ASI_LIB_PATH']
        self.embed_mask = embed_mask
        self.calculator_ll = calc_base_ll
        self.calculator_hl = calc_base_hl

    @property
    def scf_methods(self):
        return self._scf_methods

    @scf_methods.setter
    def scf_methods(self, val):

        if isinstance(self.embed_mask, int):
            assert len(val) == 2, \
                "Invalid number of methods for given n_layers."
        elif isinstance(self.embed_mask, list):
            assert len(val) == len(set(self.embed_mask)), \
                "Invalid number of methods for given n_layers."
            
        self._scf_methods = []
        
        for scf in val:
            self._scf_methods.append(scf)

    def set_layer(self, atoms, layer_name, calc, embed_mask, ghosts=0, no_scf=False):
        "Initialises the AtomsEmbed methods for a given method"
        from .atoms_embedding_asi import AtomsEmbed

        layer = AtomsEmbed(atoms, calc, embed_mask, outdir=layer_name, ghosts=ghosts, no_scf=no_scf)
        setattr(self, layer_name, layer)

    @abstractmethod
    def run(self):
        pass

class ProjectionEmbedding(EmbeddingBase):
    def __init__(self, atoms, embed_mask, calc_base_ll, calc_base_hl, frag_charge=0, post_scf=None, mu_val=1e+06, truncate_basis=False):
        """_summary_
        Intended to 


        Args:
            atoms (ASE Atoms Object): Input ASE Atoms object used to pass structural information
            embed_mask (int OR list): _description_
            calc_base_ll (ASE Calculator): _description_
            calc_base_hl (ASE Calculator): _description_
            frag_charge (int, optional): _description_. Defaults to 0.
            post_scf (_type_, optional): _description_. Defaults to None.
            mu_val (_type_, optional): _description_. Defaults to 1e+06 Ha.
            truncate_basis (bool, optional): _description_. Defaults to 1e+06 Ha.
        """
        
        
        from copy import copy, deepcopy
        from mpi4py import MPI

        # Low-level, lref, lowref, highref, 
        # lo-lev hi-lev, system_AB_
        # 

        self.calc_names = ["AB_LL","A_LL","A_HL","A_HL_PP","AB_LL_PP"]

        super(ProjectionEmbedding, self).__init__(atoms, embed_mask, calc_base_ll, calc_base_hl)
        low_level_calculator_1 = deepcopy(self.calculator_ll)
        low_level_calculator_2 = deepcopy(self.calculator_ll)
        low_level_calculator_3 = deepcopy(self.calculator_ll)

        high_level_calculator_1 = deepcopy(self.calculator_hl)
        high_level_calculator_2 = deepcopy(self.calculator_hl)

        low_level_calculator_1.set(qm_embedding_calc = 1)
        self.set_layer(atoms, self.calc_names[0], low_level_calculator_1, embed_mask, ghosts=0, no_scf=False)
        low_level_calculator_2.set(qm_embedding_calc = 2)
        low_level_calculator_2.set(charge_mix_param = 0.)
        low_level_calculator_2.set(charge = frag_charge)
        self.set_layer(atoms, self.calc_names[1], low_level_calculator_2, embed_mask, ghosts=2, no_scf=False)
        low_level_calculator_3.set(qm_embedding_calc = 2)
        low_level_calculator_3.set(charge_mix_param = 0.)
        self.set_layer(atoms, self.calc_names[4], low_level_calculator_3, embed_mask, ghosts=0, no_scf=False)
        high_level_calculator_1.set(qm_embedding_calc = 3)
        high_level_calculator_1.set(charge = frag_charge)
        self.set_layer(atoms, self.calc_names[2], high_level_calculator_1, embed_mask, ghosts=2, no_scf=False)
        high_level_calculator_2.set(qm_embedding_calc = 2)
        high_level_calculator_2.set(charge_mix_param = 0.)
        high_level_calculator_2.set(charge = frag_charge)
        self.set_layer(atoms, self.calc_names[3], high_level_calculator_2, embed_mask, ghosts=2, no_scf=False)

        self.mu_val = mu_val
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.ntasks = MPI.COMM_WORLD.Get_size()
        self.truncate_basis = truncate_basis

    @property
    def nlayers(self):
        return self._nlayers

    @nlayers.setter
    def nlayers(self, val):
        
        assert val == 2, \
                "Only two layers currently valid for projection embedding."
        return self._nlayers

    def calculate_levelshift_projector(self):

        self.P_b = self.mu_val * (self.AB_LL.overlap @ self.AB_LL.density_matrices_out[1] @ self.AB_LL.overlap)

    def calculate_huzinaga_projector(self):

        self.P_b = self.AB_S @ self.B_dm @ self.AB_Htot.T
        self.P_b = -0.5*(self.P_b + (self.AB_Htot.T @ self.B_dm @ self.AB_S.T))

    def select_atoms_basis_truncation(self, thresh):
        """_summary_
        Returns a list of corresponding atoms for which the total contribution of an atoms constituent basis functions to the total charge of subsystem A, q^{A}:
               q^{A}_{/mu, /nu} = /gamma^{A}_{/mu, /nu} S_{/mu, /nu}
        exceeds the threshold, thresh:
                      thresh < q_{/mu, /nu} 
        This is unfortunately just Mulliken analysis again.
        Args:
            thresh (float): _description_
        Returns: 
            active_mask (list): True/False mask for 
        """

        basis_charge = np.diag(self.AB_LL.density_matrices_out[0] @ self.AB_LL.overlap)
        atomic_charge = np.zeros(len(self.AB_LL.atoms))

        for idx, charge in enumerate(basis_charge):
            atomic_charge[self.AB_LL.basis_atoms[idx]] += abs(charge)

        truncated_atom_list = [ charge > thresh for charge in atomic_charge ]

        return truncated_atom_list

    def set_truncation_defaults(self, truncated_atom_list):
        """_summary_
        Necessary to maintain consistency in values needed for matrix truncation
        /expansion between the AtomsEmbed objects (eg., self.basis_atoms, 
        self.n_atoms). Failure to do so leads to unexpected behaviour.
        which 
        Args:
            truncated_atom_list (_type_): _description_
        """
        from ASI_embedding.basis_info import Basis_info

        # Establish mapping corresponding to each new atom from the 
        # truncated matrix
        active_atoms = np.array([ idx for idx, maskval in enumerate(truncated_atom_list) if maskval ])

        # Remove non-active atoms from basis_atoms to form a truncated
        # analogue
        trunc_basis_atoms = np.array([atom for atom in self.AB_LL.basis_atoms if atom in active_atoms])

        # Count number of basis functions included in truncation calculations
        new_nbasis = 0
        for atom in self.AB_LL.basis_atoms:
            if truncated_atom_list[atom]:
                new_nbasis +=1

        BasisInfo = Basis_info()
        BasisInfo.full_natoms = len(truncated_atom_list)
        BasisInfo.trunc_natoms = len(active_atoms)
        BasisInfo.active_atoms = active_atoms
        BasisInfo.full_basis_atoms = self.AB_LL.basis_atoms
        BasisInfo.trunc_basis_atoms = trunc_basis_atoms
        BasisInfo.full_nbasis = self.AB_LL.n_basis
        BasisInfo.trunc_nbasis = new_nbasis
        BasisInfo.set_basis_atom_indexes()

        self.AB_LL.truncate = False
        self.AB_LL.basis_info = BasisInfo

        self.A_LL.truncate = True
        self.A_LL.basis_info = BasisInfo

        self.A_HL.truncate = True
        self.A_HL.basis_info = BasisInfo

        self.A_HL_PP.truncate = True
        self.A_HL_PP.basis_info = BasisInfo

        self.AB_LL_PP.truncate = False
        self.AB_LL_PP.basis_info = BasisInfo

    def calc_subsys_pop(self, overlap_matrix, density_matrix):
        """ Summary
            Calculates the overall electron population of a given subsystem
            through the following relation:
                    P_{pop} = tr[S^{AB}/gamma]
            where S^{AB} is the overlap matrix for the supermolecular system
            and /gamma is the density matrix of a given subsystem.

        Args:
            overlap_matrix (np.ndarray): Supersystem overlap matrix in AO basis.
            density_matrix (np.ndarray): Subsystem density matrix in AO basis,
        Returns:
            population (int): Overall electronic population of subsystem
        """

        population = np.trace(overlap_matrix @ (density_matrix))

        return population

    def run(self):
        """ Summary
        The primary driver routine for performing QM-in-QM with a Projection-based embedding scheme. This scheme draws upon the work of Manby et al. [1, 2]. 

        The workflow within the current implementation operates as follows:
        1) Calculate the KS-DFT energy of the combined subsystems A+B. 
        ...

        (1) Manby, F. R.; Stella, M.; Goodpaster, J. D.; Miller, T. F. I. A Simple, Exact Density-Functional-Theory Embedding Scheme. J. Chem. Theory Comput. 2012, 8 (8), 2564–2568.
        (2) Lee, S. J. R.; Welborn, M.; Manby, F. R.; Miller, T. F. Projection-Based Wavefunction-in-DFT Embedding. Acc. Chem. Res. 2019, 52 (5), 1359–1368.
        """
        import numpy as np

        root_print("Embedding calculation begun...")

        ''' 
        Performs a single-point energy evaluation for a system composed of A
        and B. Returns localised density matrices for subsystems A and B, and the two-electron components of the hamiltonian (combined with nuclear-electron potential).
        '''
        self.AB_LL.run()

        ''' 
        Initialises the density matrix for subsystem A, and calculated the hamiltonian components for subsystem A at the low-level reference.
        '''
        if self.truncate_basis:
            basis_mask = self.select_atoms_basis_truncation(0.5)
            self.set_truncation_defaults(basis_mask)

        self.A_LL.density_matrix_in = self.AB_LL.density_matrices_out[0]
        self.A_LL.run()

        ''' 
        Calculates the electron count for the combined (A+B) and separated subsystems (A and B). 
        '''
        self.AB_pop = self.calc_subsys_pop(self.AB_LL.overlap, 
                            (self.AB_LL.density_matrices_out[0]
                            +self.AB_LL.density_matrices_out[1]))
        
        self.A_pop = self.calc_subsys_pop(self.AB_LL.overlap, 
                            self.A_LL.density_matrices_out[0])
        
        self.B_pop = self.calc_subsys_pop(self.AB_LL.overlap, 
                            self.AB_LL.density_matrices_out[1])
        
        ''' 
        Initialises the density matrix for subsystem A, and calculated the hamiltonian components for subsystem A at the low-level reference.
        '''
        self.calculate_levelshift_projector()

        # Calculate density matrix for A high-level with embedded Fock
        # matrix. 
        self.A_HL.density_matrix_in = \
                self.AB_LL.density_matrices_out[0]
        self.A_HL.fock_embedding_matrix = \
                self.AB_LL.hamiltonian_electrostatic - \
                    self.A_LL.hamiltonian_electrostatic + self.P_b
        self.A_HL.run()

        # Calculate high-level post-processed reference energy
        self.A_HL_PP.density_matrix_in = self.A_HL.density_matrices_out[0]
        self.A_HL_PP.run(ev_corr_scf=True)
        subsys_A_highlvl_totalen = self.A_HL_PP.ev_corr_total_energy

        # Re-normalising charge for differing atomic solvers (bad cludge)
        root_print(f" Normalizing density matrix from high-level reference...")
        self.A_HL_pop = self.calc_subsys_pop(self.AB_LL.overlap, self.A_HL.density_matrices_out[0])
        root_print(f" Population of Subystem A^[HL]: {self.A_HL_pop}")
        self.charge_renorm = (self.A_pop/self.A_HL_pop)
        root_print(f" Population of Subystem A^[HL] (post-norm): {self.calc_subsys_pop(self.AB_LL.overlap, self.charge_renorm*self.A_HL.
        density_matrices_out[0])}")

        # Calculate A low-level reference energy
        self.A_LL.density_matrix_in = self.charge_renorm * self.A_HL.density_matrices_out[0]
        self.A_LL.run(ev_corr_scf=True)
        subsys_A_lowlvl_totalen = self.A_LL.ev_corr_total_energy

        # Calculate AB low-level reference energy
        self.AB_LL_PP.density_matrix_in = (self.charge_renorm * self.A_HL.density_matrices_out[0]) + self.AB_LL.density_matrices_out[1]

        self.AB_LL_PP.run(ev_corr_scf=True)
        subsys_AB_lowlvl_totalen = self.AB_LL_PP.ev_corr_total_energy

        # Calculate projected density correction to total energy
        self.PB_corr = (np.trace(self.P_b @ self.A_HL.density_matrices_out[0]) * 27.211384500)

        self.DFT_AinB_total_energy = subsys_A_highlvl_totalen - \
            subsys_A_lowlvl_totalen + subsys_AB_lowlvl_totalen + self.PB_corr
        root_print( f" ----------- FINAL         OUTPUTS --------- " )
        root_print(f" ")
        root_print(f" Population Information:")
        root_print(f" Population of Subsystem AB: {self.AB_pop}")
        root_print(f" Population of Subsystem A: {self.A_pop}")
        root_print(f" Population of Subsystem B: {self.B_pop}")
        root_print(f" ")
        root_print(f" Intermediate Information:")
        root_print(f" WARNING: These are not faithful, ground-state KS total energies - ")
        root_print(f" In the case of low-level references, they are calculated using the ")
        root_print(f" density components of the high-level energy reference for fragment A. ")
        root_print(f" Do not naively use these energies unless you are comfortable with ")
        root_print(f" their true definition. ")
        root_print(f" Total Energy (A+B Low-Level): {subsys_AB_lowlvl_totalen} eV" )
        root_print(f" Total Energy (A Low-Level): {subsys_A_lowlvl_totalen} eV" )
        root_print(f" Total Energy (A High-Level): {subsys_A_highlvl_totalen} eV" )
        root_print(f" Projection operator energy correction DM^(A_HL) @ Pb: {self.PB_corr} eV" )
        root_print(f"  " )
        root_print(f" Final Energies Information:")
        root_print(f" Final total energy (Uncorrected): {self.DFT_AinB_total_energy - self.PB_corr} eV" )
        root_print(f" Final total energy (Projection Corrected): {self.DFT_AinB_total_energy} eV" )
        root_print(f" " )
        root_print(f" -----------======================--------- " )
        root_print(f" " )



