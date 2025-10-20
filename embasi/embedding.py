# ~ Overall Embedding object
from abc import ABC, abstractmethod
from embasi.parallel_utils import root_print
import scalapack4py.npscal.math_utils.operations as op
import copy
import time
import numpy as np

class EmbeddingBase(ABC):
    """Base object on which all embedding methods are based

    In general, QM/QM embedding methods should follow similar principles i.e.,
    the system of interest is partitioned into separate 'layers'. Usually, one
    layer is of chemical/physical interest and will be evaluated at a higher
    level of theory, while the other layers considered part of the embedding 
    environment and modelled at a lower level of theory. For now, we assumed
    only two embedding layers, described by embed_mask, and ASE calculators 
    to be assigned to each layer.

    Parameters
    ----------
    atoms: ASE Atoms
        Structural and chemical information of system
    embed_mask: int or list
        Number of atoms from index 0 in layer 1, or list of reigons assigned 
        to each atom (i.e., [1,1,1,2,2,2])
    calc_base_ll: ASE FileIOCalculator
        Calculator object for layer 1
    calc_base_hl: ASE FileIOCalculator
        Calculator object for layer 2

    """
    def __init__(self, atoms, embed_mask, calc_base_ll=None, calc_base_hl=None, run_dir="./EmbASI_calc"):
        import os

        self.asi_lib_path = os.environ['ASI_LIB_PATH']
        self.embed_mask = embed_mask
        self.calculator_ll = calc_base_ll
        self.calculator_hl = calc_base_hl
        self.run_dir=run_dir

        try:
            os.makedirs(self.run_dir, exist_ok=True)
            root_print(f"Running EmbASI calculation in directory: {self.run_dir}")
        except OSError as error:
            root_print(f"Directory {self.run_dir} can not be created")

    def set_layer(self, atoms, layer_name, calc, embed_mask, ghosts=0, 
                      no_scf=False, ctxt_tag=None, descr_tag=None):
        """Sets an AtomsEmbed object as an attribute

        Creates an AtomsEmbed object as a named attribute (layer_name) of 
        EmbeddingBase method. 

        Parameters
        ----------
        atoms: ASE Atoms
            Structural and chemical information of system
        layer_name: str
            Name assigned to attribute
        calc: ASE FileIOCalculator
            Calculator object assigned to AtomsEmbed
        embed_mask: int or list
            Number of atoms from index 0 in layer 1, or list of regions 
            assigned to each atom (i.e., [1,1,1,2,2,2])
        ghosts: int
            Layer to be considered as ghost atoms. If 0, no ghosts assigned.
        no_scf: bool
            Terminate SCF cycle at the first step

        """
        from .atoms_embedding_asi import AtomsEmbed
        import os

        outdir_name = os.path.join(self.run_dir, layer_name)

        layer = AtomsEmbed(atoms, calc, embed_mask, outdir=outdir_name, 
                           ghosts=ghosts, no_scf=no_scf, descr_tag=descr_tag,
                           ctxt_tag=ctxt_tag)
        setattr(self, layer_name, layer)

    def select_atoms_basis_truncation(self, atomsembed, densmat, overlap, thresh):
        """Lists atomic centers significant charge for a given density matrix

        Returns a list of corresponding atoms for which the total contribution 
        of an atoms constituent basis functions to the total charge of subsystem 
        A, q^{A}:

               q^{A}_{/mu, /nu} = /gamma^{A}_{/mu, /nu} S_{/mu, /nu}

        exceeds the threshold, thresh:

                      thresh < q_{/mu, /nu} 

        This is unfortunately just Mulliken analysis. Future work could use
        more sophisticated charge techniques, but this would require
        volumetric charge data.
        
        Parameters
        ----------
        atomsembed: AtomsEmbed
            AtomsEmbed object for which a QM run has been executed.
        densmat: np.ndarray
            Corresponding density matrix to AtomsEmbed
        thresh: float
            Mulliken charge threshold for 
        Returns
        -------
        active_atom_mask: bool list
            Atoms considered active (True), or truncated (False).
        
        """

        # @TODONPSCAL: REPLACE DIRECTIVE
        basis_charge = op.diag(densmat @ overlap)
        atomic_charge = np.zeros(len(atomsembed.atoms))

        for idx, charge in enumerate(basis_charge):
            atomic_charge[atomsembed.basis_atoms[idx]] += abs(charge)

        active_atom_mask = [ charge > thresh for charge in atomic_charge ]

        return active_atom_mask

    def set_basis_info(self, atomsembed):
        """Sets default BasisInfo objects for each embedding layer

        Necessary to maintain consistency in values needed for matrix truncation
        /expansion between the AtomsEmbed objects (eg., self.basis_atoms, 
        self.n_atoms). Failure to do so leads to unexpected behaviour.

        Parameters
        ----------
        atomsembed: AtomsEmbed Object
            Atoms embed object for which defaults are set
        active_atom_mask: list
            Atoms considered active (True), or truncated (False).

        Return
        ------
        BasisInfo: BasisInfo object
            Basis info object to be assigned to each layer

        """
        from embasi.basis_info import Basis_info

        active_atoms = np.array([True]*len(atomsembed.atoms))

        BasisInfo = Basis_info()
        BasisInfo.full_natoms = np.size(active_atoms)
        BasisInfo.trunc_natoms = np.size(active_atoms)
        BasisInfo.active_atoms = active_atoms
        BasisInfo.full_basis_atoms = atomsembed.basis_atoms
        BasisInfo.trunc_basis_atoms = atomsembed.basis_atoms
        BasisInfo.full_nbasis = atomsembed.n_basis
        BasisInfo.trunc_nbasis = atomsembed.n_basis

        return BasisInfo

    def set_truncation_defaults(self, atomsembed, active_atom_mask):
        """Sets default BasisInfo objects for each embedding layer

        Necessary to maintain consistency in values needed for matrix truncation
        /expansion between the AtomsEmbed objects (eg., self.basis_atoms, 
        self.n_atoms). Failure to do so leads to unexpected behaviour.

        Parameters
        ----------
        atomsembed: AtomsEmbed Object
            Atoms embed object for which defaults are set
        active_atom_mask: list
            Atoms considered active (True), or truncated (False).

        Return
        ------
        BasisInfo: BasisInfo object
            Basis info object to be assigned to each layer

        """
        from embasi.basis_info import Basis_info

        # Establish mapping corresponding to each new atom from the 
        # truncated matrix
        active_atoms = np.array([ idx for idx, maskval 
                                  in enumerate(active_atom_mask) 
                                  if (maskval or atomsembed.embed_mask[idx] == 1) ])

        # Remove non-active atoms from basis_atoms to form a truncated
        # analogue
        trunc_basis_atoms = np.array([atom for atom in atomsembed.basis_atoms 
                                      if atom in active_atoms])

        # Count number of basis functions included in truncation calculations
        new_nbasis = 0
        for atom in atomsembed.basis_atoms:
            if active_atom_mask[atom]:
                new_nbasis +=1
        
        root_print(f" " )
        root_print( f" ----------- Performing Truncation --------- " )
        root_print(f" ")
        root_print(f" Number of atoms before truncation: {len(active_atom_mask)}")
        root_print(f" Number of atoms after truncation: {len(active_atoms)}")
        root_print(f" ")
        root_print(f" Number of basis functions before truncation: {len(atomsembed.basis_atoms)}")
        root_print(f" Number of basis functions after truncation: {new_nbasis}")
        root_print(f" " )
        root_print( f" ------------------------------------------- " )
        root_print(f" " )

        BasisInfo = Basis_info()
        BasisInfo.full_natoms = len(active_atom_mask)
        BasisInfo.trunc_natoms = len(active_atoms)
        BasisInfo.active_atoms = active_atoms
        BasisInfo.full_basis_atoms = atomsembed.basis_atoms
        BasisInfo.trunc_basis_atoms = trunc_basis_atoms
        BasisInfo.full_nbasis = atomsembed.n_basis
        BasisInfo.trunc_nbasis = new_nbasis
        BasisInfo.set_basis_atom_indexes()

        return BasisInfo

    def calc_subsys_pop(self, overlap_matrix, density_matrix):
        """Calculated number of electrons in a subsystem
        
        Calculates the overall electron population of a given subsystem
        through the following relation:
                        P_{pop} = tr[S^{AB}/gamma]
        where S^{AB} is the overlap matrix for the supermolecular system
        and /gamma is the density matrix of a given subsystem.

        Parameters
        ----------
        overlap_matrix: np.ndarray 
            Supersystem overlap matrix in AO basis.
        density_matrix: np.ndarray 
            Subsystem density matrix in AO basis,

        Returns
        -------
        population: float 
            Overall electronic population of subsystem

        """

        # @TODONPSCAL: REPLACE DIRECTIVE
        population = op.trace(overlap_matrix @ (density_matrix))

        return population
    
    @abstractmethod
    def run(self):
        pass

    @property
    def nlayers(self):
        return self._nlayers

    @nlayers.setter
    def nlayers(self, val):

        assert val == 2, \
                "Only two layers currently valid for projection embedding."
        return self._nlayers

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

class StandardDFT(EmbeddingBase):

    def __init__(self, atoms, calc_base_ll, embed_mask=None, calc_base_hl=None, run_dir="./EmbASI_calc"):
        """Runs a normal DFT calculation without embedding

        A class which runs a standard DFT calculation without
        embedding - mostly useful for reference calculations.

        Parameters
        ----------
        atoms: ASE Atoms
            Structural and chemical information of system
        calc_base_ll: ASE FileIOCalculator
            Calculator object for layer 1
        embed_mask: int or list
            Number of atoms from index 0 in layer 1, or list of reigons assigned 
            to each atom (i.e., [1,1,1,2,2,2]) - does nothing for StandardDFT
        calc_base_hl: ASE FileIOCalculator
            Calculator object for layer 2 - does nothing for StandardDFT


        """

        from copy import copy, deepcopy
        from mpi4py import MPI

        self.calc_names = ["AB_LL"]

        super(StandardDFT, self).__init__(atoms, embed_mask, calc_base_ll, 
                                          calc_base_hl, run_dir=run_dir)
        low_level_calculator_1 = deepcopy(self.calculator_ll)

        self.set_layer(atoms, self.calc_names[0], low_level_calculator_1, 
                       embed_mask, ghosts=0, no_scf=False)

    def run(self):

        start = time.time()
        self.AB_LL.run()
        end = time.time()
        self.time_tot = end - start
        root_print(f" -----------======================--------- " )
        root_print(f" Total Energy (AB High-Level): {self.AB_LL.total_energy} eV" )
        root_print(f" -----------======================--------- " )


class ProjectionEmbedding(EmbeddingBase):
    """Implementation of Projeciton-Based Embedding with ASI communication

    A class controlling the interaction between two subsystems calculated at
    two levels of theory (low-level and high-level) with the
    Projection-Based Embedding (PbE) scheme of Manby et al.[1].

    Parameters
    ----------
    atoms: ASE Atoms
        Structural and chemical information of system
    embed_mask: int or list
        Number of atoms from index 0 in layer 1, or list of reigons assigned
        to each atom (i.e., [1,1,1,2,2,2])
    calc_base_ll: ASE FileIOCalculator
        Calculator object for layer 1
    calc_base_hl: ASE FileIOCalculator
        Calculator object for layer 2
    frag_charge: int
        Charge of the embedded fragment. Defaults to 0.
    post_scf: str
        Post-HF method applied to high-level calculation. Defaults to None.
    mu_val: float
        Pre-factor for level-shift orthogonalisation. Defaults to 1e+06 Ha.
    truncate_basis: float or None:
        Truncates the basis functions of the environment based on
        a Mulliken charge metrix. Turned off if None. Defaults to None.

    References
    ----------
    [1] Manby, F. R.; Stella, M.; Goodpaster, J. D.; Miller, T. F. I.
    A Simple, Exact Density-Functional-Theory Embedding Scheme. J. Chem.
    Theory Comput. 2012, 8 (8), 2564–2568.

    """

    def __init__(self, atoms, embed_mask, calc_base_ll, calc_base_hl,
                 total_charge=0, post_scf=None, total_energy_corr="1storder",
                 truncate_basis_thresh=None, truncate_basis_atoms=None,
                 localisation='SPADE', projection="level-shift",
                 mu_val=1.e+06, parallel=False, gc=True, run_dir="./EmbASI_calc"):

        from copy import copy, deepcopy
        from mpi4py import MPI

        self.total_energy_corr = total_energy_corr

        if self.total_energy_corr == "1storder":
            self.calc_names = ["AB_LL","A_LL","A_HL","A_HL_PP"]
        elif self.total_energy_corr == "nonscf": 
            self.calc_names = ["AB_LL","A_LL","A_HL","A_HL_PP","AB_LL_PP"]
        else:
            raise Exception("Invalid entry for total_energy_corr: use '1storder' or 'nonscf' ")

        super(ProjectionEmbedding, self).__init__(atoms, embed_mask,
                                                  calc_base_ll, calc_base_hl,
                                                  run_dir=run_dir)

        # Determines whether arrays will be communicated in parallel
        self.parallel = parallel
        self.gc = gc

        self.localisation = localisation
        if self.localisation == "SPADE":
            self.calculator_ll.parameters['qm_embedding_mo_localise']=".false."
            self.calculator_hl.parameters['qm_embedding_mo_localise']=".false."
        elif self.localisation == "qmcode":
            self.calculator_ll.parameters['qm_embedding_mo_localise']=".true."
            self.calculator_hl.parameters['qm_embedding_mo_localise']=".true."
        else:
            raise Exception("Invalid entry for localisation: use 'SPADE' or 'qmcode' ")
        #self.calculator_ll.parameters['override_default_empty_basis_order']=".true."
        #self.calculator_hl.parameters['override_default_empty_basis_order']=".true."

        self.projection = projection
        if self.projection == "level-shift":
            root_print(f"MO projection performed with: level-shift")
        elif self.projection == "huzinaga":
            root_print(f"MO projection performed with: huzinaga")
        else:
            raise Exception("Invalid entry for projection: use 'level-shift' or 'huzinaga' ")

        low_level_calculator_1 = deepcopy(self.calculator_ll)
        low_level_calculator_2 = deepcopy(self.calculator_ll)
        
        high_level_calculator_1 = deepcopy(self.calculator_hl)
        high_level_calculator_2 = deepcopy(self.calculator_hl)

        if self.total_energy_corr == "nonscf":
            low_level_calculator_3 = deepcopy(self.calculator_ll)

        # Determines the BLACS context and descriptors used for the communication
        # and storage of arrays in the NPScal registry structure
        if self.parallel:
            supersys_ctxt_tag = "main"
            subsys_ctxt_tag = "main"
            supersys_descr_tag = "supersystem"
            subsys_descr_tag = "subsystem"
        else:
            supersys_ctxt_tag = None
            subsys_ctxt_tag = None
            supersys_descr_tag = None
            subsys_descr_tag = None

        # Add AtomEmbedding objects assigned to each stage of the Projection
        # embedding workflow, where:
        # AB_LL:    Low-level supersystem reference
        # A_LL:     Low-level subsystem reference
        # A_HL:     High-level subsystem reference, w/ embedding pot
        # A_HL_PP:  High-level subsystem reference, post-processed w/o embedding pot
        low_level_calculator_1.parameters['qm_embedding_calc'] = 1
        self.set_layer(atoms, "AB_LL", low_level_calculator_1, 
                       embed_mask, ghosts=0, no_scf=False,
                       ctxt_tag=supersys_ctxt_tag,
                       descr_tag=supersys_descr_tag)
        self.AB_LL.input_total_charge = total_charge

        low_level_calculator_2.parameters['qm_embedding_calc'] = 2
        low_level_calculator_2.parameters['charge_mix_param'] = 0.
        low_level_calculator_2.parameters['sc_iter_limit'] = 0
        self.set_layer(atoms, "A_LL", low_level_calculator_2,
                       embed_mask, ghosts=2, no_scf=False,
                       ctxt_tag=subsys_ctxt_tag,
                       descr_tag=subsys_descr_tag)

        high_level_calculator_1.parameters['qm_embedding_calc'] = 3
        self.set_layer(atoms, "A_HL", high_level_calculator_1,
                       embed_mask, ghosts=2, no_scf=False,
                       ctxt_tag=subsys_ctxt_tag,
                       descr_tag=subsys_descr_tag)

        high_level_calculator_2.parameters['qm_embedding_calc'] = 2
        high_level_calculator_2.parameters['charge_mix_param'] = 0.
        high_level_calculator_2.parameters['sc_iter_limit'] = 0
        if "total_energy_method" in high_level_calculator_2.parameters:
            high_level_calculator_2.parameters['total_energy_method'] = high_level_calculator_2.parameters["xc"]
        self.set_layer(atoms, "A_HL_PP", high_level_calculator_2,
                       embed_mask, ghosts=2, no_scf=False,
                       ctxt_tag=subsys_ctxt_tag,
                       descr_tag=subsys_descr_tag)

        if self.total_energy_corr == "nonscf":
            low_level_calculator_3.parameters['qm_embedding_calc'] = 2
            low_level_calculator_3.parameters['charge_mix_param'] = 0.
            self.set_layer(atoms, "AB_LL_PP", low_level_calculator_3,
                           embed_mask, ghosts=0, no_scf=False,
                           ctxt_tag=supersys_ctxt_tag,
                           descr_tag=supersys_descr_tag)
            self.AB_LL.input_total_charge = total_charge

        self.mu_val = mu_val
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.ntasks = MPI.COMM_WORLD.Get_size()

        self.truncate = False
        if (truncate_basis_thresh is not None) and (truncate_basis_atoms is not None):
            raise Exception("Please only specific truncate_basis_thresh OR truncate_basis_atoms")
        
        if (truncate_basis_thresh is not None) or (truncate_basis_atoms is not None):
            self.truncate = True

        self.truncate_basis_thresh = truncate_basis_thresh
        self.truncate_basis_atoms = truncate_basis_atoms

    def calculate_levelshift_projector(self, densmat, overlap):
        """Calculates level-shift projection operator

        Calculate the level-shift based projection operator from 
        Manby et al.[1]:
                    P^{B} = /mu S^{AB} D^{B} S^{AB}
        where S^{AB} is the overlap matrix for the supermolecular system, and
        the density matrix for subsystem B.

        [1] Manby, F. R.; Stella, M.; Goodpaster, J. D.; Miller, T. F. I.
        A Simple, Exact Density-Functional-Theory Embedding Scheme.
        J. Chem. Theory Comput. 2012, 8 (8), 2564–2568.
        """

        self.P_b = self.mu_val * (overlap @ densmat @ overlap)

    def calculate_huzinaga_projector(self, hamiltonian, overlap, densmat):

        #self.P_b = -0.5*( (hamiltonian @ densmat @ overlap.T) + (overlap @ densmat @ hamiltonian.T) )
        self.P_b = -0.5*( (hamiltonian @ densmat @ overlap) + (overlap @ densmat @ hamiltonian) )
        
    def spade_localisation(self, atomsembed, hamiltonian, overlap):
        """Calculate the localised density matrix with the SPADE method

        As the eigenvectors (MO coefficient matrix) is not a part of the 
        ASI specification, we solve the Roothan-Hall eigenvalue problem
        and construct the density matrix at the wrapper level.

        """
        from embasi.roothan_hall_eigensolver import hamiltonian_eigensolv, calculate_densmat, overlap_illcondition_check
        from embasi.roothan_hall_eigensolver_scalapack import hamiltonian_eigensolv_parallel, overlap_illcondition_check_parallel
        from scalapack4py.npscal.math_utils.npscal2npscal import eig, svd
        from embasi.parallel_utils import mpi_bcast_matrix
        import copy

        root_print('Starting SPADE localisation...')
        nelecs = atomsembed.free_atom_nelectrons - atomsembed.input_total_charge
        if self.parallel:
            evals, evecs, occ_mat = hamiltonian_eigensolv_parallel(hamiltonian, \
                                                                   overlap, \
                                                                   nelecs, \
                                                                   return_orthog=False)
        else:
            evals, evecs, occ_mat = hamiltonian_eigensolv(hamiltonian, \
                                                          overlap, \
                                                          nelecs,
                                                          return_orthog=False)

        mask_val = []

        for idx, basis2atom in enumerate(atomsembed.basis_info.full_basis_atoms):
            if atomsembed.embed_mask[basis2atom]==1:
                mask_val.append(True)
            else:
                mask_val.append(False)

        mask_val = np.array(mask_val)

        #if self.parallel:
        #    u, sval, v = svd(overlap)
        #else:
        #    u, sval, v = np.linalg.svd(overlap, full_matrices=True)

        #rank = sval > 1e-5
        #n_bad_vals = np.count_nonzero([not val for val in rank])

        #if n_bad_vals > 0:
        #    raise Exception("Error in spade_localisation: Non-singular values in overlap matrix - basis likely too large.")

        #mask_val = [val for (val, good_val) in zip(mask_val, rank) if good_val]
        #mask_val = [val for val in mask_val]
        max_occ_state = np.count_nonzero(occ_mat)
        evecs_occ = evecs[:, :max_occ_state]
        evecs_occ_a = evecs_occ[mask_val, :]

        if self.parallel:
            u, svals, v = svd(evecs_occ_a)
        else:
            u, svals, v = np.linalg.svd(evecs_occ_a, full_matrices=True)

        svals_diff = np.ediff1d(svals**2.0)
        max_sval_change_idx = np.argmax(np.abs(svals_diff)) + 1

        root_print(f'Maximum SPADE state for subsystem A: {max_sval_change_idx}')

        evecs_occ_a = evecs_occ @ v[:max_sval_change_idx, :].T
        evecs_occ_ab = evecs_occ

        # @TODOSPIN: Need to redefine occupancies
        #density_matrix_subsys_a = calculate_densmat(evecs_occ, occ_mat[:max_occ_state])
        #density_matrix_subsys_b = density_matrix_supersystem - density_matrix_subsys_a
        density_matrix_supersystem = 2.0 * (evecs_occ_ab @ evecs_occ_ab.copy().T)
        density_matrix_subsys_a = 2.0 * (evecs_occ_a @ evecs_occ_a.copy().T)
        
        if not(self.parallel):
            density_matrix_supersystem = mpi_bcast_matrix(density_matrix_supersystem)
            density_matrix_subsys_a = mpi_bcast_matrix(density_matrix_subsys_a)

        density_matrix_subsys_b = density_matrix_supersystem - density_matrix_subsys_a
        root_print(f'SPADE total supersystem A+B charge: {op.trace(overlap @ density_matrix_supersystem)}')
        root_print(f'SPADE localised subsystem A charge: {op.trace(overlap @ density_matrix_subsys_a)}')
        root_print(f'SPADE localised subsystem B charge: {op.trace(overlap @ density_matrix_subsys_b)}')

        root_print('Exiting SPADE localisation...')

        return density_matrix_subsys_a, density_matrix_subsys_b

    def run(self):
        """ Summary
        The primary driver routine for performing QM-in-QM with a
        Projection-based embedding scheme. This scheme draws upon
        the work of Manby et al. [1, 2].

        The embedding scheme uses the following total energy expression...

        Importing and exporting of density matrices and hamiltonians is 
        performed with the ASI package [3].

        The workflow operates as follows:
        1) Calculate the KS-DFT energy of the combined subsystems A+B. Localised
           density matrices, /gamma^{A} and /gamma^{B} 
        2a) Extract the localised density matrices 
        2b) (Optional) Select atoms in subsystem B that contribute
            significantly to subsystem A (threshold 0.5 |e|) via 
            Mulliken analysis: 
                q^{A}_{/mu, /nu} = /gamma^{A}_{/mu, /nu} S_{/mu, /nu}
            Basis functions of said atoms within calculations of the 
            embedded subsystem will be included as ghost atoms. Other
            basis functions will be removed (i.e., associated atomic centers
            not included in the QM calculation, and associated rows and 
            columns in intermediate hamiltonians and density matrices
            deleted).
        2) Calculate the total energy for subsystem A with the density 
           matrix, /gamma^{A}
        3) 

        (For users of LaTeX, I am aware that a forward slash is used
        in place of the traditional backward slash for mathematical symbols - 
        unfortunately using backslashes in these comment blocks produces ugly
        warnings within the comment blocks.)

        
        ...

        (1) Manby, F. R.; Stella, M.; Goodpaster, J. D.; Miller, T. F. I. 
            A Simple, Exact Density-Functional-Theory Embedding Scheme. 
            J. Chem. Theory Comput. 2012, 8 (8), 2564–2568.
        (2) Lee, S. J. R.; Welborn, M.; Manby, F. R.; Miller, T. F. 
            Projection-Based Wavefunction-in-DFT Embedding. Acc. Chem. Res. 
            2019, 52 (5), 1359–1368.
        (3) TODO: REF
        """
        import numpy as np
        import tracemalloc
        tracemalloc.start(50)

        root_print("Embedding calculation begun...")

        # Performs a single-point energy evaluation for a system composed of A
        # and B. Returns localised density matrices for subsystems A and B, and
        # the two-electron components of the hamiltonian (combined with
        # nuclear-electron potential).
        start = time.time()
        self.AB_LL.run()
        self.subsys_AB_lowlvl_totalen = self.AB_LL.total_energy
        end = time.time()
        self.time_ab_lowlevel = end - start

        if self.parallel:
            overlap = self.AB_LL.overlap.copy()
            hamiltonian_AB_total = self.AB_LL.hamiltonian_total.copy()
            AB_hamiltonian_estat_plus_xc = self.AB_LL.hamiltonian_estat_plus_xc.copy()
        else:
            overlap = copy.deepcopy(self.AB_LL.overlap)
            hamiltonian_AB_total = copy.deepcopy(self.AB_LL.hamiltonian_total)
            AB_hamiltonian_estat_plus_xc = copy.deepcopy(self.AB_LL.hamiltonian_estat_plus_xc)

        # Read the localised density matrices output by the QM code or
        # perform SPADE localisation on the wrapper level.
        basis_info = self.set_basis_info(self.AB_LL)
        self.AB_LL.basis_info = basis_info
        if self.localisation == "SPADE":
            if self.gc:
                root_print(f"Pre-GC AB_LL: {tracemalloc.get_traced_memory()}")
                self.AB_LL.garbage_collect()
            start = time.time()
            densmat_A_LL, densmat_B_LL = self.spade_localisation(self.AB_LL, hamiltonian_AB_total, overlap)
            end = time.time()
            self.time_spade = end - start
        else:
            densmat_A_LL = self.AB_LL.density_matrices_out[0]
            densmat_B_LL = self.AB_LL.density_matrices_out[1]
            if self.gc:
                self.AB_LL.garbage_collect()


        # Initialises the density matrix for subsystem A, and calculates the
        # hamiltonian components for subsystem A at the low-level reference.
        if self.truncate:

            if (self.truncate_basis_thresh is not None):
                self.basis_mask = self.select_atoms_basis_truncation(self.AB_LL,
                                                                densmat_A_LL,
                                                                overlap,
                                                                self.truncate_basis_thresh)
            elif (self.truncate_basis_atoms is not None):
                self.basis_mask = [ x in self.truncate_basis_atoms for x in np.arange(len(self.AB_LL.atoms)) ]
            else:
                raise Exception("self.truncate is True, but neither self.truncate_basis_thresh or self.truncate_basis_atoms are set")

            self.basis_info = self.set_truncation_defaults(self.AB_LL, self.basis_mask)
            self.trunc_basis_atoms = self.basis_info.trunc_basis_atoms

            self.AB_LL.truncate = False
            self.A_LL.truncate = True
            self.A_HL.truncate = True
            self.A_HL_PP.truncate = True
        else:
            self.basis_info = self.set_basis_info(self.AB_LL)
            self.AB_LL.truncate = False
            self.A_LL.truncate = False
            self.A_HL.truncate = False
            self.A_HL_PP.truncate = False

        if self.total_energy_corr == "nonscf":
            self.AB_LL_PP.truncate = False
            self.AB_LL_PP.basis_info = basis_info

        self.AB_LL.basis_info = self.basis_info
        self.A_LL.basis_info = self.basis_info
        self.A_HL.basis_info = self.basis_info
        self.A_HL_PP.basis_info = self.basis_info

        # Calculates the electron count for the combined (A+B) and separated 
        # subsystems (A and B).
        self.AB_pop = self.calc_subsys_pop(overlap, \
                                           (densmat_A_LL + densmat_B_LL))

        self.A_pop = self.calc_subsys_pop(overlap, densmat_A_LL)

        self.B_pop = self.calc_subsys_pop(overlap, densmat_B_LL)

        root_print(f" Population of Subsystem AB: {self.AB_pop}")
        root_print(f" Population of Subsystem A: {self.A_pop}")
        root_print(f" Population of Subsystem B: {self.B_pop}")

        # Calculate the energy for subsystem A with the lower level of theory
        self.A_LL.density_matrix_in = densmat_A_LL
        if self.truncate:
            trunc_densmat_A_LL = self.A_LL.full_mat_to_truncated(densmat_A_LL)

        self.A_LL.input_fragment_nelectrons = self.A_pop
        start = time.time()
        self.A_LL.run(ev_corr_scf=True)
        self.subsys_A_lowlvl_totalen = self.A_LL.ev_corr_total_energy
        end = time.time()
        self.time_a_lowlevel = end - start

        # Initialises the density matrix for subsystem A, and calculated the 
        # hamiltonian components for subsystem A at the low-level reference.
        if self.projection == "level-shift":
            self.calculate_levelshift_projector(densmat_B_LL, overlap)
        elif self.projection == "huzinaga":
            self.calculate_huzinaga_projector(hamiltonian_AB_total, overlap, densmat_B_LL)
        else:
            raise Exception("Invalid entry for projection: use 'level-shift' or 'huzinaga' ")

        # Calculate density matrix for subsystem A at the higher level of 
        # theory. Two terms are added to the hamiltonian matrix of the embedded
        # subsystem to form the full embedded Fock-matrix, F^{A}:
        #   F^{A} = h^{core} + g^{high-level}[A] + v_{emb}^[A, B] + /mu P^{B}
        # 1) v_{emb}^[A, B], the embedding potential, which introduces the 
        #    Hartree and exchange-correlation potentials of the environment
        #    (in the case of FHI-aims, this includes the full electrostatic
        #    potential, i.e., the Hartree and nuclear-electron potentials of 
        #    subsystem B acting on subsystem A).
        # 2) The level-shift operator, /mu P^{B}, which orthogonalises the basis
        #    functions associated with the environment (subsystem B) from the 
        #    embedded subsystem by adding a large energy penalty to hamiltonian
        #    components associated with subsystem B.
        #
        # Registered callbacks in ASI add the above components to the Fock-matrix
        # at every SCF iteration.
        self.A_HL.density_matrix_in = densmat_A_LL
        self.A_HL.input_fragment_nelectrons = self.A_pop
        self.vemb = AB_hamiltonian_estat_plus_xc - self.A_LL.hamiltonian_estat_plus_xc
        self.A_HL.fock_embedding_matrix = self.vemb + self.P_b

        if self.gc:
            root_print(f"Pre-GC A_LL: {tracemalloc.get_traced_memory()}")
            self.A_LL.garbage_collect()
            root_print(f"Post-GC A_LL: {tracemalloc.get_traced_memory()}")
        
        start = time.time()
        self.A_HL.run()
        end = time.time()
        self.time_a_highlevel = end - start

        densmat_A_HL = copy.copy(self.A_HL.density_matrices_out[0])

        if self.truncate:
            trunc_vemb = self.A_HL.full_mat_to_truncated(self.vemb)
            trunc_P_b = self.A_HL.full_mat_to_truncated(self.P_b)
            trunc_densmat_A_HL = self.A_HL.full_mat_to_truncated(densmat_A_HL)
        
        if self.gc:
            root_print(f"Pre-GC A_HL: {tracemalloc.get_traced_memory()}")
            self.A_HL.garbage_collect()
            root_print(f"Post-GC A_HL: {tracemalloc.get_traced_memory()}")
                
        # Calculate the total energy of the embedded subsystem A at the high
        # level of theory without the associated embedding potential.        
        self.A_HL_PP.density_matrix_in = densmat_A_HL
        self.A_HL_PP.input_fragment_nelectrons = self.A_pop
        start = time.time()
        self.A_HL_PP.run(ev_corr_scf=True)
        self.subsys_A_highlvl_totalen = self.A_HL_PP.ev_corr_total_energy
        end = time.time()
        self.time_a_highlevel_pp = end - start

        if self.gc:
            root_print(f"Pre-GC A_HL_PP: {tracemalloc.get_traced_memory()}")
            self.A_HL_PP.garbage_collect()
            root_print(f"Post-GC A_HL_PP: {tracemalloc.get_traced_memory()}")

        if self.total_energy_corr == "nonscf":
            # A terrible cludge which requires improvement.
            # Re-normalising charge for differing atomic solvers (bad cludge)
            # root_print(f" Normalizing density matrix from high-level reference...")
            # self.A_HL_pop = self.calc_subsys_pop(self.AB_LL.overlap, 
            #                                 self.A_HL.density_matrices_out[0])
            # root_print(f" Population of Subystem A^[HL]: {self.A_HL_pop}")
            # self.charge_renorm = (self.A_pop/self.A_HL_pop)
            # root_print(f" Population of Subystem A^[HL] (post-norm): 
            #             {self.calc_subsys_pop(self.AB_LL.overlap, 
            #             self.charge_renorm*self.A_HL.density_matrices_out[0])}")
            self.charge_renorm = 1.0

            # Calculate A low-level reference energy
            self.A_LL.density_matrix_in = self.charge_renorm * \
                                        self.A_HL.density_matrices_out[0]
            self.A_LL.run(ev_corr_scf=True)
            start = time.time()
            self.subsys_A_lowlvl_totalen = self.A_LL.ev_corr_total_energy
            end = time.time()
            self.time_a_lowlevel_pp = end - start

            # Calculate AB low-level reference energy
            self.AB_LL_PP.density_matrix_in = \
                (self.charge_renorm * self.A_HL.density_matrices_out[0]) + densmat_B_LL

            start = time.time()
            self.AB_LL_PP.run(ev_corr_scf=True)
            self.subsys_AB_lowlvl_totalen = self.AB_LL_PP.ev_corr_total_energy
            end = time.time()
            self.time_ab_lowlevel_pp = end - start

        # Calculate projected density correction to total energy
        if self.truncate:
            self.PB_corr = \
                (op.trace(trunc_P_b @ trunc_densmat_A_HL) * 27.211384500)
        else:
            self.PB_corr = \
                (op.trace(self.P_b @ densmat_A_HL) * 27.211384500)

        if "total_energy_method" in self.A_HL.initial_calc.parameters:
            self.subsys_A_highlvl_totalen = self.subsys_A_highlvl_totalen + \
                self.A_HL.post_scf_corr_energy - self.A_HL.dft_energy

        if self.total_energy_corr == "1storder":
            if self.truncate:
                self.order_1_embedding_corr = op.trace((trunc_densmat_A_HL - trunc_densmat_A_LL) @ \
                                                       (trunc_vemb)) * 27.211384500
            else:
                self.order_1_embedding_corr = op.trace((densmat_A_HL - densmat_A_LL) @ \
                                                       (self.vemb)) * 27.211384500

            self.DFT_AinB_total_energy = self.subsys_A_highlvl_totalen - \
                                       self.subsys_A_lowlvl_totalen + self.subsys_AB_lowlvl_totalen + \
                                       self.order_1_embedding_corr + self.PB_corr

        if self.total_energy_corr == "nonscf":
            self.DFT_AinB_total_energy = self.subsys_A_highlvl_totalen - \
                self.subsys_A_lowlvl_totalen + self.subsys_AB_lowlvl_totalen + self.PB_corr

        root_print( f" ----------- FINAL         OUTPUTS --------- " )
        root_print(f" ")
        root_print(f" Population Information:")
        root_print(f" Population of Subsystem AB: {self.AB_pop}")
        root_print(f" Population of Subsystem A: {self.A_pop}")
        root_print(f" Population of Subsystem B: {self.B_pop}")
        root_print(f" ")
        root_print( f" ----------- TIMING    INFORMATION --------- " )
        if self.localisation == "SPADE":
            root_print(f" Total time: {self.time_a_lowlevel + self.time_ab_lowlevel + self.time_a_highlevel + self.time_a_highlevel_pp + self.time_spade} s")
        else:
            root_print(f" Total time: {self.time_a_lowlevel + self.time_ab_lowlevel + self.time_a_highlevel + self.time_a_highlevel_pp} s")
        root_print(f" Peak memory usage: {tracemalloc.get_traced_memory()[1]/(1024*1024)} MB")        
        root_print( f" ------------------------------------------- " )
        root_print(f" Intermediate Information:")
        root_print(f" WARNING: These are not faithful, ground-state KS total energies - ")
        root_print(f" In the case of low-level references, they are calculated using the ")
        root_print(f" density components of the high-level energy reference for fragment A. ")
        root_print(f" Do not naively use these energies unless you are comfortable with ")
        root_print(f" their true definition. ")
        root_print(f" Total Energy (A+B Low-Level): {self.subsys_AB_lowlvl_totalen} eV" )
        root_print(f" Total Energy (A Low-Level): {self.subsys_A_lowlvl_totalen} eV" )
        root_print(f" Total Energy (A High-Level): {self.subsys_A_highlvl_totalen} eV" )
        root_print(f" Projection operator energy correction DM^(A_HL) @ Pb: {self.PB_corr} eV" )
        if self.total_energy_corr == "1storder":
            root_print(f" First order energy correction (DM^(A_HL)-DM^(A_LL)) @ v_emb): {self.order_1_embedding_corr} eV" )
        root_print(f"  " )
        root_print(f" Final Energies Information:")
        root_print(f" Final total energy (Uncorrected): {self.DFT_AinB_total_energy - self.PB_corr} eV" )
        root_print(f" Final total energy (Projection Corrected): {self.DFT_AinB_total_energy} eV" )
        root_print(f" " )
        root_print(f" -----------======================--------- " )
        root_print(f" " )

        # And finally, now all the work is done, clear the ScaLAPACK
        # registers in case another calculation is ran.
        from scalapack4py.npscal.blacs_ctxt_management import CTXT_Register, DESCR_Register
        CTXT_Register.clear_register()
        DESCR_Register.clear_register()
