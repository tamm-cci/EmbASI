from asi4py.asecalc import ASI_ASE_calculator
from embasi.parallel_utils import root_print, mpi_bcast_matrix_storage, \
    mpi_bcast_integer
import scalapack4py.npscal.math_utils.operations as op
import numpy as np
from mpi4py import MPI

class AtomsEmbed():
    """A wrapper around an ASE atoms objects for ASI library calls

    The wrapper controls a number of functions needed for embedding:
    1) Passes the parameters necessary for the execution of the QM code
       for each step in the calculation. This includes the number of 
       atoms in each embedding layer, the level of theory, and the 
       atoms assigned as ghost species in a given step.
    2) Stores matrices (i.e., density matrices and hamiltonians) output
       by a given system callo
    3) Assigns which matrices are passed to the QM code for initialization
       (e.g., the density matrix)
    
    Parameters
    ----------
    atoms: ASE Atoms Object
        Atoms object set containing initial system information
    initial_calc: ASE FileIOCalculator
        Calculator object for a QM code supported by ASE and ASI
    embed_mask: int or list
        Assigns either the first in atoms to region 1, or an index of 
        int values 1 and 2 to each embedding layer. WARNING: The atoms
        object will be reordered such that embedding layer 1 appear 
        first.
    ghosts: int
        Assigns atoms to be evaluated as ghost species for the purposes
        of embedding. If ghosts are needed for BSSE, they must be 
        set in dictionary atoms.info['ghosts'] as a mask.
    outdir: str
        Name of directory output files are saved to
    no_scf: bool
        Old control flow for terminating the QM code at the first SCF 
        step.

    """

    def __init__(self, atoms, initial_calc, embed_mask, ghosts=0,
                 outdir='asi.calc', no_scf=False, ctxt_tag=None,
                 descr_tag=None):

        self.atoms = atoms.copy()
        self.initial_embed_mask = embed_mask
        self.outdir = outdir

        # Determines whether arrays are to be communicated in serial,
        # or as BLACS distributed arrays from a globally stored context
        # provided by NPScal
        if (ctxt_tag is None) and (descr_tag is None):
            self.parallel = False
            self.blacs_ctxt_tag = None
            self.blacs_descr_tag = None
        else:
            self.parallel = True
            self.blacs_ctxt_tag = ctxt_tag
            self.blacs_descr_tag = descr_tag
            
        if isinstance(embed_mask, int):
            # We hope the user knows what they are doing and
            # their atoms object is ordered accordingly.
            self.embed_mask = [1]*embed_mask
            self.embed_mask += [2]*(len(atoms)-embed_mask)
        elif isinstance(embed_mask, list):
            self.embed_mask = embed_mask
            assert len(atoms)==len(embed_mask), \
                "Length of embedding mask does not match number of atoms"
        elif embed_mask is None:
            self.embed_mask = None

        if self.embed_mask is not None:
            self.reorder_atoms_from_embed_mask()
            self.atoms.info['embedding_mask'] = self.embed_mask

        self.initial_calc = initial_calc

        self.truncate = False
        self.density_matrix_in = None
        self.fock_embedding_matrix = None

        self.no_scf = no_scf

        if self.embed_mask is not None:
            if isinstance(ghosts, int):
                ghosts = [ghosts]
            self.ghost_list = [(at in [ghosts]) for at in self.embed_mask]
        else:
            self.ghost_list = [False]*len(atoms)

    def calc_initializer(self, asi):

        calc = self.initial_calc

        if self.truncate:
            self.ghost_list = [ 
                ghst for (idx, ghst) in enumerate(self.ghost_list)
                           if idx in self.basis_info.active_atoms ]
        else:
            self.ghost_list = self.ghost_list

        # Before passing into the main input writer, we may
        # have extra ghosts needed for the CP correction to the
        # BSSE
        #if "ghosts" in self.atoms.info.keys():
        #    for idx, ghost in enumerate(self.atoms.info["ghosts"]):
        #        if ghost:
        #            ghost_list[idx] = True

        if hasattr(self, "input_total_charge"):
            total_charge = self.input_total_charge
        elif hasattr(self, "input_fragment_nelectrons"):
            total_charge = self.fragment_total_charge
        else:
            total_charge = 0.
        
        calc.parameters['charge'] = -float(total_charge)

        # Ensure the Aims template shares the input parameters of the calculator object
        calc.parameters["ghosts"] = self.ghost_list
        calc.template.parameters = calc.parameters

        calc.write_inputfiles(asi.atoms, properties=['energy'])

        if self.embed_mask is not None:
            self._insert_embedding_region_aims()

    def reorder_atoms_from_embed_mask(self):
        """ Re-orders atoms to push those in embedding region 1 to the beginning
        of the atom lists to ensure data is organised in a predictable manner.

        """

        import numpy as np

        "Check if embedding mask is in the correct order (e.g., [1,1,1,2,2,2])"
        "Ensure the next value is always ge than the current"
        idx_list = np.argsort(self.embed_mask)
        sort_embed_mask = np.sort(self.embed_mask)

        self.embed_mask = sort_embed_mask
        self.atoms = self.atoms[idx_list]

        if "ghosts" in self.atoms.info.keys():
            self.atoms.info["ghosts"] = [ 
                self.atoms.info["ghosts"][idx] for idx in idx_list ]

    def _insert_embedding_region_aims(self):
        """Places embedding regions in input file

        This functionality would require ASE support of the 
        qm_embedding_region, so instead we handle this in a rather ad hoc
        way and inserts the qm_embedding_region variable for each atom
        in the geometry.in file.

        """
        import os

        cwd = os.getcwd()
        geometry_path = os.path.join(cwd, "geometry.in")
        with open(geometry_path, 'r') as fil:
            lines = fil.readlines()
            mask = [any(s in str(line) for s in ('atom', 'empty')) for line in lines]

        shift = 0
        for idx, maskval in enumerate(mask):
            if maskval:
                embedding = self.atoms.info['embedding_mask'][shift]
                shift += 1
                lines.insert(idx + shift, f'qm_embedding_region {embedding}\n')

        with open(geometry_path, 'w') as fil:
            lines = "".join(lines)
            fil.write(lines)

    def full_mat_to_truncated(self, full_mat):
        """Converts a matrix quantity from a full to a truncated basis

        Deletes matrix elements which are not part of the active region,
        leaving rows and columns of the basis functions spanned by
        by the active_atoms.

        Parameters
        ----------
        full_mat: np.ndarray, shape(full_nbasis, full_nbasis)
            Matrix quantity for the full system

        Returns
        -------
        trunc_mat: np.ndarray shape(trunc_nbasis, trun_nbasis)
            Matrix quantity for the truncated system

        """

        import copy
        import os

        # Set to local variables to improve readability
        active_atoms = self.basis_info.active_atoms

        # TODO: Set-up for upper-triangular matrices.
        trunc_basis_min_idx = self.basis_info.trunc_basis_min_idx
        trunc_basis_max_idx = self.basis_info.trunc_basis_max_idx
        full_basis_min_idx = self.basis_info.full_basis_min_idx
        full_basis_max_idx = self.basis_info.full_basis_max_idx

        # Set-up empty matrix to read into
        full_nbasis = self.basis_info.full_nbasis
        trunc_nbasis = self.basis_info.trunc_nbasis

        if self.parallel:
            from scalapack4py.npscal import NPScal
            from scalapack4py.npscal.blacs_ctxt_management import DESCR_Register, BLACSDESCRManager
            from ctypes import cdll, CDLL, RTLD_GLOBAL

            lib = os.environ['ASI_LIB_PATH']

            trunc_mat = NPScal(ctxt_tag=self.blacs_ctxt_tag, descr_tag=self.blacs_descr_tag, lib=lib,
                               gl_m=trunc_nbasis, gl_n=trunc_nbasis, dmb=16, dnb=16)
        else:
            trunc_mat = np.zeros(shape=(full_nbasis, full_nbasis))

        for atom1 in active_atoms:

            # Skip atoms belonging to region A (or 1) as their basis
            # functions are already included

            for atom2 in active_atoms:
                # Skip core active atom blocks - they are already
                # correctly placed.

                atom2_trunc = np.min(np.where(active_atoms==atom2))
                atom1_trunc = np.min(np.where(active_atoms==atom1))

                trunc_row_min = trunc_basis_min_idx[atom2_trunc]
                trunc_row_max = trunc_basis_max_idx[atom2_trunc]
                trunc_col_min = trunc_basis_min_idx[atom1_trunc]
                trunc_col_max = trunc_basis_max_idx[atom1_trunc]

                full_row_min = full_basis_min_idx[atom2]
                full_row_max = full_basis_max_idx[atom2]
                full_col_min = full_basis_min_idx[atom1]
                full_col_max = full_basis_max_idx[atom1]

                trunc_mat[trunc_row_min:trunc_row_max, 
                          trunc_col_min:trunc_col_max] = \
                        full_mat[full_row_min:full_row_max, 
                                 full_col_min:full_col_max]

        return trunc_mat

    def truncated_mat_to_full(self, trunc_mat):
        """Converts a matrix quantity from a truncated to a full basis

        Maps matrix elements of give basis functions to their corresponding
        indices in the matrix of the full, non-trauncated basis.

        Parameters
        ----------
        trunc_mat: np.ndarray shape(trunc_nbasis, trun_nbasis)
            Matrix quantity for the truncated system

        Returns
        -------
        full_mat: np.ndarray, shape(full_nbasis, full_nbasis)
            Matrix quantity for the full system

        """

        import copy
        import os

        # Set to local variables to improve readability
        active_atoms = self.basis_info.active_atoms

        # TODO: Set-up for upper-triangular matrices.
        trunc_basis_min_idx = self.basis_info.trunc_basis_min_idx
        trunc_basis_max_idx = self.basis_info.trunc_basis_max_idx
        full_basis_min_idx = self.basis_info.full_basis_min_idx
        full_basis_max_idx = self.basis_info.full_basis_max_idx

        # Set-up empty matrix to read into
        full_nbasis = self.basis_info.full_nbasis
        if self.parallel:
            from scalapack4py.npscal import NPScal
            from scalapack4py.npscal.blacs_ctxt_management import DESCR_Register
            from ctypes import cdll, CDLL, RTLD_GLOBAL

            lib = os.environ['ASI_LIB_PATH']
            print(f"DESCR REGISTER {DESCR_Register}")
            new_descr_tag = "supersystem"

            full_mat = NPScal(ctxt_tag=self.blacs_ctxt_tag, descr_tag=new_descr_tag, lib=lib)
        else:
            full_mat = np.zeros(shape=(full_nbasis, full_nbasis))

        for atom1 in active_atoms:

            # Skip atoms belonging to region A (or 1) as their basis
            # functions are already included

            for atom2 in active_atoms:
                # Skip core active atom blocks - they are already
                # correctly placed.

                atom2_trunc = np.min(np.where(active_atoms==atom2))
                atom1_trunc = np.min(np.where(active_atoms==atom1))

                trunc_row_min = trunc_basis_min_idx[atom2_trunc]
                trunc_row_max = trunc_basis_max_idx[atom2_trunc]
                trunc_col_min = trunc_basis_min_idx[atom1_trunc]
                trunc_col_max = trunc_basis_max_idx[atom1_trunc]

                full_row_min = full_basis_min_idx[atom2]
                full_row_max = full_basis_max_idx[atom2]
                full_col_min = full_basis_min_idx[atom1]
                full_col_max = full_basis_max_idx[atom1]

                full_mat[full_row_min:full_row_max, 
                         full_col_min:full_col_max] = \
                    trunc_mat[trunc_row_min:trunc_row_max, 
                              trunc_col_min:trunc_col_max]

        return full_mat

    def extract_results(self):
        """Extracts quantities not currently supported by ASI

        An ad hoc solution to extract values unsupported by ASI for 
        FHI-aims. Currently reads values such as the kinetic energy,
        electrostatic energy, sum of eigenvalues etc,

        """

        with open(self.outdir+'/asi.log', 'r') as output:

            lines = output.readlines()

            for line in lines:
                outline = line.split()

                if '  | Kinetic energy                :' in line:
                    self.kinetic_energy = float(outline[6])

                if '  | Electrostatic energy          :' in line:
                    self.es_energy = float(outline[6])

                if '  | Sum of eigenvalues            :' in line:
                    self.ev_sum = float(outline[7])

                if '  | Total energy of the DFT' in line:
                    self.dft_energy = float(outline[11])

                if 'Total XC Energy     :' in line:
                    self.xc_energy = float(outline[6])

                if 'Total energy after the post-s.c.f.' in line:
                    self.post_scf_corr_energy = float(outline[9])

    def run(self, ev_corr_scf=False):
        """Invokes the ASI_run() call and extracts matrix quantities

        Driver routine which executes the QM code and controls which
        callbacks are registered, which matrix quantities are exported,
        and initialises relevant properties from previous calculations.
        
        Parameters
        ----------
        ev_corr_scf: bool
           Replaces energy contribution from sum of eigenvalues with
           product of the density matrix and hamiltonian

        """
        import os
        import numpy as np
        from asi4py.asecalc import ASI_ASE_calculator
        from embasi.asi_default_callbacks import dm_saving_callback, \
                                                        ham_saving_callback, \
                                                        ovlp_saving_callback, \
                                                        matrix_loading_callback

        root_print(f'Calculation {self.outdir}...')

        if self.truncate and len(self.atoms) != self.basis_info.trunc_natoms:
            self.atoms = self.atoms[self.basis_info.active_atoms]

        self.atoms.calc = ASI_ASE_calculator(os.environ['ASI_LIB_PATH'],
                                        self.calc_initializer,
                                        MPI.COMM_WORLD,
                                        self.atoms,
                                        work_dir=self.outdir)

        # Explicitly set function pointers to NULL to avoid
        # previously set function pointers from passing into
        # the present calculation.
        self.atoms.calc.asi.register_overlap_callback(0, 0)
        self.atoms.calc.asi.register_dm_callback(0, 0)
        self.atoms.calc.asi.register_DM_init(0, 0)
        self.atoms.calc.asi.register_hamiltonian_callback(0, 0)
        self.atoms.calc.asi.register_set_hamiltonian_callback(0, 0)
        self.atoms.calc.asi.register_modify_hamiltonian_callback(0, 0)

        # Register the relevant callbacks
        # self.atoms.calc.asi.keep_overlap = True
        self.atoms.calc.asi.overlap_storage = {}
        self.atoms.calc.asi.register_overlap_callback(ovlp_saving_callback, 
                                                      (self.atoms.calc.asi, 
                                                       self.atoms.calc.asi.overlap_storage,
                                                       self.blacs_ctxt_tag,
                                                       self.blacs_descr_tag,
                                                       'Ovlp calc'))


        self.atoms.calc.asi.dm_storage = {}
        self.atoms.calc.asi.dm_calc_cnt = {}
        self.atoms.calc.asi.dm_count = 0
        self.atoms.calc.asi.register_dm_callback(dm_saving_callback, 
                                                 (self.atoms.calc.asi, 
                                                  self.atoms.calc.asi.dm_storage, 
                                                  self.atoms.calc.asi.dm_calc_cnt,
                                                  self.blacs_ctxt_tag,
                                                  self.blacs_descr_tag,
                                                  'DM calc'))
        
        self.atoms.calc.asi.ham_storage = {}
        self.atoms.calc.asi.ham_calc_cnt = {}
        self.atoms.calc.asi.ham_count = 0
        self.atoms.calc.asi.register_hamiltonian_callback(ham_saving_callback, 
                                                          (self.atoms.calc.asi,
                                                           self.atoms.calc.asi.ham_storage,
                                                           self.atoms.calc.asi.ham_calc_cnt,
                                                           self.blacs_ctxt_tag,
                                                           self.blacs_descr_tag,
                                                           'Ham calc'))

        if self.density_matrix_in is not None:
            self.atoms.calc.asi.register_DM_init(matrix_loading_callback,
                                                 (self.atoms.calc.asi,
                                                 {(1,1): self.density_matrix_in},
                                                 self.blacs_ctxt_tag,
                                                 self.blacs_descr_tag,
                                                 'DM init'))

        if self.fock_embedding_matrix is not None:
            self.atoms.calc.asi.register_modify_hamiltonian_callback(matrix_loading_callback,
                                                                     (self.atoms.calc.asi,
                                                                     {(1,1): self.fock_embedding_matrix},
                                                                     self.blacs_ctxt_tag,
                                                                     self.blacs_descr_tag,
                                                                     'Modify H'))

        E0 = self.atoms.get_potential_energy()

        self.total_energy = E0
        self.basis_atoms = self.atoms.calc.asi.basis_atoms
        self.n_basis = self.atoms.calc.asi.n_basis
        
        # BROADCAST QUANTITIES ONLY CALCULATED FOR THE HEAD NODE TO ALL
        # OTHER NODES - ONLY DO THIS IN SERIAL MODE AS THE NPSCAL ARRAYS
        # ARE ALREADY DISTRIBUTED TO EACH TASK
        if not (self.parallel):
            self.atoms.calc.asi.ham_storage = \
                mpi_bcast_matrix_storage(self.atoms.calc.asi.ham_storage,
                                         self.atoms.calc.asi.n_basis,
                                         self.atoms.calc.asi.n_basis)
            self.atoms.calc.asi.dm_storage = \
                mpi_bcast_matrix_storage(self.atoms.calc.asi.dm_storage,
                                         self.atoms.calc.asi.n_basis,
                                         self.atoms.calc.asi.n_basis)

            self.atoms.calc.asi.overlap_storage = \
                mpi_bcast_matrix_storage(self.atoms.calc.asi.overlap_storage,
                                     self.atoms.calc.asi.n_basis,
                                     self.atoms.calc.asi.n_basis)

            self.atoms.calc.asi.dm_count = mpi_bcast_integer(self.atoms.calc.asi.dm_count)
            self.atoms.calc.asi.ham_count = mpi_bcast_integer(self.atoms.calc.asi.ham_count)

        self.atoms.calc.asi.close()
        MPI.COMM_WORLD.Barrier()

        self.extract_results()

        # Within the embedding workflow, we often want to calculate the total 
        # energy for a given density matrix without performing any SCF steps. 
        # Often, this includes using an input electron density constructed from 
        # a localised set of MOs for a fragment of a supermolecule. This 
        # density will be far from the ground-state density for the fragment,
        # meaning the output eigenvalues significantly deviate from those of a 
        # fully converged density.
        #
        # As the vast majority of DFT codes with the KS-eigenvalues to determine
        # the total energy, the total energies due to the eigenvalues do not 
        # formally reflect the density matrix of the initial input for 
        # iteration, n=0:
        #
        #    \gamma^{n+1} * H^{total}[\gamma^{n}] \= \gamma^{n} * H^{total}[\gamma^{n}],
        #
        # For TE-only calculations, we do not care about the SCF process - we 
        # are using the DFT code to integrate XC and electrostatic energies. As 
        # such, we 'correct' the eigenvalue portion of the total energy to reflect
        # the interaction of the input density matrix, as opposed to the first 
        # set of KS-eigenvectors resulting from the DFT code.
        if ev_corr_scf:
            if self.truncate:
                # @TODONPSCAL: REPLACE NP DIRECTIVE
                self.ev_corr_energy = \
                    27.211384500 * op.trace(self.density_matrix_in @ 
                                        self.full_mat_to_truncated(self.hamiltonian_total))
            else:
                # @TODONPSCAL: REPLACE NP DIRECTIVE
                self.ev_corr_energy = \
                    27.211384500 * op.trace(self.density_matrix_in @
                                            self.hamiltonian_total)

            self.ev_corr_total_energy = \
                self.total_energy - self.ev_sum + self.ev_corr_energy

    def garbage_collect(self):
        """Removes all stored matrices from memory

        """
        import gc

        del self.atoms.calc.asi
        self.density_matrix_in = None
        self.fock_embedding_matrix = None
        gc.collect()
            
    @property
    def hamiltonian_total(self):
        tot_idx = 3
        if self.truncate:
            return self.truncated_mat_to_full(self.atoms.calc.asi.ham_storage.get((tot_idx,1,1)))
        else:
            return self.atoms.calc.asi.ham_storage.get((tot_idx,1,1))

    @property
    def hamiltonian_estat_plus_xc(self):
        """_summary_
        Generates
        """
        estat_idx = 2
        if self.truncate:
            return self.truncated_mat_to_full(self.atoms.calc.asi.ham_storage.get((estat_idx,1,1)))
        else:
            return self.atoms.calc.asi.ham_storage.get((estat_idx,1,1))

    @property
    def hamiltonian_kinetic(self):
        kin_idx = 1
        if self.truncate:
            return self.truncated_mat_to_full(self.atoms.calc.asi.ham_storage.get((kin_idx,1,1)))
        else:
            return self.atoms.calc.asi.ham_storage.get((kin_idx,1,1))

    @property
    def fock_embedding_matrix(self):
        """The Fock embedding matrix of the environment

        Represents the Fock embedding matrix used to level-shift/orthogonalise
        the subsystem orbitals of the environment from the active system:
            (1) F^{A-in-B} = h^{core} + g^{hilev}[\gamma^{A}]
                                + v_{emb}[\gamma^{A}, \gamma^{B}] + P_{B}[1]
        where \gamma^{A} is the density matrix for the subystem, A},g[\gamma]
        are the two-electron interaction terms, is the embedding potential matrix,
            (2) v_{emb} = g^{low}[\gamma^{A} + \gamma^{B}] - g^{low}[\gamma^{A}]
        and h_core are the one-electron components of the hamiltonian (kinetic
        energy and nuclear-electron interactions).

        NOTE: For FHI-aims, (1) is not formally constructed fully on the
        wrapper level. Numerical stability in FHI-aims requires that the
        onsite potential per atom exactly cancel, precluding the clean
        separation of the nuclear-electron interactions from the total
        electrostatic matrix. As such, v_{emb} is constructed
        from all potential components.

        Formally, this is exactly the same when the embedded Fock matrix is finally
        constructed in FHI-aims  for the high-level calculation - components of
        F^{A-in-B} are calculated in this function are added to the Hamiltonian
        of FHI-aims before its entry into the eigensolver. As such, removing components of
        the nuclear-potential between atoms of A (included in g^{low}[\gamma^{A}])
        makes perfect sense, as they are are calculated natively within FHI-aims.
        For similar reasons, the kinetic energy components of h^{core} may be ignored.

        The final term calculated in the wrapper is then:
            (2) F_{wrapper}^{A-in-B} = H_{emb}^{Tot, lolev}[\gamma^{A} + \gamma^{B}]
             - H_{emb}^{Tot, lolev}[\gamma^{A}] - t_k(\gamma^{A} + \gamma^{B}}
                                  - t_k(\gamma^{A}) + P_{B}
        Where t_k is the kinetic energy contribution to the Hamiltonian and
        H_{emb}^{Tot, lolev}[\gamma] is the total hamiltonian derived from the
        density matrix, gamma at the low-level reference level of thoery.


            [1] Lee, S. et al., Acc. Chem. Res. 2019, 52 (5), 1359–1368.

        Parameters
        ----------
        vemb, np.ndarray: 
            Embedding potential of environment at calculation at
            low level of theory (ie., first four terms of equation (2)). 
        projection_matrix, np.ndarray: 
            Projection matrix to level shift P_B components of the environment
            upwards relative to the active subsystem (nbasis,nbasis)
        
        Returns
        -------
        fock_embedding_matrix, np.ndarray: fock_embedding_matrix(nbasis,nbasis)
    """
        return self._fock_embedding_matrix

    @fock_embedding_matrix.setter
    def fock_embedding_matrix(self, inp_fock_embedding_mat):

        #if (not ( isinstance(inp_fock_embedding_mat, (np.ndarray)) or 
        #    (inp_fock_embedding_mat is None) or
        #    (type(inp_fock_embedding_mat) == NPScal))):

        #    raise TypeError("Input vemb needs to be np.ndarray of dimensions nbasis*nbasis.")

        if ((inp_fock_embedding_mat is not None) and (self.truncate)):
            inp_fock_embedding_mat = self.full_mat_to_truncated(inp_fock_embedding_mat)

        self._fock_embedding_matrix = inp_fock_embedding_mat

    @property
    def density_matrix_in(self):
        """Input density matrix

            Defines the density matrix used as an input to construct the density
            upon the initialisation of a given calulation

        Returns
        -------
            np.ndarray: with dimensions (nbasis,nbasis)
        """
        return self._density_matrix_in

    @density_matrix_in.setter
    def density_matrix_in(self, densmat):

        #if (not isinstance(densmat, (list, tuple, np.ndarray)) and
        #    (not (densmat is None))):
            
        #    raise TypeError("Input needs to be np.ndarray of dimensions nbasis*nbasis.")

        # TODO: DIMENSION CHECKING
        if ((densmat is not None) and (self.truncate)):
            densmat = self.full_mat_to_truncated(densmat)

        self._density_matrix_in = densmat

    @property
    def density_matrices_out(self):
        """Output density matrix
        
        Returns a list of all density matrices within the dictionary,
        self.atoms.calc.asi.dm_storage, which stores all the matrices
        return from the calculation via ASI Callbacks.
        
        Returns
        -------
        out_mats: list of np.ndarrays
        """

        try:
            num_densmat = self.atoms.calc.asi.dm_count
        except:
            raise NameError("dm_count = 0: No density matrices stored!")

        out_mats = [ self.atoms.calc.asi.dm_storage.get((dm_num+1,1,1)) \
                     for dm_num in range(num_densmat) ]

        if self.truncate:
            for idx, trunc_mat in enumerate(out_mats):
                out_mats[idx] = self.truncated_mat_to_full(trunc_mat)

        return out_mats

    @property
    def overlap(self):
        """Overlap matrix of nbasisxnbasis

        """
        return self.atoms.calc.asi.overlap_storage.get((1,1))

    @property
    def basis_atoms(self):
        """Index map of basis functions to atoms

        """
        return self._basis_atoms

    @basis_atoms.setter
    def basis_atoms(self, val):
        self._basis_atoms = val

    @property
    def n_basis(self):
        """Number of basis functions

        """
        return self._n_basis

    @n_basis.setter
    def n_basis(self, val):
        self._n_basis = val

    @property
    def truncate(self):
        """Logical defining whether truncation is invoked.

        """
        return self._truncate

    @truncate.setter
    def truncate(self, val):
        self._truncate = val

    @property
    def basis_info(self):
        """Basisinfo object defining indices of basis elements
        
        """
        return self._basis_info

    @basis_info.setter
    def basis_info(self, val):
        self._basis_info = val

    @property
    def free_atom_nelectrons(self):
        
        tot_nelec = np.sum(self.atoms.numbers)
        ghost_nelec = np.sum(self.atoms.numbers[self.ghost_list])

        return tot_nelec - ghost_nelec

    @property
    def input_total_charge(self):
        return self._input_total_charge

    @input_total_charge.setter
    def input_total_charge(self, val):
        self._input_total_charge = val

    @property
    def input_fragment_nelectrons(self):
        return self._input_fragment_nelectrons

    @input_fragment_nelectrons.setter
    def input_fragment_nelectrons(self, val):
        self._input_fragment_nelectrons = val

    @property
    def fragment_total_charge(self):
        return +(self.input_fragment_nelectrons - self.free_atom_nelectrons)
