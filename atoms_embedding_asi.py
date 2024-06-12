from asi4py.asecalc import ASI_ASE_calculator
from ctypes import cdll, CDLL, RTLD_GLOBAL
from ctypes import POINTER, byref, c_int, c_int64, c_int32, c_bool, c_char_p, c_double, c_void_p, CFUNCTYPE, py_object, cast, byref
import ctypes

def dm_saving_callback(aux, iK, iS, descr, data):
    asi, storage_dict, cnt_dict, label = cast(aux, py_object).value
    asi.dm_count +=1
    #try:
    data = asi.scalapack.gather_numpy(descr, data, (asi.n_basis, asi.n_basis))
    if data is not None:
        storage_dict[(asi.dm_count, iK, iS)] = data.copy()
    #    if asi.implementation == "DFTB+":
    #        ltriang2herm_inplace(storage_dict[(iK, iS)])
    #        cnt_dict[(iK, iS)] = cnt_dict.get((iK, iS), 0) + 1
    #except Exception as eee:
    #    print(f"Something happened in ASI default_saving_callback {label}: {eee}\nAborting...")
    #    MPI.COMM_WORLD.Abort(1)

def ham_saving_callback(aux, iK, iS, descr, data):
    asi, storage_dict, cnt_dict, label = cast(aux, py_object).value
    asi.ham_count +=1
    #try:
    data = asi.scalapack.gather_numpy(descr, data, (asi.n_basis, asi.n_basis))
    if data is not None:
        storage_dict[(asi.ham_count, iK, iS)] = data.copy()
    #    if asi.implementation == "DFTB+":
    #        ltriang2herm_inplace(storage_dict[(iK, iS)])
    #        cnt_dict[(iK, iS)] = cnt_dict.get((iK, iS), 0) + 1
    #except Exception as eee:
    #    print(f"Something happened in ASI default_saving_callback {label}: {eee}\nAborting...")
    #    MPI.COMM_WORLD.Abort(1)


class AtomsEmbed():

    def __init__(self, atoms, initial_calc, embed_mask, no_scf=False, ghosts=0, outdir='asi.calc'):
        self.atoms = atoms
        self.initial_embed_mask = embed_mask
        "Sets which layer/layers are set to be ghost atoms"
        self.outdir = outdir

        if isinstance(embed_mask, int):
            "We hope the user knows what they are doing and" \
            "their atoms object is ordered accordingly"
            self.embed_mask = [1]*embed_mask
            self.embed_mask += [2]*(len(atoms)-embed_mask)

        if isinstance(embed_mask, list):
            self.embed_mask = embed_mask
            assert len(atoms)==len(embed_mask), \
                "Length of embedding mask does not match number of atoms"

        self.initial_calc = initial_calc
        self.reorder_atoms_from_embed_mask()
        self.atoms.info['embedding_mask'] = self.embed_mask

        self.no_scf = no_scf

        if isinstance(ghosts, int):
            ghosts = [ghosts]
        self.ghost_list = [(at in [ghosts]) for at in self.embed_mask]

    def calc_initializer(self, asi):

        calc = self.initial_calc
        if self.no_scf:
            calc.set(sc_iter_limit=0)

        calc.write_input(asi.atoms, ghosts=self.ghost_list)
        self._insert_embedding_region_aims()

    def reorder_atoms_from_embed_mask(self):
        """
        Re-orders atoms to push those in embedding region 1 to the beginning
        :return:
        """

        import numpy as np

        "Check if embedding mask is in the correct order (e.g., [1,1,1,2,2,2])"
        "Ensure the next value is always ge than the current"
        idx_list = np.argsort(self.embed_mask)
        sort_embed_mask = np.sort(self.embed_mask)

        self.embed_mask = sort_embed_mask
        self.atoms = self.atoms[idx_list]

    def _insert_embedding_region_aims(self):
        """Lazy way of placing embedding regions in input file"""
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

    def extract_results(self):
        """
        Extracts results from the DFT code output file that are otherwise unavailable
        within the ASE framework. This may need a separate module if other calculators are
        implemented.
        """

        with open('./'+self.outdir+'/asi.log') as output:
            
            lines = output.readlines()
            for line in lines:
                outline = line.split()

                if '  | Kinetic energy                :' in line:
                    self.kinetic_energy = float(outline[6])

                if '  | Electrostatic energy          :' in line:
                    self.es_energy = float(outline[6])

                if '  | Sum of eigenvalues            :' in line:
                    self.ev_sum = float(outline[7])



    def run(self, load_dm=None, load_ham=None, load_s=None, ev_corr_scf=False):
        """Actually performed a given simulation run for the calculator.
            Must be separated for indidividual system calls."""
        import os
        import numpy as np
        from mpi4py import MPI
        from asi4py.asecalc import ASI_ASE_calculator

        print(f'Calculation {self.outdir}...')

        self.atoms.calc = ASI_ASE_calculator(os.environ['ASI_LIB_PATH'],
                                        self.calc_initializer,
                                        MPI.COMM_WORLD,
                                        self.atoms,
                                        work_dir=self.outdir)

        #self.atoms.calc.asi.keep_hamiltonian = True
        self.atoms.calc.asi.keep_overlap = True
        #self.atoms.calc.asi.keep_density_matrix = True

        self.atoms.calc.asi.dm_storage = {}
        self.atoms.calc.asi.dm_calc_cnt = {}
        self.atoms.calc.asi.dm_count = 0
        self.atoms.calc.asi.register_dm_callback(dm_saving_callback, (self.atoms.calc.asi, self.atoms.calc.asi.dm_storage, self.atoms.calc.asi.dm_calc_cnt, 'DM calc'))

        self.atoms.calc.asi.ham_storage = {}
        self.atoms.calc.asi.ham_calc_cnt = {}
        self.atoms.calc.asi.ham_count = 0
        self.atoms.calc.asi.register_hamiltonian_callback(ham_saving_callback, (self.atoms.calc.asi, self.atoms.calc.asi.ham_storage, self.atoms.calc.asi.ham_calc_cnt, 'Ham calc'))

        if load_dm is not None:
            'TODO: Actual type enforcement and error handling'
            self.atoms.calc.asi.init_density_matrix = {(1,1): load_dm}
        if load_ham is not None:
            self.atoms.calc.asi.init_hamiltonian = {(1,1): load_ham}
        if load_s is not None:
            raise Exception("Loading of overlap matrix unavailable in ASI.")
            self.atoms.calc.asi.init_s = load_s

        E0 = self.atoms.get_potential_energy()

        self.n_basis = self.atoms.calc.asi.n_basis
        self.basis_atoms = self.atoms.calc.asi.basis_atoms

        self.total_energy = E0
        self.extract_results()

        # Within the embedding workflow, we often want to calculate the total energy for a
        # given density matrix without performing any SCF steps. Often, this includes using
        # an input electron density constructed from a localised set of MOs for a fragment
        # of a supermolecule. This density will be far from the ground-state density for the fragment, 
        # meaning the output eigenvalues significantly deviate from those of a fully converged density.
        # As the vast majority of DFT codes with the KS-eigenvalues to determine the total
        # energy, the total energies due to the eigenvalues do not formally reflect the 
        # density matrix of the initial input for iteration, n=0:
        #
        #    \gamma^{n+1} * H^{total}[\gamma^{n}] \= \gamma^{n} * H^{total}[\gamma^{n}], 
        #
        # For TE-only calculations, we do not care about the SCF process - we are treating the
        # DFT code as a pure integrator of the XC and electrostatic energies. As such, we 
        # 'correct' the eigenvalue portion of the total energy to reflect the interaction
        # of the input density matrix, as opposed to the first set of KS-eigenvectors resulting
        # from the DFT code.
        if ev_corr_scf:
            tot_idx = self.atoms.calc.asi.ham_count
            ham = self.atoms.calc.asi.ham_storage.get((tot_idx,1,1))
            self.ev_corr_energy = 27.211384500 * np.trace(load_dm @ ham)

            self.ev_corr_total_energy = self.total_energy - self.ev_sum + self.ev_corr_energy
            print(self.ev_sum)
            print(self.ev_corr_energy)

        self.atoms.calc.asi.close()
