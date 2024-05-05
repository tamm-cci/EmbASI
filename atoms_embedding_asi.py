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
        We must re-order the atoms
        :return:
        """

        import numpy as np

        "Check if embedding mask is in the correct order (e.g., [1,1,1,2,2,2])"
        "Ensure the next value is always ge than the current"
        idx_list = np.argsort(self.embed_mask)
        print(f"EMBED {self.embed_mask}")
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

    def run(self, load_dm=None, load_ham=None, load_s=None):
        """Actually performed a given simulation run for the calcualtor.
            Must be separated for indidividual system calls."""
        import os
        from mpi4py import MPI
        from asi4py.asecalc import ASI_ASE_calculator

        self.atoms.calc = ASI_ASE_calculator(os.environ['ASI_LIB_PATH'],
                                        self.calc_initializer,
                                        MPI.COMM_WORLD,
                                        self.atoms,
                                        work_dir=self.outdir)

        self.atoms.calc.asi.keep_hamiltonian = True
        self.atoms.calc.asi.keep_overlap = True
        self.atoms.calc.asi.keep_density_matrix = True

        self.atoms.calc.asi.dm_storage = {}
        self.atoms.calc.asi.dm_calc_cnt = {}
        self.atoms.calc.asi.dm_count = 0
        self.atoms.calc.asi.register_dm_callback(dm_saving_callback, (self.atoms.calc.asi, self.atoms.calc.asi.dm_storage, self.atoms.calc.asi.dm_calc_cnt, 'DM calc'))

        self.atoms.calc.asi.ham_storage = {}
        self.atoms.calc.asi.ham_calc_cnt = {}
        self.atoms.calc.asi.ham_count = 0
        self.atoms.calc.asi.register_hamiltonian_callback(ham_saving_callback, (self.atoms.calc.asi, self.atoms.calc.asi.ham_storage, self.atoms.calc.asi.ham_calc_cnt, 'Ha,m calc'))

        if load_dm is not None:
            'TODO: Actual type enforcemenet and error handling'
            self.atoms.calc.asi.init_density_matrix = {(1,1): load_dm}
        if load_ham is not None:
            raise Exception("Loading of hamiltonian matrix unavailable in ASI")
            self.atoms.calc.asi.init_ham = load_ham
        if load_s is not None:
            raise Exception("Loading of overlap matrix unavailable in ASI.")
            self.atoms.calc.asi.init_s = load_s

        E0 = self.atoms.get_potential_energy()

        print(f'E0={E0:.6f}')
        self.atoms.calc.asi.close()
