from asiembedding.parallel_utils import root_print

class Basis_info:
    """_summary_
    Contains the basis dimension information, which acts as a permanent store
    basis information which may dynamically change for each AtomsEmbed object
    """
    def __init__(self):

        root_print("Initialising basis info...")

    @property
    def full_natoms(self):
        return self._full_natoms

    @full_natoms.setter
    def full_natoms(self, val):
        self._full_natoms = val

    @property
    def trunc_natoms(self):
        return self._trunc_natoms

    @trunc_natoms.setter
    def trunc_natoms(self, val):
        self._trunc_natoms = val

    @property
    def active_atoms(self):
        return self._active_atoms

    @active_atoms.setter
    def active_atoms(self, val):
        self._active_atoms = val

    @property
    def full_nbasis(self):
        return self._full_nbasis

    @full_nbasis.setter
    def full_nbasis(self, val):
        self._full_nbasis = val

    @property
    def trunc_nbasis(self):
        return self._trunc_nbasis

    @trunc_nbasis.setter
    def trunc_nbasis(self, val):
        self._trunc_nbasis = val

    @property
    def full_basis_atoms(self):
        return self._full_basis_atoms

    @full_basis_atoms.setter
    def full_basis_atoms(self, val):
        self._full_basis_atoms = val

    @property
    def trunc_basis_atoms(self):
        return self._trunc_basis_atoms

    @trunc_basis_atoms.setter
    def trunc_basis_atoms(self, val):
        self._trunc_basis_atoms = val

    def set_basis_atom_indexes(self):
        """_summary_
        Sets the start and end index of the matrices with nbasis*nbasis or
        trunc_nbasis*trunc_nbasis dimensions for each atom block.
        """
        import numpy as np

        self.full_basis_max_idx = []
        self.full_basis_min_idx = []
        for atom in range(self.full_natoms):
            self.full_basis_max_idx.append(np.max(np.where(self.full_basis_atoms==atom))+1)
            self.full_basis_min_idx.append(np.min(np.where(self.full_basis_atoms==atom)))

        self.trunc_basis_max_idx = []
        self.trunc_basis_min_idx = []
        for atom in self.active_atoms:
            self.trunc_basis_max_idx.append(np.max(np.where(self.trunc_basis_atoms==atom))+1)
            self.trunc_basis_min_idx.append(np.min(np.where(self.trunc_basis_atoms==atom)))
