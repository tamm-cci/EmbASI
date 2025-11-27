class GeomUtils():

    def __init__(self, atoms):

        self.atoms = atoms
        self.get_conn_mat()

    def get_conn_mat(self):

        from ase.neighborlist import natural_cutoffs    
        from ase.neighborlist import get_connectivity_matrix
        from ase.neighborlist import NeighborList

        cutoffs = natural_cutoffs(self.atoms)
        self.nl = NeighborList(cutoffs, self_interaction=False,
                                      bothways=True)
        self.nl.update(self.atoms)

        self.conn_mat = self.nl.get_connectivity_matrix()

    def quantum_cap_region(self, embed_mask, qm_cap_species="H", qm_cap_len=1.0):

        from scipy import sparse

        from ase.build import molecule
        from ase.neighborlist import get_connectivity_matrix
        from ase.neighborlist import natural_cutoffs
        from ase.neighborlist import NeighborList
        from ase import Atoms, Atom
        import numpy as np

        nz_row = self.conn_mat.nonzero()[0]
        nz_col = self.conn_mat.nonzero()[1]

        intersect = []
        for row, col in zip(nz_row, nz_col):
            if embed_mask[row] != embed_mask[col]:
                indices = [int(row), int(col)]
                indices.sort()
                intersect.append(tuple(indices))

        intersect = list(set(intersect))

        species_link = qm_cap_species
        link_len = 1.0

        embed_mask = np.array(embed_mask)
        mask = np.where(embed_mask == 1)[0]

        reduced_atoms = self.atoms[mask]

        for link in intersect:
            pos_1 = self.atoms.positions[link[0]]
            pos_2 = self.atoms.positions[link[1]]
            
            vect = pos_2 - pos_1
            norm_vect = vect / np.linalg.norm(vect)

            if embed_mask[link[0]] == 2:
                link_pos = pos_2 - (link_len * norm_vect)
            else:
                link_pos = pos_1 + (link_len * norm_vect)
            
            link_atom = Atom(species_link, position=link_pos)

            reduced_atoms = reduced_atoms + link_atom
    
        return reduced_atoms
