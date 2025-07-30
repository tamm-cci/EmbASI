from ase.calculators.aims import Aims, AimsProfile

try:
    from embasi import ProjectionEmbedding
except (ModuleNotFoundError, ValueError):
    ProjectionEmbedding = None


class ChemShellInterface(ProjectionEmbedding):

    def __init__(self, atoms, embed_mask, mu_val=1.e+6, ll_calc_config=None, hl_calc_config=None, multipole_atoms=None, **kwargs):

        self.embed_mask = embed_mask
        self.mu_val = mu_val
        self.multipole_atoms = MultipoleAtoms()

        self.ll_calc_config = {'xc':'B3LYP',
        'profile': AimsProfile(command='asi-doesnt-need-command'),
        'KS_method': 'parallel',
        'RI_method': "LVL",
        'collect_eigenvectors': True,
        'density_update_method': 'density_matrix',
        'atomic_solver_xc': "PBE",
        'compute_kinetic': True,
        'override_initial_charge_check': True,}

        self.hl_calc_config = {'xc':'B3LYP',
        'profile': AimsProfile(command='asi-doesnt-need-command'),
        'KS_method': 'parallel',
        'RI_method': "LVL",
        'collect_eigenvectors': True,
        'density_update_method': 'density_matrix',
        'atomic_solver_xc': "PBE",
        'compute_kinetic': True,
        'override_initial_charge_check': True,}

        self.calc_ll = Aims(**self.ll_calc_config)
        self.calc_hl = Aims(**self.hl_calc_config)

        super().__init__(atoms=atoms, embed_mask=embed_mask,
            calc_base_ll=self.calc_ll, calc_base_hl=self.calc_hl, mu_val=mu_val)

    def write_multipoles(self):
        self.multipole_atoms.write_to_geometry_in()

    def add_atoms(self, coord, charge, multipole_order=0):
        self.multipole_atoms.add_atoms(coord, charge, multipole_order)

class MultipoleAtoms():

    def __init__(self):
        self._coords = []  
        self._charges = []  
        self._multipole_order = 0  
    
    def add_atoms(self, coord, charge, multipole_order=0):
        self._coords.append(coord)
        self._charges.append(charge)
        self._multipole_order.append(multipole_order)
    
    @property
    def coords(self):
        return self._coords
    
    @property
    def charges(self):
        return self._charges
    
    def write_to_geometry_in(self):
        import os 
        cwd = os.getcwd()
        geometry_path = os.path.join(cwd, "geometry.in")
        with open(geometry_path, 'a') as f:
            for coord, charge in zip(self._coords, self._charges):
                x, y, z = coord
                f.write(f"multipole {x:16.9f} {y:16.9f} {z:16.9f} {self._multipole_order:2d} {charge:12.8f}\n")
