from ase.calculators.aims import Aims, AimsProfile

try:
    from embasi import ProjectionEmbedding
except (ModuleNotFoundError, ValueError):
    ProjectionEmbedding = None

class ChemShellInterface(ProjectionEmbedding):

    def __init__(self, atoms, embed_mask, mu_val=1.e+6, ll_calc_config=None, hl_calc_config=None, multipoles=None, **kwargs):

        self.embed_mask = embed_mask
        self.mu_val = mu_val
    
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

        self.calc_ll = Aims(**self.ll_calc_config) if ll_calc_config else Aims()
        self.calc_hl = Aims(**self.hl_calc_config) if hl_calc_config else Aims()

        self._multipoles = multipoles if multipoles else MultipoleAtoms()
        print(self._multipoles)
        print(self._multipoles.coords, self._multipoles.charges)
        super().__init__(atoms=atoms, embed_mask=embed_mask,
            calc_base_ll=self.calc_ll, calc_base_hl=self.calc_hl, mu_val=mu_val, **kwargs)

    @property
    def multipoles(self):
        return self._multipoles

    def write_multipoles(self):
        self._multipoles.write_to_geometry_in()

class MultipoleAtoms():

    def __init__(self):
        self._multipoles = []
        
    def add_multipole(self, coord, charge, multipole_order=0):
        if not isinstance(coord, (list, tuple)) or len(coord) != 3:
            raise ValueError("Coordinate must be 3-element list/tuple")
   
        self._multipoles.append({'coord': [float(x) for x in coord],
            'charge': float(charge),'multipole_order': int(order)})
    
    @property
    def coords(self):
        return [e['coord'] for e in self._multipoles]

    @coords.setter
    def coords(self, val):
        self._coords=val
    
    @property
    def charges(self):
        return [e['charge'] for e in self._multipoles]

    @charges.setter
    def charges(self, val):
        self._charges=val
    
    def write_to_geometry_in(self):
        if not self._multipoles:
            print("No multipoles to write")
            return

        from pathlib import Path
        base_dir = Path("EmbASI_calc") 
        calc_dirs = ['AB_LL', 'A_HL', 'A_HL_PP', 'A_LL']

        for calc_dir in calc_dirs:
            try:
                geometry_path = base_dir / calc_dir / "geometry.in"
                geometry_path.parent.mkdir(parents=True, exist_ok=True)
                if not geometry_path.exists():
                    print(f"Warning: {geometry_path} doesn't exist - creating new file")
                    with open(geometry_path, 'w') as f:
                        f.write("# Created by EmbASI interface\n")
            
                print(f"Appending to {geometry_path}")
                with open(geometry_path, 'a') as f:  
                    for mp in self._multipoles:
                        x, y, z = mp['coord']
                        f.write(
                        f"multipole {x:16.9f} {y:16.9f} {z:16.9f} "
                        f"{mp['order']:2d} {mp['charge']:12.8f}\n")
                        if mp.get('region'):
                            f.write(f"qm_embedding_region {mp['region']}\n")
            except Exception as e:
                print(f"Error writing to {geometry_path}: {str(e)}")