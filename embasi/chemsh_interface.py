from ase.calculators.aims import Aims, AimsProfile
from ase import Atoms
import numpy as np

try:
    from embasi import ProjectionEmbedding
except (ModuleNotFoundError, ValueError):
    ProjectionEmbedding = None

class ChemShellInterface(ProjectionEmbedding):

    def __init__(self, embed_mask, mu_val=1.e+6, ll_calc_config=None, hl_calc_config=None, 
                charge=None, multipoles=None, **kwargs):
        self.embed_mask = embed_mask
        self.mu_val = mu_val
        self.charge = charge
    
        self.ll_calc_config = {'xc':'B3LYP',
        'profile': AimsProfile(command='asi-doesnt-need-command'),
        'KS_method': 'parallel',
        'RI_method': "LVL",
        'collect_eigenvectors': True,
        'density_update_method': 'density_matrix',
        'atomic_solver_xc': "PBE",
        'compute_kinetic': True,
        'qmmm':True,
        'override_initial_charge_check': True,}

        self.hl_calc_config = {'xc':'B3LYP',
        'profile': AimsProfile(command='asi-doesnt-need-command'),
        'KS_method': 'parallel',
        'RI_method': "LVL",
        'collect_eigenvectors': True,
        'density_update_method': 'density_matrix',
        'atomic_solver_xc': "PBE",
        'compute_kinetic': True,
        'qmmm': True,
        'override_initial_charge_check': True,}

        if ll_calc_config is not None:
            for key, val in ll_calc_config.items():
                self.ll_calc_config[key] = val
                
        if hl_calc_config is not None:
            for key, val in hl_calc_config.items():
                self.hl_calc_config[key] = val

        self.calc_ll = Aims(**self.ll_calc_config) 
        self.calc_hl = Aims(**self.hl_calc_config) 
        self._multipoles = MultipoleAtoms()

        super().__init__(embed_mask=self.embed_mask, calc_base_ll=self.calc_ll,
                        calc_base_hl = self.calc_hl, mu_val=mu_val, 
                        total_charge=self.charge, **kwargs)

    @property
    def multipoles(self):
        return self._multipoles

    @multipoles.setter
    def multipoles(self, val):
        self._multipoles = val
        
class MultipoleAtoms():

    def __init__(self, symbols=None, positions=None, **kwargs):
        self._multipoles = []
        self.coordinates = []
        self.charge = []

    def add_multipole(self, coord, charge, order=0, moments=None):
        self._multipoles.append({'coord': np.array(coord), 'charge': float(charge),
                                'order': int(order), 'moments': list(moments) if moments else[]})
        self.coordinates.append(coord)
        self.charge.append(charge)
        return self._multipoles