from ase.calculators.aims import Aims, AimsProfile
from ase import Atoms
import numpy as np

try:
    from embasi import ProjectionEmbedding
except (ModuleNotFoundError, ValueError):
    ProjectionEmbedding = None

class ChemShellInterface(ProjectionEmbedding):
    '''
    A class connecting the ChemShell interface to EmbASI's ProjectionEmbedding class.
    
    Parameters
    ----------
    embed_mask: list[int].
        Region assignment for the low level QM (1), and high level QM (2) for 
        each atom. (i.e., [1,1,1,2,2,2])
    mu_val: float
    ll_calc_config: dict, optional.
        Custom ASE.Aims configuration for low level QM region.
    hl_calc_config: dict, optional.
        Custom ASE.Aims configuration for high level QM region.
    charge: float, optional.
        Total system charge.
    multipoles: MultipoleAtoms, optional.
        Multipole representation for environment electrostatics.
    '''

    def __init__(self, embed_mask, mu_val=1.e+6, ll_calc_config=None, hl_calc_config=None, 
                charge=None, multipoles=None, **kwargs):

        self.embed_mask = embed_mask
        self.mu_val = mu_val
        self.charge = charge
    
        # Default FHI-aims configs for LL calculation
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

        # Default FHI-aims configs for HL calculation
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

        # Overrides custom configurations for LL and HL
        if ll_calc_config is not None:
            for key, val in ll_calc_config.items():
                self.ll_calc_config[key] = val
        if hl_calc_config is not None:
            for key, val in hl_calc_config.items():
                self.hl_calc_config[key] = val

        # Create ASE calculators
        self.calc_ll = Aims(**self.ll_calc_config) 
        self.calc_hl = Aims(**self.hl_calc_config) 

        # Multipole container for electrostatic embedding
        self._multipoles = MultipoleAtoms()

        # Initialise ProjectionEmbedding superclass
        super().__init__(embed_mask=self.embed_mask, calc_base_ll=self.calc_ll,
                        calc_base_hl = self.calc_hl, mu_val=mu_val, 
                        total_charge=self.charge, **kwargs)

    @property
    def multipoles(self):
        ''' Return stored multipole data '''
        return self._multipoles

    @multipoles.setter
    def multipoles(self, val):
        ''' Set multipole data '''
        self._multipoles = val
        
class MultipoleAtoms():
    '''
    Class containing atomic multipole data.
    
    Stores each multipole as a dictionary with:
        coord: np.ndarray, Cartesian coordinates (Å)
        charge: float
        order: int, multipole order
            (0=charge, 1=dipole, etc.)
        moments: list, optional high order moments
    '''

    def __init__(self, symbols=None, positions=None, **kwargs):
        self._multipoles = []
        self.coordinates = []
        self.charge = []

    def add_multipole(self, coord, charge, order=0, moments=None):
        '''
        Add a multipole to list, to be written into the geometry.in file.

        Parameters
        ----------
        coord: list[float]
            Cartesian coordinates (Å).
        charge: float
        order: int
        moments: list[float], optional
            Higher order moments for multipole expansion 
        '''
        self._multipoles.append({'coord': np.array(coord), 'charge': float(charge),
                                'order': int(order), 'moments': list(moments) if moments else[]})
        self.coordinates.append(coord)
        self.charge.append(charge)
        return self._multipoles