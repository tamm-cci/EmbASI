# ~ Overall Embedding object
from abc import ABC, abstractmethod

class EmbeddingBase(ABC): 

    def __init__(self, atoms, embed_mask, calc_base=None, scf_methods="pbe"):
        import os

        self.asi_lib_path = os.environ['ASI_LIB_PATH']
        self.embed_mask = embed_mask
        self.calculator = calc_base
        self.scf_methods = scf_methods

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

    def set_layer(self, atoms, layer_name, calc, embed_mask, ghosts=0, no_scf=False):
        "Initialises the AtomsEmbed methods for a given method"
        from .atoms_embedding_asi import AtomsEmbed

        layer = AtomsEmbed(atoms, calc, embed_mask, outdir=layer_name, ghosts=ghosts, no_scf=no_scf)
        setattr(self, layer_name, layer)

    @abstractmethod
    def run(self):
        pass

class ProjectionEmbedding(EmbeddingBase):
    def __init__(self, atoms, embed_mask, calc_base, scf_methods="pbe", mu_val=1e+06):
        from copy import copy, deepcopy

        'LL - low-level, HL - high-level'
        self.calc_names = ["AB_LL","A_LL","A_HL"]

        super(ProjectionEmbedding, self).__init__(atoms, embed_mask, calc_base, scf_methods=scf_methods)
        low_level_calculator_1 = deepcopy(self.calculator)
        low_level_calculator_2 = deepcopy(self.calculator)
        high_level_calculator = deepcopy(self.calculator)

        low_level_calculator_1.set(xc = self.scf_methods[0])
        low_level_calculator_1.set(xc = self.scf_methods[0])
        high_level_calculator.set(xc = self.scf_methods[1])

        low_level_calculator_1.set(qm_embedding_calc = 1)
        self.set_layer(atoms, self.calc_names[0], low_level_calculator_1, embed_mask, ghosts=0, no_scf=False)
        low_level_calculator_2.set(qm_embedding_calc = 2)
        self.set_layer(atoms, self.calc_names[1], low_level_calculator_2, embed_mask, ghosts=2, no_scf=True)
        high_level_calculator.set(qm_embedding_calc = 2)
        self.set_layer(atoms, self.calc_names[2], high_level_calculator, embed_mask, ghosts=2, no_scf=False)

        self.mu_val = mu_val

    @property
    def nlayers(self):
        return self._nlayers

    @nlayers.setter
    def nlayers(self, val):
        
        assert val == 2, \
                "Only two layers currently valid for projection embedding."
        return self._nlayers

    def run(self):
        print("I am running!")
