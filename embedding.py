# ~ Overall Embedding object
from abc import ABC, abstractmethod

class EmbeddingBase(ABC): 

    def __init__(self, atoms, embed_mask, calc_base_ll=None, calc_base_hl=None):
        import os

        self.asi_lib_path = os.environ['ASI_LIB_PATH']
        self.embed_mask = embed_mask
        self.calculator_ll = calc_base_ll
        self.calculator_hl = calc_base_hl

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
    def __init__(self, atoms, embed_mask, calc_base_ll, calc_base_hl, post_scf=None, mu_val=1e+06):
        from copy import copy, deepcopy

        'LL - low-level, HL - high-level'
        self.calc_names = ["AB_LL","A_LL","A_HL","A_HL_PP"]

        super(ProjectionEmbedding, self).__init__(atoms, embed_mask, calc_base_ll, calc_base_hl)
        low_level_calculator_1 = deepcopy(self.calculator_ll)
        low_level_calculator_2 = deepcopy(self.calculator_ll)
        high_level_calculator_1 = deepcopy(self.calculator_hl)
        high_level_calculator_2 = deepcopy(self.calculator_hl)

        low_level_calculator_1.set(qm_embedding_calc = 1)
        self.set_layer(atoms, self.calc_names[0], low_level_calculator_1, embed_mask, ghosts=0, no_scf=False)
        low_level_calculator_2.set(qm_embedding_calc = 2)
#        low_level_calculator_2.set(charge = -1)
        self.set_layer(atoms, self.calc_names[1], low_level_calculator_2, embed_mask, ghosts=2, no_scf=False)
        high_level_calculator_1.set(qm_embedding_calc = 3)
#        high_level_calculator_1.set(charge = -1)
        self.set_layer(atoms, self.calc_names[2], high_level_calculator_1, embed_mask, ghosts=2, no_scf=False)
        high_level_calculator_2.set(qm_embedding_calc = 2)
#        high_level_calculator_2.set(charge = -1)
        self.set_layer(atoms, self.calc_names[3], high_level_calculator_2, embed_mask, ghosts=2, no_scf=False)

        self.mu_val = mu_val

    @property
    def nlayers(self):
        return self._nlayers

    @nlayers.setter
    def nlayers(self, val):
        
        assert val == 2, \
                "Only two layers currently valid for projection embedding."
        return self._nlayers

    def calculate_projector(self):

        self.P_b = self.AB_S @ self.B_dm @ self.AB_S

    def run(self):
        import numpy as np

        print("I am running!")

        self.AB_LL.run()
        core_idx = self.AB_LL.atoms.calc.asi.ham_count - 1
        tot_idx = self.AB_LL.atoms.calc.asi.ham_count

        self.A_dm = self.AB_LL.atoms.calc.asi.dm_storage.get((1,1,1))
        self.B_dm = self.AB_LL.atoms.calc.asi.dm_storage.get((2,1,1))

        self.AB_S = self.AB_LL.atoms.calc.asi.overlap_storage[1,1]

        self.AB_Hcore = self.AB_LL.atoms.calc.asi.ham_storage.get((core_idx,1,1))
        self.AB_Htot = self.AB_LL.atoms.calc.asi.ham_storage.get((tot_idx,1,1))
        self.AB_Hee = self.AB_Htot - self.AB_Hcore

        self.A_LL.run(load_dm = np.asfortranarray(self.A_dm))
        core_idx = self.A_LL.atoms.calc.asi.ham_count - 1
        tot_idx = self.A_LL.atoms.calc.asi.ham_count

        self.A_Hcore = self.A_LL.atoms.calc.asi.ham_storage.get((core_idx,1,1))
        self.A_Htot = self.A_LL.atoms.calc.asi.ham_storage.get((tot_idx,1,1))
        self.A_Hee = self.A_Htot - self.A_Hcore

        self.AB_pop = np.trace(self.AB_S @ (self.A_dm+self.B_dm))
        self.A_pop = np.trace(self.AB_S @ (self.A_dm))
        self.B_pop = np.trace(self.AB_S @ (self.B_dm))

        self.calculate_projector()
        self.emb_mat = np.asfortranarray(self.AB_Hee - self.A_Hee + (self.AB_Hcore - self.A_Hcore) + (self.mu_val * self.P_b) )
#        self.emb_mat = np.asfortranarray(self.AB_Hee - self.A_Hee + self.AB_Hcore + (self.mu_val * self.P_b) )
        self.emb_raw = self.AB_Hee - self.A_Hee

        self.A_HL.run(load_ham=self.emb_mat, load_dm=np.asfortranarray(self.A_dm))
        self.A_HL_dm = self.A_HL.atoms.calc.asi.dm_storage.get((1,1,1))

        self.A_HL_PP.run(load_dm=np.asfortranarray(self.A_HL_dm))
        core_idx = self.A_HL_PP.atoms.calc.asi.ham_count - 1
        tot_idx = self.A_HL_PP.atoms.calc.asi.ham_count
        self.A_HL_Hcore = self.A_HL_PP.atoms.calc.asi.ham_storage.get((core_idx,1,1))
        self.A_HL_Htot = self.A_HL_PP.atoms.calc.asi.ham_storage.get((tot_idx,1,1))
        self.A_HL_Hee = self.A_HL_Htot - self.A_HL_Hcore
        
        self.EE_corr = (np.trace((self.mu_val * self.P_b) @ self.A_HL_dm) * 27.2114079527)
        self.PB_corr = (np.trace((self.emb_raw @ (self.A_HL_dm - self.A_dm))) * 27.2114079527)

        print( np.trace((self.mu_val * self.P_b) @ self.A_HL_dm) * 27.2114079527 )
        print( np.einsum('ij,ij', self.emb_raw, (self.A_HL_dm - self.A_dm)) * 27.2114079527)




