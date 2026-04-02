from abc import ABC, abstractmethod

# Valid ASI Flavours
ASI_flavour = {"Dummy": 0, "FHIaims": 1, "DFTB+": 2}

# Implemented calculators
implemented_calculators = ["Aims"]

class ase_calc_parameter_setter(ABC):

    def __init__(self):
        pass

    @property
    @abstractmethod
    def asi_flavour(self):
        return ASI_flavour["Dummy"]

    @abstractmethod
    def set_ghost_atoms(self, calc, atoms, ghost_list):
        """Set calculator parameters for setting atoms of a given index as ghost atoms

        Changes the ASE Calculator keywords to accept the keywords to
        set certain keywords as ghost basis centers.

        Parameters
        ----------
        calc : ASECalculator
            ASE calculator to add parameters.
        ghost_list : list, len(atoms)
            A list of booleans to mark which atoms are ghost species.
        """
        pass

    @abstractmethod
    def set_postscf_keyword(self, calc, postscf_method):
        """Set calculator parameters for setting the keyword for postscf methods

        Changes the ASE Calculator keywords to accept the keywords to
        the post-scf methods (used for the calling pattern of post-HF)
        methods in FHI-aims

        Parameters
        ----------
        calc : ASECalculator
            ASE calculator to add parameters.
        postscf_method : None or str
            Post-SCF method ran on calculator execution.
        """
        pass

    @abstractmethod
    def set_scalapack_blocksize(self, calc, blacs_blocksize):
        """Set calculator parameters for setting the ScaLAPACK block size

        Sets the default block size to align the internal NPScal implementation

        Parameters
        ----------
        calc : ASECalculator
            ASE calculator to add parameters.
        blacs_blocksize : int
            BLACS blocksize
        """
        pass

    @abstractmethod
    def set_embasi_calculation_type(self, calc, calc_type):
        """Set calculator parameters for setting the EmbASI calculation control
           flow keywords

        Sets the default block size to align with the control flow keywords in
        a given QM code implementation

        Parameters
        ----------
        calc : ASECalculator
            ASE calculator to add parameters.
        calc_type : str
            Accepts - 'ks_fullscf', 'total_energy_only', 'full_scf_embed', and 'frozendensity'.
        """
        pass

    @abstractmethod
    def set_noscf_calc_params(self, calc):
        """Set calculator parameters for running a total energy only calculation

        Changes the ASE Calculator keywords to accept the keywords to
        skip the SCF cycle and only output the total energy.

        Parameters
        ----------
        calc : ASECalculator
            ASE calculator to add parameters
        """
        pass

    @abstractmethod
    def override_basis_order(self, calc):
        """Set calculator parameters for overriding basis ordering

        Sets the ASE calculator to ensure that AO basis functions are ordered
        consistently between calls.

        Parameters
        ----------
        calc : ASECalculator
            ASE calculator to add parameters.
        """
        pass

    @abstractmethod
    def set_qm_localise(self, calc):
        """Set calculator parameters for performing density matrix localisation

        Sets the ASE calculator keywords to perform a density matrix optimisation
        and output the density matrix of the embedded and environment subsystems.

        Parameters
        ----------
        calc : ASECalculator
            ASE calculator to add parameters.
        """
        pass

    @abstractmethod
    def set_qm_total_charge(self, calc, charge):
        """Set calculator parameters for specifying input total charge

        Parameters
        ----------
        calc : ASECalculator
            ASE calculator to add parameters.
        charge : int
            Total charge of the system
        """
        pass


    def set_full_scf_calc(self, calc):
        calc = self.set_embasi_calculation_type(calc, "ks_fullscf")

        return calc

    def set_no_scf_total_energy_calc(self, calc):
        calc = self.set_embasi_calculation_type(calc, "total_energy_only")

        return calc

    def set_full_scf_embed_calc(self, calc):
        calc = self.set_embasi_calculation_type(calc, "full_scf_embed")

        return calc

    def set_frozendensity_calc(self, calc):
        calc = self.set_embasi_calculation_type(calc, "frozendensity")

        return calc



class Aims_param_setter(ase_calc_parameter_setter):

    def __init__(self):
        super(Aims_param_setter, self).__init__()

    @property
    def asi_flavour(self):
        return ASI_flavour["FHIaims"]

    def set_noscf_calc_params(self, calc):
        calc.parameters['charge_mix_param'] = 0.
        calc.parameters['sc_iter_limit'] = 0
        self.set_embasi_calculation_type(calc, "total_energy_only")

        return calc

    def set_ghost_atoms(self, calc, atoms, ghost_list):
        calc.parameters["ghosts"] = ghost_list
        return calc

    def set_postscf_keyword(self, calc, postscf_method):
        calc.parameters["total_energy_method"] = postscf_method
        return calc

    def set_scalapack_blocksize(self, calc, scalapack_blocksize):
        calc.parameters["scalapack_block_size"] = scalapack_blocksize
        return calc

    def set_embasi_calculation_type(self, calc, calc_type):
        if calc_type == "ks_fullscf":
            calc.parameters["qm_embedding_calc"] = 1
        elif calc_type == "total_energy_only":
            calc.parameters["qm_embedding_calc"] = 2
        elif calc_type == "full_scf_embed":
            calc.parameters["qm_embedding_calc"] = 3
        elif calc_type == "frozendensity-write":
            calc.parameters["qm_embedding_type"] = "frozendensity write"
        elif calc_type == "frozendensity-readandwrite":
            calc.parameters["qm_embedding_type"] = "frozendensity read_and_write"
        else:
            raise Exception("Only 'ks_fullscf', 'total_energy_only', 'full_scf_embed', and 'frozendensity' supported.")

        return calc

    def override_basis_order(self, calc):
        calc.parameters['override_default_empty_basis_order']=".true."

        return calc

    def set_qm_localise(self, calc):
        calc.parameters['qm_embedding_mo_localise']=".true."

        return calc

    def set_qm_total_charge(self, calc, charge):
        calc.parameters['charge'] = charge
        return calc



def ase_calc_parameter_setter(calc):
    """Gets the ASE Calculator setting class for a given ASE Calculator implementation

    Gets the name of the ASE calculator implementation and returns the
    object needed to set keywords through the EmbASI calculation

    Parameters
    ----------
    calc : ASEIOCalculator

    Raises
    ------
    Exception
        ASE Calulator setter not found

    """

    if type(calc).__name__ not in implemented_calculators:
        raise Exception("ASE Calculator not implemented.")

    if type(calc).__name__ == "Aims":
        ASE_parameter_setter = Aims_param_setter()
    else:
        raise Exception("ASE Calculator setter not found.")

    return ASE_parameter_setter
