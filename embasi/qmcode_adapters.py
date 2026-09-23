from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from mpi4py import MPI
from embasi.parallel_utils import root_print, mpi_bcast_matrix_storage, \
    mpi_bcast_integer
from embasi.ks_array import SpinKpointArray
from embasi.asi_default_callbacks import dm_saving_callback, \
    ham_saving_callback, ham_saving_and_huzinaga_callback, \
    ovlp_saving_callback, matrix_loading_callback
import numpy as np

# Valid ASI Flavours
ASI_flavour = {"None": -1, "Dummy": 0, "FHIaims": 1, "DFTB+": 2}

# Registry of concrete QMCodeAdapter implementations, keyed by the
# __name__ of the ASE calculator class they support. Populated by
# the register_calculator decorator below.
implemented_calculators = {}


def register_calculator(ase_calc_classname):
    """Class decorator registering a QMCodeAdapter for an ASE calculator

    Parameters
    ----------
    ase_calc_classname : str
        type(calc).__name__ of the ASE calculator this adapter supports.
    """

    def decorator(cls):
        implemented_calculators[ase_calc_classname] = cls
        return cls

    return decorator


class QMCodeAdapter(ABC):
    """Uniform interface to a QM code's input parameter conventions and
    output matrix extraction mechanism.

    A concrete implementation is selected by the qm_code_adapter() factory
    based on the underlying ASE calculator implementation. It controls:

    1) The syntax used to set input parameters (total charge, ghost atoms,
       post-SCF keywords, ...) needed for each step of an embedding
       calculation, since these differ between ASE calculator backends.
    2) How matrix quantities (density matrices, Hamiltonians, overlap) are
       exported from the QM code once a calculation has run. Some codes
       (e.g., FHI-aims via ASI) export these through callbacks registered
       ahead of the run and invoked from within the SCF cycle; others may
       expose them directly as attributes on the calculator once it has
       finished running.
    """

    def __init__(self):
        pass

    @property
    @abstractmethod
    def asi_flavour(self):
        return ASI_flavour["Dummy"]

    @property
    def uses_asi_callbacks(self):
        """True for codes which export matrices via ASI's callback
        registration mechanism; False for codes which expose matrices
        directly as attributes of the calculator once it has run.
        """
        return self.asi_flavour > 0

    @property
    def needs_nonscf_ene_corr(self):
        """Indicates whether the true total energy is returned from run_scf
           for a non-self-consistent total energy evaluation
        """
        return False

    # ------------------------------------------------------------------
    # Input directives
    # ------------------------------------------------------------------

    @abstractmethod
    def set_ghost_atoms(self, atomsembed, calc, ghost_list):
        """Set calculator parameters for setting atoms of a given index as ghost atoms

        Changes the ASE Calculator keywords to accept the keywords to
        set certain keywords as ghost basis centers.

        Parameters
        ----------
        atomsembed : AtomsEmbed
            The owning AtomsEmbed instance.
        calc : ASECalculator
            ASE calculator to add parameters.
        ghost_list : list, len(atoms)
            A list of booleans to mark which atoms are ghost species.
        """
        pass

    def set_truncated_atoms(self, atomsembed, calc, active_atoms):
        """Drops non-active atoms (and their basis functions) from the calculator

        No-op by default: QM codes whose calculator is rebuilt fresh from
        the (already-truncated) ASE atoms object on every run - e.g.
        FHI-aims via ASI, which regenerates geometry.in each call - get
        truncation for free and need nothing here. Codes that instead
        read a persistent, calculator-resident geometry/basis (PySCF's
        Mole) must override this to remove the truncated atoms there
        directly, since nothing else ever re-derives it from
        atomsembed.atoms.

        Parameters
        ----------
        atomsembed : AtomsEmbed
            The owning AtomsEmbed instance.
        calc : ASECalculator
            ASE calculator to truncate.
        active_atoms : array-like of int
            Indices, into the original untruncated atom ordering, of the
            atoms to keep.
        """
        return calc

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
        postscf_method : None, str, or object
            Post-SCF method ran on calculator execution. For codes which
            drive post-SCF methods directly in Python (currently PySCF),
            this may instead be a pre-configured, unconverged post-HF
            method object (e.g., ``pyscf.cc.CCSD(mf, frozen=2)``) whose
            solver options (frozen core, convergence thresholds, DIIS
            settings, ...) are reused - only its settings are taken, its
            mean-field/mol/MO references are replaced with the actual
            embedded reference at run time. To also request a
            perturbative triples correction on such an object, set
            ``.run_ccsd_t = True`` on it.
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
    def set_max_scf_cycles(self, calc):
        """Set calculator maximum number of SCF cycles

        Changes the ASE Calculator keywords for manipulating the number
        of SCF iterations

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
    def set_qm_total_charge(self, atomsembed, calc, charge):
        """Set calculator parameters for specifying input total charge

        Parameters
        ----------
        calc : ASECalculator
            ASE calculator to add parameters.
        charge : int
            Total charge of the system
        """
        pass

    def get_qm_input_spin(self, calc):
        """Returns the spin (Nalpha - Nbeta) already configured on this
        calculator, if the underlying QM code exposes such a concept at
        the calculator level (e.g. PySCF's ``mol.spin``).

        None (the default) for QM codes which set spin through a
        different mechanism entirely (e.g. FHI-aims' raw ASE ``spin``/
        ``default_initial_moment`` keywords, passed straight through to
        control.in) - this signals to AtomsEmbed.fragment_spin that spin
        should not be touched by set_qm_total_charge for this adapter.

        Parameters
        ----------
        calc : ASECalculator
            The calculator as originally configured by the user, before
            EmbASI performs any fragment (ghost/truncation/charge)
            mutation on it.
        """
        return None

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

    # ------------------------------------------------------------------
    # Output directives
    # ------------------------------------------------------------------

    @abstractmethod
    def register_import_export_hooks(self, atoms_calc, atoms_embed, emb_pot_scf=False):
        """Prepares the calculator to export matrix quantities

        Called immediately before atoms.get_potential_energy(). For QM
        codes which export matrices via ASI callbacks, this registers the
        relevant callbacks and initialises the storage they populate. For
        QM codes which expose matrices as plain attributes once a
        calculation has run, this is a no-op.

        Parameters
        ----------
        atoms_calc : ASE Calculator
            The (ASI-wrapped, or otherwise) ASE calculator about to be run.
        atoms_embed : AtomsEmbed
            The owning AtomsEmbed instance, providing access to embedding
            state needed to prepare the export (e.g., truncation, Huzinaga
            embedding potentials, an input density matrix to initialise).
        emb_pot_scf : bool
            Whether the run to follow is an embedding-potential SCF.
        """
        pass

    @abstractmethod
    def extract_matrices(self, atomsembed):
        """Extracts matrix quantities from a completed calculation

        Called immediately after atoms.get_potential_energy(). Returns a
        dict with keys 'n_spins', 'n_kpoints', 'n_basis', 'basis_atoms',
        'ham_kin', 'ham_2ee', 'ham_tot', 'ovlp' and 'dm' -- the same
        contract regardless of how the underlying QM code exposes its
        matrices.

        Parameters
        ----------
        atomsembed : AtomsEmbed
            The owning AtomsEmbed instance, providing access to the
            calculator which has just completed a calculation.

        Returns
        -------
        dict
        """
        pass

    @abstractmethod
    def get_energy(self, atoms):
        """Extracts final total energy of a given calculation

        Parameters
        ----------
        atoms : ASE Atoms object
            ASE atoms object with attached calculator

        Returns
        -------
        float64 : Total energy
        """
        pass

    @abstractmethod
    def get_forces(self, atoms):
        """Extracts final forces for calculation

        Parameters
        ----------
        atoms : ASE Atoms object
            ASE atoms object with attached calculator

        Returns
        -------
        ndarray : Final forces
        """
        pass

    @abstractmethod
    def get_decomposed_energy_from_file(self, atomsembed):
        """Extracts energetic components from an output file

        Parameters
        ----------
        atoms : ASE Atoms object
            ASE atoms object with attached calculator

        Returns
        -------
        atomsembed : Atomsembed object
        """
        pass

    # ------------------------------------------------------------------
    # Runtime directives
    # ------------------------------------------------------------------
    def run_scf(self, atomsembed):
        """Performs a total energy evaluation only with SCF steps set in
        set_no_scf_total_energy_calc

        Parameters
        ----------
        atomsembed : AtomsEmbed
            Wrapper containing data structures for embedding
        atoms_calc : ASE Calculator
            The calculator which has just completed a calculation.

        Returns
        -------
        None
        """
        pass

class QMCodeASIAdapter(QMCodeAdapter):
    """Generic Adapter for QM Codes using ASI

    Contains functions that control callback registration and extraction
    of data quantities returned by the ASI API

    """

    def __init__(self):
        super(QMCodeASIAdapter, self).__init__()

    def register_import_export_hooks(self, atoms_calc, atoms_embed, emb_pot_scf=False):
        asi = atoms_calc.asi

        # Explicitly set function pointers to NULL to avoid
        # previously set function pointers from passing into
        # the present calculation.
        asi.register_overlap_callback(0, 0)
        asi.register_dm_callback(0, 0)
        asi.register_DM_init(0, 0)
        asi.register_hamiltonian_callback(0, 0)
        asi.register_set_hamiltonian_callback(0, 0)
        asi.register_modify_hamiltonian_callback(0, 0)

        # Register the relevant callbacks
        asi.overlap_storage = {}
        asi.register_overlap_callback(ovlp_saving_callback,
                                      (asi,
                                       asi.overlap_storage,
                                       atoms_embed.blacs_ctxt_tag,
                                       atoms_embed.blacs_descr_tag,
                                       'Ovlp calc'))

        asi.dm_storage = {}
        asi.dm_calc_cnt = {}
        asi.dm_count = 0
        asi.register_dm_callback(dm_saving_callback,
                                 (asi,
                                  asi.dm_storage,
                                  asi.dm_calc_cnt,
                                  atoms_embed.blacs_ctxt_tag,
                                  atoms_embed.blacs_descr_tag,
                                  'DM calc'))

        asi.ham_storage = {}
        asi.ham_calc_cnt = {}
        asi.ham_count = 0

        if atoms_embed.flag_huz and emb_pot_scf:
            asi.huzinaga_eq = {}
            asi.register_hamiltonian_callback(ham_saving_and_huzinaga_callback,
                                              (asi,
                                               asi.ham_storage,
                                               {"atembed": atoms_embed},
                                               asi.ham_calc_cnt,
                                               atoms_embed.blacs_ctxt_tag,
                                               atoms_embed.blacs_descr_tag,
                                               'Ham calc'))
        else:
            asi.register_hamiltonian_callback(ham_saving_callback,
                                              (asi,
                                               asi.ham_storage,
                                               asi.ham_calc_cnt,
                                               atoms_embed.blacs_ctxt_tag,
                                               atoms_embed.blacs_descr_tag,
                                               'Ham calc'))

        if atoms_embed.density_matrix_in is not None:
            asi.register_DM_init(matrix_loading_callback,
                                 (asi,
                                 atoms_embed.density_matrix_in,
                                 False,
                                 atoms_embed.blacs_ctxt_tag,
                                 atoms_embed.blacs_descr_tag,
                                 'DM init'))

        # Make sure we don't run with an already stored embedding potential unless
        # It is actually called for.
        if (atoms_embed.fock_embedding_matrix is not None) and emb_pot_scf:
            if atoms_embed.truncate:
                mat_in = atoms_embed.fock_embedding_matrix_trunc
            else:
                mat_in = atoms_embed.fock_embedding_matrix

            asi.register_modify_hamiltonian_callback(matrix_loading_callback,
                                                      (asi,
                                                      mat_in,
                                                      atoms_embed.flag_huz,
                                                      atoms_embed.blacs_ctxt_tag,
                                                      atoms_embed.blacs_descr_tag,
                                                      'Modify H'))

    def get_decomposed_energy_from_file(self, atomsembed):
        """Extracts quantities not currently supported by ASI

        An ad hoc solution to extract values unsupported by ASI for
        FHI-aims. Currently reads values such as the kinetic energy,
        electrostatic energy, sum of eigenvalues etc,

        """

        with open(atomsembed.outdir+'/asi.log', 'r') as output:
            lines = output.readlines()

            for line in lines:
                outline = line.split()

                if '  | Kinetic energy                :' in line:
                    atomsembed.kinetic_energy = float(outline[6])

                if '  | Electrostatic energy          :' in line:
                    atomsembed.es_energy = float(outline[6])

                if '  | Sum of eigenvalues            :' in line:
                    atomsembed.ev_sum = float(outline[7])

                if '  | Total energy of the DFT' in line:
                    atomsembed.dft_energy = float(outline[11])

                if 'Total XC Energy     :' in line:
                    atomsembed.xc_energy = float(outline[6])

                if 'Total energy after the post-s.c.f.' in line:
                    atomsembed.post_scf_corr_energy = float(outline[9])

        return atomsembed

    def extract_matrices(self, atomsembed):
        atoms_embed = atomsembed
        atoms_calc = atomsembed.atoms.calc
        asi = atoms_calc.asi

        n_kpoints = asi.n_kpts
        n_spins = asi.n_spin

        # BROADCAST QUANTITIES ONLY CALCULATED FOR THE HEAD NODE TO ALL
        # OTHER NODES - ONLY DO THIS IN SERIAL MODE AS THE NPSCAL ARRAYS
        # ARE ALREADY DISTRIBUTED TO EACH TASK
        if not atoms_embed.parallel:
            asi.dm_count = mpi_bcast_integer(asi.dm_count)
            asi.ham_count = mpi_bcast_integer(asi.ham_count)

            if MPI.COMM_WORLD.Get_rank() != 0:
                for iham in range(asi.ham_count):
                    asi.ham_storage[iham] = {}

            if MPI.COMM_WORLD.Get_rank() != 0:
                for idm in range(asi.dm_count):
                    asi.dm_storage[idm] = {}

            for iham in range(asi.ham_count):
                asi.ham_storage[iham] = \
                    mpi_bcast_matrix_storage(asi.ham_storage[iham],
                                             asi.n_basis,
                                             asi.n_basis)

            for idm in range(asi.dm_count):
                asi.dm_storage[idm] = \
                    mpi_bcast_matrix_storage(asi.dm_storage[idm],
                                         asi.n_basis,
                                         asi.n_basis)

            asi.overlap_storage = \
                mpi_bcast_matrix_storage(asi.overlap_storage,
                                     asi.n_basis,
                                     asi.n_basis)

        if atoms_embed.truncate:
            for idx, kspdict in asi.ham_storage.items():
                for key in kspdict.keys():
                    asi.ham_storage[idx][key] = atoms_embed.truncated_mat_to_full(kspdict.get(key))

            for idx, kspdict in asi.dm_storage.items():
                for key in kspdict.keys():
                    asi.dm_storage[idx][key] = atoms_embed.truncated_mat_to_full(kspdict.get(key))

            for key in asi.overlap_storage.keys():
                asi.overlap_storage[key] = atoms_embed.truncated_mat_to_full(asi.overlap_storage.get(key))

        # Now put all of the output arrays into a nice wrapper
        ham_kin = SpinKpointArray(asi.ham_storage[0], n_spins, n_kpoints)
        ham_estat_xc = SpinKpointArray(asi.ham_storage[1], n_spins, n_kpoints)
        ham_tot = SpinKpointArray(asi.ham_storage[2], n_spins, n_kpoints)
        ovlp = SpinKpointArray(asi.overlap_storage, n_spins, n_kpoints)

        if n_spins > 1:
            ovlp[0, 0] = ovlp[0, 0]
            ovlp[1, 0] = ovlp[0, 0]

        # THIS IS CODE BREAKING FOR QM CODE LOCALISATION - I WILL NEED A FIX TO RESTORE
        if 1 in asi.dm_storage.keys():
            dm = []
            dm.append(SpinKpointArray(asi.dm_storage[0], n_spins, n_kpoints))
            dm.append(SpinKpointArray(asi.dm_storage[1], n_spins, n_kpoints))
        else:
            dm = SpinKpointArray(asi.dm_storage[0], n_spins, n_kpoints)

        return {
            "n_spins": n_spins,
            "n_kpoints": n_kpoints,
            "n_basis": asi.n_basis,
            "basis_atoms": asi.basis_atoms,
            "ham_kin": ham_kin,
            "ham_estat_xc": ham_estat_xc,
            "ham_tot": ham_tot,
            "ovlp": ovlp,
            "dm": dm,
        }

    def run_scf(self, atomsembed):
        atomsembed.atoms.get_potential_energy()

@register_calculator("Aims")
class AimsAdapter(QMCodeASIAdapter):

    def __init__(self):
        super(AimsAdapter, self).__init__()

    @property
    def asi_flavour(self):
        return ASI_flavour["FHIaims"]

    @property
    def needs_nonscf_ene_corr(self):
        """Indicates whether the true total energy is returned from run_scf
           for a non-self-consistent total energy evaluation
        """
        return True

    def set_noscf_calc_params(self, calc):
        calc.parameters['charge_mix_param'] = 0.
        calc = self.set_max_scf_cycles(calc, 0)
        self.set_embasi_calculation_type(calc, "total_energy_only")

        return calc

    def set_max_scf_cycles(self, calc, max_cycles):
        calc.parameters['sc_iter_limit'] = max_cycles
        return calc

    def set_ghost_atoms(self, atomsembed, calc, ghost_list):
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
        calc.parameters['override_default_empty_basis_order'] = ".true."

        return calc

    def set_qm_localise(self, calc):
        calc.parameters['qm_embedding_mo_localise'] = ".true."

        return calc

    def set_qm_total_charge(self, atomsembed, calc, charge):
        calc.parameters['charge'] = charge
        return calc

    def get_energy(self, atomsembed):
        return atomsembed.atoms.calc.get_potential_energy()

    def get_forces(self, atomsembed):
        return atomsembed.atoms.calc.get_forces()

@register_calculator("PySCF")
class PySCFAdapter(QMCodeAdapter):

    def __init__(self):
        super(PySCFAdapter, self).__init__()

    @property
    def asi_flavour(self):
        return ASI_flavour["None"]

    def set_noscf_calc_params(self, calc):
        calc = self.set_max_scf_cycles(calc, 0)
        return calc

    def set_max_scf_cycles(self, calc, max_cycles):
        calc.method.max_cycle = max_cycles
        calc.method_scan.max_cycle = max_cycles
        return calc

    def get_qm_input_spin(self, calc):
        return calc.mol.spin

    def _set_parity_safe_placeholder_spin(self, calc):
        """Sets calc.mol.spin to satisfy mol.build()'s parity check

        set_ghost_atoms/set_truncated_atoms mutate calc.mol.atom (and,
        for ghosts, calc.mol.basis) *before* set_qm_total_charge (called
        right after, in calc_initializer) applies the fragment's actual
        charge and spin together. mol.build() requires
        (nelectron - spin) % 2 == 0 though, so this needs a valid
        placeholder immediately - one with the correct parity for
        whatever electron count calc.mol.atom/calc.mol.charge currently
        imply (which, at this intermediate point, is generally NOT yet
        the fragment's final electron count). The exact spin value
        doesn't matter here since set_qm_total_charge overwrites it
        with the real, final one right afterwards.

        atomsembed.fragment_spin must NOT be used here directly: it is
        the fragment's *final* target spin, which is only guaranteed
        consistent with the electron count set_qm_total_charge is about
        to apply together with it - not with whatever calc.mol
        currently (still) describes.
        """
        from pyscf.gto.mole import charge as elem_charge

        current_nelec = sum(elem_charge(entry[0]) for entry in calc.mol.atom) - calc.mol.charge
        calc.mol.spin = round(current_nelec) % 2

    def set_ghost_atoms(self, atomsembed, calc, ghost_list):
        from pyscf import gto

        ghost_species = []
        old_basis = calc.mol.basis
        new_basis = {}

        if not isinstance(old_basis, dict):
            for element in calc.mol.elements:
                new_basis[element] = old_basis

        for idx, ghost in enumerate(ghost_list):
            if ghost:
                old_species_name = calc.mol.atom[idx][0]
                calc.mol.atom[idx][0] = "ghost_" + old_species_name

                if old_species_name not in ghost_species:
                    ghost_basis = new_basis[old_species_name]
                    new_basis[calc.mol.atom[idx][0]] = gto.basis.load(ghost_basis, old_species_name)
                    ghost_species.append(old_species_name)

        calc.mol.basis = new_basis

        self._set_parity_safe_placeholder_spin(calc)

        calc.mol.build()

        return calc

    def set_truncated_atoms(self, atomsembed, calc, active_atoms):
        old_atom = calc.mol.atom
        old_basis = calc.mol.basis

        new_atom = [old_atom[idx] for idx in active_atoms]

        if isinstance(old_basis, dict):
            kept_species = {entry[0] for entry in new_atom}
            new_basis = {sp: basis for sp, basis in old_basis.items() if sp in kept_species}
        else:
            new_basis = old_basis

        calc.mol.atom = new_atom
        calc.mol.basis = new_basis

        self._set_parity_safe_placeholder_spin(calc)

        calc.mol.build()

        return calc

    def set_postscf_keyword(self, calc, postscf_method):
        calc.parameters['postscf_method'] = postscf_method
        return calc

    def set_scalapack_blocksize(self, calc, scalapack_blocksize):
        return calc

    def set_embasi_calculation_type(self, calc, calc_type):
        return calc

    def override_basis_order(self, calc):
        return calc

    def set_qm_localise(self, calc):
        return calc

    def set_qm_total_charge(self, atomsembed, calc, charge):
        calc.mol.charge = charge
        calc.mol.spin = atomsembed.fragment_spin
        calc.mol.build()
        return calc

    def run_scf(self, atomsembed):
        from pyscf.scf import uhf as pyscf_uhf
        from pyscf.scf import rohf as pyscf_rohf

        mf = atomsembed.atoms.calc.method

        # UHF/UKS and ROHF/ROKS both carry two independent spin channels
        # in dm/veff (shape (2, nao, nao)), even though ROHF/ROKS's
        # mo_coeff stays a single shared (nao, nao) set of orbitals with
        # a 1D mo_occ - ROHF does NOT subclass uhf.UHF (it subclasses
        # RHF directly), so both base classes need checking explicitly.
        # RHF/RKS carry one channel. Detected from the mf object itself
        # (the SCF method class the user built calc_base_ll/hl with),
        # not from atomsembed.n_spins - which only reflects whatever a
        # *previous* run happened to report, and isn't set at all
        # before the first run.
        n_spins = 2 if isinstance(mf, (pyscf_uhf.UHF, pyscf_rohf.ROHF)) else 1

        if atomsembed.density_matrix_in is None:
            dm_in = mf.get_init_guess()
        elif n_spins == 2:
            dm_in = np.array([atomsembed.density_matrix_in[0,0],
                              atomsembed.density_matrix_in[1,0]])
        else:
            dm_in = atomsembed.density_matrix_in[0,0]

        # Both the static embedding potential (level-shift, and the
        # 'emb_pot' term of Huzinaga projection) and the dynamic Huzinaga
        # projector (which depends on the current, per-iteration Fock, not
        # just a fixed matrix) are injected via a single get_fock override.
        # get_fock is called exactly once per SCF cycle, right before
        # diagonalisation, so this needs no replication of PySCF's own SCF
        # loop - damping/DIIS/level-shift are all still handled by
        # delegating to the original get_fock. get_hcore is deliberately
        # left untouched: energy_elec/energy_tot never call get_fock (they
        # take h1e/vhf directly, or fall back to get_hcore/get_veff), so the
        # energy and Hamiltonian extracted below (energy_tot(dm, hcore, veff),
        # hcore + veff) are built on the bare core Hamiltonian. CCSD/MP2
        # rebuild their reference Fock via mf.get_fock(vhf=vhf, dm=dm)
        # (see pyscf.cc.ccsd._ChemistsERIs._common_init_), so this is also
        # the call the post-SCF correction actually needs patched.
        needs_fock_override = (atomsembed.fock_embedding_matrix is not None) or \
            (atomsembed.flag_huz and atomsembed.huzinaga_dm_in is not None)

        if needs_fock_override:
            from embasi.pyscf_scf_hooks import embedded_get_fock_factory

            def _spinks_to_ndarray(spinks):
                # Overlap (S below) is spin-independent by construction
                # and stays a plain (nao, nao) 2D array even when
                # n_spins==2 - numpy's batched @ broadcasts a bare 2D
                # array against a (2, nao, nao) stack just fine, so
                # huzinaga_projector needs no special-casing for it.
                if n_spins == 2:
                    return np.array([spinks[0,0], spinks[1,0]])
                else:
                    return spinks[0,0]

            mat_in = None
            if atomsembed.fock_embedding_matrix is not None:
                if atomsembed.truncate:
                    mat_in = _spinks_to_ndarray(atomsembed.fock_embedding_matrix_trunc)
                else:
                    mat_in = _spinks_to_ndarray(atomsembed.fock_embedding_matrix)

            gamma_B = S = None
            if atomsembed.flag_huz and atomsembed.huzinaga_dm_in is not None:
                # huzinaga_dm_in/huzinaga_ovlp_in are always stored full-size
                # (run_embasi_diag_emb_pot's absolute-truncation projector
                # needs them that way), but the Fock override below runs
                # inside mf.kernel() on the truncated Mole, so gamma_B/S
                # need to match its (trunc_nbasis, trunc_nbasis) Fock here.
                if atomsembed.truncate:
                    gamma_B = _spinks_to_ndarray(atomsembed.full_mat_to_truncated(atomsembed.huzinaga_dm_in))
                    S = atomsembed.full_mat_to_truncated(atomsembed.huzinaga_ovlp_in)[0,0]
                else:
                    gamma_B = _spinks_to_ndarray(atomsembed.huzinaga_dm_in)
                    S = atomsembed.huzinaga_ovlp_in[0,0]

            fock_func = mf.get_fock
            mf.get_fock = embedded_get_fock_factory(mf, fock_func, mat_in=mat_in,
                                                    gamma_B=gamma_B, S=S, n_spins=n_spins)

        total_energy = mf.kernel(dm0=dm_in)

        # Post-SCF (correlated wavefunction) correction, if requested. This
        # must run here, before the get_fock override below is undone -
        # cc.CCSD/mp.MP2 rebuild their reference Fock from mf.get_fock(),
        # so the correlated method needs to see the same (possibly
        # embedded/projected) Fock that the converged orbitals above were
        # obtained from.
        postscf_method = atomsembed.atoms.calc.parameters.get('postscf_method')
        if postscf_method is not None and mf.converged:
            from embasi.pyscf_postscf import run_postscf

            e_corr = run_postscf(mf, postscf_method)

            atomsembed.dft_energy = total_energy * 27.211384500
            atomsembed.post_scf_corr_energy = atomsembed.dft_energy + e_corr * 27.211384500

        if needs_fock_override:
            mf.get_fock = fock_func

        scf_conv = mf.converged
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ

        aoslice = mf.mol.aoslice_by_atom()
        basis_atoms = []
        for atom_idx, basis in enumerate(aoslice):
            basis_atoms += len(range(basis[2],basis[3])) * [atom_idx]
        basis_atoms = np.array(basis_atoms)

        nbasis = len(basis_atoms)
        n_kpoints = 1

        kin_mat = mf.mol.intor('int1e_kin')
        ovlp_mat = mf.mol.intor('int1e_ovlp')
        vnuc = mf.mol.intor('int1e_nuc')

        if scf_conv:
            dm_out = mf.make_rdm1(mo_coeff, mo_occ)
        else:
            dm_out = dm_in

        hcore = mf.get_hcore()
        veff = mf.get_veff(dm=dm_out)

        total_energy = mf.energy_tot(dm_out, hcore, veff) * 27.211384500

        # kin/ovlp are one-electron, spin-independent operators - the
        # same matrix for both channels. Duplicated into both (0,0) and
        # (1,0) slots when n_spins==2, mirroring what
        # QMCodeASIAdapter.extract_matrices already does for FHI-aims'
        # spin-independent overlap (qmcode_adapters.py, AimsAdapter).
        if n_spins == 2:
            ham_kin = SpinKpointArray({(0,0): kin_mat, (1,0): kin_mat}, n_spins, n_kpoints)
            ovlp = SpinKpointArray({(0,0): ovlp_mat, (1,0): ovlp_mat}, n_spins, n_kpoints)
            dm = SpinKpointArray({(0,0): dm_out[0], (1,0): dm_out[1]}, n_spins, n_kpoints)
            ham_estat_xc = SpinKpointArray({(0,0): vnuc + veff[0], (1,0): vnuc + veff[1]}, n_spins, n_kpoints)
            ham_tot = SpinKpointArray({(0,0): hcore + veff[0], (1,0): hcore + veff[1]}, n_spins, n_kpoints)
        else:
            ham_kin = SpinKpointArray({(0,0): kin_mat}, n_spins, n_kpoints)
            ovlp = SpinKpointArray({(0,0): ovlp_mat}, n_spins, n_kpoints)
            dm = SpinKpointArray({(0,0): dm_out}, n_spins, n_kpoints)
            ham_estat_xc = SpinKpointArray({(0,0): vnuc + veff}, n_spins, n_kpoints)
            ham_tot = SpinKpointArray({(0,0): hcore + veff}, n_spins, n_kpoints)

        # Everything above was computed on the truncated Mole (shape
        # trunc_nbasis x trunc_nbasis). Pad back up to the full
        # supersystem basis so combining these with other, untruncated
        # layers (e.g. self.AB_LL.hamiltonian_total - self.A_LL.hamiltonian_total
        # in ProjectionEmbedding.run) doesn't hit a shape mismatch - this
        # mirrors what QMCodeASIAdapter.extract_matrices already does for
        # the ASI-based adapters via the same truncated_mat_to_full call.
        if atomsembed.truncate:
            ham_kin = atomsembed.truncated_mat_to_full(ham_kin)
            ham_estat_xc = atomsembed.truncated_mat_to_full(ham_estat_xc)
            ham_tot = atomsembed.truncated_mat_to_full(ham_tot)
            ovlp = atomsembed.truncated_mat_to_full(ovlp)
            dm = atomsembed.truncated_mat_to_full(dm)

        self.scf_rundata = SCFRunData(total_energy,
                                      n_spins,
                                      n_kpoints,
                                      nbasis,
                                      basis_atoms,
                                      ham_kin,
                                      ham_estat_xc,
                                      ham_tot,
                                      ovlp,
                                      dm
                                      )

    def get_energy(self, atomsembed):
        return self.scf_rundata.total_energy

    def get_forces(self, atomsembed):
        return None

    def extract_matrices(self, atomsembed):
        return asdict(self.scf_rundata)

    def register_import_export_hooks(self, atoms_calc, atoms_embed, emb_pot_scf=False):
        return atoms_calc

    def get_decomposed_energy_from_file(self, atomsembed):
        return atomsembed

def qm_code_adapter(calc):
    """Gets the QMCodeAdapter for a given ASE Calculator implementation

    Gets the name of the ASE calculator implementation and returns the
    adapter object needed to set input parameters and extract output
    matrices through the EmbASI calculation.

    Parameters
    ----------
    calc : ASEIOCalculator

    Raises
    ------
    Exception
        ASE Calculator not implemented.
    """

    classname = type(calc).__name__

    if classname not in implemented_calculators:
        raise Exception("ASE Calculator not implemented.")

    return implemented_calculators[classname]()

@dataclass
class SCFRunData:
    total_energy: np.float64
    n_spins: int
    n_kpoints: int
    n_basis: int
    basis_atoms: list
    ham_kin: SpinKpointArray
    ham_estat_xc: SpinKpointArray
    ham_tot: SpinKpointArray
    ovlp: SpinKpointArray
    dm: SpinKpointArray


