try:
    from ase.data.s22 import s26,create_s22_system
    import numpy as np
    import os
    import re
    import time
    import warnings
    from functools import cached_property
    from pathlib import Path
    from typing import Any, Dict, List, Union
    import numpy as np
    import decimal
    from ase import Atom, Atoms
    from ase.calculators.calculator import kpts2mp
    from ase.io.aims import get_aims_header, get_sym_block, v_unit
    from ase.calculators.singlepoint import SinglePointDFTCalculator
    from ase.constraints import FixAtoms, FixCartesian
    from ase.data import atomic_numbers
    from ase.io import ParseError
    from ase.units import Ang, fs
    from ase.io.aims import write_aims, write_control
    from ase.utils import deprecated, reader, writer
    from ase.io.aims import format_aims_control_parameter, get_species_directory, parse_species_path
    import shlex
    from abc import ABC, abstractmethod
    from contextlib import ExitStack
    from os import PathLike
    from pathlib import Path
    from typing import Any, Iterable, List, Mapping, Optional, Set

    from ase.calculators.abc import GetOutputsMixin
    from ase.calculators.calculator import (
        BadConfiguration,
        BaseCalculator,
        _validate_command,
    )
    from ase.config import cfg as _cfg

except:
    pass

#/home/dchen/Software/FHIaims/species_defaults/defaults_2020/light/00_Emptium_default
os.environ["FHI_AIMS_PATH"] = "/home/dchen/Software/FHIaims"

v_unit = Ang / (1000.0 * fs)

def write_species(control_file_descriptor, species_basis_dict, parameters):
    """Write species for the calculation depending on basis set size.

    The calculation should include certain basis set size function depending
    on the numerical settings (light, tight, really tight) and the basis set
    size (minimal, tier1, tier2, tier3, tier4). If the basis set size is not
    given then a 'standard' basis set size is used for each numerical setting.
    The species files are defined according to these standard basis set sizes
    for the numerical settings in the FHI-aims repository.

    Note, for FHI-aims in ASE, we don't explicitly give the numerical setting.
    Instead we include the numerical setting in the species path: e.g.
    `~/aims2022/FHIaims/species_defaults/defaults_2020/light` this path has
    `light`, the numerical setting, as the last folder in the path.

    Example - a basis function might be commented in the standard basis set size
        such as "#     hydro 4 f 7.4" and this basis function should be
        uncommented for another basis set size such as tier4.

    Args:
        control_file_descriptor: File descriptor for the control.in file into
            which we need to write relevant basis functions to be included for
            the calculation.
        species_basis_dict: Dictionary where keys as the species symbols and
            each value is a single string containing all the basis functions
            to be included in the caclculation.
        parameters: Calculation parameters as a dict.
    """
    # Now for every species (key) in the species_basis_dict, save the
    # relevant basis functions (values) from the species_basis_dict, by
    # writing to the file handle (species_file_descriptor) given to this
    # function.
    for species_symbol, basis_set_text in species_basis_dict.items():
        control_file_descriptor.write(basis_set_text)
        if parameters.get("plus_u") is not None:
            if species_symbol in parameters.plus_u:
                control_file_descriptor.write(
                    f"plus_u {parameters.plus_u[species_symbol]} \n")

def write_control(fd, atoms, parameters, verbose_header=False):
    """Write the control.in file for FHI-aims
    Parameters
    ----------
    fd: str
        The file object to write to
    atoms: atoms.Atoms
        The Atoms object for the requested calculation
    parameters: dict
        The dictionary of all paramters for the calculation
    verbose_header: bool
        If True then explcitly list the paramters used to generate the
         control.in file inside the header
    """

    parameters = dict(parameters)
    lim = "#" + "=" * 79

    if parameters["xc"] == "LDA":
        parameters["xc"] = "pw-lda"

    cubes = parameters.pop("cubes", None)

    for line in get_aims_header():
        fd.write(line + "\n")

    if verbose_header:
        fd.write("# \n# List of parameters used to initialize the calculator:")
        for p, v in parameters.items():
            s = f"#     {p}:{v}\n"
            fd.write(s)
    fd.write(lim + "\n")

    assert "kpts" not in parameters or "k_grid" not in parameters
    assert "smearing" not in parameters or "occupation_type" not in parameters

    for key, value in parameters.items():
        if key == "kpts":
            mp = kpts2mp(atoms, parameters["kpts"])
            dk = 0.5 - 0.5 / np.array(mp)
            fd.write(
                format_aims_control_parameter(
                    "k_grid",
                    tuple(mp),
                    "%d %d %d"))
            fd.write(
                format_aims_control_parameter(
                    "k_offset",
                    tuple(dk),
                    "%f %f %f"))
        elif key in ("species_dir", "tier"):
            continue
        elif key == "aims_command":
            continue
        elif key == "plus_u":
            continue
        elif key == "smearing":
            name = parameters["smearing"][0].lower()
            if name == "fermi-dirac":
                name = "fermi"
            width = parameters["smearing"][1]
            if name == "methfessel-paxton":
                order = parameters["smearing"][2]
                order = " %d" % order
            else:
                order = ""

            fd.write(
                format_aims_control_parameter(
                    "occupation_type", (name, width, order), "%s %f%s"
                )
            )
        elif key == "output":
            for output_type in value:
                fd.write(format_aims_control_parameter(key, output_type, "%s"))
        elif key == "vdw_correction_hirshfeld" and value:
            fd.write(format_aims_control_parameter(key, "", "%s"))
        elif isinstance(value, bool):
            fd.write(
                format_aims_control_parameter(
                    key, str(value).lower(), ".%s."))
        elif isinstance(value, (tuple, list)):
            fd.write(
                format_aims_control_parameter(
                    key, " ".join([str(x) for x in value]), "%s"
                )
            )
        elif isinstance(value, str):
            fd.write(format_aims_control_parameter(key, value, "%s"))
        else:
            fd.write(format_aims_control_parameter(key, value, "%r"))

    if cubes:
        cubes.write(fd)

    fd.write(lim + "\n\n")

    # Get the species directory
    species_dir = get_species_directory
    # dicts are ordered as of python 3.7
    species_array = np.array(list(dict.fromkeys(atoms.symbols)))
    # Grab the tier specification from the parameters. THis may either
    # be None, meaning the default should be used for all species, or a
    # list of integers/None values giving a specific basis set size
    # for each species in the calculation.
    tier = parameters.pop("tier", None)
    tier_array = np.full(len(species_array), tier)
    # Path to species files for FHI-aims. In this files are specifications
    # for the basis set sizes depending on which basis set tier is used.
    species_dir = get_species_directory(parameters.get("species_dir"))
    # Parse the species files for each species present in the calculation
    # according to the tier of each species.
    species_basis_dict = parse_species_path(
        species_array=species_array, tier_array=tier_array,
        species_dir=species_dir)
    # Write the basis functions to be included for each species in the
    # calculation into the control.in file (fd).
    write_species(fd, species_basis_dict, parameters)


def write_aims_embasi(fd,atoms, cycle, scaled=False,geo_constrain=False,write_velocities=False,velocities=False,ghosts=None,info_str=None,wrap=False):
    the = []
    if cycle == 0:
        fd = open(fd, "w")
        if scaled and not np.all(atoms.pbc):
            raise ValueError(
                "Requesting scaled for a calculation where scaled=True, but "
                "the system is not periodic")

        if geo_constrain:
            if not scaled and np.all(atoms.pbc):
                warnings.warn(
                    "Setting scaled to True because a symmetry_block is detected."
                )
                scaled = True
            elif not np.all(atoms.pbc):
                warnings.warn(
                    "Parameteric constraints can only be used in periodic systems."
                )
                geo_constrain = False

        for line in get_aims_header():
            fd.write(line + "\n")

        # If writing additional information is requested via info_str:
        if info_str is not None:
            fd.write("\n# Additional information:\n")
            if isinstance(info_str, list):
                fd.write("\n".join([f"#  {s}" for s in info_str]))
            else:
                fd.write(f"# {info_str}")
            fd.write("\n")

        fd.write("#=======================================================\n")

        i = 0
        if atoms.get_pbc().any():
            for n, vector in enumerate(atoms.get_cell()):
                fd.write("lattice_vector ")
                for i in range(3):
                    fd.write(f"{vector[i]:16.16f} ")
                fd.write("\n")

        fix_cart = np.zeros((len(atoms), 3), dtype=bool)
        for constr in atoms.constraints:
            if isinstance(constr, FixAtoms):
                fix_cart[constr.index] = (True, True, True)
            elif isinstance(constr, FixCartesian):
                fix_cart[constr.index] = constr.mask

        if ghosts is None:
            ghosts = np.zeros(len(atoms))
        else:
            assert len(ghosts) == len(atoms)

        wrap = wrap and not geo_constrain
        scaled_positions = atoms.get_scaled_positions(wrap=wrap)

        for i, atom in enumerate(atoms):
            if ghosts[i] == 1:
                atomstring = "empty "
            elif scaled:
                atomstring = "atom_frac "
            else:
                atomstring = "atom "
            fd.write(atomstring)
            if scaled:
                for pos in scaled_positions[i]:
                    fd.write(f"{pos:16.16f} ")
            else:
                for pos in atom.position:
                    fd.write(f"{pos:16.16f} ")
            fd.write(atom.symbol)
            fd.write("\n")
            the.append(atomstring + " ".join(
                [f"{float(atom.position[i]):16.16f}" for i in range(len(atom.position))]) + " " + atom.symbol)
            # (1) all coords are constrained:
            if fix_cart[i].all():
                fd.write("    constrain_relaxation .true.\n")
            # (2) some coords are constrained:
            elif fix_cart[i].any():
                xyz = fix_cart[i]
                for n in range(3):
                    if xyz[n]:
                        fd.write(f"    constrain_relaxation {'xyz'[n]}\n")
            if atom.charge:
                fd.write(f"    initial_charge {atom.charge:16.6f}\n")
            if atom.magmom:
                fd.write(f"    initial_moment {atom.magmom:16.6f}\n")

            if write_velocities and atoms.get_velocities() is not None:
                v = atoms.get_velocities()[i] / v_unit
                fd.write(f"    velocity {v[0]:.16f} {v[1]:.16f} {v[2]:.16f}\n")

        if geo_constrain:
            for line in get_sym_block(atoms):
                fd.write(line)

        fd.close()
    else:
        try:
            atoms.info["multipole-charges"]
        except:
            return

        MultipoleAtoms = atoms.info["multipole-charges"]

        fd = open(fd, "a")

        for item,v in zip(MultipoleAtoms.coordinates,MultipoleAtoms.charge):
            fd.write(f"multipole {item[0]} {item[1]} {item[2]} 0 {v}\n")
            fd.write(f"empty {item[0]} {item[1]} {item[2]} Emptium\n")
        fd.close()


import os
import re

import numpy as np

from ase.calculators.genericfileio import (
    BaseProfile,
    CalculatorTemplate,
    GenericFileIOCalculator,
    read_stdout,
)
from ase.io.aims import write_aims, write_control


def get_aims_version(string):
    match = re.search(r'\s*FHI-aims version\s*:\s*(\S+)', string, re.M)
    return match.group(1)


class AimsProfile(BaseProfile):
    configvars = {'default_species_directory'}

    def __init__(self, command, default_species_directory=None, **kwargs):
        super().__init__(command, **kwargs)
        self.default_species_directory = default_species_directory

    def get_calculator_command(self, inputfile):
        return []

    def version(self):
        return get_aims_version(read_stdout(self._split_command))

class AimsTemplate(CalculatorTemplate):
    _label = 'aims'

    def __init__(self):
        super().__init__(
            'aims',
            [
                'energy',
                'free_energy',
                'forces',
                'stress',
                'stresses',
                'dipole',
                'magmom',
            ],
        )

        self.outputname = f'{self._label}.out'
        self.errorname = f'{self._label}.err'

    def update_parameters(self, properties, parameters):
        """Check and update the parameters to match the desired calculation

        Parameters
        ----------
        properties: list of str
            The list of properties to calculate
        parameters: dict
            The parameters used to perform the calculation.

        Returns
        -------
        dict
            The updated parameters object
        """
        parameters = dict(parameters)
        property_flags = {
            'forces': 'compute_forces',
            'stress': 'compute_analytical_stress',
            'stresses': 'compute_heat_flux',
        }
        # Ensure FHI-aims will calculate all desired properties
        for property in properties:
            aims_name = property_flags.get(property, None)
            if aims_name is not None:
                parameters[aims_name] = True

        if 'dipole' in properties:
            if 'output' in parameters and 'dipole' not in parameters['output']:
                parameters['output'] = list(parameters['output'])
                parameters['output'].append('dipole')
            elif 'output' not in parameters:
                parameters['output'] = ['dipole']

        return parameters

    def write_input(self, profile, directory, atoms, parameters, properties, cycle):
        """Write the geometry.in and control.in files for the calculation

        Parameters
        ----------
        directory : Path
            The working directory to store the input files.
        atoms : atoms.Atoms
            The atoms object to perform the calculation on.
        parameters: dict
            The parameters used to perform the calculation.
        properties: list of str
            The list of properties to calculate
        """
        parameters = self.template.update_parameters(properties, parameters)

        ghosts = parameters.pop('ghosts', None)
        geo_constrain = parameters.pop('geo_constrain', None)
        scaled = parameters.pop('scaled', None)
        write_velocities = parameters.pop('write_velocities', None)

        if scaled is None:
            scaled = np.all(atoms.pbc)
        if write_velocities is None:
            write_velocities = atoms.has('momenta')

        if geo_constrain is None:
            geo_constrain = scaled and 'relax_geometry' in parameters

        have_lattice_vectors = atoms.pbc.any()
        have_k_grid = (
            'k_grid' in parameters
            or 'kpts' in parameters
            or 'k_grid_density' in parameters
        )
        if have_lattice_vectors and not have_k_grid:
            raise RuntimeError('Found lattice vectors but no k-grid!')
        if not have_lattice_vectors and have_k_grid:
            raise RuntimeError('Found k-grid but no lattice vectors!')

        geometry_in = directory / 'geometry.in'

        write_aims_embasi(
            geometry_in,
            atoms,
            cycle,
            scaled,
            geo_constrain,
            write_velocities=write_velocities,
            ghosts=ghosts
        )

        control = directory / 'control.in'

        if (
            'species_dir' not in parameters
            and profile.default_species_directory is not None
        ):
            parameters['species_dir'] = profile.default_species_directory

        write_control(control, atoms, parameters)

        try:
            dir = os.environ["FHI_AIMS_PATH"] + "/species_defaults/defaults_2020/light/00_Emptium_default"
            atoms.info["multipole-charges"]
            with open(control, "a") as f:
                with open(dir,"r") as f2:
                    f23 = f2.read()
                    f2.close()
                f.write(f23)
            f.close()
        except:
            import traceback
            print(traceback.format_exc())

































































