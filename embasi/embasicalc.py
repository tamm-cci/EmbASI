
from asi4py.asecalc import ASI_ASE_calculator
import os, shutil, re

class EmbASI_ASE_calculator(ASI_ASE_calculator):
    def write_input(self, atoms, properties=None, system_changes=None):
        super().write_input(atoms, properties, system_changes)

        # Modify species file for a specific element
        species_dir = os.environ.get('AIMS_SPECIES_DIR')
        target = 'X'  # e.g., 'H'
        src = os.path.join(species_dir, target)
        dst = os.path.join(os.getcwd(), f"{target}_empty")

        # copy first
        shutil.copy(src, dst)

        # modify the file
        with open(dst) as f:
            lines = f.readlines()

        modified = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith(("hydro", "ionic")) and not stripped.startswith("#"):
                modified.append("# " + line)
            else:
                modified.append(line)
        if not modified[-1].endswith("\n"):
            modified[-1] += "\n"
        modified.append("include_min_basis  .false.\n")

        with open(dst, "w") as f:
            f.writelines(modified)
