import pytest
import os
from pathlib import Path

from pytest_regressions.num_regression import NumericRegressionFixture
from pytest_regressions.data_regression import DataRegressionFixture

from embasi.embedding import FrozenDensityEmbedding
from ase.data.s22 import s26, create_s22_system
from ase.calculators.aims import Aims, AimsProfile

import re

def get_value(key, last_cycle_block):
    match = re.search(rf"{key}\s+([-0-9.eE]+)", last_cycle_block)
    if not match:
        raise RuntimeError(f"Missing '{key}' in last FDET cycle")
    return float(match.group(1))

def test_fdet_lda_regression(num_regression: NumericRegressionFixture,
                         data_regression: DataRegressionFixture,
                         tmp_path):

    # === Run the calculation inside temp directory ===
    cwd = os.getcwd()
    os.chdir(tmp_path)

    # Prepare calculator
    calc = Aims(profile=AimsProfile(command="asi-doesnt-need-command"),
                override_warning_libxc=True,
                override_default_empty_basis_order=True,
                xc='libxc LDA_X+LDA_C_VWN',
                relativistic="atomic_zora scalar",
                spin="none",
                xc_nakp="libxc  LDA_K_TF",
                xc_naxcp="libxc LDA_X+LDA_C_VWN",
                )

    os.makedirs("H2O_dimer", exist_ok=True)
    os.chdir("H2O_dimer")

    water_dimer_idx = s26[1]
    water_dimer = create_s22_system(water_dimer_idx)

    Projection = FrozenDensityEmbedding(
        water_dimer,
        embed_mask=[1, 1, 1, 2, 2, 2],
        calc_base_ll=calc,
        calc_base_hl=calc
    )

    Projection.run()  

    os.chdir(cwd)  

    fdet_file = tmp_path / "H2O_dimer" / "fdet.txt"
    with open(fdet_file) as f:
        text = f.read()
        pattern = r"FDET Cycle:\s*(\d+)([\s\S]*?)(?=_{3})"
        cycles = re.findall(pattern, text)

        if not cycles:
            raise RuntimeError("No FDET cycle found in fdet.txt")
        
        last_cycle_num, block = cycles[-1]

        fdet_values = {"cycle": int(last_cycle_num),
                       "etot_F2A1": get_value("etot_F2A1", block),
                       "etot_F1A2": get_value("etot_F1A2",block),
                       "etot_current": get_value("etot_current", block),
                       "ediff": get_value("ediff", block)}
        
        num_regression.check(fdet_values, default_tolerance=dict(atol=1e-6))

