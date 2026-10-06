import pytest
import os
import sys
from pytest_regressions.data_regression import DataRegressionFixture
from pytest_regressions.num_regression import NumericRegressionFixture

sys.path.append(os.getcwd()+"/../../../")
print(sys.path)
from fhiaims_meoh_test_generator import FHIaims_projection_embedding_test

def test_level_shift_projection_serial(num_regression: NumericRegressionFixture, tmp_path):

    test = FHIaims_projection_embedding_test(projection="level-shift", test_dir=tmp_path)
    test.run_test()
    test_vals = test.output_ref_values()
    # PB_corr is mu (1e6) times the residual overlap of A's density with B's
    # projector, so its ~1e-6 eV value is numerical noise at the 1e-12 level.
    num_regression.check(test_vals, default_tolerance=dict(atol=1e-6),
                         tolerances={"PB_corr_energy": dict(atol=1e-5)})

    
