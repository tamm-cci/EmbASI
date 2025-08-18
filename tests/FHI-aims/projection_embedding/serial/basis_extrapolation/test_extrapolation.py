import pytest
import os
import sys
from pytest_regressions.data_regression import DataRegressionFixture
from pytest_regressions.num_regression import NumericRegressionFixture

sys.path.append(os.getcwd() + "/../../../")
print(sys.path)
from fhiaims_extrapolate_test_generator import FHIaims_projection_embedding_extrapolate_test


def test_basis_extrapolation_serial(num_regression: NumericRegressionFixture, tmp_path):
    test = FHIaims_projection_embedding_extrapolate_test(test_dir=tmp_path)
    test.run_test()
    test_vals = test.output_ref_values()
    num_regression.check(test_vals, default_tolerance=dict(atol=1e-6))

