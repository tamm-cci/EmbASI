import pytest
import os
import sys

sys.path.append(os.getcwd()+"/../../../")
print(sys.path)
from fhiaims_meoh_test_generator import FHIaims_projection_embedding_test

def test_level_shift_projection_serial(data_regression, tmp_path):

    test = FHIaims_projection_embedding_test(projection="level-shift", test_dir=tmp_path)
    test.run_test()
    test_vals = test.output_ref_values()
    data_regression.check(test_vals)
    
