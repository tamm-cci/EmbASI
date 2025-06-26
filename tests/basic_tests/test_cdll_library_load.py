import pytest

# Basic test for ensuring the relevant QM library is correctly loaded
@pytest.mark.order(1)
def test_load_aims_library():

    def load_aims_library():
        from ctypes import CDLL, RTLD_GLOBAL
        import os

        lib_path = os.environ["ASI_LIB_PATH"]
        lib = CDLL(lib_path, mode=RTLD_GLOBAL)

    load_aims_library()
        
        
