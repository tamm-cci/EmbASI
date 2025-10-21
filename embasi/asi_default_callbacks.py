from ctypes import POINTER, byref, c_int, c_int64, c_int32, c_bool, \
                   c_char_p, c_double, c_void_p, CFUNCTYPE, py_object, \
                   cast, byref, Structure
from asi4py.pyasi import triang2herm_inplace, triang_packed2full_hermit, matrix_asi_to_numpy
from embasi.parallel_utils import root_print
from mpi4py import MPI
from scalapack4py.npscal import NPScal
from scalapack4py.npscal.blacs_ctxt_management import DESCR_Register, CTXT_Register
from scalapack4py.npscal.blacs_ctxt_management import BLACSContextManager, BLACSDESCRManager
from scalapack4py.array_types import nullable_ndpointer, ctypes2ndarray
import tracemalloc
import numpy as np
import traceback, sys

def dm_saving_callback(aux, iK, iS, descr, data, matrix_descr_ptr):
    """Default callback for saving density matrices

    Callback function from ASI to be registered and invoked by 
    a given QM code. Saves density matrices from a given ASI_run()
    call to a dictionary of np.ndarray arrays, indexed by the 
    number of density matrices exported, k-point, and spin channel.

    Code derived from the default saving callback from asi4py
    
    Parameters
    ----------
    aux: Object
        Auxiliary object passed to callback 
    iK: c_int
        k-point index of matrix
    iS: c_int
        Spin channel index of matrix        
    descr: c_types.POINTER(c_int)
        Pointer to BLACS descriptor of matrix
    data: c_types.POINTER
        Pointer to dble/cdble matrix
    matrix_descr_ptr: c_types.POINTER(c_int)
        Numerical value indexing matrix shape (See: ASI docs)

    """
    try:
        asi, storage_dict, cnt_dict, ctxt_tag, descr_tag, label = cast(aux, py_object).value

        if asi.is_hamiltonian_real:
            data_shape = (asi.n_basis,asi.n_basis)
        else:
            data_shape = (asi.n_basis,asi.n_basis, 2)

        if (matrix_descr_ptr.contents.storage_type not in {1,2}):
            if ((ctxt_tag is None) and (descr_tag is None)):
                data = matrix_asi_to_numpy(asi, descr, data, matrix_descr_ptr)
                if MPI.COMM_WORLD.Get_rank() == 0:
                    data = data.copy()
            else:
                if not(CTXT_Register.check_register(ctxt_tag)):
                    descr_cast = asi.scalapack.wrap_blacs_desc(descr)
                    MP = asi.scalapack.blacs_gridinfo(descr_cast.ctxt)[0]
                    NP = asi.scalapack.blacs_gridinfo(descr_cast.ctxt)[1]
                    ctxt = BLACSContextManager(ctxt_tag, MP, NP, asi.scalapack)
                                                
                if not(DESCR_Register.check_register(descr_tag)):
                    descr_cast = asi.scalapack.wrap_blacs_desc(descr)
                    m, n, mb, nb = descr_cast.m, descr_cast.n, descr_cast.mb, descr_cast.nb
                    rsrc, csrc, lld = descr_cast.rsrc, descr_cast.csrc, descr_cast.lld
                    descr = BLACSDESCRManager(ctxt_tag, descr_tag, asi.scalapack, m, n,
                                              mb, nb, rsrc, csrc, lld)

                data = NPScal(loc_array=data, ctxt_tag=ctxt_tag, descr_tag=descr_tag, lib=asi.scalapack)

        elif (matrix_descr_ptr.contents.storage_type in {1,2}):
            assert not descr, """default_saving_callback supports only dense full 
                                 ScaLAPACK arrays"""
            assert matrix_descr_ptr.contents.matrix_type == 1, \
                "Triangular packed storage is supported only for hermitian matrices"
            data = asi.scalapack.gather_numpy(descr, data, asi.matrix_shape)

            data = matrix_asi_to_numpy(asi, descr, data, matrix_descr_ptr)

            uplo = {1:'L',2:'U'}[matrix_descr_ptr.contents.storage_type]
            data = triang_packed2full_hermit(data, asi.n_basis,
                                             asi.is_hamiltonian_real, uplo)

        if data is not None:
            #root_print(tracemalloc.get_traced_memory()[1]/(1024*1024))
            asi.dm_count += 1
            storage_dict[(asi.dm_count, iK, iS)] = data
    except Exception as eee:
        print(f"""Something happened in ASI dm_saving_callback
                  {label}: {eee}\nAborting...""")
        traceback.print_tb(eee.__traceback__, limit=5, file=sys.stdout)
        MPI.COMM_WORLD.Abort(1)

def ovlp_saving_callback(aux, iK, iS, descr, data, matrix_descr_ptr):
    """Default callback for saving density matrices

    Callback function from ASI to be registered and invoked by 
    a given QM code. Saves density matrices from a given ASI_run()
    call to a dictionary of np.ndarray arrays, indexed by the 
    number of density matrices exported, k-point, and spin channel.

    Code derived from the default saving callback from asi4py
    
    Parameters
    ----------
    aux: Object
        Auxiliary object passed to callback 
    iK: c_int
        k-point index of matrix
    iS: c_int
        Spin channel index of matrix        
    descr: c_types.POINTER(c_int)
        Pointer to BLACS descriptor of matrix
    data: c_types.POINTER
        Pointer to dble/cdble matrix
    matrix_descr_ptr: c_types.POINTER(c_int)
        Numerical value indexing matrix shape (See: ASI docs)

    """
    try:
        asi, storage_dict, ctxt_tag, descr_tag, label = cast(aux, py_object).value

        if asi.is_hamiltonian_real:
            data_shape = (asi.n_basis,asi.n_basis)
        else:
            data_shape = (asi.n_basis,asi.n_basis, 2)

        if (matrix_descr_ptr.contents.storage_type not in {1,2}):
            if ((ctxt_tag is None) and (descr_tag is None)):
                data = matrix_asi_to_numpy(asi, descr, data, matrix_descr_ptr)
                if MPI.COMM_WORLD.Get_rank() == 0:
                    data = data.copy()
            else:
                if not(CTXT_Register.check_register(ctxt_tag)):
                    descr_cast = asi.scalapack.wrap_blacs_desc(descr)
                    MP = asi.scalapack.blacs_gridinfo(descr_cast.ctxt)[0]
                    NP = asi.scalapack.blacs_gridinfo(descr_cast.ctxt)[1]
                    ctxt = BLACSContextManager(ctxt_tag, MP, NP, asi.scalapack)
                                                
                if not(DESCR_Register.check_register(descr_tag)):
                    descr_cast = asi.scalapack.wrap_blacs_desc(descr)
                    m, n, mb, nb = descr_cast.m, descr_cast.n, descr_cast.mb, descr_cast.nb
                    rsrc, csrc, lld = descr_cast.rsrc, descr_cast.csrc, descr_cast.lld
                    descr = BLACSDESCRManager(ctxt_tag, descr_tag, asi.scalapack, m, n,
                                              mb, nb, rsrc, csrc, lld)
                data = NPScal(loc_array=data, ctxt_tag=ctxt_tag, descr_tag=descr_tag, lib=asi.scalapack)

        elif (matrix_descr_ptr.contents.storage_type in {1,2}):
            assert not descr, """default_saving_callback supports only dense full 
                                 ScaLAPACK arrays"""
            assert matrix_descr_ptr.contents.matrix_type == 1, \
                "Triangular packed storage is supported only for hermitian matrices"

            uplo = {1:'L',2:'U'}[matrix_descr_ptr.contents.storage_type]
            data = triang_packed2full_hermit(data, asi.n_basis,
                                             asi.is_hamiltonian_real, uplo)

        if data is not None:
            #assert len(data.shape) == 2
            storage_dict[(iK, iS)] = data
            #root_print(tracemalloc.get_traced_memory()[1]/(1024*1024))
    except Exception as eee:
        print(f"""Something happened in ASI ovlp_saving_callback 
                  {label}: {eee}\nAborting...""")
        traceback.print_tb(eee.__traceback__, limit=5, file=sys.stdout)
        MPI.COMM_WORLD.Abort(1)

def ham_saving_callback(aux, iK, iS, descr, data, matrix_descr_ptr):
    """Default callback for saving hamiltonian matrices

    Callback function from ASI to be registered and invoked by 
    a given QM code. Saves density matrices from a given ASI_run()
    call to a dictionary of np.ndarray arrays, indexed by the 
    number of hamiltonian matrices exported, k-point, and spin channel.

    Code derived from the default saving callback from asi4py

      Parameters
    ----------
    aux: Object
        Auxiliary object passed to callback 
    iK: c_int
        k-point index of matrix
    iS: c_int
        Spin channel index of matrix        
    descr: c_types.POINTER(c_int)
        Pointer to BLACS descriptor of matrix
    data: c_types.POINTER
        Pointer to dble/cdble matrix
    matrix_descr_ptr: c_types.POINTER(c_int)
        Numerical value indexing matrix shape (See: ASI docs)

    """
    try:
        asi, storage_dict, cnt_dict, ctxt_tag, descr_tag, label = cast(aux, py_object).value
        
        if asi.is_hamiltonian_real:
            data_shape = (asi.n_basis,asi.n_basis) 
        else:
            data_shape = (asi.n_basis,asi.n_basis, 2)

        # ASI_STORAGE_TYPE_TRIL,ASI_STORAGE_TYPE_TRIU
        if (matrix_descr_ptr.contents.storage_type not in {1,2}):
            if ((ctxt_tag is None) and (descr_tag is None)):
                data = matrix_asi_to_numpy(asi, descr, data, matrix_descr_ptr)
                if MPI.COMM_WORLD.Get_rank() == 0:
                    data = data.copy()
            else:
                if not(CTXT_Register.check_register(ctxt_tag)):
                    descr_cast = asi.scalapack.wrap_blacs_desc(descr)
                    MP = asi.scalapack.blacs_gridinfo(descr_cast.ctxt)[0]
                    NP = asi.scalapack.blacs_gridinfo(descr_cast.ctxt)[1]
                    ctxt = BLACSContextManager(ctxt_tag, MP, NP, asi.scalapack)
                                                
                if not(DESCR_Register.check_register(descr_tag)):
                    descr_cast = asi.scalapack.wrap_blacs_desc(descr)
                    m, n, mb, nb = descr_cast.m, descr_cast.n, descr_cast.mb, descr_cast.nb
                    rsrc, csrc, lld = descr_cast.rsrc, descr_cast.csrc, descr_cast.lld
                    descr = BLACSDESCRManager(ctxt_tag, descr_tag, asi.scalapack, m, n,
                                              mb, nb, rsrc, csrc, lld)
                data = NPScal(loc_array=data, ctxt_tag=ctxt_tag, descr_tag=descr_tag, lib=asi.scalapack)

        elif (matrix_descr_ptr.contents.storage_type in {1,2}): #
            assert not descr, """default_saving_callback supports only dense 
                                 full ScaLAPACK arrays"""
            assert matrix_descr_ptr.contents.matrix_type == 1, \
                "Triangular packed storage is supported only for hermitian matrices"
            uplo = {1:'L',2:'U'}[matrix_descr_ptr.contents.storage_type]
            data = triang_packed2full_hermit(data, asi.n_basis, 
                                             asi.is_hamiltonian_real, uplo)

        if data is not None:
            #assert len(data.shape) == 2
            if asi.ham_count < 3:
                asi.ham_count = asi.ham_count + 1
                storage_dict[(asi.ham_count, iK, iS)] = data
                #root_print(tracemalloc.get_traced_memory()[1]/(1024*1024))
            else:
                storage_dict.pop((1, iK, iS))
                #root_print(tracemalloc.get_traced_memory()[1]/(1024*1024))
                storage_dict[(1, iK, iS)] = storage_dict[(2, iK, iS)]
                #root_print(tracemalloc.get_traced_memory()[1]/(1024*1024))
                storage_dict[(2, iK, iS)] = storage_dict[(3, iK, iS)]
                #root_print(tracemalloc.get_traced_memory()[1]/(1024*1024))
                storage_dict[(3, iK, iS)] = data
                #root_print(tracemalloc.get_traced_memory()[1]/(1024*1024))
    except Exception as eee:
        print(f"""Something happened in ASI ham_saving_callback {label}: 
                  {eee}\nAborting...""")
        traceback.print_tb(eee.__traceback__, limit=5, file=sys.stdout)
        MPI.COMM_WORLD.Abort(1)

def matrix_loading_callback(aux, iK, iS, descr, data, matrix_descr_ptr):
    """Default callback for loading matrices

    Callback function from ASI to be registered and invoked by 
    a given QM code. Loads a given matrix at a given point in the
    code where the callback is invokes

    Code derived from the default saving callback from asi4py

    Parameters
    ----------
    aux: Object
        Auxiliary object passed to callback 
    iK: c_int
        k-point index of matrix
    iS: c_int
        Spin channel index of matrix        
    descr: c_types.POINTER(c_int)
        Pointer to BLACS descriptor of matrix
    data: c_types.POINTER
        Pointer to dble/cdble matrix
    matrix_descr_ptr: c_types.POINTER(c_int)
        Numerical value indexing matrix shape (See: ASI docs)

    """

    try:
        asi, storage_dict, ctxt_tag, descr_tag, label = cast(aux, py_object).value

        if ((ctxt_tag is None) and (descr_tag is None)):
            m = np.asfortranarray(storage_dict[(iK, iS)]) if asi.scalapack.is_root(descr) else None
        else:
            m = storage_dict[(iK, iS)]

        # ASI_STORAGE_TYPE_TRIL,ASI_STORAGE_TYPE_TRIU
        if (matrix_descr_ptr.contents.storage_type not in {1,2}):
            if ((ctxt_tag is None) and (descr_tag is None)):
                asi.scalapack.scatter_numpy(m, descr, data, asi.hamiltonian_dtype)
            else:
                if not(CTXT_Register.check_register(ctxt_tag)):
                    raise Exception("Context not recognised")
                                                
                if not(DESCR_Register.check_register(descr_tag)):
                    raise Exception("BLACS descriptor not recognised")

                src_descr = DESCR_Register.get_register(descr_tag)
                dest_descr = asi.scalapack.wrap_blacs_desc(descr)

                data = ctypes2ndarray(data, shape=(dest_descr.locrow, dest_descr.loccol)).T

                asi.scalapack.pdgemr2d(asi.n_basis, asi.n_basis,
                                       m.loc_array, 1, 1, src_descr,
                                       data, 1, 1, dest_descr,
                                       dest_descr.ctxt)

            return 1
        else:
            asi.scalapack.scatter_numpy(m, descr, data, asi.hamiltonian_dtype)
            return 1 # signal that  matrix has been loaded


    except Exception as eee:
        print(f"""Something happened in ASI matrix_loading_callback {label}: 
                  {eee}\nAborting...""")
        traceback.print_tb(eee.__traceback__, limit=5, file=sys.stdout)
        MPI.COMM_WORLD.Abort(1)
