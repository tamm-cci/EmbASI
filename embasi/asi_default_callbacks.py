from ctypes import POINTER, byref, c_int, c_int64, c_int32, c_bool, \
                   c_char_p, c_double, c_void_p, CFUNCTYPE, py_object, \
                   cast, byref, Structure
from asi4py.pyasi import triang2herm_inplace, triang_packed2full_hermit
from mpi4py import MPI
import ctypes

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
        asi, storage_dict, cnt_dict, label = cast(aux, py_object).value

        if asi.is_hamiltonian_real:
            data_shape = (asi.n_basis,asi.n_basis)
        else:
            data_shape = (asi.n_basis,asi.n_basis, 2)

        if (matrix_descr_ptr.contents.storage_type not in {1,2}):
            data = asi.scalapack.gather_numpy(descr, data, data_shape)
        elif (matrix_descr_ptr.contents.storage_type in {1,2}):
            assert not descr, """default_saving_callback supports only dense full 
                                 ScaLAPACK arrays"""
            assert matrix_descr_ptr.contents.matrix_type == 1, \
                "Triangular packed storage is supported only for hermitian matrices"

            uplo = {1:'L',2:'U'}[matrix_descr_ptr.contents.storage_type]
            data = triang_packed2full_hermit(data, asi.n_basis,
                                             asi.is_hamiltonian_real, uplo)

        if data is not None:
            asi.dm_count += 1
            assert len(data.shape) == 2
            storage_dict[(asi.dm_count, iK, iS)] = data.copy()

    except Exception as eee:
        print(f"""Something happened in ASI default_saving_callback 
                  {label}: {eee}\nAborting...""")
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
        asi, storage_dict, cnt_dict, label = cast(aux, py_object).value
        
        if asi.is_hamiltonian_real:
            data_shape = (asi.n_basis,asi.n_basis) 
        else:
            data_shape = (asi.n_basis,asi.n_basis, 2)

        # ASI_STORAGE_TYPE_TRIL,ASI_STORAGE_TYPE_TRIU
        if (matrix_descr_ptr.contents.storage_type not in {1,2}):
            data = asi.scalapack.gather_numpy(descr, data, data_shape)
        elif (matrix_descr_ptr.contents.storage_type in {1,2}): #
            assert not descr, """default_saving_callback supports only dense 
                                 full ScaLAPACK arrays"""
            assert matrix_descr_ptr.contents.matrix_type == 1, \
                "Triangular packed storage is supported only for hermitian matrices"
            uplo = {1:'L',2:'U'}[matrix_descr_ptr.contents.storage_type]
            data = triang_packed2full_hermit(data, asi.n_basis, 
                                             asi.is_hamiltonian_real, uplo)

        if data is not None:
            assert len(data.shape) == 2
            if asi.ham_count < 3:
                asi.ham_count = asi.ham_count + 1
                storage_dict[(asi.ham_count, iK, iS)] = data.copy()
            else:
                storage_dict.pop((1, iK, iS))
                storage_dict[(1, iK, iS)] = storage_dict[(2, iK, iS)].copy()
                storage_dict[(2, iK, iS)] = storage_dict[(3, iK, iS)].copy()
                storage_dict[(3, iK, iS)] = data.copy()            

    except Exception as eee:
        print(f"""Something happened in ASI default_saving_callback {label}: 
                  {eee}\nAborting...""")
        MPI.COMM_WORLD.Abort(1)
