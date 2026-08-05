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
import traceback, sys
import tracemalloc
import numpy as np
import time

def timing_val(func):
    def wrapper(*args, **kwargs):
        '''source: http://www.daniweb.com/code/snippet368.html'''
        t1 = time.time()
        try:
            func(*args, **kwargs)
        except Exception as eee:
            print(f"""Something happened in {func.__name__} ASI 
                  {label}: {eee}\nAborting...""")
            traceback.print_tb(eee.__traceback__, limit=5, file=sys.stdout)
            MPI.COMM_WORLD.Abort(1)
        t2 = time.time()
        total_time = t2 - t1
        #root_print(f'Invoking Callback {func.__name__} Took {total_time:.4f} seconds')
    return wrapper

def timing_val_ret(func):
    def wrapper(*args, **kwargs):
        '''source: http://www.daniweb.com/code/snippet368.html'''
        t1 = time.time()
        try:
            result = func(*args, **kwargs)
        except Exception as eee:
            print(f"""Something happened in {func.__name__} ASI 
                  {label}: {eee}\nAborting...""")
            traceback.print_tb(eee.__traceback__, limit=5, file=sys.stdout)
            MPI.COMM_WORLD.Abort(1)
        t2 = time.time()
        total_time = t2 - t1
        #root_print(f'Invoking Callback {func.__name__} Took {total_time:.4f} seconds')
        return result
    return wrapper

@timing_val
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
        # Python-indexed spin and kpts:
        PiS = iS - 1
        PiK = iK - 1        

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
            if not (asi.dm_count in storage_dict.keys()):
                storage_dict[asi.dm_count] = {}
            #root_print(tracemalloc.get_traced_memory()[1]/(1024*1024))
            storage_dict[asi.dm_count][(PiS, PiK)] = data
            if (iS*iK) == (asi.n_local_ks):
                asi.dm_count = asi.dm_count + 1
    except Exception as eee:
        print(f"""Something happened in ASI dm_saving_callback
                  {label}: {eee}\nAborting...""")
        traceback.print_tb(eee.__traceback__, limit=5, file=sys.stdout)
        MPI.COMM_WORLD.Abort(1)

@timing_val
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

        # Python-indexed spin and kpts:
        PiS = iS - 1
        PiK = iK - 1

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
            storage_dict[(PiS, PiK)] = data
            #root_print(tracemalloc.get_traced_memory()[1]/(1024*1024))
    except Exception as eee:
        print(f"""Something happened in ASI ovlp_saving_callback 
                  {label}: {eee}\nAborting...""")
        traceback.print_tb(eee.__traceback__, limit=5, file=sys.stdout)
        MPI.COMM_WORLD.Abort(1)

@timing_val
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

        # Python-indexed spin and kpts:
        PiS = iS - 1
        PiK = iK - 1

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
            if not (asi.ham_count in storage_dict.keys()):
                storage_dict[asi.ham_count] = {}

            if asi.ham_count <= 2:
                storage_dict[asi.ham_count][(PiS, PiK)] = data

                if (iS*iK) == (asi.n_local_ks):
                    asi.ham_count = asi.ham_count + 1
                #root_print(tracemalloc.get_traced_memory()[1]/(1024*1024))
            else:
                #root_print(tracemalloc.get_traced_memory()[1]/(1024*1024))
                storage_dict[0][(PiS, PiK)] = storage_dict[1][(PiS, PiK)]
                #root_print(tracemalloc.get_traced_memory()[1]/(1024*1024))
                storage_dict[1][(PiS, PiK)] = storage_dict[2][(PiS, PiK)]
                #root_print(tracemalloc.get_traced_memory()[1]/(1024*1024))
                storage_dict[2][(PiS, PiK)] = data

    except Exception as eee:
        print(f"""Something happened in ASI ham_saving_callback {label}: 
                  {eee}\nAborting...""")
        traceback.print_tb(eee.__traceback__, limit=5, file=sys.stdout)
        MPI.COMM_WORLD.Abort(1)

@timing_val
def ham_saving_and_huzinaga_callback(aux, iK, iS, descr, data, matrix_descr_ptr):
    """Default callback for saving hamiltonian matrices

    This routine does some additional construction of the huzinaga matrix
    to allow the code to solve the Huzinaga problem.

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
    from embasi.huzinaga_projector import huzinaga_projector, get_abs_trunc_indices

    try:
        #asi, storage_dict, vemb, huz_dm, huz_ovlp, cnt_dict, ctxt_tag, descr_tag, label = cast(aux, py_object).value
        asi, storage_dict, atomsembed, cnt_dict, ctxt_tag, descr_tag, label = cast(aux, py_object).value

        # Python-indexed spin and kpts:
        PiS = iS - 1
        PiK = iK - 1

        atomsembed = atomsembed["atembed"]
        
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
            if not (asi.ham_count in storage_dict.keys()):
                storage_dict[asi.ham_count] = {}

            if asi.ham_count <= 3:
                storage_dict[asi.ham_count][(PiS, PiK)] = data

                if (iS*iK) == (asi.n_local_ks):
                    asi.ham_count = asi.ham_count + 1
                #root_print(tracemalloc.get_traced_memory()[1]/(1024*1024))
            else:
                #root_print(tracemalloc.get_traced_memory()[1]/(1024*1024))
                storage_dict[0][(PiS, PiK)] = storage_dict[1][(PiS, PiK)]
                #root_print(tracemalloc.get_traced_memory()[1]/(1024*1024))
                storage_dict[1][(PiS, PiK)] = storage_dict[2][(PiS, PiK)]
                #root_print(tracemalloc.get_traced_memory()[1]/(1024*1024))
                storage_dict[2][(PiS, PiK)] = data

            if atomsembed.truncate:
                vemb_supermol = atomsembed.fock_embedding_matrix[PiS, PiK]
                ovlp_supermol = atomsembed.huzinaga_ovlp_in[PiS, PiK]
                dm_supermol = atomsembed.huzinaga_dm_in[PiS, PiK]

                if atomsembed.abs_truncate:
                    fock_supermol = atomsembed.embedding_ham_in[PiS, PiK]

                    A_block_min, A_block_max, B_block_min, B_block_max = get_abs_trunc_indices(atomsembed)

                    fmat_supermol = fock_supermol[A_block_min:A_block_max,B_block_min:B_block_max]
                    dm_supermol = dm_supermol[B_block_min:B_block_max,B_block_min:B_block_max]
                    ovlp_supermol = ovlp_supermol[A_block_min:A_block_max,B_block_min:B_block_max]
                else:
                    fock_supermol = atomsembed.truncated_mat_to_full(data)
                    fmat_supermol = (fock_supermol + vemb_supermol)

                projector = huzinaga_projector(fmat_supermol, ovlp_supermol, dm_supermol,
                                               n_spins=atomsembed.fock_embedding_matrix.n_spins)

                if not(atomsembed.abs_truncate):
                    projector = atomsembed.full_mat_to_truncated(projector)

                asi.huzinaga_eq[(PiS, PiK)] = atomsembed.fock_embedding_matrix_trunc[PiS, PiK] + projector

            else:
                vemb = atomsembed.fock_embedding_matrix[PiS, PiK]
                ovlp = atomsembed.huzinaga_ovlp_in[PiS, PiK]
                dm = atomsembed.huzinaga_dm_in[PiS, PiK]

                asi.huzinaga_eq[(PiS, PiK)] = vemb + huzinaga_projector(data + vemb, ovlp, dm,
                                                                        n_spins=atomsembed.fock_embedding_matrix.n_spins)

    except Exception as eee:
        print(f"""Something happened in ASI ham_saving_callback {label}: 
                  {eee}\nAborting...""")
        traceback.print_tb(eee.__traceback__, limit=5, file=sys.stdout)
        MPI.COMM_WORLD.Abort(1)

@timing_val_ret
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
        asi, storage_dict, flag_huz, ctxt_tag, descr_tag, label = cast(aux, py_object).value

        # Python-indexed spin and kpts:
        PiS = iS - 1
        PiK = iK - 1

        # This is unfortunately very opaque - essentially, the huzinaga equation is formulated
        # in the hamiltonian saving callback as we need the fock matrix constructed during the
        # SCF cycle, which cannot be set outside of the invoked QM code.
        if flag_huz:
            if ((ctxt_tag is None) and (descr_tag is None)):
                if (PiS, PiK) in asi.huzinaga_eq.keys():
                    m = np.asfortranarray(asi.huzinaga_eq[(PiS, PiK)])
                else:
                    m = None
            else:
                m = asi.huzinaga_eq[(PiS, PiK)]
        else:
            if ((ctxt_tag is None) and (descr_tag is None)):
                m = np.asfortranarray(storage_dict[PiS, PiK]) if asi.scalapack.is_root(descr) else None
            else:
                m = storage_dict[PiS, PiK]

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

        # Set to empty dictionary again for re-population
        #asi.huzinaga_eq = {}

    except Exception as eee:
        print(f"""Something happened in ASI matrix_loading_callback {label}: 
                  {eee}\nAborting...""")
        traceback.print_tb(eee.__traceback__, limit=5, file=sys.stdout)
        MPI.COMM_WORLD.Abort(1)
