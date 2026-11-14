from ctypes import POINTER, byref, c_int, c_int64, c_int32, c_bool, \
                   c_char_p, c_double, c_void_p, CFUNCTYPE, py_object, \
                   cast, byref, Structure
from asi4py.pyasi import triang2herm_inplace, triang_packed2full_hermit, matrix_asi_to_numpy
from embasi.parallel_utils import root_print
from mpi4py import MPI
from scalapack4py.npscal import NPScal
from scalapack4py.npscal.npscal import _GEMR2D_NAMES
from embasi.kpoint_parallel import check_kpoint_grid
from scalapack4py.npscal.blacs_ctxt_management import BLACSGrid, descriptor_registry
from scalapack4py.array_types import nullable_ndpointer, ctypes2ndarray
import traceback, sys
import tracemalloc
import numpy as np


def _kpoint_tag(asi, tag, PiK):
    """Grid / descriptor tag of the matrices of k-point PiK

    With several k-points, FHI-aims may put each k-point on its own BLACS
    grid (a subset of the processes), so grids and descriptors are tagged
    per k-point. Unchanged for a single k-point.
    """
    return tag if (tag is None or asi.n_kpts == 1) else f"{tag}_k{PiK}"


def _debug_callback(asi, label, iK, iS, descr, data_ptr=None, descr_ptr=None):
    """EMBASI_DEBUG_CALLBACKS=1: print which (k, spin) blocks each rank receives"""
    import os
    if os.environ.get("EMBASI_DEBUG_CALLBACKS") not in ("1", "2"):
        return
    info = "descr=NULL"
    if not descr and os.environ.get("EMBASI_DEBUG_CHECKSUM") and data_ptr is not None:
        m = matrix_asi_to_numpy(asi, descr, data_ptr, descr_ptr)
        m = np.asarray(m)
        info += f" sum={np.sum(m).real:.10e} |m|={np.linalg.norm(m):.10e} m00={complex(m.flat[0]).real:.6e}"
    if descr:
        d = asi.scalapack.wrap_blacs_desc(descr)
        nprow, npcol, myrow, mycol = asi.scalapack.blacs_gridinfo(d.ctxt)[:4]
        info = f"ctxt={d.ctxt} grid={nprow}x{npcol} me=({myrow},{mycol}) m={d.m} mb={d.mb}"
    print(f"[rank {MPI.COMM_WORLD.Get_rank()}] {label} iK={iK} iS={iS} n_local_ks={asi.n_local_ks} "
          f"local_ks={list(asi.local_ks) if hasattr(asi, 'local_ks') else '?'} {info}", flush=True)


class RawKBlock:
    """A local block FHI-aims passed before EmbASI's k-point group grid
    existed (the first run with per-k-point grids): the copied local array
    and its layout, wrapped as NPScal once the grid is made (kpoint_parallel)"""
    def __init__(self, loc, descr, layout, PiK, tag):
        self.loc, self.descr, self.layout, self.PiK, self.tag = loc, descr, layout, PiK, tag


def _local_block(asi, data, descr_cast):
    """This process's local block of a BLACS-distributed matrix from the QM
    code, as a Fortran-ordered ndarray (a copy: the buffer is the QM code's).

    For complex (k-point) matrices the buffer holds (Re, Im) pairs of doubles,
    so twice locrow * loccol doubles are read and viewed as complex128. For
    real (Gamma) matrices this is the copy NPScal made from the pointer.
    """
    n_loc = descr_cast.locrow * descr_cast.loccol
    if asi.is_hamiltonian_real:
        flat = np.ctypeslib.as_array(data, shape=(n_loc,)).copy()
    else:
        flat = np.ctypeslib.as_array(data, shape=(2 * n_loc,)).copy().view(np.complex128)
    return flat.reshape((descr_cast.locrow, descr_cast.loccol), order="F")


def _wrap_block(asi, data, descr_cast, ctxt_tag, descr_tag, PiK, label):
    """NPScal (or, see RawKBlock, a deferred block) for a BLACS-distributed matrix"""
    MP, NP, myrow, mycol = asi.scalapack.blacs_gridinfo(descr_cast.ctxt)[:4]
    m, n, mb, nb = descr_cast.m, descr_cast.n, descr_cast.mb, descr_cast.nb
    rsrc, csrc = descr_cast.rsrc, descr_cast.csrc
    tag = _kpoint_tag(asi, descr_tag, PiK)
    loc = _local_block(asi, data, descr_cast)
    if asi.n_kpts > 1 and asi.n_local_ks != asi.n_spin * asi.n_kpts:
        # One BLACS grid per k-point on a subset of the processes
        kname = f"{ctxt_tag}_kgrid"
        if not BLACSGrid.is_registered(kname):
            # EmbASI's grid is made after this run, collectively (BLACS needs
            # every process; callbacks run only on a k-point's own processes)
            return RawKBlock(loc, (m, n, mb, nb, rsrc, csrc), (MP, NP, myrow, mycol), PiK, tag)
        grid = check_kpoint_grid(BLACSGrid.get(kname), asi, descr_cast, label)
    else:
        grid = BLACSGrid.get_or_create(_kpoint_tag(asi, ctxt_tag, PiK), asi.scalapack, nprow=MP, npcol=NP)

    # Stash the first ASI-native descriptor seen for this context as a
    # reference on the grid (see rechunk(x, like=grid))
    if grid.default_mb is None:
        grid.set_default_block_size(mb, nb)
    descriptor_registry.get_or_create(tag, m, n, mb, nb, rsrc, csrc)
    # From the ndarray, whose dtype (float64 or complex128) NPScal keeps;
    # from the pointer it would read the data as c_double
    arr = NPScal(loc_array=loc, grid=grid, descriptor=tag, lib=asi.scalapack)
    return _on_grid_blocks(_full_hermitian(arr), grid)


def _full_hermitian(arr):
    """The full Hermitian matrix from one passed as a single triangle

    On per-k-point ScaLAPACK grids FHI-aims passes the complex overlap with
    only its upper triangle filled (the strictly lower one is zero), as
    ScaLAPACK's Hermitian solvers read one triangle. A Hermitian matrix
    cannot have one strict triangle zero and the other not, so that is taken
    as the sign; full matrices (e.g. the Hamiltonian) are returned as they are.
    Collective over the matrix's grid.
    """
    from embasi.ks_array import dagger
    lower = float(np.linalg.norm(np.tril(arr, -1)))
    upper = float(np.linalg.norm(np.triu(arr, 1)))
    if lower == 0.0 and upper > 0.0:
        return np.triu(arr) + dagger(np.triu(arr, 1))
    if upper == 0.0 and lower > 0.0:
        return np.tril(arr) + dagger(np.tril(arr, -1))
    return arr


def _on_grid_blocks(arr, grid):
    """arr on the grid's default block size: the QM code may pass matrices
    of one grid with different block sizes (FHI-aims on 2 x 1 per-k-point
    grids: 16 x 16 and 8 x 16), which NPScal cannot combine element-wise.
    A no-op when they already match, as at Gamma."""
    from scalapack4py.npscal import rechunk
    if (arr.descr.mb, arr.descr.nb) != (grid.default_mb, grid.default_nb):
        arr = rechunk(arr, mb=grid.default_mb, nb=grid.default_nb)
    return arr


def _block_received(asi, counter, expected):
    """Count a received (spin, k-point) block; True once this process has
    received the `expected` blocks of the current step"""
    received = getattr(asi, counter + "_blocks", 0) + 1
    complete = received == expected
    setattr(asi, counter + "_blocks", 0 if complete else received)
    return complete
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
        _debug_callback(asi, label, iK, iS, descr, data, matrix_descr_ptr)
        if __import__('os').environ.get('EMBASI_DEBUG_CALLBACKS') == '2':
            return   # log-only: see every callback without processing it
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
                # Several k-points: FHI-aims may hold different k-points on
                # different ranks, so every rank keeps what it is passed
                if MPI.COMM_WORLD.Get_rank() == 0 or asi.n_kpts > 1:
                    data = data.copy()
            else:
                descr_cast = asi.scalapack.wrap_blacs_desc(descr)
                data = _wrap_block(asi, data, descr_cast, ctxt_tag, descr_tag, PiK, label)

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
            if _block_received(asi, "dm", asi.n_local_ks):
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
        _debug_callback(asi, label, iK, iS, descr, data, matrix_descr_ptr)
        if __import__('os').environ.get('EMBASI_DEBUG_CALLBACKS') == '2':
            return   # log-only: see every callback without processing it

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
                # Several k-points: FHI-aims may hold different k-points on
                # different ranks, so every rank keeps what it is passed
                if MPI.COMM_WORLD.Get_rank() == 0 or asi.n_kpts > 1:
                    data = data.copy()
            else:
                descr_cast = asi.scalapack.wrap_blacs_desc(descr)
                data = _wrap_block(asi, data, descr_cast, ctxt_tag, descr_tag, PiK, label)

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
        _debug_callback(asi, label, iK, iS, descr, data, matrix_descr_ptr)
        if __import__('os').environ.get('EMBASI_DEBUG_CALLBACKS') == '2':
            return   # log-only: see every callback without processing it

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
                # Several k-points: FHI-aims may hold different k-points on
                # different ranks, so every rank keeps what it is passed
                if MPI.COMM_WORLD.Get_rank() == 0 or asi.n_kpts > 1:
                    data = data.copy()
            else:
                descr_cast = asi.scalapack.wrap_blacs_desc(descr)
                data = _wrap_block(asi, data, descr_cast, ctxt_tag, descr_tag, PiK, label)

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

                if _block_received(asi, "ham", asi.n_local_ks if descr else asi.n_spin * asi.n_kpts):
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
    from embasi.embedding_projectors import huzinaga_projector, get_abs_trunc_indices

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
                # Several k-points: FHI-aims may hold different k-points on
                # different ranks, so every rank keeps what it is passed
                if MPI.COMM_WORLD.Get_rank() == 0 or asi.n_kpts > 1:
                    data = data.copy()
            else:
                descr_cast = asi.scalapack.wrap_blacs_desc(descr)
                data = _wrap_block(asi, data, descr_cast, ctxt_tag, descr_tag, PiK, label)

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

                if _block_received(asi, "ham", asi.n_local_ks if descr else asi.n_spin * asi.n_kpts):
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
                if not isinstance(m, NPScal):
                    raise Exception("BLACS-distributed data not recognised")

                src_descr = m.descr
                dest_descr = asi.scalapack.wrap_blacs_desc(descr)

                # A trailing axis of 2 (Re, Im) gives a complex view for k-point matrices
                loc_shape = (dest_descr.locrow, dest_descr.loccol) if asi.is_hamiltonian_real \
                    else (dest_descr.locrow, dest_descr.loccol, 2)
                data = ctypes2ndarray(data, shape=loc_shape).T

                # p?gemr2d for the matrix's dtype: pzgemr2d for complex
                # (k-point) matrices, pdgemr2d for real (Gamma) ones
                gemr2d = getattr(asi.scalapack, _GEMR2D_NAMES[np.dtype(m.loc_array.dtype)])
                gemr2d(asi.n_basis, asi.n_basis,
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
