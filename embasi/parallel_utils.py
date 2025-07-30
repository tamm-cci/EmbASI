from mpi4py import MPI
import numpy as np

def root_print(*args, **kwargs):
    """Executes print function on head node only.

    Parameters
    ----------
    *args
        Standard arguments for built-in print().
    **kwargs
        Standard  additional arguments for built-in print().

    """
    import sys

    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        print(*args, **kwargs)

    sys.stdout.flush()

def mpi_bcast_matrix_storage(data_dict, nrows, ncols):
    """Broadcasts a dictionary of numpy arrays from root node.

    The dictionaries of matrix quantities communicated by ASI 
    are only stored on the root MPI process. This means each
    node does not store its own copy of the relevant property.
    If the calculation is initialised from quantities communicated 
    from a previous ASI library call, a crash will ensue because
    processes other than the head node do not have a copy of 
    the relevant matrix. This process function is necessary to
    ensure each MPI processs posssesses a copy of an input 
    data_dictionary.

    Parameters
    ----------
    data_dict: dict of nrowsxncols np.ndarrays
        Dictionary of numpy arrays communicate from ASI
    nrows: int
        Number of rows in the matrices of data_dict
    ncols: int
        Number of columns in the matrices of data_dict
    Returns
    -------
    data_dict: dict of nrowsxncols np.ndarrays
        Input dictionary broadcast to all nodes

    """

    if MPI.COMM_WORLD.Get_rank() == 0:
        storage_keys = data_dict
        data = np.array(list(storage_keys), dtype=np.int16)
        data_shape = np.array(data.shape, dtype=np.int16)
    else:
        data_shape = np.array([0,0], dtype=np.int16)

    # Broadcast the size of the data
    MPI.COMM_WORLD.Bcast([data_shape, MPI.INT16_T], root=0)

    if MPI.COMM_WORLD.Get_rank() == 0:
        data = np.array(list(storage_keys), dtype=np.int16)
    else:
        data = np.zeros(tuple(data_shape), dtype=np.int16)

    MPI.COMM_WORLD.Bcast([data, MPI.INT16_T], root=0)
    data_dict_keys = data

    for data_key in data_dict_keys:

        if MPI.COMM_WORLD.Get_rank() == 0:
            data_buf = data_dict[tuple(data_key)]
        else:
            data_buf = np.zeros((nrows, ncols), dtype=np.float64)

        MPI.COMM_WORLD.Bcast([data_buf, MPI.DOUBLE], root=0)

        if MPI.COMM_WORLD.Get_rank() != 0:
            data_dict[tuple(data_key)] = data_buf.copy()

    return data_dict

def mpi_bcast_matrix(array):
    """Broadcasts a numpy arrays from root node.

    Parameters
    ----------
    array: nrowsxncols np.ndarrays
        ndarray to communicate to all
    nrows: int
        Number of rows in the matrices of data_dict
    ncols: int
        Number of columns in the matrices of data_dict
    Returns
    -------
    data_dict: nrowsxncols np.ndarrays
        np array on all nodes

    """

    if MPI.COMM_WORLD.Get_rank() == 0:
        dim = len(np.shape(array))
    else:
        dim = 0

    dim = mpi_bcast_integer(dim)

    if MPI.COMM_WORLD.Get_rank() == 0:
        data_shape = np.array(np.shape(array), dtype=np.int16)
    else:
        data_shape = np.zeros((dim), dtype=np.int16)

    # Broadcast the size of the data
    MPI.COMM_WORLD.Bcast([data_shape, MPI.INT16_T], root=0)
    
    if MPI.COMM_WORLD.Get_rank() == 0:
        data_buf = array
    else:
        data_buf = np.zeros(data_shape, dtype=np.float64)

    MPI.COMM_WORLD.Bcast([data_buf, MPI.DOUBLE], root=0)

    return data_buf

def mpi_bcast_integer(val):
    """Broadcasts an integer from the root process to all other processes.

    Parameters
    ----------
    val: int
        Integer value to communicate
    Returns
    -------
    int_buf[0]: int
        Input integer broadcast to all nodes
    """

    if MPI.COMM_WORLD.Get_rank() == 0:
        int_buf = np.full(1, val, dtype=int)
    else:
        int_buf = np.full(1, 0, dtype=int)

    MPI.COMM_WORLD.Bcast(int_buf)

    return int_buf[0]
 
