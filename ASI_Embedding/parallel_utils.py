from mpi4py import MPI
import numpy as np

def root_print(*args, **kwargs):
    "Prints only from the root node"

    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        print(*args, **kwargs)

def mpi_bcast_matrix_storage(data_dict, nrows, ncols):
    # Temporary cludge to communicate DMs from root to other processes,
    # which somehow aren't stored on any rank other than 0.

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

def mpi_bcast_integer(val):

    if MPI.COMM_WORLD.Get_rank() == 0:
        int_buf = np.full(1, val, dtype=int)
    else:
        int_buf = np.full(1, 0, dtype=int)

    MPI.COMM_WORLD.Bcast(int_buf)

    return int_buf[0]
