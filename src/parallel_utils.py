from mpi4py import MPI

def root_print(*args, **kwargs):
    "Prints only from the root node"

    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        print(*args, **kwargs)

# TODO: Dynamic printing routine for given tasks/all tasks.

#def parprint(*args, **kwargs):
#    "Prints on given task, list of tasks, for all tasks"
#
#    rank = MPI.COMM_WORLD.Get_rank()