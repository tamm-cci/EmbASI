"""k-point parallelism: combining per-k-point quantities across process groups

When there are fewer k-points than MPI processes, FHI-aims splits the
processes into groups, one BLACS grid per k-point, so each process holds
only its own (spin, k-point) blocks; with more k-points than processes,
each process holds whole k-points. Every matrix operation in EmbASI is
then local to a (spin, k-point) block, and only scalars and short vectors
(traces, band energies, SPADE singular values, band edges, DIIS products)
couple the k-points. These are combined here.

With every block on every process (serial runs, or parallel runs at a
single k-point) the result equals the local one.
"""
from mpi4py import MPI


def gather_by_key(local, comm=MPI.COMM_WORLD):
    """{(spin, k-point): value} from every process, one value per key

    Every process of a k-point group holds the same value for its blocks
    (NPScal reductions are already complete over the group's grid), so the
    first copy of each key is kept rather than summing over processes.
    Collective over comm; values should be small (scalars, short vectors).
    """
    merged = {}
    for part in comm.allgather(local):
        for key, value in part.items():
            merged.setdefault(key, value)
    return merged


def adopt_raw_kblocks(asi, ctxt_tag, stores):
    """Make EmbASI's grid for this process's k-point group and wrap deferred blocks

    Collective over MPI.COMM_WORLD. With no more k-points than processes,
    FHI-aims puts each k-point on its own BLACS grid over a subset of the
    processes. Which processes hold which k-point is only known once its
    first run has started (ASI reports placeholders before), so during that
    run the callbacks keep the blocks raw (RawKBlock) with FHI-aims' layout.
    Here every process takes part in one BLACSGrid construction, each with
    the communicator of its k-point group ordered by its (row, column) in
    FHI-aims' grid, which reproduces that grid exactly. Later runs reuse it.
    """
    from scalapack4py.npscal import NPScal
    from scalapack4py.npscal.blacs_ctxt_management import BLACSGrid, descriptor_registry
    from embasi.asi_default_callbacks import RawKBlock

    name = f"{ctxt_tag}_kgrid"
    raws = [(store, key) for store in stores for key, val in store.items() if isinstance(val, RawKBlock)]
    if not BLACSGrid.is_registered(name):
        if not raws:
            raise RuntimeError(f"rank {MPI.COMM_WORLD.rank}: no deferred k-point block to take the grid layout from")
        layouts = {(val.PiK, val.layout) for store, key in raws for val in [store[key]]}
        if len(layouts) != 1:
            raise NotImplementedError(f"rank {MPI.COMM_WORLD.rank} is on several k-point grids {sorted(layouts)}: "
                                      "one k-point per process is assumed")
        (PiK, (nprow, npcol, myrow, mycol)), = layouts
        comm_k = MPI.COMM_WORLD.Split(PiK, myrow * npcol + mycol)
        grid = BLACSGrid(asi.scalapack, comm=comm_k, name=name, nprow=nprow, npcol=npcol)
        if (grid.nprow, grid.npcol, grid.myrow, grid.mycol) != (nprow, npcol, myrow, mycol):
            raise RuntimeError(f"rank {MPI.COMM_WORLD.rank}: EmbASI's k-point group grid "
                               f"{(grid.nprow, grid.npcol, grid.myrow, grid.mycol)} != FHI-aims' "
                               f"{(nprow, npcol, myrow, mycol)}")
    grid = BLACSGrid.get(name)
    for store, key in raws:
        raw = store[key]
        if grid.default_mb is None:
            grid.set_default_block_size(raw.descr[2], raw.descr[3])
        descriptor_registry.get_or_create(raw.tag, *raw.descr)
        from embasi.asi_default_callbacks import _on_grid_blocks, _full_hermitian
        store[key] = _on_grid_blocks(_full_hermitian(
            NPScal(loc_array=raw.loc, grid=grid, descriptor=raw.tag, lib=asi.scalapack)), grid)


def check_kpoint_grid(grid, asi, descr_cast, label):
    """The k-point group grid must place this process where FHI-aims' grid does"""
    aims = tuple(asi.scalapack.blacs_gridinfo(descr_cast.ctxt)[:4])
    ours = (grid.nprow, grid.npcol, grid.myrow, grid.mycol)
    if aims != ours:
        raise RuntimeError(f"{label}: FHI-aims' grid (nprow, npcol, myrow, mycol) = {aims}, "
                           f"EmbASI's k-point group grid = {ours} on rank {MPI.COMM_WORLD.rank}")
    return grid
