from bodo.mpi4py import MPI
import numpy as np


def test_allgather():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    work_size = 100
    work = np.zeros(work_size)

    base = work_size / mpi_size
    leftover = work_size % mpi_size
    sizes = np.ones(mpi_size) * base
    sizes[:leftover] += 1

    offsets = np.zeros(mpi_size)
    offsets[1:] = np.cumsum(sizes)[:-1]

    start = offsets[rank]
    local_size = sizes[rank]
    work_local = np.arange(start, start + local_size, dtype=np.float64)

    comm.Allgatherv(work_local, [work, sizes, offsets, MPI.DOUBLE])
    summe = np.empty(1, dtype=np.float64)
    comm.Allreduce(np.sum(work_local), summe, op=MPI.SUM)
    total_work = np.sum(work)
    assert total_work == np.sum(summe)


if __name__ == "__main__":
    test_allgather()
