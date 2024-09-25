from bodo.mpi4py import MPI
import numpy as np


def test_mpi():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    ############### Broadcasting a python dict ######################
    if rank == 0:
        data = {"key1": [7, 2.72, 2 + 3j], "key2": ("abc", "xyz")}
    else:
        data = None
    data = comm.bcast(data, root=0)
    print(data)
    ############### Broadcasting numpy array ########################
    if rank == 0:
        data = np.arange(100, dtype="i")
    else:
        data = np.empty(100, dtype="i")
    comm.Bcast(data, root=0)
    for i in range(100):
        assert data[i] == i
    ############### Scattering python objects #######################
    if rank == 0:
        data = [(i + 1) ** 2 for i in range(size)]
    else:
        data = None
    data = comm.scatter(data, root=0)
    assert data == (rank + 1) ** 2
    ############### Scattering numpy #######################
    sendbuf = None
    if rank == 0:
        sendbuf = np.empty([size, 100], dtype="i")
        sendbuf.T[:, :] = range(size)
    recvbuf = np.empty(100, dtype="i")
    comm.Scatter(sendbuf, recvbuf, root=0)
    assert np.allclose(recvbuf, rank)
    ############### Gathering python objects ########################
    data = (rank + 1) ** 2
    data = comm.gather(data, root=0)
    if rank == 0:
        for i in range(size):
            assert data[i] == (i + 1) ** 2
    else:
        assert data is None
    ############### Gathering numpy ########################
    sendbuf = np.zeros(100, dtype="i") + rank
    recvbuf = None
    if rank == 0:
        recvbuf = np.empty([size, 100], dtype="i")
    comm.Gather(sendbuf, recvbuf, root=0)
    if rank == 0:
        for i in range(size):
            assert np.allclose(recvbuf[i, :], i)


if __name__ == "__main__":
    test_mpi()
