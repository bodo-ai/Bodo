import ipyparallel as ipp
import bodo_platform_ipyparallel_kernel as bpik
from bodo.mpi4py import MPI


def mpi_example():
    comm = MPI.COMM_WORLD
    return f"Hello World from rank {comm.Get_rank()}. total ranks={comm.Get_size()}"


def test_ipyparallel():
    # request an MPI cluster with 4 engines
    with ipp.Cluster(engines="mpi", n=4) as rc:
        # get a broadcast_view on the cluster which is best
        # suited for MPI style computation
        view = rc.broadcast_view()
        # run the mpi_example function on all engines in parallel
        r = view.apply_sync(mpi_example)
        # Retrieve and print the result from the engines
        print("\n".join(r))
        # at this point, the cluster processes have been shutdown


def test_bodo_platform_ipyparallel():
    assert hasattr(bpik, "IPyParallelKernel")
    # request an MPI cluster with 4 engines
    with ipp.Cluster(engines=bpik.BodoPlatformMPIEngineSetLauncher, n=4) as rc:
        # get a broadcast_view on the cluster which is best
        # suited for MPI style computation
        view = rc.broadcast_view()
        # run the mpi_example function on all engines in parallel
        r = view.apply_sync(mpi_example)
        # Retrieve and print the result from the engines
        print("\n".join(r))
        # at this point, the cluster processes have been shutdown


if __name__ == "__main__":
    test_ipyparallel()
