"""Support distributed deep learning with Horovod
"""
import numba
import numpy as np
from mpi4py import MPI

import bodo
from bodo.libs.distributed_api import (
    create_subcomm_mpi4py,
    get_host_ranks,
    get_nodes_first_ranks,
)

horovod_initialized = False
gpu_ranks = None  # sorted list of ranks pinned to GPUs for deep learning


def get_num_gpus():
    """ Get number of GPU devices on this host """
    try:
        import torch

        return torch.cuda.device_count()
    except:  # pragma: no cover
        # TODO: test this
        import tensorflow as tf

        return tf.config.experimental.list_physical_devices("GPU")


def get_gpu_ranks():
    """ Calculate and return the global list of ranks to pin to GPUs """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    host_ranks = get_host_ranks()
    nodes_first_ranks = get_nodes_first_ranks()
    if rank in nodes_first_ranks:
        # the first rank on each host collects the number of GPUs on the host
        # and sends them to rank 0. rank 0 will calculate global gpu rank list
        try:
            num_gpus_in_node = get_num_gpus()
        except Exception as e:  # pragma: no cover
            num_gpus_in_node = e
        subcomm = create_subcomm_mpi4py(nodes_first_ranks)
        num_gpus_per_node = subcomm.gather(num_gpus_in_node)
        if rank == 0:
            gpu_ranks = []
            error = None
            # TODO: Test CUDA on CI
            for i, ranks in enumerate(host_ranks.values()):  # pragma: no cover
                n_gpus = num_gpus_per_node[i]
                if isinstance(n_gpus, Exception):
                    error = n_gpus
                    break
                if n_gpus == 0:
                    continue
                # TODO? more GPUs than cores on a single host
                cores_per_gpu = len(ranks) // n_gpus
                for local_rank, global_rank in enumerate(ranks):
                    if local_rank % cores_per_gpu == 0:
                        # pin this rank to GPU
                        my_gpu = local_rank / cores_per_gpu
                        if my_gpu < n_gpus:
                            gpu_ranks.append(global_rank)
            if error:  # pragma: no cover
                comm.bcast(error)
                raise error
            else:
                comm.bcast(gpu_ranks)
    if rank != 0:  # pragma: no cover
        # wait for global list of GPU ranks from rank 0.
        gpu_ranks = comm.bcast(None)
        if isinstance(gpu_ranks, Exception):
            e = gpu_ranks
            raise e
    return gpu_ranks


def is_cuda_available():
    """ Return true if the cluster on which Bodo is running has GPUs available """
    # XXX Should maybe raise an error instead of doing `not horovod_initialized`
    if not horovod_initialized:
        initialize_horovod()
    return len(gpu_ranks) > 0


# TODO: support other DL frameworks: tensorflow, keras, etc.
def initialize_horovod():
    """Initialization for distributed deep learning:
    1) Get global list of ranks to pin to GPUs (one GPU per process)
    2) Limit PyTorch workers to 1 thread (in case CPU DL is used)
    3) Initialize horovod with list of gpu ranks (if cuda) otherwise with COMM_WORLD
    Returns list of gpu ranks (empty list if no GPUs in cluster)
    """
    global horovod_initialized, gpu_ranks
    if horovod_initialized:
        return np.array(gpu_ranks, dtype=np.int32)

    gpu_ranks = get_gpu_ranks()

    import horovod.torch as hvd
    import torch

    # Limit # of CPU threads to be used per worker
    torch.set_num_threads(1)

    myrank = MPI.COMM_WORLD.rank
    if len(gpu_ranks) > 0:  # pragma: no cover
        # Split COMM_WORLD into subcommunicators
        subcomm = MPI.COMM_WORLD.Split(
            color=(0 if myrank in gpu_ranks else MPI.UNDEFINED), key=myrank
        )

        if subcomm != MPI.COMM_NULL:
            hvd.init(comm=subcomm)

            # Pin a GPU to this rank (one GPU per process)
            torch.cuda.set_device(hvd.local_rank())
    else:
        if myrank == 0:
            print("[BODO-DL]: No GPUs found in cluster. Using CPUs")
        hvd.init()

    horovod_initialized = True
    return np.array(gpu_ranks, dtype=np.int32)


# XXX What if data is already prepared and all that we need is to call
# initialize_horovod?
@numba.njit
def prepare_data(X_train, y_train):  # pragma: no cover
    """This function is called by user code to redistribute the data to
    GPU ranks and initialize horovod"""
    with numba.objmode(gpu_ranks="int32[:]"):
        gpu_ranks = initialize_horovod()

    if len(gpu_ranks) > 0:
        X_train = bodo.rebalance(X_train, dests=list(gpu_ranks), parallel=True)
        y_train = bodo.rebalance(y_train, dests=list(gpu_ranks), parallel=True)
    else:
        X_train = bodo.rebalance(X_train, parallel=True)
        y_train = bodo.rebalance(y_train, parallel=True)
    return X_train, y_train
