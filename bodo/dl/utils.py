"""Support distributed deep learning with Horovod
"""
import time

import numba
import numpy as np

import bodo
from bodo.libs.distributed_api import (
    create_subcomm_mpi4py,
    get_host_ranks,
    get_nodes_first_ranks,
)
from bodo.mpi4py import MPI

dl_status = None


def assert_dl_initialized():  # pragma: no cover
    assert (
        dl_status is not None
    ), "Horovod has not been initialized. Call bodo.dl.start() first"


class DLStatus(object):  # pragma: no cover
    def __init__(self, framework, gpu_ranks):
        # will be one of "torch", "tensorflow" after initialization
        self.framework = framework
        # sorted list of ranks pinned to GPUs for deep learning
        self.gpu_ranks = gpu_ranks


def get_num_gpus(framework):  # pragma: no cover
    """Get number of GPU devices on this host"""
    if framework == "torch":
        import torch

        return torch.cuda.device_count()
    elif framework == "tensorflow":
        import tensorflow as tf

        return len(tf.config.experimental.list_physical_devices("GPU"))
    else:
        raise RuntimeError("Framework {} not recognized".format(framework))


def get_gpu_ranks(framework):  # pragma: no cover
    """Calculate and return the global list of ranks to pin to GPUs"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    host_ranks = get_host_ranks()
    nodes_first_ranks = get_nodes_first_ranks()
    if rank in nodes_first_ranks:
        # the first rank on each host collects the number of GPUs on the host
        # and sends them to rank 0. rank 0 will calculate global gpu rank list
        try:
            num_gpus_in_node = get_num_gpus(framework)
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


def is_cuda_available():  # pragma: no cover
    """Return true if the cluster on which Bodo is running has GPUs available"""
    assert_dl_initialized()
    return len(dl_status.gpu_ranks) > 0


def initialize_horovod(framework):  # pragma: no cover
    """Initialization for distributed deep learning:
    1) Get global list of ranks to pin to GPUs (one GPU per process)
    2) Resource-level initialization for DL framework as needed
    3) Initialize horovod with list of gpu ranks (if cuda) otherwise with COMM_WORLD
    Returns list of gpu ranks (empty list if no GPUs in cluster)
    """
    global dl_status
    if dl_status is not None:
        assert (
            dl_status.framework == framework
        ), "Attempted to initialize Horovod with different DL frameworks"
        return np.array(dl_status.gpu_ranks, dtype=np.int32)

    gpu_ranks = get_gpu_ranks(framework)

    if framework == "torch":
        import horovod.torch as hvd
        import torch

        # Limit # of CPU threads to be used per worker
        torch.set_num_threads(1)
    elif framework == "tensorflow":
        import horovod.tensorflow as hvd
        import tensorflow as tf
    else:
        raise RuntimeError("Framework {} not recognized".format(framework))

    myrank = MPI.COMM_WORLD.rank
    if len(gpu_ranks) > 0:  # pragma: no cover
        # Split COMM_WORLD into subcommunicators
        subcomm = MPI.COMM_WORLD.Split(
            color=(0 if myrank in gpu_ranks else MPI.UNDEFINED), key=myrank
        )

        if subcomm != MPI.COMM_NULL:
            hvd.init(comm=subcomm)

            # Pin a GPU to this rank (one GPU per process)
            if framework == "torch":
                torch.cuda.set_device(hvd.local_rank())
            elif framework == "tensorflow":
                gpus = tf.config.experimental.list_physical_devices("GPU")
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_visible_devices(
                    gpus[hvd.local_rank()], "GPU"
                )
    else:
        if myrank == 0:
            print("[BODO-DL]: No GPUs found in cluster. Using CPUs")
        hvd.init()

    dl_status = DLStatus(framework, np.array(gpu_ranks, dtype=np.int32))


@numba.njit
def start(framework):  # pragma: no cover
    """Called when user wants to begin DL. Will initialize Horovod if not
    done so already"""
    with bodo.no_warning_objmode:
        initialize_horovod(framework)


@numba.njit
def end():  # pragma: no cover
    """Called when user wants to end DL. This causes ranks that are not
    participanting in DL (e.g. non-gpu ranks to idle wait for the other workers)"""
    with bodo.no_warning_objmode:
        end_py()


def end_py():  # pragma: no cover
    if is_cuda_available():
        WAKE_UP_TAG = 17
        # The first rank on each node sends a point-to-point message to all
        # other ranks in the node. These ranks are waiting in an idle loop
        # (using sleep) for message to arrive. They probe for the message
        # periodically.
        comm = MPI.COMM_WORLD
        myhost = MPI.Get_processor_name()
        local_ranks = get_host_ranks()[myhost]
        assert_dl_initialized()
        if bodo.get_rank() == local_ranks[0]:
            assert bodo.get_rank() in dl_status.gpu_ranks
            for rank in local_ranks[1:]:
                comm.isend(1, dest=rank, tag=WAKE_UP_TAG)
        else:
            while True:
                status = MPI.Status()
                f = comm.Iprobe(MPI.ANY_SOURCE, MPI.ANY_TAG, status)
                if f:
                    assert status.source == local_ranks[0]
                    assert status.tag == WAKE_UP_TAG
                    comm.recv(source=0, tag=WAKE_UP_TAG)
                    break
                time.sleep(1.0)
    else:
        # in CPU mode everybody does work so no need to idle wait
        bodo.barrier()


def _prepare_data_get_gpu_ranks():  # pragma: no cover
    """Helper for prepare_data. Asserts DL has been initialized and returns
    list of GPU ranks"""
    assert_dl_initialized()
    return dl_status.gpu_ranks


@numba.njit
def prepare_data(data):  # pragma: no cover
    """This function is called by the user to redistribute the data to
    GPU ranks and initialize horovod"""
    with bodo.no_warning_objmode(gpu_ranks="int32[:]"):
        gpu_ranks = _prepare_data_get_gpu_ranks()

    if len(gpu_ranks) > 0:
        data = bodo.rebalance(data, dests=list(gpu_ranks), parallel=True)
    else:
        data = bodo.rebalance(data, parallel=True)
    return data
