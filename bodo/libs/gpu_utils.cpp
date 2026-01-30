#include "gpu_utils.h"

#ifdef USE_CUDF
#include <mpi_proto.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <cassert>
#include <cudf/column/column_factories.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/hashing.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <rmm/device_uvector.hpp>
#include "../libs/_distributed.h"
#include "_utils.h"

rmm::cuda_device_id get_gpu_id() {
    // TODO: Fix hang in collective call
    auto [n_ranks, rank_on_node] = dist_get_ranks_on_node();
    // int rank_on_node, n_ranks;
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank_on_node);
    // MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    int device_count;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0) {
        throw std::runtime_error("No CUDA devices available");
    }
    assert(n_ranks > device_count &&
           "More MPI ranks than available GPUs on node");
    rmm::cuda_device_id device_id(rank_on_node < device_count ? rank_on_node
                                                              : -1);

    return device_id;
}

int get_cluster_cuda_device_count() {
    int local_device_count, device_count;
    cudaGetDeviceCount(&device_count);
    CHECK_MPI(MPI_Allreduce(&local_device_count, &device_count, 1, MPI_INT,
                            MPI_SUM, MPI_COMM_WORLD),
              "get_cluster_cuda_device_count: MPI error on MPI_Allreduce:");
    return device_count;
}

MPI_Comm get_gpu_mpi_comm() {
    MPI_Comm gpu_comm;
    int has_gpu = 0;
    rmm::cuda_device_id gpu_id = get_gpu_id();
    if (gpu_id.value() >= 0) {
        has_gpu = 1;
    }
    CHECK_MPI(MPI_Comm_split(MPI_COMM_WORLD, has_gpu, 0, &gpu_comm),
              "get_gpu_mpi_comm: MPI error on MPI_Comm_split:");
    if (has_gpu == 0) {
        return MPI_COMM_NULL;
    }
    return gpu_comm;
}

#endif  // USE_CUDF
