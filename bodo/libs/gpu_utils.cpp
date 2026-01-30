#include "gpu_utils.h"

// #ifdef USE_CUDF
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

GpuShuffleManager::GpuShuffleManager() {
    // Create a subcommunicator with only ranks that have GPUs assigned
    mpi_comm = get_gpu_mpi_comm();
    if (mpi_comm == MPI_COMM_NULL) {
        return;
    }

    // Get rank and size
    MPI_Comm_rank(mpi_comm, &rank);
    MPI_Comm_size(mpi_comm, &n_ranks);

    // Create CUDA stream
    cudaStreamCreate(&stream);

    // Initialize NCCL
    initialize_nccl();
}

GpuShuffleManager::~GpuShuffleManager() {
    // Destroy NCCL communicator
    ncclCommDestroy(nccl_comm);

    // Destroy CUDA stream
    if (stream) {
        cudaStreamDestroy(stream);
    }
}

void GpuShuffleManager::initialize_nccl() {
    ncclUniqueId nccl_id;

    if (rank == 0) {
        // Generate unique ID on root rank
        CHECK_NCCL(ncclGetUniqueId(&nccl_id));
    }

    // Broadcast the unique ID to all ranks
    int ret = MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, mpi_comm);
    if (ret != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Bcast failed");
    }

    // Initialize NCCL communicator
    CHECK_NCCL(ncclCommInitRank(&nccl_comm, n_ranks, nccl_id, rank));
}

// std::vector<std::unique_ptr<cudf::table>> GpuShuffleManager::shuffle_table(
//     std::shared_ptr<cudf::table> table,
//     const std::vector<cudf::size_type>& partition_indices) {

    // if (mpi_comm == MPI_COMM_NULL) {
    //     return {};
    // }
    // // Hash partition the table
    // auto [partitioned_table, partition_sizes] =
    //     hash_partition_table(table, partition_indices, n_ranks);
    // // Pack the tables for sending
    // std::vector<cudf::packed_table> packed_tables = cudf::contiguous_split(
    //     partitioned_table->view(), partition_sizes, stream);
    // for (size_t i = 0; i < packed_tables.size(); ++i) {
    //     shuffle_packed_table(nccl_comm, stream, packed_tables[i], i);
    // }
    // // Receive the tables from all ranks
    // for (size_t i = 0; i < n_ranks; ++i) {
    //     packed_tables[i] = receive_packed_table(nccl_comm, stream, i);
    // }
    // std::vector<std::unique_ptr<cudf::table>> received_tables(n_ranks);
    // // Unpack the received tables
    // for (size_t i = 0; i < n_ranks; ++i) {
    //     received_tables[i] = std::make_unique<cudf::table>(
    //         cudf::unpack_table(packed_tables[i], stream));
    // }

    // return received_tables;
// }

std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>>
hash_partition_table(std::shared_ptr<cudf::table> table,
                     const std::vector<cudf::size_type>& column_indices,
                     cudf::size_type num_partitions) {
    if (column_indices.empty()) {
        throw std::invalid_argument("Column indices cannot be empty");
    }

    // Extract the columns to hash
    std::vector<cudf::size_type> indices(column_indices.begin(),
                                         column_indices.end());

    // Partition the table based on the hash of the selected columns
    return cudf::hash_partition(table->view(), indices, num_partitions);
}

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

void shuffle_packed_table(ncclComm_t comm, cudaStream_t stream,
                          cudf::packed_table& packed_table, int dest_rank) {}

cudf::packed_table receive_packed_table(ncclComm_t comm, cudaStream_t stream,
                                        int src_rank) {
    return cudf::packed_table{};
}

// #endif // USE_CUDF
