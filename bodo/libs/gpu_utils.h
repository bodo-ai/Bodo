#pragma once

// #ifdef USE_CUDF
#include <mpi.h>
#include <nccl.h>
#include <cudf/table/table.hpp>

// Error checking macros for NCCL
#define CHECK_NCCL(call)                                                       \
    do {                                                                       \
        ncclResult_t result = call;                                            \
        if (result != ncclSuccess) {                                           \
            throw std::runtime_error("NCCL error: " +                          \
                                     std::string(ncclGetErrorString(result))); \
        }                                                                      \
    } while (0)

/**
 * @brief Class for managing async shuffle of cudf::tables using NCCL
 */
class GpuShuffleManager {
   private:
    // NCCL communicator
    ncclComm_t nccl_comm;

    // MPI communicator for CPU communication between ranks
    // with GPUs assgined
    MPI_Comm mpi_comm;

    // Number of processes
    int n_ranks;

    // Current rank
    int rank;

    // Stream for CUDA operations
    cudaStream_t stream;

    /**
     * @brief Initialize NCCL communicator
     */
    void initialize_nccl();

   public:
    GpuShuffleManager();
    ~GpuShuffleManager();

    /**
     * @brief Shuffle a cudf table across all ranks
     * @param table Input table to shuffle
     * @param partition_indices Column indices to use for partitioning
     * @return Vector of tables received from all ranks
     */
    std::vector<std::unique_ptr<cudf::table>> shuffle_table(
        std::shared_ptr<cudf::table> table,
        const std::vector<cudf::size_type>& partition_indices);

    /**
     * @brief Get the underlying NCCL communicator
     * @return ncclComm_t
     */
    ncclComm_t get_nccl_comm() const { return nccl_comm; }

    /**
     * @brief Get the underlying CUDA stream
     * @return cudaStream_t
     */
    cudaStream_t get_stream() const { return stream; }
};

/**
 * @brief Hash partition function for GPU tables
 * @param table Input table
 * @param column_indices Column indices to hash
 * @param num_partitions Number of partitions
 * @return Pair of partitioned table and partition indices
 */
std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>>
hash_partition_table(std::shared_ptr<cudf::table> table,
                     const std::vector<cudf::size_type>& column_indices,
                     cudf::size_type num_partitions);

/**
 * @brief Get the GPU device ID for the current process. All ranks must call
 * this function.
 * @return rmm::cuda_device_id, -1 if no GPU is assigned to this rank
 */
rmm::cuda_device_id get_gpu_id();
/**
 * @brief Get the number of CUDA devices available in the cluster. All ranks
 * must call this function.
 * @return Number of CUDA devices
 */
int get_cluster_cuda_device_count();
/**
 * @brief Get the MPI communicator for ranks with GPUs assigned
 * @return MPI_Comm
 */
MPI_Comm get_gpu_mpi_comm();

// #else
//// Empty implementation when CUDF is not available
// class GpuShuffleManager {
// public:
//     explicit GpuShuffleManager(MPI_Comm mpi_comm_) {}
//     bool is_available() const { return false; }
// };
// #endif
