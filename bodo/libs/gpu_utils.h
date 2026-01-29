#pragma once

// #ifdef USE_CUDF
#include <mpi.h>
#include <nccl.h>
#include <cudf/contiguous_split.hpp>
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
#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            throw std::runtime_error("CUDA error: " +                       \
                                     std::string(cudaGetErrorString(err))); \
        }                                                                   \
    } while (0)

enum class GpuShuffleState {
    WAITING_FOR_SIZES = 0,
    INFLIGHT = 1,
    COMPLETED = 2
};
/**
 * @brief Holds information for inflight shuffle operations
 */
struct GpuShuffle {
    GpuShuffleState state = GpuShuffleState::WAITING_FOR_SIZES;
    std::vector<cudf::packed_table> packed_tables;
    MPI_Comm mpi_comm;
    ncclComm_t nccl_comm;
    cudaStream_t stream;
    // MPI Requests for sizes of gpu buffers from other ranks
    // Indexed by sending rank
    std::vector<MPI_Request> gpu_sizes_recv_reqs;
    // MPI Requests for sizes of gpu buffers to other ranks
    // Indexed by destination rank
    std::vector<MPI_Request> gpu_sizes_send_reqs;
    // MPI Requests for sizes of metadata from other ranks
    // Indexed by sending rank
    std::vector<MPI_Request> metadata_sizes_recv_reqs;
    // MPI Requests for sizes of metadata to other ranks
    // Indexed by destination rank
    std::vector<MPI_Request> metadata_sizes_send_reqs;
    // MPI_Requests for metadata transfers from other ranks
    // Indexed by sending rank
    std::vector<MPI_Request> metadata_recv_reqs;
    // MPI_Requests for metadata transfers to other ranks
    // Indexed by destination rank
    std::vector<MPI_Request> metadata_send_reqs;
    // Event marker for all nccl operations needed for this shuffle.
    // When this is finished all GPU buffers are in the correct place.
    cudaEvent_t nccl_event;
    // Buffers for metadata from other ranks, these are used to construct
    // packed_columns. Indexed by sending rank
    std::vector<std::unique_ptr<std::vector<uint8_t>>> metadata_recv_buffers;
    // Buffers for metadata transfers toother ranks, these are used to construct
    // packed_columns. Indexed by destination rank
    std::vector<std::unique_ptr<std::vector<uint8_t>>> metadata_send_buffers;
    // Buffers for column data from other ranks, these are used to construct
    // packed_columns. Indexed by sending rank
    std::vector<rmm::device_buffer> packed_recv_buffers;
    // Buffers for column data sent to other ranks, these are used to construct
    // packed_columns. Indexed by destination rank
    std::vector<rmm::device_buffer> packed_send_buffers;

    GpuShuffle(std::vector<cudf::packed_table> packed_tables,
               MPI_Comm mpi_comm_, ncclComm_t nccl_comm_, cudaStream_t stream_,
               int n_ranks)
        : packed_tables(std::move(packed_tables)),
          mpi_comm(mpi_comm_),
          nccl_comm(nccl_comm_),
          stream(stream_),
          gpu_sizes_recv_reqs(n_ranks),
          gpu_sizes_send_reqs(n_ranks),
          metadata_sizes_recv_reqs(n_ranks),
          metadata_sizes_send_reqs(n_ranks),
          metadata_recv_reqs(n_ranks),
          metadata_send_reqs(n_ranks),
          metadata_recv_buffers(n_ranks),
          metadata_send_buffers(n_ranks),
          packed_recv_buffers(n_ranks),
          packed_send_buffers(n_ranks) {
        CHECK_CUDA(
            cudaEventCreateWithFlags(&nccl_event, cudaEventDisableTiming));
    }

    std::optional<cudf::table_view> progress();
};

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

    // GPU device ID
    rmm::cuda_device_id gpu_id;
    // Ensure the correct device is set for the lifetime of this manager
    rmm::cuda_set_device_raii cuda_device_raii;

    std::vector<GpuShuffle> inflight_shuffles;

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
     */
    void shuffle_table(std::shared_ptr<cudf::table> table,
                       const std::vector<cudf::size_type>& partition_indices);

    /**
     * @brief Progress any inflight shuffles
     * @return Optional vector of tables received from all ranks if any were
     * received.
     */
    std::vector<std::unique_ptr<cudf::table>> progress();

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

    /**
     * @brief Check if there are any inflight shuffles
     * @return true if there are inflight shuffles, false otherwise
     */
    bool inflight_exists() const { return !inflight_shuffles.empty(); }
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
 * @param gpu_id GPU device ID
 * @return MPI_Comm
 */
MPI_Comm get_gpu_mpi_comm(rmm::cuda_device_id gpu_id);

// #else
//// Empty implementation when CUDF is not available
// class GpuShuffleManager {
// public:
//     explicit GpuShuffleManager(MPI_Comm mpi_comm_) {}
//     bool is_available() const { return false; }
// };
// #endif
