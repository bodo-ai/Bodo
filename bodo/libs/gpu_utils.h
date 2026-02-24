#pragma once

extern bool G_USE_ASYNC;

#ifdef USE_CUDF
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

struct cuda_event_wrapper {
    std::shared_ptr<cudaEvent_t> ev;

    cuda_event_wrapper() {
        // Allocate storage for the event handle
        cudaEvent_t raw;
        cudaEventCreateWithFlags(&raw, cudaEventDisableTiming);

        // Wrap it in a shared_ptr with a custom deleter
        ev = std::shared_ptr<cudaEvent_t>(new cudaEvent_t(raw),
                                          [](cudaEvent_t* p) {
                                              if (p && *p) {
                                                  cudaEventDestroy(*p);
                                              }
                                              delete p;
                                          });
    }

    // Copy constructor: shared ownership
    cuda_event_wrapper(const cuda_event_wrapper&) = default;

    // Copy assignment: shared ownership
    cuda_event_wrapper& operator=(const cuda_event_wrapper&) = default;

    // Disable move semantics (optional but recommended for clarity)
    cuda_event_wrapper(cuda_event_wrapper&&) noexcept = default;
    cuda_event_wrapper& operator=(cuda_event_wrapper&&) noexcept = default;

    // Record event on a stream
    void record(rmm::cuda_stream_view stream) const {
        if (ev && *ev) {
            CHECK_CUDA(cudaEventRecord(*ev, stream.value()));
        }
    }

    // Make a stream wait on this event
    void wait(rmm::cuda_stream_view stream) const {
        if (ev && *ev) {
            CHECK_CUDA(cudaStreamWaitEvent(stream.value(), *ev, 0));
        }
    }

    cudaError_t query() const {
        if (ev && *ev) {
            return cudaEventQuery(*ev);
        } else {
            throw std::runtime_error(
                "cuda_event_wrapper query on invalid state");
        }
    }

    bool ready() const {
        if (ev && *ev) {
            cudaError_t status = cudaEventQuery(*ev);
            return status == cudaSuccess;
        } else {
            throw std::runtime_error(
                "cuda_event_wrapper ready on invalid state");
        }
    }
};

struct StreamAndEvent {
    rmm::cuda_stream_view stream;
    cuda_event_wrapper event;

    StreamAndEvent(rmm::cuda_stream_view s, cuda_event_wrapper e)
        : stream(s), event(e) {}
};

inline std::shared_ptr<StreamAndEvent> make_stream_and_event(bool use_async) {
    if (use_async) {
        // Create a new non-blocking CUDA stream
        rmm::cuda_stream_view s{rmm::cuda_stream_per_thread};

        // Create an unsignaled event (default constructor)
        cuda_event_wrapper e;

        return std::make_shared<StreamAndEvent>(s, e);
    } else {
        // Synchronous mode: use default stream
        rmm::cuda_stream_view s = rmm::cuda_stream_default;

        // Event is already completed
        cuda_event_wrapper e;
        e.record(s);

        return std::make_shared<StreamAndEvent>(s, e);
    }
}

enum class GpuShuffleState {
    SIZES_INFLIGHT = 0,
    DATA_INFLIGHT = 1,
    COMPLETED = 2
};

/**
 * @brief Holds information for inflight shuffle operations
 */
struct GpuShuffle {
    GpuShuffleState send_state = GpuShuffleState::SIZES_INFLIGHT;
    GpuShuffleState recv_state = GpuShuffleState::SIZES_INFLIGHT;
    MPI_Comm mpi_comm = MPI_COMM_NULL;
    ncclComm_t nccl_comm = nullptr;
    cudaStream_t stream = nullptr;
    // These need to be unique_ptrs to vectors to guarantee they don't
    // move if GpuShuffle is moved
    // MPI Requests for sizes of gpu buffers from other ranks
    // Indexed by sending rank
    std::unique_ptr<std::vector<MPI_Request>> gpu_sizes_recv_reqs;
    // MPI Requests for sizes of gpu buffers to other ranks
    // Indexed by destination rank
    std::unique_ptr<std::vector<MPI_Request>> gpu_sizes_send_reqs;
    // MPI Requests for sizes of metadata from other ranks
    // Indexed by sending rank
    std::unique_ptr<std::vector<MPI_Request>> metadata_sizes_recv_reqs;
    // MPI Requests for sizes of metadata to other ranks
    // Indexed by destination rank
    std::unique_ptr<std::vector<MPI_Request>> metadata_sizes_send_reqs;
    // MPI_Requests for metadata transfers from other ranks
    // Indexed by sending rank
    std::unique_ptr<std::vector<MPI_Request>> metadata_recv_reqs;
    // MPI_Requests for metadata transfers to other ranks
    // Indexed by destination rank
    std::unique_ptr<std::vector<MPI_Request>> metadata_send_reqs;
    // Event markers for all nccl operations needed for this shuffle.
    // When this is finished all GPU buffers are in the correct place.
    cuda_event_wrapper nccl_send_event;
    cuda_event_wrapper nccl_recv_event;
    // We need to keep sizes around while the transfers are inflight
    std::unique_ptr<std::vector<uint64_t>> send_metadata_sizes;
    std::unique_ptr<std::vector<uint64_t>> recv_metadata_sizes;
    std::unique_ptr<std::vector<uint64_t>> send_gpu_sizes;
    std::unique_ptr<std::vector<uint64_t>> recv_gpu_sizes;
    // Buffers for metadata from other ranks, these are used to construct
    // packed_columns. Indexed by sending rank
    std::vector<std::unique_ptr<std::vector<uint8_t>>> metadata_recv_buffers;
    // Buffers for metadata transfers to other ranks, these are used to
    // construct packed_columns. Indexed by destination rank
    std::vector<std::unique_ptr<std::vector<uint8_t>>> metadata_send_buffers;
    // Buffers for column data from other ranks, these are used to construct
    // packed_columns. Indexed by sending rank
    std::vector<std::unique_ptr<rmm::device_buffer>> packed_recv_buffers;
    // Buffers for column data sent to other ranks, these are used to construct
    // packed_columns. Indexed by destination rank
    std::vector<std::unique_ptr<rmm::device_buffer>> packed_send_buffers;

    GpuShuffle(std::vector<cudf::packed_table> packed_tables,
               MPI_Comm mpi_comm_, ncclComm_t nccl_comm_, cudaStream_t stream_,
               int n_ranks, int start_tag)
        : mpi_comm(mpi_comm_),
          nccl_comm(nccl_comm_),
          stream(stream_),
          gpu_sizes_recv_reqs(
              std::make_unique<std::vector<MPI_Request>>(n_ranks)),
          gpu_sizes_send_reqs(
              std::make_unique<std::vector<MPI_Request>>(n_ranks)),
          metadata_sizes_recv_reqs(
              std::make_unique<std::vector<MPI_Request>>(n_ranks)),
          metadata_sizes_send_reqs(
              std::make_unique<std::vector<MPI_Request>>(n_ranks)),
          metadata_recv_reqs(
              std::make_unique<std::vector<MPI_Request>>(n_ranks)),
          metadata_send_reqs(
              std::make_unique<std::vector<MPI_Request>>(n_ranks)),
          send_metadata_sizes(
              std::make_unique<std::vector<uint64_t>>(n_ranks, 0)),
          recv_metadata_sizes(
              std::make_unique<std::vector<uint64_t>>(n_ranks, 0)),
          send_gpu_sizes(std::make_unique<std::vector<uint64_t>>(n_ranks, 0)),
          recv_gpu_sizes(std::make_unique<std::vector<uint64_t>>(n_ranks, 0)),
          metadata_recv_buffers(n_ranks),
          metadata_send_buffers(n_ranks),
          packed_recv_buffers(n_ranks),
          packed_send_buffers(n_ranks),
          start_tag(start_tag),
          n_ranks(n_ranks) {
        for (size_t dest_rank = 0; dest_rank < packed_tables.size();
             dest_rank++) {
            cudf::packed_table& table = packed_tables[dest_rank];
            // Prepare send buffers
            packed_send_buffers[dest_rank] = std::move(table.data.gpu_data);
            metadata_send_buffers[dest_rank] =
                std::make_unique<std::vector<uint8_t>>(
                    std::move(*table.data.metadata));
        }

        this->send_sizes();
        this->recv_sizes();
        this->send_metadata();
    }

    // Enable move constructors
    GpuShuffle(GpuShuffle&&) = default;
    GpuShuffle& operator=(GpuShuffle&&) = default;

    // Disable copy constructors
    GpuShuffle(const GpuShuffle&) = delete;
    GpuShuffle& operator=(const GpuShuffle&) = delete;

    ~GpuShuffle() {}

    /*
     * @brief Progress the shuffle operation
     * @return Optional unique_ptr to cudf::table if shuffle is complete
     */
    std::optional<std::unique_ptr<cudf::table>> progress();

   private:
    int start_tag;
    int n_ranks;
    void send_sizes();
    void recv_sizes();
    void send_metadata();
    void recv_metadata();
    void send_data();
    void recv_data();
    void progress_waiting_for_sizes();
    std::optional<std::unique_ptr<cudf::table>> progress_waiting_for_data();
    void progress_sending_sizes();
    void progress_sending_data();
};

struct DoShuffleCoordination {
    MPI_Request req = MPI_REQUEST_NULL;
    int has_data;
};

class ShuffleTableInfo {
   public:
    std::shared_ptr<cudf::table> table;
    std::vector<cudf::size_type> partition_indices;
    cuda_event_wrapper event;

    ShuffleTableInfo(std::shared_ptr<cudf::table> t,
                     const std::vector<cudf::size_type>& v,
                     cuda_event_wrapper e)
        : table(t), partition_indices(v), event(e) {}
};

/**
 * @brief Class for managing async shuffle of cudf::tables using NCCL
 */
class GpuShuffleManager {
   private:
    // NCCL communicator
    ncclComm_t nccl_comm = nullptr;

    // MPI communicator for CPU communication between ranks
    // with GPUs assigned
    MPI_Comm mpi_comm = MPI_COMM_NULL;

    // Number of processes
    int n_ranks;

    // Current rank
    int rank;

    // Stream for CUDA operations
    cudaStream_t stream = nullptr;

    // GPU device ID
    rmm::cuda_device_id gpu_id;

    std::vector<GpuShuffle> inflight_shuffles;

    // Tag counter for shuffles, each shuffle uses 3 tags
    // and they can't overlap
    int curr_tag = 0;

    const int MAX_TAG_VAL;

    // This is used to coordinate the start of shuffles across ranks
    DoShuffleCoordination shuffle_coordination;

    // IBarrier to know when all ranks are done sending data
    MPI_Request global_completion_req = MPI_REQUEST_NULL;
    int global_completion = false;
    bool complete_signaled = false;

    std::vector<ShuffleTableInfo> tables_to_shuffle;

    /**
     * @brief Initialize NCCL communicator
     */
    void initialize_nccl();

    /**
     * @brief Once we've determined we will shuffle, start the shuffle by
     * partitioning the table and posting sends/receives
     */
    void do_shuffle();

    bool data_ready_to_send() {
        return !this->tables_to_shuffle.empty() &&
               this->tables_to_shuffle.back().event.ready();
    }

   public:
    GpuShuffleManager();
    ~GpuShuffleManager();

    /**
     * @brief Shuffle a cudf table across all ranks
     * @param table Input table to shuffle
     * @param partition_indices Column indices to use for partitioning
     */
    void shuffle_table(std::shared_ptr<cudf::table> table,
                       const std::vector<cudf::size_type>& partition_indices,
                       cuda_event_wrapper event);

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
     * @brief Get the underlying MPI communicator
     * @return MPI_Comm
     */
    MPI_Comm get_mpi_comm() const { return mpi_comm; }

    /**
     * @brief Check if there are any inflight shuffles
     * @return true if there are inflight shuffles, false otherwise
     */
    bool all_complete();

    /**
     * @brief Idempotent call to signify that this rank has no more data to send
     */
    void complete();

    bool is_available() const { return true; }
};

/**
 * @brief Hash partition function for GPU tables
 * @param table Input table
 * @param column_indices Column indices to hash
 * @param num_partitions Number of partitions
 * @param stream CUDA stream to use for partitioning
 * @return Pair of partitioned table and partition indices
 */
std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>>
hash_partition_table(std::shared_ptr<cudf::table> table,
                     const std::vector<cudf::size_type>& column_indices,
                     cudf::size_type num_partitions,
                     cudaStream_t stream = cudf::get_default_stream());

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

#else
// Empty implementation when CUDF is not available
class GpuShuffleManager {
   public:
    explicit GpuShuffleManager() = default;
    bool is_available() const { return false; }
};
#endif
