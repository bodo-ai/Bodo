#pragma once

#include <deque>
extern bool g_use_async;

#ifdef USE_CUDF
#include <mpi.h>
#include <cudf/contiguous_split.hpp>
#include <cudf/table/table.hpp>

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
 * @brief Holds buffers and MPI requests for async shuffle sends for cudf tables
 (MPI_Issend calls). Buffers cannot be freed until send is completed.
 *
 */
class GpuShuffleSendState {
   public:
    /**
     * @brief Construct a new send state.
     *
     * @param starting_msg_tag Starting message tag to use for posting the
     * messages that send the data buffers.
     */
    explicit GpuShuffleSendState(std::vector<cudf::packed_table> tables,
                                 int starting_msg_tag_, MPI_Comm shuffle_comm,
                                 int n_ranks);

    /**
     * @brief Getter for starting_msg_tag.
     *
     * @return int
     */
    int get_starting_msg_tag() const { return this->starting_msg_tag; }

    /**
     * @brief Returns true if send is done which allows this state to be freed.
     *
     */
    bool sendDone() {
        int flag;
        CHECK_MPI_TEST_ALL(
            send_requests, flag,
            "[GpuShuffleSendState::sendDone] MPI error on MPI_Testall: ")
        return flag;
    }

   private:
    std::vector<MPI_Request> send_requests;
    // Starting message tag to use for posting the messages that send the data
    // buffers. This enables sending multiple tables to ranks (as long as the
    // sender, e.g. StreamSort, maintains sufficient state to not re-use tags
    // for concurrent sends).
    int starting_msg_tag = -1;

    // Buffers for metadata transfers to other ranks, these are used to
    // construct packed_columns. Indexed by destination rank
    std::vector<std::unique_ptr<std::vector<uint8_t>>> metadata_send_buffers;

    // Buffers for column data sent to other ranks, these are used to construct
    // packed_columns. Indexed by destination rank
    std::vector<std::unique_ptr<rmm::device_buffer>> packed_send_buffers;

    // We need to keep metadata around while the transfers are inflight
    std::vector<uint64_t> send_metadata_sizes;
};

/**
 * @brief Holds buffers and MPI requests for async shuffle recvs for a cudf
 table (MPI_Irecv calls). Buffers cannot be used or freed until recvs are
 completed.
 *
 */
class GpuShuffleRecvState {
   public:
    GpuShuffleRecvState(MPI_Status& status, MPI_Message& m);

    /**
     * @brief Returns a tuple of a boolean and a shared_ptr to a table. If
     * recvs of all arrays are done, the boolean will be true, and the table
     * will be non-null, otherwise the return value will be (false, NULL).
     * When the boolean is true this state can be freed.
     */
    std::pair<bool, std::shared_ptr<cudf::table>> recvDone(
        MPI_Comm shuffle_comm);

    /**
     * @brief test if the initial sizes metadata request posted is complete and
     * if so, get the sizes metadata and post receives for the data.
     *
     * @param shuffle_comm MPI communicator for shuffle
     */
    void TryRecvMetadataAndAllocArrs(MPI_Comm& shuffle_comm);

   private:
    MPI_Request metadata_request = MPI_REQUEST_NULL;
    std::vector<MPI_Request> recv_requests;
    int source;
    std::vector<uint64_t> sizes_vec;

    std::vector<uint8_t> recv_metadata_buffer;
    std::unique_ptr<rmm::device_buffer> packed_recv_buffer;
};

/**
 * @brief Holds information for inflight shuffle operations
 */
struct GpuShuffle {
    GpuShuffleState send_state = GpuShuffleState::SIZES_INFLIGHT;
    GpuShuffleState recv_state = GpuShuffleState::SIZES_INFLIGHT;
    MPI_Comm mpi_comm = MPI_COMM_NULL;
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
    // MPI_Requests for data transfers from other ranks
    // Indexed by sending rank
    std::unique_ptr<std::vector<MPI_Request>> data_send_reqs;
    // MPI_Requests for data transfers to other ranks
    // Indexed by destination rank
    std::unique_ptr<std::vector<MPI_Request>> data_recv_reqs;

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
               MPI_Comm mpi_comm_, cudaStream_t stream_, int n_ranks,
               int start_tag)
        : mpi_comm(mpi_comm_),
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
          data_send_reqs(std::make_unique<std::vector<MPI_Request>>(n_ranks)),
          data_recv_reqs(std::make_unique<std::vector<MPI_Request>>(n_ranks)),
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
 * @brief Class for handling mpi communication between GPU nodes.
 */
class GpuMpiManager {
   protected:
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

   public:
    GpuMpiManager();
    ~GpuMpiManager();

    int get_rank() const { return rank; }

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
     * @brief All GPU ranks send a device buffer to all other GPU ranks
     * @param local_buf - the local device buffer to send to all GPU ranks
     * @param stream - stream to perform the operations on
     * @return number of GPU rank length vector of their device_buffers
     */
    std::vector<std::unique_ptr<rmm::device_buffer>> all_gather_device_buffers(
        rmm::device_buffer const& local_buf, cudaStream_t stream);

    /**
     * @brief Sum all the local values from the GPU ranks
     * @param local - the local value to be added to the sum
     * @return the sum
     */
    uint64_t allreduce(uint64_t local);
};

/**
 * @brief Class for managing async shuffle of cudf::tables using MPI
 */
class GpuShuffleManager : public GpuMpiManager {
   private:
    // IBarrier to know when all ranks are fully done
    bool global_is_last = false;
    bool is_last_barrier_started = false;
    MPI_Request is_last_request = MPI_REQUEST_NULL;

    // Keep track of inflight tags to avoid tag collisions.
    std::unordered_set<int> inflight_tags;

    std::vector<GpuShuffleSendState> send_states;
    std::vector<GpuShuffleRecvState> recv_states;

    std::vector<ShuffleTableInfo> tables_to_shuffle;

    /**
     * @brief Once we've determined we will shuffle, start the shuffle by
     * partitioning the table and posting sends/receives
     */
    void do_shuffle();

    bool data_ready_to_send() {
        return !this->tables_to_shuffle.empty() &&
               this->tables_to_shuffle.back().event.ready();
    }

    void shuffle_irecv();

    std::vector<std::unique_ptr<cudf::table>> consume_completed_recvs();

   public:
    GpuShuffleManager();

    /**
     * @brief Shuffle a cudf table across all ranks
     * @param table Input table to shuffle
     * @param partition_indices Column indices to use for partitioning
     */
    void append_batch(std::shared_ptr<cudf::table> table,
                      const std::vector<cudf::size_type>& partition_indices,
                      std::shared_ptr<StreamAndEvent> se);

    /**
     * @brief Progress any inflight shuffles
     * @return Optional vector of tables received from all ranks if any were
     * received.
     */
    std::vector<std::unique_ptr<cudf::table>> progress(const bool is_last);

    bool SendRecvEmpty();

    /**
     * @brief TODO(ehsan): Return true if send/recv buffer sizes are over a
     threshold and this rank shouldn't get more input data to avoid potential
     OOM.
     */
    bool BuffersFull() { return false; }

    /**
     * @brief Sync local is_last flags across ranks to determine if all ranks
     * are fully done.
     *
     */
    bool sync_is_last(bool local_is_last);

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

/**
 * @brief Allgather a device buffer from each GPU-enabled rank to every other
 * GPU-enabled rank.
 * @param stream: CUDA stream to perform operations on.
 * @param local_buf: device buffer owned by this rank to send (may be size 0).
 * @return vector of length comm_size where element i is a unique_ptr to the
 * buffer sent by rank i. If a rank sent size 0, the corresponding vector
 * element will be nullptr.
 */
std::vector<std::unique_ptr<rmm::device_buffer>>
allgather_device_buffers_across_ranks(rmm::device_buffer const& local_buf,
                                      cudaStream_t stream);

/**
 * @brief Return whether the current rank has a GPU assigned (i.e. should
 * participate in GPU compute)
 *
 */
bool is_gpu_rank();

#else
// Empty implementation when CUDF is not available
class GpuShuffleManager {
   public:
    explicit GpuShuffleManager() = default;
    bool is_available() const { return false; }
};
#endif
