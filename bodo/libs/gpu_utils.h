#pragma once

#include <deque>
extern bool g_use_async;

#ifdef USE_CUDF
#include <mpi.h>
#include <cudf/contiguous_split.hpp>
#include <cudf/table/table.hpp>
#include <unordered_set>

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
     * @param tables Packed tables to send.
     * @param stream CUDA stream for synchronizing packed tables with MPI.
     * @param starting_msg_tag Starting message tag to use for posting the
     * messages that send the data buffers.
     * @param shuffle_comm MPI communicator for shuffle.
     * @param n_ranks Number of ranks in the shuffle.
     * @param broadcast If true, only one table is expected and it is sent to
     * all ranks. Otherwise, one table per rank is expected and each is sent to
     * the corresponding rank.
     */
    explicit GpuShuffleSendState(std::vector<cudf::packed_table> tables,
                                 cudaStream_t stream, int starting_msg_tag_,
                                 MPI_Comm shuffle_comm, size_t n_ranks,
                                 bool broadcast);

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
    bool sendDone();

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
    GpuShuffleRecvState(MPI_Status& status, MPI_Message& m,
                        cudaStream_t stream);

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
    int source;
    cudaStream_t stream;
    MPI_Request metadata_request = MPI_REQUEST_NULL;
    std::vector<MPI_Request> recv_requests;
    std::vector<uint64_t> sizes_vec;

    std::unique_ptr<std::vector<uint8_t>> recv_metadata_buffer;
    std::unique_ptr<rmm::device_buffer> packed_recv_buffer;
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

class BroadcastTableInfo {
   public:
    std::shared_ptr<cudf::table> table;
    cuda_event_wrapper event;

    BroadcastTableInfo(std::shared_ptr<cudf::table> t, cuda_event_wrapper e)
        : table(t), event(e) {}
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
class GpuTableManager : public GpuMpiManager {
   private:
    // IBarrier to know when all ranks are fully done
    bool is_last_barrier_started = false;
    MPI_Request is_last_request = MPI_REQUEST_NULL;

    // Keep track of inflight tags to avoid tag collisions.
    std::unordered_set<int> inflight_tags;

    std::vector<GpuShuffleSendState> send_states;
    std::vector<GpuShuffleRecvState> recv_states;

    /**
     * @brief Once we've determined we will shuffle, start the shuffle by
     * partitioning the table and posting sends/receives
     */
    void do_shuffle();

   protected:
    virtual std::vector<cudf::packed_table> getNextPerRankTables(
        bool& do_broadcast) = 0;
    virtual bool hasMoreTables() = 0;
    virtual bool tableReadyToSend() = 0;
    virtual std::vector<std::shared_ptr<cudf::table>> ownAndClear() = 0;

    void shuffle_irecv();

    std::vector<std::shared_ptr<cudf::table>> consume_completed_recvs();

   public:
    bool global_is_last = false;

    /**
     * @brief Progress any inflight shuffles
     * @return Optional vector of tables received from all ranks if any were
     * received.
     */
    std::vector<std::shared_ptr<cudf::table>> progress(const bool is_last);

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
};

class GpuShuffleManager : public GpuTableManager {
   private:
    std::vector<ShuffleTableInfo> tables_to_shuffle;

    bool tableReadyToSend() {
        return !this->tables_to_shuffle.empty() &&
               this->tables_to_shuffle.back().event.ready();
    }

    std::vector<cudf::packed_table> getNextPerRankTables(bool& do_broadcast);

    bool hasMoreTables() { return !tables_to_shuffle.empty(); }

   public:
    /**
     * @brief Shuffle a cudf table across all ranks
     * @param table Input table to shuffle
     * @param partition_indices Column indices to use for partitioning
     */
    void append_batch(std::shared_ptr<cudf::table> table,
                      const std::vector<cudf::size_type>& partition_indices,
                      std::shared_ptr<StreamAndEvent> se);

    bool is_available() const { return true; }

    std::vector<std::shared_ptr<cudf::table>> ownAndClear() {
        std::vector<std::shared_ptr<cudf::table>> out_tables;
        for (auto& shuffle_info : this->tables_to_shuffle) {
            out_tables.push_back(shuffle_info.table);
        }
        this->tables_to_shuffle.clear();
        return out_tables;
    }
};

class GpuTableBroadcastManager : public GpuTableManager {
   private:
    std::vector<BroadcastTableInfo> tables_to_broadcast;

    bool tableReadyToSend() {
        return !this->tables_to_broadcast.empty() &&
               this->tables_to_broadcast.back().event.ready();
    }

    std::vector<cudf::packed_table> getNextPerRankTables(bool& do_broadcast);

    bool hasMoreTables() { return !tables_to_broadcast.empty(); }

   public:
    void broadcast_table(std::shared_ptr<cudf::table> table,
                         std::shared_ptr<StreamAndEvent> se);

    bool is_available() const { return true; }

    std::vector<std::shared_ptr<cudf::table>> ownAndClear() override {
        std::vector<std::shared_ptr<cudf::table>> out_tables;
        for (auto& shuffle_info : this->tables_to_broadcast) {
            out_tables.push_back(shuffle_info.table);
        }
        this->tables_to_broadcast.clear();
        return out_tables;
    }
};

/**
 * @brief Class for managing async all-gather of cudf::tables using MPI.
 * Every rank broadcasts its table to all other ranks.
 */
class GpuTableAllGatherManager : public GpuTableBroadcastManager {
   public:
    void append_batch(std::shared_ptr<cudf::table> table,
                      std::shared_ptr<StreamAndEvent> se) {
        this->broadcast_table(std::move(table), std::move(se));
    }
};

/**
 * @brief Class for managing async shuffle of cudf::tables using MPI where
 * tables are shuffled based on range partitioning. Each rank sends one table
 * to each other rank, where the table sent to rank i contains rows for which
 * the partitioning column value falls within the range assigned to rank i.
 */
class GpuRangeShuffleManager : public GpuTableManager {
   private:
    struct RangeShuffleTableInfo {
        std::shared_ptr<cudf::table> table;
        std::vector<cudf::size_type> split_indices;
        cuda_event_wrapper event;

        RangeShuffleTableInfo(std::shared_ptr<cudf::table> t,
                              std::vector<cudf::size_type> s,
                              cuda_event_wrapper e)
            : table(std::move(t)),
              split_indices(std::move(s)),
              event(std::move(e)) {}
    };
    std::vector<RangeShuffleTableInfo> tables_to_shuffle;

    bool tableReadyToSend() override {
        return !this->tables_to_shuffle.empty() &&
               this->tables_to_shuffle.back().event.ready();
    }

    std::vector<cudf::packed_table> getNextPerRankTables(
        bool& do_broadcast) override;

    bool hasMoreTables() override { return !tables_to_shuffle.empty(); }

   public:
    /**
     * @brief Shuffle a cudf table across all ranks using range partitioning,
     * essentially a IMPIAlltoallv. We use this since CUDA aware MPI
     * doesn't support non-blocking alltoallv.
     * @param table Input table to shuffle
     * @param split_indices Row indices where the table should be split for each
     * rank
     * @param se Stream and event for synchronization
     */
    void append_batch(std::shared_ptr<cudf::table> table,
                      std::vector<cudf::size_type> split_indices,
                      std::shared_ptr<StreamAndEvent> se);

    bool is_available() const { return true; }

    std::vector<std::shared_ptr<cudf::table>> ownAndClear() override {
        std::vector<std::shared_ptr<cudf::table>> out_tables;
        for (auto& info : this->tables_to_shuffle) {
            out_tables.push_back(info.table);
        }
        this->tables_to_shuffle.clear();
        return out_tables;
    }
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
 * @param local_buf: device buffer owned by this rank to send (may be size 0).
 * @param stream: CUDA stream to perform operations on.
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

/**
 * @brief Sets specific elements in a boolean column to `Value` based on an
 * array of indices.
 *
 * @details This function iterates over the `indices` column and sets the
 * corresponding row in `target_bools` to `Value`. If the `indices` column
 * contains null values, those specific indices are safely ignored.
 * * @warning This function does **not** perform bounds checking. The caller is
 * strictly responsible for ensuring that all valid values in the `indices`
 * column are `>= 0` and
 * `< target_bools.size()`. Out-of-bounds indices will result in undefined
 * behavior or memory access violations.
 * * @note This function only updates the data buffer of `target_bools`. It does
 * not modify the validity bitmask of the target column.
 *
 * @tparam Value               The boolean value to set.
 * @param[in,out] target_bools A mutable view of the boolean column to update.
 * Must be of type `cudf::type_id::BOOL8`.
 * @param[in] indices          A view of the indices to set. Must be of
 * type `cudf::type_id::INT32`. Can contain nulls.
 * @param[in] stream           CUDA stream used for device memory operations and
 * kernel launches.
 */
template <bool Value>
void cudf_set_bools_from_indices(cudf::mutable_column_view target_bools,
                                 cudf::column_view const indices,
                                 rmm::cuda_stream_view stream);

/**
 * @brief Get a cuda asynchronous memory resource instance.
 *
 * @note This function must be called after a rank's device id is set.
 *
 * @return std::shared_ptr<rmm::mr::device_memory_resource>
 */
std::shared_ptr<rmm::mr::device_memory_resource>
get_gpu_async_memory_resource();

/**
 * @brief Get a static Cuda memory resource reference for allocating buffers for
 * MPI to enable GPU Direct paths.
 *
 * @note Device id must remain the same for all calls.
 *
 * @return rmm::device_async_resource_ref
 */
rmm::device_async_resource_ref get_cuda_memory_resource_ref();

/**
 * @brief Get a device_uvector containing an iota sequence from 0 to n-1.
 * @param n The length of the iota sequence.
 * @param stream The CUDA stream to use for any device memory operations.
 * @return the device_uvector containing the iota sequence.
 */
rmm::device_uvector<cudf::size_type> make_uvector_iota(
    cudf::size_type n, rmm::cuda_stream_view stream);

#else
// Empty implementation when CUDF is not available
class GpuShuffleManager {
   public:
    explicit GpuShuffleManager() = default;
    bool is_available() const { return false; }
};
#endif
