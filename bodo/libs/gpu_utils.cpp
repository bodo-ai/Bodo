#include "gpu_utils.h"
#include "vendored/simd-block-fixed-fpp.h"

bool g_use_async = false;

#ifdef USE_CUDF
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <cassert>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/hashing.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include "../libs/_distributed.h"
#include "../libs/streaming/_shuffle.h"
#include "_utils.h"
#include "cuda_runtime_api.h"

GpuMpiManager::GpuMpiManager() : gpu_id(get_gpu_id()) {
    // Create a subcommunicator with only ranks that have GPUs assigned
    this->mpi_comm = get_gpu_mpi_comm(this->gpu_id);
    if (mpi_comm == MPI_COMM_NULL) {
        return;
    }

    // Get rank and size
    MPI_Comm_rank(mpi_comm, &this->rank);
    MPI_Comm_size(mpi_comm, &this->n_ranks);

    // Create CUDA stream
    CHECK_CUDA(cudaStreamCreateWithFlags(&this->stream, cudaStreamNonBlocking));
}

GpuMpiManager::~GpuMpiManager() {
    if (mpi_comm == MPI_COMM_NULL) {
        return;
    }

    // Destroy CUDA stream
    if (stream) {
        cudaStreamDestroy(stream);
    }

    // Free MPI communicator
    MPI_Comm_free(&mpi_comm);
}

void GpuShuffleManager::append_batch(
    std::shared_ptr<cudf::table> table,
    const std::vector<cudf::size_type>& partition_indices,
    std::shared_ptr<StreamAndEvent> se) {
    if (mpi_comm == MPI_COMM_NULL) {
        return;
    }
    if (table->num_rows() == 0) {
        return;
    }
    this->tables_to_shuffle.emplace_back(table, partition_indices, se->event);
}

void GpuTableBroadcastManager::broadcast_table(
    std::shared_ptr<cudf::table> table, std::shared_ptr<StreamAndEvent> se) {
    if (mpi_comm == MPI_COMM_NULL) {
        return;
    }
    if (table->num_rows() == 0) {
        return;
    }
    this->tables_to_broadcast.emplace_back(table, se->event);
}

std::vector<cudf::packed_table> GpuShuffleManager::getNextPerRankTables() {
    std::vector<cudf::packed_table> packed_tables;
    if (!tableReadyToSend()) {
        throw std::runtime_error("getNextPerRankTables has no data");
    }

    ShuffleTableInfo shuffle_table_info = this->tables_to_shuffle.back();
    this->tables_to_shuffle.pop_back();

    // Hash partition the table
    auto [partitioned_table, partition_start_rows] = hash_partition_table(
        shuffle_table_info.table, shuffle_table_info.partition_indices, n_ranks,
        this->stream);

    assert(partition_start_rows.size() == static_cast<size_t>(n_ranks));
    // Contiguous splits requires the split indices excluding the first 0
    // So we create a new vector from partition_start_rows[1..end]
    std::vector<cudf::size_type> splits = std::vector<cudf::size_type>(
        partition_start_rows.begin() + 1, partition_start_rows.end());
    // Pack the tables for sending
    packed_tables =
        cudf::contiguous_split(partitioned_table->view(), splits, stream);
    return packed_tables;
}

std::vector<cudf::packed_table> make_replicas(cudf::table_view const& t,
                                              std::size_t N,
                                              rmm::cuda_stream_view stream) {
    // replicate N times
    std::vector<cudf::packed_table> out;
    out.reserve(N);
    for (std::size_t i = 0; i < N; ++i) {
        out.emplace_back(t, cudf::pack(t, stream));
    }

    return out;
}

std::vector<cudf::packed_table>
GpuTableBroadcastManager::getNextPerRankTables() {
    std::vector<cudf::packed_table> packed_tables;
    packed_tables.reserve(1);
    if (!tableReadyToSend()) {
        throw std::runtime_error("getNextPerRankTables has no data");
    }

    BroadcastTableInfo broadcast_table_info = this->tables_to_broadcast.back();
    this->tables_to_broadcast.pop_back();

    // packed_tables =
    //     make_replicas(broadcast_table_info.table->view(), n_ranks, stream);
    cudf::table_view tv = broadcast_table_info.table->view();
    // By doing only one return value signifies a broadcast.
    packed_tables.emplace_back(tv, cudf::pack(tv, stream));
    return packed_tables;
}

void GpuTableManager::do_shuffle() {
    std::vector<cudf::packed_table> packed_tables = getNextPerRankTables();
    assert(packed_tables.size() == static_cast<size_t>(n_ranks) ||
           packed_tables.size() == 1);

    int start_tag = get_next_available_tag(this->inflight_tags);
    if (start_tag == -1) {
        throw std::runtime_error(
            "[GpuShuffleManager::do_shuffle] Unable to get "
            "available MPI tag for shuffle send. All tags are inflight.");
    }

    this->send_states.emplace_back(std::move(packed_tables), start_tag,
                                   mpi_comm, static_cast<size_t>(n_ranks));
    this->inflight_tags.insert(start_tag);
}

// Similar to CPU version here:
// https://github.com/bodo-ai/Bodo/blob/5be77dc4ee731f674f679a4ff6f60ac2f231d326/bodo/libs/streaming/_shuffle.cpp#L688
std::vector<std::shared_ptr<cudf::table>> GpuTableManager::progress(
    const bool is_last) {
    if (mpi_comm == MPI_COMM_NULL || this->global_is_last) {
        return {};
    }

    // Short circuit shuffle if running on a single GPU
    if (n_ranks == 1) {
        std::vector<std::shared_ptr<cudf::table>> out_tables;
        for (auto& shuffle_info : this->tables_to_shuffle) {
            out_tables.push_back(shuffle_info.table);
        }
        this->tables_to_shuffle.clear();
        return out_tables;
    }

    // recv data first, but avoid receiving too much data at once
    if ((this->recv_states.size() == 0) || !this->BuffersFull()) {
        this->shuffle_irecv();
    }

    std::vector<std::shared_ptr<cudf::table>> received_tables =
        this->consume_completed_recvs();

    // Remove send state if recv done
    std::erase_if(this->send_states, [&](GpuShuffleSendState& s) {
        bool done = s.sendDone();
        if (done) {
            inflight_tags.erase(s.get_starting_msg_tag());
        }
        return done;
    });

    // TODO(ehsan): decide when to shuffle based on buffer size
    if (this->tableReadyToSend()) {
        this->do_shuffle();
    }

    return received_tables;
}

bool GpuTableManager::SendRecvEmpty() {
    return (this->send_states.empty() && this->recv_states.empty() &&
            !this->hasMoreTables());
}

// Similar to the CPU version here:
// https://github.com/bodo-ai/Bodo/blob/8706d2d4b4f957023090834b430682c09a275012/bodo/libs/streaming/_join.cpp#L3092
bool GpuTableManager::sync_is_last(bool local_is_last) {
    if (this->global_is_last) {
        return true;
    }

    local_is_last = local_is_last && this->SendRecvEmpty();

    if (!local_is_last) {
        return false;
    }

    if (!this->is_last_barrier_started) {
        CHECK_MPI(
            MPI_Ibarrier(MPI_COMM_WORLD, &this->is_last_request),
            "GpuShuffleManager::sync_is_last: MPI error on MPI_Ibarrier:");
        this->is_last_barrier_started = true;
        return false;
    } else {
        int flag = 0;
        CHECK_MPI(MPI_Test(&this->is_last_request, &flag, MPI_STATUS_IGNORE),
                  "GpuShuffleManager::sync_is_last: MPI error on MPI_Test:");
        if (flag) {
            this->global_is_last = true;
        }
        return flag;
    }
}

// Similar to the CPU version here:
// https://github.com/bodo-ai/Bodo/blob/b5f3663d4744982528c91d06bb437713a1d707b1/bodo/libs/streaming/_shuffle.cpp#L613
void GpuTableManager::shuffle_irecv() {
    while (true) {
        int flag;
        MPI_Status status;

        // NOTE: We use Improbe instead of Iprobe intentionally. Iprobe can
        // return true for the same message even when an Irecv for the message
        // has been posted (until the receive has actually begun). This can
        // cause hangs since we could end up posting two Irecvs for the same
        // message. Therefore, for robustness, we use Improbe, which returns a
        // message handle directly and exactly once.
        // 'PostLensRecv' uses `Imrecv` which will start receive on the
        // message using the message handle returned by Improbe.
        // Reference:
        // https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report/node70.htm which
        // states that "Unlike MPI_IPROBE, no other probe or receive operation
        // may match the message returned by MPI_IMPROBE.".
        MPI_Message m;
        CHECK_MPI(MPI_Improbe(MPI_ANY_SOURCE, SHUFFLE_METADATA_MSG_TAG,
                              this->mpi_comm, &flag, &m, &status),
                  "GpuShuffleManager::shuffle_irecv: MPI error on MPI_Improbe:")
        if (!flag) {
            break;
        }
        recv_states.emplace_back(status, m, stream);
    }

    for (auto& recv_state : recv_states) {
        recv_state.TryRecvMetadataAndAllocArrs(mpi_comm);
    }
}

std::vector<std::shared_ptr<cudf::table>>
GpuTableManager::consume_completed_recvs() {
    std::vector<std::shared_ptr<cudf::table>> out_tables;
    std::erase_if(recv_states, [&](GpuShuffleRecvState& s) {
        auto [done, table] = s.recvDone(mpi_comm);
        if (done) {
            out_tables.push_back(std::move(table));
        }
        return done;
    });
    return out_tables;
}

GpuShuffleSendState::GpuShuffleSendState(
    std::vector<cudf::packed_table> packed_tables, int starting_msg_tag_,
    MPI_Comm shuffle_comm, size_t n_ranks)
    : starting_msg_tag(starting_msg_tag_),
      metadata_send_buffers(n_ranks),
      packed_send_buffers(n_ranks),
      send_metadata_sizes(3 * n_ranks, 0) {
    bool broadcast;
    if (packed_tables.size() == 1) {
        broadcast = true;
        cudf::packed_table& table = packed_tables[0];
        packed_send_buffers[0] = std::move(table.data.gpu_data);
        metadata_send_buffers[0] = std::make_unique<std::vector<uint8_t>>(
            std::move(*table.data.metadata));

        // Prepare send buffers
        for (size_t dest_rank = 0; dest_rank < n_ranks; dest_rank++) {
            send_metadata_sizes[3 * dest_rank + 0] =
                static_cast<uint64_t>(starting_msg_tag);
            send_metadata_sizes[3 * dest_rank + 1] =
                metadata_send_buffers[0]->size();
            send_metadata_sizes[3 * dest_rank + 2] =
                packed_send_buffers[0]->size();
        }
    } else {
        broadcast = false;
        // Prepare send buffers
        for (size_t dest_rank = 0; dest_rank < packed_tables.size();
             dest_rank++) {
            cudf::packed_table& table = packed_tables[dest_rank];
            packed_send_buffers[dest_rank] = std::move(table.data.gpu_data);
            metadata_send_buffers[dest_rank] =
                std::make_unique<std::vector<uint8_t>>(
                    std::move(*table.data.metadata));

            send_metadata_sizes[3 * dest_rank + 0] =
                static_cast<uint64_t>(starting_msg_tag);
            send_metadata_sizes[3 * dest_rank + 1] =
                metadata_send_buffers[dest_rank]->size();
            send_metadata_sizes[3 * dest_rank + 2] =
                packed_send_buffers[dest_rank]->size();
        }
    }

    // Send sizes
    for (size_t dest_rank = 0; dest_rank < n_ranks; dest_rank++) {
        MPI_Request req;
        CHECK_MPI(
            MPI_Issend(&send_metadata_sizes[3 * dest_rank], 3, MPI_UINT64_T,
                       dest_rank, SHUFFLE_METADATA_MSG_TAG, shuffle_comm, &req),
            "GpuShuffleSendState: MPI_Issend for sizes failed:");
        this->send_requests.push_back(req);
    }

    // Send metadata
    for (size_t dest_rank = 0; dest_rank < metadata_send_buffers.size();
         dest_rank++) {
        MPI_Request req;
        CHECK_MPI(
            MPI_Issend(
                this->metadata_send_buffers[broadcast ? 0 : dest_rank]->data(),
                this->metadata_send_buffers[broadcast ? 0 : dest_rank]->size(),
                MPI_UINT8_T, dest_rank, starting_msg_tag, shuffle_comm, &req),
            "GpuShuffleSendState: MPI_Issend for metadata failed:");
        this->send_requests.push_back(req);
    }

    // Send data
    for (size_t dest_rank = 0; dest_rank < packed_send_buffers.size();
         dest_rank++) {
        MPI_Request req;
        CHECK_MPI(
            MPI_Issend(packed_send_buffers[broadcast ? 0 : dest_rank]->data(),
                       packed_send_buffers[broadcast ? 0 : dest_rank]->size(),
                       MPI_UINT8_T, dest_rank, starting_msg_tag + 1,
                       shuffle_comm, &req),
            "GpuShuffleSendState: MPI_Issend for data failed:");
        this->send_requests.push_back(req);
    }
}

bool GpuShuffleSendState::sendDone() {
    int flag;
    CHECK_MPI_TEST_ALL(
        send_requests, flag,
        "[GpuShuffleSendState::sendDone] MPI error on MPI_Testall: ")
    return flag;
}

GpuShuffleRecvState::GpuShuffleRecvState(MPI_Status& status, MPI_Message& m,
                                         cudaStream_t stream)
    : source(status.MPI_SOURCE), stream(stream) {
    assert(this->metadata_request == MPI_REQUEST_NULL);

    sizes_vec.resize(3);

    CHECK_MPI(MPI_Imrecv(sizes_vec.data(), 3, MPI_UINT64_T, &m,
                         &this->metadata_request),
              "GpuShuffleRecvState: MPI error on MPI_Imrecv:");
}

void GpuShuffleRecvState::TryRecvMetadataAndAllocArrs(MPI_Comm& shuffle_comm) {
    // Only post irecv if we haven't already
    if (!recv_requests.empty()) {
        return;
    }

    assert(this->metadata_request != MPI_REQUEST_NULL);

    int flag;
    CHECK_MPI(MPI_Test(&this->metadata_request, &flag, MPI_STATUS_IGNORE),
              "GpuShuffleRecvState::GetRecvMetadata: MPI error on MPI_Test:");
    if (!flag) {
        return;
    }
    this->metadata_request = MPI_REQUEST_NULL;

    // In the metadata, the starting tag to use is the first element, followed
    // by the lengths.
    int curr_tag = static_cast<int>(sizes_vec[0]);
    uint64_t metadata_size = sizes_vec[1];
    uint64_t data_size = sizes_vec[2];

    this->recv_metadata_buffer =
        std::make_unique<std::vector<uint8_t>>(metadata_size);
    this->packed_recv_buffer =
        std::make_unique<rmm::device_buffer>(data_size, stream);

    // recv metadata
    MPI_Request recv_req;
    CHECK_MPI(MPI_Irecv(this->recv_metadata_buffer->data(),
                        this->recv_metadata_buffer->size(), MPI_UINT8_T, source,
                        curr_tag, shuffle_comm, &recv_req),
              "GpuShuffle::recv_metadata: MPI_Irecv failed:");
    this->recv_requests.push_back(recv_req);

    // recv data
    MPI_Request data_recv_req;
    CHECK_MPI(MPI_Irecv(this->packed_recv_buffer->data(),
                        this->packed_recv_buffer->size(), MPI_UINT8_T, source,
                        curr_tag + 1, shuffle_comm, &data_recv_req),
              "GpuShuffle::recv_data: MPI_Irecv failed:");
    this->recv_requests.push_back(data_recv_req);
}

std::pair<bool, std::shared_ptr<cudf::table>> GpuShuffleRecvState::recvDone(
    MPI_Comm shuffle_comm) {
    if (recv_requests.empty()) {
        // Try receiving the length again and see if we can populate the data
        // requests.
        TryRecvMetadataAndAllocArrs(shuffle_comm);
        if (recv_requests.empty()) {
            return std::make_pair(false, nullptr);
        }
    }

    int flag;
    CHECK_MPI_TEST_ALL(
        recv_requests, flag,
        "[GpuShuffleRecvState::recvDone] MPI Error on MPI_Testall: ");

    if (!flag) {
        return std::make_pair(false, nullptr);
    }

    // Unpack received table
    cudf::packed_columns packed_recv_column =
        cudf::packed_columns(std::move(this->recv_metadata_buffer),
                             std::move(this->packed_recv_buffer));
    cudf::table_view table_view = cudf::unpack(packed_recv_column);

    // TODO(ehsan): avoid copy if possible
    std::shared_ptr<cudf::table> shuffle_res =
        std::make_shared<cudf::table>(table_view, stream);

    return std::make_pair(true, shuffle_res);
}

std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>>
hash_partition_table(std::shared_ptr<cudf::table> table,
                     const std::vector<cudf::size_type>& column_indices,
                     cudf::size_type num_partitions, cudaStream_t stream) {
    if (column_indices.empty()) {
        throw std::invalid_argument("Column indices cannot be empty");
    }

    // Extract the columns to hash
    std::vector<cudf::size_type> indices(column_indices.begin(),
                                         column_indices.end());

    // Partition the table based on the hash of the selected columns
    return cudf::hash_partition(table->view(), indices, num_partitions,
                                cudf::hash_id::HASH_MURMUR3,
                                cudf::DEFAULT_HASH_SEED, stream);
}

rmm::cuda_device_id get_gpu_id() {
    auto [n_ranks, rank_on_node] = dist_get_ranks_on_node();

    int device_count;
    cudaGetDeviceCount(&device_count);

    rmm::cuda_device_id device_id(rank_on_node < device_count ? rank_on_node
                                                              : -1);

    return device_id;
}

int get_cluster_cuda_device_count() {
    auto [n_ranks, rank_on_node] = dist_get_ranks_on_node();

    int local_device_count = 0;

    if (rank_on_node == 0) {
        // Note: We ignore the return code here to default to 0 if no driver
        // exists
        cudaGetDeviceCount(&local_device_count);
    }

    int device_count;
    CHECK_MPI(MPI_Allreduce(&local_device_count, &device_count, 1, MPI_INT,
                            MPI_SUM, MPI_COMM_WORLD),
              "get_cluster_cuda_device_count: MPI error on MPI_Allreduce:");
    return device_count;
}

MPI_Comm get_gpu_mpi_comm(rmm::cuda_device_id gpu_id) {
    MPI_Comm gpu_comm;
    int has_gpu = 0;
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

std::vector<std::unique_ptr<rmm::device_buffer>>
GpuMpiManager::all_gather_device_buffers(rmm::device_buffer const& local_buf,
                                         cudaStream_t stream) {
    // Exchange sizes (uint64_t)
    uint64_t local_size = static_cast<uint64_t>(local_buf.size());
    std::vector<uint64_t> all_sizes(static_cast<size_t>(n_ranks), 0);
    CHECK_MPI(MPI_Allgather(&local_size, 1, MPI_UINT64_T, all_sizes.data(), 1,
                            MPI_UINT64_T, mpi_comm),
              "allgather_device_buffers_across_ranks: MPI_Allgather failed:");

    // Allocate receive buffers for each rank (on device, using provided stream)
    std::vector<std::unique_ptr<rmm::device_buffer>> recv_buffers;
    recv_buffers.reserve(static_cast<size_t>(n_ranks));
    for (int i = 0; i < n_ranks; ++i) {
        uint64_t sz = all_sizes[static_cast<size_t>(i)];
        if (sz == 0) {
            recv_buffers.push_back(nullptr);
        } else {
            // allocate device buffer on the provided stream
            recv_buffers.push_back(
                std::make_unique<rmm::device_buffer>(sz, stream));
        }
    }

    // Wait for buffers to be ready.
    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::vector<MPI_Request> recv_reqs(n_ranks, MPI_REQUEST_NULL);

    for (int src = 0; src < n_ranks; ++src) {
        uint64_t sz = all_sizes[static_cast<size_t>(src)];
        if (sz == 0) {
            continue;
        }
        void* dst_ptr = recv_buffers[static_cast<size_t>(src)]->data();
        CHECK_MPI(MPI_Irecv(dst_ptr, static_cast<int>(sz), MPI_BYTE, src,
                            /*tag=*/0, mpi_comm, &recv_reqs[src]),
                  "MPI_Irecv failed:");
    }

    std::vector<MPI_Request> send_reqs(n_ranks, MPI_REQUEST_NULL);

    // Post sends: send this rank's buffer to every rank (including self)
    if (local_size > 0) {
        for (int dst = 0; dst < n_ranks; ++dst) {
            CHECK_MPI(
                MPI_Issend(local_buf.data(), static_cast<int>(local_size),
                           MPI_BYTE, dst, /*tag=*/0, mpi_comm, &send_reqs[dst]),
                "MPI_Issend failed:");
        }
    }

    std::vector<MPI_Request> all_reqs;
    all_reqs.reserve(2 * n_ranks);

    for (auto& r : recv_reqs)
        if (r != MPI_REQUEST_NULL)
            all_reqs.push_back(r);
    for (auto& r : send_reqs)
        if (r != MPI_REQUEST_NULL)
            all_reqs.push_back(r);

    CHECK_MPI(MPI_Waitall(static_cast<int>(all_reqs.size()), all_reqs.data(),
                          MPI_STATUSES_IGNORE),
              "MPI_Waitall failed:");

    return recv_buffers;
}

uint64_t GpuMpiManager::allreduce(uint64_t local) {
    uint64_t allsum = 0;
    CHECK_MPI(MPI_Allreduce(&local, &allsum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                            mpi_comm),
              "GpuMpiManager::allreduce: MPI error on MPI_Allreduce:");
    return allsum;
}

bool is_gpu_rank() {
    static bool is_gpu_rank = (get_gpu_id().value() != -1);
    return is_gpu_rank;
}

std::shared_ptr<rmm::mr::device_memory_resource>
get_gpu_async_memory_resource() {
    return std::make_shared<rmm::mr::cuda_async_memory_resource>();
}

#endif  // USE_CUDF
