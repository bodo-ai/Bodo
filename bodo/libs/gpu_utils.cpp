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

GpuShuffleManager::GpuShuffleManager()
    : MAX_TAG_VAL(get_max_allowed_tag_value()) {}

void GpuShuffleManager::shuffle_table(
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

void GpuShuffleManager::do_shuffle() {
    std::vector<cudf::packed_table> packed_tables;
    if (data_ready_to_send()) {
        ShuffleTableInfo shuffle_table_info = this->tables_to_shuffle.back();
        this->tables_to_shuffle.pop_back();

        // Hash partition the table
        auto [partitioned_table, partition_start_rows] = hash_partition_table(
            shuffle_table_info.table, shuffle_table_info.partition_indices,
            n_ranks, this->stream);

        assert(partition_start_rows.size() == static_cast<size_t>(n_ranks));
        // Contiguous splits requires the split indices excluding the first 0
        // So we create a new vector from partition_start_rows[1..end]
        std::vector<cudf::size_type> splits = std::vector<cudf::size_type>(
            partition_start_rows.begin() + 1, partition_start_rows.end());
        // Pack the tables for sending
        packed_tables =
            cudf::contiguous_split(partitioned_table->view(), splits, stream);
    } else {
        // If we have no data to shuffle, we still need to create empty packed
        // tables for each rank so that the shuffle can proceed without special
        // casing empty sends/receives
        cudf::table empty_table(
            cudf::table_view(std::vector<cudf::column_view>{}));

        for (int i = 0; i < n_ranks; i++) {
            cudf::packed_columns empty_packed_columns =
                cudf::pack(empty_table.view(), stream);
            cudf::packed_table empty_packed_table(
                empty_table.view(), std::move(empty_packed_columns));
            packed_tables.push_back(std::move(empty_packed_table));
        }
    }

    assert(packed_tables.size() == static_cast<size_t>(n_ranks));

    this->inflight_shuffles.emplace_back(std::move(packed_tables), mpi_comm,
                                         stream, this->n_ranks, this->curr_tag);

    // Each shuffle will use 4 tags for shuffling metadata/gpu data
    // sizes and metadata buffers
    if (inflight_shuffles.size() * 4 > static_cast<size_t>(MAX_TAG_VAL)) {
        throw std::runtime_error(
            "Exceeded maximum number of inflight shuffles");
    }
    this->curr_tag = (this->curr_tag + 4) % MAX_TAG_VAL;
}

std::vector<std::unique_ptr<cudf::table>> GpuShuffleManager::progress() {
    // If complete has been signaled and there are no inflight shuffles or
    // tables to shuffle, we can start the global completion barrier. This needs
    // to be called on all ranks so they know when
    // they can exit the pipeline.
    if (is_gpu_rank() && this->complete_signaled && inflight_shuffles.empty() &&
        tables_to_shuffle.empty() &&
        global_completion_req == MPI_REQUEST_NULL && !global_completion) {
        CHECK_MPI(MPI_Ibarrier(mpi_comm, &global_completion_req),
                  "GpuShuffleManager::complete: MPI_Ibarrier failed:");
    }

    if (mpi_comm == MPI_COMM_NULL || this->all_complete()) {
        return {};
    }

    if (this->shuffle_coordination.req == MPI_REQUEST_NULL) {
        // Coordinate when to shuffle by doing an allreduce, ranks with data
        // send 1, ranks without data send 0, this way all ranks will know when
        // a shuffle is needed and can call progress to start it
        this->shuffle_coordination.has_data =
            this->data_ready_to_send() ? 1 : 0;
        CHECK_MPI(
            MPI_Iallreduce(MPI_IN_PLACE, &this->shuffle_coordination.has_data,
                           1, MPI_INT, MPI_MAX, mpi_comm,
                           &this->shuffle_coordination.req),
            "GpuShuffleManager::progress: MPI_Iallreduce failed:");
    } else {
        int coordination_finished;
        CHECK_MPI(MPI_Test(&this->shuffle_coordination.req,
                           &coordination_finished, MPI_STATUS_IGNORE),
                  "GpuShuffleManager::progress: MPI_Test failed:");
        if (coordination_finished) {
            if (this->shuffle_coordination.has_data) {
                // If a shuffle is needed, start it
                this->do_shuffle();
            }
            // Reset coordination for next shuffle
            this->shuffle_coordination.req = MPI_REQUEST_NULL;
        }
    }

    std::vector<std::unique_ptr<cudf::table>> received_tables;
    while (!this->inflight_shuffles.empty()) {
        auto progress_res = this->inflight_shuffles.front().progress();
        if (progress_res.has_value()) {
            received_tables.push_back(std::move(progress_res.value()));
        }

        if (this->inflight_shuffles.front().send_state ==
                GpuShuffleState::COMPLETED &&
            this->inflight_shuffles.front().recv_state ==
                GpuShuffleState::COMPLETED) {
            this->inflight_shuffles.pop_front();
        } else {
            // Only progress one inflight_shuffle at a time to ensure consistent
            // ordering across ranks (TODO(ehsan): update since we replaced NCCL
            // with MPI?).
            break;
        }
    }
    return received_tables;
}

std::optional<std::unique_ptr<cudf::table>> GpuShuffle::progress() {
    switch (this->send_state) {
        case GpuShuffleState::SIZES_INFLIGHT: {
            this->progress_sending_sizes();
            break;
        }
        case GpuShuffleState::DATA_INFLIGHT: {
            this->progress_sending_data();
            break;
        }
        case GpuShuffleState::COMPLETED: {
            break;
        }
    }

    switch (this->recv_state) {
        case GpuShuffleState::SIZES_INFLIGHT: {
            this->progress_waiting_for_sizes();
            return std::nullopt;
        } break;
        case GpuShuffleState::DATA_INFLIGHT: {
            return this->progress_waiting_for_data();
        } break;
        case GpuShuffleState::COMPLETED: {
            return std::nullopt;
        } break;
    }
    return std::nullopt;
}

void GpuShuffle::send_sizes() {
    // Send metadata sizes
    for (size_t dest_rank = 0; dest_rank < metadata_send_buffers.size();
         dest_rank++) {
        (*this->send_metadata_sizes)[dest_rank] =
            this->metadata_send_buffers[dest_rank]->size();
        CHECK_MPI(MPI_Issend(&(*this->send_metadata_sizes)[dest_rank], 1,
                             MPI_UINT64_T, dest_rank, this->start_tag, mpi_comm,
                             &(*this->metadata_sizes_send_reqs)[dest_rank]),
                  "GpuShuffle::send_sizes: MPI_Issend failed:");
    }
    // Send GPU data sizes
    for (size_t dest_rank = 0; dest_rank < packed_send_buffers.size();
         dest_rank++) {
        (*this->send_gpu_sizes)[dest_rank] =
            packed_send_buffers[dest_rank]->size();
        CHECK_MPI(
            MPI_Issend(&(*this->send_gpu_sizes)[dest_rank], 1, MPI_UINT64_T,
                       dest_rank, this->start_tag + 1, mpi_comm,
                       &(*this->gpu_sizes_send_reqs)[dest_rank]),
            "GpuShuffle::send_sizes: MPI_Issend failed:");
    }
}
void GpuShuffle::recv_sizes() {
    for (size_t src_rank = 0; src_rank < metadata_recv_buffers.size();
         src_rank++) {
        CHECK_MPI(MPI_Irecv(&(*this->recv_metadata_sizes)[src_rank], 1,
                            MPI_UINT64_T, src_rank, this->start_tag, mpi_comm,
                            &(*this->metadata_sizes_recv_reqs)[src_rank]),
                  "GpuShuffle::recv_sizes: MPI_Irecv failed:");
    }
    for (size_t src_rank = 0; src_rank < packed_recv_buffers.size();
         src_rank++) {
        CHECK_MPI(MPI_Irecv(&(*this->recv_gpu_sizes)[src_rank], 1, MPI_UINT64_T,
                            src_rank, this->start_tag + 1, mpi_comm,
                            &(*this->gpu_sizes_recv_reqs)[src_rank]),
                  "GpuShuffle::recv_sizes: MPI_Irecv failed:");
    }
}

void GpuShuffle::send_metadata() {
    for (size_t dest_rank = 0; dest_rank < metadata_send_buffers.size();
         dest_rank++) {
        CHECK_MPI(MPI_Issend(this->metadata_send_buffers[dest_rank]->data(),
                             this->metadata_send_buffers[dest_rank]->size(),
                             MPI_UINT8_T, dest_rank, this->start_tag + 2,
                             mpi_comm, &(*this->metadata_send_reqs)[dest_rank]),
                  "GpuShuffle::send_metadata: MPI_Issend failed:");
    }
}

void GpuShuffle::recv_metadata() {
    for (size_t src_rank = 0; src_rank < metadata_recv_buffers.size();
         src_rank++) {
        CHECK_MPI(MPI_Irecv(this->metadata_recv_buffers[src_rank]->data(),
                            this->metadata_recv_buffers[src_rank]->size(),
                            MPI_UINT8_T, src_rank, this->start_tag + 2,
                            mpi_comm, &(*this->metadata_recv_reqs)[src_rank]),
                  "GpuShuffle::recv_metadata: MPI_Irecv failed:");
    }
}

void GpuShuffle::send_data() {
    for (size_t dest_rank = 0; dest_rank < packed_send_buffers.size();
         dest_rank++) {
        CHECK_MPI(MPI_Issend(packed_send_buffers[dest_rank]->data(),
                             packed_send_buffers[dest_rank]->size(),
                             MPI_UINT8_T, dest_rank, this->start_tag + 3,
                             mpi_comm, &(*this->data_send_reqs)[dest_rank]),
                  "GpuShuffle::send_data: MPI_Issend failed:");
    }
}

void GpuShuffle::recv_data() {
    for (size_t src_rank = 0; src_rank < packed_recv_buffers.size();
         src_rank++) {
        CHECK_MPI(MPI_Irecv(packed_recv_buffers[src_rank]->data(),
                            packed_recv_buffers[src_rank]->size(), MPI_UINT8_T,
                            src_rank, this->start_tag + 3, mpi_comm,
                            &(*this->data_recv_reqs)[src_rank]),
                  "GpuShuffle::recv_data: MPI_Irecv failed:");
    }
}

void GpuShuffle::progress_waiting_for_sizes() {
    // Check if all sizes have been received
    assert(this->recv_state == GpuShuffleState::SIZES_INFLIGHT);
    int all_metadata_sizes_received;
    CHECK_MPI_TEST_ALL((*this->metadata_sizes_recv_reqs),
                       all_metadata_sizes_received,
                       "GpuShuffle::progress_waiting_for_sizes: MPI_Test for "
                       "metadata sizes failed:");
    int all_gpu_sizes_received;
    CHECK_MPI_TEST_ALL((*this->gpu_sizes_recv_reqs), all_gpu_sizes_received,
                       "GpuShuffle::progress_waiting_for_sizes: MPI_Test for "
                       "recv sizes failed:");
    if (all_metadata_sizes_received && all_gpu_sizes_received) {
        // Allocate receive buffers based on received sizes
        for (size_t src_rank = 0; src_rank < packed_recv_buffers.size();
             src_rank++) {
            this->packed_recv_buffers[src_rank] =
                std::make_unique<rmm::device_buffer>(
                    (*this->recv_gpu_sizes)[src_rank], stream);
        }
        for (size_t src_rank = 0; src_rank < metadata_recv_buffers.size();
             src_rank++) {
            this->metadata_recv_buffers[src_rank] =
                std::make_unique<std::vector<uint8_t>>(
                    (*this->recv_metadata_sizes)[src_rank]);
        }
        // Deallocate size data
        this->recv_metadata_sizes->clear();
        this->recv_gpu_sizes->clear();
        this->metadata_sizes_recv_reqs->clear();
        this->gpu_sizes_recv_reqs->clear();

        // Start receiving metadata and data and send gpu data
        this->recv_metadata();
        this->recv_data();
        this->send_data();

        // Move to next state
        this->recv_state = GpuShuffleState::DATA_INFLIGHT;
    }
}

std::optional<std::unique_ptr<cudf::table>>
GpuShuffle::progress_waiting_for_data() {
    assert(this->recv_state == GpuShuffleState::DATA_INFLIGHT);
    int all_metadata_received;
    CHECK_MPI_TEST_ALL(
        (*this->metadata_recv_reqs), all_metadata_received,
        "GpuShuffle::progress_waiting_for_data: MPI_Test for metadata failed:");

    int gpu_data_received;
    CHECK_MPI_TEST_ALL(
        (*this->data_recv_reqs), gpu_data_received,
        "GpuShuffle::progress_waiting_for_data: MPI_Test for data failed:");

    if (all_metadata_received && gpu_data_received) {
        // Unpack received tables
        std::vector<cudf::table_view> table_views;
        std::vector<cudf::packed_columns> packed_recv_columns;
        for (size_t src_rank = 0; src_rank < packed_recv_buffers.size();
             src_rank++) {
            if (this->packed_recv_buffers[src_rank]->size() == 0) {
                continue;
            }
            packed_recv_columns.emplace_back(
                std::move(this->metadata_recv_buffers[src_rank]),
                std::move(this->packed_recv_buffers[src_rank]));
            table_views.push_back(cudf::unpack(packed_recv_columns.back()));
        }
        // Deallocate all receive data
        this->metadata_recv_buffers.clear();
        this->packed_recv_buffers.clear();
        this->metadata_recv_reqs->clear();
        this->data_recv_reqs->clear();
        // Move to completed state
        this->recv_state = GpuShuffleState::COMPLETED;

        std::unique_ptr<cudf::table> shuffle_res =
            cudf::concatenate(table_views);

        return {std::move(shuffle_res)};
    }
    return std::nullopt;
}

void GpuShuffle::progress_sending_sizes() {
    assert(this->send_state == GpuShuffleState::SIZES_INFLIGHT);
    int all_metadata_sizes_sent;
    CHECK_MPI_TEST_ALL((*this->metadata_sizes_send_reqs),
                       all_metadata_sizes_sent,
                       "GpuShuffle::progress_sending_sizes: MPI_Test failed:");
    int all_gpu_sizes_sent;
    CHECK_MPI_TEST_ALL(
        (*this->gpu_sizes_send_reqs), all_gpu_sizes_sent,
        "GpuShuffle::progress_sending_sizes: MPI_Test for GPU sizes failed:");
    if (all_metadata_sizes_sent && all_gpu_sizes_sent &&
        this->recv_state != GpuShuffleState::SIZES_INFLIGHT) {
        // Deallocate all size data
        this->send_metadata_sizes->clear();
        this->send_gpu_sizes->clear();
        this->metadata_sizes_send_reqs->clear();
        this->gpu_sizes_send_reqs->clear();
        // Move to next state
        this->send_state = GpuShuffleState::DATA_INFLIGHT;
    }
}

void GpuShuffle::progress_sending_data() {
    assert(this->send_state == GpuShuffleState::DATA_INFLIGHT);
    int all_metadata_sent;
    CHECK_MPI_TEST_ALL(
        (*this->metadata_send_reqs), all_metadata_sent,
        "GpuShuffle::progress_sending_data: MPI_Test for metadata failed:");
    int gpu_data_sent;
    CHECK_MPI_TEST_ALL(
        (*this->data_send_reqs), gpu_data_sent,
        "GpuShuffle::progress_sending_data: MPI_Test for GPU data failed:");

    if (all_metadata_sent && gpu_data_sent) {
        // Deallocate all send data
        this->metadata_send_buffers.clear();
        this->packed_send_buffers.clear();
        this->metadata_send_reqs->clear();
        this->data_send_reqs->clear();
        // Move to completed state
        this->send_state = GpuShuffleState::COMPLETED;
    }
}

void GpuShuffleManager::complete() { this->complete_signaled = true; }

bool GpuShuffleManager::all_complete() {
    if (!is_gpu_rank()) {
        return true;
    }
    if (global_completion_req != MPI_REQUEST_NULL) {
        CHECK_MPI(MPI_Test(&global_completion_req, &this->global_completion,
                           MPI_STATUS_IGNORE),
                  "GpuShuffleManager::all_complete: MPI_Test failed:");
        if (global_completion) {
            // If global completion is reached, we can cancel any inflight
            // shuffle coordination since we know all data has been sent
            if (this->shuffle_coordination.req != MPI_REQUEST_NULL) {
                // CHECK_MPI(
                //     MPI_Cancel(&this->shuffle_coordination.req),
                //     "GpuShuffleManager::all_complete: MPI_Cancel failed:");
            }
            this->shuffle_coordination.req = MPI_REQUEST_NULL;
            this->global_completion_req = MPI_REQUEST_NULL;
        }
    }
    return this->global_completion && inflight_shuffles.empty() &&
           tables_to_shuffle.empty();
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

#endif  // USE_CUDF
