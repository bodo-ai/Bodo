#include "gpu_utils.h"

// #ifdef USE_CUDF
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
#include "_utils.h"
#include "cuda_runtime_api.h"

GpuShuffleManager::GpuShuffleManager() : gpu_id(get_gpu_id()) {
    // There's probably a more robust way to handle this

    // Create a subcommunicator with only ranks that have GPUs assigned
    this->mpi_comm = get_gpu_mpi_comm(this->gpu_id);
    if (mpi_comm == MPI_COMM_NULL) {
        return;
    }
    CHECK_CUDA(cudaSetDevice(this->gpu_id.value()));

    // Get rank and size
    MPI_Comm_rank(mpi_comm, &this->rank);
    MPI_Comm_size(mpi_comm, &this->n_ranks);

    // Create CUDA stream
    cudaStreamCreateWithFlags(&this->stream, cudaStreamNonBlocking);

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
    CHECK_MPI(MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, mpi_comm),
              "GpuShuffleManager::initialize_nccl: MPI_Bcast failed:");

    // Initialize NCCL communicator
    CHECK_NCCL(ncclCommInitRank(&nccl_comm, n_ranks, nccl_id, rank));
}

void GpuShuffleManager::shuffle_table(
    std::shared_ptr<cudf::table> table,
    const std::vector<cudf::size_type>& partition_indices) {
    if (mpi_comm == MPI_COMM_NULL) {
        return;
    }
    // Hash partition the table
    auto [partitioned_table, partition_start_rows] =
        hash_partition_table(table, partition_indices, n_ranks);

    assert(partition_start_rows.size() == static_cast<size_t>(n_ranks));
    // Contiguous splits requires the split indices excluding the first 0
    // So we create a new vector from partition_start_rows[1..end]
    std::vector<cudf::size_type> splits = std::vector<cudf::size_type>(
        partition_start_rows.begin() + 1, partition_start_rows.end());
    // Pack the tables for sending
    std::vector<cudf::packed_table> packed_tables =
        cudf::contiguous_split(partitioned_table->view(), splits, stream);

    assert(packed_tables.size() == static_cast<size_t>(n_ranks));

    this->inflight_shuffles.emplace_back(std::move(packed_tables), mpi_comm,
                                         nccl_comm, stream, this->n_ranks,
                                         this->curr_tag);

    // Each shuffle will use 3 tags for shuffling metadata/gpu data
    // sizes and metadata buffers
    this->curr_tag += 3;
}

std::vector<std::unique_ptr<cudf::table>> GpuShuffleManager::progress() {
    std::vector<std::unique_ptr<cudf::table>> received_tables;
    for (GpuShuffle& shuffle : this->inflight_shuffles) {
        std::optional<std::unique_ptr<cudf::table>> progress_res =
            shuffle.progress();
        if (progress_res.has_value()) {
            received_tables.push_back(std::move(progress_res.value()));
        }
    }

    // Remove completed shuffles
    size_t i = 0;
    while (i < this->inflight_shuffles.size()) {
        if (this->inflight_shuffles[i].send_state ==
                GpuShuffleState::COMPLETED &&
            this->inflight_shuffles[i].recv_state ==
                GpuShuffleState::COMPLETED) {
            this->inflight_shuffles.erase(this->inflight_shuffles.begin() + i);
        } else {
            i++;
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
        CHECK_MPI(MPI_Isend(&(*this->send_metadata_sizes)[dest_rank], 1,
                            MPI_UINT64_T, dest_rank, this->start_tag, mpi_comm,
                            &(*this->metadata_sizes_send_reqs)[dest_rank]),
                  "GpuShuffle::send_sizes: MPI_Isend failed:");
        std::cout << "Sending metadata size: "
                  << (*this->send_metadata_sizes)[dest_rank] << " to rank "
                  << dest_rank << std::endl;
    }
    // Send GPU data sizes
    for (size_t dest_rank = 0; dest_rank < packed_send_buffers.size();
         dest_rank++) {
        (*this->send_gpu_sizes)[dest_rank] =
            packed_send_buffers[dest_rank]->size();
        CHECK_MPI(MPI_Isend(&(*this->send_gpu_sizes)[dest_rank], 1,
                            MPI_UINT64_T, dest_rank, this->start_tag + 1,
                            mpi_comm, &(*this->gpu_sizes_send_reqs)[dest_rank]),
                  "GpuShuffle::send_sizes: MPI_Isend failed:");
        std::cout << "Sending GPU data size: "
                  << (*this->send_gpu_sizes)[dest_rank] << " to rank "
                  << dest_rank << std::endl;
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
        CHECK_MPI(MPI_Isend(this->metadata_send_buffers[dest_rank]->data(),
                            this->metadata_send_buffers[dest_rank]->size(),
                            MPI_UINT8_T, dest_rank, this->start_tag + 2,
                            mpi_comm, &(*this->metadata_send_reqs)[dest_rank]),
                  "GpuShuffle::send_metadata: MPI_Isend failed:");
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
        if (packed_send_buffers[dest_rank]->size() == 0) {
            continue;
        }
        CHECK_NCCL(ncclSend(packed_send_buffers[dest_rank]->data(),
                            packed_send_buffers[dest_rank]->size(), ncclChar,
                            dest_rank, this->nccl_comm, this->stream));
    }
}

void GpuShuffle::recv_data() {
    for (size_t src_rank = 0; src_rank < packed_recv_buffers.size();
         src_rank++) {
        if (packed_recv_buffers[src_rank]->size() == 0) {
            continue;
        }
        CHECK_NCCL(ncclRecv(packed_recv_buffers[src_rank]->data(),
                            packed_recv_buffers[src_rank]->size(), ncclChar,
                            src_rank, this->nccl_comm, this->stream));
    }
}

void GpuShuffle::progress_waiting_for_sizes() {
    // Check if all sizes have been received
    assert(this->recv_state == GpuShuffleState::SIZES_INFLIGHT);
    int all_metadata_sizes_received;
    CHECK_MPI_TEST_ALL(
        (*this->metadata_sizes_recv_reqs), all_metadata_sizes_received,
        "GpuShuffle::progress_waiting_for_sizes: MPI_Test failed:");
    int all_gpu_sizes_received;
    CHECK_MPI_TEST_ALL(
        (*this->gpu_sizes_recv_reqs), all_gpu_sizes_received,
        "GpuShuffle::progress_waiting_for_sizes: MPI_Test failed:");
    if (all_metadata_sizes_received && all_gpu_sizes_received) {
        // Allocate receive buffers based on received sizes
        for (size_t src_rank = 0; src_rank < packed_recv_buffers.size();
             src_rank++) {
            this->packed_recv_buffers[src_rank] =
                std::make_unique<rmm::device_buffer>(
                    (*this->recv_gpu_sizes)[src_rank], stream);
            std::cout << "Allocated GPU recv buffer from rank " << src_rank
                      << " of size " << (*this->recv_gpu_sizes)[src_rank]
                      << std::endl;
        }
        for (size_t src_rank = 0; src_rank < metadata_recv_buffers.size();
             src_rank++) {
            this->metadata_recv_buffers[src_rank] =
                std::make_unique<std::vector<uint8_t>>(
                    (*this->recv_metadata_sizes)[src_rank]);
            std::cout << "Allocated metadata recv buffer from rank " << src_rank
                      << " of size " << (*this->recv_metadata_sizes)[src_rank]
                      << std::endl;
        }
        // Deallocate size data
        this->recv_metadata_sizes->clear();
        this->recv_gpu_sizes->clear();
        this->metadata_sizes_recv_reqs->clear();
        this->gpu_sizes_recv_reqs->clear();

        // Start receiving metadata and data and send gpu data
        this->recv_metadata();
        CHECK_NCCL(ncclGroupStart());
        // 1. Post ALL Receives first
        for (int i = 0; i < n_ranks; ++i) {
            if (packed_recv_buffers[i]->size() > 0) {
                CHECK_NCCL(ncclRecv(packed_recv_buffers[i]->data(),
                                    packed_recv_buffers[i]->size(), ncclChar, i,
                                    nccl_comm, stream));
                std::cout << "Posted NCCL recv from rank " << i << " of size "
                          << packed_recv_buffers[i]->size() << std::endl;
            }
        }
        // 2. Post ALL Sends next
        for (int i = 0; i < n_ranks; ++i) {
            if (packed_send_buffers[i]->size() > 0) {
                CHECK_NCCL(ncclSend(packed_send_buffers[i]->data(),
                                    packed_send_buffers[i]->size(), ncclChar, i,
                                    nccl_comm, stream));
                std::cout << "Posted NCCL send to rank " << i << "of size "
                          << packed_send_buffers[i]->size() << std::endl;
            }
        }

        CHECK_NCCL(ncclGroupEnd());
        CHECK_CUDA(cudaEventRecord(this->nccl_recv_event, this->stream));
        CHECK_CUDA(cudaEventRecord(this->nccl_send_event, this->stream));

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
        "GpuShuffle::progress_waiting_for_data: MPI_Test failed:");
    // Check if NCCL event has completed, this will return cudaSuccess if the
    // event has completed, cudaErrorNotReady if not yet completed,
    // or another error code if an error occurred.
    cudaError_t event_status = cudaEventQuery(nccl_recv_event);
    // Check for errors
    if (event_status != cudaErrorNotReady) {
        CHECK_CUDA(event_status);
    }
    bool gpu_data_received = (event_status == cudaSuccess);

    if (all_metadata_received && gpu_data_received) {
        // Unpack received tables
        std::vector<cudf::table_view> table_views(n_ranks);
        std::vector<cudf::packed_columns> packed_recv_columns(n_ranks);
        for (size_t src_rank = 0; src_rank < packed_recv_buffers.size();
             src_rank++) {
            packed_recv_columns[src_rank] = cudf::packed_columns(
                std::move(this->metadata_recv_buffers[src_rank]),
                std::move(this->packed_recv_buffers[src_rank]));
            table_views[src_rank] = cudf::unpack(packed_recv_columns[src_rank]);
        }
        // Deallocate all receive data
        this->metadata_recv_buffers.clear();
        this->packed_recv_buffers.clear();
        this->metadata_recv_reqs->clear();
        // Move to completed state
        this->recv_state = GpuShuffleState::COMPLETED;

        cudaStreamSynchronize(this->stream);
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
    CHECK_MPI_TEST_ALL((*this->gpu_sizes_send_reqs), all_gpu_sizes_sent,
                       "GpuShuffle::progress_sending_sizes: MPI_Test failed:");
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
    CHECK_MPI_TEST_ALL((*this->metadata_send_reqs), all_metadata_sent,
                       "GpuShuffle::progress_sending_data: MPI_Test failed:");
    // Check if NCCL event has completed, this will return cudaSuccess if the
    // event has completed, cudaErrorNotReady if not yet completed,
    // or another error code if an error occurred.
    cudaError_t event_status = cudaEventQuery(nccl_send_event);
    // Check for errors
    if (event_status != cudaErrorNotReady) {
        CHECK_CUDA(event_status);
    }
    bool gpu_data_sent = (event_status == cudaSuccess);

    if (all_metadata_sent && gpu_data_sent) {
        // Deallocate all send data
        this->metadata_send_buffers.clear();
        this->packed_send_buffers.clear();
        this->metadata_send_reqs->clear();
        // Move to completed state
        this->send_state = GpuShuffleState::COMPLETED;
    }
}

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

// #endif // USE_CUDF
