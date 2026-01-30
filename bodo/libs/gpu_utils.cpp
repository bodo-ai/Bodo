#include "gpu_utils.h"

// #ifdef USE_CUDF
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
#include <rmm/cuda_device.hpp>
#include <rmm/device_uvector.hpp>
#include "../libs/_distributed.h"
#include "_utils.h"
#include "cuda_runtime_api.h"

GpuShuffleManager::GpuShuffleManager()
    : gpu_id(get_gpu_id()),
      cuda_device_raii(rmm::cuda_set_device_raii(gpu_id)) {
    // Create a subcommunicator with only ranks that have GPUs assigned
    this->mpi_comm = get_gpu_mpi_comm(this->gpu_id);
    if (mpi_comm == MPI_COMM_NULL) {
        return;
    }

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
    int ret = MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, mpi_comm);
    if (ret != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Bcast failed");
    }

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
    auto [partitioned_table, partition_sizes] =
        hash_partition_table(table, partition_indices, n_ranks);
    // Pack the tables for sending
    std::vector<cudf::packed_table> packed_tables = cudf::contiguous_split(
        partitioned_table->view(), partition_sizes, stream);

    GpuShuffle(std::move(packed_tables), mpi_comm, nccl_comm, stream,
               this->rank, this->curr_tag);
    // Each shuffle will use nranks * 3 tags for shuffling metadata/gpu data
    // sizes and metadata buffers
    this->curr_tag += this->n_ranks * 3;
}

std::vector<std::unique_ptr<cudf::table>> GpuShuffleManager::progress() {
    std::vector<std::unique_ptr<cudf::table>> received_tables;
    for (GpuShuffle& shuffle : this->inflight_shuffles) {
        std::optional<cudf::table_view> progress_res = shuffle.progress();
        // This makes a copy, not sure how to avoid it, we get packed_columns
        // back after the shuffle which have internal buffers that own the data
        // and only supports creating a cudf::table_view. I can't find a way to
        // move those buffers into a cudf::table without copying.
        if (progress_res.has_value()) {
            received_tables.push_back(
                std::make_unique<cudf::table>(progress_res.value()));
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

std::optional<cudf::table_view> GpuShuffle::progress() {
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
        this->send_metadata_sizes[dest_rank] =
            this->metadata_send_buffers[dest_rank]->size();
        CHECK_MPI(
            MPI_Issend(&this->send_metadata_sizes[dest_rank], 1, MPI_UINT64_T,
                       dest_rank, this->start_tag + dest_rank, mpi_comm,
                       &this->metadata_sizes_send_reqs[dest_rank]),
            "GpuShuffle::send_sizes: MPI_Issend failed:");
    }
    // Send GPU data sizes
    for (size_t dest_rank = 0; dest_rank < packed_send_buffers.size();
         dest_rank++) {
        this->send_gpu_sizes[dest_rank] =
            packed_send_buffers[dest_rank]->size();
        CHECK_MPI(
            MPI_Issend(&this->send_metadata_sizes[dest_rank], 1, MPI_UINT64_T,
                       dest_rank, this->start_tag + this->n_ranks + dest_rank,
                       mpi_comm, &this->gpu_sizes_send_reqs[dest_rank]),
            "GpuShuffle::send_sizes: MPI_Issend failed:");
    }
}
void GpuShuffle::recv_sizes() {
    for (size_t src_rank = 0; src_rank < metadata_recv_buffers.size();
         src_rank++) {
        CHECK_MPI(
            MPI_Irecv(&this->recv_metadata_sizes[src_rank], 1, MPI_UINT64_T,
                      src_rank, this->start_tag + src_rank, mpi_comm,
                      &this->metadata_sizes_recv_reqs[src_rank]),
            "GpuShuffle::recv_sizes: MPI_Irecv failed:");
    }
    for (size_t src_rank = 0; src_rank < packed_recv_buffers.size();
         src_rank++) {
        CHECK_MPI(
            MPI_Irecv(&this->recv_gpu_sizes[src_rank], 1, MPI_UINT64_T,
                      src_rank, this->start_tag + this->n_ranks + src_rank,
                      mpi_comm, &this->gpu_sizes_recv_reqs[src_rank]),
            "GpuShuffle::recv_sizes: MPI_Irecv failed:");
    }
}

void GpuShuffle::send_metadata() {
    for (size_t dest_rank = 0; dest_rank < metadata_send_buffers.size();
         dest_rank++) {
        CHECK_MPI(MPI_Issend(this->metadata_send_buffers[dest_rank]->data(),
                             this->metadata_send_buffers[dest_rank]->size(),
                             MPI_UINT8_T, dest_rank,
                             this->start_tag + 2 * this->n_ranks + dest_rank,
                             mpi_comm, &this->metadata_send_reqs[dest_rank]),
                  "GpuShuffle::send_metadata: MPI_Issend failed:");
    }
}

void GpuShuffle::recv_metadata() {
    for (size_t src_rank = 0; src_rank < metadata_recv_buffers.size();
         src_rank++) {
        CHECK_MPI(MPI_Irecv(this->metadata_recv_buffers[src_rank]->data(),
                            this->metadata_recv_buffers[src_rank]->size(),
                            MPI_UINT8_T, src_rank,
                            this->start_tag + 2 * this->n_ranks + src_rank,
                            mpi_comm, &this->metadata_recv_reqs[src_rank]),
                  "GpuShuffle::recv_metadata: MPI_Irecv failed:");
    }
}

void GpuShuffle::send_data() {
    // Send GPU data using NCCL
}

void GpuShuffle::recv_data() {
    // Receive GPU data using NCCL
}

void GpuShuffle::progress_waiting_for_sizes() {
    // Check if all sizes have been received
    assert(this->recv_state == GpuShuffleState::SIZES_INFLIGHT);
    int all_metadata_sizes_received;
    CHECK_MPI_TEST_ALL(
        this->metadata_sizes_recv_reqs, all_metadata_sizes_received,
        "GpuShuffle::progress_waiting_for_sizes: MPI_Test failed:");
    int all_gpu_sizes_received;
    CHECK_MPI_TEST_ALL(
        this->gpu_sizes_recv_reqs, all_gpu_sizes_received,
        "GpuShuffle::progress_waiting_for_sizes: MPI_Test failed:");
    if (all_metadata_sizes_received && all_gpu_sizes_received) {
        // Allocate receive buffers based on received sizes
        for (size_t src_rank = 0; src_rank < packed_recv_buffers.size();
             src_rank++) {
            this->packed_recv_buffers[src_rank] =
                std::make_unique<rmm::device_buffer>(
                    this->recv_gpu_sizes[src_rank], stream);
        }
        for (size_t src_rank = 0; src_rank < metadata_recv_buffers.size();
             src_rank++) {
            this->metadata_recv_buffers[src_rank] =
                std::make_unique<std::vector<uint8_t>>(
                    this->recv_metadata_sizes[src_rank]);
        }
        // Deallocate size data
        this->recv_metadata_sizes.clear();
        this->recv_gpu_sizes.clear();
        this->metadata_sizes_recv_reqs.clear();
        this->gpu_sizes_recv_reqs.clear();

        // Start receiving metadata and data
        this->recv_metadata();
        this->recv_data();

        // Move to next state
        this->recv_state = GpuShuffleState::DATA_INFLIGHT;
    }
}

std::optional<cudf::table_view> GpuShuffle::progress_waiting_for_data() {
    return std::nullopt;
}

void GpuShuffle::progress_sending_sizes() {
    assert(this->send_state == GpuShuffleState::SIZES_INFLIGHT);
    int all_metadata_sizes_sent;
    CHECK_MPI_TEST_ALL(this->metadata_sizes_send_reqs, all_metadata_sizes_sent,
                       "GpuShuffle::progress_sending_sizes: MPI_Test failed:");
    int all_gpu_sizes_sent;
    CHECK_MPI_TEST_ALL(this->gpu_sizes_send_reqs, all_gpu_sizes_sent,
                       "GpuShuffle::progress_sending_sizes: MPI_Test failed:");
    if (all_metadata_sizes_sent && all_gpu_sizes_sent) {
        // Deallocate all size data
        this->send_metadata_sizes.clear();
        this->send_gpu_sizes.clear();
        this->metadata_sizes_send_reqs.clear();
        this->gpu_sizes_send_reqs.clear();
        // Move to next state
        this->send_state = GpuShuffleState::DATA_INFLIGHT;
    }
}

void GpuShuffle::progress_sending_data() {
    assert(this->send_state == GpuShuffleState::DATA_INFLIGHT);
    int all_metadata_sent;
    CHECK_MPI_TEST_ALL(this->metadata_send_reqs, all_metadata_sent,
                       "GpuShuffle::progress_sending_data: MPI_Test failed:");
    bool gpu_data_sent = 1;  // Placeholder for NCCL send completion check
    if (all_metadata_sent && gpu_data_sent) {
        // Deallocate all send data
        this->metadata_send_buffers.clear();
        this->packed_send_buffers.clear();
        this->metadata_send_reqs.clear();
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

    if (device_count == 0) {
        throw std::runtime_error("No CUDA devices available");
    }
    assert(n_ranks > device_count &&
           "More MPI ranks than available GPUs on node");
    rmm::cuda_device_id device_id(rank_on_node ? rank_on_node % device_count
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
