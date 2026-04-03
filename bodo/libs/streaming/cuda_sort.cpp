#include "cuda_sort.h"
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/merge.hpp>
#include <cudf/search.hpp>
#include <cudf/table/table_view.hpp>
#include <rmm/device_uvector.hpp>
#include "../_bodo_common.h"
#include "../_utils.h"
#include "../gpu_utils.h"
#include "_util.h"
#include "physical/operator.h"

#ifdef USE_CUDF

CudaSortState::CudaSortState(
    std::shared_ptr<bodo::Schema> schema,
    std::vector<cudf::size_type> const& key_indices,
    std::vector<cudf::order> const& column_order,
    std::vector<cudf::null_order> const& null_precedence)
    : schema(std::move(schema)),
      key_indices(key_indices),
      column_order(column_order),
      null_precedence(null_precedence) {
    // Schema for sort keys (used for sampling and pivots)
    std::vector<std::shared_ptr<arrow::Field>> key_fields;
    auto full_arrow_schema = this->schema->ToArrowSchema();
    for (auto idx : this->key_indices) {
        key_fields.push_back(full_arrow_schema->field(idx));
    }
    this->key_schema = std::make_shared<arrow::Schema>(std::move(key_fields));
}

void CudaSortState::ConsumeBatch(std::shared_ptr<cudf::table> table,
                                 std::shared_ptr<StreamAndEvent> input_se) {
    if (table->num_rows() == 0) {
        return;
    }
    accumulation_buffer.push_back(std::move(table));
}

bool CudaSortState::FinalizeAccumulation(bool local_is_last) {
    if (state == State::ACCUMULATING && local_is_last) {
        // Perform PSRS on the accumulated data
        ExecutePsrs(cudf::get_default_stream());
        state = State::SHUFFLING;
    }

    // Progress the shuffle
    std::vector<std::shared_ptr<cudf::table>> shuffled_chunks =
        shuffle_manager.progress(local_is_last);

    for (auto& chunk : shuffled_chunks) {
        if (chunk->num_rows() > 0) {
            received_tables.push_back(std::move(chunk));
        }
    }

    bool global_is_last = shuffle_manager.sync_is_last(local_is_last);
    if (global_is_last && state == State::SHUFFLING) {
        state = State::MERGING;
    }

    return global_is_last;
}

void CudaSortState::ExecutePsrs(rmm::cuda_stream_view stream) {
    if (!is_gpu_rank()) {
        return;
    }

    // Concatenate all accumulated batches
    std::shared_ptr<cudf::table> local_table;
    if (!accumulation_buffer.empty()) {
        std::vector<cudf::table_view> views;
        for (const auto& table : accumulation_buffer) {
            views.push_back(table->view());
        }
        local_table = cudf::concatenate(views, stream);
        accumulation_buffer.clear();

        // Local Sort
        local_table = cudf::sort_by_key(local_table->view(),
                                        local_table->select(key_indices),
                                        column_order, null_precedence, stream);
        auto local_arrow_table = convertGPUToArrow(
            {local_table, this->schema->ToArrowSchema(),
             std::make_shared<StreamAndEvent>(stream, cuda_event_wrapper())});
        int rank;
        MPI_Comm_rank(shuffle_manager.get_mpi_comm(), &rank);
        std::cout << "local table Rank " << rank << std::endl;
        std::cout << local_arrow_table->ToString() << std::endl;
    }

    MPI_Comm comm = shuffle_manager.get_mpi_comm();
    int n_ranks = 0;
    MPI_Comm_size(comm, &n_ranks);
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    // Regular Sampling
    size_t n = local_table->num_rows();
    std::vector<cudf::size_type> sample_indices;
    for (int i = 0; i < n_ranks; ++i) {
        if (n > 0) {
            sample_indices.push_back(
                static_cast<cudf::size_type>(i * n / n_ranks));
        }
    }

    std::unique_ptr<cudf::table> sample_table;
    if (!sample_indices.empty()) {
        rmm::device_uvector<cudf::size_type> d_sample_indices(
            sample_indices.size(), stream);
        CHECK_CUDA(
            cudaMemcpyAsync(d_sample_indices.data(), sample_indices.data(),
                            sample_indices.size() * sizeof(cudf::size_type),
                            cudaMemcpyHostToDevice, stream.value()));

        cudf::column_view sample_indices_view(
            cudf::data_type{cudf::type_to_id<cudf::size_type>()},
            d_sample_indices.size(), d_sample_indices.data(), nullptr, 0);

        sample_table =
            cudf::gather(local_table->select(key_indices), sample_indices_view,
                         cudf::out_of_bounds_policy::DONT_CHECK, stream);
    } else {
        sample_table = empty_table_from_arrow_schema(this->key_schema);
    }

    //  Global Pivot Selection
    // Convert samples to arrow for CPU processing on Rank 0
    std::shared_ptr<arrow::Table> local_samples = convertGPUToArrow(
        {std::shared_ptr<cudf::table>(std::move(sample_table)),
         this->key_schema,
         std::make_shared<StreamAndEvent>(stream, cuda_event_wrapper())});

    // Serialize local samples
    auto local_buf = SerializeTableToIPC(local_samples);
    int local_size = static_cast<int>(local_buf->size());

    // Gather sizes from all ranks
    std::vector<int> recv_counts(n_ranks);
    CHECK_MPI(MPI_Gather(&local_size, 1, MPI_INT, recv_counts.data(), 1,
                         MPI_INT, 0, comm),
              "ExecutePsrs: Failed to gather sample sizes");

    std::vector<int> displs(n_ranks, 0);
    int total_bytes = 0;
    if (rank == 0) {
        for (int i = 0; i < n_ranks; ++i) {
            displs[i] = total_bytes;
            total_bytes += recv_counts[i];
        }
    }

    std::vector<uint8_t> recv_buffer(total_bytes);
    CHECK_MPI(
        MPI_Gatherv(local_buf->data(), local_size, MPI_BYTE, recv_buffer.data(),
                    recv_counts.data(), displs.data(), MPI_BYTE, 0, comm),
        "ExecutePsrs: Failed to gather sample buffers");

    std::shared_ptr<cudf::table> global_pivots = nullptr;

    if (rank == 0) {
        std::vector<std::shared_ptr<arrow::Table>> all_samples;
        for (int i = 0; i < n_ranks; ++i) {
            if (recv_counts[i] > 0) {
                auto buf = arrow::Buffer::Wrap(recv_buffer.data() + displs[i],
                                               recv_counts[i]);
                all_samples.push_back(DeserializeIPC(std::move(buf)));
            }
        }

        auto combined_samples =
            arrow::ConcatenateTables(all_samples).ValueOrDie();

        GPU_DATA samples_gpu = convertArrowTableToGPU(
            combined_samples,
            std::make_shared<StreamAndEvent>(stream, cuda_event_wrapper()));
        std::unique_ptr<cudf::table> sorted_samples = cudf::sort(
            samples_gpu.table->view(), column_order, null_precedence, stream);

        // Select P-1 pivots
        cudf::size_type n_total_samples = sorted_samples->num_rows();
        std::vector<cudf::size_type> pivot_indices;
        for (int i = 1; i < n_ranks; ++i) {
            cudf::size_type idx = static_cast<cudf::size_type>(i * n_ranks - 1);
            if (idx < n_total_samples) {
                pivot_indices.push_back(idx);
            }
        }

        if (!pivot_indices.empty()) {
            rmm::device_uvector<cudf::size_type> d_pivot_indices(
                pivot_indices.size(), stream);
            CHECK_CUDA(
                cudaMemcpyAsync(d_pivot_indices.data(), pivot_indices.data(),
                                pivot_indices.size() * sizeof(cudf::size_type),
                                cudaMemcpyHostToDevice, stream.value()));

            cudf::column_view pivot_indices_view(
                cudf::data_type{cudf::type_to_id<cudf::size_type>()},
                d_pivot_indices.size(), d_pivot_indices.data(), nullptr, 0);

            global_pivots =
                cudf::gather(sorted_samples->view(), pivot_indices_view,
                             cudf::out_of_bounds_policy::DONT_CHECK, stream);
        } else {
            global_pivots = empty_table_from_arrow_schema(this->key_schema);
        }
    }

    // Broadcast pivots
    std::shared_ptr<arrow::Buffer> pivot_buf;
    int pivot_buf_size = 0;
    if (rank == 0) {
        auto h_pivots = convertGPUToArrow(
            {global_pivots, this->key_schema,
             std::make_shared<StreamAndEvent>(stream, cuda_event_wrapper())});
        std::cout << "Pivots: \n" << h_pivots->ToString() << std::endl;
        pivot_buf = SerializeTableToIPC(h_pivots);
        pivot_buf_size = static_cast<int>(pivot_buf->size());
    }

    CHECK_MPI(MPI_Bcast(&pivot_buf_size, 1, MPI_INT, 0, comm),
              "ExecutePsrs: Failed to broadcast pivot buffer size");

    std::vector<uint8_t> pivot_recv_buffer;
    if (rank != 0) {
        pivot_recv_buffer.resize(pivot_buf_size);
    }
    CHECK_MPI(MPI_Bcast(rank == 0 ? (void*)pivot_buf->data()
                                  : pivot_recv_buffer.data(),
                        pivot_buf_size, MPI_BYTE, 0, comm),
              "ExecutePsrs: Failed to broadcast pivots");

    if (rank != 0) {
        auto buf =
            arrow::Buffer::Wrap(pivot_recv_buffer.data(), pivot_buf_size);
        auto h_pivots = DeserializeIPC(std::move(buf));
        GPU_DATA pivots_gpu = convertArrowTableToGPU(
            h_pivots,
            std::make_shared<StreamAndEvent>(stream, cuda_event_wrapper()));
        global_pivots = std::move(pivots_gpu.table);
    }

    //  Partitioning
    std::vector<cudf::size_type> split_indices;
    if (global_pivots && global_pivots->num_rows() > 0) {
        auto d_splits = cudf::upper_bound(local_table->select(key_indices),
                                          global_pivots->view(), column_order,
                                          null_precedence, stream);
        // Copy d_splits back to host
        split_indices.resize(d_splits->size());
        CHECK_CUDA(cudaMemcpyAsync(split_indices.data(),
                                   d_splits->view().head<cudf::size_type>(),
                                   d_splits->size() * sizeof(cudf::size_type),
                                   cudaMemcpyDeviceToHost, stream.value()));
        // Ensure split_indices are ready on host before using them
        CHECK_CUDA(cudaStreamSynchronize(stream.value()));
    } else {
        // Fallback for single rank or no pivots
        for (int i = 1; i < n_ranks; i++) {
            split_indices.push_back(
                static_cast<cudf::size_type>(local_table->num_rows()));
        }
    }
    std::cout << "Rank " << rank << " split indices: ";
    for (auto idx : split_indices) {
        std::cout << idx << " ";
    }
    std::cout << std::endl;

    // 6. Start Shuffle
    auto se = std::make_shared<StreamAndEvent>(stream, cuda_event_wrapper());
    se->event.record(stream);
    shuffle_manager.append_batch(std::move(local_table),
                                 std::move(split_indices), se);
}

std::unique_ptr<cudf::table> CudaSortState::GetOutputBatch(
    bool& out_is_last, rmm::cuda_stream_view stream) {
    if (state != State::MERGING) {
        out_is_last = false;
        return empty_table_from_arrow_schema(schema->ToArrowSchema());
    }

    if (final_result == nullptr) {
        if (received_tables.empty()) {
            final_result =
                empty_table_from_arrow_schema(schema->ToArrowSchema());
        } else {
            std::vector<cudf::table_view> views;
            for (const auto& table : received_tables) {
                views.push_back(table->view());
            }
            // Multi-way Merge
            final_result = cudf::merge(views, key_indices, column_order,
                                       null_precedence, stream);
        }
        received_tables.clear();
    }

    out_is_last = true;
    state = State::DONE;
    return std::move(final_result);
}

#endif
