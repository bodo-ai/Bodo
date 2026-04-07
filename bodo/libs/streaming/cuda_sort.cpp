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
    std::unique_ptr<cudf::table> sorted_table =
        cudf::sort_by_key(table->view(), table->select(key_indices),
                          column_order, null_precedence, input_se->stream);
    accumulation_buffer.push_back(std::move(sorted_table));
}

bool CudaSortState::FinalizeAccumulation(
    bool local_is_last, std::shared_ptr<StreamAndEvent> input_se) {
    if (state == State::ACCUMULATING && local_is_last) {
        if (is_gpu_rank()) {
            this->ExecutePsrsStep1(input_se->stream);
        }
        std::cout << "State transition: ACCUMULATING -> GATHERING_SAMPLES"
                  << std::endl;
        state = State::GATHERING_SAMPLES;
    }

    if (state == State::GATHERING_SAMPLES) {
        std::vector<std::shared_ptr<cudf::table>> received =
            sample_gatherer.progress(true);
        for (auto& t : received) {
            received_samples.push_back(std::move(t));
        }

        if (sample_gatherer.sync_is_last(true)) {
            if (is_gpu_rank()) {
                this->ExecutePsrsStep2(input_se->stream);
            }
            std::cout << "State transition: GATHERING_SAMPLES -> SHUFFLING"
                      << std::endl;
            state = State::SHUFFLING;
        }
    }

    if (state == State::SHUFFLING) {
        std::vector<std::shared_ptr<cudf::table>> shuffled_chunks =
            shuffle_manager.progress(true);

        for (auto& chunk : shuffled_chunks) {
            if (chunk && chunk->num_rows() > 0) {
                received_tables.push_back(std::move(chunk));
                std::cout << "Received shuffled chunk with "
                          << chunk->num_rows() << " rows" << std::endl;
            }
        }

        if (shuffle_manager.sync_is_last(true)) {
            std::cout << "State transition: SHUFFLING -> MERGING" << std::endl;
            state = State::MERGING;
        }
    }

    return (state == State::MERGING);
}

void CudaSortState::ExecutePsrsStep1(rmm::cuda_stream_view stream) {
    if (!is_gpu_rank()) {
        return;
    }

    // Merge all accumulated (sorted) batches
    if (!accumulation_buffer.empty()) {
        std::vector<cudf::table_view> views;
        for (const auto& table : accumulation_buffer) {
            views.push_back(table->view());
        }
        local_table = cudf::merge(views, key_indices, column_order,
                                  null_precedence, stream);
        accumulation_buffer.clear();
    } else {
        local_table = empty_table_from_arrow_schema(schema->ToArrowSchema());
    }

    MPI_Comm comm = sample_gatherer.get_mpi_comm();
    int n_ranks = 0;
    MPI_Comm_size(comm, &n_ranks);

    // Regular Sampling
    size_t n = local_table->num_rows();
    std::vector<cudf::size_type> sample_indices;
    for (int i = 0; i < n_ranks; ++i) {
        if (n > 0) {
            // PSRS says sample local indices by
            // i(n/p**2) where n is the global data size
            // and p is the number of ranks. We don't know n so we approximate
            // it by assuming an even distribution of n/p per rank, which
            // simplifies the formula to i(n/p)
            sample_indices.push_back(
                static_cast<cudf::size_type>(i * (n / n_ranks)));
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

    // Start Broadcast of local samples
    local_samples = std::move(sample_table);
    auto se = std::make_shared<StreamAndEvent>(stream, cuda_event_wrapper());
    sample_gatherer.append_batch(local_samples, se);
}

void CudaSortState::ExecutePsrsStep2(rmm::cuda_stream_view stream) {
    if (!is_gpu_rank()) {
        return;
    }

    MPI_Comm comm = shuffle_manager.get_mpi_comm();
    int n_ranks = 0;
    MPI_Comm_size(comm, &n_ranks);

    // Combine received samples
    std::vector<cudf::table_view> all_sample_views;
    for (const auto& t : received_samples) {
        if (t && t->num_rows() > 0) {
            all_sample_views.push_back(t->view());
        }
    }

    std::unique_ptr<cudf::table> global_pivots = nullptr;
    if (!all_sample_views.empty()) {
        // Since each rank's samples are already sorted, we can merge them
        // instead of concatenating and re-sorting.
        std::vector<cudf::size_type> sample_key_indices(key_indices.size());
        std::iota(sample_key_indices.begin(), sample_key_indices.end(), 0);
        std::unique_ptr<cudf::table> sorted_samples =
            cudf::merge(all_sample_views, sample_key_indices, column_order,
                        null_precedence, stream);

        // Select P-1 pivots
        cudf::size_type n_total_samples = sorted_samples->num_rows();
        std::vector<cudf::size_type> pivot_indices;
        for (int i = 1; i < n_ranks; ++i) {
            size_t idx = static_cast<size_t>(i) * n_ranks + n_ranks / 2 - 1;
            // If we have a rank with less data than nranks
            // we may end up with pivot indices that are out of bounds, so we
            // clamp to n_total_samples
            if (idx < static_cast<size_t>(n_total_samples)) {
                pivot_indices.push_back(static_cast<cudf::size_type>(idx));
            } else {
                pivot_indices.push_back(n_total_samples - 1);
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
        }
    }

    if (!global_pivots) {
        global_pivots = empty_table_from_arrow_schema(this->key_schema);
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

    // Start Shuffle
    auto se = std::make_shared<StreamAndEvent>(stream, cuda_event_wrapper());
    shuffle_manager.append_batch(std::move(local_table),
                                 std::move(split_indices), se);
}

std::unique_ptr<cudf::table> CudaSortState::GetOutputBatch(
    bool& out_is_last, rmm::cuda_stream_view stream) {
    std::cout << "GetOutputBatch called in state: " << static_cast<int>(state)
              << std::endl;
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
