#include "cuda_join.h"
#include <arrow/array/util.h>
#include <arrow/compute/api_aggregate.h>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/reduction.hpp>
#include <rmm/cuda_stream_view.hpp>
#include "../../pandas/physical/operator.h"
#include "../_utils.h"
#include "_util.h"

std::shared_ptr<arrow::Table> SyncAndReduceGlobalStats(
    std::shared_ptr<arrow::Table> local_stats) {
    // Serialize local stats to bytes
    auto local_buf = SerializeTableToIPC(local_stats);
    int local_size = static_cast<int>(local_buf->size());

    // Gather sizes from all ranks
    int n_pes;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    std::vector<int> recv_counts(n_pes);
    CHECK_MPI(MPI_Allgather(&local_size, 1, MPI_INT, recv_counts.data(), 1,
                            MPI_INT, MPI_COMM_WORLD),
              "Failed to gather local stats sizes");

    // Calculate displacements
    std::vector<int> displs(n_pes);
    int total_bytes = 0;
    for (int i = 0; i < n_pes; ++i) {
        displs[i] = total_bytes;
        total_bytes += recv_counts[i];
    }

    // Gather all IPC buffers into one large blob on each rank
    std::vector<uint8_t> recv_buffer(total_bytes);
    CHECK_MPI(MPI_Allgatherv(local_buf->data(), local_size, MPI_BYTE,
                             recv_buffer.data(), recv_counts.data(),
                             displs.data(), MPI_BYTE, MPI_COMM_WORLD),
              "Failed to gather local stats buffers");

    // Deserialize all tables
    std::vector<std::shared_ptr<arrow::Table>> all_tables;
    for (int i = 0; i < n_pes; ++i) {
        if (recv_counts[i] > 0) {
            std::shared_ptr<arrow::Buffer> buf = arrow::Buffer::Wrap(
                recv_buffer.data() + displs[i], recv_counts[i]);
            all_tables.push_back(DeserializeIPC(std::move(buf)));
        }
    }

    auto concat_options =
        arrow::ConcatenateTablesOptions{.unify_schemas = true};
    auto combined_table =
        arrow::ConcatenateTables(all_tables, concat_options).ValueOrDie();

    auto min_col = combined_table->column(0);
    auto max_col = combined_table->column(1);

    auto global_min_scalar =
        std::static_pointer_cast<arrow::StructScalar>(
            arrow::compute::MinMax(min_col).ValueOrDie().scalar())
            ->field(0)
            .ValueOrDie();
    auto global_max_scalar =
        std::static_pointer_cast<arrow::StructScalar>(
            arrow::compute::MinMax(max_col).ValueOrDie().scalar())
            ->field(1)
            .ValueOrDie();

    // Construct the final 1 row table
    auto schema = local_stats->schema();

    return arrow::Table::Make(
        schema,
        {arrow::MakeArrayFromScalar(*global_min_scalar, 1).ValueOrDie(),
         arrow::MakeArrayFromScalar(*global_max_scalar, 1).ValueOrDie()});
}

void CudaHashJoin::build_hash_table(
    const std::vector<std::shared_ptr<cudf::table>>& build_chunks) {
    std::vector<cudf::table_view> build_views;
    // 1. Concatenate all build chunks into one contiguous table
    //    This is necessary because cudf::hash_join expects a single table_view
    for (const auto& chunk : build_chunks) {
        build_views.push_back(chunk->view());
    }
    this->_build_table = cudf::concatenate(build_views);
    if (this->_build_table->num_rows() == 0) {
        // If we don't have chunks we don't need a build table, we won't match
        // anything
        return;
    }

    // 2. Create the hash_join object
    //    This triggers the kernel that builds the hash table on the GPU.
    //    We maintain ownership of _join_handle to reuse it for probing.
    cudf::table_view build_view = _build_table->view();

    this->_join_handle = std::make_unique<cudf::hash_join>(
        build_view.select(this->build_key_indices), this->null_equality);
}

void CudaHashJoin::FinalizeBuild() {
    // Build the hash table if we have a gpu assigned to us
    if (this->build_shuffle_manager.get_mpi_comm() != MPI_COMM_NULL) {
        this->build_hash_table(this->_build_chunks);
    }

    std::shared_ptr<arrow::Schema> build_table_arrow_schema =
        this->build_table_schema->ToArrowSchema();

    for (const auto& col_idx : this->build_key_indices) {
        std::shared_ptr<arrow::Table> local_stats;
        if (this->build_shuffle_manager.get_mpi_comm() != MPI_COMM_NULL &&
            this->_build_table->num_rows()) {
            auto [min, max] =
                cudf::minmax(this->_build_table->get_column(col_idx).view());
            std::vector<std::unique_ptr<cudf::column>> columns;
            columns.emplace_back(cudf::make_column_from_scalar(*min, 1));
            columns.emplace_back(cudf::make_column_from_scalar(*max, 1));
            std::shared_ptr<cudf::table> stats_table =
                std::make_shared<cudf::table>(std::move(columns));

            std::vector<std::shared_ptr<arrow::Field>> fields = {
                arrow::field("min",
                             build_table_arrow_schema->field(col_idx)->type()),
                arrow::field("max",
                             build_table_arrow_schema->field(col_idx)->type())};
            GPU_DATA stats_gpu_data = {
                stats_table, std::make_shared<arrow::Schema>(std::move(fields)),
                make_stream_and_event(false)};
            local_stats = convertGPUToArrow(stats_gpu_data);
        } else {
            // If we don't have a GPU, we still need to participate in the
            // global stats reduction, so we create an table with null vals
            std::vector<std::shared_ptr<arrow::Field>> fields = {
                arrow::field("min",
                             build_table_arrow_schema->field(col_idx)->type()),
                arrow::field("max",
                             build_table_arrow_schema->field(col_idx)->type())};
            local_stats = arrow::Table::Make(
                std::make_shared<arrow::Schema>(std::move(fields)),
                {arrow::MakeArrayOfNull(
                     build_table_arrow_schema->field(col_idx)->type(), 1)
                     .ValueOrDie(),
                 arrow::MakeArrayOfNull(
                     build_table_arrow_schema->field(col_idx)->type(), 1)
                     .ValueOrDie()});
        }
        std::shared_ptr<arrow::Table> global_stats =
            SyncAndReduceGlobalStats(std::move(local_stats));
        this->min_max_stats.push_back(global_stats);
    }

    // Clear build chunks to free memory
    this->_build_chunks.clear();
}

void CudaHashJoin::BuildConsumeBatch(std::shared_ptr<cudf::table> build_chunk,
                                     cuda_event_wrapper event) {
    // TODO: remove unused columns before shuffling to save network bandwidth
    // and GPU memory.
    // Store the incoming build chunk for later finalization
    this->build_shuffle_manager.shuffle_table(build_chunk,
                                              this->build_key_indices, event);
    std::vector<std::unique_ptr<cudf::table>> shuffled_build_chunks =
        build_shuffle_manager.progress();
    for (auto& chunk : shuffled_build_chunks) {
        this->_build_chunks.emplace_back(std::move(chunk));
    }
}

std::unique_ptr<cudf::table> CudaHashJoin::ProbeProcessBatch(
    const std::shared_ptr<cudf::table>& probe_chunk, cuda_event_wrapper event,
    rmm::cuda_stream_view& stream) {
    // TODO: remove unused columns before shuffling to save network bandwidth
    // and GPU memory Send local data to appropriate ranks
    probe_shuffle_manager.shuffle_table(probe_chunk, this->probe_key_indices,
                                        event);

    //    Receive data destined for this rank
    std::vector<std::unique_ptr<cudf::table>> shuffled_probe_chunks =
        probe_shuffle_manager.progress();
    if (shuffled_probe_chunks.empty() || this->_join_handle == nullptr ||
        this->probe_shuffle_manager.get_mpi_comm() == MPI_COMM_NULL) {
        return empty_table_from_arrow_schema(
            this->output_schema->ToArrowSchema());
    }

    // Concatenate all incoming chunks into one contiguous table and join
    // against it
    std::vector<cudf::table_view> probe_views;
    probe_views.reserve(shuffled_probe_chunks.size());
    for (const auto& chunk : shuffled_probe_chunks) {
        probe_views.push_back(chunk->view());
    }
    std::unique_ptr<cudf::table> coalesced_probe =
        cudf::concatenate(probe_views, stream);

    auto [probe_indices, build_indices] = _join_handle->inner_join(
        coalesced_probe->select(this->probe_key_indices), {}, stream);

    if (probe_indices->size() == 0) {
        return empty_table_from_arrow_schema(
            this->output_schema->ToArrowSchema());
    }

    // Create views for the columns we want to keep
    cudf::table_view probe_kept_view = coalesced_probe->select(
        this->probe_kept_cols.begin(), this->probe_kept_cols.end());
    cudf::table_view build_kept_view = _build_table->select(
        this->build_kept_cols.begin(), this->build_kept_cols.end());

    // Create column views of the indices from the indices buffers
    cudf::column_view probe_idx_view(cudf::data_type{cudf::type_id::INT32},
                                     probe_indices->size(),
                                     probe_indices->data(), nullptr, 0);

    cudf::column_view build_idx_view(cudf::data_type{cudf::type_id::INT32},
                                     build_indices->size(),
                                     build_indices->data(), nullptr, 0);

    // Materialize the selected rows
    auto gathered_probe =
        cudf::gather(probe_kept_view, probe_idx_view,
                     cudf::out_of_bounds_policy::DONT_CHECK, stream);
    auto gathered_build =
        cudf::gather(build_kept_view, build_idx_view,
                     cudf::out_of_bounds_policy::DONT_CHECK, stream);

    // Assemble Final Result
    std::vector<std::unique_ptr<cudf::column>> final_columns;

    for (auto& col : gathered_probe->release()) {
        final_columns.push_back(std::move(col));
    }
    for (auto& col : gathered_build->release()) {
        final_columns.push_back(std::move(col));
    }

    return std::make_unique<cudf::table>(std::move(final_columns));
}
