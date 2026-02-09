#include "cuda_join.h"
#include <arrow/array/util.h>
#include <arrow/compute/api_aggregate.h>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/reduction.hpp>
#include "../../pandas/physical/operator.h"
#include "../_utils.h"

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
    // 1. Concatenate all build chunks into one contiguous table
    //    This is necessary because cudf::hash_join expects a single table_view
    std::vector<cudf::table_view> build_views;
    for (const auto& chunk : build_chunks) {
        build_views.push_back(chunk->view());
    }
    this->_build_table = cudf::concatenate(build_views);

    // 2. Create the hash_join object
    //    This triggers the kernel that builds the hash table on the GPU.
    //    We maintain ownership of _join_handle to reuse it for probing.
    cudf::table_view build_view = _build_table->view();

    this->_join_handle = std::make_unique<cudf::hash_join>(
        build_view.select(this->build_key_indices), this->null_equality);
}

void CudaHashJoin::FinalizeBuild() {
    this->build_hash_table(this->_build_chunks);

    std::shared_ptr<arrow::Schema> build_table_arrow_schema =
        this->build_table_schema->ToArrowSchema();
    for (const auto& col_idx : this->build_key_indices) {
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
        std::shared_ptr<arrow::Table> local_stats =
            convertGPUToArrow(stats_gpu_data);
        std::shared_ptr<arrow::Table> global_stats =
            SyncAndReduceGlobalStats(local_stats);
        this->min_max_stats.push_back(global_stats);
    }

    // Clear build chunks to free memory
    this->_build_chunks.clear();
}

void CudaHashJoin::BuildConsumeBatch(std::shared_ptr<cudf::table> build_chunk) {
    // Store the incoming build chunk for later finalization
    _build_chunks.push_back(std::move(build_chunk));
}

std::unique_ptr<cudf::table> CudaHashJoin::ProbeProcessBatch(
    const std::shared_ptr<cudf::table>& probe_chunk) {
    if (!_join_handle) {
        throw std::runtime_error(
            "Hash table not built. Call FinalizeBuild first.");
    }

    // Perform the join using the pre-built hash table
    cudf::table_view probe_view = probe_chunk->view();

    auto [probe_indices_ptr, build_indices_ptr] =
        _join_handle->inner_join(probe_view.select(this->probe_key_indices));

    cudf::table_view selected_probe_view = probe_chunk->select(
        this->probe_kept_cols.begin(), this->probe_kept_cols.end());
    cudf::table_view selected_build_view = _build_table->select(
        this->build_kept_cols.begin(), this->build_kept_cols.end());

    // Check for empty result to avoid errors
    if (probe_indices_ptr->size() == 0) {
        std::vector<std::unique_ptr<cudf::column>> final_columns;
        for (auto& col : selected_probe_view) {
            std::unique_ptr<cudf::column> empty_col = cudf::empty_like(col);
            final_columns.push_back(std::move(empty_col));
        }

        // Move columns from build side
        for (auto& col : selected_build_view) {
            std::unique_ptr<cudf::column> empty_col = cudf::empty_like(col);
            final_columns.push_back(std::move(empty_col));
        }

        // Return empty table
        std::unique_ptr<cudf::table> empty_out =
            std::make_unique<cudf::table>(std::move(final_columns));
        return empty_out;
    }

    // 2. Create column_views from the raw indices
    // We wrap the raw device memory in a column_view.
    // NOTE: The underlying uvectors must outlive these views (they do here).
    cudf::column_view probe_idx_view(
        cudf::data_type{
            cudf::type_id::INT32},  // indices are always size_type (int32)
        probe_indices_ptr->size(), probe_indices_ptr->data(),
        nullptr,  // no null mask
        0         // null count
    );

    cudf::column_view build_idx_view(cudf::data_type{cudf::type_id::INT32},
                                     build_indices_ptr->size(),
                                     build_indices_ptr->data(), nullptr, 0);

    // Gather the actual data
    // This creates new tables containing only the matching rows
    std::unique_ptr<cudf::table> gathered_probe =
        cudf::gather(selected_probe_view, probe_idx_view);
    std::unique_ptr<cudf::table> gathered_build =
        cudf::gather(selected_build_view, build_idx_view);

    // Assemble the final result
    // We extract the columns from the gathered tables and combine them into one
    // vector.
    std::vector<std::unique_ptr<cudf::column>> final_columns;

    // Move columns from probe side
    for (auto& col : gathered_probe->release()) {
        final_columns.push_back(std::move(col));
    }

    // Move columns from build side
    for (auto& col : gathered_build->release()) {
        final_columns.push_back(std::move(col));
    }

    // Construct the final joined table
    std::unique_ptr<cudf::table> result_table =
        std::make_unique<cudf::table>(std::move(final_columns));

    return result_table;
}
