#include "cuda_join.h"
#include <arrow/array/util.h>
#include <arrow/compute/api_aggregate.h>
#include <mpi.h>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/join/conditional_join.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/join/mixed_join.hpp>
#include <cudf/reduction.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/types.hpp>
#include <memory>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <stdexcept>
#include "../../pandas/physical/gpu_expression.h"
#include "../../pandas/physical/operator.h"
#include "../_utils.h"
#include "_util.h"
#include "duckdb/common/enum_util.hpp"
#include "duckdb/common/enums/join_type.hpp"

constexpr float FALSE_POSITIVE_RATE = 0.01;

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

CudaJoin::CudaJoin(std::shared_ptr<bodo::Schema> build_schema,
                   std::shared_ptr<bodo::Schema> probe_schema,
                   std::vector<int64_t> build_kept_cols,
                   std::vector<int64_t> probe_kept_cols,
                   std::shared_ptr<bodo::Schema> output_schema,
                   duckdb::JoinType join_type,
                   std::unique_ptr<CudfASTOwner> non_equi_expression,
                   bool is_broadcast)
    : output_schema(std::move(output_schema)),
      build_kept_cols(std::move(build_kept_cols)),
      probe_kept_cols(std::move(probe_kept_cols)),
      build_table_schema(std::move(build_schema)),
      probe_table_schema(std::move(probe_schema)),
      join_type(join_type),
      non_equi_expression(std::move(non_equi_expression)),
      is_broadcast_join(is_broadcast) {
    probe_shuffle_manager = std::make_shared<GpuShuffleManager>();

    if (is_broadcast_join) {
        build_broadcast_manager = std::make_shared<GpuTableBroadcastManager>();
        if (duckdb::PropagatesBuildSide(this->join_type)) {
            // This is the only case we need to sync build matches
            this->build_matches_synced = false;
        }
    } else {
        build_shuffle_manager = std::make_shared<GpuShuffleManager>();
    }
}

bool CudaJoin::BuildConsumeBatch(
    std::shared_ptr<cudf::table> build_chunk,
    std::shared_ptr<StreamAndEvent> input_stream_event, bool local_is_last) {
    if (is_broadcast_join) {
        this->build_broadcast_manager->broadcast_table(build_chunk,
                                                       input_stream_event);
        std::vector<std::shared_ptr<cudf::table>> received_build_chunks =
            build_broadcast_manager->progress(local_is_last);
        for (auto& chunk : received_build_chunks) {
            this->_build_chunks.emplace_back(std::move(chunk));
        }
        return this->build_broadcast_manager->sync_is_last(local_is_last);
    } else {
        // Use an empty vector for key indices if it's not a hash join
        // (will be overridden in CudaHashJoin)
        std::vector<cudf::size_type> build_key_indices;
        if (auto hash_join = dynamic_cast<CudaHashJoin*>(this)) {
            build_key_indices = hash_join->build_key_indices;
        }

        this->build_shuffle_manager->append_batch(
            build_chunk, build_key_indices, input_stream_event);
        std::vector<std::shared_ptr<cudf::table>> shuffled_build_chunks =
            build_shuffle_manager->progress(local_is_last);
        for (auto& chunk : shuffled_build_chunks) {
            this->_build_chunks.emplace_back(std::move(chunk));
        }
        return this->build_shuffle_manager->sync_is_last(local_is_last);
    }
}

std::unique_ptr<cudf::table> CudaJoin::produce_unmatched_build_rows(
    std::unique_ptr<cudf::table> table, bool global_is_last,
    rmm::cuda_stream_view stream) {
    if (!global_is_last || !duckdb::PropagatesBuildSide(this->join_type)) {
        return table;
    }

    const MPI_Comm comm = this->probe_shuffle_manager->get_mpi_comm();
    if (this->is_broadcast_join && !this->build_matches_synced) {
        if (this->sync_build_matches_req == MPI_REQUEST_NULL) {
            CHECK_MPI(
                MPI_Iallreduce(MPI_IN_PLACE, this->unmatched_build_rows.get(),
                               this->unmatched_build_rows->size(), MPI_UINT8_T,
                               MPI_BAND, comm, &this->sync_build_matches_req),
                "produce_unmatched_build_rows: MPI error on MPI_Iallreduce ");
        } else {
            int flag = 0;
            CHECK_MPI(MPI_Test(&this->sync_build_matches_req, &flag,
                               MPI_STATUS_IGNORE),
                      "produce_unmatched_build_rows: MPI error on MPI_Test ");
            if (flag) {
                this->build_matches_synced = true;
            } else {
                return table;
            }
        }
    }

    int rank;
    MPI_Comm_rank(comm, &rank);
    // If it's a broadcast join we only want to
    // produce the unmatched rows on one rank since
    // all ranks share a build table
    if (this->is_broadcast_join && rank > 0) {
        return table;
    }

    cudf::table_view build_kept_view = _build_table->select(
        this->build_kept_cols.begin(), this->build_kept_cols.end());
    // For right and outer joins, we need to output unmatched build rows
    // at the end. We can identify these using the matched_build_rows
    // boolean mask.
    std::unique_ptr<cudf::table> unmatched_build_build_side =
        cudf::apply_boolean_mask(build_kept_view,
                                 this->unmatched_build_rows->view(), stream);

    // Then we need to construct null columns for the probe side for
    // these unmatched build rows, and concatenate them with the
    // unmatched build rows to add to the final output
    std::vector<cudf::table_view> unmatched_build_cols;
    std::unique_ptr<cudf::table> unmatched_build_probe_side;
    if (this->join_type != duckdb::JoinType::RIGHT_ANTI) {
        std::vector<std::unique_ptr<cudf::column>> null_probe_columns;
        for (size_t i = 0; i < probe_kept_cols.size(); i++) {
            std::shared_ptr<arrow::Field> field =
                this->probe_table_schema->ToArrowSchema()->field(
                    this->probe_kept_cols[i]);
            std::shared_ptr<arrow::Scalar> arrow_scalar =
                arrow::MakeNullScalar(field->type());
            std::unique_ptr<cudf::scalar> cudf_scalar =
                arrow_scalar_to_cudf(arrow_scalar);
            null_probe_columns.push_back(cudf::make_column_from_scalar(
                *cudf_scalar, unmatched_build_build_side->num_rows()));
        }
        unmatched_build_probe_side =
            std::make_unique<cudf::table>(std::move(null_probe_columns));
        unmatched_build_cols.push_back(unmatched_build_probe_side->view());
    }
    if (this->join_type != duckdb::JoinType::ANTI) {
        unmatched_build_cols.push_back(unmatched_build_build_side->view());
    }

    // Zip up the two tables into a table view so we can concatenate it
    // with the main output
    cudf::table_view unmatched_build_output_view =
        cudf::table_view(unmatched_build_cols);

    // Concatenate with the main output
    std::vector<cudf::table_view> output_views = {table->view(),
                                                  unmatched_build_output_view};
    table = cudf::concatenate(output_views, stream);

    // Set unmatched_build_rows to nullptr so we only add the unmatched
    // rows once
    this->unmatched_build_rows.reset(nullptr);
    return table;
}

std::pair<std::unique_ptr<cudf::table>, bool> CudaJoin::get_empty_output_table(
    bool global_is_last, rmm::cuda_stream_view stream) {
    return {
        produce_unmatched_build_rows(
            empty_table_from_arrow_schema(this->output_schema->ToArrowSchema()),
            global_is_last, stream),
        global_is_last && this->build_matches_synced};
}

std::pair<std::unique_ptr<cudf::table>, bool> CudaJoin::materialize_and_output(
    cudf::table_view const& probe_kept_view,
    cudf::column_view const& probe_idx_view,
    cudf::table_view const& build_kept_view,
    cudf::column_view const& build_idx_view, bool global_is_last,
    rmm::cuda_stream_view stream) {
    std::vector<std::unique_ptr<cudf::column>> final_columns;

    // Materialize the selected rows
    cudf::out_of_bounds_policy oob_policy =
        this->join_type == duckdb::JoinType::INNER ||
                this->join_type == duckdb::JoinType::RIGHT ||
                this->join_type == duckdb::JoinType::RIGHT_ANTI
            ? cudf::out_of_bounds_policy::DONT_CHECK
            : cudf::out_of_bounds_policy::NULLIFY;

    auto gathered_probe =
        cudf::gather(probe_kept_view, probe_idx_view, oob_policy, stream);

    for (auto& col : gathered_probe->release()) {
        final_columns.push_back(std::move(col));
    }

    if (this->join_type != duckdb::JoinType::ANTI) {
        auto gathered_build =
            cudf::gather(build_kept_view, build_idx_view, oob_policy, stream);
        for (auto& col : gathered_build->release()) {
            final_columns.push_back(std::move(col));
        }
    }
    std::unique_ptr<cudf::table> output_table =
        std::make_unique<cudf::table>(std::move(final_columns));

    return {produce_unmatched_build_rows(std::move(output_table),
                                         global_is_last, stream),
            global_is_last && this->build_matches_synced};
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

    // 2. Create the hash_join object
    //    This triggers the kernel that builds the hash table on the GPU.
    //    We maintain ownership of _join_handle to reuse it for probing.
    cudf::table_view build_view = _build_table->view();
    cudf::table_view selected_build_view =
        build_view.select(this->build_key_indices);

    if (build_view.num_rows() != 0 && this->build_key_indices.size() > 0) {
        if (this->join_type == duckdb::JoinType::MARK ||
            this->join_type == duckdb::JoinType::ANTI) {
            this->_join_handle = std::make_unique<cudf::filtered_join>(
                selected_build_view, this->null_equality,
                /* default args otherwise the compiler can't figure out which
                   constructor to call */
                cudf::set_as_build_table::RIGHT, 0.5);
        } else {
            this->_join_handle = std::make_unique<cudf::hash_join>(
                selected_build_view, this->null_equality);
        }
    }

    uint64_t build_total_size = gather_blooms.allreduce(build_view.num_rows());
    // Generate local bloom filter.
    if (build_view.num_rows() != 0) {
        this->_build_bloom_filter = build_bloom_filter_from_table(
            build_view.select(this->build_key_indices), build_total_size,
            FALSE_POSITIVE_RATE, cudf::get_default_stream());
    } else {
        this->_build_bloom_filter = build_empty_bloom_filter(
            build_total_size, FALSE_POSITIVE_RATE, cudf::get_default_stream());
    }

    if (!is_broadcast_join) {
        // Get all GPU nodes' bloom filters.
        std::vector<std::unique_ptr<rmm::device_buffer>> all_blooms =
            gather_blooms.all_gather_device_buffers(
                this->_build_bloom_filter->bitset, cudf::get_default_stream());
        // AtomicOR them all together.
        for (auto& one_bloom : all_blooms) {
            if (one_bloom) {
                size_t one_bloom_size = one_bloom->size();
                if (one_bloom_size % 8 != 0) {
                    throw std::runtime_error(
                        "Received bloom filter that isn't a multiple of "
                        "64-bits");
                }
                mergeBloomBitset(this->_build_bloom_filter->bitset, *one_bloom,
                                 cudf::get_default_stream());
            }
        }
    }
}

void CudaHashJoin::runtime_filter(
    cudf::table_view const& probe_table,
    std::vector<cudf::size_type> const& probe_key_indices,
    std::unique_ptr<cudf::column>& prev_mask, rmm::cuda_stream_view stream) {
    if (_build_bloom_filter) {
        filter_table_with_bloom(probe_table, probe_key_indices,
                                *_build_bloom_filter, prev_mask, stream);
    }
}

void CudaHashJoin::FinalizeBuild() {
    if (is_gpu_rank()) {
        this->build_hash_table(this->_build_chunks);
    }

    std::shared_ptr<arrow::Schema> build_table_arrow_schema =
        this->build_table_schema->ToArrowSchema();

    for (const auto& col_idx : this->build_key_indices) {
        std::shared_ptr<arrow::Table> local_stats;
        if (is_gpu_rank() && this->_build_table->num_rows()) {
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
        if (is_broadcast_join) {
            this->min_max_stats.push_back(std::move(local_stats));
        } else {
            std::shared_ptr<arrow::Table> global_stats =
                SyncAndReduceGlobalStats(std::move(local_stats));
            this->min_max_stats.push_back(global_stats);
        }
    }

    // Clear build chunks to free memory
    this->_build_chunks.clear();

    if (duckdb::PropagatesBuildSide(this->join_type) && this->_build_table) {
        // For right and outer joins we need to track which build rows have been
        // matched so we can output unmatched build rows at the end
        this->unmatched_build_rows = cudf::make_column_from_scalar(
            cudf::numeric_scalar(true), this->_build_table->num_rows(),
            cudf::get_default_stream());
    }
}

std::pair<std::unique_ptr<cudf::table>, bool> CudaHashJoin::ProbeProcessBatch(
    const std::shared_ptr<cudf::table>& probe_chunk,
    std::shared_ptr<StreamAndEvent> input_stream_event,
    rmm::cuda_stream_view& stream, bool local_is_last) {
    bool global_is_last;
    std::shared_ptr<cudf::table> probe_to_select;

    if (is_broadcast_join) {
        // In broadcast join mode, we don't need to wait for other workers to
        // send probe data to us. If this is a right/outer join, we need to
        // synchronize which build rows have been globally matched before
        // terminating this operator.
        global_is_last =
            duckdb::PropagatesBuildSide(this->join_type)
                ? probe_shuffle_manager->sync_is_last(local_is_last)
                : local_is_last;

        if (!is_gpu_rank()) {
            return {nullptr, global_is_last};
        }

        if (probe_chunk->num_rows() == 0) {
            return get_empty_output_table(global_is_last, stream);
        }

        probe_to_select = probe_chunk;
    } else {
        // TODO: remove unused columns before shuffling to save network
        // bandwidth and GPU memory Send local data to appropriate ranks
        probe_shuffle_manager->append_batch(
            probe_chunk, this->probe_key_indices, input_stream_event);

        // Receive data destined for this rank
        std::vector<std::shared_ptr<cudf::table>> shuffled_probe_chunks =
            probe_shuffle_manager->progress(local_is_last);

        global_is_last = probe_shuffle_manager->sync_is_last(local_is_last);

        if (!is_gpu_rank()) {
            return {nullptr, global_is_last};
        }
        if (shuffled_probe_chunks.empty()) {
            return get_empty_output_table(global_is_last, stream);
        }

        // Concatenate all incoming chunks into one contiguous table and join
        // against it
        std::vector<cudf::table_view> probe_views;
        probe_views.reserve(shuffled_probe_chunks.size());
        for (const auto& chunk : shuffled_probe_chunks) {
            probe_views.push_back(chunk->view());
        }
        probe_to_select = cudf::concatenate(probe_views, stream);

        bool null_handle = std::visit(
            [](auto& handle) { return handle == nullptr; }, this->_join_handle);
        // ANTI joins with an empty build table (null join handle) should output
        // all probe rows, and for other join types we can just return early
        // since we know the probe rows can't match.
        if (probe_to_select->num_rows() == 0 ||
            (null_handle && this->join_type != duckdb::JoinType::ANTI)) {
            return get_empty_output_table(global_is_last, stream);
        }
    }

    cudf::table_view selected =
        probe_to_select->select(this->probe_key_indices);

    // Create views for the columns we want to keep
    cudf::table_view probe_kept_view = probe_to_select->select(
        this->probe_kept_cols.begin(), this->probe_kept_cols.end());
    cudf::table_view build_kept_view = _build_table->select(
        this->build_kept_cols.begin(), this->build_kept_cols.end());

    std::unique_ptr<rmm::device_uvector<cudf::size_type>> probe_indices,
        build_indices;

    cudf::join_kind cudf_join_kind;
    switch (this->join_type) {
        case duckdb::JoinType::RIGHT_ANTI:
        case duckdb::JoinType::RIGHT:
        case duckdb::JoinType::INNER: {
            auto& join_handle =
                std::get<std::unique_ptr<cudf::hash_join>>(this->_join_handle);
            std::tie(probe_indices, build_indices) =
                join_handle->inner_join(selected, {}, stream);
            cudf_join_kind = cudf::join_kind::INNER_JOIN;
        } break;
        // Use left join for outer because it will give us all probe rows,
        // and for the build rows we can use the matched_build_rows boolean
        // mask to determine which ones are unmatched and should be included
        // in the output. If we used cudf's full join it would output
        // unmatched build rows every batch
        case duckdb::JoinType::OUTER:
        case duckdb::JoinType::LEFT: {
            auto& join_handle =
                std::get<std::unique_ptr<cudf::hash_join>>(this->_join_handle);
            std::tie(probe_indices, build_indices) =
                join_handle->left_join(selected, {}, stream);
            cudf_join_kind = cudf::join_kind::LEFT_JOIN;
        } break;
        case duckdb::JoinType::ANTI: {
            bool null_handle =
                std::visit([](auto& handle) { return handle == nullptr; },
                           this->_join_handle);
            if (null_handle) {
                probe_indices =
                    std::make_unique<rmm::device_uvector<cudf::size_type>>(
                        make_uvector_iota(selected.num_rows(), stream));
            } else {
                auto& join_handle =
                    std::get<std::unique_ptr<cudf::filtered_join>>(
                        this->_join_handle);
                probe_indices = join_handle->anti_join(selected, stream);
            }
            build_indices =
                std::make_unique<rmm::device_uvector<cudf::size_type>>(0,
                                                                       stream);
            cudf_join_kind = cudf::join_kind::LEFT_ANTI_JOIN;
        } break;
        case duckdb::JoinType::MARK: {
            auto& join_handle = std::get<std::unique_ptr<cudf::filtered_join>>(
                this->_join_handle);
            probe_indices = join_handle->semi_join(selected, stream);
            build_indices =
                std::make_unique<rmm::device_uvector<cudf::size_type>>(0,
                                                                       stream);
            cudf_join_kind = cudf::join_kind::LEFT_SEMI_JOIN;
        } break;
        default: {
            throw std::runtime_error(
                "Unsupported join type " +
                duckdb::EnumUtil::ToString(this->join_type));
        }
    }

    if (this->non_equi_expression != nullptr) {
        std::tie(probe_indices, build_indices) = cudf::filter_join_indices(
            probe_to_select->view(), this->_build_table->view(), *probe_indices,
            *build_indices, this->non_equi_expression->get_root(),
            cudf_join_kind, stream);
    }

    if (this->join_type == duckdb::JoinType::MARK) {
        // Create the mark column (all false)
        auto mark_col = cudf::make_fixed_width_column(
            cudf::data_type{cudf::type_id::BOOL8}, probe_to_select->num_rows(),
            cudf::mask_state::ALL_VALID, stream);
        cudaMemsetAsync(mark_col->mutable_view().head<uint8_t>(), 0,
                        probe_to_select->num_rows() * sizeof(uint8_t),
                        stream.value());

        // Set matched indices to true
        cudf_set_bools_from_indices<true>(
            mark_col->mutable_view(),
            cudf::column_view(cudf::data_type{cudf::type_id::INT32},
                              probe_indices->size(), probe_indices->data(),
                              nullptr, 0),
            stream);

        // Prepare final columns: probe_kept_cols + mark_col
        std::vector<std::unique_ptr<cudf::column>> final_columns;
        for (auto const& i : this->probe_kept_cols) {
            final_columns.push_back(std::make_unique<cudf::column>(
                probe_to_select->get_column(i), stream));
        }
        final_columns.push_back(std::move(mark_col));

        std::unique_ptr<cudf::table> output_table =
            std::make_unique<cudf::table>(std::move(final_columns));
        return {std::move(output_table),
                global_is_last && this->build_matches_synced};
    }

    // Create column views of the indices from the indices buffers
    cudf::column_view probe_idx_view(cudf::data_type{cudf::type_id::INT32},
                                     probe_indices->size(),
                                     probe_indices->data(), nullptr, 0);

    cudf::column_view build_idx_view(cudf::data_type{cudf::type_id::INT32},
                                     build_indices->size(),
                                     build_indices->data(), nullptr, 0);

    // Update which build table indices we've matched if it's relevant
    if (duckdb::PropagatesBuildSide(this->join_type) &&
        // If this is nullptr we either don't have a build table on this rank
        // or we've already produced the unmatched output
        this->unmatched_build_rows && build_idx_view.size()) {
        cudf_set_bools_from_indices<false>(
            this->unmatched_build_rows->mutable_view(), build_idx_view, stream);
    }

    // Right anti joins only output unmatched build rows, so we can return early
    if (this->join_type == duckdb::JoinType::RIGHT_ANTI) {
        return get_empty_output_table(global_is_last, stream);
    }

    return materialize_and_output(probe_kept_view, probe_idx_view,
                                  build_kept_view, build_idx_view,
                                  global_is_last, stream);
}

CudaHashJoin::CudaHashJoin(std::vector<cudf::size_type> build_keys,
                           std::vector<cudf::size_type> probe_keys,
                           std::shared_ptr<bodo::Schema> build_schema,
                           std::shared_ptr<bodo::Schema> probe_schema,
                           std::vector<int64_t> build_kept_cols,
                           std::vector<int64_t> probe_kept_cols,
                           std::shared_ptr<bodo::Schema> output_schema,
                           duckdb::JoinType join_type,
                           std::unique_ptr<CudfASTOwner> non_equi_expression,
                           cudf::null_equality null_eq, bool is_broadcast)
    : CudaJoin(std::move(build_schema), std::move(probe_schema),
               std::move(build_kept_cols), std::move(probe_kept_cols),
               std::move(output_schema), join_type,
               std::move(non_equi_expression), is_broadcast),
      build_key_indices(std::move(build_keys)),
      probe_key_indices(std::move(probe_keys)),
      null_equality(null_eq) {}

CudaNonEquiJoin::CudaNonEquiJoin(
    std::shared_ptr<bodo::Schema> build_schema,
    std::shared_ptr<bodo::Schema> probe_schema,
    std::vector<int64_t> build_kept_cols, std::vector<int64_t> probe_kept_cols,
    std::shared_ptr<bodo::Schema> output_schema, duckdb::JoinType join_type,
    std::unique_ptr<CudfASTOwner> non_equi_expression, bool is_broadcast)
    : CudaJoin(std::move(build_schema), std::move(probe_schema),
               std::move(build_kept_cols), std::move(probe_kept_cols),
               std::move(output_schema), join_type,
               // Non-equi joins always broadcast the build table
               std::move(non_equi_expression), true) {}

void CudaNonEquiJoin::FinalizeBuild() {
    if (is_gpu_rank()) {
        std::vector<cudf::table_view> build_views;
        for (const auto& chunk : this->_build_chunks) {
            build_views.push_back(chunk->view());
        }
        this->_build_table = cudf::concatenate(build_views);
    }
    // Clear build chunks to free memory
    this->_build_chunks.clear();

    if (duckdb::PropagatesBuildSide(this->join_type) && this->_build_table) {
        // For right and outer joins we need to track which build rows have been
        // matched so we can output unmatched build rows at the end
        this->unmatched_build_rows = cudf::make_column_from_scalar(
            cudf::numeric_scalar(true), this->_build_table->num_rows(),
            cudf::get_default_stream());
    }
}

std::pair<std::unique_ptr<cudf::table>, bool>
CudaNonEquiJoin::ProbeProcessBatch(
    const std::shared_ptr<cudf::table>& probe_chunk,
    std::shared_ptr<StreamAndEvent> input_stream_event,
    rmm::cuda_stream_view& stream, bool local_is_last) {
    bool global_is_last;

    global_is_last = duckdb::PropagatesBuildSide(this->join_type)
                         ? probe_shuffle_manager->sync_is_last(local_is_last)
                         : local_is_last;

    if (!is_gpu_rank()) {
        return {nullptr, global_is_last};
    }

    if (probe_chunk->num_rows() == 0) {
        return get_empty_output_table(global_is_last, stream);
    }

    std::shared_ptr<cudf::table> probe_to_select = probe_chunk;

    // Create views for the columns we want to keep
    cudf::table_view probe_kept_view = probe_to_select->select(
        this->probe_kept_cols.begin(), this->probe_kept_cols.end());
    cudf::table_view build_kept_view = _build_table->select(
        this->build_kept_cols.begin(), this->build_kept_cols.end());

    std::unique_ptr<rmm::device_uvector<cudf::size_type>> probe_indices,
        build_indices;

    const cudf::ast::expression& root = this->non_equi_expression->get_root();

    switch (this->join_type) {
        case duckdb::JoinType::RIGHT_ANTI:
        case duckdb::JoinType::RIGHT:
        case duckdb::JoinType::INNER: {
            std::tie(probe_indices, build_indices) =
                cudf::conditional_inner_join(probe_to_select->view(),
                                             this->_build_table->view(), root,
                                             {}, stream);
        } break;
        case duckdb::JoinType::OUTER:
        case duckdb::JoinType::LEFT: {
            std::tie(probe_indices, build_indices) =
                cudf::conditional_left_join(probe_to_select->view(),
                                            this->_build_table->view(), root,
                                            {}, stream);
        } break;
        case duckdb::JoinType::ANTI: {
            probe_indices = cudf::conditional_left_anti_join(
                probe_to_select->view(), this->_build_table->view(), root, {},
                stream);
            build_indices =
                std::make_unique<rmm::device_uvector<cudf::size_type>>(0,
                                                                       stream);
        } break;
        case duckdb::JoinType::MARK: {
            probe_indices = cudf::conditional_left_semi_join(
                probe_to_select->view(), this->_build_table->view(), root, {},
                stream);
        } break;
        default: {
            throw std::runtime_error(
                "Unsupported join type " +
                duckdb::EnumUtil::ToString(this->join_type));
        }
    }

    if (this->join_type == duckdb::JoinType::MARK) {
        // Create the mark column (all false)
        auto mark_col = cudf::make_fixed_width_column(
            cudf::data_type{cudf::type_id::BOOL8}, probe_to_select->num_rows(),
            cudf::mask_state::ALL_VALID, stream);
        cudaMemsetAsync(mark_col->mutable_view().head<uint8_t>(), 0,
                        probe_to_select->num_rows() * sizeof(uint8_t),
                        stream.value());

        // Set matched indices to true
        cudf_set_bools_from_indices<true>(
            mark_col->mutable_view(),
            cudf::column_view(cudf::data_type{cudf::type_id::INT32},
                              probe_indices->size(), probe_indices->data(),
                              nullptr, 0),
            stream);

        // Prepare final columns: probe_kept_cols + mark_col
        std::vector<std::unique_ptr<cudf::column>> final_columns;
        for (auto const& i : this->probe_kept_cols) {
            final_columns.push_back(std::make_unique<cudf::column>(
                probe_to_select->get_column(i), stream));
        }
        final_columns.push_back(std::move(mark_col));

        std::unique_ptr<cudf::table> output_table =
            std::make_unique<cudf::table>(std::move(final_columns));
        return {std::move(output_table),
                global_is_last && this->build_matches_synced};
    }

    // Create column views of the indices from the indices buffers
    cudf::column_view probe_idx_view(cudf::data_type{cudf::type_id::INT32},
                                     probe_indices->size(),
                                     probe_indices->data(), nullptr, 0);

    cudf::column_view build_idx_view(cudf::data_type{cudf::type_id::INT32},
                                     build_indices->size(),
                                     build_indices->data(), nullptr, 0);

    // Update which build table indices we've matched if it's relevant
    if (duckdb::PropagatesBuildSide(this->join_type) &&
        this->unmatched_build_rows && build_idx_view.size()) {
        cudf_set_bools_from_indices<false>(
            this->unmatched_build_rows->mutable_view(), build_idx_view, stream);
    }

    // Only output unmatched build rows for right anti joins, so we can return
    // early
    if (this->join_type == duckdb::JoinType::RIGHT_ANTI) {
        return get_empty_output_table(global_is_last, stream);
    }

    return materialize_and_output(probe_kept_view, probe_idx_view,
                                  build_kept_view, build_idx_view,
                                  global_is_last, stream);
}
