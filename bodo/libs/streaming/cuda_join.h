#pragma once
#include <arrow/scalar.h>
#include <mpi.h>
#include <cudf/column/column_factories.hpp>
#include "../_bodo_common.h"
#ifdef USE_CUDF
#include <cudf/join/hash_join.hpp>
#include <cudf/table/table.hpp>
#include "../gpu_bloom_filter.h"
#include "../gpu_utils.h"
#include "duckdb/common/enums/join_type.hpp"

// Forward declaration to avoid import loop
class CudfASTOwner;

struct CudaHashJoin {
    std::unique_ptr<cudf::table> _build_table;

    // Store build chunks until FinalizeBuild is called
    std::vector<std::shared_ptr<cudf::table>> _build_chunks;
    std::shared_ptr<CudfBloomFilter> _build_bloom_filter;

    // The hash map object (opaque handle to the GPU hash table)
    std::unique_ptr<cudf::hash_join> _join_handle;

    // The output schema of the join probe phase, which is needed for
    // constructing empty result tables when there are no matches
    std::shared_ptr<bodo::Schema> output_schema;

    /**
     * @brief Build the hash table from the accumulated build chunks
     * @param build_chunks Vector of build table chunks
     */
    void build_hash_table(
        const std::vector<std::shared_ptr<cudf::table>>& build_chunks);

    // What input columns to join on
    std::vector<cudf::size_type> build_key_indices;
    std::vector<cudf::size_type> probe_key_indices;
    // What columns to keep in the output
    std::vector<int64_t> build_kept_cols;
    std::vector<int64_t> probe_kept_cols;

    // Stats for runtime join filter
    std::vector<std::shared_ptr<arrow::Table>> min_max_stats;

    std::shared_ptr<bodo::Schema> build_table_schema;
    std::shared_ptr<bodo::Schema> probe_table_schema;

    duckdb::JoinType join_type;

    // Optional expression to evaluate
    std::unique_ptr<CudfASTOwner> non_equi_expression;

    cudf::null_equality null_equality = cudf::null_equality::EQUAL;

    std::unique_ptr<cudf::column> unmatched_build_rows =
        nullptr;  // Used for right/outer joins to track which
                  // build rows have been matched
    // For broadcast joins on RIGHT/OUTER joins we need to sync the build table
    // matches globally to only produce unmatched build rows once.
    MPI_Request sync_build_matches_req = MPI_REQUEST_NULL;
    // Flag to determine if unmatched_build_rows can be used to produce output
    // rows, if false this means this is a broadcast join and we need to wait
    // for sync_build_matches_req to complete before proceeding.
    bool build_matches_synced = true;

    /**
     * @brief Appends unmatched build-side rows to the output on the final batch
     * of a RIGHT or FULL OUTER join.
     *
     * On the last global probe batch, gathers all build rows that were never
     * matched (tracked via @c unmatched_build_rows), pairs them with
     * null-filled probe-side columns, and concatenates them onto @p table.
     * Resets @c unmatched_build_rows to @c nullptr after emission to prevent
     * duplicate output.
     *
     * Returns @p table unmodified if @p global_is_last is @c false, the join
     * type is not RIGHT or OUTER, or @c unmatched_build_rows is
     * @c nullptr.
     *
     * @param table Accumulated join output for this batch; unmatched
     * rows are appended.
     * @param global_is_last True if all ranks have finished producing probe
     * data.
     * @param stream CUDA stream on which device operations are
     * enqueued.
     *
     * @return The input @p table with unmatched build rows appended, or
     * unmodified if the emission conditions are not met.
     */
    std::unique_ptr<cudf::table> produce_unmatched_build_rows(
        std::unique_ptr<cudf::table> table, bool global_is_last,
        rmm::cuda_stream_view stream);
    bool is_broadcast_join;

    std::shared_ptr<GpuShuffleManager> build_shuffle_manager;
    std::shared_ptr<GpuShuffleManager> probe_shuffle_manager;
    GpuMpiManager gather_blooms;
    std::shared_ptr<GpuTableBroadcastManager> build_broadcast_manager;

    bool hasComm() {
        if (is_broadcast_join) {
            return build_broadcast_manager->get_mpi_comm() != MPI_COMM_NULL;
        } else {
            return build_shuffle_manager->get_mpi_comm() != MPI_COMM_NULL;
        }
    }

   public:
    CudaHashJoin(std::vector<cudf::size_type> build_keys,
                 std::vector<cudf::size_type> probe_keys,
                 std::shared_ptr<bodo::Schema> build_schema,
                 std::shared_ptr<bodo::Schema> probe_schema,
                 std::vector<int64_t> build_kept_cols,
                 std::vector<int64_t> probe_kept_cols,
                 std::shared_ptr<bodo::Schema> output_schema,
                 duckdb::JoinType join_type,
                 std::unique_ptr<CudfASTOwner> non_equi_expression,
                 cudf::null_equality null_eq = cudf::null_equality::EQUAL,
                 bool is_broadcast = false);

    CudaHashJoin() = default;

    /**
     * @brief Finalize the build phase by constructing the hash table
     * and collecting statistics
     */
    void FinalizeBuild();

    /**
     * @brief Process input tables to build side of join
     */
    bool BuildConsumeBatch(std::shared_ptr<cudf::table> build_chunk,
                           std::shared_ptr<StreamAndEvent> input_stream_event,
                           bool local_is_last);

    /**
     * @brief Run join probe on the input batch
     * @param probe_chunk input batch to probe
     * @param input_stream_event stream and event associated with the input
     * batch
     * @param stream CUDA stream to execute on
     * @param local_is_last whether this is the last input batch on this rank
     * @return output batch of probe and global is last flag
     */
    std::pair<std::unique_ptr<cudf::table>, bool> ProbeProcessBatch(
        const std::shared_ptr<cudf::table>& probe_chunk,
        std::shared_ptr<StreamAndEvent> input_stream_event,
        rmm::cuda_stream_view& stream, bool local_is_last);

    /**
     * @brief Add to the previous mask of rows.
     */
    void runtime_filter(cudf::table_view const& probe_table,
                        std::vector<cudf::size_type> const& probe_key_indices,
                        std::unique_ptr<cudf::column>& prev_mask,
                        rmm::cuda_stream_view stream);

    /**
     * @brief Get the min-max statistics for runtime join filters
     *
     * @return std::vector<std::shared_ptr<arrow::Table>> vector of min-max
     * stats one per build key column. The first column is "min" and the second
     * column is "max".
     */
    std::vector<std::shared_ptr<arrow::Table>> get_min_max_stats() {
        return min_max_stats;
    }

    /**
     * @brief Signal this rank has completed its portion of the join.
     */
    bool is_build_complete() {
        if (is_broadcast_join) {
            return build_broadcast_manager->BuffersFull();
        } else {
            return build_shuffle_manager->BuffersFull();
        }
    }

    /**
     * @brief Signal this rank has completed its portion of the join.
     */
    bool is_probe_complete() {
        if (is_broadcast_join) {
            return true;
        } else {
            return probe_shuffle_manager->BuffersFull();
        }
    }
};
#else
struct CudaHashJoin {};
#endif
