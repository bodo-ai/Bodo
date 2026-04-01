#pragma once
#include <arrow/scalar.h>
#include <mpi.h>
#include <cudf/column/column_factories.hpp>
#include "../_bodo_common.h"
#ifdef USE_CUDF
#include <cudf/join/filtered_join.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/table/table.hpp>
#include "../gpu_bloom_filter.h"
#include "../gpu_utils.h"
#include "duckdb/common/enums/join_type.hpp"

// Forward declaration to avoid import loop
class CudfASTOwner;

/**
 * @brief Base class for CUDA join operators.
 *
 * This class contains common functionality for both hash joins and non-equi
 * joins.
 */
struct CudaJoin {
    std::unique_ptr<cudf::table> _build_table;

    // Store build chunks until FinalizeBuild is called
    std::vector<std::shared_ptr<cudf::table>> _build_chunks;

    // The output schema of the join probe phase, which is needed for
    // constructing empty result tables when there are no matches
    std::shared_ptr<bodo::Schema> output_schema;

    // What columns to keep in the output
    std::vector<int64_t> build_kept_cols;
    std::vector<int64_t> probe_kept_cols;

    std::shared_ptr<bodo::Schema> build_table_schema;
    std::shared_ptr<bodo::Schema> probe_table_schema;

    duckdb::JoinType join_type;

    // Optional expression to evaluate
    std::unique_ptr<CudfASTOwner> non_equi_expression;

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

    bool is_broadcast_join;

    std::shared_ptr<GpuShuffleManager> build_shuffle_manager;
    std::shared_ptr<GpuShuffleManager> probe_shuffle_manager;
    GpuMpiManager gather_blooms;
    std::shared_ptr<GpuTableBroadcastManager> build_broadcast_manager;

    CudaJoin(std::shared_ptr<bodo::Schema> build_schema,
             std::shared_ptr<bodo::Schema> probe_schema,
             std::vector<int64_t> build_kept_cols,
             std::vector<int64_t> probe_kept_cols,
             std::shared_ptr<bodo::Schema> output_schema,
             duckdb::JoinType join_type,
             std::unique_ptr<CudfASTOwner> non_equi_expression,
             bool is_broadcast = false);

    virtual ~CudaJoin() = default;

    /**
     * @brief Finalize the build phase by constructing the join structures
     * and collecting statistics
     */
    virtual void FinalizeBuild() = 0;

    /**
     * @brief Process input tables to build side of join
     */
    virtual bool BuildConsumeBatch(
        std::shared_ptr<cudf::table> build_chunk,
        std::shared_ptr<StreamAndEvent> input_stream_event, bool local_is_last);

    /**
     * @brief Run join probe on the input batch
     * @param probe_chunk input batch to probe
     * @param input_stream_event stream and event associated with the input
     * batch
     * @param stream CUDA stream to execute on
     * @param local_is_last whether this is the last input batch on this rank
     * @return output batch of probe and global is last flag
     */
    virtual std::pair<std::unique_ptr<cudf::table>, bool> ProbeProcessBatch(
        const std::shared_ptr<cudf::table>& probe_chunk,
        std::shared_ptr<StreamAndEvent> input_stream_event,
        rmm::cuda_stream_view& stream, bool local_is_last) = 0;

    /**
     * @brief Add to the previous mask of rows.
     */
    virtual void runtime_filter(
        cudf::table_view const& probe_table,
        std::vector<cudf::size_type> const& probe_key_indices,
        std::unique_ptr<cudf::column>& prev_mask,
        rmm::cuda_stream_view stream) {}

    /**
     * @brief Get the min-max statistics for runtime join filters
     *
     * @return std::vector<std::shared_ptr<arrow::Table>> vector of min-max
     * stats one per build key column. The first column is "min" and the second
     * column is "max".
     */
    virtual std::vector<std::shared_ptr<arrow::Table>> get_min_max_stats() {
        return {};
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
     * @brief Signal this rank cannot accept new input since shuffle buffer is
     * full.
     */
    bool shuffle_buffer_full() {
        if (is_broadcast_join) {
            // There is no shuffling in broadcast join
            return false;
        } else {
            return probe_shuffle_manager->BuffersFull();
        }
    }

    bool hasComm() {
        if (is_broadcast_join) {
            return build_broadcast_manager->get_mpi_comm() != MPI_COMM_NULL;
        } else {
            return build_shuffle_manager->get_mpi_comm() != MPI_COMM_NULL;
        }
    }

    /**
     * @brief Appends unmatched build-side rows to the output on the final batch
     * of a join that propagates the build side (e.g. RIGHT, OUTER, or
     * RIGHT_ANTI).
     *
     * On the last global probe batch, gathers all build rows that were never
     * matched (tracked via @c unmatched_build_rows), pairs them with
     * null-filled probe-side columns, and concatenates them onto @p table.
     * Resets @c unmatched_build_rows to @c nullptr after emission to prevent
     * duplicate output.
     *
     * Returns @p table unmodified if @p global_is_last is @c false, the join
     * type does not propagate the build side, or @c unmatched_build_rows is
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

   protected:
    /**
     * @brief Return an empty output table for this join operator,
     * potentially including unmatched build side rows.
     */
    std::pair<std::unique_ptr<cudf::table>, bool> get_empty_output_table(
        bool global_is_last, rmm::cuda_stream_view stream);

    /**
     * @brief Materialize the output table using the provided indices and
     * views, and append unmatched build side rows if necessary.
     */
    std::pair<std::unique_ptr<cudf::table>, bool> materialize_and_output(
        cudf::table_view const& probe_kept_view,
        cudf::column_view const& probe_idx_view,
        cudf::table_view const& build_kept_view,
        cudf::column_view const& build_idx_view, bool global_is_last,
        rmm::cuda_stream_view stream);
};

struct CudaHashJoin : public CudaJoin {
    std::shared_ptr<CudfBloomFilter> _build_bloom_filter;

    // The hash map object (opaque handle to the GPU hash table)
    std::variant<std::unique_ptr<cudf::hash_join>,
                 std::unique_ptr<cudf::filtered_join>>
        _join_handle;

    /**
     * @brief Build the hash table from the accumulated build chunks
     * @param build_chunks Vector of build table chunks
     */
    void build_hash_table(
        const std::vector<std::shared_ptr<cudf::table>>& build_chunks);

    // What input columns to join on
    std::vector<cudf::size_type> build_key_indices;
    std::vector<cudf::size_type> probe_key_indices;

    // Stats for runtime join filter
    std::vector<std::shared_ptr<arrow::Table>> min_max_stats;

    cudf::null_equality null_equality = cudf::null_equality::EQUAL;

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

    CudaHashJoin() = delete;

    void FinalizeBuild() override;

    std::pair<std::unique_ptr<cudf::table>, bool> ProbeProcessBatch(
        const std::shared_ptr<cudf::table>& probe_chunk,
        std::shared_ptr<StreamAndEvent> input_stream_event,
        rmm::cuda_stream_view& stream, bool local_is_last) override;

    void runtime_filter(cudf::table_view const& probe_table,
                        std::vector<cudf::size_type> const& probe_key_indices,
                        std::unique_ptr<cudf::column>& prev_mask,
                        rmm::cuda_stream_view stream) override;

    std::vector<std::shared_ptr<arrow::Table>> get_min_max_stats() override {
        return min_max_stats;
    }
};

/**
 * @brief CUDA join operator for pure non-equi joins.
 */
struct CudaNonEquiJoin : public CudaJoin {
   public:
    CudaNonEquiJoin(std::shared_ptr<bodo::Schema> build_schema,
                    std::shared_ptr<bodo::Schema> probe_schema,
                    std::vector<int64_t> build_kept_cols,
                    std::vector<int64_t> probe_kept_cols,
                    std::shared_ptr<bodo::Schema> output_schema,
                    duckdb::JoinType join_type,
                    std::unique_ptr<CudfASTOwner> non_equi_expression,
                    bool is_broadcast = false);

    void FinalizeBuild() override;

    std::pair<std::unique_ptr<cudf::table>, bool> ProbeProcessBatch(
        const std::shared_ptr<cudf::table>& probe_chunk,
        std::shared_ptr<StreamAndEvent> input_stream_event,
        rmm::cuda_stream_view& stream, bool local_is_last) override;
};

#else
struct CudaJoin {};
struct CudaHashJoin : public CudaJoin {};
struct CudaNonEquiJoin : public CudaJoin {};
#endif
