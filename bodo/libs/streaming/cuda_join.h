#pragma once
#include <arrow/scalar.h>
#include "../_bodo_common.h"
#include "../gpu_utils.h"
#ifdef USE_CUDF
#include <cudf/join/hash_join.hpp>
#include <cudf/table/table.hpp>

struct CudaHashJoin {
   private:
    // Storage for the finalized build table
    std::unique_ptr<cudf::table> _build_table;

    // Store build chunks until FinalizeBuild is called
    std::vector<std::shared_ptr<cudf::table>> _build_chunks;

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

    cudf::null_equality null_equality = cudf::null_equality::EQUAL;

   public:
    CudaHashJoin(std::vector<cudf::size_type> build_keys,
                 std::vector<cudf::size_type> probe_keys,
                 std::shared_ptr<bodo::Schema> build_schema,
                 std::shared_ptr<bodo::Schema> probe_schema,
                 std::vector<int64_t> build_kept_cols,
                 std::vector<int64_t> probe_kept_cols,
                 std::shared_ptr<bodo::Schema> output_schema,
                 cudf::null_equality null_eq = cudf::null_equality::EQUAL)
        : output_schema(std::move(output_schema)),
          build_key_indices(std::move(build_keys)),
          probe_key_indices(std::move(probe_keys)),
          build_kept_cols(std::move(build_kept_cols)),
          probe_kept_cols(std::move(probe_kept_cols)),
          build_table_schema(std::move(build_schema)),
          probe_table_schema(std::move(probe_schema)),
          null_equality(null_eq) {}
    CudaHashJoin() = default;
    /**
     * @brief Finalize the build phase by constructing the hash table
     * and collecting statistics
     */
    void FinalizeBuild();
    /**
     * @brief Process input tables to build side of join
     */
    void BuildConsumeBatch(std::shared_ptr<cudf::table> build_chunk,
                           cuda_event_wrapper event);
    /**
     * @brief Run join probe on the input batch
     * @param probe_chunk input batch to probe
     * @return output batch of probe
     */
    std::unique_ptr<cudf::table> ProbeProcessBatch(
        const std::shared_ptr<cudf::table>& probe_chunk,
        cuda_event_wrapper event);

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

    // Public so PhysicalGPUJoin can access to determine if there are pending
    // shuffles
    GpuShuffleManager build_shuffle_manager;
    GpuShuffleManager probe_shuffle_manager;
};
#else
struct CudaHashJoin {};
#endif
