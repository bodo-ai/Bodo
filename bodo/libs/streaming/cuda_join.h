#pragma once
#include <../../bodo/libs/_bodo_common.h>
#include <arrow/scalar.h>
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
    void build_hash_table(
        const std::vector<std::shared_ptr<cudf::table>>& build_chunks);

    std::vector<cudf::size_type> build_key_indices;
    std::vector<cudf::size_type> probe_key_indices;
    std::vector<int64_t> build_kept_cols;
    std::vector<int64_t> probe_kept_cols;

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
                 cudf::null_equality null_eq = cudf::null_equality::EQUAL)
        : build_key_indices(std::move(build_keys)),
          probe_key_indices(std::move(probe_keys)),
          build_kept_cols(std::move(build_kept_cols)),
          probe_kept_cols(std::move(probe_kept_cols)),
          build_table_schema(std::move(build_schema)),
          probe_table_schema(std::move(probe_schema)),
          null_equality(null_eq) {}
    CudaHashJoin() = default;
    void FinalizeBuild();
    void BuildConsumeBatch(std::shared_ptr<cudf::table> build_chunk);
    std::unique_ptr<cudf::table> ProbeProcessBatch(
        const std::shared_ptr<cudf::table>& probe_chunk);
    std::vector<std::shared_ptr<arrow::Table>> get_min_max_stats() {
        return min_max_stats;
    }
};
