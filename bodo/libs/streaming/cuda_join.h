#pragma once
#include <cudf/join/hash_join.hpp>
#include <cudf/table/table.hpp>


struct CudaHashJoin {

private:
    // Storage for the finalized build table
    std::unique_ptr<cudf::table> _build_table; 

    // Store build chunks until FinalizeBuild is called
    std::vector<std::unique_ptr<cudf::table>> _build_chunks;
    
    // The hash map object (opaque handle to the GPU hash table)
    std::unique_ptr<cudf::hash_join> _join_handle;
    void build_hash_table(const std::vector<std::unique_ptr<cudf::table>>& build_chunks);
    

    std::vector<cudf::size_type> build_key_indices;
    std::vector<cudf::size_type> probe_key_indices;

    cudf::null_equality null_equality = cudf::null_equality::EQUAL;

public:
    CudaHashJoin(
        std::vector<cudf::size_type> build_keys,
        std::vector<cudf::size_type> probe_keys,
        cudf::null_equality null_eq = cudf::null_equality::EQUAL)
        : build_key_indices(std::move(build_keys)),
          probe_key_indices(std::move(probe_keys)),
          null_equality(null_eq) {}
    CudaHashJoin()= default;
    void FinalizeBuild();
    void BuildConsumeBatch(std::unique_ptr<cudf::table> build_chunk);
    std::unique_ptr<cudf::table> ProbeProcessBatch(const cudf::table& probe_chunk);
};
