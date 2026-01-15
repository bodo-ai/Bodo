#include "cuda_join.h"
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>



void CudaHashJoin::build_hash_table(const std::vector<std::unique_ptr<cudf::table>>& build_chunks) {
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
            build_view.select(this->build_key_indices),
            this->null_equality
        );

    }

void CudaHashJoin::FinalizeBuild() {
    this->build_hash_table(this->_build_chunks);
    // Clear build chunks to free memory
    this->_build_chunks.clear();
}

void CudaHashJoin::BuildConsumeBatch(std::unique_ptr<cudf::table> build_chunk) {
    // Store the incoming build chunk for later finalization
    _build_chunks.push_back(std::move(build_chunk));
}

std::unique_ptr<cudf::table> CudaHashJoin::ProbeProcessBatch(const cudf::table& probe_chunk) {
    if (!_join_handle) {
        throw std::runtime_error("Hash table not built. Call FinalizeBuild first.");
    }

    // Perform the join using the pre-built hash table
    cudf::table_view probe_view = probe_chunk.view();
    
     auto [probe_indices_ptr, build_indices_ptr] = _join_handle->inner_join(
        probe_view.select(this->probe_key_indices)
    );

    // Check for empty result to avoid errors
    if (probe_indices_ptr->size() == 0) {
        // Return empty table
    }

    // 2. Create column_views from the raw indices
    // We wrap the raw device memory in a column_view. 
    // NOTE: The underlying uvectors must outlive these views (they do here).
    cudf::column_view probe_idx_view(
        cudf::data_type{cudf::type_id::INT32}, // indices are always size_type (int32)
        probe_indices_ptr->size(),
        probe_indices_ptr->data(),
        nullptr, // no null mask
        0        // null count
    );
    
    cudf::column_view build_idx_view(
        cudf::data_type{cudf::type_id::INT32},
        build_indices_ptr->size(),
        build_indices_ptr->data(),
        nullptr,
        0
    );

    // 3. Gather the actual data
    // This creates new tables containing only the matching rows
    auto gathered_probe = cudf::gather(probe_chunk, probe_idx_view);
    auto gathered_build = cudf::gather(this->_build_table->view(), build_idx_view);
    // 4. Assemble the final result
    // We extract the columns from the gathered tables and combine them into one vector.
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
    std::unique_ptr<cudf::table> result_table = std::make_unique<cudf::table>(std::move(final_columns));

    return result_table;
}


