// Copyright (C) 2024 Bodo Inc. All rights reserved.
#include "../_bodo_common.h"
#include "../_dict_builder.h"

/**
 * @breif Computes the outputs of a collection of window functions
 * via the window calculator infrastructure.
 *
 * @param[in] in_schema: the schema of the input data.
 * @param[in] in_chunks: the vector of table_info representing
 * the sorted table spread out across multiple in_chunks.
 * @param[in] partition_col_indices: the indices of columns
 * in in_chunks that correspond to partition columns.
 * @param[in] order_col_indices: the indices of columns
 * in in_chunks that correspond to order columns.
 * @param[in] keep_indices: the indices of columns in in_chunks
 * that should be kept in the final output.
 * @param[in] input_col_indices: a vector of vectors where
 * each inner vector contains the indices of columns in
 * in_chunks that are associated with that window function.
 * @param[in] window_funcs: the ftypes for the window functions
 * being calculated.
 * @param[in] builders: the dictionary builders used for in_chunks.
 * These should be used to ensure unification of columns.
 * @param[out] out_chunks: the vector of table_infos to push the
 * result rows into.
 * @param[in] partition_arr_type: the array type of all the partition
 * columns. Should be UNKNOWN if there are no partition arrays, or
 * multiple different types.
 * @param[in] order_arr_type: the array type of all the order by
 * columns. Should be UNKNOWN if there are no order arrays, or
 * multiple different types.
 * @param[in] is_parallel: is the computation happening in parallel.
 */
void compute_window_functions_via_calculators(
    std::shared_ptr<bodo::Schema> schema,
    std::vector<std::shared_ptr<table_info>> in_chunks,
    std::vector<int32_t> partition_col_indices,
    std::vector<int32_t> order_col_indices, std::vector<int32_t> keep_indices,
    const std::vector<std::vector<int32_t>> &input_col_indices,
    const std::vector<int32_t> &window_funcs,
    std::vector<std::shared_ptr<DictionaryBuilder>> builders,
    bodo_array_type::arr_type_enum partition_arr_type,
    bodo_array_type::arr_type_enum order_arr_type,
    std::vector<std::shared_ptr<table_info>> &out_chunks, bool is_parallel,
    bodo::IBufferPool *const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());
