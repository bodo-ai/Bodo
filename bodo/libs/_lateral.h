// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include "_array_utils.h"
#include "_bodo_common.h"

/**
 * @brief Computes the lateral flatten operation on a table on an array column.
 *
 * @param[in] in_table the table to be exploded. The first column is the one
 *            that gets exploded, all others have their rows replicated.
 * @param[in,out] n_rows pointer to store the number of rows in the final table.
 * @param[in] output_seq whether to output a column that associates each
 * exploded row with a specific row in the original table.
 * @param[in] output_key whether to output a column representing the keys in
 * JSON key-value pairs, or null if the column is an array.
 * @param[in] output_path whether to output a column representing the string
 * used to access the flattened element from the original array.
 * @param[in] output_index whether to output an index column based on the
 *            indices of values within each inner-array.
 * @param[in] output_value whether to include the exploded values from the
 *            exploded column in the output.
 * @param[in] output_this whether to include a copy of the original array along
 * with each exploded row.
 * @return the exploded table.
 */
std::unique_ptr<table_info> lateral_flatten_array(
    const std::unique_ptr<table_info> &in_table, int64_t *n_rows,
    bool output_seq, bool output_key, bool output_path, bool output_index,
    bool output_value, bool output_this, bool outer,
    bodo::IBufferPool *const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

/**
 * @brief Computes the lateral flatten operation on a table on a map array.
 *
 * @param[in] in_table the table to be exploded. The first column is the one
 *            that gets exploded, all others have their rows replicated.
 * @param[in,out] n_rows pointer to store the number of rows in the final table.
 * @param[in] output_seq whether to output a column that associates each
 * exploded row with a specific row in the original table.
 * @param[in] output_key whether to output a column representing the keys in
 * JSON key-value pairs, or null if the column is an array.
 * @param[in] output_path whether to output a column representing the string
 * used to access the flattened element from the original array.
 * @param[in] output_index whether to output an index column based on the
 *            indices of values within each inner-array.
 * @param[in] output_value whether to include the exploded values from the
 *            exploded column in the output.
 * @param[in] output_this whether to include a copy of the original array along
 * with each exploded row.
 * @return the exploded table.
 */
std::unique_ptr<table_info> lateral_flatten_map(
    const std::unique_ptr<table_info> &in_table, int64_t *n_rows,
    bool output_seq, bool output_key, bool output_path, bool output_index,
    bool output_value, bool output_this, bool outer,
    bodo::IBufferPool *const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

/**
 * @brief Computes the lateral flatten operation on a table on a struct column.
 *
 * @param[in] in_table the table to be exploded. The first column is the one
 *            that gets exploded, all others have their rows replicated.
 * @param[in,out] n_rows pointer to store the number of rows in the final table.
 * @param[in] output_seq whether to output a column that associates each
 * exploded row with a specific row in the original table.
 * @param[in] output_key whether to output a column representing the keys in
 * JSON key-value pairs, or null if the column is an array.
 * @param[in] output_path whether to output a column representing the string
 * used to access the flattened element from the original array.
 * @param[in] output_index whether to output an index column based on the
 *            indices of values within each inner-array.
 * @param[in] output_value whether to include the exploded values from the
 *            exploded column in the output.
 * @param[in] output_this whether to include a copy of the original array along
 * with each exploded row.
 * @return the exploded table.
 */
std::unique_ptr<table_info> lateral_flatten_struct(
    const std::unique_ptr<table_info> &in_table, int64_t *n_rows,
    bool output_seq, bool output_key, bool output_path, bool output_index,
    bool output_value, bool output_this, bool outer,
    bodo::IBufferPool *const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());
