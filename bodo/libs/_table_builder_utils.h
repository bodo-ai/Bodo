#pragma once
#include "_bodo_common.h"

/// Helper functions

/**
 * @brief Allocate an empty table with provided column types
 *
 * @param schema bodo::Schema of the table to create
 * bodo_array_type format)
 * @param pool IBufferPool to use for allocating the underlying data
 * buffers.
 * @param mm MemoryManager for the 'pool'.
 * @return std::shared_ptr<table_info> allocated table
 */
std::shared_ptr<table_info> alloc_table(
    const std::shared_ptr<bodo::Schema>& schema,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

/**
 * @brief Allocate an empty table with the same schema
 * (arr_types and dtypes) as 'table'.
 *
 * @param table Reference table
 * @param reuse_dictionaries Whether we should reuse the
 * dictionaries from the input table in the new table.
 * @return std::shared_ptr<table_info> Allocated table
 */
std::shared_ptr<table_info> alloc_table_like(
    const std::shared_ptr<table_info>& table,
    const bool reuse_dictionaries = true,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());
