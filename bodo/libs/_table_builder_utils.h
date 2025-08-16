#pragma once
#include "_bodo_common.h"
#include "_dict_builder.h"

/// Helper functions

/**
 * @brief Allocate an empty table with provided column types
 *
 * @param schema bodo::Schema of the table to create
 * bodo_array_type format)
 * @param pool IBufferPool to use for allocating the underlying data
 * buffers.
 * @param mm MemoryManager for the 'pool'.
 * @param dict builder dictionary buiders to use for populating dictionary typed
 * arrays
 * @return std::shared_ptr<table_info> allocated table
 */
std::shared_ptr<table_info> alloc_table(
    const std::shared_ptr<bodo::Schema>& schema,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager(),
    std::vector<std::shared_ptr<DictionaryBuilder>>* dict_builders = nullptr);

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

/**
 * @brief Helper for UnifyBuildTableDictionaryArrays and
 * UnifyProbeTableDictionaryArrays. Unifies dictionaries of input table with
 * dictionaries in dict_builders by appending its new dictionary values to
 * buffer's dictionaries and transposing input's indices.
 *
 * @param in_table input table
 * @param dict_builders Dictionary builders to unify with. The dict builders
 * will be appended with the new values from dictionaries in input_table.
 * @param n_keys number of key columns
 * @param only_transpose_existing_on_key_cols For key columns, whether or not to
 * only transpose the values that already exist in the dictionary-builder (and
 * set the rest as nulls) instead of full unification which would append any
 * values not already in the dictionary-builder to the dictionary-builder. This
 * is used in the Probe-Inner case in Join since we statically know that the
 * dictionary should not grow because if a value doesn't already exist, it will
 * get filtered out by the Join anyway.
 * @return std::shared_ptr<table_info> input table with dictionaries unified
 * with build table dictionaries.
 */
std::shared_ptr<table_info> unify_dictionary_arrays_helper(
    const std::shared_ptr<table_info>& in_table,
    std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders,
    uint64_t n_keys, bool only_transpose_existing_on_key_cols = false);
