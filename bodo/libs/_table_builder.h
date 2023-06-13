#pragma once
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_bodo_to_arrow.h"
#include "_dict_builder.h"

/**
 * @brief Wrapper around array_info to turn it into build buffer.
 * It allows appending elements while also providing random access, which is
 * necessary when used with a hash table. See
 * https://bodo.atlassian.net/wiki/spaces/B/pages/1351974913/Implementation+Notes
 *
 */
struct ArrayBuildBuffer {
    // internal array with data values
    std::shared_ptr<array_info> data_array;
    // Current number of elements in the buffer
    int64_t size;
    // Total capacity for data elements (including current elements,
    // capacity>=size should always be true)
    int64_t capacity;

    // Shared dictionary builder.
    std::shared_ptr<DictionaryBuilder> dict_builder = nullptr;
    // dictionary indices buffer for appending dictionary indices (only for
    // dictionary-encoded string arrays)
    std::shared_ptr<ArrayBuildBuffer> dict_indices;

    /**
     * @brief Construct a new ArrayBuildBuffer for the provided data array.
     *
     * @param _data_array Data array that we will be appending to. This is
     * expected to be an empty array.
     * @param dict_builder If this is a dictionary encoded string array,
     * a DictBuilder must be provided that will be used as the dictionary.
     * The dictionary of the data_array (_data_array->child_arrays[0]) must
     * be the dictionary in dict_builder (_dict_builder->dict_buff->data_array).
     */
    ArrayBuildBuffer(
        std::shared_ptr<array_info> _data_array,
        std::shared_ptr<DictionaryBuilder> _dict_builder = nullptr);

    /**
     * @brief Append a new data element to the buffer, assuming
     * there is already enough space reserved (with ReserveArray).
     *
     * @param in_arr input table with the new data element
     * @param row_ind index of data in input
     */
    void AppendRow(const std::shared_ptr<array_info>& in_arr, int64_t row_ind);

    /**
     * @brief Reserve enough space to potentially append all contents of input
     * array to buffer. NOTE: This requires reserving space for variable-sized
     * elements like strings and nested arrays.
     *
     * @param in_arr input array used for finding new buffer sizes to reserve
     */
    void ReserveArray(const std::shared_ptr<array_info>& in_arr);

    /**
     * @brief Clear the buffers, i.e. set size to 0.
     * Capacity is not changed and memory is not released.
     * For DICT arrays, the dictionary state is also reset.
     * In particular, the reset to point to the dictionary of the original
     * dictionary-builder which was provided during creation and the
     * dictionary related flags are reset.
     */
    void Reset();
};

/**
 * @brief Wrapper around table_info to turn it into build buffer.
 * It allows appending rows while also providing random access, which is
 * necessary when used with a hash table. See
 * https://bodo.atlassian.net/wiki/spaces/B/pages/1351974913/Implementation+Notes
 *
 */
struct TableBuildBuffer {
    // internal data table with values
    std::shared_ptr<table_info> data_table;
    // buffer wrappers around arrays of data table
    std::vector<ArrayBuildBuffer> array_buffers;

    // Only used for temporary objects. In particular,
    // in HashJoinState constructor, we cannot initialize
    // the shuffle buffers in the initialization list since
    // we need to build the dict_builders first. So we need
    // to provide this default constructor so that it
    // is initialized to an empty buffer by default and then
    // we can create and replace it with the actual TableBuildBuffer
    // later in the constructor.
    TableBuildBuffer() = default;

    /**
     * @brief Constructor for a TableBuildBuffer.
     *
     * @param arr_c_types Data Types for the columns.
     * @param arr_array_types Array Types for the columns.
     * @param dict_builders DictBuilders for the columns.
     * Element corresponding to a column must be provided in the
     * DICT array case and should be nullptr otherwise.
     */
    TableBuildBuffer(
        const std::vector<int8_t>& arr_c_types,
        const std::vector<int8_t>& arr_array_types,
        const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders);

    /**
     * @brief Append a row of data to the buffer, assuming
     * there is already enough space reserved (with ReserveTable).
     *
     * @param in_table input table with the new row
     * @param row_ind index of new row in input table
     */
    void AppendRow(const std::shared_ptr<table_info>& in_table,
                   int64_t row_ind);

    /**
     * @brief Reserve enough space to potentially append all contents of input
     * table to buffer. NOTE: This requires reserving space for variable-sized
     * elements like strings and nested arrays.
     *
     * @param in_table input table used for finding new buffer sizes to reserve
     */
    void ReserveTable(const std::shared_ptr<table_info>& in_table);

    /**
     * @brief Clear the buffers, i.e. set size to 0.
     * Capacity is not changed and memory is not released.
     * For DICT arrays, the dictionaries are reset to
     * use the original dictionary builders provided during
     * creation.
     */
    void Reset();
};

/// Helper functions

/**
 * @brief allocate an empty table with provided column types
 *
 * @param arr_c_types vector of ints for column dtypes (in Bodo_CTypes format)
 * @param arr_array_types vector of ints for colmun array types (in
 * bodo_array_type format)
 * @return std::shared_ptr<table_info> allocated table
 */
std::shared_ptr<table_info> alloc_table(
    const std::vector<int8_t>& arr_c_types,
    const std::vector<int8_t>& arr_array_types);

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
    const bool reuse_dictionaries = true);
