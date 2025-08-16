// Function to convert Bodo arrays to Arrow arrays

#pragma once

#include "_bodo_common.h"

#if _MSC_VER >= 1900
#undef timezone
#endif

// if status of arrow::Result is not ok, form an err msg and raise a
// runtime_error with it. If it is ok, get value using ValueOrDie
// and assign it to lhs using std::move
#ifndef CHECK_ARROW_AND_ASSIGN
#undef CHECK_ARROW_AND_ASSIGN
#define CHECK_ARROW_AND_ASSIGN(res, msg, lhs)                  \
    if (!(res.status().ok())) {                                \
        std::string err_msg = std::string("Error in arrow ") + \
                              " write: " + msg + " " +         \
                              res.status().ToString();         \
        throw std::runtime_error(err_msg);                     \
    }                                                          \
    lhs = std::move(res).ValueOrDie();
#endif

/**
 * @brief Convert array_info to equivalent Arrow array.
 *
 * @return std::shared_ptr<arrow::Array> equivalent Array array
 */
std::shared_ptr<arrow::Array> to_arrow(const std::shared_ptr<array_info> info);

std::shared_ptr<arrow::Array> bodo_array_to_arrow(
    arrow::MemoryPool *pool, const std::shared_ptr<array_info> array,
    bool convert_timedelta_to_int64, const std::string &tz,
    arrow::TimeUnit::type &time_unit, bool downcast_time_ns_to_us,
    std::shared_ptr<::arrow::MemoryManager> mm);

/**
 * @brief convert Bodo table to Arrow table
 *
 * @param table input Bodo table
 * @return std::shared_ptr<arrow::Table> Arrow table
 */
std::shared_ptr<arrow::Table> bodo_table_to_arrow(
    std::shared_ptr<table_info> table);

/**
 * @brief convert Bodo table to Arrow table with the
 * provided names for the fields.
 *
 * @param table input Bodo table
 * @param field_names The field names to append.
 * @param schema_metadata Additional metadata to include in the arrow schema.
 * @param convert_timedelta_to_int64 : cast timedelta to int64.
 * @param tz Timezone to use for Datetime (/timestamp) arrays. Provide an empty
 * string ("") to not specify one. This is primarily required for
 * Iceberg/Snowflake, for which we specify "UTC".
 * @param time_unit Time-Unit (NANO / MICRO / MILLI / SECOND) to use for
 * Datetime (/timestamp) arrays. Bodo arrays store information in nanoseconds.
 * When this is not nanoseconds, the data is converted to the specified type
 * before being copied to the Arrow array. Note that in case it's not
 * nanoseconds, we make a copy of the integer array (array->data1()) since we
 * cannot modify the existing array, as it might be used elsewhere. This is
 * primarily required for Iceberg which requires data to be written in
 * microseconds.
 * @param downcast_time_ns_to_us (default False): Is time data required to be
 * written in microseconds? NOTE: this is needed for snowflake write operation.
 * See gen_snowflake_schema comments.
 * @param pool Memory pool to use for allocating memory.
 * @param mm Memory manager to use for allocating memory.

 * @return std::shared_ptr<arrow::Table> Arrow table
 */
std::shared_ptr<arrow::Table> bodo_table_to_arrow(
    std::shared_ptr<table_info> table, std::vector<std::string> field_names,
    const std::shared_ptr<const arrow::KeyValueMetadata> schema_metadata = {},
    bool convert_timedelta_to_int64 = false, std::string tz = "",
    arrow::TimeUnit::type time_unit = arrow::TimeUnit::NANO,
    bool downcast_time_ns_to_us = false,
    bodo::IBufferPool *const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

/**
 * @brief Convert Arrow array to Bodo array_info with zero-copy.
 * The output Bodo array holds references to the Arrow array's buffers and
 * releases them when deleted. Currently, only string, dict-encoded string,
 * and numeric arrays are zero-copied. For decimal arrays if the result
 * we get from arrow is not 16 byte aligned then the result isn't zero-copied.
 * This has occasionally happened with decimal arrays that are the result of
 * the Snowflake connector and can lead to undefined behavior in other kernels.
 *
 * @param arrow_arr Input Arrow array
 * @param src_pool Pointer to BufferPool used to allocate `buf` if allocation
 * came from Bodo or known Arrow source
 * @param array_id Identifier for an array to say two arrays are equivalent.
 * Currently only used for string arrays that function as dictionaries.
 * @param dicts_ref_arr Array with same type as output used for replacing output
 * array's dictionaries (used in RetrieveArray for nested arrays since array
 * builder doesn't set dictionaries)
 * @return std::shared_ptr<array_info> Output Bodo array
 */
std::shared_ptr<array_info> arrow_array_to_bodo(
    std::shared_ptr<arrow::Array> arrow_arr, bodo::IBufferPool *src_pool,
    int64_t array_id = -1, std::shared_ptr<array_info> dicts_ref_arr = nullptr);

/**
 * @brief Convert Arrow table to Bodo table_info with zero-copy as much as
 * possible.
 *
 * @param table input Arrow table
 * @param src_pool Pointer to BufferPool used for allocations
 * @return std::shared_ptr<table_info> Bodo output table
 */
std::shared_ptr<table_info> arrow_table_to_bodo(
    std::shared_ptr<arrow::Table> table, bodo::IBufferPool *src_pool);

/**
 * @brief Convert Arrow DataType to Bodo DataType, including nested types
 *
 * @param arrow_type input Arrow DataType
 * @return std::unique_ptr<bodo::DataType> equivalent Bodo DataType
 */
std::unique_ptr<bodo::DataType> arrow_type_to_bodo_data_type(
    const std::shared_ptr<arrow::DataType> arrow_type);

/**
 * @brief Convert Arrow RecordBatch to Bodo table_info with zero-copy as much as
 * possible. Uses arrow_array_to_bodo to zero-copy all Arrow Arrays in the
 * table. Does not support reusing string array references for
 * dictionary-encoding
 *
 * Note, we are not using Arrow tables because they are chunked as RecordBatches
 * under the hood
 * - If the intention is to have 1 table_info out, use TableBuilder.
 * - If each piece can be handled separately, use this function
 *
 * @param arrow_rb Input Arrow RecordBatch
 * @param length Number of rows in the output table.
 * @return std::shared_ptr<table_into> Output Bodo table
 */
std::shared_ptr<table_info> arrow_recordbatch_to_bodo(
    std::shared_ptr<arrow::RecordBatch> arrow_rb, int64_t length);

/**
 * @brief Returns a vector whose data is a bitmask indicating
 * which indices of a dictionary array were used at least once.
 * Assumes that UnifyDictionary has already been called.
 *
 * @param[in] column Arrow chunked array of dictionary data
 */
std::optional<std::shared_ptr<arrow::Buffer>> get_dictionary_hits(
    const std::shared_ptr<arrow::ChunkedArray> &column);
