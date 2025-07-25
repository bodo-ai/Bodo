#pragma once
#include <deque>

#include "_array_build_buffer.h"
#include "_array_utils.h"
#include "_dict_builder.h"
#include "_query_profile_collector.h"

// Pre-allocate 32 bytes per string for now.
// Keep in sync with value in
// test_stream_join.py::test_long_strings_chunked_table_builder
#define CHUNKED_TABLE_DEFAULT_STRING_PREALLOCATION 32LL
// Keep in sync with value in
// test_stream_join.py::test_long_strings_chunked_table_builder
#define DEFAULT_MAX_RESIZE_COUNT_FOR_VARIABLE_SIZE_DTYPES 2

#ifndef CASE_APPEND_ROWS_DTYPE
#define CASE_APPEND_ROWS_DTYPE(ARRAY_TYPE, IN_ARRAY_TYPE, DTYPE)  \
    case Bodo_CTypes::DTYPE: {                                    \
        this->UnsafeAppendRows<ARRAY_TYPE, IN_ARRAY_TYPE, DTYPE>( \
            in_arr, idxs, idx_start, idx_length);                 \
    } break;
#endif

#ifndef CASE_APPEND_ROWS
#define CASE_APPEND_ROWS(ARRAY_TYPE, IN_ARRAY_TYPE)                            \
    if (this->data_array->arr_type == ARRAY_TYPE &&                            \
        in_arr->arr_type == IN_ARRAY_TYPE) {                                   \
        switch (this->data_array->dtype) {                                     \
            CASE_APPEND_ROWS_DTYPE(ARRAY_TYPE, IN_ARRAY_TYPE,                  \
                                   Bodo_CTypes::INT8)                          \
            CASE_APPEND_ROWS_DTYPE(ARRAY_TYPE, IN_ARRAY_TYPE,                  \
                                   Bodo_CTypes::UINT8)                         \
            CASE_APPEND_ROWS_DTYPE(ARRAY_TYPE, IN_ARRAY_TYPE,                  \
                                   Bodo_CTypes::INT32)                         \
            CASE_APPEND_ROWS_DTYPE(ARRAY_TYPE, IN_ARRAY_TYPE,                  \
                                   Bodo_CTypes::UINT32)                        \
            CASE_APPEND_ROWS_DTYPE(ARRAY_TYPE, IN_ARRAY_TYPE,                  \
                                   Bodo_CTypes::INT64)                         \
            CASE_APPEND_ROWS_DTYPE(ARRAY_TYPE, IN_ARRAY_TYPE,                  \
                                   Bodo_CTypes::FLOAT32)                       \
            CASE_APPEND_ROWS_DTYPE(ARRAY_TYPE, IN_ARRAY_TYPE,                  \
                                   Bodo_CTypes::FLOAT64)                       \
            CASE_APPEND_ROWS_DTYPE(ARRAY_TYPE, IN_ARRAY_TYPE,                  \
                                   Bodo_CTypes::UINT64)                        \
            CASE_APPEND_ROWS_DTYPE(ARRAY_TYPE, IN_ARRAY_TYPE,                  \
                                   Bodo_CTypes::INT16)                         \
            CASE_APPEND_ROWS_DTYPE(ARRAY_TYPE, IN_ARRAY_TYPE,                  \
                                   Bodo_CTypes::UINT16)                        \
            CASE_APPEND_ROWS_DTYPE(ARRAY_TYPE, IN_ARRAY_TYPE,                  \
                                   Bodo_CTypes::STRING)                        \
            CASE_APPEND_ROWS_DTYPE(ARRAY_TYPE, IN_ARRAY_TYPE,                  \
                                   Bodo_CTypes::_BOOL)                         \
            CASE_APPEND_ROWS_DTYPE(ARRAY_TYPE, IN_ARRAY_TYPE,                  \
                                   Bodo_CTypes::DECIMAL)                       \
            CASE_APPEND_ROWS_DTYPE(ARRAY_TYPE, IN_ARRAY_TYPE,                  \
                                   Bodo_CTypes::DATE)                          \
            CASE_APPEND_ROWS_DTYPE(ARRAY_TYPE, IN_ARRAY_TYPE,                  \
                                   Bodo_CTypes::TIME)                          \
            CASE_APPEND_ROWS_DTYPE(ARRAY_TYPE, IN_ARRAY_TYPE,                  \
                                   Bodo_CTypes::DATETIME)                      \
            CASE_APPEND_ROWS_DTYPE(ARRAY_TYPE, IN_ARRAY_TYPE,                  \
                                   Bodo_CTypes::TIMEDELTA)                     \
            CASE_APPEND_ROWS_DTYPE(ARRAY_TYPE, IN_ARRAY_TYPE,                  \
                                   Bodo_CTypes::INT128)                        \
            CASE_APPEND_ROWS_DTYPE(ARRAY_TYPE, IN_ARRAY_TYPE,                  \
                                   Bodo_CTypes::BINARY)                        \
            default:                                                           \
                throw std::runtime_error(                                      \
                    "ChunkedTableArrayBuilder::UnsafeAppendRows: data type " + \
                    GetDtype_as_string(this->data_array->dtype) +              \
                    " not supported!");                                        \
        }                                                                      \
        return;                                                                \
    }
#endif

/**
 * @brief Array Builder for Chunked Table Builder.
 * Tracks size and capacity of the buffers.
 * For fixed size data types, the required capacity is
 * reserved up front.
 * For strings, we allocate certain amount up front,
 * and then resize as needed/allowed.
 * See design doc here: https://bodo.atlassian.net/l/cp/mzidHW9G.
 *
 */
struct ChunkedTableArrayBuilder {
    // Internal array with data values
    std::shared_ptr<array_info> data_array;

    // Child array builders
    // TODO[BSE-1258]: change to chunked buffer to allow pin/unpin each chunk
    // separately
    std::vector<ArrayBuildBuffer> child_array_builders;

    // Shared dictionary builder
    std::shared_ptr<DictionaryBuilder> dict_builder;
    // Dictionary indices buffer for appending dictionary indices (only for
    // dictionary-encoded string arrays)
    std::shared_ptr<ChunkedTableArrayBuilder> dict_indices;

    // Current number of elements in the buffers.
    const uint64_t& size;
    // Maximum number of rows this array is allowed to have.
    const size_t capacity;

    /* Only applicable to variable size data types like strings */

    // Number of times this array's buffers have been resized.
    size_t resize_count = 0;
    // Maximum number of times that this array's buffers can be
    // resized.
    const size_t max_resize_count = 2;

    // XXX In the future, we can generally keep track of the amount
    // of memory that has been allocated, etc. and use that to determine
    // when to stop building a chunk.

    /**
     * @brief Construct a new Chunked Table Array Builder object
     *
     * @param _data_array Underlying array_info into whose buffers we will
     * insert the data.
     * @param _dict_builder If this is a dictionary encoded string array,
     * a DictBuilder must be provided that will be used as the dictionary.
     * The dictionary of the data_array (_data_array->child_arrays[0]) must
     * be the dictionary in dict_builder (_dict_builder->dict_buff->data_array).
     * @param chunk_size Maximum number of rows this chunk is allowed to have.
     * @param max_resize_count Maximum number of times the buffers of this array
     * can be resized (grow by x2). This is only applicable to variable size
     * data types like strings. Even in those cases, only the variable sized
     * buffers (e.g. data1 for strings) will be resized.
     */
    ChunkedTableArrayBuilder(
        std::shared_ptr<array_info> _data_array,
        std::shared_ptr<DictionaryBuilder> _dict_builder = nullptr,
        size_t chunk_size = 8192, size_t max_resize_count = 2);

    /**
     * @brief Get the total number of bytes that all buffers
     * of this array have allocated (so far).
     *
     * @return size_t
     */
    size_t GetTotalBytes();

    /**
     * @brief Check if this array can resize (grow by x2).
     *
     * @return true If it has variable-size data (e.g. string) and resize_count
     * < max_resize_count.
     * @return false Otherwise.
     */
    bool CanResize();

    template <typename IndexT>
    void UnsafeAppendRows(const std::shared_ptr<array_info>& in_arr,
                          const std::span<const IndexT> idxs, size_t idx_start,
                          size_t idx_length) {
        CASE_APPEND_ROWS(bodo_array_type::NUMPY, bodo_array_type::NUMPY)
        CASE_APPEND_ROWS(bodo_array_type::NULLABLE_INT_BOOL,
                         bodo_array_type::NUMPY)
        CASE_APPEND_ROWS(bodo_array_type::NULLABLE_INT_BOOL,
                         bodo_array_type::NULLABLE_INT_BOOL)
        if (this->data_array->arr_type == bodo_array_type::TIMESTAMPTZ &&
            in_arr->arr_type == bodo_array_type::TIMESTAMPTZ) {
            switch (this->data_array->dtype) {
                CASE_APPEND_ROWS_DTYPE(bodo_array_type::TIMESTAMPTZ,
                                       bodo_array_type::TIMESTAMPTZ,
                                       Bodo_CTypes::TIMESTAMPTZ)
                default:
                    throw std::runtime_error(
                        "ChunkedTableArrayBuilder::UnsafeAppendRows: data "
                        "type " +
                        GetDtype_as_string(this->data_array->dtype) +
                        " not supported!");
            }
        } else if (this->data_array->arr_type == bodo_array_type::STRING &&
                   in_arr->arr_type == bodo_array_type::STRING) {
            switch (this->data_array->dtype) {
                CASE_APPEND_ROWS_DTYPE(bodo_array_type::STRING,
                                       bodo_array_type::STRING,
                                       Bodo_CTypes::STRING)
                CASE_APPEND_ROWS_DTYPE(bodo_array_type::STRING,
                                       bodo_array_type::STRING,
                                       Bodo_CTypes::BINARY)
                default:
                    throw std::runtime_error(
                        "ChunkedTableArrayBuilder::UnsafeAppendRows: data "
                        "type " +
                        GetDtype_as_string(this->data_array->dtype) +
                        " not supported!");
            }
        } else if (this->data_array->arr_type == bodo_array_type::DICT &&
                   in_arr->arr_type == bodo_array_type::DICT) {
            switch (this->data_array->dtype) {
                CASE_APPEND_ROWS_DTYPE(bodo_array_type::DICT,
                                       bodo_array_type::DICT,
                                       Bodo_CTypes::STRING)
                default:
                    throw std::runtime_error(
                        "ChunkedTableArrayBuilder::UnsafeAppendRows: data "
                        "type " +
                        GetDtype_as_string(this->data_array->dtype) +
                        " not supported!");
            }
        } else if (this->data_array->arr_type == bodo_array_type::ARRAY_ITEM &&
                   in_arr->arr_type == bodo_array_type::ARRAY_ITEM) {
            switch (this->data_array->dtype) {
                CASE_APPEND_ROWS_DTYPE(bodo_array_type::ARRAY_ITEM,
                                       bodo_array_type::ARRAY_ITEM,
                                       Bodo_CTypes::LIST)
                default:
                    throw std::runtime_error(
                        "ChunkedTableArrayBuilder::UnsafeAppendRows: data "
                        "type " +
                        GetDtype_as_string(this->data_array->dtype) +
                        " not supported!");
            }
        } else if (this->data_array->arr_type == bodo_array_type::MAP &&
                   in_arr->arr_type == bodo_array_type::MAP) {
            switch (this->data_array->dtype) {
                CASE_APPEND_ROWS_DTYPE(bodo_array_type::MAP,
                                       bodo_array_type::MAP, Bodo_CTypes::MAP)
                default:
                    throw std::runtime_error(
                        "ChunkedTableArrayBuilder::UnsafeAppendRows: data "
                        "type " +
                        GetDtype_as_string(this->data_array->dtype) +
                        " not supported!");
            }
        } else if (this->data_array->arr_type == bodo_array_type::STRUCT &&
                   in_arr->arr_type == bodo_array_type::STRUCT) {
            switch (this->data_array->dtype) {
                CASE_APPEND_ROWS_DTYPE(bodo_array_type::STRUCT,
                                       bodo_array_type::STRUCT,
                                       Bodo_CTypes::STRUCT)
                default:
                    throw std::runtime_error(
                        "ChunkedTableArrayBuilder::UnsafeAppendRows: data "
                        "type " +
                        GetDtype_as_string(this->data_array->dtype) +
                        " not supported!");
            }
        } else {
            throw std::runtime_error(
                "ChunkedTableArrayBuilder::UnsafeAppendRows: array type " +
                GetArrType_as_string(this->data_array->arr_type) +
                " not supported!");
        }
    }

    /**
     * @brief Append the rows from in_arr found via
     * idxs[idx_start: idx_start + idx_length] into this array.
     * This assumes that enough space is available in the buffers
     * without need to resize. This is useful internally as well
     * as externally when the caller is tracking available space
     * themselves.
     *
     * This is the implementation where both arrays are nullable arrays
     * and not booleans.
     *
     * @param in_arr The array from which we are inserting.
     * @param idxs The indices giving which rows in in_arr we want to insert.
     * @param idx_start The start location in idxs from which to insert.
     * @param idx_length The number of rows we will insert.
     */
    template <bodo_array_type::arr_type_enum out_arr_type,
              bodo_array_type::arr_type_enum in_arr_type,
              Bodo_CTypes::CTypeEnum dtype, typename IndexT>
        requires(out_arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
                 in_arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
                 dtype != Bodo_CTypes::_BOOL)
    void UnsafeAppendRows(const std::shared_ptr<array_info>& in_arr,
                          const std::span<const IndexT> idxs, size_t idx_start,
                          size_t idx_length) {
        using T = typename dtype_to_type<dtype>::type;
        T* out_data = (T*)this->data_array->data1<out_arr_type>();
        T* in_data = (T*)in_arr->data1<in_arr_type>();
        uint8_t* out_bitmask =
            (uint8_t*)this->data_array->null_bitmask<out_arr_type>();
        const uint8_t* in_bitmask =
            (uint8_t*)in_arr->null_bitmask<in_arr_type>();

        for (size_t i = 0; i < idx_length; i++) {
            int64_t row_idx = idxs[i + idx_start];
            T new_data = row_idx < 0 ? static_cast<T>(0) : in_data[row_idx];
            out_data[this->size + i] = new_data;
        }
        for (size_t i = 0; i < idx_length; i++) {
            int64_t row_idx = idxs[i + idx_start];
            bool null_bit = (row_idx >= 0) && GetBit(in_bitmask, row_idx);
            SetBitTo(out_bitmask, this->size + i, null_bit);
        }
        this->data_array->length += idx_length;
    }

    /**
     * @brief Append the rows from in_arr found via
     * idxs[idx_start: idx_start + idx_length] into this array.
     * This assumes that enough space is available in the buffers
     * without need to resize. This is useful internally as well
     * as externally when the caller is tracking available space
     * themselves.
     *
     * This is the implementation where both arrays are TIMESTAMPTZ arrays.
     *
     * @param in_arr The array from which we are inserting.
     * @param idxs The indices giving which rows in in_arr we want to insert.
     * @param idx_start The start location in idxs from which to insert.
     * @param idx_length The number of rows we will insert.
     */
    template <bodo_array_type::arr_type_enum out_arr_type,
              bodo_array_type::arr_type_enum in_arr_type,
              Bodo_CTypes::CTypeEnum dtype, typename IndexT>
        requires(out_arr_type == bodo_array_type::TIMESTAMPTZ &&
                 in_arr_type == bodo_array_type::TIMESTAMPTZ &&
                 dtype == Bodo_CTypes::TIMESTAMPTZ)
    void UnsafeAppendRows(const std::shared_ptr<array_info>& in_arr,
                          const std::span<const IndexT> idxs, size_t idx_start,
                          size_t idx_length) {
        using T_utc = typename dtype_to_type<Bodo_CTypes::TIMESTAMPTZ>::type;
        using T_offset = typename dtype_to_type<Bodo_CTypes::INT16>::type;
        T_utc* out_data = (T_utc*)this->data_array->data1<out_arr_type>();
        T_utc* in_data = (T_utc*)in_arr->data1<in_arr_type>();
        T_offset* out_offsets =
            (T_offset*)this->data_array->data2<out_arr_type>();
        T_offset* in_offsets = (T_offset*)in_arr->data2<in_arr_type>();
        uint8_t* out_bitmask =
            (uint8_t*)this->data_array->null_bitmask<out_arr_type>();
        const uint8_t* in_bitmask =
            (uint8_t*)in_arr->null_bitmask<in_arr_type>();

        for (size_t i = 0; i < idx_length; i++) {
            int64_t row_idx = idxs[i + idx_start];
            T_utc new_data = row_idx < 0 ? 0 : in_data[row_idx];
            T_offset new_offset = row_idx < 0 ? 0 : in_offsets[row_idx];
            out_data[this->size + i] = new_data;
            out_offsets[this->size + i] = new_offset;
        }
        for (size_t i = 0; i < idx_length; i++) {
            int64_t row_idx = idxs[i + idx_start];
            bool null_bit = (row_idx >= 0) && GetBit(in_bitmask, row_idx);
            SetBitTo(out_bitmask, this->size + i, null_bit);
        }
        this->data_array->length += idx_length;
    }

    /**
     * @brief Append the rows from in_arr found via
     * idxs[idx_start: idx_start + idx_length] into this array.
     * This assumes that enough space is available in the buffers
     * without need to resize. This is useful internally as well
     * as externally when the caller is tracking available space
     * themselves.
     *
     * This is the implementation where both arrays are nullable
     * boolean arrays.
     *
     * @param in_arr The array from which we are inserting.
     * @param idxs The indices giving which rows in in_arr we want to insert.
     * @param idx_start The start location in idxs from which to insert.
     * @param idx_length The number of rows we will insert.
     */
    template <bodo_array_type::arr_type_enum out_arr_type,
              bodo_array_type::arr_type_enum in_arr_type,
              Bodo_CTypes::CTypeEnum dtype, typename IndexT>
        requires(out_arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
                 in_arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
                 dtype == Bodo_CTypes::_BOOL)
    void UnsafeAppendRows(const std::shared_ptr<array_info>& in_arr,
                          const std::span<const IndexT> idxs, size_t idx_start,
                          size_t idx_length) {
        uint8_t* out_data = (uint8_t*)this->data_array->data1<out_arr_type>();
        uint8_t* in_data = (uint8_t*)in_arr->data1<in_arr_type>();
        uint8_t* out_bitmask =
            (uint8_t*)this->data_array->null_bitmask<out_arr_type>();
        const uint8_t* in_bitmask =
            (uint8_t*)in_arr->null_bitmask<in_arr_type>();

        for (size_t i = 0; i < idx_length; i++) {
            int64_t row_idx = idxs[i + idx_start];
            bool bit = row_idx < 0 ? false : GetBit(in_data, row_idx);
            arrow::bit_util::SetBitTo(out_data, this->size + i, bit);
        }
        for (size_t i = 0; i < idx_length; i++) {
            int64_t row_idx = idxs[i + idx_start];
            bool null_bit = (row_idx >= 0) && GetBit(in_bitmask, row_idx);
            SetBitTo(out_bitmask, this->size + i, null_bit);
        }
        this->data_array->length += idx_length;
    }

    /**
     * @brief Append the rows from in_arr found via
     * idxs[idx_start: idx_start + idx_length] into this array.
     * This assumes that enough space is available in the buffers
     * without need to resize. This is useful internally as well
     * as externally when the caller is tracking available space
     * themselves.
     *
     * This is the implementation where a non-boolean NUMPY array is
     * upcast to a nullable array.
     *
     * @param in_arr The array from which we are inserting.
     * @param idxs The indices giving which rows in in_arr we want to insert.
     * @param idx_start The start location in idxs from which to insert.
     * @param idx_length The number of rows we will insert.
     */
    template <bodo_array_type::arr_type_enum out_arr_type,
              bodo_array_type::arr_type_enum in_arr_type,
              Bodo_CTypes::CTypeEnum dtype, typename IndexT>
        requires(out_arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
                 in_arr_type == bodo_array_type::NUMPY &&
                 dtype != Bodo_CTypes::_BOOL)
    void UnsafeAppendRows(const std::shared_ptr<array_info>& in_arr,
                          const std::span<const IndexT> idxs, size_t idx_start,
                          size_t idx_length) {
        using T = typename dtype_to_type<dtype>::type;
        T* out_data = (T*)this->data_array->data1<out_arr_type>();
        T* in_data = (T*)in_arr->data1<in_arr_type>();
        uint8_t* out_bitmask =
            (uint8_t*)this->data_array->null_bitmask<out_arr_type>();

        for (size_t i = 0; i < idx_length; i++) {
            int64_t row_idx = idxs[i + idx_start];
            T new_data = row_idx < 0 ? static_cast<T>(0) : in_data[row_idx];
            out_data[this->size + i] = new_data;
        }
        for (size_t i = 0; i < idx_length; i++) {
            int64_t row_idx = idxs[i + idx_start];
            bool null_bit = (row_idx >= 0);
            SetBitTo(out_bitmask, this->size + i, null_bit);
        }
        this->data_array->length += idx_length;
    }

    /**
     * @brief Append the rows from in_arr found via
     * idxs[idx_start: idx_start + idx_length] into this array.
     * This assumes that enough space is available in the buffers
     * without need to resize. This is useful internally as well
     * as externally when the caller is tracking available space
     * themselves.
     *
     * This is the implementation where a boolean NUMPY array is
     * upcast to a nullable array.
     *
     * @param in_arr The array from which we are inserting.
     * @param idxs The indices giving which rows in in_arr we want to insert.
     * @param idx_start The start location in idxs from which to insert.
     * @param idx_length The number of rows we will insert.
     */
    template <bodo_array_type::arr_type_enum out_arr_type,
              bodo_array_type::arr_type_enum in_arr_type,
              Bodo_CTypes::CTypeEnum dtype, typename IndexT>
        requires(out_arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
                 in_arr_type == bodo_array_type::NUMPY &&
                 dtype == Bodo_CTypes::_BOOL)
    void UnsafeAppendRows(const std::shared_ptr<array_info>& in_arr,
                          const std::span<const IndexT> idxs, size_t idx_start,
                          size_t idx_length) {
        uint8_t* out_data = (uint8_t*)this->data_array->data1<out_arr_type>();
        uint8_t* in_data = (uint8_t*)in_arr->data1<in_arr_type>();
        uint8_t* out_bitmask =
            (uint8_t*)this->data_array->null_bitmask<out_arr_type>();

        for (size_t i = 0; i < idx_length; i++) {
            int64_t row_idx = idxs[i + idx_start];
            bool bit = row_idx < 0 ? false : in_data[row_idx];
            arrow::bit_util::SetBitTo(out_data, this->size + i, bit);
        }
        for (size_t i = 0; i < idx_length; i++) {
            int64_t row_idx = idxs[i + idx_start];
            bool null_bit = (row_idx >= 0);
            SetBitTo(out_bitmask, this->size + i, null_bit);
        }
        this->data_array->length += idx_length;
    }

    /**
     * @brief Append the rows from in_arr found via
     * idxs[idx_start: idx_start + idx_length] into this array.
     * This assumes that enough space is available in the buffers
     * without need to resize. This is useful internally as well
     * as externally when the caller is tracking available space
     * themselves.
     *
     * This is the implementation where both arrays are numpy arrays
     * without SQL NULL sentinels.
     *
     * @param in_arr The array from which we are inserting.
     * @param idxs The indices giving which rows in in_arr we want to insert.
     * @param idx_start The start location in idxs from which to insert.
     * @param idx_length The number of rows we will insert.
     */
    template <bodo_array_type::arr_type_enum out_arr_type,
              bodo_array_type::arr_type_enum in_arr_type,
              Bodo_CTypes::CTypeEnum dtype, typename IndexT>
        requires(out_arr_type == bodo_array_type::NUMPY &&
                 in_arr_type == bodo_array_type::NUMPY &&
                 !SQLNASentinelDtype<dtype>)
    void UnsafeAppendRows(const std::shared_ptr<array_info>& in_arr,
                          const std::span<const IndexT> idxs, size_t idx_start,
                          size_t idx_length) {
        using T = typename dtype_to_type<dtype>::type;
        T* out_data = (T*)this->data_array->data1<out_arr_type>();
        T* in_data = (T*)in_arr->data1<in_arr_type>();

        for (size_t i = 0; i < idx_length; i++) {
            int64_t row_idx = idxs[i + idx_start];
            T new_data = row_idx < 0 ? static_cast<T>(0) : in_data[row_idx];
            out_data[this->size + i] = new_data;
        }
        this->data_array->length += idx_length;
    }

    /**
     * @brief Append the rows from in_arr found via
     * idxs[idx_start: idx_start + idx_length] into this array.
     * This assumes that enough space is available in the buffers
     * without need to resize. This is useful internally as well
     * as externally when the caller is tracking available space
     * themselves.
     *
     * This is the implementation where input is a nullable array, output is
     * numpy, dtype isn't bool and without SQL NULL sentinels.
     *
     * @param in_arr The array from which we are inserting.
     * @param idxs The indices giving which rows in in_arr we want to insert.
     * @param idx_start The start location in idxs from which to insert.
     * @param idx_length The number of rows we will insert.
     */
    template <bodo_array_type::arr_type_enum out_arr_type,
              bodo_array_type::arr_type_enum in_arr_type,
              Bodo_CTypes::CTypeEnum dtype, typename IndexT>
        requires(out_arr_type == bodo_array_type::NUMPY &&
                 in_arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
                 !SQLNASentinelDtype<dtype> && dtype != Bodo_CTypes::_BOOL)
    void UnsafeAppendRows(const std::shared_ptr<array_info>& in_arr,
                          const std::span<const IndexT> idxs, size_t idx_start,
                          size_t idx_length) {
        using T = typename dtype_to_type<dtype>::type;
        T* out_data = (T*)this->data_array->data1<out_arr_type>();
        T* in_data = (T*)in_arr->data1<in_arr_type>();
        uint8_t* in_bitmask = (uint8_t*)in_arr->null_bitmask<in_arr_type>();
        for (size_t i = 0; i < idx_length; i++) {
            if (!arrow::bit_util::GetBit(in_bitmask, idxs[i + idx_start])) {
                throw std::runtime_error(
                    "ChunkedTableArrayBuilder::UnsafeAppendRows: Cannot append "
                    "NULL value to non-nullable array");
            }
        }

        for (size_t i = 0; i < idx_length; i++) {
            int64_t row_idx = idxs[i + idx_start];
            T new_data = row_idx < 0 ? static_cast<T>(0) : in_data[row_idx];
            out_data[this->size + i] = new_data;
        }
        this->data_array->length += idx_length;
    }

    /**
     * @brief Append the rows from in_arr found via
     * idxs[idx_start: idx_start + idx_length] into this array.
     * This assumes that enough space is available in the buffers
     * without need to resize. This is useful internally as well
     * as externally when the caller is tracking available space
     * themselves.
     *
     * This is the implementation where input is a nullable array, output is
     * numpy, dtype is bool and without SQL NULL sentinels.
     *
     * @param in_arr The array from which we are inserting.
     * @param idxs The indices giving which rows in in_arr we want to insert.
     * @param idx_start The start location in idxs from which to insert.
     * @param idx_length The number of rows we will insert.
     */
    template <bodo_array_type::arr_type_enum out_arr_type,
              bodo_array_type::arr_type_enum in_arr_type,
              Bodo_CTypes::CTypeEnum dtype, typename IndexT>
        requires(out_arr_type == bodo_array_type::NUMPY &&
                 in_arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
                 !SQLNASentinelDtype<dtype> && dtype == Bodo_CTypes::_BOOL)
    void UnsafeAppendRows(const std::shared_ptr<array_info>& in_arr,
                          const std::span<const IndexT> idxs, size_t idx_start,
                          size_t idx_length) {
        uint8_t* out_data = (uint8_t*)this->data_array->data1<out_arr_type>();
        uint8_t* in_data = (uint8_t*)in_arr->data1<in_arr_type>();
        uint8_t* in_bitmask = (uint8_t*)in_arr->null_bitmask<in_arr_type>();
        for (size_t i = 0; i < idx_length; i++) {
            if (!arrow::bit_util::GetBit(in_bitmask, idxs[i + idx_start])) {
                throw std::runtime_error(
                    "ChunkedTableArrayBuilder::UnsafeAppendRows: Cannot append "
                    "NULL value to non-nullable array");
            }
        }

        for (size_t i = 0; i < idx_length; i++) {
            int64_t row_idx = idxs[i + idx_start];
            bool new_data =
                row_idx < 0 ? 0 : arrow::bit_util::GetBit(in_data, row_idx);
            out_data[this->size + i] = new_data;
        }
        this->data_array->length += idx_length;
    }

    /**
     * @brief Append the rows from in_arr found via
     * idxs[idx_start: idx_start + idx_length] into this array.
     * This assumes that enough space is available in the buffers
     * without need to resize. This is useful internally as well
     * as externally when the caller is tracking available space
     * themselves.
     *
     * This is the implementation where both arrays are numpy arrays
     * with SQL NULL sentinels.
     *
     * @param in_arr The array from which we are inserting.
     * @param idxs The indices giving which rows in in_arr we want to insert.
     * @param idx_start The start location in idxs from which to insert.
     * @param idx_length The number of rows we will insert.
     */
    template <bodo_array_type::arr_type_enum out_arr_type,
              bodo_array_type::arr_type_enum in_arr_type,
              Bodo_CTypes::CTypeEnum dtype, typename IndexT>
        requires(out_arr_type == bodo_array_type::NUMPY &&
                 in_arr_type == bodo_array_type::NUMPY &&
                 SQLNASentinelDtype<dtype>)
    void UnsafeAppendRows(const std::shared_ptr<array_info>& in_arr,
                          const std::span<const IndexT> idxs, size_t idx_start,
                          size_t idx_length) {
        // TODO: Remove when Timedelta can be nullable.
        using T = typename dtype_to_type<dtype>::type;
        T* out_data = (T*)this->data_array->data1<out_arr_type>();
        T* in_data = (T*)in_arr->data1<in_arr_type>();

        for (size_t i = 0; i < idx_length; i++) {
            int64_t row_idx = idxs[i + idx_start];
            // Timedelta Sentinel is std::numeric_limits<int64_t>::min()
            T new_data = row_idx < 0 ? std::numeric_limits<int64_t>::min()
                                     : in_data[row_idx];
            out_data[this->size + i] = new_data;
        }
        this->data_array->length += idx_length;
    }

    /**
     * @brief Append the rows from in_arr found via
     * idxs[idx_start: idx_start + idx_length] into this array.
     * This assumes that enough space is available in the buffers
     * without need to resize. This is useful internally as well
     * as externally when the caller is tracking available space
     * themselves.
     *
     * This is the implementation where input is a nullable array and output is
     * numpy with SQL NULL sentinels.
     *
     * @param in_arr The array from which we are inserting.
     * @param idxs The indices giving which rows in in_arr we want to insert.
     * @param idx_start The start location in idxs from which to insert.
     * @param idx_length The number of rows we will insert.
     */
    template <bodo_array_type::arr_type_enum out_arr_type,
              bodo_array_type::arr_type_enum in_arr_type,
              Bodo_CTypes::CTypeEnum dtype, typename IndexT>
        requires(out_arr_type == bodo_array_type::NUMPY &&
                 in_arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
                 SQLNASentinelDtype<dtype>)
    void UnsafeAppendRows(const std::shared_ptr<array_info>& in_arr,
                          const std::span<const IndexT> idxs, size_t idx_start,
                          size_t idx_length) {
        // TODO: Remove when Timedelta can be nullable.
        using T = typename dtype_to_type<dtype>::type;
        T* out_data = (T*)this->data_array->data1<out_arr_type>();
        T* in_data = (T*)in_arr->data1<in_arr_type>();
        uint8_t* in_bitmask = (uint8_t*)in_arr->null_bitmask<in_arr_type>();

        for (size_t i = 0; i < idx_length; i++) {
            int64_t row_idx = idxs[i + idx_start];
            // Timedelta Sentinel is std::numeric_limits<int64_t>::min()
            out_data[this->size + i] =
                ((row_idx < 0) ||
                 (!arrow::bit_util::GetBit(in_bitmask, row_idx)))
                    ? std::numeric_limits<int64_t>::min()
                    : in_data[row_idx];
        }
        this->data_array->length += idx_length;
    }

    /**
     * @brief Append the rows from in_arr found via
     * idxs[idx_start: idx_start + idx_length] into this array.
     * This assumes that enough space is available in the buffers
     * without need to resize. This is useful internally as well
     * as externally when the caller is tracking available space
     * themselves.
     *
     * This is the implementation where both arrays are strings.
     *
     * @param in_arr The array from which we are inserting.
     * @param idxs The indices giving which rows in in_arr we want to insert.
     * @param idx_start The start location in idxs from which to insert.
     * @param idx_length The number of rows we will insert.
     */
    template <bodo_array_type::arr_type_enum out_arr_type,
              bodo_array_type::arr_type_enum in_arr_type,
              Bodo_CTypes::CTypeEnum dtype, typename IndexT>
        requires(out_arr_type == bodo_array_type::STRING &&
                 in_arr_type == bodo_array_type::STRING)
    void UnsafeAppendRows(const std::shared_ptr<array_info>& in_arr,
                          const std::span<const IndexT> idxs, size_t idx_start,
                          size_t idx_length) {
        // Copy the offsets
        offset_t* curr_offsets =
            (offset_t*)this->data_array->data2<out_arr_type>() + this->size;
        offset_t* in_offsets = (offset_t*)in_arr->data2<in_arr_type>();
        for (size_t i = 0; i < idx_length; i++) {
            int64_t row_idx = idxs[i + idx_start];
            offset_t new_length =
                row_idx < 0 ? 0 : in_offsets[row_idx + 1] - in_offsets[row_idx];
            curr_offsets[i + 1] = curr_offsets[i] + new_length;
        }
        // Copy the data
        char* out_data = this->data_array->data1<out_arr_type>();
        char* in_data = in_arr->data1<in_arr_type>();
        for (size_t i = 0; i < idx_length; i++) {
            int64_t row_idx = idxs[i + idx_start];
            offset_t out_offset = curr_offsets[i];
            offset_t in_offset = row_idx < 0 ? 0 : in_offsets[row_idx];
            size_t copy_len =
                row_idx < 0 ? 0 : in_offsets[row_idx + 1] - in_offsets[row_idx];
            memcpy(out_data + out_offset, in_data + in_offset, copy_len);
        }
        // Copy the null bitmap
        uint8_t* out_bitmask =
            (uint8_t*)this->data_array->null_bitmask<out_arr_type>();
        const uint8_t* in_bitmask =
            (uint8_t*)in_arr->null_bitmask<in_arr_type>();
        for (size_t i = 0; i < idx_length; i++) {
            int64_t row_idx = idxs[i + idx_start];
            bool null_bit = (row_idx >= 0) && GetBit(in_bitmask, row_idx);
            SetBitTo(out_bitmask, this->size + i, null_bit);
        }
        this->data_array->length += idx_length;
    }

    /**
     * @brief Append the rows from in_arr found via
     * idxs[idx_start: idx_start + idx_length] into this array.
     * This assumes that enough space is available in the buffers
     * without need to resize. This is useful internally as well
     * as externally when the caller is tracking available space
     * themselves.
     *
     * This is the implementation where both arrays are dicts.
     *
     * @param in_arr The array from which we are inserting.
     * @param idxs The indices giving which rows in in_arr we want to insert.
     * @param idx_start The start location in idxs from which to insert.
     * @param idx_length The number of rows we will insert.
     */
    template <bodo_array_type::arr_type_enum out_arr_type,
              bodo_array_type::arr_type_enum in_arr_type,
              Bodo_CTypes::CTypeEnum dtype, typename IndexT>
        requires(out_arr_type == bodo_array_type::DICT &&
                 in_arr_type == bodo_array_type::DICT)
    void UnsafeAppendRows(const std::shared_ptr<array_info>& in_arr,
                          const std::span<const IndexT> idxs, size_t idx_start,
                          size_t idx_length) {
        if (!is_matching_dictionary(this->data_array->child_arrays[0],
                                    in_arr->child_arrays[0])) {
            throw std::runtime_error(
                "ChunkedTableArrayBuilder::UnsafeAppendRows: "
                "Dictionaries "
                "not unified!");
        }
        this->dict_indices->UnsafeAppendRows<bodo_array_type::NULLABLE_INT_BOOL,
                                             bodo_array_type::NULLABLE_INT_BOOL,
                                             Bodo_CTypes::INT32>(
            in_arr->child_arrays[1], idxs, idx_start, idx_length);
        this->data_array->length += idx_length;
    }

    /**
     * @brief UnsafeAppendRows implementation where both arrays are nested
     * arrays.
     *
     * @param in_arr The array from which we are inserting.
     * @param idxs The indices giving which rows in in_arr we want to insert.
     * @param idx_start The start location in idxs from which to insert.
     * @param idx_length The number of rows we will insert.
     */
    template <bodo_array_type::arr_type_enum out_arr_type,
              bodo_array_type::arr_type_enum in_arr_type,
              Bodo_CTypes::CTypeEnum dtype, typename IndexT>
        requires(out_arr_type == bodo_array_type::ARRAY_ITEM &&
                 in_arr_type == bodo_array_type::ARRAY_ITEM)
    void UnsafeAppendRows(const std::shared_ptr<array_info>& in_arr,
                          const std::span<const IndexT> idxs, size_t idx_start,
                          size_t idx_length);

    /**
     * @brief UnsafeAppendRows implementation where both arrays are map
     * arrays.
     *
     * @param in_arr The array from which we are inserting.
     * @param idxs The indices giving which rows in in_arr we want to insert.
     * @param idx_start The start location in idxs from which to insert.
     * @param idx_length The number of rows we will insert.
     */
    template <bodo_array_type::arr_type_enum out_arr_type,
              bodo_array_type::arr_type_enum in_arr_type,
              Bodo_CTypes::CTypeEnum dtype, typename IndexT>
        requires(out_arr_type == bodo_array_type::MAP &&
                 in_arr_type == bodo_array_type::MAP)
    void UnsafeAppendRows(const std::shared_ptr<array_info>& in_arr,
                          const std::span<const IndexT> idxs, size_t idx_start,
                          size_t idx_length);

    /**
     * @brief UnsafeAppendRows implementation where both arrays are struct
     * arrays.
     *
     * @param in_arr The array from which we are inserting.
     * @param idxs The indices giving which rows in in_arr we want to insert.
     * @param idx_start The start location in idxs from which to insert.
     * @param idx_length The number of rows we will insert.
     */
    template <bodo_array_type::arr_type_enum out_arr_type,
              bodo_array_type::arr_type_enum in_arr_type,
              Bodo_CTypes::CTypeEnum dtype, typename IndexT>
        requires(out_arr_type == bodo_array_type::STRUCT &&
                 in_arr_type == bodo_array_type::STRUCT)
    void UnsafeAppendRows(const std::shared_ptr<array_info>& in_arr,
                          const std::span<const IndexT> idxs, size_t idx_start,
                          size_t idx_length);

    /**
     * @brief Determine how many rows can be appended to a given array buffer
     * from in_arr with the given indices without resizing the underlying
     * buffers. The attempted append is for idxs[idx_start: idx_start +
     * idx_length] and the maximum value this will return is idx_length.
     *
     * This is the implementation for non-string arrays.
     *
     * @param in_arr The array from which we are inserting.
     * @param idxs The indices giving which rows in in_arr we want to insert.
     * @param idx_start The start location in idxs from which to insert.
     * @param idx_length The number of rows we want to insert.
     * @return size_t The number of rows that fit in the current chunk in the
     * interval [0, idx_length].
     */
    template <bodo_array_type::arr_type_enum arr_type, typename IndexT>
        requires(arr_type != bodo_array_type::STRING)
    size_t NumRowsCanAppendWithoutResizing(
        const std::shared_ptr<array_info>& in_arr,
        const std::span<const IndexT> idxs, size_t idx_start,
        size_t idx_length) {
        return std::min(this->capacity - static_cast<size_t>(this->size),
                        idx_length);
    }

    /**
     * @brief Determine how many rows can be appended to a given array buffer
     * from in_arr with the given indices without resizing the underlying
     * buffers. The attempted append is for idxs[idx_start: idx_start +
     * idx_length] and the maximum value this will return is idx_length.
     *
     * This is the implementation for string arrays.
     *
     * @param in_arr The array from which we are inserting.
     * @param idxs The indices giving which rows in in_arr we want to insert.
     * @param idx_start The start location in idxs from which to insert.
     * @param idx_length The number of rows we want to insert.
     * @return size_t The number of rows that fit in the current chunk in the
     * interval [0, idx_length].
     */
    template <bodo_array_type::arr_type_enum arr_type, typename IndexT>
        requires(arr_type == bodo_array_type::STRING)
    size_t NumRowsCanAppendWithoutResizing(
        const std::shared_ptr<array_info>& in_arr,
        const std::span<const IndexT> idxs, size_t idx_start,
        size_t idx_length) {
        // All types can only return at most as many rows as the capacity
        // allows.
        size_t max_rows = std::min(
            this->capacity - static_cast<size_t>(this->size), idx_length);
        // If we have a string array check the memory usage.
        int64_t buffer_size = this->data_array->buffers[0]->size();
        // Compute the total amount of memory needed.
        int64_t total_memory = this->data_array->n_sub_elems();
        const offset_t* offsets = (offset_t*)in_arr->data2<arr_type>();
        // Iterate over the max_rows we might keep and check the memory.
        size_t idx_end = idx_start + max_rows;
        for (size_t i = idx_start; i < idx_end; i++) {
            int64_t row_ind = idxs[i];
            // Compute the memory for the current string.
            const int64_t new_str_len =
                row_ind < 0 ? 0 : offsets[row_ind + 1] - offsets[row_ind];
            total_memory += new_str_len;
            // Check if we can fit with resizing.
            if (total_memory > buffer_size) {
                return i - idx_start;
            }
        }
        return max_rows;
    }

    /**
     * @brief Determine how many rows can be appended to a given array buffer
     * from in_arr with the given indices. The attempted append is for
     * idxs[idx_start: idx_start + idx_length] and the maximum value this will
     * return is idx_length.
     *
     * This is the implementation for non-string arrays.
     *
     * @param in_arr The array from which we are inserting.
     * @param idxs The indices giving which rows in in_arr we want to insert.
     * @param idx_start The start location in idxs from which to insert.
     * @param idx_length The number of rows we want to insert.
     * @return size_t The number of rows that fit in the current chunk in the
     * interval [0, idx_length].
     */
    template <bodo_array_type::arr_type_enum arr_type, typename IndexT>
        requires(arr_type != bodo_array_type::STRING)
    size_t NumRowsCanAppend(const std::shared_ptr<array_info>& in_arr,
                            const std::span<const IndexT> idxs,
                            size_t idx_start, size_t idx_length) {
        return std::min(this->capacity - static_cast<size_t>(this->size),
                        idx_length);
    }

    /**
     * @brief Determine how many rows can be appended to a given array buffer
     * from in_arr with the given indices. The attempted append is for
     * idxs[idx_start: idx_start + idx_length] and the maximum value this will
     * return is idx_length.
     *
     * This is the implementation for string arrays.
     *
     * @param in_arr The array from which we are inserting.
     * @param idxs The indices giving which rows in in_arr we want to insert.
     * @param idx_start The start location in idxs from which to insert.
     * @param idx_length The number of rows we want to insert.
     * @return size_t The number of rows that fit in the current chunk in the
     * interval [0, idx_length].
     */
    template <bodo_array_type::arr_type_enum arr_type, typename IndexT>
        requires(arr_type == bodo_array_type::STRING)
    size_t NumRowsCanAppend(const std::shared_ptr<array_info>& in_arr,
                            const std::span<const IndexT> idxs,
                            size_t idx_start, size_t idx_length) {
        // String is capped by memory and number of rows.
        size_t max_rows = std::min(
            this->capacity - static_cast<size_t>(this->size), idx_length);
        // If we have a string array check the memory usage.
        // TODO: Move to the constructor.
        int64_t max_possible_buffer_size =
            this->data_array->buffers[0]->size() *
            std::pow(2, this->max_resize_count - this->resize_count);
        // Since we can't track actual resizes, we will track the amount of
        // memory we would have used if we had resizes remaining.
        int64_t max_possible_remaining_resize =
            this->data_array->buffers[0]->size() *
            std::pow(2, (this->max_resize_count - 1) - (this->resize_count));
        // Compute the total amount of memory needed.
        int64_t total_memory = this->data_array->n_sub_elems();
        const offset_t* offsets = (offset_t*)in_arr->data2<arr_type>();
        // Iterate over the max_rows we might keep and check the memory.
        size_t idx_end = idx_start + max_rows;
        for (size_t i = idx_start; i < idx_end; i++) {
            int64_t row_ind = idxs[i];
            // Compute the memory for the current string.
            const int64_t new_str_len =
                row_ind < 0 ? 0 : offsets[row_ind + 1] - offsets[row_ind];
            total_memory += new_str_len;
            // Check if we can fit with resizing.
            if (total_memory > max_possible_buffer_size) {
                size_t allowed_length = i - idx_start;
                // Check for the case where this string is very large
                // on its own. We have to support this case in some form
                // for reliability.
                // We will allow resizing this buffer endlessly if we
                // have resize attempts remaining, else we will let the
                // next chunk handle it. Since other steps may have required
                // resizing we will check the previous amount of memory would
                // have fit if we had 1 resize left.
                int64_t old_memory = total_memory - new_str_len;
                if ((new_str_len > max_possible_buffer_size) &&
                    (old_memory <= max_possible_remaining_resize)) {
                    return allowed_length + 1;
                } else {
                    return allowed_length;
                }
            }
        }
        return max_rows;
    }

    /**
     * @brief Append rows into the array buffer from in_arr with the given
     * indices. The append is for idxs[idx_start: idx_start + idx_length].
     *
     * This is the implementation for non-string arrays.
     *
     * @param in_arr The array from which we are inserting.
     * @param idxs The indices giving which rows in in_arr we want to insert.
     * @param idx_start The start location in idxs from which to insert.
     * @param idx_length The number of rows we will insert.
     */
    template <bodo_array_type::arr_type_enum out_arr_type,
              bodo_array_type::arr_type_enum in_arr_type,
              Bodo_CTypes::CTypeEnum dtype, typename IndexT>
        requires(in_arr_type != bodo_array_type::STRING)
    void AppendRows(const std::shared_ptr<array_info>& in_arr,
                    const std::span<const IndexT> idxs, size_t idx_start,
                    size_t idx_length) {
        bool can_append =
            idx_length == this->NumRowsCanAppendWithoutResizing<in_arr_type>(
                              in_arr, idxs, idx_start, idx_length);
        if (!can_append) {
            throw std::runtime_error(
                "ChunkedTableArrayBuilder::AppendRows: Cannot append row!");
        }
        this->UnsafeAppendRows<out_arr_type, in_arr_type, dtype>(
            in_arr, idxs, idx_start, idx_length);
    }

    /**
     * @brief Append rows into the array buffer from in_arr with the given
     * indices. The append is for idxs[idx_start: idx_start + idx_length].
     *
     * This is the implementation for string arrays.
     *
     * @param in_arr The array from which we are inserting.
     * @param idxs The indices giving which rows in in_arr we want to insert.
     * @param idx_start The start location in idxs from which to insert.
     * @param idx_length The number of rows we will insert.
     */
    template <bodo_array_type::arr_type_enum out_arr_type,
              bodo_array_type::arr_type_enum in_arr_type,
              Bodo_CTypes::CTypeEnum dtype, typename IndexT>
        requires(in_arr_type == bodo_array_type::STRING)
    void AppendRows(const std::shared_ptr<array_info>& in_arr,
                    const std::span<const IndexT> idxs, size_t idx_start,
                    size_t idx_length) {
        // Fast path for most appends.
        if (idx_length == this->NumRowsCanAppendWithoutResizing<in_arr_type>(
                              in_arr, idxs, idx_start, idx_length)) {
            // Fast path for most appends.
            this->UnsafeAppendRows<out_arr_type, in_arr_type, dtype>(
                in_arr, idxs, idx_start, idx_length);
            return;
        }
        // Check if it can be appended after resizing:
        if (idx_length != this->NumRowsCanAppend<in_arr_type, IndexT>(
                              in_arr, idxs, idx_start, idx_length)) {
            throw std::runtime_error(
                "ChunkedTableArrayBuilder::AppendRow: Cannot append row!");
        }
        // TODO: Make this more efficient by getting the string
        // resize information from CanAppend or similar helper so we don't
        // have to recompute that
        // (https://bodo.atlassian.net/browse/BSE-556).

        // Compute the total amount of memory needed.
        int64_t total_memory = this->data_array->n_sub_elems();
        const offset_t* offsets = (offset_t*)in_arr->data2<in_arr_type>();
        // Iterate over the max_rows we might keep and check the memory.
        size_t idx_end = idx_start + idx_length;
        for (size_t i = idx_start; i < idx_end; i++) {
            int64_t row_ind = idxs[i];
            // Compute the memory for the current string.
            const int64_t new_str_len =
                row_ind < 0 ? 0 : offsets[row_ind + 1] - offsets[row_ind];
            total_memory += new_str_len;
        }
        int64_t buffer_size = this->data_array->buffers[0]->size();
        // We must resize because NumRowsCanAppendWithoutResizing failed.
        while ((total_memory > buffer_size) &&
               (this->resize_count < this->max_resize_count)) {
            buffer_size *= 2;
            this->resize_count += 1;
        }
        // In case it's still not sufficient (single very large string
        // case), just resize to required length.
        buffer_size = std::max(buffer_size, total_memory);
        CHECK_ARROW_BASE(
            this->data_array->buffers[0]->Resize(buffer_size, false),
            "ChunkedTableArrayBuilder::AppendRows: Resize failed!");
        // Now simply append into the buffer.
        this->UnsafeAppendRows<out_arr_type, in_arr_type, dtype>(
            in_arr, idxs, idx_start, idx_length);
    }

    /**
     * @brief Finalize this array. Once done, no more inserts are allowed (not
     * enforced). If shrink_to_fit == true, all buffers will be resized to
     * appropriate size based on the number of rows inserted. The minimum size
     * of any buffer in bytes is still the smallest block size in the Buffer
     * Pool.
     *
     * @param shrink_to_fit Whether to resize (i.e. shrink) buffers for
     * compactness.
     *
     */
    void Finalize(bool shrink_to_fit = true);

    /**
     * @brief Clear the buffers, i.e. set size to 0.
     * Capacity is not changed and memory is not released.
     * For DICT arrays, the dictionary state is also reset.
     * In particular, they reset to point to the dictionary of the original
     * dictionary-builder which was provided during creation and the
     * dictionary related flags are reset.
     */
    void Reset();
};

/**
 * @brief Iterator of inactive chunks for ChunkedTableBuilder. It supports
 * iterations over deque of inactive chunks and uses bodo::pin_guard to
 * automatically pin and unpin the chunks.
 * @param Iterator the iterator type. It must be a deque iterator and is either
 * std::deque<std::shared_ptr<table_info>>::iterator or
 * std::deque<std::shared_ptr<table_info>>::const_iterator
 */
template <typename Iterator>
class ChunkedTableBuilderIterator {
   public:
    using value_type = Iterator::value_type;
    using reference = Iterator::reference;

    static_assert(
        std::is_same_v<value_type, std::shared_ptr<table_info>>,
        "Iterator must be a iterator of deque<std::shared_ptr<table_info>>");

    ChunkedTableBuilderIterator(Iterator&& iter, std::deque<value_type>& chunks)
        : iter(iter), chunks(chunks) {
        if (this->iter != chunks.end()) {
            chunk_guard.emplace(**this->iter);
        }
    }

    ChunkedTableBuilderIterator& operator++() {
        // Just like any STL container, attempting to increment or access the
        // value of the end() iterator of a ChunkedTableBuilder is undefined
        // behavior
        if (++iter == chunks.end()) {
            chunk_guard.reset();
        } else {
            chunk_guard.emplace(**iter);
        }
        return *this;
    }

    bool operator==(const ChunkedTableBuilderIterator& other) const {
        return iter == other.iter;
    }

    bool operator!=(const ChunkedTableBuilderIterator& other) const {
        return iter != other.iter;
    }

    reference operator*() const { return *iter; }

   private:
    Iterator iter;
    std::deque<value_type>& chunks;
    std::optional<bodo::pin_guard<table_info>> chunk_guard;
};

/**
 * @brief Interface and implementation of a table builder that consumes tables
 * or rows and produces a stream of tables with a bound on the number of rows.
 * This stream will be consumed and output by implementations in child classes
 * which allows customizing the behavior of newly produced tables. (e.g.
 * ChunkedTableBuilder will unpin tables as they are produced)
 */
class AbstractChunkedTableBuilder {
   public:
    /* Active chunk state */

    // TODO Add an option to re-use the "active chunk" since some
    // PushActiveChunk implementations make a copy of the data, which allows us
    // to reset and reuse the buffers. Similarly, we can try to not shrink the
    // buffers in these cases since the buffers will be reused.

    // Active chunk
    std::shared_ptr<table_info> active_chunk;
    // Number of rows inserted into the active chunk
    size_t active_chunk_size = 0;
    // Max number of rows that can be inserted into the chunk
    const size_t active_chunk_capacity;
    // Keep a handle on the arrays in the table so we can
    // append and finalize them correctly.
    std::vector<ChunkedTableArrayBuilder> active_chunk_array_builders;
    // Maximum number of times that array build buffers are allowed to
    // grow by factor of 2x. Only applicable for variable size
    // data types like strings.
    const size_t max_resize_count_for_variable_size_dtypes;

    // Dummy output chunk that will be returned when there are
    // no more rows left in the buffer.
    std::shared_ptr<table_info> dummy_output_chunk;

    // Dictionary builders for the arrays. This is nullptr for non-dict columns.
    const std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders;

    // XXX In the future, we can keep track of the amount
    // of memory that has been allocated, etc. and use that to determine
    // when to stop building the active chunk.

    /* Aggregate statistics */

    // Total rows inserted into the table across its lifetime.
    size_t total_size = 0;
    // Total rows that are in "un-popped" chunks (including the active chunk)
    size_t total_remaining = 0;
    // Max number of rows after any append
    size_t max_reached_size = 0;
    // Total append time
    MetricBase::TimerValue append_time = 0;

    /* Memory buffer pool */
    bodo::IBufferPool* const pool;
    std::shared_ptr<::arrow::MemoryManager> mm;

    // XXX In the future, we could keep track of the
    // allocated memory as well. This might be useful
    // for the build table buffers of inactive partitions
    // since we can use that information to decide
    // whether to re-partition or not.

    /**
     * @brief Construct a new Chunked Table with the given bodo::Schema.
     *
     * @param schema Data types of the columns as a bodo::Schema
     * @param dict_builders_ Dictionary builders to use for DICT arrays.
     * @param chunk_size Max number of rows allowed in each chunk.
     * @param max_resize_count_for_variable_size_dtypes How many times are
     * we allowed to resize (grow by 2x) buffers for variable size
     * data types like strings. 0 means resizing is not allowed. 2 means
     * that the final size could be 4x of the original.
     * @param pool Memory pool to use for allocations during the execution of
     * this function.
     * @param mm Memory manager associated with the pool.
     */
    AbstractChunkedTableBuilder(
        const std::shared_ptr<bodo::Schema>& schema,
        const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders_,
        size_t chunk_size,
        size_t max_resize_count_for_variable_size_dtypes =
            DEFAULT_MAX_RESIZE_COUNT_FOR_VARIABLE_SIZE_DTYPES,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager());

    virtual ~AbstractChunkedTableBuilder() = default;

    /**
     * @brief Finalize the active chunk and create a new active chunk.
     * This will call Finalize on all the arrays in this chunk, and then
     * insert the resulting table_info into this->chunks.
     * If the active chunk is empty, this will be a NOP.
     * In the future, it will also unpin the finalized chunk.
     *
     * @param shrink_to_fit Whether the array buffers should be
     * shrunk to fit the required capacity.
     *
     */
    void FinalizeActiveChunk(bool shrink_to_fit = true);

    /**
     * @brief Append a batch of data to the buffer
     *
     * @param in_table Input table with the new rows.
     * @param append_rows Bit-vector indicating which rows to append.
     * @param in_table_start_offset Starting offset into the 'in_table'. e.g. if
     * 'append_rows' is [t, f, t, f, t] and 'in_table_start_offset' is 4, then
     * we will append rows 4, 6 & 8 from the input table.
     */
    void AppendBatch(const std::shared_ptr<table_info>& in_table,
                     const std::vector<bool>& append_rows,
                     const int64_t in_table_start_offset = 0);

    void AppendBatch(const std::shared_ptr<table_info>& in_table,
                     const std::vector<bool>& append_rows,
                     const size_t num_append_rows,
                     const int64_t in_table_start_offset = 0);

    void AppendBatch(const std::shared_ptr<table_info>& in_table,
                     const std::unique_ptr<uint8_t[]>& append_rows,
                     const int64_t in_table_start_offset = 0);

    void AppendBatch(const std::shared_ptr<table_info>& in_table,
                     const std::unique_ptr<uint8_t[]>& append_rows,
                     const size_t num_append_rows,
                     const int64_t in_table_start_offset);

    void AppendBatch(const std::shared_ptr<table_info>& in_table);

    void AppendBatch(const std::shared_ptr<table_info>& in_table,
                     const std::span<const int64_t> idxs);

    void AppendBatch(const std::shared_ptr<table_info>& build_table,
                     const std::span<const int64_t> idxs,
                     const std::span<const uint64_t> cols);

    void AppendBatch(const std::shared_ptr<table_info>& build_table,
                     const std::span<const int32_t> idxs,
                     const std::span<const uint64_t> cols);

    /**
     * @brief Similar to AppendRow, but specifically for
     * Join Output Buffer use case. In join computation, we collect
     * the row indices from the build and probe tables which need
     * to be inserted together. Furthermore, due to column pruning,
     * we need to only append from required columns.
     *
     * @param build_table Build/Left table to insert rows from.
     * @param probe_table Probe/Right table to insert rows from.
     * @param build_idxs Indices from the build table.
     * @param probe_idxs Corresponding indices from the probe table.
     * @param build_kept_cols Indices of the columns from the build table.
     * @param probe_kept_cols Indices of the columns from the probe table.
     * @param is_mark_join If true, this is a mark join and we only append
     * the probe table rows and mark column.
     */
    void AppendJoinOutput(std::shared_ptr<table_info> build_table,
                          std::shared_ptr<table_info> probe_table,
                          const std::span<const int64_t> build_idxs,
                          const std::span<const int64_t> probe_idxs,
                          const std::vector<uint64_t>& build_kept_cols,
                          const std::vector<uint64_t>& probe_kept_cols,
                          bool is_mark_join = false);

    /**
     * @brief Finalize this chunked table. This will finalize
     * the active chunk. If the active chunk is empty, it will
     * be discarded and its memory will be freed.
     * No more rows should be appended once it is finalized
     * (not enforced at this point).
     *
     * @param shrink_to_fit If we finalize the active chunk, should
     * we shrink the buffers to fit the required size.
     */
    void Finalize(bool shrink_to_fit = true);

    /**
     * @brief Get the first available chunk. It is assumed that the output table
     * is pinned.
     *
     * @param force_return If true, when no complete chunks are availible, it
     * will finalize and return the active chunk (useful in
     * the "is_last" case), else it will return an empty table. Note that it
     * might return an empty table even in the force_return case if the active
     * chunk is empty.
     * @return std::tuple<std::shared_ptr<table_info>, int64_t> Tuple
     * of the chunk and the size of the chunk (in case all columns are dead).
     */
    std::tuple<std::shared_ptr<table_info>, int64_t> PopChunk(
        bool force_return = false);

    /**
     * @brief Clear the buffers, i.e. set size to 0.
     * Capacity is not changed and memory is not released.
     * For DICT arrays, the dictionary state is also reset.
     * In particular, they reset to point to the dictionary of the original
     * dictionary-builder which was provided during creation and the
     * dictionary related flags are reset.
     */
    void Reset();

    /// @brief True if there are 0 rows in the table, prepared or active
    bool empty() const;

    /**
     * @brief Wrapper method over AppendBatch to append an input table
     * with string columns that should be converted to dictionary-encoded
     * string columns. Currently only used for SnowflakeReader
     *
     * @param in_table Input with string columns to convert
     */
    void UnifyDictionariesAndAppend(
        const std::shared_ptr<table_info>& in_table);

   private:
    template <typename IndexT>
    void AppendBatch(const std::shared_ptr<table_info>& build_table,
                     const std::span<const IndexT> idxs,
                     const std::span<const uint64_t> cols);

    /**
     * @brief number of chunks that are ready for output
     */
    virtual size_t NumReadyChunks() = 0;

    /**
     * @brief Pop the first chunk for output
     */
    virtual std::shared_ptr<table_info> PopFront() = 0;

    /**
     * @brief trigger consumption of this->active_chunk. After this method is
     * called, the reference in this->active_chunk will be dropped.
     */
    virtual void PushActiveChunk() = 0;

    /**
     * @brief Reset any state used by a child class
     */
    virtual void ResetInternal() = 0;
};

/**
 * @brief Chunked Table Builder for use cases like outputs
 * of streaming operators, etc.
 * Columnar table chunks (essentially PAX) are maintained such that
 * each chunk is of size at most 'active_chunk_capacity'
 * rows. The chunks are stored in a std::deque to provide
 * queue like behavior, while allowing iteration over the
 * chunks without removing them. All finalized chunks are kept unpinned and we
 * use PopChunk to pin and return chunks. See design doc here:
 * https://bodo.atlassian.net/l/cp/mzidHW9G.
 */
class ChunkedTableBuilder : public AbstractChunkedTableBuilder {
   public:
    using iterator = ChunkedTableBuilderIterator<
        std::deque<std::shared_ptr<table_info>>::iterator>;
    using const_iterator = ChunkedTableBuilderIterator<
        std::deque<std::shared_ptr<table_info>>::const_iterator>;

    // Queue of finalized chunks. We use a deque instead of
    // a regular queue since it gives us ability to both
    // iterate over elements as well as pop/push. Finalized chunks are unpinned.
    // If we want to access finalized chunks, we need to pin and unpin them
    // manually.
    std::deque<std::shared_ptr<table_info>> chunks;

    ChunkedTableBuilder(
        const std::shared_ptr<bodo::Schema>& schema,
        const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders,
        size_t chunk_size,
        size_t max_resize_count_for_variable_size_dtypes =
            DEFAULT_MAX_RESIZE_COUNT_FOR_VARIABLE_SIZE_DTYPES,
        bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
        std::shared_ptr<::arrow::MemoryManager> mm =
            bodo::default_buffer_memory_manager())
        : AbstractChunkedTableBuilder(schema, dict_builders, chunk_size,
                                      max_resize_count_for_variable_size_dtypes,
                                      pool, mm) {}

    iterator begin() { return iterator(chunks.begin(), chunks); }

    iterator end() { return iterator(chunks.end(), chunks); }

    const_iterator cbegin() { return const_iterator(chunks.cbegin(), chunks); }

    const_iterator cend() { return const_iterator(chunks.cend(), chunks); }

   private:
    // Gets the size of chunks
    size_t NumReadyChunks() final;

    // Removes the first element of chunks, pins it, and returns it
    std::shared_ptr<table_info> PopFront() final;

    // Unpins the active chunks and appends to chunks
    void PushActiveChunk() final;

    // Clears chunks
    void ResetInternal() final;
};
