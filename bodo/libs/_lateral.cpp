// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include "_lateral.h"
#include <Python.h>
#include "_array_utils.h"
#include "_bodo_common.h"

/**
 * Implements the rules for combining 2 numeric dtypes:
 * - Signed int vs signed int: goes with the larger bit size, signed
 * - Unsigned int vs unsigned int: goes with the larger bit size, unsigned
 * - Signed int vs unsigned int: if the unsigned is the larger bit size, make
 * keep that but make it signed. If the two have the same bit sizes or the
 * signed is the larger bit size, go with 1 size larger than the larger size,
 * and keep it signed.
 * - Float vs float: goes with the larger bit size
 * - Decimal, Int128, Date, Time, Datetime, Timedelta, Boolean: only allowed
 * with itself, for now
 */
Bodo_CTypes::CTypeEnum combine_numeric_dtypes(Bodo_CTypes::CTypeEnum dtype_0,
                                              Bodo_CTypes::CTypeEnum dtype_1) {
#define combination_case(typ_1, out_typ) \
    case typ_1: {                        \
        return out_typ;                  \
    }
#define error_default()           \
    default:                      \
        throw std::runtime_error( \
            "lateral_flatten_struct: unable to combine numeric dtypes");
    switch (dtype_0) {
        case Bodo_CTypes::_BOOL: {
            switch (dtype_1) {
                combination_case(Bodo_CTypes::_BOOL, Bodo_CTypes::_BOOL);
                error_default();
            }
        }
        case Bodo_CTypes::DATE: {
            switch (dtype_1) {
                combination_case(Bodo_CTypes::DATE, Bodo_CTypes::DATE);
                error_default();
            }
        }
        case Bodo_CTypes::TIME: {
            switch (dtype_1) {
                combination_case(Bodo_CTypes::TIME, Bodo_CTypes::TIME);
                error_default();
            }
        }
        case Bodo_CTypes::DATETIME: {
            switch (dtype_1) {
                combination_case(Bodo_CTypes::DATETIME, Bodo_CTypes::DATETIME);
                error_default();
            }
        }
        case Bodo_CTypes::TIMEDELTA: {
            switch (dtype_1) {
                combination_case(Bodo_CTypes::TIMEDELTA,
                                 Bodo_CTypes::TIMEDELTA);
                error_default();
            }
        }
        case Bodo_CTypes::DECIMAL: {
            switch (dtype_1) {
                combination_case(Bodo_CTypes::DECIMAL, Bodo_CTypes::DECIMAL);
                error_default();
            }
        }
        case Bodo_CTypes::INT128: {
            switch (dtype_1) {
                combination_case(Bodo_CTypes::INT128, Bodo_CTypes::INT128);
                error_default();
            }
        }
        case Bodo_CTypes::INT8: {
            switch (dtype_1) {
                combination_case(Bodo_CTypes::INT8, Bodo_CTypes::INT8);
                combination_case(Bodo_CTypes::INT16, Bodo_CTypes::INT16);
                combination_case(Bodo_CTypes::INT32, Bodo_CTypes::INT32);
                combination_case(Bodo_CTypes::INT64, Bodo_CTypes::INT64);
                combination_case(Bodo_CTypes::UINT8, Bodo_CTypes::INT16);
                combination_case(Bodo_CTypes::UINT16, Bodo_CTypes::INT32);
                combination_case(Bodo_CTypes::UINT32, Bodo_CTypes::INT64);
                combination_case(Bodo_CTypes::UINT64, Bodo_CTypes::INT64);
                error_default();
            }
        }
        case Bodo_CTypes::UINT8: {
            switch (dtype_1) {
                combination_case(Bodo_CTypes::INT8, Bodo_CTypes::INT16);
                combination_case(Bodo_CTypes::INT16, Bodo_CTypes::INT16);
                combination_case(Bodo_CTypes::INT32, Bodo_CTypes::INT32);
                combination_case(Bodo_CTypes::INT64, Bodo_CTypes::INT64);
                combination_case(Bodo_CTypes::UINT8, Bodo_CTypes::UINT8);
                combination_case(Bodo_CTypes::UINT16, Bodo_CTypes::UINT16);
                combination_case(Bodo_CTypes::UINT32, Bodo_CTypes::UINT32);
                combination_case(Bodo_CTypes::UINT64, Bodo_CTypes::UINT64);
                error_default();
            }
        }
        case Bodo_CTypes::INT16: {
            switch (dtype_1) {
                combination_case(Bodo_CTypes::INT8, Bodo_CTypes::INT16);
                combination_case(Bodo_CTypes::INT16, Bodo_CTypes::INT16);
                combination_case(Bodo_CTypes::INT32, Bodo_CTypes::INT32);
                combination_case(Bodo_CTypes::INT64, Bodo_CTypes::INT64);
                combination_case(Bodo_CTypes::UINT8, Bodo_CTypes::INT16);
                combination_case(Bodo_CTypes::UINT16, Bodo_CTypes::INT32);
                combination_case(Bodo_CTypes::UINT32, Bodo_CTypes::INT64);
                combination_case(Bodo_CTypes::UINT64, Bodo_CTypes::INT64);
                error_default();
            }
        }
        case Bodo_CTypes::UINT16: {
            switch (dtype_1) {
                combination_case(Bodo_CTypes::INT8, Bodo_CTypes::INT32);
                combination_case(Bodo_CTypes::INT16, Bodo_CTypes::INT32);
                combination_case(Bodo_CTypes::INT32, Bodo_CTypes::INT32);
                combination_case(Bodo_CTypes::INT64, Bodo_CTypes::INT64);
                combination_case(Bodo_CTypes::UINT8, Bodo_CTypes::UINT16);
                combination_case(Bodo_CTypes::UINT16, Bodo_CTypes::UINT16);
                combination_case(Bodo_CTypes::UINT32, Bodo_CTypes::UINT32);
                combination_case(Bodo_CTypes::UINT64, Bodo_CTypes::UINT64);
                error_default();
            }
        }
        case Bodo_CTypes::INT32: {
            switch (dtype_1) {
                combination_case(Bodo_CTypes::INT8, Bodo_CTypes::INT32);
                combination_case(Bodo_CTypes::INT16, Bodo_CTypes::INT32);
                combination_case(Bodo_CTypes::INT32, Bodo_CTypes::INT32);
                combination_case(Bodo_CTypes::INT64, Bodo_CTypes::INT64);
                combination_case(Bodo_CTypes::UINT8, Bodo_CTypes::INT32);
                combination_case(Bodo_CTypes::UINT16, Bodo_CTypes::INT32);
                combination_case(Bodo_CTypes::UINT32, Bodo_CTypes::INT64);
                combination_case(Bodo_CTypes::UINT64, Bodo_CTypes::INT64);
                error_default();
            }
        }
        case Bodo_CTypes::UINT32: {
            switch (dtype_1) {
                combination_case(Bodo_CTypes::INT8, Bodo_CTypes::INT64);
                combination_case(Bodo_CTypes::INT16, Bodo_CTypes::INT64);
                combination_case(Bodo_CTypes::INT32, Bodo_CTypes::INT64);
                combination_case(Bodo_CTypes::INT64, Bodo_CTypes::INT64);
                combination_case(Bodo_CTypes::UINT8, Bodo_CTypes::UINT32);
                combination_case(Bodo_CTypes::UINT16, Bodo_CTypes::UINT32);
                combination_case(Bodo_CTypes::UINT32, Bodo_CTypes::UINT32);
                combination_case(Bodo_CTypes::UINT64, Bodo_CTypes::UINT64);
                error_default();
            }
        }
        case Bodo_CTypes::INT64: {
            switch (dtype_1) {
                combination_case(Bodo_CTypes::INT8, Bodo_CTypes::INT64);
                combination_case(Bodo_CTypes::INT16, Bodo_CTypes::INT64);
                combination_case(Bodo_CTypes::INT32, Bodo_CTypes::INT64);
                combination_case(Bodo_CTypes::INT64, Bodo_CTypes::INT64);
                combination_case(Bodo_CTypes::UINT8, Bodo_CTypes::INT64);
                combination_case(Bodo_CTypes::UINT16, Bodo_CTypes::INT64);
                combination_case(Bodo_CTypes::UINT32, Bodo_CTypes::INT64);
                combination_case(Bodo_CTypes::UINT64, Bodo_CTypes::INT64);
                error_default();
            }
        }
        case Bodo_CTypes::UINT64: {
            switch (dtype_1) {
                combination_case(Bodo_CTypes::INT8, Bodo_CTypes::INT64);
                combination_case(Bodo_CTypes::INT16, Bodo_CTypes::INT64);
                combination_case(Bodo_CTypes::INT32, Bodo_CTypes::INT64);
                combination_case(Bodo_CTypes::INT64, Bodo_CTypes::INT64);
                combination_case(Bodo_CTypes::UINT8, Bodo_CTypes::UINT64);
                combination_case(Bodo_CTypes::UINT16, Bodo_CTypes::UINT64);
                combination_case(Bodo_CTypes::UINT32, Bodo_CTypes::UINT64);
                combination_case(Bodo_CTypes::UINT64, Bodo_CTypes::UINT64);
                error_default();
            }
        }
        case Bodo_CTypes::FLOAT32: {
            switch (dtype_1) {
                combination_case(Bodo_CTypes::FLOAT32, Bodo_CTypes::FLOAT32);
                combination_case(Bodo_CTypes::FLOAT64, Bodo_CTypes::FLOAT64);
                error_default();
            }
        }
        case Bodo_CTypes::FLOAT64: {
            switch (dtype_1) {
                combination_case(Bodo_CTypes::FLOAT32, Bodo_CTypes::FLOAT64);
                combination_case(Bodo_CTypes::FLOAT64, Bodo_CTypes::FLOAT64);
                error_default();
            }
        }
            error_default();
    }
}

/**
 * Takes in an array of nullable/numpy arrays and finds a common numeric
 * dtype for them to use when interleaved into one combined array, if possible.
 */
Bodo_CTypes::CTypeEnum get_common_numeric_dtype(
    const std::vector<std::shared_ptr<array_info>> &arrs) {
    Bodo_CTypes::CTypeEnum common_dtype = arrs[0]->dtype;
    for (size_t i = 1; i < arrs.size(); i++) {
        common_dtype = combine_numeric_dtypes(common_dtype, arrs[i]->dtype);
    }
    return common_dtype;
}

/**
 * Helper for lateral_flatten_copy_numeric_value that knows the dtypes
 * of the input array and the output array.
 */
template <Bodo_CTypes::CTypeEnum out_dtype, Bodo_CTypes::CTypeEnum in_dtype>
void lateral_flatten_copy_numeric_value_with_dtypes(
    std::shared_ptr<array_info> &out_arr, size_t write_idx,
    const std::shared_ptr<array_info> &in_arr, size_t read_idx) {
    using T_out = typename dtype_to_type<out_dtype>::type;
    using T_in = typename dtype_to_type<in_dtype>::type;
    T_in val;
    if (in_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        val = get_arr_item<bodo_array_type::NULLABLE_INT_BOOL, T_in, in_dtype>(
            *in_arr, read_idx);
    } else {
        val = get_arr_item<bodo_array_type::NUMPY, T_in, in_dtype>(*in_arr,
                                                                   read_idx);
    }
    T_out casted_val = static_cast<T_out>(val);
    if (out_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        set_arr_item<bodo_array_type::NULLABLE_INT_BOOL, T_out, in_dtype>(
            *out_arr, write_idx, casted_val);
    } else {
        set_arr_item<bodo_array_type::NUMPY, T_out, in_dtype>(
            *out_arr, write_idx, casted_val);
    }
    getv<T_out>(out_arr, write_idx) = casted_val;
}

/**
 * Reads a value from one numpy/nullable array and writes it into
 * another one, where the two arrays have possibly different dtypes.
 *
 * @param[in,out] out_arr: the array to write the numeric data to.
 * @param[in] write_idx: the row of out_arr to write to.
 * @param[in] in_arr: the array to extract the numeric data from.
 * @param[in] read_idx: the row of in_arr to read from.
 */
void lateral_flatten_copy_numeric_value(
    std::shared_ptr<array_info> &out_arr, size_t write_idx,
    const std::shared_ptr<array_info> &in_arr, size_t read_idx) {
#define LFCNV_IN_DTYPE(out_dtype, in_dtype)                                  \
    case in_dtype: {                                                         \
        lateral_flatten_copy_numeric_value_with_dtypes<out_dtype, in_dtype>( \
            out_arr, write_idx, in_arr, read_idx);                           \
        return;                                                              \
    }

#define LFCNV_OUT_DTYPE(out_dtype)                                         \
    case out_dtype:                                                        \
        switch (in_arr->dtype) {                                           \
            LFCNV_IN_DTYPE(out_dtype, Bodo_CTypes::INT8);                  \
            LFCNV_IN_DTYPE(out_dtype, Bodo_CTypes::UINT8);                 \
            LFCNV_IN_DTYPE(out_dtype, Bodo_CTypes::INT16);                 \
            LFCNV_IN_DTYPE(out_dtype, Bodo_CTypes::UINT16);                \
            LFCNV_IN_DTYPE(out_dtype, Bodo_CTypes::INT32);                 \
            LFCNV_IN_DTYPE(out_dtype, Bodo_CTypes::UINT32);                \
            LFCNV_IN_DTYPE(out_dtype, Bodo_CTypes::INT64);                 \
            LFCNV_IN_DTYPE(out_dtype, Bodo_CTypes::UINT64);                \
            LFCNV_IN_DTYPE(out_dtype, Bodo_CTypes::FLOAT32);               \
            LFCNV_IN_DTYPE(out_dtype, Bodo_CTypes::FLOAT64);               \
            LFCNV_IN_DTYPE(out_dtype, Bodo_CTypes::DECIMAL);               \
            LFCNV_IN_DTYPE(out_dtype, Bodo_CTypes::INT128);                \
            LFCNV_IN_DTYPE(out_dtype, Bodo_CTypes::DATE);                  \
            LFCNV_IN_DTYPE(out_dtype, Bodo_CTypes::TIME);                  \
            LFCNV_IN_DTYPE(out_dtype, Bodo_CTypes::DATETIME);              \
            LFCNV_IN_DTYPE(out_dtype, Bodo_CTypes::TIMEDELTA);             \
            LFCNV_IN_DTYPE(out_dtype, Bodo_CTypes::_BOOL);                 \
            default:                                                       \
                throw std::runtime_error(                                  \
                    "lateral_flatten_copy_numeric_value: invalid numeric " \
                    "dtype");                                              \
        }

    switch (out_arr->dtype) {
        LFCNV_OUT_DTYPE(Bodo_CTypes::INT8);
        LFCNV_OUT_DTYPE(Bodo_CTypes::UINT8);
        LFCNV_OUT_DTYPE(Bodo_CTypes::INT16);
        LFCNV_OUT_DTYPE(Bodo_CTypes::UINT16);
        LFCNV_OUT_DTYPE(Bodo_CTypes::INT32);
        LFCNV_OUT_DTYPE(Bodo_CTypes::UINT32);
        LFCNV_OUT_DTYPE(Bodo_CTypes::INT64);
        LFCNV_OUT_DTYPE(Bodo_CTypes::UINT64);
        LFCNV_OUT_DTYPE(Bodo_CTypes::FLOAT32);
        LFCNV_OUT_DTYPE(Bodo_CTypes::FLOAT64);
        LFCNV_OUT_DTYPE(Bodo_CTypes::DECIMAL);
        LFCNV_OUT_DTYPE(Bodo_CTypes::INT128);
        LFCNV_OUT_DTYPE(Bodo_CTypes::DATE);
        LFCNV_OUT_DTYPE(Bodo_CTypes::TIME);
        LFCNV_OUT_DTYPE(Bodo_CTypes::DATETIME);
        LFCNV_OUT_DTYPE(Bodo_CTypes::TIMEDELTA);
        LFCNV_OUT_DTYPE(Bodo_CTypes::_BOOL);
        default:
            throw std::runtime_error(
                "lateral_flatten_copy_numeric_value: invalid numeric dtype");
    }
}

/**
 * Recursive procedure to interleave the fields of a struct array.
 *
 * @param[in] fields_to_pick: array mapping each row in the output array
 * to which of the columns should a value be picked from.
 * @param[in] rows_to_pick: array mapping each row in the output array
 * to which of the rows in the desired column should a value be picked from.
 * @param[in] inner_arrs: the current child arrays being interleaved.
 *
 * @return the interleaved column of the original field columns.
 */
std::shared_ptr<array_info> interleave_struct_columns(
    const std::vector<size_t> &fields_to_pick,
    const std::vector<size_t> &rows_to_pick,
    const std::vector<std::shared_ptr<array_info>> &inner_arrs,
    bodo::IBufferPool *const pool, std::shared_ptr<::arrow::MemoryManager> mm);

/**
 * Procedure to interleave the fields of a struct array, invoked on a collection
 * of string/dictionary-encoded arrays.
 *
 * @param[in] fields_to_pick: array mapping each row in the output array
 * to which of the columns should a value be picked from.
 * @param[in] rows_to_pick: array mapping each row in the output array
 * to which of the rows in the desired column should a value be picked from.
 * @param[in] inner_arrs: the current child arrays being interleaved.
 *
 * @return the interleaved column of the original field columns.
 */
std::shared_ptr<array_info> interleave_string_arrays(
    const std::vector<size_t> &fields_to_pick,
    const std::vector<size_t> &rows_to_pick,
    const std::vector<std::shared_ptr<array_info>> &inner_arrs,
    bodo::IBufferPool *const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    // Setup the string and null vectors.
    size_t rows_kept = fields_to_pick.size();
    bodo::vector<uint8_t> null_vector((rows_kept + 7) >> 3, 0, pool);
    bodo::vector<std::string> string_vector;

    // For each row in the output, identify the corresponding column from
    // the inputs and the row from the column to select, and add that entry
    // to the null and string vectors.
    for (size_t row = 0; row < rows_kept; row++) {
        size_t field_to_pick = fields_to_pick[row];
        size_t row_to_pick = rows_to_pick[row];
        if (inner_arrs[field_to_pick]->get_null_bit(row_to_pick)) {
            SetBitTo(null_vector.data(), row, 1);
            std::string s;
            if (inner_arrs[field_to_pick]->arr_type ==
                bodo_array_type::STRING) {
                s = get_arr_item_str<bodo_array_type::STRING, std::string_view,
                                     Bodo_CTypes::STRING>(
                    *inner_arrs[field_to_pick], row_to_pick);
            } else {
                s = get_arr_item_str<bodo_array_type::DICT, std::string_view,
                                     Bodo_CTypes::STRING>(
                    *inner_arrs[field_to_pick], row_to_pick);
            }
            string_vector.push_back(s);
        } else {
            SetBitTo(null_vector.data(), row, 0);
            string_vector.push_back("");
        }
    }
    std::shared_ptr<array_info> res = create_string_array(
        Bodo_CTypes::STRING, null_vector, string_vector, -1, pool, mm);
    return res;
}

/**
 * Procedure to interleave the fields of a struct array, invoked on a collection
 * of array item arrays.
 *
 * @param[in] fields_to_pick: array mapping each row in the output array
 * to which of the columns should a value be picked from.
 * @param[in] rows_to_pick: array mapping each row in the output array
 * to which of the rows in the desired column should a value be picked from.
 * @param[in] inner_arrs: the current child arrays being interleaved.
 *
 * @return the interleaved column of the original field columns.
 */
std::shared_ptr<array_info> interleave_array_item_arrays(
    const std::vector<size_t> &fields_to_pick,
    const std::vector<size_t> &rows_to_pick,
    const std::vector<std::shared_ptr<array_info>> &inner_arrs,
    bodo::IBufferPool *const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    // Extrapolate the fields to pick and rows to pick to the child
    // arrays of the columns.
    size_t n_fields = inner_arrs.size();
    size_t rows_kept = fields_to_pick.size();
    std::vector<size_t> sub_fields_to_pick;
    std::vector<size_t> sub_rows_to_pick;
    std::vector<size_t> array_sizes;
    std::vector<bool> is_null;
    for (size_t row = 0; row < rows_kept; row++) {
        size_t field_to_pick = fields_to_pick[row];
        size_t row_to_pick = rows_to_pick[row];
        offset_t *offset_buffer =
            (offset_t *)(inner_arrs[field_to_pick]->buffers[0]->mutable_data() +
                         inner_arrs[field_to_pick]->offset);
        size_t start_idx = offset_buffer[row_to_pick];
        size_t end_idx = offset_buffer[row_to_pick + 1];
        for (size_t idx = start_idx; idx < end_idx; idx++) {
            sub_fields_to_pick.push_back(field_to_pick);
            sub_rows_to_pick.push_back(idx);
        }
        array_sizes.push_back(end_idx - start_idx);
        is_null.push_back(
            !(inner_arrs[field_to_pick]->get_null_bit(row_to_pick)));
    }

    // Recursively interleave the child arrays.
    std::vector<std::shared_ptr<array_info>> child_arrs;
    for (size_t i = 0; i < n_fields; i++) {
        child_arrs.push_back(inner_arrs[i]->child_arrays[0]);
    }
    std::shared_ptr<array_info> combined_inner_arrs = interleave_struct_columns(
        sub_fields_to_pick, sub_rows_to_pick, child_arrs, pool, mm);

    // Reconstruct a new array item arary using the interleaved child arrays
    // and the array sizes calculated earlier.
    size_t n_arrays = array_sizes.size();
    std::shared_ptr<array_info> result =
        alloc_array_item(n_arrays, combined_inner_arrs, 0, pool, mm);
    offset_t *offset_buffer =
        (offset_t *)(result->buffers[0]->mutable_data() + result->offset);
    offset_buffer[0] = 0;
    size_t curr_offset = 0;
    for (size_t row = 0; row < n_arrays; row++) {
        curr_offset += array_sizes[row];
        offset_buffer[row + 1] = curr_offset;
        if (is_null[row]) {
            result->set_null_bit(row, false);
        }
    }
    return result;
    // throw std::runtime_error("interleave_array_item_arrays: TODO");
}

/**
 * Procedure to interleave the fields of a struct array, invoked on a collection
 * of nullable or numpy arrays.
 *
 * @param[in] fields_to_pick: array mapping each row in the output array
 * to which of the columns should a value be picked from.
 * @param[in] rows_to_pick: array mapping each row in the output array
 * to which of the rows in the desired column should a value be picked from.
 * @param[in] inner_arrs: the current child arrays being interleaved.
 *
 * @return the interleaved column of the original field columns.
 */
std::shared_ptr<array_info> interleave_nullable_arrays(
    const std::vector<size_t> &fields_to_pick,
    const std::vector<size_t> &rows_to_pick,
    const std::vector<std::shared_ptr<array_info>> &inner_arrs,
    bodo::IBufferPool *const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    // Find the numerical type that all the values will be casted to,
    // and allocate a numpy array with the correct size.
    size_t rows_kept = fields_to_pick.size();
    Bodo_CTypes::CTypeEnum dtype = get_common_numeric_dtype(inner_arrs);
    std::shared_ptr out_arr =
        alloc_nullable_array_no_nulls(rows_kept, dtype, 0);

    // For each row, iterate across all of the fields and write them
    // to the output array at the corresponding index.
    for (size_t row = 0; row < rows_kept; row++) {
        size_t field_to_pick = fields_to_pick[row];
        size_t row_to_pick = rows_to_pick[row];
        if ((inner_arrs[field_to_pick]->arr_type == bodo_array_type::NUMPY) ||
            (inner_arrs[field_to_pick]->get_null_bit(row_to_pick))) {
            lateral_flatten_copy_numeric_value(
                out_arr, row, inner_arrs[field_to_pick], row_to_pick);
        } else {
            out_arr->set_null_bit(row, false);
        }
    }
    return out_arr;
}

/**
 * Procedure to interleave the fields of a struct array, invoked on a collection
 * of numpy arrays.
 *
 * @param[in] fields_to_pick: array mapping each row in the output array
 * to which of the columns should a value be picked from.
 * @param[in] rows_to_pick: array mapping each row in the output array
 * to which of the rows in the desired column should a value be picked from.
 * @param[in] inner_arrs: the current child arrays being interleaved.
 *
 * @return the interleaved column of the original field columns.
 */
std::shared_ptr<array_info> interleave_numpy_arrays(
    const std::vector<size_t> &fields_to_pick,
    const std::vector<size_t> &rows_to_pick,
    const std::vector<std::shared_ptr<array_info>> &inner_arrs,
    bodo::IBufferPool *const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    // Find the numerical type that all the values will be casted to,
    // and allocate a numpy array with the correct size.
    size_t rows_kept = fields_to_pick.size();
    Bodo_CTypes::CTypeEnum dtype = get_common_numeric_dtype(inner_arrs);
    std::shared_ptr out_arr = alloc_numpy(rows_kept, dtype);

    // For each row, iterate across all of the fields and write them
    // to the output array at the corresponding index.
    for (size_t row = 0; row < rows_kept; row++) {
        size_t field_to_pick = fields_to_pick[row];
        size_t row_to_pick = rows_to_pick[row];
        lateral_flatten_copy_numeric_value(
            out_arr, row, inner_arrs[field_to_pick], row_to_pick);
    }
    return out_arr;
}

std::shared_ptr<array_info> interleave_struct_columns(
    const std::vector<size_t> &fields_to_pick,
    const std::vector<size_t> &rows_to_pick,
    const std::vector<std::shared_ptr<array_info>> &inner_arrs,
    bodo::IBufferPool *const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    /* Identify which type of array combination procedure will
     * be required to interleave the child arrays:
     * - string_mode: combine string arrays (possibly with dictionary-encoded
     * arrays)
     * - dict_mode: combine only dictionary-encoded arrays
     * - array_mode: combine only array item arrays
     * - nullable_mode: combine nullable arrays (possibly with numpy arrays)
     * - numpy_mode: combine only numpy arrays
     * - none of the above: implies combining only numpy arrays
     * All other combinations are currently not allowed.
     */
    bool string_mode = false;
    bool array_mode = false;
    bool nullable_mode = false;
    bool numpy_mode = false;
    size_t n_fields = inner_arrs.size();
    for (size_t i = 0; i < n_fields; i++) {
        switch (inner_arrs[i]->arr_type) {
            case bodo_array_type::DICT:
            case bodo_array_type::STRING: {
                string_mode = true;
                if (array_mode || nullable_mode || numpy_mode)
                    throw std::runtime_error(
                        "lateral_flatten_struct: invalid struct child arrays");
                break;
            }
            case bodo_array_type::NULLABLE_INT_BOOL: {
                nullable_mode = true;
                numpy_mode = false;
                if (string_mode || array_mode)
                    throw std::runtime_error(
                        "lateral_flatten_struct: invalid struct child arrays");
                break;
            }
            case bodo_array_type::NUMPY: {
                if (inner_arrs[i]->dtype == Bodo_CTypes::FLOAT32 ||
                    inner_arrs[i]->dtype == Bodo_CTypes::FLOAT64) {
                    // If the input arrays are float arrays, the combined array
                    // should be a nullable array even if the inputs are numpy
                    // arrays.
                    nullable_mode = true;
                }
                if (!nullable_mode)
                    numpy_mode = true;
                if (string_mode || array_mode)
                    throw std::runtime_error(
                        "lateral_flatten_struct: invalid struct child arrays");
                break;
            }
            case bodo_array_type::ARRAY_ITEM: {
                array_mode = true;
                if (string_mode || nullable_mode || numpy_mode)
                    throw std::runtime_error(
                        "lateral_flatten_struct: invalid struct child arrays");
                break;
            }
            default: {
                throw std::runtime_error(
                    "lateral_flatten_struct: invalid struct child arrays");
            }
        }
    }

    // Route to one of the helpers to interleave the arrays based on which
    // compatible type combination was found from iterating across the arrays.
    if (string_mode) {
        return interleave_string_arrays(fields_to_pick, rows_to_pick,
                                        inner_arrs, pool, mm);
    } else if (array_mode) {
        return interleave_array_item_arrays(fields_to_pick, rows_to_pick,
                                            inner_arrs, pool, mm);
    } else if (nullable_mode) {
        return interleave_nullable_arrays(fields_to_pick, rows_to_pick,
                                          inner_arrs, pool, mm);
    } else if (numpy_mode) {
        return interleave_numpy_arrays(fields_to_pick, rows_to_pick, inner_arrs,
                                       pool, mm);
    } else {
        throw std::runtime_error("lateral_flatten_struct: invalid state");
    }
}

std::unique_ptr<table_info> lateral_flatten_array(
    const std::unique_ptr<table_info> &in_table, int64_t *n_rows,
    bool output_seq, bool output_key, bool output_path, bool output_index,
    bool output_value, bool output_this, bodo::IBufferPool *const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    // Create the table where we will place all of the out columns in.
    std::unique_ptr<table_info> out_table = std::make_unique<table_info>();

    // Find the column to be exploded and calculate the number of times each
    // row should be duplicated.
    std::shared_ptr<array_info> explode_arr = in_table->columns[0];
    size_t n_inner_arrs = explode_arr->length;
    offset_t *offset_buffer =
        (offset_t *)(explode_arr->buffers[0]->mutable_data() +
                     explode_arr->offset);
    size_t exploded_size = offset_buffer[n_inner_arrs] - offset_buffer[0];
    std::vector<int64_t> rows_to_copy(exploded_size);
    size_t write_idx = 0;
    for (size_t i = 0; i < n_inner_arrs; i++) {
        size_t n_rows = offset_buffer[i + 1] - offset_buffer[i];
        for (size_t j = 0; j < n_rows; j++) {
            rows_to_copy[write_idx] = i;
            write_idx++;
        }
    }

    // If we need to output the index, then create a column with the
    // indices within each inner array.
    if (output_index) {
        std::shared_ptr<array_info> idx_arr =
            alloc_numpy(exploded_size, Bodo_CTypes::INT64);
        offset_t *offset_buffer =
            (offset_t *)(explode_arr->buffers[0]->mutable_data() +
                         explode_arr->offset);
        for (size_t i = 0; i < explode_arr->length; i++) {
            offset_t start_offset = offset_buffer[i];
            offset_t end_offset = offset_buffer[i + 1];
            for (offset_t j = start_offset; j < end_offset; j++) {
                getv<int64_t>(idx_arr, j) = j - start_offset;
            }
        }
        out_table->columns.push_back(idx_arr);
    }

    // If we need to output the array, then append the inner array
    // from the exploded column
    if (output_value) {
        out_table->columns.push_back(explode_arr->child_arrays[0]);
    }

    // If we need to output the 'this' column, then repeat the replication
    // procedure on the input column
    if (output_this) {
        std::shared_ptr<array_info> old_col = in_table->columns[0];
        std::shared_ptr<array_info> new_col =
            RetrieveArray_SingleColumn(old_col, rows_to_copy, false, pool, mm);
        out_table->columns.push_back(new_col);
    }

    // For each column in the table (except the one to be exploded) create
    // a copy with the rows duplicated the specified number of times and
    // append it to the output table.
    for (size_t i = 1; i < in_table->columns.size(); i++) {
        std::shared_ptr<array_info> old_col = in_table->columns[i];
        std::shared_ptr<array_info> new_col =
            RetrieveArray_SingleColumn(old_col, rows_to_copy, false, pool, mm);
        out_table->columns.push_back(new_col);
    }

    *n_rows = exploded_size;
    return out_table;
}

std::unique_ptr<table_info> lateral_flatten_map(
    const std::unique_ptr<table_info> &in_table, int64_t *n_rows,
    bool output_seq, bool output_key, bool output_path, bool output_index,
    bool output_value, bool output_this, bodo::IBufferPool *const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    // Create the table where we will place all of the out columns in.
    std::unique_ptr<table_info> out_table = std::make_unique<table_info>();

    // Find the column to be exploded and calculate the number of times each
    // row should be duplicated.
    std::shared_ptr<array_info> explode_arr =
        in_table->columns[0]->child_arrays[0];
    size_t n_inner_arrs = explode_arr->length;
    offset_t *offset_buffer =
        (offset_t *)(explode_arr->buffers[0]->mutable_data() +
                     explode_arr->offset);
    size_t exploded_size = offset_buffer[n_inner_arrs] - offset_buffer[0];
    std::vector<int64_t> rows_to_copy(exploded_size);
    size_t write_idx = 0;
    for (size_t i = 0; i < n_inner_arrs; i++) {
        size_t n_rows = offset_buffer[i + 1] - offset_buffer[i];
        for (size_t j = 0; j < n_rows; j++) {
            rows_to_copy[write_idx] = i;
            write_idx++;
        }
    }

    // If we need to output the key, then add the first column
    // of the inner struct array.
    if (output_key) {
        std::shared_ptr<array_info> key_arr =
            explode_arr->child_arrays[0]->child_arrays[0];
        out_table->columns.push_back(key_arr);
    }

    // If we need to output the key, then add the second column
    // of the inner struct array.
    if (output_value) {
        std::shared_ptr<array_info> value_arr =
            explode_arr->child_arrays[0]->child_arrays[1];
        out_table->columns.push_back(value_arr);
    }

    // If we need to output the 'this' column, then repeat the replication
    // procedure on the input column
    if (output_this) {
        std::shared_ptr<array_info> old_col =
            in_table->columns[0]->child_arrays[0];
        std::shared_ptr<array_info> new_col =
            RetrieveArray_SingleColumn(old_col, rows_to_copy, false, pool, mm);
        out_table->columns.push_back(alloc_map(new_col->length, new_col));
    }

    // For each column in the table (except the one to be exploded) create
    // a copy with the rows duplicated the specified number of times and
    // append it to the output table.
    for (size_t i = 1; i < in_table->columns.size(); i++) {
        std::shared_ptr<array_info> old_col = in_table->columns[i];
        std::shared_ptr<array_info> new_col =
            RetrieveArray_SingleColumn(old_col, rows_to_copy, false, pool, mm);
        out_table->columns.push_back(new_col);
    }

    *n_rows = exploded_size;
    return out_table;
}

std::unique_ptr<table_info> lateral_flatten_struct(
    const std::unique_ptr<table_info> &in_table, int64_t *n_rows,
    bool output_seq, bool output_key, bool output_path, bool output_index,
    bool output_value, bool output_this, bodo::IBufferPool *const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    // Create the table where we will place all of the out columns in.
    std::unique_ptr<table_info> out_table = std::make_unique<table_info>();

    // Find the column to be exploded and calculate the number of times each
    // row should be duplicated.
    std::shared_ptr<array_info> explode_arr = in_table->columns[0];
    size_t n_structs = explode_arr->length;
    size_t n_fields = (explode_arr->child_arrays).size();
    size_t exploded_size = 0;
    std::vector<int64_t> rows_to_copy;
    for (size_t i = 0; i < n_structs; i++) {
        if (!(explode_arr->get_null_bit(i)))
            continue;
        for (size_t j = 0; j < n_fields; j++) {
            rows_to_copy.push_back(i);
            exploded_size++;
        }
    }

    // If we need to output the key, then add the first column
    // of the inner struct array.
    if (output_key) {
        bodo::vector<uint8_t> null_vector((n_fields + 7) >> 3, 0, pool);
        bodo::vector<std::string> string_vector;
        for (size_t i = 0; i < n_fields; i++) {
            SetBitTo(null_vector.data(), i, 1);
            string_vector.push_back(explode_arr->field_names[i]);
        }

        std::shared_ptr<array_info> dict_arr = create_string_array(
            Bodo_CTypes::STRING, null_vector, string_vector, -1, pool, mm);
        std::shared_ptr<array_info> idx_arr = alloc_nullable_array_no_nulls(
            exploded_size, Bodo_CTypes::INT32, 0, pool, mm);
        for (size_t i = 0; i < n_structs; i++) {
            for (size_t j = 0; j < n_fields; j++) {
                size_t idx = (n_fields * i) + j;
                getv<int32_t>(idx_arr, idx) = j;
            }
        }

        std::shared_ptr<array_info> key_arr =
            create_dict_string_array(dict_arr, idx_arr);
        out_table->columns.push_back(key_arr);
    }

    // If we need to output the key, then add the second column
    // of the inner struct array.
    if (output_value) {
        std::vector<size_t> fields_to_pick;
        std::vector<size_t> rows_to_pick;
        for (size_t row = 0; row < n_structs; row++) {
            if (!explode_arr->get_null_bit(row))
                continue;
            for (size_t field = 0; field < n_fields; field++) {
                fields_to_pick.push_back(field);
                rows_to_pick.push_back(row);
            }
        }
        std::shared_ptr<array_info> val_arr = interleave_struct_columns(
            fields_to_pick, rows_to_pick, explode_arr->child_arrays, pool, mm);
        out_table->columns.push_back(val_arr);
    }

    // If we need to output the 'this' column, then repeat the replication
    // procedure on the input column
    if (output_this) {
        std::shared_ptr<array_info> old_col = in_table->columns[0];
        std::shared_ptr<array_info> new_col =
            RetrieveArray_SingleColumn(old_col, rows_to_copy, false, pool, mm);
        out_table->columns.push_back(new_col);
    }

    // For each column in the table (except the one to be exploded) create
    // a copy with the rows duplicated the specified number of times and
    // append it to the output table.
    for (size_t i = 1; i < in_table->columns.size(); i++) {
        std::shared_ptr<array_info> old_col = in_table->columns[i];
        std::shared_ptr<array_info> new_col =
            RetrieveArray_SingleColumn(old_col, rows_to_copy, false, pool, mm);
        out_table->columns.push_back(new_col);
    }

    *n_rows = exploded_size;
    return out_table;
}

// Python entrypoint for lateral_flatten
table_info *lateral_flatten_py_entrypt(table_info *in_table, int64_t *n_rows,
                                       bool output_seq, bool output_key,
                                       bool output_path, bool output_index,
                                       bool output_value, bool output_this,
                                       bool json_mode) {
    try {
        std::unique_ptr<table_info> tab(in_table);
        std::unique_ptr<table_info> result;
        if (json_mode) {
            switch (tab->columns[0]->arr_type) {
                case bodo_array_type::MAP: {
                    result = lateral_flatten_map(
                        tab, n_rows, output_seq, output_key, output_path,
                        output_index, output_value, output_this);
                    break;
                }
                case bodo_array_type::STRUCT: {
                    result = lateral_flatten_struct(
                        tab, n_rows, output_seq, output_key, output_path,
                        output_index, output_value, output_this);
                    break;
                }
                default: {
                    throw std::runtime_error(
                        "lateral flatten with json mode requires a map array "
                        "or struct array as an input");
                }
            }
        } else {
            if (tab->columns[0]->arr_type != bodo_array_type::ARRAY_ITEM) {
                throw std::runtime_error(
                    "lateral flatten with array mode requires an array item "
                    "array as an input");
            }
            result = lateral_flatten_array(tab, n_rows, output_seq, output_key,
                                           output_path, output_index,
                                           output_value, output_this);
        }
        table_info *raw_result = result.release();
        return raw_result;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

PyMODINIT_FUNC PyInit_lateral(void) {
    PyObject *m;
    MOD_DEF(m, "lateral", "No docs", NULL);
    if (m == NULL) {
        return NULL;
    }

    bodo_common_init();

    SetAttrStringFromVoidPtr(m, lateral_flatten_py_entrypt);

    return m;
}
