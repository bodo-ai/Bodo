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
    const std::vector<int64_t> &fields_to_pick,
    const std::vector<int64_t> &rows_to_pick,
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
    const std::vector<int64_t> &fields_to_pick,
    const std::vector<int64_t> &rows_to_pick,
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
        int64_t field_to_pick = fields_to_pick[row];
        int64_t row_to_pick = rows_to_pick[row];
        if (row_to_pick != -1 &&
            // TODO XXX Can this get_null_bit be templated?
            inner_arrs[field_to_pick]->get_null_bit(row_to_pick)) {
            SetBitTo(null_vector.data(), row, true);
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
            SetBitTo(null_vector.data(), row, false);
            string_vector.emplace_back("");
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
    const std::vector<int64_t> &fields_to_pick,
    const std::vector<int64_t> &rows_to_pick,
    const std::vector<std::shared_ptr<array_info>> &inner_arrs,
    bodo::IBufferPool *const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    // Extrapolate the fields to pick and rows to pick to the child
    // arrays of the columns.
    size_t n_fields = inner_arrs.size();
    size_t rows_kept = fields_to_pick.size();
    std::vector<int64_t> sub_fields_to_pick;
    std::vector<int64_t> sub_rows_to_pick;
    std::vector<size_t> array_sizes;
    std::vector<bool> is_null;
    for (size_t row = 0; row < rows_kept; row++) {
        int64_t field_to_pick = fields_to_pick[row];
        int64_t row_to_pick = rows_to_pick[row];
        if (row_to_pick == -1) {
            array_sizes.push_back(0);
            is_null.push_back(true);
            continue;
        }
        offset_t *offset_buffer =
            (offset_t *)(inner_arrs[field_to_pick]->buffers[0]->mutable_data() +
                         inner_arrs[field_to_pick]->offset);
        size_t start_idx = offset_buffer[row_to_pick];
        size_t end_idx = offset_buffer[row_to_pick + 1];
        for (size_t idx = start_idx; idx < end_idx; idx++) {
            sub_fields_to_pick.push_back(static_cast<int64_t>(field_to_pick));
            sub_rows_to_pick.push_back(static_cast<int64_t>(idx));
        }
        array_sizes.push_back(end_idx - start_idx);
        // TODO XXX Can this get_null_bit be templated?
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
            result->set_null_bit<bodo_array_type::ARRAY_ITEM>(row, false);
        }
    }
    return result;
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
std::shared_ptr<array_info> interleave_numeric_arrays(
    const std::vector<int64_t> &fields_to_pick,
    const std::vector<int64_t> &rows_to_pick,
    const std::vector<std::shared_ptr<array_info>> &inner_arrs,
    bodo::IBufferPool *const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    // Find the numerical type that all the values will be casted to,
    // and allocate a numpy array with the correct size.
    size_t rows_kept = fields_to_pick.size();
    Bodo_CTypes::CTypeEnum dtype = get_common_numeric_dtype(inner_arrs);
    std::shared_ptr out_arr = alloc_nullable_array_no_nulls(rows_kept, dtype);

    // For each row, iterate across all of the fields and write them
    // to the output array at the corresponding index.
    for (size_t row = 0; row < rows_kept; row++) {
        int64_t field_to_pick = fields_to_pick[row];
        int64_t row_to_pick = rows_to_pick[row];
        // TODO XXX Can this get_null_bit be templated?
        if (row_to_pick != -1 &&
            ((inner_arrs[field_to_pick]->arr_type == bodo_array_type::NUMPY) ||
             (inner_arrs[field_to_pick]->get_null_bit(row_to_pick)))) {
            lateral_flatten_copy_numeric_value(
                out_arr, row, inner_arrs[field_to_pick], row_to_pick);
        } else {
            out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(row,
                                                                      false);
        }
    }
    return out_arr;
}

std::shared_ptr<array_info> interleave_struct_columns(
    const std::vector<int64_t> &fields_to_pick,
    const std::vector<int64_t> &rows_to_pick,
    const std::vector<std::shared_ptr<array_info>> &inner_arrs,
    bodo::IBufferPool *const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    /* Identify which type of array combination procedure will
     * be required to interleave the child arrays:
     * - string_mode: combine string arrays (possibly with dictionary-encoded
     * arrays)
     * - dict_mode: combine only dictionary-encoded arrays
     * - array_mode: combine only array item arrays
     * - numeric_mode: combine nullable arrays (possibly with numpy arrays)
     * - none of the above: implies combining only numpy arrays
     * All other combinations are currently not allowed.
     */
    bool string_mode = false;
    bool array_mode = false;
    bool numeric_mode = false;
    size_t n_fields = inner_arrs.size();
    for (size_t i = 0; i < n_fields; i++) {
        switch (inner_arrs[i]->arr_type) {
            case bodo_array_type::DICT:
            case bodo_array_type::STRING: {
                string_mode = true;
                if (array_mode || numeric_mode) {
                    throw std::runtime_error(
                        "lateral_flatten_struct: invalid struct child arrays");
                }
                break;
            }
            case bodo_array_type::NUMPY:
            case bodo_array_type::NULLABLE_INT_BOOL: {
                numeric_mode = true;
                if (string_mode || array_mode) {
                    throw std::runtime_error(
                        "lateral_flatten_struct: invalid struct child arrays");
                }
                break;
            }
            case bodo_array_type::ARRAY_ITEM: {
                array_mode = true;
                if (string_mode || numeric_mode) {
                    throw std::runtime_error(
                        "lateral_flatten_struct: invalid struct child arrays");
                }
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
    } else if (numeric_mode) {
        return interleave_numeric_arrays(fields_to_pick, rows_to_pick,
                                         inner_arrs, pool, mm);
    } else {
        throw std::runtime_error("lateral_flatten_struct: invalid state");
    }
}

/*
 * Helper utility to transform one of the output columns (e.g. the index
 * or key column) when the OUTER keyword is set to true by injecting
 * nulls at every row where the original array/map was null/empty.
 *
 * @param[in] column_to_transform the column to inject the nulls into.
 * @param[in] exploded_rows_to_copy the vector storing the mapping of
 * rows in the new column to the row from the original column to copy,
 * or -1 if a null is to be inserted. The first time this function is called,
 * this will be empty and should be populated by this function so later calls
 * can re-use it.
 * @param[in] offset_buffer the buffer used to determine which arrays/maps
 * don't contain any elements and therefore need a null injected.
 * @param[in] outer_elems the number of arrays/maps represented by
 * offset_buffer.
 * @param[in] exploded_size the number of rows in the final answer.
 */
std::shared_ptr<array_info> outer_transform_flatten_cols(
    std::shared_ptr<array_info> column_to_transform,
    std::vector<int64_t> &exploded_rows_to_copy, offset_t *offset_buffer,
    size_t outer_elems, bodo::IBufferPool *const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    // If this is the first time calling the function with this
    // vector, populate the vector with a mapping back to the rows
    // of the regular column, with -1 inserted for each empty
    // array/map.
    if (exploded_rows_to_copy.size() == 0) {
        size_t inner_read_idx = 0;
        for (size_t i = 0; i < outer_elems; i++) {
            size_t n_rows = offset_buffer[i + 1] - offset_buffer[i];
            if (n_rows == 0) {
                exploded_rows_to_copy.push_back(-1);
            } else {
                for (size_t idx = 0; idx < n_rows; idx++) {
                    exploded_rows_to_copy.push_back(inner_read_idx);
                    inner_read_idx++;
                }
            }
        }
    }
    // Then use RetrieveArray to get the desired transformation
    return RetrieveArray_SingleColumn(column_to_transform,
                                      exploded_rows_to_copy, false, pool, mm);
}

std::unique_ptr<table_info> lateral_flatten_array(
    const std::unique_ptr<table_info> &in_table, int64_t *n_rows,
    bool output_seq, bool output_key, bool output_path, bool output_index,
    bool output_value, bool output_this, bool is_outer,
    bodo::IBufferPool *const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    // Create the table where we will place all of the out columns in.
    std::unique_ptr<table_info> out_table = std::make_unique<table_info>();

    // Find the column to be exploded and calculate the number of times each
    // row should be duplicated.
    std::shared_ptr<array_info> explode_arr = in_table->columns[0];
    size_t n_inner_arrs = explode_arr->length;
    offset_t *offset_buffer =
        (offset_t *)(explode_arr->buffers[0]->mutable_data() +
                     explode_arr->offset);

    size_t exploded_size = 0;
    std::vector<int64_t> rows_to_copy;
    // Calculate the normal size and times each row is copied based
    // on the offset buffers which indicate how many elements are in
    // each row of the explode array.
    for (size_t i = 0; i < n_inner_arrs; i++) {
        size_t n_rows = offset_buffer[i + 1] - offset_buffer[i];
        // If in outer mode, each row gets copied once even if the
        // corresponding row of the explode column is null or empty.
        if (is_outer && n_rows == 0) {
            n_rows = 1;
        }
        exploded_size += n_rows;
        for (size_t j = 0; j < n_rows; j++) {
            rows_to_copy.push_back(i);
        }
    }

    std::vector<int64_t> exploded_rows_to_copy;

    // If we need to output the index, then create a column with the
    // indices within each inner array.
    if (output_index) {
        std::shared_ptr<array_info> idx_arr =
            alloc_nullable_array_no_nulls(exploded_size, Bodo_CTypes::INT64);
        for (size_t i = 0; i < explode_arr->length; i++) {
            offset_t start_offset = offset_buffer[i];
            offset_t end_offset = offset_buffer[i + 1];
            for (offset_t j = start_offset; j < end_offset; j++) {
                getv<int64_t, bodo_array_type::NULLABLE_INT_BOOL>(idx_arr, j) =
                    j - start_offset;
            }
        }
        if (is_outer) {
            idx_arr = outer_transform_flatten_cols(
                idx_arr, exploded_rows_to_copy, offset_buffer, n_inner_arrs,
                pool, mm);
        }
        out_table->columns.push_back(idx_arr);
    }

    // If we need to output the array, then append the inner array
    // from the exploded column
    if (output_value) {
        std::shared_ptr<array_info> value_arr = explode_arr->child_arrays[0];
        if (is_outer) {
            value_arr = outer_transform_flatten_cols(
                value_arr, exploded_rows_to_copy, offset_buffer, n_inner_arrs,
                pool, mm);
        }
        out_table->columns.push_back(value_arr);
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
    bool output_value, bool output_this, bool is_outer,
    bodo::IBufferPool *const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    // Create the table where we will place all of the out columns in.
    std::unique_ptr<table_info> out_table = std::make_unique<table_info>();

    // Calculate the normal size and times each row is copied based
    // on the offset buffers which indicate how many elements are in
    // each row of the explode map.
    std::shared_ptr<array_info> explode_arr =
        in_table->columns[0]->child_arrays[0];
    size_t n_inner_arrs = explode_arr->length;
    offset_t *offset_buffer =
        (offset_t *)(explode_arr->buffers[0]->mutable_data() +
                     explode_arr->offset);
    size_t exploded_size = 0;
    std::vector<int64_t> rows_to_copy;
    for (size_t i = 0; i < n_inner_arrs; i++) {
        size_t n_rows = offset_buffer[i + 1] - offset_buffer[i];
        // If in outer mode, each row gets copied once even if the
        // corresponding row of the explode column is null or empty.
        if (is_outer && n_rows == 0) {
            n_rows = 1;
        }
        exploded_size += n_rows;
        for (size_t j = 0; j < n_rows; j++) {
            rows_to_copy.push_back(i);
        }
    }

    std::vector<int64_t> exploded_rows_to_copy;

    // If we need to output the key, then add the first column
    // of the inner struct array.
    if (output_key) {
        std::shared_ptr<array_info> key_arr =
            explode_arr->child_arrays[0]->child_arrays[0];
        if (is_outer) {
            key_arr = outer_transform_flatten_cols(
                key_arr, exploded_rows_to_copy, offset_buffer, n_inner_arrs,
                pool, mm);
        }
        out_table->columns.push_back(key_arr);
    }

    // If we need to output the key, then add the second column
    // of the inner struct array.
    if (output_value) {
        std::shared_ptr<array_info> value_arr =
            explode_arr->child_arrays[0]->child_arrays[1];
        if (is_outer) {
            value_arr = outer_transform_flatten_cols(
                value_arr, exploded_rows_to_copy, offset_buffer, n_inner_arrs,
                pool, mm);
        }
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
    bool output_value, bool output_this, bool is_outer,
    bodo::IBufferPool *const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    // Create the table where we will place all of the out columns in.
    std::unique_ptr<table_info> out_table = std::make_unique<table_info>();

    // Find the column to be exploded and calculate the number of times each
    // row should be duplicated.
    std::shared_ptr<array_info> explode_arr = in_table->columns[0];
    size_t n_structs = explode_arr->length;
    size_t n_fields = (explode_arr->child_arrays).size();
    size_t exploded_size = 0;
    std::vector<int64_t> rows_to_copy;
    const uint8_t *explode_arr_null_bitmask =
        (uint8_t *)explode_arr->null_bitmask();
    for (size_t i = 0; i < n_structs; i++) {
        if (!(GetBit(explode_arr_null_bitmask, i))) {
            // If in outer mode, add 1 dummy for any null rows.
            if (is_outer) {
                rows_to_copy.push_back(i);
                exploded_size++;
            }
            continue;
        }
        for (size_t j = 0; j < n_fields; j++) {
            rows_to_copy.push_back(i);
            exploded_size++;
        }
    }

    // If we need to output the key, then add a series of interleaved
    // copies of the field names.
    if (output_key) {
        bodo::vector<uint8_t> null_vector((n_fields + 7) >> 3, 0, pool);
        bodo::vector<std::string> string_vector;
        for (size_t i = 0; i < n_fields; i++) {
            SetBitTo(null_vector.data(), i, true);
            string_vector.push_back(explode_arr->field_names[i]);
        }

        std::shared_ptr<array_info> dict_arr = create_string_array(
            Bodo_CTypes::STRING, null_vector, string_vector, -1, pool, mm);
        std::shared_ptr<array_info> idx_arr = alloc_nullable_array_no_nulls(
            exploded_size, Bodo_CTypes::INT32, 0, pool, mm);
        size_t write_idx = 0;
        for (size_t i = 0; i < n_structs; i++) {
            if (!GetBit(explode_arr_null_bitmask, i)) {
                if (is_outer) {
                    idx_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(
                        write_idx, false);
                    write_idx++;
                }
                continue;
            }
            for (size_t j = 0; j < n_fields; j++) {
                getv<int32_t, bodo_array_type::NULLABLE_INT_BOOL>(
                    idx_arr, write_idx) = j;
                write_idx++;
            }
        }

        std::shared_ptr<array_info> key_arr =
            create_dict_string_array(dict_arr, idx_arr);
        out_table->columns.push_back(key_arr);
    }

    // If we need to output the value, then add the interleaved values
    // from the field columns.
    if (output_value) {
        std::vector<int64_t> fields_to_pick;
        std::vector<int64_t> rows_to_pick;
        for (size_t row = 0; row < n_structs; row++) {
            if (!GetBit(explode_arr_null_bitmask, row)) {
                if (is_outer) {
                    fields_to_pick.push_back(-1);
                    rows_to_pick.push_back(-1);
                }
                continue;
            }
            for (size_t field = 0; field < n_fields; field++) {
                fields_to_pick.push_back(static_cast<int64_t>(field));
                rows_to_pick.push_back(static_cast<int64_t>(row));
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
                                       bool json_mode, bool is_outer) {
    try {
        std::unique_ptr<table_info> tab(in_table);
        std::unique_ptr<table_info> result;
        if (json_mode) {
            switch (tab->columns[0]->arr_type) {
                case bodo_array_type::MAP: {
                    result = lateral_flatten_map(
                        tab, n_rows, output_seq, output_key, output_path,
                        output_index, output_value, output_this, is_outer);
                    break;
                }
                case bodo_array_type::STRUCT: {
                    result = lateral_flatten_struct(
                        tab, n_rows, output_seq, output_key, output_path,
                        output_index, output_value, output_this, is_outer);
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
                                           output_value, output_this, is_outer);
        }
        table_info *raw_result = result.release();
        return raw_result;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

#ifdef IS_TESTING
PyMODINIT_FUNC PyInit_test_cpp(void);
#endif

PyMODINIT_FUNC PyInit_lateral_cpp(void) {
    PyObject *m;
    MOD_DEF(m, "lateral_cpp", "No docs", nullptr);
    if (m == nullptr) {
        return nullptr;
    }

    bodo_common_init();

    SetAttrStringFromVoidPtr(m, lateral_flatten_py_entrypt);

#ifdef IS_TESTING
    SetAttrStringFromPyInit(m, test_cpp);
#endif

    return m;
}
