#include "../_array_utils.h"
#include "../_bodo_common.h"

/**
 * This file contains groupby update functions that still correspond to the
 * activity on a single row but are too complicated to be inlined. In general
 * this file will consists of implementations that require multiple columns of
 * variable type and therefore cannot be templated.
 */

bool idx_compare_column(const std::shared_ptr<array_info>& out_arr,
                        int64_t grp_num,
                        const std::shared_ptr<array_info>& in_arr,
                        int64_t in_idx, bool asc, bool na_pos) {
    if (in_arr->arr_type != bodo_array_type::STRING &&
        in_arr->arr_type != bodo_array_type::DICT &&
        in_arr->arr_type != bodo_array_type::NULLABLE_INT_BOOL &&
        in_arr->arr_type != bodo_array_type::NUMPY) {
        throw std::runtime_error(
            "Unsupported array type for idx_compare_column: " +
            GetArrType_as_string(in_arr->arr_type));
    }
    int64_t& curr_idx = getv<int64_t>(out_arr, grp_num);
    // Check for nulls.
    if (in_arr->arr_type == bodo_array_type::STRING ||
        in_arr->arr_type == bodo_array_type::DICT ||
        in_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        bool bit1 = in_arr->get_null_bit(curr_idx);
        bool bit2 = in_arr->get_null_bit(in_idx);
        if (!bit1 && !bit2) {
            // Both values are null. We have a tie.
            return false;
        } else if (!bit1) {
            // Old value is null. We now look at na_pos.
            if (na_pos) {
                // NAs last
                curr_idx = in_idx;
            }
            return true;
        } else if (!bit2) {
            // New value is null. We now look at na_pos.
            if (!na_pos) {
                // NAs first
                curr_idx = in_idx;
            }
            return true;
        }
    }
    if (in_arr->arr_type == bodo_array_type::STRING) {
        char* data = in_arr->data1<bodo_array_type::STRING>();
        offset_t* offsets = (offset_t*)in_arr->data2<bodo_array_type::STRING>();
        // Load the old data.
        offset_t start_offset_old = offsets[curr_idx];
        offset_t end_offset_old = offsets[curr_idx + 1];
        offset_t len = end_offset_old - start_offset_old;
        std::string_view old_str(&data[start_offset_old], len);
        // Load the new data
        offset_t start_offset_new = offsets[in_idx];
        offset_t end_offset_new = offsets[in_idx + 1];
        offset_t len_org = end_offset_new - start_offset_new;
        std::string_view new_str(&data[start_offset_new], len_org);
        int cmp = old_str.compare(new_str);
        if (cmp == 0) {
            // Strings are equal
            return false;
        } else if (cmp > 0) {
            // asc = True means idxmin
            if (asc) {
                curr_idx = in_idx;
            }
            return true;
        } else {
            // asc = False means idxmax
            if (!asc) {
                curr_idx = in_idx;
            }
            return true;
        }
    } else if (in_arr->arr_type == bodo_array_type::DICT) {
        int32_t old_index =
            getv<dict_indices_t, bodo_array_type::NULLABLE_INT_BOOL>(
                in_arr->child_arrays[1], curr_idx);
        int32_t new_index =
            getv<dict_indices_t, bodo_array_type::NULLABLE_INT_BOOL>(
                in_arr->child_arrays[1], in_idx);
        // Fast path via index comparison.
        if (old_index == new_index) {
            return false;
        }
        // If the index is sorted we can just compare the indices.
        if (in_arr->child_arrays[0]->is_locally_sorted) {
            if (old_index > new_index) {
                // asc = True means idxmin
                if (asc) {
                    curr_idx = in_idx;
                }
                return true;
            } else {
                // asc = False means idxmax
                if (!asc) {
                    curr_idx = in_idx;
                }
                return true;
            }
        } else {
            // We need to load the actual data and compare it.
            char* data =
                in_arr->child_arrays[0]->data1<bodo_array_type::STRING>();
            offset_t* offsets = (offset_t*)in_arr->child_arrays[0]
                                    ->data2<bodo_array_type::STRING>();
            // Load the old data.
            offset_t start_offset_old = offsets[old_index];
            offset_t end_offset_old = offsets[old_index + 1];
            offset_t len = end_offset_old - start_offset_old;
            std::string_view old_str(&data[start_offset_old], len);
            // Load the new data
            offset_t start_offset_new = offsets[new_index];
            offset_t end_offset_new = offsets[new_index + 1];
            offset_t len_org = end_offset_new - start_offset_new;
            std::string_view new_str(&data[start_offset_new], len_org);
            int cmp = old_str.compare(new_str);
            if (cmp == 0) {
                // Strings are equal
                return false;
            } else if (cmp > 0) {
                // asc = True means idxmin
                if (asc) {
                    curr_idx = in_idx;
                }
                return true;
            } else {
                // asc = False means idxmax
                if (!asc) {
                    curr_idx = in_idx;
                }
                return true;
            }
        }
    } else {
        // Define a macro for a reused standard comparison across several
        // types. Note that asc = True means idxmin and asc = False means
        // idxmax.
#ifndef STANDARD_EQUALITY_CHECK
#define STANDARD_EQUALITY_CHECK         \
    if (new_value == old_value) {       \
        return false;                   \
    } else if (old_value > new_value) { \
        if (asc) {                      \
            curr_idx = in_idx;          \
        }                               \
        return true;                    \
    } else {                            \
        if (!asc) {                     \
            curr_idx = in_idx;          \
        }                               \
        return true;                    \
    }
#endif

        switch (in_arr->dtype) {
            // TODO: Add separate bool handling for nullable booleans
            // once that PR is ready.
            case Bodo_CTypes::_BOOL: {
                if (in_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
                    // Nullable boolean arrays store 1 bit per boolean, so we
                    // need a separate path to get the values.
                    bool old_value = GetBit(
                        (uint8_t*)
                            in_arr->data1<bodo_array_type::NULLABLE_INT_BOOL>(),
                        curr_idx);
                    bool new_value = GetBit(
                        (uint8_t*)
                            in_arr->data1<bodo_array_type::NULLABLE_INT_BOOL>(),
                        in_idx);
                    STANDARD_EQUALITY_CHECK
                } else {
                    bool old_value = getv<bool>(in_arr, curr_idx);
                    bool new_value = getv<bool>(in_arr, in_idx);
                    STANDARD_EQUALITY_CHECK
                }
            }
            case Bodo_CTypes::INT8: {
                int8_t old_value = getv<int8_t>(in_arr, curr_idx);
                int8_t new_value = getv<int8_t>(in_arr, in_idx);
                STANDARD_EQUALITY_CHECK
            }
            case Bodo_CTypes::UINT8: {
                uint8_t old_value = getv<uint8_t>(in_arr, curr_idx);
                uint8_t new_value = getv<uint8_t>(in_arr, in_idx);
                STANDARD_EQUALITY_CHECK
            }
            case Bodo_CTypes::INT16: {
                int16_t old_value = getv<int16_t>(in_arr, curr_idx);
                int16_t new_value = getv<int16_t>(in_arr, in_idx);
                STANDARD_EQUALITY_CHECK
            }
            case Bodo_CTypes::UINT16: {
                uint16_t old_value = getv<uint16_t>(in_arr, curr_idx);
                uint16_t new_value = getv<uint16_t>(in_arr, in_idx);
                STANDARD_EQUALITY_CHECK
            }
            case Bodo_CTypes::DATE:
            case Bodo_CTypes::INT32: {
                int32_t old_value = getv<int32_t>(in_arr, curr_idx);
                int32_t new_value = getv<int32_t>(in_arr, in_idx);
                STANDARD_EQUALITY_CHECK
            }
            case Bodo_CTypes::UINT32: {
                uint32_t old_value = getv<uint32_t>(in_arr, curr_idx);
                uint32_t new_value = getv<uint32_t>(in_arr, in_idx);
                STANDARD_EQUALITY_CHECK
            }
            case Bodo_CTypes::INT64:
            case Bodo_CTypes::TIME: {
                int64_t old_value = getv<int64_t>(in_arr, curr_idx);
                int64_t new_value = getv<int64_t>(in_arr, in_idx);
                STANDARD_EQUALITY_CHECK
            }
            case Bodo_CTypes::UINT64: {
                uint64_t old_value = getv<uint64_t>(in_arr, curr_idx);
                uint64_t new_value = getv<uint64_t>(in_arr, in_idx);
                STANDARD_EQUALITY_CHECK
            }
            // TODO: Fuse with int64 once it uses the nullable array
            // type.
            case Bodo_CTypes::DATETIME:
            case Bodo_CTypes::TIMEDELTA: {
                int64_t old_value = getv<int64_t>(in_arr, curr_idx);
                int64_t new_value = getv<int64_t>(in_arr, in_idx);
                bool isna1;
                bool isna2;
                // Note: These are the same but this should be more resilient to
                // refactoring.
                if (in_arr->dtype == Bodo_CTypes::DATETIME) {
                    isna1 = isnan_alltype<int64_t, Bodo_CTypes::DATETIME>(
                        old_value);
                    isna2 = isnan_alltype<int64_t, Bodo_CTypes::DATETIME>(
                        new_value);
                } else {
                    isna1 = isnan_alltype<int64_t, Bodo_CTypes::TIMEDELTA>(
                        old_value);
                    isna2 = isnan_alltype<int64_t, Bodo_CTypes::TIMEDELTA>(
                        new_value);
                }
                // NA handling
                if (isna1 && isna2) {
                    return false;
                } else if (isna1) {
                    if (na_pos) {
                        // Old value is NA and we want
                        // NAs last
                        curr_idx = in_idx;
                    }
                    return true;
                } else if (isna2) {
                    if (!na_pos) {
                        // New value is NA and we want
                        // NAs first
                        curr_idx = in_idx;
                    }
                    return true;
                }
                // Compare values
                STANDARD_EQUALITY_CHECK
            }
            case Bodo_CTypes::FLOAT32: {
                // Note in SQL NaN is not NA and is instead always the
                // largest value.
                // https://docs.snowflake.com/en/sql-reference/data-types-numeric#special-values
                float old_value = getv<float>(in_arr, curr_idx);
                float new_value = getv<float>(in_arr, in_idx);
                // NaN handling
                bool is_nan1 = isnan(old_value);
                bool is_nan2 = isnan(new_value);
                if (is_nan1 && is_nan2) {
                    return false;
                } else if (is_nan1) {
                    // asc = True means idxmin
                    if (asc) {
                        curr_idx = in_idx;
                    }
                    return true;
                } else if (is_nan2) {
                    // asc = False means idxmax
                    if (!asc) {
                        curr_idx = in_idx;
                    }
                    return true;
                }
                // Other values
                STANDARD_EQUALITY_CHECK
            }
            case Bodo_CTypes::FLOAT64: {
                // Note in SQL NaN is not NA and is instead always the
                // largest value.
                // https://docs.snowflake.com/en/sql-reference/data-types-numeric#special-values
                double old_value = getv<double>(in_arr, curr_idx);
                double new_value = getv<double>(in_arr, in_idx);
                // NaN handling
                bool is_nan1 = isnan(old_value);
                bool is_nan2 = isnan(new_value);
                if (is_nan1 && is_nan2) {
                    return false;
                } else if (is_nan1) {
                    // asc = True means idxmin
                    if (asc) {
                        curr_idx = in_idx;
                    }
                    return true;
                } else if (is_nan2) {
                    // asc = False means idxmax
                    if (!asc) {
                        curr_idx = in_idx;
                    }
                    return true;
                }
                // Other values
                STANDARD_EQUALITY_CHECK
            }
            case Bodo_CTypes::DECIMAL: {
                // Decimal can just compare the underlying integers
                // as the types must be the same.
                __int128 old_value = getv<__int128>(in_arr, curr_idx);
                __int128 new_value = getv<__int128>(in_arr, in_idx);
                STANDARD_EQUALITY_CHECK
            }
            default:
                throw std::runtime_error(
                    "Invalid dtype for idx_compare_column: " +
                    GetDtype_as_string(in_arr->dtype));
        }
    }
#undef STANDARD_EQUALITY_CHECK
}
