// Copyright (C) 2023 Bodo Inc. All rights reserved.
#include "_groupby_common.h"
#include <numeric>
#include "../_array_operations.h"
#include "../_array_utils.h"
#include "../_bodo_common.h"
#include "_groupby_ftypes.h"
#include "_groupby_update.h"

/**
 * This file contains helper functions that are shared by multiple possible
 * groupby paths.
 */

/**
 * Initialize an output column that will be used to store the result of an
 * aggregation function. Initialization depends on the function:
 * default: zero initialization
 * prod: 1
 * min: max dtype value, or quiet_NaN if float (so that result is nan if all
 * input values are nan) max: min dtype value, or quiet_NaN if float (so that
 * result is nan if all input values are nan)
 * Arrays of type STRING, STRUCT, and ARRAY_ITEM are always
 * initalized to all NULL
 *
 * @param output column
 * @param function identifier
 * @param use_sql_rules: If true, use SQL rules for null handling. If false, use
 * Pandas rules.
 */
void aggfunc_output_initialize_kernel(
    const std::shared_ptr<array_info>& out_col, int ftype, bool use_sql_rules,
    int64_t start_row = 0) {
    // Generate an error message for unsupported paths that includes the name
    // of the function and the dtype.
    std::string error_msg = std::string("unsupported aggregate function: ") +
                            get_name_for_Bodo_FTypes(ftype) +
                            std::string(" for column dtype: ") +
                            std::string(GetDtype_as_string(out_col->dtype));

    // TODO: Move to an arg from Python
    if (out_col->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        bool init_val;
        // TODO: Template on use_sql_rules
        if (use_sql_rules) {
            // All nullable outputs in SQL output null for empty groups
            // except for count, count_if and min_row_number_filter.
            init_val = (ftype == Bodo_FTypes::count) ||
                       (ftype == Bodo_FTypes::count_if) ||
                       (ftype == Bodo_FTypes::min_row_number_filter);
        } else {
            if (ftype == Bodo_FTypes::min || ftype == Bodo_FTypes::max ||
                ftype == Bodo_FTypes::first || ftype == Bodo_FTypes::last ||
                ftype == Bodo_FTypes::boolor_agg ||
                ftype == Bodo_FTypes::booland_agg ||
                ftype == Bodo_FTypes::boolxor_agg ||
                ftype == Bodo_FTypes::bitor_agg ||
                ftype == Bodo_FTypes::bitand_agg ||
                ftype == Bodo_FTypes::bitxor_agg ||
                ftype == Bodo_FTypes::mean || ftype == Bodo_FTypes::var ||
                ftype == Bodo_FTypes::var_pop ||
                ftype == Bodo_FTypes::std_pop ||
                ftype == Bodo_FTypes::kurtosis || ftype == Bodo_FTypes::skew ||
                ftype == Bodo_FTypes::std || ftype == Bodo_FTypes::median ||
                ftype == Bodo_FTypes::mode) {
                // if input is all nulls, max, min, first, last, kurtosis, skew,
                // boolor_agg, boolxor_agg, booland_agg, or bitor_agg, the
                // output will be null. We null initialize median, mean, var,
                // and std as well since we always output a nullable float at
                // this time.
                init_val = false;
            } else {
                init_val = true;
            }
        }
        InitializeBitMask(
            (uint8_t*)
                out_col->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>(),
            out_col->length, init_val, start_row);
    }

    if (out_col->arr_type == bodo_array_type::STRING ||
        out_col->arr_type == bodo_array_type::STRUCT ||
        out_col->arr_type == bodo_array_type::ARRAY_ITEM) {
        InitializeBitMask((uint8_t*)out_col->null_bitmask(), out_col->length,
                          false, start_row);
    }

    if (out_col->arr_type == bodo_array_type::MAP) {
        InitializeBitMask((uint8_t*)out_col->child_arrays[0]->null_bitmask(),
                          out_col->length, false, start_row);
    }

    if (out_col->arr_type == bodo_array_type::CATEGORICAL) {
        if (ftype == Bodo_FTypes::min || ftype == Bodo_FTypes::max ||
            ftype == Bodo_FTypes::first || ftype == Bodo_FTypes::last) {
            int init_val = -1;
            // if input is all nulls, max, first and last output will be -1
            // min will be num of categories.
            if (ftype == Bodo_FTypes::min) {
                init_val = out_col->num_categories;
            }
            switch (out_col->dtype) {
                case Bodo_CTypes::INT8:
                    std::fill(
                        (int8_t*)
                                out_col->data1<bodo_array_type::CATEGORICAL>() +
                            start_row,
                        (int8_t*)
                                out_col->data1<bodo_array_type::CATEGORICAL>() +
                            out_col->length,
                        init_val);
                    return;
                case Bodo_CTypes::INT16:
                    std::fill(
                        (int16_t*)
                                out_col->data1<bodo_array_type::CATEGORICAL>() +
                            start_row,
                        (int16_t*)
                                out_col->data1<bodo_array_type::CATEGORICAL>() +
                            out_col->length,
                        init_val);
                    return;
                case Bodo_CTypes::INT32:
                    std::fill(
                        (int32_t*)
                                out_col->data1<bodo_array_type::CATEGORICAL>() +
                            start_row,
                        (int32_t*)
                                out_col->data1<bodo_array_type::CATEGORICAL>() +
                            out_col->length,
                        init_val);
                    return;
                case Bodo_CTypes::INT64:
                    std::fill(
                        (int64_t*)
                                out_col->data1<bodo_array_type::CATEGORICAL>() +
                            start_row,
                        (int64_t*)
                                out_col->data1<bodo_array_type::CATEGORICAL>() +
                            out_col->length,
                        init_val);
                    return;
                default:
                    Bodo_PyErr_SetString(PyExc_RuntimeError, error_msg.c_str());
                    return;
            }
        }
    }
    switch (ftype) {
        case Bodo_FTypes::booland_agg: {
            InitializeBitMask((uint8_t*)out_col->data1(), out_col->length, true,
                              start_row);
            return;
        }
        case Bodo_FTypes::prod:
            switch (out_col->dtype) {
                case Bodo_CTypes::_BOOL:
                    if (out_col->arr_type ==
                        bodo_array_type::NULLABLE_INT_BOOL) {
                        // Nullable booleans store 1 bit per value
                        InitializeBitMask(
                            (uint8_t*)out_col
                                ->data1<bodo_array_type::NULLABLE_INT_BOOL>(),
                            out_col->length, true, start_row);
                    } else {
                        std::fill((bool*)out_col->data1() + start_row,
                                  (bool*)out_col->data1() + out_col->length,
                                  true);
                    }
                    return;
                case Bodo_CTypes::INT8:
                    std::fill((int8_t*)out_col->data1() + start_row,
                              (int8_t*)out_col->data1() + out_col->length, 1);
                    return;
                case Bodo_CTypes::UINT8:
                    std::fill((uint8_t*)out_col->data1() + start_row,
                              (uint8_t*)out_col->data1() + out_col->length, 1);
                    return;
                case Bodo_CTypes::INT16:
                    std::fill((int16_t*)out_col->data1() + start_row,
                              (int16_t*)out_col->data1() + out_col->length, 1);
                    return;
                case Bodo_CTypes::UINT16:
                    std::fill((uint16_t*)out_col->data1() + start_row,
                              (uint16_t*)out_col->data1() + out_col->length, 1);
                    return;
                case Bodo_CTypes::INT32:
                    std::fill((int32_t*)out_col->data1() + start_row,
                              (int32_t*)out_col->data1() + out_col->length, 1);
                    return;
                case Bodo_CTypes::UINT32:
                    std::fill((uint32_t*)out_col->data1() + start_row,
                              (uint32_t*)out_col->data1() + out_col->length, 1);
                    return;
                case Bodo_CTypes::INT64:
                    std::fill((int64_t*)out_col->data1() + start_row,
                              (int64_t*)out_col->data1() + out_col->length, 1);
                    return;
                case Bodo_CTypes::UINT64:
                    std::fill((uint64_t*)out_col->data1() + start_row,
                              (uint64_t*)out_col->data1() + out_col->length, 1);
                    return;
                case Bodo_CTypes::FLOAT32:
                    std::fill((float*)out_col->data1() + start_row,
                              (float*)out_col->data1() + out_col->length, 1);
                    return;
                case Bodo_CTypes::FLOAT64:
                    std::fill((double*)out_col->data1() + start_row,
                              (double*)out_col->data1() + out_col->length, 1);
                    return;
                case Bodo_CTypes::STRING:
                case Bodo_CTypes::BINARY:
                default:
                    Bodo_PyErr_SetString(PyExc_RuntimeError, error_msg.c_str());
                    return;
            }
        case Bodo_FTypes::bitand_agg:
            switch (out_col->dtype) {
                case Bodo_CTypes::INT8:
                    std::fill((int8_t*)out_col->data1() + start_row,
                              (int8_t*)out_col->data1() + out_col->length, -1);
                    return;
                case Bodo_CTypes::UINT8:
                    std::fill((uint8_t*)out_col->data1() + start_row,
                              (uint8_t*)out_col->data1() + out_col->length,
                              ~0u);
                    return;
                case Bodo_CTypes::INT16:
                    std::fill((int16_t*)out_col->data1() + start_row,
                              (int16_t*)out_col->data1() + out_col->length, -1);
                    return;
                case Bodo_CTypes::UINT16:
                    std::fill((uint16_t*)out_col->data1() + start_row,
                              (uint16_t*)out_col->data1() + out_col->length,
                              ~0u);
                    return;
                case Bodo_CTypes::INT32:
                    std::fill((int32_t*)out_col->data1() + start_row,
                              (int32_t*)out_col->data1() + out_col->length, -1);
                    return;
                case Bodo_CTypes::UINT32:
                    std::fill((uint32_t*)out_col->data1() + start_row,
                              (uint32_t*)out_col->data1() + out_col->length,
                              ~0u);
                    return;
                case Bodo_CTypes::INT64:
                    std::fill((int64_t*)out_col->data1() + start_row,
                              (int64_t*)out_col->data1() + out_col->length, -1);
                    return;
                case Bodo_CTypes::UINT64:
                    std::fill((uint64_t*)out_col->data1() + start_row,
                              (uint64_t*)out_col->data1() + out_col->length,
                              ~0u);
                    return;
                default:
                    std::fill((int64_t*)out_col->data1() + start_row,
                              (int64_t*)out_col->data1() + out_col->length, -1);
                    return;
            }
        case Bodo_FTypes::min:
            switch (out_col->dtype) {
                case Bodo_CTypes::_BOOL:
                    if (out_col->arr_type ==
                        bodo_array_type::NULLABLE_INT_BOOL) {
                        // Nullable booleans store 1 bit per value
                        InitializeBitMask(
                            (uint8_t*)out_col
                                ->data1<bodo_array_type::NULLABLE_INT_BOOL>(),
                            out_col->length, true, start_row);
                    } else {
                        std::fill((bool*)out_col->data1() + start_row,
                                  (bool*)out_col->data1() + out_col->length,
                                  true);
                    }
                    return;
                case Bodo_CTypes::INT8:
                    std::fill((int8_t*)out_col->data1() + start_row,
                              (int8_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<int8_t>::max());
                    return;
                case Bodo_CTypes::UINT8:
                    std::fill((uint8_t*)out_col->data1() + start_row,
                              (uint8_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<uint8_t>::max());
                    return;
                case Bodo_CTypes::INT16:
                    std::fill((int16_t*)out_col->data1() + start_row,
                              (int16_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<int16_t>::max());
                    return;
                case Bodo_CTypes::UINT16:
                    std::fill((uint16_t*)out_col->data1() + start_row,
                              (uint16_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<uint16_t>::max());
                    return;
                case Bodo_CTypes::INT32:
                case Bodo_CTypes::DATE:
                    std::fill((int32_t*)out_col->data1() + start_row,
                              (int32_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<int32_t>::max());
                    return;
                case Bodo_CTypes::UINT32:
                    std::fill((uint32_t*)out_col->data1() + start_row,
                              (uint32_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<uint32_t>::max());
                    return;
                case Bodo_CTypes::INT64:
                    std::fill((int64_t*)out_col->data1() + start_row,
                              (int64_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<int64_t>::max());
                    return;
                case Bodo_CTypes::UINT64:
                    std::fill((uint64_t*)out_col->data1() + start_row,
                              (uint64_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<uint64_t>::max());
                    return;
                case Bodo_CTypes::DATETIME:
                case Bodo_CTypes::TIMEDELTA:
                    int64_t default_value;
                    if (out_col->arr_type ==
                        bodo_array_type::NULLABLE_INT_BOOL) {
                        // Default to max to simplify compute for the nullable
                        // version. Nulls are already tracked in the null
                        // bitmap.
                        default_value = std::numeric_limits<int64_t>::max();
                    } else {
                        // We must initialize to NaT for the numpy version.
                        default_value = std::numeric_limits<int64_t>::min();
                    }
                    std::fill((int64_t*)out_col->data1() + start_row,
                              (int64_t*)out_col->data1() + out_col->length,
                              default_value);
                    return;
                case Bodo_CTypes::TIMESTAMPTZ:
                    std::fill((int64_t*)out_col->data1() + start_row,
                              (int64_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<int64_t>::max());
                    std::fill((int64_t*)out_col->data2() + start_row,
                              (int64_t*)out_col->data2() + out_col->length, 0);
                    InitializeBitMask((uint8_t*)out_col->null_bitmask(),
                                      out_col->length, false, start_row);
                    return;
                // TODO: [BE-4106] Split Time into Time32 and Time64
                case Bodo_CTypes::TIME:
                    std::fill((int64_t*)out_col->data1() + start_row,
                              (int64_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<int64_t>::max());
                    return;
                case Bodo_CTypes::FLOAT32:
                    // initialize to quiet_NaN so that result is nan if all
                    // input values are nan
                    std::fill((float*)out_col->data1() + start_row,
                              (float*)out_col->data1() + out_col->length,
                              std::numeric_limits<float>::quiet_NaN());
                    return;
                case Bodo_CTypes::FLOAT64:
                    // initialize to quiet_NaN so that result is nan if all
                    // input values are nan
                    std::fill((double*)out_col->data1() + start_row,
                              (double*)out_col->data1() + out_col->length,
                              std::numeric_limits<double>::quiet_NaN());
                    return;
                case Bodo_CTypes::DECIMAL:
                    std::fill((int64_t*)out_col->data1() + 2 * start_row,
                              (int64_t*)out_col->data1() + 2 * out_col->length,
                              std::numeric_limits<int64_t>::max());
                    return;
                case Bodo_CTypes::STRING:
                case Bodo_CTypes::BINARY:
                    // Nothing to initilize with in the case of strings.
                    return;
                default:
                    Bodo_PyErr_SetString(PyExc_RuntimeError, error_msg.c_str());
                    return;
            }
        // MODE has the same numpy initialization rules as MAX
        case Bodo_FTypes::mode:
        case Bodo_FTypes::max:
            switch (out_col->dtype) {
                case Bodo_CTypes::_BOOL:
                    if (out_col->arr_type ==
                        bodo_array_type::NULLABLE_INT_BOOL) {
                        // Nullable booleans store 1 bit per value
                        InitializeBitMask(
                            (uint8_t*)out_col
                                ->data1<bodo_array_type::NULLABLE_INT_BOOL>(),
                            out_col->length, false, start_row);
                    } else {
                        std::fill((bool*)out_col->data1() + start_row,
                                  (bool*)out_col->data1() + out_col->length,
                                  false);
                    }
                    return;
                case Bodo_CTypes::INT8:
                    std::fill((int8_t*)out_col->data1() + start_row,
                              (int8_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<int8_t>::min());
                    return;
                case Bodo_CTypes::UINT8:
                    std::fill((uint8_t*)out_col->data1() + start_row,
                              (uint8_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<uint8_t>::min());
                    return;
                case Bodo_CTypes::INT16:
                    std::fill((int16_t*)out_col->data1() + start_row,
                              (int16_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<int16_t>::min());
                    return;
                case Bodo_CTypes::UINT16:
                    std::fill((uint16_t*)out_col->data1() + start_row,
                              (uint16_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<uint16_t>::min());
                    return;
                case Bodo_CTypes::INT32:
                case Bodo_CTypes::DATE:
                    std::fill((int32_t*)out_col->data1() + start_row,
                              (int32_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<int32_t>::min());
                    return;
                case Bodo_CTypes::UINT32:
                    std::fill((uint32_t*)out_col->data1() + start_row,
                              (uint32_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<uint32_t>::min());
                    return;
                case Bodo_CTypes::INT64:
                    std::fill((int64_t*)out_col->data1() + start_row,
                              (int64_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<int64_t>::min());
                    return;
                case Bodo_CTypes::UINT64:
                    std::fill((uint64_t*)out_col->data1() + start_row,
                              (uint64_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<uint64_t>::min());
                    return;
                case Bodo_CTypes::DATETIME:
                case Bodo_CTypes::TIMEDELTA:
                // TODO: [BE-4106] Split Time into Time32 and Time64
                case Bodo_CTypes::TIME:
                    std::fill((int64_t*)out_col->data1() + start_row,
                              (int64_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<int64_t>::min());
                    return;
                case Bodo_CTypes::TIMESTAMPTZ:
                    std::fill((int64_t*)out_col->data1() + start_row,
                              (int64_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<int64_t>::min());
                    std::fill((int64_t*)out_col->data2() + start_row,
                              (int64_t*)out_col->data2() + out_col->length, 0);
                    InitializeBitMask((uint8_t*)out_col->null_bitmask(),
                                      out_col->length, false, start_row);
                    return;
                case Bodo_CTypes::FLOAT32:
                    // initialize to quiet_NaN so that result is nan if all
                    // input values are nan
                    std::fill((float*)out_col->data1() + start_row,
                              (float*)out_col->data1() + out_col->length,
                              std::numeric_limits<float>::quiet_NaN());
                    return;
                case Bodo_CTypes::FLOAT64:
                    // initialize to quiet_NaN so that result is nan if all
                    // input values are nan
                    std::fill((double*)out_col->data1() + start_row,
                              (double*)out_col->data1() + out_col->length,
                              std::numeric_limits<double>::quiet_NaN());
                    return;
                case Bodo_CTypes::DECIMAL:
                    std::fill((int64_t*)out_col->data1() + 2 * start_row,
                              (int64_t*)out_col->data1() + 2 * out_col->length,
                              std::numeric_limits<int64_t>::min());
                    return;
                case Bodo_CTypes::STRING:
                case Bodo_CTypes::BINARY:
                    // nothing to initialize in the case of strings
                    return;
                default:
                    Bodo_PyErr_SetString(PyExc_RuntimeError, error_msg.c_str());
                    return;
            }
        case Bodo_FTypes::first:
        case Bodo_FTypes::last:
            switch (out_col->dtype) {
                // for first & last, we only need an initial value for the
                // non-null bitmask cases where the datatype has a nan
                // representation
                case Bodo_CTypes::DATETIME:
                case Bodo_CTypes::TIMEDELTA:
                // TODO: [BE-4106] Split Time into Time32 and Time64
                case Bodo_CTypes::TIME:
                    // nat representation for date values is int64_t min value
                    std::fill((int64_t*)out_col->data1() + start_row,
                              (int64_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<int64_t>::min());
                    return;
                case Bodo_CTypes::DATE:
                    // nat representation for date values is int32_t min value
                    std::fill((int32_t*)out_col->data1() + start_row,
                              (int32_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<int32_t>::min());
                    return;
                case Bodo_CTypes::FLOAT32:
                    // initialize to quiet_NaN so that result is nan if all
                    // input values are nan
                    std::fill((float*)out_col->data1() + start_row,
                              (float*)out_col->data1() + out_col->length,
                              std::numeric_limits<float>::quiet_NaN());
                    return;
                case Bodo_CTypes::FLOAT64:
                    // initialize to quiet_NaN so that result is nan if all
                    // input values are nan
                    std::fill((double*)out_col->data1() + start_row,
                              (double*)out_col->data1() + out_col->length,
                              std::numeric_limits<double>::quiet_NaN());
                    return;
                case Bodo_CTypes::TIMESTAMPTZ:
                    // store nat (int64_t min value) in data1 (timestamp) and 0
                    // in data2 (offset)
                    std::fill((int64_t*)out_col->data1() + start_row,
                              (int64_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<int64_t>::min());
                    std::fill((int64_t*)out_col->data2() + start_row,
                              (int64_t*)out_col->data2() + out_col->length, 0);
                    InitializeBitMask((uint8_t*)out_col->null_bitmask(),
                                      out_col->length, false, start_row);
                    return;
                default:
                    // for most cases we don't need an initial value, first/last
                    // will just replace that with the first/last value
                    return;
            }
        case Bodo_FTypes::min_row_number_filter: {
            // Initialize all values to false
            InitializeBitMask((uint8_t*)out_col->data1(), out_col->length,
                              false, start_row);
            return;
        }
        default:
            // zero initialize
            if (out_col->arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
                out_col->dtype == Bodo_CTypes::_BOOL) {
                // Nullable booleans store 1 bit per value
                InitializeBitMask(
                    (uint8_t*)
                        out_col->data1<bodo_array_type::NULLABLE_INT_BOOL>(),
                    out_col->length, false, start_row);
            } else {
                memset(out_col->data1() +
                           numpy_item_size[out_col->dtype] * start_row,
                       0,
                       numpy_item_size[out_col->dtype] *
                           (out_col->length - start_row));
            }
    }
}

void aggfunc_output_initialize(const std::shared_ptr<array_info>& out_col,
                               int ftype, bool use_sql_rules,
                               int64_t start_row) {
    aggfunc_output_initialize_kernel(out_col, ftype, use_sql_rules, start_row);
}

std::tuple<bodo_array_type::arr_type_enum, Bodo_CTypes::CTypeEnum>
get_groupby_output_dtype(int ftype, bodo_array_type::arr_type_enum array_type,
                         Bodo_CTypes::CTypeEnum dtype) {
    bodo_array_type::arr_type_enum out_array_type = array_type;
    Bodo_CTypes::CTypeEnum out_dtype = dtype;
    switch (ftype) {
        case Bodo_FTypes::nunique:
        case Bodo_FTypes::count:
        case Bodo_FTypes::count_if:
        case Bodo_FTypes::size:
        case Bodo_FTypes::ngroup:
            out_array_type = bodo_array_type::NUMPY;
            out_dtype = Bodo_CTypes::INT64;
            break;
        case Bodo_FTypes::var_pop:
        case Bodo_FTypes::std_pop:
        case Bodo_FTypes::var:
        case Bodo_FTypes::std:
        case Bodo_FTypes::kurtosis:
        case Bodo_FTypes::skew:
            out_array_type = bodo_array_type::NULLABLE_INT_BOOL;
            out_dtype = Bodo_CTypes::FLOAT64;
            break;
        case Bodo_FTypes::median:
        case Bodo_FTypes::percentile_disc:
        case Bodo_FTypes::percentile_cont:
        case Bodo_FTypes::mean:
            out_array_type = bodo_array_type::NULLABLE_INT_BOOL;
            if (dtype == Bodo_CTypes::DECIMAL) {
                // mean, median, percentile functions have dedicated versions
                // that return decimal
                out_dtype = Bodo_CTypes::DECIMAL;
            } else {
                out_dtype = Bodo_CTypes::FLOAT64;
            }
            break;
        case Bodo_FTypes::listagg:
            out_array_type = bodo_array_type::STRING;
            break;
        case Bodo_FTypes::cumsum:
        case Bodo_FTypes::sum:
            // This is safe even for cumsum because a boolean cumsum is not yet
            // supported on the Python side, so an error will be raised there
            if (dtype == Bodo_CTypes::_BOOL) {
                out_array_type = bodo_array_type::NULLABLE_INT_BOOL;
                out_dtype = Bodo_CTypes::INT64;
            } else if (dtype == Bodo_CTypes::STRING) {
                out_array_type = bodo_array_type::STRING;
            } else if (is_integer(dtype) && (dtype != Bodo_CTypes::INT128) &&
                       (ftype == Bodo_FTypes::sum)) {
                // Upcast the output to 64 bit version to avoid
                // overflow issues.
                if (is_unsigned_integer(dtype)) {
                    out_dtype = Bodo_CTypes::UINT64;
                } else {
                    out_dtype = Bodo_CTypes::INT64;
                }
            }
            break;
        case Bodo_FTypes::boolor_agg:
        case Bodo_FTypes::booland_agg:
        case Bodo_FTypes::boolxor_agg:
            out_array_type = bodo_array_type::NULLABLE_INT_BOOL;
            out_dtype = Bodo_CTypes::_BOOL;
            break;
        case Bodo_FTypes::bitor_agg:
        case Bodo_FTypes::bitand_agg:
        case Bodo_FTypes::bitxor_agg:
            out_array_type = bodo_array_type::NULLABLE_INT_BOOL;
            // If we have a float or string, then bitX_agg will round/convert,
            // so we output int64 always.
            if (dtype == Bodo_CTypes::FLOAT32 ||
                dtype == Bodo_CTypes::FLOAT64 || dtype == Bodo_CTypes::STRING) {
                out_dtype = Bodo_CTypes::INT64;
            }
            // otherwise, output type should be whatever the input is (dtype =
            // dtype)
            break;
        case Bodo_FTypes::mode:
            // Keep the input dtype and array type
            out_dtype = dtype;
            out_array_type = array_type;
            break;
        case Bodo_FTypes::row_number:
        case Bodo_FTypes::rank:
        case Bodo_FTypes::dense_rank:
        case Bodo_FTypes::ntile:
        case Bodo_FTypes::conditional_true_event:
        case Bodo_FTypes::conditional_change_event:
            out_array_type = bodo_array_type::NUMPY;
            out_dtype = Bodo_CTypes::UINT64;
            break;
        case Bodo_FTypes::percent_rank:
        case Bodo_FTypes::cume_dist:
            out_array_type = bodo_array_type::NUMPY;
            out_dtype = Bodo_CTypes::FLOAT64;
            break;
        case Bodo_FTypes::ratio_to_report:
            out_array_type = bodo_array_type::NULLABLE_INT_BOOL;
            out_dtype = Bodo_CTypes::FLOAT64;
            break;
        case Bodo_FTypes::min_row_number_filter:
            out_array_type = bodo_array_type::NULLABLE_INT_BOOL;
            out_dtype = Bodo_CTypes::_BOOL;
            break;
    }
    return std::tuple(out_array_type, out_dtype);
}

/**
 * @brief Get key_col given a group number
 *
 * @param group[in]: group number
 * @param from_tables[in] list of tables
 * @param key_col_idx[in]
 * @return std::tuple<array_info*, int64_t> Tuple of the
 * column and the row containing the group. Note that we're
 * returning an unowned pointer to the column. The column
 * is only guaranteed to be alive for the lifetime of
 * 'from_tables'.
 */
std::tuple<array_info*, int64_t> find_key_for_group(
    int64_t group, const std::vector<std::shared_ptr<table_info>>& from_tables,
    int64_t key_col_idx, const std::vector<grouping_info>& grp_infos) {
    for (size_t k = 0; k < grp_infos.size(); k++) {
        int64_t key_row = grp_infos[k].group_to_first_row[group];
        if (key_row >= 0) {
            array_info* key_col = (from_tables[k]->columns[key_col_idx]).get();
            return {key_col, key_row};
        }
    }
    throw std::runtime_error("No valid row found for group: " +
                             std::to_string(group));
}

/**
 * Allocate and fill key columns, based on grouping info. It uses the
 * values of key columns from from_table to populate out_table.
 */
void alloc_init_keys(
    const std::vector<std::shared_ptr<table_info>>& from_tables,
    const std::shared_ptr<table_info>& out_table,
    const std::vector<grouping_info>& grp_infos, int64_t num_keys,
    size_t num_groups, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    int64_t key_row = 0;
    for (int64_t i = 0; i < num_keys; i++) {
        // Use a raw pointer since we only need temporary read access.
        // The column is guaranteed to be live for the duration
        // of the loop since 'from_tables' has a live reference
        // to it.
        array_info* key_col = (from_tables[0]->columns[i]).get();
        std::shared_ptr<array_info> new_key_col;
        switch (key_col->arr_type) {
            case bodo_array_type::NULLABLE_INT_BOOL: {
                new_key_col = alloc_array_top_level(
                    num_groups, 1, 1, key_col->arr_type, key_col->dtype, -1, 0,
                    key_col->num_categories, false, false, false, pool, mm);
                if (key_col->dtype == Bodo_CTypes::_BOOL) {
                    // Nullable booleans store 1 bit per boolean
                    for (size_t j = 0; j < num_groups; j++) {
                        std::tie(key_col, key_row) =
                            find_key_for_group(j, from_tables, i, grp_infos);
                        bool bit = GetBit(
                            (uint8_t*)key_col
                                ->data1<bodo_array_type::NULLABLE_INT_BOOL>(),
                            key_row);
                        SetBitTo(
                            (uint8_t*)new_key_col
                                ->data1<bodo_array_type::NULLABLE_INT_BOOL>(),
                            j, bit);
                    }
                } else {
                    int64_t dtype_size = numpy_item_size[key_col->dtype];
                    char* new_data1 = new_key_col->data1();
                    char* old_data1 = key_col->data1();
                    for (size_t j = 0; j < num_groups; j++) {
                        std::tie(key_col, key_row) =
                            find_key_for_group(j, from_tables, i, grp_infos);
                        memcpy(new_data1 + j * dtype_size,
                               old_data1 + key_row * dtype_size, dtype_size);
                    }
                }

                for (size_t j = 0; j < num_groups; j++) {
                    std::tie(key_col, key_row) =
                        find_key_for_group(j, from_tables, i, grp_infos);
                    bool bit =
                        key_col
                            ->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(
                                key_row);
                    new_key_col
                        ->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(j,
                                                                           bit);
                }

                break;
            }
            case bodo_array_type::CATEGORICAL:
            case bodo_array_type::NUMPY: {
                new_key_col = alloc_array_top_level(
                    num_groups, 1, 1, key_col->arr_type, key_col->dtype, -1, 0,
                    key_col->num_categories, false, false, false, pool, mm);

                int64_t dtype_size = numpy_item_size[key_col->dtype];
                char* new_data1 = new_key_col->data1();
                char* old_data1 = key_col->data1();
                for (size_t j = 0; j < num_groups; j++) {
                    std::tie(key_col, key_row) =
                        find_key_for_group(j, from_tables, i, grp_infos);
                    memcpy(new_data1 + j * dtype_size,
                           old_data1 + key_row * dtype_size, dtype_size);
                }

                break;
            }
            case bodo_array_type::TIMESTAMPTZ: {
                new_key_col = alloc_array_top_level(
                    num_groups, 1, 1, key_col->arr_type, key_col->dtype, -1, 0,
                    key_col->num_categories, false, false, false, pool, mm);
                int64_t dtype_size = numpy_item_size[key_col->dtype];
                for (size_t j = 0; j < num_groups; j++) {
                    std::tie(key_col, key_row) =
                        find_key_for_group(j, from_tables, i, grp_infos);
                    // data1 (stores UTC timestamp)
                    memcpy(new_key_col->data1<bodo_array_type::TIMESTAMPTZ>() +
                               j * dtype_size,
                           key_col->data1<bodo_array_type::TIMESTAMPTZ>() +
                               key_row * dtype_size,
                           dtype_size);
                    // data2 (stores offset from UTC)
                    memcpy(new_key_col->data2<bodo_array_type::TIMESTAMPTZ>() +
                               j * sizeof(int16_t),
                           key_col->data2<bodo_array_type::TIMESTAMPTZ>() +
                               key_row * sizeof(int16_t),
                           sizeof(int16_t));
                    // null bitmask
                    bool bit =
                        key_col->get_null_bit<bodo_array_type::TIMESTAMPTZ>(
                            key_row);
                    new_key_col->set_null_bit<bodo_array_type::TIMESTAMPTZ>(
                        j, bit);
                }

                break;
            }
            case bodo_array_type::DICT: {
                array_info* key_indices = (key_col->child_arrays[1]).get();
                std::shared_ptr<array_info> new_key_indices =
                    alloc_array_top_level(num_groups, -1, -1,
                                          key_indices->arr_type,
                                          key_indices->dtype, -1, 0, 0, false,
                                          false, false, pool, mm);
                for (size_t j = 0; j < num_groups; j++) {
                    std::tie(key_col, key_row) =
                        find_key_for_group(j, from_tables, i, grp_infos);
                    // Update key_indices with the new key col
                    key_indices = (key_col->child_arrays[1]).get();
                    new_key_indices->at<dict_indices_t,
                                        bodo_array_type::NULLABLE_INT_BOOL>(j) =
                        key_indices->at<dict_indices_t,
                                        bodo_array_type::NULLABLE_INT_BOOL>(
                            key_row);
                    bool bit =
                        key_indices
                            ->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(
                                key_row);
                    new_key_indices
                        ->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(j,
                                                                           bit);
                }
                new_key_col = create_dict_string_array(key_col->child_arrays[0],
                                                       new_key_indices);

                break;
            }
            case bodo_array_type::STRING: {
                // new key col will have num_groups rows containing the
                // string for each group
                int64_t n_chars = 0;  // total number of chars of all keys for
                                      // this column
                offset_t* in_offsets;
                for (size_t j = 0; j < num_groups; j++) {
                    std::tie(key_col, key_row) =
                        find_key_for_group(j, from_tables, i, grp_infos);
                    in_offsets =
                        (offset_t*)key_col->data2<bodo_array_type::STRING>();
                    n_chars += in_offsets[key_row + 1] - in_offsets[key_row];
                }
                // XXX Shouldn't we forward the array_id here?
                new_key_col = alloc_array_top_level(
                    num_groups, n_chars, 1, key_col->arr_type, key_col->dtype,
                    -1, 0, key_col->num_categories,
                    key_col->is_globally_replicated, key_col->is_locally_unique,
                    key_col->is_locally_sorted, pool, mm);

                offset_t* out_offsets =
                    (offset_t*)new_key_col->data2<bodo_array_type::STRING>();
                offset_t pos = 0;
                for (size_t j = 0; j < num_groups; j++) {
                    std::tie(key_col, key_row) =
                        find_key_for_group(j, from_tables, i, grp_infos);
                    in_offsets =
                        (offset_t*)key_col->data2<bodo_array_type::STRING>();
                    offset_t start_offset = in_offsets[key_row];
                    offset_t str_len = in_offsets[key_row + 1] - start_offset;
                    out_offsets[j] = pos;
                    memcpy(
                        &new_key_col->data1<bodo_array_type::STRING>()[pos],
                        &key_col
                             ->data1<bodo_array_type::STRING>()[start_offset],
                        str_len);
                    pos += str_len;
                    bool bit =
                        key_col->get_null_bit<bodo_array_type::STRING>(key_row);
                    new_key_col->set_null_bit<bodo_array_type::STRING>(j, bit);
                }
                out_offsets[num_groups] = pos;

                break;
            }
            case bodo_array_type::ARRAY_ITEM: {
                // Allocate the ARRAY_ITEM array
                new_key_col =
                    alloc_array_item(num_groups, nullptr, 0, pool, mm);

                // Setup the offsets in data1
                offset_t* new_key_offset_arr =
                    (offset_t*)
                        new_key_col->data1<bodo_array_type::ARRAY_ITEM>();

                offset_t pos = 0;

                bool all_groups_have_same_key_col =
                    (from_tables.size() == 1 && grp_infos.size() == 1);

                // List of arrays to concat
                std::vector<std::shared_ptr<array_info>> inner_arr_parts;

                std::vector<int64_t> idxs;
                for (size_t j = 0; j < num_groups; j++) {
                    std::tie(key_col, key_row) =
                        find_key_for_group(j, from_tables, i, grp_infos);
                    offset_t* key_offset_arr =
                        (offset_t*)
                            key_col->data1<bodo_array_type::ARRAY_ITEM>();

                    // construct new offsets in data1
                    offset_t start_offset = key_offset_arr[key_row];
                    offset_t end_offset = key_offset_arr[key_row + 1];
                    offset_t length = end_offset - start_offset;
                    new_key_offset_arr[j] = pos;
                    pos += length;

                    if (all_groups_have_same_key_col) {
                        // Collect the indices to make a single call to
                        // RetrieveArray_SingleColumn later.

                        // TODO(aneesh) this could be made more efficient by
                        // using a custom iterator type and making
                        // RetrieveArray_SingleColumn take an arbitrary
                        // iterator, but this probably doesn't have a huge
                        // impact on memory usage/time.
                        for (offset_t k = start_offset; k < end_offset; k++) {
                            idxs.push_back(k);
                        }
                    } else {
                        idxs.resize(length);
                        // fills idxs with [start_offset, start_offset + 1,
                        // ..., end_offset - 1]
                        std::iota(idxs.begin(), idxs.end(), start_offset);
                        // add key_col->child_arrays[0][idxs] to inner_arr_parts
                        inner_arr_parts.emplace_back(RetrieveArray_SingleColumn(
                            key_col->child_arrays[0], idxs, false, pool, mm));
                        idxs.clear();
                    }

                    // null bitmask
                    bool bit =
                        key_col->get_null_bit<bodo_array_type::ARRAY_ITEM>(
                            key_row);
                    new_key_col->set_null_bit<bodo_array_type::ARRAY_ITEM>(j,
                                                                           bit);
                }
                new_key_offset_arr[num_groups] = pos;

                if (all_groups_have_same_key_col && !idxs.empty()) {
                    new_key_col->child_arrays[0] = RetrieveArray_SingleColumn(
                        key_col->child_arrays[0], idxs, false, pool, mm);
                } else {
                    // Assemble child array fragments into a contiguous array
                    if (inner_arr_parts.empty()) {
                        // concat_arrays expects a non-empty vector, so we
                        // instead allocate an empty array that matches the type
                        // of the input.
                        new_key_col->child_arrays[0] =
                            alloc_array_like(key_col->child_arrays[0], false);
                    } else {
                        new_key_col->child_arrays[0] =
                            concat_arrays(inner_arr_parts);
                    }
                }
                break;
            }
            default: {
                throw new std::runtime_error(
                    "_groupby_common.cpp::alloc_init_keys: Unsupported array "
                    "type " +
                    GetArrType_as_string(key_col->arr_type) + ".");
            }
        }
        out_table->columns.push_back(std::move(new_key_col));
    }
}

std::shared_ptr<table_info> grouped_sort(
    const grouping_info& grp_info,
    const std::vector<std::shared_ptr<array_info>>& orderby_cols,
    const std::vector<std::shared_ptr<array_info>>& extra_cols,
    const std::vector<bool>& asc_vect, const std::vector<bool>& na_pos_vect,
    int64_t order_offset, bool is_parallel, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    // Create a new table. We want to sort the table first by
    // the groups and second by the orderby_arr.
    std::shared_ptr<table_info> sort_table = std::make_shared<table_info>();
    const bodo::vector<int64_t>& row_to_group = grp_info.row_to_group;
    int64_t num_rows = row_to_group.size();
    int64_t n_keys = orderby_cols.size() + 1;
    std::vector<int64_t> vect_ascending(n_keys);
    std::vector<int64_t> na_position(n_keys);

    // Wrap the row_to_group in an array info so we can use it
    // to sort.
    std::shared_ptr<array_info> group_arr =
        alloc_numpy(num_rows, Bodo_CTypes::INT64, pool, mm);
    memcpy(group_arr->data1<bodo_array_type::NUMPY>(), row_to_group.data(),
           sizeof(int64_t) * group_arr->length);

    sort_table->columns.push_back(group_arr);

    // Push each orderby column into the sort table
    for (std::shared_ptr<array_info> orderby_arr : orderby_cols) {
        sort_table->columns.push_back(orderby_arr);
    }

    // Push each extra column into the sort table
    for (std::shared_ptr<array_info> extra_arr : extra_cols) {
        sort_table->columns.push_back(extra_arr);
    }

    /* Populate the buffers to indicate which columns are
     * ascending/descending and which have nulls first/last
     * according to the input specifications, plus the
     * group key column in front which is hardcoded.
     */
    vect_ascending[0] = 1;
    na_position[0] = 1;
    for (int64_t i = 1; i < n_keys; i++) {
        vect_ascending[i] = asc_vect[i + order_offset - 1];
        na_position[i] = na_pos_vect[i + order_offset - 1];
    }
    // Sort the table so that all window functions that use the
    // sorted table can access it
    std::shared_ptr<table_info> iter_table = sort_values_table_local(
        sort_table, n_keys, vect_ascending.data(), na_position.data(), nullptr,
        is_parallel /* This is just used for tracing */, pool, std::move(mm));
    sort_table.reset();
    return iter_table;
}
