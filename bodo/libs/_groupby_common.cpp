// Copyright (C) 2023 Bodo Inc. All rights reserved.
#include "_groupby_common.h"
#include "_array_utils.h"
#include "_groupby_ftypes.h"
#include "_groupby_update.h"

/**
 * This function contains helper functions that are shared by multiple possible
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
 *
 * @param output column
 * @param function identifier
 * @param use_sql_rules: If true, use SQL rules for null handling. If false, use
 * Pandas rules.
 */
void aggfunc_output_initialize_kernel(array_info* out_col, int ftype,
                                      bool use_sql_rules) {
    // Generate an error message for unsupported paths that includes the name
    // of the function and the dtype.
    std::string error_msg = std::string("unsupported aggregate function: ") +
                            std::string(get_name_for_Bodo_FTypes(ftype)) +
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
                ftype == Bodo_FTypes::mean || ftype == Bodo_FTypes::var ||
                ftype == Bodo_FTypes::std || ftype == Bodo_FTypes::median) {
                // if input is all nulls, max, min, first, last, and boolor_agg
                // output will be null. We null initialize median, mean, var,
                // and std as well since we always output a nullable float at
                // this time.
                init_val = false;
            } else {
                init_val = true;
            }
        }
        InitializeBitMask((uint8_t*)out_col->null_bitmask(), out_col->length,
                          init_val);
    }
    if (out_col->arr_type == bodo_array_type::STRING ||
        out_col->arr_type == bodo_array_type::LIST_STRING) {
        InitializeBitMask((uint8_t*)out_col->null_bitmask(), out_col->length,
                          false);
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
                    std::fill((int8_t*)out_col->data1(),
                              (int8_t*)out_col->data1() + out_col->length,
                              init_val);
                    return;
                case Bodo_CTypes::INT16:
                    std::fill((int16_t*)out_col->data1(),
                              (int16_t*)out_col->data1() + out_col->length,
                              init_val);
                    return;
                case Bodo_CTypes::INT32:
                    std::fill((int32_t*)out_col->data1(),
                              (int32_t*)out_col->data1() + out_col->length,
                              init_val);
                    return;
                case Bodo_CTypes::INT64:
                    std::fill((int64_t*)out_col->data1(),
                              (int64_t*)out_col->data1() + out_col->length,
                              init_val);
                    return;
                default:
                    Bodo_PyErr_SetString(PyExc_RuntimeError, error_msg.c_str());
                    return;
            }
        }
    }
    switch (ftype) {
        case Bodo_FTypes::prod:
            switch (out_col->dtype) {
                case Bodo_CTypes::_BOOL:
                    if (out_col->arr_type ==
                        bodo_array_type::NULLABLE_INT_BOOL) {
                        // Nullable booleans store 1 bit per value
                        InitializeBitMask((uint8_t*)out_col->data1(),
                                          out_col->length, true);
                    } else {
                        std::fill((bool*)out_col->data1(),
                                  (bool*)out_col->data1() + out_col->length,
                                  true);
                    }
                    return;
                case Bodo_CTypes::INT8:
                    std::fill((int8_t*)out_col->data1(),
                              (int8_t*)out_col->data1() + out_col->length, 1);
                    return;
                case Bodo_CTypes::UINT8:
                    std::fill((uint8_t*)out_col->data1(),
                              (uint8_t*)out_col->data1() + out_col->length, 1);
                    return;
                case Bodo_CTypes::INT16:
                    std::fill((int16_t*)out_col->data1(),
                              (int16_t*)out_col->data1() + out_col->length, 1);
                    return;
                case Bodo_CTypes::UINT16:
                    std::fill((uint16_t*)out_col->data1(),
                              (uint16_t*)out_col->data1() + out_col->length, 1);
                    return;
                case Bodo_CTypes::INT32:
                    std::fill((int32_t*)out_col->data1(),
                              (int32_t*)out_col->data1() + out_col->length, 1);
                    return;
                case Bodo_CTypes::UINT32:
                    std::fill((uint32_t*)out_col->data1(),
                              (uint32_t*)out_col->data1() + out_col->length, 1);
                    return;
                case Bodo_CTypes::INT64:
                    std::fill((int64_t*)out_col->data1(),
                              (int64_t*)out_col->data1() + out_col->length, 1);
                    return;
                case Bodo_CTypes::UINT64:
                    std::fill((uint64_t*)out_col->data1(),
                              (uint64_t*)out_col->data1() + out_col->length, 1);
                    return;
                case Bodo_CTypes::FLOAT32:
                    std::fill((float*)out_col->data1(),
                              (float*)out_col->data1() + out_col->length, 1);
                    return;
                case Bodo_CTypes::FLOAT64:
                    std::fill((double*)out_col->data1(),
                              (double*)out_col->data1() + out_col->length, 1);
                    return;
                case Bodo_CTypes::STRING:
                case Bodo_CTypes::BINARY:
                default:
                    Bodo_PyErr_SetString(PyExc_RuntimeError, error_msg.c_str());
                    return;
            }
        case Bodo_FTypes::min:
            switch (out_col->dtype) {
                case Bodo_CTypes::_BOOL:
                    if (out_col->arr_type ==
                        bodo_array_type::NULLABLE_INT_BOOL) {
                        // Nullable booleans store 1 bit per value
                        InitializeBitMask((uint8_t*)out_col->data1(),
                                          out_col->length, true);
                    } else {
                        std::fill((bool*)out_col->data1(),
                                  (bool*)out_col->data1() + out_col->length,
                                  true);
                    }
                    return;
                case Bodo_CTypes::INT8:
                    std::fill((int8_t*)out_col->data1(),
                              (int8_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<int8_t>::max());
                    return;
                case Bodo_CTypes::UINT8:
                    std::fill((uint8_t*)out_col->data1(),
                              (uint8_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<uint8_t>::max());
                    return;
                case Bodo_CTypes::INT16:
                    std::fill((int16_t*)out_col->data1(),
                              (int16_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<int16_t>::max());
                    return;
                case Bodo_CTypes::UINT16:
                    std::fill((uint16_t*)out_col->data1(),
                              (uint16_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<uint16_t>::max());
                    return;
                case Bodo_CTypes::INT32:
                    std::fill((int32_t*)out_col->data1(),
                              (int32_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<int32_t>::max());
                    return;
                case Bodo_CTypes::UINT32:
                    std::fill((uint32_t*)out_col->data1(),
                              (uint32_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<uint32_t>::max());
                    return;
                case Bodo_CTypes::INT64:
                    std::fill((int64_t*)out_col->data1(),
                              (int64_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<int64_t>::max());
                    return;
                case Bodo_CTypes::UINT64:
                    std::fill((uint64_t*)out_col->data1(),
                              (uint64_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<uint64_t>::max());
                    return;
                case Bodo_CTypes::DATE:
                case Bodo_CTypes::DATETIME:
                case Bodo_CTypes::TIMEDELTA:
                // TODO: [BE-4106] Split Time into Time32 and Time64
                case Bodo_CTypes::TIME:
                    std::fill((int64_t*)out_col->data1(),
                              (int64_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<int64_t>::max());
                    return;
                case Bodo_CTypes::FLOAT32:
                    // initialize to quiet_NaN so that result is nan if all
                    // input values are nan
                    std::fill((float*)out_col->data1(),
                              (float*)out_col->data1() + out_col->length,
                              std::numeric_limits<float>::quiet_NaN());
                    return;
                case Bodo_CTypes::FLOAT64:
                    // initialize to quiet_NaN so that result is nan if all
                    // input values are nan
                    std::fill((double*)out_col->data1(),
                              (double*)out_col->data1() + out_col->length,
                              std::numeric_limits<double>::quiet_NaN());
                    return;
                case Bodo_CTypes::DECIMAL:
                    std::fill((int64_t*)out_col->data1(),
                              (int64_t*)out_col->data1() + 2 * out_col->length,
                              std::numeric_limits<int64_t>::max());
                    return;
                case Bodo_CTypes::STRING:
                case Bodo_CTypes::BINARY:
                    // Nothing to initilize with in the case of strings.
                    return;
                case Bodo_CTypes::LIST_STRING:
                    // Nothing to initilize with in the case of list strings.
                    return;
                default:
                    Bodo_PyErr_SetString(PyExc_RuntimeError, error_msg.c_str());
                    return;
            }
        case Bodo_FTypes::max:
            switch (out_col->dtype) {
                case Bodo_CTypes::_BOOL:
                    if (out_col->arr_type ==
                        bodo_array_type::NULLABLE_INT_BOOL) {
                        // Nullable booleans store 1 bit per value
                        InitializeBitMask((uint8_t*)out_col->data1(),
                                          out_col->length, false);
                    } else {
                        std::fill((bool*)out_col->data1(),
                                  (bool*)out_col->data1() + out_col->length,
                                  false);
                    }
                    return;
                case Bodo_CTypes::INT8:
                    std::fill((int8_t*)out_col->data1(),
                              (int8_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<int8_t>::min());
                    return;
                case Bodo_CTypes::UINT8:
                    std::fill((uint8_t*)out_col->data1(),
                              (uint8_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<uint8_t>::min());
                    return;
                case Bodo_CTypes::INT16:
                    std::fill((int16_t*)out_col->data1(),
                              (int16_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<int16_t>::min());
                    return;
                case Bodo_CTypes::UINT16:
                    std::fill((uint16_t*)out_col->data1(),
                              (uint16_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<uint16_t>::min());
                    return;
                case Bodo_CTypes::INT32:
                    std::fill((int32_t*)out_col->data1(),
                              (int32_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<int32_t>::min());
                    return;
                case Bodo_CTypes::UINT32:
                    std::fill((uint32_t*)out_col->data1(),
                              (uint32_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<uint32_t>::min());
                    return;
                case Bodo_CTypes::INT64:
                    std::fill((int64_t*)out_col->data1(),
                              (int64_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<int64_t>::min());
                    return;
                case Bodo_CTypes::UINT64:
                    std::fill((uint64_t*)out_col->data1(),
                              (uint64_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<uint64_t>::min());
                    return;
                case Bodo_CTypes::DATE:
                case Bodo_CTypes::DATETIME:
                case Bodo_CTypes::TIMEDELTA:
                // TODO: [BE-4106] Split Time into Time32 and Time64
                case Bodo_CTypes::TIME:
                    std::fill((int64_t*)out_col->data1(),
                              (int64_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<int64_t>::min());
                    return;
                case Bodo_CTypes::FLOAT32:
                    // initialize to quiet_NaN so that result is nan if all
                    // input values are nan
                    std::fill((float*)out_col->data1(),
                              (float*)out_col->data1() + out_col->length,
                              std::numeric_limits<float>::quiet_NaN());
                    return;
                case Bodo_CTypes::FLOAT64:
                    // initialize to quiet_NaN so that result is nan if all
                    // input values are nan
                    std::fill((double*)out_col->data1(),
                              (double*)out_col->data1() + out_col->length,
                              std::numeric_limits<double>::quiet_NaN());
                    return;
                case Bodo_CTypes::DECIMAL:
                    std::fill((int64_t*)out_col->data1(),
                              (int64_t*)out_col->data1() + 2 * out_col->length,
                              std::numeric_limits<int64_t>::min());
                    return;
                case Bodo_CTypes::STRING:
                case Bodo_CTypes::BINARY:
                    // nothing to initialize in the case of strings
                    return;
                case Bodo_CTypes::LIST_STRING:
                    // nothing to initialize in the case of list strings
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
                case Bodo_CTypes::DATE:
                case Bodo_CTypes::DATETIME:
                case Bodo_CTypes::TIMEDELTA:
                // TODO: [BE-4106] Split Time into Time32 and Time64
                case Bodo_CTypes::TIME:
                    // nat representation for date values is int64_t min value
                    std::fill((int64_t*)out_col->data1(),
                              (int64_t*)out_col->data1() + out_col->length,
                              std::numeric_limits<int64_t>::min());
                    return;
                case Bodo_CTypes::FLOAT32:
                    // initialize to quiet_NaN so that result is nan if all
                    // input values are nan
                    std::fill((float*)out_col->data1(),
                              (float*)out_col->data1() + out_col->length,
                              std::numeric_limits<float>::quiet_NaN());
                    return;
                case Bodo_CTypes::FLOAT64:
                    // initialize to quiet_NaN so that result is nan if all
                    // input values are nan
                    std::fill((double*)out_col->data1(),
                              (double*)out_col->data1() + out_col->length,
                              std::numeric_limits<double>::quiet_NaN());
                    return;
                default:
                    // for most cases we don't need an initial value, first/last
                    // will just replace that with the first/last value
                    return;
            }
        case Bodo_FTypes::min_row_number_filter: {
            // Initialize all values to false
            InitializeBitMask((uint8_t*)out_col->data1(), out_col->length,
                              false);
            return;
        }
        default:
            // zero initialize
            if (out_col->arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
                out_col->dtype == Bodo_CTypes::_BOOL) {
                // Nullable booleans store 1 bit per value
                InitializeBitMask((uint8_t*)out_col->data1(), out_col->length,
                                  false);
            } else {
                memset(out_col->data1(), 0,
                       numpy_item_size[out_col->dtype] * out_col->length);
            }
    }
}

void aggfunc_output_initialize(array_info* out_col, int ftype,
                               bool use_sql_rules) {
    aggfunc_output_initialize_kernel(out_col, ftype, use_sql_rules);
}

void get_groupby_output_dtype(int ftype,
                              bodo_array_type::arr_type_enum& array_type,
                              Bodo_CTypes::CTypeEnum& dtype) {
    switch (ftype) {
        case Bodo_FTypes::nunique:
        case Bodo_FTypes::count:
        case Bodo_FTypes::count_if:
        case Bodo_FTypes::size:
        case Bodo_FTypes::ngroup:
            array_type = bodo_array_type::NUMPY;
            dtype = Bodo_CTypes::INT64;
            return;
        case Bodo_FTypes::median:
        case Bodo_FTypes::mean:
        case Bodo_FTypes::var:
        case Bodo_FTypes::std:
            array_type = bodo_array_type::NULLABLE_INT_BOOL;
            dtype = Bodo_CTypes::FLOAT64;
            return;
        case Bodo_FTypes::cumsum:
        case Bodo_FTypes::sum:
            // This is safe even for cumsum because a boolean cumsum is not yet
            // supported on the Python side, so an error will be raised there
            if (dtype == Bodo_CTypes::_BOOL) {
                array_type = bodo_array_type::NULLABLE_INT_BOOL;
                dtype = Bodo_CTypes::INT64;
            } else if (dtype == Bodo_CTypes::STRING) {
                array_type = bodo_array_type::STRING;
            }
            return;
        case Bodo_FTypes::boolor_agg:
            array_type = bodo_array_type::NULLABLE_INT_BOOL;
            dtype = Bodo_CTypes::_BOOL;
            return;
        case Bodo_FTypes::row_number:
            array_type = bodo_array_type::NUMPY;
            dtype = Bodo_CTypes::UINT64;
            return;
        case Bodo_FTypes::min_row_number_filter:
            array_type = bodo_array_type::NULLABLE_INT_BOOL;
            dtype = Bodo_CTypes::_BOOL;
            return;
    }
}
