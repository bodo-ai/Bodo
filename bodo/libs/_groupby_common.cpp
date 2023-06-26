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
void aggfunc_output_initialize_kernel(
    const std::shared_ptr<array_info>& out_col, int ftype, bool use_sql_rules,
    int64_t start_row = 0) {
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
                ftype == Bodo_FTypes::booland_agg ||
                ftype == Bodo_FTypes::boolxor_agg ||
                ftype == Bodo_FTypes::bitor_agg ||
                ftype == Bodo_FTypes::bitand_agg ||
                ftype == Bodo_FTypes::bitxor_agg ||
                ftype == Bodo_FTypes::mean || ftype == Bodo_FTypes::var ||
                ftype == Bodo_FTypes::var_pop ||
                ftype == Bodo_FTypes::std_pop ||
                ftype == Bodo_FTypes::kurtosis || ftype == Bodo_FTypes::skew ||
                ftype == Bodo_FTypes::std || ftype == Bodo_FTypes::median) {
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
        InitializeBitMask((uint8_t*)out_col->null_bitmask(), out_col->length,
                          init_val, start_row);
    }
    if (out_col->arr_type == bodo_array_type::STRING ||
        out_col->arr_type == bodo_array_type::LIST_STRING) {
        InitializeBitMask((uint8_t*)out_col->null_bitmask(), out_col->length,
                          false, start_row);
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
                    std::fill((int8_t*)out_col->data1() + start_row,
                              (int8_t*)out_col->data1() + out_col->length,
                              init_val);
                    return;
                case Bodo_CTypes::INT16:
                    std::fill((int16_t*)out_col->data1() + start_row,
                              (int16_t*)out_col->data1() + out_col->length,
                              init_val);
                    return;
                case Bodo_CTypes::INT32:
                    std::fill((int32_t*)out_col->data1() + start_row,
                              (int32_t*)out_col->data1() + out_col->length,
                              init_val);
                    return;
                case Bodo_CTypes::INT64:
                    std::fill((int64_t*)out_col->data1() + start_row,
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
                        InitializeBitMask((uint8_t*)out_col->data1(),
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
                        InitializeBitMask((uint8_t*)out_col->data1(),
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
                InitializeBitMask((uint8_t*)out_col->data1(), out_col->length,
                                  false, start_row);
            } else {
                memset(out_col->data1() +
                           numpy_item_size[out_col->dtype] * start_row,
                       0, numpy_item_size[out_col->dtype] * out_col->length);
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
        case Bodo_FTypes::median:
        case Bodo_FTypes::mean:
        case Bodo_FTypes::var_pop:
        case Bodo_FTypes::std_pop:
        case Bodo_FTypes::var:
        case Bodo_FTypes::std:
        case Bodo_FTypes::kurtosis:
        case Bodo_FTypes::skew:
            out_array_type = bodo_array_type::NULLABLE_INT_BOOL;
            out_dtype = Bodo_CTypes::FLOAT64;
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
        case Bodo_FTypes::row_number:
            out_array_type = bodo_array_type::NUMPY;
            out_dtype = Bodo_CTypes::UINT64;
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
    size_t num_groups) {
    int64_t key_row = 0;
    for (int64_t i = 0; i < num_keys; i++) {
        // Use a raw pointer since we only need temporary read access.
        // The column is guaranteed to be live for the duration
        // of the loop since 'from_tables' has a live reference
        // to it.
        array_info* key_col = (from_tables[0]->columns[i]).get();
        std::shared_ptr<array_info> new_key_col = nullptr;
        if (key_col->arr_type == bodo_array_type::NUMPY ||
            key_col->arr_type == bodo_array_type::CATEGORICAL ||
            key_col->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
            new_key_col =
                alloc_array(num_groups, 1, 1, key_col->arr_type, key_col->dtype,
                            0, key_col->num_categories);
            if (key_col->arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
                key_col->dtype == Bodo_CTypes::_BOOL) {
                // Nullable booleans store 1 bit per boolean
                for (size_t j = 0; j < num_groups; j++) {
                    std::tie(key_col, key_row) =
                        find_key_for_group(j, from_tables, i, grp_infos);
                    bool bit = GetBit((uint8_t*)key_col->data1(), key_row);
                    SetBitTo((uint8_t*)new_key_col->data1(), j, bit);
                }
            } else {
                int64_t dtype_size = numpy_item_size[key_col->dtype];
                for (size_t j = 0; j < num_groups; j++) {
                    std::tie(key_col, key_row) =
                        find_key_for_group(j, from_tables, i, grp_infos);
                    memcpy(new_key_col->data1() + j * dtype_size,
                           key_col->data1() + key_row * dtype_size, dtype_size);
                }
            }
            if (key_col->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
                for (size_t j = 0; j < num_groups; j++) {
                    std::tie(key_col, key_row) =
                        find_key_for_group(j, from_tables, i, grp_infos);
                    bool bit = key_col->get_null_bit(key_row);
                    new_key_col->set_null_bit(j, bit);
                }
            }
        }
        if (key_col->arr_type == bodo_array_type::DICT) {
            array_info* key_indices = (key_col->child_arrays[1]).get();
            std::shared_ptr<array_info> new_key_indices =
                alloc_array(num_groups, -1, -1, key_indices->arr_type,
                            key_indices->dtype, 0, 0);
            for (size_t j = 0; j < num_groups; j++) {
                std::tie(key_col, key_row) =
                    find_key_for_group(j, from_tables, i, grp_infos);
                // Update key_indices with the new key col
                key_indices = (key_col->child_arrays[1]).get();
                new_key_indices->at<dict_indices_t>(j) =
                    key_indices->at<dict_indices_t>(key_row);
                bool bit = key_indices->get_null_bit(key_row);
                new_key_indices->set_null_bit(j, bit);
            }
            new_key_col = create_dict_string_array(
                key_col->child_arrays[0], new_key_indices,
                key_col->has_global_dictionary,
                key_col->has_deduped_local_dictionary,
                key_col->has_sorted_dictionary);
        }
        if (key_col->arr_type == bodo_array_type::STRING) {
            // new key col will have num_groups rows containing the
            // string for each group
            int64_t n_chars = 0;  // total number of chars of all keys for
                                  // this column
            offset_t* in_offsets;
            for (size_t j = 0; j < num_groups; j++) {
                std::tie(key_col, key_row) =
                    find_key_for_group(j, from_tables, i, grp_infos);
                in_offsets = (offset_t*)key_col->data2();
                n_chars += in_offsets[key_row + 1] - in_offsets[key_row];
            }
            new_key_col =
                alloc_array(num_groups, n_chars, 1, key_col->arr_type,
                            key_col->dtype, 0, key_col->num_categories);

            offset_t* out_offsets = (offset_t*)new_key_col->data2();
            offset_t pos = 0;
            for (size_t j = 0; j < num_groups; j++) {
                std::tie(key_col, key_row) =
                    find_key_for_group(j, from_tables, i, grp_infos);
                in_offsets = (offset_t*)key_col->data2();
                offset_t start_offset = in_offsets[key_row];
                offset_t str_len = in_offsets[key_row + 1] - start_offset;
                out_offsets[j] = pos;
                memcpy(&new_key_col->data1()[pos],
                       &key_col->data1()[start_offset], str_len);
                pos += str_len;
                bool bit = key_col->get_null_bit(key_row);
                new_key_col->set_null_bit(j, bit);
            }
            out_offsets[num_groups] = pos;
        }
        if (key_col->arr_type == bodo_array_type::LIST_STRING) {
            // new key col will have num_groups rows containing the
            // list string for each group
            int64_t n_strings = 0;  // total number of strings of all keys
                                    // for this column
            int64_t n_chars = 0;    // total number of chars of all keys for
                                    // this column
            offset_t* in_index_offsets;
            offset_t* in_data_offsets;
            for (size_t j = 0; j < num_groups; j++) {
                std::tie(key_col, key_row) =
                    find_key_for_group(j, from_tables, i, grp_infos);
                in_index_offsets = (offset_t*)key_col->data3();
                in_data_offsets = (offset_t*)key_col->data2();
                n_strings +=
                    in_index_offsets[key_row + 1] - in_index_offsets[key_row];
                n_chars += in_data_offsets[in_index_offsets[key_row + 1]] -
                           in_data_offsets[in_index_offsets[key_row]];
            }
            new_key_col =
                alloc_array(num_groups, n_strings, n_chars, key_col->arr_type,
                            key_col->dtype, 0, key_col->num_categories);
            uint8_t* in_sub_null_bitmask =
                (uint8_t*)key_col->sub_null_bitmask();
            uint8_t* out_sub_null_bitmask =
                (uint8_t*)new_key_col->sub_null_bitmask();
            offset_t* out_index_offsets = (offset_t*)new_key_col->data3();
            offset_t* out_data_offsets = (offset_t*)new_key_col->data2();
            offset_t pos_data = 0;
            offset_t pos_index = 0;
            out_data_offsets[0] = 0;
            out_index_offsets[0] = 0;
            for (size_t j = 0; j < num_groups; j++) {
                std::tie(key_col, key_row) =
                    find_key_for_group(j, from_tables, i, grp_infos);
                in_index_offsets = (offset_t*)key_col->data3();
                in_data_offsets = (offset_t*)key_col->data2();
                offset_t size_index =
                    in_index_offsets[key_row + 1] - in_index_offsets[key_row];
                offset_t pos_start = in_index_offsets[key_row];
                for (offset_t i_str = 0; i_str < size_index; i_str++) {
                    offset_t len_str = in_data_offsets[pos_start + i_str + 1] -
                                       in_data_offsets[pos_start + i_str];
                    pos_index++;
                    out_data_offsets[pos_index] =
                        out_data_offsets[pos_index - 1] + len_str;
                    bool bit = GetBit(in_sub_null_bitmask, pos_start + i_str);
                    SetBitTo(out_sub_null_bitmask, pos_index, bit);
                }
                out_index_offsets[j + 1] = pos_index;
                // Now the strings themselves
                offset_t in_start_offset =
                    in_data_offsets[in_index_offsets[key_row]];
                offset_t n_chars_o =
                    in_data_offsets[in_index_offsets[key_row + 1]] -
                    in_data_offsets[in_index_offsets[key_row]];
                memcpy(&new_key_col->data1()[pos_data],
                       &key_col->data1()[in_start_offset], n_chars_o);
                pos_data += n_chars_o;
                bool bit = key_col->get_null_bit(key_row);
                new_key_col->set_null_bit(j, bit);
            }
        }
        out_table->columns.push_back(std::move(new_key_col));
    }
}
