// Copyright (C) 2023 Bodo Inc. All rights reserved.
#include "_groupby_update.h"
#include "_array_operations.h"
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_distributed.h"
#include "_groupby_common.h"
#include "_groupby_do_apply_to_column.h"
#include "_groupby_hashing.h"

/**
 * The file contains the aggregate functions that are used
 * for the update step of groupby but are too complicated to
 * be inlined.
 */

// Remapping Update -> Combine for when a local update has occurred

// this mapping is used by BasicColSet operations to know what combine (i.e.
// step (c)) function to use for a given aggregation function
static UNORD_MAP_CONTAINER<int, int> combine_funcs = {
    {Bodo_FTypes::size, Bodo_FTypes::sum},
    {Bodo_FTypes::sum, Bodo_FTypes::sum},
    {Bodo_FTypes::count, Bodo_FTypes::sum},
    {Bodo_FTypes::mean, Bodo_FTypes::sum},  // sum totals and counts
    {Bodo_FTypes::min, Bodo_FTypes::min},
    {Bodo_FTypes::max, Bodo_FTypes::max},
    {Bodo_FTypes::prod, Bodo_FTypes::prod},
    {Bodo_FTypes::first, Bodo_FTypes::first},
    {Bodo_FTypes::last, Bodo_FTypes::last},
    {Bodo_FTypes::nunique, Bodo_FTypes::sum},  // used in nunique_mode = 2
    {Bodo_FTypes::boolor_agg, Bodo_FTypes::boolor_agg},
    {Bodo_FTypes::booland_agg, Bodo_FTypes::booland_agg},
    {Bodo_FTypes::bitor_agg, Bodo_FTypes::bitor_agg},
    {Bodo_FTypes::bitand_agg, Bodo_FTypes::bitand_agg},
    {Bodo_FTypes::bitxor_agg, Bodo_FTypes::bitxor_agg},
    {Bodo_FTypes::count_if, Bodo_FTypes::sum}};

int get_combine_func(int update_ftype) { return combine_funcs[update_ftype]; }

// Cumulative OPs

/**
 * The cumulative_computation function. It uses the symbolic information
 * to compute the cumsum/cumprod/cummin/cummax
 *
 * @param[in] arr: input array, string array
 * @param[out] out_arr: output array, regular string array
 * @param[in] grp_info: groupby information
 * @param[in] ftype: THe function type.
 * @param[in] skipna: Whether to skip NA values.
 */
template <typename T, int dtype>
void cumulative_computation_T(std::shared_ptr<array_info> arr,
                              std::shared_ptr<array_info> out_arr,
                              grouping_info const& grp_info,
                              int32_t const& ftype, bool const& skipna) {
    size_t num_group = grp_info.group_to_first_row.size();
    if (arr->arr_type == bodo_array_type::STRING ||
        arr->arr_type == bodo_array_type::LIST_STRING ||
        arr->arr_type == bodo_array_type::DICT) {
        throw std::runtime_error(
            "There is no cumulative operation for the string or list string "
            "case");
    }
    auto cum_computation = [&](auto const& get_entry,
                               auto const& set_entry) -> void {
        for (size_t igrp = 0; igrp < num_group; igrp++) {
            int64_t i = grp_info.group_to_first_row[igrp];
            T initVal = 0;
            if (ftype == Bodo_FTypes::cumsum) {
                initVal = 0;
            } else if (ftype == Bodo_FTypes::cummin) {
                initVal = std::numeric_limits<T>::max();
            } else if (ftype == Bodo_FTypes::cummax) {
                initVal = std::numeric_limits<T>::min();
            } else if (ftype == Bodo_FTypes::cumprod) {
                initVal = 1;
            }
            std::pair<bool, T> ePair{false, initVal};
            while (true) {
                if (i == -1) {
                    break;
                }
                std::pair<bool, T> fPair = get_entry(i);
                if (fPair.first) {  // the value is a NaN.
                    if (skipna) {
                        set_entry(i, fPair);
                    } else {
                        ePair = fPair;
                        set_entry(i, ePair);
                    }
                } else {  // The value is a normal one.
                    if (ftype == Bodo_FTypes::cumsum)
                        ePair.second += fPair.second;
                    if (ftype == Bodo_FTypes::cumprod)
                        ePair.second *= fPair.second;
                    if (ftype == Bodo_FTypes::cummin)
                        ePair.second = std::min(ePair.second, fPair.second);
                    if (ftype == Bodo_FTypes::cummax)
                        ePair.second = std::max(ePair.second, fPair.second);
                    set_entry(i, ePair);
                }
                i = grp_info.next_row_in_group[i];
            }
        }
        T eVal_nan = GetTentry<T>(
            RetrieveNaNentry((Bodo_CTypes::CTypeEnum)dtype).data());
        std::pair<bool, T> pairNaN{true, eVal_nan};
        for (auto& idx_miss : grp_info.list_missing) {
            set_entry(idx_miss, pairNaN);
        }
    };

    if (arr->arr_type == bodo_array_type::NUMPY) {
        if (dtype == Bodo_CTypes::DATETIME || dtype == Bodo_CTypes::TIMEDELTA) {
            cum_computation(
                [=](int64_t pos) -> std::pair<bool, T> {
                    // in DATETIME/TIMEDELTA case, the types is necessarily
                    // int64_t
                    T eVal = arr->at<T>(pos);
                    bool isna = (eVal == std::numeric_limits<T>::min());
                    return {isna, eVal};
                },
                [=](int64_t pos, std::pair<bool, T> const& ePair) -> void {
                    if (ePair.first)
                        out_arr->at<T>(pos) = std::numeric_limits<T>::min();
                    else
                        out_arr->at<T>(pos) = ePair.second;
                });
        } else {
            cum_computation(
                [=](int64_t pos) -> std::pair<bool, T> {
                    T eVal = arr->at<T>(pos);
                    bool isna = isnan_alltype<T, dtype>(eVal);
                    return {isna, eVal};
                },
                [=](int64_t pos, std::pair<bool, T> const& ePair) -> void {
                    out_arr->at<T>(pos) = ePair.second;
                });
        }
    }
    if (arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        cum_computation(
            [=](int64_t pos) -> std::pair<bool, T> {
                return {!arr->get_null_bit(pos), arr->at<T>(pos)};
            },
            [=](int64_t pos, std::pair<bool, T> const& ePair) -> void {
                out_arr->set_null_bit(pos, !ePair.first);
                out_arr->at<T>(pos) = ePair.second;
            });
    }
}

/**
 * @brief Perform cumulative computation on list of string columns.
 * We define a helper function get_entry(i) that returns the
 * null_bit and list value of row i and then within each list returns
 * of pair of the list entry value and if the entry is null. We then iterate
 * through each group to calculate the cumulative sums, and the intermediate
 * result for each row is stored in a temporary vector null_bit_val_vec.
 *
 * @param[in] arr: input array, list of string array
 * @param[out] out_arr: output array, list of string array
 * @param[in] grp_info: groupby information
 * @param[in] ftype: for list of strings only cumsum is supported
 * @param[in] skipna: Whether to skip NA values.
 */
void cumulative_computation_list_string(std::shared_ptr<array_info> arr,
                                        std::shared_ptr<array_info> out_arr,
                                        grouping_info const& grp_info,
                                        int32_t const& ftype,
                                        bool const& skipna) {
    if (ftype != Bodo_FTypes::cumsum) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "So far only cumulative sums for list-strings");
    }
    int64_t n = arr->length;
    using T = std::pair<bool, bodo::vector<std::pair<std::string, bool>>>;
    bodo::vector<T> null_bit_val_vec(n);
    uint8_t* null_bitmask = (uint8_t*)arr->null_bitmask();
    uint8_t* sub_null_bitmask = (uint8_t*)arr->sub_null_bitmask();
    char* data = arr->data1();
    offset_t* data_offsets = (offset_t*)arr->data2();
    offset_t* index_offsets = (offset_t*)arr->data3();
    auto get_entry = [&](int64_t i) -> T {
        bool isna = !GetBit(null_bitmask, i);
        offset_t start_idx_offset = index_offsets[i];
        offset_t end_idx_offset = index_offsets[i + 1];
        bodo::vector<std::pair<std::string, bool>> LEnt;
        for (offset_t idx = start_idx_offset; idx < end_idx_offset; idx++) {
            offset_t str_len = data_offsets[idx + 1] - data_offsets[idx];
            offset_t start_data_offset = data_offsets[idx];
            bool bit = GetBit(sub_null_bitmask, idx);
            std::string val(&data[start_data_offset], str_len);
            std::pair<std::string, bool> eEnt = {val, bit};
            LEnt.push_back(eEnt);
        }
        return {isna, LEnt};
    };
    size_t num_group = grp_info.group_to_first_row.size();
    for (size_t igrp = 0; igrp < num_group; igrp++) {
        int64_t i = grp_info.group_to_first_row[igrp];
        T ePair{false, {}};
        while (true) {
            if (i == -1) {
                break;
            }
            T fPair = get_entry(i);
            if (fPair.first) {  // the value is a NaN.
                if (skipna) {
                    null_bit_val_vec[i] = fPair;
                } else {
                    ePair = fPair;
                    null_bit_val_vec[i] = ePair;
                }
            } else {  // The value is a normal one.
                for (auto& eStr : fPair.second)
                    ePair.second.push_back(eStr);
                null_bit_val_vec[i] = ePair;
            }
            i = grp_info.next_row_in_group[i];
        }
    }
    T pairNaN{true, {}};
    for (auto& idx_miss : grp_info.list_missing) {
        null_bit_val_vec[idx_miss] = pairNaN;
    }
    //
    size_t n_bytes = (n + 7) >> 3;
    bodo::vector<uint8_t> Vmask(n_bytes, 0);
    bodo::vector<bodo::vector<std::pair<std::string, bool>>> ListListPair(n);
    for (int i = 0; i < n; i++) {
        SetBitTo(Vmask.data(), i, !null_bit_val_vec[i].first);
        ListListPair[i] = null_bit_val_vec[i].second;
    }
    std::shared_ptr<array_info> new_out_col =
        create_list_string_array(Vmask, ListListPair);
    *out_arr = std::move(*new_out_col);
}

/**
 * @brief Perform cumulative computation on string columns.
 * We define a helper function get_entry(i) that returns the
 * null_bit and string value of row i. We then iterate through each group to
 * calculate the cumulative sums, and the intermediate result for each row is
 * stored in a temporary vector null_bit_val_vec.
 *
 * @param[in] arr: input array, string array
 * @param[out] out_arr: output array, regular string array
 * @param[in] grp_info: groupby information
 * @param[in] ftype: for string only cumsum is supported
 * @param[in] skipna: Whether to skip NA values.
 */
void cumulative_computation_string(std::shared_ptr<array_info> arr,
                                   std::shared_ptr<array_info> out_arr,
                                   grouping_info const& grp_info,
                                   int32_t const& ftype, bool const& skipna) {
    if (ftype != Bodo_FTypes::cumsum) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "So far only cumulative sums for strings");
    }
    int64_t n = arr->length;
    using T = std::pair<bool, std::string>;
    bodo::vector<T> null_bit_val_vec(n);
    uint8_t* null_bitmask = (uint8_t*)arr->null_bitmask();
    char* data = arr->data1();
    offset_t* offsets = (offset_t*)arr->data2();
    auto get_entry = [&](int64_t i) -> T {
        bool isna = !GetBit(null_bitmask, i);
        offset_t start_offset = offsets[i];
        offset_t end_offset = offsets[i + 1];
        offset_t len = end_offset - start_offset;
        std::string val(&data[start_offset], len);
        return {isna, val};
    };
    size_t num_group = grp_info.group_to_first_row.size();
    for (size_t igrp = 0; igrp < num_group; igrp++) {
        int64_t i = grp_info.group_to_first_row[igrp];
        T ePair{false, ""};
        while (true) {
            if (i == -1) {
                break;
            }
            T fPair = get_entry(i);
            if (fPair.first) {  // the value is a NaN.
                if (skipna) {
                    null_bit_val_vec[i] = fPair;
                } else {
                    ePair = fPair;
                    null_bit_val_vec[i] = ePair;
                }
            } else {  // The value is a normal one.
                ePair.second += fPair.second;
                null_bit_val_vec[i] = ePair;
            }
            i = grp_info.next_row_in_group[i];
        }
    }
    T pairNaN{true, ""};
    for (auto& idx_miss : grp_info.list_missing) {
        null_bit_val_vec[idx_miss] = pairNaN;
    }
    // Now writing down in the array.
    size_t n_bytes = (n + 7) >> 3;
    bodo::vector<uint8_t> Vmask(n_bytes, 0);
    bodo::vector<std::string> ListString(n);
    for (int64_t i = 0; i < n; i++) {
        SetBitTo(Vmask.data(), i, !null_bit_val_vec[i].first);
        ListString[i] = null_bit_val_vec[i].second;
    }
    std::shared_ptr<array_info> new_out_col =
        create_string_array(out_arr->dtype, Vmask, ListString);
    *out_arr = std::move(*new_out_col);
}

/**
 * @brief Perform cumulative computation on dictionary encoded columns.
 * The function follows the same logic as that for the regular strings. More
 * specifically, we define a helper function get_entry(i) that returns the
 * null_bit and string value of row i. We then iterate through each group to
 * calculate the cumulative sums, and the intermediate result for each row is
 * stored in a temporary vector null_bit_val_vec. We choose to return a regular
 * string array instead of a dictionary-encoded one mostly because the return
 * array is likely to have a large cardinality and wouldn't benefit from
 * dictionary encoding.
 *
 * @param[in] arr: input array, dictionary encoded
 * @param[out] out_arr: output array, regulat string array
 * @param[in] grp_info: groupby information
 * @param[in] ftype: for dictionary encoded strings, only cumsum is supported
 * @param[in] skipna: Whether to skip NA values.
 */
void cumulative_computation_dict_encoded_string(
    std::shared_ptr<array_info> arr, std::shared_ptr<array_info> out_arr,
    grouping_info const& grp_info, int32_t const& ftype, bool const& skipna) {
    if (ftype != Bodo_FTypes::cumsum) {
        Bodo_PyErr_SetString(
            PyExc_RuntimeError,
            "So far only cumulative sums for dictionary-encoded strings");
    }
    int64_t n = arr->length;
    using T = std::pair<bool, std::string>;
    bodo::vector<T> null_bit_val_vec(n);  // a temporary vector that stores the
                                          // null bit and value for each row
    uint8_t* null_bitmask = (uint8_t*)arr->child_arrays[1]->null_bitmask();
    char* data = arr->child_arrays[0]->data1();
    offset_t* offsets = (offset_t*)arr->child_arrays[0]->data2();
    auto get_entry = [&](int64_t i) -> T {
        bool isna = !GetBit(null_bitmask, i);
        if (isna) {
            return {isna, ""};
        }
        int32_t dict_ind = ((int32_t*)arr->child_arrays[1]->data1())[i];
        offset_t start_offset = offsets[dict_ind];
        offset_t end_offset = offsets[dict_ind + 1];
        offset_t len = end_offset - start_offset;
        std::string val(&data[start_offset], len);
        return {isna, val};
    };
    size_t num_group = grp_info.group_to_first_row.size();
    for (size_t igrp = 0; igrp < num_group; igrp++) {
        int64_t i = grp_info.group_to_first_row[igrp];
        T ePair{false, ""};
        while (true) {
            if (i == -1) {
                break;
            }
            T fPair = get_entry(i);
            if (fPair.first) {  // the value is a NaN.
                if (skipna) {
                    null_bit_val_vec[i] = fPair;
                } else {
                    ePair = fPair;
                    null_bit_val_vec[i] = ePair;
                }
            } else {  // The value is a normal one.
                ePair.second += fPair.second;
                null_bit_val_vec[i] = ePair;
            }
            i = grp_info.next_row_in_group[i];
        }
    }
    T pairNaN{true, ""};
    for (auto& idx_miss : grp_info.list_missing) {
        null_bit_val_vec[idx_miss] = pairNaN;
    }
    // Now writing down in the array.
    size_t n_bytes = (n + 7) >> 3;
    bodo::vector<uint8_t> Vmask(n_bytes, 0);
    bodo::vector<std::string> ListString(n);
    for (int64_t i = 0; i < n; i++) {
        SetBitTo(Vmask.data(), i, !null_bit_val_vec[i].first);
        ListString[i] = null_bit_val_vec[i].second;
    }
    std::shared_ptr<array_info> new_out_col =
        create_string_array(out_arr->dtype, Vmask, ListString);
    *out_arr = std::move(*new_out_col);
}

void cumulative_computation(std::shared_ptr<array_info> arr,
                            std::shared_ptr<array_info> out_arr,
                            grouping_info const& grp_info, int32_t const& ftype,
                            bool const& skipna) {
    Bodo_CTypes::CTypeEnum dtype = arr->dtype;
    if (arr->arr_type == bodo_array_type::STRING) {
        return cumulative_computation_string(arr, out_arr, grp_info, ftype,
                                             skipna);
    } else if (arr->arr_type == bodo_array_type::DICT) {
        return cumulative_computation_dict_encoded_string(
            arr, out_arr, grp_info, ftype, skipna);
    } else if (arr->arr_type == bodo_array_type::LIST_STRING) {
        return cumulative_computation_list_string(arr, out_arr, grp_info, ftype,
                                                  skipna);
    } else {
        switch (dtype) {
            case Bodo_CTypes::INT8:
                return cumulative_computation_T<int8_t, Bodo_CTypes::INT8>(
                    arr, out_arr, grp_info, ftype, skipna);
            case Bodo_CTypes::UINT8:
                return cumulative_computation_T<uint8_t, Bodo_CTypes::UINT8>(
                    arr, out_arr, grp_info, ftype, skipna);
            case Bodo_CTypes::INT16:
                return cumulative_computation_T<int16_t, Bodo_CTypes::INT16>(
                    arr, out_arr, grp_info, ftype, skipna);
            case Bodo_CTypes::UINT16:
                return cumulative_computation_T<uint16_t, Bodo_CTypes::UINT16>(
                    arr, out_arr, grp_info, ftype, skipna);
            case Bodo_CTypes::INT32:
                return cumulative_computation_T<int32_t, Bodo_CTypes::INT32>(
                    arr, out_arr, grp_info, ftype, skipna);
            case Bodo_CTypes::UINT32:
                return cumulative_computation_T<uint32_t, Bodo_CTypes::UINT32>(
                    arr, out_arr, grp_info, ftype, skipna);
            case Bodo_CTypes::INT64:
                return cumulative_computation_T<int64_t, Bodo_CTypes::INT64>(
                    arr, out_arr, grp_info, ftype, skipna);
            case Bodo_CTypes::UINT64:
                return cumulative_computation_T<uint64_t, Bodo_CTypes::UINT64>(
                    arr, out_arr, grp_info, ftype, skipna);
            case Bodo_CTypes::FLOAT32:
                return cumulative_computation_T<float, Bodo_CTypes::FLOAT32>(
                    arr, out_arr, grp_info, ftype, skipna);
            case Bodo_CTypes::FLOAT64:
                return cumulative_computation_T<double, Bodo_CTypes::FLOAT64>(
                    arr, out_arr, grp_info, ftype, skipna);
            case Bodo_CTypes::DATE:
                return cumulative_computation_T<int32_t, Bodo_CTypes::DATE>(
                    arr, out_arr, grp_info, ftype, skipna);
            case Bodo_CTypes::TIME:
                return cumulative_computation_T<int64_t, Bodo_CTypes::TIME>(
                    arr, out_arr, grp_info, ftype, skipna);
            case Bodo_CTypes::DATETIME:
                return cumulative_computation_T<int64_t, Bodo_CTypes::DATETIME>(
                    arr, out_arr, grp_info, ftype, skipna);
            case Bodo_CTypes::TIMEDELTA:
                return cumulative_computation_T<int64_t,
                                                Bodo_CTypes::TIMEDELTA>(
                    arr, out_arr, grp_info, ftype, skipna);
            default:
                throw std::runtime_error(
                    "Unsupported dtype for cumulative operations: " +
                    GetDtype_as_string(dtype));
        }
    }
}

// HEAD

void head_computation(std::shared_ptr<array_info> arr,
                      std::shared_ptr<array_info> out_arr,
                      const bodo::vector<int64_t>& row_list) {
    std::shared_ptr<array_info> updated_col =
        RetrieveArray_SingleColumn(std::move(arr), row_list);
    *out_arr = std::move(*updated_col);
}

// NGROUP

void ngroup_computation(std::shared_ptr<array_info> arr,
                        std::shared_ptr<array_info> out_arr,
                        grouping_info const& grp_info, bool is_parallel) {
    //
    size_t num_group = grp_info.group_to_first_row.size();
    int64_t start_ngroup = 0;
    if (is_parallel) {
        MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::INT64);
        MPI_Exscan(&num_group, &start_ngroup, 1, mpi_typ, MPI_SUM,
                   MPI_COMM_WORLD);
    }
    for (size_t i = 0; i < arr->length; i++) {
        int64_t i_grp = grp_info.row_to_group[i];
        if (i_grp != -1) {
            int64_t val = i_grp + start_ngroup;
            getv<int64_t>(out_arr, i) = val;
        }
    }
}

// MEDIAN

void median_computation(std::shared_ptr<array_info> arr,
                        std::shared_ptr<array_info> out_arr,
                        grouping_info const& grp_info, bool const& skipna,
                        bool const use_sql_rules) {
    size_t num_group = grp_info.group_to_first_row.size();
    size_t siztype = numpy_item_size[arr->dtype];
    std::string error_msg = std::string("There is no median for the ") +
                            std::string(GetDtype_as_string(arr->dtype));
    if (arr->arr_type == bodo_array_type::STRING ||
        arr->arr_type == bodo_array_type::LIST_STRING) {
        Bodo_PyErr_SetString(PyExc_RuntimeError, error_msg.c_str());
        return;
    }
    if (arr->dtype == Bodo_CTypes::DATE || arr->dtype == Bodo_CTypes::TIME ||
        arr->dtype == Bodo_CTypes::DATETIME ||
        arr->dtype == Bodo_CTypes::TIMEDELTA) {
        Bodo_PyErr_SetString(PyExc_RuntimeError, error_msg.c_str());
        return;
    }
    auto median_operation = [&](auto const& isnan_entry) -> void {
        for (size_t igrp = 0; igrp < num_group; igrp++) {
            int64_t i = grp_info.group_to_first_row[igrp];
            std::vector<double> ListValue;
            bool HasNaN = false;
            while (true) {
                if (i == -1) {
                    break;
                }
                if (!isnan_entry(i)) {
                    char* ptr = arr->data1() + i * siztype;
                    double eVal = GetDoubleEntry(arr->dtype, ptr);
                    ListValue.emplace_back(eVal);
                } else {
                    if (!skipna) {
                        HasNaN = true;
                        break;
                    }
                }
                i = grp_info.next_row_in_group[i];
            }
            auto GetKthValue = [&](size_t const& pos) -> double {
                std::nth_element(ListValue.begin(), ListValue.begin() + pos,
                                 ListValue.end());
                return ListValue[pos];
            };
            double valReturn = 0;
            // a group can be empty if it has all NaNs so output will be NaN
            // even if skipna=True
            if (HasNaN || ListValue.size() == 0) {
                // We always set the output to NA.
                out_arr->set_null_bit(igrp, false);
                continue;
            } else {
                size_t len = ListValue.size();
                size_t res = len % 2;
                if (res == 0) {
                    size_t kMid1 = len / 2;
                    size_t kMid2 = kMid1 - 1;
                    valReturn = (GetKthValue(kMid1) + GetKthValue(kMid2)) / 2;
                } else {
                    size_t kMid = len / 2;
                    valReturn = GetKthValue(kMid);
                }
            }
            // The output array is always a nullable float64 array
            out_arr->set_null_bit(igrp, true);
            out_arr->at<double>(igrp) = valReturn;
        }
    };
    if (arr->arr_type == bodo_array_type::NUMPY) {
        median_operation([=](size_t pos) -> bool {
            if (arr->dtype == Bodo_CTypes::FLOAT32) {
                return isnan(arr->at<float>(pos));
            }
            if (arr->dtype == Bodo_CTypes::FLOAT64) {
                return isnan(arr->at<double>(pos));
            }
            return false;
        });
    }
    if (arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        median_operation(
            [=](size_t pos) -> bool { return !arr->get_null_bit(pos); });
    }
}

// SHIFT

void shift_computation(std::shared_ptr<array_info> arr,
                       std::shared_ptr<array_info> out_arr,
                       grouping_info const& grp_info, int64_t const& periods) {
    size_t num_rows = grp_info.row_to_group.size();
    size_t num_groups = grp_info.num_groups;
    int64_t tmp_periods = periods;
    // 1. Shift operation taken from pandas
    // https://github.com/pandas-dev/pandas/blob/master/pandas/_libs/groupby.pyx#L293
    size_t ii, offset;
    int sign;
    // If periods<0, shift up (i.e. iterate backwards)
    if (periods < 0) {
        tmp_periods = -periods;
        offset = num_rows - 1;
        sign = -1;
    } else {
        offset = 0;
        sign = 1;
    }

    bodo::vector<int64_t> row_list(num_rows);
    if (tmp_periods == 0) {
        for (size_t i = 0; i < num_rows; i++) {
            row_list[i] = i;
        }
    } else {
        int64_t gid;       // group number
        int64_t cur_pos;   // new value for current row
        int64_t prev_val;  // previous value
        bodo::vector<int64_t> nrows_per_group(
            num_groups);  // array holding number of rows per group
        bodo::vector<std::vector<int64_t>> p_values(
            num_groups,
            std::vector<int64_t>(tmp_periods));  // 2d array holding most recent
                                                 // N=periods elements per group
        // For each row value, find if it should be NaN or it will get a value
        // that is N=periods away from it. It's a NaN if it's row_number <
        // periods, otherwise get it's new value from (row_number -/+ periods)
        for (size_t i = 0; i < num_rows; i++) {
            ii = offset + sign * i;
            gid = grp_info.row_to_group[ii];
            if (gid == -1) {
                row_list[ii] = -1;
                continue;
            }

            nrows_per_group[gid]++;
            cur_pos = nrows_per_group[gid] % tmp_periods;
            prev_val = p_values[gid][cur_pos];
            if (nrows_per_group[gid] > tmp_periods) {
                row_list[ii] = prev_val;
            } else {
                row_list[ii] = -1;
            }
            p_values[gid][cur_pos] = ii;
        }  // end-row_loop
    }
    // 2. Retrieve column and put it in update_cols
    std::shared_ptr<array_info> updated_col =
        RetrieveArray_SingleColumn(std::move(arr), row_list);
    *out_arr = std::move(*updated_col);
}

// Variance
void var_combine(const std::shared_ptr<array_info>& count_col_in,
                 const std::shared_ptr<array_info>& mean_col_in,
                 const std::shared_ptr<array_info>& m2_col_in,
                 const std::shared_ptr<array_info>& count_col_out,
                 const std::shared_ptr<array_info>& mean_col_out,
                 const std::shared_ptr<array_info>& m2_col_out,
                 grouping_info const& grp_info) {
    for (size_t i = 0; i < count_col_in->length; i++) {
        // Var always has null compute columns even
        // if there is an original numpy input. All arrays
        // will have the same null bit value so just check 1.
        if (count_col_in->get_null_bit(i)) {
            int64_t group_num = grp_info.row_to_group[i];
            uint64_t& count_a = getv<uint64_t>(count_col_out, group_num);
            uint64_t& count_b = getv<uint64_t>(count_col_in, i);
            // TODO: Can I delete this comment + condition
            // in the pivot case we can receive dummy values from other ranks
            // when combining the results (in this case with count == 0). This
            // is because the pivot case groups on index and creates n_pivot
            // columns, and so for each of its index values a rank will have
            // n_pivot fields, even if a rank does not have a particular (index,
            // pivot_value) pair
            if (count_b == 0) {
                continue;
            }
            double& mean_a = getv<double>(mean_col_out, group_num);
            double& mean_b = getv<double>(mean_col_in, i);
            double& m2_a = getv<double>(m2_col_out, group_num);
            double& m2_b = getv<double>(m2_col_in, i);
            uint64_t count = count_a + count_b;
            double delta = mean_b - mean_a;
            mean_a = (count_a * mean_a + count_b * mean_b) / count;
            m2_a = m2_a + m2_b + delta * delta * count_a * count_b / count;
            count_a = count;
            // Set all the null bits to true.
            count_col_out->set_null_bit(i, true);
            mean_col_out->set_null_bit(i, true);
            m2_col_out->set_null_bit(i, true);
        }
    }
}

// boolxor_agg
void boolxor_combine(const std::shared_ptr<array_info>& one_col_in,
                     const std::shared_ptr<array_info>& two_col_in,
                     const std::shared_ptr<array_info>& one_col_out,
                     const std::shared_ptr<array_info>& two_col_out,
                     grouping_info const& grp_info) {
    for (size_t i = 0; i < one_col_in->length; i++) {
        if (one_col_in->get_null_bit(i)) {
            int64_t group_num = grp_info.row_to_group[i];

            // Fetch the input data
            bool one_in = GetBit((uint8_t*)one_col_in->data1(), i);
            bool two_in = GetBit((uint8_t*)two_col_in->data1(), i);

            // Get the existing group values
            bool one_out = GetBit((uint8_t*)one_col_out->data1(), group_num);
            bool two_out = GetBit((uint8_t*)two_col_out->data1(), group_num);
            two_out = two_out || two_in || (one_in && one_out);

            // Update the group values.
            one_out = one_out || one_in;
            SetBitTo((uint8_t*)one_col_out->data1(), group_num, one_out);
            SetBitTo((uint8_t*)two_col_out->data1(), group_num, two_out);
            // Set all the null bits to true.
            one_col_out->set_null_bit(group_num, true);
            two_col_out->set_null_bit(group_num, true);
        }
    }
}

// Skew
void skew_combine(const std::shared_ptr<array_info>& count_col_in,
                  const std::shared_ptr<array_info>& m1_col_in,
                  const std::shared_ptr<array_info>& m2_col_in,
                  const std::shared_ptr<array_info>& m3_col_in,
                  const std::shared_ptr<array_info>& count_col_out,
                  const std::shared_ptr<array_info>& m1_col_out,
                  const std::shared_ptr<array_info>& m2_col_out,
                  const std::shared_ptr<array_info>& m3_col_out,
                  grouping_info const& grp_info) {
    for (size_t i = 0; i < count_col_in->length; i++) {
        if (count_col_in->get_null_bit(i)) {
            int64_t group_num = grp_info.row_to_group[i];
            uint64_t& count_a = getv<uint64_t>(count_col_out, group_num);
            uint64_t& count_b = getv<uint64_t>(count_col_in, i);
            if (count_b == 0) {
                continue;
            }
            double& m1_a = getv<double>(m1_col_out, group_num);
            double& m1_b = getv<double>(m1_col_in, i);
            double& m2_a = getv<double>(m2_col_out, group_num);
            double& m2_b = getv<double>(m2_col_in, i);
            double& m3_a = getv<double>(m3_col_out, group_num);
            double& m3_b = getv<double>(m3_col_in, i);
            count_a += count_b;
            m1_a += m1_b;
            m2_a += m2_b;
            m3_a += m3_b;

            // Set all the null bits to true.
            count_col_out->set_null_bit(group_num, true);
            m1_col_out->set_null_bit(group_num, true);
            m2_col_out->set_null_bit(group_num, true);
            m3_col_out->set_null_bit(group_num, true);
        }
    }
}

// Kurtosis
void kurt_combine(const std::shared_ptr<array_info>& count_col_in,
                  const std::shared_ptr<array_info>& m1_col_in,
                  const std::shared_ptr<array_info>& m2_col_in,
                  const std::shared_ptr<array_info>& m3_col_in,
                  const std::shared_ptr<array_info>& m4_col_in,
                  const std::shared_ptr<array_info>& count_col_out,
                  const std::shared_ptr<array_info>& m1_col_out,
                  const std::shared_ptr<array_info>& m2_col_out,
                  const std::shared_ptr<array_info>& m3_col_out,
                  const std::shared_ptr<array_info>& m4_col_out,
                  grouping_info const& grp_info) {
    for (size_t i = 0; i < count_col_in->length; i++) {
        if (count_col_in->get_null_bit(i)) {
            int64_t group_num = grp_info.row_to_group[i];
            uint64_t& count_a = getv<uint64_t>(count_col_out, group_num);
            uint64_t& count_b = getv<uint64_t>(count_col_in, i);
            if (count_b == 0) {
                continue;
            }
            double& m1_a = getv<double>(m1_col_out, group_num);
            double& m1_b = getv<double>(m1_col_in, i);
            double& m2_a = getv<double>(m2_col_out, group_num);
            double& m2_b = getv<double>(m2_col_in, i);
            double& m3_a = getv<double>(m3_col_out, group_num);
            double& m3_b = getv<double>(m3_col_in, i);
            double& m4_a = getv<double>(m4_col_out, group_num);
            double& m4_b = getv<double>(m4_col_in, i);
            count_a += count_b;
            m1_a += m1_b;
            m2_a += m2_b;
            m3_a += m3_b;
            m4_a += m4_b;

            // Set all the null bits to true.
            count_col_out->set_null_bit(group_num, true);
            m1_col_out->set_null_bit(group_num, true);
            m2_col_out->set_null_bit(group_num, true);
            m3_col_out->set_null_bit(group_num, true);
            m4_col_out->set_null_bit(group_num, true);
        }
    }
}

// NUNIQUE
void nunique_computation(std::shared_ptr<array_info> arr,
                         std::shared_ptr<array_info> out_arr,
                         grouping_info const& grp_info, bool const& dropna,
                         bool const& is_parallel) {
    tracing::Event ev("nunique_computation", is_parallel);
    size_t num_group = grp_info.group_to_first_row.size();
    if (num_group == 0) {
        return;
    }
    // Note: Dictionary encoded is supported because we just
    // call nunique on the indices. See update that converts
    // the dict array to its indices. This is tested with
    // test_nunique_dict.
    if (arr->arr_type == bodo_array_type::NUMPY ||
        arr->arr_type == bodo_array_type::CATEGORICAL) {
        /**
         * Check if a pointer points to a NaN or not
         *
         * @param the char* pointer
         * @param the type of the data in input
         */
        auto isnan_entry = [&](char* ptr) -> bool {
            if (arr->dtype == Bodo_CTypes::FLOAT32) {
                float* ptr_f = (float*)ptr;
                return isnan(*ptr_f);
            }
            if (arr->dtype == Bodo_CTypes::FLOAT64) {
                double* ptr_d = (double*)ptr;
                return isnan(*ptr_d);
            }
            if (arr->dtype == Bodo_CTypes::DATETIME ||
                arr->dtype == Bodo_CTypes::TIMEDELTA) {
                int64_t* ptr_i = (int64_t*)ptr;
                return *ptr_i == std::numeric_limits<int64_t>::min();
            }
            if (arr->arr_type == bodo_array_type::CATEGORICAL) {
                return isnan_categorical_ptr(arr->dtype, ptr);
            }
            return false;
        };
        const size_t siztype = numpy_item_size[arr->dtype];
        const uint32_t seed = SEED_HASH_CONTAINER;

        HashNuniqueComputationNumpyOrNullableIntBool hash_fct{arr, siztype,
                                                              seed};
        KeyEqualNuniqueComputationNumpyOrNullableIntBool equal_fct{arr,
                                                                   siztype};
        bodo::unord_set_container<
            int64_t, HashNuniqueComputationNumpyOrNullableIntBool,
            KeyEqualNuniqueComputationNumpyOrNullableIntBool>
            eset({}, hash_fct, equal_fct);
        eset.reserve(double(arr->length) / num_group);  // NOTE: num_group > 0
        eset.max_load_factor(UNORDERED_MAP_MAX_LOAD_FACTOR);

        for (size_t igrp = 0; igrp < num_group; igrp++) {
            int64_t i = grp_info.group_to_first_row[igrp];
            // with nunique mode=2 some groups might not be present in the
            // nunique table
            if (i < 0) {
                continue;
            }
            eset.clear();
            bool HasNullRow = false;
            while (true) {
                char* ptr = arr->data1() + (i * siztype);
                if (!isnan_entry(ptr)) {
                    eset.insert(i);
                } else {
                    HasNullRow = true;
                }
                i = grp_info.next_row_in_group[i];
                if (i == -1)
                    break;
            }
            int64_t size = eset.size();
            if (HasNullRow && !dropna)
                size++;
            out_arr->at<int64_t>(igrp) = size;
        }
    } else if (arr->arr_type == bodo_array_type::LIST_STRING) {
        offset_t* in_index_offsets = (offset_t*)arr->data3();
        offset_t* in_data_offsets = (offset_t*)arr->data2();
        uint8_t* sub_null_bitmask = (uint8_t*)arr->sub_null_bitmask();
        const uint32_t seed = SEED_HASH_CONTAINER;

        HashNuniqueComputationListString hash_fct{arr, in_index_offsets,
                                                  in_data_offsets, seed};
        KeyEqualNuniqueComputationListString equal_fct{
            arr, in_index_offsets, in_data_offsets, sub_null_bitmask, seed};
        bodo::unord_set_container<int64_t, HashNuniqueComputationListString,
                                  KeyEqualNuniqueComputationListString>
            eset({}, hash_fct, equal_fct);
        eset.reserve(double(arr->length) / num_group);  // NOTE: num_group > 0
        eset.max_load_factor(UNORDERED_MAP_MAX_LOAD_FACTOR);

        for (size_t igrp = 0; igrp < num_group; igrp++) {
            int64_t i = grp_info.group_to_first_row[igrp];
            // with nunique mode=2 some groups might not be present in the
            // nunique table
            if (i < 0) {
                continue;
            }
            eset.clear();
            bool HasNullRow = false;
            while (true) {
                if (arr->get_null_bit(i)) {
                    eset.insert(i);
                } else {
                    HasNullRow = true;
                }
                i = grp_info.next_row_in_group[i];
                if (i == -1)
                    break;
            }
            int64_t size = eset.size();
            if (HasNullRow && !dropna)
                size++;
            out_arr->at<int64_t>(igrp) = size;
        }
    } else if (arr->arr_type == bodo_array_type::STRING) {
        offset_t* in_offsets = (offset_t*)arr->data2();
        const uint32_t seed = SEED_HASH_CONTAINER;

        HashNuniqueComputationString hash_fct{arr, in_offsets, seed};
        KeyEqualNuniqueComputationString equal_fct{arr, in_offsets};
        bodo::unord_set_container<int64_t, HashNuniqueComputationString,
                                  KeyEqualNuniqueComputationString>
            eset({}, hash_fct, equal_fct);
        eset.reserve(double(arr->length) / num_group);  // NOTE: num_group > 0
        eset.max_load_factor(UNORDERED_MAP_MAX_LOAD_FACTOR);

        for (size_t igrp = 0; igrp < num_group; igrp++) {
            int64_t i = grp_info.group_to_first_row[igrp];
            // with nunique mode=2 some groups might not be present in the
            // nunique table
            if (i < 0) {
                continue;
            }
            eset.clear();
            bool HasNullRow = false;
            while (true) {
                if (arr->get_null_bit(i)) {
                    eset.insert(i);
                } else {
                    HasNullRow = true;
                }
                i = grp_info.next_row_in_group[i];
                if (i == -1)
                    break;
            }
            int64_t size = eset.size();
            if (HasNullRow && !dropna)
                size++;
            out_arr->at<int64_t>(igrp) = size;
        }
    } else if (arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        const size_t siztype = numpy_item_size[arr->dtype];
        HashNuniqueComputationNumpyOrNullableIntBool hash_fct{arr, siztype};
        KeyEqualNuniqueComputationNumpyOrNullableIntBool equal_fct{arr,
                                                                   siztype};
        bodo::unord_set_container<
            int64_t, HashNuniqueComputationNumpyOrNullableIntBool,
            KeyEqualNuniqueComputationNumpyOrNullableIntBool>
            eset({}, hash_fct, equal_fct);
        eset.reserve(double(arr->length) / num_group);  // NOTE: num_group > 0
        eset.max_load_factor(UNORDERED_MAP_MAX_LOAD_FACTOR);

        for (size_t igrp = 0; igrp < num_group; igrp++) {
            int64_t i = grp_info.group_to_first_row[igrp];
            // with nunique mode=2 some groups might not be present in the
            // nunique table
            if (i < 0) {
                continue;
            }
            eset.clear();
            bool HasNullRow = false;
            while (true) {
                if (arr->get_null_bit(i)) {
                    eset.insert(i);
                } else {
                    HasNullRow = true;
                }
                i = grp_info.next_row_in_group[i];
                if (i == -1)
                    break;
            }
            int64_t size = eset.size();
            if (HasNullRow && !dropna)
                size++;
            out_arr->at<int64_t>(igrp) = size;
        }
    } else {
        throw std::runtime_error(
            "Unsupported array type encountered with nunique. Found type: " +
            GetArrType_as_string(arr->arr_type));
    }
}

// WINDOW

/**
 * min_row_number_filter is used to evaluate the following type
 * of expression in BodoSQL:
 *
 * row_number() over (partition by ... order by ...) == 1.
 *
 * This function creates a boolean array of all-false, then finds the indices
 * corresponding to the idxmin/idxmax of the orderby columns and sets those
 * indices to true. This implementaiton does so without sorting the array since
 * if no other window functions being calculated require sorting, then we can
 * find the idxmin/idxmax without bothering to sort the whole table seciton.
 *
 * @param[in] orderby_arrs: the columns used in the order by clause of the query
 * @param[in,out] out_arr: output array where the true values will be placed
 * @param[in] grp_info: groupby information
 * @param[in] asc_vect: vector indicating which of the orderby columns are
 * ascending
 * @param[in] na_pos_vect: vector indicating which of the orderby columns are
 * null first/last
 * @param[in] is_parallel: is the function being run in parallel?
 * @param[in] use_sql_rules: should initialization functions obey SQL semantics?
 */
void min_row_number_filter_window_computation_no_sort(
    std::vector<std::shared_ptr<array_info>>& orderby_arrs,
    std::shared_ptr<array_info> out_arr, grouping_info const& grp_info,
    std::vector<bool>& asc_vect, std::vector<bool>& na_pos_vect,
    bool is_parallel, bool use_sql_rules) {
    // To compute min_row_number_filter we want to find the
    // idxmin/idxmax based on the orderby columns. Then in the output
    // array those locations will have the value true. We have already
    // initialized all other locations to false.
    size_t num_groups = grp_info.num_groups;
    int64_t ftype;
    std::shared_ptr<array_info> idx_col;
    if (orderby_arrs.size() == 1) {
        // We generate an optimized and templated path for 1 column.
        std::shared_ptr<array_info> orderby_arr = orderby_arrs[0];
        bool asc = asc_vect[0];
        bool na_pos = na_pos_vect[0];
        bodo_array_type::arr_type_enum idx_arr_type;
        if (asc) {
            // The first value of an array in ascending order is the
            // min.
            if (na_pos) {
                ftype = Bodo_FTypes::idxmin;
                // We don't need null values for indices
                idx_arr_type = bodo_array_type::NUMPY;
            } else {
                ftype = Bodo_FTypes::idxmin_na_first;
                // We need null values to signal we found an NA
                // value.
                idx_arr_type = bodo_array_type::NULLABLE_INT_BOOL;
            }
        } else {
            // The first value of an array in descending order is the
            // max.
            if (na_pos) {
                ftype = Bodo_FTypes::idxmax;
                // We don't need null values for indices
                idx_arr_type = bodo_array_type::NUMPY;
            } else {
                ftype = Bodo_FTypes::idxmax_na_first;
                // We need null values to signal we found an NA
                // value.
                idx_arr_type = bodo_array_type::NULLABLE_INT_BOOL;
            }
        }
        // Allocate intermediate buffer to find the true element for
        // each group.
        idx_col = alloc_array(num_groups, 1, 1, idx_arr_type,
                              Bodo_CTypes::UINT64, 0, 0);
        // create array to store min/max value
        std::shared_ptr<array_info> data_col = alloc_array(
            num_groups, 1, 1, orderby_arr->arr_type, orderby_arr->dtype, 0, 0);
        // Initialize the index column. This is 0 initialized and will
        // not initialize the null values.
        aggfunc_output_initialize(idx_col, Bodo_FTypes::count, use_sql_rules);
        std::vector<std::shared_ptr<array_info>> aux_cols = {idx_col};
        // Initialize the min/max column
        if (ftype == Bodo_FTypes::idxmax ||
            ftype == Bodo_FTypes::idxmax_na_first) {
            aggfunc_output_initialize(data_col, Bodo_FTypes::max,
                                      use_sql_rules);
        } else {
            aggfunc_output_initialize(data_col, Bodo_FTypes::min,
                                      use_sql_rules);
        }
        // Compute the idxmin/idxmax
        do_apply_to_column(orderby_arr, data_col, aux_cols, grp_info, ftype);
    } else {
        ftype = Bodo_FTypes::idx_n_columns;
        // We don't need null for indices
        // We only allocate an index column.
        idx_col = alloc_array(num_groups, 1, 1, bodo_array_type::NUMPY,
                              Bodo_CTypes::UINT64, 0, 0);
        aggfunc_output_initialize(idx_col, Bodo_FTypes::count, use_sql_rules);
        // Call the idx_n_columns function path.
        idx_n_columns_apply(idx_col, orderby_arrs, asc_vect, na_pos_vect,
                            grp_info, ftype);
    }
    // Now we have the idxmin/idxmax in the idx_col for each group.
    // We need to set the corresponding indices in the final array to true.
    for (size_t group_idx = 0; group_idx < idx_col->length; group_idx++) {
        int64_t row_one_idx = getv<int64_t>(idx_col, group_idx);
        SetBitTo((uint8_t*)out_arr->data1(), row_one_idx, true);
    }
}

/**
 * Alternative implementaiton for min_row_number_filter if another window
 * computation already requires sorting the table: iterates through
 * the sorted groups and sets the corresponding row to true if the
 * current row belongs to a differnet group than the previous row.
 *
 * @param[in,out] out_arr: output array where the row numbers are stored
 * @param[in] sorted_groups: sorted array of group numbers
 * @param[in] sorted_idx: array mapping each index in the sorted table back
 * to its location in the original unsorted table
 */
void min_row_number_filter_window_computation_already_sorted(
    std::shared_ptr<array_info> out_arr,
    std::shared_ptr<array_info> sorted_groups,
    std::shared_ptr<array_info> sorted_idx) {
    int64_t prev_group = -1;
    for (int64_t i = 0; i < out_arr->length; i++) {
        int64_t curr_group = getv<int64_t>(sorted_groups, i);
        // If the current group is differne from the group of the previous row,
        // then this row is the row where the row number is 1
        if (curr_group != prev_group) {
            int64_t row_one_idx = getv<int64_t>(sorted_idx, i);
            SetBitTo((uint8_t*)out_arr->data1(), row_one_idx, true);
            prev_group = curr_group;
        }
    }
}

/**
 * Computes the BodoSQL window function ROW_NUMBER() on a subset of a table
 * containing complete partitions, where the rows are sorted first by group
 * (so each partition is clustered togeher) and then by the columns in the
 * orderby clause.
 *
 * The computaiton works by having a counter that starts at 1, increases by
 * 1 each row, and resets to 1 whenever a new group is reached.
 *
 * @param[in,out] out_arr: output array where the row numbers are stored
 * @param[in] sorted_groups: sorted array of group numbers
 * @param[in] sorted_idx: array mapping each index in the sorted table back
 * to its location in the original unsorted table
 */
void row_number_computation(std::shared_ptr<array_info> out_arr,
                            std::shared_ptr<array_info> sorted_groups,
                            std::shared_ptr<array_info> sorted_idx) {
    // Start the counter at 1, increase by 1 each time, and reset
    // the counter to 1 when we reach a new group
    int64_t prev_group = -1;
    int64_t row_num = 1;
    for (int64_t i = 0; i < out_arr->length; i++) {
        int64_t curr_group = getv<int64_t>(sorted_groups, i);
        if (curr_group != prev_group) {
            row_num = 1;
        } else {
            row_num++;
        }
        // Get the index in the output array.
        int64_t idx = getv<int64_t>(sorted_idx, i);
        getv<int64_t>(out_arr, idx) = row_num;
        // Set the prev group
        prev_group = curr_group;
    }
}

/** Returns whether the current row of orderby keys is distinct from
 * the previous row when performing a rank computation.
 *
 * @param[in] sorted_orderbys: the columns used to order the table
 * when performing a window computation.
 * @param[in] i: the row that is being queried to see if it is distinct
 * from the previous row.
 */
inline bool distinct_from_previous_row(
    std::vector<std::shared_ptr<array_info>> sorted_orderbys, int64_t i) {
    if (i == 0) {
        return true;
    }
    for (auto arr : sorted_orderbys) {
        if (!TestEqualColumn(arr, i, arr, i - 1, true)) {
            return true;
        }
    }
    return false;
}

/**
 * Perform the division step for a group once an entire group has had
 * its regular rank values calculated.
 *
 * @param[in,out] out_arr input value, and holds the result
 * @param[in] sorted_idx the array mapping sorted rows back to locaitons in
 * the original array
 * @param[in] rank_arr stores the regular rank values for each row
 * @param[in] group_start_idx the index in rank_arr where the current group
 * begins
 * @param[in] group_end_idx the index in rank_arr where the current group ends
 * @param[in] numerator_offset the amount to subtract from the regular rank
 * @param[in] denominator_offset the amount to subtract from the group size
 *
 * The formula for each row is:
 * out_arr[i] = (r - numerator_offset) / (n - denominator_offset)
 * where r is the rank and  n is the size of the group
 */
inline void rank_division_batch_update(std::shared_ptr<array_info> out_arr,
                                       std::shared_ptr<array_info> sorted_idx,
                                       std::shared_ptr<array_info> rank_arr,
                                       int64_t group_start_idx,
                                       int64_t group_end_idx,
                                       int64_t numerator_offset,
                                       int64_t denominator_offset) {
    // Special case: if the group has size below the offset, set the result to
    // zero
    int64_t group_size = group_end_idx - group_start_idx;
    if (group_size <= denominator_offset) {
        int64_t idx = getv<int64_t>(sorted_idx, group_start_idx);
        getv<double>(out_arr, idx) = 0.0;
    } else {
        // Otherwise, iterate through the entire group that just finished
        for (int64_t j = group_start_idx; j < group_end_idx; j++) {
            int64_t idx = getv<int64_t>(sorted_idx, j);
            getv<double>(out_arr, idx) =
                (static_cast<double>(getv<int64_t>(rank_arr, j)) -
                 numerator_offset) /
                (group_size - denominator_offset);
        }
    }
}

/**
 * Populate the rank values from tie_start_idx to tie_end_idx with
 * the current rank value using the rule all ties are brought upward.
 *
 * @param[in] rank_arr stores the regular rank values for each row
 * @param[in] group_start_idx the index in rank_arr where the current group
 * begins
 * @param[in] tie_start_idx the index in rank_arr where the tie group begins
 * @param[in] tie_end_idx the index in rank_arr where the tie group ends
 */
inline void rank_tie_upward_batch_update(std::shared_ptr<array_info> rank_arr,
                                         int64_t group_start_idx,
                                         int64_t tie_start_idx,
                                         int64_t tie_end_idx) {
    int64_t fill_value = tie_end_idx - group_start_idx;
    std::fill((int64_t*)(rank_arr->data1()) + tie_start_idx,
              (int64_t*)(rank_arr->data1()) + tie_end_idx, fill_value);
}

/**
 * Computes the BodoSQL window function RANK() on a subset of a table
 * containing complete partitions, where the rows are sorted first by group
 * (so each partition is clustered togeher) and then by the columns in the
 * orderby clause.
 *
 * The computaiton works by having a counter that starts at 1, increases
 * whenever the orderby columns change values, and resets to 1 whenever a new
 * group is reached.
 *
 * @param[in,out] out_arr: output array where the row numbers are stored
 * @param[in] sorted_groups: sorted array of group numbers
 * @param[in] sorted_orderbys the vector of sorted orderby columns, used to
 * keep track of when we have encountered a distinct row
 * @param[in] sorted_idx: array mapping each index in the sorted table back
 * to its location in the original unsorted table
 */
void rank_computation(std::shared_ptr<array_info> out_arr,
                      std::shared_ptr<array_info> sorted_groups,
                      std::vector<std::shared_ptr<array_info>> sorted_orderbys,
                      std::shared_ptr<array_info> sorted_idx) {
    // Start the counter at 1, snap upward each we encounter a row
    // that is distinct from the previous row, and reset the counter
    // to 1 when we reach a new group
    int64_t prev_group = -1;
    int64_t rank_val = 1;
    int64_t group_start_idx = 0;
    int64_t n = out_arr->length;
    for (int64_t i = 0; i < n; i++) {
        int64_t curr_group = getv<int64_t>(sorted_groups, i);
        if (curr_group != prev_group) {
            rank_val = 1;
            group_start_idx = i;
            // Set the prev group
            prev_group = curr_group;
        } else if (distinct_from_previous_row(sorted_orderbys, i)) {
            rank_val = i - group_start_idx + 1;
        }
        // Get the index in the output array.
        int64_t idx = getv<int64_t>(sorted_idx, i);
        getv<int64_t>(out_arr, idx) = rank_val;
    }
};

/**
 * Computes the BodoSQL window function DENSE_RANK() on a subset of a table
 * containing complete partitions, where the rows are sorted first by group
 * (so each partition is clustered togeher) and then by the columns in the
 * orderby clause.
 *
 * The computaiton works by having a counter that starts at 1, increases by 1
 * the orderby columns change values, and resets to 1 whenever a new group is
 * reached.
 *
 * @param[in,out] out_arr: output array where the row numbers are stored
 * @param[in] sorted_groups: sorted array of group numbers
 * @param[in] sorted_orderbys the vector of sorted orderby columns, used to
 * keep track of when we have encountered a distinct row
 * @param[in] sorted_idx: array mapping each index in the sorted table back
 * to its location in the original unsorted table
 */
void dense_rank_computation(
    std::shared_ptr<array_info> out_arr,
    std::shared_ptr<array_info> sorted_groups,
    std::vector<std::shared_ptr<array_info>> sorted_orderbys,
    std::shared_ptr<array_info> sorted_idx) {
    // Start the counter at 1, increase by 1 each we encounter a row
    // that is distinct from the previous row, and reset the counter
    // to 1 when we reach a new group
    int64_t prev_group = -1;
    int64_t rank_val = 1;
    int64_t n = out_arr->length;
    for (int64_t i = 0; i < n; i++) {
        int64_t curr_group = getv<int64_t>(sorted_groups, i);
        if (curr_group != prev_group) {
            rank_val = 1;
            // Set the prev group
            prev_group = curr_group;
        } else if (distinct_from_previous_row(sorted_orderbys, i)) {
            rank_val++;
        }
        // Get the index in the output array.
        int64_t idx = getv<int64_t>(sorted_idx, i);
        getv<int64_t>(out_arr, idx) = rank_val;
    }
};

/**
 * Computes the BodoSQL window function PERCENT_RANK() on a subset of a table
 * containing complete partitions, where the rows are sorted first by group
 * (so each partition is clustered togeher) and then by the columns in the
 * orderby clause.
 *
 * The computaiton works by calculating the regular rank for each row. Then,
 * after a group is finished, the percent rank for each row is calculated
 * via the formula (r-1)/(n-1) where r is the rank and n is the group size.
 *
 * @param[in,out] out_arr: output array where the row numbers are stored
 * @param[in] sorted_groups: sorted array of group numbers
 * @param[in] sorted_orderbys the vector of sorted orderby columns, used to
 * keep track of when we have encountered a distinct row
 * @param[in] sorted_idx: array mapping each index in the sorted table back
 * to its location in the original unsorted table
 */
void percent_rank_computation(
    std::shared_ptr<array_info> out_arr,
    std::shared_ptr<array_info> sorted_groups,
    std::vector<std::shared_ptr<array_info>> sorted_orderbys,
    std::shared_ptr<array_info> sorted_idx) {
    // Create an intermediary column to store the regular rank
    int64_t n = out_arr->length;
    std::shared_ptr<array_info> rank_arr =
        alloc_array(n, 1, 1, bodo_array_type::NUMPY, Bodo_CTypes::INT64, 0, 0);
    // Start the counter at 1, snap upward each we encounter a row
    // that is distinct from the previous row, and reset the counter
    // to 1 when we reach a new group. When a group ends, set
    // all rows in the output table to (r-1)/(n-1) where r is
    // the regular rank value and n is the group size (or 0 if n=1)
    int64_t prev_group = -1;
    int64_t rank_val = 1;
    int64_t group_start_idx = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t curr_group = getv<int64_t>(sorted_groups, i);
        if ((curr_group != prev_group)) {
            rank_val = 1;
            // Update the group that just finished by calculating
            // (r-1)/(n-1)
            rank_division_batch_update(out_arr, sorted_idx, rank_arr,
                                       group_start_idx, i, 1, 1);
            group_start_idx = i;
            // Set the prev group
            prev_group = curr_group;
        } else if (distinct_from_previous_row(sorted_orderbys, i)) {
            rank_val = i - group_start_idx + 1;
        }
        getv<int64_t>(rank_arr, i) = rank_val;
    }
    // Repeat the group ending procedure after the main loop finishes
    rank_division_batch_update(out_arr, sorted_idx, rank_arr, group_start_idx,
                               n, 1, 1);
};

/**
 * Computes the BodoSQL window function CUME_DIST() on a subset of a table
 * containing complete partitions, where the rows are sorted first by group
 * (so each partition is clustered togeher) and then by the columns in the
 * orderby clause.
 *
 * The computaiton works by calculating the tie-upward rank for each row. Then,
 * after a group is finished, the percent rank for each row is calculated
 * via the formula r/n where r is the tie-upward rank and n is the group size.
 *
 * Suppose the sorted values in an array are as follows:
 * ["A", "B", "B", "B", "C", "C", "D", "E", "E", "E"]
 *
 * The regular rank would be as follows:
 * [1, 2, 2, 2, 5, 5, 7, 8, 8, 8]
 *
 * But the tie-upward rank is as follows:
 * [1, 4, 4, 4, 6, 6, 7, 10, 10, 10]
 *
 * @param[in,out] out_arr: output array where the row numbers are stored
 * @param[in] sorted_groups: sorted array of group numbers
 * @param[in] sorted_orderbys the vector of sorted orderby columns, used to
 * keep track of when we have encountered a distinct row
 * @param[in] sorted_idx: array mapping each index in the sorted table back
 * to its location in the original unsorted table
 */
void cume_dist_computation(
    std::shared_ptr<array_info> out_arr,
    std::shared_ptr<array_info> sorted_groups,
    std::vector<std::shared_ptr<array_info>> sorted_orderbys,
    std::shared_ptr<array_info> sorted_idx) {
    // Create an intermediary column to store the tie-up rank
    int64_t n = out_arr->length;
    std::shared_ptr<array_info> rank_arr =
        alloc_array(n, 1, 1, bodo_array_type::NUMPY, Bodo_CTypes::INT64, 0, 0);
    // Start the counter at 1, snap upward each we encounter a row
    // that is distinct from the previous row, and reset the counter
    // to 1 when we reach a new group. When a group ends, set
    // all rows in the output table to r/n where r is
    // the tie-up rank value and n is the group size
    int64_t prev_group = -1;
    int64_t group_start_idx = 0;
    int64_t tie_start_idx = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t curr_group = getv<int64_t>(sorted_groups, i);
        if ((curr_group != prev_group)) {
            // Update the group of ties that just finished by setting
            // all of them to the rank value of the last position
            rank_tie_upward_batch_update(rank_arr, group_start_idx,
                                         tie_start_idx, i);
            // Update the group that just finished by calculating
            // r/n
            rank_division_batch_update(out_arr, sorted_idx, rank_arr,
                                       group_start_idx, i, 0, 0);
            group_start_idx = i;
            tie_start_idx = i;
            // Set the prev group
            prev_group = curr_group;
        } else if (distinct_from_previous_row(sorted_orderbys, i)) {
            // Update the group of ties that just finished by setting
            // all of them to the rank value of the last position
            rank_tie_upward_batch_update(rank_arr, group_start_idx,
                                         tie_start_idx, i);
            tie_start_idx = i;
        }
    }
    // Repeat the tie ending procedure after the final group finishes
    rank_tie_upward_batch_update(rank_arr, group_start_idx, tie_start_idx, n);
    // Repeat the group ending procedure after the final group finishes
    rank_division_batch_update(out_arr, sorted_idx, rank_arr, group_start_idx,
                               n, 0, 0);
}

void ntile_batch_update(std::shared_ptr<array_info> out_arr,
                        std::shared_ptr<array_info> sorted_idx,
                        int64_t group_start_idx, int64_t group_end_idx,
                        int num_bins) {
    // Calculate the number of items in the group, the number
    // of elements that will go into small vs large groups, and
    // the number of bins that will require a larger group
    int n = group_end_idx - group_start_idx;
    if (n == 0)
        return;
    int remainder = n % num_bins;
    int n_smaller = n / num_bins;
    int n_larger = n_smaller + (remainder ? 1 : 0);

    // Calculate the indices of bins that will use the large group
    // vs the small group
    int hi_cutoff = std::min(n, num_bins) + 1;
    int lo_cutoff = std::min(remainder, hi_cutoff) + 1;

    // For each bin from 1 to lo_cutoff, assign the next n_larger
    // indices to the current bin
    int bin_start_index = group_start_idx;
    for (int bin = 1; bin < lo_cutoff; bin++) {
        int bin_end_index = bin_start_index + n_larger;
        for (int i = bin_start_index; i < bin_end_index; i++) {
            // Get the index in the output array.
            int64_t idx = getv<int64_t>(sorted_idx, i);
            getv<int64_t>(out_arr, idx) = bin;
        }
        bin_start_index = bin_end_index;
    }

    // For each bin from lo_cutoff to hi_cutoff, assign the next n_smaller
    // indices to the current bin
    for (int64_t bin = lo_cutoff; bin < hi_cutoff; bin++) {
        int bin_end_index = bin_start_index + n_smaller;
        for (int i = bin_start_index; i < bin_end_index; i++) {
            // Get the index in the output array.
            int64_t idx = getv<int64_t>(sorted_idx, i);
            getv<int64_t>(out_arr, idx) = bin;
        }
        bin_start_index = bin_end_index;
    }
}

void ntile_computation(std::shared_ptr<array_info> out_arr,
                       std::shared_ptr<array_info> sorted_groups,
                       std::shared_ptr<array_info> sorted_idx,
                       int64_t num_bins) {
    // Each time we find the end of a group, invoke the ntile
    // procedure on all the rows between that index and
    // the index where the group started
    int64_t n = out_arr->length;
    int64_t prev_group = -1;
    int64_t group_start_idx = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t curr_group = getv<int64_t>(sorted_groups, i);
        if (curr_group != prev_group) {
            ntile_batch_update(out_arr, sorted_idx, group_start_idx, i,
                               num_bins);
            // Set the prev group
            prev_group = curr_group;
            group_start_idx = i;
        }
    }
    // Repeat the ntile batch procedure at the end on the final group
    ntile_batch_update(out_arr, sorted_idx, group_start_idx, n, num_bins);
}

/**
 * Computes the BodoSQL window function CONDITIONAL_TRUE_EVENT(A) on a
 * subset of a table containing complete partitions, where the rows are
 * sorted first by group (so each partition is clustered togeher) and then
 * by the columns in the orderby clause.
 *
 * The computaiton works by starting a counter at zero, resetting it to
 * zero each time a new group is reached, and otherwise only incrementing
 * the counter when the current row of the input column is set to true.
 *
 * @param[in,out] out_arr: output array where the row numbers are stored
 * @param[in] sorted_groups: sorted array of group numbers
 * @param[in] sorted_idx: array mapping each index in the sorted table back
 * to its location in the original unsorted table
 * @param[in] input_col: the boolean array whose values are used to calculate
 * the conditional_true_event calculation
 */
void conditional_true_event_computation(
    std::shared_ptr<array_info> out_arr,
    std::shared_ptr<array_info> sorted_groups,
    std::shared_ptr<array_info> sorted_idx,
    std::shared_ptr<array_info> input_col) {
    int64_t n = out_arr->length;
    int64_t prev_group = -1;
    int64_t counter = 0;
    for (int64_t i = 0; i < n; i++) {
        // Get the index in the output array.
        int64_t idx = getv<int64_t>(sorted_idx, i);
        // If we have crossed the threshold into a new group,
        // reset the counter to zero
        int64_t curr_group = getv<int64_t>(sorted_groups, i);
        if (curr_group != prev_group) {
            prev_group = curr_group;
            counter = 0;
        }
        // If the current row is true, increment the counter
        if (input_col->arr_type == bodo_array_type::NUMPY) {
            if (getv<uint8_t>(input_col, idx))
                counter++;
        } else {
            if (GetBit((uint8_t*)input_col->data1(), idx))
                counter++;
        }
        getv<int64_t>(out_arr, idx) = counter;
    }
}

/**
 * Computes the BodoSQL window function CONDITIONAL_CHANGE_EVENT(A) on a
 * subset of a table containing complete partitions, where the rows are
 * sorted first by group (so each partition is clustered togeher) and then
 * by the columns in the orderby clause.
 *
 * The computaiton works by starting a counter at zero, resetting it to
 * zero each time a new group is reached, and otherwise only incrementing
 * the counter when the current row of the input column is a non-null
 * value that is distinct from the most recent non-null value.
 *
 * @param[in,out] out_arr: output array where the row numbers are stored
 * @param[in] sorted_groups: sorted array of group numbers
 * @param[in] sorted_idx: array mapping each index in the sorted table back
 * to its location in the original unsorted table
 * @param[in] input_col: the array whose values are used to calculate
 * the conditional_true_event calculation
 */
void conditional_change_event_computation(
    std::shared_ptr<array_info> out_arr,
    std::shared_ptr<array_info> sorted_groups,
    std::shared_ptr<array_info> sorted_idx,
    std::shared_ptr<array_info> input_col) {
    int64_t n = out_arr->length;
    int64_t prev_group = -1;
    int64_t counter = 0;
    int64_t last_non_null = -1;
    for (int64_t i = 0; i < n; i++) {
        // Get the index in the output array.
        int64_t idx = getv<int64_t>(sorted_idx, i);
        // If we have crossed the threshold into a new group,
        // reset the counter to zero
        int64_t curr_group = getv<int64_t>(sorted_groups, i);
        if (curr_group != prev_group) {
            prev_group = curr_group;
            counter = 0;
            last_non_null = -1;
        }
        // If the current row is non-null and does not equal
        // the most recent non-null row, increment the counter
        if (input_col->arr_type == bodo_array_type::NUMPY ||
            input_col->get_null_bit(idx)) {
            if (last_non_null != -1 &&
                !TestEqualColumn(input_col, idx, input_col, last_non_null,
                                 true)) {
                counter++;
            }
            last_non_null = idx;
        }
        getv<int64_t>(out_arr, idx) = counter;
    }
}

/**
 * Computes a batch of window functions for BodoSQL on a subset of a table
 * containing complete partitions. All of the window functions in the
 * batch use the same partitioning and orderby scheme. If any of the window
 * functions require the table to be sorted in order for the result to be
 * calculated, performs a sort on the table first by group and then by
 * the orderby columns.
 *
 * @param[in] input_arrs: the columns being used to order the rows of
 * the table when computing a window function, as well as any additional
 * columns that are being aggregated by the window functions.
 * @param[in] window_funcs: the vector of window functions being computed
 * @param[in,out] out_arrs: the arrays where the final answer for each window
 * function computed will be stored
 * @param[in] asc_vect: vector indicating which of the orderby columns are
 * to be sorted in ascending vs descending order
 * @param[in] na_pos_vect: vector indicating which of the orderby columns are
 * to place nulls first vs last
 * @param[in] window_args: vector of any scalar arguments for the window
 * functions being calculated. It is the responsibility of each function to know
 * how many arguments it is expected and to cast them to the correct type.
 * @param[in] n_input_cols: the number of arrays from input_arrs that correspond
 * to inputs to the window functions. If there are any, they are at the end
 * of the vector in the same order as the functions in window_funcs.
 * @param[in] is_parallel: is the function being run in parallel?
 * @param[in] use_sql_rules: should initialization functions obey SQL semantics?
 */
void window_computation(std::vector<std::shared_ptr<array_info>>& input_arrs,
                        std::vector<int64_t> window_funcs,
                        std::vector<std::shared_ptr<array_info>> out_arrs,
                        grouping_info const& grp_info,
                        std::vector<bool>& asc_vect,
                        std::vector<bool>& na_pos_vect,
                        std::vector<void*>& window_args, int n_input_cols,
                        bool is_parallel, bool use_sql_rules) {
    int64_t window_arg_offset = 0;
    int64_t window_col_offset = input_arrs.size() - n_input_cols;
    std::vector<std::shared_ptr<array_info>> orderby_arrs(
        input_arrs.begin(), input_arrs.begin() + window_col_offset);
    // Create a new table. We want to sort the table first by
    // the groups and second by the orderby_arr.
    std::shared_ptr<table_info> sort_table = std::make_shared<table_info>();
    std::shared_ptr<table_info> iter_table = nullptr;
    const bodo::vector<int64_t>& row_to_group = grp_info.row_to_group;
    int64_t num_rows = row_to_group.size();
    int64_t n_keys = orderby_arrs.size() + 1;
    int64_t vect_ascending[n_keys];
    int64_t na_position[n_keys];
    // The sort table will be the same for every window function call that uses
    // it, so the table will be unitialized until one of the calls specifies
    // that we do need to sort
    bool needs_sort = false;
    bool sort_has_required_cols = false;
    int64_t idx_col = 0;
    for (size_t i = 0; i < window_funcs.size(); i++) {
        switch (window_funcs[i]) {
            case Bodo_FTypes::min_row_number_filter: {
                // Window functions that do not require the sorted table
                break;
            }
            case Bodo_FTypes::row_number:
            case Bodo_FTypes::rank:
            case Bodo_FTypes::dense_rank:
            case Bodo_FTypes::percent_rank:
            case Bodo_FTypes::cume_dist:
            case Bodo_FTypes::ntile:
            case Bodo_FTypes::conditional_true_event:
            case Bodo_FTypes::conditional_change_event: {
                needs_sort = true;
                /* If this is the first function encountered that requires,
                 * a sort, populate sort_table with the following columns:
                 * - 1 column containing the group numbers so that when the
                 *   table is sorted, each partition has its rows clustered
                 * together
                 * - 1 column for each of the orderby cols so that within each
                 *   partition, the values are sorted as desired
                 * - 1 extra column that is set to 0...n-1 so that when it is
                 *   sorted, we have a way of converting rows in the sorted
                 *   table back to rows in the original table.
                 */
                if (!sort_has_required_cols) {
                    // Wrap the row_to_group in an array info so we can use it
                    // to sort.
                    std::shared_ptr<array_info> group_arr =
                        alloc_numpy(num_rows, Bodo_CTypes::INT64);
                    // TODO: Reuse the row_to_group buffer
                    for (int64_t i = 0; i < num_rows; i++) {
                        getv<int64_t>(group_arr, i) = row_to_group[i];
                    }
                    sort_table->columns.push_back(group_arr);
                    // Push each orderby column into the sort table
                    for (std::shared_ptr<array_info> orderby_arr :
                         orderby_arrs) {
                        sort_table->columns.push_back(orderby_arr);
                    }
                    // Append an index column so we can find the original
                    // index in the out array, and mark which column
                    // is the index column
                    idx_col = sort_table->columns.size();
                    std::shared_ptr<array_info> idx_arr =
                        alloc_numpy(num_rows, Bodo_CTypes::INT64);
                    for (int64_t i = 0; i < num_rows; i++) {
                        getv<int64_t>(idx_arr, i) = i;
                    }
                    sort_table->columns.push_back(idx_arr);
                    /* Populate the buffers to indicate which columns are
                     * ascending/descending and which have nulls first/last
                     * according to the input specifications, plus the
                     * group key column in front which is hardcoded to
                     * use the same rules as the first orderby column
                     */
                    vect_ascending[0] = asc_vect[0];
                    na_position[0] = na_pos_vect[0];
                    for (int64_t i = 1; i < n_keys; i++) {
                        vect_ascending[i] = asc_vect[i - 1];
                        na_position[i] = na_pos_vect[i - 1];
                    }
                    sort_has_required_cols = true;
                }
                break;
            }
            default:
                throw std::runtime_error("Invalid window function");
        }
    }
    if (needs_sort) {
        // Sort the table so that all window functions that use the
        // sorted table can access it
        iter_table = sort_values_table_local(
            sort_table, n_keys, vect_ascending, na_position, nullptr,
            is_parallel /* This is just used for tracing */);
        sort_table.reset();
    }
    // For each window function call, compute the answer using the
    // sorted table to lookup the rows in the original ordering
    // that are to be modified
    for (size_t i = 0; i < window_funcs.size(); i++) {
        switch (window_funcs[i]) {
            // min_row_number_filter uses a sort-less implementaiton if no
            // other window functions being calculated require sorting. However,
            // if another window funciton in this computation requires sorting
            // the table, then we can just use the sorted groups isntead.
            case Bodo_FTypes::min_row_number_filter: {
                if (needs_sort) {
                    min_row_number_filter_window_computation_already_sorted(
                        out_arrs[i], iter_table->columns[0],
                        iter_table->columns[idx_col]);
                } else {
                    min_row_number_filter_window_computation_no_sort(
                        orderby_arrs, out_arrs[i], grp_info, asc_vect,
                        na_pos_vect, is_parallel, use_sql_rules);
                }
                break;
            }
            case Bodo_FTypes::row_number: {
                row_number_computation(out_arrs[i], iter_table->columns[0],
                                       iter_table->columns[idx_col]);
                break;
            }
            case Bodo_FTypes::rank: {
                rank_computation(out_arrs[i], iter_table->columns[0],
                                 std::vector<std::shared_ptr<array_info>>(
                                     iter_table->columns.begin() + 1,
                                     iter_table->columns.begin() + idx_col),
                                 iter_table->columns[idx_col]);
                break;
            }
            case Bodo_FTypes::dense_rank: {
                dense_rank_computation(
                    out_arrs[i], iter_table->columns[0],
                    std::vector<std::shared_ptr<array_info>>(
                        iter_table->columns.begin() + 1,
                        iter_table->columns.begin() + idx_col),
                    iter_table->columns[idx_col]);
                break;
            }
            case Bodo_FTypes::percent_rank: {
                percent_rank_computation(
                    out_arrs[i], iter_table->columns[0],
                    std::vector<std::shared_ptr<array_info>>(
                        iter_table->columns.begin() + 1,
                        iter_table->columns.begin() + idx_col),
                    iter_table->columns[idx_col]);
                break;
            }
            case Bodo_FTypes::cume_dist: {
                cume_dist_computation(
                    out_arrs[i], iter_table->columns[0],
                    std::vector<std::shared_ptr<array_info>>(
                        iter_table->columns.begin() + 1,
                        iter_table->columns.begin() + idx_col),
                    iter_table->columns[idx_col]);
                break;
            }
            case Bodo_FTypes::ntile: {
                ntile_computation(out_arrs[i], iter_table->columns[0],
                                  iter_table->columns[idx_col],
                                  *((int64_t*)window_args[window_arg_offset]));
                window_arg_offset += 1;
                break;
            }
            case Bodo_FTypes::conditional_true_event: {
                conditional_true_event_computation(
                    out_arrs[i], iter_table->columns[0],
                    iter_table->columns[idx_col],
                    input_arrs[window_col_offset]);
                window_col_offset += 1;
                break;
            }
            case Bodo_FTypes::conditional_change_event: {
                conditional_change_event_computation(
                    out_arrs[i], iter_table->columns[0],
                    iter_table->columns[idx_col],
                    input_arrs[window_col_offset]);
                window_col_offset += 1;
                break;
            }
            default:
                throw std::runtime_error("Invalid window function");
        }
    }
}
