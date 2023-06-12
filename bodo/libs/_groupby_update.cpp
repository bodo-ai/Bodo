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

void window_computation(std::vector<std::shared_ptr<array_info>>& orderby_arrs,
                        int64_t window_func,
                        std::shared_ptr<array_info> out_arr,
                        grouping_info const& grp_info,
                        std::vector<bool>& asc_vect,
                        std::vector<bool>& na_pos_vect, bool is_parallel,
                        bool use_sql_rules) {
    switch (window_func) {
        case Bodo_FTypes::row_number: {
            const bodo::vector<int64_t>& row_to_group = grp_info.row_to_group;
            int64_t num_rows = row_to_group.size();
            // Wrap the row_to_group in an array info so we can use it to sort.
            std::shared_ptr<array_info> group_arr =
                alloc_numpy(num_rows, Bodo_CTypes::INT64);
            // TODO: Reuse the row_to_group buffer
            for (int64_t i = 0; i < num_rows; i++) {
                getv<int64_t>(group_arr, i) = row_to_group[i];
            }
            // Create a new table. We want to sort the table first by
            // the groups and second by the orderby_arr.
            std::shared_ptr<table_info> sort_table =
                std::make_shared<table_info>();
            sort_table->columns.push_back(group_arr);
            for (std::shared_ptr<array_info> orderby_arr : orderby_arrs) {
                sort_table->columns.push_back(orderby_arr);
            }
            // Append an index column so we can find the original
            // index in the out array.
            std::shared_ptr<array_info> idx_arr =
                alloc_numpy(num_rows, Bodo_CTypes::INT64);
            for (int64_t i = 0; i < num_rows; i++) {
                getv<int64_t>(idx_arr, i) = i;
            }
            sort_table->columns.push_back(idx_arr);
            int64_t n_keys = orderby_arrs.size() + 1;
            int64_t vect_ascending[n_keys];
            int64_t na_position[n_keys];
            // Initialize the group ordering
            vect_ascending[0] = asc_vect[0];
            na_position[0] = na_pos_vect[0];
            for (int64_t i = 1; i < n_keys; i++) {
                vect_ascending[i] = asc_vect[i - 1];
                na_position[i] = na_pos_vect[i - 1];
            }

            // Sort the table
            // XXX: We don't need the entire chunk of data sorted,
            // just the groups. We could do a partial sort to avoid
            // the overhead of sorting the data and in the future
            // we may be want to explore if we can use hashing
            // instead to avoid sort overhead.
            std::shared_ptr<table_info> iter_table = sort_values_table_local(
                sort_table, n_keys, vect_ascending, na_position, nullptr,
                is_parallel /* This is just used for tracing */);
            std::shared_ptr<array_info> sorted_groups = iter_table->columns[0];
            std::shared_ptr<array_info> sorted_idx =
                iter_table->columns[n_keys];
            sort_table.reset();

            int64_t prev_group = -1;
            int64_t row_num = 1;
            for (int64_t i = 0; i < num_rows; i++) {
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
            break;
        }
        case Bodo_FTypes::min_row_number_filter: {
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
                // each group. Indices
                idx_col = alloc_array(num_groups, 1, 1, idx_arr_type,
                                      Bodo_CTypes::UINT64, 0, 0);
                // create array to store min/max value
                std::shared_ptr<array_info> data_col =
                    alloc_array(num_groups, 1, 1, orderby_arr->arr_type,
                                orderby_arr->dtype, 0, 0);
                // Initialize the index column. This is 0 initialized and will
                // not initial the null values.
                aggfunc_output_initialize(idx_col, Bodo_FTypes::count,
                                          use_sql_rules);
                std::vector<std::shared_ptr<array_info>> aux_cols = {idx_col};
                // Initialize the max column
                if (ftype == Bodo_FTypes::idxmax ||
                    ftype == Bodo_FTypes::idxmax_na_first) {
                    aggfunc_output_initialize(data_col, Bodo_FTypes::max,
                                              use_sql_rules);
                } else {
                    aggfunc_output_initialize(data_col, Bodo_FTypes::min,
                                              use_sql_rules);
                }
                // Compute the idxmin/idxmax
                do_apply_to_column(orderby_arr, data_col, aux_cols, grp_info,
                                   ftype);
            } else {
                ftype = Bodo_FTypes::idx_n_columns;
                // We don't need null for indices
                // We only allocate an index column.
                idx_col = alloc_array(num_groups, 1, 1, bodo_array_type::NUMPY,
                                      Bodo_CTypes::UINT64, 0, 0);
                aggfunc_output_initialize(idx_col, Bodo_FTypes::count,
                                          use_sql_rules);
                // Call the idx_n_columns function path.
                idx_n_columns_apply(idx_col, orderby_arrs, asc_vect,
                                    na_pos_vect, grp_info, ftype);
            }
            // Now we have the idxmin/idxmax in the idx_col. We need to set the
            // indices to true.
            for (size_t i = 0; i < idx_col->length; i++) {
                int64_t idx = getv<int64_t>(idx_col, i);
                SetBitTo((uint8_t*)out_arr->data1(), idx, true);
            }
            break;
        }
        default:
            throw std::runtime_error("Invalid window function");
    }
}
