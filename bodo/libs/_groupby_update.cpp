// Copyright (C) 2023 Bodo Inc. All rights reserved.
#include "_groupby_update.h"
#include "_array_operations.h"
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_distributed.h"
#include "_groupby_common.h"
#include "_groupby_do_apply_to_column.h"
#include "_groupby_hashing.h"
#include "_shuffle.h"

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
template <typename T, Bodo_CTypes::CTypeEnum DType>
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
        T eVal_nan = GetTentry<T>(RetrieveNaNentry(DType).data());
        std::pair<bool, T> pairNaN{true, eVal_nan};
        for (auto& idx_miss : grp_info.list_missing) {
            set_entry(idx_miss, pairNaN);
        }
    };

    if (arr->arr_type == bodo_array_type::NUMPY) {
        if (DType == Bodo_CTypes::DATETIME || DType == Bodo_CTypes::TIMEDELTA) {
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
                    bool isna = isnan_alltype<T, DType>(eVal);
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
                return {
                    !arr->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(pos),
                    arr->at<T>(pos)};
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

// PERCENTILE

template <bodo_array_type::arr_type_enum ArrayType,
          Bodo_CTypes::CTypeEnum DType>
void percentile_computation_util(std::shared_ptr<array_info> arr,
                                 std::shared_ptr<array_info> out_arr,
                                 double percentile, bool interpolate,
                                 grouping_info const& grp_info) {
    using T = typename dtype_to_type<DType>::type;
    // Iterate across each group to find its percentile value
    for (size_t igrp = 0; igrp < grp_info.num_groups; igrp++) {
        // Set up a vector that will contain all of the values from the current
        // group as a double.
        std::vector<double> vect;
        for (int64_t i = grp_info.group_to_first_row[igrp]; i != -1;
             i = grp_info.next_row_in_group[i]) {
            if (!is_null_at<ArrayType, T, DType>(*arr, i)) {
                T val = get_arr_item<ArrayType, T, DType>(*arr, i);
                vect.emplace_back(to_double<T, DType>(val));
            }
        }
        // If there were no non-null entries, output null.
        if (vect.size() == 0) {
            set_to_null<bodo_array_type::NULLABLE_INT_BOOL, double,
                        Bodo_CTypes::FLOAT64>(*out_arr, igrp);
            continue;
        }
        double valReturn = 0;
        size_t len = vect.size();
        if (interpolate) {
            // Algorithm for PERCENTILE_CONT

            // Calculate the index in the array that corresponds to
            // the desired percentile. Cast to int64_t to round down.
            double k_approx = (len - 1) * percentile;
            int64_t k_exact = (int64_t)k_approx;
            if (k_approx == k_exact) {
                // If rounding down does not change the value, then
                // k_exact is the exact index we want. Obtain
                // the k_exact-largest element by using std::nth
                // element to partially sort about that index.
                std::nth_element(vect.begin(), vect.begin() + k_exact,
                                 vect.end());
                valReturn = vect[k_exact];
            } else {
                // Otherwise, we want a weighted average of the
                // values at k_exact and the next index.  Obtain
                // the k_exact-largest element by using std::nth
                // element to partially sort about that index.
                std::nth_element(vect.begin(), vect.begin() + k_exact,
                                 vect.end());
                double v1 = vect[k_exact];
                // Then, find the minimum of the remaining elements
                double v2 =
                    *std::min_element(vect.begin() + k_exact + 1, vect.end());
                // Linearly interpolate between v1 and v2 where
                // the weight is based on how close k_approx is
                // to k_exact vs k_exact + 1. E.g. if k_exact
                // is 12 and k_approx = 12.25, then the formula
                // is v1 + 0.25 * (v2 - v1) = 0.75*v1 + 0.25*v2
                valReturn = v1 + (k_approx - k_exact) * (v2 - v1);
            }
        } else {
            // Algorithm for PERCENTILE_DISC

            // Calculate the index in the array that corresponds to
            // the desired percentile. Cast to int64_t to round down.
            double k_approx = len * percentile;
            int64_t k_exact = (int64_t)k_approx;
            // The following formula will choose the ordinal formula
            // corresponding to the first location whose cumulative distribution
            // is >= percentile.
            int64_t k_select = (k_approx == k_exact) ? (k_exact - 1) : k_exact;
            if (k_select < 0)
                k_select = 0;
            std::nth_element(vect.begin(), vect.begin() + k_select, vect.end());
            valReturn = vect[k_select];
        }
        // Store the answer in the output array
        set_non_null<bodo_array_type::NULLABLE_INT_BOOL, double,
                     Bodo_CTypes::FLOAT64>(*out_arr, igrp);
        set_arr_item<bodo_array_type::NULLABLE_INT_BOOL, double,
                     Bodo_CTypes::FLOAT64>(*out_arr, igrp, valReturn);
    }
}

template <bodo_array_type::arr_type_enum ArrayType>
void percentile_computation_dtype_helper(std::shared_ptr<array_info> arr,
                                         std::shared_ptr<array_info> out_arr,
                                         double percentile, bool interpolate,
                                         grouping_info const& grp_info) {
    switch (arr->dtype) {
        case Bodo_CTypes::INT8: {
            percentile_computation_util<ArrayType, Bodo_CTypes::INT8>(
                arr, out_arr, percentile, interpolate, grp_info);
            break;
        }
        case Bodo_CTypes::UINT8: {
            percentile_computation_util<ArrayType, Bodo_CTypes::UINT8>(
                arr, out_arr, percentile, interpolate, grp_info);
            break;
        }
        case Bodo_CTypes::INT16: {
            percentile_computation_util<ArrayType, Bodo_CTypes::INT16>(
                arr, out_arr, percentile, interpolate, grp_info);
            break;
        }
        case Bodo_CTypes::UINT16: {
            percentile_computation_util<ArrayType, Bodo_CTypes::UINT16>(
                arr, out_arr, percentile, interpolate, grp_info);
            break;
        }
        case Bodo_CTypes::INT32: {
            percentile_computation_util<ArrayType, Bodo_CTypes::INT32>(
                arr, out_arr, percentile, interpolate, grp_info);
            break;
        }
        case Bodo_CTypes::UINT32: {
            percentile_computation_util<ArrayType, Bodo_CTypes::UINT32>(
                arr, out_arr, percentile, interpolate, grp_info);
            break;
        }
        case Bodo_CTypes::INT64: {
            percentile_computation_util<ArrayType, Bodo_CTypes::INT64>(
                arr, out_arr, percentile, interpolate, grp_info);
            break;
        }
        case Bodo_CTypes::UINT64: {
            percentile_computation_util<ArrayType, Bodo_CTypes::UINT64>(
                arr, out_arr, percentile, interpolate, grp_info);
            break;
        }
        case Bodo_CTypes::FLOAT32: {
            percentile_computation_util<ArrayType, Bodo_CTypes::FLOAT32>(
                arr, out_arr, percentile, interpolate, grp_info);
            break;
        }
        case Bodo_CTypes::FLOAT64: {
            percentile_computation_util<ArrayType, Bodo_CTypes::FLOAT64>(
                arr, out_arr, percentile, interpolate, grp_info);
            break;
        }
        case Bodo_CTypes::DECIMAL: {
            percentile_computation_util<ArrayType, Bodo_CTypes::DECIMAL>(
                arr, out_arr, percentile, interpolate, grp_info);
            break;
        }
        default: {
            throw std::runtime_error(
                "Unsupported dtype for percentile_computation: " +
                GetDtype_as_string(arr->dtype));
        }
    }
}

void percentile_computation(std::shared_ptr<array_info> arr,
                            std::shared_ptr<array_info> out_arr,
                            double percentile, bool interpolate,
                            grouping_info const& grp_info) {
    switch (arr->arr_type) {
        case bodo_array_type::NUMPY: {
            percentile_computation_dtype_helper<bodo_array_type::NUMPY>(
                arr, out_arr, percentile, interpolate, grp_info);
            break;
        }
        case bodo_array_type::NULLABLE_INT_BOOL: {
            percentile_computation_dtype_helper<
                bodo_array_type::NULLABLE_INT_BOOL>(arr, out_arr, percentile,
                                                    interpolate, grp_info);
            break;
        }
        default: {
            throw std::runtime_error(
                "Unsupported array type for percentile_computation: " +
                GetArrType_as_string(arr->arr_type));
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
        median_operation([=](size_t pos) -> bool {
            return !arr->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(pos);
        });
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
            if (group_num == -1) {
                continue;
            }
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
            if (group_num == -1) {
                continue;
            }

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
            if (group_num == -1) {
                continue;
            }

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
            if (group_num == -1) {
                continue;
            }

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

// ARRAY_AGG

template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType>
void array_agg_operation(
    const std::shared_ptr<array_info>& in_arr,
    std::shared_ptr<array_info> out_arr,
    const std::vector<std::shared_ptr<array_info>>& orderby_cols,
    const std::vector<bool>& ascending, const std::vector<bool>& na_position,
    const grouping_info& grp_info, bool is_parallel) {
    using T = typename dtype_to_type<DType>::type;

    size_t num_group = grp_info.group_to_first_row.size();
    size_t n_total = in_arr->length;

    // Step 1: Sort the table first by group (so each group has its elements
    // contiguous) and then by the orderby columns (so each group's elements
    // are internally sorted in the desired manner)
    std::shared_ptr<table_info> sorted_table =
        grouped_sort(grp_info, orderby_cols, {in_arr}, ascending, na_position,
                     1, is_parallel);
    int64_t n_sort_cols = orderby_cols.size() + 1;

    // At this stage, the implementations diverge depending on the input
    // data's array type
    if constexpr (ArrType == bodo_array_type::NUMPY) {
        // Numpy arrays do not have any nulls to remove, so the last column
        // of the sorted table can be used directly as the inner data.

        // Step 2: Make the last column from the sorted table the new inner
        // array, since it has all of the data from each group stored
        // contiguously and in the correct order.
        std::shared_ptr<array_info> inner_arr = out_arr->child_arrays[0];
        *inner_arr = *(sorted_table->columns[n_sort_cols]);

        // Step 3: Update the offsets to match the cutoffs where each contiguous
        // group in the sorted table begins/ends
        offset_t* offset_buffer =
            (offset_t*)(out_arr->buffers[0]->mutable_data() + out_arr->offset);
        offset_buffer[0] = 0;
        offset_buffer[num_group] = n_total;
        std::shared_ptr<array_info> sorted_groups = sorted_table->columns[0];
        int64_t cur_idx = 1;
        for (size_t i = 1; i < n_total; i++) {
            if (getv<int64_t>(sorted_groups, i) !=
                getv<int64_t>(sorted_groups, i - 1)) {
                offset_buffer[cur_idx] = i;
                cur_idx++;
            }
        }
    } else {
        // For nullable arrays, all the rows that contain a null in the
        // data column must first be removed, but groups that have all of their
        // rows removed must not have the group itself removed.

        // Step 2: Calculate the number of non-null rows and create a new
        // arrays for the data excluding nulls.
        int64_t non_null_count = 0;
        std::shared_ptr<array_info> sorted_data =
            sorted_table->columns[n_sort_cols];
        for (size_t i = 0; i < n_total; i++) {
            if (non_null_at<ArrType, T, DType>(*sorted_data, i)) {
                non_null_count++;
            }
        }
        std::shared_ptr<array_info> data_without_nulls =
            alloc_nullable_array_no_nulls(non_null_count, in_arr->dtype, 0);

        // If the input data is Decimal, ensure the output array has the same
        // precision/scale.
        if constexpr (DType == Bodo_CTypes::DECIMAL) {
            data_without_nulls->precision = in_arr->precision;
            data_without_nulls->scale = in_arr->scale;
        }

        // Step 3: Move the non null elements from the data column to the new
        // array, and copy over the offsets each time we enter a new group.
        offset_t* offset_buffer =
            (offset_t*)(out_arr->buffers[0]->mutable_data() + out_arr->offset);
        offset_buffer[0] = 0;
        offset_buffer[num_group] = non_null_count;
        std::shared_ptr<array_info> sorted_groups = sorted_table->columns[0];
        int64_t group_idx = 1;
        offset_t curr_offset = 0;
        for (size_t i = 0; i < n_total; i++) {
            // If the current group has just ended, the next offset will be the
            // previous offset plus the number of non null elements since the
            // current group started.
            if (i > 1 && getv<int64_t>(sorted_groups, i) !=
                             getv<int64_t>(sorted_groups, i - 1)) {
                offset_buffer[group_idx] = curr_offset;
                group_idx++;
            }
            // If the current element is non null, write it to the next empty
            // position in data_without_nulls
            if (non_null_at<ArrType, T, DType>(*sorted_data, i)) {
                set_arr_item<ArrType, T, DType>(
                    *data_without_nulls, curr_offset,
                    get_arr_item<ArrType, T, DType>(*sorted_data, i));
                curr_offset++;
            }
        }
        std::shared_ptr<array_info> inner_arr = out_arr->child_arrays[0];
        *inner_arr = *data_without_nulls;
    }
}

template <bodo_array_type::arr_type_enum ArrType>
void array_agg_dtype_helper(
    const std::shared_ptr<array_info>& in_arr,
    std::shared_ptr<array_info> out_arr,
    const std::vector<std::shared_ptr<array_info>>& orderby_cols,
    const std::vector<bool>& ascending, const std::vector<bool>& na_position,
    const grouping_info& grp_info, bool is_parallel) {
#define ARRAY_AGG_DTYPE_CASE(dtype)                                           \
    case dtype: {                                                             \
        array_agg_operation<ArrType, dtype>(in_arr, out_arr, orderby_cols,    \
                                            ascending, na_position, grp_info, \
                                            is_parallel);                     \
        break;                                                                \
    }
    switch (in_arr->dtype) {
        ARRAY_AGG_DTYPE_CASE(Bodo_CTypes::UINT8)
        ARRAY_AGG_DTYPE_CASE(Bodo_CTypes::UINT16)
        ARRAY_AGG_DTYPE_CASE(Bodo_CTypes::UINT32)
        ARRAY_AGG_DTYPE_CASE(Bodo_CTypes::UINT64)
        ARRAY_AGG_DTYPE_CASE(Bodo_CTypes::INT8)
        ARRAY_AGG_DTYPE_CASE(Bodo_CTypes::INT16)
        ARRAY_AGG_DTYPE_CASE(Bodo_CTypes::INT32)
        ARRAY_AGG_DTYPE_CASE(Bodo_CTypes::INT64)
        ARRAY_AGG_DTYPE_CASE(Bodo_CTypes::FLOAT32)
        ARRAY_AGG_DTYPE_CASE(Bodo_CTypes::FLOAT64)
        ARRAY_AGG_DTYPE_CASE(Bodo_CTypes::DECIMAL)
        ARRAY_AGG_DTYPE_CASE(Bodo_CTypes::_BOOL)
        default: {
            throw std::runtime_error(
                "Unsupported dtype encountered with array_agg. Found type: " +
                GetDtype_as_string(in_arr->dtype));
        }
    }
}

void array_agg_computation(
    const std::shared_ptr<array_info>& in_arr,
    std::shared_ptr<array_info> out_arr,
    const std::vector<std::shared_ptr<array_info>>& orderby_cols,
    const std::vector<bool>& ascending, const std::vector<bool>& na_position,
    const grouping_info& grp_info, bool is_parallel) {
    switch (in_arr->arr_type) {
        case bodo_array_type::NUMPY: {
            array_agg_dtype_helper<bodo_array_type::NUMPY>(
                in_arr, out_arr, orderby_cols, ascending, na_position, grp_info,
                is_parallel);
            break;
        }
        case bodo_array_type::NULLABLE_INT_BOOL: {
            array_agg_dtype_helper<bodo_array_type::NULLABLE_INT_BOOL>(
                in_arr, out_arr, orderby_cols, ascending, na_position, grp_info,
                is_parallel);
            break;
        }
        default: {
            throw std::runtime_error(
                "Unsupported array type encountered with array_agg. Found "
                "type: " +
                GetArrType_as_string(in_arr->arr_type));
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
                if (arr->get_null_bit<bodo_array_type::LIST_STRING>(i)) {
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
                if (arr->get_null_bit<bodo_array_type::STRING>(i)) {
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
                if (arr->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i)) {
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
