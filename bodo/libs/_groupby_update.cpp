// Copyright (C) 2023 Bodo Inc. All rights reserved.
#include "_groupby_update.h"
#include <arrow/util/decimal.h>
#include "_array_operations.h"
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_decimal_ext.h"
#include "_distributed.h"
#include "_gandiva_decimal_copy.h"
#include "_groupby_common.h"
#include "_groupby_do_apply_to_column.h"
#include "_groupby_hashing.h"
#include "_shuffle.h"

// Really should be defined in something like _decimal_ext.h,
// but there seem to be many other files that redefine CHECK_ARROW.
#ifndef CHECK_ARROW
#define CHECK_ARROW(expr, msg)                                              \
    if (!(expr.ok())) {                                                     \
        std::string err_msg = std::string("Error in decimal utilities: ") + \
                              msg + " " + expr.ToString();                  \
        throw std::runtime_error(err_msg);                                  \
    }
#endif

#ifndef CHECK_ARROW_AND_ASSIGN
#define CHECK_ARROW_AND_ASSIGN(res, msg, lhs) \
    CHECK_ARROW(res.status(), msg)            \
    lhs = std::move(res).ValueOrDie();
#endif

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
 * @param[in] arr: input array, cannot be a string (or dict-encoded string)
 * array
 * @param[out] out_arr: output array, cannot be a string (or dict-encoded
 * string) array
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
        arr->arr_type == bodo_array_type::DICT) {
        throw std::runtime_error(
            "There is no cumulative operation for the string or dict-encoded "
            "string case");
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
                    T eVal = arr->at<T, bodo_array_type::NUMPY>(pos);
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
                    T eVal = arr->at<T, bodo_array_type::NUMPY>(pos);
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
                    arr->at<T, bodo_array_type::NULLABLE_INT_BOOL>(pos)};
            },
            [=](int64_t pos, std::pair<bool, T> const& ePair) -> void {
                // XXX TODO These need to be templated!
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
 * @param[in]  arr: input array, list of string array
 * @param[out] out_arr: output array, list of string array
 * @param[in]  grp_info: groupby information
 * @param[in]  ftype: for list of strings only cumsum is supported
 * @param[in]  skipna: Whether to skip NA values.
 */
void cumulative_computation_list_string(std::shared_ptr<array_info> arr,
                                        std::shared_ptr<array_info> out_arr,
                                        grouping_info const& grp_info,
                                        int32_t const& ftype,
                                        bool const& skipna) {
    assert(arr->arr_type == bodo_array_type::ARRAY_ITEM);
    if (ftype != Bodo_FTypes::cumsum) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "So far only cumulative sums for list-strings");
    }
    int64_t n = arr->length;
    using T = std::pair<bool, bodo::vector<std::pair<std::string, bool>>>;
    bodo::vector<T> null_bit_val_vec(n);
    uint8_t* null_bitmask =
        (uint8_t*)arr->null_bitmask<bodo_array_type::ARRAY_ITEM>();

    uint8_t* sub_null_bitmask;
    char* data;
    offset_t* data_offsets;
    offset_t* index_offsets;

    sub_null_bitmask = (uint8_t*)arr->child_arrays[0]->null_bitmask();
    data = arr->child_arrays[0]->data1();
    data_offsets = (offset_t*)arr->child_arrays[0]->data2();
    index_offsets = (offset_t*)arr->data1<bodo_array_type::ARRAY_ITEM>();

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
 * @param pool Memory pool to use for allocations during the execution of this
 * function.
 * @param mm Memory manager associated with the pool.
 */
void cumulative_computation_string(
    std::shared_ptr<array_info> arr, std::shared_ptr<array_info> out_arr,
    grouping_info const& grp_info, int32_t const& ftype, bool const& skipna,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager()) {
    assert(arr->arr_type == bodo_array_type::STRING);
    if (ftype != Bodo_FTypes::cumsum) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "So far only cumulative sums for strings");
    }
    int64_t n = arr->length;
    using T = std::pair<bool, std::string>;
    bodo::vector<T> null_bit_val_vec(n, pool);
    uint8_t* null_bitmask =
        (uint8_t*)arr->null_bitmask<bodo_array_type::STRING>();
    char* data = arr->data1<bodo_array_type::STRING>();
    offset_t* offsets = (offset_t*)arr->data2<bodo_array_type::STRING>();
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
    bodo::vector<uint8_t> Vmask(n_bytes, 0, pool);
    bodo::vector<std::string> ListString(n, pool);
    for (int64_t i = 0; i < n; i++) {
        SetBitTo(Vmask.data(), i, !null_bit_val_vec[i].first);
        ListString[i] = null_bit_val_vec[i].second;
    }
    std::shared_ptr<array_info> new_out_col = create_string_array(
        out_arr->dtype, Vmask, ListString, -1, pool, std::move(mm));
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
 * @param pool Memory pool to use for allocations during the execution of this
 * function.
 * @param mm Memory manager associated with the pool.
 */
void cumulative_computation_dict_encoded_string(
    std::shared_ptr<array_info> arr, std::shared_ptr<array_info> out_arr,
    grouping_info const& grp_info, int32_t const& ftype, bool const& skipna,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager()) {
    if (ftype != Bodo_FTypes::cumsum) {
        Bodo_PyErr_SetString(
            PyExc_RuntimeError,
            "So far only cumulative sums for dictionary-encoded strings");
    }
    int64_t n = arr->length;
    using T = std::pair<bool, std::string>;
    bodo::vector<T> null_bit_val_vec(
        n, pool);  // a temporary vector that stores the
                   // null bit and value for each row
    uint8_t* null_bitmask = (uint8_t*)arr->child_arrays[1]->null_bitmask();
    char* data = arr->child_arrays[0]->data1();
    offset_t* offsets = (offset_t*)arr->child_arrays[0]->data2();
    dict_indices_t* arr_indices_data1 =
        (dict_indices_t*)arr->child_arrays[1]->data1();
    auto get_entry = [&](int64_t i) -> T {
        bool isna = !GetBit(null_bitmask, i);
        if (isna) {
            return {isna, ""};
        }
        int32_t dict_ind = arr_indices_data1[i];
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
    bodo::vector<uint8_t> Vmask(n_bytes, 0, pool);
    bodo::vector<std::string> ListString(n, pool);
    for (int64_t i = 0; i < n; i++) {
        SetBitTo(Vmask.data(), i, !null_bit_val_vec[i].first);
        ListString[i] = null_bit_val_vec[i].second;
    }
    std::shared_ptr<array_info> new_out_col = create_string_array(
        out_arr->dtype, Vmask, ListString, -1, pool, std::move(mm));
    *out_arr = std::move(*new_out_col);
}

void cumulative_computation(std::shared_ptr<array_info> arr,
                            std::shared_ptr<array_info> out_arr,
                            grouping_info const& grp_info, int32_t const& ftype,
                            bool const& skipna, bodo::IBufferPool* const pool,
                            std::shared_ptr<::arrow::MemoryManager> mm) {
    Bodo_CTypes::CTypeEnum dtype = arr->dtype;
    if (arr->arr_type == bodo_array_type::STRING) {
        return cumulative_computation_string(arr, out_arr, grp_info, ftype,
                                             skipna, pool, std::move(mm));
    } else if (arr->arr_type == bodo_array_type::DICT) {
        return cumulative_computation_dict_encoded_string(
            arr, out_arr, grp_info, ftype, skipna, pool, std::move(mm));
    } else if (arr->arr_type == bodo_array_type::ARRAY_ITEM &&
               arr->child_arrays[0]->arr_type == bodo_array_type::STRING) {
        // We don't support LIST_STRING in streaming yet, so we don't
        // need to pass the Op-Pool here.
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
                      const bodo::vector<int64_t>& row_list,
                      bodo::IBufferPool* const pool,
                      std::shared_ptr<::arrow::MemoryManager> mm) {
    std::shared_ptr<array_info> updated_col = RetrieveArray_SingleColumn(
        std::move(arr), row_list, false, pool, std::move(mm));
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
        HANDLE_MPI_ERROR(MPI_Exscan(&num_group, &start_ngroup, 1, mpi_typ,
                                    MPI_SUM, MPI_COMM_WORLD),
                         "ngroup_computation: MPI error on MPI_Exscan:");
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
void percentile_computation_util(
    std::shared_ptr<array_info> arr, std::shared_ptr<array_info> out_arr,
    double percentile, bool interpolate, grouping_info const& grp_info,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr()) {
    using T = typename dtype_to_type<DType>::type;
    // Iterate across each group to find its percentile value
    for (size_t igrp = 0; igrp < grp_info.num_groups; igrp++) {
        // Set up a vector that will contain all of the values from the current
        // group as a double.
        bodo::vector<double> vect(pool);
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

void percentile_computation_decimal_util(
    std::shared_ptr<array_info> arr, std::shared_ptr<array_info> out_arr,
    double percentile, bool interpolate, grouping_info const& grp_info,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr()) {
    using T = typename arrow::Decimal128;
    const bodo_array_type::arr_type_enum ArrayType =
        bodo_array_type::NULLABLE_INT_BOOL;
    const Bodo_CTypes::CTypeEnum DType = Bodo_CTypes::DECIMAL;
    // Iterate across each group to find its percentile value
    for (size_t igrp = 0; igrp < grp_info.num_groups; igrp++) {
        // Set up a vector that will contain all of the values from the current
        // group as a Decimal128.
        bodo::vector<arrow::Decimal128> vect(pool);
        for (int64_t i = grp_info.group_to_first_row[igrp]; i != -1;
             i = grp_info.next_row_in_group[i]) {
            if (!is_null_at<ArrayType, T, DType>(*arr, i)) {
                T val = get_arr_item<ArrayType, T, DType>(*arr, i);
                vect.emplace_back(val);
            }
        }
        // If there were no non-null entries, output null.
        if (vect.size() == 0) {
            set_to_null<bodo_array_type::NULLABLE_INT_BOOL, arrow::Decimal128,
                        Bodo_CTypes::DECIMAL>(*out_arr, igrp);
            continue;
        }
        arrow::Decimal128 valReturn = arrow::Decimal128(0);
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
                auto res = valReturn.Rescale(arr->scale, out_arr->scale);
                CHECK_ARROW_AND_ASSIGN(
                    res, "Overflow in intermediate value calculation",
                    valReturn)
            } else {
                // Otherwise, we want a weighted average of the
                // values at k_exact and the next index.  Obtain
                // the k_exact-largest element by using std::nth
                // element to partially sort about that index.
                std::nth_element(vect.begin(), vect.begin() + k_exact,
                                 vect.end());
                arrow::Decimal128 v1 = vect[k_exact];
                // Then, find the minimum of the remaining elements
                arrow::Decimal128 v2 =
                    *std::min_element(vect.begin() + k_exact + 1, vect.end());
                // Linearly interpolate between v1 and v2 where
                // the weight is based on how close k_approx is
                // to k_exact vs k_exact + 1. E.g. if k_exact
                // is 12 and k_approx = 12.25, then the formula
                // is v1 + 0.25 * (v2 - v1) = 0.75*v1 + 0.25*v2

                // We first convert the interpolation factor into a Decimal128.
                arrow::Decimal128 interpolation_factor;
                auto conversion_result = arrow::Decimal128::FromReal(
                    k_approx - k_exact, arr->precision, arr->scale);
                CHECK_ARROW_AND_ASSIGN(conversion_result,
                                       "failed to convert float to decimal",
                                       interpolation_factor)

                // Compute valReturn = v1 + (interpolation_factor) * (v2 - v1)

                // From here on out, we need to do some intermediate
                // calculations that involve manipulating the precision and
                // scale such that the output precision matches Snowflake's
                // implementation.

                bool overflow = false;

                // Do addition with normal addition precision/scale rules.
                auto diff = add_or_subtract_decimal_scalars_util(
                    v2, arr->precision, arr->scale, v1, arr->precision,
                    arr->scale, std::min(arr->precision + 1, 38), arr->scale,
                    false, &overflow);
                if (overflow) {
                    throw std::runtime_error(
                        "Overflow in intermediate value calculation");
                }

                // Let delta = (interpolation_factor) * (v2 - v1).
                // The scale and precision of delta will need to follow the
                // multiplication rules, which will result in a higher scale
                // than desired.
                int delta_leading = (arr->precision - arr->scale) * 2;
                int delta_scale =
                    std::min(arr->scale * 2, std::max(arr->scale, 12));
                int delta_precision = std::min(delta_leading + delta_scale, 38);
                auto delta = multiply_decimal_scalars_util(
                    diff, std::min(arr->precision + 1, 38), arr->scale,
                    interpolation_factor, arr->precision, arr->scale,
                    delta_precision, delta_scale, &overflow);
                if (overflow) {
                    throw std::runtime_error(
                        "Overflow in intermediate value calculation");
                }

                // Thus, we need to truncate delta to the desired scale.
                // Cut off delta to out_arr->precision, out_arr->scale
                delta = decimalops::Truncate<false>(
                    delta, std::min(delta_scale + 3, 38), delta_scale,
                    out_arr->precision, out_arr->scale, out_arr->scale,
                    &overflow);

                // Finally, we can perform addition with the desired precision
                // and scale.
                auto finalValue = add_or_subtract_decimal_scalars_util(
                    v1, arr->precision, arr->scale, delta, out_arr->precision,
                    out_arr->scale, out_arr->precision, out_arr->scale, true,
                    &overflow);
                if (overflow) {
                    throw std::runtime_error(
                        "Overflow in intermediate value calculation");
                }

                valReturn = finalValue;
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
        set_non_null<bodo_array_type::NULLABLE_INT_BOOL, arrow::Decimal128,
                     Bodo_CTypes::DECIMAL>(*out_arr, igrp);
        set_arr_item<bodo_array_type::NULLABLE_INT_BOOL, arrow::Decimal128,
                     Bodo_CTypes::DECIMAL>(*out_arr, igrp, valReturn);
    }
}

template <bodo_array_type::arr_type_enum ArrayType>
void percentile_computation_dtype_helper(
    std::shared_ptr<array_info> arr, std::shared_ptr<array_info> out_arr,
    double percentile, bool interpolate, grouping_info const& grp_info,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr()) {
    switch (arr->dtype) {
        case Bodo_CTypes::INT8: {
            percentile_computation_util<ArrayType, Bodo_CTypes::INT8>(
                arr, out_arr, percentile, interpolate, grp_info, pool);
            break;
        }
        case Bodo_CTypes::UINT8: {
            percentile_computation_util<ArrayType, Bodo_CTypes::UINT8>(
                arr, out_arr, percentile, interpolate, grp_info, pool);
            break;
        }
        case Bodo_CTypes::INT16: {
            percentile_computation_util<ArrayType, Bodo_CTypes::INT16>(
                arr, out_arr, percentile, interpolate, grp_info, pool);
            break;
        }
        case Bodo_CTypes::UINT16: {
            percentile_computation_util<ArrayType, Bodo_CTypes::UINT16>(
                arr, out_arr, percentile, interpolate, grp_info, pool);
            break;
        }
        case Bodo_CTypes::INT32: {
            percentile_computation_util<ArrayType, Bodo_CTypes::INT32>(
                arr, out_arr, percentile, interpolate, grp_info, pool);
            break;
        }
        case Bodo_CTypes::UINT32: {
            percentile_computation_util<ArrayType, Bodo_CTypes::UINT32>(
                arr, out_arr, percentile, interpolate, grp_info, pool);
            break;
        }
        case Bodo_CTypes::INT64: {
            percentile_computation_util<ArrayType, Bodo_CTypes::INT64>(
                arr, out_arr, percentile, interpolate, grp_info, pool);
            break;
        }
        case Bodo_CTypes::UINT64: {
            percentile_computation_util<ArrayType, Bodo_CTypes::UINT64>(
                arr, out_arr, percentile, interpolate, grp_info, pool);
            break;
        }
        case Bodo_CTypes::FLOAT32: {
            percentile_computation_util<ArrayType, Bodo_CTypes::FLOAT32>(
                arr, out_arr, percentile, interpolate, grp_info, pool);
            break;
        }
        case Bodo_CTypes::FLOAT64: {
            percentile_computation_util<ArrayType, Bodo_CTypes::FLOAT64>(
                arr, out_arr, percentile, interpolate, grp_info, pool);
            break;
        }
        case Bodo_CTypes::DECIMAL: {
            // Specialized decimal implementation
            percentile_computation_decimal_util(arr, out_arr, percentile,
                                                interpolate, grp_info, pool);
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
                            grouping_info const& grp_info,
                            bodo::IBufferPool* const pool) {
    switch (arr->arr_type) {
        case bodo_array_type::NUMPY: {
            percentile_computation_dtype_helper<bodo_array_type::NUMPY>(
                arr, out_arr, percentile, interpolate, grp_info, pool);
            break;
        }
        case bodo_array_type::NULLABLE_INT_BOOL: {
            percentile_computation_dtype_helper<
                bodo_array_type::NULLABLE_INT_BOOL>(
                arr, out_arr, percentile, interpolate, grp_info, pool);
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
                        bool const use_sql_rules,
                        bodo::IBufferPool* const pool) {
    size_t num_group = grp_info.group_to_first_row.size();
    size_t siztype = numpy_item_size[arr->dtype];
    std::string error_msg = std::string("There is no median for the ") +
                            std::string(GetDtype_as_string(arr->dtype));
    if (arr->arr_type == bodo_array_type::STRING) {
        Bodo_PyErr_SetString(PyExc_RuntimeError, error_msg.c_str());
        return;
    }
    if (arr->dtype == Bodo_CTypes::DATE || arr->dtype == Bodo_CTypes::TIME ||
        arr->dtype == Bodo_CTypes::DATETIME ||
        arr->dtype == Bodo_CTypes::TIMEDELTA) {
        Bodo_PyErr_SetString(PyExc_RuntimeError, error_msg.c_str());
        return;
    }
    assert(out_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL);
    assert(out_arr->dtype == Bodo_CTypes::FLOAT64 ||
           (arr->dtype == Bodo_CTypes::DECIMAL &&
            out_arr->dtype == Bodo_CTypes::DECIMAL));

    // Median operation lambda for float columns
    auto median_operation = [&](auto const& isnan_entry) -> void {
        for (size_t igrp = 0; igrp < num_group; igrp++) {
            int64_t i = grp_info.group_to_first_row[igrp];
            bodo::vector<double> ListValue(pool);
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
                out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(
                    igrp, false);
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
            out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(igrp,
                                                                      true);
            out_arr->at<double, bodo_array_type::NULLABLE_INT_BOOL>(igrp) =
                valReturn;
        }
    };

    // Median operation lambda for decimal columns
    auto median_operation_decimal = [&](auto const& isnan_entry) -> void {
        for (size_t igrp = 0; igrp < num_group; igrp++) {
            int64_t i = grp_info.group_to_first_row[igrp];
            bodo::vector<arrow::Decimal128> ListValue(pool);
            bool HasNaN = false;
            while (true) {
                if (i == -1) {
                    break;
                }
                if (!isnan_entry(i)) {
                    char* ptr = arr->data1() + i * siztype;
                    arrow::Decimal128 eVal = GetTentry<arrow::Decimal128>(ptr);
                    ListValue.emplace_back(eVal);
                } else {
                    if (!skipna) {
                        HasNaN = true;
                        break;
                    }
                }
                i = grp_info.next_row_in_group[i];
            }
            auto GetKthValue = [&](size_t const& pos) -> arrow::Decimal128 {
                std::nth_element(ListValue.begin(), ListValue.begin() + pos,
                                 ListValue.end());
                return ListValue[pos];
            };
            arrow::Decimal128 decimal_val = 0;
            // a group can be empty if it has all NaNs so output will be NaN
            // even if skipna=True
            if (HasNaN || ListValue.size() == 0) {
                // We always set the output to NA.
                out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(
                    igrp, false);
                continue;
            } else {
                size_t len = ListValue.size();
                size_t res = len % 2;
                if (res == 0) {
                    size_t kMid1 = len / 2;
                    size_t kMid2 = kMid1 - 1;
                    decimal_val = (GetKthValue(kMid1) + GetKthValue(kMid2)) / 2;
                } else {
                    size_t kMid = len / 2;
                    decimal_val = GetKthValue(kMid);
                }
            }
            // Rescale the decimal appropriately
            bool overflow = false;
            boost::multiprecision::int256_t decimal_val_int256 =
                decimalops::ConvertToInt256(decimal_val);
            int scale_delta = out_arr->scale - arr->scale;
            decimal_val_int256 =
                decimalops::IncreaseScaleBy(decimal_val_int256, scale_delta);
            decimal_val =
                decimalops::ConvertToDecimal128(decimal_val_int256, &overflow);
            if (overflow) {
                std::string err_msg =
                    "Intermediate values for MEDIAN do not fit within "
                    "Decimal(" +
                    std::to_string(out_arr->precision) + ", " +
                    std::to_string(out_arr->scale) + ")";
                Bodo_PyErr_SetString(PyExc_RuntimeError, err_msg.c_str());
                return;
            }
            // The output array is always a nullable decimal array
            out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(igrp,
                                                                      true);
            out_arr->at<arrow::Decimal128, bodo_array_type::NULLABLE_INT_BOOL>(
                igrp) = decimal_val;
        }
    };

    if (arr->arr_type == bodo_array_type::NUMPY) {
        median_operation([=](size_t pos) -> bool {
            if (arr->dtype == Bodo_CTypes::FLOAT32) {
                return isnan(arr->at<float, bodo_array_type::NUMPY>(pos));
            }
            if (arr->dtype == Bodo_CTypes::FLOAT64) {
                return isnan(arr->at<double, bodo_array_type::NUMPY>(pos));
            }
            return false;
        });
    }
    if (arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        if (arr->dtype == Bodo_CTypes::DECIMAL) {
            median_operation_decimal([=](size_t pos) -> bool {
                return !arr->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(
                    pos);
            });
        } else {
            median_operation([=](size_t pos) -> bool {
                return !arr->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(
                    pos);
            });
        }
    }
}

// SHIFT

void shift_computation(std::shared_ptr<array_info> arr,
                       std::shared_ptr<array_info> out_arr,
                       grouping_info const& grp_info, int64_t const& periods,
                       bodo::IBufferPool* const pool,
                       std::shared_ptr<::arrow::MemoryManager> mm) {
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

    bodo::vector<int64_t> row_list(num_rows, pool);
    if (tmp_periods == 0) {
        for (size_t i = 0; i < num_rows; i++) {
            row_list[i] = i;
        }
    } else {
        int64_t gid;       // group number
        int64_t cur_pos;   // new value for current row
        int64_t prev_val;  // previous value
        bodo::vector<int64_t> nrows_per_group(
            num_groups, pool);  // array holding number of rows per group
        bodo::vector<std::vector<int64_t>> p_values(
            num_groups, std::vector<int64_t>(tmp_periods),
            pool);  // 2d array holding most recent
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
    std::shared_ptr<array_info> updated_col = RetrieveArray_SingleColumn(
        std::move(arr), row_list, false, pool, std::move(mm));
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
    const uint8_t* count_col_in_null_bitmask =
        (uint8_t*)count_col_in->null_bitmask();
    uint8_t* count_col_out_null_bitmask =
        (uint8_t*)count_col_out->null_bitmask();
    uint8_t* mean_col_out_null_bitmask = (uint8_t*)mean_col_out->null_bitmask();
    uint8_t* m2_col_out_null_bitmask = (uint8_t*)m2_col_out->null_bitmask();
    for (size_t i = 0; i < count_col_in->length; i++) {
        // Var always has null compute columns even
        // if there is an original numpy input. All arrays
        // will have the same null bit value so just check 1.
        if (GetBit(count_col_in_null_bitmask, i)) {
            int64_t group_num = grp_info.row_to_group[i];
            if (group_num == -1) {
                continue;
            }
            // TODO XXX If the arr types of count_col_out, count_col_in, etc.
            // are known, we need to template the getv calls.
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
            SetBitTo(count_col_out_null_bitmask, i, true);
            SetBitTo(mean_col_out_null_bitmask, i, true);
            SetBitTo(m2_col_out_null_bitmask, i, true);
        }
    }
}

// boolxor_agg
void boolxor_combine(const std::shared_ptr<array_info>& one_col_in,
                     const std::shared_ptr<array_info>& two_col_in,
                     const std::shared_ptr<array_info>& one_col_out,
                     const std::shared_ptr<array_info>& two_col_out,
                     grouping_info const& grp_info) {
    const uint8_t* one_col_in_null_bitmask =
        (uint8_t*)(one_col_in->null_bitmask());
    const uint8_t* one_col_in_data1 = (uint8_t*)(one_col_in->data1());
    const uint8_t* two_col_in_data1 = (uint8_t*)(two_col_in->data1());
    uint8_t* one_col_out_data1 = (uint8_t*)(one_col_out->data1());
    uint8_t* two_col_out_data1 = (uint8_t*)(two_col_out->data1());
    uint8_t* one_col_out_null_bitmask = (uint8_t*)(one_col_out->null_bitmask());
    uint8_t* two_col_out_null_bitmask = (uint8_t*)(two_col_out->null_bitmask());
    for (size_t i = 0; i < one_col_in->length; i++) {
        if (GetBit(one_col_in_null_bitmask, i)) {
            int64_t group_num = grp_info.row_to_group[i];
            if (group_num == -1) {
                continue;
            }

            // Fetch the input data
            bool one_in = GetBit(one_col_in_data1, i);
            bool two_in = GetBit(two_col_in_data1, i);

            // Get the existing group values
            bool one_out = GetBit(one_col_out_data1, group_num);
            bool two_out = GetBit(two_col_out_data1, group_num);
            two_out = two_out || two_in || (one_in && one_out);

            // Update the group values.
            one_out = one_out || one_in;
            SetBitTo(one_col_out_data1, group_num, one_out);
            SetBitTo(two_col_out_data1, group_num, two_out);
            // Set all the null bits to true.
            SetBitTo(one_col_out_null_bitmask, group_num, true);
            SetBitTo(two_col_out_null_bitmask, group_num, true);
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
    const uint8_t* count_col_in_null_bitmask =
        (uint8_t*)count_col_in->null_bitmask();
    uint8_t* count_col_out_null_bitmask =
        (uint8_t*)count_col_out->null_bitmask();
    uint8_t* m1_col_out_null_bitmask = (uint8_t*)m1_col_out->null_bitmask();
    uint8_t* m2_col_out_null_bitmask = (uint8_t*)m2_col_out->null_bitmask();
    uint8_t* m3_col_out_null_bitmask = (uint8_t*)m3_col_out->null_bitmask();

    for (size_t i = 0; i < count_col_in->length; i++) {
        if (GetBit(count_col_in_null_bitmask, i)) {
            int64_t group_num = grp_info.row_to_group[i];
            if (group_num == -1) {
                continue;
            }

            // XXX TODO Are the types of count_col_out, count_col_in, etc.
            // known? If they are we need to template the getv calls.
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
            SetBitTo(count_col_out_null_bitmask, group_num, true);
            SetBitTo(m1_col_out_null_bitmask, group_num, true);
            SetBitTo(m2_col_out_null_bitmask, group_num, true);
            SetBitTo(m3_col_out_null_bitmask, group_num, true);
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
    const uint8_t* count_col_in_null_bitmask =
        (uint8_t*)count_col_in->null_bitmask();
    uint8_t* count_col_out_null_bitmask =
        (uint8_t*)count_col_out->null_bitmask();
    uint8_t* m1_col_out_null_bitmask = (uint8_t*)m1_col_out->null_bitmask();
    uint8_t* m2_col_out_null_bitmask = (uint8_t*)m2_col_out->null_bitmask();
    uint8_t* m3_col_out_null_bitmask = (uint8_t*)m3_col_out->null_bitmask();
    uint8_t* m4_col_out_null_bitmask = (uint8_t*)m4_col_out->null_bitmask();
    for (size_t i = 0; i < count_col_in->length; i++) {
        if (GetBit(count_col_in_null_bitmask, i)) {
            int64_t group_num = grp_info.row_to_group[i];
            if (group_num == -1) {
                continue;
            }

            // XXX TODO Are arr types of count_col_out, etc. known statically?
            // If so, we need to template the getv calls! Else, we need to use
            // macros for the different possible arr types.
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
            SetBitTo(count_col_out_null_bitmask, group_num, true);
            SetBitTo(m1_col_out_null_bitmask, group_num, true);
            SetBitTo(m2_col_out_null_bitmask, group_num, true);
            SetBitTo(m3_col_out_null_bitmask, group_num, true);
            SetBitTo(m4_col_out_null_bitmask, group_num, true);
        }
    }
}

// ARRAY_AGG

/**
 * @brief performs the step of the ARRAY_AGG computation where the offsets
 * of the inner arrays are calculated based on the number of non-null entries
 * in each sorted group, and the non-null entries in the sorted groups are
 * copied over into a new inner array.
 *
 * This implementation is used for a nullable or numpy array without any nulls,
 * in which case the inner array can just be the same array as the sorted data.
 * Not allowed when using DISTINCT.
 *
 * @param[in] sorted_data the data from the column that is to be aggregated,
 * sorted first by group and then by the orderby columns.
 * @param[in] sorted_groups the sorted indices of groups, indicating where each
 *            array will begin/end.
 * @param[in,out] offset_buffer the buffer that we must write to in order to
 * specify the begining/ending cutoffs of each array in the inner array.
 * @param[in] num_group the number of distinct groups.
 * @returns the new inner array, e.g. a copy of sorted_data with nulls dropped.
 */
template <bodo_array_type::arr_type_enum ArrType, typename T,
          Bodo_CTypes::CTypeEnum DType>
    requires(nullable_array<ArrType> || numpy_array<ArrType>)
std::shared_ptr<array_info> array_agg_calculate_inner_arr_and_offsets_no_nulls(
    const std::shared_ptr<array_info>& sorted_data,
    const std::shared_ptr<array_info>& sorted_groups, offset_t* offset_buffer,
    size_t num_group) {
    size_t n_total = sorted_data->length;
    offset_buffer[0] = 0;
    offset_buffer[num_group] = n_total;
    int64_t cur_idx = 1;
    for (size_t i = 1; i < n_total; i++) {
        if (getv<int64_t>(sorted_groups, i) !=
            getv<int64_t>(sorted_groups, i - 1)) {
            offset_buffer[cur_idx] = i;
            cur_idx++;
        }
    }
    return sorted_data;
}

/**
 * @brief Determines whether a specific row from the sorted data should
 * be kept when copying over rows into the new inner array.
 *
 * This implementation is used for non-DISTINCT, so the only condition
 * is that the rows are non-null.
 *
 * @param[in] sorted_data the column of data sorted first by group and then
 * by orderby columns.
 * @param[in] sorted_groups the column of sorted group ids, ensuring that we
 * know that sorted_groups[i] is the group that sorted_data[i] belongs to.
 * @param[in] idx the index we are determining if we want to keep.
 * returns whether idx should be kept when copying over.
 */
template <bodo_array_type::arr_type_enum ArrType, typename T,
          Bodo_CTypes::CTypeEnum DType, bool is_distinct>
    requires(!is_distinct)
inline bool should_keep_row(const std::shared_ptr<array_info>& sorted_data,
                            const std::shared_ptr<array_info>& sorted_groups,
                            size_t idx) {
    return non_null_at<ArrType, T, DType>(*sorted_data, idx);
}

/**
 * @brief Determines whether a specific row from the sorted data should
 * be kept when copying over rows into the new inner array.
 *
 * This implementation is used for DISTINCT, so the conditions are that the
 * row must be non-null, and must be distinct from the previous row unless
 * this is the first row in the current group.
 *
 * @param[in] sorted_data the column of data sorted first by group and then
 * by orderby columns.
 * @param[in] sorted_groups the column of sorted group ids, ensuring that we
 * know that sorted_groups[i] is the group that sorted_data[i] belongs to.
 * @param[in] idx the index we are determining if we want to keep.
 * returns whether idx should be kept when copying over.
 */
template <bodo_array_type::arr_type_enum ArrType, typename T,
          Bodo_CTypes::CTypeEnum DType, bool is_distinct>
    requires(is_distinct)
inline bool should_keep_row(const std::shared_ptr<array_info>& sorted_data,
                            const std::shared_ptr<array_info>& sorted_groups,
                            size_t idx) {
    if (is_null_at<ArrType, T, DType>(*sorted_data, idx)) {
        return false;
    }
    // If this is the first row, or the previous row belongs to a different
    // group, then we don't need to check to see if it is distinct from the
    // previous row.
    if (idx == 0 || (getv<int64_t>(sorted_groups, idx) !=
                     getv<int64_t>(sorted_groups, idx - 1))) {
        return true;
    }
    return !TestEqualColumn<ArrType>(sorted_data, idx, sorted_data, idx - 1,
                                     true);
}

/**
 * @brief Performs the step of the ARRAY_AGG computation where the new inner
 * array is calculated. This is a special case where the input is a string
 * array but the only rows being dropped are nulls, so the entire char buffer
 * can be copied.
 *
 * @param[in] sorted_data the data from the column that is to be aggregated,
 * sorted first by group and then by the orderby columns.
 * @returns aa copy of sorted_data with nulls dropped.
 */
std::shared_ptr<array_info> array_agg_drop_string_nulls(
    const std::shared_ptr<array_info>& sorted_data, size_t non_null_count) {
    assert(sorted_data->arr_type == bodo_array_type::STRING);
    // Allocate a new offset buffer with the number of rows minus the nulls.
    size_t n_total = sorted_data->length;
    offset_t* in_offsets =
        (offset_t*)sorted_data->data2<bodo_array_type::STRING>();
    int64_t n_chars = in_offsets[n_total];
    std::shared_ptr<array_info> data_without_nulls =
        alloc_string_array(Bodo_CTypes::STRING, non_null_count, n_chars);
    offset_t* out_offsets =
        (offset_t*)data_without_nulls->data2<bodo_array_type::STRING>();
    out_offsets[non_null_count] = n_chars;

    // Move the non null offsets from the data column to the new array.
    size_t write_idx = 0;
    for (size_t i = 0; i < n_total; i++) {
        // If the current element is non null, write it to the next empty
        // position in data_without_nulls
        if (sorted_data->get_null_bit<bodo_array_type::STRING>(i)) {
            out_offsets[write_idx] = in_offsets[i];
            write_idx++;
        }
    }
    out_offsets[write_idx] = in_offsets[n_total];
    char* in_data = sorted_data->data1();
    char* out_data = data_without_nulls->data1<bodo_array_type::STRING>();
    memcpy(out_data, in_data, n_chars);
    return data_without_nulls;
}

template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType,
          bool is_distinct>
void array_agg_operation(
    const std::shared_ptr<array_info>& in_arr,
    std::shared_ptr<array_info> out_arr,
    const std::vector<std::shared_ptr<array_info>>& orderby_cols,
    const std::vector<bool>& ascending, const std::vector<bool>& na_position,
    const grouping_info& grp_info, const bool is_parallel,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager()) {
    using T = typename dtype_to_type<DType>::type;

    size_t num_group = grp_info.group_to_first_row.size();

    // Sort the table first by group (so each group has its elements
    // contiguous) and then by the orderby columns (so each group's elements
    // are internally sorted in the desired manner)
    std::shared_ptr<table_info> sorted_table =
        grouped_sort(grp_info, orderby_cols, {in_arr}, ascending, na_position,
                     1, is_parallel, pool, mm);
    int64_t n_sort_cols = orderby_cols.size() + 1;

    // Populate the offset buffer and calculate the indices that are to be
    // copied over.
    std::shared_ptr<array_info> sorted_data =
        sorted_table->columns[n_sort_cols];
    std::shared_ptr<array_info> sorted_groups = sorted_table->columns[0];
    offset_t* offset_buffer = (offset_t*)(out_arr->buffers[0]->mutable_data());
    std::vector<int64_t> rows_to_copy;
    size_t group_idx = 1;
    for (size_t i = 0; i < in_arr->length; i++) {
        // If the current group has just ended, then i is the next offset value
        if (i > 0 && getv<int64_t>(sorted_groups, i) !=
                         getv<int64_t>(sorted_groups, i - 1)) {
            offset_buffer[group_idx] = (rows_to_copy.size());
            group_idx++;
        }
        if (should_keep_row<ArrType, T, DType, is_distinct>(sorted_data,
                                                            sorted_groups, i)) {
            rows_to_copy.emplace_back(i);
        }
    }
    size_t kept_rows = rows_to_copy.size();
    offset_buffer[0] = 0;
    offset_buffer[num_group] = kept_rows;

    // Calculate the new inner array
    std::shared_ptr<array_info> inner_arr = out_arr->child_arrays[0];
    if (kept_rows == in_arr->length) {
        // If nothing was dropped, then we can keep the original array as the
        // new inner array
        *inner_arr = *sorted_data;
    } else if (ArrType == bodo_array_type::STRING && !is_distinct) {
        // If the input was a string array and we are only dropping nulls, a
        // special code path can be taken that can just copy the entire
        // character buffer.
        *inner_arr = *(array_agg_drop_string_nulls(sorted_data, kept_rows));
    } else {
        // Otherwise, use the kept rows vector to select that subset of rows
        *inner_arr = *(RetrieveArray_SingleColumn(sorted_data, rows_to_copy));
    }
}

template <bodo_array_type::arr_type_enum ArrType>
void array_agg_dtype_helper(
    const std::shared_ptr<array_info>& in_arr,
    std::shared_ptr<array_info> out_arr,
    const std::vector<std::shared_ptr<array_info>>& orderby_cols,
    const std::vector<bool>& ascending, const std::vector<bool>& na_position,
    const grouping_info& grp_info, const bool is_parallel,
    const bool is_distinct,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager()) {
#define ARRAY_AGG_DTYPE_CASE(dtype)                                    \
    case dtype: {                                                      \
        if (is_distinct) {                                             \
            array_agg_operation<ArrType, dtype, true>(                 \
                in_arr, out_arr, orderby_cols, ascending, na_position, \
                grp_info, is_parallel, pool, std::move(mm));           \
        } else {                                                       \
            array_agg_operation<ArrType, dtype, false>(                \
                in_arr, out_arr, orderby_cols, ascending, na_position, \
                grp_info, is_parallel, pool, std::move(mm));           \
        }                                                              \
        break;                                                         \
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
        ARRAY_AGG_DTYPE_CASE(Bodo_CTypes::INT128)
        ARRAY_AGG_DTYPE_CASE(Bodo_CTypes::FLOAT32)
        ARRAY_AGG_DTYPE_CASE(Bodo_CTypes::FLOAT64)
        ARRAY_AGG_DTYPE_CASE(Bodo_CTypes::DECIMAL)
        ARRAY_AGG_DTYPE_CASE(Bodo_CTypes::_BOOL)
        ARRAY_AGG_DTYPE_CASE(Bodo_CTypes::DATE)
        ARRAY_AGG_DTYPE_CASE(Bodo_CTypes::TIME)
        ARRAY_AGG_DTYPE_CASE(Bodo_CTypes::DATETIME)
        ARRAY_AGG_DTYPE_CASE(Bodo_CTypes::TIMEDELTA)
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
    const grouping_info& grp_info, const bool is_parallel,
    const bool is_distinct, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    switch (in_arr->arr_type) {
        case bodo_array_type::NUMPY: {
            array_agg_dtype_helper<bodo_array_type::NUMPY>(
                in_arr, out_arr, orderby_cols, ascending, na_position, grp_info,
                is_parallel, is_distinct, pool, std::move(mm));
            break;
        }
        case bodo_array_type::NULLABLE_INT_BOOL: {
            array_agg_dtype_helper<bodo_array_type::NULLABLE_INT_BOOL>(
                in_arr, out_arr, orderby_cols, ascending, na_position, grp_info,
                is_parallel, is_distinct, pool, std::move(mm));
            break;
        }
        case bodo_array_type::STRING: {
            if (is_distinct) {
                array_agg_operation<bodo_array_type::STRING,
                                    Bodo_CTypes::STRING, true>(
                    in_arr, out_arr, orderby_cols, ascending, na_position,
                    grp_info, is_parallel, pool, std::move(mm));
            } else {
                array_agg_operation<bodo_array_type::STRING,
                                    Bodo_CTypes::STRING, false>(
                    in_arr, out_arr, orderby_cols, ascending, na_position,
                    grp_info, is_parallel, pool, std::move(mm));
            }
            break;
        }
        case bodo_array_type::DICT: {
            if (is_distinct) {
                array_agg_operation<bodo_array_type::DICT, Bodo_CTypes::STRING,
                                    true>(in_arr, out_arr, orderby_cols,
                                          ascending, na_position, grp_info,
                                          is_parallel, pool, std::move(mm));
            } else {
                array_agg_operation<bodo_array_type::DICT, Bodo_CTypes::STRING,
                                    false>(in_arr, out_arr, orderby_cols,
                                           ascending, na_position, grp_info,
                                           is_parallel, pool, std::move(mm));
            }
            break;
        }
        case bodo_array_type::MAP: {
            if (is_distinct) {
                array_agg_operation<bodo_array_type::MAP, Bodo_CTypes::STRUCT,
                                    true>(in_arr, out_arr, orderby_cols,
                                          ascending, na_position, grp_info,
                                          is_parallel, pool, std::move(mm));
            } else {
                array_agg_operation<bodo_array_type::MAP, Bodo_CTypes::STRUCT,
                                    false>(in_arr, out_arr, orderby_cols,
                                           ascending, na_position, grp_info,
                                           is_parallel, pool, std::move(mm));
            }
            break;
        }
        case bodo_array_type::STRUCT: {
            if (is_distinct) {
                array_agg_operation<bodo_array_type::STRUCT,
                                    Bodo_CTypes::STRUCT, true>(
                    in_arr, out_arr, orderby_cols, ascending, na_position,
                    grp_info, is_parallel, pool, std::move(mm));
            } else {
                array_agg_operation<bodo_array_type::STRUCT,
                                    Bodo_CTypes::STRUCT, false>(
                    in_arr, out_arr, orderby_cols, ascending, na_position,
                    grp_info, is_parallel, pool, std::move(mm));
            }
            break;
        }
        case bodo_array_type::ARRAY_ITEM: {
            if (is_distinct) {
                array_agg_operation<bodo_array_type::ARRAY_ITEM,
                                    Bodo_CTypes::LIST, true>(
                    in_arr, out_arr, orderby_cols, ascending, na_position,
                    grp_info, is_parallel, pool, std::move(mm));
            } else {
                array_agg_operation<bodo_array_type::ARRAY_ITEM,
                                    Bodo_CTypes::LIST, false>(
                    in_arr, out_arr, orderby_cols, ascending, na_position,
                    grp_info, is_parallel, pool, std::move(mm));
            }
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

// OBJECT_AGG
void object_agg_computation(const std::shared_ptr<array_info>& key_col,
                            const std::shared_ptr<array_info>& val_col,
                            std::shared_ptr<array_info> out_arr,
                            const grouping_info& grp_info,
                            const bool is_parallel,
                            bodo::IBufferPool* const pool,
                            std::shared_ptr<::arrow::MemoryManager> mm) {
    std::shared_ptr<table_info> sorted_table = grouped_sort(
        grp_info, {}, {key_col, val_col}, {}, {}, 0, is_parallel, pool, mm);
    std::shared_ptr<array_info> groups = sorted_table->columns[0];
    std::shared_ptr<array_info> keys = sorted_table->columns[1];
    std::shared_ptr<array_info> values = sorted_table->columns[2];

    // Populate the offset buffer based on the number of pairs in each group
    size_t num_group = grp_info.group_to_first_row.size();
    size_t n_total = keys->length;
    offset_t* offset_buffer =
        (offset_t*)(out_arr->child_arrays[0]->buffers[0]->mutable_data());
    offset_buffer[0] = 0;
    size_t offset_idx = 1;
    // Declare a cutoff each time the group number (which the sorted table
    // was sorted by) changes, but ignore any rows along the way that were
    // null in either the key or value column. Also keep track of which rows
    // should be copied over.
    size_t non_null_count = 0;
    std::vector<int64_t> rows_to_copy;
    const uint8_t* keys_null_bitmask = (uint8_t*)keys->null_bitmask();
    // NOTE: values_null_bitmask will be nullptr in the NUMPY case.
    const uint8_t* values_null_bitmask = (uint8_t*)values->null_bitmask();
    const bool all_values_non_null = (values_null_bitmask == nullptr);
    assert(groups->arr_type == bodo_array_type::NUMPY);
    for (size_t i = 0; i < n_total; i++) {
        if (i > 0 && !TestEqualColumn<bodo_array_type::NUMPY>(groups, i, groups,
                                                              i - 1, true)) {
            offset_buffer[offset_idx] = non_null_count;
            offset_idx++;
        }
        if (GetBit(keys_null_bitmask, i) &&
            (all_values_non_null || GetBit(values_null_bitmask, i))) {
            non_null_count++;
            rows_to_copy.push_back(i);
        }
    }
    offset_buffer[num_group] = non_null_count;
    std::shared_ptr<array_info> key_array =
        RetrieveArray_SingleColumn(keys, rows_to_copy);
    std::shared_ptr<array_info> val_array =
        RetrieveArray_SingleColumn(values, rows_to_copy);
    std::vector<std::shared_ptr<array_info>> child_arrays(2);
    child_arrays[0] = key_array;
    child_arrays[1] = val_array;
    std::shared_ptr<array_info> inner_arr =
        alloc_struct(non_null_count, child_arrays);
    out_arr->child_arrays[0]->child_arrays[0] = inner_arr;
}

// NUNIQUE
void nunique_computation(std::shared_ptr<array_info> arr,
                         std::shared_ptr<array_info> out_arr,
                         grouping_info const& grp_info, bool const& dropna,
                         bool const& is_parallel,
                         bodo::IBufferPool* const pool) {
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
            eset({}, hash_fct, equal_fct, pool);
        eset.reserve(double(arr->length) / num_group);  // NOTE: num_group > 0
        eset.max_load_factor(UNORDERED_MAP_MAX_LOAD_FACTOR);

        char* arr_data1 = arr->data1();
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
                char* ptr = arr_data1 + (i * siztype);
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
    } else if (arr->arr_type == bodo_array_type::STRING) {
        offset_t* in_offsets = (offset_t*)arr->data2<bodo_array_type::STRING>();
        const uint32_t seed = SEED_HASH_CONTAINER;

        HashNuniqueComputationString hash_fct{arr, in_offsets, seed};
        KeyEqualNuniqueComputationString equal_fct{arr, in_offsets};
        bodo::unord_set_container<int64_t, HashNuniqueComputationString,
                                  KeyEqualNuniqueComputationString>
            eset({}, hash_fct, equal_fct, pool);
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
            eset({}, hash_fct, equal_fct, pool);
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

#undef CHECK_ARROW
#undef CHECK_ARROW_AND_ASSIGN
