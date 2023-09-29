// Copyright (C) 2023 Bodo Inc. All rights reserved.

#include "_groupby_do_apply_to_column.h"
#include "_groupby_agg_funcs.h"
#include "_groupby_eval.h"
#include "_groupby_ftypes.h"

/**
 * This file defines the functions that create the
 * general infrastructure used to apply most operations
 * to an individual column. This infrastructure is used
 * by update, combine, and eval.
 */

/**
 * @brief Gets the group number for a given row.
 *
 * @param[in] grp_info The grouping info.
 * @param row_num Row number.
 * @return int64_t The group number.
 */
inline int64_t get_group_for_row(const grouping_info& grp_info,
                                 int64_t const& row_num) {
    return grp_info.row_to_group[row_num];
}

/**
 * @brief Code path for applying the supported aggregation
 * functions to columns containing lists of strings.
 *
 * @tparam ftype The function to execute.
 * @param[in] in_col The input column. This must be a column with a list of
 * strings type.
 * @param[in, out] out_col The output column.
 * @param[in] grp_info The grouping information.
 */
template <int ftype>
void apply_to_column_list_string(std::shared_ptr<array_info> in_col,
                                 std::shared_ptr<array_info> out_col,
                                 const grouping_info& grp_info) {
    // for list strings, we are supporting count, sum, max, min, first, last
    // Optimized implementation for count.
    if (ftype == Bodo_FTypes::count) {
        for (size_t i = 0; i < in_col->length; i++) {
            int64_t i_grp = get_group_for_row(grp_info, i);
            if ((i_grp != -1) && in_col->get_null_bit(i)) {
                // Note: int is unused since we would only use an NA check.
                count_agg<int, Bodo_CTypes::LIST_STRING>::apply(
                    getv<int64_t>(out_col, i_grp), getv<int>(in_col, i));
            }
        }
        return;
    }
    size_t num_groups = grp_info.num_groups;
    bodo::vector<bodo::vector<std::pair<std::string, bool>>> ListListPair(
        num_groups);
    char* data_i = in_col->data1();
    offset_t* index_offsets_i = (offset_t*)in_col->data3();
    offset_t* data_offsets_i = (offset_t*)in_col->data2();
    uint8_t* sub_null_bitmask_i = (uint8_t*)in_col->sub_null_bitmask();
    // Computing the strings used in output.
    uint64_t n_bytes = (num_groups + 7) >> 3;
    bodo::vector<uint8_t> Vmask(n_bytes, 0);
    for (size_t i = 0; i < in_col->length; i++) {
        int64_t i_grp = get_group_for_row(grp_info, i);
        if ((i_grp != -1) && in_col->get_null_bit(i)) {
            bool out_bit_set = out_col->get_null_bit(i_grp);
            if (ftype == Bodo_FTypes::first && out_bit_set)
                continue;
            offset_t start_offset = index_offsets_i[i];
            offset_t end_offset = index_offsets_i[i + 1];
            offset_t len = end_offset - start_offset;
            bodo::vector<std::pair<std::string, bool>> LStrB(len);
            for (offset_t i = 0; i < len; i++) {
                offset_t len_str = data_offsets_i[start_offset + i + 1] -
                                   data_offsets_i[start_offset + i];
                offset_t pos_start = data_offsets_i[start_offset + i];
                std::string val(&data_i[pos_start], len_str);
                bool str_bit = GetBit(sub_null_bitmask_i, start_offset + i);
                LStrB[i] = {val, str_bit};
            }
            if (out_bit_set) {
                aggliststring<ftype>::apply(ListListPair[i_grp], LStrB);
            } else {
                ListListPair[i_grp] = LStrB;
                out_col->set_null_bit(i_grp, true);
            }
        }
    }
    // Allocate the new column. Since these column are immutable
    // we don't modify the column directly.
    std::shared_ptr<array_info> new_out_col =
        create_list_string_array(Vmask, ListListPair);
    // Copy the contents into out column.
    *out_col = std::move(*new_out_col);
}

/**
 * @brief Helper function used to determine the valid groups for the series of
 * idx*** functions. This function needs to be inlined because it is called from
 * a tight loop.
 *
 * @tparam ftype The function type.
 * @param i_grp The group number.
 * @param row_num The row_number
 * @param[in] in_col The original input column.
 * @param[in] index_pos The original index column.
 * @return true This group is valid.
 * @return false This group is not valid and should be skipped.
 */
template <int ftype>
inline bool idx_func_valid_group(int64_t i_grp, size_t row_num,
                                 const std::shared_ptr<array_info>& in_col,
                                 const std::shared_ptr<array_info>& index_pos) {
    switch (ftype) {
        case Bodo_FTypes::idxmax_na_first:
        case Bodo_FTypes::idxmin_na_first:
            // If we are putting NA values first then we stop
            // visiting a group once we see a NA value. We
            // initialize the output values to all be non-NA values
            // to ensure this.
            return (i_grp != -1) & index_pos->get_null_bit(i_grp);
        default:
            // idxmin and idxmax only work on non-null entries
            return (i_grp != -1) & in_col->get_null_bit(row_num);
    }
}

/**
 * @brief Determine when an idx*** function should take the non-na path.
 * The only functions that have an NA path are idxmax_na_first and
 * idxmin_na_first, so all other functions return true. This function needs to
 * be inlined because it is called from a tight loop.
 *
 * @tparam ftype The function type.
 * @param row_num The current row number.
 * @param[in] in_col The input column.
 * @return true Should the non-NA input path be taken.
 * @return false Should the NA input path be taken?
 */
template <int ftype>
inline bool idx_func_take_non_na_path(
    size_t row_num, const std::shared_ptr<array_info>& in_col) {
    switch (ftype) {
        case Bodo_FTypes::idxmax_na_first:
        case Bodo_FTypes::idxmin_na_first:
            // idxmax_na_first and idxmin_na_first handle
            // null values as the selected values.
            return in_col->get_null_bit(row_num);
        default:
            // If a group is valid idxmin and idxmax
            // only work on non-NA values.
            return true;
    }
}

/**
 * @brief Generate the function call for the idx*** functions with a string
 * input. This either generates the appropriate max or min call.
 * This function needs to be inlined because it is called from a tight loop.
 *
 * @tparam ftype The function type.
 * @param[in, out] ListString The output vector to update.
 * @param[in, out] index_pos The column of index values to update.
 * @param[in] val The current row's value.
 * @param row_num The current row number.
 * @param i_grp The current group number.
 */
template <int ftype>
inline void idx_string_func_apply_to_row(std::string& orig_val,
                                         std::string& new_val,
                                         uint64_t& index_pos, size_t row_num) {
    switch (ftype) {
        case Bodo_FTypes::idxmax_na_first:
        case Bodo_FTypes::idxmax:
            return idxmax_string(orig_val, new_val, index_pos, row_num);
        default:
            return idxmin_string(orig_val, new_val, index_pos, row_num);
    }
}

/**
 * @brief Generate the function call for the idx*** functions with a dict
 * encoded string input. This either generates the appropriate max or min call.
 * This function needs to be inlined because it is called from a tight loop.
 *
 * @tparam ftype The function type.
 * @param[in, out] org_ind The current dictionary value.
 * @param[in, out] dict_ind The candidate dictionary value.
 * @param[in] s1 The current value for the group.
 * @param[in] s2 The candidate value for the group.
 * @param[in, out] index_pos Tracking the row number for the group output.
 * @param row_num The current row number.
 * @param i_grp The current group number.
 */
template <int ftype>
inline void idx_dict_func_apply_to_row(int32_t& org_ind, int32_t& dict_ind,
                                       std::string& s1, std::string& s2,
                                       uint64_t& index_pos, size_t row_num) {
    switch (ftype) {
        case Bodo_FTypes::idxmax_na_first:
        case Bodo_FTypes::idxmax:
            return idxmax_dict(org_ind, dict_ind, s1, s2, index_pos, row_num);
        default:
            return idxmin_dict(org_ind, dict_ind, s1, s2, index_pos, row_num);
    }
}

/**
 * @brief Perform SUM on a column of string arrays and updates the output
 * column.
 *
 * @param[in] in_col The input column. This must be a string column.
 * @param[in, out] out_col The output column.
 * @param[in] grp_info The grouping information.
 */
void apply_sum_to_column_string(std::shared_ptr<array_info> in_col,
                                std::shared_ptr<array_info> out_col,
                                const grouping_info& grp_info) {
    // allocate output array (length is number of groups, number of chars same
    // as input)
    size_t num_groups = grp_info.num_groups;
    int64_t n_chars = in_col->n_sub_elems();
    std::shared_ptr<array_info> out_arr =
        alloc_string_array(out_col->dtype, num_groups, n_chars);
    size_t n_bytes = (num_groups + 7) >> 3;
    memset(out_arr->null_bitmask(), 0xff, n_bytes);  // null not possible

    // find offsets for each output string
    bodo::vector<offset_t> str_offsets(num_groups + 1, 0);
    char* data_i = in_col->data1();
    offset_t* offsets_i = (offset_t*)in_col->data2();
    char* data_o = out_arr->data1();
    offset_t* offsets_o = (offset_t*)out_arr->data2();

    for (size_t i = 0; i < in_col->length; i++) {
        int64_t i_grp = get_group_for_row(grp_info, i);
        if ((i_grp != -1) && in_col->get_null_bit(i)) {
            offset_t len = offsets_i[i + 1] - offsets_i[i];
            str_offsets[i_grp + 1] += len;
        }
    }
    std::partial_sum(str_offsets.begin(), str_offsets.end(),
                     str_offsets.begin());
    memcpy(offsets_o, str_offsets.data(), (num_groups + 1) * sizeof(offset_t));

    // copy characters to output
    for (size_t i = 0; i < in_col->length; i++) {
        int64_t i_grp = get_group_for_row(grp_info, i);
        if ((i_grp != -1) && in_col->get_null_bit(i)) {
            offset_t len = offsets_i[i + 1] - offsets_i[i];
            memcpy(&data_o[str_offsets[i_grp]], data_i + offsets_i[i], len);
            str_offsets[i_grp] += len;
        }
    }
    // Create an intermediate array since the output array is immutable
    // and copy the data over.
    *out_col = std::move(*out_arr);
}

/**
 * @brief Codepath for applying the supported aggregation functions
 * to columns of strings.
 *
 * @tparam ftype The function type
 * @param[in] in_col The input column. This must be a string column.
 * @param[in, out] out_col The output column.
 * @param[in, out] aux_cols Any auxillary columns used for computing the
 * outputs. This is used by operations like idxmax that require tracking more
 * than 1 value.
 * @param[in] grp_info The grouping information.
 */
template <int ftype>
void apply_to_column_string(std::shared_ptr<array_info> in_col,
                            std::shared_ptr<array_info> out_col,
                            std::vector<std::shared_ptr<array_info>>& aux_cols,
                            const grouping_info& grp_info) {
    size_t num_groups = grp_info.num_groups;
    size_t n_bytes = (num_groups + 7) >> 3;
    bodo::vector<uint8_t> V(n_bytes, 0);
    bodo::vector<std::string> ListString(num_groups);
    char* data_i = in_col->data1();
    offset_t* offsets_i = (offset_t*)in_col->data2();
    switch (ftype) {
        case Bodo_FTypes::count: {
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = get_group_for_row(grp_info, i);
                if ((i_grp != -1) && in_col->get_null_bit(i)) {
                    // Note: int is unused since we would only use an NA check.
                    count_agg<int, Bodo_CTypes::STRING>::apply(
                        getv<int64_t>(out_col, i_grp), getv<int>(in_col, i));
                }
            }
            break;
        }
        case Bodo_FTypes::bitor_agg:
        case Bodo_FTypes::bitand_agg:
        case Bodo_FTypes::bitxor_agg: {
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = get_group_for_row(grp_info, i);
                if ((i_grp != -1) && in_col->get_null_bit(i)) {
                    // Get isolated string from data_i
                    offset_t start_offset = offsets_i[i];
                    offset_t end_offset = offsets_i[i + 1];
                    offset_t len = end_offset - start_offset;
                    std::string substr(&data_i[start_offset], len);

                    double double_substr = 0;
                    try {
                        double_substr = stod(substr);
                    } catch (...) {
                        throw std::runtime_error(
                            "Failed to cast string to double. Use of bitor_agg "
                            "with strings requires that the strings are "
                            "castable to numeric types.");
                    }

                    // Get output value and perform operation on it
                    int64_t& out_val = getv<int64_t>(out_col, i_grp);
                    casted_aggfunc<int64_t, double, Bodo_CTypes::FLOAT64,
                                   ftype>::apply(out_val, double_substr);
                    out_col->set_null_bit(i_grp, true);
                }
            }
            break;
        }
        // optimized groupby sum for strings (concatenation)
        case Bodo_FTypes::sum: {
            apply_sum_to_column_string(in_col, out_col, grp_info);
            break;
        }
        case Bodo_FTypes::idxmax:
        case Bodo_FTypes::idxmin:
        case Bodo_FTypes::idxmax_na_first:
        case Bodo_FTypes::idxmin_na_first: {
            // General framework for the idxmin/idxmax functions. We use
            // inlined functions to handle the differences between these
            // cases and avoid function call overhead
            std::shared_ptr<array_info> index_pos = aux_cols[0];
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = get_group_for_row(grp_info, i);
                // Code path taken by all code when we might want to look
                // at a row.
                if (idx_func_valid_group<ftype>(i_grp, i, in_col, index_pos)) {
                    if (idx_func_take_non_na_path<ftype>(i, in_col)) {
                        // This assumes the input row is not NA.
                        bool out_bit_set = GetBit(V.data(), i_grp);
                        offset_t start_offset = offsets_i[i];
                        offset_t end_offset = offsets_i[i + 1];
                        offset_t len = end_offset - start_offset;
                        std::string val(&data_i[start_offset], len);
                        if (out_bit_set) {
                            std::string& orig_val = ListString[i_grp];
                            uint64_t& index_val =
                                getv<uint64_t>(index_pos, i_grp);
                            idx_string_func_apply_to_row<ftype>(orig_val, val,
                                                                index_val, i);
                        } else {
                            ListString[i_grp] = val;
                            getv<uint64_t>(index_pos, i_grp) = i;
                            SetBitTo(V.data(), i_grp, true);
                        }
                    } else {
                        // idxmax_na_first and idxmin_na_first only!
                        // If we have an NA value mark this group as
                        // done and update the index
                        getv<uint64_t>(index_pos, i_grp) = i;
                        // set the null bit for the count so we know
                        // to stop visiting the group. The data is still
                        // valid.
                        index_pos->set_null_bit(i_grp, false);
                    }
                }
            }
            // TODO: Avoid creating the output array for
            // idxmin/idxmax/idxmin_na_first/idxmax_na_first
            // Determining the number of characters in output.
            // Create an intermediate array since the output array is immutable
            // and copy the data over.
            std::shared_ptr<array_info> new_out_col =
                create_string_array(out_col->dtype, V, ListString);
            *out_col = std::move(*new_out_col);
            break;
        }
        default:
            // Computing the strings used in output.
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = get_group_for_row(grp_info, i);
                if ((i_grp != -1) && in_col->get_null_bit(i)) {
                    bool out_bit_set = GetBit(V.data(), i_grp);
                    if (ftype == Bodo_FTypes::first && out_bit_set) {
                        continue;
                    }
                    offset_t start_offset = offsets_i[i];
                    offset_t end_offset = offsets_i[i + 1];
                    offset_t len = end_offset - start_offset;
                    std::string val(&data_i[start_offset], len);
                    if (out_bit_set) {
                        aggstring<ftype>::apply(ListString[i_grp], val);
                    } else {
                        ListString[i_grp] = val;
                        SetBitTo(V.data(), i_grp, true);
                    }
                }
            }
            std::shared_ptr<array_info> new_out_col =
                create_string_array(out_col->dtype, V, ListString);
            *out_col = std::move(*new_out_col);
            break;
    }
}

/**
 * @brief Apply sum operation on a dictionary-encoded string column. The
 * partial_sum trick used for regular string columns to calculate the offsets is
 * also used here. out_col is a regular string array instead of a
 * dictionary-encoded string array.
 *
 * @param[in] in_col: the input dictionary-encoded string array
 * @param[in, out] out_col: the output string array
 * @param[in] grp_info: groupby information
 */
void apply_sum_to_column_dict(std::shared_ptr<array_info> in_col,
                              std::shared_ptr<array_info> out_col,
                              const grouping_info& grp_info) {
    size_t num_groups = grp_info.num_groups;
    int64_t n_chars = 0;
    size_t n_bytes = (num_groups + 7) >> 3;

    // calculate the total number of characters in the dict-encoded array
    // and find offsets for each output string
    // every string has a start and end offset so len(offsets) == (len(data) +
    // 1)
    bodo::vector<offset_t> str_offsets(num_groups + 1, 0);
    char* data_i = in_col->child_arrays[0]->data1();
    offset_t* offsets_i = (offset_t*)in_col->child_arrays[0]->data2();
    for (size_t i = 0; i < in_col->length; i++) {
        int64_t i_grp = get_group_for_row(grp_info, i);
        if ((i_grp != -1) && in_col->child_arrays[1]->get_null_bit(i)) {
            int32_t dict_ind = getv<int32_t>(in_col->child_arrays[1], i);
            offset_t len = offsets_i[dict_ind + 1] - offsets_i[dict_ind];
            str_offsets[i_grp + 1] += len;
            n_chars += len;
        }
    }

    std::shared_ptr<array_info> out_arr =
        alloc_string_array(out_col->dtype, num_groups, n_chars);
    memset(out_arr->null_bitmask(), 0xff, n_bytes);  // null not possible
    char* data_o = out_arr->data1();
    offset_t* offsets_o = (offset_t*)out_arr->data2();

    std::partial_sum(str_offsets.begin(), str_offsets.end(),
                     str_offsets.begin());
    memcpy(offsets_o, str_offsets.data(), (num_groups + 1) * sizeof(offset_t));

    // copy characters to output
    for (size_t i = 0; i < in_col->length; i++) {
        int64_t i_grp = get_group_for_row(grp_info, i);
        if ((i_grp != -1) && in_col->child_arrays[1]->get_null_bit(i)) {
            int32_t dict_ind = getv<int32_t>(in_col->child_arrays[1], i);
            offset_t len = offsets_i[dict_ind + 1] - offsets_i[dict_ind];
            memcpy(&data_o[str_offsets[i_grp]], data_i + offsets_i[dict_ind],
                   len);
            str_offsets[i_grp] += len;
        }
    }
    // Create an intermediate array since the output array is immutable
    // and copy the data over.
    *out_col = std::move(*out_arr);
}

/**
 * @brief Applies the supported aggregate functions
 * on a dictionary encoded string column.
 *
 * @tparam ftype The function type.
 * @param[in] in_col: the input dictionary encoded string column
 * @param[in, out] out_col: the output dictionary encoded string column
 * @param[in] grp_info: groupby information
 */
template <int ftype>
void apply_to_column_dict(std::shared_ptr<array_info> in_col,
                          std::shared_ptr<array_info> out_col,
                          std::vector<std::shared_ptr<array_info>>& aux_cols,
                          const grouping_info& grp_info) {
    size_t num_groups = grp_info.num_groups;
    size_t n_bytes = (num_groups + 7) >> 3;
    std::shared_ptr<array_info> indices_arr = nullptr;
    if (ftype != Bodo_FTypes::count && ftype != Bodo_FTypes::sum &&
        ftype != Bodo_FTypes::bitor_agg && ftype != Bodo_FTypes::bitand_agg &&
        ftype != Bodo_FTypes::bitxor_agg) {
        // Allocate the indices. Count and sum don't use this array.
        indices_arr = alloc_nullable_array(num_groups, Bodo_CTypes::INT32, 0);
    }
    bodo::vector<uint8_t> V(n_bytes,
                            0);  // bitmask to mark if group's been updated
    char* data_i = in_col->child_arrays[0]->data1();
    offset_t* offsets_i = (offset_t*)in_col->child_arrays[0]->data2();
    switch (ftype) {
        case Bodo_FTypes::count: {
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = get_group_for_row(grp_info, i);
                if (i_grp != -1 && in_col->child_arrays[1]->get_null_bit(i)) {
                    // Note: int is unused since we would only use an NA check.
                    count_agg<int, Bodo_CTypes::STRING>::apply(
                        getv<int64_t>(out_col, i_grp), getv<int>(in_col, i));
                }
            }
            return;
        }
        case Bodo_FTypes::bitor_agg:
        case Bodo_FTypes::bitand_agg:
        case Bodo_FTypes::bitxor_agg: {
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = get_group_for_row(grp_info, i);
                // Get index of the string data
                int32_t& dict_idx = getv<int32_t>(in_col->child_arrays[1], i);
                if ((i_grp != -1) && in_col->child_arrays[1]->get_null_bit(i)) {
                    // Get isolated string from data_i
                    offset_t start_offset = offsets_i[dict_idx];
                    offset_t end_offset = offsets_i[dict_idx + 1];
                    offset_t len = end_offset - start_offset;
                    std::string substr(&data_i[start_offset], len);

                    // Can potentially cache converted doubles due to properties
                    // of dictionary encoded strings:
                    // https://github.com/Bodo-inc/Bodo/pull/5471/files/f14a47db40b3abbcb968b6de84ab4ddb0b2f1e83#r1212472922
                    double double_substr = 0;
                    try {
                        double_substr = stod(substr);
                    } catch (...) {
                        throw std::runtime_error(
                            "Failed to cast string to double. Use of bitor_agg "
                            "with strings requires that the strings are "
                            "castable to numeric types.");
                    }

                    // Get output value and perform operation on it
                    int64_t& out_val = getv<int64_t>(out_col, i_grp);
                    casted_aggfunc<int64_t, double, Bodo_CTypes::FLOAT64,
                                   ftype>::apply(out_val, double_substr);
                    out_col->set_null_bit(i_grp, true);
                }
            }
            return;
        }
        // optimized groupby sum for strings (concatenation)
        case Bodo_FTypes::sum: {
            return apply_sum_to_column_dict(in_col, out_col, grp_info);
        }
        case Bodo_FTypes::idxmax:
        case Bodo_FTypes::idxmin:
        case Bodo_FTypes::idxmax_na_first:
        case Bodo_FTypes::idxmin_na_first: {
            std::shared_ptr<array_info> index_pos = aux_cols[0];
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = get_group_for_row(grp_info, i);
                // Code path taken by all code when we might want to look
                // at a row.
                if (idx_func_valid_group<ftype>(i_grp, i, in_col, index_pos)) {
                    if (idx_func_take_non_na_path<ftype>(i, in_col)) {
                        // This assumes the input row is not NA.
                        bool out_bit_set = GetBit(V.data(), i_grp);
                        int32_t& dict_ind =
                            getv<int32_t>(in_col->child_arrays[1], i);
                        int32_t& org_ind = getv<int32_t>(indices_arr, i_grp);
                        if (out_bit_set) {
                            // Get the address and length of the new value to be
                            // compared
                            offset_t start_offset = offsets_i[dict_ind];
                            offset_t end_offset = offsets_i[dict_ind + 1];
                            offset_t len = end_offset - start_offset;
                            std::string s2(&data_i[start_offset], len);
                            // Get the address and length of the cumulative
                            // result
                            offset_t start_offset_org = offsets_i[org_ind];
                            offset_t end_offset_org = offsets_i[org_ind + 1];
                            offset_t len_org =
                                end_offset_org - start_offset_org;
                            std::string s1(&data_i[start_offset_org], len_org);
                            uint64_t& index_val =
                                getv<uint64_t>(index_pos, i_grp);
                            idx_dict_func_apply_to_row<ftype>(
                                org_ind, dict_ind, s1, s2, index_val, i);
                        } else {
                            org_ind = dict_ind;
                            getv<uint64_t>(index_pos, i_grp) = i;
                            SetBitTo(V.data(), i_grp, true);
                            indices_arr->set_null_bit(i_grp, true);
                        }
                    } else {
                        // idxmax_na_first and idxmin_na_first only!
                        // If we have an NA value mark this group as
                        // done and update the index
                        getv<uint64_t>(index_pos, i_grp) = i;
                        // set the null bit for the count so we know
                        // to stop visiting the group. The data is still
                        // valid.
                        index_pos->set_null_bit(i_grp, false);
                    }
                }
            }
            break;
        }
        case Bodo_FTypes::last: {
            // Define a specialized implementation of last
            // so we avoid allocating for the underlying strings.
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = get_group_for_row(grp_info, i);
                if ((i_grp != -1) && in_col->child_arrays[1]->get_null_bit(i)) {
                    bool out_bit_set = GetBit(V.data(), i_grp);
                    int32_t& dict_ind =
                        getv<int32_t>(in_col->child_arrays[1], i);
                    int32_t& org_ind = getv<int32_t>(indices_arr, i_grp);
                    if (out_bit_set) {
                        aggfunc<int32_t, Bodo_CTypes::STRING,
                                Bodo_FTypes::last>::apply(org_ind, dict_ind);
                    } else {
                        org_ind = dict_ind;
                        SetBitTo(V.data(), i_grp, true);
                        indices_arr->set_null_bit(i_grp, true);
                    }
                }
            }
            break;
        }
        default:
            // Populate the new indices array.
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = get_group_for_row(grp_info, i);
                if ((i_grp != -1) && in_col->child_arrays[1]->get_null_bit(i)) {
                    bool out_bit_set = GetBit(V.data(), i_grp);
                    if (ftype == Bodo_FTypes::first && out_bit_set) {
                        continue;
                    }
                    int32_t& dict_ind =
                        getv<int32_t>(in_col->child_arrays[1], i);
                    int32_t& org_ind = getv<int32_t>(indices_arr, i_grp);
                    if (out_bit_set) {
                        // Get the address and length of the new value to be
                        // compared
                        offset_t start_offset = offsets_i[dict_ind];
                        offset_t end_offset = offsets_i[dict_ind + 1];
                        offset_t len = end_offset - start_offset;
                        std::string s2(&data_i[start_offset], len);
                        // Get the address and length of the cumulative result
                        offset_t start_offset_org = offsets_i[org_ind];
                        offset_t end_offset_org = offsets_i[org_ind + 1];
                        offset_t len_org = end_offset_org - start_offset_org;
                        std::string s1(&data_i[start_offset_org], len_org);
                        aggdict<ftype>::apply(org_ind, dict_ind, s1, s2);
                    } else {
                        org_ind = dict_ind;
                        SetBitTo(V.data(), i_grp, true);
                        indices_arr->set_null_bit(i_grp, true);
                    }
                }
            }
    }
    // TODO: Avoid creating the output dict array if have
    // idxmin/idxmax/idxmin_na_first/idxmax_na_first
    // Start at 1 since 0 is is returned by the hashmap if data needs to be
    // inserted.
    int32_t k = 1;
    bodo::unord_map_container<int32_t, int32_t>
        old_to_new;  // Maps old index to new index
    old_to_new.reserve(num_groups);
    for (size_t i = 0; i < num_groups; i++) {
        // check if the value for the group is NaN
        if (!indices_arr->get_null_bit(i)) {
            continue;
        }
        // Insert 0 into the map if key is not in it.
        int32_t& old_ind = getv<int32_t>(indices_arr, i);
        int32_t& new_ind = old_to_new[old_ind];
        if (new_ind == 0) {
            new_ind =
                k++;  // Updates the value in the map without another lookup
        }
        old_ind =
            old_to_new[old_ind] - 1;  // map back from 1-indexing to 0-indexing
    }
    // Create new dict string array from map
    size_t n_dict = old_to_new.size();
    n_bytes = (n_dict + 7) >> 3;
    bodo::vector<uint8_t> bitmask_vec(n_bytes, 0);
    bodo::vector<std::string> ListString(n_dict);
    for (auto& it : old_to_new) {
        offset_t start_offset = offsets_i[it.first];
        offset_t end_offset = offsets_i[it.first + 1];
        offset_t len = end_offset - start_offset;
        std::string val(&data_i[start_offset], len);
        ListString[it.second - 1] = val;  // -1 to account for the 1 offset
        SetBitTo(bitmask_vec.data(), it.second - 1, true);
    }
    std::shared_ptr<array_info> dict_arr =
        create_string_array(Bodo_CTypes::STRING, bitmask_vec, ListString);
    std::shared_ptr<array_info> new_out_col =
        create_dict_string_array(dict_arr, indices_arr);
    // Create an intermediate array since the output array is immutable
    // and copy the data over.
    *out_col = std::move(*new_out_col);
}

/**
 * @brief Apply the given aggregate function to the categorical
 * array.
 *
 * @tparam T The actual C++ type for the array values.
 * @tparam ftype The aggregation function type.
 * @tparam dtype The array dtype.
 * @param[in] in_col The input array. Must be a categorical array.
 * @param[in, out] out_col The output array.
 * @param[in] grp_info The grouping information.
 */
template <typename T, int ftype, Bodo_CTypes::CTypeEnum DType>
void apply_to_column_categorical(std::shared_ptr<array_info> in_col,
                                 std::shared_ptr<array_info> out_col,
                                 const grouping_info& grp_info) {
    switch (ftype) {
        case Bodo_FTypes::count: {
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = get_group_for_row(grp_info, i);
                if (i_grp != -1) {
                    T& val = getv<T>(in_col, i);
                    if (!isnan_categorical<T, DType>(val)) {
                        count_agg<T, DType>::apply(
                            getv<int64_t>(out_col, i_grp), val);
                    }
                }
            }
            return;
        }
        case Bodo_FTypes::min: {
            // NOTE: Bodo_FTypes::max is handled for categorical type
            // since NA is -1.
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = get_group_for_row(grp_info, i);
                if (i_grp != -1) {
                    T& val = getv<T>(in_col, i);
                    if (!isnan_categorical<T, DType>(val)) {
                        aggfunc<T, DType, ftype>::apply(getv<T>(out_col, i_grp),
                                                        val);
                    }
                }
            }
            // aggfunc_output_initialize_kernel, min defaults
            // to num_categories if all entries are NA
            // this needs to be replaced with -1
            for (size_t i = 0; i < out_col->length; i++) {
                T& val = getv<T>(out_col, i);
                set_na_if_num_categories<T, DType>(val,
                                                   out_col->num_categories);
            }
            return;
        }
        case Bodo_FTypes::max:
        case Bodo_FTypes::last: {
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = get_group_for_row(grp_info, i);
                if (i_grp != -1) {
                    T& val = getv<T>(in_col, i);
                    if (!isnan_categorical<T, DType>(val)) {
                        aggfunc<T, DType, ftype>::apply(getv<T>(out_col, i_grp),
                                                        val);
                    }
                }
            }
            return;
        }
        case Bodo_FTypes::first: {
            int64_t n_bytes = ((out_col->length + 7) >> 3);
            bodo::vector<uint8_t> bitmask_vec(n_bytes, 0);
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = get_group_for_row(grp_info, i);
                T val = getv<T>(in_col, i);
                if ((i_grp != -1) && !GetBit(bitmask_vec.data(), i_grp) &&
                    !isnan_categorical<T, DType>(val)) {
                    getv<T>(out_col, i_grp) = val;
                    SetBitTo(bitmask_vec.data(), i_grp, true);
                }
            }
            return;
        }
        default:
            throw std::runtime_error(
                std::string(
                    "do_apply_to_column: unsupported Categorical function: ") +
                get_name_for_Bodo_FTypes(ftype));
    }
}

/**
 * @brief Apply the given aggregation function to an input Numpy
 * array.
 *
 * @tparam T The actual C++ type for the underlying array data.
 * @tparam ftype The function type.
 * @tparam dtype The Bodo dtype for the array.
 * @param[in] in_col The input column. This must be a numpy array type.
 * @param[in, out] out_col The output column.
 * @param[in, out] aux_cols Auxillary columns used to for certain operations
 * that require multiple columns (e.g. mean).
 * @param[in] grp_info The grouping information.
 */
template <typename T, int ftype, Bodo_CTypes::CTypeEnum DType>
void apply_to_column_numpy(std::shared_ptr<array_info> in_col,
                           std::shared_ptr<array_info> out_col,
                           std::vector<std::shared_ptr<array_info>>& aux_cols,
                           const grouping_info& grp_info) {
    switch (ftype) {
        case Bodo_FTypes::mean: {
            std::shared_ptr<array_info> count_col = aux_cols[0];
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = get_group_for_row(grp_info, i);
                if (i_grp != -1) {
                    mean_agg<T, DType>::apply(getv<double>(out_col, i_grp),
                                              getv<T>(in_col, i),
                                              getv<uint64_t>(count_col, i_grp));
                    // Mean always has a nullable output even
                    // if there is a numpy input.
                    out_col->set_null_bit(i_grp, true);
                    count_col->set_null_bit(i_grp, true);
                }
            }
            break;
        }
        case Bodo_FTypes::mean_eval: {
            for (size_t i = 0; i < in_col->length; i++) {
                mean_eval(getv<double>(out_col, i), getv<uint64_t>(in_col, i));
            }
            break;
        }
        case Bodo_FTypes::var_pop:
        case Bodo_FTypes::std_pop:
        case Bodo_FTypes::var:
        case Bodo_FTypes::std: {
            std::shared_ptr<array_info> count_col = aux_cols[0];
            std::shared_ptr<array_info> mean_col = aux_cols[1];
            std::shared_ptr<array_info> m2_col = aux_cols[2];
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = get_group_for_row(grp_info, i);
                if (i_grp != -1) {
                    var_agg<T, DType>::apply(getv<T>(in_col, i),
                                             getv<uint64_t>(count_col, i_grp),
                                             getv<double>(mean_col, i_grp),
                                             getv<double>(m2_col, i_grp));
                    // Var always has a nullable output even
                    // if there is a numpy input
                    out_col->set_null_bit(i_grp, true);
                    count_col->set_null_bit(i_grp, true);
                    mean_col->set_null_bit(i_grp, true);
                    m2_col->set_null_bit(i_grp, true);
                }
            }
            break;
        }
        case Bodo_FTypes::skew: {
            std::shared_ptr<array_info> count_col = aux_cols[0];
            std::shared_ptr<array_info> m1_col = aux_cols[1];
            std::shared_ptr<array_info> m2_col = aux_cols[2];
            std::shared_ptr<array_info> m3_col = aux_cols[3];
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = get_group_for_row(grp_info, i);
                if (i_grp != -1) {
                    skew_agg<T, DType>::apply(getv<T>(in_col, i),
                                              getv<uint64_t>(count_col, i_grp),
                                              getv<double>(m1_col, i_grp),
                                              getv<double>(m2_col, i_grp),
                                              getv<double>(m3_col, i_grp));
                    out_col->set_null_bit(i_grp, true);
                    count_col->set_null_bit(i_grp, true);
                    m1_col->set_null_bit(i_grp, true);
                    m2_col->set_null_bit(i_grp, true);
                    m3_col->set_null_bit(i_grp, true);
                }
            }
            break;
        }
        case Bodo_FTypes::kurtosis: {
            std::shared_ptr<array_info> count_col = aux_cols[0];
            std::shared_ptr<array_info> m1_col = aux_cols[1];
            std::shared_ptr<array_info> m2_col = aux_cols[2];
            std::shared_ptr<array_info> m3_col = aux_cols[3];
            std::shared_ptr<array_info> m4_col = aux_cols[4];
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = get_group_for_row(grp_info, i);
                if (i_grp != -1) {
                    kurt_agg<T, DType>::apply(getv<T>(in_col, i),
                                              getv<uint64_t>(count_col, i_grp),
                                              getv<double>(m1_col, i_grp),
                                              getv<double>(m2_col, i_grp),
                                              getv<double>(m3_col, i_grp),
                                              getv<double>(m4_col, i_grp));
                    out_col->set_null_bit(i_grp, true);
                    count_col->set_null_bit(i_grp, true);
                    m1_col->set_null_bit(i_grp, true);
                    m2_col->set_null_bit(i_grp, true);
                    m3_col->set_null_bit(i_grp, true);
                    m4_col->set_null_bit(i_grp, true);
                }
            }
            break;
        }
        case Bodo_FTypes::var_pop_eval: {
            std::shared_ptr<array_info> count_col = aux_cols[0];
            std::shared_ptr<array_info> m2_col = aux_cols[2];
            for (size_t i = 0; i < in_col->length; i++) {
                var_pop_eval(getv<double>(out_col, i),
                             getv<uint64_t>(count_col, i),
                             getv<double>(m2_col, i));
            }
            break;
        }
        case Bodo_FTypes::std_pop_eval: {
            std::shared_ptr<array_info> count_col = aux_cols[0];
            std::shared_ptr<array_info> m2_col = aux_cols[2];
            for (size_t i = 0; i < in_col->length; i++) {
                std_pop_eval(getv<double>(out_col, i),
                             getv<uint64_t>(count_col, i),
                             getv<double>(m2_col, i));
            }
            break;
        }
        case Bodo_FTypes::var_eval: {
            std::shared_ptr<array_info> count_col = aux_cols[0];
            std::shared_ptr<array_info> m2_col = aux_cols[2];
            for (size_t i = 0; i < in_col->length; i++) {
                var_eval(getv<double>(out_col, i), getv<uint64_t>(count_col, i),
                         getv<double>(m2_col, i));
            }
            break;
        }
        case Bodo_FTypes::std_eval: {
            std::shared_ptr<array_info> count_col = aux_cols[0];
            std::shared_ptr<array_info> m2_col = aux_cols[2];
            for (size_t i = 0; i < in_col->length; i++) {
                std_eval(getv<double>(out_col, i), getv<uint64_t>(count_col, i),
                         getv<double>(m2_col, i));
            }
            break;
        }
        case Bodo_FTypes::count: {
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = get_group_for_row(grp_info, i);
                if (i_grp != -1) {
                    count_agg<T, DType>::apply(getv<int64_t>(out_col, i_grp),
                                               getv<T>(in_col, i));
                }
            }
            break;
        }
        case Bodo_FTypes::first: {
            int64_t n_bytes = ((out_col->length + 7) >> 3);
            bodo::vector<uint8_t> bitmask_vec(n_bytes, 0);
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = get_group_for_row(grp_info, i);
                T val = getv<T>(in_col, i);
                if ((i_grp != -1) && !GetBit(bitmask_vec.data(), i_grp) &&
                    !isnan_alltype<T, DType>(val)) {
                    getv<T>(out_col, i_grp) = val;
                    SetBitTo(bitmask_vec.data(), i_grp, true);
                }
            }
            break;
        }
        case Bodo_FTypes::idxmax: {
            std::shared_ptr<array_info> index_pos = aux_cols[0];
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = get_group_for_row(grp_info, i);
                if (i_grp != -1) {
                    idxmax_agg<T, DType>::apply(
                        getv<T>(out_col, i_grp), getv<T>(in_col, i),
                        getv<uint64_t>(index_pos, i_grp), i);
                }
            }
            break;
        }
        case Bodo_FTypes::idxmin: {
            std::shared_ptr<array_info> index_pos = aux_cols[0];
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = get_group_for_row(grp_info, i);
                if (i_grp != -1) {
                    idxmin_agg<T, DType>::apply(
                        getv<T>(out_col, i_grp), getv<T>(in_col, i),
                        getv<uint64_t>(index_pos, i_grp), i);
                }
            }
            break;
        }
        case Bodo_FTypes::idxmax_na_first: {
            // Datetime64 and Timedelta64 represent NA values in the array.
            // We need to handle these the same as the nullable case. For
            // all other NA like values (e.g. floats) the relative value of
            // NaN should be based upon wherever they would be sorted. This
            // may need to be handled to match SQL, but is a separate issue.
            std::shared_ptr<array_info> index_pos = aux_cols[0];
            if (DType == Bodo_CTypes::DATETIME ||
                DType == Bodo_CTypes::TIMEDELTA) {
                for (size_t i = 0; i < in_col->length; i++) {
                    int64_t i_grp = get_group_for_row(grp_info, i);
                    // If we are putting NA values first then we stop
                    // visiting a group once we see a NA value. We
                    // initialize the output values to all be non-NA values
                    // to ensure this.
                    if (i_grp != -1 && index_pos->get_null_bit(i_grp)) {
                        // If we see NA/NaN mark this as the match.
                        T input_val = getv<T>(in_col, i);
                        if (!isnan_alltype<T, DType>(input_val)) {
                            idxmax_agg<T, DType>::apply(
                                getv<T>(out_col, i_grp), input_val,
                                getv<uint64_t>(index_pos, i_grp), i);
                        } else {
                            // If we have an NA value mark this group as
                            // done and update the index
                            getv<uint64_t>(index_pos, i_grp) = i;
                            // set the null bit for the count so we know
                            // to stop visiting the group. The data is still
                            // valid.
                            index_pos->set_null_bit(i_grp, false);
                        }
                    }
                }
            } else {
                for (size_t i = 0; i < in_col->length; i++) {
                    int64_t i_grp = get_group_for_row(grp_info, i);
                    if (i_grp != -1) {
                        idxmax_agg<T, DType>::apply(
                            getv<T>(out_col, i_grp), getv<T>(in_col, i),
                            getv<uint64_t>(index_pos, i_grp), i);
                    }
                }
            }
            break;
        }
        case Bodo_FTypes::idxmin_na_first: {
            // Datetime64 and Timedelta64 represent NA values in the array.
            // We need to handle these the same as the nullable case. For
            // all other NA like values (e.g. floats) the relative value of
            // NaN should be based upon wherever they would be sorted. This
            // may need to be handled to match SQL, but is a separate issue.
            std::shared_ptr<array_info> index_pos = aux_cols[0];
            if (DType == Bodo_CTypes::DATETIME ||
                DType == Bodo_CTypes::TIMEDELTA) {
                for (size_t i = 0; i < in_col->length; i++) {
                    int64_t i_grp = get_group_for_row(grp_info, i);
                    // If we are putting NA values first then we stop
                    // visiting a group once we see a NA value. We
                    // initialize the output values to all be non-NA values
                    // to ensure this.
                    if (i_grp != -1 && index_pos->get_null_bit(i_grp)) {
                        // If we see NA/NaN mark this as the match.
                        T input_val = getv<T>(in_col, i);
                        if (!isnan_alltype<T, DType>(input_val)) {
                            idxmin_agg<T, DType>::apply(
                                getv<T>(out_col, i_grp), input_val,
                                getv<uint64_t>(index_pos, i_grp), i);
                        } else {
                            // If we have an NA value mark this group as
                            // done and update the index
                            getv<uint64_t>(index_pos, i_grp) = i;
                            // set the null bit for the count so we know
                            // to stop visiting the group. The data is still
                            // valid.
                            index_pos->set_null_bit(i_grp, false);
                        }
                    }
                }
            } else {
                for (size_t i = 0; i < in_col->length; i++) {
                    int64_t i_grp = get_group_for_row(grp_info, i);
                    if (i_grp != -1) {
                        idxmin_agg<T, DType>::apply(
                            getv<T>(out_col, i_grp), getv<T>(in_col, i),
                            getv<uint64_t>(index_pos, i_grp), i);
                    }
                }
            }
            break;
        }
        case Bodo_FTypes::boolor_agg:
        case Bodo_FTypes::booland_agg: {
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = get_group_for_row(grp_info, i);
                if (i_grp != -1) {
                    T val2 = getv<T>(in_col, i);
                    // Skip NA values
                    if (!isnan_alltype<T, DType>(val2)) {
                        bool_aggfunc<T, DType, ftype>::apply(out_col, i_grp,
                                                             val2);
                        out_col->set_null_bit(i_grp, true);
                    }
                }
            }
            break;
        }
        case Bodo_FTypes::boolxor_agg: {
            std::shared_ptr<array_info> one_col = out_col;
            std::shared_ptr<array_info> two_col = aux_cols[0];
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = get_group_for_row(grp_info, i);
                if (i_grp != -1) {
                    T val = getv<T>(in_col, i);
                    boolxor_agg<T, DType>::apply(val, one_col, two_col, i_grp);
                    one_col->set_null_bit(i_grp, true);
                    two_col->set_null_bit(i_grp, true);
                }
            }
            break;
        }
        case Bodo_FTypes::bitor_agg:
        case Bodo_FTypes::bitand_agg:
        case Bodo_FTypes::bitxor_agg: {
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = get_group_for_row(grp_info, i);
                if (i_grp != -1) {
                    T val2 = getv<T>(in_col, i);
                    // Skip NA values
                    // We treat NaN as NA in this case, since it is a runtime
                    // error in snowflake anyway.
                    if (!isnan_alltype<T, DType>(val2)) {
                        // If we have an integer
                        if (std::is_integral<T>::value) {
                            // output type = input type
                            T& val1 = getv<T>(out_col, i_grp);
                            casted_aggfunc<T, T, DType, ftype>::apply(val1,
                                                                      val2);
                        } else {
                            // otherwise, always use int64_t
                            int64_t& val1 = getv<int64_t>(out_col, i_grp);
                            casted_aggfunc<int64_t, T, DType, ftype>::apply(
                                val1, val2);
                        }
                        out_col->set_null_bit(i_grp, true);
                    }
                }
            }
            break;
        }
        case Bodo_FTypes::count_if: {
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = get_group_for_row(grp_info, i);
                if (i_grp != -1) {
                    bool_sum(getv<int64_t>(out_col, i_grp),
                             getv<bool>(in_col, i));
                }
            }
            break;
        }
        default: {
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = get_group_for_row(grp_info, i);
                if (i_grp != -1) {
                    aggfunc<T, DType, ftype>::apply(getv<T>(out_col, i_grp),
                                                    getv<T>(in_col, i));
                }
            }
        }
    }
}

/**
 * @brief Apply the given aggregation function to an input nullable
 * array.
 *
 * @tparam T The actual C++ type for the underlying array data.
 * @tparam ftype The function type.
 * @tparam dtype The Bodo dtype for the array.
 * @param[in] in_col The input column. This must be a nullable array type.
 * @param[in, out] out_col The output column.
 * @param[in, out] aux_cols Auxillary columns used to for certain operations
 * that require multiple columns (e.g. mean).
 * @param[in] grp_info The grouping information.
 */
template <typename T, int ftype, Bodo_CTypes::CTypeEnum DType>
void apply_to_column_nullable(
    std::shared_ptr<array_info> in_col, std::shared_ptr<array_info> out_col,
    std::vector<std::shared_ptr<array_info>>& aux_cols,
    const grouping_info& grp_info) {
// macros to reduce code duplication

// Find the group number. Eval doesn't care about group number.
#ifndef APPLY_TO_COLUMN_FIND_GROUP
#define APPLY_TO_COLUMN_FIND_GROUP                  \
    switch (ftype) {                                \
        case Bodo_FTypes::mean_eval:                \
        case Bodo_FTypes::var_pop_eval:             \
        case Bodo_FTypes::std_pop_eval:             \
        case Bodo_FTypes::var_eval:                 \
        case Bodo_FTypes::std_eval:                 \
        case Bodo_FTypes::skew_eval:                \
        case Bodo_FTypes::kurt_eval:                \
        case Bodo_FTypes::boolxor_eval:             \
            break;                                  \
        default:                                    \
            i_grp = get_group_for_row(grp_info, i); \
    }
#endif

// Find the valid group condition. The eval functions check if the sample
// is large enough. First checks if we have already seen a value.
// idx***_na_first check if we have already found NA.
#ifndef APPLY_TO_COLUMN_VALID_GROUP
#define APPLY_TO_COLUMN_VALID_GROUP                                         \
    switch (ftype) {                                                        \
        case Bodo_FTypes::mean_eval:                                        \
            valid_group =                                                   \
                in_col->get_null_bit(i) && getv<uint64_t>(in_col, i) > 0;   \
            break;                                                          \
        case Bodo_FTypes::boolxor_eval:                                     \
            valid_group = aux_cols[0]->get_null_bit(i);                     \
            break;                                                          \
        case Bodo_FTypes::var_pop_eval:                                     \
        case Bodo_FTypes::std_pop_eval:                                     \
            valid_group = aux_cols[0]->get_null_bit(i) &&                   \
                          getv<uint64_t>(aux_cols[0], i) > 0;               \
            break;                                                          \
        case Bodo_FTypes::var_eval:                                         \
        case Bodo_FTypes::std_eval:                                         \
            valid_group = aux_cols[0]->get_null_bit(i) &&                   \
                          getv<uint64_t>(aux_cols[0], i) > 1;               \
            break;                                                          \
        case Bodo_FTypes::skew_eval:                                        \
            valid_group = aux_cols[0]->get_null_bit(i) &&                   \
                          getv<uint64_t>(aux_cols[0], i) > 2;               \
            break;                                                          \
        case Bodo_FTypes::kurt_eval:                                        \
            valid_group = aux_cols[0]->get_null_bit(i) &&                   \
                          getv<uint64_t>(aux_cols[0], i) > 3;               \
            break;                                                          \
        case Bodo_FTypes::first:                                            \
            valid_group = (i_grp != -1) && !out_col->get_null_bit(i_grp) && \
                          in_col->get_null_bit(i);                          \
            break;                                                          \
        case Bodo_FTypes::idxmax_na_first:                                  \
        case Bodo_FTypes::idxmin_na_first:                                  \
            valid_group = i_grp != -1 && aux_cols[0]->get_null_bit(i_grp);  \
            break;                                                          \
        default:                                                            \
            valid_group = (i_grp != -1) && in_col->get_null_bit(i);         \
    }
#endif

// Generate the NA special path check. This is only ever
// true for idxmax_na_first and idxmax_na_first
#ifndef APPLY_TO_COLUMN_IS_NA_SPECIAL_CASE
#define APPLY_TO_COLUMN_IS_NA_SPECIAL_CASE    \
    switch (ftype) {                          \
        case Bodo_FTypes::idxmax_na_first:    \
        case Bodo_FTypes::idxmin_na_first:    \
            is_na = !in_col->get_null_bit(i); \
            break;                            \
        default:                              \
            is_na = false;                    \
    }
#endif

// Generate the NA special path. idx***_na_first
// initialize the group with this value.
#ifndef APPLY_TO_COLUMN_NA_SPECIAL_CASE
#define APPLY_TO_COLUMN_NA_SPECIAL_CASE              \
    switch (ftype) {                                 \
        case Bodo_FTypes::idxmax_na_first:           \
        case Bodo_FTypes::idxmin_na_first:           \
            getv<uint64_t>(aux_cols[0], i_grp) = i;  \
            aux_cols[0]->set_null_bit(i_grp, false); \
            break;                                   \
        default:;                                    \
    }
#endif

// Generate the regular codegen path for each function type.
// This controls the aggregation function run on a valid group.
// These just call the dedicated kernels and set values to be
// non-null.
#ifndef APPLY_TO_COLUMN_REGULAR_CASE
#define APPLY_TO_COLUMN_REGULAR_CASE                                           \
    switch (ftype) {                                                           \
        case Bodo_FTypes::count:                                               \
            if (DType == Bodo_CTypes::_BOOL) {                                 \
                bool data_bit = GetBit((uint8_t*)in_col->data1(), i);          \
                count_agg<bool, DType>::apply(getv<int64_t>(out_col, i_grp),   \
                                              data_bit);                       \
            } else {                                                           \
                count_agg<T, DType>::apply(getv<int64_t>(out_col, i_grp),      \
                                           getv<T>(in_col, i));                \
            }                                                                  \
            break;                                                             \
        case Bodo_FTypes::mean:                                                \
            mean_agg<T, DType>::apply(getv<double>(out_col, i_grp),            \
                                      getv<T>(in_col, i),                      \
                                      getv<uint64_t>(aux_cols[0], i_grp));     \
            out_col->set_null_bit(i_grp, true);                                \
            aux_cols[0]->set_null_bit(i_grp, true);                            \
            break;                                                             \
        case Bodo_FTypes::var_pop:                                             \
        case Bodo_FTypes::std_pop:                                             \
        case Bodo_FTypes::var:                                                 \
        case Bodo_FTypes::std:                                                 \
            var_agg<T, DType>::apply(getv<T>(in_col, i),                       \
                                     getv<uint64_t>(aux_cols[0], i_grp),       \
                                     getv<double>(aux_cols[1], i_grp),         \
                                     getv<double>(aux_cols[2], i_grp));        \
            out_col->set_null_bit(i_grp, true);                                \
            aux_cols[0]->set_null_bit(i_grp, true);                            \
            aux_cols[1]->set_null_bit(i_grp, true);                            \
            aux_cols[2]->set_null_bit(i_grp, true);                            \
            break;                                                             \
        case Bodo_FTypes::skew:                                                \
            skew_agg<T, DType>::apply(getv<T>(in_col, i),                      \
                                      getv<uint64_t>(aux_cols[0], i_grp),      \
                                      getv<double>(aux_cols[1], i_grp),        \
                                      getv<double>(aux_cols[2], i_grp),        \
                                      getv<double>(aux_cols[3], i_grp));       \
            out_col->set_null_bit(i_grp, true);                                \
            aux_cols[0]->set_null_bit(i_grp, true);                            \
            aux_cols[1]->set_null_bit(i_grp, true);                            \
            aux_cols[2]->set_null_bit(i_grp, true);                            \
            aux_cols[3]->set_null_bit(i_grp, true);                            \
            break;                                                             \
        case Bodo_FTypes::kurtosis:                                            \
            kurt_agg<T, DType>::apply(getv<T>(in_col, i),                      \
                                      getv<uint64_t>(aux_cols[0], i_grp),      \
                                      getv<double>(aux_cols[1], i_grp),        \
                                      getv<double>(aux_cols[2], i_grp),        \
                                      getv<double>(aux_cols[3], i_grp),        \
                                      getv<double>(aux_cols[4], i_grp));       \
            out_col->set_null_bit(i_grp, true);                                \
            aux_cols[0]->set_null_bit(i_grp, true);                            \
            aux_cols[1]->set_null_bit(i_grp, true);                            \
            aux_cols[2]->set_null_bit(i_grp, true);                            \
            aux_cols[3]->set_null_bit(i_grp, true);                            \
            aux_cols[4]->set_null_bit(i_grp, true);                            \
            break;                                                             \
        case Bodo_FTypes::boolxor_eval:                                        \
            boolxor_eval(out_col, aux_cols[0], i);                             \
            out_col->set_null_bit(i, true);                                    \
            break;                                                             \
        case Bodo_FTypes::mean_eval:                                           \
            mean_eval(getv<double>(out_col, i), getv<uint64_t>(in_col, i));    \
            out_col->set_null_bit(i, true);                                    \
            break;                                                             \
        case Bodo_FTypes::var_pop_eval:                                        \
            var_pop_eval(getv<double>(out_col, i),                             \
                         getv<uint64_t>(aux_cols[0], i),                       \
                         getv<double>(aux_cols[2], i));                        \
            out_col->set_null_bit(i, true);                                    \
            break;                                                             \
        case Bodo_FTypes::std_pop_eval:                                        \
            std_pop_eval(getv<double>(out_col, i),                             \
                         getv<uint64_t>(aux_cols[0], i),                       \
                         getv<double>(aux_cols[2], i));                        \
            out_col->set_null_bit(i, true);                                    \
            break;                                                             \
        case Bodo_FTypes::var_eval:                                            \
            var_eval(getv<double>(out_col, i), getv<uint64_t>(aux_cols[0], i), \
                     getv<double>(aux_cols[2], i));                            \
            out_col->set_null_bit(i, true);                                    \
            break;                                                             \
        case Bodo_FTypes::std_eval:                                            \
            std_eval(getv<double>(out_col, i), getv<uint64_t>(aux_cols[0], i), \
                     getv<double>(aux_cols[2], i));                            \
            out_col->set_null_bit(i, true);                                    \
            break;                                                             \
        case Bodo_FTypes::skew_eval:                                           \
            skew_eval(                                                         \
                getv<double>(out_col, i), getv<uint64_t>(aux_cols[0], i),      \
                getv<double>(aux_cols[1], i), getv<double>(aux_cols[2], i),    \
                getv<double>(aux_cols[3], i));                                 \
            out_col->set_null_bit(i, true);                                    \
            break;                                                             \
        case Bodo_FTypes::kurt_eval:                                           \
            kurt_eval(                                                         \
                getv<double>(out_col, i), getv<uint64_t>(aux_cols[0], i),      \
                getv<double>(aux_cols[1], i), getv<double>(aux_cols[2], i),    \
                getv<double>(aux_cols[3], i), getv<double>(aux_cols[4], i));   \
            out_col->set_null_bit(i, true);                                    \
            break;                                                             \
        case Bodo_FTypes::first:                                               \
            if (DType == Bodo_CTypes::_BOOL) {                                 \
                bool data_bit = GetBit((uint8_t*)in_col->data1(), i);          \
                SetBitTo((uint8_t*)out_col->data1(), i_grp, data_bit);         \
            } else {                                                           \
                getv<T>(out_col, i_grp) = getv<T>(in_col, i);                  \
            }                                                                  \
            out_col->set_null_bit(i_grp, true);                                \
            break;                                                             \
        case Bodo_FTypes::idxmax:                                              \
        case Bodo_FTypes::idxmax_na_first:                                     \
            if (DType == Bodo_CTypes::_BOOL) {                                 \
                bool data_bit = GetBit((uint8_t*)in_col->data1(), i);          \
                idxmax_bool(out_col, i_grp, data_bit,                          \
                            getv<uint64_t>(aux_cols[0], i_grp), i);            \
            } else {                                                           \
                idxmax_agg<T, DType>::apply(                                   \
                    getv<T>(out_col, i_grp), getv<T>(in_col, i),               \
                    getv<uint64_t>(aux_cols[0], i_grp), i);                    \
            }                                                                  \
            out_col->set_null_bit(i_grp, true);                                \
            break;                                                             \
        case Bodo_FTypes::idxmin:                                              \
        case Bodo_FTypes::idxmin_na_first:                                     \
            if (DType == Bodo_CTypes::_BOOL) {                                 \
                bool data_bit = GetBit((uint8_t*)in_col->data1(), i);          \
                idxmin_bool(out_col, i_grp, data_bit,                          \
                            getv<uint64_t>(aux_cols[0], i_grp), i);            \
            } else {                                                           \
                idxmin_agg<T, DType>::apply(                                   \
                    getv<T>(out_col, i_grp), getv<T>(in_col, i),               \
                    getv<uint64_t>(aux_cols[0], i_grp), i);                    \
            }                                                                  \
            out_col->set_null_bit(i_grp, true);                                \
            break;                                                             \
        case Bodo_FTypes::boolor_agg:                                          \
        case Bodo_FTypes::booland_agg:                                         \
            if (DType == Bodo_CTypes::_BOOL) {                                 \
                bool data_bit = GetBit((uint8_t*)in_col->data1(), i);          \
                bool_aggfunc<bool, DType, ftype>::apply(out_col, i_grp,        \
                                                        data_bit);             \
            } else {                                                           \
                bool_aggfunc<T, DType, ftype>::apply(out_col, i_grp,           \
                                                     getv<T>(in_col, i));      \
            }                                                                  \
            out_col->set_null_bit(i_grp, true);                                \
            break;                                                             \
        case Bodo_FTypes::boolxor_agg:                                         \
            if (DType == Bodo_CTypes::_BOOL) {                                 \
                bool data_bit = GetBit((uint8_t*)in_col->data1(), i);          \
                boolxor_agg<bool, DType>::apply(data_bit, out_col,             \
                                                aux_cols[0], i_grp);           \
            } else {                                                           \
                T val = getv<T>(in_col, i);                                    \
                boolxor_agg<T, DType>::apply(val, out_col, aux_cols[0],        \
                                             i_grp);                           \
            }                                                                  \
            out_col->set_null_bit(i_grp, true);                                \
            aux_cols[0]->set_null_bit(i_grp, true);                            \
            break;                                                             \
        case Bodo_FTypes::bitor_agg:                                           \
        case Bodo_FTypes::bitand_agg:                                          \
        case Bodo_FTypes::bitxor_agg: {                                        \
            T val2 = getv<T>(in_col, i);                                       \
            if (!isnan_alltype<T, DType>(val2)) {                              \
                if (std::is_integral<T>::value) {                              \
                    T& val1 = getv<T>(out_col, i_grp);                         \
                    casted_aggfunc<T, T, DType, ftype>::apply(val1, val2);     \
                } else {                                                       \
                    int64_t& val1 = getv<int64_t>(out_col, i_grp);             \
                    casted_aggfunc<int64_t, T, DType, ftype>::apply(val1,      \
                                                                    val2);     \
                }                                                              \
                out_col->set_null_bit(i_grp, true);                            \
            }                                                                  \
            break;                                                             \
        }                                                                      \
        case Bodo_FTypes::count_if: {                                          \
            bool data_bit = GetBit((uint8_t*)in_col->data1(), i);              \
            bool_sum(getv<int64_t>(out_col, i_grp), data_bit);                 \
            break;                                                             \
        }                                                                      \
        case Bodo_FTypes::sum:                                                 \
            if (DType == Bodo_CTypes::_BOOL) {                                 \
                bool data_bit = GetBit((uint8_t*)in_col->data1(), i);          \
                bool_sum(getv<int64_t>(out_col, i_grp), data_bit);             \
                out_col->set_null_bit(i_grp, true);                            \
                break;                                                         \
            }                                                                  \
        case Bodo_FTypes::min:                                                 \
            if (DType == Bodo_CTypes::_BOOL) {                                 \
                bool data_bit = GetBit((uint8_t*)in_col->data1(), i);          \
                bool_aggfunc<bool, DType, ftype>::apply(out_col, i_grp,        \
                                                        data_bit);             \
                out_col->set_null_bit(i_grp, true);                            \
                break;                                                         \
            }                                                                  \
        default:                                                               \
            if (DType == Bodo_CTypes::_BOOL) {                                 \
                bool data_bit = GetBit((uint8_t*)in_col->data1(), i);          \
                bool_aggfunc<bool, DType, ftype>::apply(out_col, i_grp,        \
                                                        data_bit);             \
                out_col->set_null_bit(i_grp, true);                            \
            } else {                                                           \
                aggfunc<T, DType, ftype>::apply(getv<T>(out_col, i_grp),       \
                                                getv<T>(in_col, i));           \
                out_col->set_null_bit(i_grp, true);                            \
            }                                                                  \
    }

#endif

    for (size_t i = 0; i < in_col->length; i++) {
        int64_t i_grp = -1;
        APPLY_TO_COLUMN_FIND_GROUP
        bool valid_group = true;
        APPLY_TO_COLUMN_VALID_GROUP
        if (valid_group) {
            bool is_na = false;
            APPLY_TO_COLUMN_IS_NA_SPECIAL_CASE
            if (is_na) {
                APPLY_TO_COLUMN_NA_SPECIAL_CASE
            } else {
                APPLY_TO_COLUMN_REGULAR_CASE
            }
        }
    }
#undef APPLY_TO_COLUMN_FIND_GROUP
#undef APPLY_TO_COLUMN_VALID_GROUP
#undef APPLY_TO_COLUMN_IS_NA_SPECIAL_CASE
#undef APPLY_TO_COLUMN_NA_SPECIAL_CASE
#undef APPLY_TO_COLUMN_REGULAR_CASE
}

/**
 * @brief Apply the given aggregation function to an input array item array.
 *
 * Currently only supported for first, which is used to implement ANY_VALUE.
 *
 * @tparam ftype The function type.
 * @param[in] in_col The input column. This must be a nullable array type.
 * @param[in, out] out_col The output column.
 * @param[in, out] aux_cols Auxiliary columns used to for certain operations
 * that require multiple columns (e.g. mean).
 * @param[in] grp_info The grouping information.
 */
template <int ftype>
    requires(ftype == Bodo_FTypes::first)
void apply_to_column_array_item(
    std::shared_ptr<array_info> in_col, std::shared_ptr<array_info> out_col,
    std::vector<std::shared_ptr<array_info>>& aux_cols,
    const grouping_info& grp_info) {
    offset_t* in_offset_buffer =
        (offset_t*)(in_col->buffers[0]->mutable_data());
    offset_t* out_offset_buffer =
        (offset_t*)(out_col->buffers[0]->mutable_data());
    out_offset_buffer[0] = 0;
    offset_t curr_offset = 0;
    // Build a vector of which rows to copy to build the new inner array.
    std::vector<size_t> rows_to_copy;
    size_t n_groups = grp_info.num_groups;
    for (size_t i_grp = 0; i_grp < n_groups; i_grp++) {
        size_t i = grp_info.group_to_first_row[i_grp];
        if (in_col->get_null_bit(i)) {
            // If the row is non-null, then the group will map to this inner
            // array entry.
            offset_t start_offset = in_offset_buffer[i];
            offset_t end_offset = in_offset_buffer[i + 1];
            curr_offset += end_offset - start_offset;
            for (size_t idx = start_offset; idx < end_offset; idx++) {
                rows_to_copy.push_back(idx);
            }
        } else {
            // Otherwise, the group will map to null.
            out_col->set_null_bit(i_grp, false);
        }
        out_offset_buffer[i_grp + 1] = curr_offset;
    }
    std::shared_ptr<array_info> in_inner_arr = in_col->child_arrays[0];
    std::shared_ptr<array_info> out_inner_arr = out_col->child_arrays[0];
    // Copy over the desired subset of the inner array
    *out_inner_arr = *(select_subset_of_rows(in_inner_arr, rows_to_copy));
}

/**
 * @brief Apply a function to a column(s), save result to (possibly reduced)
 * output column(s) Semantics of this function right now vary depending on
 * function type (ftype).
 *
 * @tparam T The data type of the input column
 * @tparam ftype The function type being called.
 * @tparam dtype The Bodo dtype of the array.
 * @param[in] in_col Column containing input values.
 * @param[in,out] out_col Column containing output values
 * @param[in,out] aux_cols Addition input/output columns used for
 * the intermediate steps of certain operations, such as mean, var, std.
 * @param grp_info The grouping information to determine the row->group
 * mapping.
 */
template <typename T, int ftype, Bodo_CTypes::CTypeEnum DType>
void apply_to_column(const std::shared_ptr<array_info>& in_col,
                     const std::shared_ptr<array_info>& out_col,
                     std::vector<std::shared_ptr<array_info>>& aux_cols,
                     const grouping_info& grp_info) {
    switch (in_col->arr_type) {
        case bodo_array_type::CATEGORICAL:
            return apply_to_column_categorical<T, ftype, DType>(in_col, out_col,
                                                                grp_info);
        case bodo_array_type::NUMPY:
            return apply_to_column_numpy<T, ftype, DType>(in_col, out_col,
                                                          aux_cols, grp_info);
        case bodo_array_type::DICT:
            return apply_to_column_dict<ftype>(in_col, out_col, aux_cols,
                                               grp_info);
        case bodo_array_type::LIST_STRING:
            return apply_to_column_list_string<ftype>(in_col, out_col,
                                                      grp_info);

        // For the STRING we compute the count, sum, max, min, first, last,
        // idxmin, idxmax, idxmin_na_first, idxmax_na_first
        case bodo_array_type::STRING:
            return apply_to_column_string<ftype>(in_col, out_col, aux_cols,
                                                 grp_info);
        case bodo_array_type::NULLABLE_INT_BOOL:
            return apply_to_column_nullable<T, ftype, DType>(
                in_col, out_col, aux_cols, grp_info);
        default:
            throw std::runtime_error("apply_to_column: incorrect array type");
    }
}

void do_apply_to_column(const std::shared_ptr<array_info>& in_col,
                        const std::shared_ptr<array_info>& out_col,
                        std::vector<std::shared_ptr<array_info>>& aux_cols,
                        const grouping_info& grp_info, int ftype) {
    // macro to reduce code duplication
#ifndef APPLY_TO_COLUMN_CALL
#define APPLY_TO_COLUMN_CALL(FTYPE, CTYPE)                                  \
    if (ftype == FTYPE && in_col->dtype == CTYPE) {                         \
        return apply_to_column<typename dtype_to_type<CTYPE>::type, FTYPE,  \
                               CTYPE>(in_col, out_col, aux_cols, grp_info); \
    }
#endif
    // Handle nested array cases separately
    if (in_col->arr_type == bodo_array_type::ARRAY_ITEM ||
        in_col->arr_type == bodo_array_type::LIST_STRING) {
        switch (ftype) {
            case Bodo_FTypes::first: {
                return apply_to_column_array_item<Bodo_FTypes::first>(
                    in_col, out_col, aux_cols, grp_info);
            }
        }
    }
    // Handle string functions where we care about the input separately.
    if (in_col->arr_type == bodo_array_type::STRING ||
        in_col->arr_type == bodo_array_type::LIST_STRING ||
        in_col->arr_type == bodo_array_type::DICT) {
        switch (ftype) {
            // NOTE: The int template argument is not used in this call to
            // apply_to_column
            case Bodo_FTypes::sum:
                return apply_to_column<int, Bodo_FTypes::sum,
                                       Bodo_CTypes::STRING>(in_col, out_col,
                                                            aux_cols, grp_info);
            case Bodo_FTypes::min:
                return apply_to_column<int, Bodo_FTypes::min,
                                       Bodo_CTypes::STRING>(in_col, out_col,
                                                            aux_cols, grp_info);
            case Bodo_FTypes::max:
                return apply_to_column<int, Bodo_FTypes::max,
                                       Bodo_CTypes::STRING>(in_col, out_col,
                                                            aux_cols, grp_info);
            case Bodo_FTypes::first:
                return apply_to_column<int, Bodo_FTypes::first,
                                       Bodo_CTypes::STRING>(in_col, out_col,
                                                            aux_cols, grp_info);
            case Bodo_FTypes::last:
                return apply_to_column<int, Bodo_FTypes::last,
                                       Bodo_CTypes::STRING>(in_col, out_col,
                                                            aux_cols, grp_info);
            case Bodo_FTypes::idxmin:
                // idxmin handles the na_last case
                return apply_to_column<int, Bodo_FTypes::idxmin,
                                       Bodo_CTypes::STRING>(in_col, out_col,
                                                            aux_cols, grp_info);
            case Bodo_FTypes::idxmax:
                // idxmax handles the na_last case
                return apply_to_column<int, Bodo_FTypes::idxmax,
                                       Bodo_CTypes::STRING>(in_col, out_col,
                                                            aux_cols, grp_info);
            case Bodo_FTypes::idxmin_na_first:
                return apply_to_column<int, Bodo_FTypes::idxmin_na_first,
                                       Bodo_CTypes::STRING>(in_col, out_col,
                                                            aux_cols, grp_info);
            case Bodo_FTypes::idxmax_na_first:
                return apply_to_column<int, Bodo_FTypes::idxmax_na_first,
                                       Bodo_CTypes::STRING>(in_col, out_col,
                                                            aux_cols, grp_info);
            case Bodo_FTypes::bitor_agg:
                return apply_to_column<int, Bodo_FTypes::bitor_agg,
                                       Bodo_CTypes::STRING>(in_col, out_col,
                                                            aux_cols, grp_info);

            case Bodo_FTypes::bitand_agg:
                return apply_to_column<int, Bodo_FTypes::bitand_agg,
                                       Bodo_CTypes::STRING>(in_col, out_col,
                                                            aux_cols, grp_info);

            case Bodo_FTypes::bitxor_agg:
                return apply_to_column<int, Bodo_FTypes::bitxor_agg,
                                       Bodo_CTypes::STRING>(in_col, out_col,
                                                            aux_cols, grp_info);
        }
    }
    // All other types.
    // NOTE: only min/max/first/last are supported for time.
    // Other types are restricted in Python.
    switch (ftype) {
        case Bodo_FTypes::size:
            // SIZE
            // size operation is the same regardless of type of data.
            // Hence, just compute number of rows per group here.
            // TODO: Move to a helper function to simplify this code?
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = grp_info.row_to_group[i];
                if (i_grp != -1) {
                    size_agg<int64_t, Bodo_CTypes::INT64>::apply(
                        getv<int64_t>(out_col, i_grp),
                        getv<int64_t>(in_col, i));
                }
            }
            return;
        case Bodo_FTypes::count:
            // Count
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::count, Bodo_CTypes::FLOAT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::count, Bodo_CTypes::FLOAT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::count, Bodo_CTypes::DATETIME)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::count, Bodo_CTypes::TIMEDELTA)
            // If APPLY_TO_COLUMN_CALL doesn't match then we hit this case.
            // data will be ignored in this case, so type doesn't matter
            return apply_to_column<int8_t, Bodo_FTypes::count,
                                   Bodo_CTypes::INT8>(in_col, out_col, aux_cols,
                                                      grp_info);
        case Bodo_FTypes::sum:
            // SUM
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::sum, Bodo_CTypes::_BOOL)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::sum, Bodo_CTypes::INT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::sum, Bodo_CTypes::UINT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::sum, Bodo_CTypes::INT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::sum, Bodo_CTypes::UINT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::sum, Bodo_CTypes::INT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::sum, Bodo_CTypes::UINT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::sum, Bodo_CTypes::INT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::sum, Bodo_CTypes::UINT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::sum, Bodo_CTypes::DATE)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::sum, Bodo_CTypes::TIME)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::sum, Bodo_CTypes::DATETIME)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::sum, Bodo_CTypes::TIMEDELTA)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::sum, Bodo_CTypes::FLOAT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::sum, Bodo_CTypes::FLOAT64)
            break;
        case Bodo_FTypes::min:
            // MIN
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::min, Bodo_CTypes::_BOOL)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::min, Bodo_CTypes::INT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::min, Bodo_CTypes::UINT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::min, Bodo_CTypes::INT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::min, Bodo_CTypes::UINT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::min, Bodo_CTypes::INT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::min, Bodo_CTypes::UINT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::min, Bodo_CTypes::INT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::min, Bodo_CTypes::UINT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::min, Bodo_CTypes::DATE)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::min, Bodo_CTypes::TIME)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::min, Bodo_CTypes::DATETIME)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::min, Bodo_CTypes::TIMEDELTA)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::min, Bodo_CTypes::FLOAT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::min, Bodo_CTypes::FLOAT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::min, Bodo_CTypes::DECIMAL)
            break;
        case Bodo_FTypes::max:
            // MAX
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::max, Bodo_CTypes::_BOOL)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::max, Bodo_CTypes::INT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::max, Bodo_CTypes::UINT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::max, Bodo_CTypes::INT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::max, Bodo_CTypes::UINT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::max, Bodo_CTypes::INT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::max, Bodo_CTypes::UINT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::max, Bodo_CTypes::INT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::max, Bodo_CTypes::UINT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::max, Bodo_CTypes::DATE)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::max, Bodo_CTypes::TIME)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::max, Bodo_CTypes::DATETIME)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::max, Bodo_CTypes::TIMEDELTA)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::max, Bodo_CTypes::FLOAT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::max, Bodo_CTypes::FLOAT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::max, Bodo_CTypes::DECIMAL)
            break;
        case Bodo_FTypes::prod:
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::prod, Bodo_CTypes::_BOOL)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::prod, Bodo_CTypes::INT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::prod, Bodo_CTypes::UINT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::prod, Bodo_CTypes::INT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::prod, Bodo_CTypes::UINT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::prod, Bodo_CTypes::INT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::prod, Bodo_CTypes::UINT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::prod, Bodo_CTypes::INT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::prod, Bodo_CTypes::UINT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::prod, Bodo_CTypes::DATE)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::prod, Bodo_CTypes::TIME)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::prod, Bodo_CTypes::DATETIME)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::prod, Bodo_CTypes::TIMEDELTA)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::prod, Bodo_CTypes::FLOAT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::prod, Bodo_CTypes::FLOAT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::prod, Bodo_CTypes::DECIMAL)
            break;
        case Bodo_FTypes::first:
            // FIRST
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::first, Bodo_CTypes::_BOOL)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::first, Bodo_CTypes::INT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::first, Bodo_CTypes::UINT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::first, Bodo_CTypes::INT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::first, Bodo_CTypes::UINT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::first, Bodo_CTypes::INT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::first, Bodo_CTypes::UINT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::first, Bodo_CTypes::INT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::first, Bodo_CTypes::UINT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::first, Bodo_CTypes::DATE)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::first, Bodo_CTypes::TIME)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::first, Bodo_CTypes::DATETIME)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::first, Bodo_CTypes::TIMEDELTA)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::first, Bodo_CTypes::FLOAT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::first, Bodo_CTypes::FLOAT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::first, Bodo_CTypes::DECIMAL)
            break;
        case Bodo_FTypes::last:
            // LAST
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::last, Bodo_CTypes::_BOOL)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::last, Bodo_CTypes::INT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::last, Bodo_CTypes::UINT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::last, Bodo_CTypes::INT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::last, Bodo_CTypes::UINT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::last, Bodo_CTypes::INT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::last, Bodo_CTypes::UINT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::last, Bodo_CTypes::INT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::last, Bodo_CTypes::UINT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::last, Bodo_CTypes::DATE)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::last, Bodo_CTypes::TIME)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::last, Bodo_CTypes::DATETIME)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::last, Bodo_CTypes::TIMEDELTA)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::last, Bodo_CTypes::FLOAT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::last, Bodo_CTypes::FLOAT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::last, Bodo_CTypes::DECIMAL)
            break;
        case Bodo_FTypes::mean:
            // MEAN
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::mean, Bodo_CTypes::INT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::mean, Bodo_CTypes::UINT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::mean, Bodo_CTypes::INT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::mean, Bodo_CTypes::UINT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::mean, Bodo_CTypes::INT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::mean, Bodo_CTypes::UINT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::mean, Bodo_CTypes::INT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::mean, Bodo_CTypes::UINT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::mean, Bodo_CTypes::DATE)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::mean, Bodo_CTypes::TIME)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::mean, Bodo_CTypes::DATETIME)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::mean, Bodo_CTypes::TIMEDELTA)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::mean, Bodo_CTypes::FLOAT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::mean, Bodo_CTypes::FLOAT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::mean, Bodo_CTypes::DECIMAL)
            break;
        case Bodo_FTypes::var_pop:
            // VAR
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::var_pop, Bodo_CTypes::INT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::var_pop, Bodo_CTypes::UINT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::var_pop, Bodo_CTypes::INT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::var_pop, Bodo_CTypes::UINT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::var_pop, Bodo_CTypes::INT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::var_pop, Bodo_CTypes::UINT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::var_pop, Bodo_CTypes::INT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::var_pop, Bodo_CTypes::UINT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::var_pop, Bodo_CTypes::DATE)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::var_pop, Bodo_CTypes::TIME)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::var_pop, Bodo_CTypes::DATETIME)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::var_pop, Bodo_CTypes::TIMEDELTA)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::var_pop, Bodo_CTypes::FLOAT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::var_pop, Bodo_CTypes::FLOAT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::var_pop, Bodo_CTypes::DECIMAL)
            break;
        case Bodo_FTypes::std_pop:
            // STDDEV
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::std_pop, Bodo_CTypes::INT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::std_pop, Bodo_CTypes::UINT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::std_pop, Bodo_CTypes::INT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::std_pop, Bodo_CTypes::UINT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::std_pop, Bodo_CTypes::INT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::std_pop, Bodo_CTypes::UINT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::std_pop, Bodo_CTypes::INT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::std_pop, Bodo_CTypes::UINT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::std_pop, Bodo_CTypes::DATE)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::std_pop, Bodo_CTypes::TIME)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::std_pop, Bodo_CTypes::DATETIME)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::std_pop, Bodo_CTypes::TIMEDELTA)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::std_pop, Bodo_CTypes::FLOAT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::std_pop, Bodo_CTypes::FLOAT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::std_pop, Bodo_CTypes::DECIMAL)
            break;
        case Bodo_FTypes::var:
            // VAR
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::var, Bodo_CTypes::INT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::var, Bodo_CTypes::UINT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::var, Bodo_CTypes::INT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::var, Bodo_CTypes::UINT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::var, Bodo_CTypes::INT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::var, Bodo_CTypes::UINT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::var, Bodo_CTypes::INT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::var, Bodo_CTypes::UINT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::var, Bodo_CTypes::DATE)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::var, Bodo_CTypes::TIME)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::var, Bodo_CTypes::DATETIME)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::var, Bodo_CTypes::TIMEDELTA)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::var, Bodo_CTypes::FLOAT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::var, Bodo_CTypes::FLOAT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::var, Bodo_CTypes::DECIMAL)
            break;
        case Bodo_FTypes::std:
            // STDDEV
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::std, Bodo_CTypes::INT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::std, Bodo_CTypes::UINT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::std, Bodo_CTypes::INT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::std, Bodo_CTypes::UINT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::std, Bodo_CTypes::INT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::std, Bodo_CTypes::UINT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::std, Bodo_CTypes::INT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::std, Bodo_CTypes::UINT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::std, Bodo_CTypes::DATE)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::std, Bodo_CTypes::TIME)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::std, Bodo_CTypes::DATETIME)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::std, Bodo_CTypes::TIMEDELTA)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::std, Bodo_CTypes::FLOAT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::std, Bodo_CTypes::FLOAT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::std, Bodo_CTypes::DECIMAL)
            break;
        case Bodo_FTypes::skew:
            // SKEW
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::skew, Bodo_CTypes::INT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::skew, Bodo_CTypes::UINT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::skew, Bodo_CTypes::INT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::skew, Bodo_CTypes::UINT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::skew, Bodo_CTypes::INT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::skew, Bodo_CTypes::UINT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::skew, Bodo_CTypes::INT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::skew, Bodo_CTypes::UINT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::skew, Bodo_CTypes::FLOAT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::skew, Bodo_CTypes::FLOAT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::skew, Bodo_CTypes::DECIMAL)
            break;
        case Bodo_FTypes::kurtosis:
            // KURTOSIS
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::kurtosis, Bodo_CTypes::INT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::kurtosis, Bodo_CTypes::UINT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::kurtosis, Bodo_CTypes::INT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::kurtosis, Bodo_CTypes::UINT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::kurtosis, Bodo_CTypes::INT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::kurtosis, Bodo_CTypes::UINT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::kurtosis, Bodo_CTypes::INT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::kurtosis, Bodo_CTypes::UINT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::kurtosis, Bodo_CTypes::FLOAT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::kurtosis, Bodo_CTypes::FLOAT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::kurtosis, Bodo_CTypes::DECIMAL)
            break;
        case Bodo_FTypes::idxmin:
            // IDXMIN
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmin, Bodo_CTypes::_BOOL)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmin, Bodo_CTypes::INT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmin, Bodo_CTypes::UINT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmin, Bodo_CTypes::INT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmin, Bodo_CTypes::UINT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmin, Bodo_CTypes::INT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmin, Bodo_CTypes::UINT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmin, Bodo_CTypes::INT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmin, Bodo_CTypes::UINT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmin, Bodo_CTypes::DATE)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmin, Bodo_CTypes::TIME)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmin, Bodo_CTypes::DATETIME)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmin, Bodo_CTypes::TIMEDELTA)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmin, Bodo_CTypes::FLOAT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmin, Bodo_CTypes::FLOAT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmin, Bodo_CTypes::DECIMAL)
            break;
        case Bodo_FTypes::idxmax:
            // IDXMAX
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmax, Bodo_CTypes::_BOOL)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmax, Bodo_CTypes::INT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmax, Bodo_CTypes::UINT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmax, Bodo_CTypes::INT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmax, Bodo_CTypes::UINT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmax, Bodo_CTypes::INT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmax, Bodo_CTypes::UINT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmax, Bodo_CTypes::INT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmax, Bodo_CTypes::UINT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmax, Bodo_CTypes::DATE)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmax, Bodo_CTypes::TIME)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmax, Bodo_CTypes::DATETIME)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmax, Bodo_CTypes::TIMEDELTA)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmax, Bodo_CTypes::FLOAT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmax, Bodo_CTypes::FLOAT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmax, Bodo_CTypes::DECIMAL)
            break;
        case Bodo_FTypes::idxmin_na_first:
            // IDXMIN_NA_FIRST
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmin_na_first,
                                 Bodo_CTypes::_BOOL)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmin_na_first,
                                 Bodo_CTypes::INT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmin_na_first,
                                 Bodo_CTypes::UINT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmin_na_first,
                                 Bodo_CTypes::INT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmin_na_first,
                                 Bodo_CTypes::UINT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmin_na_first,
                                 Bodo_CTypes::INT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmin_na_first,
                                 Bodo_CTypes::UINT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmin_na_first,
                                 Bodo_CTypes::INT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmin_na_first,
                                 Bodo_CTypes::UINT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmin_na_first,
                                 Bodo_CTypes::DATE)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmin_na_first,
                                 Bodo_CTypes::TIME)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmin_na_first,
                                 Bodo_CTypes::DATETIME)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmin_na_first,
                                 Bodo_CTypes::TIMEDELTA)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmin_na_first,
                                 Bodo_CTypes::FLOAT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmin_na_first,
                                 Bodo_CTypes::FLOAT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmin_na_first,
                                 Bodo_CTypes::DECIMAL)
            break;
        case Bodo_FTypes::idxmax_na_first:
            // IDXMAX_NA_FIRST
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmax_na_first,
                                 Bodo_CTypes::_BOOL)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmax_na_first,
                                 Bodo_CTypes::INT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmax_na_first,
                                 Bodo_CTypes::UINT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmax_na_first,
                                 Bodo_CTypes::INT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmax_na_first,
                                 Bodo_CTypes::UINT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmax_na_first,
                                 Bodo_CTypes::INT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmax_na_first,
                                 Bodo_CTypes::UINT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmax_na_first,
                                 Bodo_CTypes::INT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmax_na_first,
                                 Bodo_CTypes::UINT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmax_na_first,
                                 Bodo_CTypes::DATE)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmax_na_first,
                                 Bodo_CTypes::TIME)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmax_na_first,
                                 Bodo_CTypes::DATETIME)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmax_na_first,
                                 Bodo_CTypes::TIMEDELTA)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmax_na_first,
                                 Bodo_CTypes::FLOAT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmax_na_first,
                                 Bodo_CTypes::FLOAT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::idxmax_na_first,
                                 Bodo_CTypes::DECIMAL)
            break;
        case Bodo_FTypes::boolor_agg:
            // boolor_agg
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::boolor_agg, Bodo_CTypes::_BOOL)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::boolor_agg, Bodo_CTypes::INT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::boolor_agg, Bodo_CTypes::UINT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::boolor_agg, Bodo_CTypes::INT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::boolor_agg, Bodo_CTypes::UINT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::boolor_agg, Bodo_CTypes::INT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::boolor_agg, Bodo_CTypes::UINT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::boolor_agg, Bodo_CTypes::INT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::boolor_agg, Bodo_CTypes::UINT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::boolor_agg, Bodo_CTypes::FLOAT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::boolor_agg, Bodo_CTypes::FLOAT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::boolor_agg, Bodo_CTypes::DECIMAL)
            break;
        case Bodo_FTypes::booland_agg:
            // BOOLAND_AGG
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::booland_agg, Bodo_CTypes::_BOOL)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::booland_agg, Bodo_CTypes::INT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::booland_agg, Bodo_CTypes::UINT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::booland_agg, Bodo_CTypes::INT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::booland_agg, Bodo_CTypes::UINT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::booland_agg, Bodo_CTypes::INT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::booland_agg, Bodo_CTypes::UINT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::booland_agg, Bodo_CTypes::INT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::booland_agg, Bodo_CTypes::UINT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::booland_agg, Bodo_CTypes::FLOAT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::booland_agg, Bodo_CTypes::FLOAT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::booland_agg, Bodo_CTypes::DECIMAL)
            break;
        case Bodo_FTypes::boolxor_agg:
            // BOOLXOR_AGG
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::boolxor_agg, Bodo_CTypes::_BOOL)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::boolxor_agg, Bodo_CTypes::INT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::boolxor_agg, Bodo_CTypes::UINT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::boolxor_agg, Bodo_CTypes::INT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::boolxor_agg, Bodo_CTypes::UINT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::boolxor_agg, Bodo_CTypes::INT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::boolxor_agg, Bodo_CTypes::UINT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::boolxor_agg, Bodo_CTypes::INT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::boolxor_agg, Bodo_CTypes::UINT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::boolxor_agg, Bodo_CTypes::FLOAT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::boolxor_agg, Bodo_CTypes::FLOAT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::boolxor_agg, Bodo_CTypes::DECIMAL)
            break;
        case Bodo_FTypes::bitor_agg:
            // BITOR_AGG
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::bitor_agg, Bodo_CTypes::INT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::bitor_agg, Bodo_CTypes::UINT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::bitor_agg, Bodo_CTypes::INT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::bitor_agg, Bodo_CTypes::UINT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::bitor_agg, Bodo_CTypes::INT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::bitor_agg, Bodo_CTypes::UINT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::bitor_agg, Bodo_CTypes::INT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::bitor_agg, Bodo_CTypes::UINT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::bitor_agg, Bodo_CTypes::FLOAT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::bitor_agg, Bodo_CTypes::FLOAT64)
            // TODO: add decimal????
            break;
        case Bodo_FTypes::bitand_agg:
            // BITAND_AGG
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::bitand_agg, Bodo_CTypes::INT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::bitand_agg, Bodo_CTypes::UINT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::bitand_agg, Bodo_CTypes::INT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::bitand_agg, Bodo_CTypes::UINT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::bitand_agg, Bodo_CTypes::INT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::bitand_agg, Bodo_CTypes::UINT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::bitand_agg, Bodo_CTypes::INT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::bitand_agg, Bodo_CTypes::UINT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::bitand_agg, Bodo_CTypes::FLOAT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::bitand_agg, Bodo_CTypes::FLOAT64)
            // TODO: add decimal????
            break;
        case Bodo_FTypes::bitxor_agg:
            // BITXOR_AGG
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::bitxor_agg, Bodo_CTypes::INT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::bitxor_agg, Bodo_CTypes::UINT8)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::bitxor_agg, Bodo_CTypes::INT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::bitxor_agg, Bodo_CTypes::UINT16)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::bitxor_agg, Bodo_CTypes::INT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::bitxor_agg, Bodo_CTypes::UINT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::bitxor_agg, Bodo_CTypes::INT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::bitxor_agg, Bodo_CTypes::UINT64)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::bitxor_agg, Bodo_CTypes::FLOAT32)
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::bitxor_agg, Bodo_CTypes::FLOAT64)
            // TODO: add decimal????
            break;
        case Bodo_FTypes::count_if:
            // count_if requires a boolean input.
            APPLY_TO_COLUMN_CALL(Bodo_FTypes::count_if, Bodo_CTypes::_BOOL)
            break;
        case Bodo_FTypes::mean_eval:
            // EVAL step for MEAN. This transforms each group from several
            // arrays to the final single array output. Note we don't care about
            // the input types here as the supported types are always
            // hard coded.
            // TODO: Move elsewhere? Every row is processed instead of reduce
            // to groups.
            return apply_to_column<double, Bodo_FTypes::mean_eval,
                                   Bodo_CTypes::FLOAT64>(in_col, out_col,
                                                         aux_cols, grp_info);

        case Bodo_FTypes::boolxor_eval:
            // EVAL step for BOOLXOR_AGG. This transforms each group from
            // several arrays to the final single array output. Note we don't
            // care about the input types here as the supported types are always
            // hard coded.
            // TODO: Move elsewhere? Every row is processed instead of reduce
            // to groups.
            return apply_to_column<bool, Bodo_FTypes::boolxor_eval,
                                   Bodo_CTypes::_BOOL>(in_col, out_col,
                                                       aux_cols, grp_info);
        case Bodo_FTypes::var_pop_eval:
            // EVAL step for VAR_POP. This transforms each group from several
            // arrays to the final single array output. Note we don't care about
            // the input types here as the supported types are always
            // hard coded.
            // TODO: Move elsewhere? Every row is processed instead of reduce
            // to groups.
            return apply_to_column<double, Bodo_FTypes::var_pop_eval,
                                   Bodo_CTypes::FLOAT64>(in_col, out_col,
                                                         aux_cols, grp_info);
        case Bodo_FTypes::std_pop_eval:
            // EVAL step for STDDEV_POP. This transforms each group from several
            // arrays to the final single array output. Note we don't care about
            // the input types here as the supported types are always
            // hard coded.
            // TODO: Move elsewhere? Every row is processed instead of reduce
            // to groups.
            return apply_to_column<double, Bodo_FTypes::std_pop_eval,
                                   Bodo_CTypes::FLOAT64>(in_col, out_col,
                                                         aux_cols, grp_info);
        case Bodo_FTypes::var_eval:
            // EVAL step for VAR. This transforms each group from several
            // arrays to the final single array output. Note we don't care about
            // the input types here as the supported types are always
            // hard coded.
            // TODO: Move elsewhere? Every row is processed instead of reduce
            // to groups.
            return apply_to_column<double, Bodo_FTypes::var_eval,
                                   Bodo_CTypes::FLOAT64>(in_col, out_col,
                                                         aux_cols, grp_info);
        case Bodo_FTypes::std_eval:
            // EVAL step for STDDEV. This transforms each group from several
            // arrays to the final single array output. Note we don't care about
            // the input types here as the supported types are always
            // hard coded.
            // TODO: Move elsewhere? Every row is processed instead of reduce
            // to groups.
            return apply_to_column<double, Bodo_FTypes::std_eval,
                                   Bodo_CTypes::FLOAT64>(in_col, out_col,
                                                         aux_cols, grp_info);
        case Bodo_FTypes::skew_eval:
            // EVAL step for SKEW. This transforms each group from several
            // arrays to the final single array output. Note we don't care about
            // the input types here as the supported types are always
            // hard coded.
            // TODO: Move elsewhere? Every row is processed instead of reduce
            // to groups.
            return apply_to_column<double, Bodo_FTypes::skew_eval,
                                   Bodo_CTypes::FLOAT64>(in_col, out_col,
                                                         aux_cols, grp_info);
        case Bodo_FTypes::kurt_eval:
            // EVAL step for KURTOSIS. This transforms each group from several
            // arrays to the final single array output. Note we don't care about
            // the input types here as the supported types are always
            // hard coded.
            // TODO: Move elsewhere? Every row is processed instead of reduce
            // to groups.
            return apply_to_column<double, Bodo_FTypes::kurt_eval,
                                   Bodo_CTypes::FLOAT64>(in_col, out_col,
                                                         aux_cols, grp_info);
        default:
            throw std::runtime_error(
                std::string("do_apply_to_column: unsupported function: ") +
                get_name_for_Bodo_FTypes(ftype));
    }
#undef APPLY_TO_COLUMN_CALL
    throw std::runtime_error(
        std::string("do_apply_to_column: unsupported array dtype: ") +
        std::string(GetDtype_as_string(in_col->dtype)));
}

void idx_n_columns_apply(std::shared_ptr<array_info> out_arr,
                         std::vector<std::shared_ptr<array_info>>& orderby_arrs,
                         const std::vector<bool>& asc_vect,
                         const std::vector<bool>& na_pos_vect,
                         grouping_info const& grp_info, int ftype) {
    // Add a sanity check to confirm we never call this in the wrong place.
    if (ftype != Bodo_FTypes::idx_n_columns) {
        throw std::runtime_error(
            "Invalid function type for idx_n_columns_computation");
    }
    // Select the first column to allow iterating over the groups.
    std::shared_ptr<array_info> iter_column = orderby_arrs[0];
    for (size_t i = 0; i < iter_column->length; i++) {
        int64_t i_grp = get_group_for_row(grp_info, i);
        if (i_grp != -1) {
            // Verify the existing value is a valid group member.
            // This is basically a "not null check".
            int64_t curr_grp =
                get_group_for_row(grp_info, getv<int64_t>(out_arr, i_grp));
            if (curr_grp == i_grp) {
                for (size_t j = 0; j < orderby_arrs.size(); j++) {
                    bool found =
                        idx_compare_column(out_arr, i_grp, orderby_arrs[j], i,
                                           asc_vect[j], na_pos_vect[j]);
                    if (found) {
                        break;
                    }
                }
            } else {
                // Initialize the indices
                getv<uint64_t>(out_arr, i_grp) = i;
            }
        }
    }
}
