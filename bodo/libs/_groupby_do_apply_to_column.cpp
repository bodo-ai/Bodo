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

template <typename F, int ftype>
array_info* apply_to_column_list_string(array_info* in_col, array_info* out_col,
                                        const grouping_info& grp_info, F f) {
    size_t num_groups = grp_info.num_groups;
    std::vector<std::vector<std::pair<std::string, bool>>> ListListPair(
        num_groups);
    char* data_i = in_col->data1;
    offset_t* index_offsets_i = (offset_t*)in_col->data3;
    offset_t* data_offsets_i = (offset_t*)in_col->data2;
    uint8_t* sub_null_bitmask_i = (uint8_t*)in_col->sub_null_bitmask;
    // Computing the strings used in output.
    uint64_t n_bytes = (num_groups + 7) >> 3;
    std::vector<uint8_t> Vmask(n_bytes, 0);
    for (size_t i = 0; i < in_col->length; i++) {
        int64_t i_grp = f(i);
        if ((i_grp != -1) && in_col->get_null_bit(i)) {
            bool out_bit_set = out_col->get_null_bit(i_grp);
            if (ftype == Bodo_FTypes::first && out_bit_set) continue;
            offset_t start_offset = index_offsets_i[i];
            offset_t end_offset = index_offsets_i[i + 1];
            offset_t len = end_offset - start_offset;
            std::vector<std::pair<std::string, bool>> LStrB(len);
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
    return create_list_string_array(Vmask, ListListPair);
}

template <typename F, int ftype>
array_info* apply_to_column_string(array_info* in_col, array_info* out_col,
                                   std::vector<array_info*>& aux_cols,
                                   const grouping_info& grp_info, F f) {
    size_t num_groups = grp_info.num_groups;
    size_t n_bytes = (num_groups + 7) >> 3;
    std::vector<uint8_t> V(n_bytes, 0);
    std::vector<std::string> ListString(num_groups);
    char* data_i = in_col->data1;
    offset_t* offsets_i = (offset_t*)in_col->data2;
    switch (ftype) {
        case Bodo_FTypes::idxmax: {
            array_info* index_pos = aux_cols[0];
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = f(i);
                if ((i_grp != -1) && in_col->get_null_bit(i)) {
                    bool out_bit_set = GetBit(V.data(), i_grp);
                    offset_t start_offset = offsets_i[i];
                    offset_t end_offset = offsets_i[i + 1];
                    offset_t len = end_offset - start_offset;
                    std::string val(&data_i[start_offset], len);
                    if (out_bit_set) {
                        idxmax_string(ListString[i_grp], val,
                                      getv<uint64_t>(index_pos, i_grp), i);
                    } else {
                        ListString[i_grp] = val;
                        getv<uint64_t>(index_pos, i_grp) = i;
                        SetBitTo(V.data(), i_grp, true);
                    }
                }
            }
            break;
        }
        case Bodo_FTypes::idxmin: {
            array_info* index_pos = aux_cols[0];
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = f(i);
                if ((i_grp != -1) && in_col->get_null_bit(i)) {
                    bool out_bit_set = GetBit(V.data(), i_grp);
                    offset_t start_offset = offsets_i[i];
                    offset_t end_offset = offsets_i[i + 1];
                    offset_t len = end_offset - start_offset;
                    std::string val(&data_i[start_offset], len);
                    if (out_bit_set) {
                        idxmin_string(ListString[i_grp], val,
                                      getv<uint64_t>(index_pos, i_grp), i);
                    } else {
                        ListString[i_grp] = val;
                        getv<uint64_t>(index_pos, i_grp) = i;
                        SetBitTo(V.data(), i_grp, true);
                    }
                }
            }
            break;
        }
        case Bodo_FTypes::idxmax_na_first: {
            array_info* index_pos = aux_cols[0];
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = f(i);
                // If we are putting NA values first then we stop
                // visiting a group once we see a NA value. We
                // initialize the output values to all be non-NA values
                // to ensure this.
                if (i_grp != -1 && index_pos->get_null_bit(i_grp)) {
                    if (in_col->get_null_bit(i)) {
                        bool out_bit_set = GetBit(V.data(), i_grp);
                        offset_t start_offset = offsets_i[i];
                        offset_t end_offset = offsets_i[i + 1];
                        offset_t len = end_offset - start_offset;
                        std::string val(&data_i[start_offset], len);
                        if (out_bit_set) {
                            idxmax_string(ListString[i_grp], val,
                                          getv<uint64_t>(index_pos, i_grp), i);
                        } else {
                            ListString[i_grp] = val;
                            getv<uint64_t>(index_pos, i_grp) = i;
                            SetBitTo(V.data(), i_grp, true);
                        }
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
            break;
        }
        case Bodo_FTypes::idxmin_na_first: {
            array_info* index_pos = aux_cols[0];
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = f(i);
                // If we are putting NA values first then we stop
                // visiting a group once we see a NA value. We
                // initialize the output values to all be non-NA values
                // to ensure this.
                if (i_grp != -1 && index_pos->get_null_bit(i_grp)) {
                    if (in_col->get_null_bit(i)) {
                        bool out_bit_set = GetBit(V.data(), i_grp);
                        offset_t start_offset = offsets_i[i];
                        offset_t end_offset = offsets_i[i + 1];
                        offset_t len = end_offset - start_offset;
                        std::string val(&data_i[start_offset], len);
                        if (out_bit_set) {
                            idxmin_string(ListString[i_grp], val,
                                          getv<uint64_t>(index_pos, i_grp), i);
                        } else {
                            ListString[i_grp] = val;
                            getv<uint64_t>(index_pos, i_grp) = i;
                            SetBitTo(V.data(), i_grp, true);
                        }
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
            break;
        }
        default:
            // Computing the strings used in output.
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = f(i);
                if ((i_grp != -1) && in_col->get_null_bit(i)) {
                    bool out_bit_set = GetBit(V.data(), i_grp);
                    if (ftype == Bodo_FTypes::first && out_bit_set) continue;
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
    }
    // TODO: Avoid creating the output array for
    // idxmin/idxmax/idxmin_na_first/idxmax_na_first
    // Determining the number of characters in output.
    return create_string_array(V, ListString);
}

template <typename F, int ftype>
array_info* apply_sum_to_column_string(array_info* in_col, array_info* out_col,
                                       const grouping_info& grp_info, F f) {
    // allocate output array (length is number of groups, number of chars same
    // as input)
    size_t num_groups = grp_info.num_groups;
    int64_t n_chars = in_col->n_sub_elems;
    array_info* out_arr = alloc_string_array(num_groups, n_chars, 0);
    size_t n_bytes = (num_groups + 7) >> 3;
    memset(out_arr->null_bitmask, 0xff, n_bytes);  // null not possible

    // find offsets for each output string
    std::vector<offset_t> str_offsets(num_groups + 1, 0);
    char* data_i = in_col->data1;
    offset_t* offsets_i = (offset_t*)in_col->data2;
    char* data_o = out_arr->data1;
    offset_t* offsets_o = (offset_t*)out_arr->data2;

    for (size_t i = 0; i < in_col->length; i++) {
        int64_t i_grp = f(i);
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
        int64_t i_grp = f(i);
        if ((i_grp != -1) && in_col->get_null_bit(i)) {
            offset_t len = offsets_i[i + 1] - offsets_i[i];
            memcpy(&data_o[str_offsets[i_grp]], data_i + offsets_i[i], len);
            str_offsets[i_grp] += len;
        }
    }
    return out_arr;
}

/**
 * @brief Applies max/min/first/last to dictionary encoded string column
 *
 * @tparam F
 * @tparam ftype
 * @param in_col: the input dictionary encoded string column
 * @param out_col: the output dictionary encoded string column
 * @param grp_info: groupby information
 * @param f: a function that returns group index given row index
 * @return array_info*
 */
template <typename F, int ftype>
array_info* apply_to_column_dict(array_info* in_col, array_info* out_col,
                                 std::vector<array_info*>& aux_cols,
                                 const grouping_info& grp_info, F f) {
    size_t num_groups = grp_info.num_groups;
    size_t n_bytes = (num_groups + 7) >> 3;
    array_info* indices_arr =
        alloc_nullable_array(num_groups, Bodo_CTypes::INT32, 0);
    std::vector<uint8_t> V(n_bytes,
                           0);  // bitmask to mark if group's been updated
    char* data_i = in_col->info1->data1;
    offset_t* offsets_i = (offset_t*)in_col->info1->data2;
    switch (ftype) {
        case Bodo_FTypes::idxmax: {
            array_info* index_pos = aux_cols[0];
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = f(i);
                if ((i_grp != -1) && in_col->get_null_bit(i)) {
                    bool out_bit_set = GetBit(V.data(), i_grp);
                    int32_t& dict_ind = getv<int32_t>(in_col->info2, i);
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
                        idxmax_dict(org_ind, dict_ind, s1, s2,
                                    getv<uint64_t>(index_pos, i_grp), i);
                    } else {
                        org_ind = dict_ind;
                        getv<uint64_t>(index_pos, i_grp) = i;
                        SetBitTo(V.data(), i_grp, true);
                        indices_arr->set_null_bit(i_grp, true);
                    }
                }
            }
            break;
        }
        case Bodo_FTypes::idxmin: {
            array_info* index_pos = aux_cols[0];
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = f(i);
                if ((i_grp != -1) && in_col->get_null_bit(i)) {
                    bool out_bit_set = GetBit(V.data(), i_grp);
                    int32_t& dict_ind = getv<int32_t>(in_col->info2, i);
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
                        idxmin_dict(org_ind, dict_ind, s1, s2,
                                    getv<uint64_t>(index_pos, i_grp), i);
                    } else {
                        org_ind = dict_ind;
                        getv<uint64_t>(index_pos, i_grp) = i;
                        SetBitTo(V.data(), i_grp, true);
                        indices_arr->set_null_bit(i_grp, true);
                    }
                }
            }
            break;
        }
        case Bodo_FTypes::idxmax_na_first: {
            array_info* index_pos = aux_cols[0];
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = f(i);
                // If we are putting NA values first then we stop
                // visiting a group once we see a NA value. We
                // initialize the output values to all be non-NA values
                // to ensure this.
                if (i_grp != -1 && index_pos->get_null_bit(i_grp)) {
                    if (in_col->get_null_bit(i)) {
                        bool out_bit_set = GetBit(V.data(), i_grp);
                        int32_t& dict_ind = getv<int32_t>(in_col->info2, i);
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
                            idxmax_dict(org_ind, dict_ind, s1, s2,
                                        getv<uint64_t>(index_pos, i_grp), i);
                        } else {
                            org_ind = dict_ind;
                            getv<uint64_t>(index_pos, i_grp) = i;
                            SetBitTo(V.data(), i_grp, true);
                            indices_arr->set_null_bit(i_grp, true);
                        }
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
            break;
        }
        case Bodo_FTypes::idxmin_na_first: {
            array_info* index_pos = aux_cols[0];
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = f(i);
                // If we are putting NA values first then we stop
                // visiting a group once we see a NA value. We
                // initialize the output values to all be non-NA values
                // to ensure this.
                if (i_grp != -1 && index_pos->get_null_bit(i_grp)) {
                    if (in_col->get_null_bit(i)) {
                        bool out_bit_set = GetBit(V.data(), i_grp);
                        int32_t& dict_ind = getv<int32_t>(in_col->info2, i);
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
                            idxmin_dict(org_ind, dict_ind, s1, s2,
                                        getv<uint64_t>(index_pos, i_grp), i);
                        } else {
                            org_ind = dict_ind;
                            getv<uint64_t>(index_pos, i_grp) = i;
                            SetBitTo(V.data(), i_grp, true);
                            indices_arr->set_null_bit(i_grp, true);
                        }
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
            break;
        }
        case Bodo_FTypes::last: {
            // Define a specialized implementation of last
            // so we avoid allocating for the underlying strings.
            for (size_t i = 0; i < in_col->length; i++) {
                int64_t i_grp = f(i);
                if ((i_grp != -1) && in_col->info2->get_null_bit(i)) {
                    bool out_bit_set = GetBit(V.data(), i_grp);
                    int32_t& dict_ind = getv<int32_t>(in_col->info2, i);
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
                int64_t i_grp = f(i);
                if ((i_grp != -1) && in_col->info2->get_null_bit(i)) {
                    bool out_bit_set = GetBit(V.data(), i_grp);
                    if (ftype == Bodo_FTypes::first && out_bit_set) continue;
                    int32_t& dict_ind = getv<int32_t>(in_col->info2, i);
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
    UNORD_MAP_CONTAINER<int32_t, int32_t>
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
    std::vector<uint8_t> bitmask_vec(n_bytes, 0);
    std::vector<std::string> ListString(n_dict);
    for (auto& it : old_to_new) {
        offset_t start_offset = offsets_i[it.first];
        offset_t end_offset = offsets_i[it.first + 1];
        offset_t len = end_offset - start_offset;
        std::string val(&data_i[start_offset], len);
        ListString[it.second - 1] = val;  // -1 to account for the 1 offset
        SetBitTo(bitmask_vec.data(), it.second - 1, true);
    }
    array_info* dict_arr = create_string_array(bitmask_vec, ListString);
    return create_dict_string_array(dict_arr, indices_arr, num_groups);
}

/**
 * @brief Apply sum operation on a dictionary-encoded string column. The
 * partial_sum trick used for regular string columns to calculate the offsets is
 * also used here. Returns a regulare string array instead of a
 * dictionary-encoded string array.
 *
 * @tparam F
 * @tparam ftype
 * @param in_col: the input dictionary-encoded string array
 * @param out_col: the output string array
 * @param grp_info: groupby information
 * @param f: a function that returns the group index given the row index
 * @return array_info*
 */
template <typename F, int ftype>
array_info* apply_sum_to_column_dict(array_info* in_col, array_info* out_col,
                                     const grouping_info& grp_info, F f) {
    size_t num_groups = grp_info.num_groups;
    int64_t n_chars = 0;
    size_t n_bytes = (num_groups + 7) >> 3;

    // calculate the total number of characters in the dict-encoded array
    // and find offsets for each output string
    // every string has a start and end offset so len(offsets) == (len(data) +
    // 1)
    std::vector<offset_t> str_offsets(num_groups + 1, 0);
    char* data_i = in_col->info1->data1;
    offset_t* offsets_i = (offset_t*)in_col->info1->data2;
    for (size_t i = 0; i < in_col->length; i++) {
        int64_t i_grp = f(i);
        if ((i_grp != -1) && in_col->info2->get_null_bit(i)) {
            int32_t dict_ind = getv<int32_t>(in_col->info2, i);
            offset_t len = offsets_i[dict_ind + 1] - offsets_i[dict_ind];
            str_offsets[i_grp + 1] += len;
            n_chars += len;
        }
    }

    array_info* out_arr = alloc_string_array(num_groups, n_chars, 0);
    memset(out_arr->null_bitmask, 0xff, n_bytes);  // null not possible
    char* data_o = out_arr->data1;
    offset_t* offsets_o = (offset_t*)out_arr->data2;

    std::partial_sum(str_offsets.begin(), str_offsets.end(),
                     str_offsets.begin());
    memcpy(offsets_o, str_offsets.data(), (num_groups + 1) * sizeof(offset_t));

    // copy characters to output
    for (size_t i = 0; i < in_col->length; i++) {
        int64_t i_grp = f(i);
        if ((i_grp != -1) && in_col->info2->get_null_bit(i)) {
            int32_t dict_ind = getv<int32_t>(in_col->info2, i);
            offset_t len = offsets_i[dict_ind + 1] - offsets_i[dict_ind];
            memcpy(&data_o[str_offsets[i_grp]], data_i + offsets_i[dict_ind],
                   len);
            str_offsets[i_grp] += len;
        }
    }
    return out_arr;
}

/**
 * Apply a function to a column(s), save result to (possibly reduced) output
 * column(s) Semantics of this function right now vary depending on function
 * type (ftype).
 *
 * @param column containing input values
 * @param output column
 * @param auxiliary input/output columns used for mean, var, std
 * @param maps row numbers in input columns to group numbers (for reduction
 * operations)
 */
template <typename F, typename T, int ftype, int dtype>
void apply_to_column_F(array_info* in_col, array_info* out_col,
                       std::vector<array_info*>& aux_cols,
                       const grouping_info& grp_info, F f, bool use_sql_rules) {
    switch (in_col->arr_type) {
        case bodo_array_type::CATEGORICAL:
            if (ftype == Bodo_FTypes::count) {
                for (size_t i = 0; i < in_col->length; i++) {
                    int64_t i_grp = f(i);
                    if (i_grp != -1) {
                        T& val = getv<T>(in_col, i);
                        if (!isnan_categorical<T, dtype>(val)) {
                            count_agg<T, dtype>::apply(
                                getv<int64_t>(out_col, i_grp), val);
                        }
                    }
                }
                return;
            } else if (ftype == Bodo_FTypes::min ||
                       ftype == Bodo_FTypes::last) {
                // NOTE: Bodo_FTypes::max is handled for categorical type
                // since NA is -1.
                for (size_t i = 0; i < in_col->length; i++) {
                    int64_t i_grp = f(i);
                    if (i_grp != -1) {
                        T& val = getv<T>(in_col, i);
                        if (!isnan_categorical<T, dtype>(val)) {
                            aggfunc<T, dtype, ftype>::apply(
                                getv<T>(out_col, i_grp), val);
                        }
                    }
                }
                // aggfunc_output_initialize_kernel, min defaults
                // to num_categories if all entries are NA
                // this needs to be replaced with -1
                if (ftype == Bodo_FTypes::min) {
                    for (size_t i = 0; i < out_col->length; i++) {
                        T& val = getv<T>(out_col, i);
                        set_na_if_num_categories<T, dtype>(
                            val, out_col->num_categories);
                    }
                }
                return;
            } else if (ftype == Bodo_FTypes::first) {
                int64_t n_bytes = ((out_col->length + 7) >> 3);
                std::vector<uint8_t> bitmask_vec(n_bytes, 0);
                for (size_t i = 0; i < in_col->length; i++) {
                    int64_t i_grp = f(i);
                    T val = getv<T>(in_col, i);
                    if ((i_grp != -1) && !GetBit(bitmask_vec.data(), i_grp) &&
                        !isnan_categorical<T, dtype>(val)) {
                        getv<T>(out_col, i_grp) = val;
                        SetBitTo(bitmask_vec.data(), i_grp, true);
                    }
                }
            }
        case bodo_array_type::NUMPY:
            if (ftype == Bodo_FTypes::mean) {
                array_info* count_col = aux_cols[0];
                for (size_t i = 0; i < in_col->length; i++) {
                    int64_t i_grp = f(i);
                    if (i_grp != -1) {
                        mean_agg<T, dtype>::apply(
                            getv<double>(out_col, i_grp), getv<T>(in_col, i),
                            getv<uint64_t>(count_col, i_grp));
                        // Mean always has a nullable output even
                        // if there is a numpy input.
                        out_col->set_null_bit(i_grp, true);
                        count_col->set_null_bit(i_grp, true);
                    }
                }
            } else if (ftype == Bodo_FTypes::mean_eval) {
                for (size_t i = 0; i < in_col->length; i++) {
                    mean_eval(getv<double>(out_col, i),
                              getv<uint64_t>(in_col, i));
                }
            } else if (ftype == Bodo_FTypes::var) {
                array_info* count_col = aux_cols[0];
                array_info* mean_col = aux_cols[1];
                array_info* m2_col = aux_cols[2];
                for (size_t i = 0; i < in_col->length; i++) {
                    int64_t i_grp = f(i);
                    if (i_grp != -1) {
                        var_agg<T, dtype>::apply(
                            getv<T>(in_col, i),
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
            } else if (ftype == Bodo_FTypes::var_eval) {
                array_info* count_col = aux_cols[0];
                array_info* m2_col = aux_cols[2];
                for (size_t i = 0; i < in_col->length; i++)
                    var_eval(getv<double>(out_col, i),
                             getv<uint64_t>(count_col, i),
                             getv<double>(m2_col, i));
            } else if (ftype == Bodo_FTypes::std_eval) {
                array_info* count_col = aux_cols[0];
                array_info* m2_col = aux_cols[2];
                for (size_t i = 0; i < in_col->length; i++)
                    std_eval(getv<double>(out_col, i),
                             getv<uint64_t>(count_col, i),
                             getv<double>(m2_col, i));
            } else if (ftype == Bodo_FTypes::count) {
                for (size_t i = 0; i < in_col->length; i++) {
                    int64_t i_grp = f(i);
                    if (i_grp != -1)
                        count_agg<T, dtype>::apply(
                            getv<int64_t>(out_col, i_grp), getv<T>(in_col, i));
                }
            } else if (ftype == Bodo_FTypes::first) {
                // create a temporary bitmask to know if we have set a
                // value for each row/group
                int64_t n_bytes = ((out_col->length + 7) >> 3);
                std::vector<uint8_t> bitmask_vec(n_bytes, 0);
                for (size_t i = 0; i < in_col->length; i++) {
                    int64_t i_grp = f(i);
                    T val = getv<T>(in_col, i);
                    if ((i_grp != -1) && !GetBit(bitmask_vec.data(), i_grp) &&
                        !isnan_alltype<T, dtype>(val)) {
                        getv<T>(out_col, i_grp) = val;
                        SetBitTo(bitmask_vec.data(), i_grp, true);
                    }
                }
            } else if (ftype == Bodo_FTypes::idxmax) {
                array_info* index_pos = aux_cols[0];
                for (size_t i = 0; i < in_col->length; i++) {
                    int64_t i_grp = f(i);
                    if (i_grp != -1) {
                        idxmax_agg<T, dtype>::apply(
                            getv<T>(out_col, i_grp), getv<T>(in_col, i),
                            getv<uint64_t>(index_pos, i_grp), i);
                    }
                }
            } else if (ftype == Bodo_FTypes::idxmin) {
                array_info* index_pos = aux_cols[0];
                for (size_t i = 0; i < in_col->length; i++) {
                    int64_t i_grp = f(i);
                    if (i_grp != -1) {
                        idxmin_agg<T, dtype>::apply(
                            getv<T>(out_col, i_grp), getv<T>(in_col, i),
                            getv<uint64_t>(index_pos, i_grp), i);
                    }
                }
            } else if (ftype == Bodo_FTypes::idxmax_na_first) {
                // Datetime64 and Timedelta64 represent NA values in the array.
                // We need to handle these the same as the nullable case. For
                // all other NA like values (e.g. floats) the relative value of
                // NaN should be based upon wherever they would be sorted. This
                // may need to be handled to match SQL, but is a separate issue.
                array_info* index_pos = aux_cols[0];
                if (dtype == Bodo_CTypes::DATETIME ||
                    dtype == Bodo_CTypes::TIMEDELTA) {
                    for (size_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        // If we are putting NA values first then we stop
                        // visiting a group once we see a NA value. We
                        // initialize the output values to all be non-NA values
                        // to ensure this.
                        if (i_grp != -1 && index_pos->get_null_bit(i_grp)) {
                            // If we see NA/NaN mark this as the match.
                            T input_val = getv<T>(in_col, i);
                            if (!isnan_alltype<T, dtype>(input_val)) {
                                idxmax_agg<T, dtype>::apply(
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
                        int64_t i_grp = f(i);
                        if (i_grp != -1) {
                            idxmax_agg<T, dtype>::apply(
                                getv<T>(out_col, i_grp), getv<T>(in_col, i),
                                getv<uint64_t>(index_pos, i_grp), i);
                        }
                    }
                }
            } else if (ftype == Bodo_FTypes::idxmin_na_first) {
                // Datetime64 and Timedelta64 represent NA values in the array.
                // We need to handle these the same as the nullable case. For
                // all other NA like values (e.g. floats) the relative value of
                // NaN should be based upon wherever they would be sorted. This
                // may need to be handled to match SQL, but is a separate issue.
                array_info* index_pos = aux_cols[0];
                if (dtype == Bodo_CTypes::DATETIME ||
                    dtype == Bodo_CTypes::TIMEDELTA) {
                    for (size_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        // If we are putting NA values first then we stop
                        // visiting a group once we see a NA value. We
                        // initialize the output values to all be non-NA values
                        // to ensure this.
                        if (i_grp != -1 && index_pos->get_null_bit(i_grp)) {
                            // If we see NA/NaN mark this as the match.
                            T input_val = getv<T>(in_col, i);
                            if (!isnan_alltype<T, dtype>(input_val)) {
                                idxmin_agg<T, dtype>::apply(
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
                        int64_t i_grp = f(i);
                        if (i_grp != -1) {
                            idxmin_agg<T, dtype>::apply(
                                getv<T>(out_col, i_grp), getv<T>(in_col, i),
                                getv<uint64_t>(index_pos, i_grp), i);
                        }
                    }
                }
            } else if (ftype == Bodo_FTypes::boolor_agg) {
                for (size_t i = 0; i < in_col->length; i++) {
                    int64_t i_grp = f(i);
                    if ((i_grp != -1)) {
                        T val2 = getv<T>(in_col, i);
                        // Skip NA values
                        if (!isnan_alltype<T, dtype>(val2)) {
                            bool_aggfunc<T, dtype, ftype>::apply(
                                getv<bool>(out_col, i_grp), val2);
                            out_col->set_null_bit(i_grp, true);
                        }
                    }
                }
            } else {
                for (size_t i = 0; i < in_col->length; i++) {
                    int64_t i_grp = f(i);
                    if (i_grp != -1) {
                        aggfunc<T, dtype, ftype>::apply(getv<T>(out_col, i_grp),
                                                        getv<T>(in_col, i));
                    }
                }
            }
            return;
        // For DICT(dictionary-encoded string) we support count
        case bodo_array_type::DICT:
            switch (ftype) {
                case Bodo_FTypes::count: {
                    for (size_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        if (i_grp != -1 && in_col->info2->get_null_bit(i))
                            count_agg<T, dtype>::apply(
                                getv<int64_t>(out_col, i_grp),
                                getv<T>(in_col, i));
                    }
                    return;
                }
                // optimized groupby sum for strings (concatenation)
                case Bodo_FTypes::sum: {
                    array_info* new_out_col =
                        apply_sum_to_column_dict<F, ftype>(in_col, out_col,
                                                           grp_info, f);
                    *out_col = std::move(*new_out_col);
                    delete new_out_col;
                    return;
                }
                default:
                    array_info* new_out_col = apply_to_column_dict<F, ftype>(
                        in_col, out_col, aux_cols, grp_info, f);
                    *out_col = std::move(*new_out_col);
                    delete new_out_col;
                    return;
            }
        // for list strings, we are supporting count, sum, max, min, first, last
        case bodo_array_type::LIST_STRING:
            switch (ftype) {
                case Bodo_FTypes::count: {
                    for (size_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        if ((i_grp != -1) && in_col->get_null_bit(i))
                            count_agg<T, dtype>::apply(
                                getv<int64_t>(out_col, i_grp),
                                getv<T>(in_col, i));
                    }
                    return;
                }
                default:
                    array_info* new_out_col =
                        apply_to_column_list_string<F, ftype>(in_col, out_col,
                                                              grp_info, f);
                    *out_col = std::move(*new_out_col);
                    delete new_out_col;
                    return;
            }

        // For the STRING we compute the count, sum, max, min, first, last,
        // idxmin, idxmax, idxmin_na_first, idxmax_na_first
        case bodo_array_type::STRING:
            switch (ftype) {
                case Bodo_FTypes::count: {
                    for (size_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        if ((i_grp != -1) && in_col->get_null_bit(i))
                            count_agg<T, dtype>::apply(
                                getv<int64_t>(out_col, i_grp),
                                getv<T>(in_col, i));
                    }
                    return;
                }
                // optimized groupby sum for strings (concatenation)
                case Bodo_FTypes::sum: {
                    array_info* new_out_col =
                        apply_sum_to_column_string<F, ftype>(in_col, out_col,
                                                             grp_info, f);
                    *out_col = std::move(*new_out_col);
                    delete new_out_col;
                    return;
                }
                default:
                    array_info* new_out_col = apply_to_column_string<F, ftype>(
                        in_col, out_col, aux_cols, grp_info, f);
                    *out_col = std::move(*new_out_col);
                    delete new_out_col;
                    return;
            }
        case bodo_array_type::NULLABLE_INT_BOOL:
            switch (ftype) {
                case Bodo_FTypes::count: {
                    for (size_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        if ((i_grp != -1) && in_col->get_null_bit(i))
                            count_agg<T, dtype>::apply(
                                getv<int64_t>(out_col, i_grp),
                                getv<T>(in_col, i));
                    }
                    return;
                }
                case Bodo_FTypes::mean: {
                    array_info* count_col = aux_cols[0];
                    for (size_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        if ((i_grp != -1) && in_col->get_null_bit(i)) {
                            mean_agg<T, dtype>::apply(
                                getv<double>(out_col, i_grp),
                                getv<T>(in_col, i),
                                getv<uint64_t>(count_col, i_grp));
                            out_col->set_null_bit(i_grp, true);
                            count_col->set_null_bit(i_grp, true);
                        }
                    }
                    return;
                }
                case Bodo_FTypes::mean_eval: {
                    for (size_t i = 0; i < in_col->length; i++) {
                        if (in_col->get_null_bit(i) &&
                            getv<uint64_t>(in_col, i) > 0) {
                            mean_eval(getv<double>(out_col, i),
                                      getv<uint64_t>(in_col, i));
                            out_col->set_null_bit(i, true);
                        }
                    }
                    return;
                }
                case Bodo_FTypes::var: {
                    array_info* count_col = aux_cols[0];
                    array_info* mean_col = aux_cols[1];
                    array_info* m2_col = aux_cols[2];
                    for (size_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        if ((i_grp != -1) && in_col->get_null_bit(i)) {
                            var_agg<T, dtype>::apply(
                                getv<T>(in_col, i),
                                getv<uint64_t>(count_col, i_grp),
                                getv<double>(mean_col, i_grp),
                                getv<double>(m2_col, i_grp));
                            out_col->set_null_bit(i_grp, true);
                            count_col->set_null_bit(i_grp, true);
                            mean_col->set_null_bit(i_grp, true);
                            m2_col->set_null_bit(i_grp, true);
                        }
                    }
                    return;
                }
                case Bodo_FTypes::var_eval: {
                    array_info* count_col = aux_cols[0];
                    array_info* m2_col = aux_cols[2];
                    for (size_t i = 0; i < in_col->length; i++) {
                        // TODO: Template on use_sql_rules
                        if (count_col->get_null_bit(i) &&
                            getv<uint64_t>(count_col, i) > 1) {
                            var_eval(getv<double>(out_col, i),
                                     getv<uint64_t>(count_col, i),
                                     getv<double>(m2_col, i));
                            out_col->set_null_bit(i, true);
                        }
                    }
                    return;
                }
                case Bodo_FTypes::std_eval: {
                    array_info* count_col = aux_cols[0];
                    array_info* m2_col = aux_cols[2];
                    for (size_t i = 0; i < in_col->length; i++) {
                        // TODO: Template on use_sql_rules
                        if (count_col->get_null_bit(i) &&
                            getv<uint64_t>(count_col, i) > 1) {
                            std_eval(getv<double>(out_col, i),
                                     getv<uint64_t>(count_col, i),
                                     getv<double>(m2_col, i));
                            out_col->set_null_bit(i, true);
                        }
                    }
                    return;
                }
                case Bodo_FTypes::first:
                    for (size_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        if ((i_grp != -1) && !out_col->get_null_bit(i_grp) &&
                            in_col->get_null_bit(i)) {
                            getv<T>(out_col, i_grp) = getv<T>(in_col, i);
                            out_col->set_null_bit(i_grp, true);
                        }
                    }
                    return;
                case Bodo_FTypes::idxmax:
                    for (size_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        if ((i_grp != -1) && in_col->get_null_bit(i)) {
                            idxmax_agg<T, dtype>::apply(
                                getv<T>(out_col, i_grp), getv<T>(in_col, i),
                                getv<uint64_t>(aux_cols[0], i_grp), i);
                            out_col->set_null_bit(i_grp, true);
                        }
                    }
                    return;
                case Bodo_FTypes::idxmax_na_first: {
                    array_info* index_pos = aux_cols[0];
                    for (size_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        // If we are putting NA values first then we stop
                        // visiting a group once we see a NA value. We
                        // initialize the output values to all be non-NA values
                        // to ensure this.
                        if (i_grp != -1 && index_pos->get_null_bit(i_grp)) {
                            if (in_col->get_null_bit(i)) {
                                idxmax_agg<T, dtype>::apply(
                                    getv<T>(out_col, i_grp), getv<T>(in_col, i),
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
                    return;
                }
                case Bodo_FTypes::idxmin:
                    for (size_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        if ((i_grp != -1) && in_col->get_null_bit(i)) {
                            idxmin_agg<T, dtype>::apply(
                                getv<T>(out_col, i_grp), getv<T>(in_col, i),
                                getv<uint64_t>(aux_cols[0], i_grp), i);
                            out_col->set_null_bit(i_grp, true);
                        }
                    }
                    return;
                case Bodo_FTypes::idxmin_na_first: {
                    array_info* index_pos = aux_cols[0];
                    for (size_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        // If we are putting NA values first then we stop
                        // visiting a group once we see a NA value. We
                        // initialize the output values to all be non-NA values
                        // to ensure this.
                        if (i_grp != -1 && index_pos->get_null_bit(i_grp)) {
                            if (in_col->get_null_bit(i)) {
                                idxmin_agg<T, dtype>::apply(
                                    getv<T>(out_col, i_grp), getv<T>(in_col, i),
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
                    return;
                }
                case Bodo_FTypes::boolor_agg:
                    for (size_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        if ((i_grp != -1) && in_col->get_null_bit(i)) {
                            bool_aggfunc<T, dtype, ftype>::apply(
                                getv<bool>(out_col, i_grp), getv<T>(in_col, i));
                            out_col->set_null_bit(i_grp, true);
                        }
                    }
                    return;
                default: {
                    if (ftype == Bodo_FTypes::sum &&
                        dtype == Bodo_CTypes::_BOOL) {
                        for (size_t i = 0; i < in_col->length; i++) {
                            int64_t i_grp = f(i);
                            if ((i_grp != -1) && in_col->get_null_bit(i)) {
                                bool_sum<bool, dtype>::apply(
                                    getv<int64_t>(out_col, i_grp),
                                    getv<bool>(in_col, i));
                            }
                            // The output is never null for count_if, which is
                            // implemented by bool_sum.
                            // TODO: Replace with an explicit count_if function
                            // to avoid NULL issues with SUM(BOOLEAN) column.
                            out_col->set_null_bit(i_grp, true);
                        }
                        return;
                    }
                    for (size_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        if ((i_grp != -1) && in_col->get_null_bit(i)) {
                            aggfunc<T, dtype, ftype>::apply(
                                getv<T>(out_col, i_grp), getv<T>(in_col, i));
                            out_col->set_null_bit(i_grp, true);
                        }
                    }
                    return;
                }
            }
        default:
            Bodo_PyErr_SetString(PyExc_RuntimeError,
                                 "apply_to_column: incorrect array type");
            return;
    }
}

template <typename T, int ftype, int dtype>
void apply_to_column(array_info* in_col, array_info* out_col,
                     std::vector<array_info*>& aux_cols,
                     const grouping_info& grp_info, bool use_sql_rules) {
    auto f = [&](int64_t const& i_row) -> int64_t {
        return grp_info.row_to_group[i_row];
    };
    return apply_to_column_F<decltype(f), T, ftype, dtype>(
        in_col, out_col, aux_cols, grp_info, f, use_sql_rules);
}

/**
 * Invokes the correct template instance of apply_to_column depending on
 * function (ftype) and dtype. See 'apply_to_column'
 *
 * @param column containing input values
 * @param output column
 * @param auxiliary input/output columns used for mean, var, std
 * @param maps row numbers in input columns to group numbers (for reduction
 * operations)
 * @param function to apply
 */
void do_apply_to_column(array_info* in_col, array_info* out_col,
                        std::vector<array_info*>& aux_cols,
                        const grouping_info& grp_info, int ftype,
                        bool use_sql_rules) {
    // size operation is the same regardless of type of data.
    // Hence, just compute number of rows per group here.
    if (ftype == Bodo_FTypes::size) {
        for (size_t i = 0; i < in_col->length; i++) {
            int64_t i_grp = grp_info.row_to_group[i];
            if (i_grp != -1)
                size_agg<int64_t, Bodo_CTypes::INT64>::apply(
                    getv<int64_t>(out_col, i_grp), getv<int64_t>(in_col, i));
        }
        return;
    }
    if (in_col->arr_type == bodo_array_type::STRING ||
        in_col->arr_type == bodo_array_type::LIST_STRING ||
        in_col->arr_type == bodo_array_type::DICT) {
        switch (ftype) {
            // NOTE: The int template argument is not used in this call to
            // apply_to_column
            case Bodo_FTypes::sum:
                return apply_to_column<int, Bodo_FTypes::sum,
                                       Bodo_CTypes::STRING>(
                    in_col, out_col, aux_cols, grp_info, use_sql_rules);
            case Bodo_FTypes::min:
                return apply_to_column<int, Bodo_FTypes::min,
                                       Bodo_CTypes::STRING>(
                    in_col, out_col, aux_cols, grp_info, use_sql_rules);
            case Bodo_FTypes::max:
                return apply_to_column<int, Bodo_FTypes::max,
                                       Bodo_CTypes::STRING>(
                    in_col, out_col, aux_cols, grp_info, use_sql_rules);
            case Bodo_FTypes::first:
                return apply_to_column<int, Bodo_FTypes::first,
                                       Bodo_CTypes::STRING>(
                    in_col, out_col, aux_cols, grp_info, use_sql_rules);
            case Bodo_FTypes::last:
                return apply_to_column<int, Bodo_FTypes::last,
                                       Bodo_CTypes::STRING>(
                    in_col, out_col, aux_cols, grp_info, use_sql_rules);
            case Bodo_FTypes::idxmin:
                // idxmin handles the na_last case
                return apply_to_column<int, Bodo_FTypes::idxmin,
                                       Bodo_CTypes::STRING>(
                    in_col, out_col, aux_cols, grp_info, use_sql_rules);
            case Bodo_FTypes::idxmax:
                // idxmax handles the na_last case
                return apply_to_column<int, Bodo_FTypes::idxmax,
                                       Bodo_CTypes::STRING>(
                    in_col, out_col, aux_cols, grp_info, use_sql_rules);
            case Bodo_FTypes::idxmin_na_first:
                return apply_to_column<int, Bodo_FTypes::idxmin_na_first,
                                       Bodo_CTypes::STRING>(
                    in_col, out_col, aux_cols, grp_info, use_sql_rules);
            case Bodo_FTypes::idxmax_na_first:
                return apply_to_column<int, Bodo_FTypes::idxmax_na_first,
                                       Bodo_CTypes::STRING>(
                    in_col, out_col, aux_cols, grp_info, use_sql_rules);
        }
    }
    if (ftype == Bodo_FTypes::count) {
        switch (in_col->dtype) {
            case Bodo_CTypes::FLOAT32:
                // data will only be used to check for nans
                return apply_to_column<float, Bodo_FTypes::count,
                                       Bodo_CTypes::FLOAT32>(
                    in_col, out_col, aux_cols, grp_info, use_sql_rules);
            case Bodo_CTypes::FLOAT64:
                // data will only be used to check for nans
                return apply_to_column<double, Bodo_FTypes::count,
                                       Bodo_CTypes::FLOAT64>(
                    in_col, out_col, aux_cols, grp_info, use_sql_rules);
            case Bodo_CTypes::DATETIME:
            case Bodo_CTypes::TIMEDELTA:
                // data will only be used to check for NATs
                return apply_to_column<int64_t, Bodo_FTypes::count,
                                       Bodo_CTypes::DATETIME>(
                    in_col, out_col, aux_cols, grp_info, use_sql_rules);
            default:
                // data will be ignored in this case, so type doesn't matter
                return apply_to_column<int8_t, Bodo_FTypes::count,
                                       Bodo_CTypes::INT8>(
                    in_col, out_col, aux_cols, grp_info, use_sql_rules);
        }
    }

    switch (in_col->dtype) {
        case Bodo_CTypes::_BOOL:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<bool, Bodo_FTypes::sum,
                                           Bodo_CTypes::_BOOL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::min:
                    return apply_to_column<bool, Bodo_FTypes::min,
                                           Bodo_CTypes::_BOOL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::max:
                    return apply_to_column<bool, Bodo_FTypes::max,
                                           Bodo_CTypes::_BOOL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::prod:
                    return apply_to_column<bool, Bodo_FTypes::prod,
                                           Bodo_CTypes::_BOOL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::first:
                    return apply_to_column<bool, Bodo_FTypes::first,
                                           Bodo_CTypes::_BOOL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::last:
                    return apply_to_column<bool, Bodo_FTypes::last,
                                           Bodo_CTypes::_BOOL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<bool, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::_BOOL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<bool, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::_BOOL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin_na_first:
                    return apply_to_column<bool, Bodo_FTypes::idxmin_na_first,
                                           Bodo_CTypes::_BOOL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax_na_first:
                    return apply_to_column<bool, Bodo_FTypes::idxmax_na_first,
                                           Bodo_CTypes::_BOOL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::boolor_agg:
                    return apply_to_column<bool, Bodo_FTypes::boolor_agg,
                                           Bodo_CTypes::_BOOL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                default:
                    Bodo_PyErr_SetString(
                        PyExc_RuntimeError,
                        "unsupported aggregation for boolean type column");
                    return;
            }
        case Bodo_CTypes::INT8:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<int8_t, Bodo_FTypes::sum,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::min:
                    return apply_to_column<int8_t, Bodo_FTypes::min,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::max:
                    return apply_to_column<int8_t, Bodo_FTypes::max,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::prod:
                    return apply_to_column<int8_t, Bodo_FTypes::prod,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::first:
                    return apply_to_column<int8_t, Bodo_FTypes::first,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::last:
                    return apply_to_column<int8_t, Bodo_FTypes::last,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::mean:
                    return apply_to_column<int8_t, Bodo_FTypes::mean,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<int8_t, Bodo_FTypes::var,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<int8_t, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<int8_t, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin_na_first:
                    return apply_to_column<int8_t, Bodo_FTypes::idxmin_na_first,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax_na_first:
                    return apply_to_column<int8_t, Bodo_FTypes::idxmax_na_first,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::boolor_agg:
                    return apply_to_column<int8_t, Bodo_FTypes::boolor_agg,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
            }
        case Bodo_CTypes::UINT8:
            switch (ftype) {
                case Bodo_FTypes::sum: {
                    return apply_to_column<uint8_t, Bodo_FTypes::sum,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                }
                case Bodo_FTypes::min:
                    return apply_to_column<uint8_t, Bodo_FTypes::min,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::max:
                    return apply_to_column<uint8_t, Bodo_FTypes::max,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::prod:
                    return apply_to_column<uint8_t, Bodo_FTypes::prod,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::first:
                    return apply_to_column<uint8_t, Bodo_FTypes::first,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::last:
                    return apply_to_column<uint8_t, Bodo_FTypes::last,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::mean:
                    return apply_to_column<uint8_t, Bodo_FTypes::mean,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<uint8_t, Bodo_FTypes::var,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<uint8_t, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<uint8_t, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin_na_first:
                    return apply_to_column<uint8_t,
                                           Bodo_FTypes::idxmin_na_first,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax_na_first:
                    return apply_to_column<uint8_t,
                                           Bodo_FTypes::idxmax_na_first,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::boolor_agg:
                    return apply_to_column<uint8_t, Bodo_FTypes::boolor_agg,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
            }
        case Bodo_CTypes::INT16:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<int16_t, Bodo_FTypes::sum,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::min:
                    return apply_to_column<int16_t, Bodo_FTypes::min,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::max:
                    return apply_to_column<int16_t, Bodo_FTypes::max,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::prod:
                    return apply_to_column<int16_t, Bodo_FTypes::prod,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::first:
                    return apply_to_column<int16_t, Bodo_FTypes::first,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::last:
                    return apply_to_column<int16_t, Bodo_FTypes::last,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::mean:
                    return apply_to_column<int16_t, Bodo_FTypes::mean,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<int16_t, Bodo_FTypes::var,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<int16_t, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<int16_t, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin_na_first:
                    return apply_to_column<int16_t,
                                           Bodo_FTypes::idxmin_na_first,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax_na_first:
                    return apply_to_column<int16_t,
                                           Bodo_FTypes::idxmax_na_first,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::boolor_agg:
                    return apply_to_column<int16_t, Bodo_FTypes::boolor_agg,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
            }
        case Bodo_CTypes::UINT16:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<uint16_t, Bodo_FTypes::sum,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::min:
                    return apply_to_column<uint16_t, Bodo_FTypes::min,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::max:
                    return apply_to_column<uint16_t, Bodo_FTypes::max,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::prod:
                    return apply_to_column<uint16_t, Bodo_FTypes::prod,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::first:
                    return apply_to_column<uint16_t, Bodo_FTypes::first,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::last:
                    return apply_to_column<uint16_t, Bodo_FTypes::last,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::mean:
                    return apply_to_column<uint16_t, Bodo_FTypes::mean,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<uint16_t, Bodo_FTypes::var,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<uint16_t, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<uint16_t, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin_na_first:
                    return apply_to_column<uint16_t,
                                           Bodo_FTypes::idxmin_na_first,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax_na_first:
                    return apply_to_column<uint16_t,
                                           Bodo_FTypes::idxmax_na_first,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::boolor_agg:
                    return apply_to_column<uint16_t, Bodo_FTypes::boolor_agg,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
            }
        case Bodo_CTypes::INT32:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<int32_t, Bodo_FTypes::sum,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::min:
                    return apply_to_column<int32_t, Bodo_FTypes::min,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::max:
                    return apply_to_column<int32_t, Bodo_FTypes::max,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::prod:
                    return apply_to_column<int32_t, Bodo_FTypes::prod,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::first:
                    return apply_to_column<int32_t, Bodo_FTypes::first,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::last:
                    return apply_to_column<int32_t, Bodo_FTypes::last,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::mean:
                    return apply_to_column<int32_t, Bodo_FTypes::mean,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<int32_t, Bodo_FTypes::var,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<int32_t, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<int32_t, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin_na_first:
                    return apply_to_column<int32_t,
                                           Bodo_FTypes::idxmin_na_first,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax_na_first:
                    return apply_to_column<int32_t,
                                           Bodo_FTypes::idxmax_na_first,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::boolor_agg:
                    return apply_to_column<int32_t, Bodo_FTypes::boolor_agg,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
            }
        case Bodo_CTypes::UINT32:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<uint32_t, Bodo_FTypes::sum,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::min:
                    return apply_to_column<uint32_t, Bodo_FTypes::min,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::max:
                    return apply_to_column<uint32_t, Bodo_FTypes::max,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::prod:
                    return apply_to_column<uint32_t, Bodo_FTypes::prod,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::first:
                    return apply_to_column<uint32_t, Bodo_FTypes::first,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::last:
                    return apply_to_column<uint32_t, Bodo_FTypes::last,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::mean:
                    return apply_to_column<uint32_t, Bodo_FTypes::mean,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<uint32_t, Bodo_FTypes::var,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<uint32_t, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<uint32_t, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin_na_first:
                    return apply_to_column<uint32_t,
                                           Bodo_FTypes::idxmin_na_first,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax_na_first:
                    return apply_to_column<uint32_t,
                                           Bodo_FTypes::idxmax_na_first,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::boolor_agg:
                    return apply_to_column<uint32_t, Bodo_FTypes::boolor_agg,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
            }
        case Bodo_CTypes::INT64:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<int64_t, Bodo_FTypes::sum,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::min:
                    return apply_to_column<int64_t, Bodo_FTypes::min,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::max:
                    return apply_to_column<int64_t, Bodo_FTypes::max,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::prod:
                    return apply_to_column<int64_t, Bodo_FTypes::prod,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::first:
                    return apply_to_column<int64_t, Bodo_FTypes::first,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::last:
                    return apply_to_column<int64_t, Bodo_FTypes::last,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::mean:
                    return apply_to_column<int64_t, Bodo_FTypes::mean,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<int64_t, Bodo_FTypes::var,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<int64_t, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<int64_t, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin_na_first:
                    return apply_to_column<int64_t,
                                           Bodo_FTypes::idxmin_na_first,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax_na_first:
                    return apply_to_column<int64_t,
                                           Bodo_FTypes::idxmax_na_first,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::boolor_agg:
                    return apply_to_column<int64_t, Bodo_FTypes::boolor_agg,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
            }
        case Bodo_CTypes::UINT64:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<uint64_t, Bodo_FTypes::sum,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::min:
                    return apply_to_column<uint64_t, Bodo_FTypes::min,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::max:
                    return apply_to_column<uint64_t, Bodo_FTypes::max,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::prod:
                    return apply_to_column<uint64_t, Bodo_FTypes::prod,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::first:
                    return apply_to_column<uint64_t, Bodo_FTypes::first,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::last:
                    return apply_to_column<uint64_t, Bodo_FTypes::last,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::mean:
                    return apply_to_column<uint64_t, Bodo_FTypes::mean,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<uint64_t, Bodo_FTypes::var,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<uint64_t, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<uint64_t, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin_na_first:
                    return apply_to_column<uint64_t,
                                           Bodo_FTypes::idxmin_na_first,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax_na_first:
                    return apply_to_column<uint64_t,
                                           Bodo_FTypes::idxmax_na_first,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::boolor_agg:
                    return apply_to_column<uint64_t, Bodo_FTypes::boolor_agg,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
            }
        case Bodo_CTypes::DATE:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<int64_t, Bodo_FTypes::sum,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::min:
                    return apply_to_column<int64_t, Bodo_FTypes::min,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::max:
                    return apply_to_column<int64_t, Bodo_FTypes::max,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::prod:
                    return apply_to_column<int64_t, Bodo_FTypes::prod,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::first:
                    return apply_to_column<int64_t, Bodo_FTypes::first,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::last:
                    return apply_to_column<int64_t, Bodo_FTypes::last,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::mean:
                    return apply_to_column<int64_t, Bodo_FTypes::mean,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<int64_t, Bodo_FTypes::var,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<int64_t, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<int64_t, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin_na_first:
                    return apply_to_column<int64_t,
                                           Bodo_FTypes::idxmin_na_first,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax_na_first:
                    return apply_to_column<int64_t,
                                           Bodo_FTypes::idxmax_na_first,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
            }
        // TODO: [BE-4106] Split Time into Time32 and Time64
        case Bodo_CTypes::TIME:
            // NOTE: only min/max/first/last are supported.
            // Others, the column will be dropped on Python side.
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<int64_t, Bodo_FTypes::sum,
                                           Bodo_CTypes::TIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::min:
                    return apply_to_column<int64_t, Bodo_FTypes::min,
                                           Bodo_CTypes::TIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::max:
                    return apply_to_column<int64_t, Bodo_FTypes::max,
                                           Bodo_CTypes::TIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::prod:
                    return apply_to_column<int64_t, Bodo_FTypes::prod,
                                           Bodo_CTypes::TIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::first:
                    return apply_to_column<int64_t, Bodo_FTypes::first,
                                           Bodo_CTypes::TIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::last:
                    return apply_to_column<int64_t, Bodo_FTypes::last,
                                           Bodo_CTypes::TIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::mean:
                    return apply_to_column<int64_t, Bodo_FTypes::mean,
                                           Bodo_CTypes::TIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<int64_t, Bodo_FTypes::var,
                                           Bodo_CTypes::TIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<int64_t, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::TIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<int64_t, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::TIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin_na_first:
                    return apply_to_column<int64_t,
                                           Bodo_FTypes::idxmin_na_first,
                                           Bodo_CTypes::TIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax_na_first:
                    return apply_to_column<int64_t,
                                           Bodo_FTypes::idxmax_na_first,
                                           Bodo_CTypes::TIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
            }
        case Bodo_CTypes::DATETIME:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<int64_t, Bodo_FTypes::sum,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::min:
                    return apply_to_column<int64_t, Bodo_FTypes::min,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::max:
                    return apply_to_column<int64_t, Bodo_FTypes::max,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::prod:
                    return apply_to_column<int64_t, Bodo_FTypes::prod,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::first:
                    return apply_to_column<int64_t, Bodo_FTypes::first,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::last:
                    return apply_to_column<int64_t, Bodo_FTypes::last,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::mean:
                    return apply_to_column<int64_t, Bodo_FTypes::mean,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<int64_t, Bodo_FTypes::var,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<int64_t, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<int64_t, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin_na_first:
                    return apply_to_column<int64_t,
                                           Bodo_FTypes::idxmin_na_first,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax_na_first:
                    return apply_to_column<int64_t,
                                           Bodo_FTypes::idxmax_na_first,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
            }
        case Bodo_CTypes::TIMEDELTA:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<int64_t, Bodo_FTypes::sum,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::min:
                    return apply_to_column<int64_t, Bodo_FTypes::min,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::max:
                    return apply_to_column<int64_t, Bodo_FTypes::max,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::prod:
                    return apply_to_column<int64_t, Bodo_FTypes::prod,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::first:
                    return apply_to_column<int64_t, Bodo_FTypes::first,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::last:
                    return apply_to_column<int64_t, Bodo_FTypes::last,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::mean:
                    return apply_to_column<int64_t, Bodo_FTypes::mean,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<int64_t, Bodo_FTypes::var,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<int64_t, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<int64_t, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin_na_first:
                    return apply_to_column<int64_t,
                                           Bodo_FTypes::idxmin_na_first,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax_na_first:
                    return apply_to_column<int64_t,
                                           Bodo_FTypes::idxmax_na_first,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
            }
        case Bodo_CTypes::FLOAT32:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<float, Bodo_FTypes::sum,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::min:
                    return apply_to_column<float, Bodo_FTypes::min,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::max:
                    return apply_to_column<float, Bodo_FTypes::max,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::prod:
                    return apply_to_column<float, Bodo_FTypes::prod,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::first:
                    return apply_to_column<float, Bodo_FTypes::first,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::last:
                    return apply_to_column<float, Bodo_FTypes::last,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::mean:
                    return apply_to_column<float, Bodo_FTypes::mean,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::mean_eval:
                    return apply_to_column<float, Bodo_FTypes::mean_eval,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<float, Bodo_FTypes::var,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::var_eval:
                    return apply_to_column<float, Bodo_FTypes::var_eval,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::std_eval:
                    return apply_to_column<float, Bodo_FTypes::std_eval,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<float, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<float, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin_na_first:
                    return apply_to_column<float, Bodo_FTypes::idxmin_na_first,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax_na_first:
                    return apply_to_column<float, Bodo_FTypes::idxmax_na_first,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::boolor_agg:
                    return apply_to_column<float, Bodo_FTypes::boolor_agg,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
            }
        case Bodo_CTypes::FLOAT64:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<double, Bodo_FTypes::sum,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::min:
                    return apply_to_column<double, Bodo_FTypes::min,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::max:
                    return apply_to_column<double, Bodo_FTypes::max,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::prod:
                    return apply_to_column<double, Bodo_FTypes::prod,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::first:
                    return apply_to_column<double, Bodo_FTypes::first,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::last:
                    return apply_to_column<double, Bodo_FTypes::last,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::mean:
                    return apply_to_column<double, Bodo_FTypes::mean,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::mean_eval:
                    return apply_to_column<double, Bodo_FTypes::mean_eval,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<double, Bodo_FTypes::var,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::var_eval:
                    return apply_to_column<double, Bodo_FTypes::var_eval,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::std_eval:
                    return apply_to_column<double, Bodo_FTypes::std_eval,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<double, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<double, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin_na_first:
                    return apply_to_column<double, Bodo_FTypes::idxmin_na_first,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax_na_first:
                    return apply_to_column<double, Bodo_FTypes::idxmax_na_first,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::boolor_agg:
                    return apply_to_column<double, Bodo_FTypes::boolor_agg,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
            }
        case Bodo_CTypes::DECIMAL:
            switch (ftype) {
                case Bodo_FTypes::first:
                    return apply_to_column<decimal_value_cpp,
                                           Bodo_FTypes::first,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::last:
                    return apply_to_column<decimal_value_cpp, Bodo_FTypes::last,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::min:
                    return apply_to_column<decimal_value_cpp, Bodo_FTypes::min,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::max:
                    return apply_to_column<decimal_value_cpp, Bodo_FTypes::max,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::mean:
                    return apply_to_column<decimal_value_cpp, Bodo_FTypes::mean,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::mean_eval:
                    return apply_to_column<decimal_value_cpp,
                                           Bodo_FTypes::mean_eval,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<decimal_value_cpp, Bodo_FTypes::var,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::var_eval:
                    return apply_to_column<decimal_value_cpp,
                                           Bodo_FTypes::var_eval,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::std_eval:
                    return apply_to_column<decimal_value_cpp,
                                           Bodo_FTypes::std_eval,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<decimal_value_cpp,
                                           Bodo_FTypes::idxmin,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<decimal_value_cpp,
                                           Bodo_FTypes::idxmax,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmin_na_first:
                    return apply_to_column<decimal_value_cpp,
                                           Bodo_FTypes::idxmin_na_first,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::idxmax_na_first:
                    return apply_to_column<decimal_value_cpp,
                                           Bodo_FTypes::idxmax_na_first,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
                case Bodo_FTypes::boolor_agg:
                    return apply_to_column<decimal_value_cpp,
                                           Bodo_FTypes::boolor_agg,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info, use_sql_rules);
            }
        default:
            Bodo_PyErr_SetString(
                PyExc_RuntimeError,
                (std::string("do_apply_to_column: unsuported array dtype: ") +
                 std::string(GetDtype_as_string(in_col->dtype)))
                    .c_str());
            return;
    }
}
