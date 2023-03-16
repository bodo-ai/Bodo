// Copyright (C) 2023 Bodo Inc. All rights reserved.

#include "_array_utils.h"
#include "_groupby.h"
#include "_shuffle.h"

/**
 * This files contains the functions used for the eval step of
 * hashed based groupby that are too complex to be inlined.
 */

/**
 * @brief Copy nullable value from tmp_col to all the rows in the
 * corresponding group update_col.
 *
 * @param update_col[out] output column
 * @param tmp_col[in] input column (one value per group)
 * @param grouping_info[in] structures used to get rows for each group
 *
 */
template <typename T>
void copy_nullable_values_transform(array_info* update_col, array_info* tmp_col,
                                    const grouping_info& grp_info) {
    int64_t nrows = update_col->length;
    bool bit = false;
    for (int64_t iRow = 0; iRow < nrows; iRow++) {
        int64_t igrp = grp_info.row_to_group[iRow];
        bit = tmp_col->get_null_bit(igrp);
        T val = tmp_col->template at<T>(igrp);
        update_col->set_null_bit(iRow, bit);
        update_col->template at<T>(iRow) = val;
    }
}
/**
 * Propagate value from the row in the tmp_col to all the rows in the
 * group update_col.
 * @param update_col[out]: column that has the final result for all rows
 * @param tmp_col[in]: column that has the result per group
 * @param grouping_info[in]: structures used to get rows for each group
 *
 * */
void copy_string_values_transform(array_info* update_col, array_info* tmp_col,
                                  const grouping_info& grp_info) {
    int64_t num_groups = grp_info.num_groups;
    array_info* out_arr = NULL;
    // first we have to deal with offsets first so we
    // need one first loop to determine the needed length. In the second
    // loop, the assignation is made. If the entries are missing then the
    // bitmask is set to false.
    bodo_array_type::arr_type_enum arr_type = tmp_col->arr_type;
    Bodo_CTypes::CTypeEnum dtype = tmp_col->dtype;
    int64_t n_chars = 0;
    int64_t nRowOut = update_col->length;
    // Store size of data per row
    std::vector<offset_t> ListSizes(nRowOut);
    offset_t* in_offsets = (offset_t*)tmp_col->data2;
    char* in_data1 = tmp_col->data1;
    // 1. Determine needed length (total number of characters)
    // and number of characters per element/row
    // All rows in same group gets same data
    for (int64_t igrp = 0; igrp < num_groups; igrp++) {
        offset_t size = 0;
        offset_t start_offset = in_offsets[igrp];
        offset_t end_offset = in_offsets[igrp + 1];
        size = end_offset - start_offset;
        int64_t idx = grp_info.group_to_first_row[igrp];
        while (true) {
            if (idx == -1) break;
            ListSizes[idx] = size;
            n_chars += size;
            idx = grp_info.next_row_in_group[idx];
        }
    }
    out_arr = alloc_array(nRowOut, n_chars, -1, arr_type, dtype, 0, 0);
    offset_t* out_offsets = (offset_t*)out_arr->data2;
    char* out_data1 = out_arr->data1;
    // keep track of output array position
    offset_t pos = 0;
    // 2. Copy data from tmp_col to corresponding rows in out_arr
    bool bit = false;
    for (int64_t iRow = 0; iRow < nRowOut; iRow++) {
        offset_t size = ListSizes[iRow];
        int64_t igrp = grp_info.row_to_group[iRow];
        offset_t start_offset = in_offsets[igrp];
        char* in_ptr = in_data1 + start_offset;
        char* out_ptr = out_data1 + pos;
        out_offsets[iRow] = pos;
        memcpy(out_ptr, in_ptr, size);
        pos += size;
        bit = tmp_col->get_null_bit(igrp);
        out_arr->set_null_bit(iRow, bit);
    }
    out_offsets[nRowOut] = pos;
    *update_col = std::move(*out_arr);
    delete out_arr;
}

/**
 * @brief Propagate value from the row in the tmp_col to all the rows in the
 * group update_col.
 *
 * @param update_col[out]: column that has the final result for all rows
 * @param tmp_col[in]: column that has the result per group
 * @param grouping_info[in]: structures used to get rows for each group
 *
 */
template <typename T>
void copy_values(array_info* update_col, array_info* tmp_col,
                 const grouping_info& grp_info) {
    if (tmp_col->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        copy_nullable_values_transform<T>(update_col, tmp_col, grp_info);
        return;
    }
    // Copy result from tmp_col to corresponding group rows in
    // update_col.
    int64_t nrows = update_col->length;
    for (int64_t iRow = 0; iRow < nrows; iRow++) {
        int64_t igrp = grp_info.row_to_group[iRow];
        T& val = getv<T>(tmp_col, igrp);
        T& val2 = getv<T>(update_col, iRow);
        val2 = val;
    }
}

/**
 * @brief Propagate value from the row in the tmp_col to all the rows in the
 * group update_col. Both tmp_col and update_col are dictionary encoded.
 *
 * @param update_col[out]: column that has the final result for all rows
 * @param tmp_col[in]: column that has the result per group
 * @param grouping_info[in]: structures used to get rows for each group
 *
 * */
void copy_dict_string_values_transform(array_info* update_col,
                                       array_info* tmp_col,
                                       const grouping_info& grp_info) {
    int64_t length = update_col->length;
    array_info* indices_arr =
        alloc_nullable_array(length, Bodo_CTypes::INT32, 0);
    copy_values<int32_t>(indices_arr, tmp_col->info2, grp_info);
    incref_array(tmp_col->info1);  // increase reference because we reuse the
                                   // underlying data array.
    array_info* out_col =
        create_dict_string_array(tmp_col->info1, indices_arr, length);
    *update_col = std::move(*out_col);
    // reverse_shuffle_table needs the dictionary to be global
    // copy_dict_string_values_transform is only called on distributed data
    // Does this implementation require the dictionary is sorted.
    // Similarly does it require that there are no duplicates.
    make_dictionary_global_and_unique(update_col, true);
    delete out_col;
}

void copy_values_transform(array_info* update_col, array_info* tmp_col,
                           const grouping_info& grp_info) {
    switch (tmp_col->dtype) {
        case Bodo_CTypes::_BOOL:
            copy_values<bool>(update_col, tmp_col, grp_info);
            break;
        case Bodo_CTypes::INT8:
            copy_values<int8_t>(update_col, tmp_col, grp_info);
            break;
        case Bodo_CTypes::UINT8:
            copy_values<uint8_t>(update_col, tmp_col, grp_info);
            break;
        case Bodo_CTypes::INT16:
            copy_values<int16_t>(update_col, tmp_col, grp_info);
            break;
        case Bodo_CTypes::UINT16:
            copy_values<uint16_t>(update_col, tmp_col, grp_info);
            break;
        case Bodo_CTypes::INT32:
            copy_values<int32_t>(update_col, tmp_col, grp_info);
            break;
        case Bodo_CTypes::UINT32:
            copy_values<uint32_t>(update_col, tmp_col, grp_info);
            break;
        case Bodo_CTypes::DATE:
        case Bodo_CTypes::DATETIME:
        case Bodo_CTypes::TIMEDELTA:
        case Bodo_CTypes::INT64:
        // TODO: [BE-4106] Split Time into Time32 and Time64
        case Bodo_CTypes::TIME:
            copy_values<int64_t>(update_col, tmp_col, grp_info);
            break;
        case Bodo_CTypes::UINT64:
            copy_values<uint64_t>(update_col, tmp_col, grp_info);
            break;
        case Bodo_CTypes::FLOAT32:
            copy_values<float>(update_col, tmp_col, grp_info);
            break;
        case Bodo_CTypes::FLOAT64:
            copy_values<double>(update_col, tmp_col, grp_info);
            break;
        case Bodo_CTypes::STRING:
            if (tmp_col->arr_type == bodo_array_type::DICT) {
                copy_dict_string_values_transform(update_col, tmp_col,
                                                  grp_info);
            } else {
                copy_string_values_transform(update_col, tmp_col, grp_info);
            }
            break;
        default:
            throw new std::runtime_error("unsupported data type for eval: " +
                                         GetDtype_as_string(tmp_col->dtype));
    }
}
