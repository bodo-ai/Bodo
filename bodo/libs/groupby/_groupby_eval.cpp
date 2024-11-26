// Copyright (C) 2023 Bodo Inc. All rights reserved.

#include "../_array_utils.h"
#include "../_dict_builder.h"
#include "_groupby.h"

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
void copy_nullable_values_transform(std::shared_ptr<array_info> update_col,
                                    std::shared_ptr<array_info> tmp_col,
                                    const grouping_info& grp_info) {
    assert(tmp_col->arr_type == bodo_array_type::NULLABLE_INT_BOOL);
    assert(update_col->arr_type == bodo_array_type::NULLABLE_INT_BOOL);
    int64_t nrows = update_col->length;
    for (int64_t iRow = 0; iRow < nrows; iRow++) {
        int64_t igrp = grp_info.row_to_group[iRow];
        // Update the bitmap
        bool bit =
            tmp_col->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(igrp);
        update_col->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(iRow, bit);
        // Update the value
        T& val = getv<T, bodo_array_type::NULLABLE_INT_BOOL>(tmp_col, igrp);
        T& val2 = getv<T, bodo_array_type::NULLABLE_INT_BOOL>(update_col, iRow);
        val2 = val;
    }
}
/**
 * Propagate value from the row in the tmp_col to all the rows in the
 * group update_col.
 * @param update_col[out]: column that has the final result for all rows
 * @param tmp_col[in]: column that has the result per group
 * @param grouping_info[in]: structures used to get rows for each group
 * @param pool Memory pool to use for allocations during the execution of
 * this function.
 * @param mm Memory manager associated with the pool.
 *
 * */
void copy_string_values_transform(
    std::shared_ptr<array_info> update_col, std::shared_ptr<array_info> tmp_col,
    const grouping_info& grp_info,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager()) {
    assert(tmp_col->arr_type == bodo_array_type::STRING);
    int64_t num_groups = grp_info.num_groups;
    std::shared_ptr<array_info> out_arr = nullptr;
    // first we have to deal with offsets first so we
    // need one first loop to determine the needed length. In the second
    // loop, the assignation is made. If the entries are missing then the
    // bitmask is set to false.
    bodo_array_type::arr_type_enum arr_type = tmp_col->arr_type;
    Bodo_CTypes::CTypeEnum dtype = tmp_col->dtype;
    int64_t n_chars = 0;
    int64_t nRowOut = update_col->length;
    // Store size of data per row
    bodo::vector<offset_t> ListSizes(nRowOut, pool);
    offset_t* in_offsets = (offset_t*)tmp_col->data2<bodo_array_type::STRING>();
    char* in_data1 = tmp_col->data1<bodo_array_type::STRING>();
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
            if (idx == -1)
                break;
            ListSizes[idx] = size;
            n_chars += size;
            idx = grp_info.next_row_in_group[idx];
        }
    }
    out_arr =
        alloc_array_top_level(nRowOut, n_chars, -1, arr_type, dtype, -1, 0, 0,
                              false, false, false, pool, std::move(mm));
    offset_t* out_offsets =
        (offset_t*)out_arr->data2<bodo_array_type::STRING>();
    char* out_data1 = out_arr->data1<bodo_array_type::STRING>();
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
        bit = tmp_col->get_null_bit<bodo_array_type::STRING>(igrp);
        out_arr->set_null_bit<bodo_array_type::STRING>(iRow, bit);
    }
    out_offsets[nRowOut] = pos;
    *update_col = std::move(*out_arr);
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
void copy_values(std::shared_ptr<array_info> update_col,
                 std::shared_ptr<array_info> tmp_col,
                 const grouping_info& grp_info) {
    if (tmp_col->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        assert(update_col->arr_type == bodo_array_type::NULLABLE_INT_BOOL);
        if (tmp_col->dtype == Bodo_CTypes::_BOOL) {
            // Nullable booleans store 1 bit per boolean so we have
            // a separate code path.
            int64_t nrows = update_col->length;
            for (int64_t iRow = 0; iRow < nrows; iRow++) {
                int64_t igrp = grp_info.row_to_group[iRow];
                // Update the bitmap
                bool null_bit =
                    tmp_col->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(
                        igrp);
                update_col->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(
                    iRow, null_bit);
                // Update the value
                bool data_bit = GetBit(
                    (uint8_t*)
                        tmp_col->data1<bodo_array_type::NULLABLE_INT_BOOL>(),
                    igrp);
                SetBitTo((uint8_t*)update_col
                             ->data1<bodo_array_type::NULLABLE_INT_BOOL>(),
                         iRow, data_bit);
            }
        } else {
            copy_nullable_values_transform<T>(update_col, tmp_col, grp_info);
        }
    } else {
        // Numpy array. No bitmap
        // Copy result from tmp_col to corresponding group rows in
        // update_col.
        int64_t nrows = update_col->length;
        for (int64_t iRow = 0; iRow < nrows; iRow++) {
            int64_t igrp = grp_info.row_to_group[iRow];
            // TODO XXX getv needs to be templated with the arr type!
            T& val = getv<T>(tmp_col, igrp);
            T& val2 = getv<T>(update_col, iRow);
            val2 = val;
        }
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
void copy_dict_string_values_transform(std::shared_ptr<array_info> update_col,
                                       std::shared_ptr<array_info> tmp_col,
                                       const grouping_info& grp_info,
                                       bool is_parallel) {
    copy_values<int32_t>(update_col->child_arrays[1], tmp_col->child_arrays[1],
                         grp_info);
    update_col->child_arrays[0] = tmp_col->child_arrays[0];
    // reverse_shuffle_table needs the dictionary to be global
    // copy_dict_string_values_transform is only called on distributed data
    // Does this implementation require the dictionary is sorted.
    // Similarly does it require that there are no duplicates.
    make_dictionary_global_and_unique(update_col, is_parallel);
}

void copy_values_transform(std::shared_ptr<array_info> update_col,
                           std::shared_ptr<array_info> tmp_col,
                           const grouping_info& grp_info, bool is_parallel,
                           bodo::IBufferPool* const pool,
                           std::shared_ptr<::arrow::MemoryManager> mm) {
    if (tmp_col->arr_type == bodo_array_type::DICT) {
        // XXX Technically this calls make_dictionary_global_and_unique which
        // may make allocations, so we should be passing the pool and mm,
        // but it's not a high priority right now since this is not
        // supported in the streaming case anyway.
        copy_dict_string_values_transform(update_col, tmp_col, grp_info,
                                          is_parallel);
    } else if (tmp_col->arr_type == bodo_array_type::STRING) {
        copy_string_values_transform(update_col, tmp_col, grp_info, pool,
                                     std::move(mm));
    } else {
        // macro to reduce code duplication
#ifndef COPY_VALUES_CALL
#define COPY_VALUES_CALL(CTYPE)                                               \
    if (tmp_col->dtype == CTYPE) {                                            \
        copy_values<typename dtype_to_type<CTYPE>::type>(update_col, tmp_col, \
                                                         grp_info);           \
        return;                                                               \
    }
#endif
        COPY_VALUES_CALL(Bodo_CTypes::_BOOL)
        COPY_VALUES_CALL(Bodo_CTypes::INT8)
        COPY_VALUES_CALL(Bodo_CTypes::UINT8)
        COPY_VALUES_CALL(Bodo_CTypes::INT16)
        COPY_VALUES_CALL(Bodo_CTypes::UINT16)
        COPY_VALUES_CALL(Bodo_CTypes::INT32)
        COPY_VALUES_CALL(Bodo_CTypes::UINT32)
        COPY_VALUES_CALL(Bodo_CTypes::INT64)
        COPY_VALUES_CALL(Bodo_CTypes::UINT64)
        COPY_VALUES_CALL(Bodo_CTypes::DATE)
        COPY_VALUES_CALL(Bodo_CTypes::DATETIME)
        COPY_VALUES_CALL(Bodo_CTypes::TIMEDELTA)
        COPY_VALUES_CALL(Bodo_CTypes::TIME)
        COPY_VALUES_CALL(Bodo_CTypes::FLOAT32)
        COPY_VALUES_CALL(Bodo_CTypes::FLOAT64)
        // None of the calls match. If any matched we return from the macro.
        throw std::runtime_error("unsupported data type for eval: " +
                                 GetDtype_as_string(tmp_col->dtype));
    }
#undef COPY_VALUES_CALL
}
