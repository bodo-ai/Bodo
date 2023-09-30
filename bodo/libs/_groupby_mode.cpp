// Copyright (C) 2023 Bodo Inc. All rights reserved.
#include "_array_hash.h"
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_groupby_common.h"
#include "_groupby_hashing.h"
#include "_shuffle.h"

/**
 * @brief Perform mode operation on an array and store the result in out_arr.
 *
 * @param[in] arr The array to operate on.
 * @param[in, out] out_arr The array to store the result.
 * @param[in] grp_info Information about groupings that are to be operated on.
 * @param pool Memory pool to use for allocations during the execution of this
 * function.
 */
template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType>
void mode_operation(
    std::shared_ptr<array_info> arr, std::shared_ptr<array_info> out_arr,
    const grouping_info& grp_info,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr()) {
    using T = typename dtype_to_type<DType>::type;
    size_t num_group = grp_info.group_to_first_row.size();
    for (size_t igrp = 0; igrp < num_group; igrp++) {
        // Set up a hashtable with key type T and value type integer
        bodo::unord_map_container<T, int> counts(pool);
        // Keep track of NaN values seperately
        size_t nan_count = 0;
        // Iterate over all the elements in the group by starting
        // at the first row in the group and following the
        // next_row_in_group array until we reach -1, indicating
        // that every row in the group has been traversed.
        int64_t i = grp_info.group_to_first_row[igrp];
        while (i != -1) {
            // If the current entry is non-null, increment
            // the hashtable or the nan count
            if (non_null_at<ArrType, T, DType>(*arr, i)) {
                if (isnan_alltype<T, DType>(getv<T>(arr, i))) {
                    nan_count++;
                } else {
                    ++counts[get_arr_item<ArrType, T, DType>(*arr, i)];
                }
            }
            i = grp_info.next_row_in_group[i];
        }
        // If at least 1 non-null element was found, find
        // the value with the largest count and set it as
        // the answer for the current group
        if (counts.size() > 0 || nan_count > 0) {
            set_non_null<ArrType, T, DType>(*out_arr, igrp);
            // Set up the best element and best count, which
            // should start as NaN with the NaN count if
            // there were any NaN elements
            T best_elem = {};
            int best_count = 0;
            if (nan_count > 0) {
                best_elem = nan_val<T, DType>();
                best_count = nan_count;
            }
            // Loop through every non-NaN element
            // to find the one with the highest count
            for (const auto& [elem, count] : counts) {
                if (count > best_count) {
                    best_count = count;
                    best_elem = elem;
                }
            }
            set_arr_item<ArrType, T, DType>(*out_arr, igrp, best_elem);
        }
    }
}

/**
 * @brief Compute the mode within each group for an array of strings. Within
 * each group, a hashtable mapping each string in the group to its count is
 * built, then the string with the highest count is found. The most frquent
 * strings for each group are placed into a vector which is then converted into
 * the final array.
 *
 * @param[in] arr: the array to process.
 * @param[in] out_arr: the array to write the result to.
 * @param[in] grp_info: information about the grouping.
 * @param pool Memory pool to use for allocations during the execution of this
 * function.
 * @param mm Memory manager associated with the pool.
 */
void mode_operation_strings(
    std::shared_ptr<array_info> arr, std::shared_ptr<array_info> out_arr,
    const grouping_info& grp_info,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager()) {
    char* data = arr->data1();
    offset_t* offsets = (offset_t*)arr->data2();
    size_t num_groups = out_arr->length;
    // Set up the vectors to store the null bits and the most
    // frequent string for each group
    bodo::vector<uint8_t> nulls((num_groups + 7) >> 3, 0, pool);
    bodo::vector<std::string> strings(num_groups, pool);
    size_t num_group = grp_info.group_to_first_row.size();
    for (size_t igrp = 0; igrp < num_group; igrp++) {
        // Set up a hashtable mapping each string seen in the group
        // to its count
        bodo::unord_map_container<std::string_view, int> counts(pool);
        // Iterate over all the elements in the group by starting
        // at the first row in the group and following the
        // next_row_in_group array until we reach -1, indicating
        // that every row in the group has been traversed.
        int64_t i = grp_info.group_to_first_row[igrp];
        while (i != -1) {
            // If the current element in the group is not null,
            // extract the corresponding string from the character
            // buffer and increment its count in the hashtable
            if (arr->get_null_bit(i)) {
                offset_t start_offset = offsets[i];
                offset_t end_offset = offsets[i + 1];
                offset_t len = end_offset - start_offset;
                std::string_view substr(&data[start_offset], len);
                ++counts[substr];
            }
            i = grp_info.next_row_in_group[i];
        }
        // If at least 1 non-null element was found, find
        // the value with the largest count and set it as
        // the answer for the current group, storing it
        // in the vector
        if (counts.size() > 0) {
            SetBitTo(nulls.data(), igrp, true);
            std::string best_elem = {};
            int best_count = 0;
            // Find string with the highest count
            for (auto it = counts.begin(); it != counts.end(); it++) {
                int count = it->second;
                if (count > best_count) {
                    best_count = count;
                    best_elem = it->first;
                }
            }
            strings[igrp] = best_elem;
        } else {
            // Otherwise, the mode of the group is set to null
            SetBitTo(nulls.data(), igrp, false);
        }
    }
    // Convert the vectors into a proper string array, and then
    // replace the dummy output array with the new one
    std::shared_ptr<array_info> new_out_arr = create_string_array(
        Bodo_CTypes::STRING, nulls, strings, -1, pool, std::move(mm));
    *out_arr = std::move(*new_out_arr);
}

/**
 * @brief Compute the mode within each group for an array of strings using
 * dictionary encoding. This is done by finding the mode of the indices
 * (instead of the strings themselves), and creating a new dictionary encoded
 * array using the same dictionary as the input but with the indices from the
 * mode calculation.
 *
 * TODO: possibly drop unused indices from the output dictionary array?
 *
 * @param[in] arr: the array to process.
 * @param[in,out] out_arr: the array to write the result to.
 * @param[in] grp_info: information about the grouping.
 */
void mode_operation_strings_dict(std::shared_ptr<array_info> arr,
                                 std::shared_ptr<array_info> out_arr,
                                 const grouping_info& grp_info) {
    drop_duplicates_local_dictionary(arr);
    // after make_dictionary_global_and_unique has already been called, we can
    // find the mode of the strings by finding the mode of the indices, then
    // extracting the corresponding string for each of them.
    std::shared_ptr<array_info> indices = arr->child_arrays[1];
    size_t num_groups = out_arr->length;
    // Create an array of nullable integers that will store the mode
    // of the indices within each group
    std::shared_ptr<array_info> out_indices =
        alloc_nullable_array_all_nulls(num_groups, Bodo_CTypes::INT32, 0);
    mode_operation<bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::INT32>(
        indices, out_indices, grp_info);
    // Create a new dictionary encoded array using the indices derived
    // from the mode computation
    std::shared_ptr<array_info> new_out_arr =
        create_dict_string_array(arr->child_arrays[0], out_indices);
    *out_arr = std::move(*new_out_arr);
}

/**
 * @brief Case on the dtype of a numpy/nullable array in order to
 * perform the mode operation using the correct templated arguments.
 *
 * @param[in] arr the input array to perform the operation on.
 * @param[in, out] out_arr the output array to write the result.
 * @param[in] grp_info Information about groupings that are to be operated on.
 * @param pool Memory pool to use for allocations during the execution of this
 * function.
 */
template <bodo_array_type::arr_type_enum ArrType>
void do_mode_computation(
    std::shared_ptr<array_info> arr, std::shared_ptr<array_info> out_arr,
    const grouping_info& grp_info,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr()) {
    switch (arr->dtype) {
        case Bodo_CTypes::UINT8: {
            mode_operation<ArrType, Bodo_CTypes::UINT8>(arr, out_arr, grp_info,
                                                        pool);
            break;
        }
        case Bodo_CTypes::UINT16: {
            mode_operation<ArrType, Bodo_CTypes::UINT16>(arr, out_arr, grp_info,
                                                         pool);
            break;
        }
        case Bodo_CTypes::UINT32: {
            mode_operation<ArrType, Bodo_CTypes::UINT32>(arr, out_arr, grp_info,
                                                         pool);
            break;
        }
        case Bodo_CTypes::UINT64: {
            mode_operation<ArrType, Bodo_CTypes::UINT64>(arr, out_arr, grp_info,
                                                         pool);
            break;
        }
        case Bodo_CTypes::INT8: {
            mode_operation<ArrType, Bodo_CTypes::INT8>(arr, out_arr, grp_info,
                                                       pool);
            break;
        }
        case Bodo_CTypes::INT16: {
            mode_operation<ArrType, Bodo_CTypes::INT16>(arr, out_arr, grp_info,
                                                        pool);
            break;
        }
        case Bodo_CTypes::INT32: {
            mode_operation<ArrType, Bodo_CTypes::INT32>(arr, out_arr, grp_info,
                                                        pool);
            break;
        }
        case Bodo_CTypes::INT64: {
            mode_operation<ArrType, Bodo_CTypes::INT64>(arr, out_arr, grp_info,
                                                        pool);
            break;
        }
        case Bodo_CTypes::FLOAT32: {
            mode_operation<ArrType, Bodo_CTypes::FLOAT32>(arr, out_arr,
                                                          grp_info, pool);
            break;
        }
        case Bodo_CTypes::FLOAT64: {
            mode_operation<ArrType, Bodo_CTypes::FLOAT64>(arr, out_arr,
                                                          grp_info, pool);
            break;
        }
        case Bodo_CTypes::DECIMAL: {
            mode_operation<ArrType, Bodo_CTypes::DECIMAL>(arr, out_arr,
                                                          grp_info, pool);
            break;
        }
        case Bodo_CTypes::DATETIME: {
            mode_operation<ArrType, Bodo_CTypes::DATETIME>(arr, out_arr,
                                                           grp_info, pool);
            break;
        }
        case Bodo_CTypes::TIMEDELTA: {
            mode_operation<ArrType, Bodo_CTypes::TIMEDELTA>(arr, out_arr,
                                                            grp_info, pool);
            break;
        }
        case Bodo_CTypes::TIME: {
            mode_operation<ArrType, Bodo_CTypes::TIME>(arr, out_arr, grp_info,
                                                       pool);
            break;
        }
        case Bodo_CTypes::DATE: {
            mode_operation<ArrType, Bodo_CTypes::DATE>(arr, out_arr, grp_info,
                                                       pool);
            break;
        }
        case Bodo_CTypes::_BOOL: {
            mode_operation<ArrType, Bodo_CTypes::_BOOL>(arr, out_arr, grp_info,
                                                        pool);
            break;
        }
        default: {
            throw std::runtime_error(
                "_groupby_update.cpp::do_mode_computation: Unsupported dtype "
                "encountered. Found type: " +
                GetDtype_as_string(arr->dtype));
        }
    }
}

void mode_computation(std::shared_ptr<array_info> arr,
                      std::shared_ptr<array_info> out_arr,
                      const grouping_info& grp_info,
                      bodo::IBufferPool* const pool,
                      std::shared_ptr<::arrow::MemoryManager> mm) {
    switch (arr->arr_type) {
        case bodo_array_type::NUMPY: {
            do_mode_computation<bodo_array_type::NUMPY>(arr, out_arr, grp_info,
                                                        pool);
            break;
        }
        case bodo_array_type::NULLABLE_INT_BOOL: {
            do_mode_computation<bodo_array_type::NULLABLE_INT_BOOL>(
                arr, out_arr, grp_info, pool);
            break;
        }
        case bodo_array_type::STRING: {
            mode_operation_strings(arr, out_arr, grp_info, pool, std::move(mm));
            break;
        }
        case bodo_array_type::DICT: {
            // NOTE: We don't support custom pools for the DICT case yet.
            // This is fine since streaming groupby doesn't support
            // MODE yet anyway (https://bodo.atlassian.net/browse/BSE-1139).
            mode_operation_strings_dict(arr, out_arr, grp_info);
            break;
        }
        default: {
            throw std::runtime_error(
                "Unsupported array type encountered with mode. Found type: " +
                GetArrType_as_string(arr->arr_type));
        }
    }
}
