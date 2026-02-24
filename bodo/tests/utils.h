#pragma once

#include "../libs/_array_utils.h"
#include "../libs/_bodo_common.h"

/* @brief Helper utility to create an array from vectors of dtype_to_type<dtype>
 * and nulls
 * @param numbers vector of integers
 * @param nulls vector of booleans indicating nulls
 * @return shared pointer to the created array
 */
template <Bodo_CTypes::CTypeEnum dtype>
    requires(dtype != Bodo_CTypes::_BOOL)
std::shared_ptr<array_info> nullable_array_from_vector(
    std::vector<typename dtype_to_type<dtype>::type> numbers,
    std::vector<bool> nulls) {
    using T = typename dtype_to_type<dtype>::type;
    size_t length = numbers.size();
    auto result = alloc_nullable_array_no_nulls(length, dtype);
    T *buffer = result->data1<bodo_array_type::NULLABLE_INT_BOOL, T>();
    for (size_t i = 0; i < length; i++) {
        if (nulls[i]) {
            buffer[i] = (T)numbers[i];
        } else {
            result->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i, false);
        }
    }
    return result;
}

// Special case of nullable_array_from_vector for booleans
template <Bodo_CTypes::CTypeEnum dtype>
    requires(dtype == Bodo_CTypes::_BOOL)
std::shared_ptr<array_info> nullable_array_from_vector(
    std::vector<bool> booleans, std::vector<bool> nulls) {
    size_t length = booleans.size();
    auto result = alloc_nullable_array_no_nulls(length, dtype);
    uint8_t *buffer =
        result->data1<bodo_array_type::NULLABLE_INT_BOOL, uint8_t>();
    for (size_t i = 0; i < length; i++) {
        if (nulls[i]) {
            SetBitTo(buffer, i, booleans[i]);
        } else {
            result->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i, false);
        }
    }
    return result;
}

/* @brief Helper utility to create a string array from vectors of strings and
 * nulls
 * @param strings vector of strings
 * @param nulls vector of booleans indicating nulls
 * @param dtype type of the string array
 * @return shared pointer to the created array
 */
inline std::shared_ptr<array_info> string_array_from_vector(
    bodo::vector<std::string> strings, std::vector<bool> nulls,
    Bodo_CTypes::CTypeEnum dtype) {
    size_t length = strings.size();

    bodo::vector<uint8_t> null_bitmask((length + 7) >> 3, 0);
    for (size_t i = 0; i < length; i++) {
        SetBitTo(null_bitmask.data(), i, nulls[i]);
    }
    return create_string_array(dtype, null_bitmask, strings, -1);
}

/* @brief Helper utility to create a dictionary array from vectors of strings
 * and indices
 * @param strings vector of strings
 * @param indices vector of integers
 * @param nulls vector of booleans indicating nulls
 * @return shared pointer to the created array
 */
inline std::shared_ptr<array_info> dict_array_from_vector(
    bodo::vector<std::string> strings, std::vector<int32_t> indices,
    std::vector<bool> nulls) {
    std::vector<bool> string_nulls(strings.size(), true);
    std::shared_ptr<array_info> dict_arr =
        string_array_from_vector(strings, string_nulls, Bodo_CTypes::STRING);
    std::shared_ptr<array_info> index_arr =
        nullable_array_from_vector<Bodo_CTypes::INT32>(indices, nulls);
    return create_dict_string_array(dict_arr, index_arr);
}
