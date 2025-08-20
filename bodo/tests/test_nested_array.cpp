#include <fmt/core.h>
#include <algorithm>
#include <sstream>

#include "../libs/_array_hash.h"
#include "../libs/_array_utils.h"
#include "../libs/_bodo_common.h"
#include "./test.hpp"

std::shared_ptr<array_info> get_sample_array_item() {
    // Create the internal array containing numbers from 0 to 9
    std::shared_ptr<array_info> inner_arr =
        alloc_nullable_array_no_nulls(10, Bodo_CTypes::INT64);
    for (int64_t i = 0; i < 10; i++) {
        getv<int64_t>(inner_arr, i) = i;
    }
    // Create the array item array that has 5 sub-arrays
    std::shared_ptr<array_info> arr = alloc_array_item(5, inner_arr);
    // Populate the offsets to indicate which of the 10 elements from
    // the original array belong to which of the 5 sub-arrays.
    offset_t *offset_buffer =
        (offset_t *)(arr->buffers[0]->mutable_data() + arr->offset);
    offset_buffer[0] = 0;
    offset_buffer[1] = 1;
    offset_buffer[2] = 5;
    offset_buffer[3] = 7;
    offset_buffer[4] = 7;
    offset_buffer[5] = 10;
    return arr;
}

std::shared_ptr<array_info> get_sample_struct(size_t nfields = 1) {
    std::vector<std::shared_ptr<array_info>> children =
        std::vector<std::shared_ptr<array_info>>(nfields);
    for (size_t i = 0; i < nfields; ++i) {
        // Create the child array containing numbers from 0 to 9
        std::shared_ptr<array_info> child_array =
            alloc_nullable_array_no_nulls(10, Bodo_CTypes::INT64);
        for (int64_t i = 0; i < 10; i++) {
            getv<int64_t>(child_array, i) = i;
        }
        children[i] = child_array;
    }
    // Create the array item array with one child array
    auto struct_arr = alloc_struct(10, children);
    // Name the fields alphabetically
    for (size_t i = 0; i < nfields; ++i) {
        struct_arr->field_names.emplace_back(static_cast<uint64_t>(i / 26 + 1),
                                             'a' + i % 26);
    }
    return struct_arr;
}

// Produces a sample map array with 10 elements of the form
// [{"0":0}, {"1":1, "2":2, "3":3, "4":4}, {"5":5, "6":6}, {}, {"7":7, "8":8,
// "9":9}]
std::shared_ptr<array_info> get_sample_map() {
    std::shared_ptr<array_info> keys =
        alloc_string_array(Bodo_CTypes::STRING, 10, 10);
    char *str_data = keys->data1();
    std::string("0123456789").copy(str_data, 10);
    offset_t *keys_offset_buffer = (offset_t *)(keys->data2() + keys->offset);
    for (int64_t i = 0; i <= 10; i++) {
        keys_offset_buffer[i] = i;
    }
    std::shared_ptr<array_info> values =
        alloc_nullable_array_no_nulls(10, Bodo_CTypes::INT64);
    for (int64_t i = 0; i < 10; i++) {
        getv<int64_t>(values, i) = i;
    }

    // Create the struct mapping string to int
    std::shared_ptr<array_info> struct_child = alloc_struct(10, {keys, values});
    struct_child->field_names = {"keys", "values"};
    std::shared_ptr<array_info> list_child = alloc_array_item(5, struct_child);
    // Populate the offsets to indicate which of the 10 elements from
    // the original array belong to which of the 5 sub-arrays.
    offset_t *offset_buffer =
        (offset_t *)(list_child->buffers[0]->mutable_data() +
                     list_child->offset);
    offset_buffer[0] = 0;
    offset_buffer[1] = 1;
    offset_buffer[2] = 5;
    offset_buffer[3] = 7;
    offset_buffer[4] = 7;
    offset_buffer[5] = 10;
    // Create the array item array with one child array
    return alloc_map(5, list_child);
}

void check_row_helper(std::shared_ptr<array_info> arr1,
                      std::shared_ptr<array_info> arr2,
                      std::vector<bool> test_vals) {
    if (test_vals.size() != arr1->length || arr1->length != arr2->length) {
        throw std::runtime_error(
            "test_array_item_array_comparison: invalid lengths in "
            "check_row_helper");
    }
    for (uint64_t i = 0; i < arr1->length; i++) {
        bodo::tests::check(
            TestEqualColumn(arr1, i, arr2, i, true) == test_vals[i],
            fmt::format("check failed for idx {}", i).c_str());
    }
};

void check_hashes_helper(std::shared_ptr<uint32_t[]> hashes1,
                         std::shared_ptr<uint32_t[]> hashes2,
                         std::vector<bool> test_vals) {
    for (size_t i = 0; i < test_vals.size(); i++) {
        bodo::tests::check((hashes1[i] == hashes2[i]) == test_vals[i],
                           fmt::format("check failed for idx {}", i).c_str());
    }
};

const std::vector<int8_t> sample_array_types{bodo_array_type::NUMPY,
                                             bodo_array_type::ARRAY_ITEM,
                                             bodo_array_type::ARRAY_ITEM,
                                             bodo_array_type::DICT,
                                             bodo_array_type::ARRAY_ITEM,
                                             bodo_array_type::STRING,
                                             bodo_array_type::STRUCT,
                                             2,
                                             bodo_array_type::NULLABLE_INT_BOOL,
                                             bodo_array_type::ARRAY_ITEM,
                                             bodo_array_type::CATEGORICAL,
                                             bodo_array_type::NULLABLE_INT_BOOL,
                                             bodo_array_type::ARRAY_ITEM,
                                             bodo_array_type::STRUCT,
                                             2,
                                             bodo_array_type::NULLABLE_INT_BOOL,
                                             bodo_array_type::NUMPY};

static bodo::tests::suite tests([] {
    bodo::tests::test("test_array_item_construction", [] {
        std::shared_ptr<array_info> arr = get_sample_array_item();
        // Check to make sure the lengths match.
        int64_t n_rows = arr->length;
        bodo::tests::check(n_rows == 5);
        std::stringstream ss;
        // Dumping the column to a string to ensure it matches.
        DEBUG_PrintColumn(ss, arr);
        bodo::tests::check(ss.str() ==
                           "ARRAY_INFO: Column n=5 arr=ARRAY_ITEM dtype=LIST\n"
                           "i_row=0 S=[[0]]\n"
                           "i_row=1 S=[[1,2,3,4]]\n"
                           "i_row=2 S=[[5,6]]\n"
                           "i_row=3 S=[[]]\n"
                           "i_row=4 S=[[7,8,9]]\n");
    });

    bodo::tests::test("test_alloc_array_like", [] {
        // ARRAY_ITEM
        std::shared_ptr<array_info> arr =
            alloc_array_like(get_sample_array_item());
        bodo::tests::check(arr->length == 0);
        bodo::tests::check(
            ((offset_t *)arr->data1<bodo_array_type::ARRAY_ITEM>())[0] == 0);
        bodo::tests::check(arr->arr_type == bodo_array_type::ARRAY_ITEM);
        bodo::tests::check(arr->child_arrays[0]->arr_type ==
                           bodo_array_type::NULLABLE_INT_BOOL);
        bodo::tests::check(arr->child_arrays[0]->length == 0);
        // STRUCT
        arr = alloc_array_like(get_sample_struct());
        bodo::tests::check(arr->length == 0);
        bodo::tests::check(arr->arr_type == bodo_array_type::STRUCT);
        bodo::tests::check(arr->dtype == Bodo_CTypes::STRUCT);
        bodo::tests::check(arr->child_arrays.size() == 1);
        bodo::tests::check(arr->child_arrays[0]->arr_type ==
                           bodo_array_type::NULLABLE_INT_BOOL);
        bodo::tests::check(arr->child_arrays[0]->length == 0);
    });

    bodo::tests::test("test_get_col_idx_map", [] {
        std::span<const int8_t> arr_array_types(sample_array_types);
        bodo::tests::check(get_col_idx_map(arr_array_types) ==
                           std::vector<size_t>{0, 1, 4, 6, 11, 12});
        bodo::tests::check(get_col_idx_map(arr_array_types.subspan<8, 3>()) ==
                           std::vector<size_t>{0, 1});
        bool exception_caught = false;
        try {
            get_col_idx_map(arr_array_types.subspan<0, 13>());
        } catch (std::runtime_error &ex) {
            bodo::tests::check(std::string(ex.what()) ==
                               "The last array type cannot be ARRAY_ITEM: "
                               "inner array type needs to be provided!");
            exception_caught = true;
        }
        bodo::tests::check(exception_caught);
    });

    bodo::tests::test("test_empty_col_idx_map", [] {
        bodo::tests::check(get_col_idx_map(std::span<int8_t>()) ==
                           std::vector<size_t>());
    });

    bodo::tests::test("test_array_item_array_comparison", [] {
        // arr1 = [[0], [1,2,3,4], [5,6], [], [7,8,9]]
        // arr2 = [[0], [1,2,3,4], [5,6], [], [7,8,9]]
        std::shared_ptr<array_info> arr1 = get_sample_array_item();
        std::shared_ptr<array_info> arr2 = get_sample_array_item();
        check_row_helper(arr1, arr2, {true, true, true, true, true});
        // Change arr1 to have a different value
        // arr1 = [[1], [1,2,3,4], [5,6], [], [7,8,9]]
        getv<int64_t>(arr1->child_arrays[0], 0) = 1;
        check_row_helper(arr1, arr2, {false, true, true, true, true});
        getv<int64_t>(arr1->child_arrays[0], 0) = 0;

        // Change arr1 to have a different offset
        // arr1 = [[0], [1,2,3], [4, 5,6], [], [7,8,9]]
        ((offset_t *)arr1->buffers[0]->mutable_data())[2] = 4;
        check_row_helper(arr1, arr2, {true, false, false, true, true});
        ((offset_t *)arr1->buffers[0]->mutable_data())[2] = 5;

        // Reorder arr1[1]
        // arr1 = [[0], [4, 3,2,1], [5,6], [], [7,8,9]]
        getv<int64_t>(arr1->child_arrays[0], 1) = 4;
        getv<int64_t>(arr1->child_arrays[0], 2) = 3;
        getv<int64_t>(arr1->child_arrays[0], 3) = 2;
        getv<int64_t>(arr1->child_arrays[0], 4) = 1;
        check_row_helper(arr1, arr2, {true, false, true, true, true});
        getv<int64_t>(arr1->child_arrays[0], 1) = 1;
        getv<int64_t>(arr1->child_arrays[0], 2) = 2;
        getv<int64_t>(arr1->child_arrays[0], 3) = 3;
        getv<int64_t>(arr1->child_arrays[0], 4) = 4;

        // Set a value to null and ensure it is not equal
        // arr1 = [[0], [1,2,3,4], [5,6], null, [7,8,9]]
        arr1->set_null_bit(3, false);
        check_row_helper(arr1, arr2, {true, true, true, false, true});
    });
    bodo::tests::test("test_struct_array_comparison", [] {
        // struct_arr1 = struct_arr2 =[{"a":0, "b":0}, {"a": 1, "b":1}, {"a":2,
        // "b":2}, {"a": 3, "b":3}, {"a":4, "b":4}, {"a":5, "b":5}, {"a":6,
        // "b":6}, {"a":7, "b":7}, {"a":8, "b":8}, {"a":9, "b": 9}]
        std::shared_ptr<array_info> struct_arr1 = get_sample_struct(2);
        std::shared_ptr<array_info> struct_arr2 = get_sample_struct(2);
        auto truth_vec = std::vector<bool>(10, true);
        check_row_helper(struct_arr1, struct_arr2, truth_vec);

        // Change values and ensure the appropriate rows are not equal
        // struct_arr1 = [{"a":-1, "b":0}, {"a": 1, "b":-1}, {"a":0, "b":1} ...]
        int64_t *a_data = (int64_t *)struct_arr1->child_arrays[0]->data1();
        int64_t *b_data = (int64_t *)struct_arr1->child_arrays[1]->data1();
        // Change one field in the first and second row
        a_data[0] = -1;
        b_data[1] = -1;
        // Change both fields in the third row
        a_data[2] = 0;
        b_data[2] = 1;

        truth_vec[0] = false;
        truth_vec[1] = false;
        truth_vec[2] = false;
        check_row_helper(struct_arr1, struct_arr2, truth_vec);

        // Set a value to null and ensure it is not equal
        // struct_arr1 = [{"a":-1, "b":0}, null, {"a":0, "b":1} ...]
        struct_arr1->set_null_bit(1, false);
        truth_vec[1] = false;
        check_row_helper(struct_arr1, struct_arr2, truth_vec);

        // Check structs with different numbers of fields are not equal
        auto struct_arr3 = get_sample_struct(1);
        std::fill(truth_vec.begin(), truth_vec.end(), false);  // NOLINT
        check_row_helper(struct_arr2, struct_arr3, truth_vec);
    });
    bodo::tests::test("test_map_array_comparison", [] {
        // map_arr1 = map_arr2 = [{"0":0}, {"1":1, "2":2, "3":3, "4":4}, {"5":5,
        // "6":6}, {}, {"7":7, "8":8, "9":9}]
        std::shared_ptr<array_info> map_arr1 = get_sample_map();
        std::shared_ptr<array_info> map_arr2 = get_sample_map();
        check_row_helper(map_arr1, map_arr2, {true, true, true, true, true});
        // Reorder the a key to test comparison is key order independent
        // map_arr1 = [{"0":0}, {"2":2, "1":1, "3":3, "4":4}, {"5":5, "6":6},
        // {}, {"7":7, "8":8, "9":9}]
        auto key_data1 = (char *)map_arr1->child_arrays[0]
                             ->child_arrays[0]
                             ->child_arrays[0]
                             ->data1();
        auto value_data1 = (int64_t *)map_arr1->child_arrays[0]
                               ->child_arrays[0]
                               ->child_arrays[1]
                               ->data1();
        key_data1[1] = '2';
        key_data1[2] = '1';
        value_data1[1] = 2;
        value_data1[2] = 1;
        check_row_helper(map_arr1, map_arr2, {true, true, true, true, true});
        // Change the offsets to make sure they're properly accounted for
        // map_arr1 = [{"0":0}, {"2":2, "1":1, "3":3}, {"4":4, "5":5, "6":6},
        // {}, {"7":7, "8":8, "9":9}]
        ((offset_t *)map_arr1->child_arrays[0]->buffers[0]->mutable_data())[2] =
            4;
        check_row_helper(map_arr1, map_arr2, {true, false, false, true, true});
        ((offset_t *)map_arr1->child_arrays[0]->buffers[0]->mutable_data())[2] =
            5;

        // Set a value to null and ensure it is not equal
        // map_arr1 = [{"0":0}, {"2":2, "1":1, "3":3}, {"5":5, "6":6}, null,
        // {"7":7, "8":8, "9":9}]
        map_arr1->set_null_bit(3, false);
        check_row_helper(map_arr1, map_arr2, {true, true, true, false, true});
    });
    bodo::tests::test("test_array_item_array_hashing", [] {
        // arr1 = [[0], [1,2,3,4], [5,6], [], [7,8,9]]
        // arr2 = [[0], [1,2,3,4], [5,6], [], [7,8,9]]
        auto arr1 = get_sample_array_item();
        auto arr2 = get_sample_array_item();
        uint32_t hashes1_buf[5];
        std::shared_ptr<uint32_t[]> hashes1 =
            std::shared_ptr<uint32_t[]>(hashes1_buf, [](uint32_t *) {});
        uint32_t hashes2_buf[5];
        std::shared_ptr<uint32_t[]> hashes2 =
            std::shared_ptr<uint32_t[]>(hashes2_buf, [](uint32_t *) {});
        hash_array(hashes1.get(), arr1, 5, 0, false, false);
        hash_array(hashes2.get(), arr2, 5, 0, false, false);
        bodo::tests::check(
            memcmp(hashes1.get(), hashes2.get(), 5 * sizeof(uint32_t)) == 0);

        // Change arr1 to have a different value
        // arr1 = [[1], [1,2,3,4], [5,6], [], [7,8,9]]
        getv<int64_t>(arr1->child_arrays[0], 0) = 1;
        hash_array(hashes1.get(), arr1, 5, 0, false, false);
        hash_array(hashes2.get(), arr2, 5, 0, false, false);
        check_hashes_helper(hashes1, hashes2, {false, true, true, true, true});
        getv<int64_t>(arr1->child_arrays[0], 0) = 0;

        // Change arr1 to have a different offset
        // arr1 = [[0], [1,2,3], [4, 5,6], [], [7,8,9]]
        ((offset_t *)arr1->buffers[0]->mutable_data())[2] = 4;
        hash_array(hashes1.get(), arr1, 5, 0, false, false);
        hash_array(hashes2.get(), arr2, 5, 0, false, false);
        check_hashes_helper(hashes1, hashes2, {true, false, false, true, true});
        ((offset_t *)arr1->buffers[0]->mutable_data())[2] = 5;

        // Reorder arr1[1]
        // arr1 = [[0], [4, 3,2,1], [5,6], [], [7,8,9]]
        ((offset_t *)arr1->buffers[0]->mutable_data())[2] = 5;
        getv<int64_t>(arr1->child_arrays[0], 1) = 4;
        getv<int64_t>(arr1->child_arrays[0], 2) = 3;
        getv<int64_t>(arr1->child_arrays[0], 3) = 2;
        getv<int64_t>(arr1->child_arrays[0], 4) = 1;
        hash_array(hashes1.get(), arr1, 5, 0, false, false);
        hash_array(hashes2.get(), arr2, 5, 0, false, false);
        check_hashes_helper(hashes1, hashes2, {true, false, true, true, true});
        getv<int64_t>(arr1->child_arrays[0], 1) = 1;
        getv<int64_t>(arr1->child_arrays[0], 2) = 2;
        getv<int64_t>(arr1->child_arrays[0], 3) = 3;
        getv<int64_t>(arr1->child_arrays[0], 4) = 4;

        // Set a value to null and ensure it is not equal
        // arr1 = [[0], [1,2,3,4], [5,6], null, [7,8,9]]
        arr1->set_null_bit(3, false);
        hash_array(hashes1.get(), arr1, 5, 0, false, false);
        hash_array(hashes2.get(), arr2, 5, 0, false, false);
        check_hashes_helper(hashes1, hashes2, {true, true, true, false, true});
        arr1->set_null_bit(3, true);

        // Check arrays compared from different start_row_offsets hash the same
        hash_array(hashes1.get(), arr1, 5, 0, false, false);
        hash_array(hashes2.get(), arr1, 4, 0, false, false, {}, 1);
        bodo::tests::check(hashes1[1] == hashes2[0]);
    });
    bodo::tests::test("test_struct_array_hashing", [] {
        // struct_arr1 = struct_arr2 =[{"a":0, "b":0}, {"a": 1, "b":1}, {"a":2,
        // "b":2}, {"a": 3, "b":3}, {"a":4, "b":4}, {"a":5, "b":5}, {"a":6,
        // "b":6}, {"a":7, "b":7}, {"a":8, "b":8}, {"a":9, "b": 9}]
        auto struct_arr1 = get_sample_struct(2);
        auto struct_arr2 = get_sample_struct(2);

        uint32_t hashes1_buf[10];
        std::shared_ptr<uint32_t[]> hashes1 =
            std::shared_ptr<uint32_t[]>(hashes1_buf, [](uint32_t *) {});
        uint32_t hashes2_buf[10];
        std::shared_ptr<uint32_t[]> hashes2 =
            std::shared_ptr<uint32_t[]>(hashes2_buf, [](uint32_t *) {});
        hash_array(hashes1.get(), struct_arr1, 10, 0, false, false);
        hash_array(hashes2.get(), struct_arr2, 10, 0, false, false);

        bodo::tests::check(
            memcmp(hashes1.get(), hashes2.get(), 10 * sizeof(uint32_t)) == 0);
        auto a_data1 = (int64_t *)struct_arr1->child_arrays[0]->data1();
        auto b_data1 = (int64_t *)struct_arr1->child_arrays[1]->data1();

        // Change one field in the first and second row
        // and both fields in the third row
        // struct_arr1 = [{"a":-1, "b":0}, {"a": 1, "b":-1}, {"a":0, "b":1} ...]
        a_data1[0] = -1;
        b_data1[1] = -1;
        a_data1[2] = 0;
        b_data1[2] = 1;
        hash_array(hashes1.get(), struct_arr1, 10, 0, false, false);
        hash_array(hashes2.get(), struct_arr2, 10, 0, false, false);
        check_hashes_helper(
            hashes1, hashes2,
            {false, false, false, true, true, true, true, true, true, true});
        a_data1[0] = 0;
        b_data1[1] = 1;
        a_data1[2] = 2;
        b_data1[2] = 2;

        // Set a value to null and ensure it is not equal
        // struct_arr1 = [{"a":-1, "b":0}, null, {"a":0, "b":1} ...]
        struct_arr1->set_null_bit(1, false);
        hash_array(hashes1.get(), struct_arr1, 10, 0, false, false);
        hash_array(hashes2.get(), struct_arr2, 10, 0, false, false);
        check_hashes_helper(
            hashes1, hashes2,
            {true, false, true, true, true, true, true, true, true, true});
        struct_arr1->set_null_bit(1, true);

        // Check structs with different numbers of fields hashes are different
        auto struct_arr3 = get_sample_struct(1);
        hash_array(hashes1.get(), struct_arr2, 10, 0, false, false);
        hash_array(hashes2.get(), struct_arr3, 10, 0, false, false);
        check_hashes_helper(hashes1, hashes2,
                            {false, false, false, false, false, false, false,
                             false, false, false});

        // Check structs compared from different start_row_offsets hash the same
        hash_array(hashes1.get(), struct_arr1, 10, 0, false, false);
        hash_array(hashes2.get(), struct_arr1, 9, 0, false, false, {}, 1);
        bodo::tests::check(hashes1[1] == hashes2[0]);
    });
    bodo::tests::test("test_map_array_hashing", [] {
        // Check that the hash of a map array is the same as the hash of the
        // same map array.
        auto map_arr = get_sample_map();
        auto map_arr2 = get_sample_map();
        uint32_t hashes1_buf[5];
        std::shared_ptr<uint32_t[]> hashes1 =
            std::shared_ptr<uint32_t[]>(hashes1_buf, [](uint32_t *) {});
        uint32_t hashes2_buf[5];
        std::shared_ptr<uint32_t[]> hashes2 =
            std::shared_ptr<uint32_t[]>(hashes2_buf, [](uint32_t *) {});
        hash_array(hashes1.get(), map_arr, 5, 0, false, false);
        hash_array(hashes2.get(), map_arr2, 5, 0, false, false);
        bodo::tests::check(
            memcmp(hashes1.get(), hashes2.get(), 5 * sizeof(uint32_t)) == 0);

        // Reorder the a key to test hashing is key order independent
        // map_arr = [[{'0':0}, {'2':2, '1':1}, {'3':3, '4':4, '5':5, '6':6},
        // {}, {'7':7, '8':8, '9':9}]]
        auto key_data1 = (char *)map_arr->child_arrays[0]
                             ->child_arrays[0]
                             ->child_arrays[0]
                             ->data1();
        auto value_data1 = (int64_t *)map_arr->child_arrays[0]
                               ->child_arrays[0]
                               ->child_arrays[1]
                               ->data1();
        key_data1[1] = '2';
        key_data1[2] = '1';
        value_data1[1] = 2;
        value_data1[2] = 1;

        hash_array(hashes1.get(), map_arr, 5, 0, false, false);
        hash_array(hashes2.get(), map_arr2, 5, 0, false, false);
        bodo::tests::check(
            memcmp(hashes1.get(), hashes2.get(), 5 * sizeof(uint32_t)) == 0);
        key_data1[1] = '1';
        key_data1[2] = '2';
        value_data1[1] = 1;
        value_data1[2] = 2;

        // Change a key and make sure the hashes are different
        // map_arr = [[{'1':0}, {'2':2, '1':1}, {'3':3, '4':4, '5':5, '6':6},
        // {}, {'7':7, '8':8, '9':9}]]
        key_data1[0] = '1';
        hash_array(hashes1.get(), map_arr, 5, 0, false, false);
        hash_array(hashes2.get(), map_arr2, 5, 0, false, false);
        check_hashes_helper(hashes1, hashes2, {false, true, true, true, true});
        key_data1[0] = '0';
        // Change a value and make sure the hashes are different
        // map_arr = [[{'0':1}, {'2':2, '1':1}, {'3':3, '4':4, '5':5, '6':6},
        // {}, {'7':7, '8':8, '9':9}]]
        value_data1[0] = 1;
        hash_array(hashes1.get(), map_arr, 5, 0, false, false);
        hash_array(hashes2.get(), map_arr2, 5, 0, false, false);
        check_hashes_helper(hashes1, hashes2, {false, true, true, true, true});
        value_data1[0] = 0;

        // Change the offsets to make sure they're properly accounted for
        // map_arr = [[{'0':0}, {'2':2, '1':1, '3':3}, {'4':4, '5':5, '6':6},
        // {}, {'7':7, '8':8, '9':9}]]
        ((offset_t *)map_arr->child_arrays[0]->buffers[0]->mutable_data())[2] =
            4;
        hash_array(hashes1.get(), map_arr, 5, 0, false, false);
        hash_array(hashes2.get(), map_arr2, 5, 0, false, false);
        check_hashes_helper(hashes1, hashes2, {true, false, false, true, true});
        ((offset_t *)map_arr->child_arrays[0]->buffers[0]->mutable_data())[2] =
            5;

        // Set a value to null and ensure it is not equal
        // map_arr = [[{'0':0}, {'2':2, '1':1, '3':3, '4':4}, {'5':5, '6':6},
        // null, {'7':7, '8':8, '9':9}]]
        map_arr->set_null_bit(3, false);
        hash_array(hashes1.get(), map_arr, 5, 0, false, false);
        hash_array(hashes2.get(), map_arr2, 5, 0, false, false);
        check_hashes_helper(hashes1, hashes2, {true, true, true, false, true});
        map_arr->set_null_bit(3, true);

        // Check that two values with different start offsets hash the same
        hash_array(hashes1.get(), map_arr, 5, 0, false, false);
        hash_array(hashes2.get(), map_arr2, 4, 0, false, false, {}, 1);
        bodo::tests::check(hashes1[1] == hashes2[0]);
    });
    bodo::tests::test("test_nested_is_na_equal", [] {
        auto arr1 = get_sample_array_item();
        auto arr2 = get_sample_array_item();
        // Set row 0 to null
        arr1->child_arrays[0]->set_null_bit(0, false);
        arr2->child_arrays[0]->set_null_bit(0, false);
        bodo::tests::check(TestEqualColumn(arr1, 0, arr2, 0, false) == false);
        bodo::tests::check(TestEqualColumn(arr1, 0, arr2, 0, true) == true);
    });
});
