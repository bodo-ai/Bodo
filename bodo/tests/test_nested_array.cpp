#include <iostream>
#include <sstream>
#include "../libs/_array_utils.h"
#include "../libs/_bodo_common.h"
#include "../libs/_lateral.h"
#include "./test.hpp"

// Create the table used for testing in the test functions. The table
// contains a numpy array, an array_item array of integers, a string array,
// and a dictionary encoded array.
std::unique_ptr<table_info> make_array_item_testing_table() {
    // Create an numpy array that has 5 integers
    std::shared_ptr<array_info> numpy_arr = alloc_numpy(5, Bodo_CTypes::INT64);
    for (int i = 0; i < 5; i++) {
        getv<int64_t>(numpy_arr, i) = i * i;
    }

    // Create the internal array containing numbers from 0 to 9
    std::shared_ptr<array_info> inner_arr = alloc_numpy(10, Bodo_CTypes::INT64);
    for (int64_t i = 0; i < 10; i++) {
        getv<int64_t>(inner_arr, i) = i;
    }

    // Create the array item array that has 5 sub-arrays
    std::shared_ptr<array_info> array_item_arr = alloc_array_item(5, inner_arr);
    // Populate the offsets to indicate which of the 10 elements from
    // the original array belong to which of the 5 sub-arrays.
    offset_t *offset_buffer =
        (offset_t *)(array_item_arr->buffers[0]->mutable_data() +
                     array_item_arr->offset);
    offset_buffer[0] = 0;
    offset_buffer[1] = 1;
    offset_buffer[2] = 5;
    offset_buffer[3] = 7;
    offset_buffer[4] = 7;
    offset_buffer[5] = 10;

    // Create a string array
    std::shared_ptr<array_info> string_arr =
        alloc_string_array(Bodo_CTypes::STRING, 5, 10);
    string_arr->set_null_bit(2, false);
    char *str_data = string_arr->data1();
    std::string("ABCDEFGHIJ").copy(str_data, 10);
    offset_t *str_offsets = (offset_t *)(string_arr->data2());
    str_offsets[0] = 0;
    str_offsets[1] = 3;
    str_offsets[2] = 5;
    str_offsets[3] = 5;
    str_offsets[4] = 7;
    str_offsets[5] = 10;

    // Create a dictionary encoded array
    std::shared_ptr<array_info> dict_inner_arr =
        alloc_string_array(Bodo_CTypes::STRING, 2, 12);
    char *dict_inner_data = dict_inner_arr->data1();
    std::string("AlphabetSoup").copy(dict_inner_data, 12);
    offset_t *dict_inner_offsets = (offset_t *)(dict_inner_arr->data2());
    dict_inner_offsets[0] = 0;
    dict_inner_offsets[1] = 8;
    dict_inner_offsets[2] = 12;
    std::shared_ptr<array_info> dict_indices =
        alloc_nullable_array_no_nulls(5, Bodo_CTypes::INT32, 0);
    getv<int32_t>(dict_indices, 0) = 0;
    dict_indices->set_null_bit(1, false);
    getv<int32_t>(dict_indices, 2) = 1;
    getv<int32_t>(dict_indices, 3) = 0;
    getv<int32_t>(dict_indices, 4) = 0;
    std::shared_ptr<array_info> dict_arr =
        create_dict_string_array(dict_inner_arr, dict_indices);

    std::unique_ptr<table_info> table = std::make_unique<table_info>();
    table->columns.push_back(array_item_arr);
    table->columns.push_back(numpy_arr);
    table->columns.push_back(string_arr);
    table->columns.push_back(dict_arr);
    return table;
}

std::shared_ptr<array_info> get_sample_array_item() {
    // Create the internal array containing numbers from 0 to 9
    std::shared_ptr<array_info> inner_arr = alloc_numpy(10, Bodo_CTypes::INT64);
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

std::shared_ptr<array_info> get_sample_struct() {
    // Create the child array containing numbers from 0 to 9
    std::shared_ptr<array_info> child_array =
        alloc_numpy(10, Bodo_CTypes::INT64);
    for (int64_t i = 0; i < 10; i++) {
        getv<int64_t>(child_array, i) = i;
    }
    // Create the array item array with one child array
    return alloc_struct(10,
                        std::vector<std::shared_ptr<array_info>>{child_array});
}

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

    bodo::tests::test("test_lateral_array_flatten_no_index", [] {
        std::unique_ptr<table_info> in_table = make_array_item_testing_table();
        std::stringstream ss;
        int64_t n_rows = 0;

        // Call lateral flatten without the index but with the value
        std::unique_ptr<table_info> out_table = lateral_flatten(
            in_table, &n_rows, false, false, false, false, true, false);

        // Check to make sure the lengths match.
        bodo::tests::check(out_table->columns.size() == 4);
        bodo::tests::check(n_rows == 10);
        bodo::tests::check(n_rows == (int64_t)(out_table->columns[0]->length));
        bodo::tests::check(n_rows == (int64_t)(out_table->columns[1]->length));
        bodo::tests::check(n_rows == (int64_t)(out_table->columns[2]->length));
        bodo::tests::check(n_rows == (int64_t)(out_table->columns[3]->length));

        // Dumping the table to a string to ensure it matches.
        DEBUG_PrintTable(ss, out_table.release());
        bodo::tests::check(ss.str() ==
                           "Column 0 : arr_type=NUMPY dtype=INT64\n"
                           "Column 1 : arr_type=STRING dtype=STRING\n"
                           "Column 2 : arr_type=DICT dtype=STRING\n"
                           "Column 3 : arr_type=NUMPY dtype=INT64\n"
                           "nCol=4 List of number of rows: 10 10 10 10\n"
                           "0 : 0  ABC Alphabet 0\n"
                           "1 : 1  DE  NA       1\n"
                           "2 : 1  DE  NA       2\n"
                           "3 : 1  DE  NA       3\n"
                           "4 : 1  DE  NA       4\n"
                           "5 : 4  NA  Soup     5\n"
                           "6 : 4  NA  Soup     6\n"
                           "7 : 16 HIJ Alphabet 7\n"
                           "8 : 16 HIJ Alphabet 8\n"
                           "9 : 16 HIJ Alphabet 9\n");
    });

    bodo::tests::test("test_lateral_array_flatten_with_index", [] {
        std::unique_ptr<table_info> in_table = make_array_item_testing_table();
        std::stringstream ss;
        int64_t n_rows = 0;

        // Call lateral flatten with the index and the value
        std::unique_ptr<table_info> out_table = lateral_flatten(
            in_table, &n_rows, false, false, false, true, true, false);

        // Check to make sure the lengths match.
        bodo::tests::check(out_table->columns.size() == 5);
        bodo::tests::check(n_rows == 10);
        bodo::tests::check(n_rows == (int64_t)(out_table->columns[0]->length));
        bodo::tests::check(n_rows == (int64_t)(out_table->columns[1]->length));
        bodo::tests::check(n_rows == (int64_t)(out_table->columns[2]->length));
        bodo::tests::check(n_rows == (int64_t)(out_table->columns[3]->length));
        bodo::tests::check(n_rows == (int64_t)(out_table->columns[4]->length));

        // Dumping the table to a string to ensure it matches.
        DEBUG_PrintTable(ss, out_table.release());
        bodo::tests::check(ss.str() ==
                           "Column 0 : arr_type=NUMPY dtype=INT64\n"
                           "Column 1 : arr_type=STRING dtype=STRING\n"
                           "Column 2 : arr_type=DICT dtype=STRING\n"
                           "Column 3 : arr_type=NUMPY dtype=INT64\n"
                           "Column 4 : arr_type=NUMPY dtype=INT64\n"
                           "nCol=5 List of number of rows: 10 10 10 10 10\n"
                           "0 : 0  ABC Alphabet 0 0\n"
                           "1 : 1  DE  NA       0 1\n"
                           "2 : 1  DE  NA       1 2\n"
                           "3 : 1  DE  NA       2 3\n"
                           "4 : 1  DE  NA       3 4\n"
                           "5 : 4  NA  Soup     0 5\n"
                           "6 : 4  NA  Soup     1 6\n"
                           "7 : 16 HIJ Alphabet 0 7\n"
                           "8 : 16 HIJ Alphabet 1 8\n"
                           "9 : 16 HIJ Alphabet 2 9\n");
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
                           bodo_array_type::NUMPY);
        bodo::tests::check(arr->child_arrays[0]->length == 0);
        // STRUCT
        arr = alloc_array_like(get_sample_struct());
        bodo::tests::check(arr->length == 0);
        bodo::tests::check(arr->arr_type == bodo_array_type::STRUCT);
        bodo::tests::check(arr->dtype == Bodo_CTypes::STRUCT);
        bodo::tests::check(arr->child_arrays.size() == 1);
        bodo::tests::check(arr->child_arrays[0]->arr_type ==
                           bodo_array_type::NUMPY);
        bodo::tests::check(arr->child_arrays[0]->length == 0);
    });

    bodo::tests::test("test_get_col_idx_map", [] {
        std::vector<int8_t> arr_array_types(sample_array_types);
        bodo::tests::check(get_col_idx_map(arr_array_types) ==
                           std::vector<size_t>{0, 1, 4, 6, 11, 12});
        bodo::tests::check(get_col_idx_map(arr_array_types, 8, 11) ==
                           std::vector<size_t>{8, 9});
        arr_array_types.resize(13);
        bool exception_caught = false;
        try {
            get_col_idx_map(arr_array_types);
        } catch (std::runtime_error &ex) {
            bodo::tests::check(std::string(ex.what()) ==
                               "The last array type cannot be ARRAY_ITEM: "
                               "inner array type needs to be provided!");
            exception_caught = true;
        }
        bodo::tests::check(exception_caught);
    });

    bodo::tests::test("test_empty_col_idx_map", [] {
        std::vector<int8_t> arr_array_types;
        bodo::tests::check(get_col_idx_map(arr_array_types) ==
                           std::vector<size_t>());
    });
});
