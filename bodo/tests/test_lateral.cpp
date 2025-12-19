#include <fmt/core.h>
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
    std::shared_ptr<array_info> inner_arr =
        alloc_nullable_array_no_nulls(10, Bodo_CTypes::INT64);
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
        alloc_nullable_array_no_nulls(5, Bodo_CTypes::INT32);
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
    bodo::tests::test("test_lateral_flatten_array_no_index", [] {
        std::unique_ptr<table_info> in_table = make_array_item_testing_table();
        std::stringstream ss;
        int64_t n_rows = 0;

        // Call lateral flatten without the index but with the value
        std::unique_ptr<table_info> out_table = lateral_flatten_array(
            in_table, &n_rows, false, false, false, false, true, false, false);

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
                           "Column 0 : arr_type=NULLABLE dtype=INT64\n"
                           "Column 1 : arr_type=NUMPY dtype=INT64\n"
                           "Column 2 : arr_type=STRING dtype=STRING\n"
                           "Column 3 : arr_type=DICT dtype=STRING\n"
                           "nCol=4 List of number of rows: 10 10 10 10\n"
                           "0 : 0 0  ABC Alphabet\n"
                           "1 : 1 1  DE  NA      \n"
                           "2 : 2 1  DE  NA      \n"
                           "3 : 3 1  DE  NA      \n"
                           "4 : 4 1  DE  NA      \n"
                           "5 : 5 4  NA  Soup    \n"
                           "6 : 6 4  NA  Soup    \n"
                           "7 : 7 16 HIJ Alphabet\n"
                           "8 : 8 16 HIJ Alphabet\n"
                           "9 : 9 16 HIJ Alphabet\n");
    });

    bodo::tests::test("test_lateral_flatten_array_with_index", [] {
        std::unique_ptr<table_info> in_table = make_array_item_testing_table();
        std::stringstream ss;
        int64_t n_rows = 0;

        // Call lateral flatten with the index and the value
        std::unique_ptr<table_info> out_table = lateral_flatten_array(
            in_table, &n_rows, false, false, false, true, true, false, false);

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
                           "Column 0 : arr_type=NULLABLE dtype=INT64\n"
                           "Column 1 : arr_type=NULLABLE dtype=INT64\n"
                           "Column 2 : arr_type=NUMPY dtype=INT64\n"
                           "Column 3 : arr_type=STRING dtype=STRING\n"
                           "Column 4 : arr_type=DICT dtype=STRING\n"
                           "nCol=5 List of number of rows: 10 10 10 10 10\n"
                           "0 : 0 0 0  ABC Alphabet\n"
                           "1 : 0 1 1  DE  NA      \n"
                           "2 : 1 2 1  DE  NA      \n"
                           "3 : 2 3 1  DE  NA      \n"
                           "4 : 3 4 1  DE  NA      \n"
                           "5 : 0 5 4  NA  Soup    \n"
                           "6 : 1 6 4  NA  Soup    \n"
                           "7 : 0 7 16 HIJ Alphabet\n"
                           "8 : 1 8 16 HIJ Alphabet\n"
                           "9 : 2 9 16 HIJ Alphabet\n");
    });

    bodo::tests::test("test_lateral_flatten_array_outer", [] {
        std::unique_ptr<table_info> in_table = make_array_item_testing_table();
        std::stringstream ss;
        int64_t n_rows = 0;

        // Call lateral flatten with the index, value, this, & the outer
        // parameter
        std::unique_ptr<table_info> out_table = lateral_flatten_array(
            in_table, &n_rows, false, false, false, true, true, true, true);

        // Check to make sure the lengths match.
        bodo::tests::check(out_table->columns.size() == 6);
        bodo::tests::check(n_rows == 11);
        bodo::tests::check(n_rows == (int64_t)(out_table->columns[0]->length));
        bodo::tests::check(n_rows == (int64_t)(out_table->columns[1]->length));
        bodo::tests::check(n_rows == (int64_t)(out_table->columns[2]->length));
        bodo::tests::check(n_rows == (int64_t)(out_table->columns[3]->length));
        bodo::tests::check(n_rows == (int64_t)(out_table->columns[4]->length));
        bodo::tests::check(n_rows == (int64_t)(out_table->columns[5]->length));

        // Dumping the table to a string to ensure it matches.
        DEBUG_PrintTable(ss, out_table.release());
        bodo::tests::check(ss.str() ==
                           "Column 0 : arr_type=NULLABLE dtype=INT64\n"
                           "Column 1 : arr_type=NULLABLE dtype=INT64\n"
                           "Column 2 : arr_type=ARRAY_ITEM dtype=LIST\n"
                           "Column 3 : arr_type=NUMPY dtype=INT64\n"
                           "Column 4 : arr_type=STRING dtype=STRING\n"
                           "Column 5 : arr_type=DICT dtype=STRING\n"
                           "nCol=6 List of number of rows: 11 11 11 11 11 11\n"
                           "0 : 0  0  [[0]]       0  ABC Alphabet\n"
                           "1 : 0  1  [[1,2,3,4]] 1  DE  NA      \n"
                           "2 : 1  2  [[1,2,3,4]] 1  DE  NA      \n"
                           "3 : 2  3  [[1,2,3,4]] 1  DE  NA      \n"
                           "4 : 3  4  [[1,2,3,4]] 1  DE  NA      \n"
                           "5 : 0  5  [[5,6]]     4  NA  Soup    \n"
                           "6 : 1  6  [[5,6]]     4  NA  Soup    \n"
                           "7 : NA NA [[]]        9  FG  Alphabet\n"
                           "8 : 0  7  [[7,8,9]]   16 HIJ Alphabet\n"
                           "9 : 1  8  [[7,8,9]]   16 HIJ Alphabet\n"
                           "10 : 2  9  [[7,8,9]]   16 HIJ Alphabet\n");
    });
});
