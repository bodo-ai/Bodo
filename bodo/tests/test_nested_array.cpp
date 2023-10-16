#include "../libs/_array_utils.h"
#include "../libs/_bodo_common.h"
#include "./test.hpp"

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
    offset_t* offset_buffer =
        (offset_t*)(arr->buffers[0]->mutable_data() + arr->offset);
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
        // Check that each row of array when converted to a string
        // is as expected.
        std::shared_ptr<arrow::Array> as_arrow = to_arrow(arr);
        std::string row_0 = "";
        std::string row_1 = "";
        std::string row_2 = "";
        std::string row_3 = "";
        std::string row_4 = "";
        DEBUG_append_to_out_array(as_arrow, 0, 1, row_0);
        DEBUG_append_to_out_array(as_arrow, 1, 2, row_1);
        DEBUG_append_to_out_array(as_arrow, 2, 3, row_2);
        DEBUG_append_to_out_array(as_arrow, 3, 4, row_3);
        DEBUG_append_to_out_array(as_arrow, 4, 5, row_4);
        bodo::tests::check(row_0 == "[[0]]");
        bodo::tests::check(row_1 == "[[1,2,3,4]]");
        bodo::tests::check(row_2 == "[[5,6]]");
        bodo::tests::check(row_3 == "[[]]");
        bodo::tests::check(row_4 == "[[7,8,9]]");
    });

    bodo::tests::test("test_alloc_array_like", [] {
        // ARRAY_ITEM
        std::shared_ptr<array_info> arr =
            alloc_array_like(get_sample_array_item());
        bodo::tests::check(arr->length == 0);
        bodo::tests::check(
            ((offset_t*)arr->data1<bodo_array_type::ARRAY_ITEM>())[0] == 0);
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
        } catch (std::runtime_error& ex) {
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
