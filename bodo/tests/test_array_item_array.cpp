#include "../libs/_array_utils.h"
#include "../libs/_bodo_common.h"
#include "./test.hpp"

static bodo::tests::suite tests([] {
    bodo::tests::test("test_array_item_construction", [] {
        // Create the internal array containing numbers from 0 to 10
        std::shared_ptr<array_info> inner_arr =
            alloc_numpy(10, Bodo_CTypes::INT64);
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
});
