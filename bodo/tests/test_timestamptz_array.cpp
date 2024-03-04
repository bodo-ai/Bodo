#include <sstream>
#include "../libs/_array_operations.h"
#include "../libs/_array_utils.h"
#include "../libs/_bodo_common.h"
#include "./test.hpp"

std::unique_ptr<table_info> make_timestamptz_arr() {
    std::shared_ptr<array_info> arr = alloc_array_top_level(
        5, 0, 0, bodo_array_type::arr_type_enum::TIMESTAMPTZ,
        Bodo_CTypes::CTypeEnum::TIMESTAMPTZ);

    int64_t* ts_buffer = (int64_t*)(arr->data1());
    int16_t* offset_buffer = (int16_t*)(arr->data2());

    // Index 0: UTC = 2024-7-4 18:45:31.250991000, offset = +02:30
    ts_buffer[0] = 1720118731250991000;
    offset_buffer[0] = 150;

    // Index 1: Null
    arr->set_null_bit(1, 0);

    // Index 2: UTC = 2024-3-14 00:00:00.000000000, offset = -01:00
    ts_buffer[2] = 1710374400000000000;
    offset_buffer[2] = -60;

    // Index 3: Null
    arr->set_null_bit(3, 0);

    // Index 4: UTC = 1999-12-31 23:59:59.999999250, offset = +11:01
    ts_buffer[4] = 946684799999999250;
    offset_buffer[4] = 665;

    std::unique_ptr<table_info> table = std::make_unique<table_info>();
    table->columns.push_back(arr);

    return table;
}

bodo::tests::suite timestamptz_array_tests([] {
    bodo::tests::test("test_timestamptz_array_allocation", [] {
        std::unique_ptr<array_info> arr = alloc_array_top_level(
            0, 0, 0, bodo_array_type::arr_type_enum::TIMESTAMPTZ,
            Bodo_CTypes::CTypeEnum::TIMESTAMPTZ);

        bodo::tests::check(arr->buffers[0] != nullptr);
        bodo::tests::check(arr->buffers[1] != nullptr);
        bodo::tests::check(arr->buffers[2] != nullptr);
    });

    bodo::tests::test("test_timestamptz_array_to_string", [] {
        // Test to make sure that a timestamptz array is converted
        // to a string correctly for debugging purposes.
        std::unique_ptr<table_info> table = make_timestamptz_arr();
        std::stringstream ss;
        DEBUG_PrintTable(ss, table.release());
        bodo::tests::check(ss.str() ==
                           "Column 0 : arr_type=TIMESTAMPTZ dtype=TIMESTAMPTZ\n"
                           "nCol=1 List of number of rows: 5\n"
                           "0 : 2024-07-04 18:45:31.250991000 +02:30\n"
                           "1 : NA                                  \n"
                           "2 : 2024-03-14 00:00:00.000000000 -01:00\n"
                           "3 : NA                                  \n"
                           "4 : 1999-12-31 23:59:59.999999250 +11:05\n");
    });

    bodo::tests::test("test_timestamptz_array_sort", [] {
        // Test to make sure that a timestamptz array orders
        // values correctly when sorted.
        std::shared_ptr<table_info> table = make_timestamptz_arr();
        int64_t asc = 1;
        int64_t napos = 1;
        std::shared_ptr<table_info> sorted_table =
            sort_values_table_local(table, 1, &asc, &napos, nullptr, false);
        std::stringstream ss;
        DEBUG_PrintTable(ss, sorted_table);
        bodo::tests::check(ss.str() ==
                           "Column 0 : arr_type=TIMESTAMPTZ dtype=TIMESTAMPTZ\n"
                           "nCol=1 List of number of rows: 5\n"
                           "0 : 1999-12-31 23:59:59.999999250 +11:05\n"
                           "1 : 2024-03-14 00:00:00.000000000 -01:00\n"
                           "2 : 2024-07-04 18:45:31.250991000 +02:30\n"
                           "3 : NA                                  \n"
                           "4 : NA                                  \n");
    });
});
