#include "../libs/_stream_sort.h"
#include "./test.hpp"
#include "table_generator.hpp"

bodo::tests::suite external_sort_tests([] {
    bodo::tests::test("test_external_sort_empty", [] {
        std::vector<int64_t> vect_ascending{0};
        std::vector<int64_t> na_position{0};
        std::vector<int64_t> dead_keys;
        SortedChunkedTableBuilder builder(1, vect_ascending, na_position,
                                          dead_keys, 1);

        auto res = builder.Finalize();
        bodo::tests::check(res.size() == 0);
    });

    bodo::tests::test("test_external_sort_one_chunk", [] {
        std::shared_ptr<table_info> table = bodo::tests::cppToBodo(
            {"A"}, {false}, {}, std::vector<int64_t>{1, 5, 2, 4, 3});
        std::vector<int64_t> vect_ascending{0};
        std::vector<int64_t> na_position{0};
        std::vector<int64_t> dead_keys;
        SortedChunkedTableBuilder builder(1, vect_ascending, na_position,
                                          dead_keys, 3);

        builder.AppendChunk(table);
        auto res = builder.Finalize();
        bodo::tests::check(res.size() == 1);

        res[0].table->pin();
        auto arrow_arr = to_arrow(res[0].table->columns[0]);
        arrow::Int64Array* int_arr =
            static_cast<arrow::Int64Array*>(arrow_arr.get());
        bodo::tests::check(int_arr->Value(0) == 5);
        bodo::tests::check(int_arr->Value(1) == 4);
        bodo::tests::check(int_arr->Value(2) == 3);
        bodo::tests::check(int_arr->Value(3) == 2);
        bodo::tests::check(int_arr->Value(4) == 1);
    });

    bodo::tests::test("test_external_sort_two_chunks", [] {
        std::shared_ptr<table_info> table_0 = bodo::tests::cppToBodo(
            {"A"}, {false}, {}, std::vector<int64_t>{1, 5, 2, 4, 3});
        std::shared_ptr<table_info> table_1 = bodo::tests::cppToBodo(
            {"A"}, {false}, {}, std::vector<int64_t>{1, 5, 2, 4, 3});
        std::vector<int64_t> vect_ascending{0};
        std::vector<int64_t> na_position{0};
        std::vector<int64_t> dead_keys;
        SortedChunkedTableBuilder builder(1, vect_ascending, na_position,
                                          dead_keys, 5);

        builder.AppendChunk(table_0);
        builder.AppendChunk(table_1);
        auto res = builder.Finalize();
        bodo::tests::check(res.size() >= 2);

        std::vector<std::shared_ptr<array_info>> arrays;
        for (const auto& chunk : res) {
            chunk.table->pin();
            arrays.push_back(chunk.table->columns[0]);
        }
        auto bodo_arr = concat_arrays(arrays);
        auto arrow_arr = to_arrow(bodo_arr);
        arrow::Int64Array* int_arr =
            static_cast<arrow::Int64Array*>(arrow_arr.get());
        bodo::tests::check(int_arr->length() == 10);
        bodo::tests::check(int_arr->Value(0) == 5);
        bodo::tests::check(int_arr->Value(1) == 5);
        bodo::tests::check(int_arr->Value(2) == 4);
        bodo::tests::check(int_arr->Value(3) == 4);
        bodo::tests::check(int_arr->Value(4) == 3);
        bodo::tests::check(int_arr->Value(5) == 3);
        bodo::tests::check(int_arr->Value(6) == 2);
        bodo::tests::check(int_arr->Value(7) == 2);
        bodo::tests::check(int_arr->Value(8) == 1);
        bodo::tests::check(int_arr->Value(9) == 1);
    });

    bodo::tests::test("test_external_sort_three_chunks_asc", [] {
        std::shared_ptr<table_info> table_0 = bodo::tests::cppToBodo(
            {"A"}, {false}, {}, std::vector<int64_t>{1, 4, 7, 10});
        std::shared_ptr<table_info> table_1 = bodo::tests::cppToBodo(
            {"A"}, {false}, {}, std::vector<int64_t>{2, 5, 8, 11});
        std::shared_ptr<table_info> table_2 = bodo::tests::cppToBodo(
            {"A"}, {false}, {}, std::vector<int64_t>{3, 6, 9});

        std::vector<int64_t> vect_ascending{1};
        std::vector<int64_t> na_position{0};
        std::vector<int64_t> dead_keys;
        SortedChunkedTableBuilder builder(1, vect_ascending, na_position,
                                          dead_keys, 3);

        builder.AppendChunk(table_2);
        builder.AppendChunk(table_1);
        builder.AppendChunk(table_0);

        auto res = builder.Finalize();
        bodo::tests::check(res.size() >= 4);

        std::vector<std::shared_ptr<array_info>> arrays;
        for (const auto& chunk : res) {
            chunk.table->pin();
            arrays.push_back(chunk.table->columns[0]);
        }
        auto bodo_arr = concat_arrays(arrays);
        auto arrow_arr = to_arrow(bodo_arr);
        arrow::Int64Array* int_arr =
            static_cast<arrow::Int64Array*>(arrow_arr.get());
        bodo::tests::check(int_arr->length() == 11);

        bodo::tests::check(int_arr->Value(0) == 1);
        bodo::tests::check(int_arr->Value(1) == 2);
        bodo::tests::check(int_arr->Value(2) == 3);
        bodo::tests::check(int_arr->Value(3) == 4);
        bodo::tests::check(int_arr->Value(4) == 5);
        bodo::tests::check(int_arr->Value(5) == 6);
        bodo::tests::check(int_arr->Value(6) == 7);
        bodo::tests::check(int_arr->Value(7) == 8);
        bodo::tests::check(int_arr->Value(8) == 9);
        bodo::tests::check(int_arr->Value(9) == 10);
        bodo::tests::check(int_arr->Value(10) == 11);
    });
});
