#include <iostream>
#include <random>
#include "../libs/_distributed.h"
#include "../libs/_stream_sort.h"
#include "./test.hpp"
#include "table_generator.hpp"

static void unsort_vector(std::vector<int64_t> data) {
    // Randomly shuffle the input
    const int seed = 1234;
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, data.size() - 1);
    for (size_t i = 0; i < data.size(); i++) {
        // pick two indices and swap them
        int64_t idx0 = dis(gen);
        int64_t idx1 = dis(gen);
        std::swap(data[idx0], data[idx1]);
    }
}

bodo::tests::suite external_sort_tests([] {
    bodo::tests::test("test_external_sort_empty", [] {
        std::vector<int64_t> vect_ascending{0};
        std::vector<int64_t> na_position{0};
        std::vector<int64_t> dead_keys;
        // uint64_t pool_size = 0;
        SortedChunkedTableBuilder builder(1, vect_ascending, na_position,
                                          dead_keys, 2, 1);

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
                                          dead_keys, 2, 3);
        builder.UpdateChunkSize(3);
        builder.AppendChunk(table);
        builder.InitCTB(table->schema());
        auto res = builder.Finalize();
        bodo::tests::check(res.size() == 2);

        std::vector<std::shared_ptr<array_info>> arrays;
        for (const auto& chunk : res) {
            chunk.table->pin();
            arrays.push_back(chunk.table->columns[0]);
        }
        auto bodo_arr = concat_arrays(arrays);
        auto arrow_arr = to_arrow(bodo_arr);
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
                                          dead_keys, 2, 5);

        builder.AppendChunk(table_0);
        builder.UpdateChunkSize(5);
        builder.AppendChunk(table_1);
        builder.InitCTB(table_1->schema());
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
        std::vector<std::string> col_names = {"A", "B", "C"};
        std::vector<bool> nullable = {false, true, true};
        std::shared_ptr<table_info> table_0 = bodo::tests::cppToBodo(
            col_names, nullable, {}, std::vector<int64_t>{1, 4, 7, 10},
            std::vector<std::int64_t>{1, 2, 3, 4},
            std::vector<std::string>{"a", "b", "c", "d"});
        std::shared_ptr<table_info> table_1 = bodo::tests::cppToBodo(
            col_names, nullable, {}, std::vector<int64_t>{2, 5, 8, 11},
            std::vector<std::int64_t>{1, 2, 3, 4},
            std::vector<std::string>{"a", "b", "c", "d"});
        std::shared_ptr<table_info> table_2 = bodo::tests::cppToBodo(
            col_names, nullable, {}, std::vector<int64_t>{3, 6, 9},
            std::vector<std::int64_t>{1, 2, 3},
            std::vector<std::string>{"a", "b", "c"});

        std::vector<int64_t> vect_ascending{1};
        std::vector<int64_t> na_position{0};
        std::vector<int64_t> dead_keys;
        SortedChunkedTableBuilder builder(1, vect_ascending, na_position,
                                          dead_keys, 2, 3);

        builder.AppendChunk(table_2);
        builder.AppendChunk(table_1);
        builder.UpdateChunkSize(3);
        builder.AppendChunk(table_0);
        builder.InitCTB(table_0->schema());

        auto res = builder.Finalize();
        bodo::tests::check(res.size() >= 4);

        std::vector<std::shared_ptr<table_info>> tables;
        for (const auto& chunk : res) {
            chunk.table->pin();
            tables.push_back(chunk.table);
        }
        auto bodo_table = concat_tables(tables);

        auto a_arrow_arr = to_arrow(bodo_table->columns[0]);
        auto* a_arr = static_cast<arrow::Int64Array*>(a_arrow_arr.get());
        auto b_arrow_arr = to_arrow(bodo_table->columns[1]);
        auto* b_arr = static_cast<arrow::Int64Array*>(b_arrow_arr.get());
        auto c_arrow_arr = to_arrow(bodo_table->columns[2]);
        auto* c_arr = static_cast<arrow::LargeStringArray*>(c_arrow_arr.get());
        bodo::tests::check(a_arr->length() == 11);
        bodo::tests::check(b_arr->length() == 11);
        bodo::tests::check(c_arr->length() == 11);

        // Extract the idx'th element from each of the column arrays and pack
        // them into a tuple
        auto getRow = [&](int64_t idx) {
            return std::make_tuple(a_arr->Value(idx), b_arr->Value(idx),
                                   c_arr->Value(idx));
        };

        bodo::tests::check(getRow(0) == std::make_tuple(1, 1, "a"));
        bodo::tests::check(getRow(1) == std::make_tuple(2, 1, "a"));
        bodo::tests::check(getRow(2) == std::make_tuple(3, 1, "a"));
        bodo::tests::check(getRow(3) == std::make_tuple(4, 2, "b"));
        bodo::tests::check(getRow(4) == std::make_tuple(5, 2, "b"));
        bodo::tests::check(getRow(5) == std::make_tuple(6, 2, "b"));
        bodo::tests::check(getRow(6) == std::make_tuple(7, 3, "c"));
        bodo::tests::check(getRow(7) == std::make_tuple(8, 3, "c"));
        bodo::tests::check(getRow(8) == std::make_tuple(9, 3, "c"));
        bodo::tests::check(getRow(9) == std::make_tuple(10, 4, "d"));
        bodo::tests::check(getRow(10) == std::make_tuple(11, 4, "d"));
    });

    bodo::tests::test("test_sort_nulls", [] {
        std::vector<std::string> col_names = {"A", "B", "C"};
        std::vector<bool> nullable = {false, true, true};
        std::shared_ptr<table_info> table_0 = bodo::tests::cppToBodo(
            {"A"}, {true}, {}, std::vector<int64_t>{1, -1, 7, 10});
        std::shared_ptr<table_info> table_1 = bodo::tests::cppToBodo(
            {"A"}, {true}, {}, std::vector<int64_t>{2, 5, -1, 11});
        std::shared_ptr<table_info> table_2 = bodo::tests::cppToBodo(
            {"A"}, {true}, {}, std::vector<int64_t>{3, 6, 9});

        // Replace all -1's with NA
        for (const auto& table : {table_0, table_1, table_2}) {
            auto& col = table->columns[0];
            for (size_t i = 0; i < table->nrows(); i++) {
                if (((int64_t*)col->data1())[i] == -1) {
                    col->set_null_bit(i, false);
                }
            }
        }

        std::vector<int64_t> vect_ascending{1};
        std::vector<int64_t> dead_keys;
        // Test with nulls at the front
        {
            std::vector<int64_t> na_position{0};
            SortedChunkedTableBuilder builder(1, vect_ascending, na_position,
                                              dead_keys, 2, 3);

            builder.AppendChunk(table_2);
            builder.AppendChunk(table_1);
            builder.UpdateChunkSize(3);
            builder.AppendChunk(table_0);
            builder.InitCTB(table_0->schema());

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

            std::vector<std::optional<int64_t>> expected{
                std::nullopt, std::nullopt, 1, 2, 3, 5, 6, 7, 9, 10, 11};

            for (size_t i = 0; i < 11; i++) {
                std::optional<int64_t> actual = std::nullopt;
                if (int_arr->IsValid(i)) {
                    actual = int_arr->Value(i);
                }
                bodo::tests::check(actual == expected[i]);
            }
        }
        // Test with nulls at the back
        {
            std::vector<int64_t> na_position{1};
            SortedChunkedTableBuilder builder(1, vect_ascending, na_position,
                                              dead_keys, 2, 3);

            builder.AppendChunk(table_2);
            builder.AppendChunk(table_1);
            builder.UpdateChunkSize(3);
            builder.AppendChunk(table_0);
            builder.InitCTB(table_0->schema());

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

            std::vector<std::optional<int64_t>> expected{
                1, 2, 3, 5, 6, 7, 9, 10, 11, std::nullopt, std::nullopt};

            for (size_t i = 0; i < 11; i++) {
                std::optional<int64_t> actual = std::nullopt;
                if (int_arr->IsValid(i)) {
                    actual = int_arr->Value(i);
                }
                bodo::tests::check(actual == expected[i]);
            }
        }
    });

    bodo::tests::test("test_sampling", [] {
        int n_pes, myrank;
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        const int64_t n_elem = 1000;
        int64_t per_host_size = n_elem / n_pes;
        // Create a table with numbers from 1 to 1000
        std::vector<int64_t> data(per_host_size);
        std::iota(data.begin(), data.end(), myrank * per_host_size);
        unsort_vector(data);

        std::shared_ptr<table_info> table =
            bodo::tests::cppToBodo({"A"}, {false}, {}, std::move(data));

        std::vector<int64_t> vect_ascending{1};
        std::vector<int64_t> na_position{0};
        StreamSortState state(0, 1, std::move(vect_ascending),
                              std::move(na_position), table->schema(), true,
                              10);
        state.phase = StreamSortPhase::BUILD;
        state.ConsumeBatch(table, true, true);
        state.GlobalSort();
        state.phase = StreamSortPhase::PRODUCE_OUTPUT;

        auto res = state.GetParallelSortBounds();
        if (myrank == 0) {
            bodo::tests::check(static_cast<int>(res->nrows()) == (n_pes - 1));

            auto arrow_arr = to_arrow(res->columns[0]);
            arrow::Int64Array* int_arr =
                static_cast<arrow::Int64Array*>(arrow_arr.get());

            for (int64_t i = 0; i < (n_pes - 1); i++) {
                bodo::tests::check(int_arr->Value(i) ==
                                   ((i + 1) * per_host_size));
            }
        }
    });

    bodo::tests::test("test_parallel_all_local", [] {
        // This test doesn't actually require any communication because all the
        // data for a given rank is already present
        const size_t chunk_size = 10;

        int n_pes, myrank;
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        const size_t n_elems_per_host = 100 * chunk_size;
        std::vector<int64_t> data(n_elems_per_host);
        int64_t start_elem = myrank * n_elems_per_host;
        // Create a table with numbers from start_elem to (start_elem + 1000)

        std::iota(data.begin(), data.end(), start_elem);
        unsort_vector(data);

        std::shared_ptr<table_info> table =
            bodo::tests::cppToBodo({"A"}, {false}, {}, std::move(data));

        std::vector<int64_t> vect_ascending{1};
        std::vector<int64_t> na_position{0};
        StreamSortState state(0, 1, std::move(vect_ascending),
                              std::move(na_position), table->schema(), true,
                              chunk_size);
        state.phase = StreamSortPhase::BUILD;
        state.ConsumeBatch(table, true, true);

        std::shared_ptr<table_info> bounds = state.GetParallelSortBounds();
        auto bounds_arrow_arr = to_arrow(bounds->columns[0]);
        arrow::Int64Array* bounds_int_arr =
            static_cast<arrow::Int64Array*>(bounds_arrow_arr.get());
        int64_t actual_start_elem =
            (myrank == 0) ? 0 : (bounds_int_arr->Value(myrank - 1) + 1);

        state.GlobalSort();
        state.phase = StreamSortPhase::PRODUCE_OUTPUT;

        std::vector<std::shared_ptr<table_info>> tables;
        bool done = false;
        std::shared_ptr<table_info> out_table;
        do {
            std::tie(out_table, done) = state.GetOutput();
            tables.push_back(out_table);
        } while (!done);
        auto local_sorted_table = concat_tables(std::move(tables));
        auto arrow_arr = to_arrow(local_sorted_table->columns[0]);
        arrow::Int64Array* int_arr =
            static_cast<arrow::Int64Array*>(arrow_arr.get());

        for (int64_t i = 0; i < int_arr->length(); i++) {
            bodo::tests::check(int_arr->Value(i) ==
                               static_cast<int64_t>(actual_start_elem + i));
        }
    });

    bodo::tests::test("test_parallel", [] {
        const size_t chunk_size = 10;

        int n_pes, myrank;
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        const int64_t n_elems_per_host = 100 * chunk_size;
        const int64_t n_elem = n_pes * n_elems_per_host;
        std::vector<int64_t> global_data(n_elem);
        if (myrank == 0) {
            // Create an array with numbers from 0 to n_elem
            std::iota(global_data.begin(), global_data.end(), 0);
            unsort_vector(global_data);
        }
        // Broadcast data from rank0 to all hosts
        MPI_Bcast(global_data.data(), n_elem, get_MPI_typ<Bodo_CTypes::INT64>(),
                  0, MPI_COMM_WORLD);

        std::vector<int64_t> local_data(n_elems_per_host);
        int64_t start_elem = myrank * n_elems_per_host;
        for (int64_t i = start_elem; i < (start_elem + n_elems_per_host); i++) {
            local_data[i - start_elem] = global_data[i];
        }

        std::shared_ptr<table_info> table =
            bodo::tests::cppToBodo({"A"}, {false}, {}, std::move(local_data));

        std::vector<int64_t> vect_ascending{1};
        std::vector<int64_t> na_position{0};
        StreamSortState state(0, 1, std::move(vect_ascending),
                              std::move(na_position), table->schema(), true,
                              chunk_size);
        state.phase = StreamSortPhase::BUILD;
        state.ConsumeBatch(table, true, true);
        state.builder.chunk_size = chunk_size;

        state.GlobalSort();
        state.phase = StreamSortPhase::PRODUCE_OUTPUT;

        // Collect all output tables
        bool done = false;
        std::vector<std::shared_ptr<table_info>> tables;
        while (!done) {
            auto res = state.GetOutput();
            done = res.second;
            tables.push_back(res.first);
        }

        // Merge tables and get a single output array
        auto local_sorted_table = concat_tables(std::move(tables));
        auto arrow_arr = to_arrow(local_sorted_table->columns[0]);
        arrow::Int64Array* int_arr =
            static_cast<arrow::Int64Array*>(arrow_arr.get());

        int64_t max = int_arr->Value(int_arr->length() - 1);
        // Get the maximum element from every rank
        std::vector<int64_t> maximums(n_pes);
        MPI_Allgather(&max, 1, get_MPI_typ<Bodo_CTypes::INT64>(),
                      maximums.data(), 1, get_MPI_typ<Bodo_CTypes::INT64>(),
                      MPI_COMM_WORLD);

        // Check that element 0 is the max value on the previous rank + 1
        if (myrank == 0) {
            // no previous rank - the first element should be 0
            bodo::tests::check(int_arr->Value(0) == 0);
        } else {
            bodo::tests::check(int_arr->Value(0) == (maximums[myrank - 1] + 1));
        }

        // Check that all values are increasing by 1
        for (int64_t i = 1; i < int_arr->length(); i++) {
            bodo::tests::check(int_arr->Value(i) ==
                               (int_arr->Value(i - 1) + 1));
        }
    });
});
