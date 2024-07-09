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

    bodo::tests::test("test_sampling", [] {
        std::vector<int64_t> vect_ascending{1};
        std::vector<int64_t> na_position{0};
        StreamSortState state(1, std::move(vect_ascending),
                              std::move(na_position), 10);

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
        state.phase = StreamSortPhase::PRE_BUILD;
        state.consume_batch(table, true, true);

        auto res = state.get_parallel_sort_bounds();
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
        std::vector<int64_t> vect_ascending{1};
        std::vector<int64_t> na_position{0};
        const size_t chunk_size = 10;
        StreamSortState state(1, std::move(vect_ascending),
                              std::move(na_position), chunk_size);

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
        state.phase = StreamSortPhase::PRE_BUILD;
        state.consume_batch(table, true, true);
        state.global_sort();

        bool done = false;
        int64_t chunk_id = 0;
        while (!done) {
            auto res = state.get_output();
            done = res.second;

            auto output = res.first;
            auto arrow_arr = to_arrow(output->columns[0]);
            arrow::Int64Array* int_arr =
                static_cast<arrow::Int64Array*>(arrow_arr.get());

            for (int64_t i = 0; i < (n_pes - 1); i++) {
                bodo::tests::check(int_arr->Value(i) ==
                                   static_cast<int64_t>(
                                       start_elem + chunk_id * chunk_size + i));
            }
        }
    });

    bodo::tests::test("test_parallel", [] {
        std::vector<int64_t> vect_ascending{1};
        std::vector<int64_t> na_position{0};
        const size_t chunk_size = 10;
        StreamSortState state(1, std::move(vect_ascending),
                              std::move(na_position), chunk_size);

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
        state.phase = StreamSortPhase::PRE_BUILD;
        state.consume_batch(table, true, true);
        state.global_sort();

        // Collect all output tables
        bool done = false;
        std::vector<std::shared_ptr<table_info>> tables;
        while (!done) {
            auto res = state.get_output();
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
