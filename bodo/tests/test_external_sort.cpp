#include <iostream>
#include <random>
#include "../libs/_distributed.h"
#include "../libs/_shuffle.h"
#include "../libs/_stream_sort.h"
#include "./test.hpp"
#include "table_generator.hpp"

static void unsort_vector(std::vector<int64_t>& data) {
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
        auto schema = std::make_shared<bodo::Schema>();
        // uint64_t pool_size = 0;
        SortedChunkedTableBuilder builder(schema, {}, 1, vect_ascending,
                                          na_position, dead_keys, 2, 1);

        auto res = builder.Finalize();
        bodo::tests::check(res.size() == 0);
    });

    bodo::tests::test("test_external_sort_one_chunk", [] {
        std::shared_ptr<table_info> table = bodo::tests::cppToBodo(
            {"A"}, {false}, {}, std::vector<int64_t>{1, 5, 2, 4, 3});
        std::vector<int64_t> vect_ascending{0};
        std::vector<int64_t> na_position{0};
        std::vector<int64_t> dead_keys;
        std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders = {
            nullptr};
        SortedChunkedTableBuilder builder(table->schema(), dict_builders, 1,
                                          vect_ascending, na_position,
                                          dead_keys, 2, 3);
        builder.UpdateChunkSize(3);
        builder.AppendChunk(table);
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

    bodo::tests::test("test_external_sort_one_chunk_limitoffset", [] {
        std::shared_ptr<table_info> table = bodo::tests::cppToBodo(
            {"A"}, {false}, {}, std::vector<int64_t>{1, 5, 2, 4, 3});
        std::vector<int64_t> vect_ascending{0};
        std::vector<int64_t> na_position{0};
        std::vector<int64_t> dead_keys;
        std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders = {
            nullptr};
        SortedChunkedTableBuilder builder(table->schema(), dict_builders, 1,
                                          vect_ascending, na_position,
                                          dead_keys, 2, 3);
        builder.UpdateChunkSize(3);
        builder.AppendChunk(table);
        auto res =
            builder.Finalize(std::make_optional<SortLimits>(SortLimits(4, 1)));
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
        bodo::tests::check(int_arr->Value(0) == 4);
        bodo::tests::check(int_arr->Value(1) == 3);
        bodo::tests::check(int_arr->Value(2) == 2);
        bodo::tests::check(int_arr->Value(3) == 1);
    });

    bodo::tests::test("test_external_sort_two_chunks", [] {
        std::shared_ptr<table_info> table_0 = bodo::tests::cppToBodo(
            {"A"}, {false}, {}, std::vector<int64_t>{1, 5, 2, 4, 3});
        std::shared_ptr<table_info> table_1 = bodo::tests::cppToBodo(
            {"A"}, {false}, {}, std::vector<int64_t>{1, 5, 2, 4, 3});
        std::vector<int64_t> vect_ascending{0};
        std::vector<int64_t> na_position{0};
        std::vector<int64_t> dead_keys;
        std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders = {
            nullptr};
        SortedChunkedTableBuilder builder(table_0->schema(), dict_builders, 1,
                                          vect_ascending, na_position,
                                          dead_keys, 2, 5);

        builder.AppendChunk(table_0);
        builder.UpdateChunkSize(5);
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

    bodo::tests::test("test_external_sort_two_chunks_limitoffset", [] {
        std::shared_ptr<table_info> table_0 = bodo::tests::cppToBodo(
            {"A"}, {false}, {}, std::vector<int64_t>{1, 5, 2, 4, 3});
        std::shared_ptr<table_info> table_1 = bodo::tests::cppToBodo(
            {"A"}, {false}, {}, std::vector<int64_t>{1, 5, 2, 4, 3});
        std::vector<int64_t> vect_ascending{0};
        std::vector<int64_t> na_position{0};
        std::vector<int64_t> dead_keys;
        std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders = {
            nullptr};
        SortedChunkedTableBuilder builder(table_0->schema(), dict_builders, 1,
                                          vect_ascending, na_position,
                                          dead_keys, 2, 5);

        builder.AppendChunk(table_0);
        builder.UpdateChunkSize(5);
        builder.AppendChunk(table_1);
        auto res =
            builder.Finalize(std::make_optional<SortLimits>(SortLimits(6, 2)));
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
        bodo::tests::check(int_arr->length() == 6);
        bodo::tests::check(int_arr->Value(0) == 4);
        bodo::tests::check(int_arr->Value(1) == 4);
        bodo::tests::check(int_arr->Value(2) == 3);
        bodo::tests::check(int_arr->Value(3) == 3);
        bodo::tests::check(int_arr->Value(4) == 2);
        bodo::tests::check(int_arr->Value(5) == 2);
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
        std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders = {
            nullptr, nullptr, nullptr};
        SortedChunkedTableBuilder builder(table_0->schema(), dict_builders, 1,
                                          vect_ascending, na_position,
                                          dead_keys, 2, 3);

        builder.UpdateChunkSize(3);
        builder.AppendChunk(table_2);
        builder.AppendChunk(table_1);
        builder.AppendChunk(table_0);

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

    bodo::tests::test("test_external_sort_three_chunks_asc_limitoffset", [] {
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
        std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders = {
            nullptr, nullptr, nullptr};
        SortedChunkedTableBuilder builder(table_0->schema(), dict_builders, 1,
                                          vect_ascending, na_position,
                                          dead_keys, 2, 3);

        builder.AppendChunk(table_2);
        builder.AppendChunk(table_1);
        builder.UpdateChunkSize(3);
        builder.AppendChunk(table_0);

        auto res =
            builder.Finalize(std::make_optional<SortLimits>(SortLimits(3, 7)));
        bodo::tests::check(res.size() >= 1);

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
        bodo::tests::check(a_arr->length() == 3);
        bodo::tests::check(b_arr->length() == 3);
        bodo::tests::check(c_arr->length() == 3);

        // Extract the idx'th element from each of the column arrays and pack
        // // them into a tuple
        auto getRow = [&](int64_t idx) {
            return std::make_tuple(a_arr->Value(idx), b_arr->Value(idx),
                                   c_arr->Value(idx));
        };

        bodo::tests::check(getRow(0) == std::make_tuple(8, 3, "c"));
        bodo::tests::check(getRow(1) == std::make_tuple(9, 3, "c"));
        bodo::tests::check(getRow(2) == std::make_tuple(10, 4, "d"));
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
            std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders = {
                nullptr};
            SortedChunkedTableBuilder builder(table_0->schema(), dict_builders,
                                              1, vect_ascending, na_position,
                                              dead_keys, 2, 3);

            builder.AppendChunk(table_2);
            builder.AppendChunk(table_1);
            builder.UpdateChunkSize(3);
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
            std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders = {
                nullptr};
            SortedChunkedTableBuilder builder(table_0->schema(), dict_builders,
                                              1, vect_ascending, na_position,
                                              dead_keys, 2, 3);

            builder.AppendChunk(table_2);
            builder.AppendChunk(table_1);
            builder.UpdateChunkSize(3);
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
                              std::move(na_position), table->schema(), true, -1,
                              -1, 10);
        state.ConsumeBatch(table, true, true);
        state.FinalizeBuild();

        auto res = state.bounds_;
        if (myrank == 0) {
            bodo::tests::check(static_cast<int>(res->nrows()) == (n_pes - 1));

            auto arrow_arr = to_arrow(res->columns[0]);
            arrow::Int64Array* int_arr =
                static_cast<arrow::Int64Array*>(arrow_arr.get());

            // Allow imbalance of 10%
            int64_t error = std::max((per_host_size + 9) / 10, (int64_t)5);
            auto in_bound = [&](int64_t diff) -> bool {
                return std::abs(diff - per_host_size) <= error;
            };

            for (int64_t i = 0; i < (n_pes - 1); i++) {
                if (i == 0)
                    bodo::tests::check(in_bound(int_arr->Value(i)));
                else
                    bodo::tests::check(
                        in_bound(int_arr->Value(i) - int_arr->Value(i - 1)));
                if (i == n_pes - 2)
                    bodo::tests::check(
                        in_bound(per_host_size * n_pes - int_arr->Value(i)));
            }
        }
    });

    bodo::tests::test("test_sampling_hard", [] {
        int n_pes, myrank;
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        const int64_t n_elem = 100000;
        int64_t per_host_size = n_elem / n_pes;
        // Create a table with numbers from 1 to 1000
        std::vector<int64_t> all_data(n_elem);
        std::iota(all_data.begin(), all_data.end(), 0);
        unsort_vector(all_data);
        std::vector<int64_t> data{};
        for (int i = 0; i < per_host_size; i++)
            data.push_back(all_data[i + myrank * per_host_size]);

        std::shared_ptr<table_info> table =
            bodo::tests::cppToBodo({"A"}, {false}, {}, std::move(data));

        std::vector<int64_t> vect_ascending{1};
        std::vector<int64_t> na_position{0};
        StreamSortState state(0, 1, std::move(vect_ascending),
                              std::move(na_position), table->schema(), true, -1,
                              -1, 10);
        state.ConsumeBatch(table, true, true);
        state.builder.FinalizeActiveChunk();
        auto res = state.GetParallelSortBounds(state.builder.chunks);
        if (myrank == 0) {
            bodo::tests::check(static_cast<int>(res->nrows()) == (n_pes - 1));

            auto arrow_arr = to_arrow(res->columns[0]);
            arrow::Int64Array* int_arr =
                static_cast<arrow::Int64Array*>(arrow_arr.get());

            // Allow imbalance of 10%
            // Currently all input individual chunks are unsorted. There is
            // another version where input individual chunks are sorted (but not
            // globally Finalized) Random sampling on the current version
            // produces slightly worse unbalanced bounds (3%) compared to the
            // sorted version (0.5%). Although already better than theoretical
            // value (10%) with high probability
            int64_t error = std::max((per_host_size + 9) / 10, (int64_t)5);
            auto in_bound = [&](int64_t diff) -> bool {
                return std::abs(diff - per_host_size) <= error;
            };

            for (int64_t i = 0; i < (n_pes - 1); i++) {
                if (i == 0)
                    bodo::tests::check(in_bound(int_arr->Value(i)));
                else
                    bodo::tests::check(
                        in_bound(int_arr->Value(i) - int_arr->Value(i - 1)));
                if (i == n_pes - 2)
                    bodo::tests::check(
                        in_bound(per_host_size * n_pes - int_arr->Value(i)));
            }
        }
    });

    // Test when parallel flag is set to false
    bodo::tests::test("test_unparallel", [] {
        int n_pes, myrank;
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        if (n_pes > 1)
            return;

        std::vector<std::pair<int64_t, int64_t>> limitoffset{
            std::make_pair(-1, -1), std::make_pair(5, 0), std::make_pair(47, 0),
            std::make_pair(20, 27)};

        for (int trial = 0; trial < (int)limitoffset.size(); trial++) {
            std::vector<int64_t> vect_ascending{1};
            std::vector<int64_t> na_position{0};
            const size_t chunk_size = 10;

            const size_t n_elems_per_host = 47;
            std::vector<int64_t> data(n_elems_per_host), all_data;
            for (int i = 0; i < (int)n_elems_per_host; i++) {
                data[i] = i * (i - 40) + 200;
                all_data.push_back(i * (i - 40) + 200);
            }
            unsort_vector(data);
            std::shared_ptr<table_info> table =
                bodo::tests::cppToBodo({"A"}, {false}, {}, std::move(data));
            sort(all_data.begin(), all_data.end());

            StreamSortState state(0, 1, std::move(vect_ascending),
                                  std::move(na_position), table->schema(), true,
                                  limitoffset[trial].first,
                                  limitoffset[trial].second, chunk_size);
            state.ConsumeBatch(table, false, true);
            state.FinalizeBuild();

            int index =
                limitoffset[trial].second == -1 ? 0 : limitoffset[trial].second;

            bool done = false;
            while (!done) {
                auto res = state.GetOutput();
                done = res.second;

                auto output = res.first;
                auto arrow_arr = to_arrow(output->columns[0]);
                arrow::Int64Array* int_arr =
                    static_cast<arrow::Int64Array*>(arrow_arr.get());

                if (!done) {
                    for (int i = 0; i < int_arr->length(); i++) {
                        bodo::tests::check(index < (int64_t)all_data.size() &&
                                           all_data[index++] ==
                                               int_arr->Value(i));
                    }
                } else {
                    if (limitoffset[trial].first == -1)
                        bodo::tests::check(
                            index == static_cast<int64_t>(n_elems_per_host));
                    else
                        bodo::tests::check(
                            index ==
                            std::min(limitoffset[trial].first +
                                         limitoffset[trial].second,
                                     static_cast<int64_t>(n_elems_per_host)));
                    bodo::tests::check(int_arr->length() == 0);
                }
            }
        }
    });

    bodo::tests::test("test_unbalanced_data", [] {
        std::vector<int64_t> vect_ascending{1};
        std::vector<int64_t> na_position{0};
        const size_t chunk_size = 10;

        int n_pes, myrank;
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        // Each rank has different number of almost-evenly distributed data to
        // every other rank
        const size_t n_elems_per_host = n_pes * (myrank + 1) * chunk_size;
        std::vector<int64_t> data(n_elems_per_host), all_data;
        for (int64_t i = 0; i < (int64_t)n_elems_per_host; i++)
            data[i] = myrank + i * n_pes;
        unsort_vector(data);
        std::shared_ptr<table_info> table =
            bodo::tests::cppToBodo({"A"}, {false}, {}, std::move(data));
        for (int i = 0; i < n_pes; i++) {
            const size_t n_elems_per_host = n_pes * (i + 1) * chunk_size;
            for (int j = 0; j < (int)n_elems_per_host; j++)
                all_data.push_back(i + j * n_pes);
        }
        sort(all_data.begin(), all_data.end());

        StreamSortState state(0, 1, std::move(vect_ascending),
                              std::move(na_position), table->schema(), true, -1,
                              -1, chunk_size);
        state.ConsumeBatch(table, true, true);
        state.FinalizeBuild();

        std::pair<int, int> range{};
        std::vector<std::pair<int, int>> gather(n_pes);
        int index = -1;

        bool done = false;
        while (!done) {
            auto res = state.GetOutput();
            done = res.second;

            auto output = res.first;
            auto arrow_arr = to_arrow(output->columns[0]);
            arrow::Int64Array* int_arr =
                static_cast<arrow::Int64Array*>(arrow_arr.get());
            // Now the bounds don't evenly separate all data, so we can't use
            // static pre-computed bounds Each rank: 2 consecutive chunks have
            // consecutive data Across rank: 2 consecutive ranks have
            // consecutive data
            if (!done) {
                bodo::tests::check(int_arr->length() > 0);
                if (index == -1) {
                    index = 0;
                    while (index < (int)all_data.size() &&
                           all_data[index] < int_arr->Value(0))
                        index++;
                    bodo::tests::check(index < (int)all_data.size() &&
                                       all_data[index] == int_arr->Value(0));
                    range.first = index;
                }
                for (int i = 0; i < int_arr->length(); i++) {
                    bodo::tests::check(index < (int)all_data.size() &&
                                       all_data[index] == int_arr->Value(i));
                    range.second = index;
                    index++;
                }
            } else {
                bodo::tests::check(int_arr->length() == 0);
                MPI_Gather(&range, 1, MPI_2INT, gather.data(), 1, MPI_2INT, 0,
                           MPI_COMM_WORLD);
                if (myrank == 0) {
                    for (int i = 0; i < n_pes; i++) {
                        if (i == 0)
                            bodo::tests::check(gather[i].first == all_data[0]);
                        else
                            bodo::tests::check(gather[i].first ==
                                               gather[i - 1].second + 1);
                        if (i == n_pes - 1)
                            bodo::tests::check(gather[i].second ==
                                               (int)all_data.size() - 1);
                    }
                }
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
                              std::move(na_position), table->schema(), true, -1,
                              -1, chunk_size);
        state.ConsumeBatch(table, true, true);
        state.FinalizeBuild();

        std::vector<std::shared_ptr<table_info>> tables;
        bool done = false;
        std::pair<int, int> range{};
        std::vector<std::pair<int, int>> gather(n_pes);
        int index = -1;
        while (!done) {
            auto res = state.GetOutput();
            done = res.second;

            auto output = res.first;
            auto arrow_arr = to_arrow(output->columns[0]);
            arrow::Int64Array* int_arr =
                static_cast<arrow::Int64Array*>(arrow_arr.get());

            const size_t total_elems = 100 * chunk_size * n_pes;
            if (!done) {
                bodo::tests::check(int_arr->length() > 0);
                if (index == -1) {
                    index = 0;
                    while (index < (int)total_elems &&
                           index < int_arr->Value(0))
                        index++;
                    bodo::tests::check(index < (int)total_elems &&
                                       index == int_arr->Value(0));
                    range.first = index;
                }
                for (int i = 0; i < int_arr->length(); i++) {
                    bodo::tests::check(index < (int)total_elems &&
                                       index == int_arr->Value(i));
                    range.second = index;
                    index++;
                }
            } else {
                bodo::tests::check(int_arr->length() == 0);
                MPI_Gather(&range, 1, MPI_2INT, gather.data(), 1, MPI_2INT, 0,
                           MPI_COMM_WORLD);
                if (myrank == 0) {
                    for (int i = 0; i < n_pes; i++) {
                        if (i == 0)
                            bodo::tests::check(gather[i].first == 0);
                        else
                            bodo::tests::check(gather[i].first ==
                                               gather[i - 1].second + 1);
                        if (i == n_pes - 1)
                            bodo::tests::check(gather[i].second ==
                                               (int)total_elems - 1);
                    }
                }
            }
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
                              std::move(na_position), table->schema(), true, -1,
                              -1, chunk_size);
        state.ConsumeBatch(table, true, true);
        state.FinalizeBuild();

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

    // 100 elements in total. Each rank, each call to ConsumeBatch will take the
    // next array of length [5, 10]
    bodo::tests::test("test_parallel_edgecase", [] {
        const size_t chunk_size = 61;

        int n_pes, myrank;
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        const int64_t n_elem = 100;
        std::vector<int64_t> global_data(n_elem);

        // Create an array with numbers from 0 to n_elem
        std::iota(global_data.begin(), global_data.end(), 0);
        unsort_vector(global_data);

        const int seed = 5678;
        std::mt19937 gen(seed);
        std::uniform_int_distribution<> dis(5, 10);

        std::vector<int64_t> vect_ascending{1};
        std::vector<int64_t> na_position{0};

        std::vector<int64_t> schema_data{};
        std::shared_ptr<table_info> schema_table =
            bodo::tests::cppToBodo({"A"}, {false}, {}, std::move(schema_data));
        StreamSortState state(0, 1, std::move(vect_ascending),
                              std::move(na_position), schema_table->schema(),
                              true, -1, -1, chunk_size);
        int index = 0;
        while (index < n_elem) {
            std::vector<int64_t> local_data;
            for (int i = 0; i < n_pes; i++) {
                int batch_elem = dis(gen);
                batch_elem = std::min(batch_elem, (int)n_elem - index);
                if (i == myrank)
                    for (int j = index; j < index + batch_elem; j++)
                        local_data.push_back(global_data[j]);
                index += batch_elem;
            }

            std::shared_ptr<table_info> table = bodo::tests::cppToBodo(
                {"A"}, {false}, {}, std::move(local_data));
            state.ConsumeBatch(table, true, index >= n_elem);
        }
        state.FinalizeBuild();

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

    bodo::tests::test("test_dict_encoded", [] {
        std::vector<std::string> data(10);
        int n_pes, myrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);

        char data_c = 'A' + myrank;
        std::string data_s = " ";
        data_s[0] = data_c;
        std::fill(data.begin(), data.end(), data_s);

        std::shared_ptr<table_info> table =
            bodo::tests::cppToBodo({"A"}, {false}, {"A"}, std::move(data));
        std::vector<int64_t> vect_ascending{1};
        std::vector<int64_t> na_position{0};
        StreamSortState state(0, 1, std::move(vect_ascending),
                              std::move(na_position), table->schema(), true, 5);
        state.ConsumeBatch(table, true, true);

        state.FinalizeBuild();
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
        auto result = gather_table(local_sorted_table,
                                   local_sorted_table->ncols(), false, true);
        if (myrank == 0) {
            auto rescol = result->columns[0];
            bodo::tests::check(rescol->arr_type == bodo_array_type::DICT);
            bodo::tests::check(rescol->length ==
                               static_cast<uint64_t>(n_pes * 10));
            auto resdata = to_arrow(rescol->child_arrays[0]);
            arrow::LargeStringArray* str_arr =
                static_cast<arrow::LargeStringArray*>(resdata.get());
            auto* indices = (dict_indices_t*)(rescol->child_arrays[1]->data1());
            for (int i = 0; i < n_pes; i++) {
                for (int j = 0; j < 10; j++) {
                    dict_indices_t idx = indices[i * 10 + j];
                    std::string expected_s = " ";
                    expected_s[0] = 'A' + i;
                    bodo::tests::check(std::string(str_arr->Value(idx)) ==
                                       expected_s);
                }
            }
        }
    });

    // 100 elements in total. Each rank, each call to ConsumeBatch will take the
    // next array of length [5, 10]
    bodo::tests::test("test_parallel_stress", [] {
        const size_t chunk_size = 17;

        int n_pes, myrank;
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        // test all combination of limit & offset from (0, 0) to (101, 101)
        const int trial_num = 10404;
        for (int trial = 0; trial < trial_num; trial++) {
            size_t limit = (size_t)trial / 102;
            size_t offset = (size_t)trial % 102;
            const int64_t n_elem = 100;
            std::vector<int64_t> global_data(n_elem);

            // Create an array with numbers from 0 to n_elem
            std::iota(global_data.begin(), global_data.end(), 0);
            unsort_vector(global_data);

            const int seed = 5678;
            std::mt19937 gen(seed);
            std::uniform_int_distribution<> dis(5, 10);

            std::vector<int64_t> vect_ascending{1};
            std::vector<int64_t> na_position{0};

            std::vector<int64_t> schema_data{};
            std::shared_ptr<table_info> schema_table = bodo::tests::cppToBodo(
                {"A"}, {false}, {}, std::move(schema_data));
            StreamSortState state(
                0, 1, std::move(vect_ascending), std::move(na_position),
                schema_table->schema(), true, limit, offset, chunk_size);
            int index = 0;
            while (index < n_elem) {
                std::vector<int64_t> local_data;
                for (int i = 0; i < n_pes; i++) {
                    int batch_elem = dis(gen);
                    batch_elem = std::min(batch_elem, (int)n_elem - index);
                    if (i == myrank)
                        for (int j = index; j < index + batch_elem; j++)
                            local_data.push_back(global_data[j]);
                    index += batch_elem;
                }

                std::shared_ptr<table_info> table = bodo::tests::cppToBodo(
                    {"A"}, {false}, {}, std::move(local_data));
                state.ConsumeBatch(table, true, index >= n_elem);
            }
            state.FinalizeBuild();

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

            if (offset >= n_elem || limit == 0) {
                bodo::tests::check(int_arr->length() == 0);
                continue;
            }

            int64_t max = int_arr->length() > 0
                              ? int_arr->Value(int_arr->length() - 1)
                              : -1;
            std::vector<int64_t> maximums(n_pes);
            MPI_Allgather(&max, 1, get_MPI_typ<Bodo_CTypes::INT64>(),
                          maximums.data(), 1, get_MPI_typ<Bodo_CTypes::INT64>(),
                          MPI_COMM_WORLD);
            int64_t start = offset >= 0 ? offset : 0;
            for (int64_t i = 0; i < myrank; i++)
                start = std::max(start, maximums[i] + 1);
            int64_t end = (max == -1 ? start : max + 1);
            bodo::tests::check(int_arr->length() == end - start);
            for (int64_t i = 0; i < end - start; i++) {
                bodo::tests::check(int_arr->Value(i) == start + i);
            }
            int last_non_empty = n_pes - 1;
            while (maximums[last_non_empty] == -1)
                last_non_empty--;
            if (myrank == last_non_empty)
                bodo::tests::check(
                    end ==
                    std::min(static_cast<int64_t>(limit + offset), n_elem));
        }
    });
});
