#include <fmt/format.h>
#include <algorithm>
#include <random>

#include "../libs/_distributed.h"
#include "../libs/streaming/_sort.h"
#include "./test.hpp"
#include "table_generator.hpp"

template <typename T>
static void unsort_vector(std::vector<T>& data) {
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
    std::vector<bool> enable_inmem_concat_sort_opts = {true, false};
    for (bool enable_inmem_concat_sort : enable_inmem_concat_sort_opts) {
        std::string test_suffix =
            enable_inmem_concat_sort ? "inmem_concat_sort" : "kway_merge";
        bodo::tests::test(
            fmt::format("test_external_sort_empty_{}", test_suffix),
            [enable_inmem_concat_sort] {
                std::vector<int64_t> vect_ascending{0};
                std::vector<int64_t> na_position{0};
                std::vector<int64_t> dead_keys;
                auto schema = std::make_shared<bodo::Schema>();
                uint64_t pool_size = 256 * 1024 * 1024;
                ExternalKWayMergeSorter builder(
                    schema, {}, 1, vect_ascending, na_position, dead_keys,
                    pool_size, -1, -1, -1, 1, 2, 1, enable_inmem_concat_sort);

                auto res = builder.Finalize();
                bodo::tests::check(res.size() == 0);
            });

        bodo::tests::test(
            fmt::format("test_external_sort_one_chunk_{}", test_suffix),
            [enable_inmem_concat_sort] {
                std::shared_ptr<table_info> table = bodo::tests::cppToBodo(
                    {"A"}, {false}, {}, std::vector<int64_t>{1, 5, 2, 4, 3});
                std::vector<int64_t> vect_ascending{0};
                std::vector<int64_t> na_position{0};
                std::vector<int64_t> dead_keys;
                std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders =
                    {nullptr};
                uint64_t pool_size = 16 * 1024 * 1024;
                ExternalKWayMergeSorter builder(
                    table->schema(), dict_builders, 1, vect_ascending,
                    na_position, dead_keys, pool_size, -1, -1, -1, 3, 2, 3,
                    enable_inmem_concat_sort);
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

        bodo::tests::test(
            fmt::format("test_external_sort_one_chunk_limitoffset_{}",
                        test_suffix),
            [enable_inmem_concat_sort] {
                std::shared_ptr<table_info> table = bodo::tests::cppToBodo(
                    {"A"}, {false}, {}, std::vector<int64_t>{1, 5, 2, 4, 3});
                std::vector<int64_t> vect_ascending{0};
                std::vector<int64_t> na_position{0};
                std::vector<int64_t> dead_keys;
                std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders =
                    {nullptr};
                uint64_t pool_size = 16 * 1024 * 1024;
                ExternalKWayMergeSorter builder(
                    table->schema(), dict_builders, 1, vect_ascending,
                    na_position, dead_keys, pool_size, -1, 4, 1, 3, 2, 3,
                    enable_inmem_concat_sort);
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
                bodo::tests::check(int_arr->Value(0) == 4);
                bodo::tests::check(int_arr->Value(1) == 3);
                bodo::tests::check(int_arr->Value(2) == 2);
                bodo::tests::check(int_arr->Value(3) == 1);
            });

        bodo::tests::test(
            fmt::format("test_external_sort_two_chunks_{}", test_suffix),
            [enable_inmem_concat_sort] {
                std::shared_ptr<table_info> table_0 = bodo::tests::cppToBodo(
                    {"A"}, {false}, {}, std::vector<int64_t>{1, 5, 2, 4, 3});
                std::shared_ptr<table_info> table_1 = bodo::tests::cppToBodo(
                    {"A"}, {false}, {}, std::vector<int64_t>{1, 5, 2, 4, 3});
                std::vector<int64_t> vect_ascending{0};
                std::vector<int64_t> na_position{0};
                std::vector<int64_t> dead_keys;
                std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders =
                    {nullptr};
                uint64_t pool_size = 16 * 1024 * 1024;
                ExternalKWayMergeSorter builder(
                    table_0->schema(), dict_builders, 1, vect_ascending,
                    na_position, dead_keys, pool_size, -1, -1, -1, 5, 2, 5,
                    enable_inmem_concat_sort);

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

        bodo::tests::test(
            fmt::format("test_external_sort_two_chunks_limitoffset_{}",
                        test_suffix),
            [enable_inmem_concat_sort] {
                std::shared_ptr<table_info> table_0 = bodo::tests::cppToBodo(
                    {"A"}, {false}, {}, std::vector<int64_t>{1, 5, 2, 4, 3});
                std::shared_ptr<table_info> table_1 = bodo::tests::cppToBodo(
                    {"A"}, {false}, {}, std::vector<int64_t>{1, 5, 2, 4, 3});
                std::vector<int64_t> vect_ascending{0};
                std::vector<int64_t> na_position{0};
                std::vector<int64_t> dead_keys;
                std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders =
                    {nullptr};
                uint64_t pool_size = 16 * 1024 * 1024;
                ExternalKWayMergeSorter builder(
                    table_0->schema(), dict_builders, 1, vect_ascending,
                    na_position, dead_keys, pool_size, -1, 6, 2, 5, 2, 5,
                    enable_inmem_concat_sort);

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
                bodo::tests::check(int_arr->length() == 6);
                bodo::tests::check(int_arr->Value(0) == 4);
                bodo::tests::check(int_arr->Value(1) == 4);
                bodo::tests::check(int_arr->Value(2) == 3);
                bodo::tests::check(int_arr->Value(3) == 3);
                bodo::tests::check(int_arr->Value(4) == 2);
                bodo::tests::check(int_arr->Value(5) == 2);
            });

        bodo::tests::test(
            fmt::format("test_external_sort_three_chunks_asc_{}", test_suffix),
            [enable_inmem_concat_sort] {
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
                std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders =
                    {nullptr, nullptr, nullptr};
                uint64_t pool_size = 16 * 1024 * 1024;
                ExternalKWayMergeSorter builder(
                    table_0->schema(), dict_builders, 1, vect_ascending,
                    na_position, dead_keys, pool_size, -1, -1, -1, 3, 2, 3,
                    enable_inmem_concat_sort);

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
                auto* a_arr =
                    static_cast<arrow::Int64Array*>(a_arrow_arr.get());
                auto b_arrow_arr = to_arrow(bodo_table->columns[1]);
                auto* b_arr =
                    static_cast<arrow::Int64Array*>(b_arrow_arr.get());
                auto c_arrow_arr = to_arrow(bodo_table->columns[2]);
                auto* c_arr =
                    static_cast<arrow::LargeStringArray*>(c_arrow_arr.get());
                bodo::tests::check(a_arr->length() == 11);
                bodo::tests::check(b_arr->length() == 11);
                bodo::tests::check(c_arr->length() == 11);

                // Extract the idx'th element from each of the column arrays and
                // pack them into a tuple
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

        bodo::tests::test(
            fmt::format("test_external_sort_three_chunks_asc_limitoffset_{}",
                        test_suffix),
            [enable_inmem_concat_sort] {
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
                std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders =
                    {nullptr, nullptr, nullptr};
                // Each rank has 10 rows, and we want each rank to output rows
                // [7, 10) Since limit / offset logic moves inside Finalize,
                // need some different inputs per rank to maintain correctness.
                uint64_t pool_size = 16 * 1024 * 1024;
                ExternalKWayMergeSorter builder(
                    table_0->schema(), dict_builders, 1, vect_ascending,
                    na_position, dead_keys, pool_size, -1, 3, 7, 3, 2, 3,
                    enable_inmem_concat_sort);

                builder.AppendChunk(table_2);
                builder.AppendChunk(table_1);
                builder.AppendChunk(table_0);

                auto res = builder.Finalize();
                bodo::tests::check(res.size() >= 1);

                std::vector<std::shared_ptr<table_info>> tables;
                for (const auto& chunk : res) {
                    chunk.table->pin();
                    tables.push_back(chunk.table);
                }
                auto bodo_table = concat_tables(tables);

                auto a_arrow_arr = to_arrow(bodo_table->columns[0]);
                auto* a_arr =
                    static_cast<arrow::Int64Array*>(a_arrow_arr.get());
                auto b_arrow_arr = to_arrow(bodo_table->columns[1]);
                auto* b_arr =
                    static_cast<arrow::Int64Array*>(b_arrow_arr.get());
                auto c_arrow_arr = to_arrow(bodo_table->columns[2]);
                auto* c_arr =
                    static_cast<arrow::LargeStringArray*>(c_arrow_arr.get());
                bodo::tests::check(a_arr->length() == 3);
                bodo::tests::check(b_arr->length() == 3);
                bodo::tests::check(c_arr->length() == 3);

                // Extract the idx'th element from each of the column arrays and
                // pack them into a tuple
                auto getRow = [&](int64_t idx) {
                    return std::make_tuple(a_arr->Value(idx), b_arr->Value(idx),
                                           c_arr->Value(idx));
                };

                bodo::tests::check(getRow(0) == std::make_tuple(8, 3, "c"));
                bodo::tests::check(getRow(1) == std::make_tuple(9, 3, "c"));
                bodo::tests::check(getRow(2) == std::make_tuple(10, 4, "d"));
            });

        bodo::tests::test(
            fmt::format("test_sort_nulls_{}", test_suffix),
            [enable_inmem_concat_sort] {
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
                uint64_t pool_size = 16 * 1024 * 1024;
                // Test with nulls at the front
                {
                    std::vector<int64_t> na_position{0};
                    std::vector<std::shared_ptr<DictionaryBuilder>>
                        dict_builders = {nullptr};
                    ExternalKWayMergeSorter builder(
                        table_0->schema(), dict_builders, 1, vect_ascending,
                        na_position, dead_keys, pool_size, -1, -1, -1, 3, 2, 3,
                        enable_inmem_concat_sort);

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

                    std::vector<std::optional<int64_t>> expected{std::nullopt,
                                                                 std::nullopt,
                                                                 1,
                                                                 2,
                                                                 3,
                                                                 5,
                                                                 6,
                                                                 7,
                                                                 9,
                                                                 10,
                                                                 11};

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
                    std::vector<std::shared_ptr<DictionaryBuilder>>
                        dict_builders = {nullptr};
                    ExternalKWayMergeSorter builder(
                        table_0->schema(), dict_builders, 1, vect_ascending,
                        na_position, dead_keys, pool_size, -1, -1, -1, 3, 2, 3,
                        enable_inmem_concat_sort);

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

                    std::vector<std::optional<int64_t>> expected{
                        1,           2, 3, 5, 6, 7, 9, 10, 11, std::nullopt,
                        std::nullopt};

                    for (size_t i = 0; i < 11; i++) {
                        std::optional<int64_t> actual = std::nullopt;
                        if (int_arr->IsValid(i)) {
                            actual = int_arr->Value(i);
                        }
                        bodo::tests::check(actual == expected[i]);
                    }
                }
            });
    }

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
                              std::move(na_position), table->schema(), true, 10,
                              4096, 10, 2);
        state.ConsumeBatch(table);
        state.FinalizeBuild();

        auto res = state.bounds_;
        if (myrank == 0) {
            bodo::tests::check(static_cast<int>(res->nrows()) == (n_pes - 1));

            auto arrow_arr = to_arrow(res->columns[0]);
            arrow::Int64Array* int_arr =
                static_cast<arrow::Int64Array*>(arrow_arr.get());

            // Allow relative error of 40%
            double error = 0.40;
            auto in_bound = [&](int64_t diff) -> bool {
                double rel_error = std::abs((double)(diff - per_host_size) /
                                            (double)per_host_size);
                return rel_error <= error;
            };

            for (int64_t i = 0; i < (n_pes - 1); i++) {
                if (i == 0) {
                    bodo::tests::check(in_bound(int_arr->Value(i)));
                } else {
                    bodo::tests::check(
                        in_bound(int_arr->Value(i) - int_arr->Value(i - 1)));
                }
                if (i == n_pes - 2) {
                    bodo::tests::check(
                        in_bound(per_host_size * n_pes - int_arr->Value(i)));
                }
            }
        }
    });

    bodo::tests::test("test_sampling_hard", [] {
        int n_pes, myrank;
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        if (n_pes == 1) {
            // parallel only
            return;
        }

        const int64_t n_elem = 100000;
        int64_t per_host_size = n_elem / n_pes;
        // Create a table with numbers from 1 to 1000
        std::vector<int64_t> all_data(n_elem);
        std::iota(all_data.begin(), all_data.end(), 0);
        unsort_vector(all_data);
        std::vector<int64_t> data{};
        for (int i = 0; i < per_host_size; i++) {
            data.push_back(all_data[i + myrank * per_host_size]);
        }

        std::shared_ptr<table_info> table =
            bodo::tests::cppToBodo({"A"}, {false}, {}, std::move(data));

        std::vector<int64_t> vect_ascending{1};
        std::vector<int64_t> na_position{0};
        StreamSortState state(0, 1, std::move(vect_ascending),
                              std::move(na_position), table->schema(), true, 10,
                              4096, 10, 2);
        state.ConsumeBatch(table);
        // This is okay for testing, but calling Finalize here means that this
        // test cannot safely call GlobalSort.
        auto local_bounds = state.reservoir_sampling_state.Finalize();
        auto res = state.GetParallelSortBounds(std::move(local_bounds));
        if (myrank == 0) {
            bodo::tests::check(static_cast<int>(res->nrows()) == (n_pes - 1));

            auto arrow_arr = to_arrow(res->columns[0]);
            auto* int_arr = static_cast<arrow::Int64Array*>(arrow_arr.get());

            bodo::tests::check(int_arr->Value(0) > 0);
            for (int64_t i = 1; i < (n_pes - 1); i++) {
                bodo::tests::check(int_arr->Value(i) > int_arr->Value(i - 1));
            }
        }
    });

    // Test when parallel flag is set to false
    bodo::tests::test("test_unparallel", [] {
        int n_pes, myrank;
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        if (n_pes > 1) {
            return;
        }

        std::vector<std::pair<int64_t, int64_t>> limitoffset{
            std::make_pair(-1, -1), std::make_pair(5, 0), std::make_pair(47, 0),
            std::make_pair(20, 27)};

        for (auto& trial : limitoffset) {
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
            std::ranges::sort(all_data);

            if (trial.second == -1) {
                StreamSortState state(0, 1, std::move(vect_ascending),
                                      std::move(na_position), table->schema(),
                                      false, chunk_size, 4096, chunk_size, 2);
                state.ConsumeBatch(table);
                state.FinalizeBuild();

                int index = 0;

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
                            bodo::tests::check(
                                index < (int64_t)all_data.size() &&
                                all_data[index++] == int_arr->Value(i));
                        }
                    } else {
                        if (trial.first == -1) {
                            bodo::tests::check(index == static_cast<int64_t>(
                                                            n_elems_per_host));
                        } else {
                            bodo::tests::check(
                                index == std::min(trial.first + trial.second,
                                                  static_cast<int64_t>(
                                                      n_elems_per_host)));
                        }
                        bodo::tests::check(int_arr->length() == 0);
                    }
                }
            } else {
                StreamSortLimitOffsetState state(
                    -1, 1, std::move(vect_ascending), std::move(na_position),
                    table->schema(), false, trial.first, trial.second, 4096,
                    chunk_size, 2);
                state.ConsumeBatch(table);
                state.FinalizeBuild();

                int index = trial.second;

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
                            bodo::tests::check(
                                index < (int64_t)all_data.size() &&
                                all_data[index++] == int_arr->Value(i));
                        }
                    } else {
                        if (trial.first == -1) {
                            bodo::tests::check(index == static_cast<int64_t>(
                                                            n_elems_per_host));
                        } else {
                            bodo::tests::check(
                                index == std::min(trial.first + trial.second,
                                                  static_cast<int64_t>(
                                                      n_elems_per_host)));
                        }
                        bodo::tests::check(int_arr->length() == 0);
                    }
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
        for (int64_t i = 0; i < (int64_t)n_elems_per_host; i++) {
            data[i] = myrank + i * n_pes;
        }
        unsort_vector(data);
        std::shared_ptr<table_info> table =
            bodo::tests::cppToBodo({"A"}, {false}, {}, std::move(data));
        for (int i = 0; i < n_pes; i++) {
            const size_t n_elems_per_host = n_pes * (i + 1) * chunk_size;
            for (int j = 0; j < (int)n_elems_per_host; j++) {
                all_data.push_back(i + j * n_pes);
            }
        }
        std::ranges::sort(all_data);

        StreamSortState state(0, 1, std::move(vect_ascending),
                              std::move(na_position), table->schema(), true,
                              chunk_size, 4096, chunk_size, 2);
        state.ConsumeBatch(table);
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
                           all_data[index] < int_arr->Value(0)) {
                        index++;
                    }
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
                CHECK_MPI(MPI_Gather(&range, 1, MPI_2INT, gather.data(), 1,
                                     MPI_2INT, 0, MPI_COMM_WORLD),
                          "test_external_sort::test_unbalanced_data: MPI error "
                          "on MPI_Gather:");
                if (myrank == 0) {
                    for (int i = 0; i < n_pes; i++) {
                        if (i == 0) {
                            bodo::tests::check(gather[i].first == all_data[0]);
                        } else {
                            bodo::tests::check(gather[i].first ==
                                               gather[i - 1].second + 1);
                        }
                        if (i == n_pes - 1) {
                            bodo::tests::check(gather[i].second ==
                                               (int)all_data.size() - 1);
                        }
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
                              std::move(na_position), table->schema(), true,
                              chunk_size, 4096, chunk_size, 2);
        state.ConsumeBatch(table);
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
                           index < int_arr->Value(0)) {
                        index++;
                    }
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
                CHECK_MPI(MPI_Gather(&range, 1, MPI_2INT, gather.data(), 1,
                                     MPI_2INT, 0, MPI_COMM_WORLD),
                          "test_external_sort::test_parallel_all_local: MPI "
                          "error on MPI_Gather:");
                if (myrank == 0) {
                    for (int i = 0; i < n_pes; i++) {
                        if (i == 0) {
                            bodo::tests::check(gather[i].first == 0);
                        } else {
                            bodo::tests::check(gather[i].first ==
                                               gather[i - 1].second + 1);
                        }
                        if (i == n_pes - 1) {
                            bodo::tests::check(gather[i].second ==
                                               (int)total_elems - 1);
                        }
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
        CHECK_MPI(
            MPI_Bcast(global_data.data(), n_elem,
                      get_MPI_typ<Bodo_CTypes::INT64>(), 0, MPI_COMM_WORLD),
            "test_external_sort::test_parallel: MPI error on MPI_Bcast:");

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
                              chunk_size, 4096, chunk_size, 2);
        state.ConsumeBatch(table);
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
        CHECK_MPI(
            MPI_Allgather(&max, 1, get_MPI_typ<Bodo_CTypes::INT64>(),
                          maximums.data(), 1, get_MPI_typ<Bodo_CTypes::INT64>(),
                          MPI_COMM_WORLD),
            "test_external_sort::test_parallel: MPI error on MPI_Allgather:");

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
                              true, chunk_size, 4096, chunk_size, 2);
        int index = 0;
        while (index < n_elem) {
            std::vector<int64_t> local_data;
            for (int i = 0; i < n_pes; i++) {
                int batch_elem = dis(gen);
                batch_elem = std::min(batch_elem, (int)n_elem - index);
                if (i == myrank) {
                    for (int j = index; j < index + batch_elem; j++) {
                        local_data.push_back(global_data[j]);
                    }
                }
                index += batch_elem;
            }

            std::shared_ptr<table_info> table = bodo::tests::cppToBodo(
                {"A"}, {false}, {}, std::move(local_data));
            state.ConsumeBatch(table);
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
        CHECK_MPI(
            MPI_Allgather(&max, 1, get_MPI_typ<Bodo_CTypes::INT64>(),
                          maximums.data(), 1, get_MPI_typ<Bodo_CTypes::INT64>(),
                          MPI_COMM_WORLD),
            "test_external_sort::test_parallel_edgecase: MPI error on "
            "MPI_Allgather:");

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
    bodo::tests::test("test_parallel_stress", [] {
        const size_t chunk_size = 17;

        int n_pes, myrank;
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        const int64_t n_elem = 100;
        std::vector<int64_t> global_data(n_elem);

        // Create an array with numbers from 0 to n_elem
        std::iota(global_data.begin(), global_data.end(), 0);
        unsort_vector(global_data);

        bool test_small_limit_optim = true;

        // test all combination of limit & offset from (0, 0) to (101, 101)
        const int trial_num = 10404;
        for (int trial = 0; trial < trial_num; trial++) {
            size_t limit = (size_t)trial / 102;
            size_t offset = (size_t)trial % 102;

            const int seed = 5678;
            std::mt19937 gen(seed);
            std::uniform_int_distribution<> dis(5, 10);

            std::vector<int64_t> vect_ascending{1};
            std::vector<int64_t> na_position{0};

            std::vector<int64_t> schema_data{};
            std::shared_ptr<table_info> schema_table = bodo::tests::cppToBodo(
                {"A"}, {false}, {}, std::move(schema_data));
            StreamSortLimitOffsetState state(
                0, 1, std::move(vect_ascending), std::move(na_position),
                schema_table->schema(), true, limit, offset, 4096, chunk_size,
                2, true, test_small_limit_optim);
            int index = 0;
            while (index < n_elem) {
                std::vector<int64_t> local_data;
                for (int i = 0; i < n_pes; i++) {
                    int batch_elem = dis(gen);
                    batch_elem = std::min(batch_elem, (int)n_elem - index);
                    if (i == myrank) {
                        for (int j = index; j < index + batch_elem; j++) {
                            local_data.push_back(global_data[j]);
                        }
                    }
                    index += batch_elem;
                }

                std::shared_ptr<table_info> table = bodo::tests::cppToBodo(
                    {"A"}, {false}, {}, std::move(local_data));
                state.ConsumeBatch(table);
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
            CHECK_MPI(MPI_Allgather(&max, 1, get_MPI_typ<Bodo_CTypes::INT64>(),
                                    maximums.data(), 1,
                                    get_MPI_typ<Bodo_CTypes::INT64>(),
                                    MPI_COMM_WORLD),
                      "test_external_sort::test_parallel_stress: MPI error on "
                      "MPI_Allgather:");
            int64_t start = offset >= 0 ? offset : 0;
            for (int64_t i = 0; i < myrank; i++) {
                start = std::max(start, maximums[i] + 1);
            }
            int64_t end = (max == -1 ? start : max + 1);
            bodo::tests::check(int_arr->length() == end - start);
            for (int64_t i = 0; i < end - start; i++) {
                bodo::tests::check(int_arr->Value(i) == start + i);
            }
            int last_non_empty = n_pes - 1;
            while (maximums[last_non_empty] == -1) {
                last_non_empty--;
            }
            if (myrank == last_non_empty) {
                bodo::tests::check(
                    end ==
                    std::min(static_cast<int64_t>(limit + offset), n_elem));
            }
        }
    });

    // Stress test the chunked async shuffle. We test this by using a small
    // shuffle chunk size and allowing it to have many concurrent sends at once.
    bodo::tests::test("test_shuffle_stress", [] {
        int n_pes, myrank;
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        // Restrict to the 2 ranks case
        if (n_pes != 2) {
            return;
        }

        // Rank 0: 1001-2000 (2M rows), Rank 1: 0-1000 (2M rows).
        // This will lead to a lot of data moving from rank 0 to 1 and
        // vice-versa.
        size_t n_elem = 2000000;
        std::vector<int64_t> local_data_colA(n_elem);
        std::vector<std::string> local_data_colB(n_elem);
        int64_t left = myrank == 0 ? 1001 : 1;
        for (size_t i = 0; i < n_elem; i++) {
            local_data_colA[i] = left + (i % 1000);
            local_data_colB[i] = std::to_string(local_data_colA[i]);
        }
        unsort_vector(local_data_colA);
        unsort_vector(local_data_colB);

        std::vector<int64_t> vect_ascending{1};
        std::vector<int64_t> na_position{0};
        std::vector<int64_t> schema_data_colA{};
        std::vector<std::string> schema_data_colB{};
        std::shared_ptr<table_info> schema_table = bodo::tests::cppToBodo(
            {"A", "B"}, {false, true}, {}, std::move(schema_data_colA),
            std::move(schema_data_colB));

        // shuffle_chunk_size=100 and shuffle_max_concurrent_sends=1000 to force
        // it to do a lot of small concurrent shuffles.
        StreamSortState state(-1, 1, std::move(vect_ascending),
                              std::move(na_position), schema_table->schema(),
                              true, DEFAULT_SAMPLE_SIZE, STREAMING_BATCH_SIZE,
                              -1, -1, true, /*shuffle_chunksize_*/ 100,
                              /*shuffle_max_concurrent_sends_*/ 5000);

        const size_t input_batch_size = STREAMING_BATCH_SIZE;
        size_t index = 0;
        while (index < n_elem) {
            size_t right = std::min(index + input_batch_size, n_elem);
            std::vector<int64_t> batch_colA;
            std::vector<std::string> batch_colB;
            batch_colA.insert(batch_colA.end(), local_data_colA.begin() + index,
                              local_data_colA.begin() + right);
            batch_colB.insert(batch_colB.end(), local_data_colB.begin() + index,
                              local_data_colB.begin() + right);

            std::shared_ptr<table_info> table = bodo::tests::cppToBodo(
                {"A", "B"}, {false, true}, {}, std::move(batch_colA),
                std::move(batch_colB));
            state.ConsumeBatch(table);
            index = right;
        }

        // This is where the shuffle will occur.
        state.FinalizeBuild();

        // Collect all output tables
        bool done = false;
        std::vector<std::shared_ptr<table_info>> output_tables;
        while (!done) {
            auto res = state.GetOutput();
            done = res.second;
            output_tables.push_back(res.first);
        }

        /// Verify some of the metrics
        bodo::tests::check(state.metrics.shuffle_chunk_size == 100);
        // Depending on the calculated bounds, the number should be around 20000
        // on both.
        bodo::tests::check(state.metrics.n_shuffle_send >= 19000);
        bodo::tests::check(state.metrics.n_shuffle_recv >= 19000);
        // Similarly, these should be around 2000000 on both ranks.
        bodo::tests::check(state.metrics.shuffle_total_sent_nrows >= 1990000);
        bodo::tests::check(state.metrics.shuffle_total_recv_nrows >= 1990000);
        // It will typically be able to post all allowed (5000) sends, and have
        // >500 receives inflight at once.
        // Intel MPI on Windows seems to have a different behavior and have
        // fewer inflight sends/recvs.
#ifndef _WIN32
        bodo::tests::check(state.metrics.max_concurrent_sends >= 4500);
        bodo::tests::check(state.metrics.max_concurrent_recvs >= 500);
#endif

        // Merge tables and get a single output array
        std::shared_ptr<table_info> local_sorted_table =
            concat_tables(std::move(output_tables));
        std::shared_ptr<arrow::Array> arrow_arr =
            to_arrow(local_sorted_table->columns[0]);
        arrow::Int64Array* int_arr =
            static_cast<arrow::Int64Array*>(arrow_arr.get());

        int64_t max = int_arr->Value(int_arr->length() - 1);
        // Get the maximum element from every rank
        std::vector<int64_t> maximums(n_pes);
        CHECK_MPI(
            MPI_Allgather(&max, 1, get_MPI_typ<Bodo_CTypes::INT64>(),
                          maximums.data(), 1, get_MPI_typ<Bodo_CTypes::INT64>(),
                          MPI_COMM_WORLD),
            "test_external_sort::test_shuffle_stress: MPI error on "
            "MPI_Allgather:");

        // Check that element 0 is the max value on the previous rank + 1
        if (myrank != 0) {
            bodo::tests::check(int_arr->Value(0) == (maximums[myrank - 1] + 1));
        }

        // Check that all values are increasing
        for (int64_t i = 1; i < int_arr->length(); i++) {
            bodo::tests::check(int_arr->Value(i) >= int_arr->Value(i - 1));
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
        std::ranges::fill(data, data_s);

        std::shared_ptr<table_info> table =
            bodo::tests::cppToBodo({"A"}, {false}, {"A"}, std::move(data));
        std::vector<int64_t> vect_ascending{1};
        std::vector<int64_t> na_position{0};
        StreamSortState state(0, 1, std::move(vect_ascending),
                              std::move(na_position), table->schema(), true, 5,
                              4096, 5, 2);
        state.ConsumeBatch(table);

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

    bodo::tests::test("test_dict_encoded_nested", [] {
        int n_pes, myrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);

        const size_t n_elems = 10;
        auto dict_arr = alloc_dict_string_array(n_elems, 1, 1, 0);
        ((char*)dict_arr->child_arrays[0]->data1())[0] = 'A' + myrank;
        ((offset_t*)dict_arr->child_arrays[0]->data2())[0] = 0;
        ((offset_t*)dict_arr->child_arrays[0]->data2())[1] = 1;
        for (size_t i = 0; i < n_elems; i++) {
            ((dict_indices_t*)dict_arr->child_arrays[1]->data1())[i] = 0;
            SetBitTo((uint8_t*)dict_arr->null_bitmask(), i, true);
        }

        auto list_arr = alloc_array_item(1, std::move(dict_arr), 0);
        ((offset_t*)list_arr->data1())[0] = 0;
        ((offset_t*)list_arr->data1())[1] = 10;

        std::vector<std::shared_ptr<array_info>> columns{std::move(list_arr)};
        std::shared_ptr<table_info> table =
            std::make_shared<table_info>(std::move(columns));

        std::vector<int64_t> vect_ascending{1};
        std::vector<int64_t> na_position{0};
        StreamSortState state(0, 1, std::move(vect_ascending),
                              std::move(na_position), table->schema(), true, 5);
        state.ConsumeBatch(table);

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
            // Assert that we have all the child elements from all ranks
            bodo::tests::check(result->columns[0]->child_arrays[0]->length ==
                               static_cast<uint64_t>(n_pes * 10));
            // Check that there's one elements in the dictionary per rank
            bodo::tests::check(
                result->columns[0]->child_arrays[0]->child_arrays[0]->length ==
                static_cast<uint64_t>(n_pes));
        }
    });
});
