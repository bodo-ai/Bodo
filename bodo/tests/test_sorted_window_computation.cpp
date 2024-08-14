#include <mpi.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <tuple>
#include <vector>
#include "../libs/_array_utils.h"
#include "../libs/_bodo_common.h"
#include "../libs/_memory.h"
#include "../libs/_window_calculator.h"
#include "../libs/_window_compute.h"
#include "./table_generator.hpp"
#include "./test.hpp"
#include "utils.h"

// creates a NUMPY/int64 array that goes from min..(min + length - 1), could be
// templated
std::shared_ptr<array_info> int64_row_numbers_arr(int64_t min, size_t length) {
    std::shared_ptr<array_info> res = alloc_numpy(length, Bodo_CTypes::INT64);
    for (size_t i = 0; i < length; i++) {
        getv<int64_t>(res, i) = min + i;
    }
    return res;
}

// create a NUMPY/int64 array and initialize all values to val
std::shared_ptr<array_info> const_int64_arr(size_t length, int64_t val) {
    std::shared_ptr<array_info> res = alloc_numpy(length, Bodo_CTypes::INT64);
    for (size_t i = 0; i < length; i++) {
        getv<int64_t>(res, i) = val;
    }
    return res;
}

// add a const int64 val to a (valid) interval in an array
// from start to stop (not including stop)
void add_int64_offset_to_interval(std::shared_ptr<array_info> arr, size_t start,
                                  size_t stop, int64_t val) {
    assert(start <= stop && stop <= arr->length);
    for (size_t i = start; i < stop; i++) {
        getv<int64_t>(arr, i) += val;
    }
}

// verify the output of a sorted with function is correct.
void verify_sorted_window_output(
    int32_t window_func,
    std::vector<std::shared_ptr<array_info>> partition_by_arrs,
    std::vector<std::shared_ptr<array_info>> order_by_arrs,
    std::vector<std::shared_ptr<array_info>> window_args,
    std::vector<int32_t> window_offset_indices,
    std::vector<std::shared_ptr<array_info>> out_arrs,
    std::shared_ptr<array_info> expected_out, bool is_parallel = true) {
    size_t num_columns =
        partition_by_arrs.size() + order_by_arrs.size() + window_args.size();
    std::vector<std::shared_ptr<DictionaryBuilder>> builders(num_columns,
                                                             nullptr);

    sorted_window_computation(partition_by_arrs, order_by_arrs, window_args,
                              window_offset_indices, {window_func}, out_arrs,
                              expected_out->length, builders, is_parallel);

    std::stringstream ss1;
    std::stringstream ss2;
    DEBUG_PrintColumn(ss1, out_arrs[0]);
    DEBUG_PrintColumn(ss2, expected_out);

    bodo::tests::check(ss1.str() == ss2.str());
}

// verify the output of a collection of window calculators is correct.
void verify_window_calculators(
    std::vector<std::shared_ptr<table_info>> in_chunks,
    std::vector<int32_t> partition_col_indices,
    std::vector<int32_t> order_col_indices, std::vector<int32_t> keep_indices,
    std::vector<std::vector<int32_t>> input_col_indices,
    std::vector<int32_t> window_funcs,
    bodo_array_type::arr_type_enum partition_arr_type,
    bodo_array_type::arr_type_enum order_arr_type,
    std::vector<std::shared_ptr<table_info>> expected_out_chunks,
    bool is_parallel = true,
    bodo::IBufferPool *const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager()) {
    size_t num_columns = in_chunks[0]->columns.size();
    std::vector<std::shared_ptr<DictionaryBuilder>> builders(num_columns,
                                                             nullptr);
    std::vector<int8_t> arr_types;
    std::vector<int8_t> c_types;
    std::tie(arr_types, c_types) = in_chunks[0]->schema()->Serialize();
    std::shared_ptr<bodo::Schema> schema =
        bodo::Schema::Deserialize(arr_types, c_types);

    // Ensure the input chunks are unpinned before doing any computation.
    for (auto &it : in_chunks) {
        it->unpin();
    }

    // Feed the input data to the implementation and have it populate a vector
    // with the output chunks.
    std::vector<std::shared_ptr<table_info>> actual_out_chunks;
    compute_window_functions_via_calculators(
        schema, in_chunks, partition_col_indices, order_col_indices,
        keep_indices, input_col_indices, window_funcs, builders,
        partition_arr_type, order_arr_type, actual_out_chunks, is_parallel,
        pool, mm);

    // Iterate across the chunks to ensure they match the expected values.
    size_t n_chunks = expected_out_chunks.size();
    bodo::tests::check_parallel(actual_out_chunks.size() == n_chunks);
    bool chunks_all_match = true;
    for (size_t i = 0; i < n_chunks; i++) {
        std::shared_ptr<table_info> expected_chunk = expected_out_chunks[i];
        std::shared_ptr<table_info> actual_chunk = actual_out_chunks[i];
        expected_chunk->pin();
        actual_chunk->pin();
        std::stringstream ss1;
        std::stringstream ss2;
        DEBUG_PrintTable(ss1, actual_chunk);
        DEBUG_PrintTable(ss2, expected_chunk);
        chunks_all_match = chunks_all_match && (ss1.str() == ss2.str());
        expected_chunk->unpin();
        actual_chunk->unpin();
    }
    bodo::tests::check_parallel(chunks_all_match);
}

/**
 * @brief Creates a common edge case example where some ranks have empty data
 *
 * @param n_partition_cols Number of partition columns
 * @param n_order_by_cols Number of order by columns
 * @return std::tuple<std::vector<std::shared_ptr<array_info>>,
 * std::vector<std::shared_ptr<array_info>>> The partition by and order by
 * columns
 */
std::tuple<std::vector<std::shared_ptr<array_info>>,
           std::vector<std::shared_ptr<array_info>>>
create_int_example_with_holes(size_t n_partition_cols, size_t n_order_by_cols) {
    int myrank;
    int n_pes;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);

    size_t n = myrank % 2 == 0 ? 4 : 0;

    std::vector<std::shared_ptr<array_info>> partition_by_arrs, order_by_arrs;

    std::shared_ptr<array_info> empty_arr = alloc_numpy(0, Bodo_CTypes::INT64);

    for (size_t i = 0; i < n_partition_cols; i++) {
        auto part_arr = const_int64_arr(n, myrank / 2);
        // partition should look like (rank 0) 0..0 1..1, (rank
        // 1) 1..1 2..2, etc
        add_int64_offset_to_interval(part_arr, part_arr->length / 2,
                                     part_arr->length, 1);
        partition_by_arrs.push_back(part_arr);
    }

    for (size_t i = 0; i < n_order_by_cols; i++) {
        if (n > 0) {
            order_by_arrs.push_back(bodo::tests::cppToBodoArr({3, 4, 1, 2}));
        } else {
            order_by_arrs.push_back(empty_arr);
        }
    }

    return std::make_tuple(partition_by_arrs, order_by_arrs);
}

/**
 * @brief Create a common edge case for testing rank functions where groups and
 * orderby columns match across ranks
 *
 * @param n_partition_cols Number of partition columns
 * @param n_order_by_cols Number of order by columns
 * @return std::tuple<std::vector<std::shared_ptr<array_info>>,
 * std::vector<std::shared_ptr<array_info>>> The partition and order by columns
 */
std::tuple<std::vector<std::shared_ptr<array_info>>,
           std::vector<std::shared_ptr<array_info>>>
create_int_example_multiple_groups(size_t n_partition_cols,
                                   size_t n_order_by_cols) {
    std::vector<std::shared_ptr<array_info>> partition_by_arrs;
    std::vector<std::shared_ptr<array_info>> order_by_arrs;
    size_t n = 4;

    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    for (size_t i = 0; i < n_partition_cols; i++) {
        auto part_arr = const_int64_arr(n, myrank);
        // partition should look like (rank 0) 0..0 1..1, (rank
        // 1) 1..1 2..2, etc
        add_int64_offset_to_interval(part_arr, part_arr->length / 2,
                                     part_arr->length, 1);
        partition_by_arrs.push_back(part_arr);
    }

    for (size_t i = 0; i < n_order_by_cols; i++) {
        order_by_arrs.push_back(bodo::tests::cppToBodoArr({3, 4, 1, 3}));
    }

    return std::make_tuple(partition_by_arrs, order_by_arrs);
}

// TODO: template these tests and/or add parameter to test different types
static bodo::tests::suite tests([] {
    bodo::tests::test("test_par_row_number_single_group", [] {
        // tests example of a (parallel) row number where each rank gets
        // a list of values that are all the same + in the same group
        std::vector<std::shared_ptr<array_info>> partition_by_arrs;
        std::vector<std::shared_ptr<array_info>> order_by_arrs;
        std::vector<std::shared_ptr<array_info>> out_arrs;
        size_t n = 3;
        size_t n_partition_cols = 0;
        size_t n_order_by_cols = 1;

        int myrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        for (size_t i = 0; i < n_partition_cols; i++) {
            partition_by_arrs.push_back(
                alloc_nullable_array_all_nulls(n, Bodo_CTypes::INT8));
        }
        for (size_t i = 0; i < n_order_by_cols; i++) {
            order_by_arrs.push_back(
                alloc_nullable_array_all_nulls(n, Bodo_CTypes::INT8));
        }
        out_arrs.push_back(alloc_numpy(n, Bodo_CTypes::INT64));
        std::shared_ptr<array_info> expected_out =
            int64_row_numbers_arr(n * myrank + 1, n);

        verify_sorted_window_output(Bodo_FTypes::row_number, partition_by_arrs,
                                    order_by_arrs, {}, {}, out_arrs,
                                    expected_out);
    });

    bodo::tests::test("test_par_row_number_diff_groups", [] {
        // tests example of a (parallel) row number where each rank gets
        // a list of values that are all the same but in diff groups
        std::vector<std::shared_ptr<array_info>> partition_by_arrs;
        std::vector<std::shared_ptr<array_info>> order_by_arrs;
        std::vector<std::shared_ptr<array_info>> out_arrs;
        size_t n = 3;
        size_t n_partition_cols = 1;
        size_t n_order_by_cols = 1;

        int myrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        for (size_t i = 0; i < n_partition_cols; i++) {
            partition_by_arrs.push_back(const_int64_arr(n, myrank));
        }

        for (size_t i = 0; i < n_order_by_cols; i++) {
            order_by_arrs.push_back(
                alloc_nullable_array_all_nulls(n, Bodo_CTypes::INT8));
        }
        out_arrs.push_back(alloc_numpy(n, Bodo_CTypes::INT64));
        std::shared_ptr<array_info> expected_out = int64_row_numbers_arr(1, n);

        verify_sorted_window_output(Bodo_FTypes::row_number, partition_by_arrs,
                                    order_by_arrs, {}, {}, out_arrs,
                                    expected_out);
    });
    bodo::tests::test("test_par_row_number_empty_arrs", [] {
        // all ranks have empty tables
        std::vector<std::shared_ptr<array_info>> partition_by_arrs;
        std::vector<std::shared_ptr<array_info>> order_by_arrs;
        std::vector<std::shared_ptr<array_info>> out_arrs;
        size_t n = 0;
        size_t n_partition_cols = 0;
        size_t n_order_by_cols = 1;

        int myrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        for (size_t i = 0; i < n_partition_cols; i++) {
            partition_by_arrs.push_back(
                alloc_nullable_array_all_nulls(n, Bodo_CTypes::INT8));
        }
        for (size_t i = 0; i < n_order_by_cols; i++) {
            order_by_arrs.push_back(
                alloc_nullable_array_all_nulls(n, Bodo_CTypes::INT8));
        }
        out_arrs.push_back(alloc_numpy(n, Bodo_CTypes::INT64));
        std::shared_ptr<array_info> expected_out =
            int64_row_numbers_arr(n * myrank + 1, n);

        verify_sorted_window_output(Bodo_FTypes::row_number, partition_by_arrs,
                                    order_by_arrs, {}, {}, out_arrs,
                                    expected_out);
    });
    bodo::tests::test("test_par_row_number_every_other_empty", [] {
        // every other rank has empty table
        std::vector<std::shared_ptr<array_info>> partition_by_arrs;
        std::vector<std::shared_ptr<array_info>> order_by_arrs;
        std::vector<std::shared_ptr<array_info>> out_arrs;
        size_t n = 3;
        size_t n_partition_cols = 0;
        size_t n_order_by_cols = 1;

        int myrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        if (myrank % 2 == 0) {
            n = 0;
        }

        for (size_t i = 0; i < n_partition_cols; i++) {
            partition_by_arrs.push_back(
                alloc_nullable_array_all_nulls(n, Bodo_CTypes::INT8));
        }
        for (size_t i = 0; i < n_order_by_cols; i++) {
            order_by_arrs.push_back(
                alloc_nullable_array_all_nulls(n, Bodo_CTypes::INT8));
        }
        out_arrs.push_back(alloc_numpy(n, Bodo_CTypes::INT64));
        std::shared_ptr<array_info> expected_out =
            int64_row_numbers_arr(n * (myrank / 2) + 1, n);

        verify_sorted_window_output(Bodo_FTypes::row_number, partition_by_arrs,
                                    order_by_arrs, {}, {}, out_arrs,
                                    expected_out);
    });
    bodo::tests::test("test_par_row_number_multiple_groups_per_rank", [] {
        // more than one group and they spill over
        std::vector<std::shared_ptr<array_info>> out_arrs;
        size_t n = 4;
        size_t n_partition_cols = 1;
        size_t n_order_by_cols = 1;

        int myrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        auto [partition_by_arrs, order_by_arrs] =
            create_int_example_multiple_groups(n_partition_cols,
                                               n_order_by_cols);
        out_arrs.push_back(alloc_numpy(n, Bodo_CTypes::INT64));

        // generate expected output
        // should look something like (rank0) 1 2 .. n/2 1 2 .. n/2 (rank1)
        // n/2+1 .. n 1 .. n/2 ...
        std::shared_ptr<array_info> expected_out =
            alloc_numpy(n, Bodo_CTypes::INT64);
        int64_t val = myrank == 0 ? 0 : (n / 2) + (n % 2);
        for (size_t i = 0; i < n; i++) {
            if (i == n / 2) {
                val = 1;
            } else {
                val += 1;
            }
            getv<int64_t>(expected_out, i) = val;
        }

        verify_sorted_window_output(Bodo_FTypes::row_number, partition_by_arrs,
                                    order_by_arrs, {}, {}, out_arrs,
                                    expected_out);
    });

    bodo::tests::test("test_local_rank", [] {
        // verifies that rank works locally
        std::vector<std::shared_ptr<array_info>> partition_by_arrs;
        std::vector<std::shared_ptr<array_info>> order_by_arrs;
        std::vector<std::shared_ptr<array_info>> out_arrs;
        size_t n = 7;
        size_t n_partition_cols = 1;

        int myrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        std::shared_ptr<array_info> order_by_col =
            bodo::tests::cppToBodoArr({1, 1, 1, 2, 2, 3, 4});
        std::shared_ptr<array_info> expected_out =
            bodo::tests::cppToBodoArr({1, 1, 1, 4, 4, 6, 7});

        for (size_t i = 0; i < n_partition_cols; i++) {
            partition_by_arrs.push_back(
                alloc_nullable_array_all_nulls(n, Bodo_CTypes::INT8));
        }
        order_by_arrs.push_back(order_by_col);

        out_arrs.push_back(alloc_numpy(n, Bodo_CTypes::INT64));

        verify_sorted_window_output(Bodo_FTypes::rank, partition_by_arrs,
                                    order_by_arrs, {}, {}, out_arrs,
                                    expected_out, false);
    });

    bodo::tests::test("test_par_rank_orders_match", [] {
        // tests case where orders/groups match across ranks and each rank has
        // multiple orders on it orders look something like: rank 0: 0 1 1, rank
        // 1: 1 2 2, rank 2: 2 3 3, ...
        std::vector<std::shared_ptr<array_info>> partition_by_arrs;
        std::vector<std::shared_ptr<array_info>> order_by_arrs;
        std::vector<std::shared_ptr<array_info>> out_arrs;
        size_t n = 3;
        size_t n_partition_cols = 1;

        int myrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        int64_t k = myrank + 1;
        int64_t first_val = k == 1 ? 1 : 3 * (k - 1) - 1;
        int64_t second_val = k == 1 ? first_val + 1 : first_val + 3;
        std::shared_ptr<array_info> order_by_col =
            bodo::tests::cppToBodoArr({k - 1, k, k});
        std::shared_ptr<array_info> expected_out =
            bodo::tests::cppToBodoArr({first_val, second_val, second_val});

        for (size_t i = 0; i < n_partition_cols; i++) {
            partition_by_arrs.push_back(
                alloc_nullable_array_all_nulls(n, Bodo_CTypes::INT64));
        }
        order_by_arrs.push_back(order_by_col);

        out_arrs.push_back(alloc_numpy(n, Bodo_CTypes::INT64));

        verify_sorted_window_output(Bodo_FTypes::rank, partition_by_arrs,
                                    order_by_arrs, {}, {}, out_arrs,
                                    expected_out);
    });

    bodo::tests::test("test_par_rank_single_orders_match", [] {
        // tests case where each rank has one order and it matches with
        // with the neighbor to the left. i.e. it looks like:
        // rank 0: 0 0 0, rank 1: 0 0 0, rank 2: 1 1 1, ...
        std::vector<std::shared_ptr<array_info>> partition_by_arrs;
        std::vector<std::shared_ptr<array_info>> order_by_arrs;
        std::vector<std::shared_ptr<array_info>> out_arrs;
        size_t n = 3;
        size_t n_partition_cols = 1;

        int myrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        int64_t k = myrank / 2;
        int64_t window_rank = k * 6 + 1;
        std::shared_ptr<array_info> order_by_col =
            bodo::tests::cppToBodoArr({k, k, k});
        std::shared_ptr<array_info> expected_out =
            bodo::tests::cppToBodoArr({window_rank, window_rank, window_rank});

        for (size_t i = 0; i < n_partition_cols; i++) {
            partition_by_arrs.push_back(
                alloc_nullable_array_all_nulls(n, Bodo_CTypes::INT64));
        }
        order_by_arrs.push_back(order_by_col);

        out_arrs.push_back(alloc_numpy(n, Bodo_CTypes::INT64));

        verify_sorted_window_output(Bodo_FTypes::rank, partition_by_arrs,
                                    order_by_arrs, {}, {}, out_arrs,
                                    expected_out);
    });
    bodo::tests::test("test_par_rank_multiple_groups", [] {
        // tests rank in the case of multiple groups on each rank
        // same set up as test_par_row_number_multiple_groups_per_rank
        std::vector<std::shared_ptr<array_info>> out_arrs;
        size_t n = 4;
        size_t n_partition_cols = 1;
        size_t n_order_by_cols = 1;

        int myrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        auto [partition_by_arrs, order_by_arrs] =
            create_int_example_multiple_groups(n_partition_cols,
                                               n_order_by_cols);

        out_arrs.push_back(alloc_numpy(n, Bodo_CTypes::INT64));

        // generate expected output
        std::vector<int64_t> expected_out_vec = {1, 2, 1, 2};
        // apply expected offsets
        if (myrank > 0) {
            expected_out_vec[0] += 1;
            expected_out_vec[1] += 2;
        }
        std::shared_ptr<array_info> expected_out =
            bodo::tests::cppToBodoArr(expected_out_vec);

        verify_sorted_window_output(Bodo_FTypes::rank, partition_by_arrs,
                                    order_by_arrs, {}, {}, out_arrs,
                                    expected_out);
    });
    bodo::tests::test("test_par_rank_with_holes", [] {
        // same set up as test_par_rank_multiple_groups except every other rank
        // is empty
        std::vector<std::shared_ptr<array_info>> out_arrs;
        size_t n_partition_cols = 1;
        size_t n_order_by_cols = 1;

        int myrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        size_t n = myrank % 2 == 0 ? 4 : 0;

        std::shared_ptr<array_info> empty_arr =
            alloc_numpy(0, Bodo_CTypes::INT64);

        auto [partition_by_arrs, order_by_arrs] =
            create_int_example_with_holes(n_partition_cols, n_order_by_cols);
        out_arrs.push_back(alloc_numpy(n, Bodo_CTypes::INT64));

        std::shared_ptr<array_info> expected_out;
        if (n == 0) {
            expected_out = empty_arr;
        } else {
            // generate expected output
            std::vector<int64_t> expected_out_vec = {1, 2, 1, 2};
            // apply expected offsets
            if (myrank > 0) {
                expected_out_vec[0] += 2;
                expected_out_vec[1] += 2;
            }
            expected_out = bodo::tests::cppToBodoArr(expected_out_vec);
        }

        verify_sorted_window_output(Bodo_FTypes::rank, partition_by_arrs,
                                    order_by_arrs, {}, {}, out_arrs,
                                    expected_out);
    });
    bodo::tests::test("test_par_rank_single_order_multiple_groups", [] {
        // test where the orderby's are all the same and the groups are
        // different
        std::vector<std::shared_ptr<array_info>> partition_by_arrs;
        std::vector<std::shared_ptr<array_info>> order_by_arrs;
        std::vector<std::shared_ptr<array_info>> out_arrs;
        size_t n = 3;
        int64_t val = 4;

        int myrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        // each rank has a different partition
        partition_by_arrs.push_back(const_int64_arr(n, myrank));
        std::shared_ptr<array_info> order_by_arr = const_int64_arr(n, val);

        // every rank has the same orderby val
        order_by_arrs.push_back(order_by_arr);
        std::shared_ptr<array_info> expected_out = const_int64_arr(n, 1);

        out_arrs.push_back(alloc_numpy(n, Bodo_CTypes::INT64));

        verify_sorted_window_output(Bodo_FTypes::rank, partition_by_arrs,
                                    order_by_arrs, {}, {}, out_arrs,
                                    expected_out);
    });
    bodo::tests::test("test_local_dense_rank", [] {
        // verifies that rank works locally
        std::vector<std::shared_ptr<array_info>> partition_by_arrs;
        std::vector<std::shared_ptr<array_info>> order_by_arrs;
        std::vector<std::shared_ptr<array_info>> out_arrs;
        size_t n = 7;
        size_t n_partition_cols = 1;

        int myrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        std::shared_ptr<array_info> order_by_col =
            bodo::tests::cppToBodoArr({1, 1, 1, 2, 2, 3, 4});
        std::shared_ptr<array_info> expected_out =
            bodo::tests::cppToBodoArr({1, 1, 1, 2, 2, 3, 4});

        for (size_t i = 0; i < n_partition_cols; i++) {
            partition_by_arrs.push_back(
                alloc_nullable_array_all_nulls(n, Bodo_CTypes::INT8));
        }
        order_by_arrs.push_back(order_by_col);

        out_arrs.push_back(alloc_numpy(n, Bodo_CTypes::INT64));

        verify_sorted_window_output(Bodo_FTypes::dense_rank, partition_by_arrs,
                                    order_by_arrs, {}, {}, out_arrs,
                                    expected_out, false);
    });

    bodo::tests::test(
        "calculate_single_function-row_number-single_chunk-single_rank", [] {
            // Ensure we spill on unpin to verify the correctness of the
            // pinning/unpinning behavior.
            bodo::BufferPoolOptions options;
            options.spill_on_unpin = true;
            bodo::BufferPool pool = bodo::BufferPool(options);
            std::shared_ptr<::arrow::MemoryManager> mm =
                buffer_memory_manager(&pool);

            std::vector<std::shared_ptr<table_info>> in_chunks;
            std::vector<int32_t> partition_col_indices;
            std::vector<int32_t> order_col_indices;
            std::vector<int32_t> keep_indices;
            std::vector<std::vector<int32_t>> input_col_indices;
            std::vector<int32_t> window_funcs;
            std::vector<std::shared_ptr<table_info>> expected_out_chunks;

            partition_col_indices.push_back(0);
            order_col_indices.push_back(1);
            keep_indices.push_back(0);
            keep_indices.push_back(2);
            input_col_indices.push_back({});
            window_funcs.push_back(Bodo_FTypes::row_number);

            std::shared_ptr<array_info> partition =
                bodo::tests::cppToBodoArr({1, 1, 1, 2, 2, 3, 4, 5, 5, 5, 5, 5});
            std::shared_ptr<array_info> order = bodo::tests::cppToBodoArr(
                {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
            std::shared_ptr<array_info> misc = bodo::tests::cppToBodoArr(
                {1, 20, 3, 40, 5, 60, 7, 80, 9, 100, 11, 120});
            std::shared_ptr<array_info> row_number =
                bodo::tests::cppToBodoArr<uint64_t>(
                    {1, 2, 3, 1, 2, 1, 1, 1, 2, 3, 4, 5});

            std::vector<std::shared_ptr<array_info>> chunk_cols = {partition,
                                                                   order, misc};
            std::shared_ptr<table_info> chunk =
                std::make_shared<table_info>(table_info(chunk_cols));
            in_chunks.push_back(chunk);

            std::vector<std::shared_ptr<array_info>> out_chunk_cols = {
                partition, misc, row_number};
            std::shared_ptr<table_info> out_chunk =
                std::make_shared<table_info>(table_info(out_chunk_cols));
            expected_out_chunks.push_back(out_chunk);

            // Test with the templated arguments provided, and without them.
            verify_window_calculators(
                in_chunks, partition_col_indices, order_col_indices,
                keep_indices, input_col_indices, window_funcs,
                bodo_array_type::NUMPY, bodo_array_type::NUMPY,
                expected_out_chunks, false, &pool, mm);
            verify_window_calculators(
                in_chunks, partition_col_indices, order_col_indices,
                keep_indices, input_col_indices, window_funcs,
                bodo_array_type::UNKNOWN, bodo_array_type::UNKNOWN,
                expected_out_chunks, false, &pool, mm);
        });

    bodo::tests::test(
        "calculate_multiple_function-all_row_number-single_chunk-single_rank",
        [] {
            // Ensure we spill on unpin to verify the correctness of the
            // pinning/unpinning behavior.
            bodo::BufferPoolOptions options;
            options.spill_on_unpin = true;
            bodo::BufferPool pool = bodo::BufferPool(options);
            std::shared_ptr<::arrow::MemoryManager> mm =
                buffer_memory_manager(&pool);

            std::vector<std::shared_ptr<table_info>> in_chunks;
            std::vector<int32_t> partition_col_indices;
            std::vector<int32_t> order_col_indices;
            std::vector<int32_t> keep_indices;
            std::vector<std::vector<int32_t>> input_col_indices;
            std::vector<int32_t> window_funcs;
            std::vector<std::shared_ptr<table_info>> expected_out_chunks;

            partition_col_indices.push_back(0);
            order_col_indices.push_back(1);
            keep_indices.push_back(0);
            keep_indices.push_back(1);
            keep_indices.push_back(2);
            input_col_indices.push_back({});
            input_col_indices.push_back({});
            input_col_indices.push_back({});
            window_funcs.push_back(Bodo_FTypes::row_number);
            window_funcs.push_back(Bodo_FTypes::row_number);
            window_funcs.push_back(Bodo_FTypes::row_number);

            std::shared_ptr<array_info> partition =
                bodo::tests::cppToBodoArr({1, 1, 1, 2, 2, 3, 4, 5, 5, 5, 5, 5});
            std::shared_ptr<array_info> order = bodo::tests::cppToBodoArr(
                {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
            std::shared_ptr<array_info> misc = bodo::tests::cppToBodoArr(
                {1, 20, 3, 40, 5, 60, 7, 80, 9, 100, 11, 120});
            std::shared_ptr<array_info> row_number =
                bodo::tests::cppToBodoArr<uint64_t>(
                    {1, 2, 3, 1, 2, 1, 1, 1, 2, 3, 4, 5});

            std::vector<std::shared_ptr<array_info>> chunk_cols = {partition,
                                                                   order, misc};
            std::shared_ptr<table_info> chunk =
                std::make_shared<table_info>(table_info(chunk_cols));
            in_chunks.push_back(chunk);

            std::vector<std::shared_ptr<array_info>> out_cols = {
                partition, order, misc, row_number, row_number, row_number};
            std::shared_ptr<table_info> out_chunk =
                std::make_shared<table_info>(table_info(out_cols));
            expected_out_chunks.push_back(out_chunk);

            // Test with the templated arguments provided, and without them.
            verify_window_calculators(
                in_chunks, partition_col_indices, order_col_indices,
                keep_indices, input_col_indices, window_funcs,
                bodo_array_type::NUMPY, bodo_array_type::NUMPY,
                expected_out_chunks, false, &pool, mm);
            verify_window_calculators(
                in_chunks, partition_col_indices, order_col_indices,
                keep_indices, input_col_indices, window_funcs,
                bodo_array_type::UNKNOWN, bodo_array_type::UNKNOWN,
                expected_out_chunks, false, &pool, mm);
        });

    bodo::tests::test(
        "calculate_single_function-row_number-multi_chunk-single_rank", [] {
            // Ensure we spill on unpin to verify the correctness of the
            // pinning/unpinning behavior.
            bodo::BufferPoolOptions options;
            options.spill_on_unpin = true;
            bodo::BufferPool pool = bodo::BufferPool(options);
            std::shared_ptr<::arrow::MemoryManager> mm =
                buffer_memory_manager(&pool);

            std::vector<std::shared_ptr<table_info>> in_chunks;
            std::vector<int32_t> partition_col_indices;
            std::vector<int32_t> order_col_indices;
            std::vector<int32_t> keep_indices;
            std::vector<std::vector<int32_t>> input_col_indices;
            std::vector<int32_t> window_funcs;
            std::vector<std::shared_ptr<table_info>> expected_out_chunks;

            partition_col_indices.push_back(0);
            order_col_indices.push_back(1);
            keep_indices.push_back(1);
            input_col_indices.push_back({});
            window_funcs.push_back(Bodo_FTypes::row_number);

            // Chunk 0: all the same partition
            {
                std::shared_ptr<array_info> partition =
                    bodo::tests::cppToBodoArr({1, 1, 1, 1, 1});
                std::shared_ptr<array_info> order =
                    bodo::tests::cppToBodoArr({10, 20, 30, 40, 50});
                std::shared_ptr<array_info> row_number =
                    bodo::tests::cppToBodoArr<uint64_t>({1, 2, 3, 4, 5});

                std::vector<std::shared_ptr<array_info>> chunk_cols = {
                    partition, order};
                std::shared_ptr<table_info> chunk =
                    std::make_shared<table_info>(table_info(chunk_cols));
                in_chunks.push_back(chunk);

                std::vector<std::shared_ptr<array_info>> out_cols = {
                    order, row_number};
                std::shared_ptr<table_info> out_chunk =
                    std::make_shared<table_info>(table_info(out_cols));
                expected_out_chunks.push_back(out_chunk);
            }

            // Chunk 1: all the same partition (different from chunk 0)
            {
                std::shared_ptr<array_info> partition =
                    bodo::tests::cppToBodoArr({2, 2, 2, 2, 2});
                std::shared_ptr<array_info> order =
                    bodo::tests::cppToBodoArr({60, 70, 80, 90, 100});
                std::shared_ptr<array_info> row_number =
                    bodo::tests::cppToBodoArr<uint64_t>({1, 2, 3, 4, 5});

                std::vector<std::shared_ptr<array_info>> chunk_cols = {
                    partition, order};
                std::shared_ptr<table_info> chunk =
                    std::make_shared<table_info>(table_info(chunk_cols));
                in_chunks.push_back(chunk);

                std::vector<std::shared_ptr<array_info>> out_cols = {
                    order, row_number};
                std::shared_ptr<table_info> out_chunk =
                    std::make_shared<table_info>(table_info(out_cols));
                expected_out_chunks.push_back(out_chunk);
            }

            // Chunk 2: all the same partition (same as chunk 1)
            {
                std::shared_ptr<array_info> partition =
                    bodo::tests::cppToBodoArr({2, 2, 2, 2, 2});
                std::shared_ptr<array_info> order =
                    bodo::tests::cppToBodoArr({110, 120, 130, 140, 150});
                std::shared_ptr<array_info> row_number =
                    bodo::tests::cppToBodoArr<uint64_t>({6, 7, 8, 9, 10});

                std::vector<std::shared_ptr<array_info>> chunk_cols = {
                    partition, order};
                std::shared_ptr<table_info> chunk =
                    std::make_shared<table_info>(table_info(chunk_cols));
                in_chunks.push_back(chunk);

                std::vector<std::shared_ptr<array_info>> out_cols = {
                    order, row_number};
                std::shared_ptr<table_info> out_chunk =
                    std::make_shared<table_info>(table_info(out_cols));
                expected_out_chunks.push_back(out_chunk);
            }

            // Chunk 3: two different partitions (first is the same as chunk 2)
            {
                std::shared_ptr<array_info> partition =
                    bodo::tests::cppToBodoArr({2, 2, 3, 3, 3});
                std::shared_ptr<array_info> order =
                    bodo::tests::cppToBodoArr({151, 152, 153, 154, 155});
                std::shared_ptr<array_info> row_number =
                    bodo::tests::cppToBodoArr<uint64_t>({11, 12, 1, 2, 3});

                std::vector<std::shared_ptr<array_info>> chunk_cols = {
                    partition, order};
                std::shared_ptr<table_info> chunk =
                    std::make_shared<table_info>(table_info(chunk_cols));
                in_chunks.push_back(chunk);

                std::vector<std::shared_ptr<array_info>> out_cols = {
                    order, row_number};
                std::shared_ptr<table_info> out_chunk =
                    std::make_shared<table_info>(table_info(out_cols));
                expected_out_chunks.push_back(out_chunk);
            }

            // Chunk 4: five different partitions (first is the same as chunk 3)
            {
                std::shared_ptr<array_info> partition =
                    bodo::tests::cppToBodoArr({3, 4, 5, 6, 7});
                std::shared_ptr<array_info> order =
                    bodo::tests::cppToBodoArr({156, 157, 158, 159, 160});
                std::shared_ptr<array_info> row_number =
                    bodo::tests::cppToBodoArr<uint64_t>({4, 1, 1, 1, 1});

                std::vector<std::shared_ptr<array_info>> chunk_cols = {
                    partition, order};
                std::shared_ptr<table_info> chunk =
                    std::make_shared<table_info>(table_info(chunk_cols));
                in_chunks.push_back(chunk);

                std::vector<std::shared_ptr<array_info>> out_cols = {
                    order, row_number};
                std::shared_ptr<table_info> out_chunk =
                    std::make_shared<table_info>(table_info(out_cols));
                expected_out_chunks.push_back(out_chunk);
            }

            verify_window_calculators(
                in_chunks, partition_col_indices, order_col_indices,
                keep_indices, input_col_indices, window_funcs,
                bodo_array_type::NUMPY, bodo_array_type::NUMPY,
                expected_out_chunks, false, &pool, mm);
        });

    bodo::tests::test(
        "calculate_single_function-row_number-single_chunk-multi_rank-single_"
        "partition-holes",
        [] {
            // Ensure we spill on unpin to verify the correctness of the
            // pinning/unpinning behavior.
            bodo::BufferPoolOptions options;
            options.spill_on_unpin = true;
            bodo::BufferPool pool = bodo::BufferPool(options);
            std::shared_ptr<::arrow::MemoryManager> mm =
                buffer_memory_manager(&pool);

            int myrank;
            MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

            // The ROW_NUMBER values from previous ranks includes 3 rows from
            // each preceding odd-numbered rank.
            size_t row_number_offset = 0;
            for (int i = 1; i < myrank; i += 2) {
                row_number_offset += 3;
            }

            std::vector<std::shared_ptr<table_info>> in_chunks;
            std::vector<int32_t> partition_col_indices;
            std::vector<int32_t> order_col_indices;
            std::vector<int32_t> keep_indices;
            std::vector<std::vector<int32_t>> input_col_indices;
            std::vector<int32_t> window_funcs;
            std::vector<std::shared_ptr<table_info>> expected_out_chunks;

            partition_col_indices.push_back(0);
            order_col_indices.push_back(1);
            keep_indices.push_back(1);
            input_col_indices.push_back({});
            window_funcs.push_back(Bodo_FTypes::row_number);

            std::shared_ptr<array_info> partition;
            std::shared_ptr<array_info> order;
            std::shared_ptr<array_info> row_number;
            if (myrank % 2 == 0) {
                // If this is an even-number partition, allocate all-empty.
                partition = bodo::tests::cppToBodoArr<int64_t>({});
                order = bodo::tests::cppToBodoArr<uint64_t>({});
                row_number = bodo::tests::cppToBodoArr<uint64_t>({});
            } else {
                // If this is an odd-number partition, allocate 3 rows.
                partition = bodo::tests::cppToBodoArr<int64_t>({1, 1, 1});
                order = bodo::tests::cppToBodoArr<uint64_t>(
                    {2 * row_number_offset, 2 * row_number_offset + 2,
                     2 * row_number_offset + 3});
                row_number = bodo::tests::cppToBodoArr<uint64_t>(
                    {row_number_offset + 1, row_number_offset + 2,
                     row_number_offset + 3});
            }

            std::vector<std::shared_ptr<array_info>> chunk_cols = {partition,
                                                                   order};
            std::shared_ptr<table_info> chunk =
                std::make_shared<table_info>(table_info(chunk_cols));
            in_chunks.push_back(chunk);

            std::vector<std::shared_ptr<array_info>> out_cols = {order,
                                                                 row_number};
            std::shared_ptr<table_info> out_chunk =
                std::make_shared<table_info>(table_info(out_cols));
            expected_out_chunks.push_back(out_chunk);

            verify_window_calculators(
                in_chunks, partition_col_indices, order_col_indices,
                keep_indices, input_col_indices, window_funcs,
                bodo_array_type::UNKNOWN, bodo_array_type::UNKNOWN,
                expected_out_chunks, true, &pool, mm);
        });

    bodo::tests::test(
        "calculate_single_function-row_number-multi_chunk-multi_rank-single_"
        "partition",
        [] {
            // Ensure we spill on unpin to verify the correctness of the
            // pinning/unpinning behavior.
            bodo::BufferPoolOptions options;
            options.spill_on_unpin = true;
            bodo::BufferPool pool = bodo::BufferPool(options);
            std::shared_ptr<::arrow::MemoryManager> mm =
                buffer_memory_manager(&pool);

            int myrank;
            MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

            size_t row_number_offset = static_cast<size_t>(myrank) * 5;

            std::vector<std::shared_ptr<table_info>> in_chunks;
            std::vector<int32_t> partition_col_indices;
            std::vector<int32_t> order_col_indices;
            std::vector<int32_t> keep_indices;
            std::vector<std::vector<int32_t>> input_col_indices;
            std::vector<int32_t> window_funcs;
            std::vector<std::shared_ptr<table_info>> expected_out_chunks;

            partition_col_indices.push_back(0);
            order_col_indices.push_back(1);
            keep_indices.push_back(1);
            input_col_indices.push_back({});
            window_funcs.push_back(Bodo_FTypes::row_number);

            // Chunk 0: all the same partition
            {
                std::shared_ptr<array_info> partition =
                    bodo::tests::cppToBodoArr({1, 1, 1});
                std::shared_ptr<array_info> order =
                    bodo::tests::cppToBodoArr<uint64_t>(
                        {2 * row_number_offset, 2 * row_number_offset + 2,
                         2 * row_number_offset + 3});
                std::shared_ptr<array_info> row_number =
                    bodo::tests::cppToBodoArr<uint64_t>(
                        {row_number_offset + 1, row_number_offset + 2,
                         row_number_offset + 3});

                std::vector<std::shared_ptr<array_info>> chunk_cols = {
                    partition, order};
                std::shared_ptr<table_info> chunk =
                    std::make_shared<table_info>(table_info(chunk_cols));
                in_chunks.push_back(chunk);

                std::vector<std::shared_ptr<array_info>> out_cols = {
                    order, row_number};
                std::shared_ptr<table_info> out_chunk =
                    std::make_shared<table_info>(table_info(out_cols));
                expected_out_chunks.push_back(out_chunk);
            }

            // Chunk 1: all the same partition as chunk 0 (shared across both
            // ranks)
            {
                std::shared_ptr<array_info> partition =
                    bodo::tests::cppToBodoArr({1, 1});
                std::shared_ptr<array_info> order =
                    bodo::tests::cppToBodoArr<uint64_t>(
                        {2 * row_number_offset + 5, 2 * row_number_offset + 6});
                std::shared_ptr<array_info> row_number =
                    bodo::tests::cppToBodoArr<uint64_t>(
                        {row_number_offset + 4, row_number_offset + 5});

                std::vector<std::shared_ptr<array_info>> chunk_cols = {
                    partition, order};
                std::shared_ptr<table_info> chunk =
                    std::make_shared<table_info>(table_info(chunk_cols));
                in_chunks.push_back(chunk);

                std::vector<std::shared_ptr<array_info>> out_cols = {
                    order, row_number};
                std::shared_ptr<table_info> out_chunk =
                    std::make_shared<table_info>(table_info(out_cols));
                expected_out_chunks.push_back(out_chunk);
            }

            verify_window_calculators(
                in_chunks, partition_col_indices, order_col_indices,
                keep_indices, input_col_indices, window_funcs,
                bodo_array_type::UNKNOWN, bodo_array_type::UNKNOWN,
                expected_out_chunks, true, &pool, mm);
        });

    bodo::tests::test(
        "calculate_single_function-row_number-multi_chunk-multi_rank-"
        "partitionless",
        [] {
            // Ensure we spill on unpin to verify the correctness of the
            // pinning/unpinning behavior.
            bodo::BufferPoolOptions options;
            options.spill_on_unpin = true;
            bodo::BufferPool pool = bodo::BufferPool(options);
            std::shared_ptr<::arrow::MemoryManager> mm =
                buffer_memory_manager(&pool);

            int myrank;
            MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

            size_t row_number_offset = static_cast<size_t>(myrank) * 5;

            std::vector<std::shared_ptr<table_info>> in_chunks;
            std::vector<int32_t> partition_col_indices;
            std::vector<int32_t> order_col_indices;
            std::vector<int32_t> keep_indices;
            std::vector<std::vector<int32_t>> input_col_indices;
            std::vector<int32_t> window_funcs;
            std::vector<std::shared_ptr<table_info>> expected_out_chunks;

            order_col_indices.push_back(0);
            keep_indices.push_back(0);
            input_col_indices.push_back({});
            window_funcs.push_back(Bodo_FTypes::row_number);

            // Chunk 0:
            {
                std::shared_ptr<array_info> order =
                    bodo::tests::cppToBodoArr<uint64_t>(
                        {2 * row_number_offset, 2 * row_number_offset + 2,
                         2 * row_number_offset + 3});
                std::shared_ptr<array_info> row_number =
                    bodo::tests::cppToBodoArr<uint64_t>(
                        {row_number_offset + 1, row_number_offset + 2,
                         row_number_offset + 3});

                std::vector<std::shared_ptr<array_info>> chunk_cols = {order};
                std::shared_ptr<table_info> chunk =
                    std::make_shared<table_info>(table_info(chunk_cols));
                in_chunks.push_back(chunk);

                std::vector<std::shared_ptr<array_info>> out_cols = {
                    order, row_number};
                std::shared_ptr<table_info> out_chunk =
                    std::make_shared<table_info>(table_info(out_cols));
                expected_out_chunks.push_back(out_chunk);
            }

            // Chunk 1
            {
                std::shared_ptr<array_info> order =
                    bodo::tests::cppToBodoArr<uint64_t>(
                        {2 * row_number_offset + 5, 2 * row_number_offset + 6});
                std::shared_ptr<array_info> row_number =
                    bodo::tests::cppToBodoArr<uint64_t>(
                        {row_number_offset + 4, row_number_offset + 5});

                std::vector<std::shared_ptr<array_info>> chunk_cols = {order};
                std::shared_ptr<table_info> chunk =
                    std::make_shared<table_info>(table_info(chunk_cols));
                in_chunks.push_back(chunk);

                std::vector<std::shared_ptr<array_info>> out_cols = {
                    order, row_number};
                std::shared_ptr<table_info> out_chunk =
                    std::make_shared<table_info>(table_info(out_cols));
                expected_out_chunks.push_back(out_chunk);
            }

            verify_window_calculators(
                in_chunks, partition_col_indices, order_col_indices,
                keep_indices, input_col_indices, window_funcs,
                bodo_array_type::UNKNOWN, bodo_array_type::UNKNOWN,
                expected_out_chunks, true, &pool, mm);
        });

    bodo::tests::test(
        "calculate_single_function-row_number-single_chunk-multi_rank-multiple_"
        "partition",
        [] {
            // Ensure we spill on unpin to verify the correctness of the
            // pinning/unpinning behavior.
            bodo::BufferPoolOptions options;
            options.spill_on_unpin = true;
            bodo::BufferPool pool = bodo::BufferPool(options);
            std::shared_ptr<::arrow::MemoryManager> mm =
                buffer_memory_manager(&pool);

            int myrank, num_ranks;
            MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
            MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

            size_t local_rows = 5;
            size_t total_rows = static_cast<size_t>(num_ranks) * local_rows;

            // Test with every location for where the split between partition A
            // vs partition B can possibly be located.
            for (size_t partition_cutoff = 1; partition_cutoff < total_rows;
                 partition_cutoff++) {
                size_t previous_ranks_rows =
                    static_cast<size_t>(myrank) * local_rows;

                std::vector<std::shared_ptr<table_info>> in_chunks;
                std::vector<int32_t> partition_col_indices;
                std::vector<int32_t> order_col_indices;
                std::vector<int32_t> keep_indices;
                std::vector<std::vector<int32_t>> input_col_indices;
                std::vector<int32_t> window_funcs;
                std::vector<std::shared_ptr<table_info>> expected_out_chunks;

                partition_col_indices.push_back(0);
                order_col_indices.push_back(1);
                keep_indices.push_back(0);
                keep_indices.push_back(1);
                input_col_indices.push_back({});
                window_funcs.push_back(Bodo_FTypes::row_number);

                std::vector<int64_t> partition_vect;
                std::vector<uint64_t> order_vect;
                std::vector<uint64_t> row_number_vect;

                for (size_t local_row = 0; local_row < local_rows;
                     local_row++) {
                    size_t global_row = local_row + previous_ranks_rows;
                    if ((global_row < partition_cutoff)) {
                        partition_vect.push_back(42);
                        row_number_vect.push_back(global_row + 1);
                    } else {
                        partition_vect.push_back(64);
                        row_number_vect.push_back(global_row + 1 -
                                                  partition_cutoff);
                    }
                    order_vect.push_back(global_row << 2);
                }

                std::shared_ptr<array_info> partition =
                    bodo::tests::cppToBodoArr(partition_vect);
                std::shared_ptr<array_info> order =
                    bodo::tests::cppToBodoArr<uint64_t>(order_vect);
                std::shared_ptr<array_info> row_number =
                    bodo::tests::cppToBodoArr<uint64_t>(row_number_vect);

                std::vector<std::shared_ptr<array_info>> chunk_cols = {
                    partition, order};
                std::shared_ptr<table_info> chunk =
                    std::make_shared<table_info>(table_info(chunk_cols));
                in_chunks.push_back(chunk);

                std::vector<std::shared_ptr<array_info>> out_cols = {
                    partition, order, row_number};
                std::shared_ptr<table_info> out_chunk =
                    std::make_shared<table_info>(table_info(out_cols));
                expected_out_chunks.push_back(out_chunk);

                verify_window_calculators(
                    in_chunks, partition_col_indices, order_col_indices,
                    keep_indices, input_col_indices, window_funcs,
                    bodo_array_type::NUMPY, bodo_array_type::NUMPY,
                    expected_out_chunks, true, &pool, mm);
            }
        });

    bodo::tests::test(
        "calculate_single_function-row_number-multi_chunk-multi_rank-multiple_"
        "partition",
        [] {
            // Ensure we spill on unpin to verify the correctness of the
            // pinning/unpinning behavior.
            bodo::BufferPoolOptions options;
            options.spill_on_unpin = true;
            bodo::BufferPool pool = bodo::BufferPool(options);
            std::shared_ptr<::arrow::MemoryManager> mm =
                buffer_memory_manager(&pool);

            int myrank, num_ranks;
            MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
            MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

            size_t local_rows = 5;
            size_t total_rows = static_cast<size_t>(num_ranks) * local_rows;

            // Test with every location for where the split between partition A
            // vs partition B can possibly be located.
            for (size_t partition_cutoff = 1; partition_cutoff < total_rows;
                 partition_cutoff++) {
                size_t previous_ranks_rows =
                    static_cast<size_t>(myrank) * local_rows;

                std::vector<std::shared_ptr<table_info>> in_chunks;
                std::vector<int32_t> partition_col_indices;
                std::vector<int32_t> order_col_indices;
                std::vector<int32_t> keep_indices;
                std::vector<std::vector<int32_t>> input_col_indices;
                std::vector<int32_t> window_funcs;
                std::vector<std::shared_ptr<table_info>> expected_out_chunks;

                partition_col_indices.push_back(0);
                order_col_indices.push_back(1);
                keep_indices.push_back(0);
                keep_indices.push_back(1);
                input_col_indices.push_back({});
                window_funcs.push_back(Bodo_FTypes::row_number);

                // Split up each row into a singleton chunk
                for (size_t local_row = 0; local_row < local_rows;
                     local_row++) {
                    std::vector<int64_t> partition_vect;
                    std::vector<uint64_t> order_vect;
                    std::vector<uint64_t> row_number_vect;
                    size_t global_row = local_row + previous_ranks_rows;
                    if ((global_row < partition_cutoff)) {
                        partition_vect.push_back(42);
                        row_number_vect.push_back(global_row + 1);
                    } else {
                        partition_vect.push_back(64);
                        row_number_vect.push_back(global_row + 1 -
                                                  partition_cutoff);
                    }
                    order_vect.push_back(global_row << 2);

                    std::shared_ptr<array_info> partition =
                        bodo::tests::cppToBodoArr(partition_vect);
                    std::shared_ptr<array_info> order =
                        bodo::tests::cppToBodoArr<uint64_t>(order_vect);
                    std::shared_ptr<array_info> row_number =
                        bodo::tests::cppToBodoArr<uint64_t>(row_number_vect);

                    std::vector<std::shared_ptr<array_info>> chunk_cols = {
                        partition, order};
                    std::shared_ptr<table_info> chunk =
                        std::make_shared<table_info>(table_info(chunk_cols));
                    in_chunks.push_back(chunk);

                    std::vector<std::shared_ptr<array_info>> out_cols = {
                        partition, order, row_number};
                    std::shared_ptr<table_info> out_chunk =
                        std::make_shared<table_info>(table_info(out_cols));
                    expected_out_chunks.push_back(out_chunk);
                }

                verify_window_calculators(
                    in_chunks, partition_col_indices, order_col_indices,
                    keep_indices, input_col_indices, window_funcs,
                    bodo_array_type::NUMPY, bodo_array_type::NUMPY,
                    expected_out_chunks, true, &pool, mm);
            }
        });

    bodo::tests::test(
        "calculate_single_function-sum-single_chunk-single_rank", [] {
            // Ensure we spill on unpin to verify the correctness of the
            // pinning/unpinning behavior.
            bodo::BufferPoolOptions options;
            options.spill_on_unpin = true;
            bodo::BufferPool pool = bodo::BufferPool(options);
            std::shared_ptr<::arrow::MemoryManager> mm =
                buffer_memory_manager(&pool);

            std::vector<std::shared_ptr<table_info>> in_chunks;
            std::vector<int32_t> partition_col_indices;
            std::vector<int32_t> order_col_indices;
            std::vector<int32_t> keep_indices;
            std::vector<std::vector<int32_t>> input_col_indices;
            std::vector<int32_t> window_funcs;
            std::vector<std::shared_ptr<table_info>> expected_out_chunks;

            partition_col_indices.push_back(0);
            keep_indices.push_back(0);
            keep_indices.push_back(1);
            input_col_indices.push_back({1});
            window_funcs.push_back(Bodo_FTypes::sum);

            std::shared_ptr<array_info> partition =
                bodo::tests::cppToBodoArr({1, 1, 1, 2, 2, 3, 4, 5, 5, 5, 5, 5});
            std::shared_ptr<array_info> input = bodo::tests::cppToBodoArr(
                {1, -1, 3, 4, 5, -1, 7, 8, 9, 10, 11, 12}, true);
            input->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(1, false);
            input->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(5, false);
            std::shared_ptr<array_info> sum = bodo::tests::cppToBodoArr(
                {4, 4, 4, 9, 9, -1, 7, 50, 50, 50, 50, 50}, true);
            sum->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(5, false);

            std::vector<std::shared_ptr<array_info>> chunk_cols = {partition,
                                                                   input};
            std::shared_ptr<table_info> chunk =
                std::make_shared<table_info>(table_info(chunk_cols));
            in_chunks.push_back(chunk);

            std::vector<std::shared_ptr<array_info>> out_chunk_cols = {
                partition, input, sum};
            std::shared_ptr<table_info> out_chunk =
                std::make_shared<table_info>(table_info(out_chunk_cols));
            expected_out_chunks.push_back(out_chunk);

            verify_window_calculators(
                in_chunks, partition_col_indices, order_col_indices,
                keep_indices, input_col_indices, window_funcs,
                bodo_array_type::NUMPY, bodo_array_type::UNKNOWN,
                expected_out_chunks, false, &pool, mm);
        });

    bodo::tests::test(
        "calculate_single_function-sum-multi_chunk-single_rank", [] {
            // Ensure we spill on unpin to verify the correctness of the
            // pinning/unpinning behavior.
            bodo::BufferPoolOptions options;
            options.spill_on_unpin = true;
            bodo::BufferPool pool = bodo::BufferPool(options);
            std::shared_ptr<::arrow::MemoryManager> mm =
                buffer_memory_manager(&pool);

            std::vector<std::shared_ptr<table_info>> in_chunks;
            std::vector<int32_t> partition_col_indices;
            std::vector<int32_t> order_col_indices;
            std::vector<int32_t> keep_indices;
            std::vector<std::vector<int32_t>> input_col_indices;
            std::vector<int32_t> window_funcs;
            std::vector<std::shared_ptr<table_info>> expected_out_chunks;

            partition_col_indices.push_back(0);
            keep_indices.push_back(0);
            keep_indices.push_back(1);
            input_col_indices.push_back({1});
            window_funcs.push_back(Bodo_FTypes::sum);

            // Chunk 0: two partitions
            {
                std::shared_ptr<array_info> partition =
                    bodo::tests::cppToBodoArr({1, 1, 1, 2, 2, 2});
                std::shared_ptr<array_info> input =
                    bodo::tests::cppToBodoArr({1, 2, 3, 4, 5, 6}, true);
                std::shared_ptr<array_info> sum =
                    bodo::tests::cppToBodoArr({6, 6, 6, 15, 15, 15}, true);

                std::vector<std::shared_ptr<array_info>> chunk_cols = {
                    partition, input};
                std::shared_ptr<table_info> chunk =
                    std::make_shared<table_info>(table_info(chunk_cols));
                in_chunks.push_back(chunk);

                std::vector<std::shared_ptr<array_info>> out_chunk_cols = {
                    partition, input, sum};
                std::shared_ptr<table_info> out_chunk =
                    std::make_shared<table_info>(table_info(out_chunk_cols));
                expected_out_chunks.push_back(out_chunk);
            }

            // Chunk 1: two partitions (first overlaps with chunk 0)
            {
                std::shared_ptr<array_info> partition =
                    bodo::tests::cppToBodoArr({2, 2, 2, 3, 3, 3});
                std::shared_ptr<array_info> input =
                    bodo::tests::cppToBodoArr({-1, -1, -1, 7, 8, 9}, true);
                input->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(0,
                                                                        false);
                input->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(1,
                                                                        false);
                input->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(2,
                                                                        false);
                std::shared_ptr<array_info> sum =
                    bodo::tests::cppToBodoArr({15, 15, 15, 45, 45, 45}, true);

                std::vector<std::shared_ptr<array_info>> chunk_cols = {
                    partition, input};
                std::shared_ptr<table_info> chunk =
                    std::make_shared<table_info>(table_info(chunk_cols));
                in_chunks.push_back(chunk);

                std::vector<std::shared_ptr<array_info>> out_chunk_cols = {
                    partition, input, sum};
                std::shared_ptr<table_info> out_chunk =
                    std::make_shared<table_info>(table_info(out_chunk_cols));
                expected_out_chunks.push_back(out_chunk);
            }

            // Chunk 2: one partition (overlaps with chunks 1)
            {
                std::shared_ptr<array_info> partition =
                    bodo::tests::cppToBodoArr({3, 3, 3, 3});
                std::shared_ptr<array_info> input =
                    bodo::tests::cppToBodoArr({-1, -1, -1, -1}, true);
                input->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(0,
                                                                        false);
                input->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(1,
                                                                        false);
                input->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(2,
                                                                        false);
                input->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(3,
                                                                        false);
                std::shared_ptr<array_info> sum =
                    bodo::tests::cppToBodoArr({45, 45, 45, 45}, true);

                std::vector<std::shared_ptr<array_info>> chunk_cols = {
                    partition, input};
                std::shared_ptr<table_info> chunk =
                    std::make_shared<table_info>(table_info(chunk_cols));
                in_chunks.push_back(chunk);

                std::vector<std::shared_ptr<array_info>> out_chunk_cols = {
                    partition, input, sum};
                std::shared_ptr<table_info> out_chunk =
                    std::make_shared<table_info>(table_info(out_chunk_cols));
                expected_out_chunks.push_back(out_chunk);
            }

            // Chunk 3: one partition (overlaps with chunks 1/2)
            {
                std::shared_ptr<array_info> partition =
                    bodo::tests::cppToBodoArr({3, 3, 3});
                std::shared_ptr<array_info> input =
                    bodo::tests::cppToBodoArr({10, 11, -1}, true);
                input->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(2,
                                                                        false);
                std::shared_ptr<array_info> sum =
                    bodo::tests::cppToBodoArr({45, 45, 45}, true);

                std::vector<std::shared_ptr<array_info>> chunk_cols = {
                    partition, input};
                std::shared_ptr<table_info> chunk =
                    std::make_shared<table_info>(table_info(chunk_cols));
                in_chunks.push_back(chunk);

                std::vector<std::shared_ptr<array_info>> out_chunk_cols = {
                    partition, input, sum};
                std::shared_ptr<table_info> out_chunk =
                    std::make_shared<table_info>(table_info(out_chunk_cols));
                expected_out_chunks.push_back(out_chunk);
            }

            // Chunk 4: two partition (first overlaps with chunks 1/2/3)
            {
                std::shared_ptr<array_info> partition =
                    bodo::tests::cppToBodoArr({3, 3, 4, 4});
                std::shared_ptr<array_info> input =
                    bodo::tests::cppToBodoArr({-1, -1, -1, 11}, true);
                input->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(0,
                                                                        false);
                input->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(1,
                                                                        false);
                input->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(2,
                                                                        false);
                input->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(3,
                                                                        false);
                std::shared_ptr<array_info> sum =
                    bodo::tests::cppToBodoArr({45, 45, -1, -1}, true);
                sum->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(2, false);
                sum->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(3, false);

                std::vector<std::shared_ptr<array_info>> chunk_cols = {
                    partition, input};
                std::shared_ptr<table_info> chunk =
                    std::make_shared<table_info>(table_info(chunk_cols));
                in_chunks.push_back(chunk);

                std::vector<std::shared_ptr<array_info>> out_chunk_cols = {
                    partition, input, sum};
                std::shared_ptr<table_info> out_chunk =
                    std::make_shared<table_info>(table_info(out_chunk_cols));
                expected_out_chunks.push_back(out_chunk);
            }

            // Chunk 5: one partition (overlaps with chunks 4)
            {
                std::shared_ptr<array_info> partition =
                    bodo::tests::cppToBodoArr({4});
                std::shared_ptr<array_info> input =
                    bodo::tests::cppToBodoArr({-1}, true);
                input->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(0,
                                                                        false);
                std::shared_ptr<array_info> sum =
                    bodo::tests::cppToBodoArr({-1}, true);
                sum->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(0, false);

                std::vector<std::shared_ptr<array_info>> chunk_cols = {
                    partition, input};
                std::shared_ptr<table_info> chunk =
                    std::make_shared<table_info>(table_info(chunk_cols));
                in_chunks.push_back(chunk);

                std::vector<std::shared_ptr<array_info>> out_chunk_cols = {
                    partition, input, sum};
                std::shared_ptr<table_info> out_chunk =
                    std::make_shared<table_info>(table_info(out_chunk_cols));
                expected_out_chunks.push_back(out_chunk);
            }

            // Chunk 6: one partition (overlaps with chunks 5)
            {
                std::shared_ptr<array_info> partition =
                    bodo::tests::cppToBodoArr({4});
                std::shared_ptr<array_info> input =
                    bodo::tests::cppToBodoArr({-1}, true);
                input->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(0,
                                                                        false);
                std::shared_ptr<array_info> sum =
                    bodo::tests::cppToBodoArr({-1}, true);
                sum->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(0, false);

                std::vector<std::shared_ptr<array_info>> chunk_cols = {
                    partition, input};
                std::shared_ptr<table_info> chunk =
                    std::make_shared<table_info>(table_info(chunk_cols));
                in_chunks.push_back(chunk);

                std::vector<std::shared_ptr<array_info>> out_chunk_cols = {
                    partition, input, sum};
                std::shared_ptr<table_info> out_chunk =
                    std::make_shared<table_info>(table_info(out_chunk_cols));
                expected_out_chunks.push_back(out_chunk);
            }

            verify_window_calculators(
                in_chunks, partition_col_indices, order_col_indices,
                keep_indices, input_col_indices, window_funcs,
                bodo_array_type::NUMPY, bodo_array_type::UNKNOWN,
                expected_out_chunks, false, &pool, mm);
        });

    bodo::tests::test(
        "calculate_single_function-sum-single_chunk-multi_rank-single_"
        "partition",
        [] {
            // Ensure we spill on unpin to verify the correctness of the
            // pinning/unpinning behavior.
            bodo::BufferPoolOptions options;
            options.spill_on_unpin = true;
            bodo::BufferPool pool = bodo::BufferPool(options);
            std::shared_ptr<::arrow::MemoryManager> mm =
                buffer_memory_manager(&pool);

            int myrank, num_ranks;
            MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
            MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

            int64_t total_rows = static_cast<int64_t>(num_ranks * 4);
            int64_t total_sum = (total_rows * (total_rows + 1)) / 2;

            std::vector<std::shared_ptr<table_info>> in_chunks;
            std::vector<int32_t> partition_col_indices;
            std::vector<int32_t> order_col_indices;
            std::vector<int32_t> keep_indices;
            std::vector<std::vector<int32_t>> input_col_indices;
            std::vector<int32_t> window_funcs;
            std::vector<std::shared_ptr<table_info>> expected_out_chunks;

            partition_col_indices.push_back(0);
            keep_indices.push_back(0);
            keep_indices.push_back(1);
            input_col_indices.push_back({1});
            window_funcs.push_back(Bodo_FTypes::sum);

            std::shared_ptr<array_info> partition =
                bodo::tests::cppToBodoArr<std::string>({"A", "A", "A", "A"});
            std::shared_ptr<array_info> input = bodo::tests::cppToBodoArr<int>(
                {1 + (4 * myrank), 2 + (4 * myrank), 3 + (4 * myrank),
                 4 + (4 * myrank)},
                true);
            std::shared_ptr<array_info> sum = bodo::tests::cppToBodoArr(
                {total_sum, total_sum, total_sum, total_sum}, true);

            std::vector<std::shared_ptr<array_info>> chunk_cols = {partition,
                                                                   input};
            std::shared_ptr<table_info> chunk =
                std::make_shared<table_info>(table_info(chunk_cols));
            in_chunks.push_back(chunk);

            std::vector<std::shared_ptr<array_info>> out_chunk_cols = {
                partition, input, sum};
            std::shared_ptr<table_info> out_chunk =
                std::make_shared<table_info>(table_info(out_chunk_cols));
            expected_out_chunks.push_back(out_chunk);

            verify_window_calculators(
                in_chunks, partition_col_indices, order_col_indices,
                keep_indices, input_col_indices, window_funcs,
                bodo_array_type::NUMPY, bodo_array_type::UNKNOWN,
                expected_out_chunks, true, &pool, mm);
        });

    bodo::tests::test(
        "calculate_single_function-sum-multi_chunk-multi_rank-multi_"
        "partition",
        [] {
            // Ensure we spill on unpin to verify the correctness of the
            // pinning/unpinning behavior.
            bodo::BufferPoolOptions options;
            options.spill_on_unpin = true;
            bodo::BufferPool pool = bodo::BufferPool(options);
            std::shared_ptr<::arrow::MemoryManager> mm =
                buffer_memory_manager(&pool);

            int myrank, num_ranks;
            MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
            MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

            std::vector<std::shared_ptr<table_info>> in_chunks;
            std::vector<int32_t> partition_col_indices;
            std::vector<int32_t> order_col_indices;
            std::vector<int32_t> keep_indices;
            std::vector<std::vector<int32_t>> input_col_indices;
            std::vector<int32_t> window_funcs;
            std::vector<std::shared_ptr<table_info>> expected_out_chunks;

            partition_col_indices.push_back(0);
            keep_indices.push_back(0);
            keep_indices.push_back(1);
            input_col_indices.push_back({1});
            window_funcs.push_back(Bodo_FTypes::sum);

            int64_t first_sum = 0;
            int64_t second_sum = 0;

            if (myrank > 0) {
                first_sum += static_cast<int>(-1 + (4 * myrank));
                first_sum += static_cast<int>(0 + (4 * myrank));
            }
            first_sum += static_cast<int>(1 + (4 * myrank));
            first_sum += static_cast<int>(2 + (4 * myrank));
            second_sum += static_cast<int>(3 + (4 * myrank));
            second_sum += static_cast<int>(4 + (4 * myrank));
            if (myrank < (num_ranks - 1)) {
                second_sum += static_cast<int>(5 + (4 * myrank));
                second_sum += static_cast<int>(6 + (4 * myrank));
            }

            // Chunk 0: first partition (overlaps with previous rank)
            {
                std::shared_ptr<array_info> partition =
                    bodo::tests::cppToBodoArr<int>({myrank});
                std::shared_ptr<array_info> input =
                    bodo::tests::cppToBodoArr<int>({1 + (4 * myrank)}, true);
                std::shared_ptr<array_info> sum =
                    bodo::tests::cppToBodoArr({first_sum}, true);

                std::vector<std::shared_ptr<array_info>> chunk_cols = {
                    partition, input};
                std::shared_ptr<table_info> chunk =
                    std::make_shared<table_info>(table_info(chunk_cols));
                in_chunks.push_back(chunk);

                std::vector<std::shared_ptr<array_info>> out_chunk_cols = {
                    partition, input, sum};
                std::shared_ptr<table_info> out_chunk =
                    std::make_shared<table_info>(table_info(out_chunk_cols));
                expected_out_chunks.push_back(out_chunk);
            }

            // Chunk 1: first partition (overlaps with previous rank) and
            // second partition (overlaps with next rank)
            {
                std::shared_ptr<array_info> partition =
                    bodo::tests::cppToBodoArr<int>({myrank, myrank + 1});
                std::shared_ptr<array_info> input =
                    bodo::tests::cppToBodoArr<int>(
                        {2 + (4 * myrank), 3 + (4 * myrank)}, true);
                std::shared_ptr<array_info> sum =
                    bodo::tests::cppToBodoArr({first_sum, second_sum}, true);

                std::vector<std::shared_ptr<array_info>> chunk_cols = {
                    partition, input};
                std::shared_ptr<table_info> chunk =
                    std::make_shared<table_info>(table_info(chunk_cols));
                in_chunks.push_back(chunk);

                std::vector<std::shared_ptr<array_info>> out_chunk_cols = {
                    partition, input, sum};
                std::shared_ptr<table_info> out_chunk =
                    std::make_shared<table_info>(table_info(out_chunk_cols));
                expected_out_chunks.push_back(out_chunk);
            }

            // Chunk 2: second partition (overlaps with next rank)
            {
                std::shared_ptr<array_info> partition =
                    bodo::tests::cppToBodoArr<int>({myrank + 1});
                std::shared_ptr<array_info> input =
                    bodo::tests::cppToBodoArr<int>({4 + (4 * myrank)}, true);
                std::shared_ptr<array_info> sum =
                    bodo::tests::cppToBodoArr({second_sum}, true);

                std::vector<std::shared_ptr<array_info>> chunk_cols = {
                    partition, input};
                std::shared_ptr<table_info> chunk =
                    std::make_shared<table_info>(table_info(chunk_cols));
                in_chunks.push_back(chunk);

                std::vector<std::shared_ptr<array_info>> out_chunk_cols = {
                    partition, input, sum};
                std::shared_ptr<table_info> out_chunk =
                    std::make_shared<table_info>(table_info(out_chunk_cols));
                expected_out_chunks.push_back(out_chunk);
            }

            verify_window_calculators(
                in_chunks, partition_col_indices, order_col_indices,
                keep_indices, input_col_indices, window_funcs,
                bodo_array_type::NUMPY, bodo_array_type::UNKNOWN,
                expected_out_chunks, true, &pool, mm);
        });

    bodo::tests::test(
        "calculate_single_function-sum-multi_chunk-multi_rank-multi_"
        "partition-holes",
        [] {
            // Ensure we spill on unpin to verify the correctness of the
            // pinning/unpinning behavior.
            bodo::BufferPoolOptions options;
            options.spill_on_unpin = true;
            bodo::BufferPool pool = bodo::BufferPool(options);
            std::shared_ptr<::arrow::MemoryManager> mm =
                buffer_memory_manager(&pool);

            int myrank, num_ranks;
            MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
            MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

            std::vector<std::shared_ptr<table_info>> in_chunks;
            std::vector<int32_t> partition_col_indices;
            std::vector<int32_t> order_col_indices;
            std::vector<int32_t> keep_indices;
            std::vector<std::vector<int32_t>> input_col_indices;
            std::vector<int32_t> window_funcs;
            std::vector<std::shared_ptr<table_info>> expected_out_chunks;

            partition_col_indices.push_back(0);
            keep_indices.push_back(0);
            keep_indices.push_back(1);
            input_col_indices.push_back({1});
            window_funcs.push_back(Bodo_FTypes::sum);

            // If the current rank is an even-numbered rank, it has
            // two chunks with no data.
            if (myrank % 2 == 0) {
                std::shared_ptr<array_info> partition =
                    bodo::tests::cppToBodoArr<int>({1});
                std::shared_ptr<array_info> input =
                    bodo::tests::cppToBodoArr<int>({}, true);
                std::shared_ptr<array_info> sum =
                    bodo::tests::cppToBodoArr({}, true);

                std::vector<std::shared_ptr<array_info>> chunk_cols = {
                    partition, input};
                std::shared_ptr<table_info> chunk =
                    std::make_shared<table_info>(table_info(chunk_cols));
                in_chunks.push_back(chunk);
                in_chunks.push_back(chunk);

                std::vector<std::shared_ptr<array_info>> out_chunk_cols = {
                    partition, input, sum};
                std::shared_ptr<table_info> out_chunk =
                    std::make_shared<table_info>(table_info(out_chunk_cols));
                expected_out_chunks.push_back(out_chunk);
                expected_out_chunks.push_back(out_chunk);
            } else {
                int64_t first_sum = 0;
                int64_t second_sum = 0;

                if (myrank > 2) {
                    first_sum += static_cast<int>(-1 + (4 * myrank));
                    first_sum += static_cast<int>(0 + (4 * myrank));
                }
                first_sum += static_cast<int>(1 + (4 * myrank));
                first_sum += static_cast<int>(2 + (4 * myrank));
                second_sum += static_cast<int>(3 + (4 * myrank));
                second_sum += static_cast<int>(4 + (4 * myrank));
                if (myrank < (num_ranks - 2)) {
                    second_sum += static_cast<int>(5 + (4 * myrank));
                    second_sum += static_cast<int>(6 + (4 * myrank));
                }

                // Chunk 0: first partition (overlaps with previous rank)
                {
                    std::shared_ptr<array_info> partition =
                        bodo::tests::cppToBodoArr<int>({myrank});
                    std::shared_ptr<array_info> input =
                        bodo::tests::cppToBodoArr<int>({1 + (4 * myrank)},
                                                       true);
                    std::shared_ptr<array_info> sum =
                        bodo::tests::cppToBodoArr({first_sum}, true);

                    std::vector<std::shared_ptr<array_info>> chunk_cols = {
                        partition, input};
                    std::shared_ptr<table_info> chunk =
                        std::make_shared<table_info>(table_info(chunk_cols));
                    in_chunks.push_back(chunk);

                    std::vector<std::shared_ptr<array_info>> out_chunk_cols = {
                        partition, input, sum};
                    std::shared_ptr<table_info> out_chunk =
                        std::make_shared<table_info>(
                            table_info(out_chunk_cols));
                    expected_out_chunks.push_back(out_chunk);
                }

                // Chunk 1: first partition (overlaps with previous rank) and
                // second partition (overlaps with next rank)
                {
                    std::shared_ptr<array_info> partition =
                        bodo::tests::cppToBodoArr<int>({myrank, myrank + 2});
                    std::shared_ptr<array_info> input =
                        bodo::tests::cppToBodoArr<int>(
                            {2 + (4 * myrank), 3 + (4 * myrank)}, true);
                    std::shared_ptr<array_info> sum = bodo::tests::cppToBodoArr(
                        {first_sum, second_sum}, true);

                    std::vector<std::shared_ptr<array_info>> chunk_cols = {
                        partition, input};
                    std::shared_ptr<table_info> chunk =
                        std::make_shared<table_info>(table_info(chunk_cols));
                    in_chunks.push_back(chunk);

                    std::vector<std::shared_ptr<array_info>> out_chunk_cols = {
                        partition, input, sum};
                    std::shared_ptr<table_info> out_chunk =
                        std::make_shared<table_info>(
                            table_info(out_chunk_cols));
                    expected_out_chunks.push_back(out_chunk);
                }

                // Chunk 2: second partition (overlaps with next rank)
                {
                    std::shared_ptr<array_info> partition =
                        bodo::tests::cppToBodoArr<int>({myrank + 2});
                    std::shared_ptr<array_info> input =
                        bodo::tests::cppToBodoArr<int>({4 + (4 * myrank)},
                                                       true);
                    std::shared_ptr<array_info> sum =
                        bodo::tests::cppToBodoArr({second_sum}, true);

                    std::vector<std::shared_ptr<array_info>> chunk_cols = {
                        partition, input};
                    std::shared_ptr<table_info> chunk =
                        std::make_shared<table_info>(table_info(chunk_cols));
                    in_chunks.push_back(chunk);

                    std::vector<std::shared_ptr<array_info>> out_chunk_cols = {
                        partition, input, sum};
                    std::shared_ptr<table_info> out_chunk =
                        std::make_shared<table_info>(
                            table_info(out_chunk_cols));
                    expected_out_chunks.push_back(out_chunk);
                }

                verify_window_calculators(
                    in_chunks, partition_col_indices, order_col_indices,
                    keep_indices, input_col_indices, window_funcs,
                    bodo_array_type::NUMPY, bodo_array_type::UNKNOWN,
                    expected_out_chunks, true, &pool, mm);
            }
        });

    bodo::tests::test(
        "calculate_single_function-sum-multi_chunk-multi_rank-two_partitions",
        [] {
            // Ensure we spill on unpin to verify the correctness of the
            // pinning/unpinning behavior.
            bodo::BufferPoolOptions options;
            options.spill_on_unpin = true;
            bodo::BufferPool pool = bodo::BufferPool(options);
            std::shared_ptr<::arrow::MemoryManager> mm =
                buffer_memory_manager(&pool);

            int myrank, num_ranks;
            MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
            MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

            int local_rows = 4;
            int total_rows = num_ranks * local_rows;
            int total_sum = (total_rows * (total_rows - 1)) / 2;

            std::vector<int32_t> partition_col_indices;
            std::vector<int32_t> order_col_indices;
            std::vector<int32_t> keep_indices;
            std::vector<std::vector<int32_t>> input_col_indices;
            std::vector<int32_t> window_funcs;

            partition_col_indices.push_back(0);
            keep_indices.push_back(0);
            keep_indices.push_back(1);
            input_col_indices.push_back({1});
            window_funcs.push_back(Bodo_FTypes::sum);

            // Iterate over every possible position for the cutoff between the
            // two partitions.
            for (int partition_cutoff = 1; partition_cutoff < total_rows;
                 partition_cutoff++) {
                std::vector<std::shared_ptr<table_info>> in_chunks;
                std::vector<std::shared_ptr<table_info>> expected_out_chunks;

                // Calculate the values of the sum of rows to the left vs right
                // of the cutoff.
                int prefix_sum =
                    (partition_cutoff * (partition_cutoff - 1)) / 2;
                int suffix_sum = total_sum - prefix_sum;

                // Generate `local_rows` chunks on this rank, each with 2
                // values: NULL and the global chunk idx including previous
                // ranks. The partition depends on the chunk idx relative to the
                // partition cutoff.
                for (int local_idx = 0; local_idx < local_rows; local_idx++) {
                    int global_row = local_idx + (local_rows * myrank);
                    int partition_val;
                    int sum_val;
                    if (global_row < partition_cutoff) {
                        partition_val = -1;
                        sum_val = prefix_sum;
                    } else {
                        partition_val = 1;
                        sum_val = suffix_sum;
                    }
                    std::shared_ptr<array_info> partition =
                        bodo::tests::cppToBodoArr<int>(
                            {partition_val, partition_val});
                    std::shared_ptr<array_info> input =
                        bodo::tests::cppToBodoArr<int>({global_row, -1}, true);
                    input->set_null_bit(1, false);
                    std::shared_ptr<array_info> sum =
                        bodo::tests::cppToBodoArr({sum_val, sum_val}, true);

                    std::vector<std::shared_ptr<array_info>> chunk_cols = {
                        partition, input};
                    std::shared_ptr<table_info> chunk =
                        std::make_shared<table_info>(table_info(chunk_cols));
                    in_chunks.push_back(chunk);

                    std::vector<std::shared_ptr<array_info>> out_chunk_cols = {
                        partition, input, sum};
                    std::shared_ptr<table_info> out_chunk =
                        std::make_shared<table_info>(
                            table_info(out_chunk_cols));
                    expected_out_chunks.push_back(out_chunk);
                }

                verify_window_calculators(
                    in_chunks, partition_col_indices, order_col_indices,
                    keep_indices, input_col_indices, window_funcs,
                    bodo_array_type::NUMPY, bodo_array_type::UNKNOWN,
                    expected_out_chunks, true, &pool, mm);
            }
        });

    bodo::tests::test("partitionless_sum_unsigned", [] {
        int myrank, num_ranks;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

        size_t n_total_rows = 100000;
        std::vector<uint8_t> in_integers(n_total_rows, -1);
        std::vector<bool> in_nulls(n_total_rows, true);
        for (size_t i = 0; i < n_total_rows; i++) {
            in_integers[i] = (uint8_t)(i % 256);
            in_nulls[i] = (bool)((i & 8) | (i & 4));
        }
        std::shared_ptr<array_info> in_arr =
            nullable_array_from_vector<Bodo_CTypes::UINT8>(in_integers,
                                                           in_nulls);

        std::vector<int64_t> selection_vector;
        for (size_t i = 0; i < n_total_rows; i++) {
            auto marker = static_cast<int64_t>(i + (i >> 1) * (i >> 1));
            if (marker % (int64_t)num_ranks == myrank) {
                selection_vector.push_back(i);
            }
        }
        in_arr = RetrieveArray_SingleColumn(in_arr, selection_vector);

        size_t local_length = in_arr->length;

        // Create a singleton array containing the one global answer for
        // the sum value, and broadcast the correct length to obtain the
        // refsol.
        std::shared_ptr<array_info> answer_singleton =
            alloc_nullable_array_no_nulls(1, Bodo_CTypes::UINT64);
        getv<uint64_t>(answer_singleton, 0) = 9706740;
        std::vector<int64_t> zero_idxs(local_length, 0);
        auto expected_out =
            RetrieveArray_SingleColumn(answer_singleton, zero_idxs);

        // Create an empty output array with the correct length
        std::vector<std::shared_ptr<array_info>> out_arrs;
        out_arrs.push_back(
            alloc_nullable_array_all_nulls(1, Bodo_CTypes::UINT64));
        aggfunc_output_initialize(out_arrs[0], Bodo_FTypes::sum, true);
        verify_sorted_window_output(Bodo_FTypes::sum, {}, {}, {in_arr}, {0, 1},
                                    out_arrs, expected_out);
    });

    bodo::tests::test("partitionless_count", [] {
        int myrank, num_ranks;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

        size_t n_total_rows = 10000;
        std::vector<int32_t> in_integers(n_total_rows, -1);
        std::vector<bool> in_nulls(n_total_rows, true);
        for (size_t i = 0; i < n_total_rows; i++) {
            in_integers[i] = (int32_t)(i % 4);
            in_nulls[i] = (bool)(i & 8);
        }
        std::shared_ptr<array_info> int_arr =
            nullable_array_from_vector<Bodo_CTypes::INT32>(in_integers,
                                                           in_nulls);

        std::shared_ptr<array_info> numpy_arr =
            alloc_numpy(n_total_rows, Bodo_CTypes::INT8);

        bodo::vector<std::string> in_strings(n_total_rows, "ABC");
        std::shared_ptr<array_info> string_arr =
            string_array_from_vector(in_strings, in_nulls, Bodo_CTypes::STRING);

        bodo::vector<std::string> dict_strings(4);
        dict_strings[0] = "ALPHA";
        dict_strings[1] = "BETA";
        dict_strings[2] = "GAMMA";
        dict_strings[3] = "DELTA";
        std::shared_ptr<array_info> dict_arr =
            dict_array_from_vector(dict_strings, in_integers, in_nulls);

        std::vector<std::pair<std::shared_ptr<array_info>, uint64_t>> tests = {
            {int_arr, 5000},
            {numpy_arr, 10000},
            {string_arr, 5000},
            {dict_arr, 5000},
        };

        for (auto &it : tests) {
            // Have each rank select a different subset of the rows
            std::shared_ptr<array_info> in_arr = it.first;
            std::vector<int64_t> selection_vector;
            for (size_t i = 0; i < n_total_rows; i++) {
                auto marker = static_cast<int64_t>(i + (i >> 1) * (i >> 1));
                if (marker % (int64_t)num_ranks == myrank) {
                    selection_vector.push_back(i);
                }
            }
            in_arr = RetrieveArray_SingleColumn(in_arr, selection_vector);
            size_t local_length = in_arr->length;

            // Create a singleton array containing the one global answer for
            // the count value, and broadcast the correct length to obtain the
            // refsol.
            std::shared_ptr<array_info> answer_singleton =
                alloc_numpy(1, Bodo_CTypes::UINT64);
            getv<uint64_t>(answer_singleton, 0) = it.second;
            std::vector<int64_t> zero_idxs(local_length, 0);
            auto expected_out =
                RetrieveArray_SingleColumn(answer_singleton, zero_idxs);

            // Create an empty output array with the correct length
            std::vector<std::shared_ptr<array_info>> out_arrs;
            out_arrs.push_back(alloc_numpy(1, Bodo_CTypes::UINT64));
            aggfunc_output_initialize(out_arrs[0], Bodo_FTypes::count, true);

            verify_sorted_window_output(Bodo_FTypes::count, {}, {}, {in_arr},
                                        {0, 1}, out_arrs, expected_out);
        }
    });

    bodo::tests::test("partitionless_min_max_integer", [] {
        int myrank, num_ranks;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

        std::vector<std::pair<Bodo_FTypes::FTypeEnum, int64_t>> tests = {
            {Bodo_FTypes::max, 30863580},
            {Bodo_FTypes::min, -15233730},
        };

        for (auto &it : tests) {
            // Create the total answer array, then have each rank select a
            // different subset of the rows
            size_t n_total_rows = 10000;
            std::vector<int64_t> in_integers(n_total_rows, -1);
            std::vector<bool> in_nulls(n_total_rows, true);
            for (size_t i = 0; i < n_total_rows; i++) {
                in_integers[i] = -(((int64_t)i - 12345) * ((int64_t)i - 1234));
            }
            std::shared_ptr<array_info> in_arr =
                nullable_array_from_vector<Bodo_CTypes::INT64>(in_integers,
                                                               in_nulls);
            std::vector<int64_t> selection_vector;
            for (size_t i = 0; i < n_total_rows; i++) {
                auto marker = static_cast<int64_t>(i + (i >> 1) * (i >> 1));
                if (marker % (int64_t)num_ranks == myrank) {
                    selection_vector.push_back(i);
                }
            }
            in_arr = RetrieveArray_SingleColumn(in_arr, selection_vector);

            // Create a singleton array containing the one global answer for
            // the min/max value, and broadcast the correct length to obtain the
            // refsol.
            std::shared_ptr<array_info> answer_singleton =
                nullable_array_from_vector<Bodo_CTypes::INT64>({it.second},
                                                               {true});
            size_t local_length = in_arr->length;
            std::vector<int64_t> zero_idxs(local_length, 0);
            auto expected_out =
                RetrieveArray_SingleColumn(answer_singleton, zero_idxs);

            // Create an empty output array with the correct length
            std::vector<std::shared_ptr<array_info>> out_arrs;
            std::vector<int64_t> null_idxs(1, -1);
            out_arrs.push_back(
                RetrieveArray_SingleColumn(answer_singleton, null_idxs));
            aggfunc_output_initialize(out_arrs[0], it.first, true);

            verify_sorted_window_output(it.first, {}, {}, {in_arr}, {0, 1},
                                        out_arrs, expected_out);
        }
    });

    bodo::tests::test("partitionless_min_max_all_null", [] {
        int myrank, num_ranks;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
        size_t n_local_rows = (size_t)(((myrank + num_ranks) << 4) - 1);

        std::vector<std::shared_ptr<array_info>> input_arrays;

        // Create an all-null array of type INT32, which is both the input
        // and also the correct answer for the output.
        std::vector<int32_t> in_integers(n_local_rows, -1);
        std::vector<bool> in_nulls(n_local_rows, false);
        std::shared_ptr<array_info> int32_arr =
            nullable_array_from_vector<Bodo_CTypes::INT32>(in_integers,
                                                           in_nulls);
        input_arrays.push_back(int32_arr);

        // Do the same for a string array.
        bodo::vector<std::string> in_strings(n_local_rows, "ABC");
        std::shared_ptr<array_info> string_arr =
            string_array_from_vector(in_strings, in_nulls, Bodo_CTypes::STRING);
        input_arrays.push_back(string_arr);

        // Do the same for a string array.
        bodo::vector<std::string> dict_strings(4);
        dict_strings[0] = "ALPHA";
        dict_strings[1] = "BETA";
        dict_strings[2] = "GAMMA";
        dict_strings[3] = "DELTA";
        std::shared_ptr<array_info> dict_arr =
            dict_array_from_vector(dict_strings, in_integers, in_nulls);
        input_arrays.push_back(dict_arr);

        for (auto &in_arr : input_arrays) {
            // Create an empty output array with the correct length
            std::vector<std::shared_ptr<array_info>> out_arrs;
            std::vector<int64_t> null_idxs(1, -1);
            out_arrs.push_back(RetrieveArray_SingleColumn(in_arr, null_idxs));

            // Create the refsol by cloning the all-null input array;
            std::vector<int64_t> zero_idxs(n_local_rows, 0);
            std::shared_ptr<array_info> expected =
                RetrieveArray_SingleColumn(in_arr, zero_idxs);

            verify_sorted_window_output(Bodo_FTypes::max, {}, {}, {in_arr},
                                        {0, 1}, out_arrs, expected);
            verify_sorted_window_output(Bodo_FTypes::min, {}, {}, {in_arr},
                                        {0, 1}, out_arrs, expected);
        }
    });
    bodo::tests::test("test_par_dense_rank_orders_match", [] {
        // tests case where orders/groups match across ranks and each rank has
        // multiple orders on it orders look something like: rank 0: 0 1 1, rank
        // 1: 1 2 2, rank 2: 2 3 3, ...
        std::vector<std::shared_ptr<array_info>> partition_by_arrs;
        std::vector<std::shared_ptr<array_info>> order_by_arrs;
        std::vector<std::shared_ptr<array_info>> out_arrs;
        size_t n = 3;
        size_t n_partition_cols = 1;

        int myrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        int64_t k = myrank + 1;
        std::shared_ptr<array_info> order_by_col =
            bodo::tests::cppToBodoArr({k - 1, k, k});
        std::shared_ptr<array_info> expected_out =
            bodo::tests::cppToBodoArr({k, k + 1, k + 1});

        for (size_t i = 0; i < n_partition_cols; i++) {
            partition_by_arrs.push_back(
                alloc_nullable_array_all_nulls(n, Bodo_CTypes::INT8));
        }
        order_by_arrs.push_back(order_by_col);

        out_arrs.push_back(alloc_numpy(n, Bodo_CTypes::INT64));

        verify_sorted_window_output(Bodo_FTypes::dense_rank, partition_by_arrs,
                                    order_by_arrs, {}, {}, out_arrs,
                                    expected_out);
    });
    bodo::tests::test("test_par_dense_rank_single_orders_match", [] {
        // tests case where each rank has one order and it matches with
        // with the neighbor to the left. i.e. it looks like:
        // rank 0: 0 0 0, rank 1: 0 0 0, rank 2: 1 1 1, ...
        std::vector<std::shared_ptr<array_info>> partition_by_arrs;
        std::vector<std::shared_ptr<array_info>> order_by_arrs;
        std::vector<std::shared_ptr<array_info>> out_arrs;
        size_t n = 3;
        size_t n_partition_cols = 1;

        int myrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        int64_t k = myrank / 2;
        int64_t window_rank = k + 1;
        std::shared_ptr<array_info> order_by_col =
            bodo::tests::cppToBodoArr({k, k, k});
        std::shared_ptr<array_info> expected_out =
            bodo::tests::cppToBodoArr({window_rank, window_rank, window_rank});

        for (size_t i = 0; i < n_partition_cols; i++) {
            partition_by_arrs.push_back(
                alloc_nullable_array_all_nulls(n, Bodo_CTypes::INT8));
        }
        order_by_arrs.push_back(order_by_col);

        out_arrs.push_back(alloc_numpy(n, Bodo_CTypes::INT64));

        verify_sorted_window_output(Bodo_FTypes::dense_rank, partition_by_arrs,
                                    order_by_arrs, {}, {}, out_arrs,
                                    expected_out);
    });
    bodo::tests::test("test_par_dense_rank_multiple_groups", [] {
        // tests dense_rank in the case of multiple groups on each rank
        // same set up as test_par_row_number_multiple_groups_per_rank

        std::vector<std::shared_ptr<array_info>> out_arrs;
        size_t n = 4;
        size_t n_partition_cols = 1;
        size_t n_order_by_cols = 1;

        int myrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        auto [partition_by_arrs, order_by_arrs] =
            create_int_example_multiple_groups(n_partition_cols,
                                               n_order_by_cols);

        out_arrs.push_back(alloc_numpy(n, Bodo_CTypes::INT64));

        // generate expected output
        std::vector<int64_t> expected_out_vec = {1, 2, 1, 2};
        // apply expected offsets
        if (myrank > 0) {
            expected_out_vec[0] += 1;
            expected_out_vec[1] += 1;
        }
        std::shared_ptr<array_info> expected_out =
            bodo::tests::cppToBodoArr(expected_out_vec);

        verify_sorted_window_output(Bodo_FTypes::dense_rank, partition_by_arrs,
                                    order_by_arrs, {}, {}, out_arrs,
                                    expected_out);
    });
    bodo::tests::test("test_par_dense_rank_with_holes", [] {
        // same set up as test_par_dense_rank_multiple_groups except every
        // other rank is empty
        std::vector<std::shared_ptr<array_info>> out_arrs;
        size_t n_partition_cols = 1;
        size_t n_order_by_cols = 1;

        int myrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        size_t n = myrank % 2 == 0 ? 4 : 0;

        std::shared_ptr<array_info> empty_arr =
            alloc_numpy(0, Bodo_CTypes::INT64);

        auto [partition_by_arrs, order_by_arrs] =
            create_int_example_with_holes(n_partition_cols, n_order_by_cols);
        out_arrs.push_back(alloc_numpy(n, Bodo_CTypes::INT64));

        std::shared_ptr<array_info> expected_out;
        if (n == 0) {
            expected_out = empty_arr;
        } else {
            // generate expected output
            std::vector<int64_t> expected_out_vec = {1, 2, 1, 2};
            // apply expected offsets
            if (myrank > 0) {
                expected_out_vec[0] += 2;
                expected_out_vec[1] += 2;
            }
            expected_out = bodo::tests::cppToBodoArr(expected_out_vec);
        }

        verify_sorted_window_output(Bodo_FTypes::dense_rank, partition_by_arrs,
                                    order_by_arrs, {}, {}, out_arrs,
                                    expected_out);
    });
    bodo::tests::test("test_local_pct_rank", [] {
        // verifies that pct rank works locally
        std::vector<std::shared_ptr<array_info>> partition_by_arrs;
        std::vector<std::shared_ptr<array_info>> order_by_arrs;
        std::vector<std::shared_ptr<array_info>> out_arrs;
        size_t n = 9;

        std::shared_ptr<array_info> partition_by_arr =
            bodo::tests::cppToBodoArr({1, 1, 1, 1, 2, 2, 2, 2, 3});
        std::shared_ptr<array_info> order_by_arr =
            bodo::tests::cppToBodoArr({1, 2, 2, 3, 1, 2, 2, 3, 1});

        // ranks = {1, 2, 2, 4, 1, 2, 2, 4}

        partition_by_arrs.push_back(partition_by_arr);
        order_by_arrs.push_back(order_by_arr);

        std::shared_ptr<array_info> expected_out =
            bodo::tests::cppToBodoArr<double>({0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0,
                                               0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0,
                                               0.0});
        out_arrs.push_back(alloc_numpy(n, Bodo_CTypes::FLOAT64));

        verify_sorted_window_output(Bodo_FTypes::percent_rank,
                                    partition_by_arrs, order_by_arrs, {}, {},
                                    out_arrs, expected_out, false);
    });
    bodo::tests::test("test_par_pct_rank_orders_match", [] {
        // tests case where orders/groups match across ranks and each rank has
        // multiple orders on it orders look something like: rank 0: 0 1 1, rank
        // 1: 1 2 2, rank 2: 2 3 3, ...
        std::vector<std::shared_ptr<array_info>> partition_by_arrs;
        std::vector<std::shared_ptr<array_info>> order_by_arrs;
        std::vector<std::shared_ptr<array_info>> out_arrs;
        size_t n = 3;
        size_t n_partition_cols = 1;

        int myrank;
        int n_pes;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);

        int64_t k = myrank + 1;
        int64_t first_val = k == 1 ? 1 : 3 * (k - 1) - 1;
        int64_t second_val = k == 1 ? first_val + 1 : first_val + 3;
        double d = static_cast<double>(n_pes * n) - 1.0;

        std::shared_ptr<array_info> order_by_col =
            bodo::tests::cppToBodoArr({k - 1, k, k});
        std::shared_ptr<array_info> expected_out =
            bodo::tests::cppToBodoArr<double>(
                {static_cast<double>(first_val - 1) / d,
                 static_cast<double>(second_val - 1) / d,
                 static_cast<double>(second_val - 1) / d});

        for (size_t i = 0; i < n_partition_cols; i++) {
            partition_by_arrs.push_back(
                alloc_nullable_array_all_nulls(n, Bodo_CTypes::INT64));
        }
        order_by_arrs.push_back(order_by_col);

        out_arrs.push_back(alloc_numpy(n, Bodo_CTypes::FLOAT64));

        verify_sorted_window_output(Bodo_FTypes::percent_rank,
                                    partition_by_arrs, order_by_arrs, {}, {},
                                    out_arrs, expected_out);
    });
    bodo::tests::test("test_par_pct_rank_single_orders_match", [] {
        // tests case where each rank has one order and it matches with
        // with the neighbor to the left. i.e. it looks like:
        // rank 0: 0 0 0, rank 1: 0 0 0, rank 2: 1 1 1, ...
        std::vector<std::shared_ptr<array_info>> partition_by_arrs;
        std::vector<std::shared_ptr<array_info>> order_by_arrs;
        std::vector<std::shared_ptr<array_info>> out_arrs;
        size_t n = 3;
        size_t n_partition_cols = 1;

        int myrank;
        int n_pes;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);

        int64_t k = myrank / 2;
        double d = n_pes == 1 ? 1.0 : static_cast<double>(n_pes * n) - 1.0;
        double window_pct_rank = static_cast<double>(k * 6) / d;
        std::shared_ptr<array_info> order_by_col =
            bodo::tests::cppToBodoArr({k, k, k});
        std::shared_ptr<array_info> expected_out =
            bodo::tests::cppToBodoArr<double>(
                {window_pct_rank, window_pct_rank, window_pct_rank});

        for (size_t i = 0; i < n_partition_cols; i++) {
            partition_by_arrs.push_back(
                alloc_nullable_array_all_nulls(n, Bodo_CTypes::INT64));
        }
        order_by_arrs.push_back(order_by_col);

        out_arrs.push_back(alloc_numpy(n, Bodo_CTypes::FLOAT64));

        verify_sorted_window_output(Bodo_FTypes::percent_rank,
                                    partition_by_arrs, order_by_arrs, {}, {},
                                    out_arrs, expected_out);
    });
    bodo::tests::test("test_par_pct_rank_multiple_groups", [] {
        // tests rank in the case of multiple groups on each rank
        // same set up as test_par_row_number_multiple_groups_per_rank
        std::vector<std::shared_ptr<array_info>> out_arrs;
        size_t n = 4;
        size_t n_partition_cols = 1;
        size_t n_order_by_cols = 1;

        int myrank;
        int n_pes;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);

        // test requires more than one rank
        if (n_pes == 1) {
            return;
        }

        auto [partition_by_arrs, order_by_arrs] =
            create_int_example_multiple_groups(n_partition_cols,
                                               n_order_by_cols);

        out_arrs.push_back(alloc_numpy(n, Bodo_CTypes::FLOAT64));

        // generate expected output
        std::shared_ptr<array_info> expected_out;
        // apply expected offsets
        if (myrank == 0) {
            expected_out =
                bodo::tests::cppToBodoArr<double>({0.0, 1.0, 0.0, 1.0 / 3.0});
        } else if (myrank == n_pes - 1) {
            expected_out =
                bodo::tests::cppToBodoArr<double>({1.0 / 3.0, 1.0, 0.0, 1.0});
        } else {
            expected_out = bodo::tests::cppToBodoArr<double>(
                {1.0 / 3.0, 1.0, 0.0, 1.0 / 3.0});
        }

        verify_sorted_window_output(Bodo_FTypes::percent_rank,
                                    partition_by_arrs, order_by_arrs, {}, {},
                                    out_arrs, expected_out);
    });
    bodo::tests::test("test_par_pct_rank_with_holes", [] {
        // same set up as test_par_rank_multiple_groups except every other rank
        // is empty
        std::vector<std::shared_ptr<array_info>> out_arrs;
        size_t n_partition_cols = 1;
        size_t n_order_by_cols = 1;

        int myrank;
        int n_pes;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);

        size_t n = myrank % 2 == 0 ? 4 : 0;

        std::shared_ptr<array_info> empty_arr =
            alloc_numpy(0, Bodo_CTypes::INT64);

        auto [partition_by_arrs, order_by_arrs] =
            create_int_example_with_holes(n_partition_cols, n_order_by_cols);

        out_arrs.push_back(alloc_numpy(n, Bodo_CTypes::FLOAT64));

        std::shared_ptr<array_info> expected_out;
        if (n == 0) {
            expected_out = alloc_numpy(0, Bodo_CTypes::FLOAT64);
        } else {
            // generate expected output
            std::vector<double> expected_out_vec = {0.0, 1.0, 0.0, 1.0 / 3.0};
            // for ranks after 1, the group size of the first group is 4
            if (myrank > 0) {
                expected_out_vec[0] = 2.0 / 3.0;
            }
            // if the rank is the last non-empty rank, the group size is only 2
            if (myrank / 2 == (n_pes - 1) / 2) {
                expected_out_vec[3] = 1.0;
            }
            expected_out = bodo::tests::cppToBodoArr<double>(expected_out_vec);
        }

        verify_sorted_window_output(Bodo_FTypes::percent_rank,
                                    partition_by_arrs, order_by_arrs, {}, {},
                                    out_arrs, expected_out);
    });
    bodo::tests::test("test_par_pct_rank_singleton_groups", [] {
        // test where the orderby's are all the same and the groups are
        // different
        std::vector<std::shared_ptr<array_info>> partition_by_arrs;
        std::vector<std::shared_ptr<array_info>> order_by_arrs;
        std::vector<std::shared_ptr<array_info>> out_arrs;
        size_t n = 1;
        int64_t val = 4;

        int myrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        // each rank has a different partition
        partition_by_arrs.push_back(const_int64_arr(n, myrank));

        std::shared_ptr<array_info> order_by_arr = const_int64_arr(n, val);
        // every rank has the same orderby val
        order_by_arrs.push_back(order_by_arr);

        std::shared_ptr<array_info> expected_out =
            bodo::tests::cppToBodoArr<double>({0.0});

        out_arrs.push_back(alloc_numpy(n, Bodo_CTypes::FLOAT64));

        verify_sorted_window_output(Bodo_FTypes::percent_rank,
                                    partition_by_arrs, order_by_arrs, {}, {},
                                    out_arrs, expected_out);
    });
    bodo::tests::test("test_par_pct_multiple_single_groups", [] {
        // Extra edge case in the update step where there is only one group that
        // needs to be updated and it does not match with it's neighbor to the
        // left.
        std::vector<std::shared_ptr<array_info>> partition_by_arrs;
        std::vector<std::shared_ptr<array_info>> order_by_arrs;
        std::vector<std::shared_ptr<array_info>> out_arrs;

        int myrank;
        int n_pes;
        int n = 3;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);

        // designed for 3 ranks
        if (n_pes != 3) {
            return;
        }

        std::shared_ptr<array_info> empty_arr =
            alloc_numpy(0, Bodo_CTypes::INT64);

        std::shared_ptr<array_info> expected_out;

        out_arrs.push_back(alloc_numpy(n, Bodo_CTypes::FLOAT64));

        if (myrank == 0) {
            order_by_arrs.push_back(bodo::tests::cppToBodoArr({1, 2, 3}));
            partition_by_arrs.push_back(bodo::tests::cppToBodoArr({1, 1, 1}));
            expected_out = bodo::tests::cppToBodoArr<double>({0.0, 0.5, 1.0});
        } else if (myrank == 1) {
            order_by_arrs.push_back(bodo::tests::cppToBodoArr({1, 2, 3}));
            partition_by_arrs.push_back(bodo::tests::cppToBodoArr({2, 2, 2}));
            expected_out = bodo::tests::cppToBodoArr<double>({0.0, 0.2, 0.4});
        } else if (myrank == 2) {
            order_by_arrs.push_back(bodo::tests::cppToBodoArr({3, 3, 3}));
            partition_by_arrs.push_back(bodo::tests::cppToBodoArr({2, 2, 2}));
            expected_out = bodo::tests::cppToBodoArr<double>({0.4, 0.4, 0.4});
        }

        verify_sorted_window_output(Bodo_FTypes::percent_rank,
                                    partition_by_arrs, order_by_arrs, {}, {},
                                    out_arrs, expected_out);
    });
    bodo::tests::test("test_local_cume_dist", [] {
        // verifies that pct rank works locally
        std::vector<std::shared_ptr<array_info>> partition_by_arrs;
        std::vector<std::shared_ptr<array_info>> order_by_arrs;
        std::vector<std::shared_ptr<array_info>> out_arrs;
        size_t n = 9;

        std::shared_ptr<array_info> partition_by_arr =
            bodo::tests::cppToBodoArr({1, 1, 1, 1, 2, 2, 2, 2, 3});
        std::shared_ptr<array_info> order_by_arr =
            bodo::tests::cppToBodoArr({1, 2, 2, 3, 1, 2, 2, 3, 1});

        partition_by_arrs.push_back(partition_by_arr);
        order_by_arrs.push_back(order_by_arr);

        std::shared_ptr<array_info> expected_out =
            bodo::tests::cppToBodoArr<double>({1.0 / 4.0, 3.0 / 4.0, 3.0 / 4.0,
                                               1.0, 1.0 / 4.0, 3.0 / 4.0,
                                               3.0 / 4.0, 1.0, 1.0});
        out_arrs.push_back(alloc_numpy(n, Bodo_CTypes::FLOAT64));

        verify_sorted_window_output(Bodo_FTypes::cume_dist, partition_by_arrs,
                                    order_by_arrs, {}, {}, out_arrs,
                                    expected_out, true);
    });
    bodo::tests::test("test_par_cume_dist_orders_match", [] {
        // tests case where orders/groups match across ranks and each rank has
        // multiple orders on it orders look something like: rank 0: 0 1 1, rank
        // 1: 1 2 2, rank 2: 2 3 3, ...
        std::vector<std::shared_ptr<array_info>> partition_by_arrs;
        std::vector<std::shared_ptr<array_info>> order_by_arrs;
        std::vector<std::shared_ptr<array_info>> out_arrs;
        size_t n = 3;
        size_t n_partition_cols = 1;

        int myrank;
        int n_pes;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);

        int64_t k = myrank + 1;
        int64_t first_val = k == 1 ? 1 : 3 * (k - 1) + 1;
        int64_t second_val = k == n_pes ? first_val + 2 : first_val + 3;
        double d = static_cast<double>(n_pes * n);

        std::shared_ptr<array_info> order_by_col =
            bodo::tests::cppToBodoArr({k - 1, k, k});
        std::shared_ptr<array_info> expected_out =
            bodo::tests::cppToBodoArr<double>(
                {static_cast<double>(first_val) / d,
                 static_cast<double>(second_val) / d,
                 static_cast<double>(second_val) / d});

        for (size_t i = 0; i < n_partition_cols; i++) {
            partition_by_arrs.push_back(
                alloc_nullable_array_all_nulls(n, Bodo_CTypes::INT64));
        }
        order_by_arrs.push_back(order_by_col);

        out_arrs.push_back(alloc_numpy(n, Bodo_CTypes::FLOAT64));

        verify_sorted_window_output(Bodo_FTypes::cume_dist, partition_by_arrs,
                                    order_by_arrs, {}, {}, out_arrs,
                                    expected_out);
    });
    bodo::tests::test("test_par_cume_dist_single_orders_match", [] {
        // tests case where each rank has one order and it matches with
        // with the neighbor to the left. i.e. it looks like:
        // rank 0: 0 0 0, rank 1: 0 0 0, rank 2: 1 1 1, ...
        std::vector<std::shared_ptr<array_info>> partition_by_arrs;
        std::vector<std::shared_ptr<array_info>> order_by_arrs;
        std::vector<std::shared_ptr<array_info>> out_arrs;
        size_t n = 3;
        size_t n_partition_cols = 1;

        int myrank;
        int n_pes;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);

        int64_t k = myrank / 2;
        double d = static_cast<double>(n_pes * n);
        double window_cumedist = (k + 1.0) * 6.0 / d;

        if (myrank % 2 == 0 && myrank == n_pes - 1)
            window_cumedist = (k * 6.0 + 3.0) / d;

        std::shared_ptr<array_info> order_by_col =
            bodo::tests::cppToBodoArr({k, k, k});
        std::shared_ptr<array_info> expected_out =
            bodo::tests::cppToBodoArr<double>(
                {window_cumedist, window_cumedist, window_cumedist});

        for (size_t i = 0; i < n_partition_cols; i++) {
            partition_by_arrs.push_back(
                alloc_nullable_array_all_nulls(n, Bodo_CTypes::INT64));
        }
        order_by_arrs.push_back(order_by_col);

        out_arrs.push_back(alloc_numpy(n, Bodo_CTypes::FLOAT64));

        verify_sorted_window_output(Bodo_FTypes::cume_dist, partition_by_arrs,
                                    order_by_arrs, {}, {}, out_arrs,
                                    expected_out);
    });
    bodo::tests::test("test_par_cume_dist_multiple_groups", [] {
        // tests rank in the case of multiple groups on each rank
        // same set up as test_par_row_number_multiple_groups_per_rank
        std::vector<std::shared_ptr<array_info>> out_arrs;
        size_t n = 4;
        size_t n_partition_cols = 1;
        size_t n_order_by_cols = 1;

        int myrank;
        int n_pes;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);

        // test requires more than one rank
        if (n_pes == 1) {
            return;
        }

        auto [partition_by_arrs, order_by_arrs] =
            create_int_example_multiple_groups(n_partition_cols,
                                               n_order_by_cols);

        out_arrs.push_back(alloc_numpy(n, Bodo_CTypes::FLOAT64));

        // generate expected output
        std::shared_ptr<array_info> expected_out;
        // apply expected offsets
        if (myrank == 0) {
            expected_out =
                bodo::tests::cppToBodoArr<double>({0.5, 1.0, 0.25, 0.75});
        } else if (myrank == n_pes - 1) {
            expected_out =
                bodo::tests::cppToBodoArr<double>({0.75, 1.0, 0.5, 1.0});
        } else {
            expected_out =
                bodo::tests::cppToBodoArr<double>({0.75, 1.0, 0.25, 0.75});
        }

        verify_sorted_window_output(Bodo_FTypes::cume_dist, partition_by_arrs,
                                    order_by_arrs, {}, {}, out_arrs,
                                    expected_out);
    });
    bodo::tests::test("test_par_cume_dist_with_holes", [] {
        // same set up as test_par_rank_multiple_groups except every other rank
        // is empty
        std::vector<std::shared_ptr<array_info>> out_arrs;
        size_t n_partition_cols = 1;
        size_t n_order_by_cols = 1;

        int myrank;
        int n_pes;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);

        size_t n = myrank % 2 == 0 ? 4 : 0;

        std::shared_ptr<array_info> empty_arr =
            alloc_numpy(0, Bodo_CTypes::INT64);

        auto [partition_by_arrs, order_by_arrs] =
            create_int_example_with_holes(n_partition_cols, n_order_by_cols);

        out_arrs.push_back(alloc_numpy(n, Bodo_CTypes::FLOAT64));

        std::shared_ptr<array_info> expected_out;
        if (n == 0) {
            expected_out = alloc_numpy(0, Bodo_CTypes::FLOAT64);
        } else {
            // generate expected output
            std::vector<double> expected_out_vec = {0.75, 1.0, 0.25, 0.5};
            // for ranks after 1, the group size of the first group is 4
            if (myrank == 0)
                expected_out_vec[0] = 0.5;
            // if the rank is the last non-empty rank, the group size is only 2
            if (myrank / 2 == (n_pes - 1) / 2) {
                expected_out_vec[2] = 0.5;
                expected_out_vec[3] = 1.0;
            }
            expected_out = bodo::tests::cppToBodoArr<double>(expected_out_vec);
        }

        verify_sorted_window_output(Bodo_FTypes::cume_dist, partition_by_arrs,
                                    order_by_arrs, {}, {}, out_arrs,
                                    expected_out);
    });
    bodo::tests::test("test_partitionless_avg_double", [] {
        size_t n = 5;
        std::shared_ptr<array_info> in_arr =
            bodo::tests::cppToBodoArr<double>({1.0, 2.0, 3.0, 4.0, 5.0});

        std::shared_ptr<array_info> expected =
            bodo::tests::cppToBodoArr<double>(std::vector<double>(n, 3.0),
                                              true);

        std::vector<std::shared_ptr<array_info>> out_arrs;
        out_arrs.push_back(
            alloc_nullable_array_all_nulls(1, Bodo_CTypes::FLOAT64));
        aggfunc_output_initialize(out_arrs[0], Bodo_FTypes::mean, true);

        verify_sorted_window_output(Bodo_FTypes::mean, {}, {}, {in_arr}, {0, 1},
                                    out_arrs, expected, true);
    });
    bodo::tests::test("test_partitionless_avg_with_holes", [] {
        // tests that avg works when some ranks have empty data
        int myrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        size_t n = myrank % 2 == 0 ? 0 : 3;

        std::shared_ptr<array_info> in_arr =
            alloc_nullable_array_all_nulls(n, Bodo_CTypes::FLOAT64);

        std::shared_ptr<array_info> expected =
            alloc_nullable_array_all_nulls(n, Bodo_CTypes::FLOAT64);

        std::vector<std::shared_ptr<array_info>> out_arrs;
        out_arrs.push_back(
            alloc_nullable_array_all_nulls(1, Bodo_CTypes::FLOAT64));
        aggfunc_output_initialize(out_arrs[0], Bodo_FTypes::sum, true);

        verify_sorted_window_output(Bodo_FTypes::mean, {}, {}, {in_arr}, {0, 1},
                                    out_arrs, expected, true);
    });
    bodo::tests::test("test_partitionless_avg_int_numpy", [] {
        int myrank;
        int n_pes;
        size_t n = 5;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);

        std::shared_ptr<array_info> in_arr = const_int64_arr(n, myrank + 1);

        double global_avg = (n_pes + 1) / 2.0;

        std::shared_ptr<array_info> expected =
            bodo::tests::cppToBodoArr<double>(
                std::vector<double>(n, global_avg), true);

        std::vector<std::shared_ptr<array_info>> out_arrs;
        out_arrs.push_back(
            alloc_nullable_array_all_nulls(1, Bodo_CTypes::FLOAT64));
        aggfunc_output_initialize(out_arrs[0], Bodo_FTypes::mean, true);

        verify_sorted_window_output(Bodo_FTypes::mean, {}, {}, {in_arr}, {0, 1},
                                    out_arrs, expected, true);
    });
    bodo::tests::test("test_partitionless_avg_all_null", [] {
        size_t n = 5;

        std::shared_ptr<array_info> in_arr =
            alloc_nullable_array_all_nulls(n, Bodo_CTypes::INT32);

        std::shared_ptr<array_info> expected =
            alloc_nullable_array_all_nulls(n, Bodo_CTypes::FLOAT64);

        std::vector<std::shared_ptr<array_info>> out_arrs;
        out_arrs.push_back(
            alloc_nullable_array_all_nulls(1, Bodo_CTypes::FLOAT64));
        aggfunc_output_initialize(out_arrs[0], Bodo_FTypes::mean, true);

        verify_sorted_window_output(Bodo_FTypes::mean, {}, {}, {in_arr}, {0, 1},
                                    out_arrs, expected, true);
    });
    bodo::tests::test("test_size", [] {
        // tests count(*) over () where rank r has r + 1 rows.
        int myrank;
        int n_pes;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
        size_t n_rows = myrank + 1;

        // sum ( 1, 2, ..., n )
        size_t expected_val = ((n_pes + 1) * n_pes) / 2;

        std::shared_ptr<array_info> expected =
            bodo::tests::cppToBodoArr<uint64_t>(
                std::vector<uint64_t>(n_rows, expected_val));

        std::vector<std::shared_ptr<array_info>> out_arrs;
        out_arrs.push_back(alloc_numpy(1, Bodo_CTypes::UINT64));
        aggfunc_output_initialize(out_arrs[0], Bodo_FTypes::size, true);

        verify_sorted_window_output(Bodo_FTypes::size, {}, {}, {}, {0, 0},
                                    out_arrs, expected, true);
    });
});
