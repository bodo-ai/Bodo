#include <iostream>
#include <memory>
#include <sstream>
#include "../libs/_array_utils.h"
#include "../libs/_bodo_common.h"
#include "../libs/_groupby_col_set.h"
#include "../libs/_window_compute.h"
#include "./test.hpp"

// creates a NUMPY/int64 array that goes from min..max (including max), could be
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

// TODO: template these tests and/or add parmaeter to test different types
static bodo::tests::suite tests([] {
    // tests example of a (parallel) row number where each rank gets
    // a list of values that are all the same + in the same group
    bodo::tests::test("test_par_row_number_single_group", [] {
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

        sorted_window_computation(partition_by_arrs, order_by_arrs,
                                  {Bodo_FTypes::row_number}, out_arrs, true);

        std::stringstream ss1;
        std::stringstream ss2;
        DEBUG_PrintColumn(ss1, out_arrs[0]);
        DEBUG_PrintColumn(ss2, expected_out);

        bodo::tests::check(ss1.str() == ss2.str());
    });
    // tests example of a (parallel) row number where each rank gets
    // a list of values that are all the same but in diff groups
    bodo::tests::test("test_par_row_number_diff_groups", [] {
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

        sorted_window_computation(partition_by_arrs, order_by_arrs,
                                  {Bodo_FTypes::row_number}, out_arrs, true);

        std::stringstream ss1;
        std::stringstream ss2;
        DEBUG_PrintColumn(ss1, out_arrs[0]);
        DEBUG_PrintColumn(ss2, expected_out);

        bodo::tests::check(ss1.str() == ss2.str());
    });
    // could make n_rows a parameter and combine with (1.)
    // all ranks have empty tables
    bodo::tests::test("test_par_row_number_empty_arrs", [] {
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

        sorted_window_computation(partition_by_arrs, order_by_arrs,
                                  {Bodo_FTypes::row_number}, out_arrs, true);

        std::stringstream ss1;
        std::stringstream ss2;
        DEBUG_PrintColumn(ss1, out_arrs[0]);
        DEBUG_PrintColumn(ss2, expected_out);

        bodo::tests::check(ss1.str() == ss2.str());
    });
    // every other rank has empty table
    bodo::tests::test("test_par_row_number_every_other_empty", [] {
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

        sorted_window_computation(partition_by_arrs, order_by_arrs,
                                  {Bodo_FTypes::row_number}, out_arrs, true);

        std::stringstream ss1;
        std::stringstream ss2;
        DEBUG_PrintColumn(ss1, out_arrs[0]);
        DEBUG_PrintColumn(ss2, expected_out);

        bodo::tests::check(ss1.str() == ss2.str());
    });
    // more than one group and they spill over
    bodo::tests::test("test_par_row_number_multiple_groups_per_rank", [] {
        std::vector<std::shared_ptr<array_info>> partition_by_arrs;
        std::vector<std::shared_ptr<array_info>> order_by_arrs;
        std::vector<std::shared_ptr<array_info>> out_arrs;
        size_t n = 3;
        size_t n_partition_cols = 1;
        size_t n_order_by_cols = 1;
        (void)n_order_by_cols;

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
            order_by_arrs.push_back(
                alloc_nullable_array_all_nulls(n, Bodo_CTypes::INT8));
        }
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

        sorted_window_computation(partition_by_arrs, order_by_arrs,
                                  {Bodo_FTypes::row_number}, out_arrs, true);

        std::stringstream ss1;
        std::stringstream ss2;
        DEBUG_PrintColumn(ss1, out_arrs[0]);
        DEBUG_PrintColumn(ss2, expected_out);

        bodo::tests::check(ss1.str() == ss2.str());
    });
});
