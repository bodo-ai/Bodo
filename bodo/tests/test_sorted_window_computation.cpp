#include <iostream>
#include <memory>
#include <sstream>
#include "../libs/_array_utils.h"
#include "../libs/_bodo_common.h"
#include "../libs/_groupby_col_set.h"
#include "../libs/_window_compute.h"
#include "./table_generator.hpp"
#include "./test.hpp"

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
    std::vector<std::shared_ptr<array_info>> out_arrs,
    std::shared_ptr<array_info> expected_out, bool is_parallel = true) {
    sorted_window_computation(partition_by_arrs, order_by_arrs, {window_func},
                              out_arrs, is_parallel);
    std::stringstream ss1;
    std::stringstream ss2;
    DEBUG_PrintColumn(ss1, out_arrs[0]);
    DEBUG_PrintColumn(ss2, expected_out);

    bodo::tests::check(ss1.str() == ss2.str());
}

// TODO: template these tests and/or add parmaeter to test different types
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
                                    order_by_arrs, out_arrs, expected_out);
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
                                    order_by_arrs, out_arrs, expected_out);
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
                                    order_by_arrs, out_arrs, expected_out);
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
                                    order_by_arrs, out_arrs, expected_out);
    });
    bodo::tests::test("test_par_row_number_multiple_groups_per_rank", [] {
        // more than one group and they spill over
        std::vector<std::shared_ptr<array_info>> partition_by_arrs;
        std::vector<std::shared_ptr<array_info>> order_by_arrs;
        std::vector<std::shared_ptr<array_info>> out_arrs;
        size_t n = 3;
        size_t n_partition_cols = 1;
        size_t n_order_by_cols = 1;

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

        verify_sorted_window_output(Bodo_FTypes::row_number, partition_by_arrs,
                                    order_by_arrs, out_arrs, expected_out);
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
                                    order_by_arrs, out_arrs, expected_out,
                                    false);
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
                                    order_by_arrs, out_arrs, expected_out);
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
                                    order_by_arrs, out_arrs, expected_out);
    });
    bodo::tests::test("test_par_rank_multiple_groups", [] {
        // tests rank in the case of multiple groups on each rank
        // same set up as test_par_row_number_multiple_groups_per_rank
        std::vector<std::shared_ptr<array_info>> partition_by_arrs;
        std::vector<std::shared_ptr<array_info>> order_by_arrs;
        std::vector<std::shared_ptr<array_info>> out_arrs;
        size_t n = 4;
        size_t n_partition_cols = 1;
        size_t n_order_by_cols = 1;

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
                                    order_by_arrs, out_arrs, expected_out);
    });
    bodo::tests::test("test_par_rank_with_holes", [] {
        // same set up as test_par_rank_multiple_groups except every other rank
        // is empty
        std::vector<std::shared_ptr<array_info>> partition_by_arrs;
        std::vector<std::shared_ptr<array_info>> order_by_arrs;
        std::vector<std::shared_ptr<array_info>> out_arrs;
        size_t n_partition_cols = 1;
        size_t n_order_by_cols = 1;

        int myrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        size_t n = myrank % 2 == 0 ? 4 : 0;

        std::shared_ptr<array_info> empty_arr =
            alloc_numpy(0, Bodo_CTypes::INT64);

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
                order_by_arrs.push_back(
                    bodo::tests::cppToBodoArr({3, 4, 1, 2}));
            } else {
                order_by_arrs.push_back(empty_arr);
            }
        }
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
                                    order_by_arrs, out_arrs, expected_out);
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
        sorted_window_computation(partition_by_arrs, order_by_arrs,
                                  {Bodo_FTypes::rank}, out_arrs, true);

        verify_sorted_window_output(Bodo_FTypes::rank, partition_by_arrs,
                                    order_by_arrs, out_arrs, expected_out);
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
                                    order_by_arrs, out_arrs, expected_out,
                                    false);
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
                                    order_by_arrs, out_arrs, expected_out);
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
                                    order_by_arrs, out_arrs, expected_out);
    });
    bodo::tests::test("test_par_dense_rank_multiple_groups", [] {
        // tests dense_rank in the case of multiple groups on each rank
        // same set up as test_par_row_number_multiple_groups_per_rank
        std::vector<std::shared_ptr<array_info>> partition_by_arrs;
        std::vector<std::shared_ptr<array_info>> order_by_arrs;
        std::vector<std::shared_ptr<array_info>> out_arrs;
        size_t n = 4;
        size_t n_partition_cols = 1;
        size_t n_order_by_cols = 1;

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
                                    order_by_arrs, out_arrs, expected_out);
    });
    bodo::tests::test("test_par_dense_rank_with_holes", [] {
        // same set up as test_par_dense_rank_multiple_groups except every other
        // rank is empty
        std::vector<std::shared_ptr<array_info>> partition_by_arrs;
        std::vector<std::shared_ptr<array_info>> order_by_arrs;
        std::vector<std::shared_ptr<array_info>> out_arrs;
        size_t n_partition_cols = 1;
        size_t n_order_by_cols = 1;

        int myrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        size_t n = myrank % 2 == 0 ? 4 : 0;

        std::shared_ptr<array_info> empty_arr =
            alloc_numpy(0, Bodo_CTypes::INT64);

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
                order_by_arrs.push_back(
                    bodo::tests::cppToBodoArr({3, 4, 1, 2}));
            } else {
                order_by_arrs.push_back(empty_arr);
            }
        }
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
                                    order_by_arrs, out_arrs, expected_out);
    });
});
