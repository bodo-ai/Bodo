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

// Creates an integer column from vectors of ints and nulls
template <Bodo_CTypes::CTypeEnum dtype, typename T>
    requires(dtype != Bodo_CTypes::_BOOL)
std::shared_ptr<array_info> nullable_array_from_vector(
    std::vector<T> numbers, std::vector<bool> nulls) {
    size_t length = numbers.size();
    auto result = alloc_nullable_array_no_nulls(length, dtype);
    T *buffer = result->data1<bodo_array_type::NULLABLE_INT_BOOL, T>();
    for (size_t i = 0; i < length; i++) {
        if (nulls[i]) {
            buffer[i] = (T)numbers[i];
        } else {
            result->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i, false);
        }
    }
    return result;
}

// Special case of nullable_array_from_vector for booleans
template <Bodo_CTypes::CTypeEnum dtype, typename T>
    requires(dtype == Bodo_CTypes::_BOOL)
std::shared_ptr<array_info> nullable_array_from_vector(
    std::vector<bool> booleans, std::vector<bool> nulls) {
    size_t length = booleans.size();
    auto result = alloc_nullable_array_no_nulls(length, dtype);
    uint8_t *buffer =
        result->data1<bodo_array_type::NULLABLE_INT_BOOL, uint8_t>();
    for (size_t i = 0; i < length; i++) {
        if (nulls[i]) {
            SetBitTo(buffer, i, booleans[i]);
        } else {
            result->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i, false);
        }
    }
    return result;
}

// Variant of nullable_array_from_vector to build a string array from vectors
std::shared_ptr<array_info> string_array_from_vector(
    bodo::vector<std::string> strings, std::vector<bool> nulls,
    Bodo_CTypes::CTypeEnum dtype) {
    size_t length = strings.size();

    bodo::vector<uint8_t> null_bitmask((length + 7) >> 3, 0);
    for (size_t i = 0; i < length; i++) {
        SetBitTo(null_bitmask.data(), i, nulls[i]);
    }
    return create_string_array(dtype, null_bitmask, strings, -1);
}

// Variant of nullable_array_from_vector to build a dict array from vectors
std::shared_ptr<array_info> dict_array_from_vector(
    bodo::vector<std::string> strings, std::vector<int32_t> indices,
    std::vector<bool> nulls) {
    std::vector<bool> string_nulls(strings.size(), true);
    std::shared_ptr<array_info> dict_arr =
        string_array_from_vector(strings, string_nulls, Bodo_CTypes::STRING);
    std::shared_ptr<array_info> index_arr =
        nullable_array_from_vector<Bodo_CTypes::INT32, int32_t>(indices, nulls);
    return create_dict_string_array(dict_arr, index_arr);
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
    sorted_window_computation(partition_by_arrs, order_by_arrs, window_args,
                              window_offset_indices, {window_func}, out_arrs,
                              expected_out->length, is_parallel);

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
                                    order_by_arrs, {}, {}, out_arrs,
                                    expected_out);
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

    bodo::tests::test("partitionless_max_integer", [] {
        int myrank, num_ranks;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

        // Create the total answer array, then have each rank select a
        // different subset of the rows
        size_t n_total_rows = 10000;
        std::vector<int64_t> in_integers(n_total_rows, -1);
        std::vector<bool> in_nulls(n_total_rows, true);
        for (size_t i = 0; i < n_total_rows; i++) {
            in_integers[i] = -((i - 12345) * (i - 1234));
        }
        std::shared_ptr<array_info> in_arr =
            nullable_array_from_vector<Bodo_CTypes::INT64, int64_t>(in_integers,
                                                                    in_nulls);
        std::vector<int64_t> selection_vector;
        for (size_t i = 0; i < n_total_rows; i++) {
            auto marker = static_cast<int64_t>(i + (i >> 1) * (i >> 1));
            if (marker % (int64_t)num_ranks == myrank) {
                selection_vector.push_back(i);
            }
        }
        in_arr = RetrieveArray_SingleColumn(in_arr, selection_vector);

        // Create a singleton array containing the one global answer, and
        // broadcast the correct length to obtain the refsol.
        std::shared_ptr<array_info> answer_singleton =
            nullable_array_from_vector<Bodo_CTypes::INT64, int64_t>({30863580},
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

        verify_sorted_window_output(Bodo_FTypes::max, {}, {}, {in_arr}, {0, 1},
                                    out_arrs, expected_out);
    });

    bodo::tests::test("partitionless_max_all_null", [] {
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
            nullable_array_from_vector<Bodo_CTypes::INT64, int32_t>(in_integers,
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
                                    order_by_arrs, {}, {}, out_arrs,
                                    expected_out);
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
                                    order_by_arrs, {}, {}, out_arrs,
                                    expected_out);
    });
});
