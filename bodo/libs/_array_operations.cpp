// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include <functional>
#include "_array_hash.h"
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_distributed.h"
#include "_shuffle.h"
#include "gfx/timsort.hpp"

/**
 * Compute the boolean array on output corresponds to the "isin" function in
 * matlab. each group, writes the result to a new output table containing one
 * row per group.
 *
 * @param out_arr the boolean array on output.
 * @param in_arr the list of values on input
 * @param in_values the list of values that we need to check with
 */
static void array_isin_kernel(array_info* out_arr, array_info* in_arr,
                              array_info* in_values) {
    CheckEqualityArrayType(in_arr, in_values);
    if (out_arr->dtype != Bodo_CTypes::_BOOL) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "array out_arr should be a boolean array");
        return;
    }
    uint32_t seed = 0xb0d01d80;

    int64_t len_values = in_values->length;
    uint32_t* hashes_values = new uint32_t[len_values];
    hash_array(hashes_values, in_values, (size_t)len_values, seed);

    int64_t len_in_arr = in_arr->length;
    uint32_t* hashes_in_arr = new uint32_t[len_in_arr];
    hash_array(hashes_in_arr, in_arr, (size_t)len_in_arr, seed);

    std::function<bool(int64_t, int64_t)> equal_fct =
        [&](int64_t const& pos1, int64_t const& pos2) -> bool {
        int64_t pos1_b, pos2_b;
        array_info *arr1_b, *arr2_b;
        if (pos1 < len_values) {
            arr1_b = in_values;
            pos1_b = pos1;
        } else {
            arr1_b = in_arr;
            pos1_b = pos1 - len_values;
        }
        if (pos2 < len_values) {
            arr2_b = in_values;
            pos2_b = pos2;
        } else {
            arr2_b = in_arr;
            pos2_b = pos2 - len_values;
        }
        return TestEqualColumn(arr1_b, pos1_b, arr2_b, pos2_b);
    };
    std::function<size_t(int64_t)> hash_fct =
        [&](int64_t const& pos) -> size_t {
        int64_t value;
        if (pos < len_values)
            value = hashes_values[pos];
        else
            value = hashes_in_arr[pos - len_values];
        return (size_t)value;
    };
    SET_CONTAINER<size_t, std::function<size_t(int64_t)>,
                  std::function<bool(int64_t, int64_t)>>
        eset({}, hash_fct, equal_fct);
    for (int64_t pos = 0; pos < len_values; pos++) {
        eset.insert(pos);
    }
    for (int64_t pos = 0; pos < len_in_arr; pos++) {
        bool test = eset.count(pos + len_values) == 1;
        out_arr->at<bool>(pos) = test;
    }
    delete[] hashes_in_arr;
    delete[] hashes_values;
}

template <class T>
static void fill_recv_data_inner(T* recv_buff, T* data, uint32_t* hashes,
                                 std::vector<int> const& send_disp, int n_pes,
                                 size_t n_rows) {
    std::vector<int> tmp_offset(send_disp);
    for (size_t i = 0; i < n_rows; i++) {
        size_t node = (size_t)hashes[i] % (size_t)n_pes;
        int ind = tmp_offset[node];
        data[i] = recv_buff[ind];
        tmp_offset[node]++;
    }
}

void array_isin(array_info* out_arr, array_info* in_arr, array_info* in_values,
                bool is_parallel) {
    if (!is_parallel) {
        return array_isin_kernel(out_arr, in_arr, in_values);
    }
    std::vector<array_info*> vect_in_values = {in_values};
    table_info table_in_values = table_info(vect_in_values);
    std::vector<array_info*> vect_in_arr = {in_arr};
    table_info table_in_arr = table_info(vect_in_arr);

    int64_t num_keys = 1;
    table_info* shuf_table_in_values =
        shuffle_table(&table_in_values, num_keys);
    // we need the comm_info and hashes for the reverse shuffling
    int n_pes;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    mpi_comm_info comm_info(n_pes, table_in_arr.columns);
    uint32_t seed = SEED_HASH_PARTITION;
    uint32_t* hashes = hash_keys(vect_in_arr, seed);
    comm_info.set_counts(hashes);
    table_info* shuf_table_in_arr =
        shuffle_table_kernel(&table_in_arr, hashes, n_pes, comm_info);
    // Creation of the output array.
    int64_t len = shuf_table_in_arr->columns[0]->length;
    array_info* shuf_out_arr =
        alloc_array(len, -1, out_arr->arr_type, out_arr->dtype, 0);
    // Calling isin on the shuffled info
    array_isin_kernel(shuf_out_arr, shuf_table_in_arr->columns[0],
                      shuf_table_in_values->columns[0]);

    // Deleting the data after usage
    delete_table_free_arrays(shuf_table_in_values);
    delete_table_free_arrays(shuf_table_in_arr);
    // Now the reverse shuffling operation. Since the array out_arr is not
    // directly handled by the comm_info, we have to get out hands dirty.
    MPI_Datatype mpi_typ = get_MPI_typ(out_arr->dtype);
    size_t n_rows = out_arr->length;
    std::vector<uint8_t> tmp_recv(n_rows);
    MPI_Alltoallv(shuf_out_arr->data1, comm_info.recv_count.data(),
                  comm_info.recv_disp.data(), mpi_typ, tmp_recv.data(),
                  comm_info.send_count.data(), comm_info.send_disp.data(),
                  mpi_typ, MPI_COMM_WORLD);
    fill_recv_data_inner<uint8_t>(tmp_recv.data(), (uint8_t*)out_arr->data1,
                                  hashes, comm_info.send_disp, n_pes, n_rows);
    // freeing just before returning.
    free_array(shuf_out_arr);
    delete[] hashes;
}

table_info* sort_values_table_local(table_info* in_table, int64_t n_key_t,
                                    int64_t* vect_ascending, bool na_position) {
    size_t n_rows = (size_t)in_table->nrows();
    size_t n_key = size_t(n_key_t);
#undef DEBUG_SORT_LOCAL
#ifdef DEBUG_SORT_LOCAL
    std::cout << "n_key_t=" << n_key_t << " na_position=" << na_position
              << "\n";
    for (int64_t iKey = 0; iKey < n_key_t; iKey++)
        std::cout << "iKey=" << iKey << "/" << n_key_t
                  << "  vect_ascending=" << vect_ascending[iKey] << "\n";
    std::cout << "INPUT:\n";
    DEBUG_PrintSetOfColumn(std::cout, in_table->columns);
    DEBUG_PrintRefct(std::cout, in_table->columns);
    std::cout << "n_rows=" << n_rows << " n_key=" << n_key << "\n";
#endif
    std::vector<size_t> V(n_rows);
    for (size_t i = 0; i < n_rows; i++) V[i] = i;
    std::function<bool(size_t, size_t)> f = [&](size_t const& iRow1,
                                                size_t const& iRow2) -> bool {
        size_t shift_key1 = 0, shift_key2 = 0;
        return KeyComparisonAsPython(n_key, vect_ascending, in_table->columns,
                                     shift_key1, iRow1, in_table->columns,
                                     shift_key2, iRow2, na_position);
    };
    gfx::timsort(V.begin(), V.end(), f);
    std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>> ListPairWrite(
        n_rows);
    for (size_t i = 0; i < n_rows; i++) ListPairWrite[i] = {V[i], -1};
    //
    table_info* ret_table = RetrieveTable(in_table, ListPairWrite, -1);
#ifdef DEBUG_SORT_LOCAL
    std::cout << "OUTPUT:\n";
    DEBUG_PrintSetOfColumn(std::cout, ret_table->columns);
    DEBUG_PrintRefct(std::cout, ret_table->columns);
#endif
    return ret_table;
}

table_info* sort_values_table(table_info* in_table, int64_t n_key_t,
                              int64_t* vect_ascending, bool na_position,
                              bool parallel) {
#undef DEBUG_SORT
    table_info* local_sort =
        sort_values_table_local(in_table, n_key_t, vect_ascending, na_position);
#ifdef DEBUG_SORT
    std::cout << "sort_values_table : local_sort:\n";
    DEBUG_PrintSetOfColumn(std::cout, local_sort->columns);
    DEBUG_PrintRefct(std::cout, local_sort->columns);
#endif
    if (!parallel) return local_sort;
    // preliminary definitions.
    int n_pes, myrank;
    int mpi_root = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    int64_t n_local = in_table->nrows();
    int64_t n_total;
    MPI_Allreduce(&n_local, &n_total, 1, MPI_LONG_LONG_INT, MPI_SUM,
                  MPI_COMM_WORLD);
#ifdef DEBUG_SORT
    std::cout << "n_local=" << n_local << " n_total=" << n_total << "\n";
#endif
    int MIN_SAMPLES = 1000000;
    int samplePointsPerPartitionHint = 20;
    int sampleSize =
        std::min(samplePointsPerPartitionHint * n_pes, MIN_SAMPLES);
    double fraction = std::min(
        double(sampleSize) / double(std::max(n_total, int64_t(1))), double(1));
    int64_t n_loc_sample = std::min(n_local, int64_t(ceil(fraction * n_local)));
#ifdef DEBUG_SORT
    std::cout << "sampleSize=" << sampleSize << " fraction=" << fraction
              << " n_loc_sample=" << n_loc_sample << "\n";
#endif
    //
    // building the samples.
    std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>> ListPairWrite(
        n_loc_sample);
    for (int64_t i = 0; i < n_loc_sample; i++) {
        int pos = rand() % n_local;
        ListPairWrite[i] = {std::ptrdiff_t(pos), -1};
    }
    table_info* samples = RetrieveTable(local_sort, ListPairWrite, n_key_t);
#ifdef DEBUG_SORT
    std::cout << "sort_values_table : samples:\n";
    DEBUG_PrintSetOfColumn(std::cout, samples->columns);
    DEBUG_PrintRefct(std::cout, samples->columns);
#endif
    // Collecting all samples
    table_info* all_samples = gather_table(samples, n_key_t);
#ifdef DEBUG_SORT
    if (myrank == 0) {
        std::cout << "sort_values_table : all_samples:\n";
        DEBUG_PrintSetOfColumn(std::cout, all_samples->columns);
        DEBUG_PrintRefct(std::cout, all_samples->columns);
    }
#endif

    // Computing the bounds
    table_info* pre_bounds = nullptr;
    if (myrank == mpi_root) {
        table_info* all_samples_sort = sort_values_table_local(
            all_samples, n_key_t, vect_ascending, na_position);
#ifdef DEBUG_SORT
        std::cout << "sort_values_table : all_samples_sort:\n";
        DEBUG_PrintSetOfColumn(std::cout, all_samples_sort->columns);
        DEBUG_PrintRefct(std::cout, all_samples_sort->columns);
#endif
        int64_t n_samples = all_samples_sort->nrows();
        int64_t step = ceil(double(n_samples) / double(n_pes));
        std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>>
            ListPairWriteBounds(n_pes - 1);
        for (int i = 0; i < n_pes - 1; i++) {
            int pos = std::min((i + 1) * step, n_samples - 1);
            ListPairWriteBounds[i] = {std::ptrdiff_t(pos), -1};
        }
        pre_bounds = RetrieveTable(all_samples_sort, ListPairWriteBounds, -1);
#ifdef DEBUG_SORT
        std::cout << "sort_values_table : pre_bounds:\n";
        DEBUG_PrintSetOfColumn(std::cout, pre_bounds->columns);
        DEBUG_PrintRefct(std::cout, pre_bounds->columns);
#endif
        delete_table_free_arrays(all_samples_sort);
    }
    delete_table_free_arrays(samples);
    delete_table_free_arrays(all_samples);
    // broadcasting the bounds
    table_info* bounds = broadcast_table(pre_bounds, n_key_t);
#ifdef DEBUG_SORT
    std::cout << "sort_values_table : bounds:\n";
    DEBUG_PrintSetOfColumn(std::cout, bounds->columns);
    DEBUG_PrintRefct(std::cout, bounds->columns);
#endif
    if (myrank == mpi_root) delete_table_free_arrays(pre_bounds);
    // Now computing to which process it all goes.
    std::vector<uint32_t> hashes_v(n_local);
    uint32_t node_id = 0;
    for (int64_t i = 0; i < n_local; i++) {
        size_t shift_key1 = 0, shift_key2 = 0;
        // using 'while' since a partition can be empty which needs to be skipped
        while (node_id < uint32_t(n_pes - 1) &&
               KeyComparisonAsPython(n_key_t, vect_ascending, bounds->columns,
                                     shift_key2, node_id, local_sort->columns,
                                     shift_key1, i, na_position)) {
            node_id++;
        }
        hashes_v[i] = node_id;
#ifdef DEBUG_SORT
        std::cout << "i=" << i << "  node_id=" << node_id << "\n";
#endif
    }
#ifdef DEBUG_SORT
    std::cout << "Before bounds deallocation\n";
#endif
    delete_table_free_arrays(bounds);
#ifdef DEBUG_SORT
    std::cout << " After bounds deallocation\n";
#endif
    // Now shuffling all the data
    mpi_comm_info comm_info(n_pes, local_sort->columns);
#ifdef DEBUG_SORT
    std::cout << " We have comm_info\n";
#endif
    comm_info.set_counts(hashes_v.data());
#ifdef DEBUG_SORT
    std::cout << " We have set_counts\n";
#endif
    table_info* collected_table =
        shuffle_table_kernel(local_sort, hashes_v.data(), n_pes, comm_info);
#ifdef DEBUG_SORT
    std::cout << " We have collected_table\n";
#endif
#ifdef DEBUG_SORT
    std::cout << "sort_values_table : collected_table:\n";
    DEBUG_PrintSetOfColumn(std::cout, collected_table->columns);
    DEBUG_PrintRefct(std::cout, collected_table->columns);
#endif
    delete_table_free_arrays(local_sort);
    // Now final local sorting from all the stuff we collected.
    table_info* ret_table = sort_values_table_local(
        collected_table, n_key_t, vect_ascending, na_position);
#ifdef DEBUG_SORT
    std::cout << "sort_values_table : ret_table:\n";
    DEBUG_PrintSetOfColumn(std::cout, ret_table->columns);
    DEBUG_PrintRefct(std::cout, ret_table->columns);
#endif
    delete_table_free_arrays(collected_table);
    return ret_table;
}

/** This function is the inner function for the dropping of duplicated rows.
 * This C++ code is used for the drop_duplicates.
 * Two support cases:
 * ---The local computation where we store two values (first and last) in order
 *    to deal with all eventualities
 * ---The final case where depending on the case we store the first, last or
 *    none if more than 2 are considered.
 *
 * As for the join, this relies on using hash keys for the partitionning.
 * The computation is done locally.
 *
 * External function used are "RetrieveTable" and "TestEqual"
 *
 * @param in_table : the input table
 * @param sum_value: the uint64_t containing all the values together.
 * @param keep: integer specifying the expected behavior.
 *        keep = 0 corresponds to the case of keep="first" keep first entry
 *        keep = 1 corresponds to the case of keep="last" keep last entry
 *        keep = 2 corresponds to the case of keep=False : remove all duplicates
 * @param step: integer specifying the work done
 *              2 corresponds to the first step of the operation where we
 * collate the rows on the computational node 1 corresponds to the second step
 * of the operation after the rows have been merged on the computation
 * @return the vector of pointers to be used.
 */
static table_info* drop_duplicates_table_inner(table_info* in_table,
                                               int64_t num_keys, int64_t keep,
                                               int step) {
#undef DEBUG_DD
    size_t n_rows = (size_t)in_table->nrows();
    std::vector<array_info*> key_arrs(num_keys);
    for (size_t iKey = 0; iKey < size_t(num_keys); iKey++)
        key_arrs[iKey] = in_table->columns[iKey];
#ifdef DEBUG_DD
    size_t n_col = in_table->ncols();
    std::cout << "INPUT:\n";
    std::cout << "n_col=" << n_col << " n_rows=" << n_rows
              << " num_keys=" << num_keys << "\n";
    DEBUG_PrintSetOfColumn(std::cout, in_table->columns);
    DEBUG_PrintRefct(std::cout, in_table->columns);
#endif

    uint32_t seed = 0xb0d01287;
    uint32_t* hashes = hash_keys(key_arrs, seed);
    /* This is a function for computing the hash (here returning computed value)
     * This is the first function passed as argument for the map function.
     *
     * Note that the hash is a size_t (as requested by standard and so 8 bytes
     * on x86-64) but our hashes are int32_t
     *
     * @param iRow is the first row index for the comparison
     * @return the hash itself
     */
    std::function<size_t(size_t)> hash_fct = [&](size_t const& iRow) -> size_t {
        return size_t(hashes[iRow]);
    };
    /* This is a function for testing equality of rows.
     * This is the second lambda passed to the map function.
     *
     * We use the TestEqual function precedingly defined.
     *
     * @param iRowA is the first row index for the comparison
     * @param iRowB is the second row index for the comparison
     * @return true/false depending on the case.
     */
    std::function<bool(size_t, size_t)> equal_fct =
        [&](size_t const& iRowA, size_t const& iRowB) -> bool {
        size_t shift_A = 0, shift_B = 0;
        bool test =
            TestEqual(key_arrs, num_keys, shift_A, iRowA, shift_B, iRowB);
        return test;
    };
    // The entSet contains the hash of the table.
    // We address the entry by the row index.
    MAP_CONTAINER<size_t, size_t, std::function<size_t(size_t)>,
                  std::function<bool(size_t, size_t)>>
        entSet({}, hash_fct, equal_fct);
    // The loop over the short table.
    // entries are stored one by one and all of them are put even if identical
    // in value.
    //
    // In the first case we keep only one entry.
    auto RetrievePair1 =
        [&]() -> std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>> {
        std::vector<int64_t> ListRow;
        uint64_t next_ent = 0;
        for (size_t i_row = 0; i_row < n_rows; i_row++) {
            size_t& group = entSet[i_row];
            if (group == 0) {
                next_ent++;
                group = next_ent;
                ListRow.emplace_back(i_row);
            } else {
                size_t pos = group - 1;
                if (keep == 0) {  // keep first entry. So do nothing here
                }
                if (keep == 1) {  // keep last entry. So update the list
                    ListRow[pos] = i_row;
                }
                if (keep == 2) {  // Case of False. So put it to -1.
                    ListRow[pos] = -1;
                }
            }
        }
        std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>> ListPairWrite;
        for (auto& eRow : ListRow) {
            if (eRow != -1) ListPairWrite.push_back({eRow, -1});
        }
        return std::move(ListPairWrite);
    };
    // In this case we store the pairs of values, the first and the last.
    // This allows to reach conclusions in all possible cases.
    auto RetrievePair2 =
        [&]() -> std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>> {
        std::vector<std::pair<int64_t, int64_t>> ListRowPair;
        size_t next_ent = 0;
        for (size_t i_row = 0; i_row < n_rows; i_row++) {
            size_t& group = entSet[i_row];
            if (group == 0) {
                next_ent++;
                group = next_ent;
                ListRowPair.push_back({i_row, -1});
            } else {
                size_t pos = group - 1;
                ListRowPair[pos].second = i_row;
            }
        }
        std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>> ListPairWrite;
        for (auto& eRowPair : ListRowPair) {
            if (eRowPair.first != -1)
                ListPairWrite.push_back({eRowPair.first, -1});
            if (eRowPair.second != -1)
                ListPairWrite.push_back({eRowPair.second, -1});
        }
        return std::move(ListPairWrite);
    };
    std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>> ListPairWrite;
    if (step == 1 || keep == 0 || keep == 1)
        ListPairWrite = RetrievePair1();
    else
        ListPairWrite = RetrievePair2();
#ifdef DEBUG_DD
    std::cout << "|ListPairWrite|=" << ListPairWrite.size() << "\n";
#endif
    // Now building the out_arrs array.
    table_info* ret_table = RetrieveTable(in_table, ListPairWrite, -1);
    //
    delete[] hashes;
#ifdef DEBUG_DD
    std::cout << "OUTPUT:\n";
    DEBUG_PrintSetOfColumn(std::cout, ret_table->columns);
    DEBUG_PrintRefct(std::cout, ret_table->columns);
#endif
    return ret_table;
}

table_info* drop_duplicates_table(table_info* in_table, bool is_parallel,
                                  int64_t num_keys, int64_t keep) {
#ifdef DEBUG_DD
    std::cout << "is_parallel=" << is_parallel << "\n";
#endif
    // serial case
    if (!is_parallel) {
        return drop_duplicates_table_inner(in_table, num_keys, keep, 1);
    }
        // parallel case
        // pre reduction of duplicates
#ifdef DEBUG_DD
    std::cout << "Before the drop duplicates on the local nodes\n";
#endif
    table_info* red_table =
        drop_duplicates_table_inner(in_table, num_keys, keep, 2);
    // shuffling of values
#ifdef DEBUG_DD
    std::cout << "Before the shuffling\n";
#endif
    table_info* shuf_table = shuffle_table(red_table, num_keys);
    delete_table_free_arrays(red_table);
    // reduction after shuffling
#ifdef DEBUG_DD
    std::cout << "Before the second shuffling\n";
#endif
    table_info* ret_table =
        drop_duplicates_table_inner(shuf_table, num_keys, keep, 1);
    delete_table_free_arrays(shuf_table);
#ifdef DEBUG_DD
    std::cout << "Final returning table\n";
    DEBUG_PrintSetOfColumn(std::cout, ret_table->columns);
    DEBUG_PrintRefct(std::cout, ret_table->columns);
#endif
    // returning table
    return ret_table;
}
