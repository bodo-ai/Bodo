// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include <functional>
#include <random>
#include <set>
#include "_array_hash.h"
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_distributed.h"
#include "_shuffle.h"
#include "gfx/timsort.hpp"

#undef DEBUG_DD_SYMBOL
#undef DEBUG_DD_FULL
#undef DEBUG_SORT_LOCAL_SYMBOL
#undef DEBUG_SORT_LOCAL_FULL
#undef DEBUG_SORT_SYMBOL
#undef DEBUG_SORT_FULL
#undef DEBUG_SAMPLE

//
//   ARRAY ISIN
//

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
    uint32_t seed = SEED_HASH_CONTAINER;

    int64_t len_values = in_values->length;
    std::vector<uint32_t> hashes_values(len_values);
    hash_array(hashes_values.data(), in_values, (size_t)len_values, seed);

    int64_t len_in_arr = in_arr->length;
    std::vector<uint32_t> hashes_in_arr(len_in_arr);
    hash_array(hashes_in_arr.data(), in_arr, (size_t)len_in_arr, seed);

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
        return TestEqualColumn(arr1_b, pos1_b, arr2_b, pos2_b, true);
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
    UNORD_SET_CONTAINER<size_t, std::function<size_t(int64_t)>,
                        std::function<bool(int64_t, int64_t)>>
        eset({}, hash_fct, equal_fct);
    for (int64_t pos = 0; pos < len_values; pos++) {
        eset.insert(pos);
    }
    for (int64_t pos = 0; pos < len_in_arr; pos++) {
        bool test = eset.count(pos + len_values) == 1;
        out_arr->at<bool>(pos) = test;
    }
}

void array_isin(array_info* out_arr, array_info* in_arr, array_info* in_values,
                bool is_parallel) {
    try {
        if (!is_parallel) {
            array_isin_kernel(out_arr, in_arr, in_values);
            decref_array(out_arr);
            decref_array(in_arr);
            decref_array(in_values);
            return;
        }
        std::vector<array_info*> vect_in_values = {in_values};
        table_info table_in_values = table_info(vect_in_values);
        std::vector<array_info*> vect_in_arr = {in_arr};
        table_info* table_in_arr = new table_info(vect_in_arr);

        int64_t num_keys = 1;
        table_info* shuf_table_in_values =
            shuffle_table(&table_in_values, num_keys);
        // we need the comm_info and hashes for the reverse shuffling
        mpi_comm_info comm_info(table_in_arr->columns);
        uint32_t* hashes =
            hash_keys_table(table_in_arr, 1, SEED_HASH_PARTITION);
        comm_info.set_counts(hashes);
        table_info* shuf_table_in_arr =
            shuffle_table_kernel(table_in_arr, hashes, comm_info);
        // Creation of the output array.
        int64_t len = shuf_table_in_arr->columns[0]->length;
        array_info* shuf_out_arr =
            alloc_array(len, -1, -1, out_arr->arr_type, out_arr->dtype, 0,
                        out_arr->num_categories);
        // Calling isin on the shuffled info
        array_isin_kernel(shuf_out_arr, shuf_table_in_arr->columns[0],
                          shuf_table_in_values->columns[0]);

        // Deleting the data after usage
        delete_table_decref_arrays(shuf_table_in_values);
        delete_table_decref_arrays(shuf_table_in_arr);
        // Now the reverse shuffling operation. Since the array out_arr is not
        // directly handled by the comm_info, we have to get out hands dirty.
        MPI_Datatype mpi_typ = get_MPI_typ(out_arr->dtype);
        size_t n_rows = out_arr->length;
        std::vector<uint8_t> tmp_recv(n_rows);
        bodo_alltoallv(shuf_out_arr->data1, comm_info.recv_count,
                       comm_info.recv_disp, mpi_typ, tmp_recv.data(),
                       comm_info.send_count, comm_info.send_disp, mpi_typ,
                       MPI_COMM_WORLD);
        fill_recv_data_inner<uint8_t>(tmp_recv.data(), (uint8_t*)out_arr->data1,
                                      hashes, comm_info.send_disp,
                                      comm_info.n_pes, n_rows);
        // free temporary shuffle array
        delete_info_decref_array(shuf_out_arr);
        // release extra reference for output array (array_info wrapper's
        // reference)
        decref_array(out_arr);
        delete table_in_arr;
        delete[] hashes;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return;
    }
}

//
//   SORT VALUES
//

table_info* sort_values_table_local(table_info* in_table, int64_t n_key_t,
                                    int64_t* vect_ascending, bool na_position) {
    tracing::Event ev("sort_values_table_local");
    size_t n_rows = (size_t)in_table->nrows();
    size_t n_key = size_t(n_key_t);
#ifdef DEBUG_SORT_LOCAL_SYMBOL
    std::cout << "n_key_t=" << n_key_t << " na_position=" << na_position
              << "\n";
    for (int64_t iKey = 0; iKey < n_key_t; iKey++)
        std::cout << "iKey=" << iKey << "/" << n_key_t
                  << "  vect_ascending=" << vect_ascending[iKey] << "\n";
    std::cout << "INPUT (sort_values_table):\n";
    DEBUG_PrintRefct(std::cout, in_table->columns);
    DEBUG_PrintSetOfColumn(std::cout, in_table->columns);
    std::cout << "n_rows=" << n_rows << " n_key=" << n_key << "\n";
#endif
    std::vector<int64_t> ListIdx(n_rows);
    for (size_t i = 0; i < n_rows; i++) ListIdx[i] = i;

    // The comparison operator gets called many times by timsort so any overhead
    // can influence the sort time significantly
    if (n_key == 1) {
        // comparison operator with less overhead than the general n_key > 1
        // case. We call KeyComparisonAsPython_Column directly without looping
        // through the keys, assume fixed values for some parameters and pass
        // less parameters around
        array_info* key_col = in_table->columns[0];
        bool ascending = vect_ascending[0];
        if (ascending) {
            const bool na_position_bis = (!na_position) ^ ascending;
            const auto f = [&](size_t const& iRow1,
                               size_t const& iRow2) -> bool {
                int test = KeyComparisonAsPython_Column(
                    na_position_bis, key_col, iRow1, key_col, iRow2);
                if (test) return test > 0;
                return false;
            };
            gfx::timsort(ListIdx.begin(), ListIdx.end(), f);
        } else {
            const bool na_position_bis = (!na_position) ^ ascending;
            const auto f = [&](size_t const& iRow1,
                               size_t const& iRow2) -> bool {
                int test = KeyComparisonAsPython_Column(
                    na_position_bis, key_col, iRow1, key_col, iRow2);
                if (test) return test < 0;
                return false;
            };
            gfx::timsort(ListIdx.begin(), ListIdx.end(), f);
        }
    } else {
        const auto f = [&](size_t const& iRow1, size_t const& iRow2) -> bool {
            size_t shift_key1 = 0, shift_key2 = 0;
            bool test = KeyComparisonAsPython(
                n_key, vect_ascending, in_table->columns, shift_key1, iRow1,
                in_table->columns, shift_key2, iRow2, na_position);
            return test;
        };
        gfx::timsort(ListIdx.begin(), ListIdx.end(), f);
    }
    table_info* ret_table = RetrieveTable(in_table, ListIdx, -1);
#ifdef DEBUG_SORT_LOCAL_SYMBOL
    std::cout << "OUTPUT (sort_values_table_local) in_table:\n";
    DEBUG_PrintRefct(std::cout, in_table->columns);
    std::cout << "OUTPUT (sort_values_table_local) ret_table:\n";
    DEBUG_PrintRefct(std::cout, ret_table->columns);
#ifdef DEBUG_SORT_LOCAL_FULL
    DEBUG_PrintSetOfColumn(std::cout, ret_table->columns);
#endif
#endif
    return ret_table;
}

table_info* sort_values_table(table_info* in_table, int64_t n_key_t,
                              int64_t* vect_ascending, bool na_position,
                              bool parallel) {
    try {
        tracing::Event ev("sort_values_table");
        int64_t n_local = in_table->nrows();
        table_info* local_sort = sort_values_table_local(
            in_table, n_key_t, vect_ascending, na_position);
        if (!parallel) return local_sort;
        // preliminary definitions.
        tracing::Event ev_sample("sort sampling");
        int n_pes, myrank;
        int mpi_root = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        int64_t n_total;
        MPI_Allreduce(&n_local, &n_total, 1, MPI_LONG_LONG_INT, MPI_SUM,
                      MPI_COMM_WORLD);
        if (n_total == 0) return local_sort;

        // Sample sort with random sampling (we use a fixed seed for the random
        // generator for deterministic results across runs).
        // With random sampling as described by Blelloch et al. [1], each
        // processor divides its local sorted input into s blocks of size (N/ps)
        // (where N is the global number of rows, p the number of processors,
        // s is the oversampling ratio) and samples a random key in each block.
        // The samples are gathered on rank 0 and rank 0 chooses the splitters
        // by picking evenly spaced keys from the overall sample of size ps.
        // We choose the global of number of samples according to this theorem:
        // "With O(p log N / epsilon^2) samples overall, sample sort with
        // random sampling achieves (1 + epsilon) load balance with high
        // probability." [1] Guy E. Blelloch, Charles E. Leiserson, Bruce M
        // Maggs, C Greg Plaxton, Stephen J
        //     Smith, and Marco Zagha. 1998. An experimental analysis of
        //     parallel sorting algorithms. Theory of Computing Systems 31, 2
        //     (1998), 135-167.
        std::mt19937 gen(1234567890);
        double epsilon = 0.1;
        char* bodo_sort_epsilon = std::getenv("BODO_SORT_EPSILON");
        if (bodo_sort_epsilon) {
            epsilon = std::stod(bodo_sort_epsilon);
        }
        if (epsilon < 0) throw std::runtime_error("sort_values: epsilon < 0");
        double n_global_sample = n_pes * log(n_total) / pow(epsilon, 2.0);
        n_global_sample =
            std::min(std::max(n_global_sample, double(n_pes)), double(n_total));
        int64_t n_loc_sample =
            std::min(n_global_sample / n_pes, double(n_local));
        double block_size = double(n_local) / n_loc_sample;
        std::vector<int64_t> ListIdx(n_loc_sample);
        double cur_lo = 0;
        for (int64_t i = 0; i < n_loc_sample; i++) {
            int64_t lo = round(cur_lo);
            int64_t hi = round(cur_lo + block_size) - 1;
            ListIdx[i] = std::uniform_int_distribution<int64_t>(lo, hi)(gen);
            cur_lo += block_size;
        }
        for (int64_t i_key = 0; i_key < n_key_t; i_key++)
            incref_array(local_sort->columns[i_key]);
        table_info* samples = RetrieveTable(local_sort, ListIdx, n_key_t);

        // Collecting all samples
        bool all_gather = false;
        table_info* all_samples = gather_table(samples, n_key_t, all_gather);
        delete_table(samples);

        // Computing the bounds (splitters) on root
        table_info* pre_bounds = nullptr;
        if (myrank == mpi_root) {
            table_info* all_samples_sort = sort_values_table_local(
                all_samples, n_key_t, vect_ascending, na_position);
            int64_t n_samples = all_samples_sort->nrows();
            int64_t step = ceil(double(n_samples) / double(n_pes));
            std::vector<int64_t> ListIdxBounds(n_pes - 1);
            for (int i = 0; i < n_pes - 1; i++) {
                size_t pos = std::min((i + 1) * step, n_samples - 1);
                ListIdxBounds[i] = pos;
            }
            pre_bounds = RetrieveTable(all_samples_sort, ListIdxBounds, -1);
            delete_table(all_samples_sort);
        } else {
            // all ranks need to trace the event
            // TODO the right way would be to call sort_values_table_local with an empty table
            tracing::Event ev_dummy("sort_values_table_local");
        }
        delete_table(all_samples);

        // broadcasting the bounds
        // The local_sort is used as reference for the data type of the array.
        // This is because pre_bounds is NULL for ranks != 0
        table_info* bounds = broadcast_table(local_sort, pre_bounds, n_key_t);
        if (myrank == mpi_root) delete_table(pre_bounds);
        // Now computing to which process it all goes.
        tracing::Event ev_hashes("compute_destinations");
        std::vector<uint32_t> hashes_v(n_local);
        uint32_t node_id = 0;
        for (int64_t i = 0; i < n_local; i++) {
            size_t shift_key1 = 0, shift_key2 = 0;
            // using 'while' since a partition can be empty which needs to be
            // skipped
            while (node_id < uint32_t(n_pes - 1) &&
                   KeyComparisonAsPython(n_key_t, vect_ascending,
                                         bounds->columns, shift_key2, node_id,
                                         local_sort->columns, shift_key1, i,
                                         na_position)) {
                node_id++;
            }
            hashes_v[i] = node_id;
        }
        delete_table_decref_arrays(bounds);

        // Now shuffle all the data
        mpi_comm_info comm_info(local_sort->columns);
        comm_info.set_counts(hashes_v.data());
        ev_hashes.finalize();
        ev_sample.finalize();
        table_info* collected_table =
            shuffle_table_kernel(local_sort, hashes_v.data(), comm_info);
        // NOTE: shuffle_table_kernel decrefs input arrays
        delete_table(local_sort);

        // Final local sorting
        table_info* ret_table = sort_values_table_local(
            collected_table, n_key_t, vect_ascending, na_position);
        delete_table(collected_table);
        return ret_table;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

//
//   DROP DUPLICATES
//

namespace {
/**
 * Look up a hash in a table.
 *
 * Don't use std::function to reduce call overhead.
 */
struct HashDropDuplicates {
    /* This is a function for computing the hash (here returning computed value)
     * This is the first function passed as argument for the map function.
     *
     * Note that the hash is a size_t (as requested by standard and so 8 bytes
     * on x86-64) but our hashes are int32_t
     *
     * @param iRow is the first row index for the comparison
     * @return the hash itself
     */
    size_t operator()(size_t const iRow) const {
        return static_cast<size_t>(hashes[iRow]);
    }
    uint32_t* hashes;
};

/**
 * Check if keys are equal by lookup in a table.
 *
 * Don't use std::function to reduce call overhead.
 */
struct KeyEqualDropDuplicates {
    /* This is a function for testing equality of rows.
     * This is the second lambda passed to the map function.
     *
     * We use the TestEqual function precedingly defined.
     *
     * @param iRowA is the first row index for the comparison
     * @param iRowB is the second row index for the comparison
     * @return true/false depending on the case.
     */
    bool operator()(size_t const iRowA, size_t const iRowB) const {
        size_t shift_A = 0, shift_B = 0;
        return TestEqual(*key_arrs, num_keys, shift_A, iRowA, shift_B, iRowB);
    }
    std::vector<array_info*>* key_arrs;
    int64_t num_keys;
};
}  // namespace

/** This function is for dropping duplicated keys.
 * It is used for drop_duplicates_keys and
 * then transitively by compute_categorical_index
 *
 * External function used are "RetrieveTable" and "TestEqual"
 *
 * @param in_table : the input table
 * @param num_keys : the number of keys
 * @param dropna: whether we drop null keys or not.
 * @return the table to be used.
 */
static table_info* drop_duplicates_keys_inner(table_info* in_table,
                                              int64_t num_keys,
                                              bool dropna = true) {
    size_t n_rows = (size_t)in_table->nrows();
    std::vector<array_info*> key_arrs(in_table->columns.begin(),
                                      in_table->columns.begin() + num_keys);
    uint32_t seed = SEED_HASH_CONTAINER;
    uint32_t* hashes = hash_keys(key_arrs, seed);
    HashDropDuplicates hash_fct{hashes};
    KeyEqualDropDuplicates equal_fct{&key_arrs, num_keys};
    UNORD_MAP_CONTAINER<size_t, size_t, HashDropDuplicates,
                        KeyEqualDropDuplicates>
        entSet({}, hash_fct, equal_fct);
    //
    std::vector<int64_t> ListRow;
    uint64_t next_ent = 0;
    bool has_nulls = dropna && does_keys_have_nulls(key_arrs);
    auto is_ok = [&](size_t i_row) -> bool {
        if (!has_nulls) return true;
        return !does_row_has_nulls(key_arrs, i_row);
    };
    for (size_t i_row = 0; i_row < n_rows; i_row++) {
        if (is_ok(i_row)) {
            size_t& group = entSet[i_row];
            if (group == 0) {
                next_ent++;
                group = next_ent;
                ListRow.emplace_back(i_row);
            }
        }
    }
    std::vector<int64_t> ListIdx;
    for (auto& eRow : ListRow)
        if (eRow != -1) ListIdx.push_back(eRow);
    // Now building the out_arrs array. We select only the first num_keys.
    table_info* ret_table = RetrieveTable(in_table, ListIdx, num_keys);
    //
    delete[] hashes;
#ifdef DEBUG_DD_SYMBOL
    std::cout << "drop_duplicates_keys_inner : OUTPUT:\n";
    DEBUG_PrintRefct(std::cout, ret_table->columns);
    DEBUG_PrintSetOfColumn(std::cout, ret_table->columns);
#endif
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
 * @return the table to be used.
 */
table_info* drop_duplicates_table_inner(table_info* in_table, int64_t num_keys,
                                        int64_t keep, int step, bool dropna) {
    tracing::Event ev("drop_duplicates_table_inner");
    ev.add_attribute("table_nrows_before", size_t(in_table->nrows()));
    if (ev.is_tracing()) {
        size_t global_table_nbytes = table_global_memory_size(in_table);
        ev.add_attribute("g_table_nbytes", global_table_nbytes);
    }
    size_t n_rows = (size_t)in_table->nrows();
    std::vector<array_info*> key_arrs(num_keys);
    for (size_t iKey = 0; iKey < size_t(num_keys); iKey++)
        key_arrs[iKey] = in_table->columns[iKey];
#ifdef DEBUG_DD
    std::cout << "drop_duplicates_table_inner num_keys=" << num_keys
              << " keep=" << keep << " step=" << step << " : INPUT:\n";
    DEBUG_PrintRefct(std::cout, in_table->columns);
    DEBUG_PrintSetOfColumn(std::cout, in_table->columns);
#endif

    uint32_t seed = SEED_HASH_CONTAINER;
    uint32_t* hashes = hash_keys(key_arrs, seed);
    HashDropDuplicates hash_fct{hashes};
    KeyEqualDropDuplicates equal_fct{&key_arrs, num_keys};
    UNORD_MAP_CONTAINER<size_t, size_t, HashDropDuplicates,
                        KeyEqualDropDuplicates>
        entSet({}, hash_fct, equal_fct);
    // The loop over the short table.
    // entries are stored one by one and all of them are put even if identical
    // in value. The one exception is if we are dropping NA values
    //
    // In the first case we keep only one entry.
    bool has_nulls = dropna && does_keys_have_nulls(key_arrs);
    auto is_ok = [&](size_t i_row) -> bool {
        if (!has_nulls) return true;
        return !does_row_has_nulls(key_arrs, i_row);
    };
    auto RetrieveListIdx1 = [&]() -> std::vector<int64_t> {
        std::vector<int64_t> ListRow;
        uint64_t next_ent = 0;
        for (size_t i_row = 0; i_row < n_rows; i_row++) {
            // don't add if entry is NA and dropna=true
            if (is_ok(i_row)) {
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
        }
        std::vector<int64_t> ListIdx;
        for (auto& eRow : ListRow)
            if (eRow != -1) ListIdx.push_back(eRow);
        return ListIdx;
    };
    // In this case we store the pairs of values, the first and the last.
    // This allows to reach conclusions in all possible cases.
    auto RetrieveListIdx2 = [&]() -> std::vector<int64_t> {
        std::vector<std::pair<int64_t, int64_t>> ListRowPair;
        size_t next_ent = 0;
        for (size_t i_row = 0; i_row < n_rows; i_row++) {
            // don't add if entry is NA and dropna=true
            if (is_ok(i_row)) {
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
        }
        std::vector<int64_t> ListIdx;
        for (auto& eRowPair : ListRowPair) {
            if (eRowPair.first != -1) ListIdx.push_back(eRowPair.first);
            if (eRowPair.second != -1) ListIdx.push_back(eRowPair.second);
        }
        return ListIdx;
    };
    std::vector<int64_t> ListIdx;
    if (step == 1 || keep == 0 || keep == 1)
        ListIdx = RetrieveListIdx1();
    else
        ListIdx = RetrieveListIdx2();
    // Now building the out_arrs array.
    table_info* ret_table = RetrieveTable(in_table, ListIdx, -1);
    //
    delete[] hashes;
    ev.add_attribute("table_nrows_after", size_t(ret_table->nrows()));
#ifdef DEBUG_DD
    std::cout << "drop_duplicates_table_inner : OUTPUT:\n";
    DEBUG_PrintRefct(std::cout, ret_table->columns);
    DEBUG_PrintSetOfColumn(std::cout, ret_table->columns);
#endif
    return ret_table;
}

/** This function is for dropping the null keys.
 * This C++ code is used for the compute_categorical_index
 * ---It keeps only the non-null keys
 * ---It returns only the keys.
 *
 * As for the join, this relies on using hash keys for the partitionning.
 * The computation is done locally.
 *
 * External function used are "RetrieveTable" and "TestEqual"
 *
 * @param in_table : the input table
 * @param num_keys : the number of keys
 * @param is_parallel: whether we run in parallel or not.
 * @param dropna: whether we drop null keys or not.
 * @return the vector of pointers to be used.
 */
table_info* drop_duplicates_keys(table_info* in_table, int64_t num_keys,
                                 bool is_parallel, bool dropna) {
#ifdef DEBUG_DD
    std::cout << "drop_duplicates_keys : is_parallel=" << is_parallel << "\n";
#endif
    // serial case
    if (!is_parallel) {
        return drop_duplicates_keys_inner(in_table, num_keys, dropna);
    }
    // parallel case
    // pre reduction of duplicates
    table_info* red_table =
        drop_duplicates_keys_inner(in_table, num_keys, dropna);
    // shuffling of values
    table_info* shuf_table = shuffle_table(red_table, num_keys, 0);
    // no need to decref since shuffle_table() steals a reference
    delete_table(red_table);
    // reduction after shuffling
    int keep = 0;
    // Set dropna=False for drop_duplicates_table_inner because
    // drop_duplicates_keys_inner should have already removed any NA
    // values
    table_info* ret_table =
        drop_duplicates_table_inner(shuf_table, num_keys, keep, 1, false);
    delete_table(shuf_table);
#ifdef DEBUG_DD
    std::cout << "OUTPUT : drop_duplicates_keys ret_table=\n";
    DEBUG_PrintRefct(std::cout, ret_table->columns);
    DEBUG_PrintSetOfColumn(std::cout, ret_table->columns);
#endif
    // returning table
    return ret_table;
}

/** This function is for dropping the null keys.
 * This C++ code is used for the compute_categorical_index
 * ---Even non-null keys occur in the output.
 * ---Everything is returned, not just the keys.
 *
 * @param in_table : the input table
 * @param is_parallel: the boolean specifying if the computation is parallel or
 * not.
 * @param num_keys: the number of keys used for the computation
 * @param keep: integer specifying the expected behavior.
 *        keep = 0 corresponds to the case of keep="first" keep first entry
 *        keep = 1 corresponds to the case of keep="last" keep last entry
 *        keep = 2 corresponds to the case of keep=False : remove all duplicates
 * @param total_cols: number of columns to use identifying duplicates
 * @param dropna: Should NA be included in the final table
 * @return the vector of pointers to be used.
 */
table_info* drop_duplicates_table(table_info* in_table, bool is_parallel,
                                  int64_t num_keys, int64_t keep,
                                  int64_t total_cols, bool dropna) {
    try {
#ifdef DEBUG_DD
        std::cout << "drop_duplicates_table : is_parallel=" << is_parallel
                  << "\n";
#endif
        // serial case
        if (!is_parallel) {
            if (total_cols != -1) num_keys = total_cols;
            return drop_duplicates_table_inner(in_table, num_keys, keep, 1,
                                               dropna);
        }
        // parallel case
        // pre reduction of duplicates
        table_info* red_table;
        if (total_cols == -1) total_cols = num_keys;
        red_table =
            drop_duplicates_table_inner(in_table, total_cols, keep, 2, dropna);
        // shuffling of values
        table_info* shuf_table = shuffle_table(red_table, num_keys);
        // no need to decref since shuffle_table() steals a reference
        delete_table(red_table);
        // reduction after shuffling
        // We don't drop NA values again because the first
        // drop_duplicates_table_inner should have already handled this
        table_info* ret_table =
            drop_duplicates_table_inner(shuf_table, total_cols, keep, 1, false);
        delete_table(shuf_table);
#ifdef DEBUG_DD
        std::cout << "OUTPUT drop_duplicates_table. ret_table=\n";
        DEBUG_PrintRefct(std::cout, ret_table->columns);
        DEBUG_PrintSetOfColumn(std::cout, ret_table->columns);
#endif
        // returning table
        return ret_table;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

//
//   SAMPLE
//

table_info* sample_table(table_info* in_table, int64_t n, double frac,
                         bool replace, bool parallel) {
    try {
#ifdef DEBUG_SAMPLE
        std::cout << "sample_table : in_table\n";
        DEBUG_PrintSetOfColumn(std::cout, in_table->columns);
        DEBUG_PrintRefct(std::cout, in_table->columns);
        std::cout << "sample_table : n=" << n << " frac=" << frac << "\n";
        std::cout << "sample_table : parallel=" << parallel << "\n";
#endif
        int n_local = in_table->nrows();
        std::vector<int> ListSizes;
        int n_pes = 0, myrank = 0, mpi_root = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        if (parallel) {
            // Number of rows to collect
            if (myrank == mpi_root) ListSizes.resize(n_pes);
            MPI_Gather(&n_local, 1, MPI_INT, ListSizes.data(), 1, MPI_INT,
                       mpi_root, MPI_COMM_WORLD);
        }
        // n_total is used only on node 0. If parallel then its value is correct
        // only on node 0
        int n_total;
        if (parallel)
            n_total = std::accumulate(ListSizes.begin(), ListSizes.end(), 0);
        else
            n_total = n_local;
        // The total number of sampled node. Its value is used only on node 0.
        int n_samp;
        if (frac < 0)
            n_samp = n;
        else
            n_samp = round(n_total * frac);
        std::vector<int> ListIdxChosen;
        std::vector<int> ListByProcessor;
        std::vector<int> ListCounts;
        // We compute random points. In the case of parallel=T we consider it
        // with respect to the whole array while in the case of parallel=F we
        // need to select the same rows on all nodes.
        if (myrank == 0) {
            ListIdxChosen.resize(n_samp);
            if (parallel) {
                ListByProcessor.resize(n_samp);
                ListCounts.resize(n_pes);
                for (int i_p = 0; i_p < n_pes; i_p++) ListCounts[i_p] = 0;
            }
            auto GetIProc_IPos =
                [&](int const& idx_rand) -> std::pair<int, int> {
                int new_siz, sum_siz = 0;
                int iProc = 0;
                while (true) {
                    new_siz = sum_siz + ListSizes[iProc];
                    if (idx_rand < new_siz) {
                        int pos = idx_rand - sum_siz;
                        return {iProc, pos};
                    }
                    sum_siz = new_siz;
                    iProc++;
                }
                return {-1, -1};  // This code path should not happen. Put here
                                  // to avoid warnings.
            };
            // In the case of replace, we simply have to take points at random.
            //
            // In the case of not replacing operation, this is more complicated.
            // The issue is considered in
            // ---Bentley J., Programming Pearls, 2nd, Column 12
            // ---Knuth D., TAOCP, Seminumerical algorithms, Section 3.4.2
            // The algorithms implemented here are based on the approximation of
            // m small.
            std::set<int> SetIdxChosen;
            UNORD_SET_CONTAINER<int> UnordsetIdxChosen;
            // Deterministic random number generation. See sort_values_table for
            // rationale.
            std::mt19937 gen(1234567890);
            auto get_rand = [&](int const& len) -> int {
                return std::uniform_int_distribution<>(0, len - 1)(gen);
            };
            auto GetIdx_Rand = [&]() -> int64_t {
                if (replace) return get_rand(n_total);
                // Two different algorithms according to the size
                // Complexity will be of about O(m log m)
                if (n_samp * 2 < n_total) {
                    // In the case of small sampling state we can iterate in
                    // order to conclude.
                    while (true) {
                        int idx_rand = get_rand(n_total);
                        if (UnordsetIdxChosen.count(idx_rand) == 0) {
                            UnordsetIdxChosen.insert(idx_rand);
                            return idx_rand;
                        }
                    }
                } else {
                    // If the number of sampling points is near to the size of
                    // the object then the random will be too slow. So,
                    // something more complicated is needed.
                    int64_t siz = SetIdxChosen.size();
                    int64_t idx_rand = get_rand(n_total - siz);
                    auto iter = SetIdxChosen.begin();
                    while (iter != SetIdxChosen.end()) {
                        if (idx_rand < *iter) break;
                        iter++;
                        idx_rand++;
                    }
                    SetIdxChosen.insert(idx_rand);
                    return idx_rand;
                }
            };
            for (int i_samp = 0; i_samp < n_samp; i_samp++) {
                int64_t idx_rand = GetIdx_Rand();
                if (parallel) {
                    std::pair<int, int> ePair = GetIProc_IPos(idx_rand);
                    int iProc = ePair.first;
                    int pos = ePair.second;
                    ListByProcessor[i_samp] = iProc;
                    ListIdxChosen[i_samp] = pos;
                    ListCounts[iProc]++;
                } else {
                    ListIdxChosen[i_samp] = idx_rand;
                }
            }
        }
        if (!parallel && myrank != 0) {
            ListIdxChosen.resize(n_samp);
        }
        int n_samp_out;
        if (parallel) {
            MPI_Scatter(ListCounts.data(), 1, MPI_INT, &n_samp_out, 1, MPI_INT,
                        mpi_root, MPI_COMM_WORLD);
        } else {
            n_samp_out = n_samp;
        }

        std::vector<int> ListIdxExport(n_samp_out), ListDisps,
            ListIdxChosenExport;
        if (myrank == 0 && parallel) {
            ListDisps.resize(n_pes);
            ListIdxChosenExport.resize(n_samp);
            ListDisps[0] = 0;
            for (int i_pes = 1; i_pes < n_pes; i_pes++)
                ListDisps[i_pes] = ListDisps[i_pes - 1] + ListCounts[i_pes - 1];
            std::vector<int> ListShift(n_pes, 0);
            for (int i_samp = 0; i_samp < n_samp; i_samp++) {
                int iProc = ListByProcessor[i_samp];
                ListIdxChosenExport[ListDisps[iProc] + ListShift[iProc]] =
                    ListIdxChosen[i_samp];
                ListShift[iProc]++;
            }
        } else {
            if (myrank == 0) {
                for (int i_samp = 0; i_samp < n_samp; i_samp++)
                    ListIdxExport[i_samp] = ListIdxChosen[i_samp];
            }
        }
        if (parallel) {
            // Exporting to all nodes, the data that they must extract
            MPI_Scatterv(ListIdxChosenExport.data(), ListCounts.data(),
                         ListDisps.data(), MPI_INT, ListIdxExport.data(),
                         n_samp_out, MPI_INT, mpi_root, MPI_COMM_WORLD);
        } else {
            MPI_Bcast(ListIdxExport.data(), n_samp_out, MPI_INT, mpi_root,
                      MPI_COMM_WORLD);
        }
        //
        std::vector<int64_t> ListIdx;
        for (int i_samp_out = 0; i_samp_out < n_samp_out; i_samp_out++) {
            size_t idx_export = ListIdxExport[i_samp_out];
            ListIdx.push_back(idx_export);
        }
        table_info* tab_out = RetrieveTable(in_table, ListIdx, -1);
#ifdef DEBUG_SAMPLE
        std::cout << "sample_table : tab_out\n";
        DEBUG_PrintRefct(std::cout, tab_out->columns);
        DEBUG_PrintSetOfColumn(std::cout, tab_out->columns);
#endif
        if (parallel) {
            bool all_gather = true;
            size_t n_cols = tab_out->ncols();
            table_info* tab_ret = gather_table(tab_out, n_cols, all_gather);
            delete_table(tab_out);
#ifdef DEBUG_SAMPLE
            std::cout << "sample_table : tab_ret\n";
            DEBUG_PrintRefct(std::cout, tab_ret->columns);
            DEBUG_PrintSetOfColumn(std::cout, tab_ret->columns);
#endif
            return tab_ret;
        }
        return tab_out;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}
