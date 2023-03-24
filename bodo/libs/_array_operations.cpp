// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include "_array_operations.h"
#include <boost/xpressive/xpressive.hpp>
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
                              array_info* in_values, bool is_parallel) {
    CheckEqualityArrayType(in_arr, in_values);
    if (out_arr->dtype != Bodo_CTypes::_BOOL) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "array out_arr should be a boolean array");
        return;
    }
    uint32_t seed = SEED_HASH_CONTAINER;

    int64_t len_values = in_values->length;
    std::unique_ptr<uint32_t[]> hashes_values =
        std::make_unique<uint32_t[]>(len_values);
    hash_array(hashes_values, in_values, (size_t)len_values, seed, is_parallel,
               /*global_dict_needed=*/false);

    int64_t len_in_arr = in_arr->length;
    std::unique_ptr<uint32_t[]> hashes_in_arr =
        std::make_unique<uint32_t[]>(len_in_arr);
    hash_array(hashes_in_arr, in_arr, (size_t)len_in_arr, seed, is_parallel,
               /*global_dict_needed=*/false);

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
        if (pos < len_values) {
            value = hashes_values[pos];
        } else {
            value = hashes_in_arr[pos - len_values];
        }
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
            array_isin_kernel(out_arr, in_arr, in_values, is_parallel);
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
            shuffle_table(&table_in_values, num_keys, is_parallel);
        // we need the comm_info and hashes for the reverse shuffling
        mpi_comm_info comm_info(table_in_arr->columns);
        std::shared_ptr<uint32_t[]> hashes =
            hash_keys_table(table_in_arr, 1, SEED_HASH_PARTITION, is_parallel);
        comm_info.set_counts(hashes, is_parallel);
        table_info* shuf_table_in_arr =
            shuffle_table_kernel(table_in_arr, hashes, comm_info, is_parallel);
        // Creation of the output array.
        int64_t len = shuf_table_in_arr->columns[0]->length;
        array_info* shuf_out_arr =
            alloc_array(len, -1, -1, out_arr->arr_type, out_arr->dtype, 0,
                        out_arr->num_categories);
        // Calling isin on the shuffled info
        array_isin_kernel(shuf_out_arr, shuf_table_in_arr->columns[0],
                          shuf_table_in_values->columns[0], is_parallel);

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
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return;
    }
}

//
// UTILS FOR SORT
//

/**
 * @brief Get n_loc_sample many samples from a locally sorted table.
 * This function splits the input table into roughly n_loc_sample blocks, and
 * then picks a row randomly from each block.
 *
 * @param local_sort Locally sorted table
 * @param n_key_t Number of sort keys
 * @param n_loc_sample Number of samples to get
 * @param n_local Length of the table
 * @param parallel Only for tracing purposes
 * @return table_info* Table with samples. Shape: n_loc_sample rows, n_key_t
 * columns
 */
table_info* get_samples_from_table_local(table_info* local_sort,
                                         int64_t n_key_t, int64_t n_loc_sample,
                                         int64_t n_local, bool parallel) {
    tracing::Event ev("get_samples_from_table_local", parallel);
    std::mt19937 gen(1234567890);
    double block_size = double(n_local) / n_loc_sample;
    std::vector<int64_t> ListIdx(n_loc_sample);
    double cur_lo = 0;
    for (int64_t i = 0; i < n_loc_sample; i++) {
        int64_t lo = round(cur_lo);
        int64_t hi = round(cur_lo + block_size) - 1;
        ListIdx[i] = std::uniform_int_distribution<int64_t>(lo, hi)(gen);
        cur_lo += block_size;
    }
    // Incref since RetrieveTable will steal a reference.
    for (int64_t i_key = 0; i_key < n_key_t; i_key++) {
        incref_array(local_sort->columns[i_key]);
    }
    table_info* samples = RetrieveTable(local_sort, ListIdx, n_key_t);
    return samples;
}

/**
 * @brief Get samples from locally sorted chunks of a distributed table.
 * Number of samples required for a reasonable sampling is determined
 * based on the total length of the table, number of ranks.
 *
 * @param local_sort Locally sorted table chunk.
 * @param n_key_t Number of sort keys.
 * @param n_pes Number of MPI ranks.
 * @param n_total Global table length.
 * @param n_local Local table length (=> length of local_sort)
 * @param parallel Whether the execution is happening in parallel. Passed for
 * consistency and tracing purposes.
 * @return table_info* Table of samples on rank 0, empty table on all other
 * ranks.
 */
table_info* get_samples_from_table_parallel(table_info* local_sort,
                                            int64_t n_key_t, int n_pes,
                                            int64_t n_total, int64_t n_local,
                                            bool parallel) {
    tracing::Event ev("get_samples_from_table_parallel", parallel);
    ev.add_attribute("n_key_t", n_key_t);
    ev.add_attribute("n_local", n_local);
    ev.add_attribute("n_total", n_total);
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
    double epsilon = 0.1;
    char* bodo_sort_epsilon = std::getenv("BODO_SORT_EPSILON");
    if (bodo_sort_epsilon) {
        epsilon = std::stod(bodo_sort_epsilon);
    }
    if (epsilon < 0) {
        throw std::runtime_error("sort_values: epsilon < 0");
    }

    double n_global_sample = n_pes * log(n_total) / pow(epsilon, 2.0);
    n_global_sample =
        std::min(std::max(n_global_sample, double(n_pes)), double(n_total));
    int64_t n_loc_sample = std::min(n_global_sample / n_pes, double(n_local));

    // Get n_loc_sample many local samples from the local sorted chunk
    table_info* samples = get_samples_from_table_local(
        local_sort, n_key_t, n_loc_sample, n_local, parallel);

    // Collecting all samples
    bool all_gather = false;
    table_info* all_samples =
        gather_table(samples, n_key_t, all_gather, parallel);
    delete_table(samples);
    return all_samples;
}

//
// PARALLEL SORT
//

/**
 *   SORT VALUES
 *
 * @param is_parallel: true if data is distributed (used to indicate whether
 * tracing should be parallel or not)
 */
table_info* sort_values_table_local(table_info* in_table, int64_t n_key_t,
                                    int64_t* vect_ascending,
                                    int64_t* na_position, int64_t* dead_keys,
                                    bool is_parallel) {
    tracing::Event ev("sort_values_table_local", is_parallel);
    size_t n_rows = (size_t)in_table->nrows();
    size_t n_key = size_t(n_key_t);
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
        bool na_last = na_position[0];
        if (ascending) {
            const bool na_position_bis = (!na_last) ^ ascending;
            const auto f = [&](size_t const& iRow1,
                               size_t const& iRow2) -> bool {
                int test = KeyComparisonAsPython_Column(
                    na_position_bis, key_col, iRow1, key_col, iRow2);
                if (test) return test > 0;
                return false;
            };
            gfx::timsort(ListIdx.begin(), ListIdx.end(), f);
        } else {
            const bool na_position_bis = (!na_last) ^ ascending;
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

    table_info* ret_table;
    if (dead_keys == nullptr) {
        ret_table = RetrieveTable(in_table, ListIdx, -1);
    } else {
        uint64_t n_cols = in_table->ncols();
        std::vector<size_t> colInds;
        for (uint64_t i = 0; i < n_cols; i++) {
            if (i < n_key && dead_keys[i]) {
                array_info* in_arr = in_table->columns[i];
                // decref dead keys since not used anymore
                // RetrieveTable decrefs other arrays
                decref_array(in_arr);
                continue;
            }
            colInds.push_back(i);
        }
        ret_table = RetrieveTable(in_table, ListIdx, colInds);
    }

    return ret_table;
}

/**
 * @brief Compute bounds for the ranks based on the collected samples.
 * All samples are assumed to be on rank 0. Tables on the rest of the
 * ranks are assumed to be empty.
 * The samples are first sorted, and then the bounds are computed
 * by picking the elements at the appropriate location for the rank.
 *
 * @param all_samples Table with all samples (gathered on rank 0). It is assumed
 * to be unsorted. NOTE: All arrays get decref-ed.
 * @param ref_table Reference table to use for the broadcast step. This is
 * mainly needed for dict encoded string arrays. In those cases, it is important
 * for the dictionary in this reference table to be same as the dictionary of
 * the actual array.
 * @param n_key_t Number of key columns.
 * @param vect_ascending Vector of booleans (one for each key column) describing
 * whether to sort in ascending order on the key columns.
 * @param na_position Vector of booleans (one for each key column) describing
 * where to put the NAs (last or first) in the key columns.
 * @param myrank MPI rank of the calling process.
 * @param n_pes Total number of MPI ranks.
 * @param parallel Is the process parallel.
 * @return table_info* Bounds table with n_pes-1 rows. A full bounds table is
 * computed on rank 0, broadcasted to all ranks and returned.
 */
table_info* compute_bounds_from_samples(table_info* all_samples,
                                        table_info* ref_table, int64_t n_key_t,
                                        int64_t* vect_ascending,
                                        int64_t* na_position, int myrank,
                                        int n_pes, bool parallel) {
    tracing::Event ev("compute_bounds_from_samples", parallel);
    int mpi_root = 0;
    // Computing the bounds (splitters) on root
    table_info* pre_bounds = nullptr;
    if (myrank == mpi_root) {
        table_info* all_samples_sort =
            sort_values_table_local(all_samples, n_key_t, vect_ascending,
                                    na_position, nullptr, parallel);
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
        // TODO the right way would be to call sort_values_table_local with
        // an empty table
        tracing::Event ev_dummy("sort_values_table_local", parallel);
    }

    // broadcasting the bounds
    // The local_sort is used as reference for the data type of the array.
    // This is because pre_bounds is NULL for ranks != 0
    // The underlying dictionary is the same for local_sort and pre_bounds
    // for the dict columns, as needed for broadcast_table.
    table_info* bounds =
        broadcast_table(ref_table, pre_bounds, n_key_t, parallel, mpi_root);

    if (myrank == mpi_root) {
        delete_table(pre_bounds);
    }

    return bounds;
}

/**
 * @brief Compute data redistribution boundaries for parallel sort using a
 * sample of the data.
 *
 * @param local_sort locally sorted table
 * @param n_key_t number of sort keys
 * @param vect_ascending ascending/descending order for each key
 * @param na_position NA behavior (first or last) for each key
 * @param n_local number of local rows
 * @param n_total number of global rows
 * @param myrank MPI rank
 * @param n_pes total MPI ranks
 * @param parallel parallel flag (should be true here but passing around for
 * code consistency)
 * @return table_info* Bounds table with n_pes-1 rows.
 */
table_info* get_parallel_sort_bounds(table_info* local_sort, int64_t n_key_t,
                                     int64_t* vect_ascending,
                                     int64_t* na_position, int64_t n_local,
                                     int64_t n_total, int myrank, int n_pes,
                                     bool parallel) {
    tracing::Event ev("get_parallel_sort_bounds", parallel);
    // Compute samples from the locally sorted table.
    // (Filled on rank 0, empty on all other ranks)
    table_info* all_samples = get_samples_from_table_parallel(
        local_sort, n_key_t, n_pes, n_total, n_local, parallel);

    // Compute split bounds from the samples.
    // Output is broadcasted to all ranks.
    table_info* bounds = compute_bounds_from_samples(
        all_samples, local_sort, n_key_t, vect_ascending, na_position, myrank,
        n_pes, parallel);

    delete_table(all_samples);

    return bounds;
}

table_info* sort_values_table(table_info* in_table, int64_t n_key_t,
                              int64_t* vect_ascending, int64_t* na_position,
                              int64_t* dead_keys, int64_t* out_n_rows,
                              table_info* bounds, bool parallel) {
    try {
        tracing::Event ev("sort_values_table", parallel);

        if (out_n_rows != nullptr)
            // Initialize to the input because local sort won't
            // change the number of elements. If we do a
            // distributed source the rows per rank may change.
            *out_n_rows = (int64_t)in_table->nrows();

        // Convert all local dictionaries to global for dict columns.
        // Also sort the dictionaries, so the sorting process
        // is more efficient (we can compare indices directly)
        for (array_info* arr : in_table->columns) {
            if (arr->arr_type == bodo_array_type::DICT) {
                // For dictionary encoded arrays we need the data to be unique
                // and global in case the dictionary is sorted (because we will
                // compare indices directly)
                make_dictionary_global_and_unique(arr, parallel, true);
            }
        }

        int64_t n_local = in_table->nrows();
        int64_t n_total = n_local;
        if (parallel)
            MPI_Allreduce(&n_local, &n_total, 1, MPI_LONG_LONG_INT, MPI_SUM,
                          MPI_COMM_WORLD);

        // Want to keep dead keys only when we will perform a shuffle operation
        // later in the function
        table_info* local_sort = sort_values_table_local(
            in_table, n_key_t, vect_ascending, na_position,
            (parallel && n_total != 0) ? nullptr : dead_keys, parallel);

        if (!parallel) {
            return local_sort;
        } else if (n_total == 0) {
            if (bounds != nullptr) {
                delete_table_decref_arrays(bounds);
            }
            return local_sort;
        }

        tracing::Event ev_sample("sort sampling", parallel);

        int n_pes, myrank;
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        if (bounds == nullptr) {
            bounds = get_parallel_sort_bounds(
                local_sort, n_key_t, vect_ascending, na_position, n_local,
                n_total, myrank, n_pes, parallel);
        } else if (n_key_t != 1) {
            // throw error if more than one key since the rest of manual bounds
            // handling only supports one key cases
            throw std::runtime_error(
                "sort_values_table(): passing bounds only supported when there "
                "is a single key.");
        }

        // Now computing to which process it all goes.
        tracing::Event ev_hashes("compute_destinations", parallel);
        std::shared_ptr<uint32_t[]> hashes =
            std::make_unique<uint32_t[]>(n_local);
        uint32_t rank_id = 0;
        for (int64_t i = 0; i < n_local; i++) {
            size_t shift_key1 = 0, shift_key2 = 0;
            // using 'while' since a partition can be empty which needs to be
            // skipped
            // Go to next destination rank if bound of current destination rank
            // is less than the current key. All destination keys should be less
            // than or equal its bound (k <= bounds[rank_id])
            while (rank_id < uint32_t(n_pes - 1) &&
                   KeyComparisonAsPython(n_key_t, vect_ascending,
                                         bounds->columns, shift_key2, rank_id,
                                         local_sort->columns, shift_key1, i,
                                         na_position)) {
                rank_id++;
            }
            hashes[i] = rank_id;
        }
        delete_table_decref_arrays(bounds);

        // Now shuffle all the data
        mpi_comm_info comm_info(local_sort->columns);
        comm_info.set_counts(hashes, parallel);
        ev_hashes.finalize();
        ev_sample.finalize();
        table_info* collected_table =
            shuffle_table_kernel(local_sort, hashes, comm_info, parallel);
        // NOTE: shuffle_table_kernel decrefs input arrays
        delete_table(local_sort);

        // NOTE: local sort doesn't change the number of rows
        // ret_table cannot be used since all output columns may be dead (only
        // length is needed)
        if (out_n_rows != nullptr)
            *out_n_rows = (int64_t)collected_table->nrows();

        // Final local sorting
        table_info* ret_table =
            sort_values_table_local(collected_table, n_key_t, vect_ascending,
                                    na_position, dead_keys, parallel);
        delete_table(collected_table);
        return ret_table;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

array_info* sort_values_array_local(array_info* in_arr, bool is_parallel,
                                    int64_t ascending, int64_t na_position) {
    std::vector<array_info*> cols = {in_arr};
    table_info* dummy_table = new table_info(cols);
    int64_t zero = 0;
    // sort_values_table_local will decref all arrays in dummy_table (i.e.
    // in_arr) by 1.
    table_info* sorted_table = sort_values_table_local(
        dummy_table, 1, &ascending, &na_position, &zero, is_parallel);
    // We don't want to call `delete` on `in_arr` (since we don't own it), which
    // is what `delete_table` would do.
    delete dummy_table;
    array_info* sorted_arr = sorted_table->columns[0];
    delete sorted_table;
    return sorted_arr;
}

//
// UTILS FOR SORT STEP IN INTERVAL JOIN
//

/**
 * @brief Validate inputs to get_parallel_sort_bounds_for_domain.
 * In particular we check that the key columns all have the same array and data
 * type, and that it's one of the supported ones.
 *
 * @param tables Tables to validate.
 * @param n_keys Number of keys to validate in each table.
 * @param n_local Number of local rows for each of the tables.
 * @param n_total Number of global rows for each of the tables.
 */
void validate_inputs_for_get_parallel_sort_bounds_for_domain(
    std::vector<table_info*> tables, std::vector<uint64_t> n_keys,
    std::vector<uint64_t> n_local, std::vector<uint64_t> n_total) {
    // Basic validation
    if (n_keys.size() == 0) {
        throw std::runtime_error(
            "No tables provided to get_parallel_sort_bounds_for_domain.");
    }
    if (!((tables.size() == n_keys.size()) &&
          (n_keys.size() == n_local.size()) &&
          (n_local.size() == n_total.size()))) {
        throw std::runtime_error(
            "Inconsistent size inputs provided to "
            "get_parallel_sort_bounds_for_domain.");
    }
    if (n_keys[0] <= 0) {
        throw std::runtime_error(
            "No keys specified for the first table in "
            "get_parallel_sort_bounds_for_domain.");
    }

    // Check that the arr and dtype are supported
    bodo_array_type::arr_type_enum common_arr_type =
        tables[0]->columns[0]->arr_type;
    Bodo_CTypes::CTypeEnum common_dtype = tables[0]->columns[0]->dtype;

    if ((common_arr_type != bodo_array_type::arr_type_enum::NUMPY) &&
        (common_arr_type !=
         bodo_array_type::arr_type_enum::NULLABLE_INT_BOOL)) {
        throw std::runtime_error(
            "get_parallel_sort_bounds_for_domain not supported for "
            "array "
            "type " +
            GetArrType_as_string(common_arr_type));
    }

    if (!is_numerical(common_dtype) &&
        (common_dtype != Bodo_CTypes::CTypeEnum::DATE) &&
        (common_dtype != Bodo_CTypes::CTypeEnum::DATETIME) &&
        (common_dtype != Bodo_CTypes::CTypeEnum::TIME) &&
        (common_dtype != Bodo_CTypes::CTypeEnum::TIMEDELTA)) {
        throw std::runtime_error(
            "get_parallel_sort_bounds_for_domain not supported for "
            "dtype " +
            GetDtype_as_string(common_dtype));
    }

    // Validate that all involved columns have the same arr type and dtype
    for (size_t table_idx = 0; table_idx < tables.size(); table_idx++) {
        for (uint64_t key_idx = 0; key_idx < n_keys[table_idx]; key_idx++) {
            if (tables[table_idx]->columns[key_idx]->arr_type !=
                common_arr_type) {
                throw std::runtime_error(
                    "Array type of array at index " + std::to_string(key_idx) +
                    " of table " + std::to_string(table_idx) + ": " +
                    GetArrType_as_string(
                        tables[table_idx]->columns[key_idx]->arr_type) +
                    " doesn't match expected type: " +
                    GetArrType_as_string(common_arr_type));
            };
            if (tables[table_idx]->columns[key_idx]->dtype != common_dtype) {
                throw std::runtime_error(
                    "Data type of array at index " + std::to_string(key_idx) +
                    " of table " + std::to_string(table_idx) + " :" +
                    GetDtype_as_string(
                        tables[table_idx]->columns[key_idx]->dtype) +
                    " doesn't match expected type: " +
                    GetDtype_as_string(common_dtype));
            }
        }
    }

    // For decimals, check that precision and scale
    // also match.
    if (common_dtype == Bodo_CTypes::DECIMAL) {
        int32_t common_precision = tables[0]->columns[0]->precision;
        int32_t common_scale = tables[0]->columns[0]->scale;
        for (size_t table_idx = 0; table_idx < tables.size(); table_idx++) {
            for (uint64_t key_idx = 0; key_idx < n_keys[table_idx]; key_idx++) {
                // Check against the first array
                if (tables[table_idx]->columns[key_idx]->precision !=
                    common_precision) {
                    throw std::runtime_error(
                        "Decimal precision of array at index " +
                        std::to_string(key_idx) + " of table " +
                        std::to_string(table_idx) + " :" +
                        std::to_string(
                            tables[table_idx]->columns[key_idx]->precision) +
                        " doesn't match expected precision: " +
                        std::to_string(common_precision));
                }
                // Check against the first array
                if (tables[table_idx]->columns[key_idx]->scale !=
                    common_scale) {
                    throw std::runtime_error(
                        "Decimal scale of array at index " +
                        std::to_string(key_idx) + " of table " +
                        std::to_string(table_idx) + " :" +
                        std::to_string(
                            tables[table_idx]->columns[key_idx]->scale) +
                        " doesn't match expected scale: " +
                        std::to_string(common_scale));
                }
            }
        }
    }
}

//
// SORT FOR INTERVAL JOIN
//

/**
 * @brief Construct a new table from the rank_to_row_ids vector.
 * In this new table, the rows are ordered by ranks, and may be
 * repeated in case they are going to multiple ranks.
 * e.g. If rank_to_row_ids =
 * [[0, 1], [0, 1, 2, 3], [1, 2]].
 * Then we will construct the table:
 * row-0, row-1, row-0, row-1, row-2, row-3, row-1, row-2.
 * This table can now be shuffled since the row boundaries
 * are clear.
 *
 * @param table Table to construct the new table from. (NOTE: All arrays will
 * get decref-ed)
 * @param rank_to_row_ids Vector of length `n_pes`. Each element is a vector
 * containing the rows that must go to that rank.
 * @param[out] hashes Reference to am array to fill the hashes in. In this
 * case, we directly fill the ranks (and can treat them as hashes).
 * @param parallel Only for tracing purposes
 * @return table_info* The new table and the
 * destination rank ids for rows in the new table.
 */
table_info* create_send_table_and_hashes_from_rank_to_row_ids(
    table_info* table, const std::vector<std::vector<int64_t>> rank_to_row_ids,
    std::shared_ptr<uint32_t[]>& hashes, bool parallel) {
    tracing::Event ev("create_send_table_and_hashes_from_rank_to_row_ids",
                      parallel);
    std::vector<uint64_t> rank_to_num_rows(rank_to_row_ids.size());

    // Calculate number of rows for each rank and total number of rows
    uint64_t total_rows = 0;
    size_t i = 0;
    for (auto& vec : rank_to_row_ids) {
        total_rows += vec.size();
        rank_to_num_rows[i] = vec.size();
        i++;
    }

    // Get indices for new table by flattening rank_to_row_ids
    // TODO Change to use uint64_t when possible. Initial attempt
    // had issues at linking time (symbol not found, etc.).
    std::vector<int64_t> indices =
        flatten<int64_t>(rank_to_row_ids, total_rows);
    // Create new table using the indices.
    // NOTE: RetrieveTable decrefs all arrays.
    table_info* new_table = RetrieveTable(table, indices, -1);

    // Fill hashes array (i.e. the rank that rows in the new_table should go to)
    hashes.reset(new uint32_t[total_rows]);
    uint64_t idx = 0;
    for (size_t i = 0; i < rank_to_row_ids.size(); i++) {
        // TODO: replace with std::fill
        for (size_t j = idx; j < idx + rank_to_num_rows[i]; j++) {
            hashes[j] = i;
        }
        idx += rank_to_num_rows[i];
    }

    return new_table;
}

/**
 * @brief Compute the destination rank(s) based on the provided bounds, for each
 * of the rows of a locally sorted table. We assume that the first column is the
 * start of the interval and the second column is the end of the interval.
 * Note that all "bad" intervals (i.e. start > end) are skipped.  In point in
 * interval joins, bad intervals cannot match with a point, and hence are safe
 * to skip. Also, including them would break the assumptions this algorithm
 * needs to assign the ranks correctly to the sorted rows.
 *
 * @param table Locally sorted table for whose rows we need to calculate the
 * destination rank(s).
 * @param bounds_arr Boundaries for the ranks. Should be n_pes - 1 long.
 * @param myrank MPI rank of the calling process.
 * @param n_pes Total number of MPI ranks.
 * @param parallel Only for tracing purposes.
 * @param strict Only filter strict bad intervals (where A > B instead of A >=
 * B)
 * @return const std::vector<std::vector<uint64_t>> Vector of length n_pes. i'th
 * element is a vector of row ids that must be sent to the i'th rank. Due to the
 * nature of the algorithm, each vector will be sorted.
 */
const std::vector<std::vector<int64_t>> compute_destinations_for_interval(
    table_info* table, array_info* bounds_arr, int n_pes, bool parallel,
    bool strict) {
    tracing::Event ev("compute_destinations_for_interval", parallel);
    // TODO XXX Convert to use uint64_t
    std::vector<std::vector<int64_t>> rank_to_row_ids(n_pes);
    uint32_t rank_id = 0;
    uint32_t rank_id_i = 0;
    for (uint64_t i = 0; i < table->nrows(); i++) {
        // Start at rank_id
        rank_id_i = rank_id;

        // Check if it's a bad interval, and if it is, then skip it and move to
        // the next row.
        if (is_bad_interval(table->columns[0], table->columns[1], i, strict)) {
            continue;
        }

        // Increment rank_id_i until we find the first rank that the
        // interval is within the bounds of.
        while ((rank_id_i < uint32_t(n_pes - 1)) &&
               !within_bounds_of_rank(bounds_arr, rank_id_i, n_pes,
                                      table->columns[0], table->columns[1],
                                      i)) {
            rank_id_i++;
        }

        // Update rank_id. rank_id_i rank is the first match for this row.
        // The next row can use this as its starting point since its first
        // rank cannot be lower than this.
        rank_id = rank_id_i;

        // Find all the ranks that this interval is within bounds of.
        // These ranks must be consecutive, so we can stop once we find
        // the first one where there isn't a match. We can be sure that
        // rank_id_i is a match, hence the do...while loop.
        do {
            // Add this row to the list of rows for rank rank_id_i.
            rank_to_row_ids[rank_id_i].push_back(i);
            rank_id_i++;
        } while ((rank_id_i < uint32_t(n_pes)) &&
                 within_bounds_of_rank(bounds_arr, rank_id_i, n_pes,
                                       table->columns[0], table->columns[1],
                                       i));
    }
    // C++ 11 onwards returns locally allocated vectors efficiently, i.e.
    // doesn't copy or re-allocate memory. Verified by checking the memory
    // address of the return value at the caller.
    return rank_to_row_ids;
}

/**
 * @brief Compute the destination rank based on the provided bounds, for each
 * of the rows of a locally sorted table. We assume that the first column
 * is the key column.
 *
 * @param table Locally sorted table for whose rows we need to calculate the
 * destination rank.
 * @param bounds_arr Boundaries for the ranks. Should be n_pes - 1 long.
 * @param n_pes Total number of MPI ranks.
 * @param parallel Only for tracing purposes
 * @return std::shared_ptr<uint32_t []> PTR with destination ranks for each of
 * the rows in the input table.
 */
std::shared_ptr<uint32_t[]> compute_destinations_for_point_table(
    table_info* table, array_info* bounds_arr, int n_pes, bool parallel) {
    tracing::Event ev("compute_destinations_for_point_table", parallel);
    uint64_t n_local = table->nrows();
    // We call them hashes for consistency, but in this case, they're actually
    // the ranks themselves.
    std::shared_ptr<uint32_t[]> hashes = std::make_unique<uint32_t[]>(n_local);
    uint32_t rank_id = 0;
    for (uint64_t i = 0; i < n_local; i++) {
        // Keep incrementing rank_id while the bound for rank_id
        // is less than the point value.
        // na_position_bis is true in our case since asc = true and na_last =
        // true which means that na_bis = (!na_last) ^ asc = true
        while (rank_id < uint32_t(n_pes - 1) &&
               (KeyComparisonAsPython_Column(true, bounds_arr, rank_id,
                                             table->columns[0], i) > 0)) {
            rank_id++;
        }

        // At this point, (point <= bounds[rank_id])
        hashes[i] = rank_id;
    }
    // C++ onwards vectors get returned without copying the buffers. Verified
    // manually by checking the memory addresses.
    return hashes;
}

/**
 * @brief Convert all local dictionaries to global for dict columns.
 * Sorting the dictionaries is not required since the string/dict columns
 * can never be keys.
 * Similarly, we only need the data to be global. De-duplication
 * is not necessary since the string/dict columns cannot
 * be the keys and hence will never be compared directly.
 *
 * @param tables Tables to apply the above transformations to.
 * @param parallel Whether the process is being done in parallel.
 */
inline void handle_dict_encoded_arrays_for_interval_join_sort(
    const std::vector<table_info*> tables, const std::vector<bool> parallel) {
    for (size_t i = 0; i < tables.size(); i++) {
        table_info* table = tables[i];
        for (array_info* arr : table->columns) {
            if (arr->arr_type == bodo_array_type::DICT) {
                convert_local_dictionary_to_global(arr, parallel[i]);
            }
        }
    }
}

/**
 * @brief Compute bounds for data distribution by treating all key columns in
 * all tables as a single domain. This means that we get samples from each
 * column, concatenate them all together, and then compute the split points.
 * This gives us split bounds that are useful for cases like interval joins.
 * The number of samples taken from each column are based on its length, so
 * longer columns are weighted higher than shorter ones.
 * We assume that the input tables are sorted on the key columns.
 *
 * @param tables Tables with domain columns.
 * @param n_keys Number of domain columns in each of the tables.
 * @param n_local Number of local rows in each of the tables.
 * @param n_total Total number of global rows in each of the tables.
 * @param myrank MPI rank of this process.
 * @param n_pes Total number of MPI processes.
 * @param parallel Whether the process is being done in parallel. This should
 * always be true. The variable is passed for consistency.
 * @return array_info* Array of bounds computed based on the domain columns. It
 * will be of length n_pes-1.
 */
array_info* get_parallel_sort_bounds_for_domain(
    std::vector<table_info*>& tables, std::vector<uint64_t>& n_keys,
    std::vector<uint64_t>& n_local, std::vector<uint64_t>& n_total, int myrank,
    int n_pes, bool parallel) {
    tracing::Event ev("get_parallel_sort_bounds_for_domain", parallel);
    // Validate input values
    validate_inputs_for_get_parallel_sort_bounds_for_domain(tables, n_keys,
                                                            n_local, n_total);
    int64_t ascending = 1, na_pos = 1;
    int mpi_root = 0;

    uint64_t total_cols = std::accumulate(n_keys.begin(), n_keys.end(), 0);
    std::vector<array_info*> sample_cols;
    sample_cols.reserve(total_cols);
    for (size_t table_idx = 0; table_idx < tables.size(); table_idx++) {
        for (uint64_t key_idx = 0; key_idx < n_keys[table_idx]; key_idx++) {
            array_info* col = tables[table_idx]->columns[key_idx];
            array_info* sorted_col = col;
            bool free_sorted_col = false;
            if (key_idx > 0) {
                // If it's not the first key, we need to sort it.
                // In this case, we'll need to free this array at the
                // end.
                // sort_values_array_local decrefs the array, so we need
                // to incref first.
                // If it's the first key, then we can use it as is (already
                // sorted)
                incref_array(col);
                sorted_col =
                    sort_values_array_local(col, parallel, ascending, na_pos);
                free_sorted_col = true;
            }
            // Create a dummy single column table
            table_info* dummy_table = new table_info();
            dummy_table->columns.push_back(sorted_col);

            // Output of get_samples_from_table_parallel is output of
            // gather_table, which means that on non-root ranks, the table
            // columns are just NULLs.
            table_info* all_samples = get_samples_from_table_parallel(
                dummy_table, /*n_key_t*/ 1, n_pes, n_total[table_idx],
                n_local[table_idx], parallel);
            array_info* all_samples_arr = all_samples->columns[0];
            sample_cols.push_back(all_samples_arr);

            if (free_sorted_col) {
                decref_array(sorted_col);
            }
            // Not using delete_table since we need the pointers in sample_cols
            // to stick around.
            delete all_samples;
            all_samples = NULL;
            delete dummy_table;
            dummy_table = NULL;
        }
    }

    array_info* concatenated_samples = NULL;
    if (myrank == mpi_root) {
        // Concatenate locally_sorted_cols into a single array_info, sort it and
        // then compute the split bounds on this.
        concatenated_samples = concat_arrays(sample_cols);
    }

    table_info* concatenated_samples_table = new table_info();
    concatenated_samples_table->columns.push_back(concatenated_samples);

    // Create a reference table for broadcast step in
    // compute_bounds_from_samples using the first column of the first
    // table.
    table_info* dummy_table = new table_info();
    dummy_table->columns.push_back(tables[0]->columns[0]);

    table_info* bounds = compute_bounds_from_samples(
        concatenated_samples_table, dummy_table, 1, &ascending, &na_pos, myrank,
        n_pes, parallel);

    delete dummy_table;
    dummy_table = NULL;
    // concatenated_samples will get decref-ed as part of
    // compute_bounds_from_samples, so we just need to delete the table
    delete_table(concatenated_samples_table);

    // We don't need to free arrays in sample_cols since
    // concat_arrays will already do that.

    array_info* bounds_arr = bounds->columns[0];
    // Not using delete_table since we want the bounds_arr
    // pointer to stick around.
    delete bounds;
    bounds = NULL;

    return bounds_arr;
}

/**
 * @brief Sort a table for interval join using the provided bounds.
 *
 * @param table Table to sort. NOTE: All arrays will be decref-ed, unless not
 * parallel and table_already_sorted.
 * @param table_already_sorted Whether the table is already locally sorted. This
 * should be true in case of interval join, but is needed for the Python API
 * which is used for testing this sort functionality more directly.
 * @param bounds_arr Bounds to use for data distribution. Must be of length
 * n_pes - 1.
 * @param is_table_point_side Is the table the point side in a point in interval
 * join. If so, the table is assumed to have one key, else two.
 * @param myrank MPI rank of this process.
 * @param n_pes Total number of MPI processes.
 * @param parallel Whether the table is distributed.
 * @param strict Only filter strict bad intervals (where A > B instead of A >=
 * B)
 * @return table_info* sorted table.
 */
table_info* sort_table_for_interval_join(table_info* table,
                                         bool table_already_sorted,
                                         array_info* bounds_arr,
                                         bool is_table_point_side, int myrank,
                                         int n_pes, bool parallel,
                                         bool strict = false) {
    tracing::Event ev("sort_table_for_interval_join", parallel);
    ev.add_attribute("is_table_point_side", is_table_point_side);
    ev.add_attribute("table_len_local", table->nrows());

    int64_t asc[2] = {1, 1};
    int64_t na_pos[2] = {1, 1};

    table_info* local_sort_table = table;
    if (!table_already_sorted) {
        local_sort_table =
            sort_values_table_local(table, (is_table_point_side ? 1 : 2), asc,
                                    na_pos, nullptr, parallel);
    }

    if (!parallel) {
        return local_sort_table;
    }

    //
    // 1. Compute destination ranks for each row in both tables
    //

    table_info* table_to_send = NULL;
    // In this case, we compute the ranks directly, and treat them as the hashes
    // (the modulo step later doesn't end up changing these values since they're
    // all already between 0 and n_pes-1).
    std::shared_ptr<uint32_t[]> table_to_send_hashes =
        std::shared_ptr<uint32_t[]>(nullptr);

    if (is_table_point_side) {
        table_to_send = local_sort_table;
        table_to_send_hashes = compute_destinations_for_point_table(
            local_sort_table, bounds_arr, n_pes, parallel);
    } else {
        // NOTE: This will skip bad intervals.
        std::vector<std::vector<int64_t>> rank_to_row_ids_for_table =
            compute_destinations_for_interval(local_sort_table, bounds_arr,
                                              n_pes, parallel, strict);

        // create_send_table_and_hashes_from_rank_to_row_ids will decref
        // all arrays in local_sort_table.
        table_to_send = create_send_table_and_hashes_from_rank_to_row_ids(
            local_sort_table, rank_to_row_ids_for_table, table_to_send_hashes,
            parallel);
    }

    //
    // 2. Shuffle data
    //

    mpi_comm_info comm_info_table(table_to_send->columns);
    comm_info_table.set_counts(table_to_send_hashes, parallel);
    // NOTE: shuffle_table_kernel decrefs input arrays
    table_info* collected_table = shuffle_table_kernel(
        table_to_send, table_to_send_hashes, comm_info_table, parallel);

    // There are a few cases for cleanup:
    // 1. Table was already sorted: In this case, local_sort_table = table, so
    // we don't own local_sort_table and can't delete it.
    // 1.1 If point table --> table_to_send = local_sort_table = table.
    // Shuffle will decref table_to_send (and hence local_sort_table and table),
    // so no additional clean up should be needed.
    // 1.2 If interval table -->
    // create_send_table_and_hashes_from_rank_to_row_ids will decref
    // local_sort_table (and hence table), and then shuffle will decref
    // table_to_send. We own table_to_send and will delete it.
    // 2. Table was not already sorted: sort_values_table_local will decref
    // table.
    // 2.1 If point table --> table_to_send = local_sort_table, Shuffle
    // will decref table_to_send (and hence local_sort_table). Deleting
    // local_sort_table is sufficient for cleanup.
    // 2.2 If interval table -->
    // create_send_table_and_hashes_from_rank_to_row_ids will decref
    // local_sort_table. Shuffle will decref table_to_send.
    // We own both local_sort_table and table_to_send, and can delete both.
    if (!table_already_sorted) {
        delete_table(local_sort_table);
    }
    if (!is_table_point_side) {
        delete_table(table_to_send);
    }

    //
    // 3. Sort the shuffled tables locally
    //

    table_info* ret_table =
        sort_values_table_local(collected_table, (is_table_point_side ? 1 : 2),
                                asc, na_pos, nullptr, parallel);
    delete_table(collected_table);
    return ret_table;
}

table_info* sort_table_for_interval_join_py_entrypoint(table_info* table,
                                                       array_info* bounds_arr,
                                                       bool is_table_point_side,
                                                       bool parallel) {
    int n_pes, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    try {
        table_info* sorted_table = sort_table_for_interval_join(
            table, false, bounds_arr, is_table_point_side, myrank, n_pes,
            parallel);
        delete_info_decref_array(bounds_arr);
        return sorted_table;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

/**
 * @brief Sort the tables in a point in interval or interval overlap join.
 *
 * @param table_1 First table. Could be either point side or interval side.
 * @param table_2 Second table. Always interval side.
 * @param table_1_parallel Whether the first table is distributed.
 * @param table_2_parallel Whether the second table is distributed.
 * @param is_table_1_point_side Whether the first table is the point side table
 * in a point in interval join.
 * @return std::tuple<table_info*, table_info*, array_info*> sorted table 1,
 * sorted table 2, and bounds for the ranks.
 */
std::tuple<table_info*, table_info*, array_info*>
sort_both_tables_for_interval_join(table_info* table_1, table_info* table_2,
                                   bool table_1_parallel, bool table_2_parallel,
                                   bool is_table_1_point_side, bool strict) {
    bool parallel_trace = (table_1_parallel || table_2_parallel);
    tracing::Event ev("sort_both_tables_for_interval_join", parallel_trace);
    ev.add_attribute("table_1_len", table_1->nrows());
    ev.add_attribute("table_2_len", table_2->nrows());

    tracing::Event ev_dict(
        "sort_both_tables_for_interval_join_handle_dict_encoded_arrays",
        parallel_trace);
    handle_dict_encoded_arrays_for_interval_join_sort(
        std::vector<table_info*>{table_1, table_2},
        std::vector<bool>{table_1_parallel, table_2_parallel});
    ev_dict.finalize();

    std::vector<uint64_t> n_local_vec{table_1->nrows(), table_2->nrows()};
    std::vector<uint64_t> n_total_vec{n_local_vec[0], n_local_vec[1]};
    if (table_1_parallel) {
        MPI_Allreduce(n_local_vec.data(), n_total_vec.data(), 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    }
    if (table_2_parallel) {
        MPI_Allreduce(n_local_vec.data() + 1, n_total_vec.data() + 1, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    }

    //
    // 1. Sort the tables locally
    //

    int64_t asc[2] = {1, 1};
    int64_t na_pos[2] = {1, 1};

    // sort_values_table_local will decref all arrays in table_1
    // and table_2.
    table_info* local_sort_table_1 =
        sort_values_table_local(table_1, (is_table_1_point_side ? 1 : 2), asc,
                                na_pos, nullptr, table_1_parallel);
    table_info* local_sort_table_2 = sort_values_table_local(
        table_2, 2, asc, na_pos, nullptr, table_2_parallel);

    // Return the local sort output in the non-parallel case,
    // or if both tables are empty.
    // If either table is replicated, we can return the local sort
    // directly since each rank has all the information to construct
    // its output.
    if (!table_1_parallel || !table_2_parallel ||
        (n_total_vec[0] == 0 && n_total_vec[1] == 0)) {
        return {local_sort_table_1, local_sort_table_2, nullptr};
    }

    // For all subsequent actions, we set `parallel` as true because at this
    // point both tables are guaranteed to be parallel.
    bool parallel = true;

    int n_pes, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    //
    // 2. Compute the split bounds
    //

    std::vector<table_info*> tables_vec{local_sort_table_1, local_sort_table_2};
    std::vector<uint64_t> n_keys_vec{
        static_cast<uint64_t>(is_table_1_point_side ? 1 : 2), 2};
    array_info* bounds_arr = get_parallel_sort_bounds_for_domain(
        tables_vec, n_keys_vec, n_local_vec, n_total_vec, myrank, n_pes,
        parallel);

    //
    // 3. Compute destination ranks for each row in both tables, shuffle the
    // data and then do local sort on the shuffled tables. Note that "bad"
    // intervals (i.e. start > end) will be filtered out.
    //

    // All arrays in local_sort_table_1 will be decref-ed
    table_info* ret_table_1 = sort_table_for_interval_join(
        local_sort_table_1, true, bounds_arr, is_table_1_point_side, myrank,
        n_pes, parallel, strict);
    delete_table(local_sort_table_1);

    // All arrays in local_sort_table_2 will be decref-ed
    table_info* ret_table_2 =
        sort_table_for_interval_join(local_sort_table_2, true, bounds_arr,
                                     false, myrank, n_pes, parallel, strict);
    delete_table(local_sort_table_2);

    return {ret_table_1, ret_table_2, bounds_arr};
}

std::pair<table_info*, table_info*> sort_tables_for_point_in_interval_join(
    table_info* table_point, table_info* table_interval,
    bool table_point_parallel, bool table_interval_parallel, bool strict) {
    table_info *sorted_point, *sorted_interval;
    array_info* bounds_arr;

    std::tie(sorted_point, sorted_interval, bounds_arr) =
        sort_both_tables_for_interval_join(
            table_point, table_interval, table_point_parallel,
            table_interval_parallel, true, strict);

    if (bounds_arr != nullptr) delete_info_decref_array(bounds_arr);
    return {sorted_point, sorted_interval};
}

std::tuple<table_info*, table_info*, array_info*>
sort_tables_for_interval_overlap_join(table_info* table_1, table_info* table_2,
                                      bool table_1_parallel,
                                      bool table_2_parallel) {
    return sort_both_tables_for_interval_join(
        table_1, table_2, table_1_parallel, table_2_parallel, false, true);
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
    const std::shared_ptr<uint32_t[]>& hashes;
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
 * @param is_parallel: whether we run in parallel or not.
 * @return the table to be used.
 */
static table_info* drop_duplicates_keys_inner(table_info* in_table,
                                              int64_t num_keys,
                                              bool dropna = true,
                                              bool is_parallel = true) {
    size_t n_rows = (size_t)in_table->nrows();
    std::vector<array_info*> key_arrs(in_table->columns.begin(),
                                      in_table->columns.begin() + num_keys);
    uint32_t seed = SEED_HASH_CONTAINER;
    std::shared_ptr<uint32_t[]> hashes = hash_keys(key_arrs, seed, is_parallel);
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
    hashes.reset();
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
 * As for the join, this relies on using hash keys for the partitioning.
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
 * @param is_parallel: whether we run in parallel or not.
 * @param drop_duplicates_dict: Do we need to ensure a dictionary has no
 * duplicates?
 * @param hashes: the precomputed hashes to use for the hash table. If set to
 *                `nullptr` then the hashes are computed inside this function
 *                and deleted at the end. If passed they are not deleted.
 *
 * collate the rows on the computational node 1 corresponds to the second step
 * of the operation after the rows have been merged on the computation
 * @return the table to be used.
 */
table_info* drop_duplicates_table_inner(table_info* in_table, int64_t num_keys,
                                        int64_t keep, int step,
                                        bool is_parallel, bool dropna,
                                        bool drop_duplicates_dict,
                                        std::shared_ptr<uint32_t[]> hashes) {
    tracing::Event ev("drop_duplicates_table_inner", is_parallel);
    ev.add_attribute("table_nrows_before",
                     static_cast<size_t>(in_table->nrows()));
    if (ev.is_tracing()) {
        size_t global_table_nbytes = table_global_memory_size(in_table);
        ev.add_attribute("g_table_nbytes", global_table_nbytes);
    }
    const bool delete_hashes = bool(hashes);
    size_t n_rows = (size_t)in_table->nrows();
    std::vector<array_info*> key_arrs(num_keys);
    for (size_t iKey = 0; iKey < size_t(num_keys); iKey++) {
        array_info* key = in_table->columns[iKey];
        // If we are dropping duplicates dictionary encoding assumes
        // that the dictionary values are unique.
        if (key->arr_type == bodo_array_type::DICT) {
            if (drop_duplicates_dict) {
                // Should we ensure that the dictionary values are unique?
                // If this is just a local drop duplicates then we don't
                // need to do this and can just do a best effort approach.
                drop_duplicates_local_dictionary(key);
            }
            key = key->info2;
        }
        key_arrs[iKey] = key;
    }
    uint32_t seed = SEED_HASH_CONTAINER;
    if (!hashes) {
        hashes = hash_keys(key_arrs, seed, is_parallel,
                           /*global_dict_needed=*/false);
    }
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
    if (delete_hashes) {
        hashes.reset();
    }
    ev.add_attribute("table_nrows_after",
                     static_cast<size_t>(ret_table->nrows()));
    return ret_table;
}

/** This function is for dropping the null keys.
 * This C++ code is used for the compute_categorical_index
 * ---It keeps only the non-null keys
 * ---It returns only the keys.
 *
 * As for the join, this relies on using hash keys for the partitioning.
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
        return drop_duplicates_keys_inner(in_table, num_keys, dropna,
                                          is_parallel);
    }
    // parallel case
    // pre reduction of duplicates
    table_info* red_table =
        drop_duplicates_keys_inner(in_table, num_keys, dropna, is_parallel);
    // shuffling of values
    table_info* shuf_table = shuffle_table(red_table, num_keys, is_parallel, 0);
    // no need to decref since shuffle_table() steals a reference
    delete_table(red_table);
    // reduction after shuffling
    int keep = 0;
    // Set dropna=False for drop_duplicates_table_inner because
    // drop_duplicates_keys_inner should have already removed any NA
    // values
    table_info* ret_table =
        drop_duplicates_table_inner(shuf_table, num_keys, keep, 1, is_parallel,
                                    false, /*drop_duplicates_dict=*/true);
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
 * @param num_keys: number of columns to use identifying duplicates
 * @param keep: integer specifying the expected behavior.
 *        keep = 0 corresponds to the case of keep="first" keep first entry
 *        keep = 1 corresponds to the case of keep="last" keep last entry
 *        keep = 2 corresponds to the case of keep=False : remove all duplicates
 * @param dropna: Should NA be included in the final table
 * @return the vector of pointers to be used.
 */
table_info* drop_duplicates_table(table_info* in_table, bool is_parallel,
                                  int64_t num_keys, int64_t keep, bool dropna,
                                  bool drop_local_first) {
    try {
        // serial case
        if (!is_parallel) {
            return drop_duplicates_table_inner(in_table, num_keys, keep, 1,
                                               is_parallel, dropna,
                                               /*drop_duplicates_dict=*/true);
        }
        // parallel case
        // pre reduction of duplicates
        table_info* red_table;
        if (drop_local_first) {
            red_table = drop_duplicates_table_inner(
                in_table, num_keys, keep, 2, is_parallel, dropna,
                /*drop_duplicates_dict=*/false);
        } else {
            red_table = in_table;
        }
        // shuffling of values
        table_info* shuf_table =
            shuffle_table(red_table, num_keys, is_parallel);
        // no need to decref since shuffle_table() steals a reference
        if (drop_local_first) delete_table(red_table);
        // reduction after shuffling
        // We don't drop NA values again because the first
        // drop_duplicates_table_inner should have already handled this
        if (drop_local_first) dropna = false;
        table_info* ret_table = drop_duplicates_table_inner(
            shuf_table, num_keys, keep, 1, is_parallel, dropna,
            /*drop_duplicates_dict=*/true);
        delete_table(shuf_table);
        // returning table
        return ret_table;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

// Note: union_tables steals a reference and generates a new output.
table_info* union_tables_inner(table_info** in_table, int64_t num_tables,
                               bool drop_duplicates, bool is_parallel) {
    tracing::Event ev("union_tables", is_parallel);
    // Drop duplicates locally first. This won't do a shuffle yet because we
    // only want to do 1 shuffle.
    std::vector<table_info*> locally_processed_table(num_tables);

    // All tables have the same number of columns and all columns are keys
    int64_t num_keys = in_table[0]->ncols();

    if (drop_duplicates) {
        tracing::Event ev_drop_duplicates("union_tables_drop_duplicates_local",
                                          is_parallel);
        for (int i = 0; i < num_tables; i++) {
            locally_processed_table[i] = drop_duplicates_table_inner(
                in_table[i], num_keys, 0, 1, false, false, false);
            // drop_duplicates_table_inner steals a reference on the arrays
            // but doesn't delete the input table.
            delete_table(in_table[i]);
        }
        ev_drop_duplicates.finalize();
    } else {
        for (int i = 0; i < num_tables; i++) {
            locally_processed_table[i] = in_table[i];
        }
    }
    // Unify all of the dictionaries in any dictionary encoded array columns.
    // All tables must share the same schema so we iterate over the first table
    // for types.
    tracing::Event ev_unify_dicts("union_tables_unify_dictionaries",
                                  is_parallel);
    table_info* base_table = locally_processed_table[0];
    // Optimize unify dictionaries for two arrays (common case)
    if (num_tables == 2) {
        for (int j = 0; j < num_keys; j++) {
            if (base_table->columns[j]->arr_type == bodo_array_type::DICT) {
                // Unify all of the dictionaries.
                array_info* arr1 = locally_processed_table[0]->columns[j];
                array_info* arr2 = locally_processed_table[1]->columns[j];
                // Ensure we have global/unique data for the dictionary.
                make_dictionary_global_and_unique(arr1, is_parallel);
                make_dictionary_global_and_unique(arr2, is_parallel);
                unify_dictionaries(arr1, arr2, is_parallel, is_parallel);
            }
        }
    } else {
        for (int j = 0; j < num_keys; j++) {
            if (base_table->columns[j]->arr_type == bodo_array_type::DICT) {
                // Unify all of the dictionaries.
                std::vector<array_info*> dict_arrs(num_tables);
                std::vector<bool> is_parallels(num_tables);
                for (int i = 0; i < num_tables; i++) {
                    // Ensure we have global/unique data for the dictionary.
                    array_info* arr = locally_processed_table[i]->columns[j];
                    make_dictionary_global_and_unique(arr, is_parallel);
                    dict_arrs[i] = arr;
                    is_parallels[i] = is_parallel;
                }
                unify_several_dictionaries(dict_arrs, is_parallels);
            }
        }
    }
    ev_unify_dicts.finalize();

    // Shuffle the tables
    tracing::Event ev_shuffle("union_table_shuffle", is_parallel);
    std::vector<table_info*> shuffled_tables(num_tables);
    if (is_parallel && drop_duplicates) {
        // Shuffle the tables. We don't need to shuffle the tables
        // if we aren't dropping duplicates.
        for (int i = 0; i < num_tables; i++) {
            shuffled_tables[i] =
                shuffle_table(locally_processed_table[i], num_keys, true);
            // no need to decref since shuffle_table() steals a reference
            // but we need to delete the table
            delete_table(locally_processed_table[i]);
        }
    } else {
        for (int i = 0; i < num_tables; i++) {
            shuffled_tables[i] = locally_processed_table[i];
        }
    }
    ev_shuffle.finalize();

    // Concatenate all of the tables. Note concat_tables decrefs
    // the input tables.
    tracing::Event ev_concat("union_table_concat", is_parallel);
    table_info* concatenated_table = concat_tables(shuffled_tables);
    ev_concat.finalize();
    table_info* out_table;
    if (drop_duplicates) {
        tracing::Event ev_drop_duplicates("union_tables_drop_duplicates_local",
                                          is_parallel);
        // Drop any duplicates in the concatenated tables
        out_table = drop_duplicates_table_inner(concatenated_table, num_keys, 0,
                                                1, false, false, false);
        // drop_duplicates_table_inner steals a reference on the arrays
        // but doesn't delete the input table.
        delete_table(concatenated_table);
        ev_drop_duplicates.finalize();
    } else {
        out_table = concatenated_table;
    }
    ev.finalize();
    return out_table;
}

table_info* union_tables(table_info** in_table, int64_t num_tables,
                         bool drop_duplicates, bool is_parallel) {
    try {
        return union_tables_inner(in_table, num_tables, drop_duplicates,
                                  is_parallel);
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
        if (parallel) {
            bool all_gather = true;
            size_t n_cols = tab_out->ncols();
            table_info* tab_ret =
                gather_table(tab_out, n_cols, all_gather, parallel);
            delete_table(tab_out);
            return tab_ret;
        }
        return tab_out;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}
/**
 * Search for pattern in each input element using
 * boost::xpressive::regex_search(in, pattern)
 * @param[in] in_arr input array of string  elements
 * @param[in] case_sensitive bool whether pattern is case sensitive or not
 * @param[in] match_beginning bool whether the pattern starts at the beginning
 * @param[in] regex regular expression pattern
 * @param[out] out_arr output array of bools specifying whether the pattern
 *                      matched for corresponding in_arr element
 */
void get_search_regex(array_info* in_arr, const bool case_sensitive,
                      const bool match_beginning, char const* const pat,
                      array_info* out_arr) {
    tracing::Event ev("get_search_regex");
    // See:
    // https://www.boost.org/doc/libs/1_76_0/boost/xpressive/regex_constants.hpp
    boost::xpressive::regex_constants::syntax_option_type flag =
        boost::xpressive::regex_constants::ECMAScript;  // default value
    if (!case_sensitive) {
        flag = boost::xpressive::icase;
    }
    // match_continuous specifies that the expression must match a sub-sequence
    // that begins at first
    boost::xpressive::regex_constants::match_flag_type match_flag =
        match_beginning ? boost::xpressive::regex_constants::match_continuous
                        : boost::xpressive::regex_constants::match_default;

    const boost::xpressive::cregex pattern =
        boost::xpressive::cregex::compile(pat, flag);
    // Use of cmatch is needed to achieve better performance.
    boost::xpressive::cmatch m;
    const size_t nRow = in_arr->length;
    ev.add_attribute("local_nRows", nRow);
    int64_t num_match = 0;
    if (in_arr->arr_type == bodo_array_type::STRING) {
        offset_t const* const data2 =
            reinterpret_cast<offset_t*>(in_arr->data2);
        char const* const data1 = in_arr->data1;
        for (size_t iRow = 0; iRow < nRow; iRow++) {
            bool bit = in_arr->get_null_bit(iRow);
            if (bit) {
                const offset_t start_pos = data2[iRow];
                const offset_t end_pos = data2[iRow + 1];
                if (boost::xpressive::regex_search(data1 + start_pos,
                                                   data1 + end_pos, m, pattern,
                                                   match_flag)) {
                    out_arr->at<bool>(iRow) = true;
                    num_match++;
                } else {
                    out_arr->at<bool>(iRow) = false;
                }
            }
            out_arr->set_null_bit(iRow, bit);
        }
        ev.add_attribute("local_num_match", num_match);

    } else if (in_arr->arr_type == bodo_array_type::DICT) {
        // For dictionary-encoded string arrays, we optimize
        // by first doing the computation on the dictionary (info1)
        // (which is presumably much smaller), and then
        // building the output boolean array by indexing into this
        // result.

        array_info* dict_arr = in_arr->info1;

        // Allocate boolean array to store output of get_search_regex
        // on the dictionary (dict_arr)
        array_info* dict_arr_out =
            alloc_nullable_array(dict_arr->length, Bodo_CTypes::_BOOL, 0);

        // Compute recursively on the dictionary
        // (dict_arr; which is just a string array).
        // We incref `dict_arr_out` and `dict_arr` since they'll be decrefed
        // when we call this function recursively
        incref_array(dict_arr);
        incref_array(dict_arr_out);
        get_search_regex(dict_arr, case_sensitive, match_beginning, pat,
                         dict_arr_out);

        array_info* indices_arr = in_arr->info2;

        // Iterate over the indices, and assign values to the output
        // boolean array from dict_arr_out.
        for (size_t iRow = 0; iRow < nRow; iRow++) {
            bool bit = in_arr->get_null_bit(iRow);
            if (bit) {
                // Get index in the dictionary
                int32_t iiRow = indices_arr->at<int32_t>(iRow);
                // Get output from dict_arr_out for this dict value
                bool value = dict_arr_out->at<bool>(iiRow);
                out_arr->at<bool>(iRow) = value;
                if (value) {
                    num_match++;
                }
            }
            out_arr->set_null_bit(iRow, bit);
        }
        ev.add_attribute("local_num_match", num_match);
        // Free the output of computation on dict_arr
        delete_info_decref_array(dict_arr_out);
    } else {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "array in_arr type should be string");
    }
    // decref in_arr/out_arr generated in get_search_regex
    decref_array(out_arr);
    decref_array(in_arr);
}

/**
 * @brief Compute replace_regex on a slice of a string array using
 * boost::xpressive::regex_replace. This is used to enable optimizations
 * in which only part of a string array is computed for dictionary encoding.
 *
 * @param in_arr The input string array to compute replacement elements for.
 * @param pat The string pattern used to find where to replace elements.
 * @param replacement The string used as a replacement wherever the pattern is
 * encountered.
 * @param start_idx The starting index to compute for the array slice.
 * @param end_idx The ending index to compute for the array slice (not
 * inclusive).
 * @return array_info* An output array of replaced strings the same length as
 * end_idx - start_idx. Elements not included in the slice are not included in
 * the output.
 */
array_info* get_replace_regex_slice(array_info* in_arr, char const* const pat,
                                    char const* replacement, size_t start_idx,
                                    size_t end_idx) {
    // See:
    // https://www.boost.org/doc/libs/1_76_0/boost/xpressive/regex_constants.hpp
    boost::xpressive::regex_constants::syntax_option_type flag =
        boost::xpressive::regex_constants::ECMAScript;  // default value

    const boost::xpressive::cregex pattern =
        boost::xpressive::cregex::compile(pat, flag);

    offset_t const* const in_data2 = reinterpret_cast<offset_t*>(in_arr->data2);
    char const* const in_data1 = in_arr->data1;

    // Look at the pattern length for assumptions about allocated space.
    size_t repLen = strlen(replacement);
    repLen = repLen == 0 ? 1 : repLen;
    size_t num_chars = 0;
    size_t out_arr_len = end_idx - start_idx;
    if (repLen == 1) {
        // If pattern is a single character (or empty string), we can use
        // the faster path of just allocated the same size as the input
        // array. This is because we know that in the worst case (replacing
        // each character individually with the replacement) the output will
        // be the same size as the input.
        num_chars = in_data2[end_idx] - in_data2[start_idx];
    } else {
        // If the array can grow in the worse case then we do an extra pass
        // to compute the required number of characters.

        // Allocate a working buffer to contain the result of the regex
        // replace.
        size_t buffer_size = 1000;
        char* buffer = (char*)malloc(buffer_size);
        // To construct the output array we need to know how many characters
        // are in the output. As a result we compute the result twice, once
        // to get the size of the output and once with the result.
        // TODO: Determine how to dynamically resize the output array since
        // the compute cost seems to dominate.
        for (size_t iRow = start_idx; iRow < end_idx; iRow++) {
            bool bit = in_arr->get_null_bit(iRow);
            if (bit) {
                const offset_t start_pos = in_data2[iRow];
                const offset_t end_pos = in_data2[iRow + 1];
                while (((end_pos - start_pos) * repLen) >= buffer_size) {
                    buffer_size *= 2;
                    buffer = (char*)realloc(buffer, buffer_size);
                }
                char* out_buffer = boost::xpressive::regex_replace(
                    buffer, in_data1 + start_pos, in_data1 + end_pos, pattern,
                    replacement);
                num_chars += out_buffer - buffer;
            }
        }
        // Free the buffer now that its no longer needed.
        free(buffer);
    }
    // Allocate the output array. We add 1 because regex_replace
    // may insert a null terminator
    array_info* out_arr = alloc_string_array(out_arr_len, num_chars, 0);
    offset_t* const out_data2 = reinterpret_cast<offset_t*>(out_arr->data2);
    // Initialize the first offset to 0
    out_data2[0] = 0;
    char* const out_data1 = out_arr->data1;
    for (size_t outRow = 0, iRow = start_idx; iRow < end_idx;
         iRow++, outRow++) {
        bool bit = in_arr->get_null_bit(iRow);
        const offset_t out_offset = out_data2[outRow];
        size_t num_chars = 0;
        if (bit) {
            const offset_t start_pos = in_data2[iRow];
            const offset_t end_pos = in_data2[iRow + 1];
            char* out_buffer = boost::xpressive::regex_replace(
                out_data1 + out_offset, in_data1 + start_pos,
                in_data1 + end_pos, pattern, replacement);
            num_chars = out_buffer - (out_data1 + out_offset);
        }
        out_arr->set_null_bit(outRow, bit);
        out_data2[outRow + 1] = out_offset + num_chars;
    }
    // Update the actual number of characters since there may be fewer than we
    // allocated. This is necessary if we shuffle data.
    out_arr->n_sub_elems = out_data2[out_arr_len] - out_data2[0];
    return out_arr;
}

array_info* get_replace_regex(array_info* in_arr, char const* const pat,
                              char const* replacement, const bool is_parallel) {
    array_info* out_arr = nullptr;
    if (in_arr->arr_type == bodo_array_type::STRING) {
        out_arr = get_replace_regex_slice(in_arr, pat, replacement, 0,
                                          in_arr->length);
    } else if (in_arr->arr_type == bodo_array_type::DICT) {
        // Threshold used to determine whether to use the parallel path for a
        // global dictionary. The takeaway here is that large dictionaries may
        // benefit from splitting the compute and adding extra communication.
        // TODO[BE-4418]: Benchmark and creat a reasonable threshold for
        // dividing computation that accounts for the cost to compute replace on
        // each byte, the communication initialization overhead, and the cost to
        // perform the allgatherv.
        const size_t LOCAL_COMPUTE_THRESHOLD = 1000000;
        // For dictionary-encoded string arrays, we optimize
        // by recursing on the dictionary (info1)
        // (which is presumably much smaller), and then
        // copying the null bitmap and indices.
        array_info* dict_arr = in_arr->info1;
        array_info* indices_arr = in_arr->info2;

        // We use a special path for large dictionaries that are global and
        // called in parallel.
        bool use_parallel_path = is_parallel && in_arr->has_global_dictionary &&
                                 dict_arr->length >= LOCAL_COMPUTE_THRESHOLD;
        array_info* new_dict;
        if (use_parallel_path) {
            // Compute your local slice of the dictionary.
            size_t rank = dist_get_rank();
            size_t nranks = dist_get_size();
            size_t start_idx = dist_get_start(dict_arr->length, nranks, rank);
            size_t end_idx = dist_get_end(dict_arr->length, nranks, rank);
            // Compute the length with just your chunk.
            array_info* dict_slice = get_replace_regex_slice(
                dict_arr, pat, replacement, start_idx, end_idx);
            // Wrap the dictionary in a table so we can do an allgatherv.
            table_info* dict_table = new table_info();
            dict_table->columns.push_back(dict_slice);
            table_info* gathered_table =
                gather_table(dict_table, 1, true, true);
            new_dict = gathered_table->columns[0];
            // Delete the tables and free the original slice
            delete dict_table;
            delete gathered_table;
        } else {
            // Just recurse on the dictionary locally
            new_dict = get_replace_regex_slice(dict_arr, pat, replacement, 0,
                                               dict_arr->length);
        }
        array_info* new_indices = copy_array(indices_arr);
        out_arr = new array_info(bodo_array_type::DICT, Bodo_CTypes::STRING,
                                 in_arr->length, -1, -1, NULL, NULL, NULL,
                                 new_indices->null_bitmask, NULL, {}, NULL, 0,
                                 0, 0, in_arr->has_global_dictionary,
                                 false,  // Note replace can create collisions.
                                 false, new_dict, new_indices);
    } else {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "array in_arr type should be string");
    }
    // decref in_arr generated in get_replace_regex
    decref_array(in_arr);
    return out_arr;
}
