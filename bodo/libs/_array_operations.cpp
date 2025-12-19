#include "_array_operations.h"
#include <algorithm>
#include <boost/xpressive/xpressive.hpp>
#include <functional>
#include <random>

#include "_array_build_buffer.h"
#include "_array_hash.h"
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_distributed.h"
#include "_shuffle.h"
#include "streaming/_dict_encoding.h"
#include "vendored/gfx/timsort.hpp"

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
static void array_isin_kernel(std::shared_ptr<array_info> out_arr,
                              std::shared_ptr<array_info> in_arr,
                              std::shared_ptr<array_info> in_values,
                              bool is_parallel) {
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
    hash_array(hashes_values.get(), in_values, (size_t)len_values, seed,
               is_parallel,
               /*global_dict_needed=*/false);

    int64_t len_in_arr = in_arr->length;
    std::unique_ptr<uint32_t[]> hashes_in_arr =
        std::make_unique<uint32_t[]>(len_in_arr);
    hash_array(hashes_in_arr.get(), in_arr, (size_t)len_in_arr, seed,
               is_parallel,
               /*global_dict_needed=*/false);

    std::function<bool(int64_t, int64_t)> equal_fct =
        [&](int64_t const& pos1, int64_t const& pos2) -> bool {
        int64_t pos1_b, pos2_b;
        std::shared_ptr<array_info> arr1_b, arr2_b;
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
    bodo::unord_set_container<size_t, std::function<size_t(int64_t)>,
                              std::function<bool(int64_t, int64_t)>>
        eset({}, hash_fct, equal_fct);
    for (int64_t pos = 0; pos < len_values; pos++) {
        eset.insert(pos);
    }
    uint8_t* out_arr_data = (uint8_t*)out_arr->data1();
    for (int64_t pos = 0; pos < len_in_arr; pos++) {
        bool test = eset.count(pos + len_values) == 1;
        SetBitTo(out_arr_data, pos, test);
    }
}

void array_isin_py_entry(array_info* p_out_arr, array_info* p_in_arr,
                         array_info* p_in_values, bool is_parallel) {
    try {
        // convert raw pointers coming from Python to smart pointers and take
        // ownership
        std::shared_ptr<array_info> out_arr =
            std::shared_ptr<array_info>(p_out_arr);
        std::shared_ptr<array_info> in_arr =
            std::shared_ptr<array_info>(p_in_arr);
        std::shared_ptr<array_info> in_values =
            std::shared_ptr<array_info>(p_in_values);

        if (!is_parallel) {
            array_isin_kernel(out_arr, in_arr, in_values, is_parallel);
            return;
        }
        std::vector<std::shared_ptr<array_info>> vect_in_values = {in_values};
        std::shared_ptr<table_info> table_in_values =
            std::make_shared<table_info>(vect_in_values);
        std::vector<std::shared_ptr<array_info>> vect_in_arr = {in_arr};
        std::shared_ptr<table_info> table_in_arr =
            std::make_shared<table_info>(vect_in_arr);

        int64_t num_keys = 1;
        std::shared_ptr<table_info> shuf_table_in_values =
            shuffle_table(std::move(table_in_values), num_keys, is_parallel);
        // we need the comm_info and hashes for the reverse shuffling
        std::shared_ptr<uint32_t[]> hashes =
            hash_keys_table(table_in_arr, 1, SEED_HASH_PARTITION, is_parallel);
        mpi_comm_info comm_info(table_in_arr->columns, hashes, is_parallel);
        std::shared_ptr<table_info> shuf_table_in_arr = shuffle_table_kernel(
            std::move(table_in_arr), hashes, comm_info, is_parallel);
        // Creation of the output array.
        int64_t len = shuf_table_in_arr->columns[0]->length;
        std::shared_ptr<array_info> shuf_out_arr = alloc_array_top_level(
            len, -1, -1, out_arr->arr_type, out_arr->dtype, -1, 0,
            out_arr->num_categories);
        // Calling isin on the shuffled info
        array_isin_kernel(shuf_out_arr, shuf_table_in_arr->columns[0],
                          shuf_table_in_values->columns[0], is_parallel);

        // Deleting the data after usage
        shuf_table_in_values.reset();
        shuf_table_in_arr.reset();
        // Now the reverse shuffling operation.
        reverse_shuffle_preallocated_data_array(shuf_out_arr, out_arr,
                                                comm_info);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return;
    }
}

//
// UTILS FOR SORT
//

/**
 * @brief Get the number of samples based on the size of the local table.
 *
 * @param n_pes Number of MPI ranks.
 * @param n_total Total number of rows in the table.
 * @param n_local Number of rows in the local table.
 */
int64_t get_num_samples_from_local_table(int n_pes, int64_t n_total,
                                         int64_t n_local) {
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
    return std::min(n_global_sample / n_pes, double(n_local));
}

bodo::vector<int64_t> get_sample_selection_vector(int64_t n_local,
                                                  int64_t n_loc_sample) {
    std::mt19937 gen(1234567890);
    double block_size = double(n_local) / n_loc_sample;
    bodo::vector<int64_t> ListIdx(n_loc_sample);
    double cur_lo = 0;
    for (int64_t i = 0; i < n_loc_sample; i++) {
        int64_t lo = round(cur_lo);
        int64_t hi = round(cur_lo + block_size) - 1;
        ListIdx[i] = std::uniform_int_distribution<int64_t>(lo, hi)(gen);
        cur_lo += block_size;
    }
    return ListIdx;
}

/**
 * @brief Get n_loc_sample many samples from a locally sorted table.
 * This function splits the input table into roughly n_loc_sample blocks, and
 * then picks a row randomly from each block.
 *
 * @param local_sort Locally sorted table
 * @param n_keys Number of sort keys
 * @param n_loc_sample Number of samples to get
 * @param n_local Length of the table
 * @param parallel Only for tracing purposes
 * @return std::shared_ptr<table_info> Table with samples. Shape: n_loc_sample
 * rows, n_keys columns
 */
std::shared_ptr<table_info> get_samples_from_table_local(
    std::shared_ptr<table_info> local_sort, int64_t n_keys,
    int64_t n_loc_sample, int64_t n_local, bool parallel) {
    tracing::Event ev("get_samples_from_table_local", parallel);
    std::shared_ptr<table_info> samples = RetrieveTable(
        std::move(local_sort),
        get_sample_selection_vector(n_local, n_loc_sample), n_keys);
    return samples;
}

/**
 * @brief Get samples from locally sorted chunks of a distributed table.
 * Number of samples required for a reasonable sampling is determined
 * based on the total length of the table, number of ranks.
 *
 * @param local_sort Locally sorted table chunk.
 * @param n_keys Number of sort keys.
 * @param n_pes Number of MPI ranks.
 * @param n_total Global table length.
 * @param n_local Local table length (=> length of local_sort)
 * @param parallel Whether the execution is happening in parallel. Passed for
 * consistency and tracing purposes.
 * @return std::shared_ptr<table_info> Table of samples on rank 0, empty table
 * on all other ranks.
 */
std::shared_ptr<table_info> get_samples_from_table_parallel(
    std::shared_ptr<table_info> local_sort, int64_t n_keys, int n_pes,
    int64_t n_total, int64_t n_local, bool parallel) {
    tracing::Event ev("get_samples_from_table_parallel", parallel);
    ev.add_attribute("n_keys", n_keys);
    ev.add_attribute("n_local", n_local);
    ev.add_attribute("n_total", n_total);

    int64_t n_loc_sample =
        get_num_samples_from_local_table(n_pes, n_total, n_local);

    // Get n_loc_sample many local samples from the local sorted chunk
    std::shared_ptr<table_info> samples = get_samples_from_table_local(
        std::move(local_sort), n_keys, n_loc_sample, n_local, parallel);

    // Collecting all samples
    bool all_gather = false;
    std::shared_ptr<table_info> all_samples =
        gather_table(std::move(samples), n_keys, all_gather, parallel);
    return all_samples;
}

//
// PARALLEL SORT
//

template <typename IndexT>
    requires(std::is_same_v<IndexT, int32_t> || std::is_same_v<IndexT, int64_t>)
bodo::vector<IndexT> sort_values_table_local_get_indices(
    std::shared_ptr<table_info> in_table, size_t n_keys,
    const int64_t* vect_ascending, const int64_t* na_position, bool is_parallel,
    size_t start_offset, size_t n_rows, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    tracing::Event ev("sort_values_table_local", is_parallel);
    bodo::vector<IndexT> ListIdx(n_rows, pool);
    for (size_t i = 0; i < n_rows; i++) {
        ListIdx[i] = start_offset + i;
    }

    // The comparison operator gets called many times by timsort so any overhead
    // can influence the sort time significantly
    if (n_keys == 1) {
        // comparison operator with less overhead than the general n_key > 1
        // case. We call KeyComparisonAsPython_Column directly without looping
        // through the keys, assume fixed values for some parameters and pass
        // less parameters around
        std::shared_ptr<array_info> key_col = in_table->columns[0];
        bool ascending = vect_ascending[0];
        bool na_last = na_position[0];
        if (ascending) {
            const bool na_position_bis = (!na_last) ^ ascending;
            const auto f = [&](size_t const& iRow1,
                               size_t const& iRow2) -> bool {
                int test = KeyComparisonAsPython_Column(
                    na_position_bis, key_col, iRow1, key_col, iRow2);
                return test > 0;
            };
            gfx::timsort(ListIdx.begin(), ListIdx.end(), f);
        } else {
            const bool na_position_bis = (!na_last) ^ ascending;
            const auto f = [&](size_t const& iRow1,
                               size_t const& iRow2) -> bool {
                int test = KeyComparisonAsPython_Column(
                    na_position_bis, key_col, iRow1, key_col, iRow2);
                return test < 0;
            };
            gfx::timsort(ListIdx.begin(), ListIdx.end(), f);
        }
    } else {
        const auto f = [&](size_t const& iRow1, size_t const& iRow2) -> bool {
            size_t shift_key1 = 0, shift_key2 = 0;
            bool test = KeyComparisonAsPython(
                n_keys, vect_ascending, in_table->columns, shift_key1, iRow1,
                in_table->columns, shift_key2, iRow2, na_position);
            return test;
        };
        gfx::timsort(ListIdx.begin(), ListIdx.end(), f);
    }

    return ListIdx;
}

// Explicit instantiaion of templates for linking
template bodo::vector<int32_t> sort_values_table_local_get_indices(
    std::shared_ptr<table_info> in_table, size_t n_keys,
    const int64_t* vect_ascending, const int64_t* na_position, bool is_parallel,
    size_t start_offset, size_t n_rows, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm);
template bodo::vector<int64_t> sort_values_table_local_get_indices(
    std::shared_ptr<table_info> in_table, size_t n_keys,
    const int64_t* vect_ascending, const int64_t* na_position, bool is_parallel,
    size_t start_offset, size_t n_rows, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm);

/**
 *   SORT VALUES
 *
 * Sorts the input table by the first n_keys columns. Returns a list of indices
 * into the input table that can be used to retrieve the sorted table. See
 * sort_values_table_local for a method that returns a sorted table.
 *
 * @param is_parallel: true if data is distributed (used to indicate whether
 * tracing should be parallel or not)
 */
template <typename IndexT>
    requires(std::is_same_v<IndexT, int32_t> || std::is_same_v<IndexT, int64_t>)
std::shared_ptr<table_info> sort_values_table_local(
    std::shared_ptr<table_info> in_table, int64_t n_key,
    const int64_t* vect_ascending, const int64_t* na_position,
    const int64_t* dead_keys, bool is_parallel, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    bodo::vector<IndexT> ListIdx = sort_values_table_local_get_indices<IndexT>(
        in_table, n_key, vect_ascending, na_position, is_parallel, 0,
        in_table->nrows(), pool, mm);

    std::shared_ptr<table_info> ret_table;
    if (dead_keys == nullptr) {
        ret_table = RetrieveTable(std::move(in_table), ListIdx, -1, false, pool,
                                  std::move(mm));
    } else {
        uint64_t n_cols = in_table->ncols();
        std::vector<uint64_t> colInds;
        for (int64_t i = 0; i < n_key; i++) {
            if (dead_keys[i]) {
                // If this is the last reference to this
                // table, we can safely release reference (and potentially
                // memory if any) for the dead keys at this point.
                reset_col_if_last_table_ref(in_table, i);
            } else {
                colInds.push_back(i);
            }
        }
        for (uint64_t i = n_key; i < n_cols; i++) {
            colInds.push_back(i);
        }
        ret_table = RetrieveTable(std::move(in_table), ListIdx, colInds, false,
                                  pool, std::move(mm));
    }

    return ret_table;
}

/**
 *   SORT VALUES
 *
 * Sorts the input table by the first n_keys columns.
 *
 * @param is_parallel: true if data is distributed (used to indicate whether
 * tracing should be parallel or not)
 */
std::shared_ptr<table_info> sort_values_table_local(
    std::shared_ptr<table_info> in_table, int64_t n_key,
    const int64_t* vect_ascending, const int64_t* na_position,
    const int64_t* dead_keys, bool is_parallel, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    if (in_table->nrows() < std::numeric_limits<int32_t>::max()) {
        return sort_values_table_local<int32_t>(in_table, n_key, vect_ascending,
                                                na_position, dead_keys,
                                                is_parallel, pool, mm);
    }

    return sort_values_table_local<int64_t>(in_table, n_key, vect_ascending,
                                            na_position, dead_keys, is_parallel,
                                            pool, mm);
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
 * @param n_keys Number of key columns.
 * @param vect_ascending Vector of booleans (one for each key column) describing
 * whether to sort in ascending order on the key columns.
 * @param na_position Vector of booleans (one for each key column) describing
 * where to put the NAs (last or first) in the key columns.
 * @param myrank MPI rank of the calling process.
 * @param n_pes Total number of MPI ranks.
 * @param parallel Is the process parallel.
 * @return std::shared_ptr<table_info> Bounds table with n_pes-1 rows. A full
 * bounds table is computed on rank 0, broadcasted to all ranks and returned.
 */
std::shared_ptr<table_info> compute_bounds_from_samples(
    std::shared_ptr<table_info> all_samples,
    std::shared_ptr<table_info> ref_table, int64_t n_keys,
    const int64_t* vect_ascending, const int64_t* na_position, int myrank,
    int n_pes, bool parallel) {
    tracing::Event ev("compute_bounds_from_samples", parallel);
    int mpi_root = 0;
    // Computing the bounds (splitters) on root
    std::shared_ptr<table_info> pre_bounds = nullptr;
    if (myrank == mpi_root) {
        std::shared_ptr<table_info> all_samples_sort = sort_values_table_local(
            std::move(all_samples), n_keys, vect_ascending, na_position,
            nullptr, parallel);
        int64_t n_samples = all_samples_sort->nrows();
        int64_t step = ceil(double(n_samples) / double(n_pes));
        std::vector<int64_t> ListIdxBounds(n_pes - 1);
        for (int i = 0; i < n_pes - 1; i++) {
            size_t pos = std::min((i + 1) * step, n_samples - 1);
            ListIdxBounds[i] = pos;
        }
        pre_bounds =
            RetrieveTable(std::move(all_samples_sort), ListIdxBounds, -1);
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
    std::shared_ptr<table_info> bounds =
        broadcast_table(std::move(ref_table), std::move(pre_bounds), nullptr,
                        n_keys, parallel, mpi_root);

    return bounds;
}

/**
 * @brief Compute data redistribution boundaries for parallel sort using a
 * sample of the data.
 *
 * @param local_sort locally sorted table
 * @param n_keys number of sort keys
 * @param vect_ascending ascending/descending order for each key
 * @param na_position NA behavior (first or last) for each key
 * @param n_local number of local rows
 * @param n_total number of global rows
 * @param myrank MPI rank
 * @param n_pes total MPI ranks
 * @param parallel parallel flag (should be true here but passing around for
 * code consistency)
 * @return std::shared_ptr<table_info> Bounds table with n_pes-1 rows.
 */
std::shared_ptr<table_info> get_parallel_sort_bounds(
    std::shared_ptr<table_info> local_sort, int64_t n_keys,
    int64_t* vect_ascending, int64_t* na_position, int64_t n_local,
    int64_t n_total, int myrank, int n_pes, bool parallel) {
    tracing::Event ev("get_parallel_sort_bounds", parallel);
    // Compute samples from the locally sorted table.
    // (Filled on rank 0, empty on all other ranks)
    std::shared_ptr<table_info> all_samples = get_samples_from_table_parallel(
        local_sort, n_keys, n_pes, n_total, n_local, parallel);

    // Compute split bounds from the samples.
    // Output is broadcasted to all ranks.
    std::shared_ptr<table_info> bounds = compute_bounds_from_samples(
        std::move(all_samples), std::move(local_sort), n_keys, vect_ascending,
        na_position, myrank, n_pes, parallel);

    return bounds;
}

std::shared_ptr<table_info> sort_values_table(
    std::shared_ptr<table_info> in_table, int64_t n_keys,
    int64_t* vect_ascending, int64_t* na_position, int64_t* dead_keys,
    int64_t* out_n_rows, std::shared_ptr<table_info> bounds, bool parallel) {
    tracing::Event ev("sort_values_table", parallel);

    if (out_n_rows != nullptr) {
        // Initialize to the input because local sort won't
        // change the number of elements. If we do a
        // distributed source the rows per rank may change.
        *out_n_rows = (int64_t)in_table->nrows();
    }

    // Convert all local dictionaries to global for dict columns.
    // Also sort the dictionaries, so the sorting process
    // is more efficient (we can compare indices directly)
    for (std::shared_ptr<array_info> arr : in_table->columns) {
        if (arr->arr_type == bodo_array_type::DICT) {
            // For dictionary encoded arrays we need the data to be unique
            // and global in case the dictionary is sorted (because we will
            // compare indices directly)
            make_dictionary_global_and_unique(arr, parallel, true);
        }
    }

    int64_t n_local = in_table->nrows();
    int64_t n_total = n_local;
    if (parallel) {
        CHECK_MPI(MPI_Allreduce(&n_local, &n_total, 1, MPI_LONG_LONG_INT,
                                MPI_SUM, MPI_COMM_WORLD),
                  "sort_values_table: MPI error on MPI_Allreduce:");
    }

    // Want to keep dead keys only when we will perform a shuffle operation
    // later in the function
    std::shared_ptr<table_info> local_sort = sort_values_table_local(
        std::move(in_table), n_keys, vect_ascending, na_position,
        (parallel && n_total != 0) ? nullptr : dead_keys, parallel);

    if (!parallel) {
        return local_sort;
    } else if (n_total == 0) {
        return local_sort;
    }

    tracing::Event ev_sample("sort sampling", parallel);

    int n_pes, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (bounds == nullptr) {
        bounds = get_parallel_sort_bounds(local_sort, n_keys, vect_ascending,
                                          na_position, n_local, n_total, myrank,
                                          n_pes, parallel);
    } else if (n_keys != 1) {
        // throw error if more than one key since the rest of manual bounds
        // handling only supports one key cases
        throw std::runtime_error(
            "sort_values_table(): passing bounds only supported when there "
            "is a single key.");
    }

    // Now computing to which process it all goes.
    tracing::Event ev_hashes("compute_destinations", parallel);
    std::shared_ptr<uint32_t[]> hashes = std::make_unique<uint32_t[]>(n_local);
    uint32_t rank_id = 0;
    for (int64_t i = 0; i < n_local; i++) {
        size_t shift_key1 = 0, shift_key2 = 0;
        // using 'while' since a partition can be empty which needs to be
        // skipped
        // Go to next destination rank if bound of current destination rank
        // is less than the current key. All destination keys should be less
        // than or equal its bound (k <= bounds[rank_id])
        while (rank_id < uint32_t(n_pes - 1) &&
               KeyComparisonAsPython(n_keys, vect_ascending, bounds->columns,
                                     shift_key2, rank_id, local_sort->columns,
                                     shift_key1, i, na_position)) {
            rank_id++;
        }
        hashes[i] = rank_id;
    }
    bounds.reset();

    // Now shuffle all the data
    mpi_comm_info comm_info(local_sort->columns, hashes, parallel);
    ev_hashes.finalize();
    ev_sample.finalize();
    std::shared_ptr<table_info> collected_table = shuffle_table_kernel(
        std::move(local_sort), hashes, comm_info, parallel);

    // NOTE: local sort doesn't change the number of rows
    // ret_table cannot be used since all output columns may be dead (only
    // length is needed)
    if (out_n_rows != nullptr) {
        *out_n_rows = (int64_t)collected_table->nrows();
    }

    // Final local sorting
    std::shared_ptr<table_info> ret_table = sort_values_table_local(
        std::move(collected_table), n_keys, vect_ascending, na_position,
        dead_keys, parallel);
    return ret_table;
}

table_info* sort_values_table_py_entry(table_info* in_table, int64_t n_keys,
                                       int64_t* vect_ascending,
                                       int64_t* na_position, int64_t* dead_keys,
                                       int64_t* out_n_rows, table_info* bounds,
                                       bool parallel) {
    try {
        std::shared_ptr<table_info> out = sort_values_table(
            std::shared_ptr<table_info>(in_table), n_keys, vect_ascending,
            na_position, dead_keys, out_n_rows,
            std::shared_ptr<table_info>(bounds), parallel);
        return new table_info(*out);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

std::shared_ptr<array_info> sort_values_array_local(
    std::shared_ptr<array_info> in_arr, bool is_parallel, int64_t ascending,
    int64_t na_position) {
    std::vector<std::shared_ptr<array_info>> cols = {in_arr};
    std::shared_ptr<table_info> dummy_table =
        std::make_shared<table_info>(cols);
    int64_t zero = 0;
    // sort_values_table_local will decref all arrays in dummy_table (i.e.
    // in_arr) by 1.
    std::shared_ptr<table_info> sorted_table = sort_values_table_local(
        dummy_table, 1, &ascending, &na_position, &zero, is_parallel);
    std::shared_ptr<array_info> sorted_arr = sorted_table->columns[0];
    sorted_arr->is_locally_sorted = true;
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
    std::vector<std::shared_ptr<table_info>> tables,
    std::vector<uint64_t> n_keys, std::vector<uint64_t> n_local,
    std::vector<uint64_t> n_total) {
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
 * @return std::shared_ptr<table_info> The new table and the
 * destination rank ids for rows in the new table.
 */
std::shared_ptr<table_info> create_send_table_and_hashes_from_rank_to_row_ids(
    std::shared_ptr<table_info> table,
    const std::vector<bodo::vector<int64_t>> rank_to_row_ids,
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
    bodo::vector<int64_t> indices =
        flatten<int64_t>(rank_to_row_ids, total_rows);
    // Create new table using the indices.
    // NOTE: RetrieveTable decrefs all arrays.
    std::shared_ptr<table_info> new_table =
        RetrieveTable(std::move(table), indices, -1);

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
 * @return const std::vector<bodo::vector<uint64_t>> Vector of length n_pes.
 * i'th element is a vector of row ids that must be sent to the i'th rank. Due
 * to the nature of the algorithm, each vector will be sorted.
 */
const std::vector<bodo::vector<int64_t>> compute_destinations_for_interval(
    std::shared_ptr<table_info> table, std::shared_ptr<array_info> bounds_arr,
    int n_pes, bool parallel, bool strict) {
    tracing::Event ev("compute_destinations_for_interval", parallel);
    // TODO XXX Convert to use uint64_t
    std::vector<bodo::vector<int64_t>> rank_to_row_ids(n_pes);
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
    std::shared_ptr<table_info> table,
    const std::shared_ptr<array_info>& bounds_arr, int n_pes, bool parallel) {
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
    const std::vector<std::shared_ptr<table_info>> tables,
    const std::vector<bool> parallel) {
    for (size_t i = 0; i < tables.size(); i++) {
        std::shared_ptr<table_info> table = tables[i];
        for (std::shared_ptr<array_info> arr : table->columns) {
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
 * @return std::shared_ptr<array_info> Array of bounds computed based on the
 * domain columns. It will be of length n_pes-1.
 */
std::shared_ptr<array_info> get_parallel_sort_bounds_for_domain(
    std::vector<std::shared_ptr<table_info>>& tables,
    std::vector<uint64_t>& n_keys, std::vector<uint64_t>& n_local,
    std::vector<uint64_t>& n_total, int myrank, int n_pes, bool parallel) {
    tracing::Event ev("get_parallel_sort_bounds_for_domain", parallel);
    // Validate input values
    validate_inputs_for_get_parallel_sort_bounds_for_domain(tables, n_keys,
                                                            n_local, n_total);
    int64_t ascending = 1, na_pos = 1;
    int mpi_root = 0;

    uint64_t total_cols = std::accumulate(n_keys.begin(), n_keys.end(), 0);
    std::vector<std::shared_ptr<array_info>> sample_cols;
    sample_cols.reserve(total_cols);
    for (size_t table_idx = 0; table_idx < tables.size(); table_idx++) {
        for (uint64_t key_idx = 0; key_idx < n_keys[table_idx]; key_idx++) {
            std::shared_ptr<array_info> col =
                tables[table_idx]->columns[key_idx];
            std::shared_ptr<array_info> sorted_col = col;
            if (key_idx > 0) {
                // If it's not the first key, we need to sort it.
                // In this case, we'll need to free this array at the
                // end.
                // If it's the first key, then we can use it as is (already
                // sorted)
                sorted_col =
                    sort_values_array_local(col, parallel, ascending, na_pos);
            }
            // Create a dummy single column table
            std::shared_ptr<table_info> dummy_table =
                std::make_shared<table_info>();
            dummy_table->columns.push_back(sorted_col);

            // Output of get_samples_from_table_parallel is output of
            // gather_table, which means that on non-root ranks, the table
            // columns are just NULLs.
            std::shared_ptr<table_info> all_samples =
                get_samples_from_table_parallel(
                    std::move(dummy_table), /*n_keys*/ 1, n_pes,
                    n_total[table_idx], n_local[table_idx], parallel);
            std::shared_ptr<array_info> all_samples_arr =
                all_samples->columns[0];
            sample_cols.push_back(all_samples_arr);
        }
    }

    std::shared_ptr<array_info> concatenated_samples = nullptr;
    if (myrank == mpi_root) {
        // Concatenate locally_sorted_cols into a single array_info, sort it and
        // then compute the split bounds on this.
        concatenated_samples = concat_arrays(sample_cols);
    }

    std::shared_ptr<table_info> concatenated_samples_table =
        std::make_shared<table_info>();
    concatenated_samples_table->columns.push_back(concatenated_samples);

    // Create a reference table for broadcast step in
    // compute_bounds_from_samples using the first column of the first
    // table.
    std::shared_ptr<table_info> dummy_table = std::make_shared<table_info>();
    dummy_table->columns.push_back(tables[0]->columns[0]);

    std::shared_ptr<table_info> bounds = compute_bounds_from_samples(
        concatenated_samples_table, dummy_table, 1, &ascending, &na_pos, myrank,
        n_pes, parallel);

    std::shared_ptr<array_info> bounds_arr = bounds->columns[0];
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
 * @return std::shared_ptr<table_info> sorted table.
 */
std::shared_ptr<table_info> sort_table_for_interval_join(
    std::shared_ptr<table_info> table, bool table_already_sorted,
    std::shared_ptr<array_info> bounds_arr, bool is_table_point_side,
    int myrank, int n_pes, bool parallel, bool strict = false) {
    tracing::Event ev("sort_table_for_interval_join", parallel);
    ev.add_attribute("is_table_point_side", is_table_point_side);
    ev.add_attribute("table_len_local", table->nrows());

    // Set the sort order to ascending for all columns.
    int64_t asc[2] = {1, 1};
    // Set the NaNs as last. This is done so intervals are strict
    // subsets in case we encounter Floats. NA values will be ignored.
    int64_t na_pos[2] = {1, 1};

    std::shared_ptr<table_info> local_sort_table;
    if (!table_already_sorted) {
        local_sort_table = sort_values_table_local(
            std::move(table), (is_table_point_side ? 1 : 2), asc, na_pos,
            nullptr, parallel);
    } else {
        local_sort_table = std::move(table);
    }

    if (!parallel) {
        return local_sort_table;
    }

    //
    // 1. Compute destination ranks for each row in both tables
    //

    std::shared_ptr<table_info> table_to_send = nullptr;
    // In this case, we compute the ranks directly, and treat them as the hashes
    // (the modulo step later doesn't end up changing these values since they're
    // all already between 0 and n_pes-1).
    std::shared_ptr<uint32_t[]> table_to_send_hashes =
        std::shared_ptr<uint32_t[]>(nullptr);

    if (is_table_point_side) {
        table_to_send = local_sort_table;
        table_to_send_hashes = compute_destinations_for_point_table(
            std::move(local_sort_table), bounds_arr, n_pes, parallel);
    } else {
        // NOTE: This will skip bad intervals.
        std::vector<bodo::vector<int64_t>> rank_to_row_ids_for_table =
            compute_destinations_for_interval(local_sort_table, bounds_arr,
                                              n_pes, parallel, strict);

        // create_send_table_and_hashes_from_rank_to_row_ids will decref
        // all arrays in local_sort_table.
        table_to_send = create_send_table_and_hashes_from_rank_to_row_ids(
            std::move(local_sort_table), rank_to_row_ids_for_table,
            table_to_send_hashes, parallel);
    }

    //
    // 2. Shuffle data
    //

    mpi_comm_info comm_info_table(table_to_send->columns, table_to_send_hashes,
                                  parallel);
    // NOTE: shuffle_table_kernel decrefs input arrays
    std::shared_ptr<table_info> collected_table =
        shuffle_table_kernel(std::move(table_to_send), table_to_send_hashes,
                             comm_info_table, parallel);
    table_to_send_hashes.reset();

    //
    // 3. Sort the shuffled tables locally
    //

    std::shared_ptr<table_info> ret_table = sort_values_table_local(
        std::move(collected_table), (is_table_point_side ? 1 : 2), asc, na_pos,
        nullptr, parallel);
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
        std::shared_ptr<table_info> sorted_table = sort_table_for_interval_join(
            std::shared_ptr<table_info>(table), false,
            std::shared_ptr<array_info>(bounds_arr), is_table_point_side,
            myrank, n_pes, parallel);
        return new table_info(*sorted_table);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
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
 * @return std::tuple<std::shared_ptr<table_info>, std::shared_ptr<table_info>,
 * std::shared_ptr<array_info>> sorted table 1, sorted table 2, and bounds for
 * the ranks.
 */
std::tuple<std::shared_ptr<table_info>, std::shared_ptr<table_info>,
           std::shared_ptr<array_info>>
sort_both_tables_for_interval_join(std::shared_ptr<table_info> table_1,
                                   std::shared_ptr<table_info> table_2,
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
        std::vector<std::shared_ptr<table_info>>{table_1, table_2},
        std::vector<bool>{table_1_parallel, table_2_parallel});
    ev_dict.finalize();

    std::vector<uint64_t> n_local_vec{table_1->nrows(), table_2->nrows()};
    std::vector<uint64_t> n_total_vec{n_local_vec[0], n_local_vec[1]};
    if (table_1_parallel) {
        CHECK_MPI(
            MPI_Allreduce(n_local_vec.data(), n_total_vec.data(), 1,
                          MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD),
            "sort_both_tables_for_interval_join: MPI error on "
            "MPI_Allreduce[table_1_parallel]:");
    }
    if (table_2_parallel) {
        CHECK_MPI(
            MPI_Allreduce(n_local_vec.data() + 1, n_total_vec.data() + 1, 1,
                          MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD),
            "sort_both_tables_for_interval_join: MPI error on "
            "MPI_Allreduce[table_2_parallel]:");
    }

    //
    // 1. Sort the tables locally
    //

    // Set the sort order to ascending for all columns.
    int64_t asc[2] = {1, 1};
    // Set the NaNs as last. This is done so intervals are strict
    // subsets in case we encounter Floats. NA values will be ignored.
    int64_t na_pos[2] = {1, 1};

    // sort_values_table_local will decref all arrays in table_1
    // and table_2.
    std::shared_ptr<table_info> local_sort_table_1 = sort_values_table_local(
        std::move(table_1), (is_table_1_point_side ? 1 : 2), asc, na_pos,
        nullptr, table_1_parallel);
    std::shared_ptr<table_info> local_sort_table_2 = sort_values_table_local(
        std::move(table_2), 2, asc, na_pos, nullptr, table_2_parallel);

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

    std::vector<std::shared_ptr<table_info>> tables_vec{local_sort_table_1,
                                                        local_sort_table_2};
    std::vector<uint64_t> n_keys_vec{
        static_cast<uint64_t>(is_table_1_point_side ? 1 : 2), 2};
    std::shared_ptr<array_info> bounds_arr =
        get_parallel_sort_bounds_for_domain(tables_vec, n_keys_vec, n_local_vec,
                                            n_total_vec, myrank, n_pes,
                                            parallel);

    //
    // 3. Compute destination ranks for each row in both tables, shuffle the
    // data and then do local sort on the shuffled tables. Note that "bad"
    // intervals (i.e. start > end) will be filtered out.
    //

    std::shared_ptr<table_info> ret_table_1 = sort_table_for_interval_join(
        std::move(local_sort_table_1), true, bounds_arr, is_table_1_point_side,
        myrank, n_pes, parallel, strict);

    // All arrays in local_sort_table_2 will be decref-ed
    std::shared_ptr<table_info> ret_table_2 = sort_table_for_interval_join(
        std::move(local_sort_table_2), true, bounds_arr, false, myrank, n_pes,
        parallel, strict);

    return {ret_table_1, ret_table_2, bounds_arr};
}

std::pair<std::shared_ptr<table_info>, std::shared_ptr<table_info>>
sort_tables_for_point_in_interval_join(
    std::shared_ptr<table_info> table_point,
    std::shared_ptr<table_info> table_interval, bool table_point_parallel,
    bool table_interval_parallel, bool strict) {
    std::shared_ptr<table_info> sorted_point, sorted_interval;
    std::shared_ptr<array_info> bounds_arr;

    std::tie(sorted_point, sorted_interval, bounds_arr) =
        sort_both_tables_for_interval_join(
            std::move(table_point), std::move(table_interval),
            table_point_parallel, table_interval_parallel, true, strict);

    return {sorted_point, sorted_interval};
}

std::tuple<std::shared_ptr<table_info>, std::shared_ptr<table_info>,
           std::shared_ptr<array_info>>
sort_tables_for_interval_overlap_join(std::shared_ptr<table_info> table_1,
                                      std::shared_ptr<table_info> table_2,
                                      bool table_1_parallel,
                                      bool table_2_parallel) {
    return sort_both_tables_for_interval_join(
        std::move(table_1), std::move(table_2), table_1_parallel,
        table_2_parallel, false, true);
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
    std::vector<std::shared_ptr<array_info>>* key_arrs;
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
static std::shared_ptr<table_info> drop_duplicates_keys_inner(
    std::shared_ptr<table_info> in_table, int64_t num_keys, bool dropna = true,
    bool is_parallel = true) {
    size_t n_rows = (size_t)in_table->nrows();
    std::vector<std::shared_ptr<array_info>> key_arrs(
        in_table->columns.begin(), in_table->columns.begin() + num_keys);
    uint32_t seed = SEED_HASH_CONTAINER;
    std::shared_ptr<uint32_t[]> hashes = hash_keys(key_arrs, seed, is_parallel);
    HashDropDuplicates hash_fct{hashes};
    KeyEqualDropDuplicates equal_fct{.key_arrs = &key_arrs,
                                     .num_keys = num_keys};
    bodo::unord_map_container<size_t, size_t, HashDropDuplicates,
                              KeyEqualDropDuplicates>
        entSet({}, hash_fct, equal_fct);
    //
    bodo::vector<int64_t> ListRow;
    uint64_t next_ent = 0;
    bool has_nulls = dropna && does_keys_have_nulls(key_arrs);
    auto is_ok = [&](size_t i_row) -> bool {
        return !has_nulls || !does_row_has_nulls(key_arrs, i_row);
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
    bodo::vector<int64_t> ListIdx;
    for (auto& eRow : ListRow) {
        if (eRow != -1) {
            ListIdx.push_back(eRow);
        }
    }
    // Now building the out_arrs array. We select only the first num_keys.
    std::shared_ptr<table_info> ret_table =
        RetrieveTable(std::move(in_table), ListIdx, num_keys);
    hashes.reset();
    return ret_table;
}
/**
 * @brief Helper function for drop_duplicates_table_inner. Returns a vector of
 * indices of rows to keep.
 * @param in_table Table to drop duplicates from.
 * @param num_keys Number of keys to consider for the drop duplicates operation.
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
 * @return A vector of indices of rows to keep.
 */
bodo::vector<int64_t> drop_duplicates_table_helper(
    std::shared_ptr<table_info> in_table, int64_t num_keys, int64_t keep,
    int step, bool is_parallel, bool dropna, bool drop_duplicates_dict,
    std::shared_ptr<uint32_t[]> hashes, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    size_t n_rows = (size_t)in_table->nrows();
    std::vector<std::shared_ptr<array_info>> key_arrs(num_keys);
    for (size_t iKey = 0; iKey < size_t(num_keys); iKey++) {
        std::shared_ptr<array_info> key = in_table->columns[iKey];
        // If we are dropping duplicates dictionary encoding assumes
        // that the dictionary values are unique.
        if (key->arr_type == bodo_array_type::DICT) {
            if (drop_duplicates_dict) {
                // Should we ensure that the dictionary values are unique?
                // If this is just a local drop duplicates then we don't
                // need to do this and can just do a best effort approach.
                drop_duplicates_local_dictionary(key, false, pool, mm);
            }
            key = key->child_arrays[1];
        }
        key_arrs[iKey] = key;
    }
    uint32_t seed = SEED_HASH_CONTAINER;
    if (!hashes) {
        // TODO: add pool/mm support in hash_keys
        // NOTE: pool/mm were added for streaming groupby mode
        // (https://bodo.atlassian.net/browse/BSE-1139) for dropping dict
        // duplicates, where the number of dict elements is small (limited by
        // batch size). Therefore, tracking memory of hashes isn't necessary.
        hashes = hash_keys(key_arrs, seed, is_parallel,
                           /*global_dict_needed=*/false);
    }
    HashDropDuplicates hash_fct{hashes};
    KeyEqualDropDuplicates equal_fct{.key_arrs = &key_arrs,
                                     .num_keys = num_keys};
    bodo::unord_map_container<size_t, size_t, HashDropDuplicates,
                              KeyEqualDropDuplicates>
        entSet({}, hash_fct, equal_fct, pool);
    // The loop over the short table.
    // entries are stored one by one and all of them are put even if identical
    // in value. The one exception is if we are dropping NA values
    //
    // In the first case we keep only one entry.
    bool has_nulls = dropna && does_keys_have_nulls(key_arrs);
    auto is_ok = [&](size_t i_row) -> bool {
        return !has_nulls || !does_row_has_nulls(key_arrs, i_row);
    };
    auto RetrieveListIdx1 = [&]() -> bodo::vector<int64_t> {
        bodo::vector<int64_t> ListRow(pool);
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
        bodo::vector<int64_t> ListIdx(pool);
        for (auto& eRow : ListRow) {
            if (eRow != -1) {
                ListIdx.push_back(eRow);
            }
        }
        return ListIdx;
    };
    // In this case we store the pairs of values, the first and the last.
    // This allows to reach conclusions in all possible cases.
    auto RetrieveListIdx2 = [&]() -> bodo::vector<int64_t> {
        bodo::vector<std::pair<int64_t, int64_t>> ListRowPair(pool);
        size_t next_ent = 0;
        for (size_t i_row = 0; i_row < n_rows; i_row++) {
            // don't add if entry is NA and dropna=true
            if (is_ok(i_row)) {
                size_t& group = entSet[i_row];
                if (group == 0) {
                    next_ent++;
                    group = next_ent;
                    ListRowPair.emplace_back(i_row, -1);
                } else {
                    size_t pos = group - 1;
                    ListRowPair[pos].second = i_row;
                }
            }
        }
        bodo::vector<int64_t> ListIdx(pool);
        for (auto& eRowPair : ListRowPair) {
            if (eRowPair.first != -1) {
                ListIdx.push_back(eRowPair.first);
            }
            if (eRowPair.second != -1) {
                ListIdx.push_back(eRowPair.second);
            }
        }
        return ListIdx;
    };
    bodo::vector<int64_t> ListIdx(pool);
    if (step == 1 || keep == 0 || keep == 1) {
        ListIdx = RetrieveListIdx1();
    } else {
        ListIdx = RetrieveListIdx2();
    }
    return ListIdx;
};

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
std::shared_ptr<table_info> drop_duplicates_table_inner(
    std::shared_ptr<table_info> in_table, int64_t num_keys, int64_t keep,
    int step, bool is_parallel, bool dropna, bool drop_duplicates_dict,
    std::shared_ptr<uint32_t[]> hashes, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    tracing::Event ev("drop_duplicates_table_inner", is_parallel);
    ev.add_attribute("table_nrows_before",
                     static_cast<size_t>(in_table->nrows()));
    if (ev.is_tracing()) {
        size_t global_table_nbytes = table_global_memory_size(in_table);
        ev.add_attribute("g_table_nbytes", global_table_nbytes);
    }
    bodo::vector<int64_t> ListIdx = drop_duplicates_table_helper(
        in_table, num_keys, keep, step, is_parallel, dropna,
        drop_duplicates_dict, hashes, pool, mm);
    // Now building the out_arrs array.
    std::shared_ptr<table_info> ret_table = RetrieveTable(
        std::move(in_table), ListIdx, -1, false, pool, std::move(mm));
    //
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
std::shared_ptr<table_info> drop_duplicates_keys(
    std::shared_ptr<table_info> in_table, int64_t num_keys, bool is_parallel,
    bool dropna) {
    // serial case
    if (!is_parallel) {
        return drop_duplicates_keys_inner(in_table, num_keys, dropna,
                                          is_parallel);
    }
    // parallel case
    // pre reduction of duplicates
    std::shared_ptr<table_info> red_table =
        drop_duplicates_keys_inner(in_table, num_keys, dropna, is_parallel);
    // shuffling of values
    std::shared_ptr<table_info> shuf_table =
        shuffle_table(std::move(red_table), num_keys, is_parallel, 0);
    // reduction after shuffling
    int keep = 0;
    // Set dropna=False for drop_duplicates_table_inner because
    // drop_duplicates_keys_inner should have already removed any NA
    // values
    std::shared_ptr<table_info> ret_table =
        drop_duplicates_table_inner(shuf_table, num_keys, keep, 1, is_parallel,
                                    false, /*drop_duplicates_dict=*/true);
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
std::shared_ptr<table_info> drop_duplicates_table(
    std::shared_ptr<table_info> in_table, bool is_parallel, int64_t num_keys,
    int64_t keep, bool dropna, bool drop_local_first,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    // serial case
    if (!is_parallel) {
        return drop_duplicates_table_inner(
            in_table, num_keys, keep, 1, is_parallel, dropna,
            /*drop_duplicates_dict=*/true, std::shared_ptr<uint32_t[]>(nullptr),
            pool, std::move(mm));
    }
    // parallel case
    // pre reduction of duplicates
    std::shared_ptr<table_info> red_table;
    if (drop_local_first) {
        red_table = drop_duplicates_table_inner(
            in_table, num_keys, keep, 2, is_parallel, dropna,
            /*drop_duplicates_dict=*/false,
            std::shared_ptr<uint32_t[]>(nullptr), pool, mm);
    } else {
        red_table = in_table;
    }
    // TODO: add pool/mm support in shuffle_table
    // NOTE: pool/mm were added for streaming groupby mode
    // (https://bodo.atlassian.net/browse/BSE-1139) for dropping dict
    // duplicates, which doesn't run in parallel.
    std::shared_ptr<table_info> shuf_table =
        shuffle_table(red_table, num_keys, is_parallel);
    // no need to decref since shuffle_table() steals a reference
    if (drop_local_first) {
        red_table.reset();
    }
    // reduction after shuffling
    // We don't drop NA values again because the first
    // drop_duplicates_table_inner should have already handled this
    if (drop_local_first) {
        dropna = false;
    }
    std::shared_ptr<table_info> ret_table = drop_duplicates_table_inner(
        shuf_table, num_keys, keep, 1, is_parallel, dropna,
        /*drop_duplicates_dict=*/true, std::shared_ptr<uint32_t[]>(nullptr),
        pool, std::move(mm));
    // returning table
    return ret_table;
}

table_info* drop_duplicates_table_py_entry(table_info* in_table,
                                           bool is_parallel, int64_t num_keys,
                                           int64_t keep, bool dropna,
                                           bool drop_local_first) {
    try {
        std::shared_ptr<table_info> out_table = drop_duplicates_table(
            std::shared_ptr<table_info>(in_table), is_parallel, num_keys, keep,
            dropna, drop_local_first);
        return new table_info(*out_table);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

// Note: union_tables steals a reference and generates a new output.
std::shared_ptr<table_info> union_tables_inner(
    std::vector<std::shared_ptr<table_info>> in_table, int64_t num_tables,
    bool drop_duplicates, bool is_parallel) {
    tracing::Event ev("union_tables", is_parallel);
    // Drop duplicates locally first. This won't do a shuffle yet because we
    // only want to do 1 shuffle.
    std::vector<std::shared_ptr<table_info>> locally_processed_table(
        num_tables);

    // All tables have the same number of columns and all columns are keys
    int64_t num_keys = in_table[0]->ncols();

    if (drop_duplicates) {
        tracing::Event ev_drop_duplicates("union_tables_drop_duplicates_local",
                                          is_parallel);
        for (int i = 0; i < num_tables; i++) {
            locally_processed_table[i] = drop_duplicates_table_inner(
                in_table[i], num_keys, 0, 1, false, false, false);
            in_table[i]->columns.clear();
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
    std::shared_ptr<table_info> base_table = locally_processed_table[0];
    // Optimize unify dictionaries for two arrays (common case)
    if (num_tables == 2) {
        for (int j = 0; j < num_keys; j++) {
            if (base_table->columns[j]->arr_type == bodo_array_type::DICT) {
                // Unify all of the dictionaries.
                std::shared_ptr<array_info> arr1 =
                    locally_processed_table[0]->columns[j];
                std::shared_ptr<array_info> arr2 =
                    locally_processed_table[1]->columns[j];
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
                std::vector<std::shared_ptr<array_info>> dict_arrs(num_tables);
                std::vector<bool> is_parallels(num_tables);
                for (int i = 0; i < num_tables; i++) {
                    // Ensure we have global/unique data for the dictionary.
                    std::shared_ptr<array_info> arr =
                        locally_processed_table[i]->columns[j];
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
    std::vector<std::shared_ptr<table_info>> shuffled_tables(num_tables);
    if (is_parallel && drop_duplicates) {
        // Shuffle the tables. We don't need to shuffle the tables
        // if we aren't dropping duplicates.
        for (int i = 0; i < num_tables; i++) {
            shuffled_tables[i] =
                shuffle_table(locally_processed_table[i], num_keys, true);
            clear_all_cols_if_last_table_ref(locally_processed_table[i]);
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
    std::shared_ptr<table_info> concatenated_table =
        concat_tables(shuffled_tables);
    shuffled_tables.clear();
    ev_concat.finalize();
    std::shared_ptr<table_info> out_table;
    if (drop_duplicates) {
        tracing::Event ev_drop_duplicates("union_tables_drop_duplicates_local",
                                          is_parallel);
        // Drop any duplicates in the concatenated tables
        out_table = drop_duplicates_table_inner(concatenated_table, num_keys, 0,
                                                1, false, false, false);
        ev_drop_duplicates.finalize();
    } else {
        out_table = concatenated_table;
    }
    ev.finalize();
    return out_table;
}

table_info* union_tables(table_info** in_tables, int64_t num_tables,
                         bool drop_duplicates, bool is_parallel) {
    try {
        std::vector<std::shared_ptr<table_info>> in_tables_vec;
        for (int64_t i = 0; i < num_tables; i++) {
            in_tables_vec.push_back(std::shared_ptr<table_info>(in_tables[i]));
        }
        std::shared_ptr<table_info> out_table = union_tables_inner(
            in_tables_vec, num_tables, drop_duplicates, is_parallel);
        return new table_info(*out_table);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

table_info* concat_tables_py_entry(table_info** in_tables, int64_t num_tables) {
    try {
        std::vector<std::shared_ptr<table_info>> in_tables_vec;
        for (int64_t i = 0; i < num_tables; i++) {
            in_tables_vec.push_back(std::shared_ptr<table_info>(in_tables[i]));
        }
        std::shared_ptr<table_info> out_table = concat_tables(in_tables_vec);
        return new table_info(*out_table);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

//
//   SAMPLE
//

/**
 * Helper function that samples integers without replacement, using an unordered
 * set to track previous samples. Best suited for small `n_samp`
 * @param n_samp Number of random samples
 * @param n_total Sample integers from range [0, n_total)
 * @param gen std::mt19937 generator to sample from
 * @param out Output vector that samples are written to
 */
void sample_ints_unordset(int64_t n_samp, int64_t n_total, std::mt19937& gen,
                          bodo::vector<int64_t>& out) {
    bodo::unord_set_container<int64_t> UnordsetIdxChosen;

    int64_t i_samp = 0;
    std::uniform_int_distribution<int64_t> rng(0, n_total - 1);
    while (i_samp < n_samp) {
        int64_t idx_rand = rng(gen);
        if (UnordsetIdxChosen.count(idx_rand) == 0) {
            UnordsetIdxChosen.insert(idx_rand);
            out[i_samp] = idx_rand;
            i_samp++;
        }
    }
}

/**
 * Helper function that samples integers without replacement, using a bitmask
 * to track previous samples. Best suited for medium-small `n_samp`
 * @param n_samp Number of random samples
 * @param n_total Sample integers from range [0, n_total)
 * @param gen std::mt19937 generator to sample from
 * @param out Output vector that samples are written to
 */
void sample_ints_bitmask(int64_t n_samp, int64_t n_total, std::mt19937& gen,
                         bodo::vector<int64_t>& out) {
    bodo::vector<bool> BitmaskIdxChosen(n_total);

    int64_t i_samp = 0;
    std::uniform_int_distribution<int64_t> rng(0, n_total - 1);
    while (i_samp < n_samp) {
        int64_t idx_rand = rng(gen);
        if (!BitmaskIdxChosen[idx_rand]) {
            BitmaskIdxChosen[idx_rand] = true;
            out[i_samp] = idx_rand;
            i_samp++;
        }
    }
}

/**
 * Helper function that samples integers without replacement, by permuting the
 * array of all integers. Best suited for large `n_samp`
 * @param n_samp Number of random samples
 * @param n_total Sample integers from range [0, n_total)
 * @param gen std::mt19937 generator to sample from
 * @param out Output vector that samples are written to
 */
void sample_ints_permute(int64_t n_samp, int64_t n_total, std::mt19937& gen,
                         bodo::vector<int64_t>& out) {
    bodo::vector<int64_t> AllIndices(n_total);
    for (int64_t i_total = 0; i_total < n_total; i_total++) {
        AllIndices[i_total] = i_total;
    }

    for (int64_t i_samp = 0; i_samp < n_samp; i_samp++) {
        std::uniform_int_distribution<int64_t> rng(i_samp, n_total - 1);
        int64_t idx_swap = rng(gen);
        int64_t idx_rand = AllIndices[idx_swap];
        AllIndices[idx_swap] = AllIndices[i_samp];
        out[i_samp] = idx_rand;
    }
}

/**
 * Inner implementation of parallel sample_table
 */
std::shared_ptr<table_info> sample_table_inner_parallel(
    std::shared_ptr<table_info> in_table, int64_t n, double frac, bool replace,
    int64_t random_state) {
    int64_t n_local = in_table->nrows();
    int n_pes = 0;
    int myrank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // Gather how many rows are on each rank
    std::vector<int64_t> ListSizes(n_pes);
    CHECK_MPI(MPI_Allgather(&n_local, 1, MPI_LONG_LONG, ListSizes.data(), 1,
                            MPI_LONG_LONG, MPI_COMM_WORLD),
              "sample_table_inner_parallel: MPI error on MPI_Allgather:");

    // Total number of rows across all ranks
    int64_t n_total = std::accumulate(ListSizes.begin(), ListSizes.end(), 0);

    // Total number of samples across all ranks
    int64_t n_samp;
    if (frac < 0) {
        n_samp = std::max(static_cast<int64_t>(0), std::min(n_total, n));
    } else {
        n_samp = round(n_total * std::max(0., std::min(1., frac)));
    }

    // Compute start and end of each rank's input portion (inclusive)
    std::vector<int64_t> ListStarts(n_pes);
    std::vector<int64_t> ListEnds(n_pes);

    int64_t sum_siz = 0;
    for (int iProc = 0; iProc < n_pes; iProc++) {
        ListStarts[iProc] = sum_siz;
        sum_siz += ListSizes[iProc];
        ListEnds[iProc] = sum_siz - 1;
    }

    // Compute which rank a random sample belongs to
    // Sample random points deterministically across the whole array.
    // Each RNG has the same seed on all ranks, so this is safe and requires
    // no synchronization across ranks.
    bodo::vector<int64_t> ListIdxChosen;
    std::mt19937 gen(random_state);

    if (replace) {
        // Sampling with replacement: Simply output random points. Each rank
        // runs this code separately, selecting the samples that belong to it.

        int64_t mystart = ListStarts[myrank];
        int64_t myend = ListEnds[myrank];

        std::uniform_int_distribution<int64_t> rng(0, n_total = 1);
        for (int64_t i_samp = 0; i_samp < n_samp; i_samp++) {
            int64_t idx_rand = rng(gen);
            if (mystart <= idx_rand && idx_rand <= myend) {
                ListIdxChosen.emplace_back(idx_rand - mystart);
            }
        }
    } else {
        // Sampling without replacement: Sample random points using the best
        // method for the current sample size. This code runs on rank 0 only
        // to reduce peak memory usage. Afterwards, we shuffle each sample to
        // its destination rank.
        //
        // For small n_samp, rejection sample with a bitmask/unordered set to
        // track already-selected indices. For large n_samp, permute the array
        // of all indices.
        //
        // Threshold 1 (unordered set vs. bitmask): Bitmasks require
        // `n_total / 8` bytes, and unordered sets of integers require
        // `n_samp * 8 / load_factor` bytes, where absl::flat_hash_set
        // has load factor between 7/8 and 7/16 (average 21/32)
        // (Source: https://abseil.io/docs/cpp/guides/container).
        // So on average, unordered sets are less memory than bitmasks
        // when n_samp < 21/2048 * n_total, approximately n_total / 96
        //
        // Threshold 2 (rejection sampling vs. permutation): Rejection
        // sampling becomes catastrophically slow at large n_samp: for
        // the last sample on average `n_total / (n_total - n_samp)`
        // attempts are needed. However, an array of all indices costs
        // more memory than a bitmask. Setting this threshold to the
        // halfway point gives similar performance on large arrays.

        bodo::vector<int64_t> ListIdxSampled;
        bodo::vector<int64_t> ListIdxSampledExport;

        // NOTE: MPI requires sendcounts and displs to be passed as ints, so
        // these could overflow
        std::vector<int> ListCounts;
        std::vector<int> ListDisps;

        if (myrank == 0) {
            ListIdxSampled.resize(n_samp);
            ListIdxSampledExport.resize(n_samp);
            ListCounts.resize(n_pes);
            ListDisps.resize(n_pes);

            if (n_samp * 96L <= n_total) {
                // If n_samp is very small, keep an unordered set of which
                // indices to include.
                sample_ints_unordset(n_samp, n_total, gen, ListIdxSampled);
            } else if (n_samp * 2L <= n_total) {
                // If n_samp is medium-small, keep a bitmask of which
                // indices to include
                sample_ints_bitmask(n_samp, n_total, gen, ListIdxSampled);
            } else {
                // If n_samp is large, generate a random permutation of the
                // array of all indices by swapping with subsequent values
                sample_ints_permute(n_samp, n_total, gen, ListIdxSampled);
            }

            // Compute destination rank of each sample
            bodo::vector<int> ListByProcessor(n_samp);
            for (int64_t i_samp = 0; i_samp < n_samp; i_samp++) {
                int64_t idx_rand = ListIdxSampled[i_samp];
                int i_proc =
                    std::distance(ListEnds.begin(),
                                  std::ranges::lower_bound(ListEnds, idx_rand));
                ListIdxSampled[i_samp] -= ListStarts[i_proc];
                ListByProcessor[i_samp] = i_proc;
                ListCounts[i_proc]++;
            }

            // Compute sendbufs and displacements for MPI_Scatterv
            ListDisps[0] = 0;
            for (int i_pes = 0; i_pes < n_pes - 1; i_pes++) {
                ListDisps[i_pes + 1] = ListDisps[i_pes] + ListCounts[i_pes];
            }

            std::vector<int> ListShifts(n_pes);
            for (int64_t i_samp = 0; i_samp < n_samp; i_samp++) {
                int i_proc = ListByProcessor[i_samp];
                int i_dest = ListDisps[i_proc] + ListShifts[i_proc];
                ListIdxSampledExport[i_dest] = ListIdxSampled[i_samp];
                ListShifts[i_proc]++;
            }
        }

        ListIdxSampled.clear();

        // Scatter sampled indices to each node
        int n_samp_out;
        CHECK_MPI(MPI_Scatter(ListCounts.data(), 1, MPI_INT, &n_samp_out, 1,
                              MPI_INT, 0, MPI_COMM_WORLD),
                  "sample_table_inner_parallel: MPI error on MPI_Scatter:");

        ListIdxChosen.resize(n_samp_out);
        CHECK_MPI(
            MPI_Scatterv(ListIdxSampledExport.data(), ListCounts.data(),
                         ListDisps.data(), MPI_LONG_LONG, ListIdxChosen.data(),
                         n_samp_out, MPI_LONG_LONG, 0, MPI_COMM_WORLD),
            "sample_table_inner_parallel: MPI error on MPI_Scatterv:");

        ListIdxSampledExport.clear();
        ListCounts.clear();
        ListDisps.clear();
    }

    // Retrieve the output table from sampled indices
    std::shared_ptr<table_info> tab_out =
        RetrieveTable(std::move(in_table), ListIdxChosen, -1);
    return tab_out;
}

/**
 * Inner implementation of sequential sample_table
 */
std::shared_ptr<table_info> sample_table_inner_sequential(
    std::shared_ptr<table_info> in_table, int64_t n, double frac, bool replace,
    int64_t random_state) {
    // Total number of rows
    int64_t n_total = in_table->nrows();

    // Total number of samples
    int64_t n_samp;
    if (frac < 0) {
        n_samp = std::max(static_cast<int64_t>(0), std::min(n_total, n));
    } else {
        n_samp = round(n_total * std::max(0., std::min(1., frac)));
    }

    // Sample random points deterministically across the whole array.
    // Each RNG has the same seed on all ranks, so this is safe and requires
    // no synchronization across ranks.
    bodo::vector<int64_t> ListIdxChosen(n_samp);
    std::mt19937 gen(random_state);

    if (replace) {
        // Sampling with replacement: Simply output random points.
        std::uniform_int_distribution<int64_t> rng(0, n_total - 1);
        for (int64_t i_samp = 0; i_samp < n_samp; i_samp++) {
            int64_t idx_rand = rng(gen);
            ListIdxChosen[i_samp] = idx_rand;
        }
    } else {
        // Sampling without replacement: For small n_samp, rejection sample
        // with a bitmask/unordered set to track included idxs. For large
        // n_samp, permute the array of all indices. For justification of
        // thresholds, see comments in sample_table_inner_parallel

        if (n_samp * 96L <= n_total) {
            // If n_samp is very small, keep an unordered set of which
            // indices to include.
            sample_ints_unordset(n_samp, n_total, gen, ListIdxChosen);
        } else if (n_samp * 2L <= n_total) {
            // If n_samp is medium-small, keep a bitmask of which
            // indices to include
            sample_ints_bitmask(n_samp, n_total, gen, ListIdxChosen);
        } else {
            // If n_samp is large, generate a random permutation of the
            // array of all indices by swapping with subsequent values
            sample_ints_permute(n_samp, n_total, gen, ListIdxChosen);
        }
    }

    // Retrieve the output table from sampled indices
    std::shared_ptr<table_info> tab_out =
        RetrieveTable(std::move(in_table), ListIdxChosen, -1);
    return tab_out;
}

table_info* sample_table_py_entry(table_info* in_table, int64_t n, double frac,
                                  bool replace, int64_t random_state,
                                  bool parallel) {
    try {
        if (parallel) {
            std::shared_ptr<table_info> out_table = sample_table_inner_parallel(
                std::shared_ptr<table_info>(in_table), n, frac, replace,
                random_state);
            return new table_info(*out_table);
        } else {
            std::shared_ptr<table_info> out_table =
                sample_table_inner_sequential(
                    std::shared_ptr<table_info>(in_table), n, frac, replace,
                    random_state);
            return new table_info(*out_table);
        }
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

/**
 * Search for pattern in each input element using
 * boost::xpressive::regex_search(in, pattern)
 * @tparam do_full_match if true, check that the entire input string is matched
 * by the pattern.
 * @param[in] in_arr input array of string  elements
 * @param[in] case_sensitive bool whether pattern is case sensitive or not
 * @param[in] match_beginning bool whether the pattern starts at the beginning
 * @param[in] regex regular expression pattern
 * @param[out] out_arr output array of bools specifying whether the pattern
 *                      matched for corresponding in_arr element
 */
template <bool do_full_match>
void get_search_regex(std::shared_ptr<array_info> in_arr,
                      const bool case_sensitive, const bool match_beginning,
                      char const* const pat,
                      std::shared_ptr<array_info> out_arr) {
    assert(out_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL);
    // get_search_regex can be called a different number of times per ank.
    tracing::Event ev("get_search_regex", false);
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
        offset_t const* const data2 = reinterpret_cast<offset_t*>(
            in_arr->data2<bodo_array_type::STRING>());
        char const* const data1 = in_arr->data1<bodo_array_type::STRING>();
        for (size_t iRow = 0; iRow < nRow; iRow++) {
            bool bit = in_arr->get_null_bit<bodo_array_type::STRING>(iRow);
            if (bit) {
                const offset_t start_pos = data2[iRow];
                const offset_t end_pos = data2[iRow + 1];
                // regex_match is true if the entire string matches the pattern
                if (do_full_match && boost::xpressive::regex_match(
                                         data1 + start_pos, data1 + end_pos, m,
                                         pattern, match_flag)) {
                    SetBitTo((uint8_t*)out_arr
                                 ->data1<bodo_array_type::NULLABLE_INT_BOOL>(),
                             iRow, true);
                    num_match++;
                } else if (!do_full_match &&
                           boost::xpressive::regex_search(
                               data1 + start_pos, data1 + end_pos, m, pattern,
                               match_flag)) {
                    SetBitTo((uint8_t*)out_arr
                                 ->data1<bodo_array_type::NULLABLE_INT_BOOL>(),
                             iRow, true);
                    num_match++;
                } else {
                    SetBitTo((uint8_t*)out_arr
                                 ->data1<bodo_array_type::NULLABLE_INT_BOOL>(),
                             iRow, false);
                }
            }
            out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(iRow,
                                                                      bit);
        }
        ev.add_attribute("local_num_match", num_match);

    } else if (in_arr->arr_type == bodo_array_type::DICT) {
        // For dictionary-encoded string arrays, we optimize
        // by first doing the computation on the dictionary (info1)
        // (which is presumably much smaller), and then
        // building the output boolean array by indexing into this
        // result.

        std::shared_ptr<array_info> dict_arr = in_arr->child_arrays[0];

        // Allocate boolean array to store output of get_search_regex
        // on the dictionary (dict_arr)
        std::shared_ptr<array_info> dict_arr_out =
            alloc_nullable_array(dict_arr->length, Bodo_CTypes::_BOOL, 0);

        // Compute recursively on the dictionary
        // (dict_arr; which is just a string array).
        get_search_regex<do_full_match>(dict_arr, case_sensitive,
                                        match_beginning, pat, dict_arr_out);

        std::shared_ptr<array_info> indices_arr = in_arr->child_arrays[1];

        // Iterate over the indices, and assign values to the output
        // boolean array from dict_arr_out.
        for (size_t iRow = 0; iRow < nRow; iRow++) {
            bool bit = in_arr->get_null_bit<bodo_array_type::DICT>(iRow);
            if (bit) {
                // Get index in the dictionary
                int32_t iiRow =
                    indices_arr->at<dict_indices_t,
                                    bodo_array_type::NULLABLE_INT_BOOL>(iRow);
                // Get output from dict_arr_out for this dict value
                bool value =
                    GetBit((uint8_t*)dict_arr_out
                               ->data1<bodo_array_type::NULLABLE_INT_BOOL>(),
                           iiRow);
                SetBitTo((uint8_t*)out_arr
                             ->data1<bodo_array_type::NULLABLE_INT_BOOL>(),
                         iRow, value);
                if (value) {
                    num_match++;
                }
            }
            out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(iRow,
                                                                      bit);
        }
        ev.add_attribute("local_num_match", num_match);
    } else {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "array in_arr type should be string");
    }
}

void get_search_regex_py_entry(array_info* in_arr, const bool case_sensitive,
                               const bool match_beginning,
                               char const* const pat, array_info* out_arr,
                               bool do_full_match) {
    try {
        if (do_full_match) {
            get_search_regex<true>(std::shared_ptr<array_info>(in_arr),
                                   case_sensitive, match_beginning, pat,
                                   std::shared_ptr<array_info>(out_arr));
        } else {
            get_search_regex<false>(std::shared_ptr<array_info>(in_arr),
                                    case_sensitive, match_beginning, pat,
                                    std::shared_ptr<array_info>(out_arr));
        }
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return;
    }
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
 * @return std::shared_ptr<array_info> An output array of replaced strings the
 * same length as end_idx - start_idx. Elements not included in the slice are
 * not included in the output.
 */
std::shared_ptr<array_info> get_replace_regex_slice(
    const std::shared_ptr<array_info>& in_arr, char const* const pat,
    char const* replacement, size_t start_idx, size_t end_idx) {
    assert(in_arr->arr_type == bodo_array_type::STRING);
    // See:
    // https://www.boost.org/doc/libs/1_76_0/boost/xpressive/regex_constants.hpp
    boost::xpressive::regex_constants::syntax_option_type flag =
        boost::xpressive::regex_constants::ECMAScript;  // default value

    const boost::xpressive::cregex pattern =
        boost::xpressive::cregex::compile(pat, flag);

    offset_t const* const in_data2 =
        reinterpret_cast<offset_t*>(in_arr->data2<bodo_array_type::STRING>());
    char const* const in_data1 = in_arr->data1<bodo_array_type::STRING>();

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
        std::vector<char> buffer(buffer_size);
        // To construct the output array we need to know how many characters
        // are in the output. As a result we compute the result twice, once
        // to get the size of the output and once with the result.
        // TODO: Determine how to dynamically resize the output array since
        // the compute cost seems to dominate.
        for (size_t iRow = start_idx; iRow < end_idx; iRow++) {
            bool bit = in_arr->get_null_bit<bodo_array_type::STRING>(iRow);
            if (bit) {
                const offset_t start_pos = in_data2[iRow];
                const offset_t end_pos = in_data2[iRow + 1];
                while (((end_pos - start_pos) * repLen) >= buffer_size) {
                    buffer_size *= 2;
                    buffer.resize(buffer_size);
                }
                char* out_buffer = boost::xpressive::regex_replace(
                    buffer.data(), in_data1 + start_pos, in_data1 + end_pos,
                    pattern, replacement);
                num_chars += out_buffer - buffer.data();
            }
        }
    }
    // Allocate the output array. If this was a used as the dictionary
    // in a dict-encoded array, then being "global" is preserved, but
    // uniqueness and sorting are lost.
    // TODO(njriasan): Can this ever be Binary?
    std::shared_ptr<array_info> out_arr =
        alloc_string_array(Bodo_CTypes::STRING, out_arr_len, num_chars, -1, 0,
                           in_arr->is_globally_replicated);
    offset_t* const out_data2 =
        reinterpret_cast<offset_t*>(out_arr->data2<bodo_array_type::STRING>());
    // Initialize the first offset to 0
    out_data2[0] = 0;
    char* const out_data1 = out_arr->data1<bodo_array_type::STRING>();
    for (size_t outRow = 0, iRow = start_idx; iRow < end_idx;
         iRow++, outRow++) {
        bool bit = in_arr->get_null_bit<bodo_array_type::STRING>(iRow);
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
        out_arr->set_null_bit<bodo_array_type::STRING>(outRow, bit);
        out_data2[outRow + 1] = out_offset + num_chars;
    }
    return out_arr;
}

array_info* get_replace_regex_py_entry(array_info* p_in_arr,
                                       char const* const pat,
                                       char const* replacement) {
    std::shared_ptr<array_info> in_arr = std::shared_ptr<array_info>(p_in_arr);
    std::shared_ptr<array_info> out_arr = nullptr;

    if (in_arr->arr_type == bodo_array_type::STRING) {
        out_arr = get_replace_regex_slice(in_arr, pat, replacement, 0,
                                          in_arr->length);
    } else if (in_arr->arr_type == bodo_array_type::DICT) {
        // For dictionary-encoded string arrays, we optimize
        // by recursing on the dictionary (info1)
        // (which is presumably much smaller), and then
        // copying the null bitmap and indices.
        std::shared_ptr<array_info> dict_arr = in_arr->child_arrays[0];
        std::shared_ptr<array_info> indices_arr = in_arr->child_arrays[1];
        std::shared_ptr<array_info> new_dict = get_replace_regex_slice(
            dict_arr, pat, replacement, 0, dict_arr->length);
        std::shared_ptr<array_info> new_indices = copy_array(indices_arr);
        out_arr = create_dict_string_array(new_dict, new_indices);
    } else {
        Bodo_PyErr_SetString(
            PyExc_RuntimeError,
            "get_replace_regex_py_entry: array in_arr type should be string");
    }
    if (out_arr != nullptr) {
        return new array_info(*out_arr);
    }
    return nullptr;
}

array_info* get_replace_regex_dict_state_py_entry(array_info* p_in_arr,
                                                  char const* const pat,
                                                  char const* replacement,
                                                  DictEncodingState* state,
                                                  int64_t func_id) {
    std::shared_ptr<array_info> in_arr = std::shared_ptr<array_info>(p_in_arr);
    if (in_arr->arr_type != bodo_array_type::DICT) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "get_replace_regex_dict_state_py_entry: array "
                             "in_arr type should be dictionary encoded");
        return nullptr;
    }
    // For dictionary-encoded string arrays, we optimize
    // by recursing on the dictionary (info1)
    // (which is presumably much smaller), and then
    // copying the null bitmap and indices.
    std::shared_ptr<array_info> dict_arr = in_arr->child_arrays[0];
    std::shared_ptr<array_info> new_dict;
    int64_t new_id;
    int64_t old_id = dict_arr->array_id;
    int64_t old_size = 0;
    if (state->contains(func_id, old_id)) {
        std::tie(new_dict, new_id, old_size) =
            state->get_array(func_id, old_id);
        if (static_cast<uint64_t>(old_size) < dict_arr->length) {
            auto data_to_append = get_replace_regex_slice(
                dict_arr, pat, replacement, old_size, dict_arr->length);

            std::shared_ptr<array_info> combined_arr =
                alloc_string_array(Bodo_CTypes::STRING, 0, 0, new_id);

            ArrayBuildBuffer builder(combined_arr);
            builder.ReserveArray(new_dict);
            builder.UnsafeAppendBatch<bodo_array_type::STRING,
                                      Bodo_CTypes::STRING>(new_dict);
            builder.ReserveArray(data_to_append);
            builder.UnsafeAppendBatch<bodo_array_type::STRING,
                                      Bodo_CTypes::STRING>(data_to_append);

            new_dict = builder.data_array;
        }
    } else {
        new_dict = get_replace_regex_slice(dict_arr, pat, replacement, 0,
                                           dict_arr->length);
        new_id = new_dict->array_id;
        state->set_array(func_id, old_id, std::numeric_limits<size_t>::max(),
                         new_dict, new_id);
    }

    std::shared_ptr<array_info> indices_arr = in_arr->child_arrays[1];
    std::shared_ptr<array_info> new_indices = copy_array(indices_arr);
    std::shared_ptr<array_info> out_arr =
        create_dict_string_array(new_dict, new_indices);
    return new array_info(*out_arr);
}
