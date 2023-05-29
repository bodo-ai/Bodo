#include "_stream_join.h"
#include "_array_hash.h"
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_shuffle.h"

uint32_t HashHashJoinTable::operator()(const int64_t iRow) const {
    if (iRow >= 0) {
        return this->join_state->build_table_hashes[iRow];
    } else {
        return this->join_state->probe_table_hashes[-iRow - 1];
    }
}

bool KeyEqualHashJoinTable::operator()(const int64_t iRowA,
                                       const int64_t iRowB) const {
    const std::shared_ptr<table_info>& build_table =
        this->join_state->build_table_buffer.data_table;
    const std::shared_ptr<table_info>& probe_table =
        this->join_state->probe_table;

    bool is_build_A = iRowA >= 0;
    bool is_build_B = iRowB >= 0;

    size_t jRowA = is_build_A ? iRowA : -iRowA - 1;
    size_t jRowB = is_build_B ? iRowB : -iRowB - 1;

    const std::shared_ptr<table_info>& table_A =
        is_build_A ? build_table : probe_table;
    const std::shared_ptr<table_info>& table_B =
        is_build_B ? build_table : probe_table;

    // Determine if NA columns should match. They should always
    // match when populating the hash map with the build table.
    // When comparing the build and probe tables this depends on
    // is_na_equal.
    // TODO: Eliminate groups with NA columns with is_na_equal=False
    // from the hashmap.
    bool set_na_equal = is_na_equal || (is_build_A && is_build_B);
    bool test = TestEqualJoin(table_A, table_B, jRowA, jRowB, this->n_keys,
                              set_na_equal);
    return test;
}

/**
 * @brief consume build table batch in streaming join (insert into hash table)
 *
 * @param join_state join state pointer
 * @param in_table build table batch
 * @param is_last is last batch
 */
void join_build_consume_batch(JoinState* join_state,
                              std::shared_ptr<table_info> in_table,
                              bool is_last, bool parallel) {
    int n_pes, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // get hashes of the new batch (different hashes for partitioning and hash
    // table to reduce conflict)
    std::shared_ptr<uint32_t[]> batch_hashes_partition = hash_keys_table(
        in_table, join_state->n_keys, SEED_HASH_PARTITION, parallel);
    std::shared_ptr<uint32_t[]> batch_hashes_join =
        hash_keys_table(in_table, join_state->n_keys, SEED_HASH_JOIN, parallel);

    // insert batch into hash table
    // TODO[BSE-441]: see if appending all selected rows upfront to the build
    // buffer is faster (a call like AppendSelectedRows that takes a bool vector
    // from partitioning and appends all selected input rows)
    // TODO[BSE-441]: tune initial buffer buffer size and expansion strategy
    // using heuristics (e.g. SQL planner statistics)
    join_state->build_table_hashes.reserve(
        join_state->build_table_hashes.size() + in_table->nrows());
    join_state->build_table_buffer.ReserveTable(in_table);
    join_state->build_shuffle_buffer.ReserveTable(in_table);
    int64_t curr_build_size =
        join_state->build_table_buffer.data_table->nrows();
    for (size_t i_row = 0; i_row < in_table->nrows(); i_row++) {
        if (hash_to_rank(batch_hashes_partition[i_row], n_pes) == myrank ||
            !parallel) {
            join_state->build_table_hashes.emplace_back(
                batch_hashes_join[i_row]);
            join_state->build_table_buffer.AppendRow(in_table, i_row);
            join_state->build_table.emplace(curr_build_size, curr_build_size);
            curr_build_size++;
        } else {
            join_state->build_shuffle_buffer.AppendRow(in_table, i_row);
        }
    }

    if (is_last && parallel) {
        // shuffle data of other ranks
        std::shared_ptr<table_info> shuffle_table =
            join_state->build_shuffle_buffer.data_table;
        mpi_comm_info comm_info_table(shuffle_table->columns);
        std::shared_ptr<uint32_t[]> hashes = hash_keys_table(
            shuffle_table, join_state->n_keys, SEED_HASH_PARTITION, parallel);
        comm_info_table.set_counts(hashes, parallel);
        std::shared_ptr<table_info> new_data = shuffle_table_kernel(
            shuffle_table, hashes, comm_info_table, parallel);
        shuffle_table.reset();
        // TODO: clear build_shuffle_buffer memory

        // add hashes of the new data
        join_state->build_table_hashes.reserve(
            join_state->build_table_hashes.size() + new_data->nrows());
        std::shared_ptr<uint32_t[]> batch_hashes = hash_keys_table(
            new_data, join_state->n_keys, SEED_HASH_JOIN, parallel);
        join_state->build_table_hashes.insert(
            join_state->build_table_hashes.end(), batch_hashes.get(),
            batch_hashes.get() + new_data->nrows());

        // insert received data to build table
        int64_t n_prev_build =
            join_state->build_table_buffer.data_table->nrows();
        join_state->build_table_buffer.ReserveTable(new_data);
        int64_t n_rows = n_prev_build + new_data->nrows();
        for (int64_t i_row = n_prev_build; i_row < n_rows; i_row++) {
            join_state->build_table_buffer.AppendRow(new_data,
                                                     i_row - n_prev_build);
            join_state->build_table.emplace(i_row, i_row);
        }
    }

    // initialize build_table_matched for outer joins on build_table
    if (is_last && join_state->build_table_outer) {
        join_state->build_table_matched.resize(
            join_state->build_table_buffer.data_table->nrows(), false);
    }
}

/**
 * @brief consume probe table batch in streaming join and produce output table
 * batch
 *
 * @param join_state join state pointer
 * @param in_table probe table batch
 * @param is_last is last batch
 * @return std::shared_ptr<table_info> output table batch
 */
template <bool build_table_outer, bool probe_table_outer,
          bool non_equi_condition>
std::shared_ptr<table_info> join_probe_consume_batch(
    JoinState* join_state, std::shared_ptr<table_info> in_table, bool is_last,
    bool parallel) {
    int n_pes, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // update join state for hashing and comparison functions
    join_state->probe_table = in_table;
    join_state->probe_table_hashes =
        hash_keys_table(in_table, join_state->n_keys, SEED_HASH_JOIN, parallel);
    std::shared_ptr<uint32_t[]> batch_hashes_partition = hash_keys_table(
        in_table, join_state->n_keys, SEED_HASH_PARTITION, parallel);

    bodo::vector<int64_t> build_idxs;
    bodo::vector<int64_t> probe_idxs;

    // Fetch the raw array pointers from the arrays for passing
    // to the non-equijoin condition
    std::vector<array_info*> build_table_info_ptrs, probe_table_info_ptrs;
    // Vectors for data
    std::vector<void*> build_col_ptrs, probe_col_ptrs;
    // Vectors for null bitmaps for fast null checking from the cfunc
    std::vector<void*> build_null_bitmaps, probe_null_bitmaps;
    if (non_equi_condition) {
        std::tie(build_table_info_ptrs, build_col_ptrs, build_null_bitmaps) =
            get_gen_cond_data_ptrs(join_state->build_table_buffer.data_table);
        std::tie(probe_table_info_ptrs, probe_col_ptrs, probe_null_bitmaps) =
            get_gen_cond_data_ptrs(in_table);
    }

    // probe hash table
    join_state->probe_shuffle_buffer.ReserveTable(in_table);
    for (size_t i_row = 0; i_row < in_table->nrows(); i_row++) {
        if (hash_to_rank(batch_hashes_partition[i_row], n_pes) == myrank ||
            !parallel) {
            auto range = join_state->build_table.equal_range(-i_row - 1);
            if (probe_table_outer && range.first == range.second) {
                // Add unmatched rows from probe table to output table
                build_idxs.push_back(-1);
                probe_idxs.push_back(i_row);
                continue;
            }
            // Initialize to true for pure hash join so the final branch
            // is non-equality condition only.
            bool has_match = !non_equi_condition;
            for (auto it = range.first; it != range.second; ++it) {
                if (non_equi_condition) {
                    // Check for matches with the non-equality portion.
                    bool match = join_state->cond_func(
                        build_table_info_ptrs.data(),
                        probe_table_info_ptrs.data(), build_col_ptrs.data(),
                        probe_col_ptrs.data(), build_null_bitmaps.data(),
                        probe_null_bitmaps.data(), it->second, i_row);
                    if (!match) {
                        continue;
                    }
                    has_match = true;
                }
                if (build_table_outer) {
                    join_state->build_table_matched[it->second] = true;
                }
                build_idxs.push_back(it->second);
                probe_idxs.push_back(i_row);
            }
            // non-equality condition only branch
            if (!has_match && probe_table_outer) {
                // Add unmatched rows from probe table to output table
                build_idxs.push_back(-1);
                probe_idxs.push_back(i_row);
            }
        } else {
            join_state->probe_shuffle_buffer.AppendRow(in_table, i_row);
        }
    }

    // create output table using build and probe table indices (columns appended
    // side by side)
    std::shared_ptr<table_info> build_out_table =
        RetrieveTable(join_state->build_table_buffer.data_table, build_idxs, -1,
                      probe_table_outer);
    std::shared_ptr<table_info> probe_out_table =
        RetrieveTable(in_table, probe_idxs, -1, build_table_outer);
    build_idxs.clear();
    probe_idxs.clear();

    if (is_last) {
        if (parallel) {
            // shuffle data of other ranks
            std::shared_ptr<table_info> shuffle_table =
                join_state->probe_shuffle_buffer.data_table;
            mpi_comm_info comm_info_table(shuffle_table->columns);
            std::shared_ptr<uint32_t[]> hashes =
                hash_keys_table(shuffle_table, join_state->n_keys,
                                SEED_HASH_PARTITION, parallel);
            comm_info_table.set_counts(hashes, parallel);
            std::shared_ptr<table_info> new_data = shuffle_table_kernel(
                shuffle_table, hashes, comm_info_table, parallel);
            shuffle_table.reset();
            // TODO: clear probe_shuffle_buffer memory

            // probe hash table with new data
            join_state->probe_table = new_data;
            join_state->probe_table_hashes =
                hash_keys_table(join_state->probe_table, join_state->n_keys,
                                SEED_HASH_JOIN, parallel);

            // Fetch the raw array pointers from the arrays for passing
            // to the non-equijoin condition
            std::vector<array_info*> build_table_info_ptrs,
                probe_table_info_ptrs;
            // Vectors for data
            std::vector<void*> build_col_ptrs, probe_col_ptrs;
            // Vectors for null bitmaps for fast null checking from the cfunc
            std::vector<void*> build_null_bitmaps, probe_null_bitmaps;
            if (non_equi_condition) {
                std::tie(build_table_info_ptrs, build_col_ptrs,
                         build_null_bitmaps) =
                    get_gen_cond_data_ptrs(
                        join_state->build_table_buffer.data_table);
                std::tie(probe_table_info_ptrs, probe_col_ptrs,
                         probe_null_bitmaps) =
                    get_gen_cond_data_ptrs(join_state->probe_table);
            }

            for (size_t i_row = 0; i_row < new_data->nrows(); i_row++) {
                auto range = join_state->build_table.equal_range(-i_row - 1);
                if (probe_table_outer && range.first == range.second) {
                    // Add unmatched rows from probe table to output table
                    build_idxs.push_back(-1);
                    probe_idxs.push_back(i_row);
                    continue;
                }
                // Initialize to true for pure hash join so the final branch
                // is non-equality condition only.
                bool has_match = !non_equi_condition;
                for (auto it = range.first; it != range.second; ++it) {
                    if (non_equi_condition) {
                        // Check for matches with the non-equality portion.
                        bool match = join_state->cond_func(
                            build_table_info_ptrs.data(),
                            probe_table_info_ptrs.data(), build_col_ptrs.data(),
                            probe_col_ptrs.data(), build_null_bitmaps.data(),
                            probe_null_bitmaps.data(), it->second, i_row);
                        if (!match) {
                            continue;
                        }
                        has_match = true;
                    }
                    if (build_table_outer) {
                        join_state->build_table_matched[it->second] = true;
                    }
                    build_idxs.push_back(it->second);
                    probe_idxs.push_back(i_row);
                }
                // non-equality condition only branch
                if (!has_match && probe_table_outer) {
                    // Add unmatched rows from probe table to output table
                    build_idxs.push_back(-1);
                    probe_idxs.push_back(i_row);
                }
            }
        }

        if (build_table_outer) {
            // Add unmatched rows from build table to output table
            for (size_t i_row = 0;
                 i_row < join_state->build_table_matched.size(); i_row++) {
                if (!join_state->build_table_matched[i_row]) {
                    build_idxs.push_back(i_row);
                    probe_idxs.push_back(-1);
                }
            }
        }

        // These sizes should be the same
        // Only perform the concat if there is data to concat
        if (build_idxs.size() || probe_idxs.size()) {
            // create output table using build and probe table indices (columns
            // appended side by side)
            std::shared_ptr<table_info> new_build_out_table =
                RetrieveTable(join_state->build_table_buffer.data_table,
                              build_idxs, -1, probe_table_outer);
            std::shared_ptr<table_info> new_probe_out_table = RetrieveTable(
                join_state->probe_table, probe_idxs, -1, build_table_outer);

            // append new build data
            std::vector<std::shared_ptr<table_info>> build_tables(
                {build_out_table, new_build_out_table});
            build_out_table = concat_tables(build_tables);
            build_tables.clear();
            new_build_out_table.reset();

            // append new probe data
            std::vector<std::shared_ptr<table_info>> probe_tables(
                {probe_out_table, new_probe_out_table});
            probe_out_table = concat_tables(probe_tables);
            probe_tables.clear();
            new_probe_out_table.reset();
        }
    }

    std::vector<std::shared_ptr<array_info>> out_arrs;
    out_arrs.insert(out_arrs.end(), build_out_table->columns.begin(),
                    build_out_table->columns.end());
    out_arrs.insert(out_arrs.end(), probe_out_table->columns.begin(),
                    probe_out_table->columns.end());
    return std::make_shared<table_info>(out_arrs);
}

std::shared_ptr<table_info> alloc_table(
    const std::vector<int8_t>& arr_c_types,
    const std::vector<int8_t>& arr_array_types) {
    std::vector<std::shared_ptr<array_info>> arrays;

    for (size_t i = 0; i < arr_c_types.size(); i++) {
        bodo_array_type::arr_type_enum arr_type =
            (bodo_array_type::arr_type_enum)arr_array_types[i];
        Bodo_CTypes::CTypeEnum dtype = (Bodo_CTypes::CTypeEnum)arr_c_types[i];

        arrays.push_back(alloc_array(0, 0, 0, arr_type, dtype, 0, 0));
    }
    return std::make_shared<table_info>(arrays);
}

/**
 * @brief Initialize a new streaming join state for specified array types and
 * number of keys (called from Python)
 *
 * @param arr_c_types array types of build table columns (Bodo_CTypes ints)
 * @param n_arrs number of build table columns
 * @param n_keys number of join keys
 * @param build_table_outer whether to produce left outer join
 * @param probe_table_outer whether to produce right outer join
 * @return JoinState* join state to return to Python
 */
JoinState* join_state_init_py_entry(
    int8_t* build_arr_c_types, int8_t* build_arr_array_types, int n_build_arrs,
    int8_t* probe_arr_c_types, int8_t* probe_arr_array_types, int n_probe_arrs,
    int64_t n_keys, bool build_table_outer, bool probe_table_outer,
    cond_expr_fn_t cond_func) {
    return new JoinState(
        std::vector<int8_t>(build_arr_c_types,
                            build_arr_c_types + n_build_arrs),
        std::vector<int8_t>(build_arr_array_types,
                            build_arr_array_types + n_build_arrs),
        std::vector<int8_t>(probe_arr_c_types,
                            probe_arr_c_types + n_probe_arrs),
        std::vector<int8_t>(probe_arr_array_types,
                            probe_arr_array_types + n_probe_arrs),
        n_keys, build_table_outer, probe_table_outer, cond_func);
}

/**
 * @brief Python wrapper to consume build table batch
 *
 * @param join_state join state pointer
 * @param in_table build table batch
 * @param is_last is last batch
 */
void join_build_consume_batch_py_entry(JoinState* join_state,
                                       table_info* in_table, bool is_last,
                                       bool parallel) {
    try {
        join_build_consume_batch(join_state,
                                 std::shared_ptr<table_info>(in_table), is_last,
                                 parallel);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

/**
 * @brief Python wrapper to consume probe table batch and produce output table
 * batch
 *
 * @param join_state join state pointer
 * @param in_table probe table batch
 * @param is_last is last batch
 * @return table_info* output table batch
 */
table_info* join_probe_consume_batch_py_entry(JoinState* join_state,
                                              table_info* in_table,
                                              bool is_last, bool* out_is_last,
                                              bool parallel) {
#ifndef CONSUME_PROBE_BATCH
#define CONSUME_PROBE_BATCH(build_table_outer, probe_table_outer,         \
                            has_non_equi_cond, build_table_outer_exp,     \
                            probe_table_outer_exp, has_non_equi_cond_exp) \
    if (build_table_outer == build_table_outer_exp &&                     \
        probe_table_outer == probe_table_outer_exp &&                     \
        has_non_equi_cond == has_non_equi_cond_exp) {                     \
        out = join_probe_consume_batch<build_table_outer_exp,             \
                                       probe_table_outer_exp,             \
                                       has_non_equi_cond_exp>(            \
            join_state, std::unique_ptr<table_info>(in_table), is_last,   \
            parallel);                                                    \
    }
#endif

    try {
        // TODO: Actually output out_is_last based on is_last + the state
        // of the output buffer.
        *out_is_last = is_last;
        std::shared_ptr<table_info> out;
        bool contain_non_equi_cond = join_state->cond_func != NULL;

        CONSUME_PROBE_BATCH(join_state->build_table_outer,
                            join_state->probe_table_outer,
                            contain_non_equi_cond, true, true, true)
        CONSUME_PROBE_BATCH(join_state->build_table_outer,
                            join_state->probe_table_outer,
                            contain_non_equi_cond, true, true, false)
        CONSUME_PROBE_BATCH(join_state->build_table_outer,
                            join_state->probe_table_outer,
                            contain_non_equi_cond, true, false, true)
        CONSUME_PROBE_BATCH(join_state->build_table_outer,
                            join_state->probe_table_outer,
                            contain_non_equi_cond, true, false, false)
        CONSUME_PROBE_BATCH(join_state->build_table_outer,
                            join_state->probe_table_outer,
                            contain_non_equi_cond, false, true, true)
        CONSUME_PROBE_BATCH(join_state->build_table_outer,
                            join_state->probe_table_outer,
                            contain_non_equi_cond, false, true, false)
        CONSUME_PROBE_BATCH(join_state->build_table_outer,
                            join_state->probe_table_outer,
                            contain_non_equi_cond, false, false, true)
        CONSUME_PROBE_BATCH(join_state->build_table_outer,
                            join_state->probe_table_outer,
                            contain_non_equi_cond, false, false, false)

        return new table_info(*out);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
#undef CONSUME_PROBE_BATCH
}

/**
 * @brief delete join state (called from Python after probe loop is finished)
 *
 * @param join_state join state pointer to delete
 */
void delete_join_state(JoinState* join_state) { delete join_state; }

PyMODINIT_FUNC PyInit_stream_join_cpp(void) {
    PyObject* m;
    MOD_DEF(m, "stream_join_cpp", "No docs", NULL);
    if (m == NULL)
        return NULL;

    bodo_common_init();

    SetAttrStringFromVoidPtr(m, join_state_init_py_entry);
    SetAttrStringFromVoidPtr(m, join_build_consume_batch_py_entry);
    SetAttrStringFromVoidPtr(m, join_probe_consume_batch_py_entry);
    SetAttrStringFromVoidPtr(m, delete_join_state);
    return m;
}
