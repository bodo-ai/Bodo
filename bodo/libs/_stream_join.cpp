#include "_stream_join.h"
#include "_array_hash.h"
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_shuffle.h"

uint32_t HashHashJoinTable::operator()(const int64_t iRow) const {
    if (iRow >= 0) {
        return join_state->build_table_hashes[iRow];
    } else {
        return join_state->probe_table_hashes[-iRow - 1];
    }
}

bool KeyEqualHashJoinTable::operator()(const int64_t iRowA,
                                       const int64_t iRowB) const {
    const std::shared_ptr<const table_info>& build_table =
        join_state->build_table_buffer.data_table;
    const std::shared_ptr<const table_info>& probe_table =
        join_state->probe_table;

    bool is_build_A = iRowA >= 0;
    bool is_build_B = iRowB >= 0;

    size_t jRowA = is_build_A ? iRowA : -iRowA - 1;
    size_t jRowB = is_build_B ? iRowB : -iRowB - 1;

    const std::shared_ptr<const table_info>& table_A =
        is_build_A ? build_table : probe_table;
    const std::shared_ptr<const table_info>& table_B =
        is_build_B ? build_table : probe_table;

    // Determine if NA columns should match. They should always
    // match when populating the hash map with the build table.
    // When comparing the build and probe tables this depends on
    // is_na_equal.
    // TODO: Eliminate groups with NA columns with is_na_equal=False
    // from the hashmap.
    bool set_na_equal = (is_build_A && is_build_B) || is_na_equal;
    bool test = TestEqualJoin(table_A, table_B, jRowA, jRowB,
                              join_state->n_keys, set_na_equal);
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
    for (int64_t i_row = 0; i_row < in_table->nrows(); i_row++) {
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

    // probe hash table
    join_state->probe_shuffle_buffer.ReserveTable(in_table);
    for (int64_t i_row = 0; i_row < in_table->nrows(); i_row++) {
        if (hash_to_rank(batch_hashes_partition[i_row], n_pes) == myrank ||
            !parallel) {
            auto range = join_state->build_table.equal_range(-i_row - 1);
            for (auto it = range.first; it != range.second; ++it) {
                build_idxs.push_back(it->second);
                probe_idxs.push_back(i_row);
            }
        } else {
            join_state->probe_shuffle_buffer.AppendRow(in_table, i_row);
        }
    }

    // create output table using build and probe table indices (columns appended
    // side by side)
    std::shared_ptr<table_info> build_out_table =
        RetrieveTable(join_state->build_table_buffer.data_table, build_idxs);
    std::shared_ptr<table_info> probe_out_table =
        RetrieveTable(in_table, probe_idxs);
    build_idxs.clear();
    probe_idxs.clear();

    if (is_last && parallel) {
        // shuffle data of other ranks
        std::shared_ptr<table_info> shuffle_table =
            join_state->probe_shuffle_buffer.data_table;
        mpi_comm_info comm_info_table(shuffle_table->columns);
        std::shared_ptr<uint32_t[]> hashes = hash_keys_table(
            shuffle_table, join_state->n_keys, SEED_HASH_PARTITION, parallel);
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
        for (int64_t i_row = 0; i_row < new_data->nrows(); i_row++) {
            auto range = join_state->build_table.equal_range(-i_row - 1);
            for (auto it = range.first; it != range.second; ++it) {
                build_idxs.push_back(it->second);
                probe_idxs.push_back(i_row);
            }
        }

        std::shared_ptr<table_info> new_build_out_table = RetrieveTable(
            join_state->build_table_buffer.data_table, build_idxs);
        std::shared_ptr<table_info> new_probe_out_table =
            RetrieveTable(new_data, probe_idxs);

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

    std::vector<std::shared_ptr<array_info>> out_arrs;
    out_arrs.insert(out_arrs.end(), build_out_table->columns.begin(),
                    build_out_table->columns.end());
    out_arrs.insert(out_arrs.end(), probe_out_table->columns.begin(),
                    probe_out_table->columns.end());
    return std::make_shared<table_info>(out_arrs);
}

std::shared_ptr<table_info> alloc_table(std::vector<int8_t> arr_c_types,
                                        std::vector<int8_t> arr_array_types) {
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
 * @return JoinState* join state to return to Python
 */
JoinState* join_state_init_py_entry(int8_t* build_arr_c_types,
                                    int8_t* build_arr_array_types,
                                    int n_build_arrs, int8_t* probe_arr_c_types,
                                    int8_t* probe_arr_array_types,
                                    int n_probe_arrs, int64_t n_keys) {
    return new JoinState(
        std::vector<int8_t>(build_arr_c_types,
                            build_arr_c_types + n_build_arrs),
        std::vector<int8_t>(build_arr_array_types,
                            build_arr_array_types + n_build_arrs),
        std::vector<int8_t>(probe_arr_c_types,
                            probe_arr_c_types + n_probe_arrs),
        std::vector<int8_t>(probe_arr_array_types,
                            probe_arr_array_types + n_probe_arrs),
        n_keys);
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
                                              bool is_last, bool parallel) {
    try {
        std::shared_ptr<table_info> out = join_probe_consume_batch(
            join_state, std::unique_ptr<table_info>(in_table), is_last,
            parallel);
        return new table_info(*out);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
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
