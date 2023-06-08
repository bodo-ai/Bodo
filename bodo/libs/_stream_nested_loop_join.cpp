#include "_shuffle.h"
#include "_stream_join.h"

/**
 * @brief consume build table batch in streaming nested loop join
 * Design doc:
 * https://bodo.atlassian.net/wiki/spaces/B/pages/1373896721/Vectorized+Nested+Loop+Join+Design
 *
 * @param join_state join state pointer
 * @param in_table build table batch
 * @param is_last is last batch
 */
void nested_loop_join_build_consume_batch(NestedLoopJoinState* join_state,
                                          std::shared_ptr<table_info> in_table,
                                          bool is_last, bool parallel) {
    // just add batch to build table buffer
    std::vector<std::shared_ptr<table_info>> tables(
        {join_state->build_table_buffer.data_table, in_table});
    join_state->build_table_buffer.data_table = concat_tables(tables);
    tables.clear();
    if (is_last && join_state->build_table_outer) {
        join_state->build_table_matched.resize(
            arrow::bit_util::BytesForBits(
                join_state->build_table_buffer.data_table->nrows()),
            0);
    }
}

/**
 * @brief local nested loop computation on input probe table chunk (assuming
 * join state has all of build table)
 *
 * @param join_state join state pointer
 * @param probe_table probe table batch
 * @param build_kept_cols Which columns to generate in the output on the build
 * side.
 * @param probe_kept_cols Which columns to generate in the output on the probe
 * side.
 * @param[out] total_rows Store the number of rows in the output batch in case
 *        all columns are dead. This function should increment this with the
 * size of this chunk.
 * @param is_parallel parallel flag for tracing purposes
 * @return std::shared_ptr<table_info> output table batch
 */
std::shared_ptr<table_info> nested_loop_join_local_chunk(
    NestedLoopJoinState* join_state, std::shared_ptr<table_info> probe_table,
    const std::vector<uint64_t>& build_kept_cols,
    const std::vector<uint64_t>& probe_kept_cols, int64_t* total_rows,
    bool parallel) {
    bodo::vector<int64_t> build_idxs;
    bodo::vector<int64_t> probe_idxs;

    bodo::vector<uint8_t> probe_table_matched(0, 0);
    if (join_state->probe_table_outer) {
        probe_table_matched.resize(
            arrow::bit_util::BytesForBits(probe_table->nrows()), 0);
    }

    // cfunc is passed in batch format for nested loop join
    // see here:
    // https://github.com/Bodo-inc/Bodo/blob/fd987eca2684b9178a13caf41f23349f92a0a96e/bodo/libs/stream_join.py#L470
    // TODO: template for cases without condition (cross join) to improve
    // performance
    cond_expr_fn_batch_t cond_func =
        (cond_expr_fn_batch_t)join_state->cond_func;

    nested_loop_join_table_local(
        join_state->build_table_buffer.data_table, probe_table,
        join_state->build_table_outer, join_state->probe_table_outer, cond_func,
        parallel, build_idxs, probe_idxs, join_state->build_table_matched,
        probe_table_matched);
    if (join_state->probe_table_outer) {
        add_unmatched_rows(probe_table_matched, probe_table->nrows(),
                           probe_idxs, build_idxs, parallel);
    }

    // Update the output size for this chunk.
    *total_rows += build_idxs.size();

    std::shared_ptr<table_info> build_out_table =
        RetrieveTable(join_state->build_table_buffer.data_table, build_idxs,
                      build_kept_cols, join_state->probe_table_outer);
    std::shared_ptr<table_info> probe_out_table =
        RetrieveTable(probe_table, probe_idxs, probe_kept_cols,
                      join_state->build_table_outer);
    build_idxs.clear();
    probe_idxs.clear();

    std::vector<std::shared_ptr<array_info>> out_arrs;
    out_arrs.insert(out_arrs.end(), build_out_table->columns.begin(),
                    build_out_table->columns.end());
    out_arrs.insert(out_arrs.end(), probe_out_table->columns.begin(),
                    probe_out_table->columns.end());
    return std::make_shared<table_info>(out_arrs);
}

std::shared_ptr<table_info> nested_loop_join_probe_consume_batch(
    NestedLoopJoinState* join_state, std::shared_ptr<table_info> in_table,
    const std::vector<uint64_t> build_kept_cols,
    const std::vector<uint64_t> probe_kept_cols, int64_t* total_rows,
    bool is_last, bool parallel) {
    // Initialize the output to 0.
    *total_rows = 0;
    std::shared_ptr<table_info> out_table;
    if (parallel) {
        int n_pes, myrank;
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        std::vector<std::shared_ptr<table_info>> out_table_chunks;
        out_table_chunks.reserve(n_pes);

        for (int p = 0; p < n_pes; p++) {
            std::shared_ptr<table_info> bcast_probe_chunk = broadcast_table(
                in_table, in_table, in_table->ncols(), parallel, p);
            std::shared_ptr<table_info> out_table_chunk =
                nested_loop_join_local_chunk(join_state, bcast_probe_chunk,
                                             build_kept_cols, probe_kept_cols,
                                             total_rows, parallel);
            out_table_chunks.emplace_back(out_table_chunk);
        }
        out_table = concat_tables(out_table_chunks);
    } else {
        out_table =
            nested_loop_join_local_chunk(join_state, in_table, build_kept_cols,
                                         probe_kept_cols, total_rows, parallel);
    }

    if (join_state->build_table_outer && is_last) {
        // Add unmatched rows from build table
        // for outer join
        bodo::vector<int64_t> build_idxs;
        bodo::vector<int64_t> probe_idxs;
        add_unmatched_rows(join_state->build_table_matched,
                           join_state->build_table_buffer.data_table->nrows(),
                           build_idxs, probe_idxs, false);
        // Update the total_rows
        *total_rows += build_idxs.size();

        std::shared_ptr<table_info> build_out_outer =
            RetrieveTable(join_state->build_table_buffer.data_table, build_idxs,
                          build_kept_cols, join_state->probe_table_outer);
        std::shared_ptr<table_info> probe_out_outer =
            RetrieveTable(in_table, probe_idxs, probe_kept_cols,
                          join_state->build_table_outer);
        build_idxs.clear();
        probe_idxs.clear();

        std::vector<std::shared_ptr<array_info>> out_arrs;
        out_arrs.insert(out_arrs.end(), build_out_outer->columns.begin(),
                        build_out_outer->columns.end());
        out_arrs.insert(out_arrs.end(), probe_out_outer->columns.begin(),
                        probe_out_outer->columns.end());
        std::shared_ptr<table_info> outer_table =
            std::make_shared<table_info>(out_arrs);
        out_table = concat_tables({out_table, outer_table});
    }

    return out_table;
}

void nested_loop_join_build_consume_batch_py_entry(
    NestedLoopJoinState* join_state, table_info* in_table, bool is_last,
    bool parallel) {
    try {
        nested_loop_join_build_consume_batch(
            join_state, std::shared_ptr<table_info>(in_table), is_last,
            parallel);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

table_info* nested_loop_join_probe_consume_batch_py_entry(
    NestedLoopJoinState* join_state, table_info* in_table,
    uint64_t* kept_build_col_nums, int64_t num_kept_build_cols,
    uint64_t* kept_probe_col_nums, int64_t num_kept_probe_cols,
    int64_t* total_rows, bool is_last, bool* out_is_last, bool parallel) {
    try {
        // TODO: Actually output out_is_last based on is_last + the state
        // of the output buffer.
        *out_is_last = is_last;
        std::vector<uint64_t> build_kept_cols(
            kept_build_col_nums, kept_build_col_nums + num_kept_build_cols);
        std::vector<uint64_t> probe_kept_cols(
            kept_probe_col_nums, kept_probe_col_nums + num_kept_probe_cols);
        std::shared_ptr<table_info> out = nested_loop_join_probe_consume_batch(
            join_state, std::unique_ptr<table_info>(in_table),
            std::move(build_kept_cols), std::move(probe_kept_cols), total_rows,
            is_last, parallel);

        return new table_info(*out);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}
