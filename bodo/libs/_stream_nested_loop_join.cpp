#include "_shuffle.h"
#include "_stream_join.h"

void NestedLoopJoinState::FinalizeBuild() {
    // If the build table is small enough, broadcast it to all ranks
    // so the probe table can be joined locally.
    // NOTE: broadcasting build table is incorrect if the probe table is
    // replicated.
    if (this->build_parallel && this->probe_parallel) {
        int64_t global_table_size =
            table_global_memory_size(this->build_table_buffer.data_table);
        if (global_table_size < get_bcast_join_threshold()) {
            this->build_parallel = false;
            bool all_gather = true;
            std::shared_ptr<table_info> gathered_table = gather_table(
                this->build_table_buffer.data_table, -1, all_gather, true);

            gathered_table =
                this->UnifyBuildTableDictionaryArrays(gathered_table);
            this->build_table_buffer.Reset();
            this->build_table_buffer.ReserveTable(gathered_table);
            this->build_table_buffer.AppendBatch(gathered_table);
        }
    }

    if (this->build_table_outer) {
        this->build_table_matched.resize(
            arrow::bit_util::BytesForBits(
                this->build_table_buffer.data_table->nrows()),
            0);
    }

    JoinState::FinalizeBuild();
}

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
                                          bool is_last) {
    // just add batch to build table buffer
    if (join_state->build_input_finalized) {
        // Nothing left to do for build
        return;
    }

    // Unify dictionaries to allow consistent hashing and fast key
    // comparison using indices NOTE: key columns in build_table_buffer (of
    // all partitions), probe_table_buffers (of all partitions),
    // build_shuffle_buffer and probe_shuffle_buffer use the same dictionary
    // object for consistency. Non-key DICT columns of build_table_buffer
    // and build_shuffle_buffer also share their dictionaries and will also
    // be unified.
    in_table = join_state->UnifyBuildTableDictionaryArrays(in_table);

    join_state->build_table_buffer.ReserveTable(in_table);
    join_state->build_table_buffer.AppendBatch(in_table);

    if (is_last) {
        // Finalize the join state
        join_state->FinalizeBuild();
    }
}

/**
 * @brief local nested loop computation on input probe table chunk (assuming
 * join state has all of build table)
 *
 * @param join_state join state pointer
 * @param probe_table probe table batch
 * @param build_kept_cols Which columns to generate in the output on the
 * build side.
 * @param probe_kept_cols Which columns to generate in the output on the
 * probe side.
 * @param parallel_trace parallel flag for tracing purposes
 */
void nested_loop_join_local_chunk(NestedLoopJoinState* join_state,
                                  std::shared_ptr<table_info> probe_table,
                                  const std::vector<uint64_t>& build_kept_cols,
                                  const std::vector<uint64_t>& probe_kept_cols,
                                  bool parallel_trace) {
    bodo::vector<int64_t> build_idxs;
    bodo::vector<int64_t> probe_idxs;

    bodo::vector<uint8_t> probe_table_matched(0, 0);
    if (join_state->probe_table_outer) {
        probe_table_matched.resize(
            arrow::bit_util::BytesForBits(probe_table->nrows()), 0);
    }

#ifndef JOIN_TABLE_LOCAL
#define JOIN_TABLE_LOCAL(build_table_outer, probe_table_outer,                 \
                         non_equi_condition, build_table_outer_exp,            \
                         probe_table_outer_exp, non_equi_condition_exp)        \
    if (build_table_outer == build_table_outer_exp &&                          \
        probe_table_outer == probe_table_outer_exp &&                          \
        non_equi_condition == non_equi_condition_exp) {                        \
        nested_loop_join_table_local<build_table_outer_exp,                    \
                                     probe_table_outer_exp,                    \
                                     non_equi_condition_exp>(                  \
            join_state->build_table_buffer.data_table, probe_table, cond_func, \
            parallel_trace, build_idxs, probe_idxs,                            \
            join_state->build_table_matched, probe_table_matched);             \
    }
#endif

    // cfunc is passed in batch format for nested loop join
    // see here:
    // https://github.com/Bodo-inc/Bodo/blob/fd987eca2684b9178a13caf41f23349f92a0a96e/bodo/libs/stream_join.py#L470
    cond_expr_fn_batch_t cond_func =
        (cond_expr_fn_batch_t)join_state->cond_func;

    bool non_equi_condition = cond_func != nullptr;
    JOIN_TABLE_LOCAL(join_state->build_table_outer,
                     join_state->probe_table_outer, non_equi_condition, true,
                     true, true)
    JOIN_TABLE_LOCAL(join_state->build_table_outer,
                     join_state->probe_table_outer, non_equi_condition, true,
                     true, false)
    JOIN_TABLE_LOCAL(join_state->build_table_outer,
                     join_state->probe_table_outer, non_equi_condition, true,
                     false, true)
    JOIN_TABLE_LOCAL(join_state->build_table_outer,
                     join_state->probe_table_outer, non_equi_condition, true,
                     false, false)
    JOIN_TABLE_LOCAL(join_state->build_table_outer,
                     join_state->probe_table_outer, non_equi_condition, false,
                     true, true)
    JOIN_TABLE_LOCAL(join_state->build_table_outer,
                     join_state->probe_table_outer, non_equi_condition, false,
                     true, false)
    JOIN_TABLE_LOCAL(join_state->build_table_outer,
                     join_state->probe_table_outer, non_equi_condition, false,
                     false, true)
    JOIN_TABLE_LOCAL(join_state->build_table_outer,
                     join_state->probe_table_outer, non_equi_condition, false,
                     false, false)

    if (join_state->probe_table_outer) {
        add_unmatched_rows(probe_table_matched, probe_table->nrows(),
                           probe_idxs, build_idxs,
                           // We always broadcast one of the sides. If the build
                           // side is parallel then either the probe side is
                           // replicated or we broadcast the probe side.
                           join_state->build_parallel);
    }
    join_state->output_buffer->AppendJoinOutput(
        join_state->build_table_buffer.data_table, probe_table, build_idxs,
        probe_idxs, build_kept_cols, probe_kept_cols);
#undef JOIN_TABLE_LOCAL
}

void nested_loop_join_probe_consume_batch(
    NestedLoopJoinState* join_state, std::shared_ptr<table_info> in_table,
    const std::vector<uint64_t> build_kept_cols,
    const std::vector<uint64_t> probe_kept_cols, bool is_last) {
    if (join_state->probe_input_finalized) {
        if (in_table->nrows() != 0) {
            throw std::runtime_error(
                "nested_loop_join_probe_consume_batch: Received non-empty "
                "in_table after "
                "the probe was already finalized!");
        }
        // No processing left.
        return;
    }

    // We only need to take the parallel path if both tables are parallel
    // and the build table wasn't broadcast.
    bool parallel = join_state->build_parallel && join_state->probe_parallel;
    if (parallel) {
        int n_pes, myrank;
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        // make dictionaries global for broadcast
        for (size_t i = 0; i < in_table->ncols(); i++) {
            std::shared_ptr<array_info> arr = in_table->columns[i];
            if (arr->arr_type == bodo_array_type::DICT) {
                make_dictionary_global_and_unique(arr, parallel);
            }
        }

        for (int p = 0; p < n_pes; p++) {
            std::shared_ptr<table_info> bcast_probe_chunk = broadcast_table(
                in_table, in_table, in_table->ncols(), parallel, p);
            bcast_probe_chunk =
                join_state->UnifyProbeTableDictionaryArrays(bcast_probe_chunk);
            nested_loop_join_local_chunk(join_state, bcast_probe_chunk,
                                         build_kept_cols, probe_kept_cols,
                                         parallel);
        }
    } else {
        // Unify dictionaries to allow consistent hashing and fast key
        // comparison using indices NOTE: key columns in build_table_buffer (of
        // all partitions), probe_table_buffers (of all partitions),
        // build_shuffle_buffer and probe_shuffle_buffer use the same dictionary
        // object for consistency. Non-key DICT columns of probe_table_buffer
        // and probe_shuffle_buffer also share their dictionaries and will also
        // be unified.
        in_table = join_state->UnifyProbeTableDictionaryArrays(in_table);
        nested_loop_join_local_chunk(join_state, in_table, build_kept_cols,
                                     probe_kept_cols, parallel);
    }

    if (join_state->build_table_outer && is_last) {
        // Add unmatched rows from build table
        // for outer join
        bodo::vector<int64_t> build_idxs;
        bodo::vector<int64_t> probe_idxs;

        add_unmatched_rows(
            join_state->build_table_matched,
            join_state->build_table_buffer.data_table->nrows(), build_idxs,
            probe_idxs,
            !join_state->build_parallel && join_state->probe_parallel);

        join_state->output_buffer->AppendJoinOutput(
            join_state->build_table_buffer.data_table,
            join_state->dummy_probe_table, build_idxs, probe_idxs,
            build_kept_cols, probe_kept_cols);
        build_idxs.clear();
        probe_idxs.clear();
    }
    if (is_last) {
        // Finalize the probe side
        join_state->FinalizeProbe();
    }
}

void nested_loop_join_build_consume_batch_py_entry(
    NestedLoopJoinState* join_state, table_info* in_table, bool is_last) {
    try {
        nested_loop_join_build_consume_batch(
            join_state, std::shared_ptr<table_info>(in_table), is_last);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}
