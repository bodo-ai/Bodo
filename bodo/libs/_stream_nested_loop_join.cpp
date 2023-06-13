#include "_shuffle.h"
#include "_stream_join.h"

void NestedLoopJoinState::InitOutputBuffer(
    const std::vector<uint64_t>& build_kept_cols,
    const std::vector<uint64_t>& probe_kept_cols) {
    // TODO: Move to JoinState after dict encoding support for
    // NestedLoopJoinState.
    if (this->output_buffer != nullptr) {
        // Already initialized. We only initialize on the first
        // iteration.
        return;
    }
    auto [build_arr_c_types, build_arr_array_types] =
        get_dtypes_arr_types_from_table(this->build_table_buffer.data_table);
    auto [probe_arr_c_types, probe_arr_array_types] =
        get_dtypes_arr_types_from_table(this->dummy_probe_table);
    std::vector<int8_t> arr_c_types, arr_array_types;
    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders;
    arr_c_types.reserve(build_kept_cols.size() + probe_kept_cols.size());
    arr_array_types.reserve(build_kept_cols.size() + probe_kept_cols.size());
    dict_builders.reserve(build_kept_cols.size() + probe_kept_cols.size());
    for (uint64_t i_col : build_kept_cols) {
        bodo_array_type::arr_type_enum arr_type =
            (bodo_array_type::arr_type_enum)build_arr_array_types[i_col];
        Bodo_CTypes::CTypeEnum dtype =
            (Bodo_CTypes::CTypeEnum)build_arr_c_types[i_col];
        // In the probe outer case, we need to make NUMPY arrays
        // into NULLABLE arrays. Matches the `use_nullable_arrs`
        // behavior of RetrieveTable.
        // TODO: Move to a helper function.
        if (this->probe_table_outer && ((arr_type == bodo_array_type::NUMPY) &&
                                        (is_integer(dtype) || is_float(dtype) ||
                                         dtype == Bodo_CTypes::_BOOL))) {
            arr_type = bodo_array_type::NULLABLE_INT_BOOL;
        }
        arr_c_types.push_back(dtype);
        arr_array_types.push_back(arr_type);
        // TODO: Integrate dict_builders for nested loop join.
        dict_builders.push_back(nullptr);
    }
    for (uint64_t i_col : probe_kept_cols) {
        bodo_array_type::arr_type_enum arr_type =
            (bodo_array_type::arr_type_enum)probe_arr_array_types[i_col];
        Bodo_CTypes::CTypeEnum dtype =
            (Bodo_CTypes::CTypeEnum)probe_arr_c_types[i_col];
        // In the build outer case, we need to make NUMPY arrays
        // into NULLABLE arrays. Matches the `use_nullable_arrs`
        // behavior of RetrieveTable.
        if (this->build_table_outer && ((arr_type == bodo_array_type::NUMPY) &&
                                        (is_integer(dtype) || is_float(dtype) ||
                                         dtype == Bodo_CTypes::_BOOL))) {
            arr_type = bodo_array_type::NULLABLE_INT_BOOL;
        }
        arr_c_types.push_back(dtype);
        arr_array_types.push_back(arr_type);
        // TODO: Integrate dict_builders for nested loop join.
        dict_builders.push_back(nullptr);
    }
    this->output_buffer = std::make_shared<ChunkedTableBuilder>(
        arr_c_types, arr_array_types, dict_builders, this->output_batch_size,
        DEFAULT_MAX_RESIZE_COUNT_FOR_VARIABLE_SIZE_DTYPES);
}
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
            this->build_table_buffer.data_table = gather_table(
                this->build_table_buffer.data_table, -1, all_gather, true);
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
    std::vector<std::shared_ptr<table_info>> tables(
        {join_state->build_table_buffer.data_table, in_table});
    join_state->build_table_buffer.data_table = concat_tables(tables);
    tables.clear();
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
 * @param build_kept_cols Which columns to generate in the output on the build
 * side.
 * @param probe_kept_cols Which columns to generate in the output on the probe
 * side.
 * @param is_parallel parallel flag for tracing purposes
 */
void nested_loop_join_local_chunk(NestedLoopJoinState* join_state,
                                  std::shared_ptr<table_info> probe_table,
                                  const std::vector<uint64_t>& build_kept_cols,
                                  const std::vector<uint64_t>& probe_kept_cols,
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
                           probe_idxs, build_idxs,
                           parallel && join_state->build_parallel &&
                               !join_state->probe_parallel);
    }
    join_state->output_buffer->AppendJoinOutput(
        join_state->build_table_buffer.data_table, probe_table, build_idxs,
        probe_idxs, build_kept_cols, probe_kept_cols);
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

    // We only need to take the parallel path if both tables are parallel and
    // the build table wasn't broadcast.
    bool parallel = join_state->build_parallel && join_state->probe_parallel;
    if (parallel) {
        int n_pes, myrank;
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        for (int p = 0; p < n_pes; p++) {
            std::shared_ptr<table_info> bcast_probe_chunk = broadcast_table(
                in_table, in_table, in_table->ncols(), parallel, p);
            nested_loop_join_local_chunk(join_state, bcast_probe_chunk,
                                         build_kept_cols, probe_kept_cols,
                                         parallel);
        }
    } else {
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
            join_state->build_table_buffer.data_table, in_table, build_idxs,
            probe_idxs, build_kept_cols, probe_kept_cols);
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
