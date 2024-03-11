#include "_distributed.h"
#include "_join.h"
#include "_nested_loop_join_impl.h"
#include "_shuffle.h"
#include "_stream_join.h"

void NestedLoopJoinState::FinalizeBuild() {
    // Finalize any active chunk
    this->build_table_buffer->Finalize();

    // If the build table is small enough, broadcast it to all ranks
    // so the probe table can be joined locally.
    // NOTE: broadcasting build table is incorrect if the probe table is
    // replicated.
    if (this->build_parallel && this->probe_parallel) {
        int64_t table_size = 0;
        for (const auto& table : *this->build_table_buffer) {
            // For certain data types like string, we need to load the buffers
            // (e.g. the offset buffers for strings) in memory to be able to
            // calculate the memory size.
            // https://bodo.atlassian.net/browse/BSE-874 will resolve this.
            table_size += table_local_memory_size(table, false);
        }
        MPI_Allreduce(MPI_IN_PLACE, &table_size, 1, MPI_INT64_T, MPI_SUM,
                      MPI_COMM_WORLD);
        if (table_size < get_bcast_join_threshold()) {
            this->build_parallel = false;
            // calculate the max number of chunks for all partitions
            int64_t n_chunks = this->build_table_buffer->chunks.size();
            MPI_Allreduce(MPI_IN_PLACE, &n_chunks, 1, MPI_INT64_T, MPI_MAX,
                          MPI_COMM_WORLD);
            // Create a new chunked table which will store the chunks gathered
            // from all ranks. This will eventually become the new
            // build_table_buffer of this JoinState.
            std::unique_ptr<ChunkedTableBuilder> new_build_table_buffer =
                std::make_unique<ChunkedTableBuilder>(
                    this->build_table_schema, this->build_table_dict_builders,
                    this->build_table_buffer->active_chunk_capacity,
                    this->build_table_buffer
                        ->max_resize_count_for_variable_size_dtypes);
            for (int64_t i_chunk = 0; i_chunk < n_chunks; i_chunk++) {
                std::shared_ptr<table_info> gathered_chunk = gather_table(
                    std::get<0>(this->build_table_buffer->PopChunk()), -1,
                    /*all_gather*/ true, true);
                new_build_table_buffer->AppendBatch(
                    this->UnifyBuildTableDictionaryArrays(gathered_chunk));
            }
            this->build_table_buffer = std::move(new_build_table_buffer);
            this->build_table_buffer->Finalize();
        }
    }

    if (this->build_table_outer) {
        auto build_table_matched_pin(bodo::pin(build_table_matched));
        build_table_matched_pin->resize(
            arrow::bit_util::BytesForBits(
                this->build_table_buffer->total_remaining),
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
bool nested_loop_join_build_consume_batch(NestedLoopJoinState* join_state,
                                          std::shared_ptr<table_info> in_table,
                                          bool is_last) {
    // just add batch to build table buffer
    if (join_state->build_input_finalized) {
        // Nothing left to do for build
        return true;
    }

    // Unify dictionaries to allow consistent hashing and fast key
    // comparison using indices NOTE: key columns in build_table_buffer (of
    // all partitions), probe_table_buffers (of all partitions),
    // build_shuffle_buffer and probe_shuffle_buffer use the same dictionary
    // object for consistency. Non-key DICT columns of build_table_buffer
    // and build_shuffle_buffer also share their dictionaries and will also
    // be unified.
    in_table = join_state->UnifyBuildTableDictionaryArrays(in_table);

    join_state->build_table_buffer->AppendBatch(in_table);

    // is_last can be local here because build only shuffles once at the end
    if (is_last) {
        // Finalize the join state
        join_state->FinalizeBuild();
    }
    return is_last;
}

/**
 * @brief local nested loop computation on input probe table chunk (assuming
 * join state has all of build table)
 *
 * @param join_state join state pointer
 * @param build_table build table batch
 * @param probe_table probe table batch
 * @param build_kept_cols Which columns to generate in the output on the
 * build side.
 * @param probe_kept_cols Which columns to generate in the output on the
 * probe side.
 * @param[in, out] build_table_matched_guard Bitmask to track the build table
 * matches. This will be updated in place.
 * @param[in, out] probe_table_matched Bitmask to trace the probe table matches.
 * This will be updated in place.
 * @param build_table_offset the number of bits from the start of
 * build_table_matched that belongs to previous chunks of the build table buffer
 */
void nested_loop_join_local_chunk(
    NestedLoopJoinState* join_state, std::shared_ptr<table_info> build_table,
    std::shared_ptr<table_info> probe_table,
    const std::vector<uint64_t>& build_kept_cols,
    const std::vector<uint64_t>& probe_kept_cols,
    bodo::pin_guard<decltype(NestedLoopJoinState::build_table_matched)>&
        build_table_matched_guard,
    bodo::vector<uint8_t>& probe_table_matched, int64_t build_table_offset) {
    bodo::vector<int64_t> build_idxs;
    bodo::vector<int64_t> probe_idxs;

#ifndef JOIN_TABLE_LOCAL
#define JOIN_TABLE_LOCAL(build_table_outer, probe_table_outer,              \
                         non_equi_condition, build_table_outer_exp,         \
                         probe_table_outer_exp, non_equi_condition_exp)     \
    if (build_table_outer == build_table_outer_exp &&                       \
        probe_table_outer == probe_table_outer_exp &&                       \
        non_equi_condition == non_equi_condition_exp) {                     \
        nested_loop_join_table_local<                                       \
            probe_table_outer_exp, build_table_outer_exp,                   \
            non_equi_condition_exp, bodo::PinnableAllocator<std::uint8_t>>( \
            probe_table, build_table, cond_func, false, probe_idxs,         \
            build_idxs, probe_table_matched, *build_table_matched_guard,    \
            build_table_offset);                                            \
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

    join_state->output_buffer->AppendJoinOutput(
        build_table, probe_table, build_idxs, probe_idxs, build_kept_cols,
        probe_kept_cols);
#undef JOIN_TABLE_LOCAL
}

void NestedLoopJoinState::ProcessProbeChunk(
    std::shared_ptr<table_info> probe_table,
    const std::vector<uint64_t>& build_kept_cols,
    const std::vector<uint64_t>& probe_kept_cols) {
    // Unify dictionaries to allow consistent hashing and fast key
    // comparison using indices.
    probe_table = this->UnifyProbeTableDictionaryArrays(probe_table);

    // Bitmask to track matched rows from this probe chunk.
    bodo::vector<uint8_t> probe_table_matched(0, 0);
    if (this->probe_table_outer) {
        probe_table_matched.resize(
            arrow::bit_util::BytesForBits(probe_table->nrows()), 0);
    }

    // Pin the build table matched bitmask.
    auto build_table_matched_guard(bodo::pin(this->build_table_matched));

    // Define the number of rows already processed as 0
    int64_t build_table_offset = 0;
    for (const auto& build_table : *(this->build_table_buffer)) {
        nested_loop_join_local_chunk(
            this, build_table, probe_table, build_kept_cols, probe_kept_cols,
            build_table_matched_guard, probe_table_matched, build_table_offset);
        build_table_offset += build_table->nrows();
    }

    // Add the unmatched probe rows in the probe_outer case:
    if (this->probe_table_outer) {
        bodo::vector<int64_t> build_idxs;
        bodo::vector<int64_t> probe_idxs;
        add_unmatched_rows(probe_table_matched, probe_table->nrows(),
                           probe_idxs, build_idxs,
                           // We always broadcast one of the sides. If the build
                           // side is parallel then either the probe side is
                           // replicated or we broadcast the probe side.
                           this->build_parallel);
        // We can use a dummy chunk from the build table since all build
        // indices are guaranteed to be -1 and hence the actual content
        // of the chunk doesn't matter.
        this->output_buffer->AppendJoinOutput(
            this->build_table_buffer->dummy_output_chunk, probe_table,
            build_idxs, probe_idxs, build_kept_cols, probe_kept_cols);
    }
}

bool nested_loop_join_probe_consume_batch(
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
        // When probe is finalized global is_last has been seen so no need for
        // additional synchronization
        return true;
    }

    // Make is_last global
    is_last =
        join_stream_sync_is_last(is_last, join_state->probe_iter, join_state);

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
            join_state->ProcessProbeChunk(std::move(bcast_probe_chunk),
                                          build_kept_cols, probe_kept_cols);
        }
    } else {
        join_state->ProcessProbeChunk(std::move(in_table), build_kept_cols,
                                      probe_kept_cols);
    }
    if (join_state->build_table_outer && is_last) {
        // Add unmatched rows from build table
        // for outer join

        // define the number of rows already processed as 0
        int64_t build_table_offset = 0;
        bodo::vector<int64_t> build_idxs;
        bodo::vector<int64_t> probe_idxs;
        auto build_table_matched_pin(
            bodo::pin(join_state->build_table_matched));

        for (const auto& build_table : *join_state->build_table_buffer) {
            add_unmatched_rows(
                *build_table_matched_pin, build_table->nrows(), build_idxs,
                probe_idxs,
                !join_state->build_parallel && join_state->probe_parallel,
                build_table_offset);
            join_state->output_buffer->AppendJoinOutput(
                build_table, join_state->dummy_probe_table, build_idxs,
                probe_idxs, build_kept_cols, probe_kept_cols);
            build_table_offset += build_table->nrows();
            build_idxs.clear();
            probe_idxs.clear();
        }
    }
    if (is_last) {
        // Free the build table
        join_state->build_table_buffer.reset();
        // Release memory used by the matched bitmask
        auto build_table_matched_pin(
            bodo::pin(join_state->build_table_matched));
        build_table_matched_pin->resize(0);
        build_table_matched_pin->shrink_to_fit();

        // Finalize the probe side
        join_state->FinalizeProbe();
    }
    join_state->probe_iter++;
    return is_last;
}

bool nested_loop_join_build_consume_batch_py_entry(
    NestedLoopJoinState* join_state, table_info* in_table, bool is_last) {
    try {
        return nested_loop_join_build_consume_batch(
            join_state, std::shared_ptr<table_info>(in_table), is_last);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
    return false;
}
