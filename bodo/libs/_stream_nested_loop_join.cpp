#include "_bodo_common.h"
#include "_distributed.h"
#include "_join.h"
#include "_nested_loop_join_impl.h"
#include "_query_profile_collector.h"
#include "_shuffle.h"
#include "_stream_join.h"

void NestedLoopJoinState::ReportBuildStageMetrics(
    std::vector<MetricBase>& metrics_out) {
    assert(this->build_input_finalized);
    if (this->op_id == -1) {
        return;
    }

    metrics_out.reserve(metrics_out.size() + 32);

    metrics_out.emplace_back(
        StatMetric("block_size_bytes", this->metrics.block_size_bytes, true));
    metrics_out.emplace_back(
        StatMetric("chunk_size_nrows", this->metrics.chunk_size_nrows, true));
    metrics_out.emplace_back(
        TimerMetric("finalize_time", this->metrics.build_finalize_time));
    metrics_out.emplace_back(
        StatMetric("bcast_join", this->metrics.is_build_bcast_join, true));
    metrics_out.emplace_back(
        TimerMetric("bcast_time", this->metrics.build_bcast_time));
    NestedLoopJoinMetrics::stat_t build_table_num_chunks =
        this->build_table_buffer->chunks.size();
    metrics_out.emplace_back(StatMetric("num_chunks", build_table_num_chunks));
    metrics_out.emplace_back(
        TimerMetric("append_time", this->build_table_buffer->append_time));

    // Get and combine dict-builder stats
    DictBuilderMetrics dict_builder_metrics;
    NestedLoopJoinMetrics::stat_t n_dict_builders = 0;
    for (const auto& dict_builder : this->build_table_dict_builders) {
        if (dict_builder == nullptr) {
            continue;
        }
        dict_builder_metrics.add_metrics(dict_builder->GetMetrics());
        n_dict_builders++;
    }
    metrics_out.emplace_back(
        StatMetric("n_dict_builders", n_dict_builders, true));
    metrics_out.emplace_back(
        StatMetric("dict_builders_unify_cache_id_misses",
                   dict_builder_metrics.unify_cache_id_misses));
    metrics_out.emplace_back(
        StatMetric("dict_builders_unify_cache_length_misses",
                   dict_builder_metrics.unify_cache_length_misses));
    metrics_out.emplace_back(
        StatMetric("dict_builders_transpose_filter_cache_id_misses",
                   dict_builder_metrics.transpose_filter_cache_id_misses));
    metrics_out.emplace_back(
        StatMetric("dict_builders_transpose_filter_cache_length_misses",
                   dict_builder_metrics.transpose_filter_cache_length_misses));
    metrics_out.emplace_back(
        TimerMetric("dict_builders_unify_build_transpose_map_time",
                    dict_builder_metrics.unify_build_transpose_map_time));
    metrics_out.emplace_back(
        TimerMetric("dict_builders_unify_transpose_time",
                    dict_builder_metrics.unify_transpose_time));
    metrics_out.emplace_back(
        TimerMetric("dict_builders_unify_string_arr_time",
                    dict_builder_metrics.unify_string_arr_time));
    metrics_out.emplace_back(TimerMetric(
        "dict_builders_transpose_filter_build_transpose_map_time",
        dict_builder_metrics.transpose_filter_build_transpose_map_time));
    metrics_out.emplace_back(
        TimerMetric("dict_builders_transpose_filter_transpose_time",
                    dict_builder_metrics.transpose_filter_transpose_time));
    metrics_out.emplace_back(
        TimerMetric("dict_builders_transpose_filter_string_arr_time",
                    dict_builder_metrics.transpose_filter_string_arr_time));

    JoinState::ReportBuildStageMetrics(metrics_out);
}

void NestedLoopJoinState::FinalizeBuild() {
    time_pt start = start_timer();
    // Finalize any active chunk
    this->build_table_buffer->Finalize();

    // Finalize the min/max before broadcast because it may
    // modify the value of build_parallel.
    if (!this->probe_table_outer) {
        // If this is not an outer probe, finalize the min/max
        // values for each key by shuffling across multiple ranks.
        // This is done before the broadcast handling since
        // the finalization deals with the parallel handling
        // of the accumulated min/max state.
        time_pt start_min_max = start_timer();
        this->FinalizeKeysMinMax();
        this->metrics.build_min_max_finalize_time += end_timer(start_min_max);
    }

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
        HANDLE_MPI_ERROR(
            MPI_Allreduce(MPI_IN_PLACE, &table_size, 1, MPI_INT64_T, MPI_SUM,
                          MPI_COMM_WORLD),
            "NestedLoopJoinState::FinalizeBuild: MPI error on MPI_Allreduce:");
        if (this->force_broadcast || table_size < get_bcast_join_threshold()) {
            this->build_parallel = false;
            this->metrics.is_build_bcast_join = 1;
            time_pt start_bcast = start_timer();
            // calculate the max number of chunks for all partitions
            int64_t n_chunks = this->build_table_buffer->chunks.size();
            HANDLE_MPI_ERROR(
                MPI_Allreduce(MPI_IN_PLACE, &n_chunks, 1, MPI_INT64_T, MPI_MAX,
                              MPI_COMM_WORLD),
                "NestedLoopJoinState::FinalizeBuild: MPI error on "
                "MPI_Allreduce:");
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
            this->metrics.build_bcast_time += end_timer(start_bcast);
        }
    }

    if (this->build_table_outer) {
        auto build_table_matched_pin(bodo::pin(build_table_matched));
        build_table_matched_pin->resize(
            arrow::bit_util::BytesForBits(
                this->build_table_buffer->total_remaining),
            0);
    }

    this->metrics.build_finalize_time += end_timer(start);

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
        if (in_table->nrows() != 0) {
            throw std::runtime_error(
                "nested_loop_join_build_consume_batch: Received non-empty "
                "in_table after the build was already finalized!");
        }
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

    if (!join_state->probe_table_outer) {
        // If this is not an outer probe, use the latest batch
        // to process the min/max values for each build column and
        // update the min_max_values vector. If the column is not
        // involved in an interval join it will be a no-op.
        time_pt start_min_max = start_timer();
        for (size_t col_idx = 0;
             col_idx < join_state->build_table_schema->column_types.size();
             col_idx++) {
            join_state->UpdateKeysMinMax(in_table->columns[col_idx], col_idx);
        }
        join_state->metrics.build_min_max_update_time +=
            end_timer(start_min_max);
    }

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
 * build_table_matched that belongs to previous chunks of the build table
 * buffer.
 * @param[in, out] append_time -- -- Increment this with the time spent in
 * AppendJoinOutput.
 */
void nested_loop_join_local_chunk(
    NestedLoopJoinState* join_state, std::shared_ptr<table_info> build_table,
    std::shared_ptr<table_info> probe_table,
    const std::vector<uint64_t>& build_kept_cols,
    const std::vector<uint64_t>& probe_kept_cols,
    bodo::pin_guard<decltype(NestedLoopJoinState::build_table_matched)>&
        build_table_matched_guard,
    bodo::vector<uint8_t>& probe_table_matched, int64_t build_table_offset,
    NestedLoopJoinMetrics::time_t& append_time) {
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

    time_pt start_append = start_timer();
    join_state->output_buffer->AppendJoinOutput(
        build_table, probe_table, build_idxs, probe_idxs, build_kept_cols,
        probe_kept_cols);
    append_time += end_timer(start_append);
#undef JOIN_TABLE_LOCAL
}

void NestedLoopJoinState::ProcessProbeChunk(
    std::shared_ptr<table_info> probe_table,
    const std::vector<uint64_t>& build_kept_cols,
    const std::vector<uint64_t>& probe_kept_cols) {
    // Unify dictionaries to allow consistent hashing and fast key
    // comparison using indices.
    probe_table = this->UnifyProbeTableDictionaryArrays(probe_table);

    time_pt start = start_timer();
    NestedLoopJoinMetrics::time_t append_time = 0;
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
            build_table_matched_guard, probe_table_matched, build_table_offset,
            append_time);
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
        time_pt start_append = start_timer();
        this->output_buffer->AppendJoinOutput(
            this->build_table_buffer->dummy_output_chunk, probe_table,
            build_idxs, probe_idxs, build_kept_cols, probe_kept_cols);
        append_time += end_timer(start_append);
    }
    this->metrics.probe_compute_matches_time += end_timer(start) - append_time;
}

void NestedLoopJoinState::ReportProbeStageMetrics(
    std::vector<MetricBase>& metrics_out) {
    assert(this->probe_input_finalized);
    if (this->op_id == -1) {
        return;
    }

    metrics_out.reserve(metrics_out.size() + 16);

    metrics_out.emplace_back(
        TimerMetric("global_dict_unification_time",
                    this->metrics.probe_global_dict_unification_time));
    metrics_out.emplace_back(
        StatMetric("bcast_size_bytes", this->metrics.probe_bcast_size_bytes));
    metrics_out.emplace_back(
        TimerMetric("bcast_table_time", this->metrics.probe_bcast_table_time));
    metrics_out.emplace_back(TimerMetric(
        "compute_matches_time", this->metrics.probe_compute_matches_time));
    metrics_out.emplace_back(
        TimerMetric("add_unmatched_build_rows_time",
                    this->metrics.probe_add_unmatched_build_rows_time));

    // Get and combine metrics from dict-builders
    DictBuilderMetrics dict_builder_metrics;
    MetricBase::StatValue n_dict_builders = 0;
    for (const auto& dict_builder : this->probe_table_dict_builders) {
        if (dict_builder == nullptr) {
            continue;
        }
        dict_builder_metrics.add_metrics(dict_builder->GetMetrics());
        n_dict_builders++;
    }
    metrics_out.emplace_back(
        StatMetric("n_dict_builders", n_dict_builders, true));
    metrics_out.emplace_back(
        StatMetric("dict_builders_unify_cache_id_misses",
                   dict_builder_metrics.unify_cache_id_misses));
    metrics_out.emplace_back(
        StatMetric("dict_builders_unify_cache_length_misses",
                   dict_builder_metrics.unify_cache_length_misses));
    metrics_out.emplace_back(
        StatMetric("dict_builders_transpose_filter_cache_id_misses",
                   dict_builder_metrics.transpose_filter_cache_id_misses));
    metrics_out.emplace_back(
        StatMetric("dict_builders_transpose_filter_cache_length_misses",
                   dict_builder_metrics.transpose_filter_cache_length_misses));
    metrics_out.emplace_back(
        TimerMetric("dict_builders_unify_build_transpose_map_time",
                    dict_builder_metrics.unify_build_transpose_map_time));
    metrics_out.emplace_back(
        TimerMetric("dict_builders_unify_transpose_time",
                    dict_builder_metrics.unify_transpose_time));
    metrics_out.emplace_back(
        TimerMetric("dict_builders_unify_string_arr_time",
                    dict_builder_metrics.unify_string_arr_time));
    metrics_out.emplace_back(TimerMetric(
        "dict_builders_transpose_filter_build_transpose_map_time",
        dict_builder_metrics.transpose_filter_build_transpose_map_time));
    metrics_out.emplace_back(
        TimerMetric("dict_builders_transpose_filter_transpose_time",
                    dict_builder_metrics.transpose_filter_transpose_time));
    metrics_out.emplace_back(
        TimerMetric("dict_builders_transpose_filter_string_arr_time",
                    dict_builder_metrics.transpose_filter_string_arr_time));

    JoinState::ReportProbeStageMetrics(metrics_out);
}

/**
 * @brief Synchronize is_last flag for nested loop join's probe. Regular hash
 * join is async now so this is only needed for nested loop join.
 *
 * @param local_is_last Whether we're done on this rank.
 * @param iter Current iteration counter.
 * @param[in] join_state Join state used to get the distributed information
 * and the sync_iter.
 * @return true We don't need to have any more iterations on this rank.
 * @return false We may need to have more iterations on this rank.
 */
static inline bool nested_loop_join_stream_sync_is_last(bool local_is_last,
                                                        const uint64_t iter,
                                                        JoinState* join_state) {
    // We must synchronize if either we have a distributed build or an
    // LEFT/FULL OUTER JOIN where probe is distributed.
    // When build is parallel, every iteration has blocking collectives (bcast
    // and allreduce for outer case) so is_last sync has to be blocking as well
    // to make sure the same number of blocking collectives are called on every
    // rank. When build is not parallel, is_last sync has to be non-blocking
    // since the number of iterations per rank can be variable.
    if (join_state->build_parallel) {
        bool global_is_last = false;
        if (((iter + 1) % join_state->sync_iter) ==
            0) {  // Use iter + 1 to avoid a sync on the first iteration
            HANDLE_MPI_ERROR(
                MPI_Allreduce(&local_is_last, &global_is_last, 1,
                              MPI_UNSIGNED_CHAR, MPI_LAND, MPI_COMM_WORLD),
                "nested_loop_join_stream_sync_is_last: MPI error on "
                "MPI_Allreduce:");
        }
        return global_is_last;
    } else if (join_state->build_table_outer && join_state->probe_parallel) {
        return stream_join_sync_is_last(local_is_last, join_state);
    } else {
        // If we have a broadcast join or a replicated input we don't need to be
        // synchronized because there is no shuffle.
        return local_is_last;
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

    // We only need to take the parallel path if both tables are parallel
    // and the build table wasn't broadcast.
    bool parallel = join_state->build_parallel && join_state->probe_parallel;
    if (parallel) {
        int n_pes, myrank;
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        // Make dictionaries global for broadcast.
        // TODO [BSE-2131] Replace with:
        // 1. Unify locally with Dict-Builder
        // 2. Unify Dict-Builders globally
        // 3. Transpose local to global using known transpose map
        // 4. Broadcast
        // 5. Transpose back from global to local using known transpose map
        time_pt start_dict_unif = start_timer();
        for (size_t i = 0; i < in_table->ncols(); i++) {
            std::shared_ptr<array_info> arr = in_table->columns[i];
            if (arr->arr_type == bodo_array_type::DICT) {
                make_dictionary_global_and_unique(arr, parallel);
            }
        }
        join_state->metrics.probe_global_dict_unification_time +=
            end_timer(start_dict_unif);

        time_pt start_bcast;
        join_state->metrics.probe_bcast_size_bytes +=
            table_local_memory_size(in_table, false);
        for (int p = 0; p < n_pes; p++) {
            start_bcast = start_timer();
            std::shared_ptr<table_info> bcast_probe_chunk = broadcast_table(
                in_table, in_table, nullptr, in_table->ncols(), parallel, p);
            join_state->metrics.probe_bcast_table_time +=
                end_timer(start_bcast);
            join_state->ProcessProbeChunk(std::move(bcast_probe_chunk),
                                          build_kept_cols, probe_kept_cols);
        }
    } else {
        join_state->ProcessProbeChunk(std::move(in_table), build_kept_cols,
                                      probe_kept_cols);
    }

    // Make is_last global
    is_last = nested_loop_join_stream_sync_is_last(
        is_last, join_state->probe_iter, join_state);

    if (join_state->build_table_outer && is_last) {
        // Add unmatched rows from build table
        // for outer join
        time_pt start_add_unmatched = start_timer();
        NestedLoopJoinMetrics::time_t append_time = 0;

        // define the number of rows already processed as 0
        int64_t build_table_offset = 0;
        bodo::vector<int64_t> build_idxs;
        bodo::vector<int64_t> probe_idxs;
        auto build_table_matched_pin(
            bodo::pin(join_state->build_table_matched));

        time_pt start_append;
        for (const auto& build_table : *join_state->build_table_buffer) {
            add_unmatched_rows(
                *build_table_matched_pin, build_table->nrows(), build_idxs,
                probe_idxs,
                !join_state->build_parallel && join_state->probe_parallel,
                build_table_offset);
            start_append = start_timer();
            join_state->output_buffer->AppendJoinOutput(
                build_table, join_state->dummy_probe_table, build_idxs,
                probe_idxs, build_kept_cols, probe_kept_cols);
            append_time += end_timer(start_append);
            build_table_offset += build_table->nrows();
            build_idxs.clear();
            probe_idxs.clear();
        }
        join_state->metrics.probe_add_unmatched_build_rows_time +=
            end_timer(start_add_unmatched) - append_time;
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
