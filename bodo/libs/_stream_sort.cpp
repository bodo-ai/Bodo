#include "_stream_sort.h"
#include <numeric>
#include "_array_operations.h"
#include "_bodo_common.h"
#include "_chunked_table_builder.h"
#include "_dict_builder.h"
#include "_shuffle.h"

TableAndRange::TableAndRange(std::shared_ptr<table_info> table, int64_t n_key_t,
                             int64_t offset)
    : table(table), offset(offset) {
    // Note that we assume table is already sorted and pinned
    UpdateOffset(n_key_t, offset);
}

void TableAndRange::UpdateOffset(int64_t n_key_t, int64_t offset) {
    // Assumes table is pinned
    this->offset = offset;
    uint64_t size = table->nrows();
    // get the first and last row of the sorted chunk - note that we only
    // get the columns for the keys since the rnage is only used for comparision
    std::vector<int64_t> row_indices = {offset, (int64_t)size - 1};
    range = RetrieveTable(table, row_indices, n_key_t);
}

void SortedChunkedTableBuilder::AppendChunk(std::shared_ptr<table_info> chunk) {
    if (chunk->nrows() == 0) {
        return;
    }

    if (sorted_table_builder == nullptr) {
        for (auto& col : chunk->columns) {
            // we're passing is_key as false here even for key columns because
            // we don't need the hashes.
            dict_builders.push_back(create_dict_builder_for_array(col, false));
        }

        sorted_table_builder = std::make_unique<ChunkedTableBuilder>(
            chunk->schema(), dict_builders, chunk_size);
    }

    auto sorted_chunk =
        sort_values_table_local(chunk, n_key_t, vect_ascending.data(),
                                na_position.data(), dead_keys.data(), false);
    table_heap.push_back({sorted_chunk, n_key_t});
    std::push_heap(table_heap.begin(), table_heap.end(), comp);

    sorted_chunk->unpin();
}

std::vector<TableAndRange> SortedChunkedTableBuilder::Finalize() {
    std::vector<TableAndRange> output;
    if (table_heap.size() == 0) {
        return output;
    }

    if (table_heap.size() == 1) {
        return std::move(table_heap);
    }

    // TODO(aneesh) this might cause too many pins/unpins, so we might actually
    // want to unpin some number of tables, sort them using
    // timsort (see sort_values_table_local), then merge those lists of tables
    // together - this may reduce the number of pins/unpins needed in the final
    // merge
    while (table_heap.size() > 1) {
        std::pop_heap(table_heap.begin(), table_heap.end(), comp);
        auto& min = table_heap.back();
        min.table->pin();

        std::vector<int64_t> row_idx = {min.offset};
        sorted_table_builder->AppendBatch(min.table, row_idx);

        auto size = min.table->nrows();
        int64_t next = min.offset + 1;
        if (next >= static_cast<int64_t>(size)) {
            // shrink heap
            min.table->unpin();
            table_heap.pop_back();
            continue;
        }
        min.UpdateOffset(n_key_t, next);

        // Push the table back into the heap
        min.table->unpin();
        std::push_heap(table_heap.begin(), table_heap.end(), comp);

        if (!sorted_table_builder->chunks.empty()) {
            // Try to pop immediately because the active chunk should still be
            // in memory.
            auto [chunk, _] = sorted_table_builder->PopChunk();
            if (chunk->nrows() > 0) {
                output.push_back({chunk, n_key_t});
            }
        }
    }

    // For the remaining portion of the last table, append the whole table at
    // once
    table_heap[0].table->pin();
    std::vector<bool> append_rows(
        table_heap[0].table->nrows() - table_heap[0].offset, true);
    sorted_table_builder->AppendBatch(table_heap[0].table, append_rows,
                                      table_heap[0].offset);

    sorted_table_builder->Finalize();
    table_heap.clear();

    while (!sorted_table_builder->empty()) {
        auto [chunk, _] = sorted_table_builder->PopChunk();
        output.push_back({chunk, n_key_t});
    }
    return output;
}

bool StreamSortState::consume_batch(std::shared_ptr<table_info> table,
                                    bool parallel, bool is_last) {
    if (phase == StreamSortPhase::PRE_BUILD) {
        // TODO(aneesh) fix arrow_buffer_to_bodo - currently the pool isn't
        // stored in the dtor_info, so pinning/unpinning semi-structured data
        // after calling sort_values_table_local will crash. Remove this code
        // after fixing that.
        for (auto& col : table->columns) {
            switch (col->arr_type) {
                case bodo_array_type::STRUCT:
                case bodo_array_type::MAP:
                case bodo_array_type::ARRAY_ITEM: {
                    throw std::runtime_error("Not implemented");
                }
                default: {
                    // allow all other types
                }
            }
        }

        // Populate the dummy output with the correct types
        dummy_output_chunk =
            alloc_table_like(table, /*reuse_dictionaries*/ false);

        phase = StreamSortPhase::BUILD;
    }
    builder.AppendChunk(table);
    if (is_last) {
        local_chunks = builder.Finalize();
        phase = StreamSortPhase::PRODUCE_OUTPUT;
    }
    return is_last;
}

std::shared_ptr<table_info> StreamSortState::get_parallel_sort_bounds() {
    int64_t n_local = std::accumulate(
        local_chunks.begin(), local_chunks.end(), 0,
        [](int64_t acc, const TableAndRange& tableAndRange) {
            return acc + static_cast<int64_t>(tableAndRange.table->nrows());
        });
    int64_t n_total = n_local;
    if (parallel) {
        MPI_Allreduce(&n_local, &n_total, 1, MPI_LONG_LONG_INT, MPI_SUM,
                      MPI_COMM_WORLD);
    }

    int n_pes, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    // Compute samples from the locally sorted table.
    // (Filled on rank 0, empty on all other ranks)
    int64_t n_loc_sample =
        get_num_samples_from_local_table(n_pes, n_total, n_local);
    auto sample_idxs = get_sample_selection_vector(n_local, n_loc_sample);

    // We can't directly use the indices we got above because we have a
    // collection of unpinned chunks instead of a pinned contiguous table.
    // We need to determine which indices belong to which chunk.
    std::vector<std::vector<int64_t>> indices_per_chunk(local_chunks.size());
    int64_t cursor = 0;
    size_t idx = 0;
    for (size_t chunk = 0; chunk < local_chunks.size(); chunk++) {
        int64_t next_cursor = cursor + local_chunks[chunk].table->nrows();
        while (idx < sample_idxs.size() && sample_idxs[idx] < next_cursor) {
            indices_per_chunk[chunk].push_back(sample_idxs[idx] - cursor);
            idx++;
        }
        cursor = next_cursor;
        if (idx >= sample_idxs.size()) {
            break;
        }
    }
    assert(idx == sample_idxs.size());

    // Retrieve the data, pinning and unpinning as we go
    std::vector<std::shared_ptr<table_info>> local_sample_chunks;
    for (size_t i = 0; i < local_chunks.size(); i++) {
        const auto& idxs = indices_per_chunk[i];
        if (idxs.empty()) {
            continue;
        }
        local_chunks[i].table->pin();
        local_sample_chunks.push_back(
            RetrieveTable(local_chunks[i].table, idxs, n_key_t));
        local_chunks[i].table->unpin();
    }

    auto local_samples = concat_tables(local_sample_chunks);
    local_sample_chunks.clear();

    // Collecting all samples globally
    bool all_gather = false;
    std::shared_ptr<table_info> all_samples =
        gather_table(std::move(local_samples), n_key_t, all_gather, parallel);

    // Compute split bounds from the samples.
    // Output is broadcasted to all ranks.
    std::shared_ptr<table_info> bounds = compute_bounds_from_samples(
        std::move(all_samples), dummy_output_chunk, n_key_t,
        vect_ascending.data(), na_position.data(), myrank, n_pes, parallel);

    return bounds;
}

StreamSortState* stream_sort_state_init_py_entry(int64_t n_key_t,
                                                 int64_t* vect_ascending,
                                                 int64_t* na_position) {
    // Copy the per-column configuration into owned vectors
    std::vector<int64_t> vect_ascending_(n_key_t);
    std::vector<int64_t> na_position_(n_key_t);
    for (int64_t i = 0; i < n_key_t; i++) {
        vect_ascending_[i] = vect_ascending[i];
        na_position_[i] = na_position[i];
    }
    auto* state = new StreamSortState(n_key_t, std::move(vect_ascending_),
                                      std::move(na_position_));
    state->phase = StreamSortPhase::PRE_BUILD;
    return state;
}

bool stream_sort_build_consume_batch_py_entry(StreamSortState* state,
                                              table_info* in_table,
                                              bool parallel, bool is_last) {
    try {
        std::shared_ptr<table_info> table(in_table);
        return state->consume_batch(table, parallel, is_last);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return true;
    }
}

table_info* stream_sort_product_output_batch_py_entry(StreamSortState* state,
                                                      bool produce_output,
                                                      bool parallel,
                                                      bool* out_is_last) {
    if (!parallel) {
        std::shared_ptr<table_info> output = nullptr;
        if (state->output_idx < state->output_chunks.size()) {
            std::swap(output, state->output_chunks[state->output_idx]);
            state->output_idx++;
        } else {
            *out_is_last = true;
            output = state->dummy_output_chunk;
        }
    } else {
        int n_pes, myrank;
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        // most of this needs to be moved to state
        auto bounds = state->get_parallel_sort_bounds();
        std::vector<uint32_t> chunkToStartRank;
        std::vector<size_t> rankToFirstChunk(n_pes);
        for (uint32_t rank_id = 0; rank_id < static_cast<uint32_t>(n_pes);
             rank_id++) {
            rankToFirstChunk[rank_id] = std::numeric_limits<uint32_t>::max();
        }
        uint32_t rank_id = 0;
        for (size_t chunk_id = 0; chunk_id < state->local_chunks.size();
             chunk_id++) {
            const auto& tableAndRange = state->local_chunks[chunk_id];
            // Compare the bounds with the ranges to determine which rank a
            // chunk will start at
            while (KeyComparisonAsPython(
                state->n_key_t, state->vect_ascending.data(), bounds->columns,
                0, rank_id, tableAndRange.range->columns, 0, 0,
                state->na_position.data())) {
                rank_id++;
            }
            rankToFirstChunk[rank_id] =
                std::min(rankToFirstChunk[rank_id], chunk_id);
            chunkToStartRank.push_back(rank_id);
        }

        std::vector<size_t> rankToidx = rankToFirstChunk;
        // send `n_pes` chunks to every host and return
    }
    return nullptr;
}

void delete_stream_sort_state(StreamSortState* state) { delete state; }

PyMODINIT_FUNC PyInit_stream_sort_cpp(void) {
    PyObject* m;
    MOD_DEF(m, "stream_sort_cpp", "No docs", NULL);
    if (m == NULL) {
        return NULL;
    }

    bodo_common_init();

    SetAttrStringFromVoidPtr(m, stream_sort_state_init_py_entry);
    SetAttrStringFromVoidPtr(m, stream_sort_build_consume_batch_py_entry);
    SetAttrStringFromVoidPtr(m, stream_sort_product_output_batch_py_entry);
    SetAttrStringFromVoidPtr(m, delete_stream_sort_state);
    return m;
}
