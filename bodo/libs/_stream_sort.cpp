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
        // Instead of finalizing here we could finalize after we determine the
        // bounds and then instead of building a completely sorted list locally,
        // we can partition the list to build the list of chunks that need to be
        // sent to each rank.
        local_chunks = builder.Finalize();
        phase = StreamSortPhase::GLOBAL_SORT;
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

    auto local_samples = local_sample_chunks.empty()
                             ? dummy_output_chunk
                             : concat_tables(local_sample_chunks);
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

std::vector<std::vector<std::shared_ptr<table_info>>>
StreamSortState::partition_chunks_by_rank(int n_pes,
                                          std::shared_ptr<table_info> bounds) {
    assert(n_pes == bounds->nrows() + 1);

    std::vector<std::vector<std::shared_ptr<table_info>>> rankToChunks(n_pes);

    // some chunks might need to be split across multiple ranks. For those
    // chunks we know that a prefix of the chunk will be the "largest" elements
    // for one rank, and the suffix will be the "smallest" elements for one or
    // more ranks.
    std::vector<TableBuildBuffer> smallestPartialChunks;
    std::vector<TableBuildBuffer> largestPartialChunks;
    const std::shared_ptr<table_info>& reftable = local_chunks[0].table;
    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders;
    for (auto& col : reftable->columns) {
        dict_builders.push_back(create_dict_builder_for_array(col, false));
    }
    for (int i = 0; i < n_pes; i++) {
        smallestPartialChunks.emplace_back(reftable->schema(), dict_builders);
        largestPartialChunks.emplace_back(reftable->schema(), dict_builders);
    }

    uint32_t rank_id = 0;
    for (size_t chunk_id = 0; chunk_id < local_chunks.size(); chunk_id++) {
        const auto& tableAndRange = local_chunks[chunk_id];
        // Compare the bounds with the ranges to determine which rank a
        // chunk will start at
        while (rank_id < uint32_t(n_pes - 1) &&
               KeyComparisonAsPython(
                   n_key_t, vect_ascending.data(), bounds->columns, 0, rank_id,
                   tableAndRange.range->columns, 0, 0, na_position.data())) {
            rank_id++;
        }

        // if max(table) < bounds[rank_id] we can send all of table
        if (KeyComparisonAsPython(
                n_key_t, vect_ascending.data(), tableAndRange.range->columns, 0,
                1, bounds->columns, 0, rank_id, na_position.data())) {
            rankToChunks[rank_id].push_back(tableAndRange.table);
        } else {
            // We need to split this chunk up
            tableAndRange.table->pin();
            uint32_t start_rank_id = rank_id;
            for (size_t row = 0; row < tableAndRange.table->nrows(); row++) {
                while ((rank_id < uint32_t(n_pes - 1)) &&
                       KeyComparisonAsPython(n_key_t, vect_ascending.data(),
                                             bounds->columns, 0, rank_id,
                                             tableAndRange.table->columns, 0,
                                             row, na_position.data())) {
                    rank_id++;
                }
                TableBuildBuffer& builder =
                    (rank_id == start_rank_id) ? largestPartialChunks[rank_id]
                                               : smallestPartialChunks[rank_id];
                builder.ReserveTable(tableAndRange.table);
                builder.AppendRowKeys(tableAndRange.table, row,
                                      tableAndRange.table->columns.size());
            }
            tableAndRange.table->unpin();
        }
    }

    for (int i = 0; i < n_pes; i++) {
        if (smallestPartialChunks[i].data_table->nrows()) {
            rankToChunks[i].insert(
                rankToChunks[i].begin(),
                std::move(smallestPartialChunks[i].data_table));
        }
        if (largestPartialChunks[i].data_table->nrows()) {
            rankToChunks[i].emplace_back(
                std::move(largestPartialChunks[i].data_table));
        }
    }

    return rankToChunks;
}

/**
 * @brief Construct inputs to shuffle_table_kernel by combining at most one
 * chunk per rank into a single table
 *
 * @param n_pes number of rank
 * @param myrank local rank
 * @param rankToChunks vector of length n_pes with one vector of chunks per
 * rank. Note that all chunks are unpinned.
 * @param rankToCurrentChunk for each rank, index into rankToChunks[rank] to
 * determine which chunk to send.
 * @param dummy_output_chunk empty table to use if no chunks are avalible to
 * send
 *
 * @return pair of table and hashes mapping rows to rank to input into
 * shuffle_table_kernel
 */
std::pair<std::shared_ptr<table_info>, std::shared_ptr<uint32_t[]>>
construct_table_to_send(
    int n_pes, int myrank,
    std::vector<std::vector<std::shared_ptr<table_info>>>& rankToChunks,
    std::vector<size_t>& rankToCurrentChunk,
    std::shared_ptr<table_info> dummy_output_chunk) {
    // Tables that are being sent in this round
    std::vector<std::shared_ptr<table_info>> tables_to_send;
    // Once we concat the list above, we need to know which range of rows
    // will be sent to which ranks. We maintain a list of row offsets into
    // the final table for this.
    // offsets[i] stores the index of the first row to send to rank i, and
    // the length of the list will be appended to the end, so the number of
    // rows to send to rank i will always be offsets[i + 1] - offsets[i].
    std::vector<int64_t> offsets;

    // Track how many rows will be the table after concat
    int64_t cursor = 0;
    for (int i = 0; i < n_pes; i++) {
        std::shared_ptr<table_info> table_to_send;

        if (i != myrank) {
            if (rankToCurrentChunk[i] < rankToChunks[i].size()) {
                std::swap(table_to_send,
                          rankToChunks[i][rankToCurrentChunk[i]]);
                rankToCurrentChunk[i]++;
                table_to_send->pin();
            }
        }

        offsets.push_back(cursor);
        if (table_to_send) {
            tables_to_send.push_back(table_to_send);
            cursor += table_to_send->nrows();
        }
    }
    offsets.push_back(cursor);

    auto table_to_send = tables_to_send.empty() ? dummy_output_chunk
                                                : concat_tables(tables_to_send);

    for (const auto& table : tables_to_send) {
        table->unpin();
    }
    tables_to_send.clear();

    // TODO(aneesh) Ideally we'd use point-to-point communication and just
    // sent tables directly to the rank they're destined for without setting
    // up the hashes/using shuffle_table_kernel
    std::shared_ptr<uint32_t[]> hashes =
        std::make_unique<uint32_t[]>(offsets.back());
    cursor = 0;
    for (int i = 0; i < n_pes; i++) {
        int64_t num_elems = offsets[i + 1] - offsets[i];
        for (int j = 0; j < num_elems; j++) {
            hashes[cursor] = i;
            cursor++;
        }
    }

    return std::make_pair(std::move(table_to_send), std::move(hashes));
}

/**
 * We need to know how many rounds of communication we need - this is determined
 * by the largest count of chunks that need to be sent between two hosts.
 *
 * @param n_pes number of rank
 * @param myrank local rank
 * @param rankToChunks vector of length n_pes with one vector of chunks per
 * rank. Note that all chunks are unpinned.
 *
 * @return the number of communication rounds required
 */
int64_t get_required_num_rounds_for_sort(
    int n_pes, int myrank,
    std::vector<std::vector<std::shared_ptr<table_info>>>& rankToChunks) {
    int64_t local_num_rounds = 0;
    std::vector<int64_t> send_counts(n_pes, 0);
    for (int i = 0; i < n_pes; i++) {
        if (i == myrank) {
            continue;
        }
        send_counts[i] = rankToChunks[i].size();
        local_num_rounds = std::max(local_num_rounds, send_counts[i]);
    }
    int64_t num_rounds = 0;
    MPI_Allreduce(&local_num_rounds, &num_rounds, 1, MPI_INT64_T, MPI_MAX,
                  MPI_COMM_WORLD);

    return num_rounds;
}

void StreamSortState::global_sort() {
    if (!parallel) {
        // Move the tables from the sorted local chunks to the output
        for (const auto& chunk : local_chunks) {
            output_chunks.push_back(chunk.table);
        }
        local_chunks.clear();
        return;
    }

    int n_pes, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    auto bounds = get_parallel_sort_bounds();
    auto rankToChunks = partition_chunks_by_rank(n_pes, bounds);

    auto num_rounds =
        get_required_num_rounds_for_sort(n_pes, myrank, rankToChunks);

    SortedChunkedTableBuilder global_builder(
        n_key_t, vect_ascending, na_position, dead_keys, builder.chunk_size);

    for (auto& chunk : rankToChunks[myrank]) {
        global_builder.AppendChunk(std::move(chunk));
    }

    std::vector<size_t> rankToCurrentChunk(n_pes);

    for (int64_t round = 0; round < num_rounds; round++) {
        auto [table_to_send, hashes] =
            construct_table_to_send(n_pes, myrank, rankToChunks,
                                    rankToCurrentChunk, dummy_output_chunk);

        // Shuffle all the data
        mpi_comm_info comm_info(table_to_send->columns, hashes, parallel);
        auto collected_table = shuffle_table_kernel(
            std::move(table_to_send), hashes, comm_info, parallel);
        auto sorted_chunk = sort_values_table_local(
            std::move(collected_table), n_key_t, vect_ascending.data(),
            na_position.data(), dead_keys.data(), false);
        global_builder.AppendChunk(std::move(sorted_chunk));
    }

    auto out_chunks = global_builder.Finalize();
    for (const auto& chunk : out_chunks) {
        output_chunks.push_back(std::move(chunk.table));
    }
}

std::pair<std::shared_ptr<table_info>, bool> StreamSortState::get_output() {
    std::shared_ptr<table_info> output = nullptr;
    bool out_is_last = false;
    if (output_idx < output_chunks.size()) {
        std::swap(output, output_chunks[output_idx]);
        output_idx++;
    } else {
        out_is_last = true;
        output = dummy_output_chunk;
    }
    return std::make_pair(output, out_is_last);
}

StreamSortState* stream_sort_state_init_py_entry(int64_t n_key_t,
                                                 int64_t* vect_ascending,
                                                 int64_t* na_position) {
    try {
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
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
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
    try {
        // TODO(aneesh) this can be made async - not all communication needs to
        // be done upfront
        if (state->phase == StreamSortPhase::GLOBAL_SORT) {
            state->global_sort();
            state->phase = StreamSortPhase::PRODUCE_OUTPUT;
        }

        auto [output, is_last] = state->get_output();
        *out_is_last = is_last;
        return new table_info(*output);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
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
