#include "_stream_sort.h"
#include <cstdint>
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

void SortedChunkedTableBuilder::AppendChunk(std::shared_ptr<table_info> chunk,
                                            bool sorted) {
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
        sorted ? std::move(chunk)
               : sort_values_table_local(
                     std::move(chunk), n_key_t, vect_ascending.data(),
                     na_position.data(), dead_keys.data(), false);
    table_heap.push_back({sorted_chunk, n_key_t});
    std::push_heap(table_heap.begin(), table_heap.end(), comp);

    sorted_chunk->unpin();
}

/**
 * Comparator for vectors of sorted elements. Each chunk must be sorted, but the
 * vector as a whole must be sorted in reverse. See MergeChunks for details.
 */
struct VHeapComparator {
    HeapComparator& comp;
    bool operator()(const std::vector<TableAndRange>& a,
                    const std::vector<TableAndRange>& b) {
        return comp(a.back(), b.back());
    }
};

std::vector<TableAndRange> SortedChunkedTableBuilder::MergeChunks(
    std::vector<std::vector<TableAndRange>>&& sorted_chunks) {
    // Reverse the vectors of sorted chunks - this is so that popping chunks off
    // of the list will be efficient. We will iterate over this list in reverse
    // order.
    for (auto& chunks : sorted_chunks) {
        std::reverse(chunks.begin(), chunks.end());
    }

    VHeapComparator vcomp(comp);
    std::make_heap(sorted_chunks.begin(), sorted_chunks.end(), vcomp);

    std::vector<TableAndRange> out_chunks;

    // Move all completed chunks from the CTB into the output. This currently
    // requires pinning the chunks because we need to maintain the range for
    // each table.
    // TODO(aneesh) We should extend ChunkedTableBuilder to do the conversion to
    // TableAndRange internally
    auto FlushChunkedTableBuilder = [&]() {
        while (!sorted_table_builder->chunks.empty()) {
            auto [table, nrows] = sorted_table_builder->PopChunk();
            if (nrows) {
                table->pin();
                out_chunks.push_back({table, n_key_t});
                table->unpin();
            }
        }
    };

    while (sorted_chunks.size() > 1) {
        // find the minimum row
        std::pop_heap(sorted_chunks.begin(), sorted_chunks.end(), vcomp);
        auto& min_vec = sorted_chunks.back();
        std::reference_wrapper<TableAndRange> min = min_vec.back();
        // This table will be unpinned once all rows have been appended to the
        // final output. This does mean that we might be pinning this table more
        // than once, but pinning a pinned object is a noop.
        min.get().table->pin();

        // Consume chunks from min_vec until the smallest row in min_vec (which
        // is the first row of the last chunk since we reversed the input above)
        // is larger than the smallest row from the top of the heap.
        do {
            std::vector<int64_t> row_idx;
            // Loop through rows in the current chunk, selecting them to append
            // while they are smaller than the first row of the next smallest
            // chunk in the heap.
            int64_t offset = min.get().offset;
            int64_t nrows = static_cast<int64_t>(min.get().table->nrows());
            do {
                // TODO(aneesh) this could be replaced by a binary search to
                // find the first element that is largest than the next smallest
                // chunk.
                row_idx.push_back(offset);
                offset++;
                if (offset < nrows) {
                    min.get().UpdateOffset(n_key_t, offset);
                }
            } while (offset < nrows && comp(sorted_chunks[0].back(), min));

            // TODO(aneesh) If row_idx contains all rows in the chunk, we might
            // want to directly append to the output without copying instead.
            // We'd need to first finalize the active chunk and then flush all
            // chunks to the output.
            sorted_table_builder->AppendBatch(min.get().table, row_idx);

            // If we have completely consumed all rows from the current chunk,
            // get the next chunk from the same sorted list.
            if (offset >= nrows) {
                min.get().table->unpin();
                min_vec.pop_back();
                if (!min_vec.empty()) {
                    // we need std::ref because we want to update what min is
                    // referring to, not the contents of min (which is now a
                    // reference to invalid memory)
                    min = std::ref(min_vec.back());
                }
            }
        } while (!min_vec.empty() && comp(sorted_chunks[0].back(), min));

        if (min_vec.empty()) {
            // This vector of chunks is completely consumed, so we can remove it
            // from the heap
            sorted_chunks.pop_back();
        } else {
            // We've updated the minimum row of the vector of chunks, so we need
            // to push it back onto the heap to find the next smallest row.
            std::push_heap(sorted_chunks.begin(), sorted_chunks.end(), vcomp);
        }

        // If we have a chunk we can pop from the sorted_table_builder, we
        // should pop it while it's still fresh in memory so that we don't
        // need to pay a high cost for pinning/unpinning chunks later just
        // to get the min/max value.
        FlushChunkedTableBuilder();
    }

    // Append all unconsumed rows in a chunk to the builder
    auto AppendChunkToBuilder = [&](TableAndRange& chunk) {
        chunk.table->pin();
        std::vector<int64_t> row_idx(chunk.table->nrows() - chunk.offset);
        std::iota(row_idx.begin(), row_idx.end(), chunk.offset);
        sorted_table_builder->AppendBatch(chunk.table, row_idx);
        chunk.table->unpin();
    };

    if (sorted_chunks.size() == 1) {
        auto& chunks = sorted_chunks[0];

        // If every chunk fits into the chunk size, append without copying
        if (std::all_of(chunks.begin(), chunks.end(),
                        [&](TableAndRange& chunk) {
                            return chunk.table->nrows() <= chunk_size;
                        })) {
            // The first chunk might have a non-zero offset, so we might not be
            // able to avoid a copy. However, we know that all other chunks must
            // have a 0 offset.
            if (chunks.back().offset > 0) {
                AppendChunkToBuilder(chunks.back());
                chunks.pop_back();
            }
            // flush all chunks from the CTB so we can directly append all
            // remaining tables to the output.
            sorted_table_builder->FinalizeActiveChunk();
            FlushChunkedTableBuilder();

            // Move chunks into the output
            while (!chunks.empty()) {
                auto& chunk = chunks.back();
                out_chunks.push_back(chunk);
                chunks.pop_back();
            }
        }

        // Copy all remaining chunks into the output
        while (!chunks.empty()) {
            AppendChunkToBuilder(chunks.back());
            chunks.pop_back();
        }
    }

    if (sorted_chunks.size() == 1) {
        for (auto& chunk : sorted_chunks[0]) {
            chunk.table->pin();
            sorted_table_builder->AppendBatch(chunk.table);
            chunk.table->unpin();
        }
    }

    // If we have any leftover chunks, push them into the output
    sorted_table_builder->FinalizeActiveChunk();
    FlushChunkedTableBuilder();

    return out_chunks;
}

std::vector<TableAndRange> SortedChunkedTableBuilder::Finalize() {
    if (table_heap.size() == 0) {
        std::vector<TableAndRange> output;
        return output;
    }

    std::vector<std::vector<TableAndRange>> sorted_chunks;

    constexpr size_t NUM_CHUNKS = 5;
    while (!table_heap.empty()) {
        std::vector<std::vector<TableAndRange>> chunks;
        // Take NUM_CHUNKS chunks at a time and put them each into a vector so
        // that we can call MergeChunks, which expects a vector of vector of
        // chunks. Each inner vector of chunks is expected to be sorted - a
        // vector of a single sorted chunk is by definition, sorted.
        for (size_t i = 0; i < NUM_CHUNKS && !table_heap.empty(); i++) {
            // Get the element off the top of the heap and shrink the heap
            std::pop_heap(table_heap.begin(), table_heap.end(), comp);
            chunks.push_back({table_heap.back()});
            table_heap.pop_back();
        }

        sorted_chunks.emplace_back(MergeChunks(std::move(chunks)));
    }

    while (sorted_chunks.size() > 1) {
        std::vector<std::vector<TableAndRange>> next_sorted_chunks;
        // This loop takes NUM_CHUNKS vectors and merges them into 1 on every
        // iteration.
        for (size_t i = 0; i < sorted_chunks.size(); i += NUM_CHUNKS) {
            std::vector<std::vector<TableAndRange>> merge_input;
            // Collect the next NUM_CHUNKS vectors
            size_t start = i;
            size_t end = std::min(i + NUM_CHUNKS, sorted_chunks.size());
            for (size_t j = start; j < end; j++) {
                merge_input.emplace_back(std::move(sorted_chunks[j]));
            }

            next_sorted_chunks.emplace_back(
                MergeChunks(std::move(merge_input)));
        }

        std::swap(sorted_chunks, next_sorted_chunks);
    }

    return sorted_chunks[0];
}

StreamSortState::StreamSortState(int64_t op_id, int64_t n_key_t,
                                 std::vector<int64_t>&& vect_ascending_,
                                 std::vector<int64_t>&& na_position_,
                                 std::shared_ptr<bodo::Schema> schema_,
                                 bool parallel, size_t chunk_size)
    : op_id(op_id),
      n_key_t(n_key_t),
      vect_ascending(vect_ascending_),
      na_position(na_position_),
      parallel(parallel),
      // Note that builder only stores references to the vectors owned by
      // this object, so we must refer to the instances on this class, not
      // the arguments.
      builder(n_key_t, vect_ascending, na_position, dead_keys, chunk_size),
      schema(std::move(schema_)),
      dummy_output_chunk(alloc_table(schema)) {}

void StreamSortState::ConsumeBatch(std::shared_ptr<table_info> table,
                                   bool parallel, bool is_last) {
    if (phase != StreamSortPhase::BUILD) {
        throw std::runtime_error(
            "Cannot call ConsumeBatch after is_last is set to true");
    }
    // TODO(aneesh) unify dictionaries?
    time_pt start_append = start_timer();
    builder.AppendChunk(table);
    metrics.local_sort_chunk_time += end_timer(start_append);

    if (is_last) {
        time_pt start_finalize = start_timer();
        // Instead of finalizing here we could finalize after we determine the
        // bounds and then instead of building a completely sorted list locally,
        // we can partition the list to build the list of chunks that need to be
        // sent to each rank.
        local_chunks = builder.Finalize();
        metrics.local_sort_time += end_timer(start_finalize);
    }
}

std::shared_ptr<table_info> StreamSortState::GetParallelSortBounds() {
    time_pt start_sampling = start_timer();
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

    metrics.sampling_time += end_timer(start_sampling);
    return bounds;
}

std::vector<std::vector<std::shared_ptr<table_info>>>
StreamSortState::PartitionChunksByRank(int n_pes,
                                       std::shared_ptr<table_info> bounds) {
    assert(static_cast<uint64_t>(n_pes) == bounds->nrows() + 1);

    std::vector<std::vector<std::shared_ptr<table_info>>> rankToChunks(n_pes);

    // some chunks might need to be split across multiple ranks. For those
    // chunks we know that a prefix of the chunk will be the "largest" elements
    // for one rank, and the suffix will be the "smallest" elements for one or
    // more ranks.
    std::vector<TableBuildBuffer> smallestPartialChunks;
    std::vector<TableBuildBuffer> largestPartialChunks;
    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders;
    for (auto& col : schema->column_types) {
        dict_builders.push_back(
            create_dict_builder_for_array(col->copy(), false));
    }
    for (int i = 0; i < n_pes; i++) {
        smallestPartialChunks.emplace_back(schema, dict_builders);
        largestPartialChunks.emplace_back(schema, dict_builders);
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

void StreamSortState::GlobalSort() {
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

    std::shared_ptr<table_info> bounds = GetParallelSortBounds();
    time_pt start_partition = start_timer();
    std::vector<std::vector<std::shared_ptr<table_info>>> rankToChunks =
        PartitionChunksByRank(n_pes, bounds);
    metrics.partition_chunks_time += end_timer(start_partition);

    int64_t num_rounds =
        get_required_num_rounds_for_sort(n_pes, myrank, rankToChunks);

    SortedChunkedTableBuilder global_builder(
        n_key_t, vect_ascending, na_position, dead_keys, builder.chunk_size);

    for (auto& chunk : rankToChunks[myrank]) {
        global_builder.AppendChunk(std::move(chunk), true);
    }

    std::vector<size_t> rankToCurrentChunk(n_pes);

    time_pt start_communication = start_timer();
    for (int64_t round = 0; round < num_rounds; round++) {
        auto [table_to_send, hashes] =
            construct_table_to_send(n_pes, myrank, rankToChunks,
                                    rankToCurrentChunk, dummy_output_chunk);

        // Shuffle all the data
        mpi_comm_info comm_info(table_to_send->columns, hashes, parallel);
        auto collected_table = shuffle_table_kernel(
            std::move(table_to_send), hashes, comm_info, parallel);

        time_pt start_append = start_timer();
        global_builder.AppendChunk(std::move(collected_table));
        metrics.global_append_chunk_time += end_timer(start_append);
    }
    metrics.communication_phase += end_timer(start_communication);

    time_pt start_finalize = start_timer();
    auto out_chunks = global_builder.Finalize();
    metrics.global_sort_time += end_timer(start_finalize);

    for (const auto& chunk : out_chunks) {
        output_chunks.push_back(std::move(chunk.table));
    }
}

std::pair<std::shared_ptr<table_info>, bool> StreamSortState::GetOutput() {
    if (phase != StreamSortPhase::PRODUCE_OUTPUT) {
        throw std::runtime_error(
            "Cannot call GetOutput after all chunks are consumed");
    }

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

void StreamSortState::ReportMetrics() {
    if (this->op_id == -1) {
        return;
    }

    std::vector<MetricBase> metrics;
    metrics.reserve(7);

    metrics.emplace_back(TimerMetric("local_sort_chunk_time",
                                     this->metrics.local_sort_chunk_time));
    metrics.emplace_back(
        TimerMetric("local_sort_time", this->metrics.local_sort_time));

    metrics.emplace_back(
        TimerMetric("sampling_time", this->metrics.sampling_time));
    metrics.emplace_back(TimerMetric("partition_chunks_time",
                                     this->metrics.partition_chunks_time));
    metrics.emplace_back(
        TimerMetric("communication_phase", this->metrics.communication_phase));
    metrics.emplace_back(TimerMetric("global_append_chunk_time",
                                     this->metrics.global_append_chunk_time));
    metrics.emplace_back(
        TimerMetric("global_sort_time", this->metrics.global_sort_time));

    QueryProfileCollector::Default().RegisterOperatorStageMetrics(
        QueryProfileCollector::MakeOperatorStageID(op_id, 0),
        std::move(metrics));
}

StreamSortState* stream_sort_state_init_py_entry(int64_t op_id, int64_t n_key_t,
                                                 int64_t* vect_ascending,
                                                 int64_t* na_position,
                                                 int8_t* arr_c_types,
                                                 int8_t* arr_array_types,
                                                 int n_arrs, bool parallel) {
    try {
        // Copy the per-column configuration into owned vectors
        std::vector<int64_t> vect_ascending_(n_key_t);
        std::vector<int64_t> na_position_(n_key_t);
        for (int64_t i = 0; i < n_key_t; i++) {
            vect_ascending_[i] = vect_ascending[i];
            na_position_[i] = na_position[i];
        }
        std::shared_ptr<bodo::Schema> schema = bodo::Schema::Deserialize(
            std::vector<int8_t>(arr_array_types, arr_array_types + n_arrs),
            std::vector<int8_t>(arr_c_types, arr_c_types + n_arrs));
        auto* state = new StreamSortState(
            op_id, n_key_t, std::move(vect_ascending_), std::move(na_position_),
            std::move(schema), parallel);
        state->phase = StreamSortPhase::BUILD;
        return state;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

bool stream_sort_build_consume_batch_py_entry(StreamSortState* state,
                                              table_info* in_table,
                                              bool is_last) {
    try {
        std::shared_ptr<table_info> table(in_table);
        state->ConsumeBatch(table, state->parallel, is_last);
        if (is_last) {
            // TODO(aneesh) this can be made async - not all communication needs
            // to be done upfront
            state->GlobalSort();
            state->phase = StreamSortPhase::PRODUCE_OUTPUT;
        }
        return is_last;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return true;
    }
}

table_info* stream_sort_product_output_batch_py_entry(StreamSortState* state,
                                                      bool produce_output,
                                                      bool* out_is_last) {
    try {
        auto [output, is_last] = state->GetOutput();
        state->metrics.output_row_count += output->nrows();

        *out_is_last = is_last;
        if (is_last) {
            QueryProfileCollector::Default().SubmitOperatorStageRowCounts(
                QueryProfileCollector::MakeOperatorStageID(state->op_id, 0),
                state->metrics.output_row_count);
            state->ReportMetrics();
            state->phase = StreamSortPhase::INVALID;
        }
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
