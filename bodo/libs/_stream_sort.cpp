#include "_stream_sort.h"
#include <cstdint>
#include <numeric>
#include "_array_operations.h"
#include "_bodo_common.h"
#include "_chunked_table_builder.h"
#include "_dict_builder.h"
#include "_memory_budget.h"
#include "_shuffle.h"
#include "_stream_shuffle.h"
#include "_table_builder_utils.h"

/**
 * @brief Helper function to get num_chunks for simpler initialization.
 * It is the maximum number of chunks we will pin simultaneously during sort.
 * TODO : revisit after async is done as then in AppendChunk not every rank will
 * send a chunk.
 *
 * @return Chunk number = min(# ranks, 100)
 */
size_t GetOptimalChunkNumber() {
    int npes{0};
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    return static_cast<size_t>(
        std::max(2, std::min(SORT_OPERATOR_MAX_CHUNK_NUMBER, npes)));
}

/**
 * @brief Helper function to get chunk_size.
 * Chunk size is number of rows per chunk stored in our state
 *
 * @param bytes_per_row Number of bytes required for each row
 * @param mem_budget_bytes Budget from operator comptroller
 * @param num_chunks Upper bound of number of chunks pinned simultaneously
 * @param default_chunk_size If bytes_per_row is not set (value
 * of -1) we use default chunk size of 4096. Otherwise we set it with
 * min of 4096 and max chunk size we can afford under Operator budget
 * (Operator budget >= chunk_size (# rows) * number_chunks * bytes_per_row)
 * @return new chunk size
 */
size_t GetChunkSize(int64_t bytes_per_row, uint64_t mem_budget_bytes,
                    size_t num_chunks, size_t default_chunk_size) {
    // TODO(aneesh) change chunksize and well as number of chunks
    if (mem_budget_bytes > 0 && bytes_per_row > 0) {
        return std::max(static_cast<size_t>(mem_budget_bytes) /
                            (num_chunks * static_cast<size_t>(bytes_per_row)),
                        static_cast<size_t>(4096));
    }
    return default_chunk_size;
}

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

bool SortedChunkedTableBuilder::Compare(std::shared_ptr<table_info> table1,
                                        size_t row1,
                                        std::shared_ptr<table_info> table2,
                                        size_t row2) const {
    // TODO(aneesh) we should either templetize KeyComparisonAsPython or move to
    // something like converting rows to bitstrings for faster comparision.
    return KeyComparisonAsPython(n_key_t, vect_ascending.data(),
                                 table1->columns, row1, table2->columns, row2,
                                 na_position.data());
}

// For debugging purposes only
std::ostream& operator<<(std::ostream& os, const TableAndRange& obj) {
    os << "Offset: " << obj.offset << ' ';
    os << "rows: " << obj.table->nrows() << std::endl;
    DEBUG_PrintColumn(os, obj.table->columns[0]);
    return os;
}

/**
 * @brief Helper function to unify dictionaries with the provided dict_builders.
 * Returns a table with all dict encoded columns unified.
 *
 * @param in_table input table
 * @param dict_builders dictionary builders to unify with
 * @return table with unified dict data
 */
std::shared_ptr<table_info> UnifyDictionaryArrays(
    const std::shared_ptr<table_info>&& in_table,
    const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders,
    bool unify_empty = false) {
    std::vector<std::shared_ptr<array_info>> out_arrs;
    out_arrs.reserve(in_table->ncols());
    for (size_t i = 0; i < in_table->ncols(); i++) {
        std::shared_ptr<array_info>& in_arr = in_table->columns[i];
        std::shared_ptr<array_info> out_arr;
        if (dict_builders[i] == nullptr) {
            out_arr = in_arr;
        } else {
            out_arr = dict_builders[i]->UnifyDictionaryArray(in_arr, true);
        }
        out_arrs.emplace_back(out_arr);
    }

    return std::make_shared<table_info>(out_arrs);
}

void SortedChunkedTableBuilder::AppendChunk(std::shared_ptr<table_info> chunk) {
    if (chunk->nrows() == 0) {
        return;
    }

    chunk = UnifyDictionaryArrays(std::move(chunk), dict_builders);
    auto sorted_chunk = sort_values_table_local(
        std::move(chunk), n_key_t, vect_ascending.data(), na_position.data(),
        dead_keys.data(), false);
    // TODO(aneesh) we could truncate the table here if limit+offset <
    // chunk_size
    input_chunks.push_back({sorted_chunk, n_key_t});
    sorted_chunk->unpin();
}

/**
 * Comparator for lists of sorted elements. Each chunk must be sorted, but
 * the list as a whole must be sorted in reverse. See MergeChunks for
 * details.
 */
struct VHeapComparator {
    const SortedChunkedTableBuilder::HeapComparator& comp;
    bool operator()(const std::deque<TableAndRange>& a,
                    const std::deque<TableAndRange>& b) const {
        return comp(a.front(), b.front());
    }
};

/**
 * Some invariants maintained for correctness:
 * PRECONDITION: Each sorted_chunks[i] is globally sorted (all of
 * sorted_chunks[i][j] <= sorted_chunks[i][j + 1]) and also locally sorted
 * PRECONDITION: is_last can be True only when limitoffsetflag is True
 * If is_last is false, we will keep all top [0, limit + offset) rows as they
 * are all potentially useful rows
 * If is_last is true, this is the final call to MergeChunks and thus we
 * only keep rows from [offset, limit + offset)
 */
template <bool is_last>
std::deque<TableAndRange> SortedChunkedTableBuilder::MergeChunks(
    std::vector<std::deque<TableAndRange>>&& sorted_chunks) {
    auto sortlimit = sortlimits.has_value()
                         ? (std::make_optional(SortLimits(sortlimits->limit,
                                                          sortlimits->offset)))
                         : (std::nullopt);
    ChunkedTableAndRangeBuilder out_table_builder(
        n_key_t, schema, dict_builders, chunk_size,
        DEFAULT_MAX_RESIZE_COUNT_FOR_VARIABLE_SIZE_DTYPES, pool, mm);
    VHeapComparator vcomp(comp);
    std::make_heap(sorted_chunks.begin(), sorted_chunks.end(), vcomp);

    auto [num_chunks, total_bytes] = std::accumulate(
        sorted_chunks.begin(), sorted_chunks.end(),
        std::pair<size_t, int64_t>(0, 0),
        [&](auto acc, const std::deque<TableAndRange>& chunks) {
            acc.first += chunks.size();
            for (auto& chunk : chunks) {
                acc.second += table_local_memory_size(chunk.table, false, true);
            }
            return acc;
        });

    // Note that we require the table to fit in half the budget because sort
    // will do a copy of the data. Since the sort is effectively implemented as
    // a RetrieveTable call, we know that we will need at most twice the size of
    // the table.
    // TODO: Optimize this in-memory case.
    // https://github.com/Bodo-inc/Bodo/pull/8156#discussion_r1706300054
    if (!sortlimit &&
        static_cast<uint64_t>(total_bytes) < mem_budget_bytes / 2) {
        // This is an optimization for when the table fits in memory where we
        // concat all the chunks into a single table and sort the combined
        // table. This avoids the extra overhead incurred by the MergeChunks
        // path.
        std::vector<std::shared_ptr<table_info>> tables;
        tables.reserve(num_chunks);
        for (auto& chunks : sorted_chunks) {
            for (auto& chunk : chunks) {
                chunk.table->pin();
                tables.push_back(chunk.table);
            }
        }

        auto table = concat_tables(std::move(tables), dict_builders);

        // Just to be safe, unpin all input tables before dropping refs
        for (auto& chunks : sorted_chunks) {
            for (auto& chunk : chunks) {
                chunk.table->unpin();
            }
        }
        sorted_chunks.clear();

        // TODO(aneesh) this doesn't require concatting all tables -
        // sort_values_table_local internally uses a custom comparator to
        // sort a list of indices into a table and then does a RetrieveTable
        // call. We could modify that behavior so instead all the indices
        // map into multiple tables, but we then need to also create a
        // special version of RetrieveTable that can combine indices from
        // multiple tables together.
        auto out_idxs = sort_values_table_local_get_indices(
            table, n_key_t, vect_ascending.data(), na_position.data(), false, 0,
            table->nrows());
        uint64_t n_cols = table->ncols();
        std::vector<uint64_t> col_inds;
        for (int64_t i = 0; i < n_key_t; i++) {
            if (!dead_keys.empty() && dead_keys[i]) {
                // If this is the last reference to this
                // table, we can safely release reference (and potentially
                // memory if any) for the dead keys at this point.
                reset_col_if_last_table_ref(table, i);
            } else {
                col_inds.push_back(i);
            }
        }
        for (uint64_t i = n_key_t; i < n_cols; i++) {
            col_inds.push_back(i);
        }
        out_table_builder.AppendBatch(table, out_idxs, col_inds);
    }

    while (sorted_chunks.size() > 1 &&
           (!sortlimit.has_value() || sortlimit.value().sum() > 0)) {
        // No queue should ever be empty - empty queues should be removed
        // entirely
        assert(std::all_of(sorted_chunks.begin(), sorted_chunks.end(),
                           [&](auto& chunks) { return !chunks.empty(); }));
        // find the minimum row
        std::pop_heap(sorted_chunks.begin(), sorted_chunks.end(), vcomp);
        auto& min_vec = sorted_chunks.back();
        std::reference_wrapper<TableAndRange> min = min_vec.front();
        // This table will be unpinned once all rows have been appended to
        // the final output. This does mean that we might be pinning this
        // table more than once, but pinning a pinned object is a noop.
        min.get().table->pin();

        // Consume chunks from min_vec until the smallest row in min_vec
        // is larger than the smallest row from the top of the heap.
        do {
            std::vector<int64_t> row_idx;
            // Loop through rows in the current chunk, selecting them to append
            // while they are smaller than the first row of the next smallest
            // chunk in the heap.
            int64_t offset = min.get().offset;
            int64_t nrows = static_cast<int64_t>(min.get().table->nrows());
            if (sortlimit.has_value()) {
                nrows = std::min(nrows, static_cast<int64_t>(
                                            offset + sortlimit.value().sum()));
            }
            if (!Compare(sorted_chunks[0].front().range, RANGE_MIN,
                         min.get().range, RANGE_MAX)) {
                // We can append the whole table from offset onwards. If the
                // offset is 0, then we just append the entire table
                int64_t start_index = offset + (is_last && sortlimit.has_value()
                                                    ? sortlimit.value().offset
                                                    : 0);
                if (start_index == 0 &&
                    nrows == static_cast<int64_t>(min.get().table->nrows())) {
                    // TODO(aneesh) It would be nice to do this without
                    // copying.
                    out_table_builder.AppendBatch(min.get().table);
                } else if (start_index < nrows) {
                    row_idx.resize(nrows - start_index);
                    std::iota(row_idx.begin(), row_idx.end(), start_index);
                    out_table_builder.AppendBatch(min.get().table, row_idx);
                }
                if (sortlimit.has_value())
                    sortlimit.value() -= nrows - offset;
                offset = nrows;
            } else {
                do {
                    // TODO(aneesh) this could be replaced by a binary
                    // search to find the first element that is largest than
                    // the next smallest chunk.
                    if (!sortlimit.has_value())
                        row_idx.push_back(offset);
                    else {
                        if (!is_last || sortlimit.value().offset == 0) {
                            row_idx.push_back(offset);
                        }
                        sortlimit.value() -= 1;
                    }
                    offset++;
                    if (offset < nrows) {
                        min.get().UpdateOffset(n_key_t, offset);
                    }
                } while (
                    offset < nrows && comp(sorted_chunks[0].front(), min) &&
                    (!sortlimit.has_value() || sortlimit.value().sum() > 0));

                // TODO(aneesh) If row_idx contains all rows in the chunk,
                // we might want to directly append to the output without
                // copying instead. We'd need to first finalize the active
                // chunk and then flush all chunks to the output.
                out_table_builder.AppendBatch(min.get().table, row_idx);
            }

            // If we have completely consumed all rows from the current
            // chunk, get the next chunk from the same sorted list.
            if (offset >= nrows ||
                (sortlimit.has_value() && sortlimit.value().sum() == 0)) {
                min.get().table->unpin();
                min_vec.pop_front();
                if (!min_vec.empty()) {
                    // we need std::ref because we want to update what min
                    // is referring to, not the contents of min (which is
                    // now a reference to invalid memory)
                    min = std::ref(min_vec.front());
                }
            }
        } while (!min_vec.empty() && comp(sorted_chunks[0].front(), min) &&
                 (!sortlimit.has_value() || sortlimit.value().sum() > 0));

        if (min_vec.empty()) {
            // This vector of chunks is completely consumed, so we can
            // remove it from the heap
            sorted_chunks.pop_back();
        } else {
            // We've updated the minimum row of the vector of chunks, so we
            // need to push it back onto the heap to find the next smallest
            // row.
            std::push_heap(sorted_chunks.begin(), sorted_chunks.end(), vcomp);
        }
    }

    // Append all unconsumed rows in a chunk to the builder
    auto AppendChunkToBuilder = [&](TableAndRange& chunk) {
        if (sortlimit.has_value() && is_last) {
            size_t row_size =
                chunk.table->nrows() - static_cast<size_t>(chunk.offset);
            if (row_size <= sortlimit.value().offset) {
                sortlimit.value() -= row_size;
                return;
            }
            chunk.UpdateOffset(n_key_t,
                               chunk.offset + sortlimit.value().offset);
            sortlimit.value().offset = 0;
        }
        chunk.table->pin();
        size_t row_size =
            chunk.table->nrows() - static_cast<size_t>(chunk.offset);
        if (sortlimit.has_value()) {
            row_size = std::min(sortlimit.value().sum(), row_size);
            sortlimit.value() -= row_size;
        }
        if (row_size > 0) {
            std::vector<int64_t> row_idx(row_size);
            std::iota(row_idx.begin(), row_idx.end(), chunk.offset);
            out_table_builder.AppendBatch(chunk.table, row_idx);
        }
        chunk.table->unpin();
    };

    if (sorted_chunks.size() == 1) {
        auto& chunks = sorted_chunks[0];

        // The first chunk might have a non-zero offset, so we might not be able
        // to avoid a copy. However, we know that all other chunks must have a 0
        // offset.
        if (chunks.front().offset > 0) {
            AppendChunkToBuilder(chunks.front());
            chunks.pop_front();
        }

        // If every remaining chunk fits into the chunk size, append without
        // copying
        if (std::all_of(chunks.begin(), chunks.end(),
                        [&](TableAndRange& chunk) {
                            return chunk.table->nrows() <= chunk_size;
                        })) {
            // flush all chunks from the CTB so we can directly append all
            // remaining tables to the output.
            out_table_builder.FinalizeActiveChunk();

            // Move chunks into the output
            while (!chunks.empty()) {
                // TODO(aneesh) make a cleaner API for this. This is technically
                // unsafe as it makes internal state, such as total_remaining
                // out of sync.
                auto& chunk = chunks.front();
                if (!sortlimit.has_value()) {
                    out_table_builder.chunks.push_back(chunk);
                } else {
                    if (chunk.table->nrows() <= sortlimit.value().offset) {
                        sortlimit.value() -= chunk.table->nrows();
                        if (!is_last) {
                            out_table_builder.chunks.push_back(chunk);
                        }
                    } else {
                        size_t start_index =
                            is_last ? sortlimit.value().offset : 0;
                        size_t end_index =
                            std::min(sortlimit.value().sum(),
                                     static_cast<size_t>(chunk.table->nrows()));
                        sortlimit.value() -= end_index;
                        if (start_index == 0 &&
                            end_index == chunk.table->nrows())
                            out_table_builder.chunks.push_back(chunk);
                        else {
                            std::vector<int64_t> index(end_index - start_index);
                            iota(index.begin(), index.end(), start_index);
                            out_table_builder.AppendBatch(chunk.table, index);
                            out_table_builder.FinalizeActiveChunk();
                        }
                    }
                }
                chunks.pop_front();
            }
        } else {
            while (!chunks.empty()) {
                auto& chunk = chunks.front();
                chunk.table->pin();
                if (!sortlimit.has_value()) {
                    out_table_builder.AppendBatch(chunk.table);
                } else {
                    if (chunk.table->nrows() <= sortlimit.value().offset) {
                        sortlimit.value() -= chunk.table->nrows();
                        if (!is_last) {
                            out_table_builder.AppendBatch(chunk.table);
                        }
                    } else {
                        size_t start_index =
                            is_last ? sortlimit.value().offset : 0;
                        size_t end_index =
                            std::min(sortlimit.value().sum(),
                                     static_cast<size_t>(chunk.table->nrows()));
                        sortlimit.value() -= end_index;
                        if (start_index == 0 &&
                            end_index == chunk.table->nrows())
                            out_table_builder.AppendBatch(chunk.table);
                        else {
                            std::vector<int64_t> index(end_index - start_index);
                            iota(index.begin(), index.end(), start_index);
                            out_table_builder.AppendBatch(chunk.table, index);
                            out_table_builder.FinalizeActiveChunk();
                        }
                    }
                }
                chunk.table->unpin();

                chunks.pop_front();
            }
        }
    }

    // If we have any leftover rows, push them into the output
    out_table_builder.FinalizeActiveChunk();

    return out_table_builder.chunks;
}

std::deque<TableAndRange> SortedChunkedTableBuilder::Finalize() {
    if (input_chunks.size() == 0) {
        std::deque<TableAndRange> output;
        return output;
    }

    std::vector<std::deque<TableAndRange>> sorted_chunks;

    // TODO Currently num_chunks is set to min(rank, 100) during initialization.
    // Should revisit this after benchmark tests
    bool is_last = (input_chunks.size() <= num_chunks);
    while (!input_chunks.empty()) {
        std::vector<std::deque<TableAndRange>> chunks;
        // Take num_chunks chunks at a time and put them each into a vector so
        // that we can call MergeChunks, which expects a vector of vector of
        // chunks. Each inner vector of chunks is expected to be sorted - a
        // vector of a single sorted chunk is by definition, sorted.
        for (size_t i = 0; i < num_chunks && !input_chunks.empty(); i++) {
            chunks.push_back({input_chunks.back()});
            input_chunks.pop_back();
        }

        if (is_last) {
            sorted_chunks.emplace_back(MergeChunks<true>(std::move(chunks)));
        } else {
            sorted_chunks.emplace_back(MergeChunks<false>(std::move(chunks)));
        }
    }

    while (sorted_chunks.size() > 1) {
        is_last = (sorted_chunks.size() <= num_chunks);
        std::vector<std::deque<TableAndRange>> next_sorted_chunks;
        // This loop takes num_chunks vectors and merges them into 1 on every
        // iteration.
        for (size_t i = 0; i < sorted_chunks.size(); i += num_chunks) {
            std::vector<std::deque<TableAndRange>> merge_input;
            // Collect the next num_chunks vectors
            size_t start = i;
            size_t end = std::min(i + num_chunks, sorted_chunks.size());
            for (size_t j = start; j < end; j++) {
                merge_input.emplace_back(std::move(sorted_chunks[j]));
            }

            if (is_last) {
                next_sorted_chunks.emplace_back(
                    MergeChunks<true>(std::move(merge_input)));
            } else {
                next_sorted_chunks.emplace_back(
                    MergeChunks<false>(std::move(merge_input)));
            }
        }

        std::swap(sorted_chunks, next_sorted_chunks);
    }

    return sorted_chunks[0];
}

double ReservoirSamplingState::random() { return dis(e); }

void ReservoirSamplingState::processInput(
    const std::shared_ptr<table_info>& input_chunk) {
    auto input = ProjectTable(input_chunk, column_indices);
    int64_t consumed_input_rows = 0;
    if (total_rows_seen < sample_size) {
        // Build our initial set of samples by using a prefix of the input
        uint64_t rows_to_pull =
            std::min(static_cast<uint64_t>(sample_size - total_rows_seen),
                     input->nrows());

        std::vector<bool> selection(input->nrows());
        std::fill(selection.begin(), selection.begin() + rows_to_pull, true);
        samples.ReserveTable(input, selection);
        samples.UnsafeAppendBatch(input, selection);
        for (size_t i = 0; i < rows_to_pull; i++) {
            selected_rows.push_back(consumed_input_rows + i);
        }

        total_rows_seen += rows_to_pull;
        consumed_input_rows += rows_to_pull;
        if (static_cast<uint64_t>(consumed_input_rows) == input->nrows()) {
            return;
        }
    }

    if (row_to_sample == -1) {
        // Initialize row_to_sample to determine the next row to sample and
        // replace a previously sampled row
        row_to_sample = sample_size + int64_t(log(random()) / log(1 - W)) + 1;
    }

    std::vector<int64_t> idxs;
    // The value of rows_consumed after processing the current input
    int64_t next_rows_seen =
        total_rows_seen + input->nrows() - consumed_input_rows;
    while (row_to_sample <= next_rows_seen) {
        // Generate a random uint64_t between 0 and sample_size as the sample
        // being replaced
        uint64_t target = rand() % sample_size;
        selected_rows[target] = samples.data_table->nrows() + idxs.size();
        idxs.push_back(row_to_sample - total_rows_seen + consumed_input_rows);

        W = W * exp(log(random()) / sample_size);
        row_to_sample = row_to_sample + int64_t(log(random()) / log(1 - W)) + 1;
    }
    total_rows_seen = next_rows_seen;
    if (!idxs.empty()) {
        std::vector<bool> selection(input->nrows(), false);
        for (auto idx : idxs) {
            selection[idx] = true;
        }
        samples.ReserveTable(input, selection);
        samples.UnsafeAppendBatch(input, selection);
    }

    if (samples.data_table->nrows() > static_cast<uint64_t>(sample_size * 10)) {
        // Compact the selected rows
        auto compacted_table = RetrieveTable(samples.data_table, selected_rows);
        samples.Reset();
        samples.ReserveTable(compacted_table);
        samples.UnsafeAppendBatch(compacted_table);

        std::iota(selected_rows.begin(), selected_rows.end(), 0);
    }
}

std::shared_ptr<table_info> ReservoirSamplingState::Finalize() {
    if (total_rows_seen < sample_size) {
        return samples.data_table;
    }

    return RetrieveTable(samples.data_table, selected_rows);
}

uint64_t StreamSortState::GetBudget() const {
    int64_t budget = OperatorComptroller::Default()->GetOperatorBudget(op_id);
    if (budget == -1)
        return static_cast<uint64_t>(
            bodo::BufferPool::Default()->get_memory_size_bytes() *
            SORT_OPERATOR_DEFAULT_MEMORY_FRACTION_OP_POOL);
    return static_cast<uint64_t>(budget);
}

std::vector<std::shared_ptr<DictionaryBuilder>> create_dict_builders(
    std::shared_ptr<bodo::Schema> schema) {
    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders;
    for (auto& col : schema->column_types) {
        // Note that none of the columns are "keys" from the perspective of the
        // dictionary builder, which is referring to keys for hashing/join
        dict_builders.emplace_back(
            create_dict_builder_for_array(col->copy(), false));
    }
    return dict_builders;
}

StreamSortState::StreamSortState(int64_t op_id, int64_t n_key_t,
                                 std::vector<int64_t>&& vect_ascending_,
                                 std::vector<int64_t>&& na_position_,
                                 std::shared_ptr<bodo::Schema> schema_,
                                 bool parallel, size_t chunk_size,
                                 size_t sample_size)
    : op_id(op_id),
      n_key_t(n_key_t),
      vect_ascending(vect_ascending_),
      na_position(na_position_),
      parallel(parallel),
      mem_budget_bytes(StreamSortState::GetBudget()),
      num_chunks(GetOptimalChunkNumber()),
      // Currently Operator pool and Memory manager are set to default
      // because not fully implemented. Can turn on during testing for
      // checking memory usage adding:
      // op_pool(std::make_shared<bodo::OperatorBufferPool>(op_id,
      // mem_budget_bytes, bodo::BufferPool::Default()))
      // op_mm(bodo::buffer_memory_manager(op_pool.get()))
      // op_pool->DisableThresholdEnforcement();
      op_pool(bodo::BufferPool::DefaultPtr()),
      op_mm(bodo::default_buffer_memory_manager()),
      schema(std::move(schema_)),
      dict_builders(create_dict_builders(schema)),
      dummy_output_chunk(alloc_table(schema, op_pool, op_mm, &dict_builders)),
      chunk_size(chunk_size),
      // Note that builder only stores references to the vectors owned by
      // this object, so we must refer to the instances on this class, not
      // the arguments.
      builder(schema, dict_builders, chunk_size,
              DEFAULT_MAX_RESIZE_COUNT_FOR_VARIABLE_SIZE_DTYPES, op_pool,
              op_mm),
      reservoir_sampling_state(n_key_t, sample_size, dict_builders, schema) {}

StreamSortLimitOffsetState::StreamSortLimitOffsetState(
    int64_t op_id, int64_t n_key_t, std::vector<int64_t>&& vect_ascending_,
    std::vector<int64_t>&& na_position_, std::shared_ptr<bodo::Schema> schema_,
    bool parallel, int64_t limit, int64_t offset, size_t chunk_size,
    bool enable_small_limit_optimization)
    : StreamSortState(op_id, n_key_t, std::move(vect_ascending_),
                      std::move(na_position_), std::move(schema_), parallel,
                      chunk_size),
      sortlimit(static_cast<size_t>(limit), static_cast<size_t>(offset)),
      top_k(schema, dict_builders, n_key_t, vect_ascending, na_position,
            dead_keys, num_chunks, chunk_size, mem_budget_bytes,
            sortlimit.limit + sortlimit.offset, 0, parallel, op_pool, op_mm) {
    limit_small_flag = (sortlimit.sum() <= SORT_SMALL_LIMIT_OFFSET_CAP &&
                        enable_small_limit_optimization);
}

void StreamSortState::ConsumeBatch(std::shared_ptr<table_info> table,
                                   bool is_last) {
    row_info.first += table_local_memory_size(table, false);
    row_info.second += table->nrows();

    time_pt start_append = start_timer();
    auto unified = UnifyDictionaryArrays(std::move(table), dict_builders);
    reservoir_sampling_state.processInput(unified);

    builder.AppendBatch(unified);
    metrics.local_sort_chunk_time += end_timer(start_append);
}

void StreamSortLimitOffsetState::ConsumeBatch(std::shared_ptr<table_info> table,
                                              bool is_last) {
    row_info.first += table_local_memory_size(table, false);
    row_info.second += table->nrows();

    time_pt start_append = start_timer();

    if (!limit_small_flag) {
        builder.UnifyDictionariesAndAppend(table, dict_builders);
    } else {
        // Maintain a heap of at most limit + offset elements when limit/offset
        // is small
        top_k.AppendChunk(table);
        top_k.input_chunks = top_k.Finalize();
    }
    metrics.local_sort_chunk_time += end_timer(start_append);
}

void StreamSortState::FinalizeBuild() {
    // attempt to increase the budget - this will allow for larger chunk sizes
    // and fewer pin/unpin calls.
    mem_budget_bytes =
        OperatorComptroller::Default()->RequestAdditionalBudget(op_id, -1);

    // Make all dictionaries global
    for (auto& dict_builder : dict_builders) {
        if (!dict_builder) {
            continue;
        }
        recursive_make_dict_global_and_unique(dict_builder);
    }
    dummy_output_chunk = UnifyDictionaryArrays(std::move(dummy_output_chunk),
                                               dict_builders, true);

    std::unique_ptr<int64_t[]> in_info = std::make_unique<int64_t[]>(2);
    in_info[0] = row_info.first;
    in_info[1] = row_info.second;
    std::unique_ptr<int64_t[]> sum_row_info = std::make_unique<int64_t[]>(2);
    if (parallel) {
        MPI_Allreduce(in_info.get(), sum_row_info.get(), 2, MPI_LONG_LONG_INT,
                      MPI_SUM, MPI_COMM_WORLD);
    }
    int64_t bytes_per_row = parallel ? (sum_row_info[0] / sum_row_info[1])
                                     : (row_info.first / row_info.second);

    chunk_size =
        GetChunkSize(bytes_per_row, mem_budget_bytes, num_chunks, chunk_size);
    builder.FinalizeActiveChunk();
    auto local_chunks = builder.chunks;
    GlobalSort(std::move(local_chunks));
}

std::shared_ptr<table_info> StreamSortState::GetParallelSortBounds(
    std::shared_ptr<table_info>&& local_samples) {
    int n_pes, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    // combine the dictionaries from all the local samples across all ranks
    for (size_t i = 0; i < local_samples->ncols(); i++) {
        if (dict_builders[i]) {
            recursive_make_array_global_and_unique(local_samples->columns[i],
                                                   true);
        }
    }

    auto ref_table = alloc_table_like(local_samples);
    // Collecting all samples globally
    bool all_gather = false;
    std::shared_ptr<table_info> all_samples =
        gather_table(std::move(local_samples), n_key_t, all_gather, parallel);

    // Compute split bounds from the samples.
    // Output is broadcasted to all ranks.
    bounds_ = compute_bounds_from_samples(
        std::move(all_samples), std::move(ref_table), n_key_t,
        vect_ascending.data(), na_position.data(), myrank, n_pes, parallel);
    // Transpose bounds to use the same indices as the local builders - we know
    // that the dictionary builder has all keys at this point.
    bounds_ = UnifyDictionaryArrays(std::move(bounds_), dict_builders);

    return bounds_;
}

std::vector<std::deque<std::shared_ptr<table_info>>>
StreamSortState::PartitionChunksByRank(
    SortedChunkedTableBuilder& global_builder, int n_pes,
    std::shared_ptr<table_info> bounds,
    std::deque<std::shared_ptr<table_info>>&& local_chunks) {
    std::vector<ChunkedTableBuilder> rankToChunkedBuilders;
    assert(static_cast<uint64_t>(n_pes) == bounds->nrows() + 1);

    // For each rank, we build n_pes ChunkedTableBuilder to store tables to
    // pass
    for (int rank_id = 0; rank_id < n_pes; rank_id++) {
        // Internally we will use a chunk size of 16k instead of 4k
        size_t chunk_size = 16 * 1024;
        rankToChunkedBuilders.push_back(ChunkedTableBuilder(
            schema, dict_builders, chunk_size,
            DEFAULT_MAX_RESIZE_COUNT_FOR_VARIABLE_SIZE_DTYPES, op_pool, op_mm));
    }

    // A vector containing all the possible ranks. This could be
    // std::ranges::iota_view, but it's not supported on all compilers yet.
    std::vector<int> ranks(n_pes);
    std::iota(ranks.begin(), ranks.end(), 0);

    for (auto& chunk : local_chunks) {
        chunk->pin();

        std::vector<std::vector<int64_t>> rankToRows(n_pes);
        int64_t table_size = static_cast<int64_t>(chunk->nrows());
        for (int64_t row = 0; row < table_size; row++) {
            // Find the first rank where bounds[rank] > chunk.table[row]
            auto dst_rank = std::lower_bound(
                ranks.begin(), ranks.end(), row, [&](int rank, int row) {
                    if (rank == (n_pes - 1)) {
                        return false;
                    }
                    return global_builder.Compare(bounds, rank, chunk, row);
                });
            assert(dst_rank != ranks.end());

            rankToRows[*dst_rank].push_back(row);
        }
        // TODO(aneesh): to improve IO/compute overlap we should send a table
        // as soon as it's ready
        for (int rank_id = 0; rank_id < n_pes; rank_id++) {
            rankToChunkedBuilders[rank_id].AppendBatch(chunk,
                                                       rankToRows[rank_id]);
        }
        chunk.reset();
    }

    std::vector<std::deque<std::shared_ptr<table_info>>> rankToChunks(n_pes);
    for (int rank_id = 0; rank_id < n_pes; rank_id++) {
        rankToChunkedBuilders[rank_id].FinalizeActiveChunk();
        rankToChunks[rank_id] =
            std::move(rankToChunkedBuilders[rank_id].chunks);
    }
    return rankToChunks;
}

void StreamSortLimitOffsetState::SmallLimitOptim() {
    int n_pes, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    std::vector<std::shared_ptr<table_info>> chunks;
    auto local_chunks = top_k.input_chunks;

    // Every rank concat local tables and send to rank 0 in 1 batch
    auto local_concat_tables = dummy_output_chunk;
    if (local_chunks.size() > 0) {
        std::vector<std::shared_ptr<table_info>> local_collected_tables;
        local_collected_tables.reserve(local_chunks.size());
        for (auto& i : local_chunks) {
            i.table->pin();
            local_collected_tables.push_back(std::move(i.table));
        }
        local_concat_tables =
            concat_tables(std::move(local_collected_tables), dict_builders);
    }

    SortedChunkedTableBuilder global_builder(
        schema, dict_builders, n_key_t, vect_ascending, na_position, dead_keys,
        num_chunks, chunk_size, mem_budget_bytes, sortlimit.limit,
        sortlimit.offset, parallel, op_pool, op_mm);

    // Send all data to rank 0 synchronously
    time_pt start_communication = start_timer();
    std::vector<std::shared_ptr<table_info>> collected_tables;
    local_concat_tables =
        UnifyDictionaryArrays(std::move(local_concat_tables), dict_builders);
    auto collected_table =
        gather_table(local_concat_tables, -1, false, parallel, 0);
    metrics.communication_phase += end_timer(start_communication);
    time_pt start_finalize = start_timer();

    if (myrank == 0) {
        time_pt start_append = start_timer();
        collected_tables.push_back(std::move(collected_table));
        metrics.global_append_chunk_time += end_timer(start_append);
        auto concatenated_tables =
            concat_tables(std::move(collected_tables), dict_builders);
        auto indices = sort_values_table_local_get_indices(
            concatenated_tables, n_key_t, vect_ascending.data(),
            na_position.data(), false, 0, concatenated_tables->nrows());
        uint64_t n_cols = concatenated_tables->ncols();
        std::vector<uint64_t> col_inds;
        for (int64_t i = 0; i < n_key_t; i++) {
            if (!dead_keys.empty() && dead_keys[i]) {
                // If this is the last reference to this
                // table, we can safely release reference (and potentially
                // memory if any) for the dead keys at this point.
                reset_col_if_last_table_ref(concatenated_tables, i);
            } else {
                col_inds.push_back(i);
            }
        }
        for (uint64_t i = n_key_t; i < n_cols; i++) {
            col_inds.push_back(i);
        }

        ChunkedTableBuilder out_table_builder(
            schema, dict_builders, chunk_size,
            DEFAULT_MAX_RESIZE_COUNT_FOR_VARIABLE_SIZE_DTYPES, op_pool, op_mm);

        bodo::vector<int64_t> offset_indices;
        for (size_t i = sortlimit.offset; i < sortlimit.sum(); i++) {
            if (i >= indices.size()) {
                break;
            }
            offset_indices.push_back(indices[i]);
        }
        out_table_builder.AppendBatch(concatenated_tables, offset_indices,
                                      col_inds);

        out_table_builder.FinalizeActiveChunk();
        for (auto chunk : out_table_builder.chunks) {
            output_chunks.push_back(chunk);
        }
    }
    metrics.global_sort_time += end_timer(start_finalize);
}

void StreamSortLimitOffsetState::ComputeLocalLimit(
    SortedChunkedTableBuilder& global_builder) {
    int n_pes, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    size_t limit = sortlimit.limit;
    size_t offset = sortlimit.offset;
    size_t number_rows = 0;
    std::vector<size_t> nrows_collect(n_pes);
    for (auto& i : global_builder.input_chunks)
        number_rows += static_cast<size_t>(i.table->nrows() - i.offset);
    MPI_Allgather(&number_rows, 1, MPI_UNSIGNED_LONG, nrows_collect.data(), 1,
                  MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
    size_t total_rows_before = 0;
    for (int64_t i = 0; i < myrank; i++) {
        total_rows_before += nrows_collect[i];
    }
    if (total_rows_before >= limit + offset ||
        total_rows_before + number_rows <= offset || limit == 0) {
        global_builder.sortlimits = SortLimits(0, 0);
        return;
    }
    size_t finalize_offset =
        total_rows_before >= offset ? 0 : offset - total_rows_before;
    size_t finalize_limit =
        std::min(limit + offset - total_rows_before, number_rows) -
        finalize_offset;
    global_builder.sortlimits = SortLimits(finalize_limit, finalize_offset);
    return;
}

void StreamSortState::GlobalSort_NonParallel(
    std::deque<std::shared_ptr<table_info>>&& local_chunks) {
    SortedChunkedTableBuilder global_builder = GetGlobalBuilder();
    for (auto& chunk : local_chunks) {
        chunk->pin();
        global_builder.AppendChunk(std::move(chunk));
    }
    time_pt start_finalize = start_timer();
    auto out_chunks = global_builder.Finalize();
    metrics.global_sort_time += end_timer(start_finalize);

    for (const auto& chunk : out_chunks) {
        output_chunks.push_back(std::move(chunk.table));
    }
}

SortedChunkedTableBuilder StreamSortState::GlobalSort_Partition(
    std::deque<std::shared_ptr<table_info>>&& local_chunks) {
    int n_pes, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    SortedChunkedTableBuilder global_builder = GetGlobalBuilder();

    std::shared_ptr<table_info> bounds =
        GetParallelSortBounds(reservoir_sampling_state.Finalize());
    time_pt start_partition = start_timer();

    std::vector<std::deque<std::shared_ptr<table_info>>> rankToChunks =
        PartitionChunksByRank(global_builder, n_pes, bounds,
                              std::move(local_chunks));
    metrics.partition_chunks_time = end_timer(start_partition);

    for (auto& chunk : rankToChunks[myrank]) {
        chunk->pin();
        global_builder.AppendChunk(std::move(chunk));
    }
    rankToChunks[myrank].clear();

    std::vector<size_t> rankToCurrentChunk(n_pes);

    time_pt start_communication = start_timer();
    std::vector<AsyncShuffleSendState> send_states;

    std::vector<AsyncShuffleRecvState> recv_states;

    bool have_chunks_to_send = false;
    auto HaveChunksToSend = [&]() {
        for (int i = 0; i < n_pes; i++) {
            if (i == myrank) {
                continue;
            }
            if (rankToCurrentChunk[i] < rankToChunks[i].size()) {
                return true;
            }
        }
        return false;
    };
    have_chunks_to_send = HaveChunksToSend();

    // Barrier to test for completion
    MPI_Request is_last_request = MPI_REQUEST_NULL;
    // If there's only a single rank, skip the loop below
    bool is_last = n_pes == 1;

    // Initialize every rank to send to their neighbor. This is to prevent the
    // situation where all ranks send to a single host simultaneously making it
    // harder to overlap IO and compute.
    int host_to_send_to = (myrank + 1) % n_pes;

    bool barrier_posted = false;
    // Loop until the barrier is reached and we have no outstanding IO
    // requests
    while (!is_last || have_chunks_to_send || !recv_states.empty() ||
           !send_states.empty()) {
        // Don't send unless the number of inflight sends is less than
        // the number of ranks
        if (have_chunks_to_send &&
            send_states.size() < static_cast<size_t>(n_pes)) {
            if (rankToCurrentChunk[host_to_send_to] <
                rankToChunks[host_to_send_to].size()) {
                size_t chunk_id = rankToCurrentChunk[host_to_send_to];
                auto table_to_send =
                    std::move(rankToChunks[host_to_send_to][chunk_id]);
                table_to_send->pin();

                auto nrows = table_to_send->nrows();
                auto hashes = std::make_shared<uint32_t[]>(nrows);
                std::fill(hashes.get(), hashes.get() + nrows, host_to_send_to);

                // Shuffle all the data
                send_states.emplace_back(
                    shuffle_issend(std::move(table_to_send), std::move(hashes),
                                   nullptr, MPI_COMM_WORLD));

                // Increment chunk id
                rankToCurrentChunk[host_to_send_to]++;
            }
            // Increment rank to send to
            do {
                host_to_send_to++;
                host_to_send_to %= n_pes;
                // This loop is safe because we know that if `n_pes` was 1,
                // we don't execute the outer loop
            } while (host_to_send_to == myrank);

            // Check if we have any more buffers to send on the next
            // iteration
            have_chunks_to_send = HaveChunksToSend();
        }

        // Remove send state if recv done
        std::erase_if(send_states,
                      [](AsyncShuffleSendState& s) { return s.sendDone(); });
        if (!barrier_posted && send_states.empty() && !have_chunks_to_send) {
            MPI_Ibarrier(MPI_COMM_WORLD, &is_last_request);
            barrier_posted = true;
        }

        if (recv_states.size() < static_cast<size_t>(n_pes)) {
            // Check if we can receive
            shuffle_irecv(dummy_output_chunk, MPI_COMM_WORLD, recv_states);
        }
        // If we have any completed receives, add them to the builder
        std::erase_if(recv_states, [&](AsyncShuffleRecvState& s) {
            auto [done, table] =
                s.recvDone(dict_builders, metrics.ishuffle_metrics);
            if (done && table->nrows()) {
                time_pt start_append = start_timer();
                global_builder.AppendChunk(std::move(table));
                metrics.global_append_chunk_time += end_timer(start_append);
                // TODO(aneesh) every num_chunks tables we can start
                // the sort process while we went for messages.
            }
            return done;
        });

        // If we've already posted the barrier test to see if all other ranks
        // have as well.
        if (!is_last && barrier_posted) {
            int flag = 0;
            MPI_Test(&is_last_request, &flag, MPI_STATUS_IGNORE);
            is_last = flag;
        }
    }

    metrics.communication_phase += end_timer(start_communication);
    return global_builder;
}

void StreamSortState::GlobalSort(
    std::deque<std::shared_ptr<table_info>>&& local_chunks) {
    if (!parallel) {
        GlobalSort_NonParallel(std::move(local_chunks));
        return;
    }
    auto global_builder = GlobalSort_Partition(std::move(local_chunks));
    time_pt start_finalize = start_timer();
    auto out_chunks = global_builder.Finalize();
    metrics.global_sort_time += end_timer(start_finalize);
    for (const auto& chunk : out_chunks) {
        output_chunks.push_back(std::move(chunk.table));
    }
}

void StreamSortLimitOffsetState::GlobalSort(
    std::deque<std::shared_ptr<table_info>>&& local_chunks) {
    if (sortlimit.limit == 0) {
        return;
    }

    if (limit_small_flag) {
        SmallLimitOptim();
        return;
    }

    if (!parallel) {
        GlobalSort_NonParallel(std::move(local_chunks));
        return;
    }

    auto global_builder = GlobalSort_Partition(std::move(local_chunks));
    ComputeLocalLimit(global_builder);
    time_pt start_finalize = start_timer();
    auto out_chunks = global_builder.Finalize();
    metrics.global_sort_time += end_timer(start_finalize);
    for (const auto& chunk : out_chunks) {
        output_chunks.push_back(std::move(chunk.table));
    }
}

std::pair<std::shared_ptr<table_info>, bool> StreamSortState::GetOutput() {
    std::shared_ptr<table_info> output = nullptr;
    bool out_is_last = false;
    if (output_idx < output_chunks.size()) {
        std::swap(output, output_chunks[output_idx]);
        output->pin();
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
    metrics.reserve(6);

    metrics.emplace_back(TimerMetric("local_sort_chunk_time",
                                     this->metrics.local_sort_chunk_time));

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

StreamSortState* stream_sort_state_init_py_entry(
    int64_t op_id, int64_t limit, int64_t offset, int64_t n_key_t,
    int64_t* vect_ascending, int64_t* na_position, int8_t* arr_c_types,
    int8_t* arr_array_types, int64_t n_arrs, bool parallel) {
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
        // No limit and offset. Use usual StreamSortState
        if (limit == -1 || offset == -1) {
            auto* state = new StreamSortState(
                op_id, n_key_t, std::move(vect_ascending_),
                std::move(na_position_), std::move(schema), parallel);
            return state;
        }
        // Either limit or offset is set. Use the subclass
        // StreamSortLimitOffsetState
        auto* state = new StreamSortLimitOffsetState(
            op_id, n_key_t, std::move(vect_ascending_), std::move(na_position_),
            std::move(schema), parallel, limit, offset);
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
        state->ConsumeBatch(table, is_last);
        if (is_last) {
            state->FinalizeBuild();
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
