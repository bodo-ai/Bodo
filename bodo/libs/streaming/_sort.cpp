#include "_sort.h"
#include <algorithm>
#include <cstdint>
#include <deque>
#include <memory>
#include <numeric>
#include <unordered_set>
#include <utility>

#include <fmt/format.h>

#include "../_array_operations.h"
#include "../_array_utils.h"
#include "../_bodo_common.h"
#include "../_chunked_table_builder.h"
#include "../_dict_builder.h"
#include "../_memory_budget.h"
#include "../_query_profile_collector.h"
#include "../_table_builder_utils.h"
#include "../_utils.h"
#include "_shuffle.h"

#define QUERY_PROFILE_SORT_INIT_STAGE_ID 0
#define QUERY_PROFILE_SORT_BUILD_STAGE_ID 1
#define QUERY_PROFILE_SORT_OUTPUT_STAGE_ID 2

// Min/max allowed size (#rows) of any individual chunk used during shuffle.
#define DEFAULT_SORT_MIN_SHUFFLE_CHUNK_SIZE 4096
#define DEFAULT_SORT_MAX_SHUFFLE_CHUNK_SIZE 16 * 1024 * 1024

// Min/max allowed size (#rows) of any individual chunk during the final
// external k-way merge-sort step.
#define DEFAULT_SORT_MIN_CHUNK_SIZE 4096
#define DEFAULT_SORT_MAX_CHUNK_SIZE 16 * 1024 * 1024

// Min/max allowed value of 'K' for the external K-way merge-sort step
#define DEFAULT_SORT_MIN_K 2
#define DEFAULT_SORT_MAX_K 128

// If an operator ID is not specified (and hence no budget), use this fraction
// of the buffer pool size as the budget.
#define SORT_OPERATOR_DEFAULT_MEMORY_FRACTION_OP_POOL 1.0

// Threshold for using the small limit+offset optimization.
#define SORT_SMALL_LIMIT_OFFSET_CAP 16384

/**
 * @brief Get the optimal sort shuffle chunk size based on the memory budget,
 * average size of each row and the maximum number of chunks that might be
 * inflight at once.
 *
 * @param bytes_per_row_ Average number of bytes per row. If this is -1 (i.e.
 * unknown), we return the min allowed chunk size.
 * @param mem_budget_bytes Available memory budget. If this is 0, we return the
 * min allowed chunk size.
 * @param max_inflight_sends Maximum number of concurrent send/recvs allowed.
 * @return uint64_t Optimal chunk size for shuffle.
 */
static uint64_t get_optimal_sort_shuffle_chunk_size(int64_t bytes_per_row_,
                                                    uint64_t mem_budget_bytes,
                                                    size_t max_inflight_sends) {
    uint64_t min_chunk_size = DEFAULT_SORT_MIN_SHUFFLE_CHUNK_SIZE;
    if (char* min_chunk_size_env_ =
            std::getenv("BODO_STREAM_SORT_MIN_SHUFFLE_CHUNK_SIZE")) {
        min_chunk_size = static_cast<uint64_t>(std::stoi(min_chunk_size_env_));
    }
    uint64_t max_chunk_size = DEFAULT_SORT_MAX_SHUFFLE_CHUNK_SIZE;
    if (char* max_chunk_size_env_ =
            std::getenv("BODO_STREAM_SORT_MAX_SHUFFLE_CHUNK_SIZE")) {
        max_chunk_size = static_cast<uint64_t>(std::stoi(max_chunk_size_env_));
    }
    if (bytes_per_row_ <= 0 || mem_budget_bytes == 0) {
        return min_chunk_size;
    }
    uint64_t bytes_per_row = static_cast<uint64_t>(bytes_per_row_);
    // NOTE: We divide by 2 since we need to have sufficient memory for both
    // incoming and outgoing chunks.
    uint64_t chunk_size = std::min(
        std::max(mem_budget_bytes / (2 * max_inflight_sends * bytes_per_row),
                 min_chunk_size),
        max_chunk_size);
    return chunk_size;
}

/**
 * @brief Get the optimal chunk size and k (of k-way merge) to use during the
 * k-way merge, based on the budget.
 *
 * @param bytes_per_row_ Average number of bytes per row. If this is -1 (i.e.
 * unknown), we return the min allowed chunk size and K.
 * @param mem_budget_bytes Available memory budget. If this is 0, we return the
 * min allowed chunk size and K.
 * @return std::pair<uint64_t, uint64_t> Optimal chunk size and corresponding K.
 * K will always be >=2.
 */
static std::pair<uint64_t, uint64_t> get_optimal_sort_chunk_size_and_k(
    int64_t bytes_per_row_, uint64_t mem_budget_bytes) {
    uint64_t min_chunk_size = DEFAULT_SORT_MIN_CHUNK_SIZE;
    if (char* min_chunk_size_env_ =
            std::getenv("BODO_STREAM_SORT_MIN_CHUNK_SIZE")) {
        min_chunk_size = static_cast<uint64_t>(std::stoi(min_chunk_size_env_));
    }
    uint64_t max_chunk_size = DEFAULT_SORT_MAX_CHUNK_SIZE;
    if (char* max_chunk_size_env_ =
            std::getenv("BODO_STREAM_SORT_MAX_CHUNK_SIZE")) {
        max_chunk_size = static_cast<uint64_t>(std::stoi(max_chunk_size_env_));
    }
    uint64_t min_k = DEFAULT_SORT_MIN_K;
    if (char* min_k_env_ = std::getenv("BODO_STREAM_SORT_MIN_K")) {
        min_k = static_cast<uint64_t>(std::stoi(min_k_env_));
    }
    uint64_t max_k = DEFAULT_SORT_MAX_K;
    if (char* max_k_env_ = std::getenv("BODO_STREAM_SORT_MAX_K")) {
        max_k = static_cast<uint64_t>(std::stoi(max_k_env_));
    }
    if (min_k < 2) {
        throw std::runtime_error(
            fmt::format("get_optimal_sort_chunk_size_and_k: min_k must be >=2! "
                        "Tried to set it to {}.",
                        min_k));
    }

    if (bytes_per_row_ <= 0 || mem_budget_bytes == 0) {
        return {min_chunk_size, min_k};
    }

    uint64_t bytes_per_row = static_cast<uint64_t>(bytes_per_row_);

    // NOTE: The +/-1 is because we need K+1 chunks to fit in memory (K input
    // and 1 output chunks). We want as large of a chunk as possible, so we use
    // min_k to calculate the chunk size.
    uint64_t optimal_chunk_size =
        std::min(std::max(mem_budget_bytes / (bytes_per_row * (min_k + 1)),
                          min_chunk_size),
                 max_chunk_size);

    // XXX TODO Ideally, in a columnar setting, (#key_columns_buffers * k) <
    // #cache_lines in L1/L2 cache (assuming each cache line can fit a few
    // values from a column) to avoid thrashing. Each cache line is typically
    // 64B.
    uint64_t optimal_k =
        std::min(
            std::max(mem_budget_bytes / (optimal_chunk_size * bytes_per_row),
                     min_k + 1),
            max_k + 1) -
        1;
    return {optimal_chunk_size, optimal_k};
}

TableAndRange::TableAndRange(
    std::shared_ptr<table_info> table, int64_t n_keys,
    const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders,
    int64_t offset)
    : table(table),
      offset(offset),
      range(table->schema()->Project(n_keys), dict_builders) {
    // Reserve 2 rows worth of space up front.
    this->range.ReserveTableSize(2);

    // Note that we assume table is already sorted and pinned
    this->UpdateOffset(n_keys, offset);
}

void TableAndRange::UpdateOffset(int64_t n_keys, int64_t offset_) {
    // Assumes table is pinned
    this->offset = offset_;
    uint64_t size = table->nrows();

    // XXX TODO Ideally we should do the following:
    //   1. Store the max at index 0 (since it never changes)
    //   2. Add a way to "decrement" from the bottom in TBB.
    //   3. Only update the "offset"/"min" row entry.

    // Get the first and last row of the sorted chunk - note that we only
    // get the columns for the keys since the range is only used for comparision
    range.Reset();  // This doesn't release memory
    range.ReserveTableRow(this->table, offset_);
    range.AppendRowKeys(this->table, offset_, n_keys);
    range.ReserveTableRow(this->table, size - 1);
    range.AppendRowKeys(this->table, size - 1, n_keys);
}

void SortedChunkedTableBuilder::PushActiveChunk() {
    // Sort the active chunk to create a new sorted chunk.

    // We will assume that row indices fit into 32 bits
    assert(this->active_chunk->nrows() < std::numeric_limits<int32_t>::max());

    time_pt start_sort = start_timer();
    bodo::vector<int32_t> sort_idx =
        sort_values_table_local_get_indices<int32_t>(
            this->active_chunk, this->n_keys, this->vect_ascending.data(),
            this->na_position.data(), /*is_parallel*/ false, /*start_offset*/ 0,
            this->active_chunk->nrows(), this->pool, this->mm);
    this->sort_time += end_timer(start_sort);
    time_pt start_retr = start_timer();
    std::shared_ptr<table_info> sorted_active_chunk =
        RetrieveTable(this->active_chunk, sort_idx, /*n_cols_arg*/ -1,
                      /*use_nullable_arr*/ false, this->pool, this->mm);
    this->sort_copy_time += end_timer(start_retr);

    // Get the range before we unpin the table
    TableAndRange chunk{std::move(sorted_active_chunk), this->n_keys,
                        this->dict_builders};
    chunk.table->unpin();
    this->chunks.emplace_back(std::move(chunk));
}

bool ExternalKWayMergeSorter::Compare(std::shared_ptr<table_info> table1,
                                      size_t row1,
                                      std::shared_ptr<table_info> table2,
                                      size_t row2) const {
    // TODO(aneesh) we should either template KeyComparisonAsPython or move to
    // something like converting rows to bitstrings for faster comparision.
    return KeyComparisonAsPython(n_keys, vect_ascending.data(), table1->columns,
                                 row1, table2->columns, row2,
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

void ExternalKWayMergeSorterFinalizeMetrics::ExportMetrics(
    std::vector<MetricBase>& metrics) {
#define APPEND_TIMER_METRIC(field) \
    metrics.push_back(TimerMetric(#field, this->field));

#define APPEND_STAT_METRIC(field) \
    metrics.push_back(StatMetric(#field, this->field));

    APPEND_STAT_METRIC(merge_chunks_processing_chunk_size);
    APPEND_STAT_METRIC(merge_chunks_K);
    APPEND_TIMER_METRIC(kway_merge_sort_total_time);
    APPEND_TIMER_METRIC(merge_input_builder_finalize_time);
    APPEND_TIMER_METRIC(merge_input_builder_total_sort_time);
    APPEND_TIMER_METRIC(merge_input_builder_total_sort_copy_time);
    APPEND_STAT_METRIC(merge_n_input_chunks);
    APPEND_STAT_METRIC(merge_approx_input_chunks_total_bytes);
    APPEND_STAT_METRIC(merge_approx_max_input_chunk_size_bytes);
    APPEND_STAT_METRIC(performed_inmem_concat_sort);
    APPEND_TIMER_METRIC(finalize_inmem_concat_time);
    APPEND_TIMER_METRIC(finalize_inmem_sort_time);
    APPEND_TIMER_METRIC(finalize_inmem_output_append_time);
    APPEND_STAT_METRIC(n_merge_levels);
    APPEND_STAT_METRIC(n_chunk_merges);
    APPEND_TIMER_METRIC(merge_chunks_total_time);
    APPEND_TIMER_METRIC(merge_chunks_make_heap_time);
    APPEND_TIMER_METRIC(merge_chunks_output_append_time);
    APPEND_TIMER_METRIC(merge_chunks_pop_heap_time);
    APPEND_TIMER_METRIC(merge_chunks_push_heap_time);

#undef APPEND_STAT_METRIC
#undef APPEND_TIMER_METRIC
}

void ReservoirSamplingMetrics::ExportMetrics(std::vector<MetricBase>& metrics) {
    metrics.push_back(TimerMetric("sampling_process_input_time",
                                  this->sampling_process_input_time));
    metrics.push_back(StatMetric("n_sampling_buffer_rebuilds",
                                 this->n_sampling_buffer_rebuilds));
    metrics.push_back(StatMetric("n_samples_taken", this->n_samples_taken));
}

ExternalKWayMergeSorter::ExternalKWayMergeSorter(
    std::shared_ptr<bodo::Schema> schema,
    const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders_,
    int64_t n_keys, const std::vector<int64_t>& vect_ascending,
    const std::vector<int64_t>& na_position,
    const std::vector<int64_t>& dead_keys, uint64_t mem_budget_bytes_,
    int64_t bytes_per_row_, int64_t limit, int64_t offset,
    size_t output_chunk_size_, int64_t K_, int64_t processing_chunk_size_,
    bool enable_inmem_concat_sort_, bool debug_mode_,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm)
    : schema(std::move(schema)),
      dict_builders(dict_builders_),
      n_keys(n_keys),
      vect_ascending(vect_ascending),
      na_position(na_position),
      dead_keys(dead_keys),
      mem_budget_bytes(mem_budget_bytes_),
      output_chunk_size(output_chunk_size_),
      enable_inmem_concat_sort(enable_inmem_concat_sort_),
      debug_mode(debug_mode_),
      comp(*this),
      pool(pool),
      mm(std::move(mm)) {
    // Either both limit and offset are -1, or they are both >= 0
    if (limit >= 0) {
        sortlimits = std::make_optional(SortLimits(limit, offset));
    }
    auto [optimal_chunk_size, optimal_k] =
        get_optimal_sort_chunk_size_and_k(bytes_per_row_, mem_budget_bytes_);
    this->processing_chunk_size = processing_chunk_size_ != -1
                                      ? processing_chunk_size_
                                      : optimal_chunk_size;
    this->K = K_ != -1 ? K_ : optimal_k;
    if (this->K < 2) {
        throw std::runtime_error(fmt::format(
            "ExternalKWayMergeSorter: K must be >=2! Tried to set it to {}.",
            this->K));
    }
    this->metrics.merge_chunks_K = this->K;
    this->metrics.merge_chunks_processing_chunk_size =
        this->processing_chunk_size;
    this->sorted_input_chunks_builder =
        std::make_unique<SortedChunkedTableBuilder>(
            this->schema, this->dict_builders, this->n_keys,
            this->vect_ascending, this->na_position,
            this->processing_chunk_size,
            DEFAULT_MAX_RESIZE_COUNT_FOR_VARIABLE_SIZE_DTYPES, this->pool,
            this->mm);
}

void ExternalKWayMergeSorter::AppendChunk(std::shared_ptr<table_info> chunk) {
    if (chunk->nrows() == 0) {
        return;
    }
    chunk = UnifyDictionaryArrays(std::move(chunk), this->dict_builders);
    this->sorted_input_chunks_builder->AppendBatch(std::move(chunk));
}

void ExternalKWayMergeSorter::UpdateLimitOffset(int64_t new_limit,
                                                int64_t new_offset) {
    if (new_limit == -1) {
        assert(new_offset == -1);
        this->sortlimits = std::nullopt;
    } else {
        assert(new_limit >= 0);
        assert(new_offset >= 0);
        this->sortlimits =
            std::make_optional(SortLimits(new_limit, new_offset));
    }
}

/**
 * Comparator for lists of sorted elements. Each chunk must be sorted, but
 * the list as a whole must be sorted in reverse. See MergeChunks for
 * details.
 */
struct VHeapComparator {
    const ExternalKWayMergeSorter::HeapComparator& comp;
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
template <bool is_last, bool has_limit_offset>
std::deque<TableAndRange> ExternalKWayMergeSorter::MergeChunks(
    std::vector<std::deque<TableAndRange>>&& sorted_chunks) /*const*/ {
    time_pt start = start_timer();
    assert(has_limit_offset == this->sortlimits.has_value());
    SortLimits sortlimit_ =
        has_limit_offset ? SortLimits(this->sortlimits.value().limit,
                                      this->sortlimits.value().offset)
                         : /*dummy value for type stability*/ SortLimits(0, 0);
    // If this is the last level of merging, we should output chunks of the
    // required output size directly. If we're in the middle of computation, we
    // should use the more optimal 'processing_chunk_size'.
    size_t out_chunk_size =
        is_last ? this->output_chunk_size : this->processing_chunk_size;
    ChunkedTableAndRangeBuilder out_table_builder(
        n_keys, schema, dict_builders, out_chunk_size,
        DEFAULT_MAX_RESIZE_COUNT_FOR_VARIABLE_SIZE_DTYPES, pool, mm);
    VHeapComparator vcomp(comp);
    time_pt start_make_heap = start_timer();
    std::ranges::make_heap(sorted_chunks, vcomp);
    this->metrics.merge_chunks_make_heap_time += end_timer(start_make_heap);

    time_pt start_pop, start_push;
    while (sorted_chunks.size() > 1 &&
           (!has_limit_offset || sortlimit_.sum() > 0)) {
        // No queue should ever be empty - empty queues should be removed
        // entirely
        assert(std::all_of(sorted_chunks.begin(), sorted_chunks.end(),
                           [&](auto& chunks) { return !chunks.empty(); }));
        // find the minimum row
        start_pop = start_timer();
        std::ranges::pop_heap(sorted_chunks, vcomp);
        this->metrics.merge_chunks_pop_heap_time += end_timer(start_pop);
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
            if (has_limit_offset) {
                nrows = std::min(
                    nrows, static_cast<int64_t>(offset + sortlimit_.sum()));
            }
            if (!Compare(sorted_chunks[0].front().range.data_table, RANGE_MIN,
                         min.get().range.data_table, RANGE_MAX)) {
                // We can append the whole table from offset onwards. If the
                // offset is 0, then we just append the entire table
                int64_t start_index =
                    offset +
                    (is_last && has_limit_offset ? sortlimit_.offset : 0);
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
                if (has_limit_offset) {
                    sortlimit_ -= nrows - offset;
                }
                offset = nrows;
            } else {
                do {
                    // TODO(aneesh) this could be replaced by a binary
                    // search to find the first element that is largest than
                    // the next smallest chunk.
                    if (!has_limit_offset) {
                        row_idx.push_back(offset);
                    } else {
                        if (!is_last || sortlimit_.offset == 0) {
                            row_idx.push_back(offset);
                        }
                        sortlimit_ -= 1;
                    }
                    offset++;
                } while (offset < nrows &&
                         !Compare(sorted_chunks[0].front().range.data_table,
                                  RANGE_MIN, min.get().table, offset) &&
                         (!has_limit_offset || sortlimit_.sum() > 0));
                if (offset < nrows) {
                    min.get().UpdateOffset(n_keys, offset);
                }

                // TODO(aneesh) If row_idx contains all rows in the chunk,
                // we might want to directly append to the output without
                // copying instead. We'd need to first finalize the active
                // chunk and then flush all chunks to the output.
                out_table_builder.AppendBatch(min.get().table, row_idx);
            }

            // If we have completely consumed all rows from the current
            // chunk, get the next chunk from the same sorted list.
            if (offset >= nrows ||
                (has_limit_offset && sortlimit_.sum() == 0)) {
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
                 (!has_limit_offset || sortlimit_.sum() > 0));

        if (min_vec.empty()) {
            // This vector of chunks is completely consumed, so we can
            // remove it from the heap
            sorted_chunks.pop_back();
        } else {
            // We've updated the minimum row of the vector of chunks, so we
            // need to push it back onto the heap to find the next smallest
            // row.
            start_push = start_timer();
            std::ranges::push_heap(sorted_chunks, vcomp);
            this->metrics.merge_chunks_push_heap_time += end_timer(start_push);
        }
    }

    // Append all unconsumed rows in a chunk to the builder
    auto AppendChunkToBuilder = [&](TableAndRange& chunk) {
        if (has_limit_offset && is_last) {
            size_t row_size =
                chunk.table->nrows() - static_cast<size_t>(chunk.offset);
            if (row_size <= sortlimit_.offset) {
                sortlimit_ -= row_size;
                return;
            }
            chunk.UpdateOffset(n_keys, chunk.offset + sortlimit_.offset);
            sortlimit_.offset = 0;
        }
        chunk.table->pin();
        size_t row_size =
            chunk.table->nrows() - static_cast<size_t>(chunk.offset);
        if (has_limit_offset) {
            row_size = std::min(sortlimit_.sum(), row_size);
            sortlimit_ -= row_size;
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
                            return chunk.table->nrows() <=
                                   out_table_builder.active_chunk_capacity;
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
                if (!has_limit_offset) {
                    out_table_builder.chunks.push_back(chunk);
                } else {
                    if (chunk.table->nrows() <= sortlimit_.offset) {
                        sortlimit_ -= chunk.table->nrows();
                        if (!is_last) {
                            out_table_builder.chunks.push_back(chunk);
                        }
                    } else {
                        size_t start_index = is_last ? sortlimit_.offset : 0;
                        size_t end_index =
                            std::min(sortlimit_.sum(),
                                     static_cast<size_t>(chunk.table->nrows()));
                        sortlimit_ -= end_index;
                        if (start_index == 0 &&
                            end_index == chunk.table->nrows()) {
                            out_table_builder.chunks.push_back(chunk);
                        } else {
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
                if (!has_limit_offset) {
                    out_table_builder.AppendBatch(chunk.table);
                } else {
                    if (chunk.table->nrows() <= sortlimit_.offset) {
                        sortlimit_ -= chunk.table->nrows();
                        if (!is_last) {
                            out_table_builder.AppendBatch(chunk.table);
                        }
                    } else {
                        size_t start_index = is_last ? sortlimit_.offset : 0;
                        size_t end_index =
                            std::min(sortlimit_.sum(),
                                     static_cast<size_t>(chunk.table->nrows()));
                        sortlimit_ -= end_index;
                        if (start_index == 0 &&
                            end_index == chunk.table->nrows()) {
                            out_table_builder.AppendBatch(chunk.table);
                        } else {
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

    this->metrics.merge_chunks_output_append_time +=
        out_table_builder.append_time;
    this->metrics.merge_chunks_total_time += end_timer(start);

    return out_table_builder.chunks;
}

std::deque<TableAndRange> ExternalKWayMergeSorter::Finalize(
    bool reset_input_builder) {
    const std::string DEBUG_LOG_PREFIX =
        "[DEBUG] ExternalKWayMergeSorter::Finalize:";
    ScopedTimer timer(this->metrics.kway_merge_sort_total_time);

    // Finalize any active chunks.
    time_pt start_fin_ac = start_timer();
    if (reset_input_builder) {
        this->sorted_input_chunks_builder->Finalize(/*shrink_to_fit*/ false);
    } else {
        this->sorted_input_chunks_builder->FinalizeActiveChunk(
            /*shrink_to_fit*/ false);
    }
    this->metrics.merge_input_builder_finalize_time += end_timer(start_fin_ac);
    this->metrics.merge_input_builder_total_sort_time =
        this->sorted_input_chunks_builder->sort_time;
    this->metrics.merge_input_builder_total_sort_copy_time =
        this->sorted_input_chunks_builder->sort_copy_time;

    // Get the chunks out of the builder and then reset the builder.
    std::deque<TableAndRange> input_chunks;
    std::swap(input_chunks, this->sorted_input_chunks_builder->chunks);
    if (reset_input_builder) {
        this->sorted_input_chunks_builder.reset();
    }
    this->metrics.merge_n_input_chunks = input_chunks.size();

    if (input_chunks.size() == 0 ||
        (this->sortlimits.has_value() && this->sortlimits.value().limit == 0)) {
        std::deque<TableAndRange> output;
        return output;
    }

    size_t approx_input_chunks_total_bytes = 0;
    size_t approx_max_input_chunk_size_bytes = 0;
    size_t num_input_rows = 0;
    for (const auto& chunk : input_chunks) {
        // XXX String buffer sizes in the CTB buffers might be an
        // overestimate, so we might need to modify CTB buffers to set the size
        // correctly (e.g. use Reserve/SetSize instead of Resize, etc.).
        size_t chunk_size =
            table_local_memory_size(chunk.table, /*include_dict_size*/ false,
                                    /*approximate_string_size*/ true);
        approx_input_chunks_total_bytes += chunk_size;
        approx_max_input_chunk_size_bytes =
            std::max(approx_max_input_chunk_size_bytes, chunk_size);
        num_input_rows += chunk.table->nrows();
    }
    this->metrics.merge_approx_input_chunks_total_bytes =
        approx_input_chunks_total_bytes;
    this->metrics.merge_approx_max_input_chunk_size_bytes =
        approx_max_input_chunk_size_bytes;

    // Size of each index in the output of sort_values_table_local_get_indices
    // called below.
    // TODO(aneesh) allow using i32 (or smaller) when the number of rows would
    // fit in an i32.
    size_t index_size = sizeof(int64_t);
    size_t index_vector_size = index_size * num_input_rows;
    // The maximum length index vector we want to allocate
    size_t max_index_vector_length = std::numeric_limits<int32_t>::max();

    size_t approx_mem_usage =
        2 * approx_input_chunks_total_bytes + index_vector_size;

    if (this->debug_mode) {
        std::cerr << fmt::format(
                         "{} Number of Input Chunks: {}, Approximate Input "
                         "Size: {}, Max Input Chunk Size: {}, Processing Chunk "
                         "Size: {}, K: {}.",
                         DEBUG_LOG_PREFIX, input_chunks.size(),
                         BytesToHumanReadableString(
                             approx_input_chunks_total_bytes),
                         BytesToHumanReadableString(
                             approx_max_input_chunk_size_bytes),
                         this->processing_chunk_size, this->K)
                  << std::endl;
    }

    // If all 'input_chunks' (plus space needed during sort) fit in mem_budget,
    // then concat and sort them. Similarly, if there's a single chunk, we know
    // that it's already sorted and we can simply output it (after chunking).
    // XXX TODO Technically we only need to ensure that all input chunks fit in
    // (mem_budget_bytes - max_chunk_size_bytes), however, there were some
    // regressions when doing that, particularly in the output append step. This
    // is something to revisit.
    if (this->enable_inmem_concat_sort &&
        (num_input_rows < max_index_vector_length) &&
        ((approx_mem_usage < this->mem_budget_bytes) ||
         input_chunks.size() == 1)) {
        // This is an optimization for when the table fits in memory where
        // we concat all the chunks into a single table and sort the
        // combined table. This avoids the extra overhead incurred by the
        // MergeChunks path.
        this->metrics.performed_inmem_concat_sort = 1;
        if (this->debug_mode) {
            std::cerr << fmt::format(
                             "{} Using in-memory concat-sort optimization.",
                             DEBUG_LOG_PREFIX)
                      << std::endl;
        }

        std::shared_ptr<table_info> table;

        if (input_chunks.size() == 1) {
            // If there's a single chunk, it's already sorted, so we can use it
            // as is.
            TableAndRange chunk = input_chunks.front();
            table = chunk.table;
            table->pin();
            input_chunks.pop_front();
            bodo::vector<int32_t> out_idxs;
            out_idxs.resize(table->nrows());
            std::iota(out_idxs.begin(), out_idxs.end(), 0);

            return SelectRowsAndProduceOutputInMem(table, std::move(out_idxs));
        } else {
            time_pt start_concat = start_timer();
            std::vector<std::shared_ptr<table_info>> unpinned_tables;
            unpinned_tables.reserve(input_chunks.size());
            for (auto& chunk : input_chunks) {
                unpinned_tables.push_back(std::move(chunk.table));
            }
            input_chunks.clear();

            // This will pin one chunk at a time, so the peak memory usage
            // should fit within the budget.
            // XXX If we can sort on a list of tables, then we can skip the
            // concat entirely, sort, and then append into the CTB directly.
            table =
                concat_tables(std::move(unpinned_tables), this->dict_builders,
                              /*input_is_unpinned*/ true);
            this->metrics.finalize_inmem_concat_time += end_timer(start_concat);

            // TODO [BSE-3773] This doesn't require concatenating all tables -
            // sort_values_table_local internally uses a custom comparator to
            // sort a list of indices into a table and then does a RetrieveTable
            // call. We could modify that behavior so instead all the indices
            // map into multiple tables, but we then need to also create a
            // special version of RetrieveTable that can combine indices from
            // multiple tables together.
            time_pt start_sort = start_timer();
            assert(table->nrows() < std::numeric_limits<int32_t>::max());
            bodo::vector<int32_t> out_idxs =
                sort_values_table_local_get_indices<int32_t>(
                    table, this->n_keys, this->vect_ascending.data(),
                    this->na_position.data(), /*is_parallel*/ false,
                    /*start_offset*/ 0, table->nrows());
            this->metrics.finalize_inmem_sort_time += end_timer(start_sort);
            return SelectRowsAndProduceOutputInMem(table, std::move(out_idxs));
        }
    }

    std::vector<std::deque<TableAndRange>> sorted_chunks;

    // Insert the input chunks (already sorted) as single-element deques as the
    // starting point.
    while (!input_chunks.empty()) {
        sorted_chunks.push_back({input_chunks.back()});
        input_chunks.pop_back();
    }

    // Now we will merge K chunks at a time at each level.
    while (sorted_chunks.size() > 1) {
        this->metrics.n_merge_levels++;
        bool is_last = (sorted_chunks.size() <= this->K);
        std::vector<std::deque<TableAndRange>> next_sorted_chunks;
        if (this->debug_mode) {
            std::cerr << fmt::format(
                             "{} Starting Merge Level {}. Number of Input "
                             "Sorted Ranges: {}. Number of Range Merges so "
                             "far: {}. Final Level: {}.",
                             DEBUG_LOG_PREFIX, this->metrics.n_merge_levels,
                             sorted_chunks.size(), this->metrics.n_chunk_merges,
                             is_last ? "Yes" : "No")
                      << std::endl;
        }

        // This loop takes this->K vectors and merges them into 1 on every
        // iteration.
        for (size_t i = 0; i < sorted_chunks.size(); i += this->K) {
            std::vector<std::deque<TableAndRange>> merge_input;
            // Collect the next this->K vectors
            size_t start = i;
            size_t end = std::min(i + this->K, sorted_chunks.size());
            for (size_t j = start; j < end; j++) {
                merge_input.emplace_back(std::move(sorted_chunks[j]));
            }

            if (is_last) {
                if (this->sortlimits.has_value()) {
                    next_sorted_chunks.emplace_back(
                        this->MergeChunks<true, true>(std::move(merge_input)));
                } else {
                    next_sorted_chunks.emplace_back(
                        this->MergeChunks<true, false>(std::move(merge_input)));
                }
            } else {
                if (this->sortlimits.has_value()) {
                    next_sorted_chunks.emplace_back(
                        this->MergeChunks<false, true>(std::move(merge_input)));
                } else {
                    next_sorted_chunks.emplace_back(
                        this->MergeChunks<false, false>(
                            std::move(merge_input)));
                }
            }
            this->metrics.n_chunk_merges++;
        }

        std::swap(sorted_chunks, next_sorted_chunks);
    }

    if (this->debug_mode) {
        std::cerr << fmt::format(
                         "{} Finished Merge. Number of Levels: {}. Number of "
                         "Range Merges: {}.",
                         DEBUG_LOG_PREFIX, this->metrics.n_merge_levels,
                         this->metrics.n_chunk_merges)
                  << std::endl;
    }

    return sorted_chunks[0];
}

template <typename IndexT>
    requires(std::is_same_v<IndexT, int32_t> || std::is_same_v<IndexT, int64_t>)
std::deque<TableAndRange>
ExternalKWayMergeSorter::SelectRowsAndProduceOutputInMem(
    std::shared_ptr<table_info> table, bodo::vector<IndexT>&& out_idxs) {
    const std::string DEBUG_LOG_PREFIX =
        "[DEBUG] ExternalKWayMergeSorter::SelectRowsAndProduceOutputInMem:";

    time_pt start_append = start_timer();
    uint64_t n_cols = table->ncols();
    std::vector<uint64_t> col_inds;

    if (this->debug_mode) {
        std::cerr << fmt::format(
                         "{}: Appending sort result into output buffer.",
                         DEBUG_LOG_PREFIX)
                  << std::endl;
    }

    for (int64_t i = 0; i < this->n_keys; i++) {
        if (!this->dead_keys.empty() && this->dead_keys[i]) {
            // If this is the last reference to this
            // table, we can safely release reference (and potentially
            // memory if any) for the dead keys at this point.
            reset_col_if_last_table_ref(table, i);
        } else {
            col_inds.push_back(i);
        }
    }
    for (uint64_t i = this->n_keys; i < n_cols; i++) {
        col_inds.push_back(i);
    }

    ChunkedTableAndRangeBuilder out_table_builder(
        this->n_keys, this->schema, this->dict_builders,
        this->output_chunk_size,
        DEFAULT_MAX_RESIZE_COUNT_FOR_VARIABLE_SIZE_DTYPES, this->pool,
        this->mm);

    // Use a slice of 'out_idxs' (based on limit/offset) and append it into
    // the CTB.
    size_t offset_ =
        this->sortlimits.has_value() ? this->sortlimits.value().offset : 0;
    size_t limit_ = this->sortlimits.has_value()
                        ? this->sortlimits.value().limit
                        : out_idxs.size();
    size_t left = std::min(out_idxs.size(), offset_);
    size_t right = std::min(out_idxs.size(), left + limit_);
    out_table_builder.AppendBatch(
        std::move(table),
        std::span(out_idxs.begin() + left, out_idxs.begin() + right), col_inds);
    out_table_builder.FinalizeActiveChunk();
    this->metrics.finalize_inmem_output_append_time += end_timer(start_append);
    return out_table_builder.chunks;
}

// Explicit instantiation of templates for linking
template std::deque<TableAndRange>
ExternalKWayMergeSorter::SelectRowsAndProduceOutputInMem(
    std::shared_ptr<table_info> table, bodo::vector<int32_t>&& out_idxs);
template std::deque<TableAndRange>
ExternalKWayMergeSorter::SelectRowsAndProduceOutputInMem(
    std::shared_ptr<table_info> table, bodo::vector<int64_t>&& out_idxs);

double ReservoirSamplingState::random() { return dis(e); }

void ReservoirSamplingState::processInput(
    const std::shared_ptr<table_info>& input_chunk) {
    time_pt start = start_timer();
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
    while (row_to_sample < next_rows_seen) {
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
        samples.ReserveTable(input);
        samples.UnsafeAppendBatch(input, selection);
        this->metrics.n_samples_taken += idxs.size();
    }

    if (samples.data_table->nrows() > static_cast<uint64_t>(sample_size * 10)) {
        // Compact the selected rows
        auto compacted_table = RetrieveTable(samples.data_table, selected_rows);
        samples.Reset();
        samples.ReserveTable(compacted_table);
        samples.UnsafeAppendBatch(compacted_table);
        std::iota(selected_rows.begin(), selected_rows.end(), 0);
        this->metrics.n_sampling_buffer_rebuilds++;
    }
    this->metrics.sampling_process_input_time += end_timer(start);
}

std::shared_ptr<table_info> ReservoirSamplingState::Finalize() {
    if (total_rows_seen < sample_size) {
        return samples.data_table;
    }

    return RetrieveTable(samples.data_table, selected_rows);
}

uint64_t StreamSortState::GetBudget() const {
    int64_t budget = OperatorComptroller::Default()->GetOperatorBudget(op_id);
    if (budget == -1) {
        return static_cast<uint64_t>(
            bodo::BufferPool::Default()->get_memory_size_bytes() *
            SORT_OPERATOR_DEFAULT_MEMORY_FRACTION_OP_POOL);
    }
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

/**
 * @brief Get the maximum allowed concurrent sends/receives. If 'override' is
 * not -1, it will be used. If it is, we will check the
 * 'BODO_STREAM_SORT_MAX_SHUFFLE_CONCURRENT_SENDS' env var. If that is also not
 * set, we will return `n_pes - 1`. Note that the output will always be >=1.
 *
 * @param override Override value. This is used unless it's -1.
 * @return size_t
 */
static size_t get_max_shuffle_concurrent_sends(int64_t override) {
    size_t max_concurrent_sends;
    if (override != -1) {
        max_concurrent_sends = override;
    } else if (char* max_concurrent_sends_env_ = std::getenv(
                   "BODO_STREAM_SORT_MAX_SHUFFLE_CONCURRENT_SENDS")) {
        max_concurrent_sends = std::stoi(max_concurrent_sends_env_);
    } else {
        int n_pes = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
        max_concurrent_sends = n_pes - 1;
    }
    return std::max(static_cast<size_t>(1), max_concurrent_sends);
}

/**
 * @brief Helper function to determine whether or not to use the in-memory
 * concat-sort optimization (if data fits in memory) during the final K-way
 * merge. If 'override' is provided (i.e. not std::nullopt), it will be used. If
 * it isn't, we will check the 'BODO_STREAM_SORT_DISABLE_INMEM_OPTIMIZATION' env
 * var. If it is set to 1, we will return false, else return true (the default).
 *
 * @param override Override value.
 * @return true
 * @return false
 */
static bool get_enable_inmem_concat_sort(
    std::optional<bool> override = std::nullopt) {
    if (override.has_value()) {
        return override.value();
    } else if (char* disable_inmem_concat_sort_env_ =
                   std::getenv("BODO_STREAM_SORT_DISABLE_INMEM_OPTIMIZATION")) {
        // If set to 1, return 0 (i.e. false), else return true.
        return std::strcmp(disable_inmem_concat_sort_env_, "1");
    }
    return true;
}

StreamSortState::StreamSortState(
    int64_t op_id, int64_t n_keys, std::vector<int64_t>&& vect_ascending_,
    std::vector<int64_t>&& na_position_, std::shared_ptr<bodo::Schema> schema_,
    bool parallel, size_t sample_size, size_t output_chunk_size_,
    int64_t kway_merge_chunk_size_, int64_t kway_merge_k_,
    std::optional<bool> enable_inmem_concat_sort_, int64_t shuffle_chunksize_,
    int64_t shuffle_max_concurrent_sends_)
    : op_id(op_id),
      n_keys(n_keys),
      vect_ascending(vect_ascending_),
      na_position(na_position_),
      parallel(parallel),
      output_chunk_size(output_chunk_size_),
      kway_merge_chunksize(kway_merge_chunk_size_),
      kway_merge_k(kway_merge_k_),
      enable_inmem_concat_sort(
          get_enable_inmem_concat_sort(enable_inmem_concat_sort_)),
      shuffle_chunksize(shuffle_chunksize_),
      shuffle_max_concurrent_msgs(
          get_max_shuffle_concurrent_sends(shuffle_max_concurrent_sends_)),
      mem_budget_bytes(StreamSortState::GetBudget()),
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
      // Note that builder only stores references to the vectors owned by
      // this object, so we must refer to the instances on this class, not
      // the arguments.
      builder(schema, dict_builders, STREAMING_BATCH_SIZE,
              DEFAULT_MAX_RESIZE_COUNT_FOR_VARIABLE_SIZE_DTYPES, op_pool,
              op_mm),
      reservoir_sampling_state(n_keys, sample_size, dict_builders, schema) {
    if (char* debug_mode_env_ = std::getenv("BODO_DEBUG_STREAM_SORT")) {
        this->debug_mode = !std::strcmp(debug_mode_env_, "1");
    }
}

StreamSortLimitOffsetState::StreamSortLimitOffsetState(
    int64_t op_id, int64_t n_keys, std::vector<int64_t>&& vect_ascending_,
    std::vector<int64_t>&& na_position_, std::shared_ptr<bodo::Schema> schema_,
    bool parallel, int64_t limit, int64_t offset, size_t output_chunk_size_,
    int64_t kway_merge_chunk_size_, int64_t kway_merge_k_,
    std::optional<bool> enable_inmem_concat_sort_,
    bool enable_small_limit_optimization)
    : StreamSortState(
          op_id, n_keys, std::move(vect_ascending_), std::move(na_position_),
          std::move(schema_), parallel, DEFAULT_SAMPLE_SIZE, output_chunk_size_,
          kway_merge_chunk_size_, kway_merge_k_, enable_inmem_concat_sort_),
      sortlimit(static_cast<size_t>(limit), static_cast<size_t>(offset)),
      top_k(this->schema, this->dict_builders, this->n_keys,
            this->vect_ascending, this->na_position, this->dead_keys,
            this->mem_budget_bytes, -1,
            this->sortlimit.limit + this->sortlimit.offset, 0,
            this->output_chunk_size, this->kway_merge_k,
            this->kway_merge_chunksize, this->enable_inmem_concat_sort,
            /*debug_mode_*/ false, op_pool, op_mm) {
    this->limit_small_flag = (sortlimit.sum() <= SORT_SMALL_LIMIT_OFFSET_CAP &&
                              enable_small_limit_optimization);
}

void StreamSortState::ConsumeBatch(std::shared_ptr<table_info> table) {
    row_info.first += table_local_memory_size(table, false);
    row_info.second += table->nrows();

    time_pt start = start_timer();
    std::shared_ptr<table_info> unified =
        UnifyDictionaryArrays(std::move(table), dict_builders);
    this->metrics.input_append_time += end_timer(start);

    this->reservoir_sampling_state.processInput(unified);

    start = start_timer();
    this->builder.AppendBatch(unified);
    this->metrics.input_append_time += end_timer(start);
}

void StreamSortLimitOffsetState::ConsumeBatch(
    std::shared_ptr<table_info> table) {
    row_info.first += table_local_memory_size(table, false);
    row_info.second += table->nrows();

    if (this->sortlimit.limit == 0) {
        return;
    }

    if (!limit_small_flag) {
        time_pt start = start_timer();
        std::shared_ptr<table_info> unified =
            UnifyDictionaryArrays(std::move(table), dict_builders);
        this->metrics.input_append_time += end_timer(start);

        this->reservoir_sampling_state.processInput(unified);

        start = start_timer();
        this->builder.AppendBatch(unified);
        this->metrics.input_append_time += end_timer(start);
    } else {
        // Maintain a heap of at most limit + offset elements when limit/offset
        // is small
        time_pt start = start_timer();
        this->top_k.AppendChunk(table);
        this->metrics.topk_heap_append_chunk_time += end_timer(start);
        start = start_timer();
        this->top_k.sorted_input_chunks_builder->chunks =
            top_k.Finalize(/*reset_input_builder*/ false);
        this->metrics.topk_heap_update_time += end_timer(start);
    }
}

void StreamSortState::FinalizeBuild() {
    if (this->build_finalized) {
        return;
    }
    time_pt start = start_timer();

    this->builder.FinalizeActiveChunk();
    this->metrics.n_input_chunks = this->builder.chunks.size();

    // Attempt to increase the budget - this will allow for larger chunk sizes
    // and fewer pin/unpin calls.
    OperatorComptroller::Default()->RequestAdditionalBudget(op_id, -1);
    this->mem_budget_bytes = this->GetBudget();
    this->metrics.input_chunks_size_bytes_total = this->row_info.first;

    if (this->debug_mode) {
        std::cerr
            << fmt::format(
                   "[DEBUG] StreamSortState::FinalizeBuild: Memory Budget: {}.",
                   BytesToHumanReadableString(this->mem_budget_bytes))
            << std::endl;
    }

    // Make all dictionaries global
    time_pt start_dict_unif = start_timer();
    for (auto& dict_builder : dict_builders) {
        if (!dict_builder) {
            continue;
        }
        recursive_unify_dict_builder_globally(dict_builder);
    }
    dummy_output_chunk = UnifyDictionaryArrays(std::move(dummy_output_chunk),
                                               dict_builders, true);
    this->metrics.global_dict_unification_time += end_timer(start_dict_unif);

    // Figure out optimal chunk size
    std::unique_ptr<int64_t[]> in_info = std::make_unique<int64_t[]>(2);
    in_info[0] = row_info.first;
    in_info[1] = row_info.second;
    std::unique_ptr<int64_t[]> sum_row_info = std::make_unique<int64_t[]>(2);
    if (parallel) {
        CHECK_MPI(
            MPI_Allreduce(in_info.get(), sum_row_info.get(), 2,
                          MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD),
            "StreamSortState::FinalizeBuild: MPI error on MPI_Allreduce:");
    }

    if (parallel && sum_row_info[1] > 0) {
        this->bytes_per_row =
            std::ceil(((double)sum_row_info[0]) / ((double)sum_row_info[1]));
    } else if (!parallel && row_info.second > 0) {
        this->bytes_per_row =
            std::ceil(((double)row_info.first) / ((double)row_info.second));
    }

    std::deque<std::shared_ptr<table_info>> local_chunks;
    std::swap(builder.chunks, local_chunks);
    this->GlobalSort(std::move(local_chunks));
    this->metrics.total_finalize_time += end_timer(start);
    this->build_finalized = true;

    // Report metrics
    std::vector<MetricBase> metrics_out;
    this->ReportBuildMetrics(metrics_out);
}

std::shared_ptr<table_info> StreamSortState::GetParallelSortBounds(
    std::shared_ptr<table_info>&& local_samples) {
    ScopedTimer timer(this->metrics.get_bounds_total_time);
    int n_pes, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // XXX We already unify the dictionaries in StreamSortState::FinalizeBuild,
    // so some of this might be redundant.
    // Combine the dictionaries from all the local samples across all ranks
    time_pt start = start_timer();
    for (size_t i = 0; i < local_samples->ncols(); i++) {
        if (dict_builders[i]) {
            recursive_make_array_global_and_unique(local_samples->columns[i],
                                                   true);
        }
    }
    this->metrics.get_bounds_dict_unify_time += end_timer(start);

    // Collecting all samples globally
    std::shared_ptr<table_info> ref_table = alloc_table_like(local_samples);
    bool all_gather = false;
    start = start_timer();
    std::shared_ptr<table_info> all_samples =
        gather_table(std::move(local_samples), n_keys, all_gather, parallel);
    this->metrics.get_bounds_gather_samples_time += end_timer(start);

    // Compute split bounds from the samples.
    // Output is broadcasted to all ranks.
    start = start_timer();
    this->bounds_ = compute_bounds_from_samples(
        std::move(all_samples), std::move(ref_table), n_keys,
        vect_ascending.data(), na_position.data(), myrank, n_pes, parallel);
    // Transpose bounds to use the same indices as the local builders - we know
    // that the dictionary builder has all keys at this point.
    this->bounds_ = UnifyDictionaryArrays(std::move(bounds_), dict_builders);
    this->metrics.get_bounds_compute_bounds_time += end_timer(start);

    return bounds_;
}

std::vector<std::deque<std::shared_ptr<table_info>>>
StreamSortState::PartitionChunksByRank(
    const ExternalKWayMergeSorter& kway_merge_sorter, int n_pes,
    std::shared_ptr<table_info> bounds,
    std::deque<std::shared_ptr<table_info>>&& local_chunks) {
    ScopedTimer timer(this->metrics.partition_chunks_total_time);
    // XXX Instead of using a SortedCTB and doing the sort during this
    // partitioning step, we could instead sort "during" the shuffle and right
    // before sending the chunks. That may allow better overlap. However, it may
    // also "delay" data transfer, so we may need a way to balance this.
    std::vector<SortedChunkedTableBuilder> rankToChunkedBuilders;
    assert(static_cast<uint64_t>(n_pes) == bounds->nrows() + 1);

    // For each rank, we build n_pes ChunkedTableBuilder to store tables to
    // pass. To calculate the shuffle chunk size, we get the min budget on any
    // rank. 'bytes_per_row' is already averaged globally.
    assert(this->parallel);
    uint64_t global_min_mem_budget_bytes = this->mem_budget_bytes;
    CHECK_MPI(
        MPI_Allreduce(&this->mem_budget_bytes, &global_min_mem_budget_bytes, 1,
                      MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD),
        "StreamSortState::PartitionChunksByRank: MPI error on MPI_Allreduce:");
    int myrank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    // Use the optimal chunk size unless an override has been provided.
    size_t shuffle_chunk_size_ =
        this->shuffle_chunksize != -1
            ? this->shuffle_chunksize
            : get_optimal_sort_shuffle_chunk_size(
                  this->bytes_per_row, global_min_mem_budget_bytes,
                  this->shuffle_max_concurrent_msgs);
    this->metrics.shuffle_chunk_size = shuffle_chunk_size_;
    for (int rank_id = 0; rank_id < n_pes; rank_id++) {
        rankToChunkedBuilders.emplace_back(
            this->schema, this->dict_builders, this->n_keys,
            this->vect_ascending, this->na_position, shuffle_chunk_size_,
            DEFAULT_MAX_RESIZE_COUNT_FOR_VARIABLE_SIZE_DTYPES);
    }

    if (this->debug_mode && myrank == 0) {
        std::cerr << fmt::format(
                         "[DEBUG] StreamSortState::PartitionChunksByRank: "
                         "Shuffle Chunk Size: {}.",
                         shuffle_chunk_size_)
                  << std::endl;
    }

    // A vector containing all the possible ranks. This could be
    // std::ranges::iota_view, but it's not supported on all compilers yet.
    std::vector<int> ranks(n_pes);
    std::iota(ranks.begin(), ranks.end(), 0);

    time_pt start_pin, start_append, start_check;
    for (auto& chunk : local_chunks) {
        start_pin = start_timer();
        chunk->pin();
        this->metrics.partition_chunks_pin_time += end_timer(start_pin);

        start_check = start_timer();
        std::vector<std::vector<int64_t>> rankToRows(n_pes);
        int64_t table_size = static_cast<int64_t>(chunk->nrows());
        for (int64_t row = 0; row < table_size; row++) {
            // Find the first rank where bounds[rank] > chunk.table[row]
            auto dst_rank =
                std::ranges::lower_bound(ranks, row, [&](int rank, int row) {
                    if (rank == (n_pes - 1)) {
                        return false;
                    }
                    return kway_merge_sorter.Compare(bounds, rank, chunk, row);
                });
            assert(dst_rank != ranks.end());
            rankToRows[*dst_rank].push_back(row);
        }
        this->metrics.partition_chunks_compute_dest_rank_time +=
            end_timer(start_check);

        // TODO(aneesh): to improve IO/compute overlap we should send a table
        // as soon as it's ready
        start_append = start_timer();
        for (int rank_id = 0; rank_id < n_pes; rank_id++) {
            rankToChunkedBuilders[rank_id].AppendBatch(chunk,
                                                       rankToRows[rank_id]);
        }
        this->metrics.partition_chunks_append_time += end_timer(start_append);

        chunk.reset();
    }

    std::vector<std::deque<std::shared_ptr<table_info>>> rankToChunks(n_pes);
    for (int rank_id = 0; rank_id < n_pes; rank_id++) {
        start_append = start_timer();
        rankToChunkedBuilders[rank_id].Finalize();
        this->metrics.partition_chunks_append_time += end_timer(start_append);
        this->metrics.partition_chunks_sort_time +=
            rankToChunkedBuilders[rank_id].sort_time;
        this->metrics.partition_chunks_sort_copy_time +=
            rankToChunkedBuilders[rank_id].sort_copy_time;
        for (auto& chunk : rankToChunkedBuilders[rank_id].chunks) {
            rankToChunks[rank_id].push_back(std::move(chunk.table));
        }
    }

    if (this->debug_mode) {
        std::cerr << fmt::format(
                         "[DEBUG] StreamSortState::PartitionChunksByRank: "
                         "Finished range partitioning chunks for shuffle.")
                  << std::endl;
    }

    return rankToChunks;
}

void StreamSortLimitOffsetState::SmallLimitOptim() {
    int n_pes, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    std::vector<std::shared_ptr<table_info>> chunks;
    std::deque<TableAndRange> local_chunks;
    std::swap(this->top_k.sorted_input_chunks_builder->chunks, local_chunks);

    // Every rank concat local tables and send to rank 0 in 1 batch
    time_pt start_concat = start_timer();
    std::shared_ptr<table_info> local_concat_tables = dummy_output_chunk;
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
    this->metrics.small_limit_local_concat_time += end_timer(start_concat);

    std::shared_ptr<table_info> collected_table;
    if (this->parallel) {
        // Send all data to rank 0 synchronously
        time_pt start_gather = start_timer();
        collected_table = gather_table(std::move(local_concat_tables), -1,
                                       false, parallel, 0);
        this->metrics.small_limit_gather_time += end_timer(start_gather);
    } else {
        collected_table = std::move(local_concat_tables);
    }

    if (myrank == 0 || !this->parallel) {
        collected_table =
            UnifyDictionaryArrays(std::move(collected_table), dict_builders);
        time_pt start_sort = start_timer();
        bodo::vector<int64_t> indices =
            sort_values_table_local_get_indices<int64_t>(
                collected_table, n_keys, vect_ascending.data(),
                na_position.data(), false, 0, collected_table->nrows());
        this->metrics.small_limit_rank0_sort_time += end_timer(start_sort);

        time_pt start_append = start_timer();
        uint64_t n_cols = collected_table->ncols();
        std::vector<uint64_t> col_inds;
        for (int64_t i = 0; i < n_keys; i++) {
            if (!dead_keys.empty() && dead_keys[i]) {
                // If this is the last reference to this
                // table, we can safely release reference (and potentially
                // memory if any) for the dead keys at this point.
                reset_col_if_last_table_ref(collected_table, i);
            } else {
                col_inds.push_back(i);
            }
        }
        for (uint64_t i = n_keys; i < n_cols; i++) {
            col_inds.push_back(i);
        }

        // XXX We need to use a pruned "schema" to drop the columns that are not
        // needed. Otherwise, we're allocating space we don't need
        // unnecessarily.
        ChunkedTableBuilder out_table_builder(
            schema, dict_builders, this->output_chunk_size,
            DEFAULT_MAX_RESIZE_COUNT_FOR_VARIABLE_SIZE_DTYPES, op_pool, op_mm);

        bodo::vector<int64_t> offset_indices;
        for (size_t i = sortlimit.offset; i < sortlimit.sum(); i++) {
            if (i >= indices.size()) {
                break;
            }
            offset_indices.push_back(indices[i]);
        }
        out_table_builder.AppendBatch(collected_table, offset_indices,
                                      col_inds);

        out_table_builder.FinalizeActiveChunk();
        for (auto chunk : out_table_builder.chunks) {
            this->output_chunks.push_back(chunk);
        }
        this->metrics.small_limit_rank0_output_append_time +=
            end_timer(start_append);
    }
}

SortLimits StreamSortLimitOffsetState::ComputeLocalLimit(
    size_t local_nrows) /*const*/ {
    ScopedTimer timer(this->metrics.compute_local_limit_time);
    int n_pes, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    size_t limit = this->sortlimit.limit;
    size_t offset = this->sortlimit.offset;
    std::vector<size_t> nrows_collect(n_pes);
    CHECK_MPI(MPI_Allgather(&local_nrows, 1, MPI_UNSIGNED_LONG_LONG,
                            nrows_collect.data(), 1, MPI_UNSIGNED_LONG_LONG,
                            MPI_COMM_WORLD),
              "StreamSortLimitOffsetState::ComputeLocalLimit: MPI error on "
              "MPI_Allgather:");
    size_t total_rows_before = 0;
    for (int64_t i = 0; i < myrank; i++) {
        total_rows_before += nrows_collect[i];
    }
    if (total_rows_before >= limit + offset ||
        total_rows_before + local_nrows <= offset || limit == 0) {
        return SortLimits(0, 0);
    }
    size_t finalize_offset =
        total_rows_before >= offset ? 0 : offset - total_rows_before;
    size_t finalize_limit =
        std::min(limit + offset - total_rows_before, local_nrows) -
        finalize_offset;
    return SortLimits(finalize_limit, finalize_offset);
}

void StreamSortState::GlobalSort_NonParallel(
    std::deque<std::shared_ptr<table_info>>&& local_chunks) {
    ExternalKWayMergeSorter kway_merge_sorter = this->GetKWayMergeSorter();
    time_pt start_append = start_timer();
    for (auto& chunk : local_chunks) {
        chunk->pin();
        kway_merge_sorter.AppendChunk(std::move(chunk));
    }
    this->metrics.kway_merge_sorter_append_time += end_timer(start_append);
    std::deque<TableAndRange> out_chunks = kway_merge_sorter.Finalize();
    this->metrics.external_kway_merge_sort_finalize_metrics =
        kway_merge_sorter.metrics;

    for (const auto& chunk : out_chunks) {
        output_chunks.push_back(std::move(chunk.table));
    }
}

ExternalKWayMergeSorter StreamSortState::GlobalSort_Partition(
    std::deque<std::shared_ptr<table_info>>&& local_chunks) {
    int n_pes, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    ExternalKWayMergeSorter kway_merge_sorter = this->GetKWayMergeSorter();

    std::shared_ptr<table_info> bounds =
        this->GetParallelSortBounds(reservoir_sampling_state.Finalize());

    std::vector<std::deque<std::shared_ptr<table_info>>> rankToChunks =
        this->PartitionChunksByRank(kway_merge_sorter, n_pes, bounds,
                                    std::move(local_chunks));

    time_pt start_append = start_timer();
    for (auto& chunk : rankToChunks[myrank]) {
        chunk->pin();
        this->metrics.n_rows_after_shuffle += chunk->nrows();
        kway_merge_sorter.AppendChunk(std::move(chunk));
    }
    rankToChunks[myrank].clear();
    this->metrics.kway_merge_sorter_append_time += end_timer(start_append);

    time_pt start_shuffle = start_timer();
    std::vector<size_t> rankToCurrentChunk(n_pes);
    // Tuple of destination rank and send_state.
    std::vector<std::tuple<int, AsyncShuffleSendState>> send_states;
    std::vector<AsyncShuffleRecvState> recv_states;

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
    bool have_chunks_to_send = HaveChunksToSend();

    // Barrier to test for completion
    MPI_Request is_last_request = MPI_REQUEST_NULL;
    // If there's only a single rank, skip the loop below
    bool is_last = n_pes == 1;

    // There may be multiple chunks in flight to the same destination rank. MPI
    // does guarantee that messages will be *matched* in the order that they are
    // sent, however, it doesn't guarantee that messages will *finish* receiving
    // in the same order. Let's say two chunks are being sent from A to B. A
    // will post the metadata (MD) message for the first chunk followed by its
    // buffers. It will then post the MD message for the second chunk followed
    // by its buffers. B will Improbe and get the first MD message and then the
    // second MD message (in order, as guaranteed by the MPI standard). Let's
    // say the Imrecv for the second MD message *finishes* first (permitted by
    // MPI standard). Then, B will immediately post the IRecvs based on these
    // lengths. In that case, these IRecvs (with lengths corresponding to the
    // second chunk) will match the messages containing the buffers of the first
    // chunk, leading to a size mismatch. To avoid this issue, we use a
    // separate set of tags for each concurrent chunk. See BSE-3892 for more
    // details.
    std::vector<std::unordered_set<int>> ranks_to_inflight_tags(n_pes);

    auto GetNextTagForRank = [&](int rank) {
        return get_next_available_tag(ranks_to_inflight_tags[rank]);
    };

    auto HaveChunksToSendToRankWithFreeTag = [&]() {
        for (int i = 0; i < n_pes; i++) {
            if (i == myrank) {
                continue;
            }
            if (rankToCurrentChunk[i] < rankToChunks[i].size() &&
                GetNextTagForRank(i) != -1) {
                return true;
            }
        }
        return false;
    };

    if (this->debug_mode) {
        std::cerr << "[DEBUG] StreamSortState::GlobalSort_Partition: Starting "
                     "data shuffle."
                  << std::endl;
    }

    // Initialize every rank to send to their neighbor. This is to prevent the
    // situation where all ranks send to a single host simultaneously making it
    // harder to overlap IO and compute.
    int host_to_send_to = (myrank + 1) % n_pes;

    bool barrier_posted = false;
    time_pt start_issend, start_irecv, start_barrier_test;
    // Loop until the barrier is reached and we have no outstanding IO
    // requests
    while (!is_last || have_chunks_to_send || !recv_states.empty() ||
           !send_states.empty()) {
        // Don't send unless the number of inflight sends is less than
        // the max allowed (n_pes - 1 by default) and there are chunks to send
        // to a rank with remaining MPI tags.
        // XXX TODO Since there is a max allowed chunk size, we could
        // dynamically choose a larger 'shuffle_max_concurrent_sends' to fully
        // utilize the available memory and allow more concurrent data transfer.
        while ((send_states.size() < this->shuffle_max_concurrent_msgs) &&
               HaveChunksToSendToRankWithFreeTag()) {
            int starting_msg_tag = GetNextTagForRank(host_to_send_to);
            if ((rankToCurrentChunk[host_to_send_to] <
                 rankToChunks[host_to_send_to].size()) &&
                (starting_msg_tag != -1)) {
                size_t chunk_id = rankToCurrentChunk[host_to_send_to];
                std::shared_ptr<table_info> table_to_send =
                    std::move(rankToChunks[host_to_send_to][chunk_id]);
                table_to_send->pin();

                size_t nrows = table_to_send->nrows();
                auto hashes = std::make_shared<uint32_t[]>(nrows);
                std::fill(hashes.get(), hashes.get() + nrows, host_to_send_to);

                // Shuffle all the data
                this->metrics.shuffle_total_sent_nrows += nrows;
                this->metrics.shuffle_total_approx_sent_size_bytes +=
                    table_local_memory_size(table_to_send, false);
                this->metrics.shuffle_approx_sent_size_bytes_dicts +=
                    table_local_dictionary_memory_size(table_to_send);
                start_issend = start_timer();
                // XXX shuffle_issend will make redundant copies of many of
                // these buffers even though they can be reused as is. This is
                // something to optimize.
                send_states.emplace_back(
                    host_to_send_to,
                    shuffle_issend(std::move(table_to_send), std::move(hashes),
                                   nullptr, MPI_COMM_WORLD, starting_msg_tag));
                ranks_to_inflight_tags[host_to_send_to].insert(
                    starting_msg_tag);
                this->metrics.shuffle_issend_time += end_timer(start_issend);
                this->metrics.n_shuffle_send++;

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
        }
        this->metrics.max_concurrent_sends =
            std::max(this->metrics.max_concurrent_sends,
                     static_cast<int64_t>(send_states.size()));

        // Check if we have any more buffers to send on the next
        // iteration
        have_chunks_to_send = HaveChunksToSend();

        // Remove send state if recv done
        std::erase_if(send_states, [this, &ranks_to_inflight_tags](
                                       std::tuple<int, AsyncShuffleSendState>&
                                           dest_rank_send_state_tup) {
            time_pt start_send_done = start_timer();
            auto& [dest_rank, send_state] = dest_rank_send_state_tup;
            bool done = send_state.sendDone();
            this->metrics.shuffle_send_done_check_time +=
                end_timer(start_send_done);
            this->metrics.shuffle_n_send_done_checks++;
            if (done) {
                // Remove the tag and make it available for re-use.
                ranks_to_inflight_tags[dest_rank].erase(
                    send_state.get_starting_msg_tag());
            }

            return done;
        });
        if (!barrier_posted && send_states.empty() && !have_chunks_to_send) {
            CHECK_MPI(MPI_Ibarrier(MPI_COMM_WORLD, &is_last_request),
                      "StreamSortState::GlobalSort_Partition: MPI error "
                      "on MPI_Ibarrier:");
            barrier_posted = true;
        }

        if (recv_states.size() < this->shuffle_max_concurrent_msgs) {
            // Check if we can receive
            start_irecv = start_timer();
            size_t prev_recv_states_size = recv_states.size();
            // XXX TODO Limit the number of receive states that can be posted
            // during this call (e.g. at most n_pes - 1).
            shuffle_irecv(this->dummy_output_chunk, MPI_COMM_WORLD, recv_states,
                          this->shuffle_max_concurrent_msgs);
            this->metrics.n_shuffle_recv +=
                recv_states.size() - prev_recv_states_size;
            this->metrics.shuffle_irecv_time += end_timer(start_irecv);
            this->metrics.shuffle_n_irecvs++;
        }
        this->metrics.max_concurrent_recvs =
            std::max(this->metrics.max_concurrent_recvs,
                     static_cast<int64_t>(recv_states.size()));

        // If we have any completed receives, add them to the builder
        std::erase_if(recv_states, [&](AsyncShuffleRecvState& s) {
            time_pt start_done = start_timer();
            auto [done, table] = s.recvDone(dict_builders, MPI_COMM_WORLD,
                                            this->metrics.ishuffle_metrics);
            this->metrics.shuffle_recv_done_check_time += end_timer(start_done);
            this->metrics.shuffle_n_recv_done_checks++;
            if (done && table->nrows()) {
                this->metrics.shuffle_total_recv_nrows += table->nrows();
                this->metrics.shuffle_total_recv_size_bytes +=
                    table_local_memory_size(table, false, false);
                time_pt start_append = start_timer();
                kway_merge_sorter.AppendChunk(std::move(table));
                this->metrics.kway_merge_sorter_append_time +=
                    end_timer(start_append);
                // TODO(aneesh) Every K tables, we could start the merge process
                // while we wait for messages.
            }
            return done;
        });

        // If we've already posted the barrier test to see if all other ranks
        // have as well.
        if (!is_last && barrier_posted) {
            start_barrier_test = start_timer();
            int flag = 0;
            CHECK_MPI(MPI_Test(&is_last_request, &flag, MPI_STATUS_IGNORE),
                      "StreamSortState::GlobalSort_Partition: MPI error on "
                      "MPI_Ibarrier:");
            is_last = flag;
            this->metrics.shuffle_barrier_test_time +=
                end_timer(start_barrier_test);
            this->metrics.shuffle_n_barrier_tests++;
        }
    }
    this->metrics.shuffle_total_time += end_timer(start_shuffle);
    this->metrics.n_rows_after_shuffle +=
        this->metrics.shuffle_total_recv_nrows;

    if (this->debug_mode) {
        std::cerr
            << fmt::format(
                   "[DEBUG] StreamSortState::GlobalSort_Partition: Finished "
                   "data shuffle. Number of sent chunks: {}. Number of "
                   "received chunks: {}. Number of rows after shuffle: {}. "
                   "Total Shuffle Time: {}us.",
                   this->metrics.n_shuffle_send, this->metrics.n_shuffle_recv,
                   this->metrics.n_rows_after_shuffle,
                   this->metrics.shuffle_total_time)
            << std::endl;
    }

    return kway_merge_sorter;
}

void StreamSortState::GlobalSort(
    std::deque<std::shared_ptr<table_info>>&& local_chunks) {
    if (!parallel) {
        this->GlobalSort_NonParallel(std::move(local_chunks));
        return;
    }
    ExternalKWayMergeSorter kway_merge_sorter =
        this->GlobalSort_Partition(std::move(local_chunks));
    std::deque<TableAndRange> out_chunks = kway_merge_sorter.Finalize();
    this->metrics.external_kway_merge_sort_finalize_metrics =
        kway_merge_sorter.metrics;
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
        this->SmallLimitOptim();
        return;
    }

    if (!parallel) {
        this->GlobalSort_NonParallel(std::move(local_chunks));
        return;
    }

    ExternalKWayMergeSorter kway_merge_sorter =
        this->GlobalSort_Partition(std::move(local_chunks));
    // Compute new local limit to be used during Finalize based on global sizes
    // and update the kway-merge-sorter.
    // XXX If we do this computation based on information available
    // after PartitionChunksByRank (but before the actual shuffle), we could
    // avoid some of the shuffle as well.
    SortLimits new_limit_offset = this->ComputeLocalLimit(
        kway_merge_sorter.sorted_input_chunks_builder->total_remaining);
    kway_merge_sorter.UpdateLimitOffset(new_limit_offset.limit,
                                        new_limit_offset.offset);
    std::deque<TableAndRange> out_chunks = kway_merge_sorter.Finalize();
    this->metrics.external_kway_merge_sort_finalize_metrics =
        kway_merge_sorter.metrics;
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

std::vector<std::shared_ptr<table_info>>
StreamSortState::GetAllOutputUnpinned() {
    return std::move(output_chunks);
}

void StreamSortState::ReportBuildMetrics(std::vector<MetricBase>& metrics_out) {
    assert(this->build_finalized);
    if (this->reported_build_metrics) {
        return;
    }

#define APPEND_STAT_METRIC(field) \
    metrics_out.push_back(StatMetric(#field, this->metrics.field));
#define APPEND_TIMER_METRIC(field) \
    metrics_out.push_back(TimerMetric(#field, this->metrics.field));

    metrics_out.reserve(metrics_out.size() + 128);

    // Consume step:
    APPEND_STAT_METRIC(input_chunks_size_bytes_total);
    APPEND_STAT_METRIC(n_input_chunks);
    APPEND_TIMER_METRIC(input_append_time);

    // Sampling metrics
    this->reservoir_sampling_state.metrics.ExportMetrics(metrics_out);

    // FinalizeBuild metrics:
    MetricBase::StatValue final_budget_bytes = this->mem_budget_bytes;
    metrics_out.push_back(StatMetric("final_budget_bytes", final_budget_bytes));
    MetricBase::StatValue final_bytes_per_row = this->bytes_per_row;
    metrics_out.push_back(
        StatMetric("final_bytes_per_row", final_bytes_per_row, this->parallel));
    APPEND_TIMER_METRIC(global_dict_unification_time);
    APPEND_TIMER_METRIC(total_finalize_time);
    APPEND_TIMER_METRIC(kway_merge_sorter_append_time);
    APPEND_TIMER_METRIC(get_bounds_total_time);
    APPEND_TIMER_METRIC(get_bounds_dict_unify_time);
    APPEND_TIMER_METRIC(get_bounds_gather_samples_time);
    APPEND_TIMER_METRIC(get_bounds_compute_bounds_time);
    APPEND_TIMER_METRIC(partition_chunks_total_time);
    APPEND_TIMER_METRIC(partition_chunks_pin_time);
    APPEND_TIMER_METRIC(partition_chunks_append_time);
    APPEND_TIMER_METRIC(partition_chunks_sort_time);
    APPEND_TIMER_METRIC(partition_chunks_sort_copy_time);
    APPEND_TIMER_METRIC(partition_chunks_compute_dest_rank_time);
    APPEND_TIMER_METRIC(shuffle_total_time);
    metrics_out.push_back(StatMetric("shuffle_chunk_size",
                                     this->metrics.shuffle_chunk_size, true));
    APPEND_TIMER_METRIC(shuffle_issend_time);
    APPEND_TIMER_METRIC(shuffle_send_done_check_time);
    APPEND_STAT_METRIC(shuffle_n_send_done_checks);
    APPEND_TIMER_METRIC(shuffle_irecv_time);
    APPEND_STAT_METRIC(shuffle_n_irecvs);
    APPEND_TIMER_METRIC(shuffle_recv_done_check_time);
    APPEND_STAT_METRIC(shuffle_n_recv_done_checks);
    APPEND_TIMER_METRIC(shuffle_barrier_test_time);
    APPEND_STAT_METRIC(shuffle_n_barrier_tests);
    APPEND_STAT_METRIC(n_shuffle_send);
    APPEND_STAT_METRIC(n_shuffle_recv);
    APPEND_STAT_METRIC(shuffle_total_sent_nrows);
    APPEND_STAT_METRIC(shuffle_total_recv_nrows);
    APPEND_STAT_METRIC(shuffle_total_approx_sent_size_bytes);
    APPEND_STAT_METRIC(shuffle_approx_sent_size_bytes_dicts);
    APPEND_STAT_METRIC(shuffle_total_recv_size_bytes);
    APPEND_STAT_METRIC(n_rows_after_shuffle);
    APPEND_STAT_METRIC(max_concurrent_sends);
    APPEND_STAT_METRIC(max_concurrent_recvs);
    metrics_out.push_back(StatMetric(
        "approx_recv_size_bytes_dicts",
        this->metrics.ishuffle_metrics.approx_recv_size_bytes_dicts));
    metrics_out.push_back(
        TimerMetric("dict_unification_time",
                    this->metrics.ishuffle_metrics.dict_unification_time));
    this->metrics.external_kway_merge_sort_finalize_metrics.ExportMetrics(
        metrics_out);

    // DictBuilder metrics
    DictBuilderMetrics dict_builder_metrics;
    MetricBase::StatValue n_dict_builders = 0;
    for (const auto& dict_builder : this->dict_builders) {
        if (dict_builder != nullptr) {
            dict_builder_metrics.add_metrics(dict_builder->GetMetrics());
            n_dict_builders++;
        }
    }
    if (n_dict_builders > 0) {
        metrics_out.emplace_back(
            StatMetric("n_dict_builders", n_dict_builders, true));
        dict_builder_metrics.add_to_metrics(metrics_out, "dict_builders_");
    }

    if (this->op_id != -1) {
        QueryProfileCollector::Default().RegisterOperatorStageMetrics(
            QueryProfileCollector::MakeOperatorStageID(
                this->op_id, QUERY_PROFILE_SORT_BUILD_STAGE_ID),
            std::move(metrics_out));
    }
    this->reported_build_metrics = true;

#undef APPEND_STAT_METRIC
#undef APPEND_TIMER_METRIC
}

void StreamSortLimitOffsetState::ReportBuildMetrics(
    std::vector<MetricBase>& metrics_out) {
    if (this->reported_build_metrics) {
        return;
    }

#define APPEND_TIMER_METRIC(field) \
    metrics_out.push_back(TimerMetric(#field, this->metrics.field));

    metrics_out.reserve(metrics_out.size() + 16);
    MetricBase::StatValue is_limit_offset_case = 1;
    metrics_out.push_back(
        StatMetric("is_limit_offset_case", is_limit_offset_case, true));
    MetricBase::StatValue is_small_limit_case = this->limit_small_flag;
    metrics_out.push_back(
        StatMetric("is_small_limit_case", is_small_limit_case, true));
    if (this->limit_small_flag) {
        APPEND_TIMER_METRIC(topk_heap_append_chunk_time);
        APPEND_TIMER_METRIC(topk_heap_update_time);
        APPEND_TIMER_METRIC(small_limit_local_concat_time);
        APPEND_TIMER_METRIC(small_limit_gather_time);
        APPEND_TIMER_METRIC(small_limit_rank0_sort_time);
        APPEND_TIMER_METRIC(small_limit_rank0_output_append_time);
    } else {
        APPEND_TIMER_METRIC(compute_local_limit_time);
    }

    StreamSortState::ReportBuildMetrics(metrics_out);

#undef APPEND_TIMER_METRIC
}

StreamSortState* stream_sort_state_init_py_entry(
    int64_t op_id, int64_t limit, int64_t offset, int64_t n_keys,
    int64_t* vect_ascending, int64_t* na_position, int8_t* arr_c_types,
    int8_t* arr_array_types, int64_t n_arrs, bool parallel) {
    try {
        // Copy the per-column configuration into owned vectors
        std::vector<int64_t> vect_ascending_(n_keys);
        std::vector<int64_t> na_position_(n_keys);
        for (int64_t i = 0; i < n_keys; i++) {
            vect_ascending_[i] = vect_ascending[i];
            na_position_[i] = na_position[i];
        }
        std::shared_ptr<bodo::Schema> schema = bodo::Schema::Deserialize(
            std::vector<int8_t>(arr_array_types, arr_array_types + n_arrs),
            std::vector<int8_t>(arr_c_types, arr_c_types + n_arrs));

        if ((limit != -1 || offset != -1) && (limit == -1 || offset == -1)) {
            throw std::runtime_error(fmt::format(
                "stream_sort_state_init_py_entry: Either both "
                "limit ({}) and offset ({}) must be -1 or both not -1!",
                limit, offset));
        }
        // No limit and offset. Use usual StreamSortState
        if (limit == -1) {
            auto* state = new StreamSortState(
                op_id, n_keys, std::move(vect_ascending_),
                std::move(na_position_), std::move(schema), parallel);
            return state;
        }
        // Either limit or offset is set. Use the subclass
        // StreamSortLimitOffsetState.
        // XXX It might be better to merge these two for the regular
        // limit/offset case. We can have a separate class for the SmallLimit
        // case, since that's the one with different code paths.
        auto* state = new StreamSortLimitOffsetState(
            op_id, n_keys, std::move(vect_ascending_), std::move(na_position_),
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
        state->ConsumeBatch(table);
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
        if (is_last && state->op_id != -1) {
            QueryProfileCollector::Default().SubmitOperatorStageRowCounts(
                QueryProfileCollector::MakeOperatorStageID(
                    state->op_id, QUERY_PROFILE_SORT_OUTPUT_STAGE_ID),
                state->metrics.output_row_count);
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
    MOD_DEF(m, "stream_sort_cpp", "No docs", nullptr);
    if (m == nullptr) {
        return nullptr;
    }

    bodo_common_init();

    SetAttrStringFromVoidPtr(m, stream_sort_state_init_py_entry);
    SetAttrStringFromVoidPtr(m, stream_sort_build_consume_batch_py_entry);
    SetAttrStringFromVoidPtr(m, stream_sort_product_output_batch_py_entry);
    SetAttrStringFromVoidPtr(m, delete_stream_sort_state);
    return m;
}
