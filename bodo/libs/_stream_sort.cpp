#include "_stream_sort.h"
#include "_array_operations.h"
#include "_bodo_common.h"
#include "_chunked_table_builder.h"
#include "_dict_builder.h"

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

enum class StreamSortPhase { INIT, BUILD, PRODUCE_OUTPUT, INVALID };

struct StreamSortState {
    int64_t n_key_t;
    std::vector<int64_t> vect_ascending;
    std::vector<int64_t> na_position;
    std::vector<int64_t> dead_keys;

    SortedChunkedTableBuilder builder;
    std::vector<TableAndRange> local_chunks;

    size_t output_idx = 0;
    // TODO(aneesh) it would be better to make this a ChunkedTableBuilder
    std::vector<std::shared_ptr<table_info>> output_chunks;

    StreamSortPhase phase = StreamSortPhase::INIT;

    // TODO(aneesh) populate this with an actual chunk!!!
    std::shared_ptr<table_info> dummy_output_chunk_;

    std::shared_ptr<table_info> dummy_output_chunk() {
        throw std::runtime_error("Not implemented");
    }

    StreamSortState(int64_t n_key_t_, std::vector<int64_t>&& vect_ascending_,
                    std::vector<int64_t>&& na_position_)
        : n_key_t(n_key_t_),
          vect_ascending(vect_ascending_),
          na_position(na_position_),
          // Note that builder only stores references to the vectors owned by
          // this object, so we must refer to the instances on this class, not
          // the arguments.
          builder(n_key_t, vect_ascending, na_position, dead_keys) {}
};

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
    state->phase = StreamSortPhase::BUILD;
    return state;
}

bool stream_sort_build_consume_batch_py_entry(StreamSortState* state,
                                              table_info* in_table,
                                              bool parallel, bool is_last) {
    try {
        // TODO(aneesh) fix arrow_buffer_to_bodo - currently the pool isn't
        // stored in the dtor_info, so pinning/unpinning semi-structured data
        // after calling sort_values_table_local will crash. Remove this code
        // after fixing that.
        for (auto& col : in_table->columns) {
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
        std::shared_ptr<table_info> table(in_table);
        state->builder.AppendChunk(table);
        if (is_last) {
            state->local_chunks = state->builder.Finalize();
            state->phase = StreamSortPhase::PRODUCE_OUTPUT;
        }
        return is_last;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return true;
    }
}

std::shared_ptr<table_info> get_parallel_sort_bounds() {
    throw std::runtime_error("Not implemented");
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
            output = state->dummy_output_chunk();
        }
    } else {
        int n_pes, myrank;
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        // most of this needs to be moved to state
        auto bounds = get_parallel_sort_bounds(/*TODO*/);
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
        throw std::runtime_error("Not implemented");
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
