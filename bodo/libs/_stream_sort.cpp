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

    auto sorted_chunk = sort_values_table_local(chunk, n_key_t, vect_ascending,
                                                na_position, dead_keys, false);
    table_heap.push_back({sorted_chunk, n_key_t});
    std::push_heap(table_heap.begin(), table_heap.end(), comp);

    sorted_chunk->unpin();
}

std::vector<TableAndRange> SortedChunkedTableBuilder::Finalize() {
    std::vector<TableAndRange> output;
    if (table_heap.size() == 0) {
        return output;
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

        if (sorted_table_builder->total_size %
                sorted_table_builder->active_chunk_capacity ==
            1) {
            auto [chunk, _] = sorted_table_builder->PopChunk();
            if (chunk->nrows() > 0) {
                output.push_back({chunk, n_key_t});
            }
        }

        auto size = static_cast<int64_t>(min.table->nrows());
        int64_t next = min.offset + 1;

        // Update table and range
        if (next < size) {
            min.UpdateOffset(n_key_t, next);
        }
        min.table->unpin();

        if (next >= size) {
            // shrink heap
            table_heap.pop_back();
        } else {
            // Push the table back into the heap
            std::push_heap(table_heap.begin(), table_heap.end(), comp);
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

struct StreamSortState {};

StreamSortState* stream_sort_state_init_py_entry(int8_t* arr_c_types,
                                                 int8_t* arr_array_types,
                                                 int n_arrs,
                                                 int64_t chunk_size) {
    return nullptr;
}

bool stream_sort_build_consume_batch_py_entry(StreamSortState* state,
                                              table_info* in_table,
                                              bool is_last) {
    return true;
}

table_info* stream_sort_product_output_batch_py_entry(StreamSortState* state,
                                                      bool produce_output,
                                                      bool* out_is_last) {
    *out_is_last = true;
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
