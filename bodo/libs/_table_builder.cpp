#include "_table_builder.h"
#include <memory>

#include "_table_builder_utils.h"
#include "streaming/_join.h"

/* -------------------------- TableBuildBuffer ---------------------------- */

TableBuildBuffer::TableBuildBuffer(
    const std::shared_ptr<bodo::Schema>& schema,
    const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    // allocate empty initial table with provided data types
    this->data_table = alloc_table(schema, pool, std::move(mm));

    // initialize array buffer wrappers
    for (size_t i = 0; i < this->data_table->ncols(); i++) {
        set_array_dict_from_builder(this->data_table->columns[i],
                                    dict_builders[i]);
        this->array_buffers.emplace_back(this->data_table->columns[i],
                                         dict_builders[i]);
    }
}

size_t TableBuildBuffer::EstimatedSize() const {
    size_t size = 0;
    for (const auto& arr : array_buffers) {
        size += arr.EstimatedSize();
    }
    return size;
}

void TableBuildBuffer::UnifyTablesAndAppend(
    const std::shared_ptr<table_info>& in_table,
    std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders) {
    std::shared_ptr<table_info> unified_table =
        unify_dictionary_arrays_helper(in_table, dict_builders, 0, false);
    ReserveTable(unified_table);
    UnsafeAppendBatch(unified_table);
}

void TableBuildBuffer::UnsafeAppendBatch(
    const std::shared_ptr<table_info>& in_table,
    const std::vector<bool>& append_rows, uint64_t append_rows_sum) {
#ifndef APPEND_BATCH
#define APPEND_BATCH(arr_type_exp, dtype_exp)                    \
    array_buffers[i].UnsafeAppendBatch<arr_type_exp, dtype_exp>( \
        in_arr, append_rows, append_rows_sum)
#endif
    for (size_t i = 0; i < in_table->ncols(); i++) {
        std::shared_ptr<array_info>& in_arr = in_table->columns[i];
        if (in_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
            switch (in_arr->dtype) {
                case Bodo_CTypes::INT8:
                    APPEND_BATCH(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::INT8);
                    break;
                case Bodo_CTypes::INT16:
                    APPEND_BATCH(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::INT16);
                    break;
                case Bodo_CTypes::INT32:
                    APPEND_BATCH(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::INT32);
                    break;
                case Bodo_CTypes::INT64:
                    APPEND_BATCH(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::INT64);
                    break;
                case Bodo_CTypes::UINT8:
                    APPEND_BATCH(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::UINT8);
                    break;
                case Bodo_CTypes::UINT16:
                    APPEND_BATCH(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::UINT16);
                    break;
                case Bodo_CTypes::UINT32:
                    APPEND_BATCH(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::UINT32);
                    break;
                case Bodo_CTypes::UINT64:
                    APPEND_BATCH(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::UINT64);
                    break;
                case Bodo_CTypes::FLOAT32:
                    APPEND_BATCH(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::FLOAT32);
                    break;
                case Bodo_CTypes::FLOAT64:
                    APPEND_BATCH(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::FLOAT64);
                    break;
                case Bodo_CTypes::_BOOL:
                    APPEND_BATCH(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::_BOOL);
                    break;
                case Bodo_CTypes::DATETIME:
                    APPEND_BATCH(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::DATETIME);
                    break;
                case Bodo_CTypes::TIMEDELTA:
                    APPEND_BATCH(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::TIMEDELTA);
                    break;
                case Bodo_CTypes::TIME:
                    APPEND_BATCH(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::TIME);
                    break;
                case Bodo_CTypes::DATE:
                    APPEND_BATCH(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::DATE);
                    break;
                case Bodo_CTypes::DECIMAL:
                    APPEND_BATCH(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::DECIMAL);
                    break;
                case Bodo_CTypes::INT128:
                    APPEND_BATCH(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::INT128);
                    break;
                default:
                    assert(false);
                    break;
            }
        } else if (in_arr->arr_type == bodo_array_type::NUMPY) {
            switch (in_arr->dtype) {
                case Bodo_CTypes::INT8:
                    APPEND_BATCH(bodo_array_type::NUMPY, Bodo_CTypes::INT8);
                    break;
                case Bodo_CTypes::INT16:
                    APPEND_BATCH(bodo_array_type::NUMPY, Bodo_CTypes::INT16);
                    break;
                case Bodo_CTypes::INT32:
                    APPEND_BATCH(bodo_array_type::NUMPY, Bodo_CTypes::INT32);
                    break;
                case Bodo_CTypes::INT64:
                    APPEND_BATCH(bodo_array_type::NUMPY, Bodo_CTypes::INT64);
                    break;
                case Bodo_CTypes::UINT8:
                    APPEND_BATCH(bodo_array_type::NUMPY, Bodo_CTypes::UINT8);
                    break;
                case Bodo_CTypes::UINT16:
                    APPEND_BATCH(bodo_array_type::NUMPY, Bodo_CTypes::UINT16);
                    break;
                case Bodo_CTypes::UINT32:
                    APPEND_BATCH(bodo_array_type::NUMPY, Bodo_CTypes::UINT32);
                    break;
                case Bodo_CTypes::UINT64:
                    APPEND_BATCH(bodo_array_type::NUMPY, Bodo_CTypes::UINT64);
                    break;
                case Bodo_CTypes::FLOAT32:
                    APPEND_BATCH(bodo_array_type::NUMPY, Bodo_CTypes::FLOAT32);
                    break;
                case Bodo_CTypes::FLOAT64:
                    APPEND_BATCH(bodo_array_type::NUMPY, Bodo_CTypes::FLOAT64);
                    break;
                case Bodo_CTypes::_BOOL:
                    APPEND_BATCH(bodo_array_type::NUMPY, Bodo_CTypes::_BOOL);
                    break;
                case Bodo_CTypes::DATETIME:
                    APPEND_BATCH(bodo_array_type::NUMPY, Bodo_CTypes::DATETIME);
                    break;
                case Bodo_CTypes::TIMEDELTA:
                    APPEND_BATCH(bodo_array_type::NUMPY,
                                 Bodo_CTypes::TIMEDELTA);
                    break;
                case Bodo_CTypes::TIME:
                    APPEND_BATCH(bodo_array_type::NUMPY, Bodo_CTypes::TIME);
                    break;
                case Bodo_CTypes::DATE:
                    APPEND_BATCH(bodo_array_type::NUMPY, Bodo_CTypes::DATE);
                    break;
                case Bodo_CTypes::DECIMAL:
                    APPEND_BATCH(bodo_array_type::NUMPY, Bodo_CTypes::DECIMAL);
                    break;
                case Bodo_CTypes::INT128:
                    APPEND_BATCH(bodo_array_type::NUMPY, Bodo_CTypes::INT128);
                    break;
                default:
                    assert(false);
                    break;
            }
        } else if (in_arr->arr_type == bodo_array_type::STRING) {
            if (in_arr->dtype == Bodo_CTypes::STRING) {
                APPEND_BATCH(bodo_array_type::STRING, Bodo_CTypes::STRING);
            } else {
                assert(in_arr->dtype == Bodo_CTypes::BINARY);
                APPEND_BATCH(bodo_array_type::STRING, Bodo_CTypes::BINARY);
            }
        } else if (in_arr->arr_type == bodo_array_type::DICT) {
            assert(in_arr->dtype == Bodo_CTypes::STRING);
            APPEND_BATCH(bodo_array_type::DICT, Bodo_CTypes::STRING);
        } else if (in_arr->arr_type == bodo_array_type::ARRAY_ITEM) {
            assert(in_arr->dtype == Bodo_CTypes::LIST);
            APPEND_BATCH(bodo_array_type::ARRAY_ITEM, Bodo_CTypes::LIST);
        } else if (in_arr->arr_type == bodo_array_type::MAP) {
            assert(in_arr->dtype == Bodo_CTypes::MAP);
            APPEND_BATCH(bodo_array_type::MAP, Bodo_CTypes::MAP);
        } else if (in_arr->arr_type == bodo_array_type::STRUCT) {
            assert(in_arr->dtype == Bodo_CTypes::STRUCT);
            APPEND_BATCH(bodo_array_type::STRUCT, Bodo_CTypes::STRUCT);
        } else if (in_arr->arr_type == bodo_array_type::TIMESTAMPTZ) {
            assert(in_arr->dtype == Bodo_CTypes::TIMESTAMPTZ);
            APPEND_BATCH(bodo_array_type::TIMESTAMPTZ,
                         Bodo_CTypes::TIMESTAMPTZ);
        } else {
            throw std::runtime_error(
                "TableBuildBuffer::UnsafeAppendBatch: Invalid array type " +
                GetArrType_as_string(in_arr->arr_type));
        }
    }
#undef APPEND_BATCH
}

void TableBuildBuffer::UnsafeAppendBatch(
    const std::shared_ptr<table_info>& in_table,
    const std::vector<bool>& append_rows) {
    uint64_t append_rows_sum =
        std::accumulate(append_rows.begin(), append_rows.end(), (uint64_t)0);
    this->UnsafeAppendBatch(in_table, append_rows, append_rows_sum);
}

void TableBuildBuffer::UnsafeAppendBatch(
    const std::shared_ptr<table_info>& in_table) {
#ifndef APPEND_BATCH
#define APPEND_BATCH(arr_type_exp, dtype_exp) \
    array_buffers[i].UnsafeAppendBatch<arr_type_exp, dtype_exp>(in_arr)
#endif
    for (size_t i = 0; i < in_table->ncols(); i++) {
        const std::shared_ptr<array_info>& in_arr = in_table->columns[i];
        if (in_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
            switch (in_arr->dtype) {
                case Bodo_CTypes::INT8:
                    APPEND_BATCH(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::INT8);
                    break;
                case Bodo_CTypes::INT16:
                    APPEND_BATCH(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::INT16);
                    break;
                case Bodo_CTypes::INT32:
                    APPEND_BATCH(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::INT32);
                    break;
                case Bodo_CTypes::INT64:
                    APPEND_BATCH(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::INT64);
                    break;
                case Bodo_CTypes::UINT8:
                    APPEND_BATCH(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::UINT8);
                    break;
                case Bodo_CTypes::UINT16:
                    APPEND_BATCH(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::UINT16);
                    break;
                case Bodo_CTypes::UINT32:
                    APPEND_BATCH(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::UINT32);
                    break;
                case Bodo_CTypes::UINT64:
                    APPEND_BATCH(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::UINT64);
                    break;
                case Bodo_CTypes::FLOAT32:
                    APPEND_BATCH(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::FLOAT32);
                    break;
                case Bodo_CTypes::FLOAT64:
                    APPEND_BATCH(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::FLOAT64);
                    break;
                case Bodo_CTypes::_BOOL:
                    APPEND_BATCH(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::_BOOL);
                    break;
                case Bodo_CTypes::DATETIME:
                    APPEND_BATCH(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::DATETIME);
                    break;
                case Bodo_CTypes::TIMEDELTA:
                    APPEND_BATCH(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::TIMEDELTA);
                    break;
                case Bodo_CTypes::TIME:
                    APPEND_BATCH(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::TIME);
                    break;
                case Bodo_CTypes::DATE:
                    APPEND_BATCH(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::DATE);
                    break;
                case Bodo_CTypes::DECIMAL:
                    APPEND_BATCH(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::DECIMAL);
                    break;
                case Bodo_CTypes::INT128:
                    APPEND_BATCH(bodo_array_type::NULLABLE_INT_BOOL,
                                 Bodo_CTypes::INT128);
                    break;
                default:
                    assert(false);
                    break;
            }
        } else if (in_arr->arr_type == bodo_array_type::NUMPY ||
                   in_arr->arr_type == bodo_array_type::CATEGORICAL) {
            switch (in_arr->dtype) {
                case Bodo_CTypes::INT8:
                    APPEND_BATCH(bodo_array_type::NUMPY, Bodo_CTypes::INT8);
                    break;
                case Bodo_CTypes::INT16:
                    APPEND_BATCH(bodo_array_type::NUMPY, Bodo_CTypes::INT16);
                    break;
                case Bodo_CTypes::INT32:
                    APPEND_BATCH(bodo_array_type::NUMPY, Bodo_CTypes::INT32);
                    break;
                case Bodo_CTypes::INT64:
                    APPEND_BATCH(bodo_array_type::NUMPY, Bodo_CTypes::INT64);
                    break;
                case Bodo_CTypes::UINT8:
                    APPEND_BATCH(bodo_array_type::NUMPY, Bodo_CTypes::UINT8);
                    break;
                case Bodo_CTypes::UINT16:
                    APPEND_BATCH(bodo_array_type::NUMPY, Bodo_CTypes::UINT16);
                    break;
                case Bodo_CTypes::UINT32:
                    APPEND_BATCH(bodo_array_type::NUMPY, Bodo_CTypes::UINT32);
                    break;
                case Bodo_CTypes::UINT64:
                    APPEND_BATCH(bodo_array_type::NUMPY, Bodo_CTypes::UINT64);
                    break;
                case Bodo_CTypes::FLOAT32:
                    APPEND_BATCH(bodo_array_type::NUMPY, Bodo_CTypes::FLOAT32);
                    break;
                case Bodo_CTypes::FLOAT64:
                    APPEND_BATCH(bodo_array_type::NUMPY, Bodo_CTypes::FLOAT64);
                    break;
                case Bodo_CTypes::_BOOL:
                    APPEND_BATCH(bodo_array_type::NUMPY, Bodo_CTypes::_BOOL);
                    break;
                case Bodo_CTypes::DATETIME:
                    APPEND_BATCH(bodo_array_type::NUMPY, Bodo_CTypes::DATETIME);
                    break;
                case Bodo_CTypes::TIMEDELTA:
                    APPEND_BATCH(bodo_array_type::NUMPY,
                                 Bodo_CTypes::TIMEDELTA);
                    break;
                case Bodo_CTypes::TIME:
                    APPEND_BATCH(bodo_array_type::NUMPY, Bodo_CTypes::TIME);
                    break;
                case Bodo_CTypes::DATE:
                    APPEND_BATCH(bodo_array_type::NUMPY, Bodo_CTypes::DATE);
                    break;
                case Bodo_CTypes::DECIMAL:
                    APPEND_BATCH(bodo_array_type::NUMPY, Bodo_CTypes::DECIMAL);
                    break;
                case Bodo_CTypes::INT128:
                    APPEND_BATCH(bodo_array_type::NUMPY, Bodo_CTypes::INT128);
                    break;
                default:
                    assert(false);
                    break;
            }
        } else if (in_arr->arr_type == bodo_array_type::STRING) {
            if (in_arr->dtype == Bodo_CTypes::STRING) {
                APPEND_BATCH(bodo_array_type::STRING, Bodo_CTypes::STRING);
            } else {
                assert(in_arr->dtype == Bodo_CTypes::BINARY);
                APPEND_BATCH(bodo_array_type::STRING, Bodo_CTypes::BINARY);
            }
        } else if (in_arr->arr_type == bodo_array_type::DICT) {
            assert(in_arr->dtype == Bodo_CTypes::STRING);
            APPEND_BATCH(bodo_array_type::DICT, Bodo_CTypes::STRING);
        } else if (in_arr->arr_type == bodo_array_type::ARRAY_ITEM) {
            assert(in_arr->dtype == Bodo_CTypes::LIST);
            APPEND_BATCH(bodo_array_type::ARRAY_ITEM, Bodo_CTypes::LIST);
        } else if (in_arr->arr_type == bodo_array_type::MAP) {
            assert(in_arr->dtype == Bodo_CTypes::MAP);
            APPEND_BATCH(bodo_array_type::MAP, Bodo_CTypes::MAP);
        } else if (in_arr->arr_type == bodo_array_type::STRUCT) {
            assert(in_arr->dtype == Bodo_CTypes::STRUCT);
            APPEND_BATCH(bodo_array_type::STRUCT, Bodo_CTypes::STRUCT);
        } else if (in_arr->arr_type == bodo_array_type::TIMESTAMPTZ) {
            assert(in_arr->dtype == Bodo_CTypes::TIMESTAMPTZ);
            APPEND_BATCH(bodo_array_type::TIMESTAMPTZ,
                         Bodo_CTypes::TIMESTAMPTZ);
        } else {
            throw std::runtime_error(
                "TableBuildBuffer::UnsafeAppendBatch: Invalid array type " +
                GetArrType_as_string(in_arr->arr_type));
        }
    }
#undef APPEND_BATCH
}

void TableBuildBuffer::AppendRowKeys(
    const std::shared_ptr<table_info>& in_table, int64_t row_ind,
    uint64_t n_keys) {
    for (size_t i = 0; i < (size_t)n_keys; i++) {
        const std::shared_ptr<array_info>& in_arr = in_table->columns[i];
        array_buffers[i].UnsafeAppendRow(in_arr, row_ind);
    }
}

void TableBuildBuffer::IncrementSizeDataColumns(uint64_t n_keys) {
    for (size_t i = n_keys; i < this->data_table->ncols(); i++) {
        array_buffers[i].IncrementSize();
    }
}

void TableBuildBuffer::IncrementSize(size_t new_size) {
    for (size_t i = 0; i < this->data_table->ncols(); i++) {
        array_buffers[i].IncrementSize(new_size);
    }
}

void TableBuildBuffer::ReserveTable(const std::shared_ptr<table_info>& in_table,
                                    const std::vector<bool>& reserve_rows,
                                    uint64_t reserve_rows_sum) {
    assert(in_table->nrows() == reserve_rows.size());
    for (size_t i = 0; i < in_table->ncols(); i++) {
        std::shared_ptr<array_info>& in_arr = in_table->columns[i];
        array_buffers[i].ReserveArray(in_arr, reserve_rows, reserve_rows_sum);
    }
}

void TableBuildBuffer::ReserveTable(const std::shared_ptr<table_info>& in_table,
                                    const std::vector<bool>& reserve_rows) {
    uint64_t reserve_rows_sum =
        std::accumulate(reserve_rows.begin(), reserve_rows.end(), (uint64_t)0);
    this->ReserveTable(in_table, reserve_rows, reserve_rows_sum);
}

void TableBuildBuffer::ReserveTable(
    const std::shared_ptr<table_info>& in_table) {
    for (size_t i = 0; i < in_table->ncols(); i++) {
        std::shared_ptr<array_info>& in_arr = in_table->columns[i];
        array_buffers[i].ReserveArray(in_arr);
    }
}

void TableBuildBuffer::ReserveTable(const ChunkedTableBuilder& chunked_tb) {
    for (size_t i = 0; i < this->data_table->ncols(); i++) {
        array_buffers[i].ReserveArrayChunks(chunked_tb.chunks, i,
                                            /*input_is_unpinned*/ true);
    }
}

void TableBuildBuffer::ReserveTable(
    const std::vector<std::shared_ptr<table_info>>& chunks,
    const bool input_is_unpinned) {
    for (size_t i = 0; i < this->data_table->ncols(); i++) {
        array_buffers[i].ReserveArrayChunks(chunks, i, input_is_unpinned);
    }
}

void TableBuildBuffer::ReserveTableSize(const size_t new_data_len) {
    for (size_t i = 0; i < this->data_table->ncols(); ++i) {
        array_buffers[i].ReserveSize(new_data_len);
    }
}

void TableBuildBuffer::ReserveTableRow(
    const std::shared_ptr<table_info>& in_table, size_t row_idx) {
    assert(in_table->columns.size() >= this->array_buffers.size());
    for (size_t i = 0; i < this->array_buffers.size(); i++) {
        std::shared_ptr<array_info>& in_arr = in_table->columns[i];
        array_buffers[i].ReserveArrayRow(in_arr, row_idx);
    }
}

void TableBuildBuffer::Reset() {
    for (auto& array_buffer : array_buffers) {
        array_buffer.Reset();
    }
}

void TableBuildBuffer::pin() {
    if (!this->pinned_) {
        // This will automatically pin the underlying arrays.
        // XXX We could expand it to be more explicit in
        // the future, i.e. call pin on the individual
        // ArrayBuildBuffers.
        this->data_table->pin();
        this->pinned_ = true;
    }
}

void TableBuildBuffer::unpin() {
    if (this->pinned_) {
        // This will automatically unpin the underlying arrays.
        // XXX We could expand it to be more explicit in
        // the future, i.e. call unpin on the individual
        // ArrayBuildBuffers.
        this->data_table->unpin();
        this->pinned_ = false;
    }
}

/* ------------------------------------------------------------------------ */

struct TableBuilderState {
    const std::shared_ptr<bodo::Schema> table_schema;
    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders;
    TableBuildBuffer builder;
    // true if dictionaries of input batches are already unified and no
    // unification is necessary during append. Only the dictionary array needs
    // to be set from the latest batch since the upstream operator may have
    // appended elements to it (which changes data pointers).
    bool input_dics_unified;

    TableBuilderState(const std::shared_ptr<bodo::Schema> table_schema_,
                      bool _input_dicts_unified)
        : table_schema(std::move(table_schema_)),
          input_dics_unified(_input_dicts_unified) {
        // Create dictionary builders for all columns
        for (const std::unique_ptr<bodo::DataType>& t :
             table_schema->column_types) {
            dict_builders.emplace_back(
                create_dict_builder_for_array(t->copy(), false));
        }
        builder = TableBuildBuffer(table_schema, dict_builders);
    }
};

TableBuilderState* table_builder_state_init_py_entry(int8_t* arr_c_types,
                                                     int8_t* arr_array_types,
                                                     int n_arrs,
                                                     bool input_dics_unified) {
    std::shared_ptr<bodo::Schema> schema = bodo::Schema::Deserialize(
        std::vector<int8_t>(arr_array_types, arr_array_types + n_arrs),
        std::vector<int8_t>(arr_c_types, arr_c_types + n_arrs));

    return new TableBuilderState(schema, input_dics_unified);
}

void table_builder_append_py_entry(TableBuilderState* state,
                                   table_info* in_table) {
    try {
        std::shared_ptr<table_info> tmp_table(in_table);

        if (state->input_dics_unified) {
            // No need for unification if input is already unified by an
            // upstream operator. The dictionary arrays need to be set from the
            // latest batch since the upstream operator may have appended
            // elements to them (which changes data pointers).
            for (size_t i = 0; i < tmp_table->ncols(); i++) {
                set_array_dict_from_array(state->builder.data_table->columns[i],
                                          tmp_table->columns[i]);
            }

            state->builder.ReserveTable(tmp_table);
            state->builder.UnsafeAppendBatch(tmp_table);
        } else {
            state->builder.UnifyTablesAndAppend(tmp_table,
                                                state->dict_builders);
        }
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

int64_t table_builder_nbytes_py_entry(TableBuilderState* state) {
    int64_t nbytes = 0;
    try {
        nbytes = table_local_memory_size(state->builder.data_table, true);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
    return nbytes;
}

table_info* table_builder_finalize(TableBuilderState* state) {
    auto* rettable = new table_info(*state->builder.data_table);
    delete state;
    return rettable;
}

/**
 * @brief Get the internal data table of table builder without affecting state.
 *
 * @param state table builder state
 * @return table_info* builder's data table
 */
table_info* table_builder_get_data(TableBuilderState* state) {
    return new table_info(*state->builder.data_table);
}

/**
 * @brief Reset table builder's buffer (sets array buffer sizes to zero but
 * keeps capacity the same)
 *
 * @param state table builder state
 */
void table_builder_reset(TableBuilderState* state) { state->builder.Reset(); }

void delete_table_builder_state(TableBuilderState* state) { delete state; }

ChunkedTableBuilderState* chunked_table_builder_state_init_py_entry(
    int8_t* arr_c_types, int8_t* arr_array_types, int n_arrs,
    int64_t chunk_size) {
    std::shared_ptr<bodo::Schema> schema = bodo::Schema::Deserialize(
        std::vector<int8_t>(arr_array_types, arr_array_types + n_arrs),
        std::vector<int8_t>(arr_c_types, arr_c_types + n_arrs));

    return new ChunkedTableBuilderState(schema, (size_t)chunk_size);
}

void chunked_table_builder_append_py_entry(ChunkedTableBuilderState* state,
                                           table_info* in_table) {
    const std::shared_ptr<table_info> tmp_table(in_table);
    std::shared_ptr<table_info> unified_table = unify_dictionary_arrays_helper(
        tmp_table, state->dict_builders, 0, false);
    state->builder->AppendBatch(unified_table);
}

table_info* chunked_table_builder_pop_chunk(ChunkedTableBuilderState* state,
                                            bool produce_output,
                                            bool force_return,
                                            bool* is_last_output_chunk) {
    std::shared_ptr<table_info> ret_table;
    if (!produce_output) {
        ret_table = state->builder->dummy_output_chunk;
    } else {
        ret_table = get<0>(state->builder->PopChunk(force_return));
    }
    *is_last_output_chunk = (state->builder->total_remaining == 0);
    return new table_info(*ret_table);
}

void delete_chunked_table_builder_state(ChunkedTableBuilderState* state) {
    delete state;
}

PyMODINIT_FUNC PyInit_table_builder_cpp(void) {
    PyObject* m;
    MOD_DEF(m, "table_builder_cpp", "No docs", nullptr);
    if (m == nullptr) {
        return nullptr;
    }

    bodo_common_init();

    SetAttrStringFromVoidPtr(m, table_builder_state_init_py_entry);
    SetAttrStringFromVoidPtr(m, table_builder_append_py_entry);
    SetAttrStringFromVoidPtr(m, table_builder_finalize);
    SetAttrStringFromVoidPtr(m, table_builder_get_data);
    SetAttrStringFromVoidPtr(m, table_builder_reset);
    SetAttrStringFromVoidPtr(m, table_builder_nbytes_py_entry);
    SetAttrStringFromVoidPtr(m, delete_table_builder_state);

    SetAttrStringFromVoidPtr(m, chunked_table_builder_state_init_py_entry);
    SetAttrStringFromVoidPtr(m, chunked_table_builder_append_py_entry);
    SetAttrStringFromVoidPtr(m, chunked_table_builder_pop_chunk);
    SetAttrStringFromVoidPtr(m, delete_chunked_table_builder_state);
    return m;
}
