#include "_table_builder.h"
#include "_array_hash.h"
#include "_stream_join.h"

/* -------------------------- ArrayBuildBuffer ---------------------------- */

ArrayBuildBuffer::ArrayBuildBuffer(
    std::shared_ptr<array_info> _data_array,
    std::shared_ptr<DictionaryBuilder> _dict_builder)
    : data_array(_data_array),
      size(0),
      capacity(0),
      dict_builder(_dict_builder) {
    if (_data_array->length != 0) {
        throw std::runtime_error(
            "ArrayBuildBuffer: Received a non-empty data array!");
    }
    if (_data_array->arr_type == bodo_array_type::DICT) {
        if (_dict_builder == nullptr) {
            throw std::runtime_error(
                "ArrayBuildBuffer: dict_builder is nullptr for a "
                "dict-encoded string array!");
        }
        if (_dict_builder->dict_buff->data_array.get() !=
            _data_array->child_arrays[0].get()) {
            throw std::runtime_error(
                "ArrayBuildBuffer: specified dict_builder does not "
                "match dictionary of _data_array!");
        }
        this->dict_indices = std::make_shared<ArrayBuildBuffer>(
            this->data_array->child_arrays[1]);
    }
}

size_t ArrayBuildBuffer::EstimatedSize() const {
    std::function<size_t(const array_info&)> getSizeOfArrayInfo =
        [&](const array_info& arr) -> size_t {
        size_t size = 0;
        for (const auto& buff : arr.buffers) {
            size += buff->getMeminfo()->size;
        }
        for (auto& child : arr.child_arrays) {
            size += getSizeOfArrayInfo(*child);
        }
        return size;
    };
    return getSizeOfArrayInfo(*data_array);
}

void ArrayBuildBuffer::IncrementSize() {
    switch (this->data_array->arr_type) {
        case bodo_array_type::NULLABLE_INT_BOOL: {
            if (this->data_array->dtype == Bodo_CTypes::_BOOL) {
                size++;
                data_array->length = size;
                CHECK_ARROW_MEM(
                    data_array->buffers[0]->Resize(
                        arrow::bit_util::BytesForBits(size), false),
                    "ArrayBuildBuffer::IncrementSize: Resize failed!");
                CHECK_ARROW_MEM(
                    data_array->buffers[1]->Resize(
                        arrow::bit_util::BytesForBits(size), false),
                    "ArrayBuildBuffer::IncrementSize: Resize failed!");
            } else {
                uint64_t size_type = numpy_item_size[this->data_array->dtype];
                size++;
                data_array->length = size;
                CHECK_ARROW_MEM(
                    data_array->buffers[0]->Resize(size * size_type, false),
                    "ArrayBuildBuffer::IncrementSize: Resize failed!");
                CHECK_ARROW_MEM(
                    data_array->buffers[1]->Resize(
                        arrow::bit_util::BytesForBits(size), false),
                    "ArrayBuildBuffer::IncrementSize: Resize failed!");
            }
        } break;
        case bodo_array_type::STRING: {
            // NOTE: only increments offset and null bitmap sizes but not data
            // buffer.
            size++;
            data_array->length = size;

            CHECK_ARROW_MEM(data_array->buffers[1]->Resize(
                                (size + 1) * sizeof(offset_t), false),
                            "ArrayBuildBuffer::IncrementSize: Resize failed!");
            CHECK_ARROW_MEM(data_array->buffers[2]->Resize(
                                arrow::bit_util::BytesForBits(size), false),
                            "ArrayBuildBuffer::IncrementSize: Resize failed!");
        } break;
        case bodo_array_type::DICT: {
            this->dict_indices->IncrementSize();
            size++;
            data_array->length = size;
        } break;
        case bodo_array_type::NUMPY: {
            uint64_t size_type = numpy_item_size[this->data_array->dtype];
            size++;
            data_array->length = size;
            CHECK_ARROW_MEM(
                data_array->buffers[0]->Resize(size * size_type, false),
                "ArrayBuildBuffer::IncrementSize: Resize failed!");
        } break;
        default:
            throw std::runtime_error(
                "ArrayBuildBuffer::IncrementSize: Invalid array type " +
                GetArrType_as_string(this->data_array->arr_type));
    }
}

void ArrayBuildBuffer::ReserveArrayTypeCheck(
    const std::shared_ptr<array_info>& in_arr) {
    if (in_arr->arr_type != this->data_array->arr_type) {
        throw std::runtime_error(
            "Array types don't match in ReserveArray, buffer is " +
            GetArrType_as_string(data_array->arr_type) + ", but input is " +
            GetArrType_as_string(in_arr->arr_type));
    }
    if (in_arr->dtype != this->data_array->dtype) {
        throw std::runtime_error(
            "Array dtypes don't match in ReserveArray, buffer is " +
            GetDtype_as_string(data_array->dtype) + ", but input is " +
            GetDtype_as_string(in_arr->dtype));
    }
    if (in_arr->arr_type == bodo_array_type::DICT &&
        !is_matching_dictionary(this->data_array->child_arrays[0],
                                in_arr->child_arrays[0])) {
        throw std::runtime_error("dictionary not unified in ReserveArray");
    }
}

void ArrayBuildBuffer::ReserveSpaceForStringAppend(size_t new_char_count) {
    int64_t capacity_chars = data_array->buffers[0]->capacity();
    int64_t min_capacity_chars = data_array->n_sub_elems() + new_char_count;
    if (min_capacity_chars > capacity_chars) {
        int64_t new_capacity_chars =
            std::max(min_capacity_chars, capacity_chars * 2);
        CHECK_ARROW_MEM(
            data_array->buffers[0]->Reserve(new_capacity_chars),
            "ArrayBuildBuffer::ReserveSpaceForStringAppend: Reserve failed!");
    }
}

void ArrayBuildBuffer::ReserveArray(const std::shared_ptr<array_info>& in_arr,
                                    const std::vector<bool>& reserve_rows,
                                    uint64_t reserve_rows_sum) {
    this->ReserveArrayTypeCheck(in_arr);
    this->ReserveSize(reserve_rows_sum);

    if (in_arr->arr_type == bodo_array_type::STRING) {
        if (in_arr->length != reserve_rows.size()) {
            throw std::runtime_error(
                "Array length doesn't match bitmap size in ReserveArray");
        }
        // update data buffer to be able to store all strings that have their
        // corresponding entry in reserve_rows set to true
        size_t new_capacity_chars = 0;
        offset_t* offsets = (offset_t*)in_arr->data2();
        for (uint64_t i = 0; i < in_arr->length; i++) {
            if (reserve_rows[i]) {
                new_capacity_chars += offsets[i + 1] - offsets[i];
            }
        }
        this->ReserveSpaceForStringAppend(new_capacity_chars);
    }
}

void ArrayBuildBuffer::ReserveArray(const std::shared_ptr<array_info>& in_arr) {
    this->ReserveArrayTypeCheck(in_arr);
    this->ReserveSize(in_arr->length);

    if (in_arr->arr_type == bodo_array_type::STRING) {
        // update data buffer to be able to store all strings from in_arr
        this->ReserveSpaceForStringAppend(
            static_cast<size_t>(in_arr->n_sub_elems()));
    }
}

void ArrayBuildBuffer::ReserveArray(const ChunkedTableBuilder& chunked_tb,
                                    const size_t array_idx) {
    if (chunked_tb.chunks.empty()) {
        return;
    }
    // Verify array and ctype
    const std::shared_ptr<array_info>& in_arr_first_chunk =
        chunked_tb.chunks[0]->columns[array_idx];

    this->ReserveArrayTypeCheck(in_arr_first_chunk);

    size_t total_length = 0;
    for (auto& table : chunked_tb.chunks) {
        total_length += table->columns[array_idx]->length;
    }

    this->ReserveSize(total_length);

    // In case of strings, calculate required size of characters buffer:
    if (in_arr_first_chunk->arr_type == bodo_array_type::STRING) {
        // update data buffer to be able to store all strings from the chunked
        // table
        size_t new_capacity_chars = 0;
        for (auto& table : chunked_tb.chunks) {
            const std::shared_ptr<array_info>& arr = table->columns[array_idx];
            // TODO Remove pin/unpin requirement to get this information.
            // Looking at size_ of the buffers[0] might be sufficient
            // since we maintain that information correctly.
            arr->pin();
            new_capacity_chars += arr->n_sub_elems();
            arr->unpin();
        }

        this->ReserveSpaceForStringAppend(new_capacity_chars);
    }
}

void ArrayBuildBuffer::ReserveArrayRow(
    const std::shared_ptr<array_info>& in_arr, size_t row_idx) {
    this->ReserveArrayTypeCheck(in_arr);
    this->ReserveSize(1);

    if (in_arr->arr_type == bodo_array_type::STRING) {
        // update data buffer to be able to store the string at in_arr[row_idx]
        offset_t* offsets = (offset_t*)in_arr->data2();
        size_t len = offsets[row_idx + 1] - offsets[row_idx];
        this->ReserveSpaceForStringAppend(len);
    }
}

void ArrayBuildBuffer::ReserveSize(uint64_t new_data_len) {
    int64_t min_capacity = size + new_data_len;
    switch (this->data_array->arr_type) {
        case bodo_array_type::NULLABLE_INT_BOOL:
            if (min_capacity > capacity) {
                int64_t new_capacity = std::max(min_capacity, capacity * 2);
                if (this->data_array->dtype == Bodo_CTypes::_BOOL) {
                    CHECK_ARROW_MEM(
                        this->data_array->buffers[0]->Reserve(
                            arrow::bit_util::BytesForBits(new_capacity)),
                        "ArrayBuildBuffer::ReserveSize: Reserve failed!");
                    CHECK_ARROW_MEM(
                        this->data_array->buffers[1]->Reserve(
                            arrow::bit_util::BytesForBits(new_capacity)),
                        "ArrayBuildBuffer::ReserveSize: Reserve failed!");
                } else {
                    uint64_t size_type =
                        numpy_item_size[this->data_array->dtype];
                    CHECK_ARROW_MEM(
                        data_array->buffers[0]->Reserve(new_capacity *
                                                        size_type),
                        "ArrayBuildBuffer::ReserveSize: Reserve failed!");
                    CHECK_ARROW_MEM(
                        this->data_array->buffers[1]->Reserve(
                            arrow::bit_util::BytesForBits(new_capacity)),
                        "ArrayBuildBuffer::ReserveSize: Reserve failed!");
                }
                capacity = new_capacity;
            }
            break;
        case bodo_array_type::STRING: {
            // update offset and null bitmap buffers
            if (min_capacity > capacity) {
                int64_t new_capacity = std::max(min_capacity, capacity * 2);
                CHECK_ARROW_MEM(
                    data_array->buffers[1]->Reserve((new_capacity + 1) *
                                                    sizeof(offset_t)),
                    "ArrayBuildBuffer::ReserveSize: Reserve failed!");
                CHECK_ARROW_MEM(
                    this->data_array->buffers[2]->Reserve(
                        arrow::bit_util::BytesForBits(new_capacity)),
                    "ArrayBuildBuffer::ReserveSize: Reserve failed!");
                capacity = new_capacity;
            }
        } break;
        case bodo_array_type::DICT: {
            this->dict_indices->ReserveSize(new_data_len);
        } break;
        case bodo_array_type::NUMPY: {
            uint64_t size_type = numpy_item_size[this->data_array->dtype];
            if (min_capacity > capacity) {
                int64_t new_capacity = std::max(min_capacity, capacity * 2);
                CHECK_ARROW_MEM(
                    this->data_array->buffers[0]->Reserve(new_capacity *
                                                          size_type),
                    "ArrayBuildBuffer::ReserveSize: Reserve failed!");
                capacity = new_capacity;
            }
        } break;
        default:
            throw std::runtime_error(
                "ArrayBuildBuffer::ReserveSize: Invalid array type " +
                GetArrType_as_string(this->data_array->arr_type));
    }
}

void ArrayBuildBuffer::Reset() {
    size = 0;
    data_array->length = 0;
    switch (data_array->arr_type) {
        case bodo_array_type::NULLABLE_INT_BOOL: {
            CHECK_ARROW_MEM(data_array->buffers[0]->SetSize(0),
                            "ArrayBuildBuffer::Reset: SetSize failed!");
            CHECK_ARROW_MEM(data_array->buffers[1]->SetSize(0),
                            "ArrayBuildBuffer::Reset: SetSize failed!");
        } break;
        case bodo_array_type::NUMPY: {
            CHECK_ARROW_MEM(data_array->buffers[0]->SetSize(0),
                            "ArrayBuildBuffer::Reset: SetSize failed!");

        } break;
        case bodo_array_type::STRING: {
            CHECK_ARROW_MEM(data_array->buffers[0]->SetSize(0),
                            "ArrayBuildBuffer::Reset: SetSize failed!");
            CHECK_ARROW_MEM(data_array->buffers[1]->SetSize(0),
                            "ArrayBuildBuffer::Reset: SetSize failed!");
            CHECK_ARROW_MEM(data_array->buffers[2]->SetSize(0),
                            "ArrayBuildBuffer::Reset: SetSize failed!");
        } break;
        case bodo_array_type::DICT: {
            this->dict_indices->Reset();
            // Reset the dictionary to point to the one
            // in the dict builder:
            this->data_array->child_arrays[0] =
                this->dict_builder->dict_buff->data_array;
        } break;
        default: {
            throw std::runtime_error(
                "ArrayBuildBuffer::Reset: Invalid array type " +
                GetArrType_as_string(data_array->arr_type));
        }
    }
}

/* ------------------------------------------------------------------------ */

/* -------------------------- TableBuildBuffer ---------------------------- */

TableBuildBuffer::TableBuildBuffer(
    const std::vector<int8_t>& arr_c_types,
    const std::vector<int8_t>& arr_array_types,
    const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    // allocate empty initial table with provided data types
    this->data_table =
        alloc_table(arr_c_types, arr_array_types, pool, std::move(mm));

    // initialize array buffer wrappers
    for (size_t i = 0; i < arr_c_types.size(); i++) {
        if (arr_array_types[i] == bodo_array_type::DICT) {
            // Set the dictionary to the one from the dict builder:
            this->data_table->columns[i]->child_arrays[0] =
                dict_builders[i]->dict_buff->data_array;
        }
        array_buffers.emplace_back(this->data_table->columns[i],
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
    auto unified_table =
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
                    break;
            }
        } else if (in_arr->arr_type == bodo_array_type::STRING) {
            if (in_arr->dtype == Bodo_CTypes::STRING) {
                APPEND_BATCH(bodo_array_type::STRING, Bodo_CTypes::STRING);
            } else if (in_arr->dtype == Bodo_CTypes::BINARY) {
                APPEND_BATCH(bodo_array_type::STRING, Bodo_CTypes::BINARY);
            }
        } else if (in_arr->arr_type == bodo_array_type::DICT) {
            if (in_arr->dtype == Bodo_CTypes::STRING) {
                APPEND_BATCH(bodo_array_type::DICT, Bodo_CTypes::STRING);
            }
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
                    break;
            }
        } else if (in_arr->arr_type == bodo_array_type::STRING) {
            if (in_arr->dtype == Bodo_CTypes::STRING) {
                APPEND_BATCH(bodo_array_type::STRING, Bodo_CTypes::STRING);
            } else if (in_arr->dtype == Bodo_CTypes::BINARY) {
                APPEND_BATCH(bodo_array_type::STRING, Bodo_CTypes::BINARY);
            }
        } else if (in_arr->arr_type == bodo_array_type::DICT) {
            if (in_arr->dtype == Bodo_CTypes::STRING) {
                APPEND_BATCH(bodo_array_type::DICT, Bodo_CTypes::STRING);
            }
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
        array_buffers[i].ReserveArray(chunked_tb, i);
    }
}

void TableBuildBuffer::ReserveTableKeys(
    const std::shared_ptr<table_info>& in_table, uint64_t n_keys) {
    for (size_t i = 0; i < n_keys; i++) {
        std::shared_ptr<array_info>& in_arr = in_table->columns[i];
        array_buffers[i].ReserveArray(in_arr);
    }
}

void TableBuildBuffer::ReserveSizeDataColumns(uint64_t new_data_len,
                                              uint64_t n_keys) {
    for (size_t i = n_keys; i < this->data_table->ncols(); i++) {
        array_buffers[i].ReserveSize(new_data_len);
    }
}

void TableBuildBuffer::Reset() {
    for (size_t i = 0; i < array_buffers.size(); i++) {
        array_buffers[i].Reset();
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
    std::vector<int8_t> arr_c_types;
    std::vector<int8_t> arr_array_types;
    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders;
    TableBuildBuffer builder;
    // true if dictionaries of input batches are already unified and no
    // unification is necessary during append. Only the dictionary array needs
    // to be set from the latest batch since the upstream operator may have
    // appended elements to it (which changes data pointers).
    bool input_dics_unified;

    TableBuilderState(std::vector<int8_t> _arr_c_types,
                      std::vector<int8_t> _arr_array_types,
                      bool _input_dicts_unified)
        : arr_c_types(std::move(_arr_c_types)),
          arr_array_types(std::move(_arr_array_types)),
          input_dics_unified(_input_dicts_unified) {
        // Create dictionary builders for all columns
        for (size_t i = 0; i < arr_array_types.size(); i++) {
            if (arr_array_types[i] == bodo_array_type::DICT) {
                std::shared_ptr<array_info> dict = alloc_array(
                    0, 0, 0, bodo_array_type::STRING, Bodo_CTypes::STRING);
                dict_builders.emplace_back(
                    std::make_shared<DictionaryBuilder>(dict, false));
            } else {
                dict_builders.emplace_back(nullptr);
            }
        }
        builder = TableBuildBuffer(arr_c_types, arr_array_types, dict_builders);
    }
};

TableBuilderState* table_builder_state_init_py_entry(int8_t* arr_c_types,
                                                     int8_t* arr_array_types,
                                                     int n_arrs,
                                                     bool input_dics_unified) {
    std::vector<int8_t> ctype_vec(n_arrs);
    std::vector<int8_t> ctype_arr_vec(n_arrs);
    for (int i = 0; i < n_arrs; i++) {
        ctype_vec[i] = arr_c_types[i];
        ctype_arr_vec[i] = arr_array_types[i];
    }
    auto* state =
        new TableBuilderState(ctype_vec, ctype_arr_vec, input_dics_unified);
    return state;
}

void table_builder_append_py_entry(TableBuilderState* state,
                                   table_info* in_table) {
    std::shared_ptr<table_info> tmp_table(in_table);

    if (state->input_dics_unified) {
        // No need for unification if input is already unified by an upstream
        // operator. The dictionary arrays need to be set from the latest batch
        // since the upstream operator may have appended elements to them (which
        // changes data pointers).
        for (size_t i = 0; i < tmp_table->ncols(); i++) {
            const std::shared_ptr<array_info>& arr = tmp_table->columns[i];
            if (arr->arr_type == bodo_array_type::DICT) {
                state->builder.data_table->columns[i]->child_arrays[0] =
                    arr->child_arrays[0];
            }
        }

        state->builder.ReserveTable(tmp_table);
        state->builder.UnsafeAppendBatch(tmp_table);
    } else {
        state->builder.UnifyTablesAndAppend(tmp_table, state->dict_builders);
    }
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

struct ChunkedTableBuilderState {
    std::vector<int8_t> arr_c_types;
    std::vector<int8_t> arr_array_types;
    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders;
    std::unique_ptr<ChunkedTableBuilder> builder;

    ChunkedTableBuilderState(std::vector<int8_t> _arr_c_types,
                             std::vector<int8_t> _arr_array_types,
                             size_t chunk_size)
        : arr_c_types(std::move(_arr_c_types)),
          arr_array_types(std::move(_arr_array_types)) {
        // Create dictionary builders for all columns
        for (size_t i = 0; i < arr_array_types.size(); i++) {
            if (arr_array_types[i] == bodo_array_type::DICT) {
                std::shared_ptr<array_info> dict = alloc_array(
                    0, 0, 0, bodo_array_type::STRING, Bodo_CTypes::STRING);
                dict_builders.emplace_back(
                    std::make_shared<DictionaryBuilder>(dict, false));
            } else {
                dict_builders.emplace_back(nullptr);
            }
        }
        builder = std::make_unique<ChunkedTableBuilder>(ChunkedTableBuilder(
            arr_c_types, arr_array_types, dict_builders, chunk_size,
            DEFAULT_MAX_RESIZE_COUNT_FOR_VARIABLE_SIZE_DTYPES));
    }
};

ChunkedTableBuilderState* chunked_table_builder_state_init_py_entry(
    int8_t* arr_c_types, int8_t* arr_array_types, int n_arrs,
    int64_t chunk_size) {
    std::vector<int8_t> ctype_vec(n_arrs);
    std::vector<int8_t> ctype_arr_vec(n_arrs);
    for (int i = 0; i < n_arrs; i++) {
        ctype_vec[i] = arr_c_types[i];
        ctype_arr_vec[i] = arr_array_types[i];
    }

    return new ChunkedTableBuilderState(ctype_vec, ctype_arr_vec,
                                        (size_t)chunk_size);
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
    MOD_DEF(m, "table_builder_cpp", "No docs", NULL);
    if (m == NULL) {
        return NULL;
    }

    bodo_common_init();

    SetAttrStringFromVoidPtr(m, table_builder_state_init_py_entry);
    SetAttrStringFromVoidPtr(m, table_builder_append_py_entry);
    SetAttrStringFromVoidPtr(m, table_builder_finalize);
    SetAttrStringFromVoidPtr(m, table_builder_get_data);
    SetAttrStringFromVoidPtr(m, table_builder_reset);
    SetAttrStringFromVoidPtr(m, delete_table_builder_state);

    SetAttrStringFromVoidPtr(m, chunked_table_builder_state_init_py_entry);
    SetAttrStringFromVoidPtr(m, chunked_table_builder_append_py_entry);
    SetAttrStringFromVoidPtr(m, chunked_table_builder_pop_chunk);
    SetAttrStringFromVoidPtr(m, delete_chunked_table_builder_state);
    return m;
}
