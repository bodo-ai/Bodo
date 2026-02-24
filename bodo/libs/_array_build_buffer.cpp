#include "_array_build_buffer.h"

/* -------------------------- ArrayBuildBuffer ---------------------------- */

ArrayBuildBuffer::ArrayBuildBuffer(
    std::shared_ptr<array_info> _data_array,
    std::shared_ptr<DictionaryBuilder> _dict_builder)
    : data_array(_data_array),
      size(this->data_array->length),
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
    } else if (_data_array->arr_type == bodo_array_type::ARRAY_ITEM) {
        this->child_array_builders.emplace_back(
            this->data_array->child_arrays[0],
            this->dict_builder->child_dict_builders[0]);
    } else if (_data_array->arr_type == bodo_array_type::MAP) {
        this->child_array_builders.emplace_back(
            this->data_array->child_arrays[0],
            this->dict_builder->child_dict_builders[0]);
    } else if (_data_array->arr_type == bodo_array_type::STRUCT) {
        for (size_t i = 0; i < this->data_array->child_arrays.size(); i++) {
            const std::shared_ptr<array_info>& child_array =
                this->data_array->child_arrays[i];
            this->child_array_builders.emplace_back(
                child_array, this->dict_builder->child_dict_builders[i]);
        }
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

void ArrayBuildBuffer::UnsafeAppendBatch(
    const std::shared_ptr<array_info>& in_arr,
    const std::vector<bool>& append_rows, uint64_t append_rows_sum) {
#ifndef APPEND_ROWS
#define APPEND_ROWS(arr_type_exp, dtype_exp)                              \
    this->UnsafeAppendBatch<arr_type_exp, dtype_exp>(in_arr, append_rows, \
                                                     append_rows_sum)
#endif
    if (in_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        switch (in_arr->dtype) {
            case Bodo_CTypes::INT8:
                APPEND_ROWS(bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::INT8);
                break;
            case Bodo_CTypes::INT16:
                APPEND_ROWS(bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::INT16);
                break;
            case Bodo_CTypes::INT32:
                APPEND_ROWS(bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::INT32);
                break;
            case Bodo_CTypes::INT64:
                APPEND_ROWS(bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::INT64);
                break;
            case Bodo_CTypes::UINT8:
                APPEND_ROWS(bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::UINT8);
                break;
            case Bodo_CTypes::UINT16:
                APPEND_ROWS(bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::UINT16);
                break;
            case Bodo_CTypes::UINT32:
                APPEND_ROWS(bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::UINT32);
                break;
            case Bodo_CTypes::UINT64:
                APPEND_ROWS(bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::UINT64);
                break;
            case Bodo_CTypes::FLOAT32:
                APPEND_ROWS(bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::FLOAT32);
                break;
            case Bodo_CTypes::FLOAT64:
                APPEND_ROWS(bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::FLOAT64);
                break;
            case Bodo_CTypes::_BOOL:
                APPEND_ROWS(bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::_BOOL);
                break;
            case Bodo_CTypes::DATETIME:
                APPEND_ROWS(bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::DATETIME);
                break;
            case Bodo_CTypes::TIMEDELTA:
                APPEND_ROWS(bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::TIMEDELTA);
                break;
            case Bodo_CTypes::TIME:
                APPEND_ROWS(bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::TIME);
                break;
            case Bodo_CTypes::DATE:
                APPEND_ROWS(bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::DATE);
                break;
            case Bodo_CTypes::DECIMAL:
                APPEND_ROWS(bodo_array_type::NULLABLE_INT_BOOL,
                            Bodo_CTypes::DECIMAL);
                break;
            case Bodo_CTypes::INT128:
                APPEND_ROWS(bodo_array_type::NULLABLE_INT_BOOL,
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
                APPEND_ROWS(bodo_array_type::NUMPY, Bodo_CTypes::INT8);
                break;
            case Bodo_CTypes::INT16:
                APPEND_ROWS(bodo_array_type::NUMPY, Bodo_CTypes::INT16);
                break;
            case Bodo_CTypes::INT32:
                APPEND_ROWS(bodo_array_type::NUMPY, Bodo_CTypes::INT32);
                break;
            case Bodo_CTypes::INT64:
                APPEND_ROWS(bodo_array_type::NUMPY, Bodo_CTypes::INT64);
                break;
            case Bodo_CTypes::UINT8:
                APPEND_ROWS(bodo_array_type::NUMPY, Bodo_CTypes::UINT8);
                break;
            case Bodo_CTypes::UINT16:
                APPEND_ROWS(bodo_array_type::NUMPY, Bodo_CTypes::UINT16);
                break;
            case Bodo_CTypes::UINT32:
                APPEND_ROWS(bodo_array_type::NUMPY, Bodo_CTypes::UINT32);
                break;
            case Bodo_CTypes::UINT64:
                APPEND_ROWS(bodo_array_type::NUMPY, Bodo_CTypes::UINT64);
                break;
            case Bodo_CTypes::FLOAT32:
                APPEND_ROWS(bodo_array_type::NUMPY, Bodo_CTypes::FLOAT32);
                break;
            case Bodo_CTypes::FLOAT64:
                APPEND_ROWS(bodo_array_type::NUMPY, Bodo_CTypes::FLOAT64);
                break;
            case Bodo_CTypes::_BOOL:
                APPEND_ROWS(bodo_array_type::NUMPY, Bodo_CTypes::_BOOL);
                break;
            case Bodo_CTypes::DATETIME:
                APPEND_ROWS(bodo_array_type::NUMPY, Bodo_CTypes::DATETIME);
                break;
            case Bodo_CTypes::TIMEDELTA:
                APPEND_ROWS(bodo_array_type::NUMPY, Bodo_CTypes::TIMEDELTA);
                break;
            case Bodo_CTypes::TIME:
                APPEND_ROWS(bodo_array_type::NUMPY, Bodo_CTypes::TIME);
                break;
            case Bodo_CTypes::DATE:
                APPEND_ROWS(bodo_array_type::NUMPY, Bodo_CTypes::DATE);
                break;
            case Bodo_CTypes::DECIMAL:
                APPEND_ROWS(bodo_array_type::NUMPY, Bodo_CTypes::DECIMAL);
                break;
            case Bodo_CTypes::INT128:
                APPEND_ROWS(bodo_array_type::NUMPY, Bodo_CTypes::INT128);
                break;
            default:
                assert(false);
                break;
        }
    } else if (in_arr->arr_type == bodo_array_type::STRING) {
        if (in_arr->dtype == Bodo_CTypes::STRING) {
            APPEND_ROWS(bodo_array_type::STRING, Bodo_CTypes::STRING);
        } else {
            assert(in_arr->dtype == Bodo_CTypes::BINARY);
            APPEND_ROWS(bodo_array_type::STRING, Bodo_CTypes::BINARY);
        }
    } else if (in_arr->arr_type == bodo_array_type::DICT) {
        assert(in_arr->dtype == Bodo_CTypes::STRING);
        APPEND_ROWS(bodo_array_type::DICT, Bodo_CTypes::STRING);
    } else if (in_arr->arr_type == bodo_array_type::ARRAY_ITEM) {
        assert(in_arr->dtype == Bodo_CTypes::LIST);
        APPEND_ROWS(bodo_array_type::ARRAY_ITEM, Bodo_CTypes::LIST);
    } else if (in_arr->arr_type == bodo_array_type::MAP) {
        assert(in_arr->dtype == Bodo_CTypes::MAP);
        APPEND_ROWS(bodo_array_type::MAP, Bodo_CTypes::MAP);
    } else if (in_arr->arr_type == bodo_array_type::STRUCT) {
        assert(in_arr->dtype == Bodo_CTypes::STRUCT);
        APPEND_ROWS(bodo_array_type::STRUCT, Bodo_CTypes::STRUCT);
    } else if (in_arr->arr_type == bodo_array_type::TIMESTAMPTZ) {
        assert(in_arr->dtype == Bodo_CTypes::TIMESTAMPTZ);
        APPEND_ROWS(bodo_array_type::TIMESTAMPTZ, Bodo_CTypes::TIMESTAMPTZ);
    } else {
        throw std::runtime_error(
            "ArrayBuildBuffer::UnsafeAppendBatch: array type " +
            GetArrType_as_string(this->data_array->arr_type) +
            " not supported!");
    }
#undef APPEND_ROWS
}

void ArrayBuildBuffer::UnsafeAppendBatch(
    const std::shared_ptr<array_info>& in_arr) {
#ifndef APPEND_BATCH_ARRAY
#define APPEND_BATCH_ARRAY(arr_type_exp, dtype_exp) \
    this->UnsafeAppendBatch<arr_type_exp, dtype_exp>(in_arr)
#endif
    if (in_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        switch (in_arr->dtype) {
            case Bodo_CTypes::INT8:
                APPEND_BATCH_ARRAY(bodo_array_type::NULLABLE_INT_BOOL,
                                   Bodo_CTypes::INT8);
                break;
            case Bodo_CTypes::INT16:
                APPEND_BATCH_ARRAY(bodo_array_type::NULLABLE_INT_BOOL,
                                   Bodo_CTypes::INT16);
                break;
            case Bodo_CTypes::INT32:
                APPEND_BATCH_ARRAY(bodo_array_type::NULLABLE_INT_BOOL,
                                   Bodo_CTypes::INT32);
                break;
            case Bodo_CTypes::INT64:
                APPEND_BATCH_ARRAY(bodo_array_type::NULLABLE_INT_BOOL,
                                   Bodo_CTypes::INT64);
                break;
            case Bodo_CTypes::UINT8:
                APPEND_BATCH_ARRAY(bodo_array_type::NULLABLE_INT_BOOL,
                                   Bodo_CTypes::UINT8);
                break;
            case Bodo_CTypes::UINT16:
                APPEND_BATCH_ARRAY(bodo_array_type::NULLABLE_INT_BOOL,
                                   Bodo_CTypes::UINT16);
                break;
            case Bodo_CTypes::UINT32:
                APPEND_BATCH_ARRAY(bodo_array_type::NULLABLE_INT_BOOL,
                                   Bodo_CTypes::UINT32);
                break;
            case Bodo_CTypes::UINT64:
                APPEND_BATCH_ARRAY(bodo_array_type::NULLABLE_INT_BOOL,
                                   Bodo_CTypes::UINT64);
                break;
            case Bodo_CTypes::FLOAT32:
                APPEND_BATCH_ARRAY(bodo_array_type::NULLABLE_INT_BOOL,
                                   Bodo_CTypes::FLOAT32);
                break;
            case Bodo_CTypes::FLOAT64:
                APPEND_BATCH_ARRAY(bodo_array_type::NULLABLE_INT_BOOL,
                                   Bodo_CTypes::FLOAT64);
                break;
            case Bodo_CTypes::_BOOL:
                APPEND_BATCH_ARRAY(bodo_array_type::NULLABLE_INT_BOOL,
                                   Bodo_CTypes::_BOOL);
                break;
            case Bodo_CTypes::DATETIME:
                APPEND_BATCH_ARRAY(bodo_array_type::NULLABLE_INT_BOOL,
                                   Bodo_CTypes::DATETIME);
                break;
            case Bodo_CTypes::TIMEDELTA:
                APPEND_BATCH_ARRAY(bodo_array_type::NULLABLE_INT_BOOL,
                                   Bodo_CTypes::TIMEDELTA);
                break;
            case Bodo_CTypes::TIME:
                APPEND_BATCH_ARRAY(bodo_array_type::NULLABLE_INT_BOOL,
                                   Bodo_CTypes::TIME);
                break;
            case Bodo_CTypes::DATE:
                APPEND_BATCH_ARRAY(bodo_array_type::NULLABLE_INT_BOOL,
                                   Bodo_CTypes::DATE);
                break;
            case Bodo_CTypes::DECIMAL:
                APPEND_BATCH_ARRAY(bodo_array_type::NULLABLE_INT_BOOL,
                                   Bodo_CTypes::DECIMAL);
                break;
            case Bodo_CTypes::INT128:
                APPEND_BATCH_ARRAY(bodo_array_type::NULLABLE_INT_BOOL,
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
                APPEND_BATCH_ARRAY(bodo_array_type::NUMPY, Bodo_CTypes::INT8);
                break;
            case Bodo_CTypes::INT16:
                APPEND_BATCH_ARRAY(bodo_array_type::NUMPY, Bodo_CTypes::INT16);
                break;
            case Bodo_CTypes::INT32:
                APPEND_BATCH_ARRAY(bodo_array_type::NUMPY, Bodo_CTypes::INT32);
                break;
            case Bodo_CTypes::INT64:
                APPEND_BATCH_ARRAY(bodo_array_type::NUMPY, Bodo_CTypes::INT64);
                break;
            case Bodo_CTypes::UINT8:
                APPEND_BATCH_ARRAY(bodo_array_type::NUMPY, Bodo_CTypes::UINT8);
                break;
            case Bodo_CTypes::UINT16:
                APPEND_BATCH_ARRAY(bodo_array_type::NUMPY, Bodo_CTypes::UINT16);
                break;
            case Bodo_CTypes::UINT32:
                APPEND_BATCH_ARRAY(bodo_array_type::NUMPY, Bodo_CTypes::UINT32);
                break;
            case Bodo_CTypes::UINT64:
                APPEND_BATCH_ARRAY(bodo_array_type::NUMPY, Bodo_CTypes::UINT64);
                break;
            case Bodo_CTypes::FLOAT32:
                APPEND_BATCH_ARRAY(bodo_array_type::NUMPY,
                                   Bodo_CTypes::FLOAT32);
                break;
            case Bodo_CTypes::FLOAT64:
                APPEND_BATCH_ARRAY(bodo_array_type::NUMPY,
                                   Bodo_CTypes::FLOAT64);
                break;
            case Bodo_CTypes::_BOOL:
                APPEND_BATCH_ARRAY(bodo_array_type::NUMPY, Bodo_CTypes::_BOOL);
                break;
            case Bodo_CTypes::DATETIME:
                APPEND_BATCH_ARRAY(bodo_array_type::NUMPY,
                                   Bodo_CTypes::DATETIME);
                break;
            case Bodo_CTypes::TIMEDELTA:
                APPEND_BATCH_ARRAY(bodo_array_type::NUMPY,
                                   Bodo_CTypes::TIMEDELTA);
                break;
            case Bodo_CTypes::TIME:
                APPEND_BATCH_ARRAY(bodo_array_type::NUMPY, Bodo_CTypes::TIME);
                break;
            case Bodo_CTypes::DATE:
                APPEND_BATCH_ARRAY(bodo_array_type::NUMPY, Bodo_CTypes::DATE);
                break;
            case Bodo_CTypes::DECIMAL:
                APPEND_BATCH_ARRAY(bodo_array_type::NUMPY,
                                   Bodo_CTypes::DECIMAL);
                break;
            case Bodo_CTypes::INT128:
                APPEND_BATCH_ARRAY(bodo_array_type::NUMPY, Bodo_CTypes::INT128);
                break;
            default:
                assert(false);
                break;
        }
    } else if (in_arr->arr_type == bodo_array_type::STRING) {
        if (in_arr->dtype == Bodo_CTypes::STRING) {
            APPEND_BATCH_ARRAY(bodo_array_type::STRING, Bodo_CTypes::STRING);
        } else {
            assert(in_arr->dtype == Bodo_CTypes::BINARY);
            APPEND_BATCH_ARRAY(bodo_array_type::STRING, Bodo_CTypes::BINARY);
        }
    } else if (in_arr->arr_type == bodo_array_type::DICT) {
        assert(in_arr->dtype == Bodo_CTypes::STRING);
        APPEND_BATCH_ARRAY(bodo_array_type::DICT, Bodo_CTypes::STRING);
    } else if (in_arr->arr_type == bodo_array_type::ARRAY_ITEM) {
        assert(in_arr->dtype == Bodo_CTypes::LIST);
        APPEND_BATCH_ARRAY(bodo_array_type::ARRAY_ITEM, Bodo_CTypes::LIST);
    } else if (in_arr->arr_type == bodo_array_type::MAP) {
        assert(in_arr->dtype == Bodo_CTypes::MAP);
        APPEND_BATCH_ARRAY(bodo_array_type::MAP, Bodo_CTypes::MAP);
    } else if (in_arr->arr_type == bodo_array_type::STRUCT) {
        assert(in_arr->dtype == Bodo_CTypes::STRUCT);
        APPEND_BATCH_ARRAY(bodo_array_type::STRUCT, Bodo_CTypes::STRUCT);
    } else if (in_arr->arr_type == bodo_array_type::TIMESTAMPTZ) {
        assert(in_arr->dtype == Bodo_CTypes::TIMESTAMPTZ);
        APPEND_BATCH_ARRAY(bodo_array_type::TIMESTAMPTZ,
                           Bodo_CTypes::TIMESTAMPTZ);
    } else {
        throw std::runtime_error(
            "ArrayBuildBuffer::UnsafeAppendBatch: array type " +
            GetArrType_as_string(this->data_array->arr_type) +
            " not supported!");
    }
#undef APPEND_BATCH_ARRAY
}

void ArrayBuildBuffer::IncrementSize(size_t addln_size) {
    size_t new_size = this->size + addln_size;
    switch (this->data_array->arr_type) {
        case bodo_array_type::NULLABLE_INT_BOOL: {
            if (this->data_array->dtype == Bodo_CTypes::_BOOL) {
                CHECK_ARROW_BASE(
                    data_array->buffers[0]->SetSize(
                        arrow::bit_util::BytesForBits(new_size)),
                    "ArrayBuildBuffer::IncrementSize: SetSize failed!");
                CHECK_ARROW_BASE(
                    data_array->buffers[1]->SetSize(
                        arrow::bit_util::BytesForBits(new_size)),
                    "ArrayBuildBuffer::IncrementSize: SetSize failed!");
            } else {
                uint64_t size_type = numpy_item_size[this->data_array->dtype];
                CHECK_ARROW_BASE(
                    data_array->buffers[0]->SetSize(new_size * size_type),
                    "ArrayBuildBuffer::IncrementSize: SetSize failed!");
                CHECK_ARROW_BASE(
                    data_array->buffers[1]->SetSize(
                        arrow::bit_util::BytesForBits(new_size)),
                    "ArrayBuildBuffer::IncrementSize: SetSize failed!");
            }

        } break;
        case bodo_array_type::TIMESTAMPTZ: {
            uint64_t utc_size_type = numpy_item_size[this->data_array->dtype];
            uint64_t offset_size_type = numpy_item_size[Bodo_CTypes::INT16];
            CHECK_ARROW_BASE(
                data_array->buffers[0]->SetSize(new_size * utc_size_type),
                "ArrayBuildBuffer::IncrementSize: SetSize failed!");
            CHECK_ARROW_BASE(
                data_array->buffers[1]->SetSize(
                    arrow::bit_util::BytesForBits(new_size * offset_size_type)),
                "ArrayBuildBuffer::IncrementSize: SetSize failed!");
            CHECK_ARROW_BASE(
                data_array->buffers[2]->SetSize(
                    arrow::bit_util::BytesForBits(new_size)),
                "ArrayBuildBuffer::IncrementSize: SetSize failed!");

        } break;
        case bodo_array_type::STRING: {
            throw std::runtime_error(
                "ArrayBuildBuffer::IncrementSize: String arrays unsupported");
        } break;
        case bodo_array_type::DICT: {
            this->dict_indices->IncrementSize(addln_size);
        } break;
        case bodo_array_type::CATEGORICAL:
        case bodo_array_type::NUMPY: {
            uint64_t size_type = numpy_item_size[this->data_array->dtype];
            CHECK_ARROW_BASE(
                data_array->buffers[0]->SetSize(new_size * size_type),
                "ArrayBuildBuffer::IncrementSize: SetSize failed!");
        } break;
        default:
            throw std::runtime_error(
                "ArrayBuildBuffer::IncrementSize: Invalid array type " +
                GetArrType_as_string(this->data_array->arr_type));
    }
    this->data_array->length += addln_size;
}

/**
 * @brief Make sure data types are compatible for ArrayBuildBuffer while
 * allowing int64/uint64 interchangeably (necessary since typing and execution
 * are separate in some code paths such as Arrow expression compute of df
 * library leading to mismatches)
 */
bool is_dtype_compatible(Bodo_CTypes::CTypeEnum dtype1,
                         Bodo_CTypes::CTypeEnum dtype2) {
    if (dtype1 == dtype2) {
        return true;
    }

    if ((dtype1 == Bodo_CTypes::UINT64 && dtype2 == Bodo_CTypes::INT64) ||
        (dtype1 == Bodo_CTypes::INT64 && dtype2 == Bodo_CTypes::UINT64)) {
        return true;
    }
    return false;
}

void ArrayBuildBuffer::ReserveArrayTypeCheck(
    const std::shared_ptr<array_info>& in_arr) {
    if (in_arr->arr_type != this->data_array->arr_type) {
        throw std::runtime_error(
            "Array types don't match in ReserveArray, buffer is " +
            GetArrType_as_string(data_array->arr_type) + ", but input is " +
            GetArrType_as_string(in_arr->arr_type));
    }
    if (!is_dtype_compatible(in_arr->dtype, this->data_array->dtype)) {
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
        CHECK_ARROW_BASE(
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
        offset_t* offsets = (offset_t*)in_arr->data2<bodo_array_type::STRING>();
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

template <typename T>
    requires(std::is_same_v<T, std::vector<std::shared_ptr<table_info>>> ||
             std::is_same_v<T, std::deque<std::shared_ptr<table_info>>>)
void ArrayBuildBuffer::ReserveArrayChunks(const T& chunks,
                                          const size_t array_idx,
                                          const bool input_is_unpinned) {
    if (chunks.empty()) {
        return;
    }
    // Verify array and ctype
    const std::shared_ptr<array_info>& in_arr_first_chunk =
        chunks[0]->columns[array_idx];

    this->ReserveArrayTypeCheck(in_arr_first_chunk);

    size_t total_length = 0;
    for (auto& table : chunks) {
        total_length += table->columns[array_idx]->length;
    }

    this->ReserveSize(total_length);

    // In case of strings, calculate required size of characters buffer:
    if (in_arr_first_chunk->arr_type == bodo_array_type::STRING) {
        // update data buffer to be able to store all strings from the chunked
        // table
        size_t new_capacity_chars = 0;
        for (auto& table : chunks) {
            const std::shared_ptr<array_info>& arr = table->columns[array_idx];
            // TODO Remove pin/unpin requirement to get this information.
            // Looking at size_ of the buffers[0] might be sufficient
            // since we maintain that information correctly.
            if (input_is_unpinned) {
                arr->pin();
            }
            new_capacity_chars += arr->n_sub_elems();
            if (input_is_unpinned) {
                arr->unpin();
            }
        }

        this->ReserveSpaceForStringAppend(new_capacity_chars);
    }
}

// Explicitly initialize the required templates for loader to be able to
// find them statically.
template void ArrayBuildBuffer::ReserveArrayChunks(
    const std::vector<std::shared_ptr<table_info>>& chunks,
    const size_t array_idx, const bool input_is_unpinned);

template void ArrayBuildBuffer::ReserveArrayChunks(
    const std::deque<std::shared_ptr<table_info>>& chunks,
    const size_t array_idx, const bool input_is_unpinned);

void ArrayBuildBuffer::ReserveArrayRow(
    const std::shared_ptr<array_info>& in_arr, size_t row_idx) {
    this->ReserveArrayTypeCheck(in_arr);
    this->ReserveSize(1);

    if (in_arr->arr_type == bodo_array_type::STRING) {
        // update data buffer to be able to store the string at in_arr[row_idx]
        offset_t* offsets = (offset_t*)in_arr->data2<bodo_array_type::STRING>();
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
                    CHECK_ARROW_BASE(
                        this->data_array->buffers[0]->Reserve(
                            arrow::bit_util::BytesForBits(new_capacity)),
                        "ArrayBuildBuffer::ReserveSize: Reserve failed!");
                    CHECK_ARROW_BASE(
                        this->data_array->buffers[1]->Reserve(
                            arrow::bit_util::BytesForBits(new_capacity)),
                        "ArrayBuildBuffer::ReserveSize: Reserve failed!");
                } else {
                    uint64_t size_type =
                        numpy_item_size[this->data_array->dtype];
                    CHECK_ARROW_BASE(
                        data_array->buffers[0]->Reserve(new_capacity *
                                                        size_type),
                        "ArrayBuildBuffer::ReserveSize: Reserve failed!");
                    CHECK_ARROW_BASE(
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
                CHECK_ARROW_BASE(
                    data_array->buffers[1]->Reserve((new_capacity + 1) *
                                                    sizeof(offset_t)),
                    "ArrayBuildBuffer::ReserveSize: Reserve failed!");
                CHECK_ARROW_BASE(
                    this->data_array->buffers[2]->Reserve(
                        arrow::bit_util::BytesForBits(new_capacity)),
                    "ArrayBuildBuffer::ReserveSize: Reserve failed!");
                capacity = new_capacity;
            }
        } break;
        case bodo_array_type::DICT: {
            this->dict_indices->ReserveSize(new_data_len);
        } break;
        case bodo_array_type::CATEGORICAL:
        case bodo_array_type::NUMPY: {
            uint64_t size_type = numpy_item_size[this->data_array->dtype];
            if (min_capacity > capacity) {
                int64_t new_capacity = std::max(min_capacity, capacity * 2);
                CHECK_ARROW_BASE(
                    this->data_array->buffers[0]->Reserve(new_capacity *
                                                          size_type),
                    "ArrayBuildBuffer::ReserveSize: Reserve failed!");
                capacity = new_capacity;
            }
        } break;
        case bodo_array_type::ARRAY_ITEM: {
            // update offset and null bitmap buffers
            if (min_capacity > capacity) {
                int64_t new_capacity = std::max(min_capacity, capacity * 2);
                CHECK_ARROW_BASE(
                    data_array->buffers[0]->Reserve((new_capacity + 1) *
                                                    sizeof(offset_t)),
                    "ArrayBuildBuffer::ReserveSize: Reserve failed!");
                CHECK_ARROW_BASE(
                    this->data_array->buffers[1]->Reserve(
                        arrow::bit_util::BytesForBits(new_capacity)),
                    "ArrayBuildBuffer::ReserveSize: Reserve failed!");
                capacity = new_capacity;
            }
        } break;
        case bodo_array_type::MAP: {
            this->child_array_builders[0].ReserveSize(new_data_len);
        } break;
        case bodo_array_type::STRUCT: {
            if (min_capacity > capacity) {
                int64_t new_capacity = std::max(min_capacity, capacity * 2);
                CHECK_ARROW_BASE(
                    this->data_array->buffers[0]->Reserve(
                        arrow::bit_util::BytesForBits(new_capacity)),
                    "ArrayBuildBuffer::ReserveSize: Reserve failed!");
                capacity = new_capacity;
            }
        } break;
        case bodo_array_type::TIMESTAMPTZ: {
            uint64_t ts_size_type = numpy_item_size[Bodo_CTypes::TIMESTAMPTZ];
            uint64_t offset_size_type = numpy_item_size[Bodo_CTypes::INT16];
            if (min_capacity > capacity) {
                int64_t new_capacity = std::max(min_capacity, capacity * 2);
                CHECK_ARROW_BASE(
                    this->data_array->buffers[0]->Reserve(new_capacity *
                                                          ts_size_type),
                    "ArrayBuildBuffer::ReserveSize: Reserve failed!");
                CHECK_ARROW_BASE(
                    this->data_array->buffers[1]->Reserve(new_capacity *
                                                          offset_size_type),
                    "ArrayBuildBuffer::ReserveSize: Reserve failed!");
                CHECK_ARROW_BASE(
                    this->data_array->buffers[2]->Reserve(
                        arrow::bit_util::BytesForBits(new_capacity)),
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
    // Reset the dictionary to point to the one
    // in the dict builder:
    set_array_dict_from_builder(this->data_array, this->dict_builder);
    switch (data_array->arr_type) {
        case bodo_array_type::NULLABLE_INT_BOOL: {
            CHECK_ARROW_BASE(data_array->buffers[0]->SetSize(0),
                             "ArrayBuildBuffer::Reset: SetSize failed!");
            CHECK_ARROW_BASE(data_array->buffers[1]->SetSize(0),
                             "ArrayBuildBuffer::Reset: SetSize failed!");
        } break;
        case bodo_array_type::CATEGORICAL:
        case bodo_array_type::NUMPY: {
            CHECK_ARROW_BASE(data_array->buffers[0]->SetSize(0),
                             "ArrayBuildBuffer::Reset: SetSize failed!");

        } break;
        case bodo_array_type::STRING: {
            CHECK_ARROW_BASE(data_array->buffers[0]->SetSize(0),
                             "ArrayBuildBuffer::Reset: SetSize failed!");
            CHECK_ARROW_BASE(data_array->buffers[1]->SetSize(0),
                             "ArrayBuildBuffer::Reset: SetSize failed!");
            CHECK_ARROW_BASE(data_array->buffers[2]->SetSize(0),
                             "ArrayBuildBuffer::Reset: SetSize failed!");
        } break;
        case bodo_array_type::DICT: {
            this->dict_indices->Reset();
        } break;
        case bodo_array_type::ARRAY_ITEM: {
            this->child_array_builders.front().Reset();
            CHECK_ARROW_BASE(data_array->buffers[0]->SetSize(0),
                             "ArrayBuildBuffer::Reset: SetSize failed!");
            CHECK_ARROW_BASE(data_array->buffers[1]->SetSize(0),
                             "ArrayBuildBuffer::Reset: SetSize failed!");
        } break;
        case bodo_array_type::MAP: {
            this->child_array_builders.front().Reset();
        } break;
        case bodo_array_type::STRUCT: {
            for (ArrayBuildBuffer child_array_builder :
                 this->child_array_builders) {
                child_array_builder.Reset();
            }
            CHECK_ARROW_BASE(data_array->buffers[0]->SetSize(0),
                             "ArrayBuildBuffer::Reset: SetSize failed!");
        } break;
        case bodo_array_type::TIMESTAMPTZ: {
            CHECK_ARROW_BASE(data_array->buffers[0]->SetSize(0),
                             "ArrayBuildBuffer::Reset: SetSize failed!");
            CHECK_ARROW_BASE(data_array->buffers[1]->SetSize(0),
                             "ArrayBuildBuffer::Reset: SetSize failed!");
            CHECK_ARROW_BASE(data_array->buffers[2]->SetSize(0),
                             "ArrayBuildBuffer::Reset: SetSize failed!");
        } break;
        default: {
            throw std::runtime_error(
                "ArrayBuildBuffer::Reset: Invalid array type " +
                GetArrType_as_string(data_array->arr_type));
        }
    }
    this->data_array->length = 0;
}

/* ------------------------------------------------------------------------ */
