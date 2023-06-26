#include "_table_builder.h"
#include "_array_hash.h"

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

void ArrayBuildBuffer::IncrementSize() {
    switch (this->data_array->arr_type) {
        case bodo_array_type::NULLABLE_INT_BOOL: {
            if (this->data_array->dtype == Bodo_CTypes::_BOOL) {
                size++;
                data_array->length = size;
                CHECK_ARROW_MEM(data_array->buffers[0]->Resize(
                                    arrow::bit_util::BytesForBits(size), false),
                                "Resize Failed!");
                CHECK_ARROW_MEM(data_array->buffers[1]->Resize(
                                    arrow::bit_util::BytesForBits(size), false),
                                "Resize Failed!");
            } else {
                uint64_t size_type = numpy_item_size[this->data_array->dtype];
                size++;
                data_array->length = size;
                CHECK_ARROW_MEM(
                    data_array->buffers[0]->Resize(size * size_type, false),
                    "Resize Failed!");
                CHECK_ARROW_MEM(data_array->buffers[1]->Resize(
                                    arrow::bit_util::BytesForBits(size), false),
                                "Resize Failed!");
            }
        } break;
        case bodo_array_type::STRING: {
            // NOTE: only increments offset and null bitmap sizes but not data
            // buffer.
            size++;
            data_array->length = size;

            CHECK_ARROW_MEM(data_array->buffers[1]->Resize(
                                (size + 1) * sizeof(offset_t), false),
                            "Resize Failed!");
            CHECK_ARROW_MEM(data_array->buffers[2]->Resize(
                                arrow::bit_util::BytesForBits(size), false),
                            "Resize Failed!");
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
                "Resize Failed!");
        } break;
        default:
            throw std::runtime_error(
                "invalid array type in IncrementSize " +
                GetArrType_as_string(this->data_array->arr_type));
    }
}

void ArrayBuildBuffer::ReserveArray(const std::shared_ptr<array_info>& in_arr) {
    if (in_arr->arr_type != data_array->arr_type) {
        throw std::runtime_error(
            "Array types don't match in ReserveArray, buffer is " +
            GetArrType_as_string(data_array->arr_type) + ", but input is " +
            GetArrType_as_string(in_arr->arr_type));
    }
    if (in_arr->dtype != data_array->dtype) {
        throw std::runtime_error(
            "Array dtypes don't match in ReserveArray, buffer is " +
            GetDtype_as_string(data_array->dtype) + ", but input is " +
            GetDtype_as_string(in_arr->dtype));
    }

    if (in_arr->arr_type == bodo_array_type::DICT &&
        data_array->child_arrays[0] != in_arr->child_arrays[0]) {
        throw std::runtime_error("dictionary not unified in ReserveArray");
    }

    this->ReserveSize(in_arr->length);

    if (in_arr->arr_type == bodo_array_type::STRING) {
        // update data buffer
        int64_t capacity_chars = data_array->buffers[0]->capacity();
        int64_t min_capacity_chars =
            data_array->n_sub_elems() + in_arr->n_sub_elems();
        if (min_capacity_chars > capacity_chars) {
            int64_t new_capacity_chars =
                std::max(min_capacity_chars, capacity_chars * 2);
            CHECK_ARROW_MEM(data_array->buffers[0]->Reserve(new_capacity_chars *
                                                            sizeof(int8_t)),
                            "Reserve failed!");
        }
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
                        data_array->buffers[0]->Reserve(
                            arrow::bit_util::BytesForBits(new_capacity)),
                        "Reserve failed!");
                    CHECK_ARROW_MEM(
                        data_array->buffers[1]->Reserve(
                            arrow::bit_util::BytesForBits(new_capacity)),
                        "Reserve failed!");
                } else {
                    uint64_t size_type =
                        numpy_item_size[this->data_array->dtype];
                    CHECK_ARROW_MEM(data_array->buffers[0]->Reserve(
                                        new_capacity * size_type),
                                    "Reserve failed!");
                    CHECK_ARROW_MEM(
                        data_array->buffers[1]->Reserve(
                            arrow::bit_util::BytesForBits(new_capacity)),
                        "Reserve failed!");
                }
                capacity = new_capacity;
            }
            break;
        case bodo_array_type::STRING: {
            // update offset and null bitmap buffers
            if (min_capacity > capacity) {
                int64_t new_capacity = std::max(min_capacity, capacity * 2);
                CHECK_ARROW_MEM(data_array->buffers[1]->Reserve(
                                    (new_capacity + 1) * sizeof(offset_t)),
                                "Reserve failed!");
                CHECK_ARROW_MEM(
                    data_array->buffers[2]->Reserve(
                        arrow::bit_util::BytesForBits(new_capacity)),
                    "Reserve failed!");
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
                    data_array->buffers[0]->Reserve(new_capacity * size_type),
                    "Reserve failed!");
                capacity = new_capacity;
            }
        } break;
        default:
            throw std::runtime_error(
                "invalid array type in ReserveSize " +
                GetArrType_as_string(this->data_array->arr_type));
    }
}

void ArrayBuildBuffer::Reset() {
    size = 0;
    data_array->length = 0;
    switch (data_array->arr_type) {
        case bodo_array_type::NULLABLE_INT_BOOL: {
            CHECK_ARROW_MEM(data_array->buffers[0]->Resize(0, false),
                            "Resize failed!");
            CHECK_ARROW_MEM(data_array->buffers[1]->Resize(0, false),
                            "Resize failed!");
        } break;
        case bodo_array_type::NUMPY: {
            CHECK_ARROW_MEM(data_array->buffers[0]->Resize(0, false),
                            "Resize failed!");

        } break;
        case bodo_array_type::STRING: {
            CHECK_ARROW_MEM(data_array->buffers[0]->Resize(0, false),
                            "Resize failed!");
            CHECK_ARROW_MEM(data_array->buffers[1]->Resize(0, false),
                            "Resize failed!");
            CHECK_ARROW_MEM(data_array->buffers[2]->Resize(0, false),
                            "Resize failed!");
        } break;
        case bodo_array_type::DICT: {
            this->dict_indices->Reset();
            // Reset the dictionary to point to the one
            // in the dict builder:
            this->data_array->child_arrays[0] =
                this->dict_builder->dict_buff->data_array;
            // Reset the flags:
            this->data_array->has_global_dictionary =
                false;  // by default, dictionary builders are not global.
            this->data_array->has_deduped_local_dictionary =
                true;  // the dictionary builder guarantees deduplication by
                       // design.
            this->data_array->has_sorted_dictionary =
                false;  // by default, dictionary builders are not sorted.
        } break;
        default: {
            throw std::runtime_error(
                "invalid array type in Clear " +
                GetArrType_as_string(data_array->arr_type));
        }
    }
}

/* ------------------------------------------------------------------------ */

/* -------------------------- TableBuildBuffer ---------------------------- */

TableBuildBuffer::TableBuildBuffer(
    const std::vector<int8_t>& arr_c_types,
    const std::vector<int8_t>& arr_array_types,
    const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders) {
    // allocate empty initial table with provided data types
    this->data_table = alloc_table(arr_c_types, arr_array_types);

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

void TableBuildBuffer::AppendBatch(const std::shared_ptr<table_info>& in_table,
                                   const std::vector<bool>& append_rows) {
#ifndef APPEND_ARRAY_BATCH
#define APPEND_ARRAY_BATCH(arr_type, arr_type_exp, dtype, dtype_expr) \
    if (arr_type == arr_type_exp) {                                   \
        if (dtype == dtype_expr) {                                    \
            array_buffers[i].AppendBatch<arr_type_exp, dtype_expr>(   \
                in_arr, append_rows, append_rows_sum);                \
        }                                                             \
    }
#endif

    u_int64_t append_rows_sum =
        std::accumulate(append_rows.begin(), append_rows.end(), (u_int64_t)0);

    for (size_t i = 0; i < in_table->ncols(); i++) {
        std::shared_ptr<array_info>& in_arr = in_table->columns[i];
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NULLABLE_INT_BOOL,
                           in_arr->dtype, Bodo_CTypes::INT8);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NULLABLE_INT_BOOL,
                           in_arr->dtype, Bodo_CTypes::INT16);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NULLABLE_INT_BOOL,
                           in_arr->dtype, Bodo_CTypes::INT32);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NULLABLE_INT_BOOL,
                           in_arr->dtype, Bodo_CTypes::INT64);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NULLABLE_INT_BOOL,
                           in_arr->dtype, Bodo_CTypes::UINT8);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NULLABLE_INT_BOOL,
                           in_arr->dtype, Bodo_CTypes::UINT16);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NULLABLE_INT_BOOL,
                           in_arr->dtype, Bodo_CTypes::UINT32);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NULLABLE_INT_BOOL,
                           in_arr->dtype, Bodo_CTypes::UINT64);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NULLABLE_INT_BOOL,
                           in_arr->dtype, Bodo_CTypes::FLOAT32);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NULLABLE_INT_BOOL,
                           in_arr->dtype, Bodo_CTypes::FLOAT64);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NULLABLE_INT_BOOL,
                           in_arr->dtype, Bodo_CTypes::_BOOL);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NULLABLE_INT_BOOL,
                           in_arr->dtype, Bodo_CTypes::DATETIME);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NULLABLE_INT_BOOL,
                           in_arr->dtype, Bodo_CTypes::TIMEDELTA);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NULLABLE_INT_BOOL,
                           in_arr->dtype, Bodo_CTypes::TIME);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NULLABLE_INT_BOOL,
                           in_arr->dtype, Bodo_CTypes::DATE);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NULLABLE_INT_BOOL,
                           in_arr->dtype, Bodo_CTypes::DECIMAL);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NULLABLE_INT_BOOL,
                           in_arr->dtype, Bodo_CTypes::INT128);

        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NUMPY,
                           in_arr->dtype, Bodo_CTypes::INT8);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NUMPY,
                           in_arr->dtype, Bodo_CTypes::INT16);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NUMPY,
                           in_arr->dtype, Bodo_CTypes::INT32);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NUMPY,
                           in_arr->dtype, Bodo_CTypes::INT64);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NUMPY,
                           in_arr->dtype, Bodo_CTypes::UINT8);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NUMPY,
                           in_arr->dtype, Bodo_CTypes::UINT16);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NUMPY,
                           in_arr->dtype, Bodo_CTypes::UINT32);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NUMPY,
                           in_arr->dtype, Bodo_CTypes::UINT64);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NUMPY,
                           in_arr->dtype, Bodo_CTypes::FLOAT32);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NUMPY,
                           in_arr->dtype, Bodo_CTypes::FLOAT64);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NUMPY,
                           in_arr->dtype, Bodo_CTypes::_BOOL);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NUMPY,
                           in_arr->dtype, Bodo_CTypes::DATETIME);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NUMPY,
                           in_arr->dtype, Bodo_CTypes::TIMEDELTA);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NUMPY,
                           in_arr->dtype, Bodo_CTypes::TIME);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NUMPY,
                           in_arr->dtype, Bodo_CTypes::DATE);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NUMPY,
                           in_arr->dtype, Bodo_CTypes::DECIMAL);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NUMPY,
                           in_arr->dtype, Bodo_CTypes::INT128);

        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::STRING,
                           in_arr->dtype, Bodo_CTypes::STRING);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::STRING,
                           in_arr->dtype, Bodo_CTypes::BINARY);

        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::DICT,
                           in_arr->dtype, Bodo_CTypes::STRING);
    }
#undef APPEND_ARRAY_BATCH
}

void TableBuildBuffer::AppendBatch(
    const std::shared_ptr<table_info>& in_table) {
#ifndef APPEND_ARRAY_BATCH
#define APPEND_ARRAY_BATCH(arr_type, arr_type_exp, dtype, dtype_expr)       \
    if (arr_type == arr_type_exp) {                                         \
        if (dtype == dtype_expr) {                                          \
            array_buffers[i].AppendBatch<arr_type_exp, dtype_expr>(in_arr); \
        }                                                                   \
    }
#endif

    for (size_t i = 0; i < in_table->ncols(); i++) {
        const std::shared_ptr<array_info>& in_arr = in_table->columns[i];
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NULLABLE_INT_BOOL,
                           in_arr->dtype, Bodo_CTypes::INT8);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NULLABLE_INT_BOOL,
                           in_arr->dtype, Bodo_CTypes::INT16);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NULLABLE_INT_BOOL,
                           in_arr->dtype, Bodo_CTypes::INT32);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NULLABLE_INT_BOOL,
                           in_arr->dtype, Bodo_CTypes::INT64);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NULLABLE_INT_BOOL,
                           in_arr->dtype, Bodo_CTypes::UINT8);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NULLABLE_INT_BOOL,
                           in_arr->dtype, Bodo_CTypes::UINT16);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NULLABLE_INT_BOOL,
                           in_arr->dtype, Bodo_CTypes::UINT32);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NULLABLE_INT_BOOL,
                           in_arr->dtype, Bodo_CTypes::UINT64);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NULLABLE_INT_BOOL,
                           in_arr->dtype, Bodo_CTypes::FLOAT32);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NULLABLE_INT_BOOL,
                           in_arr->dtype, Bodo_CTypes::FLOAT64);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NULLABLE_INT_BOOL,
                           in_arr->dtype, Bodo_CTypes::_BOOL);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NULLABLE_INT_BOOL,
                           in_arr->dtype, Bodo_CTypes::DATETIME);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NULLABLE_INT_BOOL,
                           in_arr->dtype, Bodo_CTypes::TIMEDELTA);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NULLABLE_INT_BOOL,
                           in_arr->dtype, Bodo_CTypes::TIME);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NULLABLE_INT_BOOL,
                           in_arr->dtype, Bodo_CTypes::DATE);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NULLABLE_INT_BOOL,
                           in_arr->dtype, Bodo_CTypes::DECIMAL);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NULLABLE_INT_BOOL,
                           in_arr->dtype, Bodo_CTypes::INT128);

        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NUMPY,
                           in_arr->dtype, Bodo_CTypes::INT8);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NUMPY,
                           in_arr->dtype, Bodo_CTypes::INT16);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NUMPY,
                           in_arr->dtype, Bodo_CTypes::INT32);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NUMPY,
                           in_arr->dtype, Bodo_CTypes::INT64);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NUMPY,
                           in_arr->dtype, Bodo_CTypes::UINT8);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NUMPY,
                           in_arr->dtype, Bodo_CTypes::UINT16);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NUMPY,
                           in_arr->dtype, Bodo_CTypes::UINT32);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NUMPY,
                           in_arr->dtype, Bodo_CTypes::UINT64);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NUMPY,
                           in_arr->dtype, Bodo_CTypes::FLOAT32);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NUMPY,
                           in_arr->dtype, Bodo_CTypes::FLOAT64);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NUMPY,
                           in_arr->dtype, Bodo_CTypes::_BOOL);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NUMPY,
                           in_arr->dtype, Bodo_CTypes::DATETIME);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NUMPY,
                           in_arr->dtype, Bodo_CTypes::TIMEDELTA);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NUMPY,
                           in_arr->dtype, Bodo_CTypes::TIME);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NUMPY,
                           in_arr->dtype, Bodo_CTypes::DATE);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NUMPY,
                           in_arr->dtype, Bodo_CTypes::DECIMAL);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::NUMPY,
                           in_arr->dtype, Bodo_CTypes::INT128);

        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::STRING,
                           in_arr->dtype, Bodo_CTypes::STRING);
        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::STRING,
                           in_arr->dtype, Bodo_CTypes::BINARY);

        APPEND_ARRAY_BATCH(in_arr->arr_type, bodo_array_type::DICT,
                           in_arr->dtype, Bodo_CTypes::STRING);
    }
#undef APPEND_ARRAY_BATCH
}

void TableBuildBuffer::AppendRowKeys(
    const std::shared_ptr<table_info>& in_table, int64_t row_ind,
    uint64_t n_keys) {
    for (size_t i = 0; i < (size_t)n_keys; i++) {
        const std::shared_ptr<array_info>& in_arr = in_table->columns[i];
        array_buffers[i].AppendRow(in_arr, row_ind);
    }
}

void TableBuildBuffer::IncrementSizeDataColumns(uint64_t n_keys) {
    for (size_t i = n_keys; i < this->data_table->ncols(); i++) {
        array_buffers[i].IncrementSize();
    }
}

void TableBuildBuffer::ReserveTable(
    const std::shared_ptr<table_info>& in_table) {
    for (size_t i = 0; i < in_table->ncols(); i++) {
        std::shared_ptr<array_info>& in_arr = in_table->columns[i];
        array_buffers[i].ReserveArray(in_arr);
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

/* ------------------------------------------------------------------------ */

/* --------------------------- Helper Functions --------------------------- */

std::shared_ptr<table_info> alloc_table(
    const std::vector<int8_t>& arr_c_types,
    const std::vector<int8_t>& arr_array_types) {
    std::vector<std::shared_ptr<array_info>> arrays;

    for (size_t i = 0; i < arr_c_types.size(); i++) {
        bodo_array_type::arr_type_enum arr_type =
            (bodo_array_type::arr_type_enum)arr_array_types[i];
        Bodo_CTypes::CTypeEnum dtype = (Bodo_CTypes::CTypeEnum)arr_c_types[i];

        arrays.push_back(alloc_array(0, 0, 0, arr_type, dtype, 0, 0));
    }
    return std::make_shared<table_info>(arrays);
}

std::shared_ptr<table_info> alloc_table_like(
    const std::shared_ptr<table_info>& table, const bool reuse_dictionaries) {
    std::vector<std::shared_ptr<array_info>> arrays;
    for (size_t i = 0; i < table->ncols(); i++) {
        bodo_array_type::arr_type_enum arr_type = table->columns[i]->arr_type;
        Bodo_CTypes::CTypeEnum dtype = table->columns[i]->dtype;
        arrays.push_back(alloc_array(0, 0, 0, arr_type, dtype, 0, 0));
        // For dict encoded columns, re-use the same dictionary
        // if reuse_dictionaries = true
        if (reuse_dictionaries && (arr_type == bodo_array_type::DICT)) {
            arrays[i]->child_arrays[0] = table->columns[i]->child_arrays[0];
        }
    }
    return std::make_shared<table_info>(arrays);
}

/* ------------------------------------------------------------------------ */
