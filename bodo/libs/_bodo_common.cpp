#include "_bodo_common.h"
#include "_bodo_to_arrow.h"
#include "_distributed.h"

std::vector<size_t> numpy_item_size(Bodo_CTypes::_numtypes);

void bodo_common_init() {
    static bool initialized = false;
    if (initialized) {
        return;
    }
    initialized = true;
    // Note: This is only true for Numpy Boolean arrays
    // and should be removed when we purge Numpy Boolean arrays
    // from C++.
    numpy_item_size[Bodo_CTypes::_BOOL] = sizeof(bool);
    numpy_item_size[Bodo_CTypes::INT8] = sizeof(int8_t);
    numpy_item_size[Bodo_CTypes::UINT8] = sizeof(uint8_t);
    numpy_item_size[Bodo_CTypes::INT16] = sizeof(int16_t);
    numpy_item_size[Bodo_CTypes::UINT16] = sizeof(uint16_t);
    numpy_item_size[Bodo_CTypes::INT32] = sizeof(int32_t);
    numpy_item_size[Bodo_CTypes::UINT32] = sizeof(uint32_t);
    numpy_item_size[Bodo_CTypes::INT64] = sizeof(int64_t);
    numpy_item_size[Bodo_CTypes::UINT64] = sizeof(uint64_t);
    numpy_item_size[Bodo_CTypes::FLOAT32] = sizeof(float);
    numpy_item_size[Bodo_CTypes::FLOAT64] = sizeof(double);
    numpy_item_size[Bodo_CTypes::DECIMAL] = BYTES_PER_DECIMAL;
    numpy_item_size[Bodo_CTypes::DATETIME] = sizeof(int64_t);
    numpy_item_size[Bodo_CTypes::DATE] = sizeof(int32_t);
    // TODO: [BE-4106] TIME size should depend on precision.
    numpy_item_size[Bodo_CTypes::TIME] = sizeof(int64_t);
    numpy_item_size[Bodo_CTypes::TIMEDELTA] = sizeof(int64_t);
    numpy_item_size[Bodo_CTypes::INT128] = BYTES_PER_DECIMAL;

    PyObject* np_mod = PyImport_ImportModule("numpy");
    PyObject* dtype_obj = PyObject_CallMethod(np_mod, "dtype", "s", "bool");
    if ((size_t)PyNumber_AsSsize_t(
            PyObject_GetAttrString(dtype_obj, "itemsize"), NULL) !=
        sizeof(bool)) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "bool size mismatch between C++ and NumPy!");
        return;
    }
    dtype_obj = PyObject_CallMethod(np_mod, "dtype", "s", "float32");
    if ((size_t)PyNumber_AsSsize_t(
            PyObject_GetAttrString(dtype_obj, "itemsize"), NULL) !=
        sizeof(float)) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "float32 size mismatch between C++ and NumPy!");
        return;
    }
    dtype_obj = PyObject_CallMethod(np_mod, "dtype", "s", "float64");
    if ((size_t)PyNumber_AsSsize_t(
            PyObject_GetAttrString(dtype_obj, "itemsize"), NULL) !=
        sizeof(double)) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "float64 size mismatch between C++ and NumPy!");
        return;
    }
}

Bodo_CTypes::CTypeEnum arrow_to_bodo_type(arrow::Type::type type) {
    switch (type) {
        case arrow::Type::INT8:
            return Bodo_CTypes::INT8;
        case arrow::Type::UINT8:
            return Bodo_CTypes::UINT8;
        case arrow::Type::INT16:
            return Bodo_CTypes::INT16;
        case arrow::Type::UINT16:
            return Bodo_CTypes::UINT16;
        case arrow::Type::INT32:
            return Bodo_CTypes::INT32;
        case arrow::Type::UINT32:
            return Bodo_CTypes::UINT32;
        case arrow::Type::INT64:
            return Bodo_CTypes::INT64;
        case arrow::Type::UINT64:
            return Bodo_CTypes::UINT64;
        case arrow::Type::FLOAT:
            return Bodo_CTypes::FLOAT32;
        case arrow::Type::DOUBLE:
            return Bodo_CTypes::FLOAT64;
        case arrow::Type::DECIMAL:
            return Bodo_CTypes::DECIMAL;
        case arrow::Type::TIMESTAMP:
            return Bodo_CTypes::DATETIME;
        case arrow::Type::STRING:
        case arrow::Type::LARGE_STRING:
            // The main difference between
            // STRING and LARGE_STRING is the offset size
            // (uint32 for STRING and uint64 for LARGE_STRING).
            // We use 64 bit offsets for everything in Bodo,
            // so these two types are equivalent for us.
            return Bodo_CTypes::STRING;
        case arrow::Type::BINARY:
            return Bodo_CTypes::BINARY;
        case arrow::Type::DATE32:
            return Bodo_CTypes::DATE;
        case arrow::Type::TIME32:
            return Bodo_CTypes::TIME;
        case arrow::Type::TIME64:
            return Bodo_CTypes::TIME;
        case arrow::Type::BOOL:
            return Bodo_CTypes::_BOOL;
        // TODO Timedelta
        default: {
            // TODO: Construct the type from the id
            throw std::runtime_error("arrow_to_bodo_type");
        }
    }
}

char* array_info::data1() const {
    switch (arr_type) {
        case bodo_array_type::NULLABLE_INT_BOOL:
        case bodo_array_type::STRING:
        case bodo_array_type::NUMPY:
        case bodo_array_type::CATEGORICAL:
        case bodo_array_type::INTERVAL:
            return (char*)this->buffers[0]->mutable_data() + this->offset;
        case bodo_array_type::LIST_STRING:
            return this->child_arrays[0]->data1();
        case bodo_array_type::DICT:
        case bodo_array_type::ARRAY_ITEM:
        case bodo_array_type::STRUCT:
        default:
            return nullptr;
    }
}

char* array_info::data2() const {
    switch (arr_type) {
        case bodo_array_type::STRING:
        case bodo_array_type::INTERVAL:
            return (char*)this->buffers[1]->mutable_data();
        case bodo_array_type::LIST_STRING:
            return this->child_arrays[0]->data2();
        case bodo_array_type::DICT:
        case bodo_array_type::ARRAY_ITEM:
        case bodo_array_type::STRUCT:
        case bodo_array_type::NULLABLE_INT_BOOL:
        case bodo_array_type::NUMPY:
        case bodo_array_type::CATEGORICAL:
        default:
            return nullptr;
    }
}

char* array_info::data3() const {
    switch (arr_type) {
        case bodo_array_type::LIST_STRING:
            return (char*)this->buffers[0]->mutable_data();
        case bodo_array_type::STRING:
        case bodo_array_type::INTERVAL:
        case bodo_array_type::DICT:
        case bodo_array_type::ARRAY_ITEM:
        case bodo_array_type::STRUCT:
        case bodo_array_type::NULLABLE_INT_BOOL:
        case bodo_array_type::NUMPY:
        case bodo_array_type::CATEGORICAL:
        default:
            return nullptr;
    }
}

char* array_info::null_bitmask() const {
    switch (arr_type) {
        case bodo_array_type::NULLABLE_INT_BOOL:
        case bodo_array_type::LIST_STRING:
        case bodo_array_type::ARRAY_ITEM:
            return (char*)this->buffers[1]->mutable_data();
        case bodo_array_type::STRING:
            return (char*)this->buffers[2]->mutable_data();
        case bodo_array_type::DICT:
            return (char*)this->child_arrays[1]->null_bitmask();
        case bodo_array_type::STRUCT:
            return (char*)this->buffers[0]->mutable_data();
        case bodo_array_type::INTERVAL:
        case bodo_array_type::NUMPY:
        case bodo_array_type::CATEGORICAL:
        default:
            return nullptr;
    }
}

std::shared_ptr<arrow::Array> to_arrow(std::shared_ptr<array_info> arr) {
    std::shared_ptr<arrow::Array> arrow_arr;
    arrow::TimeUnit::type time_unit = arrow::TimeUnit::NANO;
    bodo_array_to_arrow(::arrow::default_memory_pool(), arr, &arrow_arr,
                        false /*convert_timedelta_to_int64*/, "", time_unit,
                        false /*downcast_time_ns_to_us*/);
    return arrow_arr;
}

std::unique_ptr<BodoBuffer> AllocateBodoBuffer(const int64_t size) {
    NRT_MemInfo* meminfo = NRT_MemInfo_alloc_safe_aligned(size, ALIGNMENT);
    return std::make_unique<BodoBuffer>((uint8_t*)meminfo->data, size, meminfo,
                                        false);
}

std::unique_ptr<BodoBuffer> AllocateBodoBuffer(
    const int64_t length, Bodo_CTypes::CTypeEnum typ_enum) {
    int64_t itemsize = numpy_item_size[typ_enum];
    int64_t size = length * itemsize;
    return AllocateBodoBuffer(size);
}

std::shared_ptr<array_info> alloc_numpy(int64_t length,
                                        Bodo_CTypes::CTypeEnum typ_enum) {
    int64_t size = length * numpy_item_size[typ_enum];
    std::unique_ptr<BodoBuffer> buffer = AllocateBodoBuffer(size);
    return std::make_shared<array_info>(
        bodo_array_type::NUMPY, typ_enum, length,
        std::vector<std::shared_ptr<BodoBuffer>>({std::move(buffer)}));
}

std::shared_ptr<array_info> alloc_interval_array(
    int64_t length, Bodo_CTypes::CTypeEnum typ_enum) {
    int64_t size = length * numpy_item_size[typ_enum];
    std::unique_ptr<BodoBuffer> left_buffer = AllocateBodoBuffer(size);
    std::unique_ptr<BodoBuffer> right_buffer = AllocateBodoBuffer(size);
    return std::make_shared<array_info>(
        bodo_array_type::INTERVAL, typ_enum, length,
        std::vector<std::shared_ptr<BodoBuffer>>(
            {std::move(left_buffer), std::move(right_buffer)}));
}

std::shared_ptr<array_info> alloc_categorical(int64_t length,
                                              Bodo_CTypes::CTypeEnum typ_enum,
                                              int64_t num_categories) {
    int64_t size = length * numpy_item_size[typ_enum];
    std::unique_ptr<BodoBuffer> buffer = AllocateBodoBuffer(size);
    return std::make_shared<array_info>(
        bodo_array_type::CATEGORICAL, typ_enum, length,
        std::vector<std::shared_ptr<BodoBuffer>>({std::move(buffer)}),
        std::vector<std::shared_ptr<array_info>>({}), 0, 0, num_categories);
}

std::shared_ptr<array_info> alloc_nullable_array(
    int64_t length, Bodo_CTypes::CTypeEnum typ_enum, int64_t extra_null_bytes) {
    int64_t n_bytes = ((length + 7) >> 3) + extra_null_bytes;
    int64_t size;
    if (typ_enum == Bodo_CTypes::_BOOL) {
        // Boolean arrays store 1 bit per element.
        // Note extra_null_bytes are used for padding when we need
        // to shuffle and split the entries into separate bytes, so
        // we need these for the data as well.
        size = n_bytes;
    } else {
        size = length * numpy_item_size[typ_enum];
    }
    std::unique_ptr<BodoBuffer> buffer = AllocateBodoBuffer(size);
    std::unique_ptr<BodoBuffer> buffer_bitmask =
        AllocateBodoBuffer(n_bytes * sizeof(uint8_t));
    return std::make_shared<array_info>(
        bodo_array_type::NULLABLE_INT_BOOL, typ_enum, length,
        std::vector<std::shared_ptr<BodoBuffer>>(
            {std::move(buffer), std::move(buffer_bitmask)}));
}

std::shared_ptr<array_info> alloc_nullable_array_no_nulls(
    int64_t length, Bodo_CTypes::CTypeEnum typ_enum, int64_t extra_null_bytes) {
    // Same as alloc_nullable_array but we set the null_bitmask
    // such that there are no null values in the output.
    // Useful for cases like allocating indices array of dictionary-encoded
    // string arrays such as input_file_name column where nulls are not possible
    std::shared_ptr<array_info> arr =
        alloc_nullable_array(length, typ_enum, extra_null_bytes);
    size_t n_bytes = ((length + 7) >> 3) + extra_null_bytes;
    memset(arr->null_bitmask(), 0xff, n_bytes);  // null not possible
    return arr;
}

std::shared_ptr<array_info> alloc_nullable_array_all_nulls(
    int64_t length, Bodo_CTypes::CTypeEnum typ_enum, int64_t extra_null_bytes) {
    // Same as alloc_nullable_array but we set the null_bitmask
    // such that all values are null values in the output.
    // Useful for cases like the iceberg void transform.
    std::shared_ptr<array_info> arr =
        alloc_nullable_array(length, typ_enum, extra_null_bytes);
    size_t n_bytes = ((length + 7) >> 3) + extra_null_bytes;
    memset(arr->null_bitmask(), 0x00, n_bytes);  // all nulls
    return arr;
}

std::shared_ptr<array_info> alloc_string_array(int64_t length, int64_t n_chars,
                                               int64_t extra_null_bytes) {
    // allocate data/offsets/null_bitmap arrays
    std::unique_ptr<BodoBuffer> data_buffer =
        AllocateBodoBuffer(n_chars, Bodo_CTypes::UINT8);
    std::unique_ptr<BodoBuffer> offsets_buffer =
        AllocateBodoBuffer(length + 1, Bodo_CType_offset);
    int64_t n_bytes = (int64_t)((length + 7) / 8) + extra_null_bytes;
    std::unique_ptr<BodoBuffer> null_bitmap_buffer =
        AllocateBodoBuffer(n_bytes, Bodo_CTypes::UINT8);
    // setting all to non-null to avoid unexpected issues
    memset(null_bitmap_buffer->mutable_data(), 0xff, n_bytes);

    // set offsets for boundaries
    offset_t* offsets_ptr = (offset_t*)offsets_buffer->mutable_data();
    offsets_ptr[0] = 0;
    offsets_ptr[length] = n_chars;

    return std::make_shared<array_info>(
        bodo_array_type::STRING, Bodo_CTypes::STRING, length,
        std::vector<std::shared_ptr<BodoBuffer>>(
            {std::move(data_buffer), std::move(offsets_buffer),
             std::move(null_bitmap_buffer)}));
}

std::shared_ptr<array_info> alloc_dict_string_array(
    int64_t length, int64_t n_keys, int64_t n_chars_keys,
    bool has_global_dictionary, bool has_deduped_local_dictionary) {
    // dictionary
    std::shared_ptr<array_info> dict_data_arr =
        alloc_string_array(n_keys, n_chars_keys, 0);
    // indices
    std::shared_ptr<array_info> indices_data_arr =
        alloc_nullable_array(length, Bodo_CTypes::INT32, 0);

    return std::make_shared<array_info>(
        bodo_array_type::DICT, Bodo_CTypes::CTypeEnum::STRING, length,
        std::vector<std::shared_ptr<BodoBuffer>>({}),
        std::vector<std::shared_ptr<array_info>>(
            {dict_data_arr, indices_data_arr}),
        0, 0, 0, has_global_dictionary, has_deduped_local_dictionary, false);
}

std::shared_ptr<array_info> create_string_array(
    std::vector<uint8_t> const& null_bitmap,
    std::vector<std::string> const& list_string) {
    size_t len = list_string.size();
    // Calculate the number of characters for allocating the string.
    size_t nb_char = 0;
    std::vector<std::string>::const_iterator iter = list_string.begin();
    for (size_t i_grp = 0; i_grp < len; i_grp++) {
        if (GetBit(null_bitmap.data(), i_grp)) {
            nb_char += iter->size();
        }
        iter++;
    }
    size_t extra_bytes = 0;
    std::shared_ptr<array_info> out_col =
        alloc_string_array(len, nb_char, extra_bytes);
    // update string array payload to reflect change
    char* data_o = out_col->data1();
    offset_t* offsets_o = (offset_t*)out_col->data2();
    offset_t pos = 0;
    iter = list_string.begin();
    for (size_t i_grp = 0; i_grp < len; i_grp++) {
        offsets_o[i_grp] = pos;
        bool bit = GetBit(null_bitmap.data(), i_grp);
        if (bit) {
            size_t len_str = size_t(iter->size());
            memcpy(data_o, iter->data(), len_str);
            data_o += len_str;
            pos += len_str;
        }
        out_col->set_null_bit(i_grp, bit);
        iter++;
    }
    offsets_o[len] = pos;
    return out_col;
}

std::shared_ptr<array_info> create_list_string_array(
    std::vector<uint8_t> const& null_bitmap,
    std::vector<std::vector<std::pair<std::string, bool>>> const&
        list_list_pair) {
    size_t len = list_list_pair.size();
    // Determining the number of characters in output.
    size_t nb_string = 0;
    size_t nb_char = 0;
    std::vector<std::vector<std::pair<std::string, bool>>>::const_iterator
        iter = list_list_pair.begin();
    for (size_t i_grp = 0; i_grp < len; i_grp++) {
        if (GetBit(null_bitmap.data(), i_grp)) {
            std::vector<std::pair<std::string, bool>> e_list = *iter;
            nb_string += e_list.size();
            for (auto& e_str : e_list) {
                nb_char += e_str.first.size();
            }
        }
        iter++;
    }
    // Allocation needs to be done through
    // alloc_list_string_array, which allocates with meminfos
    // and same data structs that Python uses. We need to
    // re-allocate here because number of strings and chars has
    // been determined here (previous out_col was just an empty
    // dummy allocation).

    std::shared_ptr<array_info> new_out_col =
        alloc_list_string_array(len, nb_string, nb_char, 0);
    offset_t* index_offsets_o = (offset_t*)new_out_col->data3();
    offset_t* data_offsets_o = (offset_t*)new_out_col->data2();
    uint8_t* sub_null_bitmask_o = (uint8_t*)new_out_col->sub_null_bitmask();
    // Writing the list_strings in output
    char* data_o = new_out_col->data1();
    data_offsets_o[0] = 0;
    offset_t pos_index = 0;
    offset_t pos_data = 0;
    iter = list_list_pair.begin();
    for (size_t i_grp = 0; i_grp < len; i_grp++) {
        bool bit = GetBit(null_bitmap.data(), i_grp);
        new_out_col->set_null_bit(i_grp, bit);
        index_offsets_o[i_grp] = pos_index;
        if (bit) {
            std::vector<std::pair<std::string, bool>> e_list = *iter;
            offset_t n_string = e_list.size();
            for (offset_t i_str = 0; i_str < n_string; i_str++) {
                std::string& estr = e_list[i_str].first;
                offset_t n_char = estr.size();
                memcpy(data_o, estr.data(), n_char);
                data_o += n_char;
                pos_data++;
                data_offsets_o[pos_data] =
                    data_offsets_o[pos_data - 1] + n_char;
                bool bit = e_list[i_str].second;
                SetBitTo(sub_null_bitmask_o, pos_index + i_str, bit);
            }
            pos_index += n_string;
        }
        iter++;
    }
    index_offsets_o[len] = pos_index;
    return new_out_col;
}

std::shared_ptr<array_info> create_dict_string_array(
    std::shared_ptr<array_info> dict_arr,
    std::shared_ptr<array_info> indices_arr, bool has_global_dictionary,
    bool has_deduped_local_dictionary, bool has_sorted_dictionary) {
    std::shared_ptr<array_info> out_col = std::make_shared<array_info>(
        bodo_array_type::DICT, Bodo_CTypes::CTypeEnum::STRING,
        indices_arr->length, std::vector<std::shared_ptr<BodoBuffer>>({}),
        std::vector<std::shared_ptr<array_info>>({dict_arr, indices_arr}), 0, 0,
        0, has_global_dictionary, has_deduped_local_dictionary,
        has_sorted_dictionary);
    return out_col;
}

/**
 * Allocates memory for string allocation as a NRT_MemInfo
 */
NRT_MemInfo* alloc_meminfo(int64_t length) {
    return NRT_MemInfo_alloc_safe(length);
}

/**
 * @brief destrcutor for array(item) array meminfo. Decrefs the underlying data,
 * offsets and null_bitmap arrays.
 *
 * Note: duplicate of dtor_array_item_array but with array_item_arr_payload
 * TODO: refactor
 *
 * @param payload array(item) array meminfo payload
 * @param size payload size (ignored)
 * @param in extra info (ignored)
 */
void dtor_array_item_arr(array_item_arr_payload* payload, int64_t size,
                         void* in) {
    if (payload->data->refct != -1) {
        payload->data->refct--;
    }
    if (payload->data->refct == 0) {
        NRT_MemInfo_call_dtor(payload->data);
    }

    if (payload->offsets.meminfo->refct != -1) {
        payload->offsets.meminfo->refct--;
    }
    if (payload->offsets.meminfo->refct == 0) {
        NRT_MemInfo_call_dtor(payload->offsets.meminfo);
    }

    if (payload->null_bitmap.meminfo->refct != -1) {
        payload->null_bitmap.meminfo->refct--;
    }
    if (payload->null_bitmap.meminfo->refct == 0) {
        NRT_MemInfo_call_dtor(payload->null_bitmap.meminfo);
    }
}

std::shared_ptr<array_info> alloc_list_string_array(
    int64_t length, std::shared_ptr<array_info> string_arr,
    int64_t extra_null_bytes) {
    std::unique_ptr<BodoBuffer> offsets_buffer =
        AllocateBodoBuffer(length + 1, Bodo_CType_offset);
    int64_t n_bytes = (int64_t)((length + 7) / 8) + extra_null_bytes;
    std::unique_ptr<BodoBuffer> null_bitmap_buffer =
        AllocateBodoBuffer(n_bytes, Bodo_CTypes::UINT8);
    // setting all to non-null to avoid unexpected issues
    memset(null_bitmap_buffer->mutable_data(), 0xff, n_bytes);

    // set offsets for boundaries
    offset_t* offsets_ptr = (offset_t*)offsets_buffer->mutable_data();
    offsets_ptr[0] = 0;
    offsets_ptr[length] = string_arr->length;

    return std::make_shared<array_info>(
        bodo_array_type::LIST_STRING, Bodo_CTypes::LIST_STRING, length,
        std::vector<std::shared_ptr<BodoBuffer>>(
            {std::move(offsets_buffer), std::move(null_bitmap_buffer)}),
        std::vector<std::shared_ptr<array_info>>({string_arr}));
}

std::shared_ptr<array_info> alloc_list_string_array(int64_t n_lists,
                                                    int64_t n_strings,
                                                    int64_t n_chars,
                                                    int64_t extra_null_bytes) {
    // allocate string data array
    std::shared_ptr<array_info> data_arr = alloc_array(
        n_strings, n_chars, -1, bodo_array_type::arr_type_enum::STRING,
        Bodo_CTypes::UINT8, extra_null_bytes, 0);

    return alloc_list_string_array(n_lists, data_arr, extra_null_bytes);
}

/**
 * @brief allocate a numpy array payload
 *
 * @param length number of elements
 * @param typ_enum dtype of elements
 * @return numpy_arr_payload
 */
numpy_arr_payload allocate_numpy_payload(int64_t length,
                                         Bodo_CTypes::CTypeEnum typ_enum) {
    int64_t itemsize = numpy_item_size[typ_enum];
    int64_t size = length * itemsize;
    NRT_MemInfo* meminfo = NRT_MemInfo_alloc_safe_aligned(size, ALIGNMENT);
    char* data = (char*)meminfo->data;
    return make_numpy_array_payload(meminfo, NULL, length, itemsize, data,
                                    length, itemsize);
}

/**
 * @brief decref numpy array stored in payload and free if refcount becomes
 * zero.
 *
 * @param arr
 */
void decref_numpy_payload(numpy_arr_payload arr) {
    if (arr.meminfo->refct != -1) {
        arr.meminfo->refct--;
    }
    if (arr.meminfo->refct == 0) {
        NRT_MemInfo_call_dtor(arr.meminfo);
    }
}

/**
 * @brief destructor for array(item) array meminfo. Decrefs the underlying data,
 * offsets and null_bitmap arrays.
 *
 * @param payload array(item) array meminfo payload
 * @param size payload size (ignored)
 * @param in extra info (ignored)
 */
void dtor_array_item_array(array_item_arr_numpy_payload* payload, int64_t size,
                           void* in) {
    if (payload->data.meminfo->refct != -1) {
        payload->data.meminfo->refct--;
    }
    if (payload->data.meminfo->refct == 0) {
        NRT_MemInfo_call_dtor(payload->data.meminfo);
    }

    if (payload->offsets.meminfo->refct != -1) {
        payload->offsets.meminfo->refct--;
    }
    if (payload->offsets.meminfo->refct == 0) {
        NRT_MemInfo_call_dtor(payload->offsets.meminfo);
    }

    if (payload->null_bitmap.meminfo->refct != -1) {
        payload->null_bitmap.meminfo->refct--;
    }
    if (payload->null_bitmap.meminfo->refct == 0) {
        NRT_MemInfo_call_dtor(payload->null_bitmap.meminfo);
    }
}

/**
 * @brief create a meminfo for array(item) array
 *
 * @return NRT_MemInfo*
 */
NRT_MemInfo* alloc_array_item_arr_meminfo() {
    return NRT_MemInfo_alloc_dtor_safe(
        sizeof(array_item_arr_numpy_payload),
        (NRT_dtor_function)dtor_array_item_array);
}

/**
 * The allocations array function for the function.
 *
 * In the case of NUMPY/CATEGORICAL or NULLABLE_INT_BOOL,
 * -- length is the number of rows, and n_sub_elems, n_sub_sub_elems do not
 * matter.
 * In the case of STRING:
 * -- length is the number of rows (= number of strings)
 * -- n_sub_elems is the total number of characters.
 * In the case of LIST_STRING:
 * -- length is the number of rows.
 * -- n_sub_elems is the number of strings.
 * -- n_sub_sub_elems is the total number of characters.
 * In the case of DICT:
 * -- length is the number of rows (same as the number of indices)
 * -- n_sub_elems is the number of keys in the dictionary
 * -- n_sub_sub_elems is the total number of characters for
 *    the keys in the dictionary
 */
std::shared_ptr<array_info> alloc_array(int64_t length, int64_t n_sub_elems,
                                        int64_t n_sub_sub_elems,
                                        bodo_array_type::arr_type_enum arr_type,
                                        Bodo_CTypes::CTypeEnum dtype,
                                        int64_t extra_null_bytes,
                                        int64_t num_categories) {
    switch (arr_type) {
        case bodo_array_type::LIST_STRING:
            return alloc_list_string_array(length, n_sub_elems, n_sub_sub_elems,
                                           extra_null_bytes);

        case bodo_array_type::STRING:
            return alloc_string_array(length, n_sub_elems, extra_null_bytes);

        case bodo_array_type::NULLABLE_INT_BOOL:
            return alloc_nullable_array(length, dtype, extra_null_bytes);

        case bodo_array_type::INTERVAL:
            return alloc_interval_array(length, dtype);

        case bodo_array_type::NUMPY:
            return alloc_numpy(length, dtype);

        case bodo_array_type::CATEGORICAL:
            return alloc_categorical(length, dtype, num_categories);

        case bodo_array_type::DICT:
            return alloc_dict_string_array(length, n_sub_elems, n_sub_sub_elems,
                                           false, false);
        default:
            throw std::runtime_error("Type not covered in alloc_array");
    }
}

int64_t arrow_array_memory_size(std::shared_ptr<arrow::Array> arr) {
    int64_t n_rows = arr->length();
    int64_t n_bytes = (n_rows + 7) >> 3;
#if OFFSET_BITWIDTH == 32
    if (arr->type_id() == arrow::Type::LIST) {
        std::shared_ptr<arrow::ListArray> list_arr =
            std::dynamic_pointer_cast<arrow::ListArray>(arr);
#else
    if (arr->type_id() == arrow::Type::LARGE_LIST) {
        std::shared_ptr<arrow::LargeListArray> list_arr =
            std::dynamic_pointer_cast<arrow::LargeListArray>(arr);
#endif
        int64_t siz_offset = sizeof(offset_t) * (n_rows + 1);
        int64_t siz_null_bitmap = n_bytes;
        std::shared_ptr<arrow::Array> arr_values = list_arr->values();
        return siz_offset + siz_null_bitmap +
               arrow_array_memory_size(arr_values);
    }
    if (arr->type_id() == arrow::Type::STRUCT) {
        std::shared_ptr<arrow::StructArray> struct_arr =
            std::dynamic_pointer_cast<arrow::StructArray>(arr);
        auto struct_type =
            std::dynamic_pointer_cast<arrow::StructType>(struct_arr->type());
        int64_t num_fields = struct_type->num_fields();
        int64_t total_siz = n_bytes;
        for (int64_t i_field = 0; i_field < num_fields; i_field++)
            total_siz += arrow_array_memory_size(struct_arr->field(i_field));
        return total_siz;
    }
#if OFFSET_BITWIDTH == 32
    if (arr->type_id() == arrow::Type::STRING) {
        std::shared_ptr<arrow::StringArray> string_array =
            std::dynamic_pointer_cast<arrow::StringArray>(arr);
#else
    if (arr->type_id() == arrow::Type::LARGE_STRING) {
        std::shared_ptr<arrow::LargeStringArray> string_array =
            std::dynamic_pointer_cast<arrow::LargeStringArray>(arr);
#endif
        int64_t siz_offset = sizeof(offset_t) * (n_rows + 1);
        int64_t siz_null_bitmap = n_bytes;
        int64_t siz_character = string_array->value_offset(n_rows);
        return siz_offset + siz_null_bitmap + siz_character;
    } else {
        int64_t siz_null_bitmap = n_bytes;
        std::shared_ptr<arrow::PrimitiveArray> prim_arr =
            std::dynamic_pointer_cast<arrow::PrimitiveArray>(arr);
        std::shared_ptr<arrow::DataType> arrow_type = prim_arr->type();
        int64_t siz_primitive_data;
        if (arrow_type->id() == arrow::Type::BOOL) {
            // Arrow boolean arrays store 1 bit per boolean
            siz_primitive_data = n_bytes;
        } else {
            Bodo_CTypes::CTypeEnum bodo_typ =
                arrow_to_bodo_type(prim_arr->type()->id());
            int64_t siz_typ = numpy_item_size[bodo_typ];
            siz_primitive_data = siz_typ * n_rows;
        }
        return siz_null_bitmap + siz_primitive_data;
    }
}

int64_t array_memory_size(std::shared_ptr<array_info> earr) {
    if (earr->arr_type == bodo_array_type::NUMPY ||
        earr->arr_type == bodo_array_type::CATEGORICAL) {
        uint64_t siztype = numpy_item_size[earr->dtype];
        return siztype * earr->length;
    }
    if (earr->arr_type == bodo_array_type::NULLABLE_INT_BOOL ||
        earr->arr_type == bodo_array_type::DICT) {
        if (earr->arr_type == bodo_array_type::DICT) {
            // TODO also contribute dictionary size, but note that when
            // table_global_memory_size() calls this function, it's not supposed
            // to add the size of dictionaries of all ranks
            earr = earr->child_arrays[1];
        }
        int64_t n_bytes = ((earr->length + 7) >> 3);
        if (earr->dtype == Bodo_CTypes::_BOOL) {
            // Nullable boolean arrays store 1 bit per boolean.
            return n_bytes * 2;
        } else {
            uint64_t siztype = numpy_item_size[earr->dtype];
            return n_bytes + siztype * earr->length;
        }
    }
    if (earr->arr_type == bodo_array_type::STRING) {
        int64_t n_bytes = ((earr->length + 7) >> 3);
        return earr->n_sub_elems() + sizeof(offset_t) * (earr->length + 1) +
               n_bytes;
    }
    if (earr->arr_type == bodo_array_type::LIST_STRING) {
        int64_t n_bytes = ((earr->length + 7) >> 3);
        int64_t n_sub_bytes = ((earr->n_sub_elems() + 7) >> 3);
        return earr->n_sub_sub_elems() +
               sizeof(offset_t) * (earr->n_sub_elems() + 1) +
               sizeof(offset_t) * (earr->length + 1) + n_bytes + n_sub_bytes;
    }
    if (earr->arr_type == bodo_array_type::STRUCT ||
        earr->arr_type == bodo_array_type::ARRAY_ITEM) {
        return arrow_array_memory_size(to_arrow(earr));
    }
    Bodo_PyErr_SetString(PyExc_RuntimeError,
                         "Type not covered in array_memory_size");
    return 0;
}

int64_t table_local_memory_size(std::shared_ptr<table_info> table) {
    int64_t local_size = 0;
    for (auto& arr : table->columns) {
        local_size += array_memory_size(arr);
    }
    return local_size;
}

int64_t table_global_memory_size(std::shared_ptr<table_info> table) {
    int64_t local_size = table_local_memory_size(table);
    int64_t global_size;
    MPI_Allreduce(&local_size, &global_size, 1, MPI_LONG_LONG_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    return global_size;
}

std::shared_ptr<array_info> copy_array(std::shared_ptr<array_info> earr) {
    int64_t extra_null_bytes = 0;
    std::shared_ptr<array_info> farr;
    if (earr->arr_type == bodo_array_type::DICT) {
        std::shared_ptr<array_info> dictionary =
            copy_array(earr->child_arrays[0]);
        std::shared_ptr<array_info> indices = copy_array(earr->child_arrays[1]);
        farr = std::make_shared<array_info>(
            bodo_array_type::DICT, earr->dtype, indices->length,
            std::vector<std::shared_ptr<BodoBuffer>>({}),
            std::vector<std::shared_ptr<array_info>>({dictionary, indices}), 0,
            0, 0, earr->has_global_dictionary,
            earr->has_deduped_local_dictionary, earr->has_sorted_dictionary);
    } else {
        farr = alloc_array(earr->length, earr->n_sub_elems(),
                           earr->n_sub_sub_elems(), earr->arr_type, earr->dtype,
                           extra_null_bytes, earr->num_categories);
    }
    if (earr->arr_type == bodo_array_type::NUMPY ||
        earr->arr_type == bodo_array_type::CATEGORICAL) {
        uint64_t siztype = numpy_item_size[earr->dtype];
        memcpy(farr->data1(), earr->data1(), siztype * earr->length);
    }
    if (earr->arr_type == bodo_array_type::INTERVAL) {
        uint64_t siztype = numpy_item_size[earr->dtype];
        memcpy(farr->data1(), earr->data1(), siztype * earr->length);
        memcpy(farr->data2(), earr->data2(), siztype * earr->length);
    }
    if (earr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        int64_t data_copy_size;
        int64_t n_bytes = ((earr->length + 7) >> 3);
        if (earr->dtype == Bodo_CTypes::_BOOL) {
            data_copy_size = n_bytes;
        } else {
            data_copy_size = earr->length * numpy_item_size[earr->dtype];
        }
        memcpy(farr->data1(), earr->data1(), data_copy_size);
        memcpy(farr->null_bitmask(), earr->null_bitmask(), n_bytes);
    }
    if (earr->arr_type == bodo_array_type::STRING) {
        memcpy(farr->data1(), earr->data1(), earr->n_sub_elems());
        memcpy(farr->data2(), earr->data2(),
               sizeof(offset_t) * (earr->length + 1));
        int64_t n_bytes = ((earr->length + 7) >> 3);
        memcpy(farr->null_bitmask(), earr->null_bitmask(), n_bytes);
    }
    if (earr->arr_type == bodo_array_type::LIST_STRING) {
        memcpy(farr->data1(), earr->data1(), earr->n_sub_sub_elems());
        memcpy(farr->data2(), earr->data2(),
               sizeof(offset_t) * (earr->n_sub_elems() + 1));
        memcpy(farr->data3(), earr->data3(),
               sizeof(offset_t) * (earr->length + 1));
        int64_t n_bytes = ((earr->length + 7) >> 3);
        memcpy(farr->null_bitmask(), earr->null_bitmask(), n_bytes);
        int64_t n_sub_bytes = ((earr->n_sub_elems() + 7) >> 3);
        memcpy(farr->sub_null_bitmask(), earr->sub_null_bitmask(), n_sub_bytes);
    }
    return farr;
}

/**
 * Free underlying array of array_info pointer and delete the pointer.
 * Called from Python.
 */
void delete_info(array_info* arr) { delete arr; }

/**
 * Delete table pointer and its column array_info pointers (but not the arrays).
 * Called from Python.
 */
void delete_table(table_info* table) { delete table; }

void decref_meminfo(MemInfo* meminfo) {
    if (meminfo != NULL && meminfo->refct != -1) {
        meminfo->refct--;
        if (meminfo->refct == 0) {
            NRT_MemInfo_call_dtor(meminfo);
        }
    }
}

void incref_meminfo(MemInfo* meminfo) {
    if (meminfo != NULL && meminfo->refct != -1) {
        meminfo->refct++;
    }
}

void reset_col_if_last_table_ref(std::shared_ptr<table_info> const& table,
                                 size_t col_idx) {
    if (table.use_count() == 1) {
        table->columns[col_idx].reset();
    }
}

void clear_all_cols_if_last_table_ref(
    std::shared_ptr<table_info> const& table) {
    if (table.use_count() == 1) {
        table->columns.clear();
    }
}

// get memory alloc/free info from _meminfo.h
size_t get_stats_alloc() { return NRT_MemSys_get_stats_alloc(); }
size_t get_stats_free() { return NRT_MemSys_get_stats_free(); }
size_t get_stats_mi_alloc() { return NRT_MemSys_get_stats_mi_alloc(); }
size_t get_stats_mi_free() { return NRT_MemSys_get_stats_mi_free(); }

extern "C" {

PyMODINIT_FUNC PyInit_ext(void) {
    PyObject* m;
    MOD_DEF(m, "ext", "No docs", NULL);
    if (m == NULL)
        return NULL;

    bodo_common_init();

    SetAttrStringFromPyInit(m, hdist);
    SetAttrStringFromPyInit(m, hstr_ext);
    SetAttrStringFromPyInit(m, decimal_ext);
    SetAttrStringFromPyInit(m, quantile_alg);
    SetAttrStringFromPyInit(m, hdatetime_ext);
    SetAttrStringFromPyInit(m, hio);
    SetAttrStringFromPyInit(m, array_ext);
    SetAttrStringFromPyInit(m, s3_reader);
    SetAttrStringFromPyInit(m, fsspec_reader);
    SetAttrStringFromPyInit(m, hdfs_reader);

    SetAttrStringFromPyInit(m, _hdf5);
    SetAttrStringFromPyInit(m, arrow_cpp);
    SetAttrStringFromPyInit(m, csv_cpp);
    SetAttrStringFromPyInit(m, json_cpp);

    return m;
}

} /* extern "C" */
