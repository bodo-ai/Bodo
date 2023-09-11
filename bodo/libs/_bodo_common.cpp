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

std::shared_ptr<arrow::Array> to_arrow(const std::shared_ptr<array_info> arr) {
    std::shared_ptr<arrow::Array> arrow_arr;
    arrow::TimeUnit::type time_unit = arrow::TimeUnit::NANO;
    bodo_array_to_arrow(bodo::BufferPool::DefaultPtr(), std::move(arr),
                        &arrow_arr, false /*convert_timedelta_to_int64*/, "",
                        time_unit, false /*downcast_time_ns_to_us*/);
    return arrow_arr;
}

std::unique_ptr<BodoBuffer> AllocateBodoBuffer(
    const int64_t size, bodo::IBufferPool* const pool,
    const std::shared_ptr<::arrow::MemoryManager> mm) {
    NRT_MemInfo* meminfo =
        NRT_MemInfo_alloc_safe_aligned_pool(size, ALIGNMENT, pool);
    return std::make_unique<BodoBuffer>((uint8_t*)meminfo->data, size, meminfo,
                                        false, pool, std::move(mm));
}

std::unique_ptr<BodoBuffer> AllocateBodoBuffer(
    const int64_t length, Bodo_CTypes::CTypeEnum typ_enum,
    bodo::IBufferPool* const pool,
    const std::shared_ptr<::arrow::MemoryManager> mm) {
    int64_t itemsize = numpy_item_size[typ_enum];
    int64_t size = length * itemsize;
    return AllocateBodoBuffer(size, pool, std::move(mm));
}

std::unique_ptr<array_info> alloc_numpy(
    int64_t length, Bodo_CTypes::CTypeEnum typ_enum,
    bodo::IBufferPool* const pool,
    const std::shared_ptr<::arrow::MemoryManager> mm) {
    int64_t size = length * numpy_item_size[typ_enum];
    std::unique_ptr<BodoBuffer> buffer =
        AllocateBodoBuffer(size, pool, std::move(mm));
    return std::make_unique<array_info>(
        bodo_array_type::NUMPY, typ_enum, length,
        std::vector<std::shared_ptr<BodoBuffer>>({std::move(buffer)}));
}

std::unique_ptr<array_info> alloc_interval_array(
    int64_t length, Bodo_CTypes::CTypeEnum typ_enum,
    bodo::IBufferPool* const pool,
    const std::shared_ptr<::arrow::MemoryManager> mm) {
    int64_t size = length * numpy_item_size[typ_enum];
    std::unique_ptr<BodoBuffer> left_buffer =
        AllocateBodoBuffer(size, pool, mm);
    std::unique_ptr<BodoBuffer> right_buffer =
        AllocateBodoBuffer(size, pool, std::move(mm));
    return std::make_unique<array_info>(
        bodo_array_type::INTERVAL, typ_enum, length,
        std::vector<std::shared_ptr<BodoBuffer>>(
            {std::move(left_buffer), std::move(right_buffer)}));
}

std::unique_ptr<array_info> alloc_array_item(
    int64_t n_arrays, std::shared_ptr<array_info> inner_arr,
    bodo::IBufferPool* const pool,
    const std::shared_ptr<::arrow::MemoryManager> mm) {
    std::unique_ptr<BodoBuffer> offsets_buffer =
        AllocateBodoBuffer(n_arrays + 1, Bodo_CType_offset, pool, mm);
    int64_t n_bytes = (int64_t)((n_arrays + 7) >> 3);
    std::unique_ptr<BodoBuffer> null_bitmap_buffer =
        AllocateBodoBuffer(n_bytes, Bodo_CTypes::UINT8, pool, std::move(mm));
    // setting all to non-null to avoid unexpected issues
    memset(null_bitmap_buffer->mutable_data(), 0xff, n_bytes);
    return std::make_unique<array_info>(
        bodo_array_type::ARRAY_ITEM, Bodo_CTypes::LIST, n_arrays,
        std::vector<std::shared_ptr<BodoBuffer>>(
            {std::move(offsets_buffer), std::move(null_bitmap_buffer)}),
        std::vector<std::shared_ptr<array_info>>({inner_arr}));
}

std::unique_ptr<array_info> alloc_categorical(
    int64_t length, Bodo_CTypes::CTypeEnum typ_enum, int64_t num_categories,
    bodo::IBufferPool* const pool,
    const std::shared_ptr<::arrow::MemoryManager> mm) {
    int64_t size = length * numpy_item_size[typ_enum];
    std::unique_ptr<BodoBuffer> buffer =
        AllocateBodoBuffer(size, pool, std::move(mm));
    return std::make_unique<array_info>(
        bodo_array_type::CATEGORICAL, typ_enum, length,
        std::vector<std::shared_ptr<BodoBuffer>>({std::move(buffer)}),
        std::vector<std::shared_ptr<array_info>>({}), 0, 0, num_categories);
}

std::unique_ptr<array_info> alloc_nullable_array(
    int64_t length, Bodo_CTypes::CTypeEnum typ_enum, int64_t extra_null_bytes,
    bodo::IBufferPool* const pool,
    const std::shared_ptr<::arrow::MemoryManager> mm) {
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
    std::unique_ptr<BodoBuffer> buffer = AllocateBodoBuffer(size, pool, mm);
    std::unique_ptr<BodoBuffer> buffer_bitmask =
        AllocateBodoBuffer(n_bytes * sizeof(uint8_t), pool, std::move(mm));
    return std::make_unique<array_info>(
        bodo_array_type::NULLABLE_INT_BOOL, typ_enum, length,
        std::vector<std::shared_ptr<BodoBuffer>>(
            {std::move(buffer), std::move(buffer_bitmask)}));
}

std::unique_ptr<array_info> alloc_nullable_array_no_nulls(
    int64_t length, Bodo_CTypes::CTypeEnum typ_enum, int64_t extra_null_bytes) {
    // Same as alloc_nullable_array but we set the null_bitmask
    // such that there are no null values in the output.
    // Useful for cases like allocating indices array of dictionary-encoded
    // string arrays such as input_file_name column where nulls are not possible
    std::unique_ptr<array_info> arr =
        alloc_nullable_array(length, typ_enum, extra_null_bytes);
    size_t n_bytes = ((length + 7) >> 3) + extra_null_bytes;
    memset(arr->null_bitmask(), 0xff, n_bytes);  // null not possible
    return arr;
}

std::unique_ptr<array_info> alloc_nullable_array_all_nulls(
    int64_t length, Bodo_CTypes::CTypeEnum typ_enum, int64_t extra_null_bytes) {
    // Same as alloc_nullable_array but we set the null_bitmask
    // such that all values are null values in the output.
    // Useful for cases like the iceberg void transform.
    std::unique_ptr<array_info> arr =
        alloc_nullable_array(length, typ_enum, extra_null_bytes);
    size_t n_bytes = ((length + 7) >> 3) + extra_null_bytes;
    memset(arr->null_bitmask(), 0x00, n_bytes);  // all nulls
    return arr;
}

std::unique_ptr<array_info> alloc_string_array(
    Bodo_CTypes::CTypeEnum typ_enum, int64_t length, int64_t n_chars,
    int64_t array_id, int64_t extra_null_bytes, bool is_globally_replicated,
    bool is_locally_unique, bool is_locally_sorted,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    // allocate data/offsets/null_bitmap arrays
    std::unique_ptr<BodoBuffer> data_buffer =
        AllocateBodoBuffer(n_chars, Bodo_CTypes::UINT8, pool, mm);
    std::unique_ptr<BodoBuffer> offsets_buffer =
        AllocateBodoBuffer(length + 1, Bodo_CType_offset, pool, mm);
    int64_t n_bytes = (int64_t)((length + 7) / 8) + extra_null_bytes;
    std::unique_ptr<BodoBuffer> null_bitmap_buffer =
        AllocateBodoBuffer(n_bytes, Bodo_CTypes::UINT8, pool, std::move(mm));
    // setting all to non-null to avoid unexpected issues
    memset(null_bitmap_buffer->mutable_data(), 0xff, n_bytes);

    // set offsets for boundaries
    offset_t* offsets_ptr = (offset_t*)offsets_buffer->mutable_data();
    offsets_ptr[0] = 0;
    offsets_ptr[length] = n_chars;

    // Generate a valid array id
    if (array_id < 0) {
        array_id = generate_array_id(length);
    }

    return std::make_unique<array_info>(
        bodo_array_type::STRING, typ_enum, length,
        std::vector<std::shared_ptr<BodoBuffer>>(
            {std::move(data_buffer), std::move(offsets_buffer),
             std::move(null_bitmap_buffer)}),
        std::vector<std::shared_ptr<array_info>>({}), 0, 0, 0, array_id,
        is_globally_replicated, is_locally_unique, is_locally_sorted);
}

std::unique_ptr<array_info> alloc_dict_string_array(
    int64_t length, int64_t n_keys, int64_t n_chars_keys,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    // dictionary
    std::shared_ptr<array_info> dict_data_arr =
        alloc_string_array(Bodo_CTypes::CTypeEnum::STRING, n_keys, n_chars_keys,
                           -1, 0, false, false, false, pool, mm);
    // indices
    std::shared_ptr<array_info> indices_data_arr = alloc_nullable_array(
        length, Bodo_CTypes::INT32, 0, pool, std::move(mm));

    return std::make_unique<array_info>(
        bodo_array_type::DICT, Bodo_CTypes::CTypeEnum::STRING, length,
        std::vector<std::shared_ptr<BodoBuffer>>({}),
        std::vector<std::shared_ptr<array_info>>(
            {dict_data_arr, indices_data_arr}));
}

std::unique_ptr<array_info> create_string_array(
    Bodo_CTypes::CTypeEnum typ_enum, bodo::vector<uint8_t> const& null_bitmap,
    bodo::vector<std::string> const& list_string, int64_t array_id) {
    size_t len = list_string.size();
    // Calculate the number of characters for allocating the string.
    size_t nb_char = 0;
    bodo::vector<std::string>::const_iterator iter = list_string.begin();
    for (size_t i_grp = 0; i_grp < len; i_grp++) {
        if (GetBit(null_bitmap.data(), i_grp)) {
            nb_char += iter->size();
        }
        iter++;
    }
    std::unique_ptr<array_info> out_col =
        alloc_string_array(typ_enum, len, nb_char, array_id);
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

std::unique_ptr<array_info> create_list_string_array(
    bodo::vector<uint8_t> const& null_bitmap,
    bodo::vector<bodo::vector<std::pair<std::string, bool>>> const&
        list_list_pair) {
    size_t len = list_list_pair.size();
    // Determining the number of characters in output.
    size_t nb_string = 0;
    size_t nb_char = 0;
    bodo::vector<bodo::vector<std::pair<std::string, bool>>>::const_iterator
        iter = list_list_pair.begin();
    for (size_t i_grp = 0; i_grp < len; i_grp++) {
        if (GetBit(null_bitmap.data(), i_grp)) {
            bodo::vector<std::pair<std::string, bool>> e_list = *iter;
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

    std::unique_ptr<array_info> new_out_col =
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
            bodo::vector<std::pair<std::string, bool>> e_list = *iter;
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

std::unique_ptr<array_info> create_dict_string_array(
    std::shared_ptr<array_info> dict_arr,
    std::shared_ptr<array_info> indices_arr) {
    std::unique_ptr<array_info> out_col = std::make_unique<array_info>(
        bodo_array_type::DICT, Bodo_CTypes::CTypeEnum::STRING,
        indices_arr->length, std::vector<std::shared_ptr<BodoBuffer>>({}),
        std::vector<std::shared_ptr<array_info>>({dict_arr, indices_arr}));
    return out_col;
}

/**
 * Allocates memory for string allocation as a NRT_MemInfo
 */
NRT_MemInfo* alloc_meminfo(int64_t length) {
    return NRT_MemInfo_alloc_safe(length);
}

std::unique_ptr<array_info> alloc_list_string_array(
    int64_t length, std::shared_ptr<array_info> string_arr,
    int64_t extra_null_bytes, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    std::unique_ptr<BodoBuffer> offsets_buffer =
        AllocateBodoBuffer(length + 1, Bodo_CType_offset, pool, mm);
    int64_t n_bytes = (int64_t)((length + 7) / 8) + extra_null_bytes;
    std::unique_ptr<BodoBuffer> null_bitmap_buffer =
        AllocateBodoBuffer(n_bytes, Bodo_CTypes::UINT8, pool, std::move(mm));
    // setting all to non-null to avoid unexpected issues
    memset(null_bitmap_buffer->mutable_data(), 0xff, n_bytes);

    // set offsets for boundaries
    offset_t* offsets_ptr = (offset_t*)offsets_buffer->mutable_data();
    offsets_ptr[0] = 0;
    offsets_ptr[length] = string_arr->length;

    return std::make_unique<array_info>(
        bodo_array_type::LIST_STRING, Bodo_CTypes::LIST_STRING, length,
        std::vector<std::shared_ptr<BodoBuffer>>(
            {std::move(offsets_buffer), std::move(null_bitmap_buffer)}),
        std::vector<std::shared_ptr<array_info>>({string_arr}));
}

std::unique_ptr<array_info> alloc_list_string_array(
    int64_t n_lists, int64_t n_strings, int64_t n_chars,
    int64_t extra_null_bytes, bodo::IBufferPool* const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    // allocate string data array
    std::shared_ptr<array_info> data_arr =
        alloc_array(n_strings, n_chars, -1,
                    bodo_array_type::arr_type_enum::STRING, Bodo_CTypes::UINT8,
                    -1, extra_null_bytes, 0, false, false, false, pool, mm);

    return alloc_list_string_array(n_lists, data_arr, extra_null_bytes, pool,
                                   std::move(mm));
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
std::unique_ptr<array_info> alloc_array(
    int64_t length, int64_t n_sub_elems, int64_t n_sub_sub_elems,
    bodo_array_type::arr_type_enum arr_type, Bodo_CTypes::CTypeEnum dtype,
    int64_t array_id, int64_t extra_null_bytes, int64_t num_categories,
    bool is_globally_replicated, bool is_locally_unique, bool is_locally_sorted,
    bodo::IBufferPool* const pool, std::shared_ptr<::arrow::MemoryManager> mm) {
    switch (arr_type) {
        case bodo_array_type::LIST_STRING:
            return alloc_list_string_array(length, n_sub_elems, n_sub_sub_elems,
                                           extra_null_bytes, pool,
                                           std::move(mm));

        case bodo_array_type::STRING:
            return alloc_string_array(dtype, length, n_sub_elems, array_id,
                                      extra_null_bytes, is_globally_replicated,
                                      is_locally_unique, is_locally_sorted,
                                      pool, std::move(mm));

        case bodo_array_type::NULLABLE_INT_BOOL:
            return alloc_nullable_array(length, dtype, extra_null_bytes, pool,
                                        std::move(mm));

        case bodo_array_type::INTERVAL:
            return alloc_interval_array(length, dtype, pool, std::move(mm));

        case bodo_array_type::NUMPY:
            return alloc_numpy(length, dtype, pool, std::move(mm));

        case bodo_array_type::CATEGORICAL:
            return alloc_categorical(length, dtype, num_categories, pool,
                                     std::move(mm));

        case bodo_array_type::DICT:
            return alloc_dict_string_array(length, n_sub_elems, n_sub_sub_elems,
                                           pool, std::move(mm));
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
        farr = create_dict_string_array(dictionary, indices);
    } else {
        int64_t array_id = -1;
        if (earr->arr_type == bodo_array_type::STRING) {
            array_id = earr->array_id;
        }
        farr = alloc_array(earr->length, earr->n_sub_elems(),
                           earr->n_sub_sub_elems(), earr->arr_type, earr->dtype,
                           array_id, extra_null_bytes, earr->num_categories,
                           earr->is_globally_replicated,
                           earr->is_locally_unique, earr->is_locally_sorted);
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

size_t get_expected_bits_per_entry(int8_t arr_type, int8_t c_type) {
    // TODO: Handle nested data structure and categorical data types seperately
    size_t nullable;
    switch (arr_type) {
        case bodo_array_type::DICT:
            return 33;  // Dictionaries are fixed 32 bits + 1 null bit
        case bodo_array_type::NUMPY:
        case bodo_array_type::INTERVAL:
        case bodo_array_type::CATEGORICAL:
            nullable = 0;
            break;
        case bodo_array_type::STRING:
        case bodo_array_type::NULLABLE_INT_BOOL:
        case bodo_array_type::LIST_STRING:
        case bodo_array_type::STRUCT:
        case bodo_array_type::ARRAY_ITEM:
            nullable = 1;
            break;
        default:
            throw std::runtime_error(
                "get_expected_bits_per_entry: Invalid array type!");
    }
    switch (c_type) {
        case Bodo_CTypes::_BOOL:
            return arr_type == bodo_array_type::NUMPY ? 8 : nullable + 1;
        case Bodo_CTypes::INT8:
        case Bodo_CTypes::UINT8:
        case Bodo_CTypes::INT16:
        case Bodo_CTypes::UINT16:
        case Bodo_CTypes::INT32:
        case Bodo_CTypes::UINT32:
        case Bodo_CTypes::INT64:
        case Bodo_CTypes::UINT64:
        case Bodo_CTypes::INT128:
        case Bodo_CTypes::FLOAT32:
        case Bodo_CTypes::FLOAT64:
        case Bodo_CTypes::DECIMAL:
        case Bodo_CTypes::DATE:
        case Bodo_CTypes::TIME:
        case Bodo_CTypes::DATETIME:
        case Bodo_CTypes::TIMEDELTA:
            return (nullable + numpy_item_size[c_type]) << 3;
        case Bodo_CTypes::STRING:
        case Bodo_CTypes::LIST_STRING:
        case Bodo_CTypes::LIST:
        case Bodo_CTypes::STRUCT:
        case Bodo_CTypes::BINARY:
            return nullable +
                   256;  // 32 bytes estimate for unknown or variable size types
        default:
            throw std::runtime_error(
                "get_expected_bits_per_entry: Invalid C type!");
    }
}

size_t get_row_bytes(const std::vector<int8_t>& arr_array_types,
                     const std::vector<int8_t>& arr_c_types) {
    assert(arr_array_types.size() == arr_c_types.size());
    size_t row_bits = 0;
    for (size_t i = 0; i < arr_array_types.size(); i++) {
        row_bits +=
            get_expected_bits_per_entry(arr_array_types[i], arr_c_types[i]);
    }
    return (row_bits + 7) >> 3;
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

std::tuple<std::vector<int8_t>, std::vector<int8_t>>
get_dtypes_arr_types_from_table(const std::shared_ptr<table_info>& table) {
    size_t n_cols = table->columns.size();
    std::vector<int8_t> arr_c_types(n_cols);
    std::vector<int8_t> arr_array_types(n_cols);
    for (size_t i = 0; i < n_cols; i++) {
        arr_c_types[i] = table->columns[i]->dtype;
        arr_array_types[i] = table->columns[i]->arr_type;
    }
    return std::make_tuple(arr_c_types, arr_array_types);
}

// get memory alloc/free info from _meminfo.h
size_t get_stats_alloc() { return NRT_MemSys_get_stats_alloc(); }
size_t get_stats_free() { return NRT_MemSys_get_stats_free(); }
size_t get_stats_mi_alloc() { return NRT_MemSys_get_stats_mi_alloc(); }
size_t get_stats_mi_free() { return NRT_MemSys_get_stats_mi_free(); }

// Dictionary utilities

/**
 * @brief Generate a new local id for a dictionary. These
 * can be used to identify if dictionaries are "equivalent"
 * because they share an id. Other than ==, a particular
 * id has no significance.
 *
 * @param length The length of the dictionary being assigned
 * the id. All dictionaries of length 0 should get the same
 * id.
 * @return int64_t The new id that is generated.
 */
static int64_t generate_array_id_state(int64_t length) {
    static int64_t id_counter = 1;
    if (length == 0) {
        // Ensure we can identify all length 0 dictionaries
        // and that all can be unified without transposing.
        return 0;
    } else {
        return id_counter++;
    }
}

int64_t generate_array_id(int64_t length) {
    return generate_array_id_state(length);
}

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
    SetAttrStringFromPyInit(m, lead_lag);
    SetAttrStringFromPyInit(m, crypto_funcs);
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
    SetAttrStringFromPyInit(m, memory_budget_cpp);
    SetAttrStringFromPyInit(m, stream_join_cpp);
    SetAttrStringFromPyInit(m, stream_groupby_cpp);
    SetAttrStringFromPyInit(m, stream_dict_encoding_cpp);
    SetAttrStringFromPyInit(m, table_builder_cpp);

#ifdef IS_TESTING
    SetAttrStringFromPyInit(m, test_cpp);
#endif

    SetAttrStringFromPyInit(m, listagg);

    return m;
}

} /* extern "C" */
