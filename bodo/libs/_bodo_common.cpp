#include "_bodo_common.h"
#include "_distributed.h"

std::vector<size_t> numpy_item_size(Bodo_CTypes::_numtypes);

void bodo_common_init() {
    static bool initialized = false;
    if (initialized) {
        return;
    }
    initialized = true;

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
    numpy_item_size[Bodo_CTypes::DATE] = sizeof(int64_t);
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
    // initalize memory alloc/tracking system in _meminfo.h
    NRT_MemSys_init();
}

Bodo_CTypes::CTypeEnum arrow_to_bodo_type(
    std::shared_ptr<arrow::DataType> type) {
    switch (type->id()) {
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
        // TODO Date, Datetime, Timedelta, String, Bool
        default: {
            throw std::runtime_error("arrow_to_bodo_type : Unsupported type " +
                                     type->ToString());
        }
    }
}

array_info& array_info::operator=(array_info&& other) noexcept {
    if (this != &other) {
        // delete this array's original data
        decref_array(this);

        // copy the other array's pointers into this array_info
        this->length = other.length;
        this->arr_type = other.arr_type;
        this->dtype = other.dtype;
        this->n_sub_elems = other.n_sub_elems;
        this->n_sub_sub_elems = other.n_sub_sub_elems;
        this->data1 = other.data1;
        this->data2 = other.data2;
        this->data3 = other.data3;
        this->null_bitmask = other.null_bitmask;
        this->sub_null_bitmask = other.sub_null_bitmask;
        this->meminfo = other.meminfo;
        this->meminfo_bitmask = other.meminfo_bitmask;
        this->array = other.array;
        this->precision = other.precision;
        this->scale = other.scale;
        this->num_categories = other.num_categories;
        this->has_global_dictionary = other.has_global_dictionary;
        this->has_deduped_local_dictionary = other.has_deduped_local_dictionary;
        this->has_sorted_dictionary = other.has_sorted_dictionary;
        this->info1 = other.info1;
        this->info2 = other.info2;

        // reset the other array_info's pointers
        other.data1 = nullptr;
        other.data2 = nullptr;
        other.data3 = nullptr;
        other.null_bitmask = nullptr;
        other.sub_null_bitmask = nullptr;
        other.meminfo = nullptr;
        other.meminfo_bitmask = nullptr;
        other.array = nullptr;
        other.info1 = nullptr;
        other.info2 = nullptr;
    }
    return *this;
}

array_info* alloc_numpy(int64_t length, Bodo_CTypes::CTypeEnum typ_enum) {
    int64_t size = length * numpy_item_size[typ_enum];
    NRT_MemInfo* meminfo = NRT_MemInfo_alloc_safe_aligned(size, ALIGNMENT);
    char* data = (char*)meminfo->data;
    return new array_info(bodo_array_type::NUMPY, typ_enum, length, -1, -1,
                          data, NULL, NULL, NULL, NULL, meminfo, NULL);
}

array_info* alloc_interval_array(int64_t length,
                                 Bodo_CTypes::CTypeEnum typ_enum) {
    int64_t size = length * numpy_item_size[typ_enum];
    NRT_MemInfo* left_meminfo = NRT_MemInfo_alloc_safe_aligned(size, ALIGNMENT);
    NRT_MemInfo* right_meminfo =
        NRT_MemInfo_alloc_safe_aligned(size, ALIGNMENT);
    char* left_data = (char*)left_meminfo->data;
    char* right_data = (char*)right_meminfo->data;
    return new array_info(bodo_array_type::INTERVAL, typ_enum, length, -1, -1,
                          left_data, right_data, NULL, NULL, NULL, left_meminfo,
                          right_meminfo);
}

array_info* alloc_categorical(int64_t length, Bodo_CTypes::CTypeEnum typ_enum,
                              int64_t num_categories) {
    int64_t size = length * numpy_item_size[typ_enum];
    NRT_MemInfo* meminfo = NRT_MemInfo_alloc_safe_aligned(size, ALIGNMENT);
    char* data = (char*)meminfo->data;
    return new array_info(bodo_array_type::CATEGORICAL, typ_enum, length, -1,
                          -1, data, NULL, NULL, NULL, NULL, meminfo, NULL, NULL,
                          0, 0, num_categories);
}

array_info* alloc_nullable_array(int64_t length,
                                 Bodo_CTypes::CTypeEnum typ_enum,
                                 int64_t extra_null_bytes) {
    int64_t n_bytes = ((length + 7) >> 3) + extra_null_bytes;
    int64_t size = length * numpy_item_size[typ_enum];
    NRT_MemInfo* meminfo = NRT_MemInfo_alloc_safe_aligned(size, ALIGNMENT);
    char* data = (char*)meminfo->data;
    NRT_MemInfo* meminfo_bitmask =
        NRT_MemInfo_alloc_safe_aligned(n_bytes * sizeof(uint8_t), ALIGNMENT);
    char* null_bitmap = (char*)meminfo_bitmask->data;
    return new array_info(bodo_array_type::NULLABLE_INT_BOOL, typ_enum, length,
                          -1, -1, data, NULL, NULL, null_bitmap, NULL, meminfo,
                          meminfo_bitmask);
}

array_info* alloc_nullable_array_no_nulls(int64_t length,
                                          Bodo_CTypes::CTypeEnum typ_enum,
                                          int64_t extra_null_bytes) {
    // Same as alloc_nullable_array but we set the null_bitmask
    // such that there are no null values in the output.
    // Useful for cases like allocating indices array of dictionary-encoded
    // string arrays such as input_file_name column where nulls are not possible
    array_info* arr = alloc_nullable_array(length, typ_enum, extra_null_bytes);
    size_t n_bytes = ((length + 7) >> 3) + extra_null_bytes;
    memset(arr->null_bitmask, 0xff, n_bytes);  // null not possible
    return arr;
}

array_info* alloc_nullable_array_all_nulls(int64_t length,
                                           Bodo_CTypes::CTypeEnum typ_enum,
                                           int64_t extra_null_bytes) {
    // Same as alloc_nullable_array but we set the null_bitmask
    // such that all values are null values in the output.
    // Useful for cases like the iceberg void transform.
    array_info* arr = alloc_nullable_array(length, typ_enum, extra_null_bytes);
    size_t n_bytes = ((length + 7) >> 3) + extra_null_bytes;
    memset(arr->null_bitmask, 0x00, n_bytes);  // all nulls
    return arr;
}

array_info* alloc_string_array(int64_t length, int64_t n_chars,
                               int64_t extra_null_bytes) {
    // allocate underlying array(item) data array
    array_info* data_arr = alloc_array(
        length, n_chars, -1, bodo_array_type::arr_type_enum::ARRAY_ITEM,
        Bodo_CTypes::UINT8, extra_null_bytes, 0);

    NRT_MemInfo* out_meminfo = data_arr->meminfo;
    array_item_arr_numpy_payload* payload =
        (array_item_arr_numpy_payload*)(out_meminfo->data);

    array_info* out_arr = new array_info(
        bodo_array_type::STRING, Bodo_CTypes::STRING, length, n_chars, -1,
        payload->data.data, (char*)payload->offsets.data, NULL,
        (char*)payload->null_bitmap.data, NULL, out_meminfo, NULL);
    delete data_arr;
    return out_arr;
}

array_info* alloc_dict_string_array(int64_t length, int64_t n_keys,
                                    int64_t n_chars_keys,
                                    bool has_global_dictionary,
                                    bool has_deduped_local_dictionary) {
    // dictionary
    array_info* dict_data_arr = alloc_string_array(n_keys, n_chars_keys, 0);
    // indices
    array_info* indices_data_arr =
        alloc_nullable_array(length, Bodo_CTypes::INT32, 0);

    return new array_info(
        bodo_array_type::DICT, Bodo_CTypes::CTypeEnum::STRING, length, -1, -1,
        NULL, NULL, NULL, indices_data_arr->null_bitmask, NULL, NULL, NULL,
        NULL, 0, 0, 0, has_global_dictionary, has_deduped_local_dictionary,
        false, dict_data_arr, indices_data_arr);
}

array_info* create_string_array(std::vector<uint8_t> const& null_bitmap,
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
    array_info* out_col = alloc_string_array(len, nb_char, extra_bytes);
    // update string array payload to reflect change
    char* data_o = out_col->data1;
    offset_t* offsets_o = (offset_t*)out_col->data2;
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

array_info* create_list_string_array(
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
            for (auto& e_str : e_list) nb_char += e_str.first.size();
        }
        iter++;
    }
    // Allocation needs to be done through
    // alloc_list_string_array, which allocates with meminfos
    // and same data structs that Python uses. We need to
    // re-allocate here because number of strings and chars has
    // been determined here (previous out_col was just an empty
    // dummy allocation).

    array_info* new_out_col =
        alloc_list_string_array(len, nb_string, nb_char, 0);
    offset_t* index_offsets_o = (offset_t*)new_out_col->data3;
    offset_t* data_offsets_o = (offset_t*)new_out_col->data2;
    uint8_t* sub_null_bitmask_o = (uint8_t*)new_out_col->sub_null_bitmask;
    // Writing the list_strings in output
    char* data_o = new_out_col->data1;
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

array_info* create_dict_string_array(array_info* dict_arr,
                                     array_info* indices_arr, size_t length) {
    array_info* out_col = new array_info(
        bodo_array_type::DICT, Bodo_CTypes::CTypeEnum::STRING, length, -1, -1,
        NULL, NULL, NULL, indices_arr->null_bitmask, NULL, NULL, NULL, NULL, 0,
        0, 0, false, false, false, dict_arr, indices_arr);
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
    if (payload->data->refct != -1) payload->data->refct--;
    if (payload->data->refct == 0) NRT_MemInfo_call_dtor(payload->data);

    if (payload->offsets.meminfo->refct != -1)
        payload->offsets.meminfo->refct--;
    if (payload->offsets.meminfo->refct == 0)
        NRT_MemInfo_call_dtor(payload->offsets.meminfo);

    if (payload->null_bitmap.meminfo->refct != -1)
        payload->null_bitmap.meminfo->refct--;
    if (payload->null_bitmap.meminfo->refct == 0)
        NRT_MemInfo_call_dtor(payload->null_bitmap.meminfo);
}

array_info* alloc_list_string_array(int64_t n_lists, array_info* string_arr,
                                    int64_t extra_null_bytes) {
    int64_t n_strings = string_arr->length;
    int64_t n_chars = string_arr->n_sub_elems;
    NRT_MemInfo* meminfo_string_array = string_arr->meminfo;
    delete string_arr;

    // allocate array(item) array payload
    NRT_MemInfo* meminfo_array_item = NRT_MemInfo_alloc_dtor_safe(
        sizeof(array_item_arr_payload), (NRT_dtor_function)dtor_array_item_arr);
    array_item_arr_payload* payload =
        (array_item_arr_payload*)(meminfo_array_item->data);
    payload->n_arrays = n_lists;
    payload->offsets = allocate_numpy_payload(n_lists + 1, Bodo_CType_offset);
    int64_t n_bytes = (int64_t)((n_lists + 7) / 8) + extra_null_bytes;
    payload->null_bitmap =
        allocate_numpy_payload(n_bytes, Bodo_CTypes::CTypeEnum::UINT8);
    payload->data = meminfo_string_array;
    // setting all to non-null to avoid unexpected issues
    memset(payload->null_bitmap.data, 0xff, n_bytes);

    array_item_arr_numpy_payload* sub_payload =
        (array_item_arr_numpy_payload*)(meminfo_string_array->data);

    return new array_info(
        bodo_array_type::LIST_STRING, Bodo_CTypes::LIST_STRING, n_lists,
        n_strings, n_chars, (char*)sub_payload->data.data,
        (char*)sub_payload->offsets.data, (char*)payload->offsets.data,
        (char*)payload->null_bitmap.data, (char*)sub_payload->null_bitmap.data,
        meminfo_array_item, NULL);
}

array_info* alloc_list_string_array(int64_t n_lists, int64_t n_strings,
                                    int64_t n_chars, int64_t extra_null_bytes) {
    // allocate string data array
    array_info* data_arr = alloc_array(
        n_strings, n_chars, -1, bodo_array_type::arr_type_enum::ARRAY_ITEM,
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
    if (arr.meminfo->refct != -1) arr.meminfo->refct--;
    if (arr.meminfo->refct == 0) NRT_MemInfo_call_dtor(arr.meminfo);
}

/**
 * @brief destrcutor for array(item) array meminfo. Decrefs the underlying data,
 * offsets and null_bitmap arrays.
 *
 * @param payload array(item) array meminfo payload
 * @param size payload size (ignored)
 * @param in extra info (ignored)
 */
void dtor_array_item_array(array_item_arr_numpy_payload* payload, int64_t size,
                           void* in) {
    if (payload->data.meminfo->refct != -1) payload->data.meminfo->refct--;
    if (payload->data.meminfo->refct == 0)
        NRT_MemInfo_call_dtor(payload->data.meminfo);

    if (payload->offsets.meminfo->refct != -1)
        payload->offsets.meminfo->refct--;
    if (payload->offsets.meminfo->refct == 0)
        NRT_MemInfo_call_dtor(payload->offsets.meminfo);

    if (payload->null_bitmap.meminfo->refct != -1)
        payload->null_bitmap.meminfo->refct--;
    if (payload->null_bitmap.meminfo->refct == 0)
        NRT_MemInfo_call_dtor(payload->null_bitmap.meminfo);
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
 * @brief allocate an array(item) array and wrap in an array_info*
 * TODO: generalize beyond Numpy arrays
 *
 * @param n_arrays number of array elements
 * @param n_total_items total number of data elements
 * @param dtype dtype of data elements
 * @return array_info*
 */
array_info* alloc_array_item(int64_t n_arrays, int64_t n_total_items,
                             Bodo_CTypes::CTypeEnum dtype,
                             int64_t extra_null_bytes) {
    // allocate payload
    NRT_MemInfo* meminfo_array_item =
        NRT_MemInfo_alloc_dtor_safe(sizeof(array_item_arr_numpy_payload),
                                    (NRT_dtor_function)dtor_array_item_array);
    array_item_arr_numpy_payload* payload =
        (array_item_arr_numpy_payload*)(meminfo_array_item->data);

    payload->n_arrays = n_arrays;

    // allocate data array
    // TODO: support non-numpy data
    payload->data = allocate_numpy_payload(n_total_items, dtype);
    // TODO: support 64-bit offsets case
    payload->offsets = allocate_numpy_payload(n_arrays + 1, Bodo_CType_offset);
    int64_t n_bytes = (int64_t)((n_arrays + 7) / 8) + extra_null_bytes;
    payload->null_bitmap =
        allocate_numpy_payload(n_bytes, Bodo_CTypes::CTypeEnum::UINT8);
    // setting all to non-null to avoid unexpected issues
    memset(payload->null_bitmap.data, 0xff, n_bytes);

    // set offsets for boundaries
    offset_t* offsets_ptr = (offset_t*)payload->offsets.data;
    offsets_ptr[0] = 0;
    offsets_ptr[n_arrays] = n_total_items;

    return new array_info(bodo_array_type::ARRAY_ITEM, dtype, n_arrays,
                          n_total_items, -1, NULL, NULL, NULL, NULL, NULL,
                          meminfo_array_item, NULL);
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
array_info* alloc_array(int64_t length, int64_t n_sub_elems,
                        int64_t n_sub_sub_elems,
                        bodo_array_type::arr_type_enum arr_type,
                        Bodo_CTypes::CTypeEnum dtype, int64_t extra_null_bytes,
                        int64_t num_categories) {
    if (arr_type == bodo_array_type::LIST_STRING)
        return alloc_list_string_array(length, n_sub_elems, n_sub_sub_elems,
                                       extra_null_bytes);

    if (arr_type == bodo_array_type::STRING)
        return alloc_string_array(length, n_sub_elems, extra_null_bytes);

    if (arr_type == bodo_array_type::NULLABLE_INT_BOOL)
        return alloc_nullable_array(length, dtype, extra_null_bytes);

    if (arr_type == bodo_array_type::INTERVAL)
        return alloc_interval_array(length, dtype);

    if (arr_type == bodo_array_type::NUMPY) return alloc_numpy(length, dtype);

    if (arr_type == bodo_array_type::CATEGORICAL)
        return alloc_categorical(length, dtype, num_categories);

    if (arr_type == bodo_array_type::ARRAY_ITEM)
        return alloc_array_item(length, n_sub_elems, dtype, extra_null_bytes);

    if (arr_type == bodo_array_type::DICT)
        return alloc_dict_string_array(length, n_sub_elems, n_sub_sub_elems,
                                       false, false);

    Bodo_PyErr_SetString(PyExc_RuntimeError, "Type not covered in alloc_array");
    return nullptr;
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
        Bodo_CTypes::CTypeEnum bodo_typ = arrow_to_bodo_type(prim_arr->type());
        int64_t siz_typ = numpy_item_size[bodo_typ];
        int64_t siz_primitive_data = siz_typ * n_rows;
        return siz_null_bitmap + siz_primitive_data;
    }
}

int64_t array_memory_size(array_info* earr) {
    if (earr->arr_type == bodo_array_type::NUMPY ||
        earr->arr_type == bodo_array_type::CATEGORICAL) {
        uint64_t siztype = numpy_item_size[earr->dtype];
        return siztype * earr->length;
    }
    if (earr->arr_type == bodo_array_type::NULLABLE_INT_BOOL ||
        earr->arr_type == bodo_array_type::DICT) {
        if (earr->arr_type == bodo_array_type::DICT)
            // TODO also contribute dictionary size, but note that when
            // table_global_memory_size() calls this function, it's not supposed
            // to add the size of dictionaries of all ranks
            earr = earr->info2;
        uint64_t siztype = numpy_item_size[earr->dtype];
        int64_t n_bytes = ((earr->length + 7) >> 3);
        return n_bytes + siztype * earr->length;
    }
    if (earr->arr_type == bodo_array_type::STRING) {
        int64_t n_bytes = ((earr->length + 7) >> 3);
        return earr->n_sub_elems + sizeof(offset_t) * (earr->length + 1) +
               n_bytes;
    }
    if (earr->arr_type == bodo_array_type::LIST_STRING) {
        int64_t n_bytes = ((earr->length + 7) >> 3);
        int64_t n_sub_bytes = ((earr->n_sub_elems + 7) >> 3);
        return earr->n_sub_sub_elems +
               sizeof(offset_t) * (earr->n_sub_elems + 1) +
               sizeof(offset_t) * (earr->length + 1) + n_bytes + n_sub_bytes;
    }
    if (earr->arr_type == bodo_array_type::ARROW) {
        return arrow_array_memory_size(earr->array);
    }
    Bodo_PyErr_SetString(PyExc_RuntimeError,
                         "Type not covered in array_memory_size");
    return 0;
}

int64_t table_local_memory_size(table_info* table) {
    int64_t local_size = 0;
    for (auto& arr : table->columns) {
        local_size += array_memory_size(arr);
    }
    return local_size;
}

int64_t table_global_memory_size(table_info* table) {
    int64_t local_size = table_local_memory_size(table);
    int64_t global_size;
    MPI_Allreduce(&local_size, &global_size, 1, MPI_LONG_LONG_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    return global_size;
}

array_info* copy_array(array_info* earr) {
    int64_t extra_null_bytes = 0;
    array_info* farr;
    if (earr->arr_type == bodo_array_type::DICT) {
        array_info* dictionary = copy_array(earr->info1);
        array_info* indices = copy_array(earr->info2);
        farr = new array_info(
            bodo_array_type::DICT, earr->dtype, indices->length, -1, -1, NULL,
            NULL, NULL, indices->null_bitmask, NULL, NULL, NULL, NULL, 0, 0, 0,
            earr->has_global_dictionary, earr->has_deduped_local_dictionary,
            earr->has_sorted_dictionary, dictionary, indices);
    } else {
        farr = alloc_array(earr->length, earr->n_sub_elems,
                           earr->n_sub_sub_elems, earr->arr_type, earr->dtype,
                           extra_null_bytes, earr->num_categories);
    }
    if (earr->arr_type == bodo_array_type::NUMPY ||
        earr->arr_type == bodo_array_type::CATEGORICAL) {
        uint64_t siztype = numpy_item_size[earr->dtype];
        memcpy(farr->data1, earr->data1, siztype * earr->length);
    }
    if (earr->arr_type == bodo_array_type::INTERVAL) {
        uint64_t siztype = numpy_item_size[earr->dtype];
        memcpy(farr->data1, earr->data1, siztype * earr->length);
        memcpy(farr->data2, earr->data2, siztype * earr->length);
    }
    if (earr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        uint64_t siztype = numpy_item_size[earr->dtype];
        memcpy(farr->data1, earr->data1, siztype * earr->length);
        int64_t n_bytes = ((earr->length + 7) >> 3);
        memcpy(farr->null_bitmask, earr->null_bitmask, n_bytes);
    }
    if (earr->arr_type == bodo_array_type::STRING) {
        memcpy(farr->data1, earr->data1, earr->n_sub_elems);
        memcpy(farr->data2, earr->data2, sizeof(offset_t) * (earr->length + 1));
        int64_t n_bytes = ((earr->length + 7) >> 3);
        memcpy(farr->null_bitmask, earr->null_bitmask, n_bytes);
    }
    if (earr->arr_type == bodo_array_type::LIST_STRING) {
        memcpy(farr->data1, earr->data1, earr->n_sub_sub_elems);
        memcpy(farr->data2, earr->data2,
               sizeof(offset_t) * (earr->n_sub_elems + 1));
        memcpy(farr->data3, earr->data3, sizeof(offset_t) * (earr->length + 1));
        int64_t n_bytes = ((earr->length + 7) >> 3);
        memcpy(farr->null_bitmask, earr->null_bitmask, n_bytes);
        int64_t n_sub_bytes = ((earr->n_sub_elems + 7) >> 3);
        memcpy(farr->sub_null_bitmask, earr->sub_null_bitmask, n_sub_bytes);
    }
    return farr;
}

/**
 * Free underlying array of array_info pointer and delete the pointer
 */
void delete_info_decref_array(array_info* arr) {
    decref_array(arr);
    delete arr;
}

/**
 * delete table pointer and its column array_info pointers (but not the arrays).
 */
void delete_table(table_info* table) {
    for (array_info* a : table->columns) {
        delete a;
    }
    delete table;
}

/**
 * Free all arrays of a table and delete the table.
 */
void delete_table_decref_arrays(table_info* table) {
    for (array_info* a : table->columns) {
        if (a != NULL) {
            decref_array(a);
            delete a;
        }
    }
    delete table;
}

/**
 * Free an array of a table
 */
void decref_table_array(table_info* table, int arr_no) {
    array_info* a = table->columns[arr_no];
    if (a != NULL) {
        decref_array(a);
    }
}

/**
 * @brief incref all arrays in a table
 *
 * @param table input table
 */
void incref_table_arrays(table_info* table) {
    for (array_info* a : table->columns) {
        if (a != NULL) {
            incref_array(a);
        }
    }
}

/**
 * @brief decref all arrays in a table
 *
 * @param table input table
 */
void decref_table_arrays(table_info* table) {
    for (array_info* a : table->columns) {
        if (a != NULL) {
            decref_array(a);
        }
    }
}

/*
  Decreases refcount and frees array if refcount is zero.
  This is more complicated than one might expect since the structure needs to be
  coherent with the Python side.
  *
  On the Python side, Numba uses this function:
  def _define_nrt_decref(module, atomic_decr):
  from
  https://github.com/numba/numba/blob/ce2139c7dd93127efb04b35f28f4bebc7f44dfd5/numba/core/runtime/nrtdynmod.py#L64
  *
  On the C++ side, we cannot call this function directly. But we have to
  manually call the destructor. We cannot do operations directly on the
  arrays. For example deallocating data1 and then reallocating it is deadly,
  since the Python side tries to deallocate again the data1 (double free error)
  and leaves the current allocation intact.
  *
  The entries to be called are the NRT_MemInfo*
  An array_info contains two NRT_MemInfo*: meminfo and meminfo_bitmask
  Thus we have two calls for destructors when they are not NULL.
 */
void decref_array(array_info* arr) {
    // dictionary-encoded string array uses nested infos
    if (arr->arr_type == bodo_array_type::DICT) {
        if (arr->info1 != nullptr) decref_array(arr->info1);
        if (arr->info2 != nullptr) decref_array(arr->info2);
        return;
    }

    if (arr->meminfo != NULL && arr->meminfo->refct != -1) {
        arr->meminfo->refct--;
        if (arr->meminfo->refct == 0) NRT_MemInfo_call_dtor(arr->meminfo);
    }
    if (arr->meminfo_bitmask != NULL && arr->meminfo_bitmask->refct != -1) {
        arr->meminfo_bitmask->refct--;
        if (arr->meminfo_bitmask->refct == 0)
            NRT_MemInfo_call_dtor(arr->meminfo_bitmask);
    }
}

void incref_array(array_info* arr) {
    // dictionary-encoded string array uses nested infos
    if (arr->arr_type == bodo_array_type::DICT) {
        if (arr->info1 != nullptr) incref_array(arr->info1);
        if (arr->info2 != nullptr) incref_array(arr->info2);
        return;
    }

    if (arr->meminfo != NULL && arr->meminfo->refct != -1)
        arr->meminfo->refct++;
    if (arr->meminfo_bitmask != NULL && arr->meminfo_bitmask->refct != -1)
        arr->meminfo_bitmask->refct++;
}

// get memory alloc/free info from _meminfo.h
size_t get_stats_alloc() { return NRT_MemSys_get_stats_alloc(); }
size_t get_stats_free() { return NRT_MemSys_get_stats_free(); }
size_t get_stats_mi_alloc() { return NRT_MemSys_get_stats_mi_alloc(); }
size_t get_stats_mi_free() { return NRT_MemSys_get_stats_mi_free(); }

/**
 * Given an Arrow array, populate array of lengths and array of array_info*
 * with the data from the array and all of its descendant arrays.
 * This is called recursively, and will create one array_info for each
 * individual buffer (offsets, null_bitmaps, data).
 * @param array: The input Arrow array
 * @param lengths: The lengths array to fill
 * @param infos: The array_info* array to fill
 * @param lengths_pos: current position in lengths (this is passed by reference
 *        because we are traversing an arbitrary tree of arrays. Some arrays
 *        (like StructArray) have multiple children and upon returning from
 *        one of the child subtrees we need to know the current position in
 *        lengths.
 * @param infos_pos: same as lengths_pos but tracks the position in infos array
 */
void nested_array_to_c(std::shared_ptr<arrow::Array> array, int64_t* lengths,
                       array_info** infos, int64_t& lengths_pos,
                       int64_t& infos_pos) {
    if (array->type_id() == arrow::Type::LARGE_LIST) {
        // TODO should print error or warning if OFFSET_BITWIDTH==32
        std::shared_ptr<arrow::LargeListArray> list_array =
            std::dynamic_pointer_cast<arrow::LargeListArray>(array);
        lengths[lengths_pos++] = list_array->length();

        // allocate output arrays and copy data
        array_info* offsets = alloc_array(list_array->length() + 1, -1, -1,
                                          bodo_array_type::arr_type_enum::NUMPY,
                                          Bodo_CType_offset, 0, 0);
        int64_t n_null_bytes = (list_array->length() + 7) >> 3;
        array_info* nulls = alloc_array(n_null_bytes, -1, -1,
                                        bodo_array_type::arr_type_enum::NUMPY,
                                        Bodo_CTypes::UINT8, 0, 0);

        // NOTE: this should just do a memcpy if the bidwidths of input and
        // output match
        std::copy_n((int64_t*)(list_array->value_offsets()->data()),
                    list_array->length() + 1, (offset_t*)offsets->data1);
        memset(nulls->data1, 0, n_null_bytes);
        for (int64_t i = 0; i < list_array->length(); i++) {
            if (!list_array->IsNull(i))
                SetBitTo((uint8_t*)nulls->data1, i, true);
        }

        infos[infos_pos++] = offsets;
        infos[infos_pos++] = nulls;
        nested_array_to_c(list_array->values(), lengths, infos, lengths_pos,
                          infos_pos);
    } else if (array->type_id() == arrow::Type::LIST) {
        std::shared_ptr<arrow::ListArray> list_array =
            std::dynamic_pointer_cast<arrow::ListArray>(array);
        lengths[lengths_pos++] = list_array->length();

        // allocate output arrays and copy data
        array_info* offsets = alloc_array(list_array->length() + 1, -1, -1,
                                          bodo_array_type::arr_type_enum::NUMPY,
                                          Bodo_CType_offset, 0, 0);
        int64_t n_null_bytes = (list_array->length() + 7) >> 3;
        array_info* nulls = alloc_array(n_null_bytes, -1, -1,
                                        bodo_array_type::arr_type_enum::NUMPY,
                                        Bodo_CTypes::UINT8, 0, 0);

        std::copy_n((int32_t*)(list_array->value_offsets()->data()),
                    list_array->length() + 1, (offset_t*)offsets->data1);
        memset(nulls->data1, 0, n_null_bytes);
        for (int64_t i = 0; i < list_array->length(); i++) {
            if (!list_array->IsNull(i))
                SetBitTo((uint8_t*)nulls->data1, i, true);
        }

        infos[infos_pos++] = offsets;
        infos[infos_pos++] = nulls;
        nested_array_to_c(list_array->values(), lengths, infos, lengths_pos,
                          infos_pos);
    } else if (array->type_id() == arrow::Type::STRUCT) {
        auto struct_array =
            std::dynamic_pointer_cast<arrow::StructArray>(array);
        auto struct_type =
            std::dynamic_pointer_cast<arrow::StructType>(struct_array->type());
        lengths[lengths_pos++] = struct_array->length();

        // allocate output arrays and copy data
        int64_t n_null_bytes = (struct_array->length() + 7) >> 3;
        array_info* nulls = alloc_array(n_null_bytes, -1, -1,
                                        bodo_array_type::arr_type_enum::NUMPY,
                                        Bodo_CTypes::UINT8, 0, 0);
        memset(nulls->data1, 0, n_null_bytes);
        for (int64_t i = 0; i < struct_array->length(); i++) {
            if (!struct_array->IsNull(i))
                SetBitTo((uint8_t*)nulls->data1, i, true);
        }
        infos[infos_pos++] = nulls;
        // Now outputing the fields.
        for (int i = 0; i < struct_type->num_fields();
             i++) {  // each field is an array
            nested_array_to_c(struct_array->field(i), lengths, infos,
                              lengths_pos, infos_pos);
        }
    } else if (array->type_id() == arrow::Type::LARGE_STRING) {
        // TODO should print error or warning if OFFSET_BITWIDTH==32
        auto str_array =
            std::dynamic_pointer_cast<arrow::LargeStringArray>(array);
        lengths[lengths_pos++] = str_array->length();
        int64_t n_strings = str_array->length();
        int64_t n_chars =
            ((int64_t*)str_array->value_offsets()->data())[n_strings];
        array_info* str_arr_info = alloc_string_array(n_strings, n_chars, 0);
        memcpy(str_arr_info->data1, str_array->value_data()->data(),
               sizeof(char) * n_chars);  // data
        // NOTE: this should just do a memcpy if the bidwidths of input and
        // output match
        std::copy_n((int64_t*)(str_array->value_offsets()->data()),
                    n_strings + 1, (offset_t*)str_arr_info->data2);
        int64_t n_null_bytes = (n_strings + 7) >> 3;
        memset(str_arr_info->null_bitmask, 0, n_null_bytes);
        for (int64_t i = 0; i < n_strings; i++) {
            if (!str_array->IsNull(i))
                SetBitTo((uint8_t*)str_arr_info->null_bitmask, i, true);
        }
        infos[infos_pos++] = str_arr_info;
    } else if (array->type_id() == arrow::Type::STRING) {
        auto str_array = std::dynamic_pointer_cast<arrow::StringArray>(array);
        lengths[lengths_pos++] = str_array->length();
        int64_t n_strings = str_array->length();
        int64_t n_chars =
            ((uint32_t*)str_array->value_offsets()->data())[n_strings];
        array_info* str_arr_info = alloc_string_array(n_strings, n_chars, 0);
        memcpy(str_arr_info->data1, str_array->value_data()->data(),
               sizeof(char) * n_chars);  // data
        std::copy_n((int32_t*)(str_array->value_offsets()->data()),
                    n_strings + 1, (offset_t*)str_arr_info->data2);
        int64_t n_null_bytes = (n_strings + 7) >> 3;
        memset(str_arr_info->null_bitmask, 0, n_null_bytes);
        for (int64_t i = 0; i < n_strings; i++) {
            if (!str_array->IsNull(i))
                SetBitTo((uint8_t*)str_arr_info->null_bitmask, i, true);
        }
        infos[infos_pos++] = str_arr_info;
    } else {
        auto primitive_array =
            std::dynamic_pointer_cast<arrow::PrimitiveArray>(array);
        lengths[lengths_pos++] = primitive_array->length();

        Bodo_CTypes::CTypeEnum dtype =
            arrow_to_bodo_type(primitive_array->type());

        // allocate output arrays and copy data
        array_info* data =
            alloc_array(primitive_array->length(), -1, -1,
                        bodo_array_type::arr_type_enum::NUMPY, dtype, 0, 0);
        int64_t n_null_bytes = (primitive_array->length() + 7) >> 3;
        array_info* nulls = alloc_array(n_null_bytes, -1, -1,
                                        bodo_array_type::arr_type_enum::NUMPY,
                                        Bodo_CTypes::UINT8, 0, 0);
        uint64_t siztype = numpy_item_size[dtype];
        memcpy(data->data1, primitive_array->values()->data(),
               siztype * primitive_array->length());
        memset(nulls->data1, 0, n_null_bytes);
        std::vector<char> vectNaN = RetrieveNaNentry(dtype);
        for (int64_t i = 0; i < primitive_array->length(); i++) {
            if (!primitive_array->IsNull(i))
                SetBitTo((uint8_t*)nulls->data1, i, true);
            else
                memcpy(data->data1 + siztype * i, vectNaN.data(), siztype);
        }
        infos[infos_pos++] = nulls;
        infos[infos_pos++] = data;
    }
}
