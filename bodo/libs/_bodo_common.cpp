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

array_info& array_info::operator=(array_info&& other) noexcept {
    if (this != &other) {
        // delete this array's original data
        decref_array(this);

        // copy the other array's pointers into this array_info
        this->length = other.length;
        this->arr_type = other.arr_type;
        this->dtype = other.dtype;
        this->meminfos = std::move(other.meminfos);
        this->array = other.array;
        this->precision = other.precision;
        this->scale = other.scale;
        this->num_categories = other.num_categories;
        this->has_global_dictionary = other.has_global_dictionary;
        this->has_deduped_local_dictionary = other.has_deduped_local_dictionary;
        this->has_sorted_dictionary = other.has_sorted_dictionary;
        this->child_arrays = std::move(other.child_arrays);

        // reset the other array_info's pointers
        other.array = nullptr;
    }
    return *this;
}

std::shared_ptr<arrow::Array> array_info::to_arrow() const {
    std::shared_ptr<arrow::Array> arrow_arr;
    arrow::TimeUnit::type time_unit = arrow::TimeUnit::NANO;
    bodo_array_to_arrow(::arrow::default_memory_pool(), this, &arrow_arr,
                        false /*convert_timedelta_to_int64*/, "", time_unit,
                        false /*downcast_time_ns_to_us*/);
    return arrow_arr;
}

array_info* alloc_numpy(int64_t length, Bodo_CTypes::CTypeEnum typ_enum) {
    int64_t size = length * numpy_item_size[typ_enum];
    NRT_MemInfo* meminfo = NRT_MemInfo_alloc_safe_aligned(size, ALIGNMENT);
    return new array_info(bodo_array_type::NUMPY, typ_enum, length, {meminfo});
}

array_info* alloc_interval_array(int64_t length,
                                 Bodo_CTypes::CTypeEnum typ_enum) {
    int64_t size = length * numpy_item_size[typ_enum];
    NRT_MemInfo* left_meminfo = NRT_MemInfo_alloc_safe_aligned(size, ALIGNMENT);
    NRT_MemInfo* right_meminfo =
        NRT_MemInfo_alloc_safe_aligned(size, ALIGNMENT);
    return new array_info(bodo_array_type::INTERVAL, typ_enum, length,
                          {left_meminfo, right_meminfo});
}

array_info* alloc_categorical(int64_t length, Bodo_CTypes::CTypeEnum typ_enum,
                              int64_t num_categories) {
    int64_t size = length * numpy_item_size[typ_enum];
    NRT_MemInfo* meminfo = NRT_MemInfo_alloc_safe_aligned(size, ALIGNMENT);
    return new array_info(bodo_array_type::CATEGORICAL, typ_enum, length,
                          {meminfo}, {}, NULL, 0, 0, num_categories);
}

array_info* alloc_nullable_array(int64_t length,
                                 Bodo_CTypes::CTypeEnum typ_enum,
                                 int64_t extra_null_bytes) {
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
    NRT_MemInfo* meminfo = NRT_MemInfo_alloc_safe_aligned(size, ALIGNMENT);
    NRT_MemInfo* meminfo_bitmask =
        NRT_MemInfo_alloc_safe_aligned(n_bytes * sizeof(uint8_t), ALIGNMENT);
    return new array_info(bodo_array_type::NULLABLE_INT_BOOL, typ_enum, length,
                          {meminfo, meminfo_bitmask});
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
    memset(arr->null_bitmask(), 0xff, n_bytes);  // null not possible
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
    memset(arr->null_bitmask(), 0x00, n_bytes);  // all nulls
    return arr;
}

array_info* alloc_string_array(int64_t length, int64_t n_chars,
                               int64_t extra_null_bytes) {
    // allocate data/offsets/null_bitmap arrays
    MemInfo* data_meminfo =
        allocate_numpy_payload(n_chars, Bodo_CTypes::UINT8).meminfo;
    MemInfo* offsets_meminfo =
        allocate_numpy_payload(length + 1, Bodo_CType_offset).meminfo;
    int64_t n_bytes = (int64_t)((length + 7) / 8) + extra_null_bytes;
    MemInfo* null_bitmap_meminfo =
        allocate_numpy_payload(n_bytes, Bodo_CTypes::UINT8).meminfo;
    // setting all to non-null to avoid unexpected issues
    memset(null_bitmap_meminfo->data, 0xff, n_bytes);

    // set offsets for boundaries
    offset_t* offsets_ptr = (offset_t*)offsets_meminfo->data;
    offsets_ptr[0] = 0;
    offsets_ptr[length] = n_chars;

    return new array_info(bodo_array_type::STRING, Bodo_CTypes::STRING, length,
                          {data_meminfo, offsets_meminfo, null_bitmap_meminfo});
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

    return new array_info(bodo_array_type::DICT, Bodo_CTypes::CTypeEnum::STRING,
                          length, {}, {dict_data_arr, indices_data_arr}, NULL,
                          0, 0, 0, has_global_dictionary,
                          has_deduped_local_dictionary, false);
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

    array_info* new_out_col =
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

array_info* create_dict_string_array(array_info* dict_arr,
                                     array_info* indices_arr,
                                     bool has_global_dictionary,
                                     bool has_deduped_local_dictionary,
                                     bool has_sorted_dictionary) {
    array_info* out_col =
        new array_info(bodo_array_type::DICT, Bodo_CTypes::CTypeEnum::STRING,
                       indices_arr->length, {}, {dict_arr, indices_arr}, NULL,
                       0, 0, 0, has_global_dictionary,
                       has_deduped_local_dictionary, has_sorted_dictionary);
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

array_info* alloc_list_string_array(int64_t length, array_info* string_arr,
                                    int64_t extra_null_bytes) {
    MemInfo* offsets_meminfo =
        allocate_numpy_payload(length + 1, Bodo_CType_offset).meminfo;
    int64_t n_bytes = (int64_t)((length + 7) / 8) + extra_null_bytes;
    MemInfo* null_bitmap_meminfo =
        allocate_numpy_payload(n_bytes, Bodo_CTypes::UINT8).meminfo;
    // setting all to non-null to avoid unexpected issues
    memset(null_bitmap_meminfo->data, 0xff, n_bytes);

    // set offsets for boundaries
    offset_t* offsets_ptr = (offset_t*)offsets_meminfo->data;
    offsets_ptr[0] = 0;
    offsets_ptr[length] = string_arr->length;

    return new array_info(bodo_array_type::LIST_STRING,
                          Bodo_CTypes::LIST_STRING, length,
                          {offsets_meminfo, null_bitmap_meminfo}, {string_arr});
}

array_info* alloc_list_string_array(int64_t n_lists, int64_t n_strings,
                                    int64_t n_chars, int64_t extra_null_bytes) {
    // allocate string data array
    array_info* data_arr = alloc_array(n_strings, n_chars, -1,
                                       bodo_array_type::arr_type_enum::STRING,
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
array_info* alloc_array(int64_t length, int64_t n_sub_elems,
                        int64_t n_sub_sub_elems,
                        bodo_array_type::arr_type_enum arr_type,
                        Bodo_CTypes::CTypeEnum dtype, int64_t extra_null_bytes,
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

int64_t array_memory_size(array_info* earr) {
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
    if (earr->arr_type == bodo_array_type::ARROW ||
        earr->arr_type == bodo_array_type::ARRAY_ITEM) {
        return arrow_array_memory_size(earr->to_arrow());
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
        array_info* dictionary = copy_array(earr->child_arrays[0]);
        array_info* indices = copy_array(earr->child_arrays[1]);
        farr = new array_info(
            bodo_array_type::DICT, earr->dtype, indices->length, {},
            {dictionary, indices}, NULL, 0, 0, 0, earr->has_global_dictionary,
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
    for (array_info* inner_arr : arr->child_arrays) {
        if (inner_arr != nullptr) {
            decref_array(inner_arr);
        }
    }

    for (MemInfo* meminfo : arr->meminfos) {
        decref_meminfo(meminfo);
    }
}

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

void incref_array(const array_info* arr) {
    for (array_info* inner_arr : arr->child_arrays) {
        if (inner_arr != nullptr) {
            incref_array(inner_arr);
        }
    }

    for (MemInfo* meminfo : arr->meminfos) {
        incref_meminfo(meminfo);
    }
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
                    list_array->length() + 1, (offset_t*)offsets->data1());
        memset(nulls->data1(), 0, n_null_bytes);
        for (int64_t i = 0; i < list_array->length(); i++) {
            if (!list_array->IsNull(i))
                SetBitTo((uint8_t*)nulls->data1(), i, true);
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
                    list_array->length() + 1, (offset_t*)offsets->data1());
        memset(nulls->data1(), 0, n_null_bytes);
        for (int64_t i = 0; i < list_array->length(); i++) {
            if (!list_array->IsNull(i))
                SetBitTo((uint8_t*)nulls->data1(), i, true);
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
        memset(nulls->data1(), 0, n_null_bytes);
        for (int64_t i = 0; i < struct_array->length(); i++) {
            if (!struct_array->IsNull(i))
                SetBitTo((uint8_t*)nulls->data1(), i, true);
        }
        infos[infos_pos++] = nulls;
        // Now outputting the fields.
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
        memcpy(str_arr_info->data1(), str_array->value_data()->data(),
               sizeof(char) * n_chars);  // data
        // NOTE: this should just do a memcpy if the bidwidths of input and
        // output match
        std::copy_n((int64_t*)(str_array->value_offsets()->data()),
                    n_strings + 1, (offset_t*)str_arr_info->data2());
        int64_t n_null_bytes = (n_strings + 7) >> 3;
        memset(str_arr_info->null_bitmask(), 0, n_null_bytes);
        for (int64_t i = 0; i < n_strings; i++) {
            if (!str_array->IsNull(i))
                SetBitTo((uint8_t*)str_arr_info->null_bitmask(), i, true);
        }
        infos[infos_pos++] = str_arr_info;
    } else if (array->type_id() == arrow::Type::STRING) {
        auto str_array = std::dynamic_pointer_cast<arrow::StringArray>(array);
        lengths[lengths_pos++] = str_array->length();
        int64_t n_strings = str_array->length();
        int64_t n_chars =
            ((uint32_t*)str_array->value_offsets()->data())[n_strings];
        array_info* str_arr_info = alloc_string_array(n_strings, n_chars, 0);
        memcpy(str_arr_info->data1(), str_array->value_data()->data(),
               sizeof(char) * n_chars);  // data
        std::copy_n((int32_t*)(str_array->value_offsets()->data()),
                    n_strings + 1, (offset_t*)str_arr_info->data2());
        int64_t n_null_bytes = (n_strings + 7) >> 3;
        memset(str_arr_info->null_bitmask(), 0, n_null_bytes);
        for (int64_t i = 0; i < n_strings; i++) {
            if (!str_array->IsNull(i))
                SetBitTo((uint8_t*)str_arr_info->null_bitmask(), i, true);
        }
        infos[infos_pos++] = str_arr_info;
    } else {
        auto primitive_array =
            std::dynamic_pointer_cast<arrow::PrimitiveArray>(array);
        lengths[lengths_pos++] = primitive_array->length();

        Bodo_CTypes::CTypeEnum dtype =
            arrow_to_bodo_type(primitive_array->type()->id());
        uint64_t siztype = numpy_item_size[dtype];

        // allocate output arrays and copy data
        size_t buffer_size;
        size_t arr_len;
        if (primitive_array->type()->id() == arrow::Type::BOOL) {
            dtype = Bodo_CTypes::UINT8;
            buffer_size = (primitive_array->length() + 7) >> 3;
            arr_len = buffer_size;
        } else {
            buffer_size = primitive_array->length() * siztype;
            arr_len = primitive_array->length();
        }
        array_info* data =
            alloc_array(arr_len, -1, -1, bodo_array_type::arr_type_enum::NUMPY,
                        dtype, 0, 0);
        int64_t n_null_bytes = (primitive_array->length() + 7) >> 3;
        array_info* nulls = alloc_array(n_null_bytes, -1, -1,
                                        bodo_array_type::arr_type_enum::NUMPY,
                                        Bodo_CTypes::UINT8, 0, 0);
        memcpy(data->data1(), primitive_array->values()->data(), buffer_size);
        memset(nulls->data1(), 0, n_null_bytes);
        std::vector<char> vectNaN = RetrieveNaNentry(dtype);
        for (int64_t i = 0; i < primitive_array->length(); i++) {
            if (!primitive_array->IsNull(i)) {
                SetBitTo((uint8_t*)nulls->data1(), i, true);
            } else {
                if (primitive_array->type()->id() != arrow::Type::BOOL) {
                    memcpy(data->data1() + siztype * i, vectNaN.data(),
                           siztype);
                }
            }
        }
        infos[infos_pos++] = nulls;
        infos[infos_pos++] = data;
    }
}
