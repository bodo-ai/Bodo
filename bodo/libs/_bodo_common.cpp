#include "_bodo_common.h"

#undef DEBUG_ARROW_ARRAY

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
        // TODO Date, Datetime, Timedelta, String, Bool
        default: {
            std::string err_msg = "arrow_to_bodo_type : Unsupported type";
            Bodo_PyErr_SetString(PyExc_RuntimeError, err_msg.c_str());
        }
    }
}

array_info& array_info::operator=(array_info&& other) noexcept {
    if (this != &other) {
        // delete this array's original data
        if (this->arr_type == bodo_array_type::LIST_STRING)
            decref_list_string_array(this->meminfo);
        else
            decref_array(this);

        // copy the other array's pointers into this array_info
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

        // reset the other array_info's pointers
        other.data1 = nullptr;
        other.data2 = nullptr;
        other.data3 = nullptr;
        other.null_bitmask = nullptr;
        other.sub_null_bitmask = nullptr;
        other.meminfo = nullptr;
        other.meminfo_bitmask = nullptr;
        other.array = nullptr;
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
    payload->data->refct--;
    if (payload->data->refct == 0) NRT_MemInfo_call_dtor(payload->data);

    payload->offsets.meminfo->refct--;
    if (payload->offsets.meminfo->refct == 0)
        NRT_MemInfo_call_dtor(payload->offsets.meminfo);

    payload->null_bitmap.meminfo->refct--;
    if (payload->null_bitmap.meminfo->refct == 0)
        NRT_MemInfo_call_dtor(payload->null_bitmap.meminfo);
}

array_info* alloc_list_string_array(int64_t n_lists, int64_t n_strings,
                                    int64_t n_chars, int64_t extra_null_bytes) {
    // allocate string data array
    array_info* data_arr = alloc_array(
        n_strings, n_chars, -1, bodo_array_type::arr_type_enum::ARRAY_ITEM,
        Bodo_CTypes::UINT8, extra_null_bytes, 0);
    NRT_MemInfo* meminfo_string_array = data_arr->meminfo;
    delete data_arr;

    // allocate array(item) array payload
    NRT_MemInfo* meminfo_array_item = NRT_MemInfo_alloc_dtor_safe(
        sizeof(array_item_arr_payload), (NRT_dtor_function)dtor_array_item_arr);
    array_item_arr_payload* payload =
        (array_item_arr_payload*)(meminfo_array_item->data);
    payload->n_arrays = n_lists;
    payload->offsets =
        allocate_numpy_payload(n_lists + 1, Bodo_CTypes::CTypeEnum::UINT32);
    payload->null_bitmap =
        allocate_numpy_payload((int64_t)((n_lists + 7) / 8) + extra_null_bytes,
                               Bodo_CTypes::CTypeEnum::UINT8);
    payload->data = meminfo_string_array;

    array_item_arr_numpy_payload* sub_payload =
        (array_item_arr_numpy_payload*)(meminfo_string_array->data);

    return new array_info(
        bodo_array_type::LIST_STRING, Bodo_CTypes::LIST_STRING, n_lists,
        n_strings, n_chars, (char*)sub_payload->data.data,
        (char*)sub_payload->offsets.data, (char*)payload->offsets.data,
        (char*)payload->null_bitmap.data, (char*)sub_payload->null_bitmap.data,
        meminfo_array_item, NULL);
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
    return numpy_arr_payload(meminfo, NULL, length, itemsize, data, length,
                             itemsize);
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
    payload->data.meminfo->refct--;
    if (payload->data.meminfo->refct == 0)
        NRT_MemInfo_call_dtor(payload->data.meminfo);

    payload->offsets.meminfo->refct--;
    if (payload->offsets.meminfo->refct == 0)
        NRT_MemInfo_call_dtor(payload->offsets.meminfo);

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
    payload->offsets =
        allocate_numpy_payload(n_arrays + 1, Bodo_CTypes::CTypeEnum::UINT32);
    payload->null_bitmap =
        allocate_numpy_payload((int64_t)((n_arrays + 7) / 8) + extra_null_bytes,
                               Bodo_CTypes::CTypeEnum::UINT8);

    return new array_info(bodo_array_type::ARRAY_ITEM, dtype, n_arrays,
                          n_total_items, -1, NULL, NULL, NULL, NULL, NULL,
                          meminfo_array_item, NULL);
}

/**
 * The allocations array function for the function.
 *
 * In the case of NUMPY/CATEGORICAL or NULLABLE_INT_BOOL,
 * -- length is the number of rows, and n_sub_elems, n_sub_sub_elems do not
 * matter. In the case of STRING:
 * -- length is the number of rows (= number of strings)
 * -- n_sub_elems is the total number of characters.
 * In the case of LIST_STRING:
 * -- length is the number of rows.
 * -- n_sub_elems is the number of strings.
 * -- n_sub_sub_elems is the total number of characters.
 *
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

    if (arr_type == bodo_array_type::NUMPY) return alloc_numpy(length, dtype);

    if (arr_type == bodo_array_type::CATEGORICAL)
        return alloc_categorical(length, dtype, num_categories);

    if (arr_type == bodo_array_type::ARRAY_ITEM)
        return alloc_array_item(length, n_sub_elems, dtype, extra_null_bytes);

    Bodo_PyErr_SetString(PyExc_RuntimeError, "Type not covered in alloc_array");
    return nullptr;
}

array_info* copy_array(array_info* earr) {
    int64_t extra_null_bytes = 0;
    array_info* farr = alloc_array(
        earr->length, earr->n_sub_elems, earr->n_sub_sub_elems, earr->arr_type,
        earr->dtype, extra_null_bytes, earr->num_categories);
    if (earr->arr_type == bodo_array_type::NUMPY ||
        earr->arr_type == bodo_array_type::CATEGORICAL) {
        uint64_t siztype = numpy_item_size[earr->dtype];
        memcpy(farr->data1, earr->data1, siztype * earr->length);
    }
    if (earr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        uint64_t siztype = numpy_item_size[earr->dtype];
        memcpy(farr->data1, earr->data1, siztype * earr->length);
        int64_t n_bytes = ((earr->length + 7) >> 3);
        memcpy(farr->null_bitmask, earr->null_bitmask, n_bytes);
    }
    if (earr->arr_type == bodo_array_type::STRING) {
        memcpy(farr->data1, earr->data1, earr->n_sub_elems);
        memcpy(farr->data2, earr->data2, sizeof(uint32_t) * (earr->length + 1));
        int64_t n_bytes = ((earr->length + 7) >> 3);
        memcpy(farr->null_bitmask, earr->null_bitmask, n_bytes);
    }
    if (earr->arr_type == bodo_array_type::LIST_STRING) {
        memcpy(farr->data1, earr->data1, earr->n_sub_sub_elems);
        memcpy(farr->data2, earr->data2,
               sizeof(uint32_t) * (earr->n_sub_elems + 1));
        memcpy(farr->data3, earr->data3, sizeof(uint32_t) * (earr->length + 1));
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
    if (arr->meminfo != NULL) {
        arr->meminfo->refct--;
        if (arr->meminfo->refct == 0) NRT_MemInfo_call_dtor(arr->meminfo);
    }
    if (arr->meminfo_bitmask != NULL) {
        arr->meminfo_bitmask->refct--;
        if (arr->meminfo_bitmask->refct == 0)
            NRT_MemInfo_call_dtor(arr->meminfo_bitmask);
    }
}

void incref_array(array_info* arr) {
    if (arr->meminfo != NULL) arr->meminfo->refct++;
    if (arr->meminfo_bitmask != NULL) arr->meminfo_bitmask->refct++;
}

void decref_list_string_array(NRT_MemInfo* meminfo) {
    array_item_arr_payload* payload = (array_item_arr_payload*)(meminfo->data);

    // delete string array
    payload->data->refct--;
    if (payload->data->refct == 0) NRT_MemInfo_call_dtor(payload->data);

    // delete array item array
    payload->offsets.meminfo->refct--;
    if (payload->offsets.meminfo->refct == 0)
        NRT_MemInfo_call_dtor(payload->offsets.meminfo);
    payload->null_bitmap.meminfo->refct--;
    if (payload->null_bitmap.meminfo->refct == 0)
        NRT_MemInfo_call_dtor(payload->null_bitmap.meminfo);
    meminfo->refct--;
    if (meminfo->refct == 0) NRT_MemInfo_call_dtor(meminfo);
    // TODO: add meminfo for sub_null_bitmask in array struct and decref it here
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
#ifdef DEBUG_ARROW_ARRAY
    std::cout << "nested_array_to_c : Beginning of nested_array_to_c\n";
#endif
    if (array->type_id() == arrow::Type::LIST) {
#ifdef DEBUG_ARROW_ARRAY
        std::cout << "nested_array_to_c : LIST case\n";
#endif
        std::shared_ptr<arrow::ListArray> list_array =
            std::dynamic_pointer_cast<arrow::ListArray>(array);
        lengths[lengths_pos++] = list_array->length();

        // allocate output arrays and copy data
        array_info* offsets = alloc_array(list_array->length() + 1, -1, -1,
                                          bodo_array_type::arr_type_enum::NUMPY,
                                          Bodo_CTypes::INT32, 0, 0);
        int64_t n_null_bytes = (list_array->length() + 7) >> 3;
        array_info* nulls = alloc_array(n_null_bytes, -1, -1,
                                        bodo_array_type::arr_type_enum::NUMPY,
                                        Bodo_CTypes::UINT8, 0, 0);

        memcpy(offsets->data1, list_array->value_offsets()->data(),
               (list_array->length() + 1) * sizeof(int32_t));
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
#ifdef DEBUG_ARROW_ARRAY
        std::cout << "nested_array_to_c : STRUCT case\n";
#endif
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
    } else if (array->type_id() == arrow::Type::STRING) {
#ifdef DEBUG_ARROW_ARRAY
        std::cout << "nested_array_to_c : STRING case\n";
#endif
        auto str_array = std::dynamic_pointer_cast<arrow::StringArray>(array);
        lengths[lengths_pos++] = str_array->length();
        int64_t n_strings = str_array->length();
        int64_t n_chars =
            ((int32_t*)str_array->value_offsets()->data())[n_strings];
        array_info* str_arr_info = alloc_string_array(n_strings, n_chars, 0);
        memcpy(str_arr_info->data1, str_array->value_data()->data(),
               sizeof(char) * n_chars);  // data
        memcpy(str_arr_info->data2, str_array->value_offsets()->data(),
               sizeof(int32_t) * (n_strings + 1));  // offsets
        int64_t n_null_bytes = (n_strings + 7) >> 3;
        memset(str_arr_info->null_bitmask, 0, n_null_bytes);
        for (int64_t i = 0; i < n_strings; i++) {
            if (!str_array->IsNull(i))
                SetBitTo((uint8_t*)str_arr_info->null_bitmask, i, true);
        }
        infos[infos_pos++] = str_arr_info;
    } else {
#ifdef DEBUG_ARROW_ARRAY
        std::cout << "nested_array_to_c : PRIMITIVE case\n";
#endif
        auto primitive_array =
            std::dynamic_pointer_cast<arrow::PrimitiveArray>(array);
        lengths[lengths_pos++] = primitive_array->length();
#ifdef DEBUG_ARROW_ARRAY
        std::cout << "nested_array_to_c : PRIMITIVE case, step 1\n";
#endif

        Bodo_CTypes::CTypeEnum dtype =
            arrow_to_bodo_type(primitive_array->type_id());
#ifdef DEBUG_ARROW_ARRAY
        std::cout << "nested_array_to_c : PRIMITIVE case, step 2\n";
#endif

        // allocate output arrays and copy data
        array_info* data =
            alloc_array(primitive_array->length(), -1, -1,
                        bodo_array_type::arr_type_enum::NUMPY, dtype, 0, 0);
        int64_t n_null_bytes = (primitive_array->length() + 7) >> 3;
        array_info* nulls = alloc_array(n_null_bytes, -1, -1,
                                        bodo_array_type::arr_type_enum::NUMPY,
                                        Bodo_CTypes::UINT8, 0, 0);
#ifdef DEBUG_ARROW_ARRAY
        std::cout << "nested_array_to_c : PRIMITIVE case, step 3\n";
#endif
        uint64_t siztype = numpy_item_size[dtype];
        memcpy(data->data1, primitive_array->values()->data(),
               siztype * primitive_array->length());
#ifdef DEBUG_ARROW_ARRAY
        std::cout << "nested_array_to_c : PRIMITIVE case, step 4\n";
#endif
        memset(nulls->data1, 0, n_null_bytes);
#ifdef DEBUG_ARROW_ARRAY
        std::cout << "nested_array_to_c : PRIMITIVE case, step 5\n";
#endif
        std::vector<char> vectNaN = RetrieveNaNentry(dtype);
#ifdef DEBUG_ARROW_ARRAY
        std::cout << "nested_array_to_c : PRIMITIVE case, step 6 |vectNaN|="
                  << vectNaN.size() << " siztype=" << siztype << "\n";
#endif
        for (int64_t i = 0; i < primitive_array->length(); i++) {
            if (!primitive_array->IsNull(i))
                SetBitTo((uint8_t*)nulls->data1, i, true);
            else
                memcpy(data->data1 + siztype * i, vectNaN.data(), siztype);
        }
#ifdef DEBUG_ARROW_ARRAY
        std::cout << "nested_array_to_c : PRIMITIVE case, step 7\n";
#endif
        infos[infos_pos++] = nulls;
        infos[infos_pos++] = data;
#ifdef DEBUG_ARROW_ARRAY
        std::cout << "nested_array_to_c : PRIMITIVE case, step 8\n";
#endif
    }
}

extern "C" {

void dtor_string_array(str_arr_payload* in_str_arr, int64_t size, void* in) {
    // check for NULL since move_str_arr_payload() may set pointers to NULL
    if (in_str_arr->offsets != nullptr) delete[] in_str_arr->offsets;
    if (in_str_arr->data != nullptr) delete[] in_str_arr->data;
    if (in_str_arr->null_bitmap != nullptr) delete[] in_str_arr->null_bitmap;
}

void allocate_list_string_array(int64_t n_lists, int64_t n_strings,
                                int64_t n_chars, int64_t extra_null_bytes,
                                array_item_arr_payload* payload,
                                str_arr_payload* sub_payload) {
    // string array
    sub_payload->num_strings = n_strings;
    sub_payload->offsets = new int32_t[n_strings + 1];
    sub_payload->data = new char[n_chars];
    sub_payload->offsets[0] = 0;
    sub_payload->offsets[n_strings] =
        (int32_t)n_chars;  // in case total chars is read from here
    // allocate nulls
    int64_t n_bytes = ((n_strings + 7) >> 3) + extra_null_bytes;
    sub_payload->null_bitmap = new uint8_t[(size_t)n_bytes];
    // set all bits to 1 indicating non-null as default
    memset(sub_payload->null_bitmap, 0xff, n_bytes);

    // list array
    payload->n_arrays = n_lists;

    // allocate offsets
    NRT_MemInfo* meminfo_offsets_array = NRT_MemInfo_alloc_safe_aligned(
        sizeof(int32_t) * (n_lists + 1), ALIGNMENT);
    payload->offsets.meminfo = meminfo_offsets_array;
    payload->offsets.parent = NULL;
    payload->offsets.nitems = n_lists + 1;
    payload->offsets.itemsize = sizeof(int32_t);
    payload->offsets.data = (char*)meminfo_offsets_array->data;
    payload->offsets.shape = n_lists + 1;
    payload->offsets.strides = 1;
    ((int32_t*)payload->offsets.data)[0] = 0;
    // in case total strings is read from here
    ((int32_t*)payload->offsets.data)[n_lists] = (int32_t)n_strings;

    // allocate nulls
    n_bytes = ((n_lists + 7) >> 3) + extra_null_bytes;
    NRT_MemInfo* meminfo_nulls_array =
        NRT_MemInfo_alloc_safe_aligned(n_bytes, ALIGNMENT);
    payload->null_bitmap.meminfo = meminfo_nulls_array;
    payload->null_bitmap.parent = NULL;
    payload->null_bitmap.nitems = n_bytes;
    payload->null_bitmap.itemsize = 1;
    payload->null_bitmap.data = (char*)meminfo_nulls_array->data;
    payload->null_bitmap.shape = n_bytes;
    payload->null_bitmap.strides = 1;
    // set all bits to 1 indicating non-null as default
    memset(payload->null_bitmap.data, 0xff, n_bytes);
}
}
