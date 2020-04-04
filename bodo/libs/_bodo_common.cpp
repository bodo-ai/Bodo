#include "_bodo_common.h"

#define ALIGNMENT 64  // preferred alignment for AVX512

std::vector<size_t> numpy_item_size(Bodo_CTypes::_numtypes);

void bodo_common_init() {
    static bool initialized = false;
    if (initialized) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "bodo_common_init called multiple times");
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
    numpy_item_size[Bodo_CTypes::INT128] = BYTES_PER_DECIMAL;
    numpy_item_size[Bodo_CTypes::DATE] = sizeof(uint64_t);
    numpy_item_size[Bodo_CTypes::DATETIME] = sizeof(uint64_t);

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

array_info* alloc_numpy(int64_t length, Bodo_CTypes::CTypeEnum typ_enum) {
    int64_t size = length * numpy_item_size[typ_enum];
    NRT_MemInfo* meminfo = NRT_MemInfo_alloc_safe_aligned(size, ALIGNMENT);
    char* data = (char*)meminfo->data;
    return new array_info(bodo_array_type::NUMPY, typ_enum, length, -1, data,
                          NULL, NULL, NULL, meminfo, NULL);
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
                          -1, data, NULL, NULL, null_bitmap, meminfo,
                          meminfo_bitmask);
}

array_info* alloc_string_array(int64_t length, int64_t n_chars,
                               int64_t extra_null_bytes) {
    // extra_null_bytes are necessary for communication buffers around the edges
    NRT_MemInfo* meminfo = NRT_MemInfo_alloc_dtor_safe(
        sizeof(str_arr_payload), (NRT_dtor_function)dtor_string_array);
    str_arr_payload* payload = (str_arr_payload*)meminfo->data;
    allocate_string_array(&(payload->offsets), &(payload->data),
                          &(payload->null_bitmap), length, n_chars,
                          extra_null_bytes);
    return new array_info(bodo_array_type::STRING, Bodo_CTypes::STRING, length,
                          n_chars, payload->data, (char*)payload->offsets, NULL,
                          (char*)payload->null_bitmap, meminfo, NULL);
}

array_info* alloc_array(int64_t length, int64_t n_sub_elems,
                        bodo_array_type::arr_type_enum arr_type,
                        Bodo_CTypes::CTypeEnum dtype,
                        int64_t extra_null_bytes) {
    if (arr_type == bodo_array_type::STRING)
        return alloc_string_array(length, n_sub_elems, extra_null_bytes);

    // nullable array
    if (arr_type == bodo_array_type::NULLABLE_INT_BOOL)
        return alloc_nullable_array(length, dtype, extra_null_bytes);

    // Numpy
    // TODO: error check
    return alloc_numpy(length, dtype);
}

array_info* copy_array(array_info* earr) {
    int64_t extra_null_bytes = 0;
    array_info* farr =
        alloc_array(earr->length, earr->n_sub_elems, earr->arr_type,
                    earr->dtype, extra_null_bytes);
    if (earr->arr_type == bodo_array_type::NUMPY) {
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
    return farr;
}

void delete_table(table_info* table) {
    for (array_info* a : table->columns) {
        delete a;
    }
    delete table;
}

void delete_table_free_arrays(table_info* table) {
    for (array_info* a : table->columns) {
        if (a != NULL) {
            free_array(a);
            delete a;
        }
    }
    delete table;
}

void free_array(array_info* arr) {
    // string array
    if (arr->arr_type == bodo_array_type::STRING) {
        // data
        delete[] arr->data1;
        // offsets
        delete[] arr->data2;
        // nulls
        if (arr->null_bitmask != nullptr) delete[] arr->null_bitmask;
    } else {                 // Numpy or nullable array
        free(arr->meminfo);  // TODO: decref for cleanup?
        if (arr->meminfo_bitmask != NULL) free(arr->meminfo_bitmask);
    }
    return;
}

extern "C" {

void dtor_string_array(str_arr_payload* in_str_arr, int64_t size, void* in) {
    // printf("str arr dtor size: %lld\n", in_str_arr->size);
    // printf("num chars: %d\n", in_str_arr->offsets[in_str_arr->size]);
    delete[] in_str_arr->offsets;
    delete[] in_str_arr->data;
    if (in_str_arr->null_bitmap != nullptr) delete[] in_str_arr->null_bitmap;
}

void dtor_list_string_array(list_str_arr_payload* in_list_str_arr, int64_t size,
                            void* in) {
    delete[] in_list_str_arr->data;
    delete[] in_list_str_arr->data_offsets;
    delete[] in_list_str_arr->index_offsets;
    if (in_list_str_arr->null_bitmap != nullptr)
        delete[] in_list_str_arr->null_bitmap;
}

void allocate_string_array(uint32_t** offsets, char** data,
                           uint8_t** null_bitmap, int64_t num_strings,
                           int64_t total_size, int64_t extra_null_bytes) {
    // std::cout << "allocating string array: " << num_strings << " " <<
    //                                                 total_size << std::endl;
    *offsets = new uint32_t[num_strings + 1];
    *data = new char[total_size];
    (*offsets)[0] = 0;
    (*offsets)[num_strings] =
        (uint32_t)total_size;  // in case total chars is read from here
    // allocate nulls
    int64_t n_bytes = ((num_strings + 7) >> 3) + extra_null_bytes;
    *null_bitmap = new uint8_t[(size_t)n_bytes];
    // set all bits to 1 indicating non-null as default
    memset(*null_bitmap, 0xff, n_bytes);
    // *data = (char*) new std::string("gggg");
}

void allocate_list_string_array(char** data, uint32_t** data_offsets,
                                uint32_t** index_offsets, uint8_t** null_bitmap,
                                int64_t num_lists, int64_t num_strings,
                                int64_t num_chars, int64_t extra_null_bytes) {
    // std::cout << "allocating list string array: " << num_lists << " "<<
    // num_strings << " " <<
    //                                                 num_chars << std::endl;
    *data = new char[num_chars];

    *data_offsets = new uint32_t[num_strings + 1];
    (*data_offsets)[0] = 0;
    (*data_offsets)[num_strings] =
        (uint32_t)num_chars;  // in case total chars is read from here

    *index_offsets = new uint32_t[num_lists + 1];
    (*index_offsets)[0] = 0;
    (*index_offsets)[num_lists] =
        (uint32_t)num_strings;  // in case total strings is read from here

    // allocate nulls
    int64_t n_bytes = ((num_lists + 7) >> 3) + extra_null_bytes;
    *null_bitmap = new uint8_t[(size_t)n_bytes];
    // set all bits to 1 indicating non-null as default
    memset(*null_bitmap, 0xff, n_bytes);
}
}
