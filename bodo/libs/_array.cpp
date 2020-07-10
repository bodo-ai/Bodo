// Copyright (C) 2019 Bodo Inc. All rights reserved.
/**
 * @author Ehsan (ehsan@bodo-inc.com)
 * @brief Bodo array and table C++ library to allocate arrays, perform parallel
 * shuffle, perform array and table operations like join, groupby
 * @date 2019-10-06
 */

#include <Python.h>
#include <datetime.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <arrow/api.h>
#include <numpy/arrayobject.h>
#include "_array_operations.h"
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_distributed.h"
#include "_groupby.h"
#include "_join.h"
#include "_shuffle.h"

MPI_Datatype decimal_mpi_type = MPI_DATATYPE_NULL;

#undef USE_ARROW_FOR_LIST_STRING
#undef DEBUG_ARROW_ARRAY

struct ArrayBuildInfo {
    ArrayBuildInfo(std::shared_ptr<arrow::Array> a, int p1, int p2)
        : array(a), types_pos(p1), buf_pos(p2) {}
    std::shared_ptr<arrow::Array> array;
    int types_pos;
    int buf_pos;
};

/**
 * This is called recursively
 * @param types: array of type IDs for each array in nested structure
 * @param buffers: array of buffers for each array in nested structure
                   (includes offset, null_bitmap and data buffers)
                   NOTE that not all arrays have all three
 * @param lengths: length of each array in nested structure
 * @param types_pos: track current position in types array
 * @param buf_pos: track current position in buffers array
 */
ArrayBuildInfo nested_array_from_c(const int* types, const uint8_t** buffers,
                                   const int64_t* lengths, int types_pos,
                                   int buf_pos) {
#ifdef DEBUG_ARROW_ARRAY
    std::cout << "Begin of nested_array_from_c\n";
#endif
    Bodo_CTypes::CTypeEnum type = (Bodo_CTypes::CTypeEnum)types[types_pos];
    int64_t length = lengths[types_pos];
    if (type == Bodo_CTypes::LIST) {
#ifdef DEBUG_ARROW_ARRAY
        std::cout << "nested_array_from_c, LIST case\n";
#endif
        const uint8_t* _offsets = buffers[buf_pos++];
        const uint8_t* _null_bitmap = buffers[buf_pos++];

        ArrayBuildInfo ai = nested_array_from_c(types, buffers, lengths,
                                                types_pos + 1, buf_pos);
        types_pos = ai.types_pos;
        buf_pos = ai.buf_pos;

        std::shared_ptr<arrow::Buffer> list_offsets =
            std::make_shared<arrow::Buffer>(_offsets,
                                            sizeof(int32_t) * (length + 1));
        std::shared_ptr<arrow::Buffer> null_bitmap = NULLPTR;
        if (_null_bitmap)
            null_bitmap = std::make_shared<arrow::Buffer>(_null_bitmap,
                                                          (length + 7) >> 3);
        std::shared_ptr<arrow::Field> field =
            std::make_shared<arrow::Field>("field0", ai.array->type());
        std::shared_ptr<arrow::Array> array =
            std::make_shared<arrow::ListArray>(arrow::list(field), length,
                                               list_offsets, ai.array,
                                               null_bitmap);

        return ArrayBuildInfo(array, types_pos, buf_pos);
    } else if (type == Bodo_CTypes::STRUCT) {
#ifdef DEBUG_ARROW_ARRAY
        std::cout << "nested_array_from_c, STRUCT case\n";
#endif
        int num_fields = types[types_pos + 1];
        types_pos += 2;

        const uint8_t* _null_bitmap = buffers[buf_pos++];

        std::vector<std::shared_ptr<arrow::Array>> child_arrays;
        std::vector<std::shared_ptr<arrow::Field>> fields;
        char field_name[100];
        for (int i = 0; i < num_fields; i++) {
            ArrayBuildInfo ai = nested_array_from_c(types, buffers, lengths,
                                                    types_pos, buf_pos);
            child_arrays.push_back(ai.array);
            sprintf(field_name, "field%d", i);
            fields.push_back(
                std::make_shared<arrow::Field>(field_name, ai.array->type()));
            types_pos = ai.types_pos;
            buf_pos = ai.buf_pos;
        }

        std::shared_ptr<arrow::StructType> struct_type =
            std::make_shared<arrow::StructType>(fields);
        std::shared_ptr<arrow::Buffer> null_bitmap = NULLPTR;
        if (_null_bitmap)
            null_bitmap = std::make_shared<arrow::Buffer>(_null_bitmap,
                                                          (length + 7) >> 3);
        std::shared_ptr<arrow::Array> array =
            std::make_shared<arrow::StructArray>(struct_type, length,
                                                 child_arrays, null_bitmap);
        return ArrayBuildInfo(array, types_pos, buf_pos);
        // TODO: string array
    } else {
#ifdef DEBUG_ARROW_ARRAY
        std::cout << "nested_array_from_c, PRIMITIVE case\n";
#endif
        // numeric array
        std::shared_ptr<arrow::Array> array;
        std::shared_ptr<arrow::Buffer> data;
        const uint8_t* _null_bitmap = buffers[buf_pos++];
        std::shared_ptr<arrow::Buffer> null_bitmap = NULLPTR;
        if (_null_bitmap)
            null_bitmap = std::make_shared<arrow::Buffer>(_null_bitmap,
                                                          (length + 7) >> 3);
        if (type == Bodo_CTypes::INT32) {
            data = std::make_shared<arrow::Buffer>(buffers[buf_pos++],
                                                   length * sizeof(int32_t));
            array = std::make_shared<arrow::NumericArray<arrow::Int32Type>>(
                length, data, null_bitmap);
        } else if (type == Bodo_CTypes::INT64) {
            data = std::make_shared<arrow::Buffer>(buffers[buf_pos++],
                                                   length * sizeof(int64_t));
            array = std::make_shared<arrow::NumericArray<arrow::Int64Type>>(
                length, data, null_bitmap);
        } else if (type == Bodo_CTypes::FLOAT64) {
            data = std::make_shared<arrow::Buffer>(buffers[buf_pos++],
                                                   length * sizeof(double));
            array = std::make_shared<arrow::NumericArray<arrow::DoubleType>>(
                length, data, null_bitmap);
        } else {
            Bodo_PyErr_SetString(PyExc_RuntimeError,
                                 "nested_array_from_c unsupported type");
            return {nullptr, 0, 0};
        }
        return ArrayBuildInfo(array, types_pos + 1, buf_pos);
    }
}

array_info* nested_array_to_info(int* types, const uint8_t** buffers,
                                 int64_t* lengths, NRT_MemInfo* meminfo) {
#ifdef DEBUG_ARROW_ARRAY
    std::cout << "Beginning of nested_array_to_info\n";
#endif
    ArrayBuildInfo ai = nested_array_from_c(types, buffers, lengths, 0, 0);
    // TODO: better memory management of struct, meminfo refcount?
    return new array_info(bodo_array_type::ARROW, Bodo_CTypes::INT8 /*dummy*/,
                          lengths[0], -1, -1, NULL, NULL, NULL, NULL, meminfo,
                          NULL, ai.array);
}

array_info* list_string_array_to_info(uint64_t n_items, uint64_t n_strings,
                                      uint64_t n_chars, char* data,
                                      char* data_offsets, char* index_offsets,
                                      char* _null_bitmap,
                                      NRT_MemInfo* meminfo) {
#ifdef USE_ARROW_FOR_LIST_STRING
    // compile time assert to make sure our offset type matches Arrow's
    // TODO replace int32_t here with macro
    static_assert(std::is_same<arrow::ListType::offset_type, int32_t>::value);

    // Convert to Arrow Array to test the Arrow code path

    // make arrow string array
    std::shared_ptr<arrow::Buffer> char_buffer =
        std::make_shared<arrow::Buffer>((uint8_t*)data, n_chars);
    std::shared_ptr<arrow::Buffer> str_offsets =
        std::make_shared<arrow::Buffer>((uint8_t*)data_offsets,
                                        sizeof(int32_t) * (n_strings + 1));
    std::shared_ptr<arrow::Array> string_array =
        std::make_shared<arrow::StringArray>(n_strings, str_offsets,
                                             char_buffer);

    // make arrow list (of string) array
    std::shared_ptr<arrow::Buffer> list_offsets =
        std::make_shared<arrow::Buffer>((uint8_t*)index_offsets,
                                        sizeof(int32_t) * (n_items + 1));
    std::shared_ptr<arrow::Buffer> null_bitmap =
        std::make_shared<arrow::Buffer>((uint8_t*)_null_bitmap,
                                        (n_items + 7) >> 3);
    std::shared_ptr<arrow::Array> array = std::make_shared<arrow::ListArray>(
        arrow::list(arrow::utf8()), n_items, list_offsets, string_array,
        null_bitmap);

    return new array_info(bodo_array_type::ARROW, Bodo_CTypes::INT8 /*dummy*/,
                          n_items, -1, -1, NULL, NULL, NULL, NULL, meminfo,
                          NULL, array);
#else
    // TODO: better memory management of struct, meminfo refcount?
    return new array_info(bodo_array_type::LIST_STRING,
                          Bodo_CTypes::LIST_STRING, n_items, n_strings, n_chars, data,
                          data_offsets, index_offsets, _null_bitmap, meminfo, NULL);
#endif
}

array_info* string_array_to_info(uint64_t n_items, uint64_t n_chars, char* data,
                                 char* offsets, char* null_bitmap,
                                 NRT_MemInfo* meminfo) {
    // TODO: better memory management of struct, meminfo refcount?
    return new array_info(bodo_array_type::STRING, Bodo_CTypes::STRING, n_items,
                          n_chars, -1, data, offsets, NULL, null_bitmap,
                          meminfo, NULL);
}

array_info* numpy_array_to_info(uint64_t n_items, char* data, int typ_enum,
                                NRT_MemInfo* meminfo) {
    // TODO: better memory management of struct, meminfo refcount?
    return new array_info(bodo_array_type::NUMPY,
                          (Bodo_CTypes::CTypeEnum)typ_enum, n_items, -1, -1,
                          data, NULL, NULL, NULL, meminfo, NULL);
}

array_info* nullable_array_to_info(uint64_t n_items, char* data, int typ_enum,
                                   char* null_bitmap, NRT_MemInfo* meminfo,
                                   NRT_MemInfo* meminfo_bitmask) {
    // TODO: better memory management of struct, meminfo refcount?
    return new array_info(bodo_array_type::NULLABLE_INT_BOOL,
                          (Bodo_CTypes::CTypeEnum)typ_enum, n_items, -1, -1,
                          data, NULL, NULL, null_bitmap, meminfo,
                          meminfo_bitmask);
}

array_info* decimal_array_to_info(uint64_t n_items, char* data, int typ_enum,
                                  char* null_bitmap, NRT_MemInfo* meminfo,
                                  NRT_MemInfo* meminfo_bitmask,
                                  int32_t precision, int32_t scale) {
    // TODO: better memory management of struct, meminfo refcount?
    return new array_info(bodo_array_type::NULLABLE_INT_BOOL,
                          (Bodo_CTypes::CTypeEnum)typ_enum, n_items, -1, -1,
                          data, NULL, NULL, null_bitmap, meminfo,
                          meminfo_bitmask, NULL, precision, scale);
}

void info_to_list_string_array(array_info* info, uint64_t* n_items,
                               uint64_t* n_strings, uint64_t* n_chars,
                               char** data, char** data_offsets,
                               char** index_offsets, char** null_bitmap,
                               NRT_MemInfo** meminfo) {
#ifdef USE_ARROW_FOR_LIST_STRING
    if (info->arr_type != bodo_array_type::ARROW) {
        Bodo_PyErr_SetString(PyExc_RuntimeError, "It has to be arrow array");
        return;
    }
    if (!info->array->type()->Equals(arrow::list(arrow::utf8()))) {
      Bodo_PyErr_SetString(PyExc_RuntimeError,"info_to_list_string_array requires list string input");
      return;
    }
    auto list_array = std::dynamic_pointer_cast<arrow::ListArray>(info->array);
    auto str_array = std::dynamic_pointer_cast<arrow::StringArray>(list_array->values());
    NRT_MemInfo* _meminfo = NRT_MemInfo_alloc_dtor_safe(
         sizeof(list_str_arr_payload),
         (NRT_dtor_function)dtor_list_string_array);
    list_str_arr_payload* payload = (list_str_arr_payload*)_meminfo->data;
    allocate_list_string_array(&(payload->data), &(payload->data_offsets),
                               &(payload->index_offsets),
                               &(payload->null_bitmap),
                               list_array->length(), str_array->length(),
                               str_array->value_data()->size(), 0);

    memcpy(payload->data, str_array->value_data()->data(),
           str_array->value_data()->size());
    memcpy(payload->data_offsets, str_array->value_offsets()->data(),
           sizeof(int32_t) * (str_array->length() + 1));
    memcpy(payload->index_offsets, list_array->value_offsets()->data(),
           sizeof(int32_t) * (list_array->length() + 1));
    if (list_array->null_bitmap_data())
      memcpy(payload->null_bitmap, list_array->null_bitmap_data(),
             (list_array->length() + 7) >> 3);

    *n_items = list_array->length();
    *n_strings = str_array->length();
    *n_chars = str_array->value_data()->size();
    
    *data = payload->data;
    *data_offsets = (char*)payload->data_offsets;
    *index_offsets = (char*)payload->index_offsets;
    *null_bitmap = (char*)payload->null_bitmap;
    *meminfo = _meminfo;
#else
    if (info->arr_type != bodo_array_type::LIST_STRING) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "info_to_list_string_array requires list string input");
        return;
    }
    *n_items = info->length;
    *n_strings = info->n_sub_elems;
    *n_chars = info->n_sub_sub_elems;
    *data = info->data1;
    *data_offsets = info->data2;
    *index_offsets = info->data3;
    *null_bitmap = info->null_bitmask;
    *meminfo = info->meminfo;
#endif
}

/**
 * Given an Arrow array, populate array of lengths and array of array_info*
 * with the data from the array and all of its descendant arrays.
 * This is called recursively, and will create one array_info for each individual
 * buffer (offsets, null_bitmaps, data).
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
                       array_info** infos, int64_t &lengths_pos,
                       int64_t &infos_pos) {
#ifdef DEBUG_ARROW_ARRAY
    std::cout << "Beginning of nested_array_to_c\n";
#endif
    if (array->type_id() == arrow::Type::LIST) {
#ifdef DEBUG_ARROW_ARRAY
        std::cout << "nested_array_to_c, LIST case\n";
#endif
        std::shared_ptr<arrow::ListArray> list_array =
            std::dynamic_pointer_cast<arrow::ListArray>(array);
        lengths[lengths_pos++] = list_array->length();

        // allocate output arrays and copy data
        array_info* offsets = alloc_array(list_array->length() + 1, -1, -1,
                                          bodo_array_type::arr_type_enum::NUMPY,
                                          Bodo_CTypes::INT32, 0);
        int64_t n_null_bytes = (list_array->length() + 7) >> 3;
        array_info* nulls = alloc_array(n_null_bytes, -1, -1,
                                        bodo_array_type::arr_type_enum::NUMPY,
                                        Bodo_CTypes::UINT8, 0);

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
        std::cout << "nested_array_to_c, STRUCT case\n";
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
                                        Bodo_CTypes::UINT8, 0);
        memset(nulls->data1, 0, n_null_bytes);
        for (int64_t i = 0; i < struct_array->length(); i++) {
            if (!struct_array->IsNull(i))
                SetBitTo((uint8_t*)nulls->data1, i, true);
        }
        infos[infos_pos++] = nulls;
        for (int i = 0; i < struct_type->num_children();
             i++) {  // each field is an array
            nested_array_to_c(struct_array->field(i), lengths, infos,
                              lengths_pos, infos_pos);
        }

    } else if (array->type_id() == arrow::Type::STRING) {
        Bodo_PyErr_SetString(
            PyExc_RuntimeError,
            "nested_array_to_c: string array not supported yet");
        return;
        //array_info* str_arr = alloc_string_array(...);
        //infos[infos_pos++] = str_arr;
    } else {
#ifdef DEBUG_ARROW_ARRAY
        std::cout << "nested_array_to_c, PRIMITIVE case\n";
#endif
        auto primitive_array =
            std::dynamic_pointer_cast<arrow::PrimitiveArray>(array);
        lengths[lengths_pos++] = primitive_array->length();

        Bodo_CTypes::CTypeEnum dtype =
            arrow_to_bodo_type(primitive_array->type_id());

        // allocate output arrays and copy data
        array_info* data =
            alloc_array(primitive_array->length(), -1, -1,
                        bodo_array_type::arr_type_enum::NUMPY, dtype, 0);
        int64_t n_null_bytes = (primitive_array->length() + 7) >> 3;
        array_info* nulls = alloc_array(n_null_bytes, -1, -1,
                                        bodo_array_type::arr_type_enum::NUMPY,
                                        Bodo_CTypes::UINT8, 0);

        memcpy(data->data1, primitive_array->values()->data(),
               numpy_item_size[dtype] * primitive_array->length());
        memset(nulls->data1, 0, n_null_bytes);
        for (int64_t i = 0; i < primitive_array->length(); i++) {
            if (!primitive_array->IsNull(i))
                SetBitTo((uint8_t*)nulls->data1, i, true);
        }
        infos[infos_pos++] = nulls;
        infos[infos_pos++] = data;
    }
}

void info_to_nested_array(array_info* info, int64_t* lengths, array_info** out_infos) {
#ifdef DEBUG_ARROW_ARRAY
    std::cout << "Beginning of info_to_nested_array\n";
#endif
    int64_t lengths_pos = 0;
    int64_t infos_pos = 0;
    nested_array_to_c(info->array, lengths, out_infos, lengths_pos, infos_pos);
}

void info_to_string_array(array_info* info, NRT_MemInfo** meminfo) {
    if (info->arr_type != bodo_array_type::STRING) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "info_to_string_array requires string input");
        return;
    }
    *meminfo = info->meminfo;
}

void info_to_numpy_array(array_info* info, uint64_t* n_items, char** data,
                         NRT_MemInfo** meminfo) {
    if (info->arr_type != bodo_array_type::NUMPY) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "info_to_numpy_array requires numpy input");
        return;
    }
    *n_items = info->length;
    *data = info->data1;
    *meminfo = info->meminfo;
}

void info_to_nullable_array(array_info* info, uint64_t* n_items,
                            uint64_t* n_bytes, char** data, char** null_bitmap,
                            NRT_MemInfo** meminfo,
                            NRT_MemInfo** meminfo_bitmask) {
    if (info->arr_type != bodo_array_type::NULLABLE_INT_BOOL) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "info_to_nullable_array requires nullable input");
        return;
    }
    *n_items = info->length;
    *n_bytes = (info->length + 7) >> 3;
    *data = info->data1;
    *null_bitmap = info->null_bitmask;
    *meminfo = info->meminfo;
    *meminfo_bitmask = info->meminfo_bitmask;
}

table_info* arr_info_list_to_table(array_info** arrs, int64_t n_arrs) {
    std::vector<array_info*> columns(arrs, arrs + n_arrs);
    return new table_info(columns);
}

array_info* info_from_table(table_info* table, int64_t col_ind) {
    return table->columns[col_ind];
}

/**
 * @brief count the total number of data elements in array(item) arrays
 *
 * @param list_arr_obj array(item) array object
 * @return int64_t total number of data elements
 */
int64_t count_total_elems_list_array(PyObject* list_arr_obj) {
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        return -1;                     \
    }
    // get pd.NA object to check for new NA kind
    PyObject* pd_mod = PyImport_ImportModule("pandas");
    CHECK(pd_mod, "importing pandas module failed");
    PyObject* C_NA = PyObject_GetAttrString(pd_mod, "NA");
    CHECK(C_NA, "getting pd.NA failed");

    Py_ssize_t n = PyObject_Size(list_arr_obj);
    int64_t n_lists = 0;
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* s = PySequence_GetItem(list_arr_obj, i);
        CHECK(s, "getting element failed");
        // Pandas stores NA as either None or nan
        if (!(s == Py_None ||
              (PyFloat_Check(s) && std::isnan(PyFloat_AsDouble(s)))) ||
            s == C_NA) {
            n_lists += PyObject_Size(s);
        }
        Py_DECREF(s);
    }
    return n_lists;
#undef CHECK
}

/**
 * @brief convert Python object to native value and copies it to data buffer
 *
 * @param data data buffer for output native value
 * @param ind index within the buffer for output
 * @param item python object to read
 * @param dtype data type
 */
inline void copy_item_to_buffer(char* data, Py_ssize_t ind, PyObject* item,
                                Bodo_CTypes::CTypeEnum dtype) {
    if (dtype == Bodo_CTypes::INT64) {
        int64_t* ptr = (int64_t*)data;
        ptr[ind] = PyLong_AsLongLong(item);
    } else if (dtype == Bodo_CTypes::FLOAT64) {
        double* ptr = (double*)data;
        ptr[ind] = PyFloat_AsDouble(item);
    } else if (dtype == Bodo_CTypes::_BOOL) {
        bool* ptr = (bool*)data;
        ptr[ind] = (item == Py_True);
    } else if (dtype == Bodo_CTypes::DATE) {
        int64_t* ptr = (int64_t*)data;
        PyObject* year_obj = PyObject_GetAttrString(item, "year");
        PyObject* month_obj = PyObject_GetAttrString(item, "month");
        PyObject* day_obj = PyObject_GetAttrString(item, "day");
        int64_t year = PyLong_AsLongLong(year_obj);
        int64_t month = PyLong_AsLongLong(month_obj);
        int64_t day = PyLong_AsLongLong(day_obj);
        ptr[ind] = (year << 32) + (month << 16) + day;
        Py_DECREF(year_obj);
        Py_DECREF(month_obj);
        Py_DECREF(day_obj);
    } else
        std::cerr << "data type " << dtype
                  << " not supported for unboxing array(item) array."
                  << std::endl;
}

/**
 * @brief compute offsets, data, and null_bitmap values for array(item) array
 * from an array of lists of values. The lists inside array can have different
 * lengths.
 *
 * @param array_item_arr_obj Python Sequence object, intended to be an array of
 * lists of items.
 * @param data data buffer to be filled with all values
 * @param offsets offsets buffer to be filled with computed offsets
 * @param null_bitmap nulls buffer to be filled
 * @param dtype data type of values, currently only float64 and int64 supported.
 */
void array_item_array_from_sequence(PyObject* list_arr_obj, char* data,
                                    uint32_t* offsets, uint8_t* null_bitmap,
                                    Bodo_CTypes::CTypeEnum dtype) {
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        return;                        \
    }

    CHECK(PySequence_Check(list_arr_obj), "expecting a PySequence");
    CHECK(data && offsets && null_bitmap, "buffer arguments must not be NULL");

    // get pd.NA object to check for new NA kind
    PyObject* pd_mod = PyImport_ImportModule("pandas");
    CHECK(pd_mod, "importing pandas module failed");
    PyObject* C_NA = PyObject_GetAttrString(pd_mod, "NA");
    CHECK(C_NA, "getting pd.NA failed");

    Py_ssize_t n = PyObject_Size(list_arr_obj);

    int64_t curr_item_ind = 0;
    for (Py_ssize_t i = 0; i < n; ++i) {
        offsets[i] = curr_item_ind;
        PyObject* s = PySequence_GetItem(list_arr_obj, i);
        CHECK(s, "getting array(item) array element failed");
        // Pandas stores NA as either None or nan
        if (s == Py_None ||
            (PyFloat_Check(s) && std::isnan(PyFloat_AsDouble(s))) ||
            s == C_NA) {
            // set null bit to 0
            SetBitTo(null_bitmap, i, 0);
        } else {
            // set null bit to 1
            null_bitmap[i / 8] |= kBitmask[i % 8];
            // check list
            CHECK(PySequence_Check(s), "expecting a list");
            Py_ssize_t n_items = PyObject_Size(s);
            for (Py_ssize_t j = 0; j < n_items; j++) {
                PyObject* v = PySequence_GetItem(s, j);
                copy_item_to_buffer(data, curr_item_ind, v, dtype);
                Py_DECREF(v);
                curr_item_ind++;
            }
        }
        Py_DECREF(s);
    }
    offsets[n] = curr_item_ind;

    Py_DECREF(pd_mod);
    Py_DECREF(C_NA);
#undef CHECK
}

/**
 * @brief convert native value from buffer to Python object
 *
 * @param data data buffer for input native value (only int64/float64 currently)
 * @param ind index within the buffer
 * @param dtype data type
 * @return python object for value
 */
inline PyObject* value_to_pyobject(const char* data, int64_t ind,
                                   Bodo_CTypes::CTypeEnum dtype) {
    // TODO: support other types
    if (dtype == Bodo_CTypes::INT64) {
        int64_t* ptr = (int64_t*)data;
        return PyLong_FromLongLong(ptr[ind]);
    } else if (dtype == Bodo_CTypes::FLOAT64) {
        double* ptr = (double*)data;
        return PyFloat_FromDouble(ptr[ind]);
    } else if (dtype == Bodo_CTypes::_BOOL) {
        bool* ptr = (bool*)data;
        return PyBool_FromLong((long)(ptr[ind]));
    } else if (dtype == Bodo_CTypes::DATE) {
        int64_t* ptr = (int64_t*)data;
        int64_t val = ptr[ind];
        int year = val >> 32;
        int month = (val >> 16) & 0xFFFF;
        int day = val & 0xFFFF;
        return PyDate_FromDate(year, month, day);
    } else
        std::cerr << "data type " << dtype
                  << " not supported for boxing array(item) array."
                  << std::endl;
    return NULL;
}

/**
 * @brief create a numpy array of lists of item objects from a ArrayItemArray
 *
 * @param num_lists number of lists in input array
 * @param buffer all values
 * @param offsets offsets to data
 * @param null_bitmap null bitmask
 * @param dtype data type of values (currently, only int/float)
 * @return numpy array of list of item objects
 */
void* np_array_from_array_item_array(int64_t num_lists, const char* buffer,
                                     const uint32_t* offsets,
                                     const uint8_t* null_bitmap,
                                     Bodo_CTypes::CTypeEnum dtype) {
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        return NULL;                   \
    }

    // allocate array and get nan object
    npy_intp dims[] = {num_lists};
    PyObject* ret = PyArray_SimpleNew(1, dims, NPY_OBJECT);
    CHECK(ret, "allocating numpy array failed");
    int err;
    PyObject* np_mod = PyImport_ImportModule("numpy");
    CHECK(np_mod, "importing numpy module failed");
    PyObject* nan_obj = PyObject_GetAttrString(np_mod, "nan");
    CHECK(nan_obj, "getting np.nan failed");

    size_t curr_value = 0;
    // for each list item
    for (int64_t i = 0; i < num_lists; ++i) {
        // set nan if item is NA
        auto p = PyArray_GETPTR1((PyArrayObject*)ret, i);
        CHECK(p, "getting offset in numpy array failed");
        if (is_na(null_bitmap, i)) {
            err = PyArray_SETITEM((PyArrayObject*)ret, (char*)p, nan_obj);
            CHECK(err == 0, "setting item in numpy array failed");
            continue;
        }

        // alloc list item
        Py_ssize_t n_vals = (Py_ssize_t)(offsets[i + 1] - offsets[i]);
        PyObject* l = PyList_New(n_vals);

        for (Py_ssize_t j = 0; j < n_vals; j++) {
            PyObject* s = value_to_pyobject(buffer, curr_value, dtype);
            CHECK(s, "creating Python int/float object failed");
            PyList_SET_ITEM(l, j, s);  // steals reference to s!
            curr_value++;
        }

        err = PyArray_SETITEM((PyArrayObject*)ret, (char*)p, l);
        CHECK(err == 0, "setting item in numpy array failed");
        Py_DECREF(l);
    }

    Py_DECREF(np_mod);
    Py_DECREF(nan_obj);
    return ret;
#undef CHECK
}

/**
 * @brief extract data and null_bitmap values for struct array
 * from an array of dict of values. The dicts inside array should have
 * the same key names.
 *
 * @param struct_arr_obj Python Sequence object, intended to be an array of
 * dicts.
 * @param n_fields number of fields in struct array
 * @param data data buffers to be filled with all values (one buffer per field)
 * @param null_bitmap nulls buffer to be filled
 * @param dtype data types of field values
 * @param field_names names of struct fields.
 */
void struct_array_from_sequence(PyObject* struct_arr_obj, int n_fields,
                                char** data, uint8_t* null_bitmap,
                                int32_t* dtypes, char** field_names) {
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        return;                        \
    }

    // TODO: currently only float64 and int64 supported
    CHECK(PySequence_Check(struct_arr_obj), "expecting a PySequence");
    CHECK(data && null_bitmap, "buffer arguments must not be NULL");

    // get pd.NA object to check for new NA kind
    PyObject* pd_mod = PyImport_ImportModule("pandas");
    CHECK(pd_mod, "importing pandas module failed");
    PyObject* C_NA = PyObject_GetAttrString(pd_mod, "NA");
    CHECK(C_NA, "getting pd.NA failed");

    Py_ssize_t n = PyObject_Size(struct_arr_obj);

    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* s = PySequence_GetItem(struct_arr_obj, i);
        CHECK(s, "getting struct array element failed");
        // Pandas stores NA as either None or nan
        if (s == Py_None ||
            (PyFloat_Check(s) && std::isnan(PyFloat_AsDouble(s))) ||
            s == C_NA) {
            // set null bit to 0
            SetBitTo(null_bitmap, i, 0);
        } else {
            // set null bit to 1
            null_bitmap[i / 8] |= kBitmask[i % 8];
            CHECK(PyDict_Check(s), "invalid non-dict element in struct array");
            // set field data values
            for (Py_ssize_t j = 0; j < n_fields; j++) {
                PyObject* v = PyDict_GetItemString(
                    s, field_names[j]);  // returns borrowed reference
                copy_item_to_buffer(data[j], i, v,
                                    (Bodo_CTypes::CTypeEnum)dtypes[j]);
            }
        }
        Py_DECREF(s);
    }

    Py_DECREF(pd_mod);
    Py_DECREF(C_NA);
#undef CHECK
}

/**
 * @brief call PyArray_GETITEM() of Numpy C-API
 *
 * @param arr array object
 * @param p pointer in array object
 * @return PyObject* value returned by getitem
 */
PyObject* array_getitem(PyArrayObject* arr, const char* p) {
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        return NULL;                   \
    }
    PyObject* s = PyArray_GETITEM(arr, p);
    CHECK(s, "getting item in numpy array failed");
    return s;
#undef CHECK
}

/**
 * @brief call PySequence_GetItem() of Python C-API
 *
 * @param obj sequence object (e.g. list)
 * @param i index
 * @return PyObject* value returned by getitem
 */
PyObject* seq_getitem(PyObject* obj, Py_ssize_t i) {
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        return NULL;                   \
    }
    PyObject* s = PySequence_GetItem(obj, i);
    CHECK(s, "getting item failed");
    return s;
#undef CHECK
}

/**
 * @brief create a numpy array of dict objects from a StructArray
 *
 * @param num_structs number of structs in input array (length of array)
 * @param n_fields number of fields in structs
 * @param data data values (one array per field)
 * @param null_bitmap null bitmask
 * @param dtypes data types of field values (currently, only int/float)
 * @param field_names names of struct fields
 * @return numpy array of dict objects
 */
void* np_array_from_struct_array(int64_t num_structs, int n_fields, char** data,
                                 uint8_t* null_bitmap, int32_t* dtypes,
                                 char** field_names) {
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        return NULL;                   \
    }

    // allocate array and get nan object
    npy_intp dims[] = {num_structs};
    PyObject* ret = PyArray_SimpleNew(1, dims, NPY_OBJECT);
    CHECK(ret, "allocating numpy array failed");
    int err;
    PyObject* np_mod = PyImport_ImportModule("numpy");
    CHECK(np_mod, "importing numpy module failed");
    PyObject* nan_obj = PyObject_GetAttrString(np_mod, "nan");
    CHECK(nan_obj, "getting np.nan failed");

    // for each struct value
    for (int64_t i = 0; i < num_structs; ++i) {
        // set nan if value is NA
        auto p = PyArray_GETPTR1((PyArrayObject*)ret, i);
        CHECK(p, "getting offset in numpy array failed");
        if (is_na(null_bitmap, i)) {
            err = PyArray_SETITEM((PyArrayObject*)ret, (char*)p, nan_obj);
            CHECK(err == 0, "setting item in numpy array failed");
            continue;
        }

        // alloc dictionary
        PyObject* d = PyDict_New();

        for (Py_ssize_t j = 0; j < n_fields; j++) {
            PyObject* s = value_to_pyobject(data[j], i,
                                            (Bodo_CTypes::CTypeEnum)dtypes[j]);
            CHECK(s, "creating Python int/float object failed");
            PyDict_SetItemString(d, field_names[j], s);
            Py_DECREF(s);
        }

        err = PyArray_SETITEM((PyArrayObject*)ret, (char*)p, d);
        CHECK(err == 0, "setting item in numpy array failed");
        Py_DECREF(d);
    }

    Py_DECREF(np_mod);
    Py_DECREF(nan_obj);
    return ret;
#undef CHECK
}

/**
 * @brief check if obj is a list
 *
 * @param obj object to check
 * @return int 1 if a list, 0 otherwise
 */
int list_check(PyArrayObject* obj) { return PyList_Check(obj); }

/**
 * @brief check if object is an NA value like None or np.nan
 *
 * @param s Python object to check
 * @param C_NA pd.NA object (passed in to avoid overheads in loops)
 * @return int 1 if value is NA, 0 otherwise
 */
int is_na_value(PyObject* s, PyObject* C_NA) {
    return (s == Py_None ||
            (PyFloat_Check(s) && std::isnan(PyFloat_AsDouble(s))) || s == C_NA);
}

int is_pd_int_array(PyObject* arr) {
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        PyGILState_Release(gilstate);  \
        return false;                  \
    }

    auto gilstate = PyGILState_Ensure();
    // pd.arrays.IntegerArray
    PyObject* pd_mod = PyImport_ImportModule("pandas");
    CHECK(pd_mod, "importing pandas module failed");
    PyObject* pd_arrays_obj = PyObject_GetAttrString(pd_mod, "arrays");
    CHECK(pd_arrays_obj, "getting pd.arrays failed");
    PyObject* pd_arrays_int_arr_obj =
        PyObject_GetAttrString(pd_arrays_obj, "IntegerArray");
    CHECK(pd_arrays_obj, "getting pd.arrays.IntegerArray failed");

    // isinstance(arr, IntegerArray)
    int ret = PyObject_IsInstance(arr, pd_arrays_int_arr_obj);
    CHECK(ret >= 0, "isinstance fails");

    Py_DECREF(pd_mod);
    Py_DECREF(pd_arrays_obj);
    Py_DECREF(pd_arrays_int_arr_obj);
    PyGILState_Release(gilstate);
    return ret;

#undef CHECK
}

/**
 * @brief unbox object array into native data and null_bitmap of native nullable
 * int array
 *
 * @param arr_obj object array with int or NA values
 * @param data native int data array of output array
 * @param null_bitmap null bitmap of output array
 */
void int_array_from_sequence(PyObject* arr_obj, int64_t* data,
                             uint8_t* null_bitmap) {
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        return;                        \
    }

    CHECK(PySequence_Check(arr_obj), "expecting a PySequence");
    CHECK(data && null_bitmap, "buffer arguments must not be NULL");

    // get pd.NA object to check for new NA kind
    PyObject* pd_mod = PyImport_ImportModule("pandas");
    CHECK(pd_mod, "importing pandas module failed");
    PyObject* C_NA = PyObject_GetAttrString(pd_mod, "NA");
    CHECK(C_NA, "getting pd.NA failed");

    Py_ssize_t n = PyObject_Size(arr_obj);

    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* s = PySequence_GetItem(arr_obj, i);
        CHECK(s, "getting int array element failed");
        // Pandas stores NA as either None or nan
        if (s == Py_None ||
            (PyFloat_Check(s) && std::isnan(PyFloat_AsDouble(s))) ||
            s == C_NA) {
            // set null bit to 0
            SetBitTo(null_bitmap, i, 0);
        } else {
            // set null bit to 1
            null_bitmap[i / 8] |= kBitmask[i % 8];
            // TODO: checking int fails for some reason
            // CHECK(PyLong_Check(s), "expecting an int");
            data[i] = PyLong_AsLongLong(s);
        }
        Py_DECREF(s);
    }

    Py_DECREF(pd_mod);
    Py_DECREF(C_NA);
#undef CHECK
}

PyMODINIT_FUNC PyInit_array_ext(void) {
    PyObject* m;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, "array_ext", "No docs", -1, NULL,
    };
    m = PyModule_Create(&moduledef);
    if (m == NULL) return NULL;

    // init datetime APIs
    PyDateTime_IMPORT;

    // init numpy
    import_array();

    bodo_common_init();

    // initialize decimal_mpi_type
    // TODO: free when program exits
    if (decimal_mpi_type == MPI_DATATYPE_NULL) {
        MPI_Type_contiguous(2, MPI_LONG_LONG_INT, &decimal_mpi_type);
        MPI_Type_commit(&decimal_mpi_type);
    }

    groupby_init();

    // DEC_MOD_METHOD(string_array_to_info);
    PyObject_SetAttrString(
        m, "list_string_array_to_info",
        PyLong_FromVoidPtr((void*)(&list_string_array_to_info)));
    PyObject_SetAttrString(m, "nested_array_to_info",
                           PyLong_FromVoidPtr((void*)(&nested_array_to_info)));
    PyObject_SetAttrString(m, "string_array_to_info",
                           PyLong_FromVoidPtr((void*)(&string_array_to_info)));
    PyObject_SetAttrString(m, "numpy_array_to_info",
                           PyLong_FromVoidPtr((void*)(&numpy_array_to_info)));
    PyObject_SetAttrString(
        m, "nullable_array_to_info",
        PyLong_FromVoidPtr((void*)(&nullable_array_to_info)));
    PyObject_SetAttrString(m, "decimal_array_to_info",
                           PyLong_FromVoidPtr((void*)(&decimal_array_to_info)));
    PyObject_SetAttrString(m, "info_to_string_array",
                           PyLong_FromVoidPtr((void*)(&info_to_string_array)));
    PyObject_SetAttrString(
        m, "info_to_list_string_array",
        PyLong_FromVoidPtr((void*)(&info_to_list_string_array)));
    PyObject_SetAttrString(m, "info_to_nested_array",
                           PyLong_FromVoidPtr((void*)(&info_to_nested_array)));
    PyObject_SetAttrString(m, "info_to_numpy_array",
                           PyLong_FromVoidPtr((void*)(&info_to_numpy_array)));
    PyObject_SetAttrString(
        m, "info_to_nullable_array",
        PyLong_FromVoidPtr((void*)(&info_to_nullable_array)));
    PyObject_SetAttrString(m, "alloc_numpy",
                           PyLong_FromVoidPtr((void*)(&alloc_numpy)));
    PyObject_SetAttrString(m, "alloc_string_array",
                           PyLong_FromVoidPtr((void*)(&alloc_string_array)));
    PyObject_SetAttrString(
        m, "arr_info_list_to_table",
        PyLong_FromVoidPtr((void*)(&arr_info_list_to_table)));
    PyObject_SetAttrString(m, "info_from_table",
                           PyLong_FromVoidPtr((void*)(&info_from_table)));
    PyObject_SetAttrString(m, "delete_table",
                           PyLong_FromVoidPtr((void*)(&delete_table)));
    PyObject_SetAttrString(m, "shuffle_table",
                           PyLong_FromVoidPtr((void*)(&shuffle_table)));
    PyObject_SetAttrString(m, "hash_join_table",
                           PyLong_FromVoidPtr((void*)(&hash_join_table)));
    PyObject_SetAttrString(m, "sort_values_table",
                           PyLong_FromVoidPtr((void*)(&sort_values_table)));
    PyObject_SetAttrString(m, "drop_duplicates_table",
                           PyLong_FromVoidPtr((void*)(&drop_duplicates_table)));
    PyObject_SetAttrString(m, "groupby_and_aggregate",
                           PyLong_FromVoidPtr((void*)(&groupby_and_aggregate)));
    PyObject_SetAttrString(m, "array_isin",
                           PyLong_FromVoidPtr((void*)(&array_isin)));
    PyObject_SetAttrString(
        m, "compute_node_partition_by_hash",
        PyLong_FromVoidPtr((void*)(&compute_node_partition_by_hash)));
    PyObject_SetAttrString(
        m, "count_total_elems_list_array",
        PyLong_FromVoidPtr((void*)(&count_total_elems_list_array)));
    PyObject_SetAttrString(
        m, "array_item_array_from_sequence",
        PyLong_FromVoidPtr((void*)(&array_item_array_from_sequence)));
    PyObject_SetAttrString(
        m, "struct_array_from_sequence",
        PyLong_FromVoidPtr((void*)(&struct_array_from_sequence)));
    PyObject_SetAttrString(
        m, "np_array_from_struct_array",
        PyLong_FromVoidPtr((void*)(&np_array_from_struct_array)));
    PyObject_SetAttrString(
        m, "np_array_from_array_item_array",
        PyLong_FromVoidPtr((void*)(&np_array_from_array_item_array)));
    PyObject_SetAttrString(m, "array_getitem",
                           PyLong_FromVoidPtr((void*)(&array_getitem)));
    PyObject_SetAttrString(m, "list_check",
                           PyLong_FromVoidPtr((void*)(&list_check)));
    PyObject_SetAttrString(m, "seq_getitem",
                           PyLong_FromVoidPtr((void*)(&seq_getitem)));
    PyObject_SetAttrString(m, "is_na_value",
                           PyLong_FromVoidPtr((void*)(&is_na_value)));
    PyObject_SetAttrString(m, "is_pd_int_array",
                           PyLong_FromVoidPtr((void*)(&is_pd_int_array)));
    PyObject_SetAttrString(
        m, "int_array_from_sequence",
        PyLong_FromVoidPtr((void*)(&int_array_from_sequence)));
    return m;
}
