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

struct ArrayBuildInfo {
    ArrayBuildInfo(std::shared_ptr<arrow::Array> a, int p1, int p2, int p3,
                   int p4)
        : array(a), type_pos(p1), buf_pos(p2), length_pos(p3), name_pos(p4) {}
    std::shared_ptr<arrow::Array> array;
    int type_pos;
    int buf_pos;
    int length_pos;
    int name_pos;
};

/**
 * This is called recursively
 * @param types: array of type IDs for each array in nested structure
 * @param buffers: array of buffers for each array in nested structure
                   (includes offset, null_bitmap and data buffers)
                   NOTE that not all arrays have all three
 * @param lengths: length of each array in nested structure
 * @param type_pos: track current position in types array
 * @param buf_pos: track current position in buffers array
 * @param length_pos: track current position in lengths array
 * @param name_pos: track current position in field names
 */
ArrayBuildInfo nested_array_from_c(const int* types, const uint8_t** buffers,
                                   const int64_t* lengths, char** field_names,
                                   int type_pos, int buf_pos, int length_pos,
                                   int name_pos) {
    Bodo_CTypes::CTypeEnum type = (Bodo_CTypes::CTypeEnum)types[type_pos];
    int64_t length = lengths[length_pos];
    if (type == Bodo_CTypes::LIST) {
        const uint8_t* _offsets = buffers[buf_pos++];
        const uint8_t* _null_bitmap = buffers[buf_pos++];

        ArrayBuildInfo ai = nested_array_from_c(
            types, buffers, lengths, field_names, type_pos + 1, buf_pos,
            length_pos + 1, name_pos);
        type_pos = ai.type_pos;
        buf_pos = ai.buf_pos;
        length_pos = ai.length_pos;
        name_pos = ai.name_pos;

        std::shared_ptr<arrow::Buffer> list_offsets =
            std::make_shared<arrow::Buffer>(_offsets,
                                            sizeof(offset_t) * (length + 1));
        std::shared_ptr<arrow::Buffer> null_bitmap = NULLPTR;
        if (_null_bitmap)
            null_bitmap = std::make_shared<arrow::Buffer>(_null_bitmap,
                                                          (length + 7) >> 3);
        std::shared_ptr<arrow::Field> field =
            std::make_shared<arrow::Field>("field0", ai.array->type());
        std::shared_ptr<arrow::Array> array =
#if OFFSET_BITWIDTH == 32
            std::make_shared<arrow::ListArray>(arrow::list(field), length,
#else
            std::make_shared<arrow::LargeListArray>(arrow::large_list(field),
                                                    length,
#endif
                                               list_offsets, ai.array,
                                               null_bitmap);

        return ArrayBuildInfo(array, type_pos, buf_pos, length_pos, name_pos);
    } else if (type == Bodo_CTypes::STRUCT) {
        int num_fields = types[type_pos + 1];
        type_pos += 2;
        length_pos += 2;

        const uint8_t* _null_bitmap = buffers[buf_pos++];

        std::vector<std::shared_ptr<arrow::Array>> child_arrays;
        std::vector<std::shared_ptr<arrow::Field>> fields;
        for (int i = 0; i < num_fields; i++) {
            std::string e_name(field_names[name_pos]);
            ArrayBuildInfo ai = nested_array_from_c(
                types, buffers, lengths, field_names, type_pos, buf_pos,
                length_pos, name_pos + 1);
            child_arrays.push_back(ai.array);
            fields.push_back(
                std::make_shared<arrow::Field>(e_name, ai.array->type()));
            type_pos = ai.type_pos;
            buf_pos = ai.buf_pos;
            length_pos = ai.length_pos;
            name_pos = ai.name_pos;
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
        return ArrayBuildInfo(array, type_pos, buf_pos, length_pos, name_pos);
    } else if (type == Bodo_CTypes::STRING) {
        const uint8_t* _offsets = buffers[buf_pos++];
        const uint8_t* _null_bitmap = buffers[buf_pos++];

        std::shared_ptr<arrow::Buffer> str_offsets =
            std::make_shared<arrow::Buffer>(_offsets,
                                            sizeof(offset_t) * (length + 1));

        std::shared_ptr<arrow::Buffer> null_bitmap = NULLPTR;
        if (_null_bitmap)
            null_bitmap = std::make_shared<arrow::Buffer>(_null_bitmap,
                                                          (length + 7) >> 3);

        int64_t num_chars = ((offset_t*)_offsets)[length];
        std::shared_ptr<arrow::Buffer> data = std::make_shared<arrow::Buffer>(
            buffers[buf_pos++], num_chars * sizeof(char));

        std::shared_ptr<arrow::Array> array =
#if OFFSET_BITWIDTH == 32
            std::make_shared<arrow::StringArray>(length, str_offsets, data,
#else
            std::make_shared<arrow::LargeStringArray>(length, str_offsets, data,
#endif
                                                 null_bitmap);
        return ArrayBuildInfo(array, type_pos + 1, buf_pos, length_pos + 1,
                              name_pos);
    } else if (type == Bodo_CTypes::DECIMAL) {
        // The null bitmap
        const uint8_t* _null_bitmap = buffers[buf_pos++];
        std::shared_ptr<arrow::Buffer> null_bitmap = NULLPTR;
        int64_t null_count_ = 0;
        if (_null_bitmap) {
            null_bitmap = std::make_shared<arrow::Buffer>(_null_bitmap,
                                                          (length + 7) >> 3);
            for (int64_t i_row = 0; i_row < length; i_row++) {
                bool bit = GetBit(_null_bitmap, i_row);
                if (!bit) null_count_++;
            }
        }
        // The other buffers
        const uint8_t* data = buffers[buf_pos++];
        std::shared_ptr<arrow::Buffer> data_buf =
            std::make_shared<arrow::Buffer>(data, length);
        std::vector<std::shared_ptr<arrow::Buffer>> l_buf = {null_bitmap,
                                                             data_buf};
        // The returning array.
        int32_t precision = types[type_pos + 1];
        int32_t scale = types[type_pos + 2];
        arrow::Result<std::shared_ptr<arrow::DataType>> type_res;
        type_res = arrow::Decimal128Type::Make(precision, scale);
        std::shared_ptr<arrow::DataType> type =
            std::move(type_res).ValueOrDie();
        std::shared_ptr<arrow::ArrayData> arr =
            arrow::ArrayData::Make(type, length, l_buf, null_count_, 0);
        std::shared_ptr<arrow::Array> array =
            std::make_shared<arrow::Decimal128Array>(arr);
        return ArrayBuildInfo(array, type_pos + 3, buf_pos, length_pos + 1,
                              name_pos);
    } else {  // Case of numeric/decimal array
        // First the null bitmap of the array
        const uint8_t* _null_bitmap = buffers[buf_pos++];
        std::shared_ptr<arrow::Buffer> null_bitmap = NULLPTR;
        if (_null_bitmap)
            null_bitmap = std::make_shared<arrow::Buffer>(_null_bitmap,
                                                          (length + 7) >> 3);
        // Second the array itself
        std::shared_ptr<arrow::Array> array;
        int64_t siz_typ = numpy_item_size[type];
        std::shared_ptr<arrow::Buffer> data = std::make_shared<arrow::Buffer>(
            buffers[buf_pos++], length * siz_typ);
        // We canot change code below to something more generic since the
        // arrow::UInt8Type are really types and not enum values.
        if (type == Bodo_CTypes::_BOOL) {
            // Arrow's boolean array uses 1 bit for each bool
            // we use uint8 for now to avoid conversion
            array = std::make_shared<arrow::NumericArray<arrow::UInt8Type>>(
                length, data, null_bitmap);
        } else if (type == Bodo_CTypes::INT8) {
            array = std::make_shared<arrow::NumericArray<arrow::Int8Type>>(
                length, data, null_bitmap);
        } else if (type == Bodo_CTypes::UINT8) {
            array = std::make_shared<arrow::NumericArray<arrow::UInt8Type>>(
                length, data, null_bitmap);
        } else if (type == Bodo_CTypes::INT16) {
            array = std::make_shared<arrow::NumericArray<arrow::Int16Type>>(
                length, data, null_bitmap);
        } else if (type == Bodo_CTypes::UINT16) {
            array = std::make_shared<arrow::NumericArray<arrow::UInt16Type>>(
                length, data, null_bitmap);
        } else if (type == Bodo_CTypes::INT32) {
            array = std::make_shared<arrow::NumericArray<arrow::Int32Type>>(
                length, data, null_bitmap);
        } else if (type == Bodo_CTypes::UINT32) {
            array = std::make_shared<arrow::NumericArray<arrow::UInt32Type>>(
                length, data, null_bitmap);
        } else if (type == Bodo_CTypes::INT64) {
            array = std::make_shared<arrow::NumericArray<arrow::Int64Type>>(
                length, data, null_bitmap);
        } else if (type == Bodo_CTypes::UINT64) {
            array = std::make_shared<arrow::NumericArray<arrow::UInt64Type>>(
                length, data, null_bitmap);
        } else if (type == Bodo_CTypes::FLOAT32) {
            array = std::make_shared<arrow::NumericArray<arrow::FloatType>>(
                length, data, null_bitmap);
        } else if (type == Bodo_CTypes::FLOAT64) {
            array = std::make_shared<arrow::NumericArray<arrow::DoubleType>>(
                length, data, null_bitmap);
        } else {
            Bodo_PyErr_SetString(PyExc_RuntimeError,
                                 "nested_array_from_c unsupported type");
            return {nullptr, 0, 0, 0, 0};
        }
        return ArrayBuildInfo(array, type_pos + 1, buf_pos, length_pos + 1,
                              name_pos);
    }
}

array_info* nested_array_to_info(int* types, const uint8_t** buffers,
                                 int64_t* lengths, char** field_names,
                                 NRT_MemInfo* meminfo) {
    try {
        int type_pos = 0;
        int buf_pos = 0;
        int length_pos = 0;
        int name_pos = 0;
        ArrayBuildInfo ai =
            nested_array_from_c(types, buffers, lengths, field_names, type_pos,
                                buf_pos, length_pos, name_pos);
        // TODO: better memory management of struct, meminfo refcount?
        return new array_info(
            bodo_array_type::ARROW, Bodo_CTypes::INT8 /*dummy*/, lengths[0], -1,
            -1, NULL, NULL, NULL, NULL, NULL, meminfo, NULL, ai.array);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

array_info* list_string_array_to_info(NRT_MemInfo* meminfo) {
    array_item_arr_payload* payload = (array_item_arr_payload*)meminfo->data;
    int64_t n_items = payload->n_arrays;

    array_item_arr_numpy_payload* sub_payload =
        (array_item_arr_numpy_payload*)payload->data->data;
    int64_t n_strings = sub_payload->n_arrays;
    int64_t n_chars = ((offset_t*)sub_payload->offsets.data)[n_strings];

    return new array_info(
        bodo_array_type::LIST_STRING, Bodo_CTypes::LIST_STRING, n_items,
        n_strings, n_chars, (char*)sub_payload->data.data,
        (char*)sub_payload->offsets.data, (char*)payload->offsets.data,
        (char*)payload->null_bitmap.data, (char*)sub_payload->null_bitmap.data,
        meminfo, nullptr);
}

array_info* string_array_to_info(uint64_t n_items, uint64_t n_chars, char* data,
                                 char* offsets, char* null_bitmap,
                                 NRT_MemInfo* meminfo, int is_bytes) {
    // TODO: better memory management of struct, meminfo refcount?
    auto dtype = Bodo_CTypes::STRING;
    if (is_bytes) dtype = Bodo_CTypes::BINARY;
    return new array_info(bodo_array_type::STRING, dtype, n_items, n_chars, -1,
                          data, offsets, NULL, null_bitmap, NULL, meminfo,
                          NULL);
}

array_info* dict_str_array_to_info(array_info* str_arr, array_info* indices_arr,
                                   char* null_bitmap,
                                   int32_t has_global_dictionary) {
    // For now has_sorted_dictionary is only available and exposed in the C++
    // struct, so we set it to false
    return new array_info(
        bodo_array_type::DICT, Bodo_CTypes::STRING, indices_arr->length, -1, -1,
        NULL, NULL, NULL, null_bitmap, NULL, NULL, NULL, NULL, 0, 0, 0,
        bool(has_global_dictionary), false, str_arr, indices_arr);
}

array_info* get_nested_info(array_info* dict_arr, int32_t info_no) {
    if (info_no == 1) {
        return dict_arr->info1;
    } else if (info_no == 2) {
        return dict_arr->info2;
    } else {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "get_nested_info: invalid info_no");
        return NULL;
    }
}

int32_t get_has_global_dictionary(array_info* dict_arr) {
    return int32_t(dict_arr->has_global_dictionary);
}

array_info* numpy_array_to_info(uint64_t n_items, char* data, int typ_enum,
                                NRT_MemInfo* meminfo) {
    // TODO: better memory management of struct, meminfo refcount?
    return new array_info(bodo_array_type::NUMPY,
                          (Bodo_CTypes::CTypeEnum)typ_enum, n_items, -1, -1,
                          data, NULL, NULL, NULL, NULL, meminfo, NULL);
}

#undef DEBUG_CATEGORICAL

array_info* categorical_array_to_info(uint64_t n_items, char* data,
                                      int typ_enum, int64_t num_categories,
                                      NRT_MemInfo* meminfo) {
// TODO: better memory management of struct, meminfo refcount?
#ifdef DEBUG_CATEGORICAL
    std::cout << "num_categories=" << num_categories << " n_items=" << n_items
              << "\n";
    std::cout << "typ_enum=" << typ_enum << "\n";
#endif
    return new array_info(bodo_array_type::CATEGORICAL,
                          (Bodo_CTypes::CTypeEnum)typ_enum, n_items, -1, -1,
                          data, NULL, NULL, NULL, NULL, meminfo, NULL, nullptr,
                          0, 0, num_categories);
}

array_info* nullable_array_to_info(uint64_t n_items, char* data, int typ_enum,
                                   char* null_bitmap, NRT_MemInfo* meminfo,
                                   NRT_MemInfo* meminfo_bitmask) {
    // TODO: better memory management of struct, meminfo refcount?
    return new array_info(bodo_array_type::NULLABLE_INT_BOOL,
                          (Bodo_CTypes::CTypeEnum)typ_enum, n_items, -1, -1,
                          data, NULL, NULL, null_bitmap, NULL, meminfo,
                          meminfo_bitmask);
}

array_info* interval_array_to_info(uint64_t n_items, char* left_data,
                                   char* right_data, int typ_enum,
                                   NRT_MemInfo* left_meminfo,
                                   NRT_MemInfo* right_meminfo) {
    return new array_info(bodo_array_type::INTERVAL,
                          (Bodo_CTypes::CTypeEnum)typ_enum, n_items, -1, -1,
                          left_data, right_data, NULL, NULL, NULL, left_meminfo,
                          right_meminfo);
}

array_info* decimal_array_to_info(uint64_t n_items, char* data, int typ_enum,
                                  char* null_bitmap, NRT_MemInfo* meminfo,
                                  NRT_MemInfo* meminfo_bitmask,
                                  int32_t precision, int32_t scale) {
    // TODO: better memory management of struct, meminfo refcount?
    return new array_info(bodo_array_type::NULLABLE_INT_BOOL,
                          (Bodo_CTypes::CTypeEnum)typ_enum, n_items, -1, -1,
                          data, NULL, NULL, null_bitmap, NULL, meminfo,
                          meminfo_bitmask, NULL, precision, scale);
}

void info_to_list_string_array(array_info* info,
                               NRT_MemInfo** array_item_meminfo) {
    if (info->arr_type != bodo_array_type::LIST_STRING) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "_array.cpp::info_to_list_string_array: info_to_list_string_array "
            "requires list string input.");
        return;
    }

    *array_item_meminfo = info->meminfo;
}

void info_to_nested_array(array_info* info, int64_t* lengths,
                          array_info** out_infos) {
    try {
        int64_t lengths_pos = 0;
        int64_t infos_pos = 0;
        nested_array_to_c(info->array, lengths, out_infos, lengths_pos,
                          infos_pos);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return;
    }
}

void info_to_string_array(array_info* info, NRT_MemInfo** meminfo) {
    if (info->arr_type != bodo_array_type::STRING) {
        PyErr_SetString(PyExc_RuntimeError,
                        "_array.cpp::info_to_string_array: "
                        "info_to_string_array requires string input.");
        return;
    }
    *meminfo = info->meminfo;
}

void info_to_numpy_array(array_info* info, uint64_t* n_items, char** data,
                         NRT_MemInfo** meminfo) {
    if ((info->arr_type != bodo_array_type::NUMPY) &&
        (info->arr_type != bodo_array_type::CATEGORICAL)) {
        // TODO: print array type in the error
        PyErr_SetString(PyExc_RuntimeError,
                        "_array.cpp::info_to_numpy_array: info_to_numpy_array "
                        "requires numpy input.");
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
        PyErr_SetString(PyExc_RuntimeError,
                        "_array.cpp::info_to_nullable_array: "
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

void info_to_interval_array(array_info* info, uint64_t* n_items,
                            char** left_data, char** right_data,
                            NRT_MemInfo** left_meminfo,
                            NRT_MemInfo** right_meminfo) {
    if (info->arr_type != bodo_array_type::INTERVAL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "_array.cpp::info_to_interval_array: "
                        "info_to_interval_array requires interval input");
        return;
    }
    *n_items = info->length;
    *left_data = info->data1;
    *right_data = info->data2;
    *left_meminfo = info->meminfo;
    *right_meminfo = info->meminfo_bitmask;
}

table_info* arr_info_list_to_table(array_info** arrs, int64_t n_arrs) {
    std::vector<array_info*> columns(arrs, arrs + n_arrs);
    return new table_info(columns);
}

array_info* info_from_table(table_info* table, int64_t col_ind) {
    return table->columns[col_ind];
}

/**
 * @brief create a concatenated string and offset table from a numpy array of
 * strings
 *
 * @param obj numpy array of strings
 * @param is_bytes whether the contents are bytes objects instead of str
 * @return NRT_MemInfo* meminfo of array(item) array containing string data
 */
NRT_MemInfo* string_array_from_sequence(PyObject* obj, int is_bytes) {
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        return NULL;                   \
    }

    CHECK(PySequence_Check(obj), "expecting a PySequence");

    Py_ssize_t n = PyObject_Size(obj);
    if (n == 0) {
        // empty sequence, this is not an error, need to set size
        array_info* out_arr =
            alloc_array(0, 0, -1, bodo_array_type::arr_type_enum::ARRAY_ITEM,
                        Bodo_CTypes::UINT8, 0, 0);
        NRT_MemInfo* out_meminfo = out_arr->meminfo;
        delete out_arr;
        return out_meminfo;
    }

    // allocate null bitmap
    // same formula as BytesForBits in Arrow
    int64_t n_bytes = (n + 7) >> 3;
    numpy_arr_payload null_bitmap_payload =
        allocate_numpy_payload(n_bytes, Bodo_CTypes::UINT8);
    uint8_t* null_bitmap = (uint8_t*)null_bitmap_payload.data;
    memset(null_bitmap, 0, n_bytes);

    // get pd.NA object to check for new NA kind
    // simple equality check is enough since the object is a singleton
    // example:
    // https://github.com/pandas-dev/pandas/blob/fcadff30da9feb3edb3acda662ff6143b7cb2d9f/pandas/_libs/missing.pyx#L57
    PyObject* pd_mod = PyImport_ImportModule("pandas");
    CHECK(pd_mod, "importing pandas module failed");
    PyObject* C_NA = PyObject_GetAttrString(pd_mod, "NA");
    CHECK(C_NA, "getting pd.NA failed");

    numpy_arr_payload offsets_payload =
        allocate_numpy_payload(n + 1, Bodo_CType_offset);
    offset_t* offsets = (offset_t*)offsets_payload.data;
    std::vector<const char*> tmp_store(n);
    size_t len = 0;
    for (Py_ssize_t i = 0; i < n; ++i) {
        offsets[i] = len;
        PyObject* s = PySequence_GetItem(obj, i);
        CHECK(s, "getting element failed");
        // Pandas stores NA as either None, nan, or pd.NA
        if (s == Py_None ||
            (PyFloat_Check(s) && std::isnan(PyFloat_AsDouble(s))) ||
            s == C_NA) {
            // leave null bit as 0
            tmp_store[i] = "";
        } else {
            // set null bit to 1 (Arrow bin-util.h)
            null_bitmap[i / 8] |= kBitmask[i % 8];
            Py_ssize_t size;
            if (is_bytes) {
                // check bytes
                CHECK(PyBytes_Check(s), "expecting a bytes object");
                size = PyBytes_GET_SIZE(s);
                // get buffer pointer
                tmp_store[i] = PyBytes_AS_STRING(s);
            } else {
                // check string
                CHECK(PyUnicode_Check(s), "expecting a string");
                // convert to UTF-8 and get size
                tmp_store[i] = PyUnicode_AsUTF8AndSize(s, &size);
            }
            CHECK(tmp_store[i], "string conversion failed");
            len += size;
        }
        Py_DECREF(s);
    }
    offsets[n] = len;

    numpy_arr_payload outbuf_payload =
        allocate_numpy_payload(len, Bodo_CTypes::UINT8);
    char* outbuf = outbuf_payload.data;
    for (Py_ssize_t i = 0; i < n; ++i) {
        memcpy(outbuf + offsets[i], tmp_store[i], offsets[i + 1] - offsets[i]);
    }

    Py_DECREF(C_NA);
    Py_DECREF(pd_mod);

    NRT_MemInfo* meminfo_array_item = alloc_array_item_arr_meminfo();
    array_item_arr_numpy_payload* payload =
        (array_item_arr_numpy_payload*)(meminfo_array_item->data);

    payload->n_arrays = n;

    // allocate data array
    // TODO: support non-numpy data
    payload->data = outbuf_payload;
    // TODO: support 64-bit offsets case
    payload->offsets = offsets_payload;
    payload->null_bitmap = null_bitmap_payload;

    return meminfo_array_item;
#undef CHECK
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
                                    offset_t* offsets, uint8_t* null_bitmap,
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
                                     const offset_t* offsets,
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
 * @brief create a numpy array of dict objects from a MapArrayType
 *
 * @param num_maps number of map in input array
 * @param key_data data buffer for keys
 * @param value_data data buffer for values
 * @param offsets offsets for different map key/value pairs
 * @param null_bitmap nulls buffer
 * @param key_dtype data types of keys
 * @param value_dtype data types of values
 * @return numpy array of dict objects
 */
void* np_array_from_map_array(int64_t num_maps, const char* key_data,
                              const char* value_data, const offset_t* offsets,
                              const uint8_t* null_bitmap,
                              Bodo_CTypes::CTypeEnum key_dtype,
                              Bodo_CTypes::CTypeEnum value_dtype) {
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        return NULL;                   \
    }

    // allocate array and get nan object
    npy_intp dims[] = {num_maps};
    PyObject* ret = PyArray_SimpleNew(1, dims, NPY_OBJECT);
    CHECK(ret, "allocating numpy array failed");
    int err;
    PyObject* np_mod = PyImport_ImportModule("numpy");
    CHECK(np_mod, "importing numpy module failed");
    PyObject* nan_obj = PyObject_GetAttrString(np_mod, "nan");
    CHECK(nan_obj, "getting np.nan failed");

    size_t curr_item = 0;
    // for each map
    for (int64_t i = 0; i < num_maps; ++i) {
        // set nan if item is NA
        auto p = PyArray_GETPTR1((PyArrayObject*)ret, i);
        CHECK(p, "getting offset in numpy array failed");
        if (is_na(null_bitmap, i)) {
            err = PyArray_SETITEM((PyArrayObject*)ret, (char*)p, nan_obj);
            CHECK(err == 0, "setting item in numpy array failed");
            continue;
        }

        // create dict and fill key/value items
        Py_ssize_t n_items = (Py_ssize_t)(offsets[i + 1] - offsets[i]);
        PyObject* dict = PyDict_New();

        for (Py_ssize_t j = 0; j < n_items; j++) {
            PyObject* key_obj =
                value_to_pyobject(key_data, curr_item, key_dtype);
            CHECK(key_obj, "creating Python int/float object failed");
            PyObject* value_obj =
                value_to_pyobject(value_data, curr_item, value_dtype);
            CHECK(value_obj, "creating Python int/float object failed");
            PyDict_SetItem(dict, key_obj, value_obj);
            Py_DECREF(key_obj);
            Py_DECREF(value_obj);
            curr_item++;
        }

        err = PyArray_SETITEM((PyArrayObject*)ret, (char*)p, dict);
        CHECK(err == 0, "setting item in numpy array failed");
        Py_DECREF(dict);
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
                                int32_t* dtypes, char** field_names,
                                bool is_tuple_array) {
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
            if (is_tuple_array) {
                CHECK(PyTuple_Check(s),
                      "invalid non-tuple element in tuple array");
            } else {
                CHECK(PyDict_Check(s),
                      "invalid non-dict element in struct array");
            }
            // set field data values
            for (Py_ssize_t j = 0; j < n_fields; j++) {
                PyObject* v;
                if (is_tuple_array)
                    v = PyTuple_GET_ITEM(s, j);  // returns borrowed reference
                else
                    v = PyDict_GetItemString(
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
 * @brief extract key/value data, offsets and null_bitmap values for map array
 * from an array of dict of values.
 *
 * @param map_arr_obj Python Sequence object, intended to be an array of
 * dicts.
 * @param key_data data buffer for keys
 * @param value_data data buffer for values
 * @param offsets offsets for different map key/value pairs
 * @param null_bitmap nulls buffer to be filled
 * @param key_dtype data types of keys
 * @param value_dtype data types of values
 */
void map_array_from_sequence(PyObject* map_arr_obj, char* key_data,
                             char* value_data, offset_t* offsets,
                             uint8_t* null_bitmap,
                             Bodo_CTypes::CTypeEnum key_dtype,
                             Bodo_CTypes::CTypeEnum value_dtype) {
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        return;                        \
    }

    // TODO: currently only a few types like float64 and int64 supported
    CHECK(PySequence_Check(map_arr_obj), "expecting a PySequence");
    CHECK(key_data && value_data && offsets && null_bitmap,
          "buffer arguments must not be NULL");

    // get pd.NA object to check for new NA kind
    PyObject* pd_mod = PyImport_ImportModule("pandas");
    CHECK(pd_mod, "importing pandas module failed");
    PyObject* C_NA = PyObject_GetAttrString(pd_mod, "NA");
    CHECK(C_NA, "getting pd.NA failed");

    Py_ssize_t n = PyObject_Size(map_arr_obj);

    int64_t curr_item_ind = 0;
    for (Py_ssize_t i = 0; i < n; ++i) {
        offsets[i] = curr_item_ind;
        PyObject* s = PySequence_GetItem(map_arr_obj, i);
        CHECK(s, "getting map array element failed");
        // Pandas stores NA as either None or nan
        if (s == Py_None ||
            (PyFloat_Check(s) && std::isnan(PyFloat_AsDouble(s))) ||
            s == C_NA) {
            // set null bit to 0
            SetBitTo(null_bitmap, i, 0);
        } else {
            // set null bit to 1
            null_bitmap[i / 8] |= kBitmask[i % 8];
            CHECK(PyDict_Check(s), "invalid non-dict element in map array");
            PyObject* key_list = PyDict_Keys(s);
            PyObject* value_list = PyDict_Values(s);
            Py_ssize_t n_items = PyObject_Size(key_list);
            // set field data values
            for (Py_ssize_t j = 0; j < n_items; j++) {
                PyObject* v1 =
                    PyList_GET_ITEM(key_list, j);  // returns borrowed reference
                PyObject* v2 = PyList_GET_ITEM(
                    value_list, j);  // returns borrowed reference
                copy_item_to_buffer(key_data, curr_item_ind, v1,
                                    (Bodo_CTypes::CTypeEnum)key_dtype);
                copy_item_to_buffer(value_data, curr_item_ind, v2,
                                    (Bodo_CTypes::CTypeEnum)value_dtype);
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
                                 char** field_names, bool is_tuple_array) {
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
        PyObject* d;
        if (is_tuple_array)
            d = PyTuple_New(n_fields);
        else
            d = PyDict_New();

        for (Py_ssize_t j = 0; j < n_fields; j++) {
            PyObject* s = value_to_pyobject(data[j], i,
                                            (Bodo_CTypes::CTypeEnum)dtypes[j]);
            CHECK(s, "creating Python int/float object failed");
            if (is_tuple_array) {
                PyTuple_SET_ITEM(d, j, s);  // steals s reference
            } else {
                PyDict_SetItemString(d, field_names[j], s);
                Py_DECREF(s);
            }
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
 * @brief return key list of dict
 *
 * @param obj dict object
 * @return list of keys object
 */
PyObject* dict_keys(PyObject* obj) { return PyDict_Keys(obj); }

/**
 * @brief return values list of dict
 *
 * @param obj dict object
 * @return list of values object
 */
PyObject* dict_values(PyObject* obj) { return PyDict_Values(obj); }

/**
 * @brief call PyDict_MergeFromSeq2() to fill a dict
 *
 * @param dict_obj dict object
 * @param seq2 iterator of key/value pairs
 */
void dict_merge_from_seq2(PyObject* dict_obj, PyObject* seq2) {
    int err = PyDict_MergeFromSeq2(dict_obj, seq2, 0);
    if (err != 0) {
        Bodo_PyErr_SetString(PyExc_RuntimeError, "PyDict_MergeFromSeq2 failed");
    }
}

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

/**
 * Extract the value from an array with the given row number.
 * This function does not check for NA values (these are handled elsewhere).
 * Currently we only include data types supported in general join conds
 * that require data beyond just data1 (as those are handled by a
 * different fast path).
 *
 * Returns a pointer the the start of the data in the given row.
 * output_size is updated to give the size of the returned data.
 */
char* array_info_getitem(array_info* arr, int64_t row_num,
                         offset_t* output_size) {
    bodo_array_type::arr_type_enum arr_type = arr->arr_type;
    if (arr_type == bodo_array_type::STRING) {
        // In the first case of STRING, we have to check the offsets.
        offset_t* offsets = (offset_t*)arr->data2;
        char* in_data1 = arr->data1;
        offset_t start_offset = offsets[row_num];
        offset_t end_offset = offsets[row_num + 1];
        offset_t size = end_offset - start_offset;
        *output_size = size;
        return in_data1 + start_offset;
    }
    throw std::runtime_error("array_info_getitem : Unsupported type");
}

char* array_info_getdata1(array_info* arr) { return arr->data1; }

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
    // Not covered by error handler
    PyObject_SetAttrString(m, "string_array_to_info",
                           PyLong_FromVoidPtr((void*)(&string_array_to_info)));
    // Not covered by error handler
    PyObject_SetAttrString(
        m, "dict_str_array_to_info",
        PyLong_FromVoidPtr((void*)(&dict_str_array_to_info)));
    PyObject_SetAttrString(m, "get_nested_info",
                           PyLong_FromVoidPtr((void*)(&get_nested_info)));
    PyObject_SetAttrString(
        m, "get_has_global_dictionary",
        PyLong_FromVoidPtr((void*)(&get_has_global_dictionary)));
    // Not covered by error handler
    PyObject_SetAttrString(m, "numpy_array_to_info",
                           PyLong_FromVoidPtr((void*)(&numpy_array_to_info)));
    // Not covered by error handler
    PyObject_SetAttrString(
        m, "categorical_array_to_info",
        PyLong_FromVoidPtr((void*)(&categorical_array_to_info)));
    // Not covered by error handler
    PyObject_SetAttrString(
        m, "nullable_array_to_info",
        PyLong_FromVoidPtr((void*)(&nullable_array_to_info)));
    PyObject_SetAttrString(
        m, "interval_array_to_info",
        PyLong_FromVoidPtr((void*)(&interval_array_to_info)));
    // Not covered by error handler
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
    PyObject_SetAttrString(
        m, "info_to_interval_array",
        PyLong_FromVoidPtr((void*)(&info_to_interval_array)));
    PyObject_SetAttrString(m, "alloc_numpy",
                           PyLong_FromVoidPtr((void*)(&alloc_numpy)));
    PyObject_SetAttrString(m, "alloc_string_array",
                           PyLong_FromVoidPtr((void*)(&alloc_string_array)));
    PyObject_SetAttrString(
        m, "arr_info_list_to_table",
        PyLong_FromVoidPtr((void*)(&arr_info_list_to_table)));
    // Not covered by error handler
    PyObject_SetAttrString(m, "info_from_table",
                           PyLong_FromVoidPtr((void*)(&info_from_table)));
    // Not covered by error handler
    PyObject_SetAttrString(
        m, "delete_info_decref_array",
        PyLong_FromVoidPtr((void*)(&delete_info_decref_array)));
    // Not covered by error handler
    PyObject_SetAttrString(
        m, "delete_table_decref_arrays",
        PyLong_FromVoidPtr((void*)(&delete_table_decref_arrays)));
    PyObject_SetAttrString(
        m, "decref_table_array",
        PyLong_FromVoidPtr((void*)(&decref_table_array)));
    // Not covered by error handler
    PyObject_SetAttrString(m, "delete_table",
                           PyLong_FromVoidPtr((void*)(&delete_table)));
    PyObject_SetAttrString(
        m, "shuffle_table",
        PyLong_FromVoidPtr((void*)(&shuffle_table_py_entrypt)));
    PyObject_SetAttrString(m, "get_shuffle_info",
                           PyLong_FromVoidPtr((void*)(&get_shuffle_info)));
    PyObject_SetAttrString(m, "delete_shuffle_info",
                           PyLong_FromVoidPtr((void*)(&delete_shuffle_info)));
    PyObject_SetAttrString(m, "reverse_shuffle_table",
                           PyLong_FromVoidPtr((void*)(&reverse_shuffle_table)));
    PyObject_SetAttrString(
        m, "shuffle_renormalization",
        PyLong_FromVoidPtr((void*)(&shuffle_renormalization_py_entrypt)));
    PyObject_SetAttrString(
        m, "shuffle_renormalization_group",
        PyLong_FromVoidPtr((void*)(&shuffle_renormalization_group_py_entrypt)));
    PyObject_SetAttrString(m, "hash_join_table",
                           PyLong_FromVoidPtr((void*)(&hash_join_table)));
    PyObject_SetAttrString(m, "sample_table",
                           PyLong_FromVoidPtr((void*)(&sample_table)));
    PyObject_SetAttrString(m, "sort_values_table",
                           PyLong_FromVoidPtr((void*)(&sort_values_table)));
    PyObject_SetAttrString(m, "drop_duplicates_table",
                           PyLong_FromVoidPtr((void*)(&drop_duplicates_table)));
    PyObject_SetAttrString(m, "groupby_and_aggregate",
                           PyLong_FromVoidPtr((void*)(&groupby_and_aggregate)));
    PyObject_SetAttrString(m, "get_groupby_labels",
                           PyLong_FromVoidPtr((void*)(&get_groupby_labels)));
    PyObject_SetAttrString(m, "array_isin",
                           PyLong_FromVoidPtr((void*)(&array_isin)));
    PyObject_SetAttrString(m, "get_search_regex",
                           PyLong_FromVoidPtr((void*)(&get_search_regex)));
    // Only uses C which cannot throw exceptions, so typical exception
    // handling is not required
    PyObject_SetAttrString(
        m, "count_total_elems_list_array",
        PyLong_FromVoidPtr((void*)(&count_total_elems_list_array)));
    // Only uses C which cannot throw exceptions, so typical exception
    // handling is not required
    PyObject_SetAttrString(
        m, "array_item_array_from_sequence",
        PyLong_FromVoidPtr((void*)(&array_item_array_from_sequence)));
    // Only uses C which cannot throw exceptions, so typical exception
    // handling is not required
    PyObject_SetAttrString(
        m, "struct_array_from_sequence",
        PyLong_FromVoidPtr((void*)(&struct_array_from_sequence)));
    // Only uses C which cannot throw exceptions, so typical exception
    // handling is not required
    PyObject_SetAttrString(
        m, "map_array_from_sequence",
        PyLong_FromVoidPtr((void*)(&map_array_from_sequence)));
    // Only uses C which cannot throw exceptions, so typical exception
    // handling is not required
    PyObject_SetAttrString(
        m, "string_array_from_sequence",
        PyLong_FromVoidPtr((void*)(&string_array_from_sequence)));
    // Only uses C which cannot throw exceptions, so typical exception
    // handling is not required
    PyObject_SetAttrString(
        m, "np_array_from_struct_array",
        PyLong_FromVoidPtr((void*)(&np_array_from_struct_array)));
    // Only uses C which cannot throw exceptions, so typical exception
    // handling is not required
    PyObject_SetAttrString(
        m, "np_array_from_array_item_array",
        PyLong_FromVoidPtr((void*)(&np_array_from_array_item_array)));
    // Only uses C which cannot throw exceptions, so typical exception
    // handling is not required
    PyObject_SetAttrString(
        m, "np_array_from_map_array",
        PyLong_FromVoidPtr((void*)(&np_array_from_map_array)));
    // Only uses C which cannot throw exceptions, so typical exception
    // handling is not required
    PyObject_SetAttrString(m, "array_getitem",
                           PyLong_FromVoidPtr((void*)(&array_getitem)));
    // Only uses C which cannot throw exceptions, so typical exception
    // handling is not required
    PyObject_SetAttrString(m, "list_check",
                           PyLong_FromVoidPtr((void*)(&list_check)));
    // Only uses C which cannot throw exceptions, so typical exception
    // handling is not required
    PyObject_SetAttrString(m, "dict_keys",
                           PyLong_FromVoidPtr((void*)(&dict_keys)));
    // Only uses C which cannot throw exceptions, so typical exception
    // handling is not required
    PyObject_SetAttrString(m, "dict_values",
                           PyLong_FromVoidPtr((void*)(&dict_values)));
    // This function calls PyErr_Set_String, but the function is called inside
    // box/unbox functions in Python, where we don't yet know how best to
    // detect and raise errors. Once we do, we should raise an error in Python
    // if this function calls PyErr_Set_String. TODO
    PyObject_SetAttrString(m, "dict_merge_from_seq2",
                           PyLong_FromVoidPtr((void*)(&dict_merge_from_seq2)));
    // This function is C, but it has components that can fail, in which case
    // we should call PyErr_Set_String and detect this and raise it in Python.
    // We currently don't know the best way to detect and raise exceptions
    // in box/unbox functions which is where this function is called.
    // Once we do, we should handle this appropriately. TODO
    PyObject_SetAttrString(m, "seq_getitem",
                           PyLong_FromVoidPtr((void*)(&seq_getitem)));
    // Only uses C which cannot throw exceptions, so typical exception
    // handling is not required
    PyObject_SetAttrString(m, "is_na_value",
                           PyLong_FromVoidPtr((void*)(&is_na_value)));
    // This function is C, but it has components that can fail, in which case
    // we should call PyErr_Set_String and detect this and raise an exception in
    // Python. We currently don't know the best way to detect and raise
    // exceptions in box/unbox functions which is where this function is called.
    // Once we do, we should handle this appropriately. TODO
    PyObject_SetAttrString(m, "is_pd_int_array",
                           PyLong_FromVoidPtr((void*)(&is_pd_int_array)));
    // This function is C, but it has components that can fail, in which case
    // we should call PyErr_Set_String and detect this and raise an exception in
    // Python. We currently don't know the best way to detect and raise
    // exceptions in box/unbox functions which is where this function is called.
    // Once we do, we should handle this appropriately. TODO
    PyObject_SetAttrString(
        m, "int_array_from_sequence",
        PyLong_FromVoidPtr((void*)(&int_array_from_sequence)));
    // Only uses C which cannot throw exceptions, so typical exception
    // handling is not required
    PyObject_SetAttrString(m, "get_stats_alloc",
                           PyLong_FromVoidPtr((void*)(&get_stats_alloc)));
    // Only uses C which cannot throw exceptions, so typical exception
    // handling is not required
    PyObject_SetAttrString(m, "get_stats_free",
                           PyLong_FromVoidPtr((void*)(&get_stats_free)));
    // Only uses C which cannot throw exceptions, so typical exception
    // handling is not required
    PyObject_SetAttrString(m, "get_stats_mi_alloc",
                           PyLong_FromVoidPtr((void*)(&get_stats_mi_alloc)));
    // Only uses C which cannot throw exceptions, so typical exception
    // handling is not required
    PyObject_SetAttrString(m, "get_stats_mi_free",
                           PyLong_FromVoidPtr((void*)(&get_stats_mi_free)));

    PyObject_SetAttrString(m, "array_info_getitem",
                           PyLong_FromVoidPtr((void*)(&array_info_getitem)));

    PyObject_SetAttrString(m, "array_info_getdata1",
                           PyLong_FromVoidPtr((void*)(&array_info_getdata1)));
    return m;
}
