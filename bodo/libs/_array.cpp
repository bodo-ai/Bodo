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
#include <arrow/python/pyarrow.h>
#include <numpy/arrayobject.h>
#include "_array_hash.h"
#include "_array_operations.h"
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_bodo_to_arrow.h"
#include "_distributed.h"
#include "_groupby.h"
#include "_join.h"
#include "_shuffle.h"

array_info* struct_array_to_info(int64_t n_fields, array_info** inner_arrays,
                                 char** field_names, NRT_MemInfo* null_bitmap) {
    std::vector<std::shared_ptr<array_info>> inner_arrs_vec(
        inner_arrays, inner_arrays + n_fields);
    std::vector<std::string> field_names_vec(field_names,
                                             field_names + n_fields);
    // get length from an inner array
    int64_t n_items = 0;
    if (inner_arrs_vec.size() > 0) {
        n_items = inner_arrs_vec[0]->length;
    }

    // wrap meminfo in BodoBuffer (increfs meminfo also)
    int64_t n_bytes = arrow::bit_util::BytesForBits(n_items);
    std::shared_ptr<BodoBuffer> null_bitmap_buff = std::make_shared<BodoBuffer>(
        (uint8_t*)null_bitmap->data, n_bytes, null_bitmap);

    // Python is responsible for deleting pointer
    return new array_info(bodo_array_type::STRUCT, Bodo_CTypes::STRUCT, n_items,
                          {null_bitmap_buff}, inner_arrs_vec, 0, 0, 0, false,
                          false, false, 0, field_names_vec);
}

array_info* array_item_array_to_info(uint64_t n_items, array_info* inner_array,
                                     NRT_MemInfo* offsets,
                                     NRT_MemInfo* null_bitmap) {
    bodo_array_type::arr_type_enum array_type = bodo_array_type::ARRAY_ITEM;
    Bodo_CTypes::CTypeEnum dtype = Bodo_CTypes::LIST;
    if (inner_array->arr_type == bodo_array_type::STRING &&
        inner_array->dtype == Bodo_CTypes::STRING) {
        array_type = bodo_array_type::LIST_STRING;
        dtype = Bodo_CTypes::LIST_STRING;
    }

    // wrap meminfo in BodoBuffer (increfs meminfo also)
    std::shared_ptr<BodoBuffer> offsets_buff = std::make_shared<BodoBuffer>(
        (uint8_t*)offsets->data, (n_items + 1) * sizeof(offset_t), offsets);
    int64_t n_bytes = arrow::bit_util::BytesForBits(n_items);
    std::shared_ptr<BodoBuffer> null_bitmap_buff = std::make_shared<BodoBuffer>(
        (uint8_t*)null_bitmap->data, n_bytes, null_bitmap);

    // Python is responsible for deleting pointer
    return new array_info(array_type, dtype, n_items,
                          {offsets_buff, null_bitmap_buff},
                          {std::shared_ptr<array_info>(inner_array)});
}

array_info* string_array_to_info(uint64_t n_items, NRT_MemInfo* data,
                                 NRT_MemInfo* offsets, NRT_MemInfo* null_bitmap,
                                 int is_bytes) {
    // TODO: better memory management of struct, meminfo refcount?
    auto dtype = Bodo_CTypes::STRING;
    if (is_bytes) {
        dtype = Bodo_CTypes::BINARY;
    }

    // wrap meminfo in BodoBuffer (increfs meminfo also)
    int64_t n_chars = ((offset_t*)offsets->data)[n_items];
    std::shared_ptr<BodoBuffer> data_buff =
        std::make_shared<BodoBuffer>((uint8_t*)data->data, n_chars, data);
    std::shared_ptr<BodoBuffer> offsets_buff = std::make_shared<BodoBuffer>(
        (uint8_t*)offsets->data, (n_items + 1) * sizeof(offset_t), offsets);
    int64_t n_bytes = arrow::bit_util::BytesForBits(n_items);
    std::shared_ptr<BodoBuffer> null_bitmap_buff = std::make_shared<BodoBuffer>(
        (uint8_t*)null_bitmap->data, n_bytes, null_bitmap);

    // Python is responsible for deleting
    return new array_info(bodo_array_type::STRING, dtype, n_items,
                          {data_buff, offsets_buff, null_bitmap_buff});
}

array_info* dict_str_array_to_info(array_info* str_arr, array_info* indices_arr,
                                   int32_t has_global_dictionary,
                                   int32_t has_deduped_local_dictionary) {
    // For now has_sorted_dictionary is only available and exposed in the C++
    // struct, so we set it to false

    // Python is responsible for deleting
    return new array_info(bodo_array_type::DICT, Bodo_CTypes::STRING,
                          indices_arr->length, {},
                          {std::shared_ptr<array_info>(str_arr),
                           std::shared_ptr<array_info>(indices_arr)},
                          0, 0, 0, bool(has_global_dictionary),
                          bool(has_deduped_local_dictionary), false);
}

// Raw pointer since called from Python
int32_t get_has_global_dictionary(array_info* dict_arr) {
    return int32_t(dict_arr->has_global_dictionary);
}

// Raw pointer since called from Python
int32_t get_has_deduped_local_dictionary(array_info* dict_arr) {
    return int32_t(dict_arr->has_deduped_local_dictionary);
}

array_info* numpy_array_to_info(uint64_t n_items, char* data, int typ_enum,
                                NRT_MemInfo* meminfo) {
    // Numpy array slicing creates views on MemInfo buffers with an offset
    // from the MemInfo data pointer. For example, A[2:4] will have
    // an offset of 16 bytes for int64 arrays (and n_items=2).
    // We use pointer arithmetic to get the offset since not explicitly stored
    // in Numpy struct.

    std::shared_ptr<BodoBuffer> data_buff = std::make_shared<BodoBuffer>(
        (uint8_t*)meminfo->data, n_items * numpy_item_size[typ_enum], meminfo);

    // Python is responsible for deleting
    return new array_info(bodo_array_type::NUMPY,
                          (Bodo_CTypes::CTypeEnum)typ_enum, n_items,
                          {data_buff}, {}, 0, 0, 0, false, false, false,
                          /*offset*/ data - (char*)meminfo->data);
}

array_info* categorical_array_to_info(uint64_t n_items, char* data,
                                      int typ_enum, int64_t num_categories,
                                      NRT_MemInfo* meminfo) {
    std::shared_ptr<BodoBuffer> data_buff = std::make_shared<BodoBuffer>(
        (uint8_t*)meminfo->data, n_items * numpy_item_size[typ_enum], meminfo);
    // Python is responsible for deleting
    return new array_info(
        bodo_array_type::CATEGORICAL, (Bodo_CTypes::CTypeEnum)typ_enum, n_items,
        {data_buff}, {}, 0, 0, num_categories, false, false, false,
        /*offset*/ data - (char*)meminfo->data);
}

array_info* nullable_array_to_info(uint64_t n_items, char* data, int typ_enum,
                                   char* null_bitmap, NRT_MemInfo* meminfo,
                                   NRT_MemInfo* meminfo_bitmask) {
    // wrap meminfo in BodoBuffer (increfs meminfo also)
    std::shared_ptr<BodoBuffer> data_buff = std::make_shared<BodoBuffer>(
        (uint8_t*)meminfo->data, n_items * numpy_item_size[typ_enum], meminfo);
    int64_t n_bytes = arrow::bit_util::BytesForBits(n_items);
    std::shared_ptr<BodoBuffer> null_bitmap_buff = std::make_shared<BodoBuffer>(
        (uint8_t*)meminfo_bitmask->data, n_bytes, meminfo_bitmask);

    // Python is responsible for deleting
    return new array_info(bodo_array_type::NULLABLE_INT_BOOL,
                          (Bodo_CTypes::CTypeEnum)typ_enum, n_items,
                          {data_buff, null_bitmap_buff}, {}, 0, 0, 0, false,
                          false, false, /*offset*/ data - (char*)meminfo->data);
}

array_info* interval_array_to_info(uint64_t n_items, char* left_data,
                                   char* right_data, int typ_enum,
                                   NRT_MemInfo* left_meminfo,
                                   NRT_MemInfo* right_meminfo) {
    if (left_data - (char*)left_meminfo->data != 0 ||
        right_data - (char*)right_meminfo->data != 0) {
        throw std::runtime_error(
            "interval_array_to_info: offsets not supported for interval array");
    }
    std::shared_ptr<BodoBuffer> left_buff = std::make_shared<BodoBuffer>(
        (uint8_t*)left_meminfo->data, n_items * numpy_item_size[typ_enum],
        left_meminfo);
    std::shared_ptr<BodoBuffer> right_buff = std::make_shared<BodoBuffer>(
        (uint8_t*)right_meminfo->data, n_items * numpy_item_size[typ_enum],
        right_meminfo);
    // Python is responsible for deleting
    return new array_info(bodo_array_type::INTERVAL,
                          (Bodo_CTypes::CTypeEnum)typ_enum, n_items,
                          {left_buff, right_buff});
}

array_info* decimal_array_to_info(uint64_t n_items, char* data, int typ_enum,
                                  char* null_bitmap, NRT_MemInfo* meminfo,
                                  NRT_MemInfo* meminfo_bitmask,
                                  int32_t precision, int32_t scale) {
    // wrap meminfo in BodoBuffer (increfs meminfo also)
    std::shared_ptr<BodoBuffer> data_buff = std::make_shared<BodoBuffer>(
        (uint8_t*)meminfo->data, n_items * numpy_item_size[typ_enum], meminfo);
    int64_t n_bytes = arrow::bit_util::BytesForBits(n_items);
    std::shared_ptr<BodoBuffer> null_bitmap_buff = std::make_shared<BodoBuffer>(
        (uint8_t*)meminfo_bitmask->data, n_bytes, meminfo_bitmask);

    // Python is responsible for deleting
    return new array_info(
        bodo_array_type::NULLABLE_INT_BOOL, (Bodo_CTypes::CTypeEnum)typ_enum,
        n_items, {data_buff, null_bitmap_buff}, {}, precision, scale, 0, false,
        false, false, /*offset*/ data - (char*)meminfo->data);
}

array_info* time_array_to_info(uint64_t n_items, char* data, int typ_enum,
                               char* null_bitmap, NRT_MemInfo* meminfo,
                               NRT_MemInfo* meminfo_bitmask,
                               int32_t precision) {
    // wrap meminfo in BodoBuffer (increfs meminfo also)
    std::shared_ptr<BodoBuffer> data_buff = std::make_shared<BodoBuffer>(
        (uint8_t*)meminfo->data, n_items * numpy_item_size[typ_enum], meminfo);
    int64_t n_bytes = arrow::bit_util::BytesForBits(n_items);
    std::shared_ptr<BodoBuffer> null_bitmap_buff = std::make_shared<BodoBuffer>(
        (uint8_t*)meminfo_bitmask->data, n_bytes, meminfo_bitmask);

    // Python is responsible for deleting
    return new array_info(
        bodo_array_type::NULLABLE_INT_BOOL, (Bodo_CTypes::CTypeEnum)typ_enum,
        n_items, {data_buff, null_bitmap_buff}, {}, precision, 0, 0, false,
        false, false, /*offset*/ data - (char*)meminfo->data);
}

array_info* info_to_array_item_array(array_info* info, int64_t* length,
                                     numpy_arr_payload* offsets_arr,
                                     numpy_arr_payload* null_bitmap_arr) {
    if (info->arr_type != bodo_array_type::LIST_STRING &&
        info->arr_type != bodo_array_type::ARRAY_ITEM) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "_array.cpp::info_to_array_item_array: info_to_array_item_array "
            "requires array(array(item)) input.");
        return nullptr;
    }
    *length = info->length;

    // create Numpy arrays for offset/null_bitmap buffers as expected by
    // Python data model
    NRT_MemInfo* offsets_meminfo = info->buffers[0]->getMeminfo();
    incref_meminfo(offsets_meminfo);
    int64_t n_offsets = info->length + 1;
    int64_t offset_itemsize = numpy_item_size[Bodo_CType_offset];
    *offsets_arr = make_numpy_array_payload(
        offsets_meminfo, NULL, n_offsets, offset_itemsize,
        (char*)offsets_meminfo->data, n_offsets, offset_itemsize);

    NRT_MemInfo* nulls_meminfo = info->buffers[1]->getMeminfo();
    incref_meminfo(nulls_meminfo);
    int64_t n_null_bytes = (info->length + 7) >> 3;
    int64_t null_itemsize = numpy_item_size[Bodo_CTypes::UINT8];
    *null_bitmap_arr = make_numpy_array_payload(
        nulls_meminfo, NULL, n_null_bytes, null_itemsize,
        (char*)nulls_meminfo->data, n_null_bytes, null_itemsize);

    // Passing raw pointer to Python without giving ownership.
    // info_to_array() uses it to convert the child array recursively,
    // so the parent array shared_ptr stays alive in the process and
    // therefore the pointer stays valid.
    return info->child_arrays[0].get();
}

void info_to_struct_array(array_info* info,
                          numpy_arr_payload* null_bitmap_arr) {
    if (info->arr_type != bodo_array_type::STRUCT) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "_array.cpp::info_to_struct_array: info_to_struct_array "
            "requires struct array input.");
        return;
    }

    // create Numpy array for null_bitmap buffer as expected by
    // Python data model
    NRT_MemInfo* nulls_meminfo = info->buffers[0]->getMeminfo();
    incref_meminfo(nulls_meminfo);
    int64_t n_null_bytes = (info->length + 7) >> 3;
    int64_t null_itemsize = numpy_item_size[Bodo_CTypes::UINT8];
    *null_bitmap_arr = make_numpy_array_payload(
        nulls_meminfo, NULL, n_null_bytes, null_itemsize,
        (char*)nulls_meminfo->data, n_null_bytes, null_itemsize);
}

/**
 * @brief return array_info* for child array
 * Using raw pointers since called from Python
 *
 * @param in_info input array (must be nested)
 * @param i index of child array
 * @return array_info* child array
 */
array_info* get_child_info(array_info* in_info, int64_t i) {
    if (in_info->child_arrays.size() <= (size_t)i) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "_array.cpp::get_child_info: invalid child array index ");
        return nullptr;
    }
    return in_info->child_arrays[i].get();
}

void info_to_string_array(array_info* info, int64_t* length,
                          numpy_arr_payload* data_arr,
                          numpy_arr_payload* offsets_arr,
                          numpy_arr_payload* null_bitmap_arr) {
    if (info->arr_type != bodo_array_type::STRING) {
        PyErr_SetString(PyExc_RuntimeError,
                        "_array.cpp::info_to_string_array: "
                        "info_to_string_array requires string input.");
        return;
    }
    *length = info->length;

    // create Numpy arrays for char/offset/null_bitmap buffers as expected by
    // Python data model
    NRT_MemInfo* data_meminfo = info->buffers[0]->getMeminfo();
    incref_meminfo(data_meminfo);
    int64_t n_chars = info->n_sub_elems();
    int64_t char_itemsize = numpy_item_size[Bodo_CTypes::INT8];
    *data_arr = make_numpy_array_payload(
        data_meminfo, NULL, n_chars, char_itemsize, (char*)data_meminfo->data,
        n_chars, char_itemsize);

    NRT_MemInfo* offsets_meminfo = info->buffers[1]->getMeminfo();
    incref_meminfo(offsets_meminfo);
    int64_t n_offsets = info->length + 1;
    int64_t offset_itemsize = numpy_item_size[Bodo_CType_offset];
    *offsets_arr = make_numpy_array_payload(
        offsets_meminfo, NULL, n_offsets, offset_itemsize,
        (char*)offsets_meminfo->data, n_offsets, offset_itemsize);

    NRT_MemInfo* nulls_meminfo = info->buffers[2]->getMeminfo();
    incref_meminfo(nulls_meminfo);
    int64_t n_null_bytes = (info->length + 7) >> 3;
    int64_t null_itemsize = numpy_item_size[Bodo_CTypes::UINT8];
    *null_bitmap_arr = make_numpy_array_payload(
        nulls_meminfo, NULL, n_null_bytes, null_itemsize,
        (char*)nulls_meminfo->data, n_null_bytes, null_itemsize);
}

void info_to_numpy_array(array_info* info, uint64_t* n_items, char** data,
                         NRT_MemInfo** meminfo) {
    // arrow_array_to_bodo() always produces a nullable array but
    // Python may expect a Numpy array
    if ((info->arr_type != bodo_array_type::NUMPY) &&
        (info->arr_type != bodo_array_type::CATEGORICAL) &&
        (info->arr_type != bodo_array_type::NULLABLE_INT_BOOL)) {
        // TODO: print array type in the error
        PyErr_SetString(PyExc_RuntimeError,
                        "_array.cpp::info_to_numpy_array: info_to_numpy_array "
                        "requires numpy input.");
        return;
    }

    *n_items = info->length;
    *data = info->data1();
    NRT_MemInfo* data_meminfo = info->buffers[0]->getMeminfo();
    incref_meminfo(data_meminfo);
    *meminfo = data_meminfo;
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
    *data = info->data1();
    *null_bitmap = info->null_bitmask();
    // give Python a reference
    NRT_MemInfo* data_meminfo = info->buffers[0]->getMeminfo();
    incref_meminfo(data_meminfo);
    *meminfo = data_meminfo;
    NRT_MemInfo* nulls_meminfo = info->buffers[1]->getMeminfo();
    incref_meminfo(nulls_meminfo);
    *meminfo_bitmask = nulls_meminfo;
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
    *left_data = info->data1();
    *right_data = info->data2();
    NRT_MemInfo* l_meminfo = info->buffers[0]->getMeminfo();
    incref_meminfo(l_meminfo);
    *left_meminfo = l_meminfo;
    NRT_MemInfo* r_meminfo = info->buffers[1]->getMeminfo();
    incref_meminfo(r_meminfo);
    *right_meminfo = r_meminfo;
}

// returns raw pointer since called from Python
table_info* arr_info_list_to_table(array_info** arrs, int64_t n_arrs) {
    std::vector<std::shared_ptr<array_info>> columns(arrs, arrs + n_arrs);
    return new table_info(columns);
}

// Raw pointers since called from Python
array_info* info_from_table(table_info* table, int64_t col_ind) {
    return new array_info(*table->columns[col_ind]);
}

#define CHECK_ARROW_AND_ASSIGN(res, msg, lhs)                              \
    if (!(res.status().ok())) {                                            \
        std::string err_msg = std::string("Error in arrow ") + msg + " " + \
                              res.status().ToString();                     \
        std::cerr << msg << std::endl;                                     \
    }                                                                      \
    lhs = std::move(res).ValueOrDie();

/**
 * @brief create a Bodo string array from a PyArrow string array
 *
 * @param obj PyArrow string array
 * @return std::shared_ptr<array_info> Bodo string array
 */
std::shared_ptr<array_info> string_array_from_pyarrow(PyObject* pyarrow_arr) {
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        return nullptr;                \
    }

    // https://arrow.apache.org/docs/python/integration/extending.html
    CHECK(!arrow::py::import_pyarrow(), "importing pyarrow failed");

    // unwrap C++ Arrow array from pyarrow array
    std::shared_ptr<arrow::Array> arrow_arr;
    auto res = arrow::py::unwrap_array(pyarrow_arr);
    CHECK_ARROW_AND_ASSIGN(res, "unwrap_array(pyarrow_arr)", arrow_arr);
    CHECK(arrow_arr->offset() == 0,
          "only Arrow arrays with zero offset supported");
    std::shared_ptr<arrow::LargeStringArray> arrow_str_arr =
        std::static_pointer_cast<arrow::LargeStringArray>(arrow_arr);

    return arrow_array_to_bodo(arrow_str_arr);

#undef CHECK
}

/**
 * @brief create a concatenated string and offset table from a numpy array of
 * strings
 *
 * @param obj numpy array of strings, pd.arrays.StringArray, or
 * pd.arrays.ArrowStringArray
 * @param[out] data_arr pointer to Numpy array for characters in output string
 * array
 * @param[out] offsets_arr pointer to Numpy array for offsets in output string
 * array
 * @param[out] null_bitmap_arr pointer to Numpy array for null bitmap in output
 * string array
 * @param is_bytes whether the contents are bytes objects instead of str
 */
void string_array_from_sequence(PyObject* obj, int64_t* length,
                                numpy_arr_payload* data_arr,
                                numpy_arr_payload* offsets_arr,
                                numpy_arr_payload* null_bitmap_arr,
                                int is_bytes) {
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        return;                        \
    }

    CHECK(PySequence_Check(obj), "expecting a PySequence");

    Py_ssize_t n = PyObject_Size(obj);
    *length = n;
    if (n == 0) {
        // empty sequence, this is not an error, need to set size
        std::shared_ptr<array_info> out_arr = alloc_array(
            0, 0, -1, bodo_array_type::STRING, Bodo_CTypes::STRING, 0, 0);
        info_to_string_array(out_arr.get(), length, data_arr, offsets_arr,
                             null_bitmap_arr);
        return;
    }

    // check if obj is ArrowStringArray to unbox it properly
    // pd.arrays.ArrowStringArray
    PyObject* pandas_mod = PyImport_ImportModule("pandas");
    CHECK(pandas_mod, "importing pandas module failed");
    PyObject* pd_arrays_obj = PyObject_GetAttrString(pandas_mod, "arrays");
    CHECK(pd_arrays_obj, "getting pd.arrays failed");
    PyObject* pd_arrow_str_arr_obj =
        PyObject_GetAttrString(pd_arrays_obj, "ArrowStringArray");
    CHECK(pd_arrays_obj, "getting pd.arrays.ArrowStringArray failed");

    // isinstance(arr, ArrowStringArray)
    int is_arrow_str_arr = PyObject_IsInstance(obj, pd_arrow_str_arr_obj);
    CHECK(is_arrow_str_arr >= 0, "isinstance(obj, ArrowStringArray) fails");

    Py_DECREF(pandas_mod);
    Py_DECREF(pd_arrays_obj);
    Py_DECREF(pd_arrow_str_arr_obj);

    if (is_arrow_str_arr) {
        // pyarrow_chunked_arr = obj._data
        PyObject* pyarrow_chunked_arr = PyObject_GetAttrString(obj, "_data");
        CHECK(pyarrow_chunked_arr, "getting obj._data failed");

        // pyarrow_arr = pyarrow_chunked_arr.combine_chunks()
        PyObject* pyarrow_arr =
            PyObject_CallMethod(pyarrow_chunked_arr, "combine_chunks", "");
        CHECK(pyarrow_arr, "array.combine_chunks() failed");

        // pyarrow_arr_large_str = pyarrow_arr.cast("large_string")
        // necessary since Pandas may have regular "string" with 32-bit offsets
        PyObject* pyarrow_arr_large_str =
            PyObject_CallMethod(pyarrow_arr, "cast", "s", "large_string");
        CHECK(pyarrow_arr_large_str, "array.cast(\"large_string\") failed");

        std::shared_ptr<array_info> arr =
            string_array_from_pyarrow(pyarrow_arr_large_str);
        info_to_string_array(arr.get(), length, data_arr, offsets_arr,
                             null_bitmap_arr);
        Py_DECREF(pyarrow_chunked_arr);
        Py_DECREF(pyarrow_arr);
        Py_DECREF(pyarrow_arr_large_str);
        return;
    }

    // allocate null bitmap
    int64_t n_bytes = arrow::bit_util::BytesForBits(n);
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

    *data_arr = outbuf_payload;
    *offsets_arr = offsets_payload;
    *null_bitmap_arr = null_bitmap_payload;
    return;
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
 * @brief check if object is a pd.FloatArray
 *
 * @param arr Python object to check
 * @return int 1 if object is a pd.FloatArray, 0 otherwise
 */
int is_pd_float_array(PyObject* arr) {
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        PyGILState_Release(gilstate);  \
        return false;                  \
    }

    auto gilstate = PyGILState_Ensure();
    // pd.arrays.FloatingArray
    PyObject* pd_mod = PyImport_ImportModule("pandas");
    CHECK(pd_mod, "importing pandas module failed");
    PyObject* pd_arrays_obj = PyObject_GetAttrString(pd_mod, "arrays");
    CHECK(pd_arrays_obj, "getting pd.arrays failed");
    PyObject* pd_arrays_float_arr_obj =
        PyObject_GetAttrString(pd_arrays_obj, "FloatingArray");
    CHECK(pd_arrays_obj, "getting pd.arrays.FloatingArray failed");

    // isinstance(arr, FloatingArray)
    int ret = PyObject_IsInstance(arr, pd_arrays_float_arr_obj);
    CHECK(ret >= 0, "isinstance fails");

    Py_DECREF(pd_mod);
    Py_DECREF(pd_arrays_obj);
    Py_DECREF(pd_arrays_float_arr_obj);
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
 * @brief unbox object array into native data and null_bitmap of native nullable
 * float array
 *
 * @param arr_obj object array with float or NA values
 * @param data native float data array of output array
 * @param null_bitmap null bitmap of output array
 */
void float_array_from_sequence(PyObject* arr_obj, double* data,
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
        CHECK(s, "getting float array element failed");
        // Pandas stores NA as either None or nan
        if (s == Py_None ||
            (PyFloat_Check(s) && std::isnan(PyFloat_AsDouble(s))) ||
            s == C_NA) {
            // set null bit to 0
            SetBitTo(null_bitmap, i, 0);
        } else {
            // set null bit to 1
            null_bitmap[i / 8] |= kBitmask[i % 8];
            data[i] = PyFloat_AsDouble(s);
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
        offset_t* offsets = (offset_t*)arr->data2();
        char* in_data1 = arr->data1();
        offset_t start_offset = offsets[row_num];
        offset_t end_offset = offsets[row_num + 1];
        offset_t size = end_offset - start_offset;
        *output_size = size;
        return in_data1 + start_offset;
    }
    throw std::runtime_error("array_info_getitem : Unsupported type");
}

char* array_info_getdata1(array_info* arr) { return arr->data1(); }

PyMODINIT_FUNC PyInit_array_ext(void) {
    PyObject* m;
    MOD_DEF(m, "array_ext", "No docs", NULL);
    if (m == NULL)
        return NULL;

    // init datetime APIs
    PyDateTime_IMPORT;

    // init numpy
    import_array();

    bodo_common_init();

    // DEC_MOD_METHOD(string_array_to_info);
    SetAttrStringFromVoidPtr(m, array_item_array_to_info);
    SetAttrStringFromVoidPtr(m, struct_array_to_info);
    // Not covered by error handler
    SetAttrStringFromVoidPtr(m, string_array_to_info);
    // Not covered by error handler
    SetAttrStringFromVoidPtr(m, dict_str_array_to_info);
    SetAttrStringFromVoidPtr(m, get_has_global_dictionary);
    SetAttrStringFromVoidPtr(m, get_has_deduped_local_dictionary);
    // Not covered by error handler
    SetAttrStringFromVoidPtr(m, numpy_array_to_info);
    // Not covered by error handler
    SetAttrStringFromVoidPtr(m, categorical_array_to_info);
    // Not covered by error handler
    SetAttrStringFromVoidPtr(m, nullable_array_to_info);
    SetAttrStringFromVoidPtr(m, interval_array_to_info);
    // Not covered by error handler
    SetAttrStringFromVoidPtr(m, decimal_array_to_info);
    SetAttrStringFromVoidPtr(m, time_array_to_info);
    SetAttrStringFromVoidPtr(m, info_to_string_array);
    SetAttrStringFromVoidPtr(m, info_to_array_item_array);
    SetAttrStringFromVoidPtr(m, info_to_struct_array);
    SetAttrStringFromVoidPtr(m, get_child_info);
    SetAttrStringFromVoidPtr(m, info_to_numpy_array);
    SetAttrStringFromVoidPtr(m, info_to_nullable_array);
    SetAttrStringFromVoidPtr(m, info_to_interval_array);
    SetAttrStringFromVoidPtr(m, alloc_numpy);
    SetAttrStringFromVoidPtr(m, alloc_string_array);
    SetAttrStringFromVoidPtr(m, arr_info_list_to_table);
    // Not covered by error handler
    SetAttrStringFromVoidPtr(m, info_from_table);
    // Not covered by error handler
    SetAttrStringFromVoidPtr(m, delete_info);
    // Not covered by error handler
    SetAttrStringFromVoidPtr(m, delete_table);
    SetAttrStringFromVoidPtr(m, shuffle_table_py_entrypt);
    SetAttrStringFromVoidPtr(m, get_shuffle_info);
    SetAttrStringFromVoidPtr(m, delete_shuffle_info);
    SetAttrStringFromVoidPtr(m, reverse_shuffle_table);
    SetAttrStringFromVoidPtr(m, shuffle_renormalization_py_entrypt);
    SetAttrStringFromVoidPtr(m, shuffle_renormalization_group_py_entrypt);
    SetAttrStringFromVoidPtr(m, hash_join_table);
    SetAttrStringFromVoidPtr(m, cross_join_table);
    SetAttrStringFromVoidPtr(m, interval_join_table);
    SetAttrStringFromVoidPtr(m, sample_table_py_entry);
    SetAttrStringFromVoidPtr(m, sort_values_table_py_entry);
    SetAttrStringFromVoidPtr(m, sort_table_for_interval_join_py_entrypoint);
    SetAttrStringFromVoidPtr(m, drop_duplicates_table_py_entry);
    SetAttrStringFromVoidPtr(m, union_tables);
    SetAttrStringFromVoidPtr(m, groupby_and_aggregate);
    SetAttrStringFromVoidPtr(m, drop_duplicates_local_dictionary_py_entry);
    SetAttrStringFromVoidPtr(m, get_groupby_labels_py_entry);
    SetAttrStringFromVoidPtr(m, array_isin_py_entry);
    SetAttrStringFromVoidPtr(m, get_search_regex_py_entry);
    SetAttrStringFromVoidPtr(m, get_replace_regex_py_entry);

    // Functions in the section below only use C which cannot throw exceptions,
    // so typical exception handling is not required
    SetAttrStringFromVoidPtr(m, count_total_elems_list_array);
    SetAttrStringFromVoidPtr(m, array_item_array_from_sequence);
    SetAttrStringFromVoidPtr(m, struct_array_from_sequence);
    SetAttrStringFromVoidPtr(m, map_array_from_sequence);
    SetAttrStringFromVoidPtr(m, string_array_from_sequence);
    SetAttrStringFromVoidPtr(m, np_array_from_struct_array);
    SetAttrStringFromVoidPtr(m, np_array_from_array_item_array);
    SetAttrStringFromVoidPtr(m, np_array_from_map_array);
    SetAttrStringFromVoidPtr(m, array_getitem);
    SetAttrStringFromVoidPtr(m, list_check);
    SetAttrStringFromVoidPtr(m, dict_keys);
    SetAttrStringFromVoidPtr(m, dict_values);
    // This function calls PyErr_Set_String, but the function is called inside
    // box/unbox functions in Python, where we don't yet know how best to
    // detect and raise errors. Once we do, we should raise an error in Python
    // if this function calls PyErr_Set_String. TODO
    SetAttrStringFromVoidPtr(m, dict_merge_from_seq2);
    // This function is C, but it has components that can fail, in which case
    // we should call PyErr_Set_String and detect this and raise it in Python.
    // We currently don't know the best way to detect and raise exceptions
    // in box/unbox functions which is where this function is called.
    // Once we do, we should handle this appropriately. TODO
    SetAttrStringFromVoidPtr(m, seq_getitem);
    SetAttrStringFromVoidPtr(m, is_na_value);
    // This function is C, but it has components that can fail, in which case
    // we should call PyErr_Set_String and detect this and raise an exception in
    // Python. We currently don't know the best way to detect and raise
    // exceptions in box/unbox functions which is where this function is called.
    // Once we do, we should handle this appropriately. TODO
    SetAttrStringFromVoidPtr(m, is_pd_int_array);
    // This function is C, but it has components that can fail, in which case
    // we should call PyErr_Set_String and detect this and raise an exception in
    // Python. We currently don't know the best way to detect and raise
    // exceptions in box/unbox functions which is where this function is called.
    // Once we do, we should handle this appropriately. TODO
    SetAttrStringFromVoidPtr(m, is_pd_float_array);
    // This function is C, but it has components that can fail, in which case
    // we should call PyErr_Set_String and detect this and raise an exception in
    // Python. We currently don't know the best way to detect and raise
    // exceptions in box/unbox functions which is where this function is called.
    // Once we do, we should handle this appropriately. TODO
    SetAttrStringFromVoidPtr(m, int_array_from_sequence);
    // This function is C, but it has components that can fail, in which case
    // we should call PyErr_Set_String and detect this and raise an exception in
    // Python. We currently don't know the best way to detect and raise
    // exceptions in box/unbox functions which is where this function is called.
    // Once we do, we should handle this appropriately. TODO
    SetAttrStringFromVoidPtr(m, float_array_from_sequence);
    SetAttrStringFromVoidPtr(m, get_stats_alloc);
    SetAttrStringFromVoidPtr(m, get_stats_free);
    SetAttrStringFromVoidPtr(m, get_stats_mi_alloc);
    SetAttrStringFromVoidPtr(m, get_stats_mi_free);
    SetAttrStringFromVoidPtr(m, array_info_getitem);
    SetAttrStringFromVoidPtr(m, array_info_getdata1);
    // End section of functions which only use C and cannot throw exceptions

    // C++ Cache functions for Like Kernel with dictionary encoded inputs
    SetAttrStringFromVoidPtr(m, alloc_like_kernel_cache);
    SetAttrStringFromVoidPtr(m, add_to_like_kernel_cache);
    SetAttrStringFromVoidPtr(m, check_like_kernel_cache);
    SetAttrStringFromVoidPtr(m, dealloc_like_kernel_cache);
    SetAttrStringFromVoidPtr(m, NRT_MemInfo_alloc_safe_aligned);

    return m;
}
