#include <Python.h>
#include <datetime.h>
#include <iostream>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <arrow/api.h>
#include <arrow/python/pyarrow.h>
#include <numpy/arrayobject.h>

#include "_array_hash.h"
#include "_array_operations.h"
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_bodo_to_arrow.h"
#include "_dict_builder.h"
#include "_join.h"
#include "_shuffle.h"
#include "groupby/_groupby.h"

array_info* struct_array_to_info(int64_t n_fields, int64_t n_items,
                                 array_info** inner_arrays, char** field_names,
                                 NRT_MemInfo* null_bitmap) {
    std::vector<std::shared_ptr<array_info>> inner_arrs_vec(
        inner_arrays, inner_arrays + n_fields);
    std::vector<std::string> field_names_vec(field_names,
                                             field_names + n_fields);
    // Get length from an inner array in case n_items is set wrong since this
    // field is new and there could be gaps somewhere. See
    // https://github.com/bodo-ai/Bodo/pull/6891
    if (inner_arrs_vec.size() > 0) {
        n_items = inner_arrs_vec[0]->length;
    }

    // wrap meminfo in BodoBuffer (increfs meminfo also)
    int64_t n_bytes = arrow::bit_util::BytesForBits(n_items);
    std::shared_ptr<BodoBuffer> null_bitmap_buff = std::make_shared<BodoBuffer>(
        (uint8_t*)null_bitmap->data, n_bytes, null_bitmap);

    // Python is responsible for deleting pointer
    return new array_info(bodo_array_type::STRUCT, Bodo_CTypes::STRUCT, n_items,
                          {null_bitmap_buff}, inner_arrs_vec, 0, 0, 0, -1,
                          false, false, false, 0, field_names_vec);
}

array_info* array_item_array_to_info(uint64_t n_items, array_info* inner_array,
                                     NRT_MemInfo* offsets,
                                     NRT_MemInfo* null_bitmap) {
    // wrap meminfo in BodoBuffer (increfs meminfo also)
    std::shared_ptr<BodoBuffer> offsets_buff = std::make_shared<BodoBuffer>(
        (uint8_t*)offsets->data, (n_items + 1) * sizeof(offset_t), offsets);
    int64_t n_bytes = arrow::bit_util::BytesForBits(n_items);
    std::shared_ptr<BodoBuffer> null_bitmap_buff = std::make_shared<BodoBuffer>(
        (uint8_t*)null_bitmap->data, n_bytes, null_bitmap);

    // Python is responsible for deleting pointer
    return new array_info(bodo_array_type::ARRAY_ITEM, Bodo_CTypes::LIST,
                          n_items, {offsets_buff, null_bitmap_buff},
                          {std::shared_ptr<array_info>(inner_array)});
}

array_info* map_array_to_info(array_info* inner_array) {
    // Python is responsible for deleting pointer
    return new array_info(bodo_array_type::MAP, Bodo_CTypes::MAP,
                          inner_array->length, {},
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
                                   int32_t has_unique_local_dictionary,
                                   int64_t dict_id) {
    // Update the id for the string array
    // TODO(njriasan): Should we move this to string_array_to_info?
    str_arr->is_globally_replicated = has_global_dictionary;
    str_arr->is_locally_unique = has_unique_local_dictionary;
    str_arr->array_id = dict_id;
    // For now is_locally_sorted is only available and exposed in the C++
    // struct, so its kept as false.

    // Python is responsible for deleting
    return new array_info(bodo_array_type::DICT, Bodo_CTypes::STRING,
                          indices_arr->length, {},
                          {std::shared_ptr<array_info>(str_arr),
                           std::shared_ptr<array_info>(indices_arr)});
}

array_info* timestamp_tz_array_to_info(uint64_t n_items, char* timestamp_arr,
                                       char* offset_arr, char* null_bitmap,
                                       NRT_MemInfo* meminfo_ts,
                                       NRT_MemInfo* meminfo_offset,
                                       NRT_MemInfo* meminfo_bitmask) {
    // wrap meminfo in BodoBuffer (increfs meminfo also)
    std::shared_ptr<BodoBuffer> ts_buff = std::make_shared<BodoBuffer>(
        (uint8_t*)meminfo_ts->data,
        n_items * numpy_item_size[Bodo_CTypes::INT64], meminfo_ts);
    std::shared_ptr<BodoBuffer> offset_buff = std::make_shared<BodoBuffer>(
        (uint8_t*)meminfo_offset->data,
        n_items * numpy_item_size[Bodo_CTypes::INT16], meminfo_offset);
    int64_t n_bytes = arrow::bit_util::BytesForBits(n_items);
    std::shared_ptr<BodoBuffer> null_bitmap_buff = std::make_shared<BodoBuffer>(
        (uint8_t*)meminfo_bitmask->data, n_bytes, meminfo_bitmask);

    // Python is responsible for deleting
    return new array_info(bodo_array_type::arr_type_enum::TIMESTAMPTZ,
                          Bodo_CTypes::CTypeEnum::TIMESTAMPTZ, n_items,
                          {ts_buff, offset_buff, null_bitmap_buff}, {}, 0, 0, 0,
                          -1, false, false, false,
                          /*offset*/ timestamp_arr - (char*)meminfo_ts->data);
}

// Raw pointer since called from Python
int32_t get_has_global_dictionary(array_info* dict_arr) {
    return int32_t(dict_arr->child_arrays[0]->is_globally_replicated);
}

// Raw pointer since called from Python
int32_t get_has_unique_local_dictionary(array_info* dict_arr) {
    return int32_t(dict_arr->child_arrays[0]->is_locally_unique);
}

// Raw pointer since called from Python
int64_t get_dict_id(array_info* dict_arr) {
    return dict_arr->child_arrays[0]->array_id;
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
                          {data_buff}, {}, 0, 0, 0, -1, false, false, false,
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
        {data_buff}, {}, 0, 0, num_categories, -1, false, false, false,
        /*offset*/ data - (char*)meminfo->data);
}

array_info* null_array_to_info(uint64_t n_items) {
    // Null arrays are all null and just represented by a length. However,
    // most of the C++ code doesn't support null arrays, so we create an
    // all null boolean array.

    // TODO[BSE-433]: Avoid allocating the null bitmap and data array and create
    // a NULL CTYPE. This will require changes in the C++ code to support.
    return alloc_nullable_array_all_nulls(n_items, Bodo_CTypes::_BOOL)
        .release();
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
                          {data_buff, null_bitmap_buff}, {}, 0, 0, 0, -1, false,
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
        n_items, {data_buff, null_bitmap_buff}, {}, precision, scale, 0, -1,
        false, false, false, /*offset*/ data - (char*)meminfo->data);
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
        n_items, {data_buff, null_bitmap_buff}, {}, precision, 0, 0, -1, false,
        false, false, /*offset*/ data - (char*)meminfo->data);
}

array_info* info_to_array_item_array(array_info* info, int64_t* length,
                                     NRT_MemInfo** offsets_arr,
                                     NRT_MemInfo** null_bitmap_arr) {
    if (info->arr_type != bodo_array_type::ARRAY_ITEM) {
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
    *offsets_arr = offsets_meminfo;

    NRT_MemInfo* nulls_meminfo = info->buffers[1]->getMeminfo();
    incref_meminfo(nulls_meminfo);
    *null_bitmap_arr = nulls_meminfo;

    // Passing raw pointer to Python without giving ownership.
    // info_to_array() uses it to convert the child array recursively,
    // so the parent array shared_ptr stays alive in the process and
    // therefore the pointer stays valid.
    return info->child_arrays[0].get();
}

void info_to_struct_array(array_info* info, int64_t* n_items,
                          numpy_arr_payload* null_bitmap_arr) {
    if (info->arr_type != bodo_array_type::STRUCT) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "_array.cpp::info_to_struct_array: info_to_struct_array "
            "requires struct array input.");
        return;
    }
    *n_items = info->length;

    // create Numpy array for null_bitmap buffer as expected by
    // Python data model
    NRT_MemInfo* nulls_meminfo = info->buffers[0]->getMeminfo();
    incref_meminfo(nulls_meminfo);
    int64_t n_null_bytes = (info->length + 7) >> 3;
    int64_t null_itemsize = numpy_item_size[Bodo_CTypes::UINT8];
    *null_bitmap_arr = make_numpy_array_payload(
        nulls_meminfo, nullptr, n_null_bytes, null_itemsize,
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
                          int64_t* out_n_chars, NRT_MemInfo** data_arr,
                          NRT_MemInfo** offsets_arr,
                          NRT_MemInfo** null_bitmap_arr) {
    if (info->arr_type != bodo_array_type::STRING) {
        PyErr_SetString(
            PyExc_RuntimeError,
            ("_array.cpp::info_to_string_array: "
             "info_to_string_array requires string input. Got input " +
             GetArrType_as_string(info->arr_type))
                .c_str());
        return;
    }
    *length = info->length;
    *out_n_chars = info->n_sub_elems();

    // create Numpy arrays for char/offset/null_bitmap buffers as expected by
    // Python data model
    NRT_MemInfo* data_meminfo = info->buffers[0]->getMeminfo();
    incref_meminfo(data_meminfo);
    *data_arr = data_meminfo;

    NRT_MemInfo* offsets_meminfo = info->buffers[1]->getMeminfo();
    incref_meminfo(offsets_meminfo);
    *offsets_arr = offsets_meminfo;

    NRT_MemInfo* nulls_meminfo = info->buffers[2]->getMeminfo();
    incref_meminfo(nulls_meminfo);
    *null_bitmap_arr = nulls_meminfo;
}

void info_to_numpy_array(array_info* info, uint64_t* n_items, char** data,
                         NRT_MemInfo** meminfo, bool dict_as_int) {
    // arrow_array_to_bodo() always produces a nullable array but
    // Python may expect a Numpy array
    if ((info->arr_type != bodo_array_type::NUMPY) &&
        (info->arr_type != bodo_array_type::CATEGORICAL) &&
        (!dict_as_int || info->arr_type != bodo_array_type::DICT) &&
        (info->arr_type != bodo_array_type::NULLABLE_INT_BOOL)) {
        // TODO: print array type in the error
        PyErr_Format(PyExc_RuntimeError,
                     "_array.cpp::info_to_numpy_array: info_to_numpy_array "
                     "requires numpy input but got %s",
                     GetArrType_as_string(info->arr_type).c_str());
        return;
    }

    bool delete_info = false;

    // Treat DICT as categorical data and extract the indices.
    // Parquet reader uses DICT arrays for categorical data.
    if (info->arr_type == bodo_array_type::DICT) {
        std::shared_ptr<array_info> codes_info = info->child_arrays[1];

        // Convert data at NA positions to -1
        uint8_t* bitmap =
            reinterpret_cast<uint8_t*>(codes_info->null_bitmask());
        int32_t* codes_data = reinterpret_cast<int32_t*>(
            codes_info->data1<bodo_array_type::NULLABLE_INT_BOOL>());
        for (uint64_t i = 0; i < codes_info->length; ++i) {
            if (!GetBit(bitmap, i)) {
                codes_data[i] = -1;
            }
        }
        info = new array_info(*codes_info);
        delete_info = true;
    }

    *n_items = info->length;
    *data = info->data1();
    NRT_MemInfo* data_meminfo = info->buffers[0]->getMeminfo();
    incref_meminfo(data_meminfo);
    *meminfo = data_meminfo;

    // Delete the dict child array that was created in the DICT case above
    if (delete_info) {
        delete info;
    }
}

void info_to_null_array(array_info* info, uint64_t* n_items) {
    // TODO[BSE-433]: Replace with proper null array requirements once
    // they are integrated into C++.
    // Arrow NA type is converted to arr_type STRING.
    if (info->arr_type != bodo_array_type::NULLABLE_INT_BOOL &&
        info->arr_type != bodo_array_type::STRING) {
        PyErr_SetString(PyExc_RuntimeError,
                        "_array.cpp:: info_to_null_array: "
                        "info_to_null_array requires nullable input");
        return;
    }
    *n_items = info->length;
}

void info_to_nullable_array(array_info* info, uint64_t* n_items,
                            uint64_t* n_bytes, char** data, char** null_bitmap,
                            NRT_MemInfo** meminfo,
                            NRT_MemInfo** meminfo_bitmask) {
    std::cout << info->arr_type << " info arr type" << std::endl;
    if ((info->arr_type != bodo_array_type::NULLABLE_INT_BOOL) &&
        (info->arr_type != bodo_array_type::NUMPY)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "_array.cpp::info_to_nullable_array: "
                        "info_to_nullable_array requires nullable input");
        return;
    }

    // Handle Numpy arrays by creating a null bitmap. Necessary since dataframe
    // library uses Arrow schemas which are always nullable.
    if (info->arr_type == bodo_array_type::NUMPY) {
        // Allocate null bitmask and set all bits to 1
        size_t n_bytes = ((info->length + 7) >> 3);
        std::unique_ptr<BodoBuffer> buffer_bitmask =
            AllocateBodoBuffer(n_bytes * sizeof(uint8_t));
        memset(buffer_bitmask->mutable_data(), 0xff, n_bytes);
        // Keep the buffer in the same array_info struct for consistent memory
        // management (TODO: convert to nullable in upstream)
        info->buffers.push_back(std::move(buffer_bitmask));
    }

    if (info->dtype == Bodo_CTypes::DATETIME ||
        info->dtype == Bodo_CTypes::TIMEDELTA) {
        // Temporary fix to set invalid entries to NaT
        std::uint8_t* bitmap =
            reinterpret_cast<std::uint8_t*>(info->null_bitmask());
        std::int64_t* data = reinterpret_cast<std::int64_t*>(info->data1());
        for (std::uint64_t i = 0; i < info->length; ++i) {
            if (!GetBit(bitmap, i)) {
                data[i] = std::numeric_limits<std::int64_t>::min();
            }
        }
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

void info_to_timestamptz_array(array_info* info, uint64_t* n_items,
                               uint64_t* n_bytes, char** data_ts,
                               char** data_offset, char** null_bitmap,
                               NRT_MemInfo** meminfo_ts,
                               NRT_MemInfo** meminfo_offset,
                               NRT_MemInfo** meminfo_bitmask) {
    if (info->arr_type != bodo_array_type::TIMESTAMPTZ) {
        PyErr_SetString(PyExc_RuntimeError,
                        "_array.cpp::info_to_timestamptz_array: "
                        "info_to_nullable_array requires Timestamp TZ input");
        return;
    }
    if (info->dtype == Bodo_CTypes::TIMESTAMPTZ) {
        // Temporary fix to set invalid entries to NaT
        std::uint8_t* bitmap =
            reinterpret_cast<std::uint8_t*>(info->null_bitmask());
        std::int64_t* data = reinterpret_cast<std::int64_t*>(info->data1());
        for (std::uint64_t i = 0; i < info->length; ++i) {
            if (!GetBit(bitmap, i)) {
                data[i] = std::numeric_limits<std::int64_t>::min();
            }
        }
    }
    *n_items = info->length;
    *n_bytes = (info->length + 7) >> 3;
    *data_ts = info->data1();
    *data_offset = info->data2();
    *null_bitmap = info->null_bitmask();
    // give Python a reference
    NRT_MemInfo* data_ts_meminfo = info->buffers[0]->getMeminfo();
    incref_meminfo(data_ts_meminfo);
    *meminfo_ts = data_ts_meminfo;
    NRT_MemInfo* data_offset_meminfo = info->buffers[1]->getMeminfo();
    incref_meminfo(data_offset_meminfo);
    *meminfo_offset = data_offset_meminfo;
    NRT_MemInfo* nulls_meminfo = info->buffers[2]->getMeminfo();
    incref_meminfo(nulls_meminfo);
    *meminfo_bitmask = nulls_meminfo;
}

// returns raw pointer since called from Python
table_info* arr_info_list_to_table(array_info** arrs, int64_t n_arrs) {
    std::vector<std::shared_ptr<array_info>> columns(arrs, arrs + n_arrs);
    return new table_info(columns);
}

// Raw pointers since called from Python
void append_arr_info_list_to_cpp_table(table_info* table, array_info** arrs,
                                       int64_t n_arrs) {
    std::vector<std::shared_ptr<array_info>> columns(arrs, arrs + n_arrs);
    table->columns.insert(table->columns.end(), columns.begin(), columns.end());
}

// Raw pointers since called from Python
array_info* info_from_table(table_info* table, int64_t col_ind) {
    return new array_info(*table->columns[col_ind]);
}

/**
 * @brief create a Bodo string array from a PyArrow string array
 *
 * @param obj PyArrow string array
 * @return std::shared_ptr<array_info> Bodo string array
 */
std::shared_ptr<array_info> string_array_from_pyarrow(PyObject* pyarrow_arr) {
#undef CHECK
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

    return arrow_array_to_bodo(arrow_str_arr, nullptr);

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
                                int64_t* out_n_chars, NRT_MemInfo** data_arr,
                                NRT_MemInfo** offsets_arr,
                                NRT_MemInfo** null_bitmap_arr, int is_bytes) {
#undef CHECK
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
        std::shared_ptr<array_info> out_arr = alloc_array_top_level(
            0, 0, -1, bodo_array_type::STRING, Bodo_CTypes::STRING);
        info_to_string_array(out_arr.get(), length, out_n_chars, data_arr,
                             offsets_arr, null_bitmap_arr);
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

    PyObject* pd_arrow_ext_arr_obj =
        PyObject_GetAttrString(pd_arrays_obj, "ArrowExtensionArray");
    CHECK(pd_arrays_obj, "getting pd.arrays.ArrowExtensionArray failed");

    // isinstance(arr, ArrowStringArray)
    int is_arrow_str_arr = PyObject_IsInstance(obj, pd_arrow_str_arr_obj);
    CHECK(is_arrow_str_arr >= 0, "isinstance(obj, ArrowStringArray) fails");

    // isinstance(arr, ArrowExtensionArray)
    int is_arrow_ext_arr = PyObject_IsInstance(obj, pd_arrow_ext_arr_obj);
    CHECK(is_arrow_str_arr >= 0, "isinstance(obj, ArrowExtensionArray) fails");

    Py_DECREF(pandas_mod);
    Py_DECREF(pd_arrays_obj);
    Py_DECREF(pd_arrow_str_arr_obj);
    Py_DECREF(pd_arrow_ext_arr_obj);

    if (is_arrow_str_arr || (!is_bytes && is_arrow_ext_arr)) {
        // pyarrow_chunked_arr = obj._pa_array
        PyObject* pyarrow_chunked_arr =
            PyObject_GetAttrString(obj, "_pa_array");
        CHECK(pyarrow_chunked_arr, "getting obj._pa_array failed");

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
        info_to_string_array(arr.get(), length, out_n_chars, data_arr,
                             offsets_arr, null_bitmap_arr);
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
    bodo::vector<const char*> tmp_store(n);
    // We need to keep the references alive until we copy the data from the
    // tmp_store
    bodo::vector<const PyObject*> tmp_unicode_refs(n);
    size_t len = 0;
    for (Py_ssize_t i = 0; i < n; ++i) {
        offsets[i] = len;
        PyObject* s = PySequence_GetItem(obj, i);
        tmp_unicode_refs[i] = s;
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
    }
    offsets[n] = len;

    numpy_arr_payload outbuf_payload =
        allocate_numpy_payload(len, Bodo_CTypes::UINT8);
    char* outbuf = outbuf_payload.data;
    for (Py_ssize_t i = 0; i < n; ++i) {
        memcpy(outbuf + offsets[i], tmp_store[i], offsets[i + 1] - offsets[i]);
        Py_DECREF(tmp_unicode_refs[i]);
    }

    Py_DECREF(C_NA);
    Py_DECREF(pd_mod);

    *out_n_chars = len;
    *data_arr = outbuf_payload.meminfo;
    *offsets_arr = offsets_payload.meminfo;
    *null_bitmap_arr = null_bitmap_payload.meminfo;
    return;
#undef CHECK
}

/**
 * @brief create a Bodo array from a PyArrow array PyObject
 *
 * @param obj PyArrow array PyObject
 * @return std::shared_ptr<array_info> Bodo array
 */
array_info* bodo_array_from_pyarrow_py_entry(PyObject* pyarrow_arr) {
#undef CHECK
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        throw std::runtime_error(msg); \
    }

    try {
        // https://arrow.apache.org/docs/python/integration/extending.html
        CHECK(!arrow::py::import_pyarrow(), "importing pyarrow failed");

        // unwrap C++ Arrow array from pyarrow array
        std::shared_ptr<arrow::Array> arrow_arr;
        auto res = arrow::py::unwrap_array(pyarrow_arr);
        CHECK_ARROW_AND_ASSIGN(res, "unwrap_array(pyarrow_arr)", arrow_arr);
        CHECK(arrow_arr->offset() == 0,
              "only Arrow arrays with zero offset supported");

        return new array_info(*arrow_array_to_bodo(arrow_arr, nullptr));
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }

#undef CHECK
}

/**
 * @brief Create Pandas ArrowExtensionArray from Bodo array
 *
 * @param arr input Bodo array
 * @return void* Pandas ArrowStringArray object
 */
void* pd_pyarrow_array_from_bodo_array_py_entry(array_info* arr) {
#undef CHECK
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        throw std::runtime_error(msg); \
    }

    try {
        // convert to Arrow array
        arrow::TimeUnit::type time_unit = arrow::TimeUnit::NANO;
        auto arrow_arr = bodo_array_to_arrow(
            bodo::BufferPool::DefaultPtr(), std::shared_ptr<array_info>(arr),
            false /*convert_timedelta_to_int64*/, "", time_unit,
            false, /*downcast_time_ns_to_us*/
            bodo::default_buffer_memory_manager());

        // https://arrow.apache.org/docs/python/integration/extending.html
        CHECK(!arrow::py::import_pyarrow(), "importing pyarrow failed");

        // convert Arrow C++ to PyArrow
        PyObject* pyarrow_arr = arrow::py::wrap_array(arrow_arr);

        // call pd.arrays.ArrowStringArray(pyarrow_arr) which avoids copy
        PyObject* pd_mod = PyImport_ImportModule("pandas");
        CHECK(pd_mod, "importing pandas module failed");
        PyObject* pd_arrays_mod = PyObject_GetAttrString(pd_mod, "arrays");
        CHECK(pd_arrays_mod, "importing pandas.arrays module failed");

        PyObject* arr_obj = PyObject_CallMethod(
            pd_arrays_mod, "ArrowExtensionArray", "O", pyarrow_arr);

        Py_DECREF(pd_mod);
        Py_DECREF(pd_arrays_mod);
        Py_DECREF(pyarrow_arr);
        return arr_obj;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
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
    } else {
        std::cerr << "data type " << dtype
                  << " not supported for unboxing array(item) array."
                  << std::endl;
    }
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
    } else {
        std::cerr << "data type " << dtype
                  << " not supported for boxing array(item) array."
                  << std::endl;
    }
    return nullptr;
}

/**
 * @brief call PyArray_GETITEM() of Numpy C-API
 *
 * @param arr array object
 * @param p pointer in array object
 * @return PyObject* value returned by getitem
 */
PyObject* array_getitem(PyArrayObject* arr, const char* p) {
#undef CHECK
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
#undef CHECK
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
        offset_t* offsets = (offset_t*)arr->data2<bodo_array_type::STRING>();
        char* in_data1 = arr->data1<bodo_array_type::STRING>();
        offset_t start_offset = offsets[row_num];
        offset_t end_offset = offsets[row_num + 1];
        offset_t size = end_offset - start_offset;
        *output_size = size;
        return in_data1 + start_offset;
    }
    throw std::runtime_error("array_info_getitem : Unsupported type");
}

char* array_info_getdata1(array_info* arr) { return arr->data1(); }

/// @brief Wrapper Function to Test if Array Unpinning Works
/// as expected in our tests. See
/// test_memory.py::test_array_unpinned
void array_info_unpin(array_info* arr) { arr->unpin(); }

/**
 * @brief Python entrypoint for retrieve table.
 *
 * @param in_table The table to filter.
 * @param row_bitmask_arr The bitmask whether row is included. Expects
 * array_type=NULLABLE_INTO_BOOL, dtype=BOOL.
 * @return table_info* the filtered table after applying the row bitmask
 */
table_info* retrieve_table_py_entry(table_info* in_table,
                                    array_info* row_bitmask_arr) {
    try {
        assert(row_bitmask_arr->arr_type ==
                   bodo_array_type::NULLABLE_INT_BOOL &&
               row_bitmask_arr->dtype == Bodo_CTypes::_BOOL);
        auto row_bitmask_ptr = std::unique_ptr<array_info>(row_bitmask_arr);
        auto in_table_ptr = std::unique_ptr<table_info>(in_table);
        return new table_info(*RetrieveTable(std::move(in_table_ptr),
                                             std::move(row_bitmask_ptr)));
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

PyMODINIT_FUNC PyInit_array_ext(void) {
    PyObject* m;
    MOD_DEF(m, "array_ext", "No docs", nullptr);
    if (m == nullptr) {
        return nullptr;
    }

    // init datetime APIs
    PyDateTime_IMPORT;

    // init numpy
    import_array();

    bodo_common_init();

    // DEC_MOD_METHOD(string_array_to_info);
    SetAttrStringFromVoidPtr(m, array_item_array_to_info);
    SetAttrStringFromVoidPtr(m, struct_array_to_info);
    SetAttrStringFromVoidPtr(m, map_array_to_info);
    // Not covered by error handler
    SetAttrStringFromVoidPtr(m, string_array_to_info);
    // Not covered by error handler
    SetAttrStringFromVoidPtr(m, dict_str_array_to_info);
    SetAttrStringFromVoidPtr(m, timestamp_tz_array_to_info);
    SetAttrStringFromVoidPtr(m, get_has_global_dictionary);
    SetAttrStringFromVoidPtr(m, get_has_unique_local_dictionary);
    SetAttrStringFromVoidPtr(m, get_dict_id);
    // Not covered by error handler
    SetAttrStringFromVoidPtr(m, numpy_array_to_info);
    // Not covered by error handler
    SetAttrStringFromVoidPtr(m, categorical_array_to_info);
    // Not covered by error handler
    SetAttrStringFromVoidPtr(m, null_array_to_info);
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
    SetAttrStringFromVoidPtr(m, info_to_null_array);
    SetAttrStringFromVoidPtr(m, info_to_nullable_array);
    SetAttrStringFromVoidPtr(m, info_to_interval_array);
    SetAttrStringFromVoidPtr(m, info_to_timestamptz_array);
    SetAttrStringFromVoidPtr(m, alloc_numpy);
    SetAttrStringFromVoidPtr(m, alloc_string_array);
    SetAttrStringFromVoidPtr(m, arr_info_list_to_table);
    SetAttrStringFromVoidPtr(m, append_arr_info_list_to_cpp_table);
    // Not covered by error handler
    SetAttrStringFromVoidPtr(m, info_from_table);
    // Not covered by error handler
    SetAttrStringFromVoidPtr(m, delete_info);
    // Not covered by error handler
    SetAttrStringFromVoidPtr(m, delete_table);
    SetAttrStringFromVoidPtr(m, cpp_table_map_to_list);
    SetAttrStringFromVoidPtr(m, shuffle_table_py_entrypt);
    SetAttrStringFromVoidPtr(m, get_shuffle_info);
    SetAttrStringFromVoidPtr(m, delete_shuffle_info);
    SetAttrStringFromVoidPtr(m, reverse_shuffle_table);
    SetAttrStringFromVoidPtr(m, shuffle_renormalization_py_entrypt);
    SetAttrStringFromVoidPtr(m, shuffle_renormalization_group_py_entrypt);
    SetAttrStringFromVoidPtr(m, hash_join_table);
    SetAttrStringFromVoidPtr(m, nested_loop_join_table);
    SetAttrStringFromVoidPtr(m, interval_join_table);
    SetAttrStringFromVoidPtr(m, sample_table_py_entry);
    SetAttrStringFromVoidPtr(m, sort_values_table_py_entry);
    SetAttrStringFromVoidPtr(m, sort_table_for_interval_join_py_entrypoint);
    SetAttrStringFromVoidPtr(m, drop_duplicates_table_py_entry);
    SetAttrStringFromVoidPtr(m, union_tables);
    SetAttrStringFromVoidPtr(m, concat_tables_py_entry);
    SetAttrStringFromVoidPtr(m, groupby_and_aggregate_py_entry);
    SetAttrStringFromVoidPtr(m, drop_duplicates_local_dictionary_py_entry);
    SetAttrStringFromVoidPtr(m, get_groupby_labels_py_entry);
    SetAttrStringFromVoidPtr(m, array_isin_py_entry);
    SetAttrStringFromVoidPtr(m, get_search_regex_py_entry);
    SetAttrStringFromVoidPtr(m, get_replace_regex_py_entry);
    SetAttrStringFromVoidPtr(m, get_replace_regex_dict_state_py_entry);

    // Functions in the section below only use C which cannot throw exceptions,
    // so typical exception handling is not required
    SetAttrStringFromVoidPtr(m, bodo_array_from_pyarrow_py_entry);
    SetAttrStringFromVoidPtr(m, pd_pyarrow_array_from_bodo_array_py_entry);
    SetAttrStringFromVoidPtr(m, string_array_from_sequence);
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
    SetAttrStringFromVoidPtr(m, get_stats_alloc);
    SetAttrStringFromVoidPtr(m, get_stats_free);
    SetAttrStringFromVoidPtr(m, get_stats_mi_alloc);
    SetAttrStringFromVoidPtr(m, get_stats_mi_free);
    SetAttrStringFromVoidPtr(m, array_info_getitem);
    SetAttrStringFromVoidPtr(m, array_info_getdata1);
    SetAttrStringFromVoidPtr(m, array_info_unpin);
    // End section of functions which only use C and cannot throw exceptions

    // C++ Cache functions for Like Kernel with dictionary encoded inputs
    SetAttrStringFromVoidPtr(m, alloc_like_kernel_cache);
    SetAttrStringFromVoidPtr(m, add_to_like_kernel_cache);
    SetAttrStringFromVoidPtr(m, check_like_kernel_cache);
    SetAttrStringFromVoidPtr(m, dealloc_like_kernel_cache);
    SetAttrStringFromVoidPtr(m, NRT_MemInfo_alloc_safe_aligned);

    SetAttrStringFromVoidPtr(m, retrieve_table_py_entry);

    return m;
}
