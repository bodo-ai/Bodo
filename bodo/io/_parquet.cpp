// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include <Python.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include "mpi.h"

#if _MSC_VER >= 1900
#undef timezone
#endif
#include "arrow/record_batch.h"
#include "parquet/arrow/reader.h"

#include "../libs/_bodo_common.h"
#include "../libs/_datetime_ext.h"
#include "_fs_io.h"
#include "_parquet_reader.h"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include "parquet/arrow/writer.h"

/**
 * This holds the file readers and other information that this process needs
 * to read its chunk of a Parquet dataset.
 */
struct DatasetReader {
    /// FileReaders, only for the files that this process has to read
    std::vector<std::shared_ptr<parquet::arrow::FileReader>> readers;
    /// Starting row for first file (readers[0])
    int start_row_first_file = 0;
    /// Total number of rows this process has to read (across readers)
    int count = 0;
};

/**
 * Get DatasetReader which contains only the file readers that this process
 * needs.
 * @param file_name : file or directory of parquet files
 * @param is_parallel : true if processes will read chunks of the dataset
 */
DatasetReader *get_dataset_reader(char *file_name, bool is_parallel);
void del_dataset_reader(DatasetReader *reader);

int64_t pq_get_size(DatasetReader *reader, int64_t column_idx);
int64_t pq_read(DatasetReader *reader, int64_t column_idx, uint8_t *out_data,
                int out_dtype, uint8_t *out_nulls = nullptr);
int pq_read_string(DatasetReader *reader, int64_t column_idx,
                   uint32_t **out_offsets, uint8_t **out_data,
                   uint8_t **out_nulls);
int pq_read_list_string(DatasetReader *reader, int64_t column_idx,
                        uint32_t **out_offsets, uint32_t **index_offsets,
                        uint8_t **out_data, uint8_t **out_nulls);
int pq_read_list_item(DatasetReader *reader, int64_t column_idx, int out_dtype,
                      array_info **out_offsets, array_info **out_data,
                      array_info **out_nulls);

void pack_null_bitmap(uint8_t **out_nulls, std::vector<bool> &null_vec,
                      int64_t n_all_vals);
void pq_write(const char *filename, const table_info *table,
              const array_info *col_names, const array_info *index,
              bool write_index, const char *metadata, const char *compression,
              bool parallel, bool write_rangeindex_to_metadata, const int start,
              const int stop, const int step, const char *name);

#define CHECK(expr, msg)                                           \
    if (!(expr)) {                                                 \
        std::cerr << "Error in parquet I/O: " << msg << std::endl; \
        return;                                                    \
    }

#define CHECK_ARROW(expr, msg)                                            \
    if (!(expr.ok())) {                                                   \
        std::cerr << "Error in arrow parquet I/O: " << msg << " " << expr \
                  << std::endl;                                           \
        return;                                                           \
    }

#define CHECK_ARROW_AND_ASSIGN(res, msg, lhs) \
    CHECK_ARROW(res.status(), msg)            \
    lhs = std::move(res).ValueOrDie();

PyMODINIT_FUNC PyInit_parquet_cpp(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, "parquet_cpp", "No docs", -1, NULL,
    };
    m = PyModule_Create(&moduledef);
    if (m == NULL) return NULL;

    bodo_common_init();

    PyObject_SetAttrString(m, "get_dataset_reader",
                           PyLong_FromVoidPtr((void *)(&get_dataset_reader)));
    PyObject_SetAttrString(m, "del_dataset_reader",
                           PyLong_FromVoidPtr((void *)(&del_dataset_reader)));
    PyObject_SetAttrString(m, "pq_read", PyLong_FromVoidPtr((void *)(&pq_read)));
    PyObject_SetAttrString(m, "pq_get_size",
                           PyLong_FromVoidPtr((void *)(&pq_get_size)));
    PyObject_SetAttrString(m, "pq_read_string",
                           PyLong_FromVoidPtr((void *)(&pq_read_string)));
    PyObject_SetAttrString(m, "pq_read_list_string",
                           PyLong_FromVoidPtr((void *)(&pq_read_list_string)));
    PyObject_SetAttrString(m, "pq_read_list_item",
                           PyLong_FromVoidPtr((void *)(&pq_read_list_item)));
    PyObject_SetAttrString(m, "pq_write",
                           PyLong_FromVoidPtr((void *)(&pq_write)));

    return m;
}

DatasetReader *get_dataset_reader(char *file_name, bool parallel) {
#define PYERR_CHECK(expr, msg)         \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        PyGILState_Release(gilstate);  \
        return ds_reader;              \
    }

    auto gilstate = PyGILState_Ensure();

    DatasetReader *ds_reader = new DatasetReader();

    // import bodo.io.parquet_pio
    PyObject *pq_mod = PyImport_ImportModule("bodo.io.parquet_pio");

    // ds = bodo.io.parquet_pio.get_parquet_dataset(file_name, parallel)
    PyObject *ds = PyObject_CallMethod(pq_mod, "get_parquet_dataset", "si",
                                       file_name, int(parallel));
    PYERR_CHECK(!PyErr_Occurred(),
                "Python error during Parquet dataset metadata")
    Py_DECREF(pq_mod);

    // total_rows = ds._bodo_total_rows
    PyObject *total_rows_py = PyObject_GetAttrString(ds, "_bodo_total_rows");
    int64_t total_rows = PyLong_AsLongLong(total_rows_py);
    Py_DECREF(total_rows_py);

    // all_pieces = ds.pieces
    PyObject *all_pieces = PyObject_GetAttrString(ds, "pieces");
    Py_DECREF(ds);

    // iterate through pieces next
    PyObject *iterator = PyObject_GetIter(all_pieces);
    Py_DECREF(all_pieces);
    PyObject *piece;

    if (iterator == NULL) {
        PyGILState_Release(gilstate);
        return ds_reader;
    }

    if (!parallel) {
        // the process will read the whole dataset
        ds_reader->count = total_rows;

        if (total_rows > 0) {
            // open readers for every piece
            while ((piece = PyIter_Next(iterator))) {
                PyObject *num_rows_piece_py =
                    PyObject_GetAttrString(piece, "_bodo_num_rows");
                int64_t num_rows_piece = PyLong_AsLongLong(num_rows_piece_py);
                Py_DECREF(num_rows_piece_py);
                if (num_rows_piece > 0) {
                    // p = piece.path
                    PyObject *p = PyObject_GetAttrString(piece, "path");
                    const char *c_path = PyUnicode_AsUTF8(p);
                    std::shared_ptr<parquet::arrow::FileReader> arrow_reader;
                    // open and store file reader for this piece
                    pq_init_reader(c_path, &arrow_reader);
                    ds_reader->readers.push_back(arrow_reader);
                    Py_DECREF(p);
                }
                Py_DECREF(piece);
            }
        }

        Py_DECREF(iterator);

        PYERR_CHECK(!PyErr_Occurred(),
                    "Python error during Parquet dataset metadata")
        PyGILState_Release(gilstate);
        return ds_reader;
    }

    // is parallel (this process will read a chunk of dataset)

    // calculate the portion of rows that this process needs to read
    size_t rank = dist_get_rank();
    size_t nranks = dist_get_size();
    int64_t start_row_global = dist_get_start(total_rows, nranks, rank);
    ds_reader->count = dist_get_node_portion(total_rows, nranks, rank);

    // open file readers only for the pieces that correspond to my chunk
    if (ds_reader->count > 0) {
        int64_t count_rows =
            0;  // total number of rows of all the pieces we iterate through
        int64_t num_rows_my_files =
            0;  // number of rows in opened files (excluding any rows in the
                // first file that will be skipped if the process starts
                // reading in the middle of the file)
        while ((piece = PyIter_Next(iterator))) {
            PyObject *num_rows_piece_py =
                PyObject_GetAttrString(piece, "_bodo_num_rows");
            int64_t num_rows_piece = PyLong_AsLongLong(num_rows_piece_py);
            Py_DECREF(num_rows_piece_py);

            // we skip all initial pieces whose total row count is less than
            // start_row_global (first row of my chunk). after that, we open
            // file readers for all subsequent pieces until the number of rows
            // in opened pieces is greater or equal to number of rows in my
            // chunk
            if ((num_rows_piece > 0) &&
                (start_row_global < count_rows + num_rows_piece)) {
                if (ds_reader->readers.size() == 0) {
                    ds_reader->start_row_first_file =
                        start_row_global - count_rows;
                    num_rows_my_files +=
                        num_rows_piece - ds_reader->start_row_first_file;
                } else {
                    num_rows_my_files += num_rows_piece;
                }

                // open and store file reader for this piece
                PyObject *p = PyObject_GetAttrString(piece, "path");
                const char *c_path = PyUnicode_AsUTF8(p);
                std::shared_ptr<parquet::arrow::FileReader> arrow_reader;
                pq_init_reader(c_path, &arrow_reader);
                ds_reader->readers.push_back(arrow_reader);
                Py_DECREF(p);
            }

            Py_DECREF(piece);

            count_rows += num_rows_piece;
            // finish when number of rows of opened files covers my chunk
            if (num_rows_my_files >= ds_reader->count) break;
        }
    }

    Py_DECREF(iterator);

    PYERR_CHECK(!PyErr_Occurred(),
                "Python error during Parquet dataset metadata")
    PyGILState_Release(gilstate);
    return ds_reader;
#undef PYERR_CHECK
}

void del_dataset_reader(DatasetReader *reader) { delete reader; }

int64_t pq_get_size(DatasetReader *reader, int64_t column_idx) {
    return reader->count;
}

int64_t pq_read(DatasetReader *ds_reader, int64_t column_idx, uint8_t *out_data,
                int out_dtype, uint8_t *out_nulls) {
    if (ds_reader->count == 0) return 0;

    int64_t start = ds_reader->start_row_first_file;

    int64_t read_rows = 0;  // rows read so far
    int dtype_size = numpy_item_size[out_dtype];
    for (auto file_reader : ds_reader->readers) {
        int64_t file_size = pq_get_size_single_file(file_reader, column_idx);
        int64_t rows_to_read =
            std::min(ds_reader->count - read_rows, file_size - start);

        pq_read_single_file(file_reader, column_idx,
                            out_data + read_rows * dtype_size, out_dtype, start,
                            rows_to_read, out_nulls, read_rows);
        read_rows += rows_to_read;
        start = 0;  // start becomes 0 after reading non-empty first chunk
    }
    return 0;
}

int pq_read_string(DatasetReader *ds_reader, int64_t column_idx,
                   uint32_t **out_offsets, uint8_t **out_data,
                   uint8_t **out_nulls) {
    if (ds_reader->count == 0) return 0;

    int64_t start = ds_reader->start_row_first_file;

    int64_t n_all_vals = 0;
    std::vector<uint32_t> offset_vec;
    std::vector<uint8_t> data_vec;
    std::vector<bool> null_vec;
    int64_t last_offset = 0;
    int64_t read_rows = 0;  // rows read so far
    for (auto file_reader : ds_reader->readers) {
        int64_t file_size = pq_get_size_single_file(file_reader, column_idx);
        int64_t rows_to_read =
            std::min(ds_reader->count - read_rows, file_size - start);

        pq_read_string_single_file(file_reader, column_idx, start, rows_to_read,
                                   &offset_vec, &data_vec, &null_vec);

        size_t size = offset_vec.size();
        for (int64_t i = 1; i <= rows_to_read + 1; i++)
            offset_vec[size - i] += last_offset;
        last_offset = offset_vec[size - 1];
        offset_vec.pop_back();
        n_all_vals += rows_to_read;

        read_rows += rows_to_read;
        start = 0;  // start becomes 0 after reading non-empty first chunk
    }
    offset_vec.push_back(last_offset);

    *out_offsets = new uint32_t[offset_vec.size()];
    *out_data = new uint8_t[data_vec.size()];

    memcpy(*out_offsets, offset_vec.data(),
           offset_vec.size() * sizeof(uint32_t));
    memcpy(*out_data, data_vec.data(), data_vec.size());
    pack_null_bitmap(out_nulls, null_vec, n_all_vals);
    return n_all_vals;
}

int pq_read_list_string(DatasetReader *ds_reader, int64_t column_idx,
                        uint32_t **out_offsets, uint32_t **index_offsets,
                        uint8_t **out_data, uint8_t **out_nulls) {
    if (ds_reader->count == 0) return 0;

    int64_t start = ds_reader->start_row_first_file;

    int64_t n_all_vals = 0;
    std::vector<uint32_t> index_offset_vec;
    std::vector<uint32_t> offset_vec;
    std::vector<uint8_t> data_vec;
    std::vector<bool> null_vec;
    int64_t last_str_offset = 0;
    int64_t last_index_offset = 0;
    int64_t read_rows = 0;  // rows read so far
    for (auto file_reader : ds_reader->readers) {
        int64_t file_size = pq_get_size_single_file(file_reader, column_idx);
        int64_t rows_to_read =
            std::min(ds_reader->count - read_rows, file_size - start);

        int64_t n_strings = pq_read_list_string_single_file(
            file_reader, column_idx, start, rows_to_read, &offset_vec,
            &index_offset_vec, &data_vec, &null_vec);

        size_t size = offset_vec.size();
        for (int64_t i = 1; i <= n_strings + 1; i++)
            offset_vec[size - i] += last_str_offset;
        last_str_offset = offset_vec.back();
        offset_vec.pop_back();

        size = index_offset_vec.size();
        for (int64_t i = 1; i <= rows_to_read + 1; i++)
            index_offset_vec[size - i] += last_index_offset;
        last_index_offset = index_offset_vec[size - 1];
        index_offset_vec.pop_back();

        n_all_vals += rows_to_read;

        read_rows += rows_to_read;
        start = 0;  // start becomes 0 after reading non-empty first chunk
    }
    offset_vec.push_back(last_str_offset);
    index_offset_vec.push_back(last_index_offset);

    *out_offsets = new uint32_t[offset_vec.size()];
    *index_offsets = new uint32_t[index_offset_vec.size()];
    *out_data = new uint8_t[data_vec.size()];

    memcpy(*out_offsets, offset_vec.data(),
           offset_vec.size() * sizeof(uint32_t));
    memcpy(*index_offsets, index_offset_vec.data(),
           index_offset_vec.size() * sizeof(uint32_t));
    memcpy(*out_data, data_vec.data(), data_vec.size());
    pack_null_bitmap(out_nulls, null_vec, n_all_vals);
    return n_all_vals;
}

int pq_read_list_item(DatasetReader *ds_reader, int64_t column_idx,
                      int out_dtype, array_info **out_offsets,
                      array_info **out_data, array_info **out_nulls) {
    if (ds_reader->count == 0) return 0;

    int64_t start = ds_reader->start_row_first_file;

    int64_t n_all_vals = 0;
    std::vector<uint32_t> offset_vec;
    std::vector<uint8_t> data_vec;
    std::vector<bool> null_vec;
    int64_t last_offset = 0;
    int64_t read_rows = 0;  // rows read so far
    for (auto file_reader : ds_reader->readers) {
        int64_t file_size = pq_get_size_single_file(file_reader, column_idx);
        int64_t rows_to_read =
            std::min(ds_reader->count - read_rows, file_size - start);

        pq_read_list_item_single_file(file_reader, column_idx, out_dtype, start,
                                      rows_to_read, &offset_vec, &data_vec,
                                      &null_vec);

        size_t size = offset_vec.size();
        for (int64_t i = 1; i <= rows_to_read + 1; i++)
            offset_vec[size - i] += last_offset;
        last_offset = offset_vec[size - 1];
        offset_vec.pop_back();
        n_all_vals += rows_to_read;

        read_rows += rows_to_read;
        start = 0;  // start becomes 0 after reading non-empty first chunk
    }
    offset_vec.push_back(last_offset);

    // allocate output arrays and copy data
    *out_offsets = alloc_array(offset_vec.size(), 1, 1,
                               bodo_array_type::arr_type_enum::NUMPY,
                               Bodo_CTypes::UINT32, 0);
    *out_data = alloc_array(data_vec.size(), 1, 1,
                            bodo_array_type::arr_type_enum::NUMPY,
                            (Bodo_CTypes::CTypeEnum)out_dtype, 0);
    int64_t n_null_bytes = (n_all_vals + 7) >> 3;
    *out_nulls =
        alloc_array(n_null_bytes, 1, 1, bodo_array_type::arr_type_enum::NUMPY,
                    Bodo_CTypes::UINT8, 0);

    memcpy((*out_offsets)->data1, offset_vec.data(),
           offset_vec.size() * sizeof(uint32_t));
    memcpy((*out_data)->data1, data_vec.data(), data_vec.size());

    memset((*out_nulls)->data1, 0, n_null_bytes);
    for (int64_t i = 0; i < n_all_vals; i++) {
        if (null_vec[i])
            SetBitTo((uint8_t *)((*out_nulls)->data1), i, true);
    }

    return n_all_vals;
}

/// Convert Bodo date (year, month, day) from int64 to Arrow date32
static int32_t bodo_date64_to_arrow_date32(int64_t date) {
    int64_t year = date >> 32;
    int64_t month = (date >> 16) & 0xFFFF;
    int64_t day = date & 0xFFFF;
    // NOTE that get_days_from_date returns int64 and we are downcasting to
    // int32
    return get_days_from_date(year, month, day);
}

/// Convert Bodo date array (year, month, day elements) to Arrow date32 array
static void CastBodoDateToArrowDate32(const int64_t *input, int64_t length,
                                      int32_t *output) {
    for (int64_t i = 0; i < length; ++i) {
        *output++ = bodo_date64_to_arrow_date32(*input++);
    }
}

void bodo_array_to_arrow(
    arrow::MemoryPool *pool, const array_info *array,
    const std::string &col_name,
    std::vector<std::shared_ptr<arrow::Field>> &schema_vector,
    std::shared_ptr<arrow::ChunkedArray> *out) {
    // allocate null bitmap
    std::shared_ptr<arrow::ResizableBuffer> null_bitmap;
    int64_t null_bytes = arrow::BitUtil::BytesForBits(array->length);
    arrow::Result<std::unique_ptr<arrow::ResizableBuffer>> res =
        AllocateResizableBuffer(null_bytes, pool);
    CHECK_ARROW_AND_ASSIGN(res, "AllocateResizableBuffer", null_bitmap);
    // Padding zeroed by AllocateResizableBuffer
    memset(null_bitmap->mutable_data(), 0, static_cast<size_t>(null_bytes));

    int64_t null_count_ = 0;
    if (array->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        // set arrow bit mask based on bodo bitmask
        for (int64_t i = 0; i < array->length; i++) {
            if (!GetBit((uint8_t *)array->null_bitmask, i)) {
                null_count_++;
                SetBitTo(null_bitmap->mutable_data(), i, false);
            } else {
                SetBitTo(null_bitmap->mutable_data(), i, true);
            }
        }
        if (array->dtype == Bodo_CTypes::_BOOL) {
            // special case: nullable bool column are bit vectors in Arrow
            schema_vector.push_back(arrow::field(col_name, arrow::boolean()));
            int64_t nbytes = ::arrow::BitUtil::BytesForBits(array->length);
            std::shared_ptr<::arrow::Buffer> buffer;
            arrow::Result<std::unique_ptr<arrow::Buffer>> res =
                AllocateBuffer(nbytes, pool);
            CHECK_ARROW_AND_ASSIGN(res, "AllocateBuffer", buffer);

            int64_t i = 0;
            uint8_t *in_data = (uint8_t *)array->data1;
            const auto generate = [&in_data, &i]() -> bool {
                return in_data[i++] != 0;
            };
            ::arrow::internal::GenerateBitsUnrolled(buffer->mutable_data(), 0,
                                                    array->length, generate);

            auto arr_data =
                arrow::ArrayData::Make(arrow::boolean(), array->length,
                                       {null_bitmap, buffer}, null_count_, 0);
            *out = std::make_shared<arrow::ChunkedArray>(
                arrow::MakeArray(arr_data));
        }
    }

    if (array->arr_type == bodo_array_type::NUMPY ||
        (array->arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
         array->dtype != Bodo_CTypes::_BOOL)) {
        int64_t in_num_bytes;
        std::shared_ptr<arrow::DataType> type;
        arrow::Result<std::shared_ptr<arrow::DataType>> type_res;
        switch (array->dtype) {
            case Bodo_CTypes::INT8:
                in_num_bytes = sizeof(int8_t) * array->length;
                type = arrow::int8();
                break;
            case Bodo_CTypes::UINT8:
                in_num_bytes = sizeof(uint8_t) * array->length;
                type = arrow::uint8();
                break;
            case Bodo_CTypes::INT16:
                in_num_bytes = sizeof(int16_t) * array->length;
                type = arrow::int16();
                break;
            case Bodo_CTypes::UINT16:
                in_num_bytes = sizeof(uint16_t) * array->length;
                type = arrow::uint16();
                break;
            case Bodo_CTypes::INT32:
                in_num_bytes = sizeof(int32_t) * array->length;
                type = arrow::int32();
                break;
            case Bodo_CTypes::UINT32:
                in_num_bytes = sizeof(uint32_t) * array->length;
                type = arrow::uint32();
                break;
            case Bodo_CTypes::INT64:
                in_num_bytes = sizeof(int64_t) * array->length;
                type = arrow::int64();
                break;
            case Bodo_CTypes::UINT64:
                in_num_bytes = sizeof(uint64_t) * array->length;
                type = arrow::uint64();
                break;
            case Bodo_CTypes::FLOAT32:
                in_num_bytes = sizeof(float) * array->length;
                type = arrow::float32();
                break;
            case Bodo_CTypes::FLOAT64:
                in_num_bytes = sizeof(double) * array->length;
                type = arrow::float64();
                break;
            case Bodo_CTypes::DECIMAL:
                in_num_bytes = BYTES_PER_DECIMAL * array->length;
                type_res =
                    arrow::Decimal128Type::Make(array->precision, array->scale);
                CHECK_ARROW_AND_ASSIGN(type_res, "arrow::Decimal128Type::Make",
                                       type);
                break;
            case Bodo_CTypes::DATE:
                // input from Bodo uses int64 for dates
                in_num_bytes = sizeof(int64_t) * array->length;
                type = arrow::date32();
                break;
            case Bodo_CTypes::DATETIME:
                // input from Bodo uses int64 for datetimes (datetime64[ns])
                in_num_bytes = sizeof(int64_t) * array->length;
                type = arrow::timestamp(arrow::TimeUnit::NANO);
                break;
            default:
                std::cerr << "Fatal error: invalid dtype found in conversion"
                             " of numeric Bodo array to Arrow"
                          << std::endl;
                exit(1);
        }
        schema_vector.push_back(arrow::field(col_name, type));
        std::shared_ptr<arrow::Buffer> out_buffer;
        if (array->dtype == Bodo_CTypes::DATE) {
            // allocate buffer to store date32 values in Arrow format
            arrow::Result<std::unique_ptr<arrow::Buffer>> res =
                AllocateBuffer(sizeof(int32_t) * array->length, pool);
            CHECK_ARROW_AND_ASSIGN(res, "AllocateBuffer", out_buffer);
            CastBodoDateToArrowDate32((int64_t *)array->data1, array->length,
                                      (int32_t *)out_buffer->mutable_data());
        } else {
            // we can use the same input buffer (no need to cast or convert)
            out_buffer = std::make_shared<arrow::Buffer>(
                (uint8_t *)array->data1, in_num_bytes);
        }

        auto arr_data = arrow::ArrayData::Make(
            type, array->length, {null_bitmap, out_buffer}, null_count_, 0);
        *out =
            std::make_shared<arrow::ChunkedArray>(arrow::MakeArray(arr_data));
    } else if (array->arr_type == bodo_array_type::STRING) {
        arrow::Status status;
        schema_vector.push_back(arrow::field(col_name, arrow::utf8()));
        // Create 16MB chunks for binary data
        constexpr int32_t kBinaryChunksize = 1 << 24;
        ::arrow::internal::ChunkedStringBuilder builder(kBinaryChunksize, pool);
        char *cur_str = array->data1;
        uint32_t *offsets = (uint32_t *)array->data2;
        for (int64_t i = 0; i < array->length; i++) {
            if (!GetBit((uint8_t *)array->null_bitmask, i)) {
                status = builder.AppendNull();
                CHECK_ARROW(status, "builder.AppendNull")
            } else {
                size_t len = offsets[i + 1] - offsets[i];
                status = builder.Append((uint8_t *)cur_str, len);
                CHECK_ARROW(status, "builder.Append")
                cur_str += len;
            }
        }

        ::arrow::ArrayVector result;
        status = builder.Finish(&result);
        CHECK_ARROW(status, "builder.Finish")
        *out = std::make_shared<arrow::ChunkedArray>(result);
    }
}

/*
 * Write the Bodo table (the chunk in this process) to a parquet file.
 * @param _path_name path of output file or directory
 * @param table table to write to parquet file
 * @param col_names_arr array containing the table's column names (index not
 * included)
 * @param index array containing the table index
 * @param write_index true if we need to write index passed in 'index', false
 * otherwise
 * @param metadata string containing table metadata
 * @param is_parallel true if the table is part of a distributed table (in this
 *        case, this process writes a file named "part-000X.parquet" where X is
 *        my rank into the directory specified by 'path_name'
 * @param write_rangeindex_to_metadata : true if writing a RangeIndex to
 * metadata
 * @param ri_start,ri_stop,ri_step start,stop,step parameters of given
 * RangeIndex
 * @param idx_name name of the given index
 */
void pq_write(const char *_path_name, const table_info *table,
              const array_info *col_names_arr, const array_info *index,
              bool write_index, const char *metadata, const char *compression,
              bool is_parallel, bool write_rangeindex_to_metadata,
              const int ri_start, const int ri_stop, const int ri_step,
              const char *idx_name) {
    // Write actual values of start, stop, step to the metadata which is a
    // string that contains %d
    int check;
    std::vector<char> new_metadata;
    if (write_rangeindex_to_metadata) {
        new_metadata.resize((strlen(metadata) + strlen(idx_name) + 50));
        check = sprintf(new_metadata.data(), metadata, idx_name, ri_start,
                        ri_stop, ri_step);
    } else {
        new_metadata.resize((strlen(metadata) + strlen(idx_name) * 4));
        check = sprintf(new_metadata.data(), metadata, idx_name, idx_name,
                        idx_name, idx_name);
    }
    if (check + 1 > new_metadata.size())
        std::cerr << "Fatal error: number of written char for metadata is "
                     "greater than new_metadata size"
                  << std::endl;

    int myrank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    std::string orig_path(_path_name);  // original path passed to this function
    std::string
        path_name;  // original path passed to this function (excluding prefix)
    std::string dirname;  // path and directory name to store the parquet files
                          // (only if is_parallel=true)
    std::string fname;    // name of parquet file to write (excludes path)
    std::shared_ptr<::arrow::io::OutputStream> out_stream;
    Bodo_Fs::FsEnum fs_option;

    extract_fs_dir_path(_path_name, is_parallel, ".parquet", myrank, num_ranks,
                        &fs_option, &dirname, &fname, &orig_path, &path_name);

    open_outstream(fs_option, is_parallel, myrank, "parquet", dirname, fname,
                   orig_path, path_name, &out_stream);

    // copy column names to a std::vector<string>
    std::vector<std::string> col_names;
    char *cur_str = col_names_arr->data1;
    uint32_t *offsets = (uint32_t *)col_names_arr->data2;
    for (int64_t i = 0; i < col_names_arr->length; i++) {
        size_t len = offsets[i + 1] - offsets[i];
        col_names.emplace_back(cur_str, len);
        cur_str += len;
    }

    auto pool = ::arrow::default_memory_pool();

    // convert Bodo table to Arrow: construct Arrow Schema and ChunkedArray
    // columns
    std::vector<std::shared_ptr<arrow::Field>> schema_vector;
    std::vector<std::shared_ptr<arrow::ChunkedArray>> columns(
        table->columns.size());
    for (size_t i = 0; i < table->columns.size(); i++) {
        auto col = table->columns[i];
        bodo_array_to_arrow(pool, col, col_names[i], schema_vector,
                            &columns[i]);
    }

    // make Arrow Schema object
    std::shared_ptr<arrow::Schema> schema;
    if (write_index) {
        // if there is an index, construct ChunkedArray index column and add
        // metadata to the schema
        std::shared_ptr<arrow::ChunkedArray> chunked_arr;
        if (strcmp(idx_name, "null") != 0)
            bodo_array_to_arrow(pool, index, idx_name, schema_vector,
                                &chunked_arr);
        else
            bodo_array_to_arrow(pool, index, "__index_level_0__", schema_vector,
                                &chunked_arr);
        columns.push_back(chunked_arr);
    }

    auto schema_metadata =
        ::arrow::key_value_metadata({{"pandas", new_metadata.data()}});

    schema = std::make_shared<arrow::Schema>(schema_vector, schema_metadata);

    // make Arrow table from Schema and ChunkedArray columns
    int64_t row_group_size = table->nrows();
    std::shared_ptr<arrow::Table> arrow_table =
        arrow::Table::Make(schema, columns, row_group_size);

    // set compression option
    ::arrow::Compression::type codec_type;
    if (strcmp(compression, "snappy") == 0) {
        codec_type = ::arrow::Compression::SNAPPY;
    } else if (strcmp(compression, "brotli") == 0) {
        codec_type = ::arrow::Compression::BROTLI;
    } else if (strcmp(compression, "gzip") == 0) {
        codec_type = ::arrow::Compression::GZIP;
    } else {
        codec_type = ::arrow::Compression::UNCOMPRESSED;
    }
    parquet::WriterProperties::Builder prop_builder;
    prop_builder.compression(codec_type);
    std::shared_ptr<parquet::WriterProperties> writer_properties =
        prop_builder.build();

    // open file and write table
    arrow::Status status = parquet::arrow::WriteTable(
        *arrow_table, pool, out_stream, row_group_size, writer_properties,
        // store_schema() = true is needed to write the schema metadata to file
        // .coerce_timestamps(::arrow::TimeUnit::MICRO)->allow_truncated_timestamps()
        // not needed when moving to parquet 2.0
        ::parquet::ArrowWriterProperties::Builder()
            .coerce_timestamps(::arrow::TimeUnit::MICRO)
            ->allow_truncated_timestamps()
            ->store_schema()
            ->build());
    CHECK_ARROW(status, "parquet::arrow::WriteTable");
}

#undef CHECK
#undef CHECK_ARROW
#undef CHECK_ARROW_AND_ASSIGN
