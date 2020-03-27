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
using parquet::arrow::FileReader;
#include "../libs/_bodo_common.h"
#include "../libs/_datetime_ext.h"
#include "_parquet_reader.h"
#include "_writer.h"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include "parquet/arrow/writer.h"

typedef std::vector<std::shared_ptr<FileReader>> FileReaderVec;

FileReaderVec *get_arrow_readers(char *file_name);
void del_arrow_readers(FileReaderVec *readers);

PyObject *str_list_to_vec(PyObject *self, PyObject *str_list);
int64_t pq_get_size(FileReaderVec *readers, int64_t column_idx);
int64_t pq_read(FileReaderVec *readers, int64_t column_idx, uint8_t *out_data,
                int out_dtype, uint8_t *out_nulls = nullptr);
int pq_read_parallel(FileReaderVec *readers, int64_t column_idx,
                     uint8_t *out_data, int out_dtype, int64_t start,
                     int64_t count, uint8_t *out_nulls = nullptr);
int pq_read_string(FileReaderVec *readers, int64_t column_idx,
                   uint32_t **out_offsets, uint8_t **out_data,
                   uint8_t **out_nulls);
int pq_read_string_parallel(FileReaderVec *readers, int64_t column_idx,
                            uint32_t **out_offsets, uint8_t **out_data,
                            uint8_t **out_nulls, int64_t start, int64_t count);
int pq_read_list_string(FileReaderVec *readers, int64_t column_idx,
                        uint32_t **out_offsets, uint32_t **index_offsets,
                        uint8_t **out_data, uint8_t **out_nulls);
int pq_read_list_string_parallel(FileReaderVec *readers, int64_t column_idx,
                                 uint32_t **out_offsets,
                                 uint32_t **index_offsets, uint8_t **out_data,
                                 uint8_t **out_nulls, int64_t start,
                                 int64_t count);

void pack_null_bitmap(uint8_t **out_nulls, std::vector<bool> &null_vec,
                      int64_t n_all_vals);
void pq_write(const char *filename, const table_info *table,
              const array_info *col_names, const array_info *index,
              const char *metadata, const char *compression, bool parallel,
              bool write_rangeindex_to_metadata, const int start,
              const int stop, const int step, const char *name);

static PyMethodDef parquet_cpp_methods[] = {
    {"str_list_to_vec", str_list_to_vec, METH_O,  // METH_STATIC
     "convert Python string list to C++ std vector of strings"},
    {NULL, NULL, 0, NULL}};

PyMODINIT_FUNC PyInit_parquet_cpp(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, "parquet_cpp", "No docs", -1,
        parquet_cpp_methods,
    };
    m = PyModule_Create(&moduledef);
    if (m == NULL) return NULL;

    PyObject_SetAttrString(m, "get_arrow_readers",
                           PyLong_FromVoidPtr((void *)(&get_arrow_readers)));
    PyObject_SetAttrString(m, "del_arrow_readers",
                           PyLong_FromVoidPtr((void *)(&del_arrow_readers)));
    PyObject_SetAttrString(m, "read", PyLong_FromVoidPtr((void *)(&pq_read)));
    PyObject_SetAttrString(m, "read_parallel",
                           PyLong_FromVoidPtr((void *)(&pq_read_parallel)));
    PyObject_SetAttrString(m, "get_size",
                           PyLong_FromVoidPtr((void *)(&pq_get_size)));
    PyObject_SetAttrString(m, "read_string",
                           PyLong_FromVoidPtr((void *)(&pq_read_string)));
    PyObject_SetAttrString(
        m, "read_string_parallel",
        PyLong_FromVoidPtr((void *)(&pq_read_string_parallel)));
    PyObject_SetAttrString(m, "read_list_string",
                           PyLong_FromVoidPtr((void *)(&pq_read_list_string)));
    PyObject_SetAttrString(
        m, "read_list_string_parallel",
        PyLong_FromVoidPtr((void *)(&pq_read_list_string_parallel)));
    PyObject_SetAttrString(m, "pq_write",
                           PyLong_FromVoidPtr((void *)(&pq_write)));

    return m;
}

PyObject *str_list_to_vec(PyObject *self, PyObject *str_list) {
    Py_INCREF(str_list);  // needed?
    // TODO: need to acquire GIL?
    std::vector<std::string> *strs_vec = new std::vector<std::string>();

    PyObject *iterator = PyObject_GetIter(str_list);
    Py_DECREF(str_list);
    PyObject *l_str;

    if (iterator == NULL) {
        return PyLong_FromVoidPtr((void *)strs_vec);
    }

    while ((l_str = PyIter_Next(iterator))) {
        const char *c_path = PyUnicode_AsUTF8(l_str);
        // printf("str %s\n", c_path);
        strs_vec->push_back(std::string(c_path));
        Py_DECREF(l_str);
    }

    Py_DECREF(iterator);

    // CHECK(!PyErr_Occurred(), "Python error during Parquet dataset metadata")
    return PyLong_FromVoidPtr((void *)strs_vec);
}

std::vector<std::string> get_pq_pieces(char *file_name) {
#define CHECK(expr, msg)                   \
    if (!(expr)) {                         \
        std::cerr << msg << std::endl;     \
        PyGILState_Release(gilstate);      \
        return std::vector<std::string>(); \
    }

    std::vector<std::string> paths;

    auto gilstate = PyGILState_Ensure();

    // import bodo.io.parquet_pio
    PyObject *pq_mod = PyImport_ImportModule("bodo.io.parquet_pio");

    // ds = bodo.io.parquet_pio.get_parquet_dataset(file_name)
    PyObject *ds =
        PyObject_CallMethod(pq_mod, "get_parquet_dataset", "s", file_name);
    CHECK(!PyErr_Occurred(), "Python error during Parquet dataset metadata")
    Py_DECREF(pq_mod);

    // all_peices = ds.pieces
    PyObject *all_peices = PyObject_GetAttrString(ds, "pieces");
    Py_DECREF(ds);

    // paths.append(piece.path) for piece in all peices
    PyObject *iterator = PyObject_GetIter(all_peices);
    Py_DECREF(all_peices);
    PyObject *piece;

    if (iterator == NULL) {
        PyGILState_Release(gilstate);
        return paths;
    }

    while ((piece = PyIter_Next(iterator))) {
        PyObject *p = PyObject_GetAttrString(piece, "path");
        const char *c_path = PyUnicode_AsUTF8(p);
        paths.push_back(std::string(c_path));
        Py_DECREF(piece);
        Py_DECREF(p);
    }

    Py_DECREF(iterator);

    CHECK(!PyErr_Occurred(), "Python error during Parquet dataset metadata")
    PyGILState_Release(gilstate);
    return paths;
#undef CHECK
}

FileReaderVec *get_arrow_readers(char *file_name) {
    FileReaderVec *readers = new FileReaderVec();

    std::vector<std::string> all_files = get_pq_pieces(file_name);
    for (const auto &inner_file : all_files) {
        std::shared_ptr<FileReader> arrow_reader;
        pq_init_reader(inner_file.c_str(), &arrow_reader);
        readers->push_back(arrow_reader);
    }

    return readers;
}

void del_arrow_readers(FileReaderVec *readers) {
    delete readers;
    return;
}

int64_t pq_get_size(FileReaderVec *readers, int64_t column_idx) {
    if (readers->size() == 0) {
        printf("empty parquet dataset\n");
        return 0;
    }

    if (readers->size() > 1) {
        // std::cout << "pq path is dir" << '\n';
        int64_t ret = 0;
        for (size_t i = 0; i < readers->size(); i++) {
            ret += pq_get_size_single_file(readers->at(i), column_idx);
        }

        // std::cout << "total pq dir size: " << ret << '\n';
        return ret;
    } else {
        return pq_get_size_single_file(readers->at(0), column_idx);
    }
    return 0;
}

int64_t pq_read(FileReaderVec *readers, int64_t column_idx, uint8_t *out_data,
                int out_dtype, uint8_t *out_nulls) {
    if (readers->size() == 0) {
        printf("empty parquet dataset\n");
        return 0;
    }

    if (readers->size() > 1) {
        // std::cout << "pq path is dir" << '\n';
        int dtype_size = pq_type_sizes[out_dtype];

        int64_t num_vals = 0;  // number of values read so for
        for (size_t i = 0; i < readers->size(); i++) {
            num_vals += pq_read_single_file(readers->at(i), column_idx,
                                            out_data + num_vals * dtype_size,
                                            out_dtype, out_nulls, num_vals);
        }

        // std::cout << "total pq dir size: " << num_vals << '\n';
        return num_vals;
    } else {
        return pq_read_single_file(readers->at(0), column_idx, out_data,
                                   out_dtype, out_nulls, 0);
    }
    return 0;
}

int pq_read_parallel(FileReaderVec *readers, int64_t column_idx,
                     uint8_t *out_data, int out_dtype, int64_t start,
                     int64_t count, uint8_t *out_nulls) {
    // printf("read parquet parallel column: %lld start: %lld count: %lld\n",
    //                                                 column_idx, start,
    //                                                 count);

    if (count == 0) {
        return 0;
    }

    if (readers->size() == 0) {
        printf("empty parquet dataset\n");
        return 0;
    }

    if (readers->size() > 1) {
        // std::cout << "pq path is dir" << '\n';
        // TODO: get file sizes on root rank only

        // skip whole files if no need to read any rows
        int file_ind = 0;
        int64_t file_size = pq_get_size_single_file(readers->at(0), column_idx);
        while (start >= file_size) {
            start -= file_size;
            file_ind++;
            file_size =
                pq_get_size_single_file(readers->at(file_ind), column_idx);
        }

        int dtype_size = pq_type_sizes[out_dtype];
        // std::cout << "dtype_size: " << dtype_size << '\n';

        // read data
        int64_t read_rows = 0;  // rows read so far
        while (read_rows < count) {
            int64_t rows_to_read =
                std::min(count - read_rows, file_size - start);
            pq_read_parallel_single_file(readers->at(file_ind), column_idx,
                                         out_data + read_rows * dtype_size,
                                         out_dtype, start, rows_to_read,
                                         out_nulls, read_rows);
            read_rows += rows_to_read;
            start = 0;  // start becomes 0 after reading non-empty first chunk
            file_ind++;
            // std::cout << "next file: " << all_files[file_ind] << '\n';
            if (read_rows < count)
                file_size =
                    pq_get_size_single_file(readers->at(file_ind), column_idx);
        }
        return 0;
    } else {
        return pq_read_parallel_single_file(readers->at(0), column_idx,
                                            out_data, out_dtype, start, count,
                                            out_nulls, 0);
    }
    return 0;
}

int pq_read_string(FileReaderVec *readers, int64_t column_idx,
                   uint32_t **out_offsets, uint8_t **out_data,
                   uint8_t **out_nulls) {
    if (readers->size() == 0) {
        printf("empty parquet dataset\n");
        return 0;
    }

    if (readers->size() > 1) {
        // std::cout << "pq path is dir" << '\n';

        std::vector<uint32_t> offset_vec;
        std::vector<uint8_t> data_vec;
        std::vector<bool> null_vec;
        int32_t last_offset = 0;
        int64_t n_all_vals = 0;
        for (size_t i = 0; i < readers->size(); i++) {
            int64_t n_vals = pq_read_string_single_file(
                readers->at(i), column_idx, NULL, NULL, NULL, &offset_vec,
                &data_vec, &null_vec);
            if (n_vals == -1) continue;

            int size = offset_vec.size();
            for (int64_t i = 1; i <= n_vals + 1; i++)
                offset_vec[size - i] += last_offset;
            last_offset = offset_vec[size - 1];
            offset_vec.pop_back();
            n_all_vals += n_vals;
        }
        offset_vec.push_back(last_offset);

        *out_offsets = new uint32_t[offset_vec.size()];
        *out_data = new uint8_t[data_vec.size()];

        memcpy(*out_offsets, offset_vec.data(),
               offset_vec.size() * sizeof(uint32_t));
        memcpy(*out_data, data_vec.data(), data_vec.size());
        pack_null_bitmap(out_nulls, null_vec, n_all_vals);

        // for(int i=0; i<offset_vec.size(); i++)
        //     std::cout << (*out_offsets)[i] << ' ';
        // std::cout << '\n';
        // std::cout << "string dir read done" << '\n';
        return n_all_vals;
    } else {
        return pq_read_string_single_file(readers->at(0), column_idx,
                                          out_offsets, out_data, out_nulls);
    }
    return 0;
}

int pq_read_string_parallel(FileReaderVec *readers, int64_t column_idx,
                            uint32_t **out_offsets, uint8_t **out_data,
                            uint8_t **out_nulls, int64_t start, int64_t count) {
    // printf("read parquet parallel str file: %s column: %lld start: %lld
    // count: %lld\n",
    //                                 file_name->c_str(), column_idx, start,
    //                                 count);

    if (readers->size() == 0) {
        printf("empty parquet dataset\n");
        return 0;
    }

    if (readers->size() > 1) {
        // std::cout << "pq path is dir" << '\n';

        // skip whole files if no need to read any rows
        int file_ind = 0;
        int64_t file_size = pq_get_size_single_file(readers->at(0), column_idx);
        while (start >= file_size) {
            start -= file_size;
            file_ind++;
            file_size =
                pq_get_size_single_file(readers->at(file_ind), column_idx);
        }

        int64_t n_all_vals = 0;
        std::vector<uint32_t> offset_vec;
        std::vector<uint8_t> data_vec;
        std::vector<bool> null_vec;

        // read data
        int64_t last_offset = 0;
        int64_t read_rows = 0;
        while (read_rows < count) {
            int64_t rows_to_read =
                std::min(count - read_rows, file_size - start);
            if (rows_to_read > 0) {
                pq_read_string_parallel_single_file(
                    readers->at(file_ind), column_idx, NULL, NULL, NULL, start,
                    rows_to_read, &offset_vec, &data_vec, &null_vec);

                int size = offset_vec.size();
                for (int64_t i = 1; i <= rows_to_read + 1; i++)
                    offset_vec[size - i] += last_offset;
                last_offset = offset_vec[size - 1];
                offset_vec.pop_back();
                n_all_vals += rows_to_read;
            }

            read_rows += rows_to_read;
            start = 0;  // start becomes 0 after reading non-empty first chunk
            file_ind++;
            if (read_rows < count)
                file_size =
                    pq_get_size_single_file(readers->at(file_ind), column_idx);
        }
        offset_vec.push_back(last_offset);

        *out_offsets = new uint32_t[offset_vec.size()];
        *out_data = new uint8_t[data_vec.size()];

        memcpy(*out_offsets, offset_vec.data(),
               offset_vec.size() * sizeof(uint32_t));
        memcpy(*out_data, data_vec.data(), data_vec.size());
        pack_null_bitmap(out_nulls, null_vec, n_all_vals);
        return n_all_vals;
    } else {
        return pq_read_string_parallel_single_file(readers->at(0), column_idx,
                                                   out_offsets, out_data,
                                                   out_nulls, start, count);
    }
    return 0;
}

int pq_read_list_string(FileReaderVec *readers, int64_t column_idx,
                        uint32_t **out_offsets, uint32_t **index_offsets,
                        uint8_t **out_data, uint8_t **out_nulls) {
    if (readers->size() == 0) {
        printf("empty parquet dataset\n");
        return 0;
    }

    if (readers->size() > 1) {
        std::vector<uint32_t> offset_vec;
        std::vector<uint32_t> index_offset_vec;
        std::vector<uint8_t> data_vec;
        std::vector<bool> null_vec;
        int32_t last_index_offset = 0;
        int32_t last_str_offset = 0;
        int64_t n_all_vals = 0;
        for (size_t i = 0; i < readers->size(); i++) {
            std::pair<int64_t, int64_t> n_vals =
                pq_read_list_string_single_file(
                    readers->at(i), column_idx, NULL, NULL, NULL, NULL,
                    &offset_vec, &index_offset_vec, &data_vec, &null_vec);
            int64_t n_lists = n_vals.first;
            int64_t n_strings = n_vals.second;
            if (n_lists == -1) continue;

            size_t size = offset_vec.size();
            for (int64_t i = 1; i <= n_strings + 1; i++)
                offset_vec[size - i] += last_str_offset;
            last_str_offset = offset_vec.back();
            offset_vec.pop_back();

            size = index_offset_vec.size();
            for (int64_t i = 1; i <= n_lists + 1; i++)
                index_offset_vec[size - i] += last_index_offset;
            last_index_offset = index_offset_vec[size - 1];
            index_offset_vec.pop_back();

            n_all_vals += n_lists;
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
    } else {
        std::pair<int64_t, int64_t> n_vals = pq_read_list_string_single_file(
            readers->at(0), column_idx, out_offsets, index_offsets, out_data,
            out_nulls);
        return n_vals.first;
    }
}

int pq_read_list_string_parallel(FileReaderVec *readers, int64_t column_idx,
                                 uint32_t **out_offsets,
                                 uint32_t **index_offsets, uint8_t **out_data,
                                 uint8_t **out_nulls, int64_t start,
                                 int64_t count) {
    if (readers->size() == 0) {
        printf("empty parquet dataset\n");
        return 0;
    }

    if (readers->size() > 1) {
        // skip whole files if no need to read any rows
        int file_ind = 0;
        int64_t file_size = pq_get_size_single_file(readers->at(0), column_idx);
        while (start >= file_size) {
            start -= file_size;
            file_ind++;
            file_size =
                pq_get_size_single_file(readers->at(file_ind), column_idx);
        }

        int64_t n_all_vals = 0;
        std::vector<uint32_t> index_offset_vec;
        std::vector<uint32_t> offset_vec;
        std::vector<uint8_t> data_vec;
        std::vector<bool> null_vec;

        // read data
        int64_t last_str_offset = 0;
        int64_t last_index_offset = 0;
        int64_t read_rows = 0;
        while (read_rows < count) {
            int64_t rows_to_read =
                std::min(count - read_rows, file_size - start);
            if (rows_to_read > 0) {
                int64_t n_strings = pq_read_list_string_parallel_single_file(
                    readers->at(file_ind), column_idx, NULL, NULL, NULL, NULL,
                    start, rows_to_read, &offset_vec, &index_offset_vec,
                    &data_vec, &null_vec);

                int size = offset_vec.size();
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
            }

            read_rows += rows_to_read;
            start = 0;  // start becomes 0 after reading non-empty first chunk
            file_ind++;
            if (read_rows < count)
                file_size =
                    pq_get_size_single_file(readers->at(file_ind), column_idx);
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
    } else {
        pq_read_list_string_parallel_single_file(
            readers->at(0), column_idx, out_offsets, index_offsets, out_data,
            out_nulls, start, count);
    }
    return 0;
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
    AllocateResizableBuffer(pool, null_bytes, &null_bitmap);
    // Padding zeroed by AllocateResizableBuffer
    memset(null_bitmap->mutable_data(), 0, static_cast<size_t>(null_bytes));

    int64_t null_count_ = 0;
    if (array->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        // set arrow bit mask based on bodo bitmask
        for (int64_t i = 0; i < array->length; i++) {
            if (!GetBit((uint8_t *)array->null_bitmask, i)) {
                null_count_++;
                ::arrow::BitUtil::ClearBit(null_bitmap->mutable_data(), i);
            } else {
                ::arrow::BitUtil::SetBit(null_bitmap->mutable_data(), i);
            }
        }
        if (array->dtype == Bodo_CTypes::_BOOL) {
            // special case: nullable bool column are bit vectors in Arrow
            schema_vector.push_back(arrow::field(col_name, arrow::boolean()));
            int64_t nbytes = ::arrow::BitUtil::BytesForBits(array->length);
            std::shared_ptr<::arrow::Buffer> buffer;
            AllocateBuffer(pool, nbytes, &buffer);

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
                arrow::Decimal128Type::Make(array->precision, array->scale,
                                            &type);
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
            AllocateBuffer(pool, sizeof(int32_t) * array->length, &out_buffer);
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
        schema_vector.push_back(arrow::field(col_name, arrow::utf8()));
        // Create 16MB chunks for binary data
        constexpr int32_t kBinaryChunksize = 1 << 24;
        ::arrow::internal::ChunkedStringBuilder builder(kBinaryChunksize, pool);
        char *cur_str = array->data1;
        uint32_t *offsets = (uint32_t *)array->data2;
        for (int64_t i = 0; i < array->length; i++) {
            if (!GetBit((uint8_t *)array->null_bitmask, i)) {
                builder.AppendNull();
            } else {
                size_t len = offsets[i + 1] - offsets[i];
                builder.Append((uint8_t *)cur_str, len);
                cur_str += len;
            }
        }

        ::arrow::ArrayVector result;
        builder.Finish(&result);
        *out = std::make_shared<arrow::ChunkedArray>(result);
    }
}

#define CHECK(expr, msg)                                             \
    if (!(expr)) {                                                   \
        std::cerr << "Error in parquet write: " << msg << std::endl; \
        return;                                                      \
    }

#define CHECK_ARROW(expr, msg)                                              \
    if (!(expr.ok())) {                                                     \
        std::cerr << "Error in arrow parquet write: " << msg << " " << expr \
                  << std::endl;                                             \
        return;                                                             \
    }

#define CHECK_ARROW_AND_ASSIGN(res, msg, lhs) \
    CHECK_ARROW(res.status(), msg)            \
    lhs = std::move(res).ValueOrDie();

/*
 * Write the Bodo table (the chunk in this process) to a parquet file.
 * @param _path_name path of output file or directory
 * @param table table to write to parquet file
 * @param col_names_arr array containing the table's column names (index not
 * included)
 * @param index array containing the table index (can be an empty array if no
 * index)
 * @param metadata string containing table metadata
 * @param is_parallel true if the table is part of a distributed table (in this
 * case, this process writes a file named "part-000X.parquet" where X is my rank
 *        into the directory specified by 'path_name'
 * @param ri_start,ri_stop,ri_step start,stop,step parameters of given
 * RangeIndex
 * @param idx_name name of the given index
 */
void pq_write(const char *_path_name, const table_info *table,
              const array_info *col_names_arr, const array_info *index,
              const char *metadata, const char *compression, bool is_parallel,
              bool write_rangeindex_to_metadata, const int ri_start,
              const int ri_stop, const int ri_step, const char *idx_name) {
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

    open_outstream(fs_option, is_parallel, myrank, "parquet",
                   dirname, fname, orig_path, path_name, &out_stream);

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
    if (index->length > 0) {
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
    parquet::arrow::WriteTable(
        *arrow_table, pool, out_stream, row_group_size, writer_properties,
        // store_schema() = true is needed to write the schema metadata to file
        ::parquet::ArrowWriterProperties::Builder().store_schema()->build());
}

#undef CHECK
#undef CHECK_ARROW
