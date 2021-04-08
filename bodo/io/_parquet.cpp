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
#include "../libs/_array_hash.h"
#include "_fs_io.h"
#include "_parquet_reader.h"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include "parquet/arrow/writer.h"

#undef DEBUG_NESTED_PARQUET

/**
 * This holds the filepaths and other information that this process needs
 * to read its chunk of a Parquet dataset.
 */
struct DatasetReader {
    // Filepaths, only for the files that this process has to read
    std::vector<std::string> filepaths;
    // for each file in filepaths, store the value of each partition
    // column (value is stored as the categorical code). Note that a given
    // file has the same partition value for all of its rows
    std::vector<std::vector<int64_t>> part_vals;
    // If S3, then store the bucket region here
    std::string bucket_region = "";
    // If S3, then store if s3_reader should use the anonymous mode
    bool s3fs_anon = false;
    /// Starting row for first file (files[0])
    int start_row_first_file = 0;
    /// Total number of rows this process has to read (across files)
    int count = 0;
    // Prefix to add to each of the paths before they're appended to filepaths
    std::string prefix = "";
};

/**
 * Get DatasetReader which contains only the file readers that this process
 * needs.
 * @param file_name : file or directory of parquet files
 * @param is_parallel : true if processes will read chunks of the dataset
 * @param bucket_region : S3 bucket region (when reading from S3)
 * @param filters : PyObject passed to pyarrow.parquet.ParquetDataset filters argument
 *                  to remove rows from scanned data
 */
DatasetReader *get_dataset_reader(char *file_name, bool is_parallel,
                                  char *bucket_region, PyObject* filters,
                                  PyObject* storage_options);
void del_dataset_reader(DatasetReader *reader);

int64_t pq_get_size(DatasetReader *reader, int64_t column_idx);
int64_t pq_read(DatasetReader *reader, int64_t real_column_idx,
                int64_t column_idx, uint8_t *out_data, int out_dtype,
                uint8_t *out_nulls = nullptr);
int pq_read_string(DatasetReader *reader, int64_t real_column_idx,
                   int64_t column_idx, NRT_MemInfo **out_meminfo);
int pq_read_list_string(DatasetReader *reader, int64_t real_column_idx,
                        int64_t column_idx, NRT_MemInfo **array_item_meminfo);
int pq_read_array_item(DatasetReader *reader, int64_t real_column_idx,
                       int64_t column_idx, int out_dtype,
                       array_info **out_offsets, array_info **out_data,
                       array_info **out_nulls);
int pq_read_arrow_array(DatasetReader *reader, int64_t real_column_idx,
                        int64_t column_idx, int64_t column_siz,
                        int64_t *lengths, array_info **out_infos);
void pack_null_bitmap(uint8_t *out_nulls, std::vector<bool> &null_vec,
                      int64_t n_all_vals);
/**
 * Fill partition column for this rank's chunk (in partitioned datasets).
 * @param ds_reader : Dataset reader (only files that this rank reads)
 * @param part_col_idx : Partition column index from 0 to # partitions - 1
 * @param out_data : buffer to fill (allocated in code generated in parquet_pio.py)
 * @param cat_dtype : int categorical dtype used to fill the array
 */
void pq_gen_partition_column(DatasetReader *ds_reader, int64_t part_col_idx,
                             void *out_data, int32_t cat_dtype);

void pq_write(const char *filename, const table_info *table,
              const array_info *col_names, const array_info *index,
              bool write_index, const char *metadata, const char *compression,
              bool parallel, bool write_rangeindex_to_metadata, const int start,
              const int stop, const int step, const char *name,
              const char *bucket_region);

void pq_write_partitioned(const char *_path_name, table_info *table,
                          const array_info *col_names_arr,
                          const array_info *col_names_arr_no_partitions,
                          table_info *categories_table, int *partition_cols_idx,
                          int num_partition_cols, const char *compression,
                          bool is_parallel, const char *bucket_region);

// if expr is not true (or NULL), form an err msg and raise a
// runtime_error with it
#define CHECK(expr, msg)                                                   \
    if (!(expr)) {                                                         \
        std::string err_msg = std::string("Error in parquet I/O: ") + msg; \
        throw std::runtime_error(err_msg);                                 \
    }

// if status of arrow::Result is not ok, form an err msg and raise a
// runtime_error with it
#define CHECK_ARROW(expr, msg)                                              \
    if (!(expr.ok())) {                                                     \
        std::string err_msg = std::string("Error in arrow parquet I/O: ") + \
                              msg + " " + expr.ToString();                  \
        throw std::runtime_error(err_msg);                                  \
    }

// if status of arrow::Result is not ok, form an err msg and raise a
// runtime_error with it. If it is ok, get value using ValueOrDie
// and assign it to lhs using std::move
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
    PyObject_SetAttrString(m, "pq_read",
                           PyLong_FromVoidPtr((void *)(&pq_read)));
    PyObject_SetAttrString(m, "pq_get_size",
                           PyLong_FromVoidPtr((void *)(&pq_get_size)));
    PyObject_SetAttrString(m, "pq_read_string",
                           PyLong_FromVoidPtr((void *)(&pq_read_string)));
    PyObject_SetAttrString(m, "pq_read_list_string",
                           PyLong_FromVoidPtr((void *)(&pq_read_list_string)));
    PyObject_SetAttrString(m, "pq_read_array_item",
                           PyLong_FromVoidPtr((void *)(&pq_read_array_item)));
    PyObject_SetAttrString(m, "pq_read_arrow_array",
                           PyLong_FromVoidPtr((void *)(&pq_read_arrow_array)));
    PyObject_SetAttrString(m, "pq_gen_partition_column",
                           PyLong_FromVoidPtr((void *)(&pq_gen_partition_column)));
    PyObject_SetAttrString(m, "pq_write",
                           PyLong_FromVoidPtr((void *)(&pq_write)));
    PyObject_SetAttrString(m, "pq_write_partitioned",
                           PyLong_FromVoidPtr((void *)(&pq_write_partitioned)));
    PyObject_SetAttrString(m, "get_stats_alloc",
                           PyLong_FromVoidPtr((void*)(&get_stats_alloc)));
    PyObject_SetAttrString(m, "get_stats_free",
                           PyLong_FromVoidPtr((void*)(&get_stats_free)));
    PyObject_SetAttrString(m, "get_stats_mi_alloc",
                           PyLong_FromVoidPtr((void*)(&get_stats_mi_alloc)));
    PyObject_SetAttrString(m, "get_stats_mi_free",
                           PyLong_FromVoidPtr((void*)(&get_stats_mi_free)));

    return m;
}


/**
 * Get values for all partition columns of a piece of pyarrow.parquet.ParquetDataset
 * and store in ds_reader (see DatasetReader for details)
 * @param ds_reader : Dataset reader (only files that this rank reads)
 * @param piece : ParquetDataset piece (a single parquet file)
 */
void get_partition_info(DatasetReader *ds_reader, PyObject *piece) {
    PyObject *partition_keys_py = PyObject_GetAttrString(piece, "partition_keys");
    if (PyList_Size(partition_keys_py) > 0) {
        ds_reader->part_vals.emplace_back();
        std::vector<int64_t> &vals = ds_reader->part_vals.back();

        PyObject *part_keys_iter = PyObject_GetIter(partition_keys_py);
        PyObject *key_val_tuple;
        while ((key_val_tuple = PyIter_Next(part_keys_iter))) {
            // PyTuple_GetItem returns borrowed reference, no need to decref
            PyObject* part_val_py = PyTuple_GetItem(key_val_tuple, 1);
            int64_t part_val = PyLong_AsLongLong(part_val_py);
            vals.emplace_back(part_val);
            Py_DECREF(key_val_tuple);
        }
        Py_DECREF(part_keys_iter);
    }
    Py_DECREF(partition_keys_py);
}

DatasetReader *get_dataset_reader(char *file_name, bool parallel,
                                  char *bucket_region, PyObject* filters,
                                  PyObject* storage_options) {
    try {
#ifdef DEBUG_NESTED_PARQUET
    std::cout << "GET_DATASET_READER, beginning\n";
#endif
    auto gilstate = PyGILState_Ensure();

    DatasetReader *ds_reader = new DatasetReader();
    ds_reader->bucket_region = bucket_region;

    assert (storage_options != Py_None);

    // Extract values from the storage_options dict
    // Check that it's a dictionary, else throw an error
    if (PyDict_Check(storage_options)) {
        // Get value of "anon". Returns NULL if it doesn't exist in the dict.
        // No need to decref s3fs_anon_py, PyDict_GetItemString returns borrowed ref
        PyObject* s3fs_anon_py = PyDict_GetItemString(storage_options, "anon");
        if (s3fs_anon_py != NULL && s3fs_anon_py == Py_True) {
            ds_reader->s3fs_anon = true;
        }
    } else {
        throw std::runtime_error("parquet.cpp::get_dataset_reader: storage_options is not a python dictionary.");
    }
    
    

    // import bodo.io.parquet_pio
    PyObject *pq_mod = PyImport_ImportModule("bodo.io.parquet_pio");

    // ds = bodo.io.parquet_pio.get_parquet_dataset(file_name, true, filters=filters, storage_options)
    PyObject *ds = PyObject_CallMethod(pq_mod, "get_parquet_dataset", "sOOO",
                                       file_name, Py_True, filters, 
                                       storage_options);
    Py_DECREF(filters);
    Py_DECREF(storage_options);
    if (PyErr_Occurred()) return NULL;

    Py_DECREF(pq_mod);

    // total_rows = ds._bodo_total_rows
    PyObject *total_rows_py = PyObject_GetAttrString(ds, "_bodo_total_rows");
    int64_t total_rows = PyLong_AsLongLong(total_rows_py);
    Py_DECREF(total_rows_py);

    // prefix = ds._prefix
    PyObject *prefix_py = PyObject_GetAttrString(ds, "_prefix");
    ds_reader->prefix = PyUnicode_AsUTF8(prefix_py);
    Py_DECREF(prefix_py);

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
            // get filepath for every piece
            while ((piece = PyIter_Next(iterator))) {
                PyObject *num_rows_piece_py =
                    PyObject_GetAttrString(piece, "_bodo_num_rows");
                int64_t num_rows_piece = PyLong_AsLongLong(num_rows_piece_py);
                Py_DECREF(num_rows_piece_py);
                if (num_rows_piece > 0) {
                    // p = piece.path
                    PyObject *p = PyObject_GetAttrString(piece, "path");
                    const char *c_path = PyUnicode_AsUTF8(p);
                    // store the filename for this piece
                    // prepend the prefix to the path
                    ds_reader->filepaths.push_back(ds_reader->prefix + c_path);
                    // for this parquet file: store partition value of each
                    // partition column in ds_reader
                    get_partition_info(ds_reader, piece);
                    Py_DECREF(p);
                }
                Py_DECREF(piece);
            }
        }

        Py_DECREF(iterator);

        if (PyErr_Occurred()) return NULL;
        PyGILState_Release(gilstate);
        return ds_reader;
    }

    // is parallel (this process will read a chunk of dataset)

    // calculate the portion of rows that this process needs to read
    size_t rank = dist_get_rank();
    size_t nranks = dist_get_size();
    int64_t start_row_global = dist_get_start(total_rows, nranks, rank);
    ds_reader->count = dist_get_node_portion(total_rows, nranks, rank);

    // get file paths only for the pieces that correspond to my chunk
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
            // start_row_global (first row of my chunk). after that, we get
            // file paths for all subsequent pieces until the number of rows
            // in opened pieces is greater or equal to number of rows in my
            // chunk
            if ((num_rows_piece > 0) &&
                (start_row_global < count_rows + num_rows_piece)) {
                if (ds_reader->filepaths.size() == 0) {
                    ds_reader->start_row_first_file =
                        start_row_global - count_rows;
                    num_rows_my_files +=
                        num_rows_piece - ds_reader->start_row_first_file;
                } else {
                    num_rows_my_files += num_rows_piece;
                }

                // open and store filepath for this piece
                PyObject *p = PyObject_GetAttrString(piece, "path");
                const char *c_path = PyUnicode_AsUTF8(p);
                // prepend the prefix to the path and add it to filepaths
                ds_reader->filepaths.push_back(ds_reader->prefix + c_path);
                // for this parquet file: store partition value of each
                // partition column in ds_reader
                get_partition_info(ds_reader, piece);
                Py_DECREF(p);
            }

            Py_DECREF(piece);

            count_rows += num_rows_piece;
            // finish when number of rows of opened files covers my chunk
            if (num_rows_my_files >= ds_reader->count) break;
        }
    }

    Py_DECREF(iterator);

    if (PyErr_Occurred()) return NULL;
    PyGILState_Release(gilstate);
    return ds_reader;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

void del_dataset_reader(DatasetReader *reader) { delete reader; }

// TODO: column_idx doesn't seem to be used. Remove?
int64_t pq_get_size(DatasetReader *reader, int64_t column_idx) {
    return reader->count;
}

int64_t pq_read(DatasetReader *ds_reader, int64_t real_column_idx,
                int64_t column_idx, uint8_t *out_data, int out_dtype,
                uint8_t *out_nulls) {
    try {
        if (ds_reader->count == 0) return 0;

        int64_t start = ds_reader->start_row_first_file;

        int64_t read_rows = 0;  // rows read so far
        int dtype_size = numpy_item_size[out_dtype];
        for (auto filepath : ds_reader->filepaths) {
            // open file reader for this piece
            std::shared_ptr<parquet::arrow::FileReader> file_reader;
            pq_init_reader(filepath.c_str(), &file_reader,
                           ds_reader->bucket_region.c_str(),
                           ds_reader->s3fs_anon);

            int64_t file_size =
                pq_get_size_single_file(file_reader, column_idx);
            int64_t rows_to_read =
                std::min(ds_reader->count - read_rows, file_size - start);

            pq_read_single_file(file_reader, real_column_idx, column_idx,
                                out_data + read_rows * dtype_size, out_dtype,
                                start, rows_to_read, out_nulls, read_rows);
            read_rows += rows_to_read;
            start = 0;  // start becomes 0 after reading non-empty first chunk
        }
        return 0;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return -1;
    }
}

int pq_read_string(DatasetReader *ds_reader, int64_t real_column_idx,
                   int64_t column_idx, NRT_MemInfo **out_meminfo) {
    try {
        if (ds_reader->count == 0) return 0;

        int64_t start = ds_reader->start_row_first_file;

        int64_t n_all_vals = 0;
        std::vector<offset_t> offset_vec;
        std::vector<uint8_t> data_vec;
        std::vector<bool> null_vec;
        offset_t last_offset = 0;
        int64_t read_rows = 0;  // rows read so far
        for (auto filepath : ds_reader->filepaths) {
            std::shared_ptr<parquet::arrow::FileReader> file_reader;
            pq_init_reader(filepath.c_str(), &file_reader,
                           ds_reader->bucket_region.c_str(),
                           ds_reader->s3fs_anon);

            int64_t file_size =
                pq_get_size_single_file(file_reader, column_idx);
            int64_t rows_to_read =
                std::min(ds_reader->count - read_rows, file_size - start);

            pq_read_string_single_file(file_reader, real_column_idx, column_idx,
                                       start, rows_to_read, &offset_vec,
                                       &data_vec, &null_vec);

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

        int64_t n_strs = offset_vec.size() - 1;
        int64_t n_chars = data_vec.size();
        array_info *out_arr =
            alloc_array(n_strs, n_chars, -1, bodo_array_type::STRING,
                        Bodo_CTypes::STRING, 0, 0);

        offset_t *out_offsets = (offset_t *)out_arr->data2;
        uint8_t *out_data = (uint8_t *)out_arr->data1;
        uint8_t *out_nulls = (uint8_t *)out_arr->null_bitmask;
        *out_meminfo = out_arr->meminfo;
        delete out_arr;

        memcpy(out_offsets, offset_vec.data(),
               offset_vec.size() * sizeof(offset_t));
        memcpy(out_data, data_vec.data(), data_vec.size());
        pack_null_bitmap(out_nulls, null_vec, n_all_vals);
        return n_all_vals;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return -1;
    }
}

int pq_read_list_string(DatasetReader *ds_reader, int64_t real_column_idx,
                        int64_t column_idx, NRT_MemInfo **array_item_meminfo) {
    try {
        if (ds_reader->count == 0) return 0;

        int64_t start = ds_reader->start_row_first_file;

        // TODO get nulls for strings too (not just lists)
        int64_t n_all_vals = 0;
        std::vector<offset_t> index_offset_vec;
        std::vector<offset_t> offset_vec;
        std::vector<uint8_t> data_vec;
        std::vector<bool> null_vec;
        int64_t last_str_offset = 0;
        int64_t last_index_offset = 0;
        int64_t read_rows = 0;  // rows read so far
        for (auto filepath : ds_reader->filepaths) {
            std::shared_ptr<parquet::arrow::FileReader> file_reader;
            pq_init_reader(filepath.c_str(), &file_reader,
                           ds_reader->bucket_region.c_str(),
                           ds_reader->s3fs_anon);

            int64_t file_size =
                pq_get_size_single_file(file_reader, column_idx);
            int64_t rows_to_read =
                std::min(ds_reader->count - read_rows, file_size - start);

            int64_t n_strings = pq_read_list_string_single_file(
                file_reader, real_column_idx, column_idx, start, rows_to_read,
                &offset_vec, &index_offset_vec, &data_vec, &null_vec);

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

        int64_t n_lists = n_all_vals;
        int64_t n_strings = offset_vec.size() - 1;
        int64_t n_chars = data_vec.size();
        array_info *info =
            alloc_list_string_array(n_lists, n_strings, n_chars, 0);
        array_item_arr_payload *payload =
            (array_item_arr_payload *)(info->meminfo->data);
        array_item_arr_numpy_payload *sub_payload =
            (array_item_arr_numpy_payload *)(payload->data->data);
        memcpy(sub_payload->offsets.data, offset_vec.data(),
               offset_vec.size() * sizeof(offset_t));
        memcpy(sub_payload->data.data, data_vec.data(), data_vec.size());
        memcpy(payload->offsets.data, index_offset_vec.data(),
               index_offset_vec.size() * sizeof(offset_t));
        int64_t n_bytes = (n_all_vals + 7) >> 3;
        memset(payload->null_bitmap.data, 0, n_bytes);
        for (int64_t i = 0; i < n_all_vals; i++) {
            if (null_vec[i])
                ::arrow::BitUtil::SetBit((uint8_t *)payload->null_bitmap.data,
                                         i);
        }
        *array_item_meminfo = info->meminfo;
        delete info;
        return n_all_vals;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return -1;
    }
}

int pq_read_arrow_array(DatasetReader *ds_reader, int64_t real_column_idx,
                        int64_t column_idx, int64_t column_siz,
                        int64_t *lengths, array_info **out_infos) {
    try {
        if (ds_reader->count == 0) return 0;

        int64_t start = ds_reader->start_row_first_file;
        int64_t read_rows = 0;  // rows read so far
        arrow::ArrayVector
            parts;  // vector of arrays read, one array for each row group
        std::vector<int> column_indices(column_siz);
        for (int64_t i = 0; i < column_siz; i++)
            column_indices[i] = column_idx + i;
        for (auto filepath : ds_reader->filepaths) {
            std::shared_ptr<parquet::arrow::FileReader> file_reader;
            pq_init_reader(filepath.c_str(), &file_reader,
                           ds_reader->bucket_region.c_str(),
                           ds_reader->s3fs_anon);

            int64_t file_size =
                pq_get_size_single_file(file_reader, column_idx);
            int64_t rows_to_read =
                std::min(ds_reader->count - read_rows, file_size - start);

            pq_read_arrow_single_file(file_reader, column_indices, start,
                                      rows_to_read, parts);

            read_rows += rows_to_read;
            start = 0;  // start becomes 0 after reading non-empty first chunk
        }

        std::shared_ptr<::arrow::Array> out_array;
        arrow::Concatenate(parts, arrow::default_memory_pool(), &out_array);
        parts.clear();  // memory of each array will be freed now

        int64_t lengths_pos = 0;
        int64_t infos_pos = 0;
        nested_array_to_c(out_array, lengths, out_infos, lengths_pos,
                          infos_pos);
        return read_rows;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return -1;
    }
}

int pq_read_array_item(DatasetReader *ds_reader, int64_t real_column_idx,
                       int64_t column_idx, int out_dtype,
                       array_info **out_offsets, array_info **out_data,
                       array_info **out_nulls) {
    try {
        if (ds_reader->count == 0) return 0;

        int64_t start = ds_reader->start_row_first_file;

        int64_t n_all_vals = 0;
        std::vector<offset_t> offset_vec;
        std::vector<uint8_t> data_vec;
        std::vector<bool> null_vec;
        int64_t last_offset = 0;
        int64_t read_rows = 0;  // rows read so far
        for (auto filepath : ds_reader->filepaths) {
            std::shared_ptr<parquet::arrow::FileReader> file_reader;
            pq_init_reader(filepath.c_str(), &file_reader,
                           ds_reader->bucket_region.c_str(),
                           ds_reader->s3fs_anon);

            int64_t file_size =
                pq_get_size_single_file(file_reader, column_idx);
            int64_t rows_to_read =
                std::min(ds_reader->count - read_rows, file_size - start);

            pq_read_array_item_single_file(
                file_reader, real_column_idx, column_idx, out_dtype, start,
                rows_to_read, &offset_vec, &data_vec, &null_vec);

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
                                   Bodo_CType_offset, 0, 0);
        *out_data = alloc_array(data_vec.size(), 1, 1,
                                bodo_array_type::arr_type_enum::NUMPY,
                                (Bodo_CTypes::CTypeEnum)out_dtype, 0, 0);
        int64_t n_null_bytes = (n_all_vals + 7) >> 3;
        *out_nulls = alloc_array(n_null_bytes, 1, 1,
                                 bodo_array_type::arr_type_enum::NUMPY,
                                 Bodo_CTypes::UINT8, 0, 0);

        memcpy((*out_offsets)->data1, offset_vec.data(),
               offset_vec.size() * sizeof(offset_t));
        memcpy((*out_data)->data1, data_vec.data(), data_vec.size());

        memset((*out_nulls)->data1, 0, n_null_bytes);
        for (int64_t i = 0; i < n_all_vals; i++) {
            if (null_vec[i])
                SetBitTo((uint8_t *)((*out_nulls)->data1), i, true);
        }

        return n_all_vals;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return -1;
    }
}

template <typename T>
void pq_gen_partition_column_T(DatasetReader *ds_reader, int64_t part_col_idx,
                               T *out_data) {
    T *cur_offset = out_data;
    int64_t rows_filled = 0;
    int64_t start = ds_reader->start_row_first_file;
    for (size_t i=0; i < ds_reader->filepaths.size(); i++) {
        std::shared_ptr<parquet::arrow::FileReader> file_reader;
        pq_init_reader(ds_reader->filepaths[i].c_str(), &file_reader,
                       ds_reader->bucket_region.c_str(), ds_reader->s3fs_anon);
        // XXX get number of rows from first column in parquet file
        int64_t file_size = pq_get_size_single_file(file_reader, 0);
        int64_t rows_to_fill =
            std::min(ds_reader->count - rows_filled, file_size - start);

        int64_t part_val = ds_reader->part_vals[i][part_col_idx];
        std::fill(cur_offset, cur_offset + rows_to_fill, part_val);

        cur_offset += rows_to_fill;
        rows_filled += rows_to_fill;
        start = 0;  // start becomes 0 after reading non-empty first chunk
    }
}

// populate categorical column for partition columns
void pq_gen_partition_column(DatasetReader *ds_reader, int64_t part_col_idx,
                             void *out_data, int32_t cat_dtype) {
    try {
        switch (cat_dtype) {
            case Bodo_CTypes::INT8:
                return pq_gen_partition_column_T<int8_t>(ds_reader, part_col_idx, (int8_t*)out_data);
            case Bodo_CTypes::INT16:
                return pq_gen_partition_column_T<int16_t>(ds_reader, part_col_idx, (int16_t*)out_data);
            case Bodo_CTypes::INT32:
                return pq_gen_partition_column_T<int32_t>(ds_reader, part_col_idx, (int32_t*)out_data);
            case Bodo_CTypes::INT64:
                return pq_gen_partition_column_T<int64_t>(ds_reader, part_col_idx, (int64_t*)out_data);
            default:
                throw std::runtime_error("pq_gen_partition_column: unrecognized categorical dtype");
        }
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
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
    if (array->arr_type == bodo_array_type::ARROW) {
        schema_vector.push_back(arrow::field(col_name, array->array->type()));
        std::shared_ptr<arrow::ArrayData> typ_data = array->array->data();
        *out = std::make_shared<arrow::ChunkedArray>(array->array);
    }

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
        offset_t *offsets = (offset_t *)array->data2;
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

/**
 * Struct used during pq_write_partitioned to store the information for a
 * partition that this process is going to write: the file path of the parquet
 * file for this partition (e.g. sales_date=2020-01-01/part-00.parquet), and
 * the rows in the table that correspond to this partition.
 */
struct partition_write_info {
    std::string fpath;          // path and filename
    std::vector<int64_t> rows;  // rows in this partition
};

/*
 * Write the Bodo table (this process' chunk) to a partitioned directory of
 * parquet files. This process will write N files if it has N partitions in its
 * local data.
 * @param _path_name path of base output directory for partitioned dataset
 * @param table table to write to parquet files
 * @param col_names_arr array containing the table's column names (index not
 * included)
 * @param col_names_arr_no_partitions array containing the table's column names
 * (index and partition columns not included)
 * @param categories_table table containing categories arrays for each partition
 * column that is a categorical array. Categories could be (for example) strings
 * like "2020-01-01", "2020-01-02", etc.
 * @param partition_cols_idx indices of partition columns in table
 * @param num_partition_cols number of partition columns
 * @param is_parallel true if the table is part of a distributed table
 */
void pq_write_partitioned(const char *_path_name, table_info *table,
                          const array_info *col_names_arr,
                          const array_info *col_names_arr_no_partitions,
                          table_info *categories_table, int *partition_cols_idx,
                          int num_partition_cols, const char *compression,
                          bool is_parallel, const char *bucket_region) {
    // TODOs
    // - Do is parallel here?
    // - sequential (only rank 0 writes, or all write with same name -which?-)
    // - create directories
    //     - what if directories already have files?
    // - write index
    // - write metadata?
    // - convert values to strings for other dtypes like datetime, decimal, etc
    // (see array_info::val_to_str)

    if (!is_parallel)
        throw std::runtime_error("to_parquet partitioned not implemented in sequential mode");

    // new_table will have partition columns at the beginning and the rest after
    // (to use multi_col_key for hashing which assumes that keys are at the
    // beginning), and we will then drop the partition columns from it for
    // writing
    table_info* new_table = new table_info();
    std::vector<bool> is_part_col(table->ncols(), false);
    std::vector<array_info*> partition_cols;
    std::vector<std::string> part_col_names;
    offset_t *offsets = (offset_t *)col_names_arr->data2;
    for (int i = 0; i < num_partition_cols; i++) {
        int j = partition_cols_idx[i];
        is_part_col[j] = true;
        partition_cols.push_back(table->columns[j]);
        new_table->columns.push_back(table->columns[j]);
        char *cur_str = col_names_arr->data1 + offsets[j];
        size_t len = offsets[j + 1] - offsets[j];
        part_col_names.emplace_back(cur_str, len);
    }
    for (int64_t i = 0; i < table->ncols(); i++) {
        if (!is_part_col[i]) new_table->columns.push_back(table->columns[i]);
    }

    const uint32_t seed = SEED_HASH_PARTITION;
    uint32_t* hashes = hash_keys(partition_cols, seed);
    UNORD_MAP_CONTAINER<multi_col_key, partition_write_info, multi_col_key_hash>
        key_to_partition;

    // TODO nullable partition cols?

    std::string fname =
        gen_pieces_file_name(dist_get_rank(), dist_get_size(), ".parquet");

    new_table->num_keys = num_partition_cols;
    for (int64_t i = 0; i < new_table->nrows(); i++) {
        multi_col_key key(hashes[i], new_table, i);
        partition_write_info& p = key_to_partition[key];
        if (p.rows.size() == 0) {
            // generate output file name
            p.fpath = std::string(_path_name) + "/";
            int64_t cat_col_idx = 0;
            for (int j = 0; j < num_partition_cols; j++) {
                auto part_col = partition_cols[j];
                // convert partition col value to string
                std::string value_str;
                if (part_col->arr_type == bodo_array_type::CATEGORICAL) {
                    int64_t code = part_col->get_code_as_int64(i);
                    // TODO can code be -1 (NA) for partition columns?
                    value_str =
                        categories_table->columns[cat_col_idx++]->val_to_str(
                            code);
                } else {
                    value_str = part_col->val_to_str(i);
                }
                p.fpath += part_col_names[j] + "=" + value_str + "/";
            }
            p.fpath += fname;
        }
        p.rows.push_back(i);
    }
    delete[] hashes;

    // drop partition columns from new_table (they are not written to parquet)
    new_table->columns.erase(new_table->columns.begin(),
                             new_table->columns.begin() + num_partition_cols);

    for (auto it = key_to_partition.begin(); it != key_to_partition.end();
         it++) {
        const partition_write_info& p = it->second;
        // RetrieveTable steals the reference but we still need them
        for (auto a : new_table->columns) incref_array(a);
        table_info* part_table =
            RetrieveTable(new_table, p.rows, new_table->ncols());
        // NOTE: we pass is_parallel=False because we already took care of
        // is_parallel here
        pq_write(p.fpath.c_str(), part_table, col_names_arr_no_partitions,
                 nullptr, /*TODO*/ false, /*TODO*/ "", compression, false,
                 false, -1, -1, -1, /*TODO*/ "", bucket_region);
        delete_table_decref_arrays(part_table);
    }
    delete new_table;
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
              const char *idx_name, const char *bucket_region) {
    try {
        // Write actual values of start, stop, step to the metadata which is a
        // string that contains %d
        int check;
        std::vector<char> new_metadata;
        if (write_rangeindex_to_metadata) {
            new_metadata.resize((strlen(metadata) + strlen(idx_name) + 50));
            check = sprintf(new_metadata.data(), metadata, idx_name, ri_start,
                            ri_stop, ri_step);
        } else {
            new_metadata.resize(
                (strlen(metadata) + 1 + (strlen(idx_name) * 4)));
            check = sprintf(new_metadata.data(), metadata, idx_name, idx_name,
                            idx_name, idx_name);
        }
        if (size_t(check + 1) > new_metadata.size())
            throw std::runtime_error(
                "Fatal error: number of written char for metadata is greater "
                "than new_metadata size");

        int myrank, num_ranks;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
        std::string orig_path(
            _path_name);        // original path passed to this function
        std::string path_name;  // original path passed to this function
                                // (excluding prefix)
        std::string dirname;    // path and directory name to store the parquet
                                // files (only if is_parallel=true)
        std::string fname;      // name of parquet file to write (excludes path)
        std::shared_ptr<::arrow::io::OutputStream> out_stream;
        Bodo_Fs::FsEnum fs_option;

        extract_fs_dir_path(_path_name, is_parallel, ".parquet", myrank,
                            num_ranks, &fs_option, &dirname, &fname, &orig_path,
                            &path_name);

        open_outstream(fs_option, is_parallel, myrank, "parquet", dirname,
                       fname, orig_path, path_name, &out_stream, bucket_region);

        // copy column names to a std::vector<string>
        std::vector<std::string> col_names;
        char *cur_str = col_names_arr->data1;
        offset_t *offsets = (offset_t *)col_names_arr->data2;
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

        if (write_index) {
            // if there is an index, construct ChunkedArray index column and add
            // metadata to the schema
            std::shared_ptr<arrow::ChunkedArray> chunked_arr;
            if (strcmp(idx_name, "null") != 0)
                bodo_array_to_arrow(pool, index, idx_name, schema_vector,
                                    &chunked_arr);
            else
                bodo_array_to_arrow(pool, index, "__index_level_0__",
                                    schema_vector, &chunked_arr);
            columns.push_back(chunked_arr);
        }

        std::shared_ptr<arrow::KeyValueMetadata> schema_metadata;
        if (new_metadata.size() > 0 && new_metadata[0] != 0)
            schema_metadata =
                ::arrow::key_value_metadata({{"pandas", new_metadata.data()}});

        // make Arrow Schema object
        std::shared_ptr<arrow::Schema> schema =
            std::make_shared<arrow::Schema>(schema_vector, schema_metadata);

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
            // store_schema() = true is needed to write the schema metadata to
            // file
            // .coerce_timestamps(::arrow::TimeUnit::MICRO)->allow_truncated_timestamps()
            // not needed when moving to parquet 2.0
            ::parquet::ArrowWriterProperties::Builder()
                .coerce_timestamps(::arrow::TimeUnit::MICRO)
                ->allow_truncated_timestamps()
                ->store_schema()
                ->build());
        CHECK_ARROW(status, "parquet::arrow::WriteTable");
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

#undef CHECK
#undef CHECK_ARROW
#undef CHECK_ARROW_AND_ASSIGN
