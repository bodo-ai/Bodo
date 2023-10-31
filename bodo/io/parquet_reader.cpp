// Copyright (C) 2021 Bodo Inc. All rights reserved.

// Implementation of ParquetReader (subclass of ArrowReader) with
// functionality that is specific to reading parquet datasets

#include "parquet_reader.h"
#include <arrow/python/pyarrow.h>
#include <span>
#include "../libs/_array_utils.h"

// Silence warnings from including generated code
PUSH_IGNORED_COMPILER_ERROR("-Wunused-variable")
#include "arrow_ext.cpp"
POP_IGNORED_COMPILER_ERROR()

using arrow::Type;
using parquet::ParquetFileReader;

// -------- Helper functions --------

/**
 * Fill a range in an output partition column with a given value.
 * NOTE: partition columns are always categorical columns. The categories are
 * the partition values (typically a small set) and in C++ we populate numpy
 * arrays containing the codes (also see part_vals below)
 * @param val : value with which to fill
 * @param data : output buffer
 * @param offset : start at this position
 * @param rows_to_fill : number of elements to write
 * @param cat_dtype : dtype to use (does a cast of val to this type)
 */
static void fill_partition_column(int64_t val, char* data, int64_t offset,
                                  int64_t rows_to_fill, int32_t cat_dtype) {
    switch (cat_dtype) {
        case Bodo_CTypes::INT8:
            return std::fill((int8_t*)data + offset,
                             (int8_t*)data + offset + rows_to_fill, val);
        case Bodo_CTypes::INT16:
            return std::fill((int16_t*)data + offset,
                             (int16_t*)data + offset + rows_to_fill, val);
        case Bodo_CTypes::INT32:
            return std::fill((int32_t*)data + offset,
                             (int32_t*)data + offset + rows_to_fill, val);
        case Bodo_CTypes::INT64:
            return std::fill((int64_t*)data + offset,
                             (int64_t*)data + offset + rows_to_fill, val);
        default:
            throw std::runtime_error(
                "fill_partition_column: unrecognized categorical dtype " +
                std::to_string(cat_dtype));
    }
}

/**
 * Fill a range in the indices array of input_file_name column with a given
 * value. NOTE: input_file_name column is always a dictionary-encoded string
 * array.
 * @param val : value with which to fill
 * @param data : output buffer
 * @param offset : start at this position
 * @param rows_to_fill : number of elements to write
 */
static void fill_input_file_name_col_indices(int32_t val, char* data,
                                             int64_t offset,
                                             int64_t rows_to_fill) {
    // Reuse the partition column functionality with INT32 (since
    // that's the type of indices array of a dictionary-encoded
    // string array).
    fill_partition_column(val, data, offset, rows_to_fill, Bodo_CTypes::INT32);
}

/**
 * Fill the dictionary array of input_file_name column with given values.
 * NOTE: input_file_name column is always a dictionary-encoded string array.
 * @param file_paths : vector of file paths to fill
 * @param prefix : prefix to attach to the file-paths (for ADLS, HDFS, etc.)
 * @param data : output buffer
 * @param offsets : offsets buffer in the underlying array_info
 */
static void fill_input_file_name_col_dict(
    const std::span<const std::string> file_paths, const std::string& prefix,
    char* data, char* offsets) {
    offset_t* col_offsets = (offset_t*)offsets;
    for (size_t i = 0; i < file_paths.size(); i++) {
        std::string fname = prefix + file_paths[i];
        memcpy(data + col_offsets[i], fname.c_str(), fname.length());
        col_offsets[i + 1] = col_offsets[i] + fname.length();
    }
}

// -------- ParquetReader --------
void ParquetReader::add_piece(PyObject* piece, int64_t num_rows,
                              int64_t total_rows) {
    // p = piece.path
    PyObject* p = PyObject_GetAttrString(piece, "path");
    const char* c_path = PyUnicode_AsUTF8(p);
    file_paths.emplace_back(c_path);
    pieces_nrows.push_back(num_rows);
    if (this->input_file_name_col) {
        // allocate indices array for the dictionary-encoded
        // input_file_name col if not already allocated
        if (this->input_file_name_col_indices_arr == nullptr) {
            // use alloc_nullable_array_no_nulls since there cannot
            // be any nulls here
            this->input_file_name_col_indices_arr =
                alloc_nullable_array_no_nulls(total_rows, Bodo_CTypes::INT32,
                                              0);
        }
        // fill a range in the indices array
        fill_input_file_name_col_indices(
            file_paths.size() - 1,
            this->input_file_name_col_indices_arr->data1(),
            this->input_file_name_col_indices_offset, num_rows);
        this->input_file_name_col_indices_offset += num_rows;
        this->input_file_name_col_dict_arr_total_chars +=
            (std::strlen(c_path) + this->prefix.length());
    }

    // for this parquet file: store partition value of each partition column
    get_partition_info(piece);
    Py_DECREF(p);
}

PyObject* ParquetReader::get_dataset() {
    // import bodo.io.parquet_pio
    PyObject* pq_mod = PyImport_ImportModule("bodo.io.parquet_pio");

    // ds = bodo.io.parquet_pio.get_parquet_dataset(path, true, filters,
    // storage_options)
    PyObject* ds = PyObject_CallMethod(
        pq_mod, "get_parquet_dataset", "OOOOOOOLOO", path, Py_True, dnf_filters,
        expr_filters, storage_options, Py_False, PyBool_FromLong(parallel),
        tot_rows_to_read, pyarrow_schema, PyBool_FromLong(this->use_hive));
    if (ds == NULL && PyErr_Occurred()) {
        throw std::runtime_error("python");
    }
    Py_DECREF(path);
    Py_DECREF(dnf_filters);
    Py_DECREF(pq_mod);
    if (PyErr_Occurred())
        throw std::runtime_error("python");

    this->ds_partitioning = PyObject_GetAttrString(ds, "partitioning");
    this->set_arrow_schema(PyObject_GetAttrString(ds, "schema"));

    // prefix = ds.prefix
    PyObject* prefix_py = PyObject_GetAttrString(ds, "_prefix");
    this->prefix.assign(PyUnicode_AsUTF8(prefix_py));
    Py_DECREF(prefix_py);

    this->filesystem = PyObject_GetAttrString(ds, "filesystem");
    return ds;
}

void ParquetReader::init_pq_scanner() {
    tracing::Event ev_scanner("init_pq_scanner");
    if (get_num_pieces() == 0) {
        return;
    }

    // Construct Python lists from C++ vectors for values used in
    // get_scanner_batches
    PyObject* fnames_list_py = PyList_New(file_paths.size());
    size_t i = 0;
    for (auto p : file_paths) {
        PyList_SetItem(fnames_list_py, i++, PyUnicode_FromString(p.c_str()));
    }

    PyObject* str_as_dict_cols_py = PyList_New(str_as_dict_colnames.size());
    i = 0;
    for (auto field_name : str_as_dict_colnames) {
        PyList_SetItem(str_as_dict_cols_py, i++,
                       PyUnicode_FromString(field_name.c_str()));
    }

    PyObject* selected_fields_py = PyList_New(selected_fields.size());
    i = 0;
    for (auto field_num : selected_fields) {
        PyList_SetItem(selected_fields_py, i++, PyLong_FromLong(field_num));
    }

    PyObject* batch_size_py =
        batch_size == -1 ? Py_None : PyLong_FromLong(batch_size);
    PyObject* pq_mod = PyImport_ImportModule("bodo.io.parquet_pio");
    // This only loads record batches that match the filter.
    // get_scanner_batches returns a tuple with the dataset object
    // and the record batch reader. We really just need the batch reader,
    // but hdfs has an issue in the order in which objects are garbage
    // collected, where Java will print error on trying to close a file
    // saying that the filesystem is already closed. This happens when done
    // reading.
    // Keeping a reference to the dataset and deleting it in the last place
    // seems to help, but the problem can still occur, so this needs
    // further investigation,
    PyObject* scanner_batches_tup = PyObject_CallMethod(
        pq_mod, "get_scanner_batches", "OOOdiOOllOOO", fnames_list_py,
        expr_filters, selected_fields_py, avg_num_pieces, int(parallel),
        this->filesystem, str_as_dict_cols_py, this->start_row_first_piece,
        this->count, this->ds_partitioning, pyarrow_schema, batch_size_py);
    if (scanner_batches_tup == NULL && PyErr_Occurred()) {
        throw std::runtime_error("python");
    }

    // PyTuple_GetItem returns a borrowed reference
    this->reader = PyTuple_GetItem(scanner_batches_tup, 0);
    Py_INCREF(this->reader);  // call incref to keep the reference

    this->rows_to_skip = PyLong_AsLong(PyTuple_GetItem(scanner_batches_tup, 1));
    this->rows_left_cur_piece = pieces_nrows[0];

    Py_DECREF(pq_mod);
    Py_DECREF(expr_filters);
    Py_DECREF(selected_fields_py);
    Py_DECREF(storage_options);
    Py_DECREF(this->filesystem);
    Py_DECREF(fnames_list_py);
    Py_DECREF(str_as_dict_cols_py);
    Py_DECREF(pyarrow_schema);
    Py_DECREF(this->ds_partitioning);
    Py_DECREF(batch_size_py);

    Py_DECREF(scanner_batches_tup);
}

std::shared_ptr<table_info> ParquetReader::get_empty_out_table() {
    if (this->empty_out_table != nullptr) {
        return this->empty_out_table;
    }

    TableBuilder builder(schema, selected_fields, 0, is_nullable,
                         str_as_dict_colnames,
                         create_dict_encoding_from_strings);
    auto out_table = std::shared_ptr<table_info>(builder.get_table());

    if (part_cols.size() > 0) {
        std::vector<std::shared_ptr<array_info>> batch_part_cols;
        for (size_t i = 0; i < part_cols.size(); i++) {
            batch_part_cols.push_back(alloc_array_top_level(
                0, -1, -1, bodo_array_type::NUMPY,
                Bodo_CTypes::CTypeEnum(this->part_cols_cat_dtype[i]), 0, -1));
        }

        out_table->columns.insert(out_table->columns.end(),
                                  batch_part_cols.begin(),
                                  batch_part_cols.end());
    }
    this->empty_out_table = out_table;

    return out_table;
}

std::tuple<table_info*, bool, uint64_t> ParquetReader::read_inner() {
    // If batch_size is set, then we need to iteratively read a
    // batch at a time. For now, ignore partitioning
    if (batch_size != -1) {
        // TODO: Consolidate behavior with SnowflakeReader
        // Specifically, what does SnowflakeReader do in zero-col case?
        if (rows_left_to_emit == 0) {
            return std::make_tuple(new table_info(*get_empty_out_table()), true,
                                   0);
        }

        PyObject* batch_py =
            PyObject_CallMethod(this->reader, "read_next_batch", NULL);
        if (batch_py == NULL && PyErr_Occurred() &&
            PyErr_ExceptionMatches(PyExc_StopIteration)) {
            // StopIteration is raised at the end of iteration.
            // An iterator would clear this automatically, but we are
            // not using an interator, so we have to clear it manually
            PyErr_Clear();
        } else if (batch_py == NULL && PyErr_Occurred()) {
            throw std::runtime_error("python");
        }

        auto batch_res = arrow::py::unwrap_batch(batch_py);
        if (!batch_res.ok()) {
            throw std::runtime_error(
                "ParquetReader::read_batch(): Error unwrapping batch: " +
                batch_res.status().ToString());
        }
        auto batch = batch_res.ValueOrDie();
        if (batch == nullptr) {
            throw std::runtime_error(
                "ParquetReader::read_batch(): The next batch is null");
        }

        int64_t batch_offset = std::min(rows_to_skip, batch->num_rows());
        int64_t length =
            std::min(rows_left_to_read, batch->num_rows() - batch_offset);
        auto table = arrow::Table::Make(
            schema, batch->Slice(batch_offset, length)->columns());
        rows_left_to_read -= length;
        rows_to_skip -= batch_offset;

        // TODO: Replace with arrow_table_to_bodo once available
        TableBuilder builder(schema, selected_fields, length, is_nullable,
                             str_as_dict_colnames,
                             create_dict_encoding_from_strings);
        builder.append(table);
        table_info* out_table = builder.get_table();

        if (part_cols.size() > 0) {
            std::vector<std::shared_ptr<array_info>> batch_part_cols;
            for (size_t i = 0; i < part_cols.size(); i++) {
                batch_part_cols.push_back(alloc_array_top_level(
                    length, -1, -1, bodo_array_type::NUMPY,
                    Bodo_CTypes::CTypeEnum(this->part_cols_cat_dtype[i]), 0,
                    -1));
            }

            // TODO: I believe the reader already makes sure
            // every chunk belongs to 1 file. Can simplify
            // or use the tagged RecordBatchReader
            std::vector<int64_t> batch_offsets(part_cols.size(), 0);
            int64_t unfilled_rows_left = length;
            while (unfilled_rows_left > 0) {
                int64_t rows_read_from_piece =
                    std::min(rows_left_cur_piece, unfilled_rows_left);
                rows_left_cur_piece -= rows_read_from_piece;
                unfilled_rows_left -= rows_read_from_piece;

                // fill partition cols
                for (size_t i = 0; i < part_cols.size(); i++) {
                    int64_t part_val =
                        part_vals[cur_piece][selected_part_cols[i]];
                    fill_partition_column(
                        part_val, batch_part_cols[i]->data1(), batch_offsets[i],
                        rows_read_from_piece, part_cols[i]->dtype);
                    batch_offsets[i] += rows_read_from_piece;
                }
                if (rows_left_cur_piece == 0 &&
                    cur_piece < get_num_pieces() - 1) {
                    rows_left_cur_piece = pieces_nrows[++cur_piece];
                }
            }

            // Insert Partition Columns
            out_table->columns.insert(out_table->columns.end(),
                                      batch_part_cols.begin(),
                                      batch_part_cols.end());
        }
        rows_left_to_emit -= length;
        bool is_last = rows_left_to_emit <= 0;
        Py_DECREF(batch_py);
        return std::make_tuple(out_table, is_last, length);
    }

    TableBuilder builder(schema, selected_fields, count, is_nullable,
                         str_as_dict_colnames,
                         create_dict_encoding_from_strings);

    if (get_num_pieces() == 0) {
        table_info* table = builder.get_table();
        // Insert Blank Partition Columns
        table->columns.insert(table->columns.end(), part_cols.begin(),
                              part_cols.end());
        return std::make_tuple(table, true, 0);
    }

    size_t cur_piece = 0;
    int64_t rows_left_cur_piece = pieces_nrows[cur_piece];

    PyObject* batch_py = NULL;
    while ((batch_py =
                PyObject_CallMethod(this->reader, "read_next_batch", NULL))) {
        auto batch = arrow::py::unwrap_batch(batch_py).ValueOrDie();
        int64_t batch_offset = std::min(rows_to_skip, batch->num_rows());
        int64_t length =
            std::min(rows_left_to_read, batch->num_rows() - batch_offset);
        if (length > 0) {
            // this is zero-copy slice
            auto table = arrow::Table::Make(
                schema, batch->Slice(batch_offset, length)->columns());
            builder.append(table);
            rows_left_to_read -= length;

            // handle partition columns and input_file_name column
            if (part_cols.size() > 0) {
                while (length > 0) {
                    int64_t rows_read_from_piece =
                        std::min(rows_left_cur_piece, length);
                    rows_left_cur_piece -= rows_read_from_piece;
                    length -= rows_read_from_piece;
                    // fill partition cols
                    for (size_t i = 0; i < part_cols.size(); i++) {
                        int64_t part_val =
                            part_vals[cur_piece][selected_part_cols[i]];
                        fill_partition_column(part_val, part_cols[i]->data1(),
                                              part_cols_offset[i],
                                              rows_read_from_piece,
                                              part_cols[i]->dtype);
                        part_cols_offset[i] += rows_read_from_piece;
                    }
                    if (rows_left_cur_piece == 0 &&
                        cur_piece < get_num_pieces() - 1) {
                        rows_left_cur_piece = pieces_nrows[++cur_piece];
                    }
                }
            }
        }
        rows_to_skip -= batch_offset;
        Py_DECREF(batch_py);
    }

    if (batch_py == NULL && PyErr_Occurred() &&
        PyErr_ExceptionMatches(PyExc_StopIteration)) {
        // StopIteration is raised at the end of iteration.
        // An iterator would clear this automatically, but we are
        // not using an interator, so we have to clear it manually
        PyErr_Clear();
    } else if (batch_py == NULL && PyErr_Occurred()) {
        // If there was a Python error, use the special string "python"
        // `py_entry` functions check for this string to avoid
        // overwriting the global Python exception
        // TODO: Replace with custom exception class
        throw std::runtime_error("python");
    }

    if (this->input_file_name_col) {
        // allocate and fill the dictionary for the
        // dictionary-encoded input_file_name column
        this->input_file_name_col_dict_arr =
            alloc_string_array(Bodo_CTypes::STRING, this->file_paths.size(),
                               this->input_file_name_col_dict_arr_total_chars);
        offset_t* offsets =
            (offset_t*)this->input_file_name_col_dict_arr->data2();
        offsets[0] = 0;
        fill_input_file_name_col_dict(
            this->file_paths, this->prefix,
            this->input_file_name_col_dict_arr->data1(),
            this->input_file_name_col_dict_arr->data2());

        // create the final dictionary-encoded input_file_name
        // column from the indices array and dictionary
        // TODO(njriasan): Are the file names always unique locally?
        this->input_file_name_col_arr =
            create_dict_string_array(this->input_file_name_col_dict_arr,
                                     this->input_file_name_col_indices_arr);
    }

    table_info* table = builder.get_table();

    // Append the partition columns to the final output table
    std::vector<std::shared_ptr<array_info>>& part_cols =
        this->get_partition_cols();
    table->columns.insert(table->columns.end(), part_cols.begin(),
                          part_cols.end());

    // Append the input_file_column to the output table
    if (input_file_name_col) {
        std::shared_ptr<array_info> input_file_name_col_arr =
            this->get_input_file_name_col();
        table->columns.push_back(input_file_name_col_arr);
    }

    // rows_left_to_emit is unnecessary for non-streaming
    // so just set equal to rows_left_to_read
    rows_left_to_emit = rows_left_to_read;
    return std::make_tuple(table, true, table->nrows());
}

void ParquetReader::get_partition_info(PyObject* piece) {
    PyObject* partition_keys_py =
        PyObject_GetAttrString(piece, "partition_keys");
    if (PyList_Size(partition_keys_py) > 0) {
        part_vals.emplace_back();
        std::vector<int64_t>& vals = part_vals.back();

        PyObject* part_keys_iter = PyObject_GetIter(partition_keys_py);
        PyObject* key_val_tuple;
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

/**
 * Read a parquet dataset.
 *
 * @param path : path to parquet dataset (can be a Python string -single path-
 * or a Python list of strings -multiple files constituting a single dataset-).
 * The PyObject is passed through to parquet_pio.py::get_parquet_dataset
 * @param parallel: true if reading in parallel
 * @param filters : PyObject passed to pyarrow.parquet.ParquetDataset filters
 * argument to remove rows from scanned data
 * @param storage_options : PyDict with extra read_parquet options. See
 * https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_parquet.html
 * @param tot_rows_to_read : total number of global rows to read from the
 * dataset (starting from the beginning). Read all rows if is -1
 * @param selected_fields : Fields to select from the parquet dataset,
 * using the field ID of Arrow schema. Note that this doesn't include
 * partition columns (which are not part of the parquet files)
 * NOTE: selected_fields must be sorted
 * @param num_selected_fields : length of selected_fields array
 * @param is_nullable : array of booleans that indicates which of the
 * selected fields is nullable. Same length and order as selected_fields.
 * @param selected_part_cols : Partition columns to select, in the order
 * and index used by pyarrow.parquet.ParquetDataset (e.g. 0 is the first
 * partition col)
 * @param part_cols_cat_dtype : Bodo_CTypes::CTypeEnum dtype of categorical
 * codes array for each partition column.
 * @param num_partition_cols : number of selected partition columns
 * @param[out] total_rows_out : total number of global rows read
 * @param input_file_name_col : boolean specifying whether a column with the
 * the names of the files that each row belongs to should be created
 * @return table containing all the arrays read (first the selected fields
 * followed by selected partition columns, in same order as specified to this
 * function).
 */
table_info* pq_read_py_entry(
    PyObject* path, bool parallel, PyObject* dnf_filters,
    PyObject* expr_filters, PyObject* storage_options, PyObject* pyarrow_schema,
    int64_t tot_rows_to_read, int32_t* _selected_fields,
    int32_t num_selected_fields, int32_t* _is_nullable,
    int32_t* selected_part_cols, int32_t* part_cols_cat_dtype,
    int32_t num_partition_cols, int32_t* _str_as_dict_cols,
    int32_t num_str_as_dict_cols, int64_t* total_rows_out,
    bool input_file_name_col, bool use_hive) {
    try {
        std::set<int> selected_fields(
            {_selected_fields, _selected_fields + num_selected_fields});
        std::vector<bool> is_nullable(_is_nullable,
                                      _is_nullable + num_selected_fields);
        std::span<int32_t> str_as_dict_cols(_str_as_dict_cols,
                                            num_str_as_dict_cols);

        ParquetReader reader(path, parallel, dnf_filters, expr_filters,
                             storage_options, pyarrow_schema, tot_rows_to_read,
                             selected_fields, is_nullable, input_file_name_col,
                             -1, use_hive);

        // Initialize reader
        reader.init_pq_reader(str_as_dict_cols, part_cols_cat_dtype,
                              selected_part_cols, num_partition_cols);
        *total_rows_out = reader.get_total_rows();
        // Actually read contents
        table_info* out = reader.read_all();
        return out;
    } catch (const std::exception& e) {
        // if the error string is "python" this means the C++ exception is
        // a result of a Python exception, so we don't call PyErr_SetString
        // because we don't want to replace the original Python error
        if (std::string(e.what()) != "python") {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
        return NULL;
    }
}

/**
 * Read a parquet dataset.
 *
 * @param path : path to parquet dataset (can be a Python string -single path-
 * or a Python list of strings -multiple files constituting a single dataset-).
 * The PyObject is passed through to parquet_pio.py::get_parquet_dataset
 * @param parallel: true if reading in parallel
 * @param filters : PyObject passed to pyarrow.parquet.ParquetDataset filters
 * argument to remove rows from scanned data
 * @param storage_options : PyDict with extra read_parquet options. See
 * https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_parquet.html
 * @param tot_rows_to_read : total number of global rows to read from the
 * dataset (starting from the beginning). Read all rows if is -1
 * @param selected_fields : Fields to select from the parquet dataset,
 * using the field ID of Arrow schema. Note that this doesn't include
 * partition columns (which are not part of the parquet files)
 * NOTE: selected_fields must be sorted
 * @param num_selected_fields : length of selected_fields array
 * @param is_nullable : array of booleans that indicates which of the
 * selected fields is nullable. Same length and order as selected_fields.
 * @param selected_part_cols : Partition columns to select, in the order
 * and index used by pyarrow.parquet.ParquetDataset (e.g. 0 is the first
 * partition col)
 * @param part_cols_cat_dtype : Bodo_CTypes::CTypeEnum dtype of categorical
 * codes array for each partition column.
 * @param num_partition_cols : number of selected partition columns
 * @param[out] total_rows_out : total number of global rows read
 * @param input_file_name_col : boolean specifying whether a column with the
 * the names of the files that each row belongs to should be created
 * @return table containing all the arrays read (first the selected fields
 * followed by selected partition columns, in same order as specified to this
 * function).
 */
ArrowReader* pq_reader_init_py_entry(
    PyObject* path, bool parallel, PyObject* dnf_filters,
    PyObject* expr_filters, PyObject* storage_options, PyObject* pyarrow_schema,
    int64_t tot_rows_to_read, int32_t* _selected_fields,
    int32_t num_selected_fields, int32_t* _is_nullable,
    int32_t* selected_part_cols, int32_t* part_cols_cat_dtype,
    int32_t num_partition_cols, int32_t* _str_as_dict_cols,
    int32_t num_str_as_dict_cols, bool input_file_name_col, int64_t batch_size,
    bool use_hive) {
    try {
        std::set<int> selected_fields(
            {_selected_fields, _selected_fields + num_selected_fields});
        std::vector<bool> is_nullable(_is_nullable,
                                      _is_nullable + num_selected_fields);
        std::span<int32_t> str_as_dict_cols(_str_as_dict_cols,
                                            num_str_as_dict_cols);

        ParquetReader* reader = new ParquetReader(
            path, parallel, dnf_filters, expr_filters, storage_options,
            pyarrow_schema, tot_rows_to_read, selected_fields, is_nullable,
            input_file_name_col, batch_size, use_hive);

        // Initialize reader
        reader->init_pq_reader(str_as_dict_cols, part_cols_cat_dtype,
                               selected_part_cols, num_partition_cols);

        return static_cast<ArrowReader*>(reader);

    } catch (const std::exception& e) {
        // if the error string is "python" this means the C++ exception is
        // a result of a Python exception, so we don't call PyErr_SetString
        // because we don't want to replace the original Python error
        if (std::string(e.what()) != "python") {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
        return NULL;
    }
}
