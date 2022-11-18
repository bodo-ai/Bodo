// Copyright (C) 2021 Bodo Inc. All rights reserved.

// Implementation of ParquetReader (subclass of ArrowDataframeReader) with
// functionality that is specific to reading parquet datasets

#include "parquet_reader.h"

using arrow::Type;
using parquet::ParquetFileReader;

#define CHECK_ARROW(expr, msg)                                                 \
    if (!(expr.ok())) {                                                        \
        std::string err_msg = std::string("Error in arrow parquet reader: ") + \
                              msg + " " + expr.ToString();                     \
        throw std::runtime_error(err_msg);                                     \
    }

// -------- Helper functions --------

/**
 * Get number of columns in a parquet file that correspond to a field
 * in the Arrow schema. For example, a struct with two int64 fields consists
 * of two columns of data in the parquet file.
 * @param : field
 * @return Number of columns
 */
static int get_num_columns(const std::shared_ptr<arrow::Field> field) {
    if (field->type()->num_fields() == 0) {
        // if field has no children then it only consists of 1 column
        return 1;
    } else {
        // get number of leaves recursively. Each leaf is a column.
        int num_leaves = 0;
        for (auto f : field->type()->fields()) num_leaves += get_num_columns(f);
        return num_leaves;
    }
}

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
    const std::vector<std::string>& file_paths, const std::string& prefix,
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
            file_paths.size() - 1, this->input_file_name_col_indices_arr->data1,
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
        pq_mod, "get_parquet_dataset", "OOOOOOOLO", path, Py_True, dnf_filters,
        expr_filters, storage_options, Py_False, PyBool_FromLong(parallel),
        tot_rows_to_read, this->use_hive ? Py_True : Py_False);
    if (ds == NULL && PyErr_Occurred()) {
        throw std::runtime_error("python");
    }
    this->ds_partitioning = PyObject_GetAttrString(ds, "partitioning");
    Py_DECREF(path);
    Py_DECREF(dnf_filters);
    Py_DECREF(pq_mod);
    if (PyErr_Occurred()) throw std::runtime_error("python");

    // prefix = ds.prefix
    PyObject* prefix_py = PyObject_GetAttrString(ds, "_prefix");
    this->prefix.assign(PyUnicode_AsUTF8(prefix_py));
    Py_DECREF(prefix_py);

    this->filesystem = PyObject_GetAttrString(ds, "filesystem");

    return ds;
}

std::shared_ptr<arrow::Schema> ParquetReader::get_schema(PyObject* dataset) {
    PyObject* schema_py = PyObject_GetAttrString(dataset, "schema");
    // see
    // https://arrow.apache.org/docs/8.0/python/integration/extending.html?highlight=unwrap_schema
    auto schema_ = arrow::py::unwrap_schema(schema_py).ValueOrDie();
    // calculate selected columns (not fields)
    int col = 0;
    for (int i = 0; i < schema_->num_fields(); i++) {
        auto f = schema_->field(i);
        int field_num_columns = get_num_columns(f);
        if (selected_fields.find(i) != selected_fields.end()) {
            for (int j = 0; j < field_num_columns; j++)
                selected_columns.push_back(col + j);
        }
        col += field_num_columns;
    }
    Py_DECREF(schema_py);
    return schema_;
}

void ParquetReader::read_all(TableBuilder& builder) {
    if (get_num_pieces() == 0) {
        // get_scanner_batches trace event has to be called by all ranks
        // Adding call here to avoid hangs in tracing.
        // This event is used to track load imbalance so we can't use
        // trace(is_parallel=False) to be able to collect min/max/avg.
        tracing::Event ev_get_scanner_batches("get_scanner_batches");
        return;
    }

    size_t cur_piece = 0;
    int64_t rows_left_cur_piece = pieces_nrows[cur_piece];

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

    int64_t rows_to_skip = start_row_first_piece;
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
    tracing::Event ev_get_scanner_batches("get_scanner_batches");
    PyObject* py_schema = arrow::py::wrap_schema(this->schema);
    PyObject* dataset_batches_tup = PyObject_CallMethod(
        pq_mod, "get_scanner_batches", "OOOdiOOllOO", fnames_list_py,
        expr_filters, selected_fields_py, avg_num_pieces, int(parallel),
        this->filesystem, str_as_dict_cols_py, this->start_row_first_piece,
        this->count, this->ds_partitioning, py_schema);
    if (dataset_batches_tup == NULL && PyErr_Occurred()) {
        throw std::runtime_error("python");
    }

    ev_get_scanner_batches.finalize();
    // PyTuple_GetItem returns a borrowed reference
    PyObject* dataset = PyTuple_GetItem(dataset_batches_tup, 0);
    Py_INCREF(dataset);  // call incref to keep the reference
    PyObject* batches_it = PyTuple_GetItem(dataset_batches_tup, 1);
    rows_to_skip = PyLong_AsLong(PyTuple_GetItem(dataset_batches_tup, 2));
    Py_DECREF(pq_mod);
    Py_DECREF(expr_filters);
    Py_DECREF(selected_fields_py);
    Py_DECREF(storage_options);
    Py_DECREF(this->filesystem);
    Py_DECREF(fnames_list_py);
    Py_DECREF(str_as_dict_cols_py);
    Py_DECREF(py_schema);
    Py_DECREF(this->ds_partitioning);
    PyObject* batch_py = NULL;
    // while ((batch_py = PyIter_Next(batches_it))) {  // XXX Fails with
    // batch reader returned by scanner
    while (
        (batch_py = PyObject_CallMethod(batches_it, "read_next_batch", NULL))) {
        auto batch = arrow::py::unwrap_batch(batch_py).ValueOrDie();
        int64_t batch_offset = std::min(rows_to_skip, batch->num_rows());
        int64_t length = std::min(rows_left, batch->num_rows() - batch_offset);
        if (length > 0) {
            // this is zero-copy slice
            auto table = arrow::Table::Make(
                schema, batch->Slice(batch_offset, length)->columns());
            builder.append(table);
            rows_left -= length;

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
                        fill_partition_column(
                            part_val, part_cols[i]->data1, part_cols_offset[i],
                            rows_read_from_piece, part_cols[i]->dtype);
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
        throw std::runtime_error("python");
    }
    Py_DECREF(dataset_batches_tup);
    Py_DECREF(dataset);  // delete dataset last

    if (this->input_file_name_col) {
        // allocate and fill the dictionary for the
        // dictionary-encoded input_file_name column
        this->input_file_name_col_dict_arr = alloc_string_array(
            this->file_paths.size(),
            this->input_file_name_col_dict_arr_total_chars, 0);
        offset_t* offsets =
            (offset_t*)this->input_file_name_col_dict_arr->data2;
        offsets[0] = 0;
        fill_input_file_name_col_dict(
            this->file_paths, this->prefix,
            this->input_file_name_col_dict_arr->data1,
            this->input_file_name_col_dict_arr->data2);

        // create the final dictionary-encoded input_file_name
        // column from the indices array and dictionary
        this->input_file_name_col_arr = new array_info(
            bodo_array_type::DICT, Bodo_CTypes::CTypeEnum::STRING, this->count,
            -1, -1, NULL, NULL, NULL,
            this->input_file_name_col_indices_arr->null_bitmask, NULL, NULL,
            NULL, NULL, 0, 0, 0, false, false,
            this->input_file_name_col_dict_arr,
            this->input_file_name_col_indices_arr);
    }
}

/**
 * Get values for all partition columns of a piece of
 * pyarrow.parquet.ParquetDataset and store in part_vals.
 * @param piece : ParquetDataset piece (a single parquet file)
 */
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
table_info* pq_read(PyObject* path, bool parallel, PyObject* dnf_filters,
                    PyObject* expr_filters, PyObject* storage_options,
                    int64_t tot_rows_to_read, int32_t* selected_fields,
                    int32_t num_selected_fields, int32_t* is_nullable,
                    int32_t* selected_part_cols, int32_t* part_cols_cat_dtype,
                    int32_t num_partition_cols, int32_t* str_as_dict_cols,
                    int32_t num_str_as_dict_cols, int64_t* total_rows_out,
                    bool input_file_name_col, bool use_hive) {
    try {
        ParquetReader reader(path, parallel, dnf_filters, expr_filters,
                             storage_options, tot_rows_to_read, selected_fields,
                             num_selected_fields, is_nullable,
                             input_file_name_col, use_hive);
        // initialize reader
        reader.init_pq_reader(str_as_dict_cols, num_str_as_dict_cols,
                              part_cols_cat_dtype, selected_part_cols,
                              num_partition_cols);
        *total_rows_out = reader.get_total_rows();
        table_info* out = reader.read();
        // append the partition columns to the final output table
        std::vector<array_info*>& part_cols = reader.get_partition_cols();
        out->columns.insert(out->columns.end(), part_cols.begin(),
                            part_cols.end());
        // append the input_file_column to the output table
        if (input_file_name_col) {
            array_info* input_file_name_col_arr =
                reader.get_input_file_name_col();
            out->columns.push_back(input_file_name_col_arr);
        }
        return out;
    } catch (const std::exception& e) {
        // if the error string is "python" this means the C++ exception is
        // a result of a Python exception, so we don't call PyErr_SetString
        // because we don't want to replace the original Python error
        if (std::string(e.what()) != "python")
            PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}
