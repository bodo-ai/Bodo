// Copyright (C) 2021 Bodo Inc. All rights reserved.

// Implementation of ParquetReader (subclass of ArrowDataframeReader) with
// functionality that is specific to reading parquet datasets

#include "arrow_reader.h"
#include "_fsspec_reader.h"
#include "_hdfs_reader.h"
#include "_s3_reader.h"
#include "arrow/io/hdfs.h"
#include "parquet/api/reader.h"
#include "parquet/arrow/reader.h"

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
 * Open Arrow parquet file reader (parquet::arrow::FileReader).
 * @param[in] file_name : Path of parquet file to open
 * @param[out] a_reader : pointer to file reader
 * @param[in] bucket_region : S3 bucket region (when reading from S3)
 * @param[in] s3fs_anon : true if anonymous access (when reading from S3)
 */
void pq_init_reader(const char* file_name,
                    std::shared_ptr<parquet::arrow::FileReader>* a_reader,
                    const char* bucket_region, bool s3fs_anon);

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

// -------- ParquetReader --------

class ParquetReader : public ArrowDataframeReader {
   public:
    /**
     * Initialize ParquetReader.
     * See pq_read function below for description of arguments.
     */
    ParquetReader(char* _path, bool _parallel, char* _bucket_region,
                  PyObject* _dnf_filters, PyObject* _expr_filters,
                  PyObject* _storage_options, int64_t _tot_rows_to_read,
                  int32_t* _selected_fields, int32_t num_selected_fields,
                  int32_t* is_nullable, int32_t* _selected_part_cols,
                  int32_t* _part_cols_cat_dtype, int32_t num_partition_cols)
        : ArrowDataframeReader(_parallel, _tot_rows_to_read, _selected_fields,
                               num_selected_fields, is_nullable),
          path(_path),
          bucket_region(_bucket_region),
          dnf_filters(_dnf_filters),
          expr_filters(_expr_filters),
          storage_options(_storage_options) {
        if (storage_options == Py_None)
            throw std::runtime_error("ParquetReader: storage_options is None");

        // copy selected_fields to a Python list to pass to
        // parquet_pio.get_scanner_batches
        selected_fields_py = PyList_New(selected_fields.size());
        size_t i = 0;
        for (auto field_num : selected_fields) {
            PyList_SetItem(selected_fields_py, i++, PyLong_FromLong(field_num));
        }

        // Extract values from the storage_options dict
        // Check that it's a dictionary, else throw an error
        if (PyDict_Check(storage_options)) {
            // Get value of "anon". Returns NULL if it doesn't exist in the
            // dict. No need to decref s3fs_anon_py, PyDict_GetItemString
            // returns borrowed ref
            PyObject* s3fs_anon_py =
                PyDict_GetItemString(storage_options, "anon");
            if (s3fs_anon_py != NULL && s3fs_anon_py == Py_True) {
                this->s3fs_anon = true;
            }
        } else {
            throw std::runtime_error(
                "ParquetReader: storage_options is not a Python dictionary.");
        }

        // initialize reader
        init();

        // allocate output partition columns. These are categorical columns where
        // we only fill out the codes in C++ (see fill_partition_column comments)
        for (auto i = 0; i < num_partition_cols; i++) {
            part_cols.push_back(alloc_array(
                count, -1, -1, bodo_array_type::NUMPY,
                Bodo_CTypes::CTypeEnum(_part_cols_cat_dtype[i]), 0, -1));
            selected_part_cols.push_back(_selected_part_cols[i]);
        }
        part_cols_offset.resize(num_partition_cols, 0);
    }

    virtual ~ParquetReader() {
        Py_DECREF(expr_filters);
        Py_DECREF(selected_fields_py);
    }

    /// a piece is a single parquet file in the context of parquet
    virtual size_t get_num_pieces() const { return file_paths.size(); }

    /// returns output partition columns
    std::vector<array_info*>& get_partition_cols() { return part_cols; }

   protected:
    virtual void add_piece(PyObject* piece) {
        // p = piece.path
        PyObject* p = PyObject_GetAttrString(piece, "path");
        const char* c_path = PyUnicode_AsUTF8(p);
        file_paths.emplace_back(c_path);
        // for this parquet file: store partition value of each partition column
        get_partition_info(piece);
        Py_DECREF(p);
    }

    virtual PyObject* get_dataset() {
        // import bodo.io.parquet_pio
        PyObject* pq_mod = PyImport_ImportModule("bodo.io.parquet_pio");

        // ds = bodo.io.parquet_pio.get_parquet_dataset(path, true, filters,
        // storage_options)
        PyObject* ds = PyObject_CallMethod(
            pq_mod, "get_parquet_dataset", "sOOOOOO", path, Py_True,
            dnf_filters, expr_filters, storage_options, Py_False,
            PyBool_FromLong(parallel));
        Py_DECREF(dnf_filters);
        Py_DECREF(storage_options);
        Py_DECREF(pq_mod);
        if (PyErr_Occurred()) throw std::runtime_error("python");

        // prefix = ds.prefix
        PyObject* prefix_py = PyObject_GetAttrString(ds, "_prefix");
        this->prefix.assign(PyUnicode_AsUTF8(prefix_py));
        Py_DECREF(prefix_py);

        return ds;
    }

    virtual std::shared_ptr<arrow::Schema> get_schema(PyObject* dataset) {
        PyObject* pq_mod = PyImport_ImportModule("bodo.io.pa_parquet");
        PyObject* schema_py =
            PyObject_CallMethod(pq_mod, "get_dataset_schema", "O", dataset);
        Py_DECREF(pq_mod);
        // see
        // https://arrow.apache.org/docs/python/extending.html#using-pyarrow-from-c-and-cython-code
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

    virtual void append_piece_builder(size_t piece_idx, TableBuilder& builder) {
        int64_t rows_read_from_piece = 0;
        const std::string fpath = prefix + file_paths[piece_idx];
        if (expr_filters != nullptr && expr_filters != Py_None) {
            // Use a different path for predicate pushdown.
            // If expr_filters==None this path is slower than the original one
            // below (seems to be about 15% slower)
            // TODO: see if it can be optimized
            // Also the memory usage pattern is likely different (see next comment)
            int64_t rows_to_skip = 0;
            if (piece_idx == 0) rows_to_skip = start_row_first_piece;
            PyObject* pq_mod = PyImport_ImportModule("bodo.io.parquet_pio");
            // XXX The get_batches method of scanner loads data into memory. The
            // question is whether it loads all of its batches into memory
            // before returning (I think it does)
            // This only loads record batches that match the filter
            PyObject* batches_it = PyObject_CallMethod(
                pq_mod, "get_scanner_batches", "sOO", fpath.c_str(),
                expr_filters, selected_fields_py);
            Py_DECREF(pq_mod);
            PyObject* batch_py = NULL;
            while ((batch_py = PyIter_Next(batches_it))) {
                auto batch = arrow::py::unwrap_batch(batch_py).ValueOrDie();
                int64_t batch_offset =
                    std::min(rows_to_skip, batch->num_rows());
                int64_t length =
                    std::min(rows_left, batch->num_rows() - batch_offset);
                if (length > 0) {
                    // this is zero-copy slice
                    auto table = arrow::Table::Make(
                        schema, batch->Slice(batch_offset, length)->columns());
                    builder.append(table);
                    rows_left -= length;
                    rows_read_from_piece += length;
                }
                rows_to_skip -= batch_offset;
                Py_DECREF(batch_py);
            }
            Py_DECREF(batches_it);
        } else {
            // open file corresponding to this piece
            std::shared_ptr<parquet::arrow::FileReader> arrow_reader;
            pq_init_reader(fpath.c_str(), &arrow_reader, bucket_region.c_str(),
                           s3fs_anon);

            // get number of row groups in this file
            auto pq_metadata = arrow_reader->parquet_reader()->metadata();
            int64_t n_row_groups = pq_metadata->num_row_groups();

            // skip row groups if necessary based on start_row_first_piece
            int64_t rg_index = 0;  // current row group index
            int64_t rg_rows_to_skip = 0;
            if (piece_idx == 0 && start_row_first_piece > 0) {
                int64_t skipped_rows =
                    0;  // total number of rows I have skipped of this piece
                while (skipped_rows < start_row_first_piece) {
                    auto rg_metadata = pq_metadata->RowGroup(rg_index);
                    rg_rows_to_skip =
                        std::min(start_row_first_piece - skipped_rows,
                                 rg_metadata->num_rows());
                    skipped_rows += rg_rows_to_skip;
                    if (rg_rows_to_skip == rg_metadata->num_rows()) {
                        rg_index++;
                        rg_rows_to_skip = 0;
                    }
                }
                if (rg_index >= n_row_groups)
                    throw std::runtime_error("parquet read error");
            }

            // read data
            while (rg_index < n_row_groups && rows_left > 0) {
                std::shared_ptr<::arrow::Table> table;
                // read row group into Arrow table. We can't pass Arrow field
                // selection to ReadRowGroup, have to pass column selection
                arrow::Status status = arrow_reader->ReadRowGroup(
                    rg_index, selected_columns, &table);
                auto rg_metadata = pq_metadata->RowGroup(rg_index);
                int64_t length = std::min(
                    rows_left, rg_metadata->num_rows() - rg_rows_to_skip);
                // pass zero-copy slice to TableBuilder to append to read data
                builder.append(table->Slice(rg_rows_to_skip, length));
                rows_left -= length;
                rows_read_from_piece += length;
                // all of the next row groups start at 0
                rg_rows_to_skip = 0;
                rg_index++;
            }
        }
        // fill partition cols
        for (auto i = 0; i < part_cols.size(); i++) {
            int64_t part_val = part_vals[piece_idx][selected_part_cols[i]];
            fill_partition_column(part_val, part_cols[i]->data1,
                                  part_cols_offset[i], rows_read_from_piece,
                                  part_cols[i]->dtype);
            part_cols_offset[i] += rows_read_from_piece;
        }
    }

   private:
    const char* path;  // path passed to pd.read_parquet() call
    PyObject* dnf_filters = nullptr;
    PyObject* expr_filters = nullptr;
    PyObject* storage_options;
    PyObject* selected_fields_py;

    // Parquet files that this process has to read
    std::vector<std::string> file_paths;
    // Prefix to add to each of the file paths before they are opened
    std::string prefix;
    std::string bucket_region;  // s3 bucket region
    bool s3fs_anon = false;     // s3 anonymous mode

    // selected columns in the parquet file (not fields). For example,
    // field "struct<A: int64, B: int64>" has two int64 columns in the
    // parquet file
    std::vector<int> selected_columns;

    // selected partition columns
    std::vector<int> selected_part_cols;
    // for each piece that this process reads, store the value of each partition
    // column (value is stored as the categorical code). Note that a given
    // piece/file has the same partition value for all of its rows
    std::vector<std::vector<int64_t>> part_vals;
    // output partition columns
    std::vector<array_info*> part_cols;
    // current fill offset of each partition column
    std::vector<int64_t> part_cols_offset;

    /**
     * Get values for all partition columns of a piece of
     * pyarrow.parquet.ParquetDataset and store in part_vals.
     * @param piece : ParquetDataset piece (a single parquet file)
     */
    void get_partition_info(PyObject* piece) {
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
};

void pq_init_reader(const char* file_name,
                    std::shared_ptr<parquet::arrow::FileReader>* a_reader,
                    const char* bucket_region, bool s3fs_anon) {
    PyObject* f_mod;
    std::string f_name(file_name);
    auto pool = ::arrow::default_memory_pool();
    arrow::Status status;

    size_t colon = f_name.find_first_of(':');
    std::string protocol = f_name.substr(0, colon);
    std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
    // HDFS if starts with hdfs://, abfs:// or abfss://
    if (f_name.find("hdfs://") == 0 || f_name.find("abfs://") == 0 ||
        f_name.find("abfss://") == 0) {
        std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
        std::shared_ptr<::arrow::io::RandomAccessFile> file;
        // open Parquet file
        hdfs_open_file(file_name, &file);
        // create Arrow reader
        status = parquet::arrow::FileReader::Make(
            pool, ParquetFileReader::Open(file), &arrow_reader);
        CHECK_ARROW(status, "parquet::arrow::FileReader::Make");
        *a_reader = std::move(arrow_reader);
    // S3
    } else if (f_name.find("s3://") == 0) {
        std::shared_ptr<::arrow::io::RandomAccessFile> file;
        // remove s3://
        f_name = f_name.substr(strlen("s3://"));
        // get s3 opener function
        // open Parquet file
        s3_open_file(f_name.c_str(), &file, bucket_region, s3fs_anon);
        // create Arrow reader
        status = parquet::arrow::FileReader::Make(
            pool, ParquetFileReader::Open(file), &arrow_reader);
        CHECK_ARROW(status, "parquet::arrow::FileReader::Make");
        *a_reader = std::move(arrow_reader);
    // Google Cloud Storage
    } else if (protocol == "gcs" || pyfs.count(protocol)) {
        std::shared_ptr<::arrow::io::RandomAccessFile> file;
        fsspec_open_file(f_name, protocol, &file);
        status = parquet::arrow::FileReader::Make(
            pool, ParquetFileReader::Open(file), &arrow_reader);
        CHECK_ARROW(status, "parquet::arrow::FileReader::Make");
        *a_reader = std::move(arrow_reader);
    } else  // regular file system
    {
        status = parquet::arrow::FileReader::Make(
            pool, ParquetFileReader::OpenFile(f_name, false), &arrow_reader);
        CHECK_ARROW(status, "parquet::arrow::FileReader::Make");
        *a_reader = std::move(arrow_reader);
    }
    return;
}

/**
 * Read a parquet dataset.
 *
 * @param path : path to parquet dataset
 * @param parallel: true if reading in parallel
 * @param bucket_region : S3 bucket region (when reading from S3)
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
 * @param is_nullable : array of bools that indicates which of the
 * selected fields is nullable. Same length and order as selected_fields.
 * @param selected_part_cols : Partition columns to select, in the order
 * and index used by pyarrow.parquet.ParquetDataset (e.g. 0 is the first
 * partition col)
 * @param part_cols_cat_dtype : Bodo_CTypes::CTypeEnum dtype of categorical
 * codes array for each partition column.
 * @param num_partition_cols : number of selected partition columns
 * @param[out] total_rows_out : total number of global rows read
 * @return table containing all the arrays read (first the selected fields
 * followed by selected partition columns, in same order as specified to this
 * function).
 */
table_info* pq_read(char* path, bool parallel, char* bucket_region,
                    PyObject* dnf_filters, PyObject* expr_filters,
                    PyObject* storage_options, int64_t tot_rows_to_read,
                    int32_t* selected_fields, int32_t num_selected_fields,
                    int32_t* is_nullable, int32_t* selected_part_cols,
                    int32_t* part_cols_cat_dtype, int32_t num_partition_cols,
                    int64_t* total_rows_out) {
    try {
        ParquetReader reader(path, parallel, bucket_region, dnf_filters,
                             expr_filters, storage_options, tot_rows_to_read,
                             selected_fields, num_selected_fields, is_nullable,
                             selected_part_cols, part_cols_cat_dtype,
                             num_partition_cols);
        *total_rows_out = reader.get_total_rows();
        table_info* out = reader.read();
        // append the partition columns to the final output table
        std::vector<array_info*>& part_cols = reader.get_partition_cols();
        out->columns.insert(out->columns.end(), part_cols.begin(),
                            part_cols.end());
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
