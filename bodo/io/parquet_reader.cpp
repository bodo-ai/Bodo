// Copyright (C) 2021 Bodo Inc. All rights reserved.

// Implementation of ParquetReader (subclass of ArrowDataframeReader) with
// functionality that is specific to reading parquet datasets

#include "_fsspec_reader.h"
#include "_hdfs_reader.h"
#include "_s3_reader.h"
#include "arrow/io/hdfs.h"
#include "arrow_reader.h"
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

/**
 * Fill a range in an output input_file_name column with a given value.
 * NOTE: input_file_name columns are always string columns.
 * @param fname : name of the file to fill
 * @param data : output buffer
 * @param offsets : offsets buffer in the underlying array_info
 * @param start_idx : index to start filling at
 * @param rows_to_fill : number of elements to write
 */
static void fill_input_file_name_column(const char* fname, char* data,
                                        char* offsets, int64_t start_idx,
                                        int64_t rows_to_fill) {
    offset_t* col_offsets = (offset_t*)offsets;
    offset_t start_offset = col_offsets[start_idx];
    int64_t fname_len = std::strlen(fname);
    for (int64_t i = 0; i < rows_to_fill; i++) {
        memcpy(data + start_offset + (i * fname_len), fname,
               sizeof(char) * fname_len);
        col_offsets[start_idx + i + 1] = col_offsets[start_idx + i] + fname_len;
    }
}

// -------- ParquetReader --------

class ParquetReader : public ArrowDataframeReader {
   public:
    /**
     * Initialize ParquetReader.
     * See pq_read function below for description of arguments.
     */
    ParquetReader(PyObject* _path, bool _parallel, char* _bucket_region,
                  PyObject* _dnf_filters, PyObject* _expr_filters,
                  PyObject* _storage_options, int64_t _tot_rows_to_read,
                  int32_t* _selected_fields, int32_t num_selected_fields,
                  int32_t* is_nullable, int32_t* _selected_part_cols,
                  int32_t* _part_cols_cat_dtype, int32_t num_partition_cols,
                  bool _input_file_name_col)
        : ArrowDataframeReader(_parallel, _tot_rows_to_read, _selected_fields,
                               num_selected_fields, is_nullable),
          path(_path),
          bucket_region(_bucket_region),
          dnf_filters(_dnf_filters),
          expr_filters(_expr_filters),
          storage_options(_storage_options),
          input_file_name_col(_input_file_name_col) {
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

        if (parallel) {
            // Get the average number of pieces per rank. This is used to
            // increase the number of threads of the Arrow batch reader
            // for ranks that have to read many more files than others.
            int num_ranks;
            MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
            uint64_t num_pieces = static_cast<uint64_t>(get_num_pieces());
            MPI_Allreduce(MPI_IN_PLACE, &num_pieces, 1, MPI_UINT64_T, MPI_SUM,
                          MPI_COMM_WORLD);
            avg_num_pieces = num_pieces / static_cast<double>(num_ranks);
        }

        // allocate output partition columns. These are categorical columns
        // where we only fill out the codes in C++ (see fill_partition_column
        // comments)
        for (auto i = 0; i < num_partition_cols; i++) {
            part_cols.push_back(alloc_array(
                count, -1, -1, bodo_array_type::NUMPY,
                Bodo_CTypes::CTypeEnum(_part_cols_cat_dtype[i]), 0, -1));
            selected_part_cols.push_back(_selected_part_cols[i]);
        }
        part_cols_offset.resize(num_partition_cols, 0);

        // allocate input_file_name column if one is specified. This
        // is a string array
        // TODO Convert to Dictionary-encoded string array in the future
        if (input_file_name_col) {
            this->input_file_name_col_arr = alloc_array(
                this->count, this->input_file_name_col_total_chars, -1,
                bodo_array_type::STRING, Bodo_CTypes::CTypeEnum::STRING, 0, -1);
            offset_t* offsets = (offset_t*)input_file_name_col_arr->data2;
            offsets[0] = 0;
        }
    }

    virtual ~ParquetReader() {}

    /// a piece is a single parquet file in the context of parquet
    virtual size_t get_num_pieces() const { return file_paths.size(); }

    /// returns output partition columns
    std::vector<array_info*>& get_partition_cols() { return part_cols; }

    array_info* get_input_file_name_col() { return input_file_name_col_arr; }

   protected:
    virtual void add_piece(PyObject* piece, int64_t num_rows) {
        // p = piece.path
        PyObject* p = PyObject_GetAttrString(piece, "path");
        const char* c_path = PyUnicode_AsUTF8(p);
        file_paths.emplace_back(c_path);
        pieces_nrows.push_back(num_rows);
        this->input_file_name_col_total_chars +=
            (num_rows * (std::strlen(c_path) + this->prefix.length()));
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
            pq_mod, "get_parquet_dataset", "OOOOOOOL", path, Py_True,
            dnf_filters, expr_filters, storage_options, Py_False,
            PyBool_FromLong(parallel), tot_rows_to_read);
        Py_DECREF(path);
        Py_DECREF(dnf_filters);
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

    virtual void read_all(TableBuilder& builder) {
        if (get_num_pieces() == 0) return;

        int64_t rows_read_so_far = 0;
        size_t cur_piece = 0;
        int64_t rows_left_cur_piece = pieces_nrows[cur_piece];

        PyObject* fnames_list_py = PyList_New(file_paths.size());
        size_t i = 0;
        for (auto p : file_paths) {
            PyList_SetItem(fnames_list_py, i++,
                           PyUnicode_FromString(p.c_str()));
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
        PyObject* dataset_batches_tup = PyObject_CallMethod(
            pq_mod, "get_scanner_batches", "OOOdiOss", fnames_list_py,
            expr_filters, selected_fields_py, avg_num_pieces, int(parallel),
            storage_options, bucket_region.c_str(), prefix.c_str());
        // PyTuple_GetItem returns a borrowed reference
        PyObject* dataset = PyTuple_GetItem(dataset_batches_tup, 0);
        Py_INCREF(dataset);  // call incref to keep the reference
        PyObject* batches_it = PyTuple_GetItem(dataset_batches_tup, 1);
        Py_DECREF(pq_mod);
        Py_DECREF(expr_filters);
        Py_DECREF(selected_fields_py);
        Py_DECREF(storage_options);
        Py_DECREF(fnames_list_py);
        PyObject* batch_py = NULL;
        // while ((batch_py = PyIter_Next(batches_it))) {  // XXX Fails with
        // batch reader returned by scanner
        while ((batch_py =
                    PyObject_CallMethod(batches_it, "read_next_batch", NULL))) {
            auto batch = arrow::py::unwrap_batch(batch_py).ValueOrDie();
            int64_t batch_offset = std::min(rows_to_skip, batch->num_rows());
            int64_t length =
                std::min(rows_left, batch->num_rows() - batch_offset);
            if (length > 0) {
                // this is zero-copy slice
                auto table = arrow::Table::Make(
                    schema, batch->Slice(batch_offset, length)->columns());
                builder.append(table);
                rows_left -= length;

                // handle partition columns and input_file_name column
                if (part_cols.size() > 0 || input_file_name_col) {
                    while (length > 0) {
                        int64_t rows_read_from_piece =
                            std::min(rows_left_cur_piece, length);
                        rows_left_cur_piece -= rows_read_from_piece;
                        length -= rows_read_from_piece;
                        if (part_cols.size() > 0) {
                            // fill partition cols
                            for (auto i = 0; i < part_cols.size(); i++) {
                                int64_t part_val =
                                    part_vals[cur_piece][selected_part_cols[i]];
                                fill_partition_column(
                                    part_val, part_cols[i]->data1,
                                    part_cols_offset[i], rows_read_from_piece,
                                    part_cols[i]->dtype);
                                part_cols_offset[i] += rows_read_from_piece;
                            }
                        }
                        if (this->input_file_name_col) {
                            // fill the input_file_name column
                            std::string fname =
                                this->prefix + this->file_paths[cur_piece];
                            fill_input_file_name_column(
                                fname.c_str(),
                                this->input_file_name_col_arr->data1,
                                this->input_file_name_col_arr->data2,
                                rows_read_so_far, rows_read_from_piece);
                        }
                        if (rows_left_cur_piece == 0 &&
                            cur_piece < get_num_pieces() - 1) {
                            rows_left_cur_piece = pieces_nrows[++cur_piece];
                        }
                        rows_read_so_far += rows_read_from_piece;
                    }
                }
            }
            rows_to_skip -= batch_offset;
            Py_DECREF(batch_py);
        }
        if (batch_py == NULL && PyErr_Occurred() &&
            PyErr_ExceptionMatches(PyExc_StopIteration))
            // StopIteration is raised at the end of iteration.
            // An iterator would clear this automatically, but we are
            // not using an interator, so we have to clear it manually
            PyErr_Clear();
        Py_DECREF(dataset_batches_tup);
        Py_DECREF(dataset);  // delete dataset last
    }

   private:
    PyObject* path;  // path passed to pd.read_parquet() call
    PyObject* dnf_filters = nullptr;
    PyObject* expr_filters = nullptr;
    PyObject* storage_options;
    PyObject* selected_fields_py;
    bool input_file_name_col;

    std::vector<int64_t> pieces_nrows;
    double avg_num_pieces = 0;

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
    // output input_file_name column
    int64_t input_file_name_col_total_chars = 0;
    array_info* input_file_name_col_arr = nullptr;

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
    } else if (protocol == "gcs" || protocol == "gs" || pyfs.count(protocol)) {
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
 * @param path : path to parquet dataset (can be a Python string -single path-
 * or a Python list of strings -multiple files constituting a single dataset-).
 * The PyObject is passed throught to parquet_pio.py::get_parquet_dataset
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
 * @param input_file_name_col : boolean specifying whether a column with the
 * the names of the files that each row belongs to should be created
 * @return table containing all the arrays read (first the selected fields
 * followed by selected partition columns, in same order as specified to this
 * function).
 */
table_info* pq_read(PyObject* path, bool parallel, char* bucket_region,
                    PyObject* dnf_filters, PyObject* expr_filters,
                    PyObject* storage_options, int64_t tot_rows_to_read,
                    int32_t* selected_fields, int32_t num_selected_fields,
                    int32_t* is_nullable, int32_t* selected_part_cols,
                    int32_t* part_cols_cat_dtype, int32_t num_partition_cols,
                    int64_t* total_rows_out, bool input_file_name_col) {
    try {
        ParquetReader reader(path, parallel, bucket_region, dnf_filters,
                             expr_filters, storage_options, tot_rows_to_read,
                             selected_fields, num_selected_fields, is_nullable,
                             selected_part_cols, part_cols_cat_dtype,
                             num_partition_cols, input_file_name_col);
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
