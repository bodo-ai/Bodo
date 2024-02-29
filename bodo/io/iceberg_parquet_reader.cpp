// Copyright (C) 2022 Bodo Inc. All rights reserved.

// Implementation of IcebergParquetReader (subclass of ParquetReader) with
// functionality that is specific to reading iceberg datasets (made up of
// parquet files)

#include <arrow/record_batch.h>
#include <arrow/result.h>
#include "arrow_reader.h"

#include <numeric>

// -------- IcebergParquetReader --------

class IcebergParquetReader : public ArrowReader {
   public:
    /**
     * Initialize IcebergParquetReader.
     * See iceberg_pq_read_py_entry function below for description of arguments.
     */
    IcebergParquetReader(const char* _conn, const char* _database_schema,
                         const char* _table_name, bool _parallel,
                         int64_t tot_rows_to_read, PyObject* _dnf_filters,
                         std::string _expr_filter_f_str,
                         PyObject* _filter_scalars,
                         std::vector<int> _selected_fields,
                         std::vector<bool> is_nullable,
                         PyObject* _pyarrow_schema, int64_t batch_size)
        : ArrowReader(_parallel, _pyarrow_schema, tot_rows_to_read,
                      _selected_fields, is_nullable, batch_size),
          conn(_conn),
          database_schema(_database_schema),
          table_name(_table_name),
          dnf_filters(_dnf_filters),
          expr_filter_f_str(_expr_filter_f_str),
          filter_scalars(_filter_scalars) {}

    virtual ~IcebergParquetReader() {
        // When an unsupported schema evolution is detected in
        // `bodo.io.iceberg.get_iceberg_pq_dataset`, a Python exception
        // is thrown. That exception is detected and converted to a C++
        // exception in `IcebergParquetReader::get_dataset`. That exception
        // will be caught in `get_iceberg_pq_dataset` so this class to be
        // destructed, calling this function aka the destructor.

        // Py_XDECREF checks if the input is null,
        // while Py_DECREF doesn't and would just segfault
        Py_XDECREF(this->file_list);
        // The final reader may or may not be decref-ed during the read, so do
        // it here to be safe.
        Py_XDECREF(this->curr_reader);
        this->curr_reader = nullptr;
    }

    void init_iceberg_reader(std::span<int32_t> str_as_dict_cols) {
        ArrowReader::init_arrow_reader(str_as_dict_cols, false);

        if (parallel) {
            // Get the average number of pieces per rank. This is used to
            // increase the number of threads of the Arrow batch reader
            // for ranks that have to read many more files than others.
            int num_ranks;
            MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
            uint64_t num_pieces = static_cast<uint64_t>(get_num_pieces());
            MPI_Allreduce(MPI_IN_PLACE, &num_pieces, 1, MPI_UINT64_T, MPI_SUM,
                          MPI_COMM_WORLD);
            this->avg_num_pieces = num_pieces / static_cast<double>(num_ranks);
        }

        // Initialize the Arrow Dataset Scanners for reading the file segments
        // assigned to this rank. This will create as many scanners as the
        // number of unique SchemaGroups that these files belong to.
        this->init_scanners();
    }

    // Return and incref the file list.
    PyObject* get_file_list() {
        Py_INCREF(this->file_list);
        return this->file_list;
    }

    int64_t get_snapshot_id() { return this->snapshot_id; }

    size_t get_num_pieces() const override { return this->file_paths.size(); }

   protected:
    PyObject* get_dataset() override {
        // import bodo.io.iceberg
        PyObject* iceberg_mod = PyImport_ImportModule("bodo.io.iceberg");
        if (PyErr_Occurred()) {
            throw std::runtime_error("python");
        }

        PyObject* str_as_dict_cols_py =
            PyList_New(this->str_as_dict_colnames.size());
        size_t i = 0;
        for (auto field_name : this->str_as_dict_colnames) {
            // PyList_SetItem steals the reference created by
            // PyUnicode_FromString.
            PyList_SetItem(str_as_dict_cols_py, i++,
                           PyUnicode_FromString(field_name.c_str()));
        }

        // ds = bodo.io.iceberg.get_iceberg_pq_dataset(
        //          conn, database_schema, table_name,
        //          pyarrow_schema, dnf_filters,
        //          expr_filter_f_str, filter_scalars,
        //      )
        PyObject* ds = PyObject_CallMethod(
            iceberg_mod, "get_iceberg_pq_dataset", "sssOOOsO", this->conn,
            this->database_schema, this->table_name, this->pyarrow_schema,
            str_as_dict_cols_py, this->dnf_filters,
            this->expr_filter_f_str.c_str(), this->filter_scalars);
        if (ds == NULL && PyErr_Occurred()) {
            throw std::runtime_error("python");
        }

        Py_XDECREF(this->dnf_filters);
        Py_XDECREF(this->filter_scalars);
        Py_DECREF(str_as_dict_cols_py);
        Py_DECREF(iceberg_mod);

        this->filesystem = PyObject_GetAttrString(ds, "filesystem");
        // Save the file list and snapshot id for use later
        this->file_list = PyObject_GetAttrString(ds, "file_list");
        PyObject* py_snapshot_id = PyObject_GetAttrString(ds, "snapshot_id");
        // The snapshot Id is just an integer so store in native code.
        this->snapshot_id = PyLong_AsLong(py_snapshot_id);
        Py_DECREF(py_snapshot_id);
        // Returns a new reference.
        this->schema_groups_py = PyObject_GetAttrString(ds, "schema_groups");

        return ds;
    }

    /**
     * @brief Add an Iceberg Parquet piece/file for this rank
     * to read. The pieces must be added in the order of the
     * schema groups they belong to.
     *
     * @param piece Python object (IcebergPiece)
     * @param num_rows Number of rows to read from this file.
     * @param total_rows (ignored)
     */
    void add_piece(PyObject* piece, int64_t num_rows,
                   int64_t total_rows) override {
        // p = piece.path
        PyObject* p = PyObject_GetAttrString(piece, "path");
        const char* c_path = PyUnicode_AsUTF8(p);
        this->file_paths.emplace_back(c_path);
        Py_DECREF(p);

        // Number of rows to read from this file.
        this->pieces_nrows.push_back(num_rows);

        // Get the index of the schema group to use for this
        // file.
        PyObject* schema_group_idx_py =
            PyObject_GetAttrString(piece, "schema_group_idx");
        if (schema_group_idx_py == NULL) {
            throw std::runtime_error(
                "schema_group_idx_py attribute not in piece");
        }
        int64_t schema_group_idx = PyLong_AsLongLong(schema_group_idx_py);
        this->pieces_schema_group_idx.push_back(schema_group_idx);
        Py_DECREF(schema_group_idx_py);
    }

    std::shared_ptr<table_info> get_empty_out_table() override {
        if (this->empty_out_table != nullptr) {
            return this->empty_out_table;
        }

        TableBuilder builder(this->schema, this->selected_fields, 0,
                             this->is_nullable, this->str_as_dict_colnames,
                             this->create_dict_encoding_from_strings);
        std::shared_ptr<table_info> out_table =
            std::shared_ptr<table_info>(builder.get_table());

        this->empty_out_table = out_table;

        return out_table;
    }

    std::tuple<table_info*, bool, uint64_t> read_inner() override {
        // If batch_size is set, then we need to iteratively read a
        // batch at a time.
        if (this->batch_size != -1) {
            // TODO: Consolidate behavior with SnowflakeReader
            // Specifically, what does SnowflakeReader do in zero-col case?
            if (this->rows_left_to_emit <= 0) {
                return std::make_tuple(new table_info(*get_empty_out_table()),
                                       true, 0);
            }

            // Get the next batch.
            PyObject* batch_py = NULL;
            do {
                batch_py = this->get_next_py_batch();
            } while (batch_py == NULL);

            ::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> batch_res =
                arrow::py::unwrap_batch(batch_py);
            if (!batch_res.ok()) {
                throw std::runtime_error(
                    "IcebergParquetReader::read_batch(): Error unwrapping "
                    "batch: " +
                    batch_res.status().ToString());
            }
            std::shared_ptr<::arrow::RecordBatch> batch =
                batch_res.ValueOrDie();
            if (batch == nullptr) {
                throw std::runtime_error(
                    "IcebergParquetReader::read_batch(): The next batch is "
                    "null");
            }

            int64_t batch_offset =
                std::min(this->rows_to_skip, batch->num_rows());
            int64_t length =
                std::min(rows_left_to_read, batch->num_rows() - batch_offset);
            // This is zero-copy slice.
            std::shared_ptr<::arrow::Table> table = arrow::Table::Make(
                this->schema, batch->Slice(batch_offset, length)->columns());
            this->rows_left_to_read -= length;
            this->rows_to_skip -= batch_offset;

            // TODO This needs to be modified to use ChunkedTableBuilder.
            // TODO: Replace with arrow_table_to_bodo once available
            TableBuilder builder(this->schema, this->selected_fields, length,
                                 this->is_nullable, this->str_as_dict_colnames,
                                 this->create_dict_encoding_from_strings);
            builder.append(table);
            table_info* out_table = builder.get_table();

            this->rows_left_to_emit -= length;
            bool is_last = (this->rows_left_to_emit <= 0);
            Py_DECREF(batch_py);
            return std::make_tuple(out_table, is_last, length);
        }

        TableBuilder builder(this->schema, this->selected_fields, this->count,
                             this->is_nullable, this->str_as_dict_colnames,
                             this->create_dict_encoding_from_strings);

        if (get_num_pieces() == 0) {
            table_info* table = builder.get_table();
            return std::make_tuple(table, true, 0);
        }

        while (this->rows_left_to_read > 0) {
            PyObject* batch_py = NULL;
            do {
                batch_py = this->get_next_py_batch();
            } while (batch_py == NULL);
            std::shared_ptr<::arrow::RecordBatch> batch =
                arrow::py::unwrap_batch(batch_py).ValueOrDie();
            int64_t batch_offset =
                std::min(this->rows_to_skip, batch->num_rows());
            int64_t length = std::min(this->rows_left_to_read,
                                      batch->num_rows() - batch_offset);
            if (length > 0) {
                // This is zero-copy slice.
                std::shared_ptr<::arrow::Table> table = arrow::Table::Make(
                    this->schema,
                    batch->Slice(batch_offset, length)->columns());
                builder.append(table);
                this->rows_left_to_read -= length;
            }
            this->rows_to_skip -= batch_offset;
            Py_DECREF(batch_py);
        }

        table_info* table = builder.get_table();

        // rows_left_to_emit is unnecessary for non-streaming
        // so just set equal to rows_left_to_read
        this->rows_left_to_emit = this->rows_left_to_read;
        assert(this->rows_left_to_read == 0);
        return std::make_tuple(table, true, table->nrows());
    }

    /**
     * @brief Initialize the Arrow Dataset Scanners. This will create
     * one Scanner per unique Schema Group that the rank will read from.
     *
     */
    void init_scanners() {
        tracing::Event ev_scanner("init_scanners", this->parallel);
        if (get_num_pieces() == 0) {
            return;
        }

        // Construct Python lists from C++ vectors for values used in
        // get_dataset_scanners
        assert(this->file_paths.size() == this->pieces_nrows.size());
        assert(this->file_paths.size() == this->pieces_schema_group_idx.size());
        PyObject* fpaths_py = PyList_New(this->file_paths.size());
        PyObject* file_nrows_to_read_py = PyList_New(this->pieces_nrows.size());
        PyObject* file_schema_group_idxs_py =
            PyList_New(this->pieces_schema_group_idx.size());
        for (size_t i = 0; i < this->file_paths.size(); i++) {
            // PyList_SetItem steals the reference created by
            // PyUnicode_FromString and PyLong_FromLong.
            PyList_SetItem(fpaths_py, i,
                           PyUnicode_FromString(this->file_paths[i].c_str()));
            PyList_SetItem(file_nrows_to_read_py, i,
                           PyLong_FromLong(this->pieces_nrows[i]));
            PyList_SetItem(file_schema_group_idxs_py, i,
                           PyLong_FromLong(this->pieces_schema_group_idx[i]));
        }

        PyObject* str_as_dict_cols_py =
            PyList_New(this->str_as_dict_colnames.size());
        size_t i = 0;
        for (auto field_name : this->str_as_dict_colnames) {
            // PyList_SetItem steals the reference created by
            // PyUnicode_FromString.
            PyList_SetItem(str_as_dict_cols_py, i++,
                           PyUnicode_FromString(field_name.c_str()));
        }

        PyObject* selected_fields_py = PyList_New(selected_fields.size());
        i = 0;
        for (int field_num : selected_fields) {
            // PyList_SetItem steals the reference created by
            // PyLong_FromLong.
            PyList_SetItem(selected_fields_py, i++, PyLong_FromLong(field_num));
        }

        PyObject* batch_size_py =
            batch_size == -1 ? Py_None : PyLong_FromLong(batch_size);
        PyObject* iceberg_mod = PyImport_ImportModule("bodo.io.iceberg");
        // get_dataset_scanners returns a tuple with the a list of PyArrow
        // Scanners and the updated offset for the first batch.
        PyObject* scanners_updated_offset_tup = PyObject_CallMethod(
            iceberg_mod, "get_dataset_scanners", "OOOOOdiOOlOO", fpaths_py,
            file_nrows_to_read_py, file_schema_group_idxs_py,
            this->schema_groups_py, selected_fields_py, this->avg_num_pieces,
            int(this->parallel), this->filesystem, str_as_dict_cols_py,
            this->start_row_first_piece, this->pyarrow_schema, batch_size_py);
        if (scanners_updated_offset_tup == NULL && PyErr_Occurred()) {
            throw std::runtime_error("python");
        }

        // PyTuple_GetItem returns a borrowed reference, so we don't need to
        // DECREF it explicitly.
        PyObject* scanners_py = PyTuple_GetItem(scanners_updated_offset_tup, 0);
        size_t n_scanners = PyList_Size(scanners_py);
        this->scanners.reserve(n_scanners);
        for (size_t i = 0; i < n_scanners; i++) {
            // This returns a borrowed reference, so we must increment it to
            // keep the object around.
            this->scanners.push_back(PyList_GetItem(scanners_py, i));
            Py_INCREF(this->scanners.back());
        }

        this->rows_to_skip =
            PyLong_AsLong(PyTuple_GetItem(scanners_updated_offset_tup, 1));

        Py_DECREF(iceberg_mod);
        Py_DECREF(fpaths_py);
        Py_DECREF(file_nrows_to_read_py);
        Py_DECREF(file_schema_group_idxs_py);
        Py_DECREF(str_as_dict_cols_py);
        Py_DECREF(selected_fields_py);
        Py_XDECREF(batch_size_py);
        Py_DECREF(scanners_updated_offset_tup);
        Py_XDECREF(this->filesystem);
        Py_XDECREF(this->pyarrow_schema);
        Py_XDECREF(this->schema_groups_py);
    }

   private:
    // Table identifiers for the iceberg table
    // provided by the user.
    const char* conn;
    const char* database_schema;
    const char* table_name;

    // Average number of files that each rank will read.
    double avg_num_pieces = 0;

    /// Filter information to be passed to create
    /// the IcebergParquetDataset.

    // Filters to use for file pruning using Iceberg metadata.
    PyObject* dnf_filters = nullptr;
    // Information for constructing the filters dynamically
    // for each schema group based on their schema. See
    // description of bodo.io.iceberg.generate_expr_filter
    // for more details.
    std::string expr_filter_f_str;
    PyObject* filter_scalars;

    // Filesystem to use for reading the files.
    PyObject* filesystem = nullptr;

    // Memoized empty out table.
    std::shared_ptr<table_info> empty_out_table;

    // Parquet files that this process has to read
    std::vector<std::string> file_paths;
    // Number of rows to read from the files.
    std::vector<int64_t> pieces_nrows;
    // Index of the schema group corresponding to the files.
    std::vector<int64_t> pieces_schema_group_idx;
    // List of IcebergSchemaGroup objects.
    PyObject* schema_groups_py;

    // Scanners to use. There's one per IcebergSchemaGroup
    // that this rank will read from.
    std::vector<PyObject*> scanners;
    // Next scanner to use.
    size_t next_scanner_idx = 0;
    // Arrow Batched Reader to get next batch iteratively.
    PyObject* curr_reader = nullptr;
    // Number of remaining rows to skip outputting
    // from the first file assigned to this process.
    int64_t rows_to_skip = -1;

    // List of the original Iceberg file names as relative paths.
    // For example if the absolute path was
    // /Users/bodo/iceberg_db/my_table/part01.pq and the iceberg directory is
    // iceberg_db, then the path in the list would be
    // iceberg_db/my_table/part01.pq. These are used by merge/delete and are not
    // the same as the files we read, which are absolute paths.
    PyObject* file_list = nullptr;
    // Iceberg snapshot id for read.
    int64_t snapshot_id;

    /**
     * @brief Helper function to get the next available
     * RecordBatch (as a Python object that must be unwrapped by the caller).
     * Note that this must only be called when there are rows left to read.
     *
     * @return PyObject*
     */
    PyObject* get_next_py_batch() {
        // Create the next reader:
        if (this->curr_reader == nullptr) {
            if (this->next_scanner_idx < this->scanners.size()) {
                this->curr_reader = PyObject_CallMethod(
                    this->scanners[this->next_scanner_idx], "to_reader", NULL);
                if (this->curr_reader == NULL && PyErr_Occurred()) {
                    throw std::runtime_error("python");
                }
                // Decref the corresponding scanner since we don't
                // need it anymore.
                Py_XDECREF(this->scanners[this->next_scanner_idx]);
                this->scanners[this->next_scanner_idx] = nullptr;
                this->next_scanner_idx++;
            } else {
                // 'this->rows_left_to_emit'/'this->rows_left_to_read' should be
                // 0 by now!
                throw std::runtime_error(
                    "IcebergParquetReader::get_next_py_batch: Out of Arrow "
                    "Scanners/Readers! This is most likely a bug.");
            }
        }
        PyObject* batch_py =
            PyObject_CallMethod(this->curr_reader, "read_next_batch", NULL);
        if (batch_py == NULL && PyErr_Occurred() &&
            PyErr_ExceptionMatches(PyExc_StopIteration)) {
            // StopIteration is raised at the end of iteration.
            // An iterator would clear this automatically, but we are
            // not using an interator, so we have to clear it manually
            PyErr_Clear();
            // Reset the reader and decref it to free it. This will
            // prompt the next invocation to create a reader from the
            // next scanner.
            Py_XDECREF(this->curr_reader);
            this->curr_reader = nullptr;
        } else if (batch_py == NULL && PyErr_Occurred()) {
            throw std::runtime_error("python");
        }
        return batch_py;
    }
};

/**
 * Read a Iceberg table made up of parquet files.
 *
 * @param conn : Connection string for the iceberg table
 * @param database_schema : Database schema in which the iceberg table resides
 * @param table_name : Name of the iceberg table to read
 * @param parallel : true if reading in parallel
 * @param tot_rows_to_read : limit of rows to read or -1 if not limited
 * @param dnf_filters : filters passed to iceberg for filter pushdown
 * @param expr_filter_f_str : Format string to use to generate the Arrow
 *  expression filter dynamically.
 * @param filter_scalars : Python list of tuples of the form
 * (var_name: str, var_value: Any). These are the scalars to plug into
 * the Arrow expression filters.
 * @param selected_fields : Fields to select from the parquet dataset,
 * using the field ID of Arrow schema. Note that this *DOES* include
 * partition columns (Iceberg has hidden partitioning, so they *ARE* part of the
 * parquet files)
 * NOTE: selected_fields must be sorted
 * @param num_selected_fields : length of selected_fields array
 * @param is_nullable : array of booleans that indicates which of the
 * selected fields is nullable. Same length and order as selected_fields.
 * @param pyarrow_schema : PyArrow schema (instance of pyarrow.lib.Schema)
 * determined at compile time. Used for schema evolution detection, and for
 * evaluating transformations in the future.
 * @param is_merge_into : Is this table loaded as the target table for merge
 * into with COW. If True we will only apply filters that limit the number of
 * files and cannot filter rows within a file.
 * @param[out] file_list_ptr : Additional output of the Python list of read-in
 * files. This is currently only used for MERGE INTO COW
 * @param[out] snapshot_id_ptr : Additional output of current snapshot id
 * This is currently only used for MERGE INTO COW
 * @return Table containing all the read data
 */
table_info* iceberg_pq_read_py_entry(
    const char* conn, const char* database_schema, const char* table_name,
    bool parallel, int64_t tot_rows_to_read, PyObject* dnf_filters,
    const char* expr_filter_f_str_, PyObject* filter_scalars,
    int32_t* _selected_fields, int32_t num_selected_fields,
    int32_t* _is_nullable, PyObject* pyarrow_schema, int32_t* _str_as_dict_cols,
    int32_t num_str_as_dict_cols, bool is_merge_into_cow,
    int64_t* total_rows_out, PyObject** file_list_ptr,
    int64_t* snapshot_id_ptr) {
    try {
        std::vector<int> selected_fields(
            {_selected_fields, _selected_fields + num_selected_fields});
        std::vector<bool> is_nullable(_is_nullable,
                                      _is_nullable + num_selected_fields);
        std::span<int32_t> str_as_dict_cols(_str_as_dict_cols,
                                            num_str_as_dict_cols);

        std::string expr_filter_f_str(expr_filter_f_str_);
        if (is_merge_into_cow) {
            // If is_merge_into=True then we don't want to use any expr_filters
            // as we must load the whole file.
            expr_filter_f_str = "";
        }
        IcebergParquetReader reader(
            conn, database_schema, table_name, parallel, tot_rows_to_read,
            dnf_filters, expr_filter_f_str, filter_scalars, selected_fields,
            is_nullable, pyarrow_schema, -1);

        // Initialize reader
        reader.init_iceberg_reader(str_as_dict_cols);

        // MERGE INTO COW Output Handling
        if (is_merge_into_cow) {
            *file_list_ptr = reader.get_file_list();
            *snapshot_id_ptr = reader.get_snapshot_id();
        } else {
            *file_list_ptr = Py_None;
            *snapshot_id_ptr = -1;
        }

        *total_rows_out = reader.get_total_rows();
        table_info* read_output = reader.read_all();

        // Append the index column to the output table used for MERGE INTO COW
        // Since the MERGE INTO flag is internal, we assume that this column
        // is never dead for simplicity sake.
        if (is_merge_into_cow) {
            int64_t num_local_rows = reader.get_local_rows();
            std::shared_ptr<array_info> row_id_col_arr =
                alloc_numpy(num_local_rows, Bodo_CTypes::INT64);

            // Create the initial value on this rank
            // TODO: Replace with start_idx from ArrowReader
            int64_t init_val = 0;
            if (parallel) {
                MPI_Exscan(&num_local_rows, &init_val, 1, MPI_LONG_LONG_INT,
                           MPI_SUM, MPI_COMM_WORLD);
            }

            // Equivalent to np.arange(*total_rows_out, dtype=np.int64)
            std::iota((int64_t*)row_id_col_arr->data1(),
                      (int64_t*)row_id_col_arr->data1() + num_local_rows,
                      init_val);

            read_output->columns.push_back(row_id_col_arr);
        }

        return read_output;

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
 * Construct an Iceberg Parquet-based ArrowReader
 *
 * @param conn : Connection string for the iceberg table
 * @param database_schema : Database schema in which the iceberg table resides
 * @param table_name : Name of the iceberg table to read
 * @param parallel : true if reading in parallel
 * @param tot_rows_to_read : limit of rows to read or -1 if not limited
 * @param dnf_filters : filters passed to iceberg for filter pushdown
 * @param expr_filter_f_str : Format string to use to generate the Arrow
 *  expression filter dynamically.
 * @param filter_scalars : Python list of tuples of the form
 * (var_name: str, var_value: Any). These are the scalars to plug into
 * the Arrow expression filters.
 * @param _selected_fields : Fields to select from the parquet dataset,
 * using the field ID of Arrow schema. Note that this *DOES* include
 * partition columns (Iceberg has hidden partitioning, so they *ARE* part of the
 * parquet files)
 * NOTE: selected_fields must be sorted
 * @param num_selected_fields : length of selected_fields array
 * @param is_nullable : array of booleans that indicates which of the
 * selected fields is nullable. Same length and order as selected_fields.
 * @param pyarrow_schema : PyArrow schema (instance of pyarrow.lib.Schema)
 * determined at compile time. Used for schema evolution detection, and for
 * evaluating transformations in the future.
 * @param batch_size Reading batch size
 * @return ArrowReader to read the table's data
 */
ArrowReader* iceberg_pq_reader_init_py_entry(
    const char* conn, const char* database_schema, const char* table_name,
    bool parallel, int64_t tot_rows_to_read, PyObject* dnf_filters,
    const char* expr_filter_f_str, PyObject* filter_scalars,
    int32_t* _selected_fields, int32_t num_selected_fields,
    int32_t* _is_nullable, PyObject* pyarrow_schema, int32_t* _str_as_dict_cols,
    int32_t num_str_as_dict_cols, int64_t batch_size) {
    try {
        std::vector<int> selected_fields(
            {_selected_fields, _selected_fields + num_selected_fields});
        std::vector<bool> is_nullable(_is_nullable,
                                      _is_nullable + num_selected_fields);
        std::span<int32_t> str_as_dict_cols(_str_as_dict_cols,
                                            num_str_as_dict_cols);

        IcebergParquetReader* reader = new IcebergParquetReader(
            conn, database_schema, table_name, parallel, tot_rows_to_read,
            dnf_filters, std::string(expr_filter_f_str), filter_scalars,
            selected_fields, is_nullable, pyarrow_schema, batch_size);

        // Initialize reader
        reader->init_iceberg_reader(str_as_dict_cols);
        return reinterpret_cast<ArrowReader*>(reader);

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
