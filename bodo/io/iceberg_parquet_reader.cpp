// Copyright (C) 2022 Bodo Inc. All rights reserved.

// Implementation of IcebergParquetReader (subclass of ParquetReader) with
// functionality that is specific to reading iceberg datasets (made up of
// parquet files)

#include "parquet_reader.h"

// -------- IcebergParquetReader --------

class IcebergParquetReader : public ParquetReader {
   public:
    /**
     * Initialize IcebergParquetReader.
     * See iceberg_pq_read_py_entry function below for description of arguments.
     */
    IcebergParquetReader(const char* _conn, const char* _database_schema,
                         const char* _table_name, bool _parallel,
                         int64_t tot_rows_to_read, PyObject* _dnf_filters,
                         PyObject* _expr_filters,
                         std::set<int> _selected_fields,
                         std::vector<bool> is_nullable,
                         PyObject* pyarrow_schema, int64_t batch_size)
        : ParquetReader(/*path*/ nullptr, _parallel, _dnf_filters,
                        _expr_filters, /*storage_options*/ PyDict_New(),
                        pyarrow_schema, tot_rows_to_read, _selected_fields,
                        is_nullable, /*input_file_name_col*/ false, batch_size),
          conn(_conn),
          database_schema(_database_schema),
          table_name(_table_name) {}

    virtual ~IcebergParquetReader() {
        // When schema evolution is detected in
        // `bodo.io.iceberg.get_iceberg_pq_dataset`, a Python exception
        // is thrown. That exception is detected and converted to a C++
        // exception in `IcebergParquetReader::get_dataset`. That exception
        // will be caught in `get_iceberg_pq_dataset` so this class to be
        // destructed, calling this function aka the destructor.

        // this->file_list will be null if schema evolution is detected
        // Py_XDECREF checks if the input is null,
        // while Py_DECREF doesn't and would just segfault
        Py_XDECREF(this->file_list);
    }

    void init_iceberg_reader(std::span<int32_t> str_as_dict_cols) {
        ParquetReader::init_pq_reader(str_as_dict_cols, nullptr, nullptr, 0);
    }

    // Return and incref the file list.
    PyObject* get_file_list() {
        Py_INCREF(this->file_list);
        return this->file_list;
    }

    int64_t get_snapshot_id() { return this->snapshot_id; }

   protected:
    // We don't need a special implementation yet and can re-use
    // ParquetReader's implementation.
    // Eventually we'll want to use this for schema evolution, etc.
    // to handle file-level transformations, etc.
    // virtual void add_piece(PyObject* piece, int64_t num_rows,
    //                        int64_t total_rows) {}

    virtual PyObject* get_dataset() {
        // import bodo.io.iceberg
        PyObject* iceberg_mod = PyImport_ImportModule("bodo.io.iceberg");
        if (PyErr_Occurred())
            throw std::runtime_error("python");

        // ds = bodo.io.iceberg.get_iceberg_pq_dataset(
        //          conn, database_schema, table_name,
        //          pyarrow_schema, dnf_filters,
        //          expr_filters, tot_rows_to_read, parallel
        //      )
        PyObject* ds = PyObject_CallMethod(
            iceberg_mod, "get_iceberg_pq_dataset", "sssOOOLO", this->conn,
            this->database_schema, this->table_name, this->pyarrow_schema,
            this->dnf_filters, this->expr_filters, this->tot_rows_to_read,
            PyBool_FromLong(parallel));
        if (ds == NULL && PyErr_Occurred()) {
            throw std::runtime_error("python");
        }

        // XXX Handle ds_has_partitions?
        this->ds_partitioning = Py_None;

        // prefix = ds._prefix
        PyObject* prefix_py = PyObject_GetAttrString(ds, "_prefix");
        this->prefix.assign(PyUnicode_AsUTF8(prefix_py));
        Py_DECREF(prefix_py);
        Py_DECREF(this->dnf_filters);
        Py_DECREF(iceberg_mod);

        this->filesystem = PyObject_GetAttrString(ds, "filesystem");
        // Save the file list and snapshot id for use later
        this->file_list = PyObject_GetAttrString(ds, "file_list");
        PyObject* py_snapshot_id = PyObject_GetAttrString(ds, "snapshot_id");
        // The snapshot Id is just an integer so store in native code.
        this->snapshot_id = PyLong_AsLong(py_snapshot_id);
        Py_DECREF(py_snapshot_id);

        return ds;
    }

    // We don't need a special implementation yet and can re-use
    // ParquetReader's implementation.
    // Eventually we'll want to use this for schema evolution, etc.
    // to handle file-level transformations, etc.
    // virtual void read_all(TableBuilder& builder) {}

   private:
    // Table identifiers for the iceberg table
    // provided by the user.
    const char* conn;
    const char* database_schema;
    const char* table_name;

    // List of the original Iceberg file names as relative paths.
    // For example if the absolute path was
    // /Users/bodo/iceberg_db/my_table/part01.pq and the iceberg directory is
    // iceberg_db, then the path in the list would be
    // iceberg_db/my_table/part01.pq. These are used by merge/delete and are not
    // the same as the files we read, which are absolute paths.
    PyObject* file_list = nullptr;
    // Iceberg snapshot id for read.
    int64_t snapshot_id;
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
 * @param expr_filters : PyObject passed to pyarrow.parquet.ParquetDataset
 * filters argument to remove rows from scanned data
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
    PyObject* expr_filters, int32_t* _selected_fields,
    int32_t num_selected_fields, int32_t* _is_nullable,
    PyObject* pyarrow_schema, int32_t* _str_as_dict_cols,
    int32_t num_str_as_dict_cols, bool is_merge_into_cow,
    int64_t* total_rows_out, PyObject** file_list_ptr,
    int64_t* snapshot_id_ptr) {
    try {
        if (is_merge_into_cow) {
            // If is_merge_into=True then we don't want to use any expr_filters
            // as we must load the whole file.
            Py_DECREF(expr_filters);
            expr_filters = Py_None;
        }

        std::set<int> selected_fields(
            {_selected_fields, _selected_fields + num_selected_fields});
        std::vector<bool> is_nullable(_is_nullable,
                                      _is_nullable + num_selected_fields);
        std::span<int32_t> str_as_dict_cols(_str_as_dict_cols,
                                            num_str_as_dict_cols);

        IcebergParquetReader reader(conn, database_schema, table_name, parallel,
                                    tot_rows_to_read, dnf_filters, expr_filters,
                                    selected_fields, is_nullable,
                                    pyarrow_schema, -1);

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
        if (std::string(e.what()) != "python")
            PyErr_SetString(PyExc_RuntimeError, e.what());
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
 * @param expr_filters : PyObject passed to pyarrow.parquet.ParquetDataset
 * filters argument to remove rows from scanned data
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
    PyObject* expr_filters, int32_t* _selected_fields,
    int32_t num_selected_fields, int32_t* _is_nullable,
    PyObject* pyarrow_schema, int32_t* _str_as_dict_cols,
    int32_t num_str_as_dict_cols, int64_t batch_size) {
    try {
        std::set<int> selected_fields(
            {_selected_fields, _selected_fields + num_selected_fields});
        std::vector<bool> is_nullable(_is_nullable,
                                      _is_nullable + num_selected_fields);
        std::span<int32_t> str_as_dict_cols(_str_as_dict_cols,
                                            num_str_as_dict_cols);

        auto reader = new IcebergParquetReader(
            conn, database_schema, table_name, parallel, tot_rows_to_read,
            dnf_filters, expr_filters, selected_fields, is_nullable,
            pyarrow_schema, batch_size);

        // Initialize reader
        reader->init_iceberg_reader(str_as_dict_cols);
        return reinterpret_cast<ArrowReader*>(reader);

    } catch (const std::exception& e) {
        // if the error string is "python" this means the C++ exception is
        // a result of a Python exception, so we don't call PyErr_SetString
        // because we don't want to replace the original Python error
        if (std::string(e.what()) != "python")
            PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}
