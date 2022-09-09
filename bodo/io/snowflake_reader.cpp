// Copyright (C) 2021 Bodo Inc. All rights reserved.

// Implementation of SnowflakeReader (subclass of ArrowDataframeReader) with
// functionality that is specific to reading from Snowflake

#include "arrow_reader.h"

// -------- SnowflakeReader --------

class SnowflakeReader : public ArrowDataframeReader {
   public:
    /**
     * Initialize SnowflakeReader.
     * See snowflake_read function below for description of arguments.
     */
    SnowflakeReader(const char* _query, const char* _conn, bool _parallel,
                    int* selected_fields, int64_t num_selected_fields,
                    int32_t* is_nullable, int32_t* _str_as_bool_cols,
                    int32_t num_str_as_dict_cols, int64_t* _total_nrows)
        : ArrowDataframeReader(_parallel, -1, selected_fields,
                               num_selected_fields, is_nullable),
          query(_query),
          conn(_conn),
          total_nrows(_total_nrows) {
        // initialize reader
        init_arrow_reader(
            {_str_as_bool_cols, _str_as_bool_cols + num_str_as_dict_cols},
            true);
    }

    virtual ~SnowflakeReader() { Py_XDECREF(sf_conn); }

    /// A piece is a snowflake.connector.result_batch.ArrowResultBatch
    virtual size_t get_num_pieces() const { return batches.size(); }

   protected:
    virtual void add_piece(PyObject* piece, int64_t num_rows,
                           int64_t total_rows) {
        Py_INCREF(piece);  // keeping a reference to this piece
        batches.push_back(piece);
    }

    virtual PyObject* get_dataset() {
        // import bodo.io.snowflake
        PyObject* sf_mod = PyImport_ImportModule("bodo.io.snowflake");
        if (PyErr_Occurred()) throw std::runtime_error("python");

        // ds = bodo.io.snowflake.get_dataset(query, conn, onlyLength)
        PyObject* onlyLength = PyBool_FromLong(selected_fields.size() == 0);
        PyObject* ds_tuple = PyObject_CallMethod(sf_mod, "get_dataset", "ssO",
                                                 query, conn, onlyLength);
        if (ds_tuple == NULL && PyErr_Occurred()) {
            throw std::runtime_error("python");
        }
        Py_DECREF(sf_mod);
        // PyTuple_GetItem borrows a reference
        PyObject* ds = PyTuple_GetItem(ds_tuple, 0);
        Py_INCREF(ds);  // call incref to keep the reference
        // PyTuple_GetItem borrows a reference
        PyObject* total_len = PyTuple_GetItem(ds_tuple, 1);
        *total_nrows = PyLong_AsLong(total_len);
        Py_DECREF(ds_tuple);
        sf_conn = PyObject_GetAttrString(ds, "conn");
        if (sf_conn == NULL) {
            throw std::runtime_error(
                "Could not retrieve conn attribute of snowflake dataset");
        }
        return ds;
    }

    virtual std::shared_ptr<arrow::Schema> get_schema(PyObject* dataset) {
        PyObject* schema_py = PyObject_GetAttrString(dataset, "schema");
        // see
        // https://arrow.apache.org/docs/python/extending.html#using-pyarrow-from-c-and-cython-code
        auto schema_ = arrow::py::unwrap_schema(schema_py).ValueOrDie();
        Py_DECREF(schema_py);
        return schema_;
    }

    virtual void read_all(TableBuilder& builder) {
        for (size_t piece_idx = 0; piece_idx < get_num_pieces(); piece_idx++) {
            int64_t offset = 0;
            if (piece_idx == 0) offset = start_row_first_piece;
            PyObject* batch = batches[piece_idx];
            PyObject* arrow_table_py =
                PyObject_CallMethod(batch, "to_arrow", "O", sf_conn);
            auto table = arrow::py::unwrap_table(arrow_table_py).ValueOrDie();
            int64_t length = std::min(rows_left, table->num_rows() - offset);
            // pass zero-copy slice to TableBuilder to append to read data
            builder.append(table->Slice(offset, length));
            rows_left -= length;
            // releasing reference to batch since it's not needed anymore
            Py_DECREF(batch);
            Py_DECREF(arrow_table_py);
        }
    }

   private:
    const char* query;     // query passed to pd.read_sql()
    const char* conn;      // connection string passed to pd.read_sql()
    int64_t* total_nrows;  // Pointer to store total number of rows read.
                           // This is used when reading 0 columns.

    // batches that this process is going to read
    // A batch is a snowflake.connector.result_batch.ArrowResultBatch
    std::vector<PyObject*> batches;
    // instance of snowflake.connector.connection.SnowflakeConnection, used to
    // read the batches
    PyObject* sf_conn = nullptr;
};

/**
 * Read data from snowflake given a query.
 *
 * @param query : SQL query
 * @param conn : connection string URL
 * @param parallel: true if reading in parallel
 * @param n_fields : Number of fields (columns) in Arrow data to retrieve
 * @param is_nullable : array of bools that indicates which of the fields is
 * nullable
 * @param[out] total_nrows: Pointer used to store to total number of rows to read.
        This is used when we are loading 0 columns.
 * @return table containing all the arrays read
 */
table_info* snowflake_read(const char* query, const char* conn, bool parallel,
                           int64_t n_fields, int32_t* is_nullable,
                           int32_t* str_as_dict_cols,
                           int32_t num_str_as_dict_cols, int64_t* total_nrows) {
    try {
        std::vector<int> selected_fields(n_fields);
        for (auto i = 0; i < n_fields; i++)
            selected_fields[i] = static_cast<int>(i);
        SnowflakeReader reader(query, conn, parallel, selected_fields.data(),
                               n_fields, is_nullable, str_as_dict_cols,
                               num_str_as_dict_cols, total_nrows);
        return reader.read();
    } catch (const std::exception& e) {
        // if the error string is "python" this means the C++ exception is
        // a result of a Python exception, so we don't call PyErr_SetString
        // because we don't want to replace the original Python error
        if (std::string(e.what()) != "python")
            PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}
