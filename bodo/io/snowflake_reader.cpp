// Copyright (C) 2022 Bodo Inc. All rights reserved.

// Implementation of SnowflakeReader (subclass of ArrowDataFrameReader) with
// functionality that is specific to reading from Snowflake

#include <chrono>
#include "arrow_reader.h"

// -------- SnowflakeReader --------

class SnowflakeReader : public ArrowDataframeReader {
   public:
    /**
     * Initialize SnowflakeReader.
     * See snowflake_read_py_entry function below for description of arguments.
     */
    SnowflakeReader(const char* _query, const char* _conn, bool _parallel,
                    bool _is_independent, PyObject* pyarrow_schema,
                    std::set<int> selected_fields,
                    std::vector<bool> is_nullable,
                    std::vector<int> str_as_dict_cols,
                    std::vector<int> allow_unsafe_dt_to_ts_cast_cols,
                    int64_t* _total_nrows, bool _only_length_query,
                    bool _is_select_query)
        : ArrowDataframeReader(_parallel, pyarrow_schema, -1, selected_fields,
                               is_nullable),
          query(_query),
          conn(_conn),
          total_nrows(_total_nrows),
          only_length_query(_only_length_query),
          is_select_query(_is_select_query),
          is_independent(_is_independent) {
        // Initialize reader
        init_arrow_reader(str_as_dict_cols, true,
                          allow_unsafe_dt_to_ts_cast_cols);
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
        if (PyErr_Occurred())
            throw std::runtime_error("python");

        // ds = bodo.io.snowflake.get_dataset(query, conn, pyarrow_schema,
        //   only_length, is_select_query, is_parallel, is_independent)
        PyObject* py_only_length_query = PyBool_FromLong(only_length_query);
        PyObject* py_is_select_query = PyBool_FromLong(is_select_query);
        PyObject* py_is_parallel = PyBool_FromLong(parallel);
        PyObject* py_is_independent = PyBool_FromLong(this->is_independent);
        PyObject* ds_tuple = PyObject_CallMethod(
            sf_mod, "get_dataset", "ssOOOOO", query, conn, pyarrow_schema,
            py_only_length_query, py_is_select_query, py_is_parallel,
            py_is_independent);
        if (ds_tuple == NULL && PyErr_Occurred()) {
            throw std::runtime_error("python");
        }
        Py_DECREF(sf_mod);
        Py_DECREF(pyarrow_schema);
        Py_DECREF(py_only_length_query);
        Py_DECREF(py_is_select_query);
        Py_DECREF(py_is_parallel);
        Py_DECREF(py_is_independent);

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

    virtual void read_all(TableBuilder& builder) {
        tracing::Event ev("reader::read_all", parallel);
        size_t num_pieces = get_num_pieces();
        int64_t to_arrow_time = 0;
        int64_t append_time = 0;
        int64_t cast_arrow_table_time = 0;
        ev.add_attribute("num_pieces", num_pieces);

        for (size_t piece_idx = 0; piece_idx < num_pieces; piece_idx++) {
            int64_t offset = 0;
            if (piece_idx == 0) {
                offset = start_row_first_piece;
            }
            auto t1 = std::chrono::steady_clock::now();
            PyObject* batch = batches[piece_idx];
            PyObject* arrow_table_py =
                PyObject_CallMethod(batch, "to_arrow", "O", sf_conn);
            if (arrow_table_py == NULL && PyErr_Occurred()) {
                throw std::runtime_error("python");
            }

            auto table = arrow::py::unwrap_table(arrow_table_py).ValueOrDie();
            auto t2 = std::chrono::steady_clock::now();
            to_arrow_time +=
                std::chrono::duration_cast<std::chrono::microseconds>((t2 - t1))
                    .count();

            t1 = std::chrono::steady_clock::now();
            // Upcast Arrow table to expected schema
            // TODO: Integrate within TableBuilder
            table = ArrowDataframeReader::cast_arrow_table(table);
            t2 = std::chrono::steady_clock::now();
            cast_arrow_table_time +=
                std::chrono::duration_cast<std::chrono::microseconds>((t2 - t1))
                    .count();

            int64_t length = std::min(rows_left, table->num_rows() - offset);
            // pass zero-copy slice to TableBuilder to append to read data
            builder.append(table->Slice(offset, length));
            rows_left -= length;
            // releasing reference to batch since it's not needed anymore
            Py_DECREF(batch);
            Py_DECREF(arrow_table_py);
            auto t3 = std::chrono::steady_clock::now();
            append_time +=
                std::chrono::duration_cast<std::chrono::microseconds>((t3 - t2))
                    .count();
        }
        ev.add_attribute("total_to_arrow_time_micro", to_arrow_time);
        ev.add_attribute("total_append_time_micro", append_time);
        ev.add_attribute("total_cast_arrow_table_time_micro",
                         cast_arrow_table_time);
        ev.finalize();
    }

   private:
    const char* query;       // query passed to pd.read_sql()
    const char* conn;        // connection string passed to pd.read_sql()
    int64_t* total_nrows;    // Pointer to store total number of rows read.
                             // This is used when reading 0 columns.
    bool only_length_query;  // Is the query optimized to only compute the
                             // length.
    bool is_select_query;    // Is this query a select statement?

    // batches that this process is going to read
    // A batch is a snowflake.connector.result_batch.ArrowResultBatch
    std::vector<PyObject*> batches;
    // instance of snowflake.connector.connection.SnowflakeConnection, used to
    // read the batches
    PyObject* sf_conn = nullptr;
    bool is_independent =
        false;  // Are the ranks executing the function independently?
};

/**
 * Read data from snowflake given a query.
 *
 * @param query : SQL query
 * @param conn : connection string URL
 * @param parallel: true if reading in parallel
 * @param is_independent: true if all the ranks are executing the function
 independently
 * @param n_fields : Number of fields (columns) in Arrow data to retrieve
 * @param is_nullable : array of bools that indicates which of the fields is
 * nullable
 * @param[out] total_nrows: Pointer used to store to total number of rows to
 read. This is used when we are loading 0 columns.
 * @param _only_length_query: Boolean value for if the query was optimized to
 only compute the length.
 * @param _is_select_query: Boolean value for if the query is a select
 statement.
 * @param _allow_unsafe_dt_to_ts_cast_cols Indices of columns where we will be
 performing a cast from date to datetime64[ns] after reading, i.e. Snowflake is
 expected to send us "date" data, and we will cast it to datetime64[ns]. In
 particular, we will allow an "unsafe" cast (only overflow allowed) in these
 cases to match the behavior of Bodo's `astype` cast.
 * @param num_allow_unsafe_dt_to_ts_cast_cols Number of columns where we will be
 performing this unsafe cast from date to datetime64[ns].
 * @return table containing all the arrays read
 */
table_info* snowflake_read_py_entry(
    const char* query, const char* conn, bool parallel, bool is_independent,
    PyObject* arrow_schema, int64_t n_fields, int32_t* _is_nullable,
    int32_t* _str_as_dict_cols, int32_t num_str_as_dict_cols,
    int32_t* _allow_unsafe_dt_to_ts_cast_cols,
    int32_t num_allow_unsafe_dt_to_ts_cast_cols, int64_t* total_nrows,
    bool _only_length_query, bool _is_select_query) {
    try {
        std::set<int> selected_fields;
        for (auto i = 0; i < n_fields; i++)
            selected_fields.insert(i);
        std::vector<bool> is_nullable(_is_nullable, _is_nullable + n_fields);
        std::vector<int> str_as_dict_cols(
            {_str_as_dict_cols, _str_as_dict_cols + num_str_as_dict_cols});
        std::vector<int> allow_unsafe_dt_to_ts_cast_cols(
            {_allow_unsafe_dt_to_ts_cast_cols,
             _allow_unsafe_dt_to_ts_cast_cols +
                 num_allow_unsafe_dt_to_ts_cast_cols});

        SnowflakeReader reader(query, conn, parallel, is_independent,
                               arrow_schema, selected_fields, is_nullable,
                               str_as_dict_cols,
                               allow_unsafe_dt_to_ts_cast_cols, total_nrows,
                               _only_length_query, _is_select_query);

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
