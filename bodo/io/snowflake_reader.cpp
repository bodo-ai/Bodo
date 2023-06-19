// Copyright (C) 2022 Bodo Inc. All rights reserved.

// Implementation of SnowflakeReader (subclass of ArrowReader) with
// functionality that is specific to reading from Snowflake

#include <chrono>
#include <queue>
#include "arrow_reader.h"

#include "../libs/_distributed.h"

/**
 * @brief Class that contains the information to convert
 * Arrow string columns loaded from Snowflake into
 * Arrow Dictionary columns. This shouldn't incur additional
 * overhead because we have a zero-copy approach to convert
 * to Bodo arrays.
 *
 */
class SnowflakeDictionaryBuilder {
   public:
    SnowflakeDictionaryBuilder(std::shared_ptr<arrow::Schema> _schema,
                               const std::set<int>& _selected_fields,
                               const std::set<std::string>& _str_as_dict_cols)
        : schema(_schema),
          selected_fields(_selected_fields),
          str_as_dict_cols(_str_as_dict_cols) {}

    std::shared_ptr<arrow::Table> convert(
        std::shared_ptr<arrow::Table> in_table) {
        arrow::ChunkedArrayVector new_cols;
        for (int i = 0; i < in_table->num_columns(); i++) {
            auto col = in_table->column(i);
            if (selected_fields.contains(i) &&
                str_as_dict_cols.contains(schema->field(i)->name())) {
                // We have a dictionary column to convert.
                auto new_col = arrow::compute::DictionaryEncode(col);
                new_cols.push_back(new_col->chunked_array());
            } else {
                // For all other columns just copy the column to the table.
                new_cols.push_back(col);
            }
        }

        return arrow::Table::Make(this->schema, new_cols, in_table->num_rows());
    }

   private:
    std::shared_ptr<arrow::Schema> schema;
    const std::set<int>& selected_fields;
    const std::set<std::string>& str_as_dict_cols;
};

// -------- SnowflakeReader --------

class SnowflakeReader : public ArrowReader {
   public:
    /**
     * Initialize SnowflakeReader.
     * See snowflake_read_py_entry function below for description of arguments.
     */
    SnowflakeReader(const char* _query, const char* _conn, bool _parallel,
                    bool _is_independent, PyObject* pyarrow_schema,
                    std::set<int> selected_fields,
                    std::vector<bool> is_nullable,
                    std::vector<int> str_as_dict_cols, bool _only_length_query,
                    bool _is_select_query, int64_t batch_size = -1)
        : ArrowReader(_parallel, pyarrow_schema, -1, selected_fields,
                      is_nullable, batch_size),
          query(_query),
          conn(_conn),
          only_length_query(_only_length_query),
          is_select_query(_is_select_query),
          is_independent(_is_independent) {
        // Initialize reader
        init_arrow_reader(str_as_dict_cols, true);
    }

    virtual ~SnowflakeReader() { Py_XDECREF(sf_conn); }

    /// A piece is a snowflake.connector.result_batch.ArrowResultBatch
    virtual size_t get_num_pieces() const {
        if (!this->initialized) {
            return this->result_batches.size();
        } else {
            // While not necessary, ensures that we always return
            // the total number of local pieces during Snowflake reads
            // because it pops from the ResultBatch queue
            return this->num_pieces;
        }
    }

    int64_t get_total_source_rows() const { return this->total_nrows; }

   protected:
    virtual void add_piece(PyObject* piece, int64_t num_rows,
                           int64_t total_rows) {
        Py_INCREF(piece);  // keeping a reference to this piece
        result_batches.push(piece);
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
        this->total_nrows = PyLong_AsLong(total_len);
        Py_DECREF(ds_tuple);
        sf_conn = PyObject_GetAttrString(ds, "conn");
        if (sf_conn == NULL) {
            throw std::runtime_error(
                "Could not retrieve conn attribute of snowflake dataset");
        }

        // all_pieces = ds.pieces
        PyObject* all_pieces = PyObject_GetAttrString(ds, "pieces");
        this->num_pieces = static_cast<uint64_t>(PyObject_Length(all_pieces));
        return ds;
    }

    /**
     * @brief Convert a ResultBatch piece object the current rank should read
     * into an Arrow Table with Bodo's expected output schema
     *
     * @param next_piece A Python object pointing to the data of the next piece
     * @return std::tuple<std::shared_ptr<arrow::Table>, int64_t, int64_t>
     *  - The data of the piece as an Arrow table
     *      - Casted to the correct expected schema
     *      - Sliced for the section that this rank should read
     * - Time it took (us) to download the piece from remote storage
     * - Time it took (us) to process the piece (cast, slice)
     */
    std::tuple<std::shared_ptr<arrow::Table>, int64_t, int64_t>
    process_result_batch(PyObject* next_piece, int64_t offset) {
        using namespace std::chrono;

        int64_t to_arrow_time = 0;
        int64_t cast_arrow_table_time = 0;

        auto start = steady_clock::now();
        PyObject* arrow_table_py =
            PyObject_CallMethod(next_piece, "to_arrow", "O", sf_conn);
        if (arrow_table_py == NULL && PyErr_Occurred()) {
            throw std::runtime_error("python");
        }
        auto table = arrow::py::unwrap_table(arrow_table_py).ValueOrDie();
        auto end = steady_clock::now();
        to_arrow_time = duration_cast<microseconds>((end - start)).count();

        start = steady_clock::now();
        int64_t length = std::min(rows_left, table->num_rows() - offset);
        // TODO: Add check if length < 0, never possible
        table = table->Slice(offset, length);

        // Upcast Arrow table to expected schema
        // TODO: Integrate within TableBuilder
        table = ArrowReader::cast_arrow_table(table);
        end = steady_clock::now();
        cast_arrow_table_time =
            duration_cast<microseconds>((end - start)).count();

        Py_DECREF(next_piece);
        Py_DECREF(arrow_table_py);
        return std::make_tuple(table, to_arrow_time, cast_arrow_table_time);
    }

    // Design for Batched Snowflake Reads:
    // https://bodo.atlassian.net/l/cp/4JwaiChQ
    virtual std::tuple<table_info*, bool, uint64_t> read_inner() {
        using namespace std::chrono;

        tracing::Event ev("reader::read_inner", parallel);
        int64_t to_arrow_time = 0;
        int64_t cast_arrow_table_time = 0;
        int64_t append_time = 0;
        // Total number of rows to output from func call
        uint64_t total_read_rows = 0;

        // In the case when we only need to perform a count(*)
        // We need to still return a table with 0 columns but
        // `this->total_nrows` rows We can return a nullptr for the table_info*,
        // but still need to return `this->total_nrows / num_ranks` for the
        // length on each rank. In the batching case, this only occurs on the
        // first iteration
        if (this->only_length_query) {
            if (!result_batches.empty() || !out_batches.empty()) {
                throw std::runtime_error(
                    "SnowflakeReader: Read a batch in the "
                    "only_length_query case!");
            }

            assert(this->rows_left == 0);

            size_t rank = dist_get_rank();
            size_t nranks = dist_get_size();
            auto out_rows =
                returned_once_empty_case
                    ? 0
                    : dist_get_node_portion(this->total_nrows, nranks, rank);
            returned_once_empty_case = true;
            return std::make_tuple(nullptr, true, out_rows);
        }

        // Attempting to read the entire table at once (read_all() path)
        // Right now, the batched reader does not attempt to combine small
        // pieces together to get the expected batch size. Thus, we need this
        // special case
        // TODO: Handle when len(piece) < batch_size and remove this special
        // case
        if (batch_size == -1) {
            TableBuilder builder(schema, selected_fields, count, is_nullable,
                                 str_as_dict_colnames,
                                 create_dict_encoding_from_strings);

            while (!result_batches.empty()) {
                auto next_piece = result_batches.front();
                result_batches.pop();

                int64_t offset = 0;
                if (is_first_piece) {
                    offset = start_row_first_piece;
                    is_first_piece = false;
                }

                auto [table, to_arrow_time_, cast_arrow_table_time_] =
                    process_result_batch(next_piece, offset);
                to_arrow_time += to_arrow_time_;
                cast_arrow_table_time += cast_arrow_table_time_;

                auto start = steady_clock::now();
                this->rows_left -= table->num_rows();
                builder.append(table);
                auto end = steady_clock::now();
                append_time +=
                    duration_cast<microseconds>((end - start)).count();
            }

            ev.add_attribute("total_to_arrow_time_micro", to_arrow_time);
            ev.add_attribute("total_append_time_micro", append_time);
            ev.add_attribute("total_cast_arrow_table_time_micro",
                             cast_arrow_table_time);

            auto out_table = builder.get_table();
            ev.add_attribute("out_batch_size", out_table->nrows());
            ev.finalize();
            return std::make_tuple(out_table, true, out_table->nrows());
        }

        // If next batch not available, prepare next batches
        if (out_batches.empty() && !result_batches.empty()) {
            auto next_piece = result_batches.front();
            result_batches.pop();

            int64_t offset = 0;
            if (is_first_piece) {
                offset = start_row_first_piece;
                is_first_piece = false;
            }

            auto [table, to_arrow_time, cast_arrow_table_time] =
                process_result_batch(next_piece, offset);

            auto start = steady_clock::now();

            // Build the dictionary
            SnowflakeDictionaryBuilder dict_builder(schema, selected_fields,
                                                    str_as_dict_colnames);
            table = dict_builder.convert(table);
            // Arrow utility to iterate over row-chunks of an input table
            // Useful for us to construct batches of Bodo tables from the piece
            auto reader = arrow::TableBatchReader(table);
            reader.set_chunksize(batch_size);

            std::shared_ptr<arrow::RecordBatch> next_recordbatch;
            for (auto status = reader.ReadNext(&next_recordbatch);
                 status.ok() && next_recordbatch;
                 status = reader.ReadNext(&next_recordbatch)) {
                // Construct Builder Object for Next Batch
                TableBuilder builder(schema, selected_fields,
                                     next_recordbatch->num_rows(), is_nullable,
                                     str_as_dict_colnames, false);
                // TODO: Have TableBuilder support RecordBatches
                auto res = arrow::Table::FromRecordBatches({next_recordbatch})
                               .ValueOrDie();
                builder.append(res);
                out_batches.push(builder.get_table());
            }

            auto end = steady_clock::now();
            append_time = duration_cast<microseconds>((end - start)).count();
        }

        ev.add_attribute("total_to_arrow_time_micro", to_arrow_time);
        ev.add_attribute("total_append_time_micro", append_time);
        ev.add_attribute("total_cast_arrow_table_time_micro",
                         cast_arrow_table_time);

        table_info* next_batch = nullptr;
        if (out_batches.size() > 0) {
            next_batch = out_batches.front();
            out_batches.pop();
            total_read_rows = next_batch->nrows();
        }

        this->rows_left -= total_read_rows;

        ev.add_attribute("out_batch_size",
                         next_batch == nullptr ? 0 : next_batch->nrows());
        ev.finalize();

        bool is_last = result_batches.empty() && out_batches.empty();
        return std::make_tuple(next_batch, is_last, total_read_rows);
    }

   private:
    const char* query;    // query passed to pd.read_sql()
    const char* conn;     // connection string passed to pd.read_sql()
    int64_t total_nrows;  // Pointer to store total number of rows read.
                          // This is used when reading 0 columns.
    // In 0-col case, need this flag to track if we have
    // returned to correct row count out (occurs in first iteration)
    bool returned_once_empty_case = false;

    bool only_length_query;  // Is the query optimized to only compute the
                             // length.
    bool is_select_query;    // Is this query a select statement?

    // instance of snowflake.connector.connection.SnowflakeConnection, used to
    // read the batches
    PyObject* sf_conn = nullptr;
    // Are the ranks executing the function independently?
    bool is_independent = false;

    // Number of total pieces / ResultBatches the SnowflakeReader
    // has for all ranks
    uint64_t num_pieces;
    // Batches that this process is going to read
    // A batch is a snowflake.connector.result_batch.ArrowResultBatch
    std::queue<PyObject*> result_batches;

    // Prepared output batches (Bodo tables) ready to emit
    std::queue<table_info*> out_batches;
    bool is_first_piece = true;
};

/**
 * @brief Construct a SnowflakeReader pointer to pass to Python
 * for streaming purposes
 *
 * TODO: At some point, we should combine with the next py_entry
 * function to avoid duplication
 *
 * @param query: SQL query
 * @param conn: connection string URL
 * @param parallel: true if reading in parallel
 * @param is_independent: true if all the ranks are executing the function
 *        independently
 * @param arrow_schema: Expected schema of output Arrow tables from the
 *        Snowflake connector
 * @param n_fields : Number of fields (columns) in Arrow data to retrieve
 * @param is_nullable : array of bools that indicates which of the fields is
 *        nullable
 * @param _str_as_dict_cols
 * @param num_str_as_dict_cols
 * @param[out] total_nrows: Pointer used to store to total number of rows to
 *        read. This is used when we are loading 0 columns.
 * @param _only_length_query: Boolean value for if the query was optimized to
 *        only compute the length.
 * @param _is_select_query: Boolean value for if the query is a select
 *        statement.
 * @param batch_size Size of batches for the ArrowReader to produce
 * @return ArrowReader* Output streaming entity
 */
ArrowReader* snowflake_reader_init_py_entry(
    const char* query, const char* conn, bool parallel, bool is_independent,
    PyObject* arrow_schema, int64_t n_fields, int32_t* _is_nullable,
    int32_t num_str_as_dict_cols, int32_t* _str_as_dict_cols,
    int64_t* total_nrows, bool _only_length_query, bool _is_select_query,
    int64_t batch_size) {
    try {
        std::set<int> selected_fields;
        for (auto i = 0; i < n_fields; i++) {
            selected_fields.insert(i);
        }
        std::vector<bool> is_nullable(_is_nullable, _is_nullable + n_fields);
        std::vector<int> str_as_dict_cols(
            {_str_as_dict_cols, _str_as_dict_cols + num_str_as_dict_cols});

        SnowflakeReader* snowflake = new SnowflakeReader(
            query, conn, parallel, is_independent, arrow_schema,
            selected_fields, is_nullable, str_as_dict_cols, _only_length_query,
            _is_select_query, batch_size);
        return static_cast<ArrowReader*>(snowflake);

    } catch (const std::exception& e) {
        // if the error string is "python" this means the C++ exception is
        // a result of a Python exception, so we don't call PyErr_SetString
        // because we don't want to replace the original Python error
        if (std::string(e.what()) != "python")
            PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

// TODO: Remove
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
 * @return table containing all the arrays read
 */
table_info* snowflake_read_py_entry(
    const char* query, const char* conn, bool parallel, bool is_independent,
    PyObject* arrow_schema, int64_t n_fields, int32_t* _is_nullable,
    int32_t* _str_as_dict_cols, int32_t num_str_as_dict_cols,
    int64_t* total_nrows, bool _only_length_query, bool _is_select_query) {
    try {
        std::set<int> selected_fields;
        for (auto i = 0; i < n_fields; i++)
            selected_fields.insert(i);
        std::vector<bool> is_nullable(_is_nullable, _is_nullable + n_fields);
        std::vector<int> str_as_dict_cols(
            {_str_as_dict_cols, _str_as_dict_cols + num_str_as_dict_cols});

        SnowflakeReader reader(query, conn, parallel, is_independent,
                               arrow_schema, selected_fields, is_nullable,
                               str_as_dict_cols, _only_length_query,
                               _is_select_query);

        *total_nrows = reader.get_total_source_rows();
        return reader.read_all();

    } catch (const std::exception& e) {
        // if the error string is "python" this means the C++ exception is
        // a result of a Python exception, so we don't call PyErr_SetString
        // because we don't want to replace the original Python error
        if (std::string(e.what()) != "python")
            PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}
