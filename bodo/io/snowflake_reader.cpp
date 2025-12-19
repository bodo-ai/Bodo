
// Implementation of SnowflakeReader (subclass of ArrowReader) with
// functionality that is specific to reading from Snowflake

#include <queue>
#include "arrow_reader.h"

#include "../libs/_bodo_to_arrow.h"
#include "../libs/_dict_builder.h"
#include "../libs/_distributed.h"

/**
 * @brief Struct for storing SnowflakeReader metrics.
 *
 */
struct SnowflakeReaderMetrics : public ArrowReaderMetrics {
    /// Init stage
    time_t sf_data_prep_time = 0;

    /// Read stage
    time_t to_arrow_time = 0;
    time_t cast_arrow_table_time = 0;
    time_t total_append_time = 0;
    time_t arrow_rb_to_bodo_time = 0;
    time_t ctb_pop_chunk_time = 0;
};

// -------- SnowflakeReader --------

class SnowflakeReader : public ArrowReader {
   public:
    /**
     * Initialize SnowflakeReader.
     * See snowflake_read_py_entry function below for description of arguments.
     */
    SnowflakeReader(const char* _query, const char* _conn, bool _parallel,
                    bool _is_independent, PyObject* _pyarrow_schema,
                    std::vector<int> selected_fields,
                    std::vector<bool> is_nullable,
                    std::vector<int> str_as_dict_cols, bool _only_length_query,
                    bool _is_select_query, bool _downcast_decimal_to_double,
                    int64_t batch_size = -1, int64_t op_id = -1)
        : ArrowReader(_parallel, _pyarrow_schema, -1, selected_fields,
                      is_nullable, batch_size, op_id),
          query(_query),
          conn(_conn),
          only_length_query(_only_length_query),
          is_select_query(_is_select_query),
          is_independent(_is_independent),
          downcast_decimal_to_double(_downcast_decimal_to_double) {
        // Initialize reader
        init_arrow_reader(str_as_dict_cols, true);

        // Construct ChunkedTableBuilder for output
        this->dict_builders = std::vector<std::shared_ptr<DictionaryBuilder>>(
            schema->num_fields());
        for (int i = 0; i < schema->num_fields(); i++) {
            const std::shared_ptr<arrow::Field>& field = schema->field(i);
            this->dict_builders[i] = create_dict_builder_for_array(
                arrow_type_to_bodo_data_type(field->type()), false);
        }

        for (int str_as_dict_col : str_as_dict_cols) {
            this->dict_builders[str_as_dict_col] =
                create_dict_builder_for_array(
                    std::make_unique<bodo::DataType>(bodo_array_type::DICT,
                                                     Bodo_CTypes::STRING),
                    false);
        }

        auto empty_table = get_empty_out_table();
        this->out_batches = std::make_shared<ChunkedTableBuilder>(
            empty_table->schema(), this->dict_builders, (size_t)batch_size);
    }

    ~SnowflakeReader() override { Py_XDECREF(sf_conn); }

    /// A piece is a snowflake.connector.result_batch.ArrowResultBatch
    size_t get_num_pieces() const override {
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

    /// @brief Report SnowflakeReader specific metrics in addition to the
    /// ArrowReader metrics.
    void ReportInitStageMetrics(std::vector<MetricBase>& metrics_out) override {
        if ((this->op_id == -1) || this->reported_init_stage_metrics) {
            return;
        }

        metrics_out.reserve(metrics_out.size() + 2);

        metrics_out.emplace_back(TimerMetric(
            "sf_data_prep_time", this->metrics.sf_data_prep_time, true));

        ArrowReader::ReportInitStageMetrics(metrics_out);
    }

    /// @brief Report SnowflakeReader specific metrics in addition to the
    /// ArrowReader metrics.
    void ReportReadStageMetrics(std::vector<MetricBase>& metrics_out) override {
        if ((this->op_id == -1) || this->reported_read_stage_metrics) {
            return;
        }

        metrics_out.reserve(metrics_out.size() + 32);

        metrics_out.emplace_back(
            TimerMetric("to_arrow_time", this->metrics.to_arrow_time));
        metrics_out.emplace_back(TimerMetric(
            "cast_arrow_table_time", this->metrics.cast_arrow_table_time));
        metrics_out.emplace_back(
            TimerMetric("total_append_time", this->metrics.total_append_time));
        metrics_out.emplace_back(TimerMetric(
            "arrow_rb_to_bodo_time", this->metrics.arrow_rb_to_bodo_time));
        metrics_out.emplace_back(TimerMetric("ctb_pop_chunk_time",
                                             this->metrics.ctb_pop_chunk_time));

        // Get time spent appending to, total number of rows appended to, and
        // max reached size of the output buffer
        if (this->out_batches != nullptr) {
            metrics_out.emplace_back(TimerMetric(
                "output_append_time", this->out_batches->append_time));
            MetricBase::StatValue output_total_size =
                this->out_batches->total_size;
            metrics_out.emplace_back(
                StatMetric("output_total_nrows", output_total_size));
            MetricBase::StatValue output_peak_nrows =
                this->out_batches->max_reached_size;
            metrics_out.emplace_back(
                StatMetric("output_peak_nrows", output_peak_nrows));
        }

        // Dict builder metrics
        DictBuilderMetrics dict_builder_metrics;
        MetricBase::StatValue n_dict_builders = 0;
        for (const auto& dict_builder : this->dict_builders) {
            if (dict_builder != nullptr) {
                dict_builder_metrics.add_metrics(dict_builder->GetMetrics());
                n_dict_builders++;
            }
        }
        metrics_out.emplace_back(
            StatMetric("n_dict_builders", n_dict_builders, true));
        dict_builder_metrics.add_to_metrics(metrics_out, "dict_builders_");

        ArrowReader::ReportReadStageMetrics(metrics_out);
    }

   protected:
    void add_piece(PyObject* piece, int64_t num_rows) override {
        Py_INCREF(piece);  // keeping a reference to this piece
        result_batches.push(piece);
    }

    PyObject* get_dataset() override {
        // import bodo.io.snowflake
        PyObject* sf_mod = PyImport_ImportModule("bodo.io.snowflake");
        if (PyErr_Occurred()) {
            throw std::runtime_error("python");
        }

        // ds = bodo.io.snowflake.get_dataset(query, conn, pyarrow_schema,
        //   only_length, is_select_query, is_parallel, is_independent)
        PyObject* py_only_length_query = PyBool_FromLong(only_length_query);
        PyObject* py_is_select_query = PyBool_FromLong(is_select_query);
        PyObject* py_is_parallel = PyBool_FromLong(parallel);
        PyObject* py_is_independent = PyBool_FromLong(this->is_independent);
        PyObject* ds_tuple = PyObject_CallMethod(
            sf_mod, "get_dataset", "ssOOOOO", query, conn, this->pyarrow_schema,
            py_only_length_query, py_is_select_query, py_is_parallel,
            py_is_independent);
        if (ds_tuple == nullptr && PyErr_Occurred()) {
            throw std::runtime_error("python");
        }
        Py_DECREF(sf_mod);
        Py_DECREF(this->pyarrow_schema);
        Py_DECREF(py_only_length_query);
        Py_DECREF(py_is_select_query);
        Py_DECREF(py_is_parallel);
        Py_DECREF(py_is_independent);

        // PyTuple_GetItem borrows a reference
        PyObject* ds = PyTuple_GetItem(ds_tuple, 0);
        Py_INCREF(ds);  // call incref to keep the reference
        // PyTuple_GetItem borrows a reference
        PyObject* total_len = PyTuple_GetItem(ds_tuple, 1);
        this->total_nrows = PyLong_AsLongLong(total_len);
        // PyTuple_GetItem borrows a reference
        PyObject* sf_exec_time_us_py = PyTuple_GetItem(ds_tuple, 2);
        int64_t sf_exec_time_us = PyLong_AsLongLong(sf_exec_time_us_py);
        this->metrics.sf_data_prep_time = sf_exec_time_us;
        Py_DECREF(ds_tuple);
        this->sf_conn = PyObject_GetAttrString(ds, "conn");
        if (sf_conn == nullptr) {
            throw std::runtime_error(
                "Could not retrieve conn attribute of Snowflake dataset");
        }

        // all_pieces = ds.pieces
        PyObject* all_pieces = PyObject_GetAttrString(ds, "pieces");
        this->num_pieces = static_cast<uint64_t>(PyObject_Length(all_pieces));
        return ds;
    }

    std::shared_ptr<table_info> get_empty_out_table() override {
        if (this->empty_out_table == nullptr) {
            TableBuilder builder(this->schema, selected_fields, 0, is_nullable,
                                 str_as_dict_colnames, false);
            this->empty_out_table =
                std::shared_ptr<table_info>(builder.get_table());
        }
        return this->empty_out_table;
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
        time_pt start = start_timer();
        PyObject* arrow_table_py =
            PyObject_CallMethod(next_piece, "to_arrow", "O", sf_conn);
        if (arrow_table_py == nullptr && PyErr_Occurred()) {
            throw std::runtime_error("python");
        }

        auto table = arrow::py::unwrap_table(arrow_table_py).ValueOrDie();
        int64_t to_arrow_time = end_timer(start);

        start = start_timer();
        int64_t length =
            std::min(rows_left_to_read, table->num_rows() - offset);
        // TODO: Add check if length < 0, never possible
        table = table->Slice(offset, length);

        // Upcast Arrow table to expected schema
        // TODO: Integrate within TableBuilder
        table = ArrowReader::cast_arrow_table(table, this->schema,
                                              downcast_decimal_to_double);
        int64_t cast_arrow_table_time = end_timer(start);

        Py_DECREF(next_piece);
        Py_DECREF(arrow_table_py);
        return std::make_tuple(table, to_arrow_time, cast_arrow_table_time);
    }

    // Design for Batched Snowflake Reads:
    // https://bodo.atlassian.net/l/cp/4JwaiChQ
    // Non-Streaming Uses TableBuilder to Construct Output
    // Streaming Uses ChunkedTableBuilder to Construct Batches
    std::tuple<table_info*, bool, uint64_t> read_inner_row_level() override {
        // In the case when we only need to perform a count(*)
        // We need to still return a table with 0 columns but
        // `this->total_nrows` rows We can return a nullptr for the table_info*,
        // but still need to return `this->total_nrows / num_ranks` for the
        // length on each rank. In the batching case, this only occurs on the
        // first iteration
        if (this->only_length_query) {
            // Runtime Assertions
            // No performance penalty since ArrowReader sets up
            // call stack anyways
            if (!result_batches.empty() || !out_batches->empty()) {
                throw std::runtime_error(
                    "SnowflakeReader: Read a batch in the "
                    "only_length_query case!");
            }

            if (this->rows_left_to_read != 0) {
                throw std::runtime_error(
                    "SnowflakeReader: Expecting to read a row in the "
                    "only_length_query case!");
            }

            size_t rank = dist_get_rank();
            size_t nranks = dist_get_size();
            auto out_rows =
                returned_once_empty_case
                    ? 0
                    : dist_get_node_portion(this->total_nrows, nranks, rank);
            returned_once_empty_case = true;
            // TODO(srilman): Why does nullptr work here?
            return std::make_tuple(nullptr, true, out_rows);
        }

        // Attempting to read the entire table at once (read_all() path)
        // Right now, the batched reader does not attempt to combine small
        // pieces together to get the expected batch size. Thus, we need this
        // special case
        // TODO: Handle when len(piece) < batch_size and remove this special
        // case
        if (batch_size == -1) {
            tracing::Event ev("reader::read_inner_row_level", false);
            TableBuilder builder(this->schema, selected_fields, count,
                                 is_nullable, str_as_dict_colnames,
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
                this->metrics.to_arrow_time += to_arrow_time_;
                this->metrics.cast_arrow_table_time += cast_arrow_table_time_;

                this->rows_left_to_read -= table->num_rows();

                time_pt start = start_timer();
                builder.append(table);
                this->metrics.total_append_time += end_timer(start);
            }

            ev.add_attribute("total_to_arrow_time_micro",
                             this->metrics.to_arrow_time);
            ev.add_attribute("total_append_time_micro",
                             this->metrics.total_append_time);
            ev.add_attribute("total_cast_arrow_table_time_micro",
                             this->metrics.cast_arrow_table_time);

            auto out_table = builder.get_table();
            // TODO: This is a hack to get the correct number of rows
            // when reading 0 columns
            // Should be true in the non-streaming case anyways
            rows_left_to_emit = rows_left_to_read;
            ev.add_attribute("out_batch_size", out_table->nrows());
            ev.finalize();
            return std::make_tuple(out_table, true, out_table->nrows());
        }

        if (out_batches->chunks.empty() && !result_batches.empty()) {
            auto next_piece = result_batches.front();
            result_batches.pop();

            int64_t offset = 0;
            if (is_first_piece) {
                offset = start_row_first_piece;
                is_first_piece = false;
            }

            auto [table, to_arrow_time, cast_arrow_table_time] =
                process_result_batch(next_piece, offset);
            this->metrics.to_arrow_time += to_arrow_time;
            this->metrics.cast_arrow_table_time += cast_arrow_table_time;
            this->rows_left_to_read -= table->num_rows();

            time_pt start = start_timer();
            // Arrow utility to zero-copy construct RecordBatches tables.
            // Tables consist of chunked arrays with inconsistent boundaries
            // Difficult to directly zero-copy
            // See https://arrow.apache.org/docs/cpp/tables.html#record-batches
            // for details and rationale
            auto reader = arrow::TableBatchReader(table);
            // Want the largest contiguous chunks possible
            reader.set_chunksize(table->num_rows());

            std::shared_ptr<arrow::RecordBatch> next_recordbatch;
            for (auto status = reader.ReadNext(&next_recordbatch);
                 status.ok() && next_recordbatch;
                 status = reader.ReadNext(&next_recordbatch)) {
                time_pt start_arrow_to_bodo = start_timer();
                auto bodo_table = arrow_recordbatch_to_bodo(
                    next_recordbatch, next_recordbatch->num_rows());
                this->metrics.arrow_rb_to_bodo_time +=
                    end_timer(start_arrow_to_bodo);
                out_batches->UnifyDictionariesAndAppend(bodo_table);
            }
            this->metrics.total_append_time += end_timer(start);

            // Explicitly Finalize ChunkedTableBuilder once all data
            // has been consumed
            if (result_batches.empty()) {
                out_batches->Finalize();
            }
        }

        time_pt start_pop = start_timer();
        auto [next_batch, out_batch_size] = out_batches->PopChunk();
        this->metrics.ctb_pop_chunk_time += end_timer(start_pop);

        rows_left_to_emit -= out_batch_size;
        bool is_last = result_batches.empty() && out_batches->empty();
        return std::make_tuple(new table_info(*next_batch), is_last,
                               out_batch_size);
    }

    std::tuple<table_info*, bool, uint64_t> read_inner_piece_level() override {
        throw std::runtime_error(
            "SnowflakeReader::read_inner_piece_level: Not supported!");
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

    // instance of snowflake.connector.connection.SnowflakeConnection, used
    // to read the batches
    PyObject* sf_conn = nullptr;
    // Are the ranks executing the function independently?
    bool is_independent = false;
    // Should we unsafely downcast decimal columns to double
    bool downcast_decimal_to_double = false;

    // Number of total pieces / ResultBatches the SnowflakeReader
    // has for all ranks
    uint64_t num_pieces;
    // Batches that this process is going to read
    // A batch is a snowflake.connector.result_batch.ArrowResultBatch
    std::queue<PyObject*> result_batches;

    // If we still need to return the first batch for streaming
    bool is_first_piece = true;

    // Empty Table for Mapping Datatypes
    std::shared_ptr<table_info> empty_out_table = nullptr;

   public:
    SnowflakeReaderMetrics metrics;
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
 * @param _only_length_query: Boolean value for if the query was optimized
 * to only compute the length.
 * @param _is_select_query: Boolean value for if the query is a select
 *        statement.
 * @param downcast_decimal_to_double Always unsafely downcast double columns
 * to decimal.
 * @param batch_size Size of batches for the ArrowReader to produce
 * @param op_id Operator ID for query profile. Can be set to -1 if not
 * known/relevant.
 * @return ArrowReader* Output streaming entity
 */
ArrowReader* snowflake_reader_init_py_entry(
    const char* query, const char* conn, bool parallel, bool is_independent,
    PyObject* arrow_schema, int64_t n_fields, int32_t* _is_nullable,
    int32_t num_str_as_dict_cols, int32_t* _str_as_dict_cols,
    int64_t* total_nrows, bool _only_length_query, bool _is_select_query,
    bool downcast_decimal_to_double, int64_t batch_size, int64_t op_id) {
    try {
        std::vector<int> selected_fields;
        for (auto i = 0; i < n_fields; i++) {
            selected_fields.push_back(i);
        }
        std::vector<bool> is_nullable(_is_nullable, _is_nullable + n_fields);
        std::vector<int> str_as_dict_cols(
            {_str_as_dict_cols, _str_as_dict_cols + num_str_as_dict_cols});

        SnowflakeReader* snowflake = new SnowflakeReader(
            query, conn, parallel, is_independent, arrow_schema,
            selected_fields, is_nullable, str_as_dict_cols, _only_length_query,
            _is_select_query, downcast_decimal_to_double, batch_size, op_id);

        std::vector<MetricBase> metrics;
        snowflake->ReportInitStageMetrics(metrics);
        QueryProfileCollector::Default().RegisterOperatorStageMetrics(
            QueryProfileCollector::MakeOperatorStageID(
                snowflake->op_id, QUERY_PROFILE_INIT_STAGE_ID),
            std::move(metrics));

        return static_cast<ArrowReader*>(snowflake);

    } catch (const std::exception& e) {
        // if the error string is "python" this means the C++ exception is
        // a result of a Python exception, so we don't call PyErr_SetString
        // because we don't want to replace the original Python error
        if (std::string(e.what()) != "python") {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
        return nullptr;
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
 * @param _only_length_query: Boolean value for if the query was optimized
 to only compute the length.
 * @param _is_select_query: Boolean value for if the query is a select
 statement.
 * @param downcast_decimal_to_double Always unsafely downcast double columns
 to
 *        decimal.
 * @return table containing all the arrays read
 */
table_info* snowflake_read_py_entry(
    const char* query, const char* conn, bool parallel, bool is_independent,
    PyObject* arrow_schema, int64_t n_fields, int32_t* _is_nullable,
    int32_t* _str_as_dict_cols, int32_t num_str_as_dict_cols,
    int64_t* total_nrows, bool _only_length_query, bool _is_select_query,
    bool downcast_decimal_to_double) {
    table_info* out = nullptr;
    // Need to synchronize exceptions in the top level case
    // TODO: This would be a nice macro to have in general
    std::optional<const char*> err_str;

    try {
        std::vector<int> selected_fields;
        for (auto i = 0; i < n_fields; i++) {
            selected_fields.push_back(i);
        }
        std::vector<bool> is_nullable(_is_nullable, _is_nullable + n_fields);
        std::vector<int> str_as_dict_cols(
            {_str_as_dict_cols, _str_as_dict_cols + num_str_as_dict_cols});

        SnowflakeReader reader(query, conn, parallel, is_independent,
                               arrow_schema, selected_fields, is_nullable,
                               str_as_dict_cols, _only_length_query,
                               _is_select_query, downcast_decimal_to_double);

        *total_nrows = reader.get_total_source_rows();
        out = reader.read_all();

    } catch (const std::exception& e) {
        err_str = e.what();
    }

    bool has_err_global;
    if (is_independent) {
        has_err_global = err_str.has_value();
    } else {
        auto has_err_local = err_str.has_value();
        has_err_global = false;
        CHECK_MPI(MPI_Allreduce(&has_err_local, &has_err_global, 1, MPI_C_BOOL,
                                MPI_LOR, MPI_COMM_WORLD),
                  "snowflake_read_py_entry: MPI error on MPI_Allreduce:");
    }

    if (has_err_global) {
        // if the error string is "python" this means the C++ exception is
        // a result of a Python exception, so we don't call PyErr_SetString
        // because we don't want to replace the original Python error
        if (std::string(err_str.value_or("")) != "python") {
            PyErr_SetString(
                PyExc_RuntimeError,
                err_str.value_or("See other ranks for runtime errors"));
        }
    }
    return out;
}
