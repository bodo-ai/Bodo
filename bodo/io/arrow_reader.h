
// Here we define ArrowReader base class and TableBuilder which are
// used to read Arrow tables into Bodo. Note that we use Arrow to read parquet
// so this includes reading parquet (see ParquetReader subclass in
// parquet_reader.cpp). TableBuilder uses BuilderColumn objects to convert
// Arrow arrays to Bodo. Column builder subclasses are defined in
// arrow_reader.cpp for specific array types.

#pragma once

#include <Python.h>
#include <set>
#include <stdexcept>

#include <arrow/array.h>
#include <arrow/dataset/scanner.h>
#include <arrow/python/pyarrow.h>
#include <arrow/table.h>
#include <arrow/type.h>

#include "../libs/_bodo_common.h"
#include "../libs/_chunked_table_builder.h"
#include "../libs/_dict_builder.h"
#include "../libs/_query_profile_collector.h"
#include "../libs/_utils.h"
#include "arrow_compat.h"

#define QUERY_PROFILE_INIT_STAGE_ID 0
#define QUERY_PROFILE_READ_STAGE_ID 1

#define CHECK_ARROW_READER(expr, msg)                                        \
    if (!(expr.ok())) {                                                      \
        std::string err_msg = std::string("Error in Arrow Reader: ") + msg + \
                              " " + expr.ToString();                         \
        throw std::runtime_error(err_msg);                                   \
    }

#undef CHECK_ARROW_READER_AND_ASSIGN
#define CHECK_ARROW_READER_AND_ASSIGN(res, msg, lhs) \
    CHECK_ARROW_READER(res.status(), msg)            \
    lhs = std::move(res).ValueOrDie();

// --------- TableBuilder ---------

/**
 * Build Bodo output table from Arrow input data. Can be passed data
 * incrementally for situations where a process has to read data from
 * multiple pieces/files/batches. Note that data will be copied from Arrow
 * to Bodo and no buffers will be shared on returning the output table.
 */
class TableBuilder {
   public:
    /**
     * @param schema : Arrow schema of the input data
     * @param selected_fields : ordered set of fields to select from input data
     * @param num_rows : total number of rows that the output table will have
     * @param is_nullable : indicates which of the selected fields is nullable
     * @param str_as_dict_cols: indices of string cols to be dictionary-encoded
     * @param create_dict_from_string: whether the data source is Snowflake
     * @param dict_ids: vector of dictionary ids used to get consistent ids when
     * streaming.
     */
    TableBuilder(std::shared_ptr<arrow::Schema> schema,
                 std::vector<int>& selected_fields, const int64_t num_rows,
                 const std::vector<bool>& is_nullable,
                 const std::set<std::string>& str_as_dict_cols,
                 const bool create_dict_from_string,
                 const std::vector<int64_t>& dict_ids = {});

    /**
     * @brief Construct a new Table Builder object with the same column types as
     * 'table'. Currently used in concat_tables() but will probably be replaced
     * with a better implementation (e.g. with Arrow's Concatenate when
     * possible).
     *
     * @param table Bodo table to use for typing
     */
    TableBuilder(std::shared_ptr<table_info> table, const int64_t num_rows);

    /**
     * Append data from Arrow table to output Bodo table.
     * NOTE: Can pass Arrow table slices.
     */
    void append(std::shared_ptr<arrow::Table> table);

    /// Get output Bodo table
    /// needs to be raw pointer since will be returned to Python
    /// in ArrowReader code path
    table_info* get_table() {
        std::vector<std::shared_ptr<array_info>> arrays;
        for (auto& col : columns) {
            arrays.push_back(col->get_output());
        }
        return new table_info(arrays, get_total_rows());
    }

    inline int64_t get_total_rows() const { return this->total_rows; }

    inline int64_t get_rem_rows() const { return this->rem_rows; }

    /**
     * Output column builder.
     * This is an abstract class. Subclasses are defined in arrow_reader.cpp
     */
    class BuilderColumn {
       public:
        virtual ~BuilderColumn() {}
        /// Append data from Arrow array to output Bodo array
        virtual void append(std::shared_ptr<arrow::ChunkedArray> array) = 0;
        /// Get output Bodo array
        virtual std::shared_ptr<array_info> get_output() { return out_array; }

       protected:
        std::shared_ptr<array_info> out_array = nullptr;  // output array
    };

   private:
    std::vector<std::unique_ptr<BuilderColumn>>
        columns;  // output column builders
    int64_t total_rows;
    int64_t rem_rows;  // Remaining number of rows to build table
};

// --------- ArrowReader ---------

/**
 * @brief Struct for storing the reader metrics.
 *
 */
struct ArrowReaderMetrics {
    using stat_t = MetricBase::StatValue;
    using time_t = MetricBase::TimerValue;
    using blob_t = MetricBase::BlobValue;

    /// Required
    stat_t output_row_count = 0;

    /// Init Stage

    // Global
    stat_t n_str_as_dict_cols = 0;
    stat_t global_n_pieces = 0;

    // Local
    time_t get_ds_time = 0;
    time_t init_arrow_reader_total_time = 0;
    // In the row-level case, this is the exact number of rows that this rank
    // will output (after applying filters and skipping rows).
    // In the piece-level read case, this is the total number of rows that this
    // rank will read from the source (across all the pieces assigned to it)
    // before applying the filter.
    stat_t local_rows_to_read = 0;
    // In the row-level case, this is the number of pieces that this rank will
    // read from. However, it might only read them partially.
    // In the piece-level case, this is the number of pieces that this rank will
    // read all the rows from. For every piece, exactly one rank will read that
    // piece.
    stat_t local_n_pieces_to_read_from = 0;
    time_t distribute_pieces_or_rows_time = 0;

    /// Read Stage
    time_t read_batch_total_time = 0;
};

/**
 * Abstract class that implements all of the common functionality to read
 * DataFrames from Arrow into Bodo. Subclasses need to implement any
 * functionality that is specific to the source (like parquet).
 */
class ArrowReader {
   public:
    /**
     * @param parallel : true if reading in parallel
     * @param pyarrow_schema: PyArrow schema of source
     * @param tot_rows_to_read : total number of global rows to read from the
     *   dataset (starting from the beginning). Read all rows if is -1
     * @param selected_fields : Fields to select from the Arrow source,
     *   using the field ID of Arrow schema
     * @param is_nullable : array of bools that indicates which of the
     *   selected fields is nullable. Same length and order as selected_fields.
     * @param batch_size : Number of rows for Readers to output per iteration
     *   if they support batching. -1 means dont output in batches, output
     *   entire dataset all at once
     * @param op_id_: Operator ID generated by the planner for query profile
     *   purposes. Defaults to -1 when not generated or not relevant (e.g.
     *   non-streaming).
     */
    ArrowReader(bool parallel, PyObject* pyarrow_schema,
                int64_t tot_rows_to_read, std::vector<int> selected_fields,
                std::vector<bool> is_nullable, int64_t batch_size = -1,
                int64_t op_id_ = -1)
        : parallel(parallel),
          tot_rows_to_read(tot_rows_to_read),
          selected_fields(selected_fields),
          is_nullable(is_nullable),
          batch_size(batch_size),
          op_id(op_id_) {
        this->set_arrow_schema(pyarrow_schema);
    }

    virtual ~ArrowReader() { release_gil(); }

    inline void set_arrow_schema(PyObject* pyarrow_schema) {
        this->pyarrow_schema = pyarrow_schema;
        this->schema = unwrap_schema(pyarrow_schema);
    }

    /**
     * Return the number of pieces that this process is going to read.
     * What a piece is and how it is read depends on subclasses.
     */
    virtual size_t get_num_pieces() const = 0;

    /// Return the total number of global rows that we are reading. Only
    /// relevant in the row-level case.
    int64_t get_total_rows() const {
        assert(this->row_level);
        return this->total_rows;
    }

    /// Return the number of rows read by the current rank. Only
    /// relevant in the row-level case.
    int64_t get_local_rows() const {
        assert(this->row_level);
        return this->count;
    }

    /**
     * @brief Read a batch of data and return a Bodo table
     *
     * @param[out] is_last_out Either this is the last batch or
     *  the last batch has already been returned
     * @param[out] total_rows_out Number of rows in the output batch
     * @param produce_output If false, will not produce any output
     * @return table_info* Out table returned to Python where it will be deleted
     * after use. If there are no rows remaining to be read in, will return
     * an empty table.
     */
    table_info* read_batch(bool& is_last, uint64_t& total_rows_out,
                           bool produce_output) {
        if (!initialized) {
            throw std::runtime_error(
                "ArrowReader::read_batch(): not initialized");
        }
        ScopedTimer timer(this->metrics.read_batch_total_time);

        if (this->row_level) {
            // Handling for repeated extra calls to reader
            // TODO: Determine the general handling for the no-column case
            // (when selected_fields.empty() == true)
            // Right now, this occurs when reading a Snowflake DELETE
            // What would be the equivalent for Parquet?
            if (this->rows_left_to_emit == 0 && !selected_fields.empty()) {
                is_last = true;
                total_rows_out = 0;
                return new table_info(*this->get_empty_out_table());
            }
            if (!produce_output) {
                is_last = this->rows_left_to_emit == 0;
                total_rows_out = 0;
                return new table_info(*this->get_empty_out_table());
            }

            auto [out_table, is_last_, total_rows_out_] =
                this->read_inner_row_level();

            if (is_last_ && this->rows_left_to_emit != 0) {
                throw std::runtime_error(
                    "ArrowReader::read_batch(): did not read all rows");
            }

            total_rows_out = total_rows_out_;
            is_last = is_last_;
            return out_table;
        } else {
            if (this->emitted_all_output && !selected_fields.empty()) {
                is_last = true;
                total_rows_out = 0;
                return new table_info(*this->get_empty_out_table());
            }
            if (!produce_output) {
                is_last = this->emitted_all_output;
                total_rows_out = 0;
                return new table_info(*this->get_empty_out_table());
            }

            auto [out_table, is_last_, total_rows_out_] =
                this->read_inner_piece_level();
            total_rows_out = total_rows_out_;
            is_last = is_last_;
            return out_table;
        }
    }

    /**
     * @brief Convenience function to read the entire dataset and return a Bodo
     * table
     * @return table_info* Out table returned to Python where it will be deleted
     * after use
     */
    table_info* read_all() {
        if (!initialized) {
            throw std::runtime_error(
                "ArrowReader::read_all(): not initialized");
        }
        if (this->batch_size != -1) {
            throw std::runtime_error(
                "ArrowReader::read_all(): Expected to read input all at once, "
                "but a batch-size is defined. Use ArrowReader::read_batch() to "
                "read in batches");
        }

        tracing::Event ev("reader::read_all", parallel);

        if (this->row_level) {
            auto [out_table, is_last_, total_rows_out_] =
                read_inner_row_level();
            assert(is_last_ == true);
            if (this->selected_fields.size()) {
                // If we are not reading any columns, we don't need to check
                // the number of rows read
                assert(total_rows_out_ ==
                       static_cast<uint64_t>(this->get_local_rows()));
            }
            if (this->rows_left_to_emit != 0) {
                throw std::runtime_error(
                    "ArrowReader::read_all(): did not read all rows. " +
                    std::to_string(this->rows_left_to_emit) + " rows left!");
            }

            return out_table;
        } else {
            auto [out_table, is_last_, total_rows_out_] =
                read_inner_piece_level();
            assert(is_last_ == true);
            if (!this->emitted_all_output) {
                throw std::runtime_error(
                    "ArrowReader::read_all(): did not read all rows!");
            }
            return out_table;
        }
    }

    /*
     * @brief determine how many batches to read ahead. Arrows defaults are
     * based on a single reader and we need to adjust for the number of ranks on
     * the node while always ensuring at least one batch is read ahead.
     */
    static int32_t batch_readahead() {
        int ranksOnNode = std::get<0>(dist_get_ranks_on_node());
        return std::max(arrow::dataset::kDefaultBatchReadahead / ranksOnNode,
                        1);
    }

    /*
     * @brief determine how many batches to read ahead. Arrows defaults are
     * based on a single reader and we need to adjust for the number of ranks on
     * the node. We don't make sure this is at least one because fragments can
     * be large, and if we have a lot of ranks a full extra fragment can be too
     * much memory.
     */
    static int32_t frag_readahead() {
        int ranksOnNode = std::get<0>(dist_get_ranks_on_node());
        return arrow::dataset::kDefaultFragmentReadahead / ranksOnNode;
    }

    /**
     * @brief Report Init Stage metrics if they haven't already been reported.
     * Note that this will only report the metrics to the QueryProfileCollector
     * the first time it's called.
     *
     * @param metrics_out out param to push metrics into
     */
    virtual void ReportInitStageMetrics(std::vector<MetricBase>& metrics_out);

    /**
     * @brief Report Read Stage metrics if they haven't already been reported.
     * Note that this will only report the metrics to the QueryProfileCollector
     * the first time it's called.
     *
     * @param metrics_out out param to push metrics into
     */
    virtual void ReportReadStageMetrics(std::vector<MetricBase>& metrics_out);

    /**
     * @brief getter for reported_init_stage_metrics
     */
    bool get_reported_init_stage_metrics() const {
        return reported_init_stage_metrics;
    }

    /**
     * @brief getter for reported_read_stage_metrics
     */
    bool get_reported_read_stage_metrics() const {
        return reported_read_stage_metrics;
    }

   protected:
    const bool parallel;
    const int64_t tot_rows_to_read;  // used for df.head(N) case
    bool initialized = false;

    // During the initialization phase, we have two ways of splitting the
    // dataset between ranks. Either we apply filters at a row level to get
    // exact row counts and then split the dataset exactly between all ranks. In
    // this situation, multiple ranks may read from the same piece (disjoint
    // slices). We call this a "row-level" (i.e. row_level=True) read.
    // The other option is to only apply the filter at the piece level to prune
    // entire pieces. In these cases, we don't know the exact row count (after
    // filtering) up front and must assign every piece to a single rank. We will
    // then filter during the read and return all the rows that satisfy the
    // filter. This is called a "piece-level" (i.e. row_level=False) read.
    // Currently, only the Iceberg reader supports both row-level and
    // piece-level readers. Parquet and Snowflake always do a row-level read.
    bool row_level = true;

    /// Schema of the input, before column pruning, rearranging, or casting
    /// Note for SnowflakeReader, this is the output of the query, not a table
    ///     so it accounts for any transformations done inside of the query
    /// For Parquet & Iceberg, it is the original schema of the files / table
    PyObject* pyarrow_schema;
    std::shared_ptr<arrow::Schema> schema;

    /// Index of fields to select from the input
    /// Equivalent to the field ID of Arrow schema
    /// Note that order matters for the BodoSQL Iceberg case,
    /// so we have to use a vector. Uniqueness is enforced at compilation
    std::vector<int> selected_fields;

    std::vector<bool> is_nullable;

    // For dictionary encoded columns, load them directly as
    // dictionaries (as in the case of parquet), or convert them
    // to dictionaries from regular string arrays (as in the case
    // of Snowflake)
    bool create_dict_encoding_from_strings = false;
    std::set<std::string> str_as_dict_colnames;

    /// Output batch size of streaming tables
    int64_t batch_size;

    /* ROW LEVEL READ CASE */

    /// Total number of rows in the dataset globally (all pieces)
    int64_t total_rows = 0;

    /// Starting row for first piece that this rank will read from.
    int64_t start_row_first_piece = 0;

    /// Total number of rows this rank has to read (across pieces)
    int64_t count = 0;

    /// Number of rows left to emit out of ArrowReader
    /// Needed for streaming reads in ArrowReader::read_inner_row_level()
    int64_t rows_left_to_emit;

    /// Rows left to read from input pieces
    /// Should be equivalent to rows_left_to_emit in non-streaming
    int64_t rows_left_to_read;

    /* PIECE LEVEL READ CASE */

    /// Whether we're done reading all pieces. Note that this only means that
    /// there's no more data to read from the source. We might still have output
    /// left to emit from this operator.
    bool done_reading_pieces = false;

    /// Whether we've emitted all output. This is required since unlike the
    /// row-level case we don't know the row count up front, and so the
    /// implementation must explicitly set this once done.
    bool emitted_all_output = false;

    /// Prepared streaming output batches (Bodo tables) ready to emit. There is
    /// one dictionary builder per column in the output table, with nullptr for
    /// columns that cannot contain a dictionary.
    /// NOTE: This is only used/initialized in the streaming case.
    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders;
    std::shared_ptr<ChunkedTableBuilder> out_batches = nullptr;

    /// initialize reader
    void init_arrow_reader(std::span<int32_t> str_as_dict_cols = {},
                           bool create_dict_from_string = false);

    /**
     * @brief Distribute and assign rows to all ranks from the global set of
     * pieces. Only applicable in the row_level=True case.
     *
     * @param pieces_py List of pieces
     */
    virtual void distribute_rows(PyObject* pieces_py);

    /**
     * @brief Distribute and assign entire pieces to all ranks from the global
     * set of pieces. Only applicable in the piece_level (row_level=False) case.
     *
     * @param pieces_py List of pieces
     */
    virtual void distribute_pieces(PyObject* pieces_py);

    /**
     * Helper Function to Upcast Runtime Data to Expected Reader Schema
     * before adding to TableBuilder. This is currently only used in
     * SnowflakeReader
     * @param table : Input source table to upcast
     * @param schema : Expected schema of the table
     * @return Casted output table
     */
    std::shared_ptr<arrow::Table> cast_arrow_table(
        std::shared_ptr<arrow::Table> table,
        std::shared_ptr<arrow::Schema> schema, bool downcast_decimal_to_double);

    /**
     * Register a piece for this process to read
     * @param piece : piece to register
     * @param num_rows : number of rows from this piece to read
     */
    virtual void add_piece(PyObject* piece, int64_t num_rows) = 0;

    /// Return Python object representing Arrow dataset
    virtual PyObject* get_dataset() = 0;

    // Force a row-level read by default. Individual readers can override this.
    virtual bool force_row_level_read() const { return true; }

    /**
     * @brief Helper Interface Function for sub Readers to implement when
     * reading at a row-level. Perform the actual management and read of input
     * pieces into batches Depending on the reader, must handle batched and
     * non-batched reads.
     */
    virtual std::tuple<table_info*, bool, uint64_t> read_inner_row_level() = 0;

    /**
     * @brief Same as read_inner_row_level, expect when we're doing a
     * piece-level read instead of row-level read, i.e. the exact row counts are
     * not known up front.
     *
     * @return std::tuple<table_info*, bool, uint64_t>
     */
    virtual std::tuple<table_info*, bool, uint64_t>
    read_inner_piece_level() = 0;

    /**
     * @brief Helper function to construct an empty Bodo table with the
     * expected output columns, but zero rows
     *
     * @return table_info* Output table with correct format
     */
    virtual std::shared_ptr<table_info> get_empty_out_table() = 0;

    /**
     * @brief Unify the given table with the dictionary builders
     * for its dictionary columns. This should only be used in the streaming
     * case.
     *
     * @param table Input table.
     * @return table_info* New combined table with unified dictionary columns.
     */
    table_info* unify_table_with_dictionary_builders(
        std::shared_ptr<table_info> table);

    /// Flags to check if we've already submitted the metrics for the init and
    /// read stages to the QueryProfileCollector.
    /// NOTE: Child classes should *not* modify this. They should instead call
    /// ArrowReader::ReportInitStageMetrics and
    /// ArrowReader::ReportReadStageMetrics *after* reporting their own metrics.
    bool reported_init_stage_metrics = false;
    bool reported_read_stage_metrics = false;

   private:
    // XXX needed to call into Python?
    bool gil_held = false;
    PyGILState_STATE gilstate;

    void release_gil() {
        if (gil_held) {
            PyGILState_Release(gilstate);
            gil_held = false;
        }
    }

   public:
    // Operator ID assigned by the planner in the streaming case (for Query
    // Profile purposes).
    const int64_t op_id;
    // Metrics collected during the read.
    ArrowReaderMetrics metrics;
};
