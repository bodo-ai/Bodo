// Copyright (C) 2021 Bodo Inc. All rights reserved.

// Here we define ArrowReader base class and TableBuilder which are
// used to read Arrow tables into Bodo. Note that we use Arrow to read parquet
// so this includes reading parquet (see ParquetReader subclass in
// parquet_reader.cpp). TableBuilder uses BuilderColumn objects to convert
// Arrow arrays to Bodo. Column builder subclasses are defined in
// arrow_reader.cpp for specific array types.

#pragma once

#include <Python.h>
#include <set>

#include <arrow/array.h>
#include <arrow/python/pyarrow.h>
#include <arrow/table.h>
#include <arrow/type.h>

#include "../libs/_bodo_common.h"
#include "../libs/_dict_builder.h"
#include "../libs/_table_builder.h"

/**
 * @brief Unwrap PyArrow Schema PyObject and return the C++ value
 * NOTE: Not calling arrow::py::unwrap_schema() in ArrowReader constructor
 * directly due to a segfault with pip. See:
 * https://bodo.atlassian.net/browse/BSE-2925
 *
 * @param pyarrow_schema input PyArrow Schema
 * @return std::shared_ptr<arrow::Schema> C++ Schema
 */
std::shared_ptr<arrow::Schema> unwrap_schema(PyObject* pyarrow_schema);

#define CHECK_ARROW(expr, msg)                                               \
    if (!(expr.ok())) {                                                      \
        std::string err_msg = std::string("Error in Arrow Reader: ") + msg + \
                              " " + expr.ToString();                         \
        throw std::runtime_error(err_msg);                                   \
    }

#undef CHECK_ARROW_AND_ASSIGN
#define CHECK_ARROW_AND_ASSIGN(res, msg, lhs) \
    CHECK_ARROW(res.status(), msg)            \
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
     * entire dataset all at once
     */
    ArrowReader(bool parallel, PyObject* pyarrow_schema,
                int64_t tot_rows_to_read, std::vector<int> selected_fields,
                std::vector<bool> is_nullable, int64_t batch_size = -1)
        : parallel(parallel),
          tot_rows_to_read(tot_rows_to_read),
          selected_fields(selected_fields),
          is_nullable(is_nullable),
          batch_size(batch_size) {
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

    /// return the total number of global rows that we are reading
    int64_t get_total_rows() const { return total_rows; }

    /// Return the number of rows read by the current rank
    int64_t get_local_rows() const { return count; }

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

        // Note: These can be called in a loop by streaming.
        // Set the parallel flag to False.
        tracing::Event ev("reader::read_batch", false);
        // Handling for repeated extra calls to reader
        // TODO: Determine the general handling for the no-column case
        // (when selected_fields.empty() == true)
        // Right now, this occurs when reading a Snowflake DELETE
        // What would be the equivalent for Parquet?
        if (rows_left_to_emit == 0 && !selected_fields.empty()) {
            is_last = true;
            total_rows_out = 0;
            ev.finalize();

            return new table_info(*this->get_empty_out_table());
        }

        if (!produce_output) {
            is_last = rows_left_to_emit == 0;
            total_rows_out = 0;
            ev.finalize();

            return new table_info(*this->get_empty_out_table());
        }

        auto [out_table, is_last_, total_rows_out_] = read_inner();
        if (is_last_ && rows_left_to_emit != 0) {
            throw std::runtime_error(
                "ArrowReader::read_batch(): did not read all rows");
        }

        total_rows_out = total_rows_out_;
        is_last = is_last_;
        return out_table;
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
                "but "
                "a batch-size is defined. Use ArrowReader::read_batch() to "
                "read in batches");
        }

        tracing::Event ev("reader::read_all", parallel);
        auto [out_table, is_last_, total_rows_out_] = read_inner();
        assert(is_last_ == true);
        if (selected_fields.size()) {
            // If we are not reading any columns, we don't need to check
            // the number of rows read
            assert(total_rows_out_ ==
                   static_cast<uint64_t>(this->get_local_rows()));
        }
        if (rows_left_to_emit != 0) {
            throw std::runtime_error(
                "ArrowReader::read_all(): did not read all rows. " +
                std::to_string(rows_left_to_emit) + " rows left!");
        }

        return out_table;
    }

   protected:
    const bool parallel;
    const int64_t tot_rows_to_read;  // used for df.head(N) case
    bool initialized = false;

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

    /// Total number of rows in the dataset (all pieces)
    int64_t total_rows = 0;

    /// Starting row for first piece
    int64_t start_row_first_piece = 0;

    /// Total number of rows this process has to read (across pieces)
    int64_t count = 0;

    /// Number of rows left to emit out of ArrowReader
    /// Needed for streaming reads in ArrowReader::read_inner()
    int64_t rows_left_to_emit;

    /// Rows left to read from input pieces
    /// Should be equivalent to rows_left_to_emit in non-streaming
    int64_t rows_left_to_read;

    /// Output batch size of streaming tables
    int64_t batch_size;

    /// Prepared streaming output batches (Bodo tables) ready to emit. There is
    /// one dictionary builder per column in the output table, with nullptr for
    /// columns that cannot contain a dictionary.
    /// NOTE: This is only used/initialized in the streaming case.
    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders;
    std::shared_ptr<ChunkedTableBuilder> out_batches;

    /// initialize reader
    void init_arrow_reader(std::span<int32_t> str_as_dict_cols = {},
                           bool create_dict_from_string = false);

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
     * @param total_rows : total number of rows this process has to read (across
     * pieces)
     */
    virtual void add_piece(PyObject* piece, int64_t num_rows,
                           int64_t total_rows) = 0;

    /// Return Python object representing Arrow dataset
    virtual PyObject* get_dataset() = 0;

    /**
     * @brief Helper Interface Function for sub Readers to implement
     * Perform the actual management and read of input pieces into batches
     * Depending on the reader, must handle batched and non-batched reads
     */
    virtual std::tuple<table_info*, bool, uint64_t> read_inner() = 0;

    /**
     * @brief Helper function to construct an empty Bodo table with the
     * expected output columns, but zero rows
     *
     * @return table_info* Output table with correct format
     */
    virtual std::shared_ptr<table_info> get_empty_out_table() = 0;

   protected:
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
};
