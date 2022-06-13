// Copyright (C) 2021 Bodo Inc. All rights reserved.

// Here we define ArrowDataframeReader base class and TableBuilder which are
// used to read Arrow tables into Bodo. Note that we use Arrow to read parquet
// so this includes reading parquet (see ParquetReader subclass in
// parquet_reader.cpp). TableBuilder uses BuilderColumn objects to convert
// Arrow arrays to Bodo. Column builder subclasses are defined in
// arrow_reader.cpp for specific array types.

#ifndef _BODO_ARROW_READER_H
#define _BODO_ARROW_READER_H

#include <Python.h>
#include <set>

#include "../libs/_bodo_common.h"
#include "../libs/_datetime_ext.h"
#include "arrow/array.h"
#include "arrow/python/pyarrow.h"
#include "arrow/table.h"
#include "arrow/type.h"

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
     */
    TableBuilder(std::shared_ptr<arrow::Schema> schema,
                 std::set<int>& selected_fields, const int64_t num_rows,
                 std::vector<bool>& is_nullable,
                 const std::set<std::string>& str_as_dict_cols);

    ~TableBuilder() {
        for (auto col : columns) delete col;
    }

    /**
     * Append data from Arrow table to output Bodo table.
     * NOTE: Can pass Arrow table slices.
     */
    void append(std::shared_ptr<arrow::Table> table);

    /// Get output Bodo table
    table_info* get_table() {
        std::vector<array_info*> arrays;
        for (auto col : columns) {
            arrays.push_back(col->get_output());
        }
        return new table_info(arrays);
    }

    /**
     * Output column builder.
     * This is an abstract class. Subclasses are defined in arrow_reader.cpp
     */
    class BuilderColumn {
       public:
        virtual ~BuilderColumn() {}
        /// Append data from Arrow array to ouput Bodo array
        virtual void append(std::shared_ptr<arrow::ChunkedArray> array) = 0;
        /// Get output Bodo array
        virtual array_info* get_output() { return out_array; }

       protected:
        array_info* out_array = nullptr;  // output array
    };

   private:
    std::vector<BuilderColumn*> columns;  // output column builders
};

// --------- ArrowDataframeReader ---------

/**
 * Abstract class that implements all of the common functionality to read
 * dataframes from Arrow into Bodo. Subclasses need to implement any
 * functionality that is specific to the source (like parquet).
 */
class ArrowDataframeReader {
   public:
    /**
     * @param parallel : true if reading in parallel
     * @param tot_rows_to_read : total number of global rows to read from the
     * dataset (starting from the beginning). Read all rows if is -1
     * @param selected_fields : Fields to select from the Arrow source,
     * using the field ID of Arrow schema
     * NOTE: selected_fields must be sorted
     * @param num_selected_fields : length of selected_fields array
     * @param is_nullable : array of bools that indicates which of the
     * selected fields is nullable. Same length and order as selected_fields.
     */
    ArrowDataframeReader(bool parallel, int64_t tot_rows_to_read,
                         int* selected_fields, int num_selected_fields,
                         int* is_nullable)
        : parallel(parallel),
          tot_rows_to_read(tot_rows_to_read),
          is_nullable(is_nullable, is_nullable + num_selected_fields),
          selected_fields(selected_fields,
                          selected_fields + num_selected_fields) {}

    virtual ~ArrowDataframeReader() { release_gil(); }

    /**
     * Return the number of pieces that this process is going to read.
     * What a piece is and how it is read depends on subclasses.
     */
    virtual size_t get_num_pieces() const = 0;

    /// return the total number of global rows that we are reading
    int64_t get_total_rows() const { return total_rows; }

    /// read data and return a Bodo table
    table_info* read() {
        tracing::Event ev("reader::read", parallel);
        if (!initialized) {
            throw std::runtime_error(
                "ArrowDataframeReader::read(): not initialized");
        }
        TableBuilder builder(schema, selected_fields, count, is_nullable,
                             str_as_dict_colnames);
        rows_left = count;
        read_all(builder);
        if (rows_left != 0)
            throw std::runtime_error(
                "ArrowDataframeReader::read(): did not read all rows");
        return builder.get_table();
    }

   protected:
    const bool parallel;
    const int64_t tot_rows_to_read;  // used for df.head(N) case
    bool initialized = false;
    std::shared_ptr<arrow::Schema> schema;
    std::vector<bool> is_nullable;
    std::set<int> selected_fields;
    std::set<std::string> str_as_dict_colnames;

    /// Total number of rows in the dataset (all pieces)
    int64_t total_rows = 0;

    /// Starting row for first piece
    int64_t start_row_first_piece = 0;

    /// Total number of rows this process has to read (across pieces)
    int64_t count = 0;
    int64_t rows_left;  // only used during ArrowDataframeReader::read()

    /// initialize reader
    virtual void init(const std::vector<int32_t>& str_as_dict_cols = {});

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

    /// Get schema of data that we are reading (arrow-cpp format)
    virtual std::shared_ptr<arrow::Schema> get_schema(PyObject* dataset) = 0;

    /// Read all pieces (data read from pieces is appended to builder)
    virtual void read_all(TableBuilder& builder) = 0;

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

#endif  // _BODO_ARROW_READER_H
