#pragma once

#include <memory>
#include <utility>

#include <arrow/python/pyarrow.h>

#include "../libs/_bodo_to_arrow.h"
#include "operator.h"

/// @brief Physical node for reading Pandas dataframes in pipelines.
class PhysicalReadPandas : public PhysicalSource {
   private:
    PyObject* df;
    int64_t current_row = 0;
    int64_t num_rows;

   public:
    explicit PhysicalReadPandas(PyObject* _df,
                                std::vector<int>& selected_columns)
        : df(_df) {
        Py_INCREF(df);

        // Select only the specified columns if provided by the optimizer
        if (!selected_columns.empty()) {
            // Avoid Index columns since not supported in df.iloc
            PyObject* shape_attr = PyObject_GetAttrString(df, "shape");
            if (!shape_attr) {
                throw std::runtime_error(
                    "Failed to get DataFrame shape attribute");
            }
            // shape is a tuple (n_rows, n_cols)
            PyObject* n_cols_obj =
                PyTuple_GetItem(shape_attr, 1);  // Not a new reference
            int64_t n_cols = PyLong_AsLongLong(n_cols_obj);
            Py_DECREF(shape_attr);
            std::vector<int> selected_columns_no_indexes;
            for (size_t i = 0; i < selected_columns.size(); ++i) {
                if (selected_columns[i] < n_cols) {
                    selected_columns_no_indexes.push_back(selected_columns[i]);
                }
            }

            if (selected_columns_no_indexes.empty()) {
                throw std::runtime_error(
                    "No valid columns selected for PhysicalReadPandas");
            }

            PyObject* iloc = PyObject_GetAttrString(df, "iloc");

            // Create slice for all rows (equivalent to ':' in Python)
            PyObject* row_slice = PySlice_New(nullptr, nullptr, nullptr);

            // Convert the selected_columns_no_indexes vector to a Python list
            PyObject* col_list = PyList_New(selected_columns_no_indexes.size());
            for (size_t i = 0; i < selected_columns_no_indexes.size(); ++i) {
                PyList_SET_ITEM(
                    col_list, i,
                    PyLong_FromLong(selected_columns_no_indexes[i]));
            }

            // Create a tuple with row_slice and col_list for advanced indexing
            PyObject* idx_tuple = PyTuple_New(2);
            PyTuple_SET_ITEM(idx_tuple, 0, row_slice);
            PyTuple_SET_ITEM(idx_tuple, 1, col_list);

            // Apply the indexing
            PyObject* projected_df = PyObject_GetItem(iloc, idx_tuple);

            // Clean up and update df
            Py_DECREF(iloc);
            Py_DECREF(idx_tuple);
            Py_DECREF(df);
            df = projected_df;
        }

        num_rows = PyObject_Length(df);
    }

    virtual ~PhysicalReadPandas() { Py_DECREF(df); }

    void Finalize() override {}

    std::pair<std::shared_ptr<table_info>, ProducerResult> ProduceBatch()
        override {
        if (this->current_row >= this->num_rows) {
            throw std::runtime_error(
                "PhysicalReadPandas::ProduceBatch: No more rows to read");
        }

        // Extract slice from pandas DataFrame
        // df.iloc[current_row:current_row+batch_size]
        int64_t batch_size = get_streaming_batch_size();
        PyObject* iloc = PyObject_GetAttrString(df, "iloc");
        PyObject* slice =
            PySlice_New(PyLong_FromLongLong(this->current_row),
                        PyLong_FromLongLong(this->current_row + batch_size),
                        PyLong_FromLongLong(1));
        PyObject* batch = PyObject_GetItem(iloc, slice);

        // Convert pandas DataFrame to Arrow Table
        PyObject* pyarrow_module = PyImport_ImportModule("pyarrow");
        PyObject* table_func = PyObject_GetAttrString(pyarrow_module, "Table");
        PyObject* pa_table =
            PyObject_CallMethod(table_func, "from_pandas", "O", batch);

        // Unwrap Arrow table from Python object
        std::shared_ptr<arrow::Table> table =
            arrow::py::unwrap_table(pa_table).ValueOrDie();

        // Convert Arrow arrays to Bodo arrays
        auto* bodo_pool = bodo::BufferPool::DefaultPtr();
        std::shared_ptr<table_info> out_table =
            arrow_table_to_bodo(table, bodo_pool);

        // Clean up Python references
        Py_DECREF(iloc);
        Py_DECREF(slice);
        Py_DECREF(batch);
        Py_DECREF(pyarrow_module);
        Py_DECREF(table_func);
        Py_DECREF(pa_table);

        this->current_row += batch_size;

        ProducerResult result = this->current_row >= this->num_rows
                                    ? ProducerResult::FINISHED
                                    : ProducerResult::HAVE_MORE_OUTPUT;

        return {out_table, result};
    }
};

/// @brief Physical node for reading Pandas series in pipelines.
class PhysicalReadSeries : public PhysicalSource {
   private:
    PyObject* series;
    int64_t current_row = 0;
    int64_t num_rows;

   public:
    explicit PhysicalReadSeries(PyObject* _series,
                                std::vector<int>& selected_columns)
        : series(_series) {
        Py_INCREF(series);

        num_rows = PyObject_Length(series);
    }

    virtual ~PhysicalReadSeries() { Py_DECREF(series); }

    void Finalize() override {}

    std::pair<std::shared_ptr<table_info>, ProducerResult> ProduceBatch()
        override {
        if (this->current_row >= this->num_rows) {
            throw std::runtime_error(
                "PhysicalReadSeries::ProduceBatch: No more rows to read");
        }

        // Extract slice from pandas DataFrame
        // series.iloc[current_row:current_row+batch_size]
        int64_t batch_size = get_streaming_batch_size();
        PyObject* iloc = PyObject_GetAttrString(series, "iloc");
        PyObject* slice =
            PySlice_New(PyLong_FromLongLong(this->current_row),
                        PyLong_FromLongLong(this->current_row + batch_size),
                        PyLong_FromLongLong(1));
        PyObject* batch = PyObject_GetItem(iloc, slice);

        // Convert pandas DataFrame to Arrow Table
        PyObject* pyarrow_module = PyImport_ImportModule("pyarrow");
        PyObject* pa_array =
            PyObject_CallMethod(pyarrow_module, "array", "O", batch);

        // Unwrap Arrow table from Python object
        std::shared_ptr<arrow::Array> array =
            arrow::py::unwrap_array(pa_array).ValueOrDie();

        // Convert Arrow arrays to Bodo arrays
        auto* bodo_pool = bodo::BufferPool::DefaultPtr();
        std::vector<std::shared_ptr<array_info>> cvec = {
            arrow_array_to_bodo(array, bodo_pool)};
        std::shared_ptr<table_info> out_table =
            std::make_shared<table_info>(std::move(cvec));

        // Clean up Python references
        Py_DECREF(iloc);
        Py_DECREF(slice);
        Py_DECREF(batch);
        Py_DECREF(pyarrow_module);
        Py_DECREF(pa_array);

        this->current_row += batch_size;

        ProducerResult result = this->current_row >= this->num_rows
                                    ? ProducerResult::FINISHED
                                    : ProducerResult::HAVE_MORE_OUTPUT;

        return {out_table, result};
    }
};
