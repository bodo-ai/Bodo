#pragma once

#include <memory>
#include <utility>

#include <arrow/python/pyarrow.h>
#include <arrow/table.h>

#include "../libs/_bodo_to_arrow.h"
#include "../libs/_table_builder_utils.h"
#include "operator.h"

/// @brief Physical node for reading Parquet files in pipelines.
class PhysicalReadPandas : public PhysicalSource {
   private:
    PyObject* df;
    int64_t current_row = 0;
    int64_t num_rows;
    const std::shared_ptr<bodo::Schema> output_schema;

   public:
    explicit PhysicalReadPandas(PyObject* _df_or_series,
                                std::vector<int>& selected_columns,
                                std::shared_ptr<arrow::Schema> arrow_schema)
        : output_schema(initOutputSchema(selected_columns, arrow_schema)) {
        this->setInputDF(_df_or_series);

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

    /**
     * @brief Initialize the output schema based on the selected columns and
     * Arrow schema.
     *
     * @param selected_columns The selected columns to project.
     * @param arrow_schema The Arrow schema of the DataFrame.
     * @return std::shared_ptr<bodo::Schema> The initialized output schema.
     */
    static std::shared_ptr<bodo::Schema> initOutputSchema(
        std::vector<int>& selected_columns,
        std::shared_ptr<arrow::Schema> arrow_schema) {
        std::shared_ptr<bodo::Schema> output_schema =
            bodo::Schema::FromArrowSchema(arrow_schema);
        if (!selected_columns.empty()) {
            output_schema = output_schema->Project(selected_columns);
        }
        return output_schema;
    }

    virtual ~PhysicalReadPandas() { Py_DECREF(df); }

    void FinalizeSource() override {}

    std::pair<std::shared_ptr<table_info>, OperatorResult> ProduceBatch()
        override {
        // NOTE: current_row may be greater than num_rows since sources need to
        // be able to produce empty batches. df.iloc will return an empty
        // DataFrame if the slice is out of bounds.

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

        if (!pa_table) {
            PyErr_Print();
            throw std::runtime_error("Failed to convert pandas to Arrow table");
        }

        arrow::Result<std::shared_ptr<arrow::Table>> pa_table_result =
            arrow::py::unwrap_table(pa_table);

        if (!pa_table_result.status().ok()) {
            throw std::runtime_error(
                "Failed to convert pandas DataFrame to Arrow Table: " +
                pa_table_result.status().ToString());
        }

        // Unwrap Arrow table from Python object
        std::shared_ptr<arrow::Table> table = pa_table_result.ValueOrDie();

        std::shared_ptr<table_info> out_table;
        if (table->num_rows() == 0) {
            // Use alloc_table here since calling from_pandas on an empty slice
            // might return different types.
            out_table = alloc_table(output_schema);
        } else {
            // arrow_table_to_bodo expects a single chunk table.
            table = table->CombineChunks().ValueOrDie();
            // Convert Arrow arrays to Bodo arrays
            // (passing nullptr for pool since not allocated through Bodo so
            // can't support spilling etc,
            // TODO: pass bodo's buffer pool to Arrow)
            out_table = arrow_table_to_bodo(table, nullptr);
        }

        // Clean up Python references
        Py_DECREF(iloc);
        Py_DECREF(slice);
        Py_DECREF(batch);
        Py_DECREF(pyarrow_module);
        Py_DECREF(table_func);
        Py_DECREF(pa_table);

        this->current_row += batch_size;

        OperatorResult result = this->current_row >= this->num_rows
                                    ? OperatorResult::FINISHED
                                    : OperatorResult::HAVE_MORE_OUTPUT;

        return {out_table, result};
    }

    /**
     * @brief Get the physical schema of the dataframe/series data
     *
     * @return std::shared_ptr<bodo::Schema> physical schema
     */
    const std::shared_ptr<bodo::Schema> getOutputSchema() override {
        return output_schema;
    }

   private:
    /**
     * @brief Convert to a DataFrame if the input is a Series.
     *
     * @param _df_or_series input DataFrame or Series
     */
    void setInputDF(PyObject* _df_or_series) {
        // Check if _df_or_series is a Series
        PyObject* pandas_module = PyImport_ImportModule("pandas");
        if (!pandas_module) {
            throw std::runtime_error("Failed to import pandas module");
        }
        PyObject* series_class =
            PyObject_GetAttrString(pandas_module, "Series");
        if (!series_class) {
            Py_XDECREF(pandas_module);
            Py_XDECREF(series_class);
            throw std::runtime_error("Failed to get Series classes");
        }

        if (PyObject_IsInstance(_df_or_series, series_class) == 1) {
            PyObject* df_from_series =
                PyObject_CallMethod(_df_or_series, "to_frame", nullptr);
            if (!df_from_series) {
                Py_XDECREF(pandas_module);
                Py_XDECREF(series_class);
                throw std::runtime_error(
                    "Failed to convert Series to DataFrame");
            }
            Py_XDECREF(pandas_module);
            Py_XDECREF(series_class);
            this->df = df_from_series;
        } else {
            this->df = _df_or_series;
            Py_INCREF(df);
        }
    }
};
