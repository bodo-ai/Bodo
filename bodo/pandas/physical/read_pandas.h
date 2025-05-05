#pragma once

#include <memory>
#include <utility>

#include <arrow/python/pyarrow.h>

#include "../libs/_bodo_to_arrow.h"
#include "operator.h"

/// @brief Physical node for reading Parquet files in pipelines.
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
            PyObject* iloc = PyObject_GetAttrString(df, "iloc");

            // Create slice for all rows (equivalent to ':' in Python)
            PyObject* row_slice = PySlice_New(nullptr, nullptr, nullptr);

            // Convert the selected_columns vector to a Python list
            PyObject* col_list = PyList_New(selected_columns.size());
            for (size_t i = 0; i < selected_columns.size(); ++i) {
                PyList_SET_ITEM(col_list, i,
                                PyLong_FromLong(selected_columns[i]));
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
