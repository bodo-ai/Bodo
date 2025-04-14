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
    explicit PhysicalReadPandas(PyObject* _df) : df(_df) {
        Py_INCREF(df);
        num_rows = PyObject_Length(df);
    }

    virtual ~PhysicalReadPandas() { Py_DECREF(df); }

    void Finalize() override {}

    std::pair<std::shared_ptr<table_info>, ProducerResult> ProduceBatch()
        override {
        // Extract slice from pandas DataFrame
        // df.iloc[current_row:current_row+batch_size]
        // TODO: convert to streaming
        int64_t batch_size = this->num_rows;
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

        return {out_table, ProducerResult::FINISHED};
    }
};
