#pragma once

#include <arrow/python/pyarrow.h>
#include <iostream>
#include <memory>
#include <utility>
#include "../libs/_array_utils.h"
#include "operator.h"

/**
 * @brief Physical node for projection.
 *
 */
class PhysicalPythonScalarFunc : public PhysicalSourceSink {
   public:
    explicit PhysicalPythonScalarFunc(PyObject* args) : args(args) {
        Py_INCREF(args);
    }

    virtual ~PhysicalPythonScalarFunc() { Py_DECREF(args); }

    void Finalize() override {}

    /**
     * @brief Do projection.
     *
     * @return std::pair<int64_t, PyObject*> Bodo C++ table pointer cast to
     * int64 (to pass to Cython easily), pyarrow schema object
     */
    std::pair<std::shared_ptr<table_info>, OperatorResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch) override {
        // Call bodo.pandas.utils.run_apply_udf() to run the UDF

        // Import the bodo.pandas.utils module
        PyObject* bodo_module = PyImport_ImportModule("bodo.pandas.utils");
        if (!bodo_module) {
            PyErr_Print();
            throw std::runtime_error(
                "Failed to import bodo.pandas.utils module");
        }

        // Call the run_apply_udf() with the table_info pointer, Arrow schema
        // and UDF function
        PyObject* pyarrow_schema =
            arrow::py::wrap_schema(input_batch->schema()->ToArrowSchema());
        PyObject* result = PyObject_CallMethod(
            bodo_module, "run_func_on_table", "LOO",
            reinterpret_cast<int64_t>(new table_info(*input_batch)),
            pyarrow_schema, this->args);
        if (!result) {
            PyErr_Print();
            Py_DECREF(bodo_module);
            throw std::runtime_error("Error calling run_apply_udf");
        }

        // Parse the result (assuming it returns a tuple with (table_info,
        // arrow_schema))
        if (!PyTuple_Check(result) || PyTuple_Size(result) != 2) {
            Py_DECREF(result);
            Py_DECREF(bodo_module);
            throw std::runtime_error(
                "Expected a tuple of size 2 from run_apply_udf");
        }

        // Extract the table_info pointer and column names from the result
        PyObject* table_info_py = PyTuple_GetItem(result, 0);
        PyObject* col_names_py = PyTuple_GetItem(result, 1);

        int64_t table_info_ptr = PyLong_AsLongLong(table_info_py);
        std::shared_ptr<table_info> out_batch(
            reinterpret_cast<table_info*>(table_info_ptr));

        // Verify col_names_py is a list
        if (!PyList_Check(col_names_py)) {
            throw std::runtime_error("Expected a list of column names");
        }

        // Clear existing column names and copy from Python list
        out_batch->column_names.clear();
        Py_ssize_t num_cols = PyList_Size(col_names_py);
        for (Py_ssize_t i = 0; i < num_cols; i++) {
            PyObject* col_name = PyList_GetItem(col_names_py, i);
            if (!PyUnicode_Check(col_name)) {
                throw std::runtime_error("Column name must be a string");
            }

            const char* utf8_name = PyUnicode_AsUTF8(col_name);
            if (!utf8_name) {
                throw std::runtime_error(
                    "Failed to convert column name to UTF-8");
            }

            out_batch->column_names.push_back(std::string(utf8_name));
        }

        Py_DECREF(bodo_module);
        Py_DECREF(result);

        return {out_batch, OperatorResult::FINISHED};
    }

   private:
    PyObject* args;
};
