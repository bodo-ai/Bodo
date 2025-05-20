#include "project.h"
#include <arrow/python/pyarrow.h>

std::shared_ptr<table_info> PhysicalProjection::runPythonScalarFunction(
    std::shared_ptr<table_info> input_batch,
    const std::shared_ptr<arrow::DataType>& result_type, PyObject* args) {
    // Call bodo.pandas.utils.run_apply_udf() to run the UDF

    // Import the bodo.pandas.utils module
    PyObject* bodo_module = PyImport_ImportModule("bodo.pandas.utils");
    if (!bodo_module) {
        PyErr_Print();
        throw std::runtime_error("Failed to import bodo.pandas.utils module");
    }

    // Call the run_apply_udf() with the table_info pointer, Arrow schema
    // and UDF function
    PyObject* pyarrow_schema =
        arrow::py::wrap_schema(input_batch->schema()->ToArrowSchema());
    PyObject* result_type_py(arrow::py::wrap_data_type(result_type));
    PyObject* result = PyObject_CallMethod(
        bodo_module, "run_func_on_table", "LOOO",
        reinterpret_cast<int64_t>(new table_info(*input_batch)), pyarrow_schema,
        result_type_py, args);
    if (!result) {
        PyErr_Print();
        Py_DECREF(bodo_module);
        throw std::runtime_error("Error calling run_apply_udf");
    }

    // Result should be a pointer to a C++ table_info
    if (!PyLong_Check(result)) {
        Py_DECREF(result);
        Py_DECREF(bodo_module);
        throw std::runtime_error("Expected an integer from run_apply_udf");
    }

    int64_t table_info_ptr = PyLong_AsLongLong(result);

    std::shared_ptr<table_info> out_batch(
        reinterpret_cast<table_info*>(table_info_ptr));

    Py_DECREF(bodo_module);
    Py_DECREF(result);

    return out_batch;
}
