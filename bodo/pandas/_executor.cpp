#include "_executor.h"
#include <arrow/python/api.h>
#include <arrow/python/pyarrow.h>
#include <arrow/status.h>
#include <cstdint>
#include <memory>
#include "../io/arrow_compat.h"
#include "../libs/_bodo_to_arrow.h"
#include "../libs/streaming/_shuffle.h"
#include "_plan.h"
#include "arrow/io/api.h"
#include "parquet/arrow/reader.h"

Executor::Executor(std::unique_ptr<duckdb::LogicalOperator> plan) {
    // Convert logical plan to physical plan and create query pipelines

    // TODO: support nodes other than read parquet
    duckdb::LogicalGet& get_plan = plan->Cast<duckdb::LogicalGet>();

    std::shared_ptr<PhysicalOperator> physical_op =
        get_plan.bind_data->Cast<BodoScanFunctionData>()
            .CreatePhysicalOperator();

    pipelines.emplace_back(
        std::vector<std::shared_ptr<PhysicalOperator>>({physical_op}));
}

std::pair<int64_t, PyObject*> Executor::execute() {
    // TODO: support multiple pipelines
    return pipelines[0].execute();
}

std::pair<int64_t, PyObject*> Pipeline::execute() {
    // TODO: support multiple operators
    return operators[0]->execute();
}

std::pair<int64_t, PyObject*> PhysicalReadParquet::execute() {
    // TODO: replace with proper streaming and parallel Parquet read (using
    // Arrow for now)

    arrow::MemoryPool* pool = arrow::default_memory_pool();
    std::shared_ptr<arrow::io::RandomAccessFile> input;
    input = arrow::io::ReadableFile::Open(path).ValueOrDie();

    // Open Parquet file reader
    std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
    arrow_reader = parquet::arrow::OpenFile(input, pool).ValueOrDie();

    // Read entire file as a single Arrow table
    std::shared_ptr<arrow::Table> table;
    if (arrow_reader->ReadTable(&table) != arrow::Status::OK()) {
        throw std::runtime_error("Failed to read Parquet file");
    }

    arrow::py::import_pyarrow_wrappers();
    PyObject* pyarrow_schema = arrow::py::wrap_schema(table->schema());

    auto* bodo_pool = bodo::BufferPool::DefaultPtr();

    std::vector<std::shared_ptr<array_info>> out_arrs;
    out_arrs.reserve(table->num_columns());
    for (int64_t i = 0; i < table->num_columns(); i++) {
        std::shared_ptr<arrow::Array> arr = table->column(i)->chunk(0);
        std::shared_ptr<array_info> out_arr =
            arrow_array_to_bodo(arr, bodo_pool);
        out_arrs.push_back(out_arr);
    }

    return {reinterpret_cast<int64_t>(new table_info(out_arrs)),
            pyarrow_schema};
}

std::pair<int64_t, PyObject*> PhysicalReadPandas::execute() {
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

    // Get Arrow schema for return value
    PyObject* pyarrow_schema = arrow::py::wrap_schema(table->schema());

    // Convert Arrow arrays to Bodo arrays
    auto* bodo_pool = bodo::BufferPool::DefaultPtr();
    std::vector<std::shared_ptr<array_info>> out_arrs;
    out_arrs.reserve(table->num_columns());
    for (int64_t i = 0; i < table->num_columns(); i++) {
        std::shared_ptr<arrow::Array> arr = table->column(i)->chunk(0);
        std::shared_ptr<array_info> out_arr =
            arrow_array_to_bodo(arr, bodo_pool);
        out_arrs.push_back(out_arr);
    }

    // Clean up Python references
    Py_DECREF(iloc);
    Py_DECREF(slice);
    Py_DECREF(batch);
    Py_DECREF(pyarrow_module);
    Py_DECREF(table_func);
    Py_DECREF(pa_table);

    return {reinterpret_cast<int64_t>(new table_info(out_arrs)),
            pyarrow_schema};
}
