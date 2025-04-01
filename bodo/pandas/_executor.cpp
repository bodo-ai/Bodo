#include "_executor.h"
#include <arrow/python/api.h>
#include <arrow/python/pyarrow.h>
#include <arrow/status.h>
#include <memory>
#include "../io/arrow_compat.h"
#include "../libs/_bodo_to_arrow.h"
#include "_bodo_plan.h"
#include "arrow/io/api.h"
#include "parquet/arrow/reader.h"

Executor::Executor(std::unique_ptr<duckdb::LogicalOperator> plan) {
    // Convert logical plan to physical plan and create query pipelines

    // TODO: support nodes other than read parquet
    duckdb::LogicalGet& get_plan = plan->Cast<duckdb::LogicalGet>();
    std::string path =
        get_plan.bind_data->Cast<BodoParquetScanFunctionData>().path;
    std::shared_ptr<PhysicalReadParquet> read_parquet =
        std::make_shared<PhysicalReadParquet>(path);

    pipelines.emplace_back(
        std::vector<std::shared_ptr<PhysicalOperator>>({read_parquet}));
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
    for (uint64_t i = 0; i < table->num_columns(); i++) {
        std::shared_ptr<arrow::Array> arr = table->column(i)->chunk(0);
        std::shared_ptr<array_info> out_arr =
            arrow_array_to_bodo(arr, bodo_pool);
        out_arrs.push_back(out_arr);
    }

    return {reinterpret_cast<int64_t>(new table_info(out_arrs)),
            pyarrow_schema};
}
