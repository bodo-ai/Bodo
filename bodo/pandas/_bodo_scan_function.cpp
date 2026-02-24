#include "_bodo_scan_function.h"

#include "../libs/_utils.h"
#include "physical/read_iceberg.h"
#include "physical/read_pandas.h"
#include "physical/read_parquet.h"
#if USE_CUDF
#include "physical/gpu_read_parquet.h"
#endif

#include <fmt/format.h>

PhysicalCpuGpuSource
BodoDataFrameParallelScanFunctionData::CreatePhysicalOperator(
    std::vector<int> &selected_columns, duckdb::TableFilterSet &filter_exprs,
    duckdb::unique_ptr<duckdb::BoundLimitNode> &limit_val,
    std::shared_ptr<std::unordered_map<int, join_state_t>> join_filter_states,
    bool run_on_gpu) {
    // Read the dataframe from the result registry using
    // sys.modules["__main__"].RESULT_REGISTRY since importing
    // bodo.spawn.worker creates a new module with new empty registry.

    // Import Python sys module
    PyObjectPtr sys_module = PyImport_ImportModule("sys");
    if (!sys_module) {
        throw std::runtime_error("Failed to import sys module");
    }

    // Get sys.modules dictionary
    PyObjectPtr modules_dict = PyObject_GetAttrString(sys_module, "modules");
    if (!modules_dict) {
        Py_DECREF(sys_module);
        throw std::runtime_error("Failed to get sys.modules");
    }

    // Get __main__ module
    PyObject *main_module = PyDict_GetItemString(modules_dict, "__main__");
    if (!main_module) {
        throw std::runtime_error("Failed to get __main__ module");
    }

    // Get RESULT_REGISTRY[result_id]
    PyObjectPtr result_registry =
        PyObject_GetAttrString(main_module, "RESULT_REGISTRY");
    PyObject *df = PyDict_GetItemString(result_registry, result_id.c_str());
    if (!df) {
        throw std::runtime_error(fmt::format(
            "Result ID {} not found in result registry", result_id.c_str()));
    }

    return std::make_shared<PhysicalReadPandas>(df, selected_columns,
                                                this->arrow_schema);
}

PhysicalCpuGpuSource BodoDataFrameSeqScanFunctionData::CreatePhysicalOperator(
    std::vector<int> &selected_columns, duckdb::TableFilterSet &filter_exprs,
    duckdb::unique_ptr<duckdb::BoundLimitNode> &limit_val,
    std::shared_ptr<std::unordered_map<int, join_state_t>> join_filter_states,
    bool run_on_gpu) {
    return std::make_shared<PhysicalReadPandas>(df, selected_columns,
                                                this->arrow_schema);
}

PhysicalCpuGpuSource BodoParquetScanFunctionData::CreatePhysicalOperator(
    std::vector<int> &selected_columns, duckdb::TableFilterSet &filter_exprs,
    duckdb::unique_ptr<duckdb::BoundLimitNode> &limit_val,
    std::shared_ptr<std::unordered_map<int, join_state_t>> join_filter_states,
    bool run_on_gpu) {
    JoinFilterColStats join_filter_col_stats =
        this->rtjf_state_map.has_value()
            ? JoinFilterColStats(join_filter_states,
                                 this->rtjf_state_map.value())
            : JoinFilterColStats();
#ifdef USE_CUDF
    if (run_on_gpu) {
        return std::make_shared<PhysicalGPUReadParquet>(
            path, pyarrow_schema, storage_options, selected_columns,
            filter_exprs, limit_val, join_filter_col_stats);
    }
#endif
    return std::make_shared<PhysicalReadParquet>(
        path, pyarrow_schema, storage_options, selected_columns, filter_exprs,
        limit_val, join_filter_col_stats);
}

PhysicalCpuGpuSource BodoIcebergScanFunctionData::CreatePhysicalOperator(
    std::vector<int> &selected_columns, duckdb::TableFilterSet &filter_exprs,
    duckdb::unique_ptr<duckdb::BoundLimitNode> &limit_val,
    std::shared_ptr<std::unordered_map<int, join_state_t>> join_filter_states,
    bool run_on_gpu) {
    JoinFilterColStats join_filter_col_stats =
        this->rtjf_state_map.has_value()
            ? JoinFilterColStats(join_filter_states,
                                 this->rtjf_state_map.value())
            : JoinFilterColStats();
    return std::make_shared<PhysicalReadIceberg>(
        this->catalog, this->table_id, this->iceberg_filter,
        this->iceberg_schema, this->arrow_schema, this->snapshot_id,
        selected_columns, filter_exprs, limit_val, join_filter_col_stats);
}
