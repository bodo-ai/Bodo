#pragma once

#include <utility>
#include "_util.h"

#include "duckdb/function/function.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/planner/bound_result_modifier.hpp"
#include "fmt/core.h"
#include "optimizer/runtime_join_filter.h"
#include "physical/operator.h"

/**
 * @brief Superclass for Bodo's DuckDB TableFunction classes.
 *
 */
class BodoScanFunction : public duckdb::TableFunction {
   public:
    BodoScanFunction(std::string name)
        : TableFunction(name, {}, nullptr, nullptr, nullptr, nullptr) {}
};

/**
 * @brief Superclass for Bodo's DuckDB TableFunctionData classes.
 *
 */
class BodoScanFunctionData : public duckdb::TableFunctionData {
   public:
    BodoScanFunctionData() = default;
    /**
     * @brief Create a PhysicalOperator for reading data from this source.
     *
     * @return std::shared_ptr<PhysicalSource> read operator
     */
    virtual PhysicalCpuGpuSource CreatePhysicalOperator(
        std::vector<int> &selected_columns,
        duckdb::TableFilterSet &filter_exprs,
        duckdb::unique_ptr<duckdb::BoundLimitNode> &limit_val,
        std::shared_ptr<std::unordered_map<int, join_state_t>>
            join_filter_states,
        bool run_on_gpu) = 0;

    // This allows pushing runtime join filter state from the optimizer to the
    // physical read operators which can generate filters from join key
    // statistics.
    std::optional<JoinFilterProgramState> rtjf_state_map = std::nullopt;

    virtual bool canRunOnGPU(bool has_filters, bool has_limit) { return false; }
};

/**
 * @brief Bodo's DuckDB TableFunction for reading Parquet datasets with Bodo
 * metadata (used in LogicalGet).
 *
 */
class BodoParquetScanFunction : public BodoScanFunction {
   public:
    BodoParquetScanFunction(const std::shared_ptr<arrow::Schema> arrow_schema)

        : BodoScanFunction(
              fmt::format("bodo_read_parquet({})",
                          schemaColumnNamesToString(arrow_schema))) {
        filter_pushdown = true;
        filter_prune = true;
        projection_pushdown = true;
        limit_pushdown = true;
        // TODO: set statistics and other optimization flags as needed
        // https://github.com/duckdb/duckdb/blob/d29a92f371179170688b4df394478f389bf7d1a6/src/include/duckdb/function/table_function.hpp#L357
    }
};

/**
 * @brief Data for Bodo's DuckDB TableFunction for reading Parquet datasets.
 *
 */
class BodoParquetScanFunctionData : public BodoScanFunctionData {
   public:
    BodoParquetScanFunctionData(PyObject *path, PyObject *pyarrow_schema,
                                PyObject *storage_options)
        : path(path),
          pyarrow_schema(pyarrow_schema),
          storage_options(storage_options) {
        Py_INCREF(pyarrow_schema);
        Py_INCREF(storage_options);
        Py_INCREF(path);
    }

    ~BodoParquetScanFunctionData() override {
        Py_DECREF(pyarrow_schema);
        Py_DECREF(storage_options);
        Py_DECREF(path);
    }

    PhysicalCpuGpuSource CreatePhysicalOperator(
        std::vector<int> &selected_columns,
        duckdb::TableFilterSet &filter_exprs,
        duckdb::unique_ptr<duckdb::BoundLimitNode> &limit_val,
        std::shared_ptr<std::unordered_map<int, join_state_t>>
            join_filter_states,
        bool run_on_gpu) override;

    bool canRunOnGPU(bool has_filters, bool has_limit) override {
        return !has_limit;
    }

    // Parquet dataset path
    PyObject *path;
    PyObject *pyarrow_schema;
    PyObject *storage_options;
};

/**
 * @brief Bodo's DuckDB TableFunction for reading dataframe rows
 * (used in LogicalGet).
 *
 */

class BodoDataFrameScanFunction : public BodoScanFunction {
   public:
    BodoDataFrameScanFunction(const std::shared_ptr<arrow::Schema> arrow_schema)
        : BodoScanFunction(fmt::format(

              "bodo_read_df({})", schemaColumnNamesToString(arrow_schema))) {
        projection_pushdown = true;
        limit_pushdown = false;
    }
};

/**
 * @brief Data for Bodo's DuckDB TableFunction for reading dataframe rows on
 * spawner sequentially.
 *
 */
class BodoDataFrameSeqScanFunctionData : public BodoScanFunctionData {
   public:
    BodoDataFrameSeqScanFunctionData(
        PyObject *df, std::shared_ptr<arrow::Schema> arrow_schema)
        : df(df), arrow_schema(std::move(arrow_schema)) {
        Py_INCREF(df);
    }
    ~BodoDataFrameSeqScanFunctionData() override { Py_DECREF(df); }
    /**
     * @brief Create a PhysicalOperator for reading from the dataframe.
     *
     * @return std::shared_ptr<PhysicalOperator> dataframe read operator
     */
    PhysicalCpuGpuSource CreatePhysicalOperator(
        std::vector<int> &selected_columns,
        duckdb::TableFilterSet &filter_exprs,
        duckdb::unique_ptr<duckdb::BoundLimitNode> &limit_val,
        std::shared_ptr<std::unordered_map<int, join_state_t>>
            join_filter_states,
        bool run_on_gpu) override;

    PyObject *df;
    const std::shared_ptr<arrow::Schema> arrow_schema;
};

/**
 * @brief Data for Bodo's DuckDB TableFunction for reading dataframe rows on
 * workers in parallel.
 *
 */
class BodoDataFrameParallelScanFunctionData : public BodoScanFunctionData {
   public:
    BodoDataFrameParallelScanFunctionData(
        std::string result_id, std::shared_ptr<arrow::Schema> arrow_schema)
        : result_id(std::move(result_id)),
          arrow_schema(std::move(arrow_schema)) {}
    ~BodoDataFrameParallelScanFunctionData() override = default;
    /**
     * @brief Create a PhysicalOperator for reading from the dataframe.
     *
     * @return std::shared_ptr<PhysicalOperator> dataframe read operator
     */
    PhysicalCpuGpuSource CreatePhysicalOperator(
        std::vector<int> &selected_columns,
        duckdb::TableFilterSet &filter_exprs,
        duckdb::unique_ptr<duckdb::BoundLimitNode> &limit_val,
        std::shared_ptr<std::unordered_map<int, join_state_t>>
            join_filter_states,
        bool run_on_gpu) override;
    std::string result_id;
    const std::shared_ptr<arrow::Schema> arrow_schema;
};

/**
 * @brief Bodo's DuckDB TableFunction for reading Iceberg datasets with Bodo
 * metadata (used in LogicalGet).
 *
 */
class BodoIcebergScanFunction : public BodoScanFunction {
   public:
    BodoIcebergScanFunction(const std::shared_ptr<arrow::Schema> arrow_schema)
        : BodoScanFunction(
              fmt::format("bodo_read_iceberg({})",
                          schemaColumnNamesToString(arrow_schema))) {
        filter_pushdown = true;
        filter_prune = true;
        projection_pushdown = true;
        limit_pushdown = true;
        // TODO: set statistics and other optimization flags as needed
        // https://github.com/duckdb/duckdb/blob/d29a92f371179170688b4df394478f389bf7d1a6/src/include/duckdb/function/table_function.hpp#L357
    }
};

/**
 * @brief Data for Bodo's DuckDB TableFunction for reading Iceberg datasets.
 *
 */
class BodoIcebergScanFunctionData : public BodoScanFunctionData {
   public:
    BodoIcebergScanFunctionData(std::shared_ptr<arrow::Schema> _arrow_schema,
                                PyObject *_catalog, const std::string _table_id,
                                PyObject *_iceberg_filter,
                                PyObject *_iceberg_schema, int64_t _snapshot_id)
        : arrow_schema(std::move(_arrow_schema)),
          catalog(_catalog),
          iceberg_filter(_iceberg_filter),
          iceberg_schema(_iceberg_schema),
          table_id(_table_id),
          snapshot_id(_snapshot_id) {
        Py_INCREF(this->catalog);
        Py_INCREF(this->iceberg_filter);
        Py_INCREF(this->iceberg_schema);
    };

    ~BodoIcebergScanFunctionData() override {
        Py_DECREF(this->catalog);
        Py_DECREF(this->iceberg_filter);
        Py_DECREF(this->iceberg_schema);
    };

    PhysicalCpuGpuSource CreatePhysicalOperator(
        std::vector<int> &selected_columns,
        duckdb::TableFilterSet &filter_exprs,
        duckdb::unique_ptr<duckdb::BoundLimitNode> &limit_val,
        std::shared_ptr<std::unordered_map<int, join_state_t>>
            join_filter_states,
        bool run_on_gpu) override;
    const std::shared_ptr<arrow::Schema> arrow_schema;
    PyObject *catalog;
    PyObject *iceberg_filter;
    PyObject *iceberg_schema;
    const std::string table_id;
    const int64_t snapshot_id;
};
