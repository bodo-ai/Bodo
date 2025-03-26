
// Defines Bodo data structures and functionality for DuckDB plans

#pragma once

#include "duckdb/function/function.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/optimizer/optimizer.hpp"

/**
 * @brief Bodo's DuckDB TableFunction for reading Parquet datasets with Bodo
 * metadata (used in LogicalGet).
 *
 */
class BodoParquetScanFunction : public duckdb::TableFunction {
   public:
    BodoParquetScanFunction()
        : TableFunction("bodo_read_parquet", {}, nullptr, nullptr, nullptr,
                        nullptr) {
        filter_pushdown = true;
        filter_prune = true;
        projection_pushdown = true;
        // TODO: set statistics and other optimization flags as needed
        // https://github.com/duckdb/duckdb/blob/d29a92f371179170688b4df394478f389bf7d1a6/src/include/duckdb/function/table_function.hpp#L357
    }
};

/**
 * @brief Data for Bodo's DuckDB TableFunction for reading Parquet datasets.
 *
 */
class BodoParquetScanFunctionData : public duckdb::TableFunctionData {
   public:
    BodoParquetScanFunctionData(std::string path) : path(path) {}
    // Parquet dataset path
    std::string path;
};

/**
 * @brief Optimize a DuckDB logical plan by applying the DuckDB optimizer.
 *
 * @param plan input logical plan to be optimized
 * @return duckdb::unique_ptr<duckdb::LogicalOperator> optimized plan
 */
duckdb::unique_ptr<duckdb::LogicalOperator> optimize_plan(
    duckdb::unique_ptr<duckdb::LogicalOperator> plan);

/**
 * @brief Creates a LogicalGet node for reading a Parquet dataset in DuckDB with
 * Bodo metadata.
 *
 * @param parquet_path path to the Parquet dataset
 * @return duckdb::unique_ptr<duckdb::LogicalGet> output node
 */
duckdb::unique_ptr<duckdb::LogicalGet> make_parquet_get_node(
    std::string parquet_path);

/**
 * @brief Returns a statically created DuckDB client context.
 *
 * @return duckdb::ClientContext& static context object
 */
duckdb::ClientContext& get_duckdb_context();

/**
 * @brief Returns a statically created DuckDB binder.
 *
 * @return duckdb::shared_ptr<duckdb::Binder> static binder object
 */
duckdb::shared_ptr<duckdb::Binder> get_duckdb_binder();

/**
 * @brief Returns a statically created duckdb optimizer
 *
 * @return duckdb::Optimizer& static optimizer object
 */
duckdb::Optimizer& get_duckdb_optimizer();
