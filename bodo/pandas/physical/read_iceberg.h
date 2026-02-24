#pragma once

#include <Python.h>
#include <arrow/compute/api.h>
#include <arrow/python/pyarrow.h>
#include <arrow/type.h>
#include <memory>
#include <utility>
#include "../../io/iceberg_parquet_reader.h"
#include "../_util.h"
#include "duckdb/planner/bound_result_modifier.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "operator.h"

struct PhysicalReadIcebergMetrics {
    using stat_t = MetricBase::StatValue;
    using time_t = MetricBase::TimerValue;

    stat_t rows_read = 0;
    time_t init_time = 0;
    time_t produce_time = 0;
};

/// @brief Physical node for reading Parquet files in pipelines.
class PhysicalReadIceberg : public PhysicalSource {
   private:
    PyObject *catalog;
    const std::string table_id;
    PyObject *iceberg_filter;
    PyObject *iceberg_schema;
    const int64_t snapshot_id;
    duckdb::unique_ptr<duckdb::TableFilterSet> filter_exprs;
    const std::shared_ptr<arrow::Schema> arrow_schema;
    const std::vector<int> selected_columns;
    int64_t total_rows_to_read = -1;  // Default to read everything.

    const std::shared_ptr<arrow::Schema> out_arrow_schema;
    std::unique_ptr<IcebergParquetReader> internal_reader;
    JoinFilterColStats join_filter_col_stats;
    PhysicalReadIcebergMetrics metrics;

    static std::vector<std::string> create_out_column_names(
        const std::vector<int> &selected_columns,
        const std::shared_ptr<arrow::Schema> schema);

    std::unique_ptr<IcebergParquetReader> create_internal_reader();

    static std::shared_ptr<arrow::Schema> create_out_arrow_schema(
        std::shared_ptr<arrow::Schema> arrow_schema,
        const std::vector<int> &selected_columns);

   public:
    explicit PhysicalReadIceberg(
        PyObject *catalog, const std::string table_id, PyObject *iceberg_filter,
        PyObject *iceberg_schema,
        const std::shared_ptr<arrow::Schema> arrow_schema,
        const int64_t snapshot_id, const std::vector<int> &selected_columns,
        duckdb::TableFilterSet &filter_exprs,
        duckdb::unique_ptr<duckdb::BoundLimitNode> &limit_val,
        JoinFilterColStats join_filter_col_stats);
    virtual ~PhysicalReadIceberg() {
        Py_XDECREF(this->catalog);
        Py_XDECREF(this->iceberg_filter);
        Py_XDECREF(this->iceberg_schema);
    }

    void FinalizeSource() override;

    std::pair<std::shared_ptr<table_info>, OperatorResult> ProduceBatch()
        override;

    /**
     * @brief Get the physical schema of the Iceberg data
     *
     * @return std::shared_ptr<bodo::Schema> physical schema
     */
    const std::shared_ptr<bodo::Schema> getOutputSchema() override;

    // Column names and metadata (Pandas Index info) used for dataframe
    // construction
    const std::shared_ptr<bodo::TableMetadata> out_metadata;
    const std::vector<std::string> out_column_names;
};
