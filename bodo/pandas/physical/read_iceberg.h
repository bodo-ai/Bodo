#pragma once

#include <Python.h>
#include <arrow/compute/api.h>
#include <arrow/python/pyarrow.h>
#include <memory>
#include <utility>
#include "../../io/arrow_reader.h"
#include "duckdb/planner/bound_result_modifier.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "operator.h"

/// @brief Physical node for reading Parquet files in pipelines.
class PhysicalReadIceberg : public PhysicalSource {
   private:
    const std::shared_ptr<arrow::Schema> arrow_schema;
    std::shared_ptr<IcebergParquetReader> internal_reader;

   public:
    explicit PhysicalReadIceberg(
        std::shared_ptr<arrow::Schema> arrow_schema,
        std::vector<int> &selected_columns,
        duckdb::TableFilterSet &filter_exprs,
        duckdb::unique_ptr<duckdb::BoundLimitNode> &limit_val);
    virtual ~PhysicalReadIceberg() = default;

    void Finalize() override {}

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
    const std::shared_ptr<TableMetadata> out_metadata;
    const std::vector<std::string> out_column_names;
};
