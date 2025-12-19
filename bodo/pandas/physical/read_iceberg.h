#pragma once

#include <Python.h>
#include <arrow/compute/api.h>
#include <arrow/python/pyarrow.h>
#include <arrow/type.h>
#include <memory>
#include <utility>
#include "../../io/iceberg_parquet_reader.h"
#include "../libs/_bodo_to_arrow.h"
#include "../libs/streaming/_join.h"
#include "duckdb/planner/bound_result_modifier.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "operator.h"
#include "optimizer/runtime_join_filter.h"

struct PhysicalReadIcebergMetrics {
    using stat_t = MetricBase::StatValue;
    using time_t = MetricBase::TimerValue;

    stat_t rows_read = 0;
    time_t init_time = 0;
    time_t produce_time = 0;
};

class JoinFilterColStats {
    using col_min_max_t = std::pair<std::shared_ptr<arrow::Scalar>,
                                    std::shared_ptr<arrow::Scalar>>;
    struct col_stats_collector {
        int64_t build_key_col;
        JoinState *join_state;
        col_min_max_t collect_min_max() {
            std::unique_ptr<bodo::DataType> dt =
                join_state->build_table_schema->column_types[build_key_col]
                    ->copy();
            const auto &col_min_max = join_state->min_max_values[build_key_col];
            arrow::TimeUnit::type time_unit = arrow::TimeUnit::NANO;
            assert(col_min_max.has_value());
            std::shared_ptr<arrow::Array> arrow_array = bodo_array_to_arrow(
                bodo::BufferPool::DefaultPtr(), col_min_max.value(), false,
                dt->timezone, time_unit, false,
                bodo::default_buffer_memory_manager());

            std::shared_ptr<arrow::Scalar> min_scalar =
                arrow_array->GetScalar(0).ValueOrDie();
            std::shared_ptr<arrow::Scalar> max_scalar =
                arrow_array->GetScalar(1).ValueOrDie();
            return {min_scalar, max_scalar};
        }
    };

    std::unordered_map<int, std::vector<col_stats_collector>>
        join_col_stats_map;

   public:
    JoinFilterColStats(std::unordered_map<int, JoinState *> join_state_map,
                       JoinFilterProgramState rtjf_state_map) {
        for (const auto &[join_id, col_info] : rtjf_state_map) {
            auto join_state_it = join_state_map.find(join_id);
            if (join_state_it == join_state_map.end()) {
                throw std::runtime_error(
                    "JoinFilterColStats: join state not found for join id " +
                    std::to_string(join_id));
            }
            JoinState *join_state = join_state_it->second;
            for (size_t i = 0; i < col_info.filter_columns.size(); ++i) {
                int64_t orig_build_key = col_info.orig_build_key_cols[i];
                join_col_stats_map[join_id].push_back(col_stats_collector{
                    .build_key_col = orig_build_key, .join_state = join_state});
            }
        }
    }
    JoinFilterColStats() = default;
};

/// @brief Physical node for reading Parquet files in pipelines.
class PhysicalReadIceberg : public PhysicalSource {
   private:
    const std::shared_ptr<arrow::Schema> arrow_schema;
    const std::shared_ptr<arrow::Schema> out_arrow_schema;
    std::unique_ptr<IcebergParquetReader> internal_reader;
    JoinFilterColStats join_filter_col_stats;
    PhysicalReadIcebergMetrics metrics;

    static std::vector<std::string> create_out_column_names(
        const std::vector<int> &selected_columns,
        const std::shared_ptr<arrow::Schema> schema);

    static std::unique_ptr<IcebergParquetReader> create_internal_reader(
        PyObject *catalog, const std::string table_id, PyObject *iceberg_filter,
        PyObject *iceberg_schema, int64_t snapshot_id,
        duckdb::TableFilterSet &filter_exprs,
        std::shared_ptr<arrow::Schema> arrow_schema,
        std::vector<int> &selected_columns,
        duckdb::unique_ptr<duckdb::BoundLimitNode> &limit_val, int64_t op_id);

    static std::shared_ptr<arrow::Schema> create_out_arrow_schema(
        std::shared_ptr<arrow::Schema> arrow_schema,
        const std::vector<int> &selected_columns);

   public:
    explicit PhysicalReadIceberg(
        PyObject *catalog, const std::string table_id, PyObject *iceberg_filter,
        PyObject *iceberg_schema, std::shared_ptr<arrow::Schema> arrow_schema,
        int64_t snapshot_id, std::vector<int> &selected_columns,
        duckdb::TableFilterSet &filter_exprs,
        duckdb::unique_ptr<duckdb::BoundLimitNode> &limit_val,
        JoinFilterColStats join_filter_col_stats);
    virtual ~PhysicalReadIceberg() = default;

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
    const std::shared_ptr<TableMetadata> out_metadata;
    const std::vector<std::string> out_column_names;
};
