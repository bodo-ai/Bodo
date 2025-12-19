#pragma once

#include <Python.h>
#include <arrow/util/key_value_metadata.h>
#include <memory>
#include <utility>
#include "../_util.h"
#include "../io/parquet_reader.h"
#include "duckdb/planner/bound_result_modifier.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "operator.h"

struct PhysicalReadParquetMetrics {
    using stat_t = MetricBase::StatValue;
    using time_t = MetricBase::TimerValue;

    stat_t rows_read = 0;
    time_t init_time = 0;
    time_t produce_time = 0;
};

/// @brief Physical node for reading Parquet files in pipelines.
class PhysicalReadParquet : public PhysicalSource {
   private:
    std::shared_ptr<ParquetReader> internal_reader;
    std::shared_ptr<bodo::Schema> output_schema;

   public:
    // TODO: Fill in the contents with info from the logical operator
    explicit PhysicalReadParquet(
        PyObject *py_path, PyObject *pyarrow_schema, PyObject *storage_options,
        std::vector<int> &selected_columns,
        duckdb::TableFilterSet &filter_exprs,
        duckdb::unique_ptr<duckdb::BoundLimitNode> &limit_val,
        JoinFilterColStats join_filter_col_stats) {
        time_pt start_init = start_timer();
        // ----------------------------------------------------------
        // Handle columns.
        // ----------------------------------------------------------
        std::vector<bool> is_nullable(selected_columns.size(), true);

        // Extract metadata from pyarrow schema (for Pandas Index reconstruction
        // of dataframe later)
        std::shared_ptr<arrow::Schema> arrow_schema =
            unwrap_schema(pyarrow_schema);
        this->out_metadata =
            std::make_shared<TableMetadata>(arrow_schema->metadata()->keys(),
                                            arrow_schema->metadata()->values());
        this->output_schema = bodo::Schema::FromArrowSchema(arrow_schema)
                                  ->Project(selected_columns);

        PyObject *schema_fields =
            PyObject_GetAttrString(pyarrow_schema, "names");
        if (!schema_fields || !PyList_Check(schema_fields)) {
            throw std::runtime_error(
                "PhysicalReadParquet(): failed to get schema fields from "
                "pyarrow schema");
        }

        // ----------------------------------------------------------
        // Handle filter expressions.
        // ----------------------------------------------------------
        PyObject *arrowFilterExpr =
            tableFilterSetToArrowCompute(filter_exprs, schema_fields);

        // ----------------------------------------------------------
        // Handle limit.
        // ----------------------------------------------------------
        int64_t total_rows_to_read = -1;  // Default to read everything.
        if (limit_val) {
            // If the limit option is present...
            if (limit_val->Type() != duckdb::LimitNodeType::CONSTANT_VALUE) {
                throw std::runtime_error(
                    "PhysicalReadParquet unsupported limit type");
            }
            // Limit the rows to read to the limit value.
            total_rows_to_read = limit_val->GetConstantValue();
        }

        // ----------------------------------------------------------
        // Configure internal parquet reader.
        // ----------------------------------------------------------
        internal_reader = std::make_shared<ParquetReader>(
            py_path, true, arrowFilterExpr, storage_options, pyarrow_schema,
            total_rows_to_read, selected_columns, is_nullable, false,
            get_streaming_batch_size());
        internal_reader->init_pq_reader({}, nullptr, nullptr, 0);

        // Extract column names from pyarrow schema using selected columns
        int num_fields = PyList_Size(schema_fields);
        out_column_names.reserve(selected_columns.size());

        for (int col_idx : selected_columns) {
            if (!(col_idx >= 0 && col_idx < num_fields)) {
                throw std::runtime_error(
                    "PhysicalReadParquet(): invalid column index " +
                    std::to_string(col_idx) + " for schema with " +
                    std::to_string(num_fields) + " fields");
            }
            PyObject *name = PyList_GetItem(schema_fields, col_idx);
            if (name && PyUnicode_Check(name)) {
                out_column_names.emplace_back(PyUnicode_AsUTF8(name));
            } else {
                out_column_names.push_back("column_" + std::to_string(col_idx));
            }
        }

        Py_DECREF(schema_fields);
        this->metrics.init_time += end_timer(start_init);
    }
    virtual ~PhysicalReadParquet() = default;

    void FinalizeSource() override {
        std::vector<MetricBase> metrics_out;
        this->ReportMetrics(metrics_out);
        QueryProfileCollector::Default().SubmitOperatorName(getOpId(),
                                                            ToString());
        QueryProfileCollector::Default().SubmitOperatorStageTime(
            QueryProfileCollector::MakeOperatorStageID(getOpId(), 0),
            this->metrics.init_time);
        QueryProfileCollector::Default().SubmitOperatorStageTime(
            QueryProfileCollector::MakeOperatorStageID(getOpId(), 1),
            this->metrics.produce_time);
        QueryProfileCollector::Default().RegisterOperatorStageMetrics(
            QueryProfileCollector::MakeOperatorStageID(getOpId(), 1),
            std::move(metrics_out));
        QueryProfileCollector::Default().SubmitOperatorStageRowCounts(
            QueryProfileCollector::MakeOperatorStageID(getOpId(), 1),
            this->metrics.rows_read);
    }

    std::pair<std::shared_ptr<table_info>, OperatorResult> ProduceBatch()
        override {
        uint64_t total_rows;
        bool is_last;

        time_pt start_produce = start_timer();
        table_info *batch =
            internal_reader->read_batch(is_last, total_rows, true);
        auto result = is_last ? OperatorResult::FINISHED
                              : OperatorResult::HAVE_MORE_OUTPUT;

        batch->column_names = out_column_names;
        batch->metadata = out_metadata;
        auto ret = std::make_pair(std::shared_ptr<table_info>(batch), result);
        this->metrics.rows_read += total_rows;
        this->metrics.produce_time += end_timer(start_produce);
        return ret;
    }

    /**
     * @brief Get the physical schema of the Parquet data
     *
     * @return std::shared_ptr<bodo::Schema> physical schema
     */
    const std::shared_ptr<bodo::Schema> getOutputSchema() override {
        return output_schema;
    }

    // Column names and metadata (Pandas Index info) used for dataframe
    // construction
    std::shared_ptr<TableMetadata> out_metadata;
    std::vector<std::string> out_column_names;

   private:
    PhysicalReadParquetMetrics metrics;

    void ReportMetrics(std::vector<MetricBase> &metrics_out) {
        metrics_out.emplace_back(
            TimerMetric("produce_time", this->metrics.produce_time));
    }
};
