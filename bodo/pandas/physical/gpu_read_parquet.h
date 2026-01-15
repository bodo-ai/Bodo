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

struct PhysicalGPUReadParquetMetrics {
    using stat_t = MetricBase::StatValue;
    using time_t = MetricBase::TimerValue;

    stat_t rows_read = 0;
    time_t init_time = 0;
    time_t produce_time = 0;
};

/// @brief Physical node for reading Parquet files in pipelines.
class PhysicalGPUReadParquet : public PhysicalGPUSource {
   private:
    std::shared_ptr<bodo::Schema> output_schema;

    JoinFilterColStats join_filter_col_stats;

    PyObject *py_path;
    PyObject *pyarrow_schema;
    PyObject *storage_options;
    PyObject *schema_fields;
    const std::vector<int> selected_columns;
    duckdb::unique_ptr<duckdb::TableFilterSet> filter_exprs;
    int64_t total_rows_to_read = -1;  // Default to read everything.

   public:
    // TODO: Fill in the contents with info from the logical operator
    explicit PhysicalGPUReadParquet(
        PyObject *py_path, PyObject *pyarrow_schema, PyObject *storage_options,
        std::vector<int> &selected_columns,
        duckdb::TableFilterSet &filter_exprs,
        duckdb::unique_ptr<duckdb::BoundLimitNode> &limit_val,
        JoinFilterColStats join_filter_col_stats)
        : join_filter_col_stats(std::move(join_filter_col_stats)),
          py_path(py_path),
          pyarrow_schema(pyarrow_schema),
          storage_options(storage_options),
          selected_columns(selected_columns),
          filter_exprs(filter_exprs.Copy()) {
        time_pt start_init = start_timer();

        Py_INCREF(this->py_path);
        Py_INCREF(this->pyarrow_schema);
        Py_INCREF(this->storage_options);

        // Extract metadata from pyarrow schema (for Pandas Index reconstruction
        // of dataframe later)
        std::shared_ptr<arrow::Schema> arrow_schema =
            unwrap_schema(pyarrow_schema);
        this->out_metadata =
            std::make_shared<TableMetadata>(arrow_schema->metadata()->keys(),
                                            arrow_schema->metadata()->values());
        this->output_schema = bodo::Schema::FromArrowSchema(arrow_schema)
                                  ->Project(selected_columns);

        this->schema_fields = PyObject_GetAttrString(pyarrow_schema, "names");
        if (!this->schema_fields || !PyList_Check(this->schema_fields)) {
            throw std::runtime_error(
                "PhysicalGPUReadParquet(): failed to get schema fields from "
                "pyarrow schema");
        }

        // ----------------------------------------------------------
        // Handle limit.
        // ----------------------------------------------------------
        if (limit_val) {
            // If the limit option is present...
            if (limit_val->Type() != duckdb::LimitNodeType::CONSTANT_VALUE) {
                throw std::runtime_error(
                    "PhysicalGPUReadParquet unsupported limit type");
            }
            // Limit the rows to read to the limit value.
            total_rows_to_read = limit_val->GetConstantValue();
        }

        // Extract column names from pyarrow schema using selected columns
        int num_fields = PyList_Size(this->schema_fields);
        out_column_names.reserve(selected_columns.size());

        for (int col_idx : selected_columns) {
            if (!(col_idx >= 0 && col_idx < num_fields)) {
                throw std::runtime_error(
                    "PhysicalGPUReadParquet(): invalid column index " +
                    std::to_string(col_idx) + " for schema with " +
                    std::to_string(num_fields) + " fields");
            }
            PyObject *name = PyList_GetItem(this->schema_fields, col_idx);
            if (name && PyUnicode_Check(name)) {
                out_column_names.emplace_back(PyUnicode_AsUTF8(name));
            } else {
                out_column_names.push_back("column_" + std::to_string(col_idx));
            }
        }

        this->metrics.init_time += end_timer(start_init);
    }
    virtual ~PhysicalGPUReadParquet() {
        Py_XDECREF(this->py_path);
        Py_XDECREF(this->pyarrow_schema);
        Py_XDECREF(this->storage_options);
        Py_XDECREF(this->schema_fields);
        Py_XDECREF(grp_mod);
        Py_XDECREF(batch_gen);
    }

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

    std::pair<GPU_DATA, OperatorResult> ProduceBatch() override {
        bool is_last;

        if (!batch_gen) {
            time_pt start_init = start_timer();
            init_batch_gen();
            this->metrics.init_time += end_timer(start_init);
        }

        time_pt start_produce = start_timer();

        PyObject *next_batch_tup = PyObject_CallMethod(
            grp_mod, "get_next_batch", "O", this->batch_gen);
        if (!next_batch_tup) {
            throw std::runtime_error("python");
        }
        PyObject *item0 =
            PyTuple_GetItem(next_batch_tup, 0);  // borrowed reference
        PyObject *item1 =
            PyTuple_GetItem(next_batch_tup, 1);  // borrowed reference
        is_last = PyLong_AsLongLong(item1);

        auto result = is_last ? OperatorResult::FINISHED
                              : OperatorResult::HAVE_MORE_OUTPUT;

        std::pair<GPU_DATA, OperatorResult> ret =
            std::make_pair(GPU_DATA(item0), result);
        Py_DECREF(next_batch_tup);
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
    PhysicalGPUReadParquetMetrics metrics;
    PyObject *batch_gen = nullptr;
    PyObject *grp_mod = nullptr;

    void ReportMetrics(std::vector<MetricBase> &metrics_out) {
        metrics_out.emplace_back(
            TimerMetric("produce_time", this->metrics.produce_time));
    }

    void init_batch_gen() {
        auto batch_size =
            get_streaming_batch_size();  // TO-DO different for GPU

        grp_mod =
            PyImport_ImportModule("bodo.pandas.physical.gpu_read_parquet");
        if (!grp_mod) {
            throw std::runtime_error("python import error");
        }
        batch_gen = PyObject_CallMethod(grp_mod, "get_rank_batch_generator",
                                        "OL", this->py_path, batch_size);
        if (batch_gen == nullptr && PyErr_Occurred()) {
            throw std::runtime_error("python");
        }
        if (PyErr_Occurred()) {
            throw std::runtime_error("python");
        }
    }
};
