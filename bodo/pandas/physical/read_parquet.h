#pragma once

#include <Python.h>
#include <arrow/compute/api.h>
#include <arrow/python/pyarrow.h>
#include <memory>
#include <utility>
#include "../io/arrow_compat.h"
#include "../io/parquet_reader.h"
#include "_duckdb_util.h"
#include "arrow/util/key_value_metadata.h"
#include "operator.h"

std::function<arrow::compute::Expression(arrow::compute::Expression,
                                         arrow::compute::Expression)>
expressionTypeToArrowCompute(const duckdb::ExpressionType &expr_type) {
    switch (expr_type) {
        case duckdb::ExpressionType::COMPARE_EQUAL:
            return arrow::compute::equal;
        case duckdb::ExpressionType::COMPARE_NOTEQUAL:
            return arrow::compute::not_equal;
        case duckdb::ExpressionType::COMPARE_GREATERTHAN:
            return arrow::compute::greater;
        case duckdb::ExpressionType::COMPARE_LESSTHAN:
            return arrow::compute::less;
        case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
            return arrow::compute::greater_equal;
        case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
            return arrow::compute::less_equal;
        default:
            throw std::runtime_error("Unhandled comparison expression type.");
    }
}

PyObject *tableFilterSetToArrowCompute(duckdb::TableFilterSet &filters,
                                       PyObject *schema_fields) {
    PyObject *ret = Py_None;
    if (filters.filters.size() == 0) {
        return ret;
    }
    arrow::py::import_pyarrow_wrappers();
    std::vector<PyObject *> to_be_freed;
    std::vector<arrow::compute::Expression> parts;

    for (auto &tf : filters.filters) {
        switch (tf.second->filter_type) {
            case duckdb::TableFilterType::CONSTANT_COMPARISON: {
                duckdb::unique_ptr<duckdb::ConstantFilter> constantFilter =
                    dynamic_cast_unique_ptr<duckdb::ConstantFilter>(
                        std::move(tf.second));
                PyObject *py_selected_field =
                    PyList_GetItem(schema_fields, tf.first);
                if (!PyUnicode_Check(py_selected_field)) {
                    throw std::runtime_error(
                        "tableFilterSetToArrowCompute selected field is "
                        "not unicode object.");
                }
                auto column_ref = arrow::compute::field_ref(
                    PyUnicode_AsUTF8(py_selected_field));
                auto scalar_val = std::visit(
                    [](const auto &value) {
                        return arrow::compute::literal(value);
                    },
                    extractValue(constantFilter->constant));
                auto expr = expressionTypeToArrowCompute(
                    constantFilter->comparison_type)(column_ref, scalar_val);
                parts.push_back(expr);
            } break;
            default:
                throw std::runtime_error(
                    "tableFilterSetToArrowCompute unsupported filter "
                    "type " +
                    std::to_string(static_cast<int>(tf.second->filter_type)));
        }
    }

    arrow::compute::Expression whole = arrow::compute::and_(parts);
    ret = arrow::py::wrap_expression(whole);

    // Clean up Python objects
    for (auto &pyo : to_be_freed) {
        Py_DECREF(pyo);
    }

    return ret;
}

/// @brief Physical node for reading Parquet files in pipelines.
class PhysicalReadParquet : public PhysicalSource {
   private:
    std::shared_ptr<ParquetReader> internal_reader;
    std::shared_ptr<arrow::Schema> arrow_schema;

   public:
    // TODO: Fill in the contents with info from the logical operator
    explicit PhysicalReadParquet(
        PyObject *py_path, PyObject *pyarrow_schema, PyObject *storage_options,
        std::vector<int> &selected_columns,
        duckdb::TableFilterSet &filter_exprs,
        duckdb::unique_ptr<duckdb::BoundLimitNode> &limit_val) {
        // ----------------------------------------------------------
        // Handle columns.
        // ----------------------------------------------------------
        std::vector<bool> is_nullable(selected_columns.size(), true);

        // Extract metadata from pyarrow schema (for Pandas Index reconstruction
        // of dataframe later)
        this->arrow_schema = unwrap_schema(pyarrow_schema);
        this->out_metadata = std::make_shared<TableMetadata>(
            this->arrow_schema->metadata()->keys(),
            this->arrow_schema->metadata()->values());

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
                out_column_names.push_back(PyUnicode_AsUTF8(name));
            } else {
                out_column_names.push_back("column_" + std::to_string(col_idx));
            }
        }

        Py_DECREF(schema_fields);
    }
    virtual ~PhysicalReadParquet() = default;

    void Finalize() override {}

    std::pair<std::shared_ptr<table_info>, ProducerResult> ProduceBatch()
        override {
        uint64_t total_rows;
        bool is_last;

        table_info *batch =
            internal_reader->read_batch(is_last, total_rows, true);
        auto result = is_last ? ProducerResult::FINISHED
                              : ProducerResult::HAVE_MORE_OUTPUT;

        batch->column_names = out_column_names;
        batch->metadata = out_metadata;
        return std::make_pair(std::shared_ptr<table_info>(batch), result);
    }

    std::shared_ptr<bodo::Schema> getOutputSchema() override {
        return bodo::Schema::FromArrowSchema(this->arrow_schema);
    }

    // Column names and metadata (Pandas Index info) used for dataframe
    // construction
    std::shared_ptr<TableMetadata> out_metadata;
    std::vector<std::string> out_column_names;
};
