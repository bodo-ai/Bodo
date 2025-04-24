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

    if (filters.filters.size() > 1) {
        throw std::runtime_error(
            "tableFilterSetToPhysicalExpression currently supports only a "
            "single filter.");
    }

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
                        "tableFilterSetToPhysicalExpression selected field is "
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

                ret = arrow::py::wrap_expression(expr);
            } break;
            default:
                throw std::runtime_error(
                    "tableFilterSetToPhysicalExpression unsupported filter "
                    "type " +
                    std::to_string(static_cast<int>(tf.second->filter_type)));
        }
    }

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

   public:
    // TODO: Fill in the contents with info from the logical operator
    explicit PhysicalReadParquet(std::string _path, PyObject *pyarrow_schema,
                                 PyObject *storage_options,
                                 std::vector<int> &selected_columns,
                                 duckdb::TableFilterSet &filter_exprs) {
        PyObject *py_path = PyUnicode_FromString(_path.c_str());

        std::vector<bool> is_nullable(selected_columns.size(), true);

        // Extract metadata from pyarrow schema (for Pandas Index reconstruction
        // of dataframe later)
        std::shared_ptr<arrow::Schema> schema = unwrap_schema(pyarrow_schema);
        this->out_metadata = std::make_shared<TableMetadata>(
            schema->metadata()->keys(), schema->metadata()->values());

        PyObject *schema_fields =
            PyObject_GetAttrString(pyarrow_schema, "names");
        if (!schema_fields || !PyList_Check(schema_fields)) {
            throw std::runtime_error(
                "PhysicalReadParquet(): failed to get schema fields from "
                "pyarrow schema");
        }

        PyObject *arrowFilterExpr =
            tableFilterSetToArrowCompute(filter_exprs, schema_fields);

        internal_reader = std::make_shared<ParquetReader>(
            py_path, true, arrowFilterExpr, storage_options, pyarrow_schema, -1,
            selected_columns, is_nullable, false, -1);
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

    // Column names and metadata (Pandas Index info) used for dataframe
    // construction
    std::shared_ptr<TableMetadata> out_metadata;
    std::vector<std::string> out_column_names;
};
