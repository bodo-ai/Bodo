#pragma once

#include <Python.h>
#include <arrow/compute/api.h>
#include <arrow/python/pyarrow.h>
#include <memory>
#include <utility>
#include "../io/parquet_reader.h"
#include "_duckdb_util.h"
#include "operator.h"

// When generating the PyObject filter expression for passing to the internal
// read parquet function, we can either generate Python objects directly or
// generate C++ arrow::compute::Expression and then as a last step convert that
// to Python.  The latter approach is cleaner and preferred but the convert to
// Python function is segfaulting now.  Once that is no longer segfaulting then
// permanently remove the Python version of the code below.
#define GEN_PYTHON

#ifndef GEN_PYTHON

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

#else

std::string expressionTypeToArrowCompute(duckdb::ExpressionType &expr_type) {
    switch (expr_type) {
        case duckdb::ExpressionType::COMPARE_EQUAL:
            return "equal";
        case duckdb::ExpressionType::COMPARE_NOTEQUAL:
            return "not_equal";
        case duckdb::ExpressionType::COMPARE_GREATERTHAN:
            return "greater";
        case duckdb::ExpressionType::COMPARE_LESSTHAN:
            return "less";
        case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
            return "greater_equal";
        case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
            return "less_equal";
        default:
            throw std::runtime_error("Unhandled comparison expression type.");
    }
}

PyObject *valueToPyObject(const duckdb::Value &value) {
    duckdb::LogicalTypeId type = value.type().id();
    switch (type) {
        case duckdb::LogicalTypeId::TINYINT:
            return PyLong_FromLong(value.GetValue<int8_t>());
        case duckdb::LogicalTypeId::SMALLINT:
            return PyLong_FromLong(value.GetValue<int16_t>());
        case duckdb::LogicalTypeId::INTEGER:
            return PyLong_FromLong(value.GetValue<int32_t>());
        case duckdb::LogicalTypeId::BIGINT:
            return PyLong_FromLong(value.GetValue<int64_t>());
        case duckdb::LogicalTypeId::FLOAT:
            return PyFloat_FromDouble(value.GetValue<float>());
        case duckdb::LogicalTypeId::DOUBLE:
            return PyFloat_FromDouble(value.GetValue<double>());
        default:
            throw std::runtime_error("valueToPyObject unhandled type" +
                                     std::to_string(static_cast<int>(type)));
    }
}

#endif

PyObject *tableFilterSetToArrowCompute(duckdb::TableFilterSet &filters,
                                       PyObject *schema_fields) {
    PyObject *ret = Py_None;
    if (filters.filters.size() == 0) {
        return ret;
    }
#ifdef GEN_PYTHON
    std::vector<PyObject *> to_be_freed;

    // Import pyarrow.compute
    PyObject *pyarrow_compute = PyImport_ImportModule("pyarrow.compute");
    if (!pyarrow_compute) {
        throw std::runtime_error("Failed to import pyarrow.compute");
    }
    to_be_freed.push_back(pyarrow_compute);
#endif

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
#ifndef GEN_PYTHON
                auto column_ref = arrow::compute::field_ref(
                    PyUnicode_AsUTF8(py_selected_field));
                auto scalar_val = std::visit(
                    [](const auto &value) {
                        return arrow::compute::literal(value);
                    },
                    extractValue(constantFilter->constant));
                auto expr = expressionTypeToArrowCompute(
                    constantFilter->comparison_type)(column_ref, scalar_val);

                ret = pyarrow_wrap_expression(expr);
#else
                PyObject *field_func =
                    PyObject_GetAttrString(pyarrow_compute, "field");
                PyObject *scalar_func =
                    PyObject_GetAttrString(pyarrow_compute, "scalar");
                PyObject *comp_func = PyObject_GetAttrString(
                    pyarrow_compute, expressionTypeToArrowCompute(
                                         constantFilter->comparison_type)
                                         .c_str());
                if (!field_func || !scalar_func || !comp_func) {
                    throw std::runtime_error(
                        "tableFilterSetToPhysicalExpression failed to create "
                        "field, scalar, or comp.");
                }
                to_be_freed.push_back(field_func);
                to_be_freed.push_back(scalar_func);
                to_be_freed.push_back(comp_func);

                // Create "column_name" field object
                PyObject *field_obj = PyObject_CallFunctionObjArgs(
                    field_func, PyList_GetItem(schema_fields, tf.first), NULL);
                if (!field_obj) {
                    throw std::runtime_error(
                        "tableFilterSetToPhysicalExpression failed to create "
                        "field_obj.");
                }
                PyObject *scalar_obj = PyObject_CallFunctionObjArgs(
                    scalar_func, valueToPyObject(constantFilter->constant),
                    NULL);
                if (!scalar_obj) {
                    throw std::runtime_error(
                        "tableFilterSetToPhysicalExpression failed to create "
                        "scalar_obj.");
                }
                to_be_freed.push_back(field_obj);
                to_be_freed.push_back(scalar_obj);

                // Create final arrow.compute expression
                PyObject *expr_obj = PyObject_CallFunctionObjArgs(
                    comp_func, field_obj, scalar_obj, NULL);

                if (!expr_obj) {
                    throw std::runtime_error(
                        "tableFilterSetToPhysicalExpression failed to create "
                        "expression.");
                }
                ret = expr_obj;
#endif
            } break;
            default:
                throw std::runtime_error(
                    "tableFilterSetToPhysicalExpression unsupported filter "
                    "type " +
                    std::to_string(static_cast<int>(tf.second->filter_type)));
        }
    }

#ifdef GEN_PYTHON
    // Clean up Python objects
    for (auto &pyo : to_be_freed) {
        Py_DECREF(pyo);
    }
#endif

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
        return std::make_pair(std::shared_ptr<table_info>(batch), result);
    }

    std::vector<std::string> out_column_names;
};
