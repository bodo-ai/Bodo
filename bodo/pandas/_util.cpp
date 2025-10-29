#include "_util.h"
#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/datum.h>
#include <arrow/python/pyarrow.h>
#include <arrow/result.h>
#include <arrow/scalar.h>
#include "../io/arrow_compat.h"
#include "../libs/_utils.h"
#include "duckdb/common/types.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/filter/optional_filter.hpp"

std::variant<int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t,
             uint64_t, bool, std::string, float, double,
             std::shared_ptr<arrow::Scalar>>
extractValue(const duckdb::Value &value) {
    duckdb::LogicalTypeId type = value.type().id();
    switch (type) {
        case duckdb::LogicalTypeId::TINYINT:
            return value.GetValue<int8_t>();
        case duckdb::LogicalTypeId::SMALLINT:
            return value.GetValue<int16_t>();
        case duckdb::LogicalTypeId::INTEGER:
            return value.GetValue<int32_t>();
        case duckdb::LogicalTypeId::BIGINT:
            return value.GetValue<int64_t>();
        case duckdb::LogicalTypeId::UTINYINT:
            return value.GetValue<uint8_t>();
        case duckdb::LogicalTypeId::USMALLINT:
            return value.GetValue<uint16_t>();
        case duckdb::LogicalTypeId::UINTEGER:
            return value.GetValue<uint32_t>();
        case duckdb::LogicalTypeId::UBIGINT:
            return value.GetValue<uint64_t>();
        case duckdb::LogicalTypeId::FLOAT:
            return value.GetValue<float>();
        case duckdb::LogicalTypeId::DOUBLE:
            return value.GetValue<double>();
        case duckdb::LogicalTypeId::BOOLEAN:
            return value.GetValue<bool>();
        case duckdb::LogicalTypeId::VARCHAR:
            return value.GetValue<std::string>();
        case duckdb::LogicalTypeId::TIMESTAMP: {
            // Define a timestamp type with microsecond precision
            auto timestamp_type = arrow::timestamp(arrow::TimeUnit::MICRO);
            duckdb::timestamp_t extracted =
                value.GetValue<duckdb::timestamp_t>();
            // Create a TimestampScalar with microsecond value
            return std::make_shared<arrow::TimestampScalar>(extracted.value,
                                                            timestamp_type);
        } break;
        case duckdb::LogicalTypeId::TIMESTAMP_MS: {
            // Define a timestamp type with millisecond precision
            auto timestamp_type = arrow::timestamp(arrow::TimeUnit::MILLI);
            duckdb::timestamp_ms_t extracted =
                value.GetValue<duckdb::timestamp_ms_t>();
            // Create a TimestampScalar with millisecond value
            return std::make_shared<arrow::TimestampScalar>(extracted.value,
                                                            timestamp_type);
        } break;
        case duckdb::LogicalTypeId::TIMESTAMP_SEC: {
            // Define a timestamp type with second precision
            auto timestamp_type = arrow::timestamp(arrow::TimeUnit::SECOND);
            duckdb::timestamp_sec_t extracted =
                value.GetValue<duckdb::timestamp_sec_t>();
            // Create a TimestampScalar with second value
            return std::make_shared<arrow::TimestampScalar>(extracted.value,
                                                            timestamp_type);
        } break;
        case duckdb::LogicalTypeId::TIMESTAMP_NS: {
            // Define a timestamp type with nanosecond precision
            auto timestamp_type = arrow::timestamp(arrow::TimeUnit::NANO);
            duckdb::timestamp_ns_t extracted =
                value.GetValue<duckdb::timestamp_ns_t>();
            // Create a TimestampScalar with nanosecond value
            return std::make_shared<arrow::TimestampScalar>(extracted.value,
                                                            timestamp_type);
        } break;
        case duckdb::LogicalTypeId::DATE: {
            // Define a date type
            auto date_type = arrow::date32();
            duckdb::date_t extracted = value.GetValue<duckdb::date_t>();
            // Create a DateScalar with the date value
            return arrow::MakeScalar(date_type, extracted.days).ValueOrDie();
        } break;
        default:
            throw std::runtime_error("extractValue unhandled type." +
                                     std::to_string(static_cast<int>(type)));
    }
}

std::variant<int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t,
             uint64_t, bool, std::string, float, double,
             std::shared_ptr<arrow::Scalar>>
getDefaultValueForDuckdbValueType(const duckdb::Value &value) {
    duckdb::LogicalTypeId type = value.type().id();
    switch (type) {
        case duckdb::LogicalTypeId::TINYINT:
            return int8_t{};
        case duckdb::LogicalTypeId::SMALLINT:
            return int16_t{};
        case duckdb::LogicalTypeId::INTEGER:
            return int32_t{};
        case duckdb::LogicalTypeId::BIGINT:
            return int64_t{};
        case duckdb::LogicalTypeId::UTINYINT:
            return uint8_t{};
        case duckdb::LogicalTypeId::USMALLINT:
            return uint16_t{};
        case duckdb::LogicalTypeId::UINTEGER:
            return uint32_t{};
        case duckdb::LogicalTypeId::UBIGINT:
            return uint64_t{};
        case duckdb::LogicalTypeId::FLOAT:
            return float{};
        case duckdb::LogicalTypeId::DOUBLE:
            return double{};
        case duckdb::LogicalTypeId::BOOLEAN:
            return bool{};
        case duckdb::LogicalTypeId::VARCHAR:
            return std::string{};
        case duckdb::LogicalTypeId::TIMESTAMP_NS: {
            // Define a timestamp type with nanosecond precision
            auto timestamp_type = arrow::timestamp(arrow::TimeUnit::NANO);
            // Create a TimestampScalar with nanosecond value
            return std::make_shared<arrow::TimestampScalar>(timestamp_type);
        } break;
        case duckdb::LogicalTypeId::DATE: {
            // Define a date type
            auto date_type = arrow::date32();
            // Create a DateScalar with the date value
            return arrow::MakeNullScalar(date_type);
        } break;
        default:
            throw std::runtime_error(
                "getDefaultValueForDuckdbValueType unhandled type." +
                std::to_string(static_cast<int>(type)));
    }
}
std::string schemaColumnNamesToString(
    const std::shared_ptr<arrow::Schema> arrow_schema) {
    std::string ret = "";
    for (int i = 0; i < arrow_schema->num_fields(); i++) {
        ret += arrow_schema->field(i)->name();
        if (i != arrow_schema->num_fields() - 1) {
            ret += ", ";
        }
    }
    return ret;
}

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

arrow::compute::Expression tableFilterToArrowExpr(
    duckdb::idx_t col_idx, duckdb::unique_ptr<duckdb::TableFilter> &tf,
    PyObject *schema_fields) {
    switch (tf->filter_type) {
        case duckdb::TableFilterType::CONSTANT_COMPARISON: {
            duckdb::unique_ptr<duckdb::ConstantFilter> constantFilter =
                dynamic_cast_unique_ptr<duckdb::ConstantFilter>(std::move(tf));
            PyObject *py_selected_field =
                PyList_GetItem(schema_fields, col_idx);
            if (!PyUnicode_Check(py_selected_field)) {
                throw std::runtime_error(
                    "tableFilterSetToArrowCompute selected field is "
                    "not unicode object.");
            }
            auto column_ref =
                arrow::compute::field_ref(PyUnicode_AsUTF8(py_selected_field));
            auto scalar_val = std::visit(
                [](const auto &value) {
                    return arrow::compute::literal(value);
                },
                extractValue(constantFilter->constant));
            auto expr = expressionTypeToArrowCompute(
                constantFilter->comparison_type)(column_ref, scalar_val);
            return expr;
        } break;
        case duckdb::TableFilterType::CONJUNCTION_AND: {
            duckdb::unique_ptr<duckdb::ConjunctionAndFilter>
                conjunctionAndFilter =
                    dynamic_cast_unique_ptr<duckdb::ConjunctionAndFilter>(
                        std::move(tf));
            assert(conjunctionAndFilter->child_filters.size() >= 2);
            auto expr = arrow::compute::and_(
                tableFilterToArrowExpr(col_idx,
                                       conjunctionAndFilter->child_filters[0],
                                       schema_fields),
                tableFilterToArrowExpr(col_idx,
                                       conjunctionAndFilter->child_filters[1],
                                       schema_fields));
            for (size_t i = 2; i < conjunctionAndFilter->child_filters.size();
                 ++i) {
                expr = arrow::compute::and_(
                    expr, tableFilterToArrowExpr(
                              col_idx, conjunctionAndFilter->child_filters[i],
                              schema_fields));
            }
            return expr;
        } break;
        case duckdb::TableFilterType::CONJUNCTION_OR: {
            duckdb::unique_ptr<duckdb::ConjunctionOrFilter>
                conjunctionOrFilter =
                    dynamic_cast_unique_ptr<duckdb::ConjunctionOrFilter>(
                        std::move(tf));
            assert(conjunctionOrFilter->child_filters.size() >= 2);
            auto expr = arrow::compute::or_(
                tableFilterToArrowExpr(col_idx,
                                       conjunctionOrFilter->child_filters[0],
                                       schema_fields),
                tableFilterToArrowExpr(col_idx,
                                       conjunctionOrFilter->child_filters[1],
                                       schema_fields));
            for (size_t i = 2; i < conjunctionOrFilter->child_filters.size();
                 ++i) {
                expr = arrow::compute::or_(
                    expr, tableFilterToArrowExpr(
                              col_idx, conjunctionOrFilter->child_filters[i],
                              schema_fields));
            }
            return expr;
        } break;
        case duckdb::TableFilterType::OPTIONAL_FILTER: {
            duckdb::unique_ptr<duckdb::OptionalFilter> OptionalFilter =
                dynamic_cast_unique_ptr<duckdb::OptionalFilter>(std::move(tf));
            // We'll try to execute the filter but if some part of it is
            // unsupported then the recursive call will generate an exception
            // and we'll convert this filter into a no-op by returning true.
            try {
                return tableFilterToArrowExpr(
                    col_idx, OptionalFilter->child_filter, schema_fields);
            } catch (...) {
                return arrow::compute::literal(true);
            }
        } break;
        default:
            throw std::runtime_error(
                "tableFilterToArrowExpr unsupported filter "
                "type " +
                std::to_string(static_cast<int>(tf->filter_type)));
    }
}

PyObject *tableFilterSetToArrowCompute(duckdb::TableFilterSet &filters,
                                       PyObject *schema_fields) {
    PyObject *ret = Py_None;
    if (filters.filters.size() == 0) {
        return ret;
    }
    arrow::py::import_pyarrow_wrappers();
    std::vector<arrow::compute::Expression> parts;

    for (auto &tf : filters.filters) {
        duckdb::idx_t col_idx = tf.first;
        duckdb::unique_ptr<duckdb::TableFilter> filter = std::move(tf.second);
        parts.push_back(tableFilterToArrowExpr(col_idx, filter, schema_fields));
    }

    arrow::compute::Expression whole = arrow::compute::and_(parts);
    ret = arrow::py::wrap_expression(whole);

    return ret;
}

void initInputColumnMapping(std::vector<int64_t> &col_inds,
                            const std::vector<uint64_t> &keys, uint64_t ncols) {
    for (uint64_t i : keys) {
        col_inds.push_back(i);
    }
    for (uint64_t i = 0; i < ncols; i++) {
        if (std::find(keys.begin(), keys.end(), i) != keys.end()) {
            continue;
        }
        col_inds.push_back(i);
    }
}

std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t> getColRefMap(
    std::vector<duckdb::ColumnBinding> source_cols) {
    std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t> col_ref_map;
    for (size_t i = 0; i < source_cols.size(); i++) {
        duckdb::ColumnBinding &col = source_cols[i];
        col_ref_map[{col.table_index, col.column_index}] = i;
    }
    return col_ref_map;
}

std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t> getColRefMap(
    std::vector<std::vector<duckdb::ColumnBinding>> source_cols_vec) {
    std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t> col_ref_map;
    for (auto &source_cols : source_cols_vec) {
        for (size_t i = 0; i < source_cols.size(); i++) {
            duckdb::ColumnBinding &col = source_cols[i];
            col_ref_map[{col.table_index, col.column_index}] = i;
        }
    }
    return col_ref_map;
}

std::shared_ptr<arrow::DataType> duckdbTypeToArrow(
    const duckdb::LogicalType &type) {
    switch (type.id()) {
        case duckdb::LogicalTypeId::TINYINT:
            return arrow::int8();
        case duckdb::LogicalTypeId::SMALLINT:
            return arrow::int16();
        case duckdb::LogicalTypeId::INTEGER:
            return arrow::int32();
        case duckdb::LogicalTypeId::BIGINT:
            return arrow::int64();
        case duckdb::LogicalTypeId::UTINYINT:
            return arrow::uint8();
        case duckdb::LogicalTypeId::USMALLINT:
            return arrow::uint16();
        case duckdb::LogicalTypeId::UINTEGER:
            return arrow::uint32();
        case duckdb::LogicalTypeId::UBIGINT:
            return arrow::uint64();
        case duckdb::LogicalTypeId::FLOAT:
            return arrow::float32();
        case duckdb::LogicalTypeId::DOUBLE:
            return arrow::float64();
        case duckdb::LogicalTypeId::BOOLEAN:
            return arrow::boolean();
        case duckdb::LogicalTypeId::VARCHAR:
            return arrow::large_utf8();
        case duckdb::LogicalTypeId::DATE:
            return arrow::date32();
        default:
            throw std::runtime_error(
                "duckdbTypeToArrow unsupported LogicalType conversion " +
                std::to_string(static_cast<int>(type.id())));
    }
}

std::tuple<std::shared_ptr<table_info>, int64_t, int64_t, int64_t>
runPythonScalarFunction(std::shared_ptr<table_info> input_batch,
                        const std::shared_ptr<arrow::DataType> &result_type,
                        PyObject *args, bool has_state, PyObject *&init_state) {
    // Call bodo.pandas.utils.run_apply_udf() to run the UDF

    // Import the bodo.pandas.utils module
    PyObject *bodo_module = PyImport_ImportModule("bodo.pandas.utils");
    if (!bodo_module) {
        PyErr_Print();
        throw std::runtime_error("Failed to import bodo.pandas.utils module");
    }

    // Call the run_apply_udf() with the table_info pointer and UDF function
    PyObject *result_type_py(arrow::py::wrap_data_type(result_type));
    PyObject *result;
    if (has_state) {
        // Validate input
        if (!PyTuple_Check(args) || PyTuple_Size(args) != 5) {
            throw std::runtime_error("Expected a 5-tuple");
        }
        PyObject *args_in_func_args = PyTuple_GetItem(args, 3);  // borrowed ref
        if (!PyTuple_Check(args_in_func_args) ||
            PyTuple_Size(args_in_func_args) < 2) {
            throw std::runtime_error("args_in_func_args should be 2+-tuple");
        }

        // We haven't run the init_state function yet so run it.
        if (init_state == nullptr) {
            // Extract first tuple element and verify it's callable
            PyObject *init_func =
                PyTuple_GetItem(args_in_func_args, 0);  // borrwed ref
            if (!init_func || !PyCallable_Check(init_func)) {
                throw std::runtime_error("First element is not a callable");
            }
            Py_INCREF(init_func);  // bump to own a reference

            // Call the initialization function.
            init_state = PyObject_CallObject(init_func, nullptr);
            Py_DECREF(init_func);  // drop our extra ref to func
            if (!init_state) {
                throw std::runtime_error("Initilization function failed");
            }
        }

        PyObject *new_tuple =
            PyTuple_New(PyTuple_Size(args));  // returns a new reference
        if (!new_tuple) {
            throw std::runtime_error("New tuple creation failed");
        }

        // New tuple will be the 5 tuple with the 4th element being yet another
        // new tuple with init_state followed by the other original args in
        // func_args.
        // The original data structure can be thought of as the following:
        // (a, b, c, (init_state_fn, row_fn, na_state), d)
        // The code below creates new tuples and looks like the following:
        // (a, b, c, (init_state, row_fn, na_state), d)
        for (int i = 0; i < PyTuple_Size(args); ++i) {
            // The args part of func_args.
            if (i == 3) {
                // Create new tuple replacing init_func with init_state.
                PyObject *new_args_in_func_args =
                    PyTuple_New(PyTuple_Size(args_in_func_args));
                Py_INCREF(init_state);
                // Put init_state into the tuple.
                PyTuple_SetItem(new_args_in_func_args, 0, init_state);
                // Copy other original args to the new tuple.
                for (int j = 1; j < PyTuple_Size(args_in_func_args); ++j) {
                    PyObject *arg_elem = PyTuple_GetItem(args_in_func_args, j);
                    Py_INCREF(arg_elem);
                    PyTuple_SetItem(new_args_in_func_args, j, arg_elem);
                }
                PyTuple_SetItem(new_tuple, i, new_args_in_func_args);
                continue;
            }
            // Copy over parts of func_args into the new func_args tuple.
            PyObject *tup_elem = PyTuple_GetItem(args, i);  // borrowed ref
            Py_INCREF(tup_elem);
            PyTuple_SetItem(new_tuple, i, tup_elem);
        }
        result = PyObject_CallMethod(
            bodo_module, "run_func_on_table", "LOO",
            reinterpret_cast<int64_t>(new table_info(*input_batch)),
            result_type_py, new_tuple);
        Py_DECREF(new_tuple);
    } else {
        result = PyObject_CallMethod(
            bodo_module, "run_func_on_table", "LOO",
            reinterpret_cast<int64_t>(new table_info(*input_batch)),
            result_type_py, args);
    }
    if (!result) {
        PyErr_Print();
        Py_DECREF(bodo_module);
        throw std::runtime_error("Error calling run_apply_udf");
    }
    // Result should be a tuple with 4 elements:
    // 1. Output table_info pointer
    // 2. Time taken to convert C++ to Python
    // 3. Time taken to run the UDF
    // 4. Time taken to convert Python back to C++
    if (!PyTuple_Check(result) || PyTuple_Size(result) != 4) {
        Py_DECREF(result);
        Py_DECREF(bodo_module);
        throw std::runtime_error(
            "Expected a tuple of 4 elements from run_apply_udf");
    }
// Extract the elements from the tuple and turn them into C++ longs
#define CHECK_AND_GET_PYLONG(py_item, res)                           \
    if (!PyLong_Check(py_item)) {                                    \
        Py_DECREF(result);                                           \
        Py_DECREF(bodo_module);                                      \
        throw std::runtime_error("Expected an integer for " #py_item \
                                 " from run_apply_udf");             \
    }                                                                \
    res = PyLong_AsLongLong(py_item);
    PyObject *py_table_info_ptr = PyTuple_GetItem(result, 0);
    PyObject *py_cpp_to_py_time = PyTuple_GetItem(result, 1);
    PyObject *py_udf_execution_time = PyTuple_GetItem(result, 2);
    PyObject *py_py_to_cpp_time = PyTuple_GetItem(result, 3);
    int64_t table_info_ptr, cpp_to_py_time_val, udf_execution_time_val,
        py_to_cpp_time_val;
    CHECK_AND_GET_PYLONG(py_table_info_ptr, table_info_ptr);
    CHECK_AND_GET_PYLONG(py_cpp_to_py_time, cpp_to_py_time_val);
    CHECK_AND_GET_PYLONG(py_udf_execution_time, udf_execution_time_val);
    CHECK_AND_GET_PYLONG(py_py_to_cpp_time, py_to_cpp_time_val);

    std::shared_ptr<table_info> out_batch(
        reinterpret_cast<table_info *>(table_info_ptr));

    Py_DECREF(bodo_module);
    Py_DECREF(result);

    return {out_batch, cpp_to_py_time_val, udf_execution_time_val,
            py_to_cpp_time_val};
}

std::shared_ptr<table_info> runCfuncScalarFunction(
    std::shared_ptr<table_info> input_batch, table_udf_t cfunc_ptr) {
    table_info *in_table = new table_info(*input_batch);
    table_info *out_table = cfunc_ptr(in_table);

    if (!out_table) {
        throw std::runtime_error("Error executing cfunc.");
    }

    std::shared_ptr<table_info> out_batch(out_table);
    out_batch->column_names = input_batch->column_names;
    out_batch->metadata = input_batch->metadata;

    return out_batch;
}

std::shared_ptr<arrow::Scalar> convertDuckdbValueToArrowScalar(
    const duckdb::Value &value) {
    arrow::Result<std::shared_ptr<arrow::Scalar>> scalar_res = std::visit(
        [](const auto &&value) {
            if constexpr (std::is_same_v<decltype(value),
                                         std::shared_ptr<arrow::Scalar>>) {
                // If the value is already a scalar, we can just wrap it
                // in an arrow::Result and return it.
                arrow::Result<std::shared_ptr<arrow::Scalar>> ret =
                    arrow::ToResult(value);
                return value;
            } else {
                arrow::Result<std::shared_ptr<arrow::Scalar>> ret =
                    arrow::MakeScalar(value);
                return ret;
            }
        },
        extractValue(value));
    if (!scalar_res.ok()) {
        throw std::runtime_error("Failed to convert duckdb value to scalar: " +
                                 scalar_res.status().ToString());
    }
    return scalar_res.ValueOrDie();
}

std::shared_ptr<arrow::DataType> duckdbValueToArrowType(
    const duckdb::Value &value) {
    arrow::Result<std::shared_ptr<arrow::Scalar>> scalar_res = std::visit(
        [](const auto &&value) {
            if constexpr (std::is_same_v<decltype(value),
                                         std::shared_ptr<arrow::Scalar>>) {
                // If the value is already a scalar, we can just wrap it
                // in an arrow::Result and return it.
                arrow::Result<std::shared_ptr<arrow::Scalar>> ret =
                    arrow::ToResult(value);
                return value;
            } else {
                arrow::Result<std::shared_ptr<arrow::Scalar>> ret =
                    arrow::MakeScalar(value);
                return ret;
            }
        },
        getDefaultValueForDuckdbValueType(value));
    if (!scalar_res.ok()) {
        throw std::runtime_error(
            "Failed to convert duckdb value to arrow type: " +
            scalar_res.status().ToString());
    }
    return scalar_res.ValueOrDie()->type;
}

std::string expressionTypeToPyicebergclass(duckdb::ExpressionType expr_type) {
    switch (expr_type) {
        case duckdb::ExpressionType::COMPARE_EQUAL:
            return "EqualTo";
        case duckdb::ExpressionType::COMPARE_NOTEQUAL:
            return "NotEqualTo";
        case duckdb::ExpressionType::COMPARE_GREATERTHAN:
            return "GreaterThan";
        case duckdb::ExpressionType::COMPARE_LESSTHAN:
            return "LessThan";
        case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
            return "GreaterThanOrEqual";
        case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
            return "LessThanOrEqual";
        case duckdb::ExpressionType::OPERATOR_IS_NULL:
            return "IsNull";
        case duckdb::ExpressionType::OPERATOR_IS_NOT_NULL:
            return "IsNotNull";
        default:
            throw std::runtime_error(
                "expressionTypeToPyicebergclass unsupported expression type: " +
                std::to_string(static_cast<int>(expr_type)));
    }
}

// Helper for duckdbFilterSetToPyicebergFilter. We need to call this recursively
// for conjunction filters.
PyObject *_duckdbFilterToPyicebergFilter(
    duckdb::unique_ptr<duckdb::TableFilter> tf, const std::string field_name,
    PyObjectPtr &pyiceberg_expression_mod) {
    PyObject *py_expr = nullptr;
    switch (tf->filter_type) {
        case duckdb::TableFilterType::CONSTANT_COMPARISON: {
            duckdb::unique_ptr<duckdb::ConstantFilter> constantFilter =
                dynamic_cast_unique_ptr<duckdb::ConstantFilter>(std::move(tf));

            std::shared_ptr<arrow::Scalar> scalar =
                convertDuckdbValueToArrowScalar(constantFilter->constant);
            std::string pyiceberg_class =
                expressionTypeToPyicebergclass(constantFilter->comparison_type);
            PyObjectPtr scalar_pyobject = arrow::py::wrap_scalar(scalar);
            if (!scalar_pyobject) {
                throw std::runtime_error(
                    "Failed to convert duckdb value to pyarrow scalar");
            }
            // Call scalar.as_py() to get the Python object representation
            PyObjectPtr scalar_as_py =
                PyObject_CallMethod(scalar_pyobject.get(), "as_py", nullptr);

            py_expr = PyObject_CallMethod(
                pyiceberg_expression_mod, pyiceberg_class.c_str(), "sO",
                field_name.c_str(), scalar_as_py.get());
        } break;
        case duckdb::TableFilterType::IS_NULL: {
            py_expr = PyObject_CallMethod(pyiceberg_expression_mod, "IsNull",
                                          "s", field_name.c_str());
        } break;
        case duckdb::TableFilterType::IS_NOT_NULL: {
            py_expr = PyObject_CallMethod(pyiceberg_expression_mod, "IsNotNull",
                                          "s", field_name.c_str());
        } break;
        case duckdb::TableFilterType::CONJUNCTION_AND: {
            duckdb::unique_ptr<duckdb::ConjunctionAndFilter> conjunctionFilter =
                dynamic_cast_unique_ptr<duckdb::ConjunctionAndFilter>(
                    std::move(tf));
            for (auto &child_filter : conjunctionFilter->child_filters) {
                PyObject *child_expr = _duckdbFilterToPyicebergFilter(
                    std::move(child_filter), field_name,
                    pyiceberg_expression_mod);
                if (!py_expr) {
                    py_expr = child_expr;
                } else {
                    PyObject *original_py_expr = py_expr;
                    py_expr = PyObject_CallMethod(py_expr, "__and__", "O",
                                                  child_expr);
                    Py_DECREF(original_py_expr);
                    Py_DECREF(child_expr);
                }
            }
        } break;
        case duckdb::TableFilterType::CONJUNCTION_OR: {
            duckdb::unique_ptr<duckdb::ConjunctionOrFilter> conjunctionFilter =
                dynamic_cast_unique_ptr<duckdb::ConjunctionOrFilter>(
                    std::move(tf));
            for (auto &child_filter : conjunctionFilter->child_filters) {
                PyObject *child_expr = _duckdbFilterToPyicebergFilter(
                    std::move(child_filter), field_name,
                    pyiceberg_expression_mod);
                if (!py_expr) {
                    py_expr = child_expr;
                } else {
                    PyObject *original_py_expr = py_expr;
                    py_expr =
                        PyObject_CallMethod(py_expr, "__or__", "O", child_expr);
                    Py_DECREF(original_py_expr);
                    Py_DECREF(child_expr);
                }
            }
        } break;
        default:
            throw std::runtime_error(
                "duckdbFilterToPyicebergFilter unsupported filter "
                "type " +
                std::to_string(static_cast<int>(tf->filter_type)));
    }
    if (!py_expr) {
        throw std::runtime_error(
            "Failed to create pyiceberg expression for filter " +
            tf->DebugToString());
    }
    return py_expr;
}

PyObject *duckdbFilterSetToPyicebergFilter(
    duckdb::TableFilterSet &filters,
    std::shared_ptr<arrow::Schema> arrow_schema) {
    PyObjectPtr pyiceberg_expression_mod =
        PyImport_ImportModule("pyiceberg.expressions");
    if (!pyiceberg_expression_mod) {
        throw std::runtime_error(
            "Failed to import pyiceberg.expressions module");
    }
    // Default return is pyiceberg.expressions.AlwaysTrue()
    PyObject *ret = PyObject_CallMethod(pyiceberg_expression_mod.get(),
                                        "AlwaysTrue", nullptr);
    for (auto &tf : filters.filters) {
        std::shared_ptr<arrow::Field> field = arrow_schema->field(tf.first);
        std::string field_name = field->name();
        PyObject *py_expr = _duckdbFilterToPyicebergFilter(
            std::move(tf.second), field_name, pyiceberg_expression_mod);
        PyObject *original_ret = ret;
        ret = PyObject_CallMethod(ret, "__and__", "O", py_expr);
        if (!ret) {
            PyErr_Print();
            throw std::runtime_error(
                "Failed to combine pyiceberg expressions for filter " +
                std::to_string(tf.first));
        }
        Py_DECREF(original_ret);
        Py_DECREF(py_expr);
    }

    return ret;
}

arrow::Datum ConvertToDatum(void *raw_ptr,
                            std::shared_ptr<arrow::DataType> type) {
    using arrow::Type;

    switch (type->id()) {
        case Type::BOOL: {
            bool value = *static_cast<bool *>(raw_ptr);
            return arrow::Datum(std::make_shared<arrow::BooleanScalar>(value));
        }
        case Type::INT8: {
            int8_t value = *static_cast<int8_t *>(raw_ptr);
            return arrow::Datum(std::make_shared<arrow::Int8Scalar>(value));
        }
        case Type::UINT8: {
            uint8_t value = *static_cast<uint8_t *>(raw_ptr);
            return arrow::Datum(std::make_shared<arrow::UInt8Scalar>(value));
        }
        case Type::INT16: {
            int16_t value = *static_cast<int16_t *>(raw_ptr);
            return arrow::Datum(std::make_shared<arrow::Int16Scalar>(value));
        }
        case Type::UINT16: {
            uint16_t value = *static_cast<uint16_t *>(raw_ptr);
            return arrow::Datum(std::make_shared<arrow::UInt16Scalar>(value));
        }
        case Type::INT32: {
            int32_t value = *static_cast<int32_t *>(raw_ptr);
            return arrow::Datum(std::make_shared<arrow::Int32Scalar>(value));
        }
        case Type::UINT32: {
            uint32_t value = *static_cast<uint32_t *>(raw_ptr);
            return arrow::Datum(std::make_shared<arrow::UInt32Scalar>(value));
        }
        case Type::INT64: {
            int64_t value = *static_cast<int64_t *>(raw_ptr);
            return arrow::Datum(std::make_shared<arrow::Int64Scalar>(value));
        }
        case Type::UINT64: {
            uint64_t value = *static_cast<uint64_t *>(raw_ptr);
            return arrow::Datum(std::make_shared<arrow::UInt64Scalar>(value));
        }
        case Type::FLOAT: {
            float value = *static_cast<float *>(raw_ptr);
            return arrow::Datum(std::make_shared<arrow::FloatScalar>(value));
        }
        case Type::DOUBLE: {
            double value = *static_cast<double *>(raw_ptr);
            return arrow::Datum(std::make_shared<arrow::DoubleScalar>(value));
        }
            // ————— Date/Time types —————
        case Type::DATE32: {
            // days since UNIX epoch
            auto v = *static_cast<int32_t *>(raw_ptr);
            return arrow::Datum(std::make_shared<arrow::Date32Scalar>(v));
        }
        case Type::DATE64: {
            // milliseconds since UNIX epoch
            auto v = *static_cast<int64_t *>(raw_ptr);
            return arrow::Datum(std::make_shared<arrow::Date64Scalar>(v));
        }
        case Type::TIME32: {
            // time32 can be SECOND or MILLI unit—pass the type through
            auto time32_type =
                std::static_pointer_cast<arrow::Time32Type>(type);
            auto v = *static_cast<int32_t *>(raw_ptr);
            return arrow::Datum(
                std::make_shared<arrow::Time32Scalar>(v, time32_type));
        }
        case Type::TIME64: {
            auto time64_type =
                std::static_pointer_cast<arrow::Time64Type>(type);
            auto v = *static_cast<int64_t *>(raw_ptr);
            return arrow::Datum(
                std::make_shared<arrow::Time64Scalar>(v, time64_type));
        }
        case Type::TIMESTAMP: {
            // timestamp value in the unit of its type (s, ms, μs, or ns)
            auto ts_type = std::static_pointer_cast<arrow::TimestampType>(type);
            auto v = *static_cast<int64_t *>(raw_ptr);
            return arrow::Datum(
                std::make_shared<arrow::TimestampScalar>(v, ts_type));
        }
        case Type::DURATION: {
            // duration value in the unit of its type (s, ms, μs, or ns)
            auto dur_type = std::static_pointer_cast<arrow::DurationType>(type);
            auto v = *static_cast<int64_t *>(raw_ptr);
            return arrow::Datum(
                std::make_shared<arrow::DurationScalar>(v, dur_type));
        }
        default:
            throw std::runtime_error(
                "ConvertToDatum does not support arrow::DataType " +
                type->ToString());
    }
}
