#include "_util.h"
#include <arrow/compute/api.h>
#include <arrow/python/pyarrow.h>
#include <arrow/result.h>
#include "../io/arrow_compat.h"
#include "../libs/_utils.h"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"

std::variant<int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t,
             uint64_t, bool, std::string, float, double, arrow::TimestampScalar>
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
        case duckdb::LogicalTypeId::TIMESTAMP_NS: {
            // Define a timestamp type with nanosecond precision
            auto timestamp_type = arrow::timestamp(arrow::TimeUnit::NANO);
            duckdb::timestamp_ns_t extracted =
                value.GetValue<duckdb::timestamp_ns_t>();
            // Create a TimestampScalar with nanosecond value
            return arrow::TimestampScalar(extracted.value, timestamp_type);
        } break;
        default:
            throw std::runtime_error("extractValue unhandled type." +
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

PyObject *tableFilterSetToArrowCompute(duckdb::TableFilterSet &filters,
                                       PyObject *schema_fields) {
    PyObject *ret = Py_None;
    if (filters.filters.size() == 0) {
        return ret;
    }
    arrow::py::import_pyarrow_wrappers();
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

    return ret;
}
void initInputColumnMapping(std::vector<int64_t> &col_inds,
                            std::vector<uint64_t> &keys, uint64_t ncols) {
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
        default:
            throw std::runtime_error(
                "duckdbTypeToArrow unsupported LogicalType conversion " +
                std::to_string(static_cast<int>(type.id())));
    }
}
std::shared_ptr<arrow::Scalar> convertDuckdbValueToArrowScalar(
    const duckdb::Value &value) {
    arrow::Result<std::shared_ptr<arrow::Scalar>> scalar_res = std::visit(
        [](const auto &&value) {
            if constexpr (std::is_same_v<std::decay_t<decltype(value)>,
                                         arrow::TimestampScalar>) {
                arrow::Result<std::shared_ptr<arrow::Scalar>> ret =
                    arrow::ToResult(
                        std::make_shared<arrow::TimestampScalar>(value));
                return ret;
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
            return "GreaterThanOrEqualTo";
        case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
            return "LessThanOrEqualTo";
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

            py_expr = PyObject_CallMethod(
                pyiceberg_expression_mod, pyiceberg_class.c_str(), "sO",
                field_name.c_str(), arrow::py::wrap_scalar(scalar));
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
