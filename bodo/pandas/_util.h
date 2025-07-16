#pragma once

#include <Python.h>
#include <arrow/api.h>
#include <cstdint>
#include <map>
#include <variant>
#include "../libs/_array_utils.h"
#include "duckdb/common/types/value.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb/planner/column_binding.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/table_filter.hpp"

/**
 * @brief Convert duckdb value to C++ variant.
 *
 * @param expr - the duckdb value to convert
 * @return the C++ variant converted value
 */
std::variant<int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t,
             uint64_t, bool, std::string, float, double,
             std::shared_ptr<arrow::Scalar>>
extractValue(const duckdb::Value &value);

/**
 * @brief Convert duckdb value to null var of right type.
 *
 * @param expr - the duckdb value to convert
 * @return the C++ variant converted value
 */
std::variant<int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t,
             uint64_t, bool, std::string, float, double,
             std::shared_ptr<arrow::Scalar>>
getDefaultValueForDuckdbValueType(const duckdb::Value &value);

/**
 * @brief Return a string representation of the column names in the Arrow schema
 * for printing purposes (e.g. plan prints).
 *
 * @param arrow_schema input Arrow schema
 * @return std::string string representation of the column names
 */
std::string schemaColumnNamesToString(
    const std::shared_ptr<arrow::Schema> arrow_schema);

/**
 * @brief Dynamic cast of base pointer to derived pointer.
 *
 * @param base_ptr - the base pointer to cast from
 * @return a non-NULL pointer of the derived type if the cast is possible else
 *         NULL
 */
template <typename Derived, typename Base>
duckdb::unique_ptr<Derived> dynamic_cast_unique_ptr(
    duckdb::unique_ptr<Base> &&base_ptr) noexcept {
    // Perform dynamic_cast on the raw pointer
    if (Derived *derived_raw = dynamic_cast<Derived *>(base_ptr.get())) {
        // Release ownership from the base_ptr and transfer it to a new
        // unique_ptr
        base_ptr.release();  // Release the ownership of the raw pointer
        return duckdb::unique_ptr<Derived>(derived_raw);
    }
    // If the cast fails, return a nullptr unique_ptr
    return nullptr;
}

PyObject *tableFilterSetToArrowCompute(duckdb::TableFilterSet &filters,
                                       PyObject *schema_fields);

/**
 * @brief Initialize mapping of input column orders so that keys are in the
 * beginning of build/probe tables to match streaming join APIs. See
 * https://github.com/bodo-ai/Bodo/blob/905664de2c37741d804615cdbb3fb437621ff0bd/bodo/libs/streaming/join.py#L189
 * @param col_inds input mapping to fill
 * @param keys key column indices
 * @param ncols number of columns in the table
 */
void initInputColumnMapping(std::vector<int64_t> &col_inds,
                            const std::vector<uint64_t> &keys, uint64_t ncols);

/**
 * @brief Create a map of column bindings to column indices in physical input
 * table
 *
 * @param source_cols column bindings in source table
 * @return std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t>
 */
std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t> getColRefMap(
    std::vector<duckdb::ColumnBinding> source_cols);

/**
 * @brief Create a map of column bindings to column indices in physical input
 * table
 *
 * @param source_cols multiple sets of column bindings in source tables
 * @return std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t>
 */
std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t> getColRefMap(
    std::vector<std::vector<duckdb::ColumnBinding>> source_cols);

/**
 * @brief Convert duckdb type to arrow type.
 *
 * @param type - the duckdb type to convert
 * @return the converted type
 */
std::shared_ptr<arrow::DataType> duckdbTypeToArrow(
    const duckdb::LogicalType &type);

/**
 * @brief UDF plan node data to pass around in DuckDB plans in
 * BoundFunctionExpression.
 *
 */
struct BodoPythonScalarFunctionData : public duckdb::FunctionData {
    BodoPythonScalarFunctionData(PyObject *args,
                                 std::shared_ptr<arrow::Schema> out_schema)
        : args(args), out_schema(std::move(out_schema)) {
        if (args)
            Py_INCREF(args);
    }
    BodoPythonScalarFunctionData(std::shared_ptr<arrow::Schema> out_schema)
        : args(nullptr), out_schema(std::move(out_schema)) {}
    ~BodoPythonScalarFunctionData() override {
        if (args)
            Py_DECREF(args);
    }
    BodoPythonScalarFunctionData(const BodoPythonScalarFunctionData &other)
        : args(other.args), out_schema(other.out_schema) {
        if (args)
            Py_INCREF(args);
    }
    BodoPythonScalarFunctionData &operator=(
        const BodoPythonScalarFunctionData &other) {
        if (this != &other) {
            Py_XINCREF(other.args);
            Py_XDECREF(args);
            args = other.args;
            out_schema = other.out_schema;
        }
        return *this;
    }

    bool Equals(const FunctionData &other_p) const override {
        const BodoPythonScalarFunctionData &other =
            other_p.Cast<BodoPythonScalarFunctionData>();
        return (other.args == this->args);
    }
    duckdb::unique_ptr<duckdb::FunctionData> Copy() const override {
        return duckdb::make_uniq<BodoPythonScalarFunctionData>(this->args,
                                                               out_schema);
    }

    PyObject *args;  // If present then a UDF.
    std::shared_ptr<arrow::Schema> out_schema;
};

/**
 * @brief Aggregate node data to pass around in DuckDB plans in
 * BoundAggregateExpression.
 *
 */
struct BodoAggFunctionData : public duckdb::FunctionData {
    BodoAggFunctionData(bool dropna, std::string name,
                        std::shared_ptr<arrow::Schema> out_schema)
        : out_schema(std::move(out_schema)), dropna(dropna), name(name) {}

    bool Equals(const FunctionData &other_p) const override {
        const BodoAggFunctionData &other = other_p.Cast<BodoAggFunctionData>();
        return (other.dropna == this->dropna && other.name == this->name &&
                other.out_schema == this->out_schema);
    }
    duckdb::unique_ptr<duckdb::FunctionData> Copy() const override {
        return duckdb::make_uniq<BodoAggFunctionData>(this->dropna, this->name,
                                                      this->out_schema);
    }

    const std::shared_ptr<arrow::Schema> out_schema;
    const bool dropna;
    const std::string name;
};

/**
 * @brief Run Python scalar function on the input batch and return the
 * output table (single data column plus Index columns).
 *
 * @param input_batch input table batch
 * @param result_type The expected result type of the function
 * @param args Python arguments for the function
 * @return std::shared_ptr<table_info> output table from the Python function
 */
std::shared_ptr<table_info> runPythonScalarFunction(
    std::shared_ptr<table_info> input_batch,
    const std::shared_ptr<arrow::DataType> &result_type, PyObject *args);

/**
 * @brief Convert duckdb table filters to pyiceberg expressions.
 * @param filters - the duckdb table filters to convert
 * @param arrow_schema - the arrow schema to bind the filters to
 * @return a PyObject representing the pyiceberg expression
 */
PyObject *duckdbFilterSetToPyicebergFilter(
    duckdb::TableFilterSet &filters,
    std::shared_ptr<arrow::Schema> arrow_schema);

/**
 * @brief Convert a DuckDB Value object which is used in constant expressions
 * (BoundConstantExpression) to an Arrow scalar.
 *
 * @param value DuckDB Value object to convert
 * @return std::shared_ptr<arrow::Scalar> equivalent Arrow scalar object
 */
std::shared_ptr<arrow::Scalar> convertDuckdbValueToArrowScalar(
    const duckdb::Value &value);

/**
 * @brief Convert a DuckDB Value object to the corresponding arrow type
 *
 * @param value DuckDB Value object to convert
 * @return std::shared_ptr<arrow::DataType> Arrow type of DuckDB value
 */
std::shared_ptr<arrow::DataType> duckdbValueToArrowType(
    const duckdb::Value &value);

/**
 * @brief Convert a raw pointer to a value of a given arrow type into an arrow
 * Datum.
 *
 * @param raw_ptr pointer to the value to put into Datum
 * @param type arrow type of the value
 * @return arrow::Datum the converted value
 */
arrow::Datum ConvertToDatum(void *raw_ptr,
                            std::shared_ptr<arrow::DataType> type);
