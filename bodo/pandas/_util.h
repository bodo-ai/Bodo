#pragma once

#include <Python.h>
#include <arrow/api.h>
#include <cstdint>
#include <map>
#include <utility>
#include <variant>
#include "../libs/_bodo_to_arrow.h"
#include "../libs/streaming/_join.h"
#include "duckdb/common/types/value.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb/planner/column_binding.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_case_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "optimizer/runtime_join_filter.h"

using table_udf_t = table_info *(*)(table_info *);

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
 * @brief Convert Arrow scalar to duckdb value.
 *
 * @param scalar - the arrow scalar to convert
 * @return the converted duckdb value
 */
duckdb::Value ArrowScalarToDuckDBValue(
    const std::shared_ptr<arrow::Scalar> &scalar);

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
struct BodoScalarFunctionData : public duckdb::FunctionData {
    BodoScalarFunctionData(PyObject *args,
                           std::shared_ptr<arrow::Schema> out_schema,
                           bool is_cfunc, bool has_state)
        : args(args),
          out_schema(std::move(out_schema)),
          is_cfunc(is_cfunc),
          has_state(has_state),
          arrow_func_name("") {
        if (args) {
            Py_INCREF(args);
        }
    }

    BodoScalarFunctionData(PyObject *args,
                           std::shared_ptr<arrow::Schema> out_schema,
                           bool is_cfunc, bool has_state,
                           std::string arrow_func_name)
        : args(args),
          out_schema(std::move(out_schema)),
          is_cfunc(is_cfunc),
          has_state(has_state),
          arrow_func_name(arrow_func_name) {
        if (args) {
            Py_INCREF(args);
        }
    }

    BodoScalarFunctionData(std::shared_ptr<arrow::Schema> out_schema,
                           std::string arrow_func_name)
        : args(nullptr),
          out_schema(std::move(out_schema)),
          is_cfunc(false),
          has_state(false),
          arrow_func_name(std::move(arrow_func_name)) {}

    BodoScalarFunctionData(std::shared_ptr<arrow::Schema> out_schema)
        : args(nullptr),
          out_schema(std::move(out_schema)),
          is_cfunc(false),
          has_state(false),
          arrow_func_name("") {}

    ~BodoScalarFunctionData() override {
        if (args) {
            Py_DECREF(args);
        }
    }

    BodoScalarFunctionData(const BodoScalarFunctionData &other)
        : args(other.args),
          out_schema(other.out_schema),
          is_cfunc(other.is_cfunc),
          has_state(other.has_state),
          arrow_func_name(other.arrow_func_name) {
        if (args) {
            Py_INCREF(args);
        }
    }

    BodoScalarFunctionData &operator=(const BodoScalarFunctionData &other) {
        if (this != &other) {
            Py_XINCREF(other.args);
            Py_XDECREF(args);
            args = other.args;
            out_schema = other.out_schema;
            is_cfunc = other.is_cfunc;
            has_state = other.has_state;
            arrow_func_name = other.arrow_func_name;
        }
        return *this;
    }

    bool Equals(const FunctionData &other_p) const override {
        const auto &other = other_p.Cast<BodoScalarFunctionData>();
        return (other.args == this->args && other.is_cfunc == this->is_cfunc &&
                other.has_state == this->has_state &&
                other.arrow_func_name == this->arrow_func_name);
    }
    duckdb::unique_ptr<duckdb::FunctionData> Copy() const override {
        if (!arrow_func_name.empty()) {
            return duckdb::make_uniq<BodoScalarFunctionData>(out_schema,
                                                             arrow_func_name);
        } else {
            return duckdb::make_uniq<BodoScalarFunctionData>(
                args, out_schema, is_cfunc, has_state);
        }
    }

    PyObject *args;  // If present then a UDF.
    std::shared_ptr<arrow::Schema> out_schema;
    bool is_cfunc;
    bool has_state;
    std::string arrow_func_name;  // Non-empty means we’re using Arrow compute
};

/**
 * @brief Aggregate node data to pass around in DuckDB plans in
 * BoundAggregateExpression.
 *
 */
struct BodoAggFunctionData : public duckdb::FunctionData {
    BodoAggFunctionData(bool dropna, std::string name, PyObject *py_udf_args,
                        std::shared_ptr<arrow::Schema> out_schema)
        : out_schema(std::move(out_schema)),
          dropna(dropna),
          name(name),
          py_udf_args(py_udf_args) {
        Py_INCREF(py_udf_args);
    }

    ~BodoAggFunctionData() override { Py_DECREF(py_udf_args); }

    bool Equals(const FunctionData &other_p) const override {
        const BodoAggFunctionData &other = other_p.Cast<BodoAggFunctionData>();
        return (other.dropna == this->dropna && other.name == this->name &&
                other.py_udf_args == this->py_udf_args &&
                other.out_schema == this->out_schema);
    }
    duckdb::unique_ptr<duckdb::FunctionData> Copy() const override {
        return duckdb::make_uniq<BodoAggFunctionData>(
            this->dropna, this->name, this->py_udf_args, this->out_schema);
    }

    const std::shared_ptr<arrow::Schema> out_schema;
    const bool dropna;
    const std::string name;
    PyObject *py_udf_args;
};

/**
 * @brief Run Python scalar function on the input batch and return the
 * output table (single data column plus Index columns).
 *
 * @param input_batch input table batch
 * @param result_type The expected result type of the function
 * @param args Python arguments for the function
 * @param has_state whether the function has an initialization routine and
 *        state passed to each row evaluator
 * @param init_state if nullptr then initializer needs to be run, else use
 *        this value for state to pass to row evaluator
 * @return std::shared_ptr<table_info> output table from the Python function
 */
std::tuple<std::shared_ptr<table_info>, int64_t, int64_t, int64_t>
runPythonScalarFunction(std::shared_ptr<table_info> input_batch,
                        const std::shared_ptr<arrow::DataType> &result_type,
                        PyObject *args, bool has_state, PyObject *&init_state);

/**
 * @brief Run a cfunc scalar function on the input batch and return the
 * output table (single data column plus Index columns).
 *
 * @param input_batch input table batch
 * @param cfunc_ptr Pointer to the Cfunc to execute on the input batch.
 * @return std::shared_ptr<table_info> output table from the Python function
 */
std::shared_ptr<table_info> runCfuncScalarFunction(
    std::shared_ptr<table_info> input_batch, table_udf_t cfunc_ptr);

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
class JoinFilterColStats {
    using col_min_max_t = std::pair<std::shared_ptr<arrow::Scalar>,
                                    std::shared_ptr<arrow::Scalar>>;
    struct col_stats_collector {
        int64_t build_key_col;
        JoinState *join_state;
        std::optional<col_min_max_t> collect_min_max() const {
            std::unique_ptr<bodo::DataType> dt =
                join_state->build_table_schema->column_types[build_key_col]
                    ->copy();
            const auto &col_min_max = join_state->min_max_values[build_key_col];
            arrow::TimeUnit::type time_unit = arrow::TimeUnit::NANO;
            if (!col_min_max.has_value()) {
                return std::nullopt;
            }
            std::shared_ptr<arrow::Array> arrow_array = bodo_array_to_arrow(
                bodo::BufferPool::DefaultPtr(), col_min_max.value(), false,
                dt->timezone, time_unit, false,
                bodo::default_buffer_memory_manager());

            std::shared_ptr<arrow::Scalar> min_scalar =
                arrow_array->GetScalar(0).ValueOrDie();
            std::shared_ptr<arrow::Scalar> max_scalar =
                arrow_array->GetScalar(1).ValueOrDie();
            return std::make_optional<col_min_max_t>(min_scalar, max_scalar);
        }
    };

    const std::shared_ptr<std::unordered_map<int, JoinState *>> join_state_map;
    const JoinFilterProgramState join_filter_program_state;
    std::unordered_map<int, std::vector<col_stats_collector>>
        join_col_stats_map;

    std::optional<std::unordered_map<int, std::vector<col_min_max_t>>> result =
        std::nullopt;

   public:
    JoinFilterColStats(
        std::shared_ptr<std::unordered_map<int, JoinState *>> join_state_map,
        JoinFilterProgramState rtjf_state_map)
        : join_state_map(std::move(join_state_map)),
          join_filter_program_state(std::move(rtjf_state_map)) {}
    JoinFilterColStats() = default;

    std::unordered_map<int, std::vector<col_min_max_t>> collect_all() {
        if (result.has_value()) {
            return result.value();
        }
        result = std::unordered_map<int, std::vector<col_min_max_t>>{};

        for (const auto &[join_id, col_info] :
             this->join_filter_program_state) {
            auto join_state_it = join_state_map->find(join_id);
            if (join_state_it == join_state_map->end()) {
                throw std::runtime_error(
                    "JoinFilterColStats: join state not found for join id " +
                    std::to_string(join_id));
            }
            JoinState *join_state = join_state_it->second;
            for (size_t i = 0; i < col_info.filter_columns.size(); ++i) {
                if (col_info.filter_columns[i] < 0) {
                    continue;
                }
                const int64_t orig_build_key = col_info.orig_build_key_cols[i];
                join_col_stats_map[col_info.filter_columns[i]].push_back(
                    col_stats_collector{.build_key_col = orig_build_key,
                                        .join_state = join_state});
            }
        }

        for (const auto &[filter_col, collectors] : join_col_stats_map) {
            for (const auto &collector : collectors) {
                std::optional<col_min_max_t> col_min_max =
                    collector.collect_min_max();
                if (col_min_max.has_value()) {
                    result.value()[filter_col].push_back(col_min_max.value());
                }
            }
        }
        return result.value();
    }
};
