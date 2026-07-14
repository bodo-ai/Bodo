#pragma once

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/compute/function.h>
#include <arrow/compute/kernel.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <arrow/type_traits.h>
#include <future>
#include <rmm/cuda_stream_view.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>
#include "_util.h"
#include "duckdb/common/enums/expression_type.hpp"
#include "duckdb/planner/expression.hpp"
#include "operator.h"

#include <cmath>
#include <cstdint>
#include <cudf/types.hpp>
#include <memory>

#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/datetime.hpp>
#include <cudf/replace.hpp>
#include <cudf/round.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/search.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/strip.hpp>
#include <cudf/unary.hpp>

#include "duckdb/common/enums/expression_type.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/table_filter.hpp"

#define GPU_COLUMN std::unique_ptr<cudf::column>
#define GPU_SCALAR std::unique_ptr<cudf::scalar>

void EnsureModRegistered();

std::shared_ptr<arrow::Array> prepare_arrow_compute(GPU_COLUMN arr);

/**
 * @brief Superclass for possible results returned by nodes in Bodo
 *        Physical expression tree.
 *
 */
class ExprGPUResult {
   public:
    virtual ~ExprGPUResult() = default;
};

/**
 * @brief Result type for Physical expression tree nodes that return
 *        an entire table_info type.
 *
 */
class TableExprGPUResult : public ExprGPUResult {
   public:
    TableExprGPUResult(GPU_DATA val) : result(val) {}
    virtual ~TableExprGPUResult() = default;
    GPU_DATA result;
};

/**
 * @brief Result type for Physical expression tree nodes that return
 *        a single column in the cudf::column type.
 *
 */
class ArrayExprGPUResult : public ExprGPUResult {
   public:
    ArrayExprGPUResult(GPU_COLUMN val, std::string col)
        : result(std::move(val)), column_name(col) {}
    virtual ~ArrayExprGPUResult() = default;
    GPU_COLUMN result;
    const std::string column_name;
};

/**
 * @brief Result type for Physical expression tree nodes that return
 *        a single scalar, which for various reasons, including not
 *        having the complication of templatizing this class, is
 *        stored as a single-element cudf::column.
 *
 */
class ScalarExprGPUResult : public ExprGPUResult {
   public:
    ScalarExprGPUResult(GPU_SCALAR val) : result(std::move(val)) {}
    virtual ~ScalarExprGPUResult() = default;
    const GPU_SCALAR result;
};

template <typename U>
std::unique_ptr<cudf::scalar> make_cudf_scalar_from_value(
    U const &value, std::shared_ptr<StreamAndEvent> se) {
    using T = std::decay_t<U>;

    if constexpr (std::is_same_v<T, std::string>) {
        return std::make_unique<cudf::string_scalar>(value, true, se->stream);
    } else if constexpr (std::is_same_v<T, const char *>) {
        return std::make_unique<cudf::string_scalar>(std::string(value), true,
                                                     se->stream);
    } else if constexpr (std::is_same_v<T, bool>) {
        return std::make_unique<cudf::numeric_scalar<bool>>(value, true,
                                                            se->stream);
    } else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
        if constexpr (sizeof(T) == 1)
            return std::make_unique<cudf::numeric_scalar<int8_t>>(
                static_cast<int8_t>(value), true, se->stream);
        if constexpr (sizeof(T) == 2)
            return std::make_unique<cudf::numeric_scalar<int16_t>>(
                static_cast<int16_t>(value), true, se->stream);
        if constexpr (sizeof(T) == 4)
            return std::make_unique<cudf::numeric_scalar<int32_t>>(
                static_cast<int32_t>(value), true, se->stream);
        return std::make_unique<cudf::numeric_scalar<int64_t>>(
            static_cast<int64_t>(value), true, se->stream);
    } else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>) {
        if constexpr (sizeof(T) == 1)
            return std::make_unique<cudf::numeric_scalar<uint8_t>>(
                static_cast<uint8_t>(value), true, se->stream);
        if constexpr (sizeof(T) == 2)
            return std::make_unique<cudf::numeric_scalar<uint16_t>>(
                static_cast<uint16_t>(value), true, se->stream);
        if constexpr (sizeof(T) == 4)
            return std::make_unique<cudf::numeric_scalar<uint32_t>>(
                static_cast<uint32_t>(value), true, se->stream);
        return std::make_unique<cudf::numeric_scalar<uint64_t>>(
            static_cast<uint64_t>(value), true, se->stream);
    } else if constexpr (std::is_floating_point_v<T>) {
        if constexpr (sizeof(T) == 4)
            return std::make_unique<cudf::numeric_scalar<float>>(
                static_cast<float>(value), true, se->stream);
        return std::make_unique<cudf::numeric_scalar<double>>(
            static_cast<double>(value), true, se->stream);
    } else {
        static_assert(!sizeof(T),
                      "Unsupported type for make_cudf_scalar_from_value");
    }
    throw std::runtime_error("make_cudf_scalar_from_value failed: " +
                             std::string(typeid(T).name()));
}

/**
 * @brief Superclass for Bodo Physical expression tree nodes. Like duckdb
 *        it is convenient to store child nodes here because many expr
 *        node types have children.
 *
 */
class PhysicalGPUExpression {
   public:
    PhysicalGPUExpression() {}
    PhysicalGPUExpression(
        std::vector<std::shared_ptr<PhysicalGPUExpression>> &_children)
        : children(_children) {}
    virtual ~PhysicalGPUExpression() = default;

    /**
     * @brief Like a pipeline ProcessBatch but with more flexible return type.
     *        Process your children first and then yourself.  The input_batch
     *        from the pipeline needs to be passed down to the leaves of the
     *        expression tree so that source that use the input_batch have
     *        access.
     *
     */
    virtual std::shared_ptr<ExprGPUResult> ProcessBatch(
        GPU_DATA input_batch, std::shared_ptr<StreamAndEvent> se) = 0;

    static bool join_expr(cudf::column **left_table, cudf::column **right_table,
                          void **left_data, void **right_data,
                          void **left_null_bitmap, void **right_null_bitmap,
                          int64_t left_index, int64_t right_index);
    static void join_expr_batch(
        cudf::column **left_table, cudf::column **right_table, void **left_data,
        void **right_data, void **left_null_bitmap, void **right_null_bitmap,
        uint8_t *match_arr, int64_t left_index_start, int64_t left_index_end,
        int64_t right_index_start, int64_t right_index_end);

    static PhysicalGPUExpression *cur_join_expr;

    virtual arrow::Datum join_expr_internal(
        cudf::column **left_table, cudf::column **right_table, void **left_data,
        void **right_data, void **left_null_bitmap, void **right_null_bitmap,
        int64_t left_index, int64_t right_index) = 0;

    virtual void ReportMetrics(std::vector<MetricBase> &metrics_out) {};

   protected:
    std::vector<std::shared_ptr<PhysicalGPUExpression>> children;
};

/**
 * @brief Convert ExprGPUResult to arrow and run compute operation on it.
 *
 */
std::variant<GPU_COLUMN, GPU_SCALAR> do_cudf_compute_unary(
    std::shared_ptr<ExprGPUResult> left_res,
    const cudf::unary_operator &comparator, std::shared_ptr<StreamAndEvent> se,
    const arrow::compute::FunctionOptions *func_options = nullptr);

/**
 * @brief Convert two ExprGPUResults to arrow and run compute operation on them.
 *
 */
std::variant<GPU_COLUMN, GPU_SCALAR> do_cudf_compute_binary(
    std::shared_ptr<ExprGPUResult> left_res,
    std::shared_ptr<ExprGPUResult> right_res,
    const cudf::binary_operator &comparator, std::shared_ptr<StreamAndEvent> se,
    const cudf::data_type &cudf_result_type);

/**
 * @brief Convert ExprGPUResult to arrow and cast to the requested type.
 *
 */
std::variant<GPU_COLUMN, GPU_SCALAR> do_cudf_compute_cast(
    std::shared_ptr<ExprGPUResult> left_res, cudf::data_type &cudf_result_type,
    std::shared_ptr<StreamAndEvent> se);

/**
 * @brief Convert ExprGPUResult to arrow and run case compute on them.
 *
 */
GPU_COLUMN do_cudf_compute_case(std::shared_ptr<ExprGPUResult> when_res,
                                std::shared_ptr<ExprGPUResult> then_res,
                                std::shared_ptr<ExprGPUResult> else_res,
                                std::shared_ptr<StreamAndEvent> se);

/**
 * @brief Physical expression tree node type for comparisons resulting in
 *        boolean arrays.
 *
 */
class PhysicalGPUComparisonExpression : public PhysicalGPUExpression {
   public:
    PhysicalGPUComparisonExpression(
        std::shared_ptr<PhysicalGPUExpression> left,
        std::shared_ptr<PhysicalGPUExpression> right,
        duckdb::ExpressionType etype, duckdb::LogicalType _return_type) {
        return_type = duckdb_logicaltype_to_cudf(_return_type);

        children.push_back(left);
        children.push_back(right);
        switch (etype) {
            case duckdb::ExpressionType::COMPARE_EQUAL:
                comparator = cudf::binary_operator::EQUAL;
                break;
            case duckdb::ExpressionType::COMPARE_NOTEQUAL:
                comparator = cudf::binary_operator::NOT_EQUAL;
                break;
            case duckdb::ExpressionType::COMPARE_GREATERTHAN:
                comparator = cudf::binary_operator::GREATER;
                break;
            case duckdb::ExpressionType::COMPARE_LESSTHAN:
                comparator = cudf::binary_operator::LESS;
                break;
            case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
                comparator = cudf::binary_operator::GREATER_EQUAL;
                break;
            case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
                comparator = cudf::binary_operator::LESS_EQUAL;
                break;
            default:
                throw std::runtime_error(
                    "Unhandled comparison expression type.");
        }
    }

    virtual ~PhysicalGPUComparisonExpression() = default;

    /**
     * @brief How to process this expression tree node.
     *
     */
    std::shared_ptr<ExprGPUResult> ProcessBatch(
        GPU_DATA input_batch, std::shared_ptr<StreamAndEvent> se) override {
        // We know we have two children so process them first.
        std::shared_ptr<ExprGPUResult> left_res =
            children[0]->ProcessBatch(input_batch, se);
        std::shared_ptr<ExprGPUResult> right_res =
            children[1]->ProcessBatch(input_batch, se);

        auto result = do_cudf_compute_binary(left_res, right_res, comparator,
                                             se, return_type);
        std::shared_ptr<ExprGPUResult> ret;
        std::visit(
            [&](auto &vres) {
                using U = std::decay_t<decltype(vres)>;
                if constexpr (std::is_same_v<U, GPU_COLUMN>) {
                    ret = std::make_shared<ArrayExprGPUResult>(
                        std::move(vres),
                        "Comparison" +
                            std::to_string(static_cast<int>(comparator)));
                } else if constexpr (std::is_same_v<U, GPU_SCALAR>) {
                    ret =
                        std::make_shared<ScalarExprGPUResult>(std::move(vres));
                } else {
                    throw std::runtime_error(
                        "Got unknown type in PhysicalGPUCastExpression.");
                }
            },
            result);
        return ret;
    }

    arrow::Datum join_expr_internal(
        cudf::column **left_table, cudf::column **right_table, void **left_data,
        void **right_data, void **left_null_bitmap, void **right_null_bitmap,
        int64_t left_index, int64_t right_index) override {
        throw std::runtime_error(
            "PhysicalGPUNullExpression::join_expr_internal unimplemented ");
    }

   protected:
    cudf::data_type return_type;
    cudf::binary_operator comparator;
};

/**
 * @brief Physical expression tree node type for scalar constants.
 *
 */
template <typename T>
class PhysicalGPUNullExpression : public PhysicalGPUExpression {
   public:
    PhysicalGPUNullExpression(const T &val, bool no_scalars)
        : constant(val), generate_array(no_scalars) {}
    virtual ~PhysicalGPUNullExpression() = default;

    std::shared_ptr<ExprGPUResult> ProcessBatch(
        GPU_DATA input_batch, std::shared_ptr<StreamAndEvent> se) override {
        // The current rule is that if the expression infrastructure
        // is used for filtering then constants are treated as
        // scalars and if used for projection then constants become
        // full columns.  If used in a projection then generate_array
        // will be true and we generate an array the size of the
        // batch and return an ArrayExprGPUResult.
        std::unique_ptr<cudf::scalar> scalar;
        if constexpr (std::is_same_v<T, std::shared_ptr<arrow::Scalar>>) {
            auto make_cudf_res = arrow_scalar_to_cudf(constant, se->stream);
            scalar = std::move(make_cudf_res);
        } else {
            auto make_cudf_res = make_cudf_scalar_from_value(constant, se);
            scalar = std::move(make_cudf_res);
        }
        std::unique_ptr<cudf::scalar> invalid =
            make_invalid_cudf_scalar(scalar->type(), se->stream);

        if (generate_array) {
            std::size_t length = input_batch.table->num_rows();
            // create a column filled with that scalar
            std::unique_ptr<cudf::column> col =
                cudf::make_column_from_scalar(*invalid, length, se->stream);
            return std::make_shared<ArrayExprGPUResult>(
                std::move(col), std::string("Constant"));
        } else {
            return std::make_shared<ScalarExprGPUResult>(std::move(invalid));
        }
    }

    arrow::Datum join_expr_internal(
        cudf::column **left_table, cudf::column **right_table, void **left_data,
        void **right_data, void **left_null_bitmap, void **right_null_bitmap,
        int64_t left_index, int64_t right_index) override {
        throw std::runtime_error(
            "PhysicalGPUNullExpression::join_expr_internal unimplemented ");
    }

    friend std::ostream &operator<<(std::ostream &os,
                                    const PhysicalGPUNullExpression<T> &obj) {
        os << "PhysicalGPUNullExpression " << std::endl;
        return os;
    }

   private:
    const T constant;  // holds no real value, only for type
    const bool generate_array;
};

/**
 * @brief Physical expression tree node type for scalar constants.
 *
 */
template <typename T>
class PhysicalGPUConstantExpression : public PhysicalGPUExpression {
   public:
    PhysicalGPUConstantExpression(const T &val, bool no_scalars)
        : constant(val), generate_array(no_scalars) {}
    virtual ~PhysicalGPUConstantExpression() = default;

    virtual std::shared_ptr<ExprGPUResult> ProcessBatch(
        GPU_DATA input_batch, std::shared_ptr<StreamAndEvent> se) override {
        std::unique_ptr<cudf::scalar> scalar;
        // create cudf scalar from the constant
        if constexpr (std::is_same_v<T, std::shared_ptr<arrow::Scalar>>) {
            auto make_cudf_res = arrow_scalar_to_cudf(constant, se->stream);
            scalar = std::move(make_cudf_res);
        } else {
            auto make_cudf_res = make_cudf_scalar_from_value(constant, se);
            scalar = std::move(make_cudf_res);
        }

        if (generate_array) {
            std::size_t length = input_batch.table->num_rows();
            // create a column filled with that scalar
            std::unique_ptr<cudf::column> col =
                cudf::make_column_from_scalar(*scalar, length, se->stream);
            return std::make_shared<ArrayExprGPUResult>(
                std::move(col), std::string("Constant"));
        } else {
            return std::make_shared<ScalarExprGPUResult>(std::move(scalar));
        }
    }

    arrow::Datum join_expr_internal(
        cudf::column **left_table, cudf::column **right_table, void **left_data,
        void **right_data, void **left_null_bitmap, void **right_null_bitmap,
        int64_t left_index, int64_t right_index) override {
        throw std::runtime_error(
            "PhysicalGPUConstantExpression::join_expr_internal unimplemented ");
    }

    friend std::ostream &operator<<(
        std::ostream &os, const PhysicalGPUConstantExpression<T> &obj) {
        os << "PhysicalGPUConstantExpression " << obj.constant << std::endl;
        return os;
    }

   private:
    const T constant;
    const bool generate_array;
};

/**
 * @brief Physical expression tree node type for getting column from table.
 *
 */
class PhysicalGPUColumnRefExpression : public PhysicalGPUExpression {
   public:
    PhysicalGPUColumnRefExpression(size_t column,
                                   const std::string &_bound_name,
                                   bool _left_side = true)
        : col_idx(column), bound_name(_bound_name), left_side(_left_side) {}
    virtual ~PhysicalGPUColumnRefExpression() = default;

    std::shared_ptr<ExprGPUResult> ProcessBatch(
        GPU_DATA input_batch, std::shared_ptr<StreamAndEvent> se) override {
        GPU_COLUMN res_array = std::make_unique<cudf::column>(
            input_batch.table->view().column(col_idx), se->stream);

        std::vector<std::string> column_names =
            input_batch.schema->field_names();
        std::string column_name;
        if (column_names.size() > 0) {
            column_name = column_names[col_idx];
        } else {
            column_name = bound_name;
        }
        return std::make_shared<ArrayExprGPUResult>(std::move(res_array),
                                                    column_name);
    }

    arrow::Datum join_expr_internal(cudf::column **table, void **data,
                                    void **null_bitmap, int64_t index) {
        throw std::runtime_error(
            "PhysicalGPColumnRefExpression::join_expr_internal unimplemented ");
    }

    arrow::Datum join_expr_internal(
        cudf::column **left_table, cudf::column **right_table, void **left_data,
        void **right_data, void **left_null_bitmap, void **right_null_bitmap,
        int64_t left_index, int64_t right_index) override {
        throw std::runtime_error(
            "PhysicalGPColumnRefExpression::join_expr_internal unimplemented ");
    }

   protected:
    size_t col_idx;
    std::string bound_name;
    bool left_side;
};

/**
 * @brief Physical expression tree node type for conjunctions of
 *        boolean arrays.
 *
 */
class PhysicalGPUConjunctionExpression : public PhysicalGPUExpression {
   public:
    PhysicalGPUConjunctionExpression(
        std::shared_ptr<PhysicalGPUExpression> left,
        std::shared_ptr<PhysicalGPUExpression> right,
        duckdb::ExpressionType etype, duckdb::LogicalType _return_type) {
        return_type = duckdb_logicaltype_to_cudf(_return_type);

        children.push_back(left);
        children.push_back(right);
        switch (etype) {
            case duckdb::ExpressionType::CONJUNCTION_AND:
                comparator = cudf::binary_operator::LOGICAL_AND;
                break;
            case duckdb::ExpressionType::CONJUNCTION_OR:
                comparator = cudf::binary_operator::LOGICAL_OR;
                break;
            default:
                throw std::runtime_error(
                    "Unhandled conjunction expression type.");
        }
    }

    virtual ~PhysicalGPUConjunctionExpression() = default;

    /**
     * @brief How to process this expression tree node.
     *
     */
    std::shared_ptr<ExprGPUResult> ProcessBatch(
        GPU_DATA input_batch, std::shared_ptr<StreamAndEvent> se) override {
        // We know we have two children so process them first.
        std::shared_ptr<ExprGPUResult> left_res =
            children[0]->ProcessBatch(input_batch, se);
        std::shared_ptr<ExprGPUResult> right_res =
            children[1]->ProcessBatch(input_batch, se);

        auto result = do_cudf_compute_binary(left_res, right_res, comparator,
                                             se, return_type);
        std::shared_ptr<ExprGPUResult> ret;
        std::visit(
            [&](auto &vres) {
                using U = std::decay_t<decltype(vres)>;
                if constexpr (std::is_same_v<U, GPU_COLUMN>) {
                    ret = std::make_shared<ArrayExprGPUResult>(
                        std::move(vres),
                        "Conjunction" +
                            std::to_string(static_cast<int>(comparator)));
                } else if constexpr (std::is_same_v<U, GPU_SCALAR>) {
                    ret =
                        std::make_shared<ScalarExprGPUResult>(std::move(vres));
                } else {
                    throw std::runtime_error(
                        "Got unknown type in PhysicalGPUCastExpression.");
                }
            },
            result);
        return ret;
    }

    arrow::Datum join_expr_internal(
        cudf::column **left_table, cudf::column **right_table, void **left_data,
        void **right_data, void **left_null_bitmap, void **right_null_bitmap,
        int64_t left_index, int64_t right_index) override {
        throw std::runtime_error(
            "PhysicalGPColumnRefExpression::join_expr_internal unimplemented ");
    }

   protected:
    cudf::data_type return_type;
    cudf::binary_operator comparator;
};

/**
 * @brief Physical expression tree node type for casting.
 *
 */
class PhysicalGPUCastExpression : public PhysicalGPUExpression {
   public:
    PhysicalGPUCastExpression(std::shared_ptr<PhysicalGPUExpression> left,
                              duckdb::LogicalType _return_type) {
        children.push_back(left);
        return_type = duckdb_logicaltype_to_cudf(_return_type);
    }

    virtual ~PhysicalGPUCastExpression() = default;

    /**
     * @brief How to process this expression tree node.
     *
     */
    std::shared_ptr<ExprGPUResult> ProcessBatch(
        GPU_DATA input_batch, std::shared_ptr<StreamAndEvent> se) override {
        // Process child first.
        std::shared_ptr<ExprGPUResult> left_res =
            children[0]->ProcessBatch(input_batch, se);
        auto result = do_cudf_compute_cast(left_res, return_type, se);
        std::shared_ptr<ExprGPUResult> ret;
        std::visit(
            [&](auto &vres) {
                using U = std::decay_t<decltype(vres)>;
                if constexpr (std::is_same_v<U, GPU_COLUMN>) {
                    ret = std::make_shared<ArrayExprGPUResult>(std::move(vres),
                                                               "Cast");
                } else if constexpr (std::is_same_v<U, GPU_SCALAR>) {
                    ret =
                        std::make_shared<ScalarExprGPUResult>(std::move(vres));
                } else {
                    throw std::runtime_error(
                        "Got unknown type in PhysicalGPUCastExpression.");
                }
            },
            result);
        return ret;
    }

    arrow::Datum join_expr_internal(
        cudf::column **left_table, cudf::column **right_table, void **left_data,
        void **right_data, void **left_null_bitmap, void **right_null_bitmap,
        int64_t left_index, int64_t right_index) override {
        throw std::runtime_error(
            "PhysicalGPUCastExpression::join_expr_internal unimplemented ");
    }

   protected:
    cudf::data_type return_type;
};

enum GPU_UNARY_MODE {
    CUDF_UNARY = 0,
    IS_NOT_NULL = 1,
    IS_NULL = 2,
    IS_TRUE = 3
};

/**
 * @brief Physical expression tree node type for unary of array.
 *
 */
class PhysicalGPUUnaryExpression : public PhysicalGPUExpression {
   public:
    PhysicalGPUUnaryExpression(std::shared_ptr<PhysicalGPUExpression> left,
                               duckdb::ExpressionType etype) {
        children.push_back(left);
        switch (etype) {
            case duckdb::ExpressionType::OPERATOR_NOT:
                comparator = cudf::unary_operator::NOT;
                break;
            case duckdb::ExpressionType::OPERATOR_IS_NOT_NULL:
                // Will do separate is_null and then apply NOT.
                // Will not show up as right column name.
                comparator = cudf::unary_operator::NOT;
                mode = GPU_UNARY_MODE::IS_NOT_NULL;
                break;
            case duckdb::ExpressionType::OPERATOR_IS_NULL:
                // Will not show up as right column name.
                comparator = cudf::unary_operator::NOT;  // unused
                mode = GPU_UNARY_MODE::IS_NULL;
                break;
            case duckdb::ExpressionType::OPERATOR_IS_TRUE:
                comparator = cudf::unary_operator::NOT;  // unused
                mode = GPU_UNARY_MODE::IS_TRUE;
                break;
            case duckdb::ExpressionType::OPERATOR_NEG:
                comparator = cudf::unary_operator::NEGATE;
                break;
            default:
                throw std::runtime_error("Unhandled unary op expression type.");
        }
    }
    PhysicalGPUUnaryExpression(std::shared_ptr<PhysicalGPUExpression> left,
                               std::string &opstr)
        : opstr(opstr) {
        children.push_back(left);
        if (opstr == "floor") {
            comparator = cudf::unary_operator::FLOOR;
        } else if (opstr == "ceil") {
            comparator = cudf::unary_operator::CEIL;
        } else if (opstr == "abs") {
            comparator = cudf::unary_operator::ABS;
        } else if (opstr == "sqrt") {
            comparator = cudf::unary_operator::SQRT;
        } else if (opstr == "cbrt") {
            comparator = cudf::unary_operator::CBRT;
        } else if (opstr == "exp") {
            comparator = cudf::unary_operator::EXP;
        } else if (opstr == "ln") {
            comparator = cudf::unary_operator::LOG;
        } else if (opstr == "trunc" || opstr == "log10" || opstr == "log2" ||
                   opstr == "sign") {
            // Handled as special cases in ProcessBatch
        } else {
            throw std::runtime_error("Unhandled unary expression opstr " +
                                     opstr);
        }
    }

    virtual ~PhysicalGPUUnaryExpression() = default;

    /**
     * @brief How to process this expression tree node.
     *
     */
    std::shared_ptr<ExprGPUResult> ProcessBatch(
        GPU_DATA input_batch, std::shared_ptr<StreamAndEvent> se) override {
        // Process child first.
        std::shared_ptr<ExprGPUResult> left_res =
            children[0]->ProcessBatch(input_batch, se);
        std::variant<GPU_COLUMN, GPU_SCALAR> op_res;
        if (mode == GPU_UNARY_MODE::IS_NOT_NULL) {
            std::shared_ptr<ArrayExprGPUResult> left_as_array =
                std::dynamic_pointer_cast<ArrayExprGPUResult>(left_res);
            if (left_as_array) {
                auto is_null_res = cudf::is_null(left_as_array->result->view());
                op_res = cudf::unary_operation(is_null_res->view(), comparator,
                                               se->stream);
            } else {
                throw std::runtime_error("Left must be array for is_not_null.");
            }
        } else if (mode == GPU_UNARY_MODE::IS_NULL) {
            std::shared_ptr<ArrayExprGPUResult> left_as_array =
                std::dynamic_pointer_cast<ArrayExprGPUResult>(left_res);
            if (left_as_array) {
                op_res = cudf::is_null(left_as_array->result->view());
            } else {
                throw std::runtime_error("Left must be array for is_null.");
            }
        } else if (mode == GPU_UNARY_MODE::IS_TRUE) {
            std::shared_ptr<ArrayExprGPUResult> left_as_array =
                std::dynamic_pointer_cast<ArrayExprGPUResult>(left_res);
            if (left_as_array) {
                // Cast to boolean column if necessary.
                std::unique_ptr<cudf::column> bool_col_ptr;
                if (left_as_array->result->type().id() !=
                    cudf::type_id::BOOL8) {
                    // Cast to boolean (this will produce a new column)
                    bool_col_ptr = cudf::cast(
                        left_as_array->result->view(),
                        cudf::data_type{cudf::type_id::BOOL8}, se->stream);
                    if (!bool_col_ptr) {
                        throw std::runtime_error("cast to BOOL8 failed");
                    }
                }

                cudf::column_view bool_view =
                    bool_col_ptr ? bool_col_ptr->view()
                                 : left_as_array->result->view();

                // Create a boolean scalar 'false' to replace nulls
                std::unique_ptr<cudf::scalar> false_scalar =
                    cudf::make_numeric_scalar(
                        cudf::data_type{cudf::type_id::BOOL8});
                // set the scalar value to false
                static_cast<cudf::numeric_scalar<bool> *>(false_scalar.get())
                    ->set_value(false);

                // Replace nulls with false.
                std::unique_ptr<cudf::column> result =
                    cudf::replace_nulls(bool_view, *false_scalar, se->stream);
                if (!result) {
                    throw std::runtime_error("fill_nulls failed");
                }
                op_res = std::move(result);
            } else {
                throw std::runtime_error("Left must be array for is_true.");
            }
        } else if (opstr == "trunc") {
            // Special case since libcudf doesn't have a trunc function
            std::shared_ptr<ArrayExprGPUResult> left_as_array =
                std::dynamic_pointer_cast<ArrayExprGPUResult>(left_res);
            if (left_as_array) {
                cudf::data_type original_type = left_as_array->result->type();
                // As a workaround, so we cast to int to truncate
                auto truncated_int64 = cudf::cast(
                    left_as_array->result->view(),
                    cudf::data_type(cudf::type_id::INT64), se->stream);
                // Preserve original type
                op_res = cudf::cast(truncated_int64->view(), original_type,
                                    se->stream);
            } else {
                throw std::runtime_error("trunc: Input must be array to cast.");
            }
        } else if (opstr == "log10" || opstr == "log2") {
            // Use change of base formula: log10(x) = ln(x)/ln(10)
            std::shared_ptr<ArrayExprGPUResult> left_as_array =
                std::dynamic_pointer_cast<ArrayExprGPUResult>(left_res);
            if (left_as_array) {
                int log_base;
                if (opstr == "log10") {
                    log_base = 10;
                } else {
                    log_base = 2;
                }

                // ln(x)
                auto ln_of_left_res = cudf::unary_operation(
                    left_as_array->result->view(), cudf::unary_operator::LOG,
                    se->stream);

                // Make scalar of ln(10) or ln(2)
                GPU_SCALAR ln_of_base_scalar =
                    std::make_unique<cudf::numeric_scalar<float>>(
                        std::log(log_base), true, se->stream);

                // ln(x)/ln(10) or ln(x)/ln(2)
                op_res = cudf::binary_operation(
                    ln_of_left_res->view(), *ln_of_base_scalar,
                    cudf::binary_operator::TRUE_DIV, ln_of_left_res->type(),
                    se->stream);
            } else {
                throw std::runtime_error("Left must be array for " + opstr);
            }
        } else if (opstr == "sign") {
            std::shared_ptr<ArrayExprGPUResult> left_as_array =
                std::dynamic_pointer_cast<ArrayExprGPUResult>(left_res);
            if (left_as_array) {
                // Make scalar of 0 to compare against
                GPU_SCALAR zero_scalar =
                    std::make_unique<cudf::numeric_scalar<float>>(0.0, true,
                                                                  se->stream);

                // Get whether the input is strictly positive or negative
                auto is_positive = cudf::binary_operation(
                    left_as_array->result->view(), *zero_scalar,
                    cudf::binary_operator::GREATER,
                    cudf::data_type{cudf::type_id::BOOL8});
                auto is_negative = cudf::binary_operation(
                    left_as_array->result->view(), *zero_scalar,
                    cudf::binary_operator::LESS,
                    cudf::data_type{cudf::type_id::BOOL8});

                // Return int8 for integer input; otherwise, retain the float
                // input type
                cudf::data_type left_type = left_as_array->result->type();
                cudf::data_type return_type;
                if (cudf::is_floating_point(left_type)) {
                    return_type = left_type;
                } else {
                    return_type = cudf::data_type{cudf::type_id::INT8};
                }

                // Computes is_positive - is_negative.
                // True is treated as 1 and False as 0. Hence:
                // Positive input: 1 - 0 = 1
                // Negative input: 0 - 1 = -1
                // Zero input: 0 - 0 = 0
                op_res = cudf::binary_operation(
                    is_positive->view(), is_negative->view(),
                    cudf::binary_operator::SUB,
                    return_type  // Get output as number in (-1, 0, 1)
                );
            } else {
                throw std::runtime_error("Left must be array for " + opstr);
            }
        } else {
            op_res = do_cudf_compute_unary(left_res, comparator, se);
        }
        std::shared_ptr<ExprGPUResult> ret;
        std::visit(
            [&](auto &vres) {
                using U = std::decay_t<decltype(vres)>;
                if constexpr (std::is_same_v<U, GPU_COLUMN>) {
                    ret = std::make_shared<ArrayExprGPUResult>(
                        std::move(vres),
                        "Unary" + std::to_string(static_cast<int>(comparator)));
                } else if constexpr (std::is_same_v<U, GPU_SCALAR>) {
                    ret =
                        std::make_shared<ScalarExprGPUResult>(std::move(vres));
                } else {
                    throw std::runtime_error(
                        "Got unknown type in PhysicalGPUCastExpression.");
                }
            },
            op_res);
        return ret;
    }

    arrow::Datum join_expr_internal(
        cudf::column **left_table, cudf::column **right_table, void **left_data,
        void **right_data, void **left_null_bitmap, void **right_null_bitmap,
        int64_t left_index, int64_t right_index) override {
        throw std::runtime_error(
            "PhysicalGPUCastExpression::join_expr_internal unimplemented ");
    }

   protected:
    std::string opstr;  // opstr, if applicable
    cudf::unary_operator comparator;
    GPU_UNARY_MODE mode = GPU_UNARY_MODE::CUDF_UNARY;
};

/**
 * @brief Physical expression tree node type for binary op non-boolean arrays.
 *
 */
class PhysicalGPUBinaryExpression : public PhysicalGPUExpression {
   public:
    PhysicalGPUBinaryExpression(std::shared_ptr<PhysicalGPUExpression> left,
                                std::shared_ptr<PhysicalGPUExpression> right,
                                duckdb::ExpressionType etype,
                                duckdb::LogicalType _return_type) {
        return_type = duckdb_logicaltype_to_cudf(_return_type);

        children.push_back(left);
        children.push_back(right);
        switch (etype) {
            default:
                throw std::runtime_error(
                    "Unhandled binary expression type " +
                    std::to_string(static_cast<int>(etype)));
        }
    }

    PhysicalGPUBinaryExpression(std::shared_ptr<PhysicalGPUExpression> left,
                                std::shared_ptr<PhysicalGPUExpression> right,
                                const std::string &opstr,
                                duckdb::LogicalType _return_type)
        : opstr(opstr) {
        return_type = duckdb_logicaltype_to_cudf(_return_type);

        children.push_back(left);
        children.push_back(right);
        if (opstr == "+") {
            comparator = cudf::binary_operator::ADD;
        } else if (opstr == "-") {
            comparator = cudf::binary_operator::SUB;
        } else if (opstr == "*") {
            comparator = cudf::binary_operator::MUL;
        } else if (opstr == "/") {
            comparator = cudf::binary_operator::TRUE_DIV;
        } else if (opstr == "//") {
            comparator = cudf::binary_operator::FLOOR_DIV;
        } else if (opstr == "%") {
            comparator = cudf::binary_operator::MOD;
        } else if (opstr == "power") {
            comparator = cudf::binary_operator::POW;
        } else {
            throw std::runtime_error("Unhandled binary expression opstr " +
                                     opstr);
        }
    }

    virtual ~PhysicalGPUBinaryExpression() = default;

    /**
     * @brief How to process this expression tree node.
     *
     */
    std::shared_ptr<ExprGPUResult> ProcessBatch(
        GPU_DATA input_batch, std::shared_ptr<StreamAndEvent> se) override {
        // We know we have two children so process them first.
        std::shared_ptr<ExprGPUResult> left_res =
            children[0]->ProcessBatch(input_batch, se);
        std::shared_ptr<ExprGPUResult> right_res =
            children[1]->ProcessBatch(input_batch, se);
        std::variant<GPU_COLUMN, GPU_SCALAR> op_res;

        if (opstr == "%") {
            // Special case since cudf supports MOD, but does not return null
            // when modulus is 0

            // Compute the regular modulo result
            auto mod_result = do_cudf_compute_binary(
                left_res, right_res, comparator, se, return_type);
            auto &mod_result_col = std::get<GPU_COLUMN>(mod_result);

            // Ensure we handle both the scalar and column case
            GPU_COLUMN modulus_column;
            auto right_as_array =
                std::dynamic_pointer_cast<ArrayExprGPUResult>(right_res);
            if (right_as_array) {
                modulus_column = std::move(right_as_array->result);
            } else {
                // Modulus is a scalar; broadcast it to a column before
                // comparison
                auto right_as_scalar =
                    std::dynamic_pointer_cast<ScalarExprGPUResult>(right_res);
                modulus_column = cudf::make_column_from_scalar(
                    *(right_as_scalar->result), mod_result_col->size());
            }

            // Create zero scalar for comparison
            GPU_SCALAR zero_scalar =
                std::make_unique<cudf::numeric_scalar<float>>(0.0, true,
                                                              se->stream);
            // Check if the modulus is nonzero
            GPU_COLUMN is_modulus_nonzero =
                cudf::binary_operation(modulus_column->view(), *zero_scalar,
                                       cudf::binary_operator::NOT_EQUAL,
                                       cudf::data_type{cudf::type_id::BOOL8});

            // Create a null scalar of the desired return type.
            auto null_scalar =
                make_invalid_cudf_scalar(mod_result_col->type(), se->stream);

            // Return NULL if modulus is zero, else return the cudf MOD result
            op_res = cudf::copy_if_else(mod_result_col->view(), *null_scalar,
                                        *is_modulus_nonzero);
        } else {
            op_res = do_cudf_compute_binary(left_res, right_res, comparator, se,
                                            return_type);
        }

        std::shared_ptr<ExprGPUResult> ret;
        std::visit(
            [&](auto &vres) {
                using U = std::decay_t<decltype(vres)>;
                if constexpr (std::is_same_v<U, GPU_COLUMN>) {
                    ret = std::make_shared<ArrayExprGPUResult>(
                        std::move(vres),
                        "Binary" +
                            std::to_string(static_cast<int>(comparator)));
                } else if constexpr (std::is_same_v<U, GPU_SCALAR>) {
                    ret =
                        std::make_shared<ScalarExprGPUResult>(std::move(vres));
                } else {
                    throw std::runtime_error(
                        "Got unknown type in PhysicalGPUCastExpression.");
                }
            },
            op_res);
        return ret;
    }

    arrow::Datum join_expr_internal(
        cudf::column **left_table, cudf::column **right_table, void **left_data,
        void **right_data, void **left_null_bitmap, void **right_null_bitmap,
        int64_t left_index, int64_t right_index) override {
        throw std::runtime_error(
            "PhysicalGPUCastExpression::join_expr_internal unimplemented ");
    }

   protected:
    std::string opstr;  // opstr, if applicable
    cudf::data_type return_type;
    cudf::binary_operator comparator;
};

/**
 * @brief Physical expression tree node type for case expressions.
 *
 */
class PhysicalGPUCaseExpression : public PhysicalGPUExpression {
   public:
    PhysicalGPUCaseExpression(
        std::shared_ptr<PhysicalGPUExpression> when_expr,
        std::shared_ptr<PhysicalGPUExpression> then_expr,
        std::shared_ptr<PhysicalGPUExpression> else_expr) {
        children.push_back(when_expr);
        children.push_back(then_expr);
        children.push_back(else_expr);
    }

    virtual ~PhysicalGPUCaseExpression() = default;

    /**
     * @brief How to process this expression tree node.
     *
     */
    std::shared_ptr<ExprGPUResult> ProcessBatch(
        GPU_DATA input_batch, std::shared_ptr<StreamAndEvent> se) override {
        // Process children first.
        std::shared_ptr<ExprGPUResult> when_res =
            children[0]->ProcessBatch(input_batch, se);
        std::shared_ptr<ExprGPUResult> then_res =
            children[1]->ProcessBatch(input_batch, se);
        std::shared_ptr<ExprGPUResult> else_res =
            children[2]->ProcessBatch(input_batch, se);

        auto result = do_cudf_compute_case(when_res, then_res, else_res, se);
        return std::make_shared<ArrayExprGPUResult>(std::move(result), "Case");
    }

    arrow::Datum join_expr_internal(
        cudf::column **left_table, cudf::column **right_table, void **left_data,
        void **right_data, void **left_null_bitmap, void **right_null_bitmap,
        int64_t left_index, int64_t right_index) override {
        throw std::runtime_error(
            "PhysicalGPUCastExpression::join_expr_internal unimplemented ");
    }
};

/**
 * @brief Convert duckdb expression tree to Bodo physical expression tree.
 *
 * @param expr - the root of input duckdb expression tree
 * @param col_ref_map - mapping of table and column indices to overall index
 * @param no_scalars - true if batch sized arrays should be generated for consts
 * @return the root of output Bodo Physical expression tree
 */
std::shared_ptr<PhysicalGPUExpression> buildPhysicalGPUExprTree(
    duckdb::unique_ptr<duckdb::Expression> &expr,
    std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t> &col_ref_map,
    bool no_scalars = false);

std::shared_ptr<PhysicalGPUExpression> buildPhysicalGPUExprTree(
    duckdb::Expression &expr,
    std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t> &col_ref_map,
    bool no_scalars = false);

bool gpu_capable(duckdb::unique_ptr<duckdb::Expression> &expr);

bool gpu_capable(duckdb::Expression &expr);

struct PhysicalGPUArrowExpressionMetrics {
    using timer_t = MetricBase::TimerValue;
    timer_t arrow_compute_time = 0;
};

/**
 * @brief Physical expression tree node type for Arrow Compute functions that
 * are also available in cudf.
 *
 */
class PhysicalGPUArrowExpression : public PhysicalGPUExpression {
   public:
    PhysicalGPUArrowExpression(
        std::vector<std::shared_ptr<PhysicalGPUExpression>> &children,
        BodoScalarFunctionData &_scalar_func_data,
        duckdb::LogicalType _result_type)
        : PhysicalGPUExpression(children),
          scalar_func_data(_scalar_func_data),
          result_type(std::move(_result_type)) {
        if (scalar_func_data.arrow_func_name != "ends_with" &&
            scalar_func_data.arrow_func_name != "starts_with" &&
            scalar_func_data.arrow_func_name != "match_substring_regex" &&
            scalar_func_data.arrow_func_name != "match_substring_regex_first" &&
            scalar_func_data.arrow_func_name != "utf8_slice_codeunits" &&
            scalar_func_data.arrow_func_name != "utf8_trim_whitespace" &&
            scalar_func_data.arrow_func_name != "utf8_trim" &&
            scalar_func_data.arrow_func_name != "year" &&
            scalar_func_data.arrow_func_name != "round" &&
            scalar_func_data.arrow_func_name != "is_null" &&
            scalar_func_data.arrow_func_name != "is_not_null" &&
            scalar_func_data.arrow_func_name != "is_in") {
            throw std::runtime_error(
                "PhysicalGPUArrowExpression only supports ends_with, "
                "starts_with, match_substring_regex, "
                "match_substring_regex_first, "
                "year, round, is_null, is_not_null and is_in for now.");
        }
        if (scalar_func_data.arrow_func_name == "ends_with" ||
            scalar_func_data.arrow_func_name == "starts_with" ||
            scalar_func_data.arrow_func_name == "match_substring_regex" ||
            scalar_func_data.arrow_func_name == "match_substring_regex_first" ||
            scalar_func_data.arrow_func_name == "utf8_trim") {
            extract_string_arg_from_python();
        } else if (scalar_func_data.arrow_func_name == "utf8_trim_whitespace") {
            // Empty string which indicates strip whitespace characters in
            // cudf::strings::strip()
            str_scalar_in = std::make_shared<cudf::string_scalar>("", true);
        } else if (scalar_func_data.arrow_func_name == "round") {
            arrow::compute::RoundMode arrow_round_mode;
            std::tie(round_ndigits, arrow_round_mode) =
                get_py_round_args(scalar_func_data.args);
            if (arrow_round_mode == arrow::compute::RoundMode::HALF_TO_EVEN) {
                round_mode = cudf::rounding_method::HALF_EVEN;
            } else if (arrow_round_mode ==
                       arrow::compute::RoundMode::HALF_TOWARDS_INFINITY) {
                round_mode = cudf::rounding_method::HALF_UP;
            } else {
                throw std::invalid_argument(
                    "Only half-to-even and half-away-from-zero rounding modes "
                    "supported on GPU.");
            }
        } else if (scalar_func_data.arrow_func_name == "utf8_slice_codeunits") {
            extract_slice_arg_from_python();
        } else if (scalar_func_data.arrow_func_name == "is_in") {
            extract_isin_arg_from_python();
        }
    }

    /**
     * @brief How to process this expression tree node.
     *
     */
    std::shared_ptr<ExprGPUResult> ProcessBatch(
        GPU_DATA input_batch, std::shared_ptr<StreamAndEvent> se) override {
        std::shared_ptr<ExprGPUResult> in_res =
            children[0]->ProcessBatch(input_batch, se);

        std::shared_ptr<ArrayExprGPUResult> in_as_array =
            std::dynamic_pointer_cast<ArrayExprGPUResult>(in_res);

        std::unique_ptr<cudf::column> result;
        if (scalar_func_data.arrow_func_name == "ends_with") {
            result = cudf::strings::ends_with(in_as_array->result->view(),
                                              *str_scalar_in, se->stream);
        } else if (scalar_func_data.arrow_func_name == "starts_with") {
            result = cudf::strings::starts_with(in_as_array->result->view(),
                                                *str_scalar_in, se->stream);
        } else if (scalar_func_data.arrow_func_name ==
                   "match_substring_regex") {
            result = cudf::strings::contains_re(in_as_array->result->view(),
                                                *regex_prog, se->stream);
        } else if (scalar_func_data.arrow_func_name ==
                   "match_substring_regex_first") {
            result = cudf::strings::matches_re(in_as_array->result->view(),
                                               *regex_prog, se->stream);
        } else if (scalar_func_data.arrow_func_name == "utf8_slice_codeunits") {
            result = cudf::strings::slice_strings(
                in_as_array->result->view(), start, stop, step, se->stream);
        } else if (scalar_func_data.arrow_func_name == "utf8_trim_whitespace" ||
                   scalar_func_data.arrow_func_name == "utf8_trim") {
            result = cudf::strings::strip(in_as_array->result->view(),
                                          cudf::strings::side_type::BOTH,
                                          *str_scalar_in, se->stream);
        } else if (scalar_func_data.arrow_func_name == "year") {
            result = cudf::datetime::extract_datetime_component(
                in_as_array->result->view(),
                cudf::datetime::datetime_component::YEAR, se->stream);
            // Cast int16 to int64 to match frontend (and CPU side)
            result =
                cudf::cast(result->view(),
                           cudf::data_type(cudf::type_id::INT64), se->stream);
        } else if (scalar_func_data.arrow_func_name == "round") {
// NOTE: cudf::round is deprecated but still used here in cudf:
// https://github.com/rapidsai/cudf/blob/1ed06c6105fb31bba18fc7cf69e2529f9c6a7a22/python/cudf/cudf/core/column/numerical_base.py#L252
// cudf::round_decimal doesn't support floating point data for some reason.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
            result = cudf::round(in_as_array->result->view(), round_ndigits,
                                 round_mode, se->stream);
#pragma GCC diagnostic pop
        } else if (scalar_func_data.arrow_func_name == "is_null") {
            result = cudf::is_null(in_as_array->result->view(), se->stream);
        } else if (scalar_func_data.arrow_func_name == "is_not_null") {
            result = cudf::is_valid(in_as_array->result->view(), se->stream);

        } else if (scalar_func_data.arrow_func_name == "is_in") {
            result = cudf::contains(isin_data->get_column(0).view(),
                                    in_as_array->result->view(), se->stream);
            // handle nulls in input to match Pandas similar to cudf:
            // https://github.com/rapidsai/cudf/blob/1fd16fda34a6e78777330f4e02bdd122a6977a22/python/cudf/cudf/core/column/column.py#L2164
            if (in_as_array->result->has_nulls()) {
                std::unique_ptr<cudf::scalar> fill_scalar =
                    arrow_scalar_to_cudf(
                        arrow::MakeScalar(arrow::boolean(),
                                          isin_data->get_column(0).has_nulls())
                            .ValueOrDie(),
                        se->stream);
                result = cudf::replace_nulls(result->view(), *fill_scalar,
                                             se->stream);
            }
        } else {
            throw std::runtime_error(
                fmt::format("Unsupported Arrow function: {}",
                            scalar_func_data.arrow_func_name));
        }

        std::shared_ptr<ExprGPUResult> ret =
            std::make_shared<ArrayExprGPUResult>(
                std::move(result),
                "Arrow Compute " + scalar_func_data.arrow_func_name);
        return ret;
    }

    arrow::Datum join_expr_internal(
        cudf::column **left_table, cudf::column **right_table, void **left_data,
        void **right_data, void **left_null_bitmap, void **right_null_bitmap,
        int64_t left_index, int64_t right_index) override {
        throw std::runtime_error(
            "PhysicalGPUArrowExpression::join_expr_internal unimplemented");
    }

    void ReportMetrics(std::vector<MetricBase> &metrics_out) override {
        metrics_out.push_back(
            TimerMetric("arrow_compute_time", metrics.arrow_compute_time));
    }

   protected:
    BodoScalarFunctionData scalar_func_data;
    const duckdb::LogicalType result_type;
    PhysicalGPUArrowExpressionMetrics metrics;

    void extract_string_arg_from_python() {
        if (scalar_func_data.arrow_func_name == "match_substring_regex" ||
            scalar_func_data.arrow_func_name == "match_substring_regex_first") {
            assert_py_args_is_tuple(scalar_func_data.args,
                                    scalar_func_data.arrow_func_name.c_str());
            size_t num_args = PyTuple_Size(scalar_func_data.args);
            const char *c_str;
            bool ignore_case = false;
            std::string sstr;
            if (num_args == 1) {
                // Only string was passed
                c_str = get_py_single_arg_as_cstr(
                    scalar_func_data.args,
                    scalar_func_data.arrow_func_name.c_str());
                sstr = std::string(c_str);
            } else {
                // string and ignore_case passed
                std::tie(c_str, ignore_case) = get_py_args_as_types(
                    scalar_func_data.args,
                    scalar_func_data.arrow_func_name.c_str(),
                    get_py_object_as_cstr, get_py_object_as_bool);
                sstr = std::string(c_str);
                if (ignore_case) {
                    sstr = "(?i)" + sstr;
                }
            }

            regex_prog = cudf::strings::regex_program::create(sstr);
        } else {
            const char *c_str = get_py_single_arg_as_cstr(
                scalar_func_data.args,
                scalar_func_data.arrow_func_name.c_str());

            str_scalar_in =
                std::make_shared<cudf::string_scalar>(std::string(c_str), true);
        }
    }

    void extract_slice_arg_from_python() {
        std::tie(start, stop, step) =
            get_py_slice_args<cudf::size_type>(scalar_func_data.args);
    }

    void extract_isin_arg_from_python() {
        std::shared_ptr<arrow::Array> values_array =
            get_py_isin_arg_as_arrow_array(scalar_func_data.args);

        // Convert isin input to cudf table
        std::shared_ptr<arrow::Table> values_table = arrow::Table::Make(
            arrow::schema({arrow::field("values", values_array->type())}),
            {values_array});
        GPU_DATA gpu_data = convertArrowTableToGPU(values_table, nullptr);

        isin_data = gpu_data.table;
    }

    // Keeping reference to the cudf string scalar created from the Python
    // string argument to ensure it stays alive during processing.
    std::shared_ptr<cudf::string_scalar> str_scalar_in;

    // Needed for match_substring_regex
    std::shared_ptr<cudf::strings::regex_program> regex_prog;

    int32_t round_ndigits = 0;
    cudf::rounding_method round_mode = cudf::rounding_method::HALF_EVEN;

    // str.slice() arguments
    int64_t start = 0;
    int64_t stop = 0;
    int64_t step = 0;

    // isin argument
    std::shared_ptr<cudf::table> isin_data;
};

struct PhysicalGPUUDFExpressionMetrics {
    using timer_t = MetricBase::TimerValue;
    timer_t cpp_to_py_time = 0;
    timer_t udf_execution_time = 0;
    timer_t py_to_cpp_time = 0;
};

/**
 * @brief Physical expression tree node type for UDF.
 *
 */
class PhysicalGPUUDFExpression : public PhysicalGPUExpression {
   public:
    PhysicalGPUUDFExpression(
        std::vector<std::shared_ptr<PhysicalGPUExpression>> &children,
        BodoScalarFunctionData &_scalar_func_data,
        duckdb::LogicalType _return_type)
        : PhysicalGPUExpression(children),
          scalar_func_data(_scalar_func_data),
          cfunc_ptr(nullptr),
          init_state(nullptr) {
        return_type = duckdb_logicaltype_to_cudf(_return_type);
        throw std::runtime_error("PhysicalGPUUDFExpression unimplemented ");
    }

    virtual ~PhysicalGPUUDFExpression() {
        if (init_state != nullptr) {
            Py_DECREF(init_state);
        }
    }

    /**
     * @brief How to process this expression tree node.
     *
     */
    std::shared_ptr<ExprGPUResult> ProcessBatch(
        GPU_DATA input_batch, std::shared_ptr<StreamAndEvent> se) override;

    arrow::Datum join_expr_internal(
        cudf::column **left_table, cudf::column **right_table, void **left_data,
        void **right_data, void **left_null_bitmap, void **right_null_bitmap,
        int64_t left_index, int64_t right_index) override {
        throw std::runtime_error(
            "PhysicalGPUUDFExpression::join_expr_internal unimplemented ");
    }
    void ReportMetrics(std::vector<MetricBase> &metrics_out) override {
        metrics_out.push_back(
            TimerMetric("run_func_cpp_to_py_time", metrics.cpp_to_py_time));
        metrics_out.push_back(TimerMetric("run_func_udf_execution_time",
                                          metrics.udf_execution_time));
        metrics_out.push_back(
            TimerMetric("run_func_py_to_cpp_time", metrics.py_to_cpp_time));
    }

   protected:
    BodoScalarFunctionData scalar_func_data;
    cudf::data_type return_type;
    PhysicalGPUUDFExpressionMetrics metrics;
    std::future<table_udf_t> compile_future;
    table_udf_t cfunc_ptr;
    PyObject *init_state;
};

/**
 * @brief Convert a duckdb TableFilterSet to cudf AST expressions.
 *
 * @param filters - duckdb TableFilterSet to convert
 * @param column_names - column names of the table (before removing unused
 * columns)
 * @param filter_ast_tree - output cudf AST expressions representing the
 * filters. All expressions should be added to be kept alive.
 * @param filter_scalars - output vector of cudf scalars representing any
 * constants in the filters. All scalars should be added to be kept alive.
 */
void tableFilterSetToCudfAST(
    duckdb::TableFilterSet &filters,
    const std::vector<std::string> &column_names,
    cudf::ast::tree &filter_ast_tree,
    std::vector<std::unique_ptr<cudf::scalar>> &filter_scalars);

/**
 * @brief Owns all nodes and scalars created during a duckdb -> cudf AST
 *        conversion so that they remain alive for the lifetime of the
 *        resulting cudf::ast::expression reference.
 *
 */
class CudfASTOwner {
    cudf::ast::tree tree;
    std::vector<std::unique_ptr<cudf::scalar>> scalars;
    const cudf::ast::expression *root{nullptr};

   public:
    const cudf::ast::expression &insert_literal(const duckdb::Value &val,
                                                rmm::cuda_stream_view &stream);
    const cudf::ast::expression &insert_literal(
        std::unique_ptr<cudf::scalar> val, rmm::cuda_stream_view &stream);
    const cudf::ast::expression &get_root() const {
        if (!root) {
            throw std::runtime_error("CudfASTOwner: tree is empty");
        }
        return *root;
    }
    template <typename ExprT>
    const ExprT &push(ExprT expr) {
        const ExprT &ref = tree.push(std::move(expr));
        root = &ref;
        return ref;
    }

    friend const cudf::ast::expression &duckdb_expr_to_cudf_ast(
        const duckdb::Expression &,
        const std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t> &,
        const std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t> &,
        CudfASTOwner &, rmm::cuda_stream_view &);
};

/**
 * @brief Map a duckdb ExpressionType comparison operator to the
 *        corresponding cudf::ast::ast_operator.
 *
 * @throws std::runtime_error for unsupported operators.
 */
cudf::ast::ast_operator duckdb_etype_to_cudf_ast_op(
    duckdb::ExpressionType etype);

/**
 * @brief Recursively convert a duckdb Expression tree into a cudf AST
 *        expression suitable for use as the binary_predicate in a mixed_join.
 *
 * Column references are resolved using @p col_ref_map which maps
 * (table_index, column_index) duckdb pairs to a flat column index into the
 * *_conditional table_view passed to mixed_join.  The table_reference
 * (LEFT vs RIGHT) is determined by whether the duckdb table_index is
 * found in @p left_col_ref_map.
 *
 * @param expr              Root of the duckdb expression subtree to convert.
 * @param left_col_ref_map Duckdb column references on the left side of the
 * join.
 * @param right_col_ref_map Duckdb column references on the right side of the
 * join.
 *
 * @param stream cuda stream to create scalars on
 *
 * @return Reference to the root cudf AST expression node, owned by @p owner.
 *
 * @throws std::runtime_error for unsupported expression types or if a
 *         constant cannot be converted.
 */
const cudf::ast::expression &duckdb_expr_to_cudf_ast(
    const duckdb::Expression &expr,
    const std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t>
        &left_col_ref_map,
    const std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t>
        &right_col_ref_map,
    CudfASTOwner &owner, rmm::cuda_stream_view &stream);
/**
 * @brief Convert multiple duckdb expressions into a single CudfASTOwner,
 *        combining them with LOGICAL_AND.
 *
 * @param exprs             The duckdb expressions to combine.
 * index.
 * @param left_col_ref_map Duckdb column references on the left side of the
 * join.
 * @param right_col_ref_map Duckdb column references on the right side of the
 * join.
 * @param stream to create scalars on
 * @return CudfASTOwner whose tree root is the AND of all expressions.
 */
CudfASTOwner build_mixed_join_predicate(
    const std::vector<duckdb::unique_ptr<duckdb::Expression>> &exprs,
    const std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t>
        &left_col_ref_map,
    const std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t>
        &right_col_ref_map,
    rmm::cuda_stream_view &stream);
