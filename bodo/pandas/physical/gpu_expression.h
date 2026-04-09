#pragma once

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/compute/function.h>
#include <arrow/compute/kernel.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/type_traits.h>
#include <future>
#include <rmm/cuda_stream_view.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>
#include "../libs/_array_utils.h"
#include "../libs/_bodo_common.h"
#include "../libs/_bodo_to_arrow.h"
#include "../tests/utils.h"
#include "_util.h"
#include "duckdb/common/enums/expression_type.hpp"
#include "duckdb/planner/expression.hpp"
#include "operator.h"

#include <cstdint>
#include <cudf/types.hpp>
#include <memory>

#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/datetime.hpp>
#include <cudf/round.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/regex/regex_program.hpp>
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
        return std::make_unique<cudf::string_scalar>(value, se->stream);
    } else if constexpr (std::is_same_v<T, const char *>) {
        return std::make_unique<cudf::string_scalar>(std::string(value),
                                                     se->stream);
    } else if constexpr (std::is_same_v<T, bool>) {
        return std::make_unique<cudf::numeric_scalar<int8_t>>(
            static_cast<int8_t>(value), true, se->stream);
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
            make_invalid_like(*scalar, se->stream);

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
                is_not_null = true;
                break;
            default:
                throw std::runtime_error("Unhandled unary op expression type.");
        }
    }
    PhysicalGPUUnaryExpression(std::shared_ptr<PhysicalGPUExpression> left,
                               std::string &opstr) {
        children.push_back(left);
        if (opstr == "floor") {
            comparator = cudf::unary_operator::FLOOR;
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
        if (is_not_null) {
            std::shared_ptr<ArrayExprGPUResult> left_as_array =
                std::dynamic_pointer_cast<ArrayExprGPUResult>(left_res);
            if (left_as_array) {
                auto is_null_res = cudf::is_null(left_as_array->result->view());
                op_res = cudf::unary_operation(is_null_res->view(), comparator,
                                               se->stream);
            } else {
                throw std::runtime_error("Left must be array for is_not_null.");
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
    cudf::unary_operator comparator;
    bool is_not_null = false;
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
                                duckdb::LogicalType _return_type) {
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

        auto result = do_cudf_compute_binary(left_res, right_res, comparator,
                                             se, return_type);
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
            scalar_func_data.arrow_func_name != "year" &&
            scalar_func_data.arrow_func_name != "round" &&
            scalar_func_data.arrow_func_name != "is_null") {
            throw std::runtime_error(
                "PhysicalGPUArrowExpression only supports ends_with, "
                "starts_with, match_substring_regex, year, round and is_null "
                "for now.");
        }
        if (scalar_func_data.arrow_func_name == "ends_with" ||
            scalar_func_data.arrow_func_name == "starts_with" ||
            scalar_func_data.arrow_func_name == "match_substring_regex") {
            extract_string_arg_from_python();
        } else if (scalar_func_data.arrow_func_name == "round") {
            extract_round_arg_from_python();
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
                                 cudf::rounding_method::HALF_EVEN, se->stream);
#pragma GCC diagnostic pop
        } else if (scalar_func_data.arrow_func_name == "is_null") {
            result = cudf::is_null(in_as_array->result->view(), se->stream);
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
        // Extract string argument from scalar_func_data.args for ends_with and
        // starts_with
        if (!PyTuple_Check(scalar_func_data.args) ||
            PyTuple_Size(scalar_func_data.args) != 1) {
            throw std::runtime_error(
                fmt::format("{} args not a 1-element tuple.",
                            scalar_func_data.arrow_func_name));
        }

        // Get the first element (borrowed reference)
        PyObject *py_str = PyTuple_GetItem(scalar_func_data.args, 0);

        if (!PyUnicode_Check(py_str)) {
            throw std::runtime_error(
                fmt::format("{} args element is not a Python string.",
                            scalar_func_data.arrow_func_name));
        }

        // Convert to UTF‑8 C string
        const char *c_str = PyUnicode_AsUTF8(py_str);
        if (!c_str) {
            throw std::runtime_error(
                fmt::format("{} error extracting Python string.",
                            scalar_func_data.arrow_func_name));
        }

        if (scalar_func_data.arrow_func_name == "match_substring_regex") {
            regex_prog =
                cudf::strings::regex_program::create(std::string(c_str));
        } else {
            str_scalar_in =
                std::make_shared<cudf::string_scalar>(std::string(c_str), true);
        }
    }

    void extract_round_arg_from_python() {
        if (PyTuple_Check(scalar_func_data.args) &&
            PyTuple_Size(scalar_func_data.args) == 1) {
            // Get the first element (borrowed reference)
            PyObject *py_digits = PyTuple_GetItem(scalar_func_data.args, 0);

            if (!PyLong_Check(py_digits)) {
                throw std::runtime_error(
                    fmt::format("{} args element is not a Python int.",
                                scalar_func_data.arrow_func_name));
            }
            round_ndigits = static_cast<int32_t>(PyLong_AsLong(py_digits));
        }
    }

    // Keeping reference to the cudf string scalar created from the Python
    // string argument to ensure it stays alive during processing.
    std::shared_ptr<cudf::string_scalar> str_scalar_in;

    // Needed for match_substring_regex
    std::shared_ptr<cudf::strings::regex_program> regex_prog;

    int32_t round_ndigits = 0;
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
    void insert_literal(const duckdb::Value &val,
                        rmm::cuda_stream_view &stream);
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
        const duckdb::Expression &, const std::unordered_set<duckdb::idx_t> &,
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
 * found in @p left_table_indices.
 *
 * @param expr              Root of the duckdb expression subtree to convert.
 * @param left_table_indices
 *                          Set of duckdb table indices that belong to the left
 *                          side of the join.  Any table index NOT in this set
 *                          is treated as belonging to the right side.
 * @param owner             Keeps all allocated AST nodes and scalars alive.
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
    const std::unordered_set<duckdb::idx_t> &left_table_indices,
    CudfASTOwner &owner, rmm::cuda_stream_view &stream);
/**
 * @brief Convert multiple duckdb expressions into a single CudfASTOwner,
 *        combining them with LOGICAL_AND.
 *
 * @param exprs             The duckdb expressions to combine.
 * index.
 * @param left_table_indices Set of duckdb table indices on the left side of the
 * join.
 * @param stream to create scalars on
 * @return CudfASTOwner whose tree root is the AND of all expressions.
 */
CudfASTOwner build_mixed_join_predicate(
    const std::vector<duckdb::unique_ptr<duckdb::Expression>> &exprs,
    const std::unordered_set<duckdb::idx_t> &left_table_indices,
    rmm::cuda_stream_view &stream);
