#pragma once

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/type_traits.h>
#include <string>
#include <type_traits>
#include "../libs/_array_utils.h"
#include "../libs/_bodo_to_arrow.h"
#include "../tests/utils.h"
#include "duckdb/common/enums/expression_type.hpp"
#include "operator.h"

std::shared_ptr<arrow::Array> prepare_arrow_compute(
    std::shared_ptr<array_info> arr);

/**
 * @brief Superclass for possible results returned by nodes in Bodo
 *        Physical expression tree.
 *
 */
class ExprResult {
   public:
    virtual ~ExprResult() = default;
};

/**
 * @brief Result type for Physical expression tree nodes that return
 *        an entire table_info type.
 *
 */
class TableExprResult : public ExprResult {
   public:
    TableExprResult(std::shared_ptr<table_info> val) : result(val) {}
    virtual ~TableExprResult() = default;
    const std::shared_ptr<table_info> result;
};

/**
 * @brief Result type for Physical expression tree nodes that return
 *        a single column in the array_info type.
 *
 */
class ArrayExprResult : public ExprResult {
   public:
    ArrayExprResult(std::shared_ptr<array_info> val) : result(val) {}
    virtual ~ArrayExprResult() = default;
    const std::shared_ptr<array_info> result;
};

/**
 * @brief Result type for Physical expression tree nodes that return
 *        a single scalar, which for various reasons, including not
 *        having the complication of templatizing this class, is
 *        stored as a single-element array_info.
 *
 */
class ScalarExprResult : public ExprResult {
   public:
    ScalarExprResult(std::shared_ptr<array_info> val) : result(val) {
        assert(val->length == 1);
    }
    virtual ~ScalarExprResult() = default;
    const std::shared_ptr<array_info> result;
};

/**
 * @brief Superclass for Bodo Physical expression tree nodes. Like duckdb
 *        it is convenient to store child nodes here because many expr
 *        node types have children.
 *
 */
class PhysicalExpression {
   public:
    virtual ~PhysicalExpression() = default;

    /**
     * @brief Like a pipeline ProcessBatch but with more flexible return type.
     *        Process your children first and then yourself.  The input_batch
     *        from the pipeline needs to be passed down to the leaves of the
     *        expression tree so that source that use the input_batch have
     *        access.
     *
     */
    virtual std::shared_ptr<ExprResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch) = 0;

   protected:
    std::vector<std::shared_ptr<PhysicalExpression>> children;
};

/**
 * @brief These lambdas convert NumericComparison output to the
 *        equivalent for the given operator.
 *
 */
extern std::function<bool(int)> equal_test;
extern std::function<bool(int)> not_equal_test;
extern std::function<bool(int)> greater_test;
extern std::function<bool(int)> less_test;
extern std::function<bool(int)> greater_equal_test;
extern std::function<bool(int)> less_equal_test;

/**
 * @brief If we switch left and right operands then we need in some cases to
 *        change the operator correspondingly.
 *
 */
duckdb::ExpressionType exprSwitchLeftRight(duckdb::ExpressionType etype);

/**
 * @brief Physical expression tree node type for comparisons resulting in
 *        boolean arrays.
 *
 */
class PhysicalComparisonExpression : public PhysicalExpression {
   public:
    PhysicalComparisonExpression(std::shared_ptr<PhysicalExpression> left,
                                 std::shared_ptr<PhysicalExpression> right,
                                 duckdb::ExpressionType etype)
        : expr_type(etype),
          first_time(true),
          switchLeftRight(false),
          two_source(true) {
        children.push_back(left);
        children.push_back(right);
    }

    virtual ~PhysicalComparisonExpression() = default;

    /**
     * @brief How to process this expression tree node.
     *
     */
    virtual std::shared_ptr<ExprResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch) {
        // We know we have two children so process them first.
        std::shared_ptr<ExprResult> left_res =
            children[0]->ProcessBatch(input_batch);
        std::shared_ptr<ExprResult> right_res =
            children[1]->ProcessBatch(input_batch);
        // Try to convert the results of our children into array
        // or scalar results to see which one they are.
        std::shared_ptr<ArrayExprResult> left_as_array =
            std::dynamic_pointer_cast<ArrayExprResult>(left_res);
        std::shared_ptr<ScalarExprResult> left_as_scalar =
            std::dynamic_pointer_cast<ScalarExprResult>(left_res);
        std::shared_ptr<ArrayExprResult> right_as_array =
            std::dynamic_pointer_cast<ArrayExprResult>(right_res);
        std::shared_ptr<ScalarExprResult> right_as_scalar =
            std::dynamic_pointer_cast<ScalarExprResult>(right_res);
        // Some things we don't know at node conversion time but
        // we do know at first execution time.  So, we try to do
        // certain checks only once with the first_time flag.
        if (first_time) {
            first_time = false;
            // If at least one output of our children is an array.
            if (left_as_array || right_as_array) {
                // Save if both are array output.
                two_source = left_as_array && right_as_array;
                // If left is a scalar then indicate we will swap
                // this an all future left and right outputs and
                // make appropriate change to the operator type.
                if (left_as_scalar) {
                    switchLeftRight = true;
                    expr_type = exprSwitchLeftRight(expr_type);
                }
            } else {
                throw std::runtime_error(
                    "Don't handle scalar-scalar expressions yet.");
            }
            // Now after possible operator switching, save the
            // comparator function we'll use for this and future
            // batch processing.
            switch (expr_type) {
                case duckdb::ExpressionType::COMPARE_EQUAL:
                    comparator = "equal";
                    break;
                case duckdb::ExpressionType::COMPARE_NOTEQUAL:
                    comparator = "not_equal";
                    break;
                case duckdb::ExpressionType::COMPARE_GREATERTHAN:
                    comparator = "greater";
                    break;
                case duckdb::ExpressionType::COMPARE_LESSTHAN:
                    comparator = "less";
                    break;
                case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
                    comparator = "greater_equal";
                    break;
                case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
                    comparator = "less_equal";
                    break;
                default:
                    throw std::runtime_error(
                        "Unhandled comparison expression type.");
            }
        }
        if (switchLeftRight) {
            // Switch left and right so one fewer case to handle.
            std::swap(left_as_array, right_as_array);
            std::swap(left_as_scalar, right_as_scalar);
        }

        arrow::Datum src1 =
            arrow::Datum(prepare_arrow_compute(left_as_array->result));
        arrow::Datum src2;

        if (two_source) {
            src2 = arrow::Datum(prepare_arrow_compute(right_as_array->result));
        } else {
            src2 =
                arrow::MakeScalar(prepare_arrow_compute(right_as_scalar->result)
                                      ->GetScalar(0)
                                      .ValueOrDie());
        }

        arrow::Result<arrow::Datum> cmp_res =
            arrow::compute::CallFunction(comparator, {src1, src2});
        if (!cmp_res.ok()) [[unlikely]] {
            throw std::runtime_error(
                "PhysicalComparisonExpression: Error in Arrow compute: " +
                cmp_res.status().message());
        }

        auto result = arrow_array_to_bodo(cmp_res.ValueOrDie().make_array(),
                                          bodo::BufferPool::DefaultPtr());

        return std::make_shared<ArrayExprResult>(result);
    }

   protected:
    duckdb::ExpressionType expr_type;
    bool first_time;
    bool switchLeftRight;
    bool two_source;
    std::string comparator;
};

// Numeric type specialization
template <typename T, typename std::enable_if<std::is_arithmetic<T>::value &&
                                                  !std::is_same<T, bool>::value,
                                              int>::type = 0>
std::shared_ptr<arrow::Array> CreateOneElementArrowArray(const T &value) {
    using ArrowType = typename arrow::CTypeTraits<T>::ArrowType;
    using BuilderType = arrow::NumericBuilder<ArrowType>;

    BuilderType builder;
    arrow::Status status;
    status = builder.Append(value);
    if (!status.ok()) {
        throw std::runtime_error("builder.Append failed.");
    }
    std::shared_ptr<arrow::Array> array;
    status = builder.Finish(&array);
    if (!status.ok()) {
        throw std::runtime_error("builder.Finish failed.");
    }
    return array;
}

// String specialization
std::shared_ptr<arrow::Array> CreateOneElementArrowArray(
    const std::string &value);

// bool specialization
std::shared_ptr<arrow::Array> CreateOneElementArrowArray(bool value);

/**
 * @brief Physical expression tree node type for scalar constants.
 *
 */
template <typename T>
class PhysicalConstantExpression : public PhysicalExpression {
   public:
    PhysicalConstantExpression(const T &val) : constant(val) {}
    virtual ~PhysicalConstantExpression() = default;

    virtual std::shared_ptr<ExprResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch) {
        std::shared_ptr<arrow::Array> array =
            CreateOneElementArrowArray(constant);

        auto result =
            arrow_array_to_bodo(array, bodo::BufferPool::DefaultPtr());
        return std::make_shared<ScalarExprResult>(std::move(result));
    }

    friend std::ostream &operator<<(std::ostream &os,
                                    const PhysicalConstantExpression<T> &obj) {
        os << "PhysicalConstantExpression " << obj.constant << std::endl;
        return os;
    }

   private:
    const T constant;
};

template <>
class PhysicalConstantExpression<std::string> : public PhysicalExpression {
   public:
    PhysicalConstantExpression(const std::string &val) : constant(val) {}
    virtual ~PhysicalConstantExpression() = default;

    virtual std::shared_ptr<ExprResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch) {
        std::shared_ptr<arrow::Array> array =
            CreateOneElementArrowArray(constant);

        auto result =
            arrow_array_to_bodo(array, bodo::BufferPool::DefaultPtr());
        return std::make_shared<ScalarExprResult>(std::move(result));
    }

    friend std::ostream &operator<<(
        std::ostream &os, const PhysicalConstantExpression<std::string> &obj) {
        os << "PhysicalConstantExpression<string> " << obj.constant << std::endl;
        return os;
    }

   private:
    const std::string constant;
};

/**
 * @brief Physical expression tree node type for getting column from table.
 *
 */
class PhysicalColumnRefExpression : public PhysicalExpression {
   public:
    PhysicalColumnRefExpression(duckdb::idx_t table, duckdb::idx_t column)
        : table_index(table), selected_columns({(int64_t)column}) {}
    virtual ~PhysicalColumnRefExpression() = default;

    virtual std::shared_ptr<ExprResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch) {
        std::shared_ptr<table_info> out_table_info =
            ProjectTable(input_batch, selected_columns);
        // ProjectTable returns a table so extract the singular column
        // out since we must return array_info result here.
        return std::make_shared<ArrayExprResult>(out_table_info->columns[0]);
    }

   protected:
    duckdb::idx_t table_index;
    const std::vector<int64_t> selected_columns;
};

/**
 * @brief Physical expression tree node type for conjunctions of
 *        boolean arrays.
 *
 */
class PhysicalConjunctionExpression : public PhysicalExpression {
   public:
    PhysicalConjunctionExpression(std::shared_ptr<PhysicalExpression> left,
                                  std::shared_ptr<PhysicalExpression> right,
                                  duckdb::ExpressionType etype)
        : expr_type(etype),
          first_time(true),
          switchLeftRight(false),
          two_source(true) {
        children.push_back(left);
        children.push_back(right);
    }

    virtual ~PhysicalConjunctionExpression() = default;

    /**
     * @brief How to process this expression tree node.
     *
     */
    virtual std::shared_ptr<ExprResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch) {
        // We know we have two children so process them first.
        std::shared_ptr<ExprResult> left_res =
            children[0]->ProcessBatch(input_batch);
        std::shared_ptr<ExprResult> right_res =
            children[1]->ProcessBatch(input_batch);
        // Try to convert the results of our children into array
        // or scalar results to see which one they are.
        std::shared_ptr<ArrayExprResult> left_as_array =
            std::dynamic_pointer_cast<ArrayExprResult>(left_res);
        std::shared_ptr<ScalarExprResult> left_as_scalar =
            std::dynamic_pointer_cast<ScalarExprResult>(left_res);
        std::shared_ptr<ArrayExprResult> right_as_array =
            std::dynamic_pointer_cast<ArrayExprResult>(right_res);
        std::shared_ptr<ScalarExprResult> right_as_scalar =
            std::dynamic_pointer_cast<ScalarExprResult>(right_res);
        // Some things we don't know at node conversion time but
        // we do know at first execution time.  So, we try to do
        // certain checks only once with the first_time flag.
        if (first_time) {
            first_time = false;
            // If at least one output of our children is an array.
            if (left_as_array || right_as_array) {
                // Save if both are array output.
                two_source = left_as_array && right_as_array;
                // If left is a scalar then indicate we will swap
                // this an all future left and right outputs and
                // make appropriate change to the operator type.
                if (left_as_scalar) {
                    switchLeftRight = true;
                }
            } else {
                throw std::runtime_error(
                    "Don't handle scalar-scalar expressions yet.");
            }
            // Now after possible operator switching, save the
            // comparator function we'll use for this and future
            // batch processing.
            switch (expr_type) {
                case duckdb::ExpressionType::CONJUNCTION_AND:
                    comparator = "and";
                    break;
                case duckdb::ExpressionType::CONJUNCTION_OR:
                    comparator = "or";
                    break;
                default:
                    throw std::runtime_error(
                        "Unhandled conjunction expression type.");
            }
        }
        if (switchLeftRight) {
            // Switch left and right so one fewer case to handle.
            std::swap(left_as_array, right_as_array);
            std::swap(left_as_scalar, right_as_scalar);
        }

        arrow::Datum src1 =
            arrow::Datum(prepare_arrow_compute(left_as_array->result));
        arrow::Datum src2;
        if (two_source) {
            src2 = arrow::Datum(prepare_arrow_compute(right_as_array->result));
        } else {
            src2 =
                arrow::MakeScalar(prepare_arrow_compute(right_as_scalar->result)
                                      ->GetScalar(0)
                                      .ValueOrDie());
        }

        arrow::Result<arrow::Datum> cmp_res =
            arrow::compute::CallFunction(comparator, {src1, src2});
        if (!cmp_res.ok()) [[unlikely]] {
            throw std::runtime_error(
                "PhysicalConjunctionExpression: Error in Arrow compute: " +
                cmp_res.status().message());
        }

        auto result = arrow_array_to_bodo(cmp_res.ValueOrDie().make_array(),
                                          bodo::BufferPool::DefaultPtr());

        return std::make_shared<ArrayExprResult>(result);
    }

   protected:
    duckdb::ExpressionType expr_type;
    bool first_time;
    bool switchLeftRight;
    bool two_source;
    std::string comparator;
};

/**
 * @brief Physical expression tree node type for unary of array.
 *
 */
class PhysicalUnaryExpression : public PhysicalExpression {
   public:
    PhysicalUnaryExpression(std::shared_ptr<PhysicalExpression> left,
                            duckdb::ExpressionType etype) {
        children.push_back(left);
        switch (etype) {
            case duckdb::ExpressionType::OPERATOR_NOT:
                comparator = "invert";
                break;
            default:
                throw std::runtime_error(
                    "Unhandled unary op expression type.");
        }
    }

    virtual ~PhysicalUnaryExpression() = default;

    /**
     * @brief How to process this expression tree node.
     *
     */
    virtual std::shared_ptr<ExprResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch) {
        // Process child first.
        std::shared_ptr<ExprResult> left_res =
            children[0]->ProcessBatch(input_batch);
        // Try to convert the results of our children into array
        // or scalar results to see which one they are.
        std::shared_ptr<ArrayExprResult> left_as_array =
            std::dynamic_pointer_cast<ArrayExprResult>(left_res);
        std::shared_ptr<ScalarExprResult> left_as_scalar =
            std::dynamic_pointer_cast<ScalarExprResult>(left_res);

        arrow::Datum src1;
        if (left_as_array) {
            src1 = arrow::Datum(prepare_arrow_compute(left_as_array->result));
        } else if (left_as_scalar) {
            src1 =
                arrow::MakeScalar(prepare_arrow_compute(left_as_scalar->result)
                                      ->GetScalar(0)
                                      .ValueOrDie());
        } else {
            throw std::runtime_error(
                "PhysicalUnaryExpression left is neither array nor scalar.");
        }

        arrow::Result<arrow::Datum> cmp_res =
            arrow::compute::CallFunction(comparator, {src1});
        if (!cmp_res.ok()) [[unlikely]] {
            throw std::runtime_error(
                "PhysicalUnaryExpression: Error in Arrow compute: " +
                cmp_res.status().message());
        }

        auto result = arrow_array_to_bodo(cmp_res.ValueOrDie().make_array(),
                                          bodo::BufferPool::DefaultPtr());

        return std::make_shared<ArrayExprResult>(result);
    }

   protected:
    std::string comparator;
};

/**
 * @brief Physical expression tree node type for binary op non-boolean arrays.
 *
 */
class PhysicalBinaryExpression : public PhysicalExpression {
   public:
    PhysicalBinaryExpression(std::shared_ptr<PhysicalExpression> left,
                             std::shared_ptr<PhysicalExpression> right,
                             duckdb::ExpressionType etype) {
        children.push_back(left);
        children.push_back(right);
        switch (etype) {
            default:
                throw std::runtime_error(
                    "Unhandled binary expression type.");
        }
    }

    virtual ~PhysicalBinaryExpression() = default;

    /**
     * @brief How to process this expression tree node.
     *
     */
    virtual std::shared_ptr<ExprResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch) {
        // We know we have two children so process them first.
        std::shared_ptr<ExprResult> left_res =
            children[0]->ProcessBatch(input_batch);
        std::shared_ptr<ExprResult> right_res =
            children[1]->ProcessBatch(input_batch);
        // Try to convert the results of our children into array
        // or scalar results to see which one they are.
        std::shared_ptr<ArrayExprResult> left_as_array =
            std::dynamic_pointer_cast<ArrayExprResult>(left_res);
        std::shared_ptr<ScalarExprResult> left_as_scalar =
            std::dynamic_pointer_cast<ScalarExprResult>(left_res);
        std::shared_ptr<ArrayExprResult> right_as_array =
            std::dynamic_pointer_cast<ArrayExprResult>(right_res);
        std::shared_ptr<ScalarExprResult> right_as_scalar =
            std::dynamic_pointer_cast<ScalarExprResult>(right_res);

        arrow::Datum src1;
        if (left_as_array) {
            src1 = arrow::Datum(prepare_arrow_compute(left_as_array->result));
        } else if(left_as_scalar) {
            src1 =
                arrow::MakeScalar(prepare_arrow_compute(left_as_scalar->result)
                                      ->GetScalar(0)
                                      .ValueOrDie());
        } else {
            throw std::runtime_error(
                "PhysicalBinaryExpression left is neither array nor scalar.");
        }

        arrow::Datum src2;
        if (right_as_array) {
            src2 = arrow::Datum(prepare_arrow_compute(right_as_array->result));
        } else if (right_as_scalar) {
            src2 =
                arrow::MakeScalar(prepare_arrow_compute(right_as_scalar->result)
                                      ->GetScalar(0)
                                      .ValueOrDie());
        } else {
            throw std::runtime_error(
                "PhysicalBinaryExpression right is neither array nor scalar.");
        }

        arrow::Result<arrow::Datum> cmp_res =
            arrow::compute::CallFunction(comparator, {src1, src2});
        if (!cmp_res.ok()) [[unlikely]] {
            throw std::runtime_error(
                "PhysicalBinaryExpression: Error in Arrow compute: " +
                cmp_res.status().message());
        }

        auto result = arrow_array_to_bodo(cmp_res.ValueOrDie().make_array(),
                                          bodo::BufferPool::DefaultPtr());

        return std::make_shared<ArrayExprResult>(result);
    }

   protected:
    std::string comparator;
};
