#pragma once

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/type_traits.h>
#include <string>
#include <type_traits>
#include "../libs/_array_utils.h"
#include "../libs/_bodo_to_arrow.h"
#include "../tests/utils.h"
#include "_util.h"
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
 * @brief Convert ExprResult to arrow and run compute operation on it.
 *
 */
std::shared_ptr<array_info> do_arrow_compute_unary(
    std::shared_ptr<ExprResult> left_res, const std::string &comparator);

/**
 * @brief Convert two ExprResults to arrow and run compute operation on them.
 *
 */
std::shared_ptr<array_info> do_arrow_compute_binary(
    std::shared_ptr<ExprResult> left_res, std::shared_ptr<ExprResult> right_res,
    const std::string &comparator);

/**
 * @brief Convert ExprResult to arrow and cast to the requested type.
 *
 */
std::shared_ptr<array_info> do_arrow_compute_cast(
    std::shared_ptr<ExprResult> left_res,
    const duckdb::LogicalType &return_type);

/**
 * @brief Physical expression tree node type for comparisons resulting in
 *        boolean arrays.
 *
 */
class PhysicalComparisonExpression : public PhysicalExpression {
   public:
    PhysicalComparisonExpression(std::shared_ptr<PhysicalExpression> left,
                                 std::shared_ptr<PhysicalExpression> right,
                                 duckdb::ExpressionType etype) {
        children.push_back(left);
        children.push_back(right);
        switch (etype) {
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

        auto result = do_arrow_compute_binary(left_res, right_res, comparator);
        return std::make_shared<ArrayExprResult>(result);
    }

   protected:
    std::string comparator;
};

// Numeric type specialization
template <typename T, typename std::enable_if<std::is_arithmetic<T>::value &&
                                                  !std::is_same<T, bool>::value,
                                              int>::type = 0>
std::shared_ptr<arrow::Array> ScalarToArrowArray(const T &value,
                                                 size_t num_elements = 1) {
    using ArrowType = typename arrow::CTypeTraits<T>::ArrowType;
    using BuilderType = arrow::NumericBuilder<ArrowType>;

    BuilderType builder;
    arrow::Status status;
    for (size_t i = 0; i < num_elements; ++i) {
        status = builder.Append(value);
        if (!status.ok()) {
            throw std::runtime_error("builder.Append failed.");
        }
    }
    std::shared_ptr<arrow::Array> array;
    status = builder.Finish(&array);
    if (!status.ok()) {
        throw std::runtime_error("builder.Finish failed.");
    }
    return array;
}

// String specialization
std::shared_ptr<arrow::Array> ScalarToArrowArray(const std::string &value,
                                                 size_t num_elements = 1);

// arrow::Scalar specialization
std::shared_ptr<arrow::Array> CreateOneElementArrowArray(
    const std::shared_ptr<arrow::Scalar> &value);

std::shared_ptr<arrow::Array> ScalarToArrowArray(const arrow::Scalar &value,
                                                 size_t num_elements = 1);

// bool specialization
std::shared_ptr<arrow::Array> ScalarToArrowArray(bool value,
                                                 size_t num_elements = 1);

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
        std::shared_ptr<arrow::Array> array = ScalarToArrowArray(constant);

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
        std::shared_ptr<arrow::Array> array = ScalarToArrowArray(constant);

        auto result =
            arrow_array_to_bodo(array, bodo::BufferPool::DefaultPtr());
        return std::make_shared<ScalarExprResult>(std::move(result));
    }

    friend std::ostream &operator<<(
        std::ostream &os, const PhysicalConstantExpression<std::string> &obj) {
        os << "PhysicalConstantExpression<string> " << obj.constant
           << std::endl;
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
                                  duckdb::ExpressionType etype) {
        children.push_back(left);
        children.push_back(right);
        switch (etype) {
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

        auto result = do_arrow_compute_binary(left_res, right_res, comparator);
        return std::make_shared<ArrayExprResult>(result);
    }

   protected:
    std::string comparator;
};

/**
 * @brief Physical expression tree node type for casting.
 *
 */
class PhysicalCastExpression : public PhysicalExpression {
   public:
    PhysicalCastExpression(std::shared_ptr<PhysicalExpression> left,
                           duckdb::LogicalType _return_type)
        : return_type(_return_type) {
        children.push_back(left);
    }

    virtual ~PhysicalCastExpression() = default;

    /**
     * @brief How to process this expression tree node.
     *
     */
    virtual std::shared_ptr<ExprResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch) {
        // Process child first.
        std::shared_ptr<ExprResult> left_res =
            children[0]->ProcessBatch(input_batch);
        auto result = do_arrow_compute_cast(left_res, return_type);
        return std::make_shared<ArrayExprResult>(result);
    }

   protected:
    duckdb::LogicalType return_type;
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
                throw std::runtime_error("Unhandled unary op expression type.");
        }
    }
    PhysicalUnaryExpression(std::shared_ptr<PhysicalExpression> left,
                            std::string &opstr) {
        children.push_back(left);
        if (opstr == "floor") {
            comparator = "floor";
        } else {
            throw std::runtime_error("Unhandled unary expression opstr " +
                                     opstr);
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
        auto result = do_arrow_compute_unary(left_res, comparator);
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
                    "Unhandled binary expression type " +
                    std::to_string(static_cast<int>(etype)));
        }
    }

    PhysicalBinaryExpression(std::shared_ptr<PhysicalExpression> left,
                             std::shared_ptr<PhysicalExpression> right,
                             std::string &opstr) {
        children.push_back(left);
        children.push_back(right);
        if (opstr == "+") {
            comparator = "add";
        } else if (opstr == "-") {
            comparator = "substract";
        } else if (opstr == "*") {
            comparator = "multiply";
        } else if (opstr == "/") {
            comparator = "divide";
        } else if (opstr == "floor") {
            comparator = "floor";
        } else {
            throw std::runtime_error("Unhandled binary expression opstr " +
                                     opstr);
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

        auto result = do_arrow_compute_binary(left_res, right_res, comparator);
        return std::make_shared<ArrayExprResult>(result);
    }

   protected:
    std::string comparator;
};
