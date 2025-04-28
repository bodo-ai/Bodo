#pragma once

#include "../libs/_array_utils.h"
#include "duckdb/common/enums/expression_type.hpp"
#include "operator.h"
#include "../tests/utils.h"
#include <iostream>

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
 * @brief Handle numerical comparison operators working on a column and a scalar.
 *
 * @param arr1 - the column data input to compare
 * @param data2 - the scalar data to compare all the elements in arr1 against
 * @param output - pointer to boolean output array with 1-bit data type
 * @param comparator - function to convert NumericComparison result to
 *                     the boolean result for the desired operator
 */
template <typename T>
void compare_one_array_numeric(std::shared_ptr<array_info> arr1,
                       const T data2,
                       uint8_t *output,
                       const std::function<bool(int)> &comparator) {
    int64_t n_rows = arr1->length;
    uint64_t arr1_siztype = numpy_item_size[arr1->dtype];
    char *arr1_data1 = arr1->data1();
    char *arr1_data1_end = arr1_data1 + (n_rows * arr1_siztype);
    bool na_position = false;

    std::function<int(const char *, const char *, bool const &)> ncfunc =
        getNumericComparisonFunc(arr1->dtype);
    for (uint64_t i = 0; arr1_data1 < arr1_data1_end;
         arr1_data1 += arr1_siztype, ++i) {
        int test = ncfunc(arr1_data1, data2, na_position);
        SetBitTo(output, i, comparator(test));
    }
}

/**
 * @brief Handle string comparison operators working on a column and a scalar.
 *
 * @param arr1 - the column data input to compare
 * @param data2 - the scalar data to compare all the elements in arr1 against
 * @param output - pointer to boolean output array with 1-bit data type
 * @param comparator - function to convert comparison result to
 *                     the boolean result for the desired operator
 */
void compare_one_array_string(std::shared_ptr<array_info> arr1,
                       const std::string data2,
                       uint8_t *output,
                       const std::function<bool(int)> &comparator);

/**
 * @brief Dispatch comparison operators working on a column and a scalar to the right comparison function for that type.
 *
 * @param arr1 - the column data input to compare
 * @param data2 - the scalar data to compare all the elements in arr1 against
 * @param output - pointer to boolean output array with 1-bit data type
 * @param comparator - function to convert comparison result to
 *                     the boolean result for the desired operator
 */
template <typename T>
void compare_one_array_dispatch(std::shared_ptr<array_info> arr1,
                       const T data2,
                       uint8_t *output,
                       const std::function<bool(int)> &comparator) {
    if (is_numerical(arr1->dtype)) {
        compare_one_array_numeric(arr1, data2, output, comparator);
    } else if(arr1->arr_type == bodo_array_type::STRING) {
        compare_one_array_string(arr1, data2, output, comparator);
    }
}

/**
 * @brief Handle comparison operators that compare two columns.
 *
 * @param arr1 - the first column data input to compare
 * @param arr2 - the second column data input to compare
 * @param output - pointer to boolean output array with 1-bit data type
 * @param comparator - function to convert NumericComparison result to
 *                     the boolean result for the desired operator
 */
void compare_two_array(std::shared_ptr<array_info> arr1,
                       const std::shared_ptr<array_info> arr2,
                       uint8_t *output,
                       const std::function<bool(int)> &comparator);

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
                    comparator = equal_test;
                    break;
                case duckdb::ExpressionType::COMPARE_NOTEQUAL:
                    comparator = not_equal_test;
                    break;
                case duckdb::ExpressionType::COMPARE_GREATERTHAN:
                    comparator = greater_test;
                    break;
                case duckdb::ExpressionType::COMPARE_LESSTHAN:
                    comparator = less_test;
                    break;
                case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
                    comparator = greater_equal_test;
                    break;
                case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
                    comparator = less_equal_test;
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
        // Create output boolean array same size as input.
        std::shared_ptr<array_info> result = alloc_nullable_array_no_nulls(
            input_batch->nrows(), Bodo_CTypes::_BOOL);
        // Get uint8_t raw data pointer for the boolean output array.
        uint8_t *result_data1 =
            result->data1<bodo_array_type::NULLABLE_INT_BOOL, uint8_t>();
        // Call one array or two array comparison functions based on whether
        // we have one input array or two.
        if (two_source) {
            compare_two_array(left_as_array->result, right_as_array->result,
                              result_data1, comparator);
        } else {
            compare_one_array_dispatch(left_as_array->result,
                              right_as_scalar->result->data1(), result_data1,
                              comparator);
        }

        return std::make_shared<ArrayExprResult>(result);
    }

   protected:
    duckdb::ExpressionType expr_type;
    bool first_time;
    bool switchLeftRight;
    bool two_source;
    std::function<bool(int)> comparator;
};

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
        // Create 1 element array with same type as constant.
        std::unique_ptr<array_info> result =
            alloc_nullable_array_no_nulls(1, typeToDtype(constant));
        // Copy constant into the array.
        std::memcpy(result->data1(), &constant, sizeof(T));
        return std::make_shared<ScalarExprResult>(std::move(result));
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
        // Create 1 element array with same type as constant.
        std::unique_ptr<array_info> result =
            alloc_nullable_array_no_nulls(1, typeToDtype(constant));
        // Copy constant into the array.
        std::memcpy(result->data1(), constant.c_str(), constant.size());
        return std::make_shared<ScalarExprResult>(std::move(result));
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
