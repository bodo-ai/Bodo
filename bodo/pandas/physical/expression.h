#pragma once

#include "../libs/_array_utils.h"
#include "operator.h"

class ExprResult {
   public:
    virtual ~ExprResult() = default;
};

class TableExprResult : public ExprResult {
   public:
    TableExprResult(std::shared_ptr<table_info> val) : result(val) {}
    virtual ~TableExprResult() = default;
    std::shared_ptr<table_info> result;
};

class ArrayExprResult : public ExprResult {
   public:
    ArrayExprResult(std::shared_ptr<array_info> val) : result(val) {}
    virtual ~ArrayExprResult() = default;
    std::shared_ptr<array_info> result;
};

class ScalarExprResult : public ExprResult {
   public:
    ScalarExprResult(std::shared_ptr<array_info> val) : result(val) {
        assert(val.length == 1);
    }
    virtual ~ScalarExprResult() = default;
    std::shared_ptr<array_info> result;
};

class PhysicalExpression {
   public:
    virtual ~PhysicalExpression() = default;

    virtual std::shared_ptr<ExprResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch) = 0;

   protected:
    std::vector<std::shared_ptr<PhysicalExpression>> children;
};

bool equal_test(int test) { return test == 0; }
bool not_equal_test(int test) { return test != 0; }
bool greater_test(int test) { return test == -1; }
bool less_test(int test) { return test == 1; }
bool greater_equal_test(int test) { return test != 1; }
bool less_equal_test(int test) { return test != -1; }

duckdb::ExpressionType exprSwitchLeftRight(duckdb::ExpressionType etype) {
    switch (etype) {
        case duckdb::ExpressionType::COMPARE_EQUAL:
        case duckdb::ExpressionType::COMPARE_NOTEQUAL:
            return etype;
        case duckdb::ExpressionType::COMPARE_LESSTHAN:
            return duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO;
        case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
            return duckdb::ExpressionType::COMPARE_LESSTHAN;
        case duckdb::ExpressionType::COMPARE_GREATERTHAN:
            return duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO;
        case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
            return duckdb::ExpressionType::COMPARE_GREATERTHAN;
        default:
            throw std::runtime_error(
                "switchLeftRight doesn't handle expression type " +
                std::to_string(static_cast<int>(etype)));
    }
}

template <typename T>
void compare_one_array(std::shared_ptr<array_info> arr1, const T data2,
                       uint8_t *output, bool (*comparator)(int)) {
    int64_t n_rows = arr1->length;
    uint64_t arr1_siztype = numpy_item_size[arr1->dtype];
    char *arr1_data1 = arr1->data1();
    char *arr1_data1_end = arr1_data1 + (n_rows * arr1_siztype);
    bool na_position = false;

    for (uint64_t i = 0; arr1_data1 < arr1_data1_end;
         arr1_data1 += arr1_siztype, ++i) {
        int test =
            NumericComparison(arr1->dtype, arr1_data1, data2, na_position);
        // if (test == 0 && test != 0) return;
        SetBitTo(output, i, comparator(test));
    }
}

void compare_two_array(std::shared_ptr<array_info> arr1,
                       const std::shared_ptr<array_info> arr2, uint8_t *output,
                       bool (*comparator)(int)) {
    int64_t n_rows = arr1->length;
    uint64_t arr1_siztype = numpy_item_size[arr1->dtype];
    char *arr1_data1 = arr1->data1();
    char *arr1_data1_end = arr1_data1 + (n_rows * arr1_siztype);
    char *arr2_data1 = arr2->data1();
    uint64_t arr2_siztype = numpy_item_size[arr2->dtype];
    assert(arr1->length == arr2->length);
    bool na_position = false;

    for (uint64_t i = 0; arr1_data1 < arr1_data1_end;
         arr1_data1 += arr1_siztype, arr2_data1 += arr2_siztype, ++i) {
        int test =
            NumericComparison(arr1->dtype, arr1_data1, arr2_data1, na_position);
        SetBitTo(output, i, comparator(test));
    }
}

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
    virtual std::shared_ptr<ExprResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch) {
        auto left_res = children[0]->ProcessBatch(input_batch);
        auto right_res = children[1]->ProcessBatch(input_batch);
        auto left_as_array =
            std::dynamic_pointer_cast<ArrayExprResult>(left_res);
        auto left_as_scalar =
            std::dynamic_pointer_cast<ScalarExprResult>(left_res);
        auto right_as_array =
            std::dynamic_pointer_cast<ArrayExprResult>(right_res);
        auto right_as_scalar =
            std::dynamic_pointer_cast<ScalarExprResult>(right_res);
        if (first_time) {
            first_time = false;
            if (left_as_array || right_as_array) {
                two_source = left_as_array && right_as_array;
                if (left_as_scalar) {
                    switchLeftRight = true;
                    expr_type = exprSwitchLeftRight(expr_type);
                }
            } else {
                throw std::runtime_error(
                    "Don't handle scalar-scalar expressions yet.");
            }
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
        std::shared_ptr<array_info> result =
            std::move(alloc_nullable_array_no_nulls(input_batch->nrows(),
                                                    Bodo_CTypes::_BOOL));
        uint8_t *result_data1 =
            result->data1<bodo_array_type::NULLABLE_INT_BOOL, uint8_t>();
        if (two_source) {
            compare_two_array(left_as_array->result, right_as_array->result,
                              result_data1, comparator);
        } else {
            compare_one_array(left_as_array->result,
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
    bool (*comparator)(int);
};

template <typename T>
class PhysicalConstantExpression : public PhysicalExpression {
   public:
    PhysicalConstantExpression(const T &val) : constant(val) {}
    virtual ~PhysicalConstantExpression() = default;
    virtual std::shared_ptr<ExprResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch) {
        auto result = alloc_nullable_array_no_nulls(1, ctypeFromVal(constant));
        std::memcpy(result->data1(), &constant, sizeof(T));
        return std::make_shared<ScalarExprResult>(std::move(result));
    }

   private:
    T constant;
};

class PhysicalColumnRefExpression : public PhysicalExpression {
   public:
    PhysicalColumnRefExpression(duckdb::idx_t table, duckdb::idx_t column)
        : table_index(table) {
        selected_columns.push_back(column);
    }
    virtual ~PhysicalColumnRefExpression() = default;

    virtual std::shared_ptr<ExprResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch) {
        std::shared_ptr<table_info> out_table_info =
            ProjectTable(input_batch, selected_columns);
        return std::make_shared<ArrayExprResult>(out_table_info->columns[0]);
    }

   protected:
    duckdb::idx_t table_index;
    std::vector<int64_t> selected_columns;
};
