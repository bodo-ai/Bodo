#pragma once

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/compute/function.h>
#include <arrow/compute/kernel.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/type_traits.h>
#include <future>
#include <mutex>
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
#include "duckdb/planner/expression/bound_between_expression.hpp"
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
    ArrayExprResult(std::shared_ptr<array_info> val, std::string col)
        : result(val), column_name(col) {}
    virtual ~ArrayExprResult() = default;
    const std::shared_ptr<array_info> result;
    const std::string column_name;
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
    PhysicalExpression() {}
    PhysicalExpression(
        std::vector<std::shared_ptr<PhysicalExpression>> &_children)
        : children(_children) {}
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

    static bool join_expr(array_info **left_table, array_info **right_table,
                          void **left_data, void **right_data,
                          void **left_null_bitmap, void **right_null_bitmap,
                          int64_t left_index, int64_t right_index);
    static void join_expr_batch(
        array_info **left_table, array_info **right_table, void **left_data,
        void **right_data, void **left_null_bitmap, void **right_null_bitmap,
        uint8_t *match_arr, int64_t left_index_start, int64_t left_index_end,
        int64_t right_index_start, int64_t right_index_end);

    static PhysicalExpression *cur_join_expr;

    virtual arrow::Datum join_expr_internal(
        array_info **left_table, array_info **right_table, void **left_data,
        void **right_data, void **left_null_bitmap, void **right_null_bitmap,
        int64_t left_index, int64_t right_index) = 0;

    virtual void ReportMetrics(std::vector<MetricBase> &metrics_out) {};

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
    std::shared_ptr<ExprResult> left_res, const std::string &comparator,
    const arrow::compute::FunctionOptions *func_options = nullptr);

/**
 * @brief Convert two ExprResults to arrow and run compute operation on them.
 *
 */
std::shared_ptr<array_info> do_arrow_compute_binary(
    std::shared_ptr<ExprResult> left_res, std::shared_ptr<ExprResult> right_res,
    const std::string &comparator,
    const std::shared_ptr<arrow::DataType> result_type = nullptr);

/**
 * @brief Convert ExprResult to arrow and cast to the requested type.
 *
 */
std::shared_ptr<array_info> do_arrow_compute_cast(
    std::shared_ptr<ExprResult> left_res,
    const duckdb::LogicalType &return_type);

/**
 * @brief Convert ExprResult to arrow and run case compute on them.
 *
 */
std::shared_ptr<array_info> do_arrow_compute_case(
    std::shared_ptr<ExprResult> when_res, std::shared_ptr<ExprResult> then_res,
    std::shared_ptr<ExprResult> else_res);

/**
 * @brief Run arrow compute operation on unary Datum.
 *
 */
arrow::Datum do_arrow_compute_unary(arrow::Datum left_res,
                                    const std::string &comparator);

/**
 * @brief Run arrow compute operation on two Datums.
 *
 */
arrow::Datum do_arrow_compute_binary(arrow::Datum left_res,
                                     arrow::Datum right_res,
                                     const std::string &comparator);

/**
 * @brief Run cast on arrow Datum.
 *
 */
arrow::Datum do_arrow_compute_cast(arrow::Datum left_res,
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
        return std::make_shared<ArrayExprResult>(result,
                                                 "Comparison" + comparator);
    }

    virtual arrow::Datum join_expr_internal(
        array_info **left_table, array_info **right_table, void **left_data,
        void **right_data, void **left_null_bitmap, void **right_null_bitmap,
        int64_t left_index, int64_t right_index) {
        arrow::Datum left_datum = children[0]->join_expr_internal(
            left_table, right_table, left_data, right_data, left_null_bitmap,
            right_null_bitmap, left_index, right_index);
        arrow::Datum right_datum = children[1]->join_expr_internal(
            left_table, right_table, left_data, right_data, left_null_bitmap,
            right_null_bitmap, left_index, right_index);
        arrow::Datum ret =
            do_arrow_compute_binary(left_datum, right_datum, comparator);
        // Pandas if either is NULL then result is false.
        if (!ret.scalar()->is_valid) {
            ret = arrow::Datum(std::make_shared<arrow::BooleanScalar>(false));
        }
        return ret;
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
std::shared_ptr<arrow::Array> ScalarToArrowArray(
    const std::shared_ptr<arrow::Scalar> &value, size_t num_elements = 1);

// bool specialization
std::shared_ptr<arrow::Array> ScalarToArrowArray(bool value,
                                                 size_t num_elements = 1);

// Numeric type specialization
template <typename T, typename std::enable_if<std::is_arithmetic<T>::value &&
                                                  !std::is_same<T, bool>::value,
                                              int>::type = 0>
std::shared_ptr<arrow::Array> NullArrowArray(const T &value,
                                             size_t num_elements = 1) {
    using ArrowType = typename arrow::CTypeTraits<T>::ArrowType;
    using BuilderType = arrow::NumericBuilder<ArrowType>;

    BuilderType builder;
    arrow::Status status;
    status = builder.AppendNulls(num_elements);
    if (!status.ok()) {
        throw std::runtime_error("builder.AppendNulls failed.");
    }
    std::shared_ptr<arrow::Array> array;
    status = builder.Finish(&array);
    if (!status.ok()) {
        throw std::runtime_error("builder.Finish failed.");
    }
    return array;
}

// String specialization
std::shared_ptr<arrow::Array> NullArrowArray(const std::string &value,
                                             size_t num_elements = 1);

// arrow::Scalar specialization
std::shared_ptr<arrow::Array> NullArrowArray(
    const std::shared_ptr<arrow::Scalar> &value, size_t num_elements = 1);

// bool specialization
std::shared_ptr<arrow::Array> NullArrowArray(bool value,
                                             size_t num_elements = 1);

/**
 * @brief Physical expression tree node type for scalar constants.
 *
 */
template <typename T>
class PhysicalNullExpression : public PhysicalExpression {
   public:
    PhysicalNullExpression(const T &val, bool no_scalars)
        : constant(val), generate_array(no_scalars) {}
    virtual ~PhysicalNullExpression() = default;

    virtual std::shared_ptr<ExprResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch) {
        // The current rule is that if the expression infrastructure
        // is used for filtering then constants are treated as
        // scalars and if used for projection then constants become
        // full columns.  If used in a projection then generate_array
        // will be true and we generate an array the size of the
        // batch and return an ArrayExprResult.
        if (generate_array) {
            std::shared_ptr<arrow::Array> array =
                NullArrowArray(constant, input_batch->nrows());

            auto result =
                arrow_array_to_bodo(array, bodo::BufferPool::DefaultPtr());
            return std::make_shared<ArrayExprResult>(std::move(result), "Null");
        } else {
            std::shared_ptr<arrow::Array> array = NullArrowArray(constant, 1);

            auto result =
                arrow_array_to_bodo(array, bodo::BufferPool::DefaultPtr());
            return std::make_shared<ScalarExprResult>(std::move(result));
        }
    }

    virtual arrow::Datum join_expr_internal(
        array_info **left_table, array_info **right_table, void **left_data,
        void **right_data, void **left_null_bitmap, void **right_null_bitmap,
        int64_t left_index, int64_t right_index) {
        throw std::runtime_error(
            "PhysicalNullExpression::join_expr_internal unimplemented ");
    }

    friend std::ostream &operator<<(std::ostream &os,
                                    const PhysicalNullExpression<T> &obj) {
        os << "PhysicalNullExpression " << std::endl;
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
class PhysicalConstantExpression : public PhysicalExpression {
   public:
    PhysicalConstantExpression(const T &val, bool no_scalars)
        : constant(val), generate_array(no_scalars) {}
    virtual ~PhysicalConstantExpression() = default;

    virtual std::shared_ptr<ExprResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch) {
        // The current rule is that if the expression infrastructure
        // is used for filtering then constants are treated as
        // scalars and if used for projection then constants become
        // full columns.  If used in a projection then generate_array
        // will be true and we generate an array the size of the
        // batch and return an ArrayExprResult.
        if (generate_array) {
            std::shared_ptr<arrow::Array> array =
                ScalarToArrowArray(constant, input_batch->nrows());

            auto result =
                arrow_array_to_bodo(array, bodo::BufferPool::DefaultPtr());
            return std::make_shared<ArrayExprResult>(std::move(result),
                                                     "Constant");
        } else {
            std::shared_ptr<arrow::Array> array = ScalarToArrowArray(constant);

            auto result =
                arrow_array_to_bodo(array, bodo::BufferPool::DefaultPtr());
            return std::make_shared<ScalarExprResult>(std::move(result));
        }
    }

    virtual arrow::Datum join_expr_internal(
        array_info **left_table, array_info **right_table, void **left_data,
        void **right_data, void **left_null_bitmap, void **right_null_bitmap,
        int64_t left_index, int64_t right_index) {
        return arrow::Datum(constant);
    }

    friend std::ostream &operator<<(std::ostream &os,
                                    const PhysicalConstantExpression<T> &obj) {
        os << "PhysicalConstantExpression " << obj.constant << std::endl;
        return os;
    }

   private:
    const T constant;
    const bool generate_array;
};

template <>
class PhysicalConstantExpression<std::string> : public PhysicalExpression {
   public:
    PhysicalConstantExpression(const std::string &val, bool no_scalars)
        : constant(val), generate_array(no_scalars) {}
    virtual ~PhysicalConstantExpression() = default;

    virtual std::shared_ptr<ExprResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch) {
        if (generate_array) {
            std::shared_ptr<arrow::Array> array =
                ScalarToArrowArray(constant, input_batch->nrows());

            auto result =
                arrow_array_to_bodo(array, bodo::BufferPool::DefaultPtr());
            return std::make_shared<ArrayExprResult>(std::move(result),
                                                     "StringConstant");
        } else {
            std::shared_ptr<arrow::Array> array = ScalarToArrowArray(constant);

            auto result =
                arrow_array_to_bodo(array, bodo::BufferPool::DefaultPtr());
            return std::make_shared<ScalarExprResult>(std::move(result));
        }
    }

    virtual arrow::Datum join_expr_internal(
        array_info **left_table, array_info **right_table, void **left_data,
        void **right_data, void **left_null_bitmap, void **right_null_bitmap,
        int64_t left_index, int64_t right_index) {
        return arrow::Datum(constant);
    }

    friend std::ostream &operator<<(
        std::ostream &os, const PhysicalConstantExpression<std::string> &obj) {
        os << "PhysicalConstantExpression<string> " << obj.constant
           << std::endl;
        return os;
    }

   private:
    const std::string constant;
    bool generate_array;
};

/**
 * @brief Physical expression tree node type for getting column from table.
 *
 */
class PhysicalColumnRefExpression : public PhysicalExpression {
   public:
    PhysicalColumnRefExpression(size_t column, const std::string &_bound_name,
                                bool _left_side = true)
        : col_idx(column), bound_name(_bound_name), left_side(_left_side) {}
    virtual ~PhysicalColumnRefExpression() = default;

    virtual std::shared_ptr<ExprResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch) {
        std::shared_ptr<array_info> res_array = input_batch->columns[col_idx];

        std::string column_name;
        if (input_batch->column_names.size() > 0) {
            column_name = input_batch->column_names[col_idx];
        } else {
            column_name = bound_name;
        }
        return std::make_shared<ArrayExprResult>(res_array, column_name);
    }

    arrow::Datum join_expr_internal(array_info **table, void **data,
                                    void **null_bitmap, int64_t index) {
        array_info *sel_col = table[col_idx];
        void *sel_data = data[col_idx];
        std::unique_ptr<bodo::DataType> col_dt = sel_col->data_type();
        std::shared_ptr<::arrow::DataType> arrow_dt = col_dt->ToArrowDataType();
        int32_t dt_byte_width = arrow_dt->byte_width();
        if (dt_byte_width <= 0) {
            throw std::runtime_error(
                "Non-fixed datatype byte width in PhysicalColumnRefExpression "
                "join_expr_internal.");
        }
        void *index_ptr = ((char *)sel_data) + (index * dt_byte_width);
        arrow::Datum ret = ConvertToDatum(index_ptr, arrow_dt);
        if (null_bitmap[col_idx] != nullptr &&
            !GetBit((const uint8_t *)null_bitmap[col_idx], index)) {
            ret = arrow::Datum(arrow::MakeNullScalar(ret.type()));
        }
        return ret;
    }

    virtual arrow::Datum join_expr_internal(
        array_info **left_table, array_info **right_table, void **left_data,
        void **right_data, void **left_null_bitmap, void **right_null_bitmap,
        int64_t left_index, int64_t right_index) {
        if (left_side) {
            return join_expr_internal(left_table, left_data, left_null_bitmap,
                                      left_index);
        } else {
            return join_expr_internal(right_table, right_data,
                                      right_null_bitmap, right_index);
        }
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
        return std::make_shared<ArrayExprResult>(result,
                                                 "Conjunction" + comparator);
    }

    virtual arrow::Datum join_expr_internal(
        array_info **left_table, array_info **right_table, void **left_data,
        void **right_data, void **left_null_bitmap, void **right_null_bitmap,
        int64_t left_index, int64_t right_index) {
        arrow::Datum left_datum = children[0]->join_expr_internal(
            left_table, right_table, left_data, right_data, left_null_bitmap,
            right_null_bitmap, left_index, right_index);
        arrow::Datum right_datum = children[1]->join_expr_internal(
            left_table, right_table, left_data, right_data, left_null_bitmap,
            right_null_bitmap, left_index, right_index);
        arrow::Datum ret =
            do_arrow_compute_binary(left_datum, right_datum, comparator);
        if (!ret.scalar()->is_valid) {
            ret = arrow::Datum(std::make_shared<arrow::BooleanScalar>(false));
        }
        return ret;
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
        return std::make_shared<ArrayExprResult>(result, "Cast");
    }

    virtual arrow::Datum join_expr_internal(
        array_info **left_table, array_info **right_table, void **left_data,
        void **right_data, void **left_null_bitmap, void **right_null_bitmap,
        int64_t left_index, int64_t right_index) {
        throw std::runtime_error(
            "PhysicalCastExpression::join_expr_internal unimplemented ");
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
            case duckdb::ExpressionType::OPERATOR_IS_NOT_NULL:
                comparator = "is_not_null";
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
        return std::make_shared<ArrayExprResult>(result, "Unary" + comparator);
    }

    virtual arrow::Datum join_expr_internal(
        array_info **left_table, array_info **right_table, void **left_data,
        void **right_data, void **left_null_bitmap, void **right_null_bitmap,
        int64_t left_index, int64_t right_index) {
        arrow::Datum left_datum = children[0]->join_expr_internal(
            left_table, right_table, left_data, right_data, left_null_bitmap,
            right_null_bitmap, left_index, right_index);
        return do_arrow_compute_unary(left_datum, comparator);
    }

   protected:
    std::string comparator;
};

void EnsureModRegistered();

/**
 * @brief Physical expression tree node type for binary op non-boolean arrays.
 *
 */
class PhysicalBinaryExpression : public PhysicalExpression {
   public:
    PhysicalBinaryExpression(
        std::shared_ptr<PhysicalExpression> left,
        std::shared_ptr<PhysicalExpression> right, duckdb::ExpressionType etype,
        const std::shared_ptr<arrow::DataType> _result_type = nullptr)
        : result_type(_result_type) {
        children.push_back(left);
        children.push_back(right);
        switch (etype) {
            default:
                throw std::runtime_error(
                    "Unhandled binary expression type " +
                    std::to_string(static_cast<int>(etype)));
        }
    }

    PhysicalBinaryExpression(
        std::shared_ptr<PhysicalExpression> left,
        std::shared_ptr<PhysicalExpression> right, std::string &opstr,
        const std::shared_ptr<arrow::DataType> _result_type = nullptr)
        : result_type(_result_type) {
        children.push_back(left);
        children.push_back(right);
        if (opstr == "+") {
            comparator = "add";
        } else if (opstr == "-") {
            comparator = "subtract";
        } else if (opstr == "*") {
            comparator = "multiply";
        } else if (opstr == "/") {
            comparator = "divide";
        } else if (opstr == "//") {
            // "//" is integer division in DuckDB which is handled by divide
            // function of Arrow
            comparator = "divide";
        } else if (opstr == "floor") {
            comparator = "floor";
        } else if (opstr == "%") {
            EnsureModRegistered();
            comparator = "bodo_mod";
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

        std::shared_ptr<array_info> result = do_arrow_compute_binary(
            left_res, right_res, comparator, result_type);
        return std::make_shared<ArrayExprResult>(result, "Binary" + comparator);
    }

    virtual arrow::Datum join_expr_internal(
        array_info **left_table, array_info **right_table, void **left_data,
        void **right_data, void **left_null_bitmap, void **right_null_bitmap,
        int64_t left_index, int64_t right_index) {
        arrow::Datum left_datum = children[0]->join_expr_internal(
            left_table, right_table, left_data, right_data, left_null_bitmap,
            right_null_bitmap, left_index, right_index);
        arrow::Datum right_datum = children[1]->join_expr_internal(
            left_table, right_table, left_data, right_data, left_null_bitmap,
            right_null_bitmap, left_index, right_index);
        return do_arrow_compute_binary(left_datum, right_datum, comparator);
    }

   protected:
    std::string comparator;
    const std::shared_ptr<arrow::DataType> result_type;
};

/**
 * @brief Physical expression tree node type for case expressions.
 *
 */
class PhysicalCaseExpression : public PhysicalExpression {
   public:
    PhysicalCaseExpression(std::shared_ptr<PhysicalExpression> when_expr,
                           std::shared_ptr<PhysicalExpression> then_expr,
                           std::shared_ptr<PhysicalExpression> else_expr) {
        children.push_back(when_expr);
        children.push_back(then_expr);
        children.push_back(else_expr);
    }

    virtual ~PhysicalCaseExpression() = default;

    /**
     * @brief How to process this expression tree node.
     *
     */
    virtual std::shared_ptr<ExprResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch) {
        // Process children first.
        std::shared_ptr<ExprResult> when_res =
            children[0]->ProcessBatch(input_batch);
        std::shared_ptr<ExprResult> then_res =
            children[1]->ProcessBatch(input_batch);
        std::shared_ptr<ExprResult> else_res =
            children[2]->ProcessBatch(input_batch);

        auto result = do_arrow_compute_case(when_res, then_res, else_res);
        return std::make_shared<ArrayExprResult>(result, "Case");
    }

    virtual arrow::Datum join_expr_internal(
        array_info **left_table, array_info **right_table, void **left_data,
        void **right_data, void **left_null_bitmap, void **right_null_bitmap,
        int64_t left_index, int64_t right_index) {
        throw std::runtime_error(
            "PhysicalCastExpression::join_expr_internal unimplemented ");
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
std::shared_ptr<PhysicalExpression> buildPhysicalExprTree(
    duckdb::unique_ptr<duckdb::Expression> &expr,
    std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t> &col_ref_map,
    bool no_scalars = false);

std::shared_ptr<PhysicalExpression> buildPhysicalExprTree(
    duckdb::Expression &expr,
    std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t> &col_ref_map,
    bool no_scalars = false);

struct PhysicalUDFExpressionMetrics {
    using timer_t = MetricBase::TimerValue;
    timer_t cpp_to_py_time = 0;
    timer_t udf_execution_time = 0;
    timer_t py_to_cpp_time = 0;
};

/**
 * @brief Physical expression tree node type for UDF.
 *
 */
class PhysicalUDFExpression : public PhysicalExpression {
   public:
    PhysicalUDFExpression(
        std::vector<std::shared_ptr<PhysicalExpression>> &children,
        BodoScalarFunctionData &_scalar_func_data,
        const std::shared_ptr<arrow::DataType> &_result_type)
        : PhysicalExpression(children),
          scalar_func_data(_scalar_func_data),
          result_type(_result_type),
          cfunc_ptr(nullptr),
          init_state(nullptr) {
        if (scalar_func_data.is_cfunc) {
            this->cfunc_ptr = (table_udf_t)1;
            PyObject *bodo_module =
                PyImport_ImportModule("bodo.pandas.utils_jit");
            if (!bodo_module) {
                PyErr_Print();
                throw std::runtime_error(
                    "Failed to import bodo.pandas.utils module");
            }
            PyObject *future_args = scalar_func_data.args;
            Py_XINCREF(future_args);
            Py_INCREF(bodo_module);
            // https://docs.python.org/3/c-api/init.html#thread-state-and-the-global-interpreter-lock
            PyThreadState *save = PyEval_SaveThread();

            compile_future = std::async(
                std::launch::async,
                [bodo_module, future_args]() -> table_udf_t {
                    // Ensure we hold the GIL in this thread.
                    PyGILState_STATE gstate = PyGILState_Ensure();
                    try {
                        table_udf_t ptr = nullptr;

                        PyObject *result = PyObject_CallMethod(
                            bodo_module, "compile_cfunc", "O", future_args);
                        if (!result) {
                            PyErr_Print();
                            Py_DECREF(bodo_module);
                            Py_XDECREF(future_args);
                            throw std::runtime_error(
                                "Error calling compile_cfunc");
                        }

                        if (!PyLong_Check(result)) {
                            Py_DECREF(result);
                            Py_DECREF(bodo_module);
                            Py_XDECREF(future_args);
                            throw std::runtime_error(
                                "Expected an integer from compile_cfunc");
                        }

                        ptr = reinterpret_cast<table_udf_t>(
                            PyLong_AsLongLong(result));

                        Py_DECREF(result);
                        Py_DECREF(bodo_module);
                        Py_XDECREF(future_args);
                        PyGILState_Release(gstate);
                        return ptr;
                    } catch (...) {
                        // Release GIL and DECREF args before propagating.
                        PyGILState_Release(gstate);
                        Py_DECREF(bodo_module);
                        Py_XDECREF(future_args);
                        throw;
                    }
                });
            PyEval_RestoreThread(save);
        }
    }

    virtual ~PhysicalUDFExpression() {
        if (init_state != nullptr) {
            Py_DECREF(init_state);
        }
    }

    /**
     * @brief How to process this expression tree node.
     *
     */
    std::shared_ptr<ExprResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch) override;

    arrow::Datum join_expr_internal(array_info **left_table,
                                    array_info **right_table, void **left_data,
                                    void **right_data, void **left_null_bitmap,
                                    void **right_null_bitmap,
                                    int64_t left_index,
                                    int64_t right_index) override {
        throw std::runtime_error(
            "PhysicalUDFExpression::join_expr_internal unimplemented ");
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
    const std::shared_ptr<arrow::DataType> result_type;
    PhysicalUDFExpressionMetrics metrics;
    std::future<table_udf_t> compile_future;
    table_udf_t cfunc_ptr;
    PyObject *init_state;
};

struct PhysicalArrowExpressionMetrics {
    using timer_t = MetricBase::TimerValue;
    timer_t arrow_compute_time = 0;
};
/**
 * @brief Physical expression tree node type for Arrow Compute functions.
 *
 */
class PhysicalArrowExpression : public PhysicalExpression {
   public:
    PhysicalArrowExpression(
        std::vector<std::shared_ptr<PhysicalExpression>> &children,
        BodoScalarFunctionData &_scalar_func_data,
        const std::shared_ptr<arrow::DataType> &_result_type)
        : PhysicalExpression(children),
          scalar_func_data(_scalar_func_data),
          result_type(_result_type) {}

    /**
     * @brief How to process this expression tree node.
     *
     */
    std::shared_ptr<ExprResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch) override;

    arrow::Datum join_expr_internal(array_info **left_table,
                                    array_info **right_table, void **left_data,
                                    void **right_data, void **left_null_bitmap,
                                    void **right_null_bitmap,
                                    int64_t left_index,
                                    int64_t right_index) override {
        throw std::runtime_error(
            "PhysicalArrowExpression::join_expr_internal unimplemented ");
    }

    void ReportMetrics(std::vector<MetricBase> &metrics_out) override {
        metrics_out.push_back(
            TimerMetric("arrow_compute_time", metrics.arrow_compute_time));
    }

   protected:
    BodoScalarFunctionData scalar_func_data;
    const std::shared_ptr<arrow::DataType> result_type;
    PhysicalArrowExpressionMetrics metrics;
};
