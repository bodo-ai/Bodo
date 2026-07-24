#pragma once

#include <arrow/api.h>
#include <arrow/chunked_array.h>
#include <arrow/compute/api.h>
#include <arrow/compute/api_scalar.h>
#include <arrow/compute/function.h>
#include <arrow/compute/kernel.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/type_fwd.h>
#include <arrow/type_traits.h>
#include <mpi.h>
#include <future>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include "../libs/_array_utils.h"
#include "../libs/_bodo_common.h"
#include "../libs/_bodo_to_arrow.h"
#include "../tests/utils.h"
#include "_util.h"
#include "duckdb/common/enums/expression_type.hpp"
#include "duckdb/common/types/interval.hpp"
#include "duckdb/planner/column_binding.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/expression/bound_between_expression.hpp"
#include "operator.h"

std::shared_ptr<arrow::Array> prepare_arrow_compute(
    std::shared_ptr<array_info> arr);

// Converts a string unit to the corresponding Arrow CalendarUnit.
arrow::compute::CalendarUnit getArrowCalendarUnit(const char *unit_str);

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
 * @brief Identifies the specific subclass type of a PhysicalExpression.
 */
enum class PhysicalExpressionType {
    INVALID = 0,
    COMPARISON,
    NULL_CONSTANT,
    CONSTANT,
    COLUMN_REF,
    CONJUNCTION,
    CAST,
    UNARY,
    BINARY,
    CASE,
    UDF,
    ARROW,
    CALENDAR_INTERVAL
};

arrow::Datum fill_null(arrow::Datum &src, const arrow::Datum &val);

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
        std::vector<std::shared_ptr<PhysicalExpression>> &_children,
        PhysicalExpressionType type)
        : children(_children), phys_expr_type(type) {}
    PhysicalExpression(PhysicalExpressionType type) : phys_expr_type(type) {}
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

    virtual void Finalize() {
        // Base implementation just calls Finalize()
        // on all the children in case the child happens
        // to have an override of Finalize(), e.g.
        // PhysicalArrowExpression.
        for (const auto &child : children) {
            child->Finalize();
        }
    }

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

    PhysicalExpressionType GetExpressionType() const { return phys_expr_type; }

    std::vector<std::shared_ptr<PhysicalExpression>> &GetChildren() {
        return children;
    }

   protected:
    std::vector<std::shared_ptr<PhysicalExpression>> children;

   private:
    PhysicalExpressionType phys_expr_type = PhysicalExpressionType::INVALID;
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
 * @brief Perform an Arrow compute operation with multiple input expressions.
 *
 * @param in_expr_results A vector of input expression results.
 * @param arrow_func_name The name of the Arrow function to call.
 * @return arrow::Datum The result of the Arrow compute
 * operation.
 */
arrow::Datum do_arrow_compute_multi_input_datum(
    const std::vector<std::shared_ptr<ExprResult>> &in_expr_results,
    const std::string &arrow_func_name);

/**
 * @brief Perform an Arrow compute operation with multiple input expressions.
 *
 * @param in_expr_results A vector of input expression results.
 * @param arrow_func_name The name of the Arrow function to call.
 * @return std::shared_ptr<array_info> The result of the Arrow compute
 * operation.
 */
std::shared_ptr<array_info> do_arrow_compute_multi_input(
    const std::vector<std::shared_ptr<ExprResult>> &in_expr_results,
    const std::string &arrow_func_name);

/**
 * @brief Convert ExprResult to arrow and run compute operation on it.
 *
 */
std::shared_ptr<array_info> do_arrow_compute_unary(
    std::shared_ptr<ExprResult> left_res, const std::string &comparator,
    const arrow::compute::FunctionOptions *func_options = nullptr,
    const std::shared_ptr<arrow::DataType> result_type = nullptr);

/**
 * @brief Convert two ExprResults to arrow and run compute operation on them.
 *
 */
std::shared_ptr<array_info> do_arrow_compute_binary(
    std::shared_ptr<ExprResult> left_res, std::shared_ptr<ExprResult> right_res,
    const std::string &comparator,
    const arrow::compute::FunctionOptions *func_options = nullptr,
    const std::shared_ptr<arrow::DataType> result_type = nullptr);

/**
 * @brief Run arrow compute on the left Datum and right ExprResult after
 * converting the ExprResult to an Arrow Datum.
 *
 */
std::shared_ptr<array_info> do_arrow_compute_binary(
    arrow::Datum left_res, std::shared_ptr<ExprResult> right_res,
    const std::string &comparator,
    const arrow::compute::FunctionOptions *func_options = nullptr,
    const std::shared_ptr<arrow::DataType> result_type = nullptr);

/**
 * @brief Run arrow compute on the left ExprResult and right Datum after
 * converting the ExprResult to an Arrow Datum.
 *
 */
std::shared_ptr<array_info> do_arrow_compute_binary(
    std::shared_ptr<ExprResult> left_res, arrow::Datum right_res,
    const std::string &comparator,
    const arrow::compute::FunctionOptions *func_options = nullptr,
    const std::shared_ptr<arrow::DataType> result_type = nullptr);

/**
 * @brief Convert ExprResult to arrow and cast to the requested type.
 *
 */
std::shared_ptr<array_info> do_arrow_compute_cast(
    std::shared_ptr<ExprResult> left_res,
    const std::shared_ptr<arrow::DataType> &return_type);

/**
 * @brief Convert ExprResult to arrow and run case compute on them.
 *
 */
std::shared_ptr<array_info> do_arrow_compute_case(
    std::shared_ptr<ExprResult> when_res, std::shared_ptr<ExprResult> then_res,
    std::shared_ptr<ExprResult> else_res,
    const std::shared_ptr<arrow::DataType> result_type = nullptr);

/**
 * @brief Run arrow compute operation on unary Datum.
 *
 */
arrow::Datum do_arrow_compute_unary(
    arrow::Datum left_res, const std::string &comparator,
    const arrow::compute::FunctionOptions *func_options = nullptr,
    const std::shared_ptr<arrow::DataType> result_type = nullptr);

/**
 * @brief Run arrow compute operation on two Datums.
 *
 */
arrow::Datum do_arrow_compute_binary(
    arrow::Datum left_res, arrow::Datum right_res,
    const std::string &comparator,
    const arrow::compute::FunctionOptions *func_options = nullptr,
    const std::shared_ptr<arrow::DataType> result_type = nullptr);

/**
 * @brief Run cast on arrow Datum.
 *
 */
arrow::Datum do_arrow_compute_cast(
    arrow::Datum left_res, const std::shared_ptr<arrow::DataType> &return_type);

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
        : PhysicalExpression(PhysicalExpressionType::COMPARISON) {
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
        auto left_as_scalar =
            std::dynamic_pointer_cast<ScalarExprResult>(left_res);
        auto right_as_scalar =
            std::dynamic_pointer_cast<ScalarExprResult>(right_res);
        if (left_as_scalar && right_as_scalar) {
            return std::make_shared<ScalarExprResult>(result);
        }
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
        : PhysicalExpression(PhysicalExpressionType::NULL_CONSTANT),
          constant(val),
          generate_array(no_scalars) {}
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
        return arrow::Datum(
            arrow::MakeNullScalar(ScalarToArrowArray(constant)->type()));
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
        : PhysicalExpression(PhysicalExpressionType::CONSTANT),
          constant(val),
          generate_array(no_scalars) {}
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
        : PhysicalExpression(PhysicalExpressionType::CONSTANT),
          constant(val),
          generate_array(no_scalars) {}
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
    PhysicalColumnRefExpression(size_t column,
                                duckdb::ColumnBinding _col_binding,
                                const std::string &_bound_name,
                                bool _left_side = true)
        : PhysicalExpression(PhysicalExpressionType::COLUMN_REF),
          col_idx(column),
          col_binding(_col_binding),
          bound_name(_bound_name),
          left_side(_left_side) {}
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

        arrow::Datum ret;
        std::unique_ptr<bodo::DataType> col_dt = sel_col->data_type();
        std::shared_ptr<::arrow::DataType> arrow_dt = col_dt->ToArrowDataType();
        int32_t dt_byte_width = arrow_dt->byte_width();
        if (dt_byte_width <= 0) {
            if (col_dt->array_type == bodo_array_type::STRING) {
                // For strings we need to read the offset and then
                // For variable-width types we need to read the offset and then
                // read the value.
                char *null_bitmask = sel_col->null_bitmask();
                if (!arrow::bit_util::GetBit((const uint8_t *)null_bitmask,
                                             index)) {
                    return arrow::Datum(arrow::MakeNullScalar(arrow_dt));
                }
                offset_t start_offset =
                    sel_col->data2<bodo_array_type::STRING, offset_t>()[index];
                offset_t end_offset =
                    sel_col
                        ->data2<bodo_array_type::STRING, offset_t>()[index + 1];
                std::string str(
                    sel_col->data1<bodo_array_type::STRING>() + start_offset,
                    end_offset - start_offset);
                ret = arrow::Datum(std::make_shared<arrow::StringScalar>(str));

            } else {
                throw std::runtime_error(
                    "Unsupported variable-width type in join_expr_internal.");
            }

        } else {
            // For fixed-width types we can just calculate the offset and read
            // the value.
            void *index_ptr = ((char *)sel_data) + (index * dt_byte_width);
            ret = ConvertToDatum(index_ptr, arrow_dt);
        }
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

    void set_left_side(bool _left_side) { left_side = _left_side; }

    duckdb::ColumnBinding get_col_binding() const { return col_binding; }

   protected:
    size_t col_idx;
    duckdb::ColumnBinding col_binding;
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
                                  duckdb::ExpressionType etype)
        : PhysicalExpression(PhysicalExpressionType::CONJUNCTION) {
        children.push_back(left);
        children.push_back(right);
        switch (etype) {
            case duckdb::ExpressionType::CONJUNCTION_AND:
                // We use the Kleene logic versions to match null behavior of
                // pandas and SQL. See
                // https://github.com/pandas-dev/pandas/blob/366ccdfcd8ed1e5543bfb6d4ee0c9bc519898670/pandas/core/arrays/arrow/array.py#L110
                comparator = "and_kleene";
                break;
            case duckdb::ExpressionType::CONJUNCTION_OR:
                comparator = "or_kleene";
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
        std::shared_ptr<ScalarExprResult> left_as_scalar =
            std::dynamic_pointer_cast<ScalarExprResult>(left_res);
        // Implement short-circuit for scalars.
        if (left_as_scalar) {
            bool left_bool = left_as_scalar->result->at<bool>(0);
            if (comparator == "and" && !left_bool) {
                return left_res;
            }
            if (comparator == "or" && left_bool) {
                return left_res;
            }
        }
        std::shared_ptr<ExprResult> right_res =
            children[1]->ProcessBatch(input_batch);

        auto result = do_arrow_compute_binary(left_res, right_res, comparator);
        auto right_as_scalar =
            std::dynamic_pointer_cast<ScalarExprResult>(right_res);
        if (left_as_scalar && right_as_scalar) {
            return std::make_shared<ScalarExprResult>(result);
        }
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
        // Implement short-circuit for scalars.
        if (left_datum.is_scalar()) {
            auto left_scalar = left_datum.scalar();
            if (left_scalar->type->id() == arrow::Type::BOOL) {
                auto bool_sc =
                    std::static_pointer_cast<arrow::BooleanScalar>(left_scalar);
                if (bool_sc->is_valid) {
                    bool left_bool = bool_sc->value;
                    if (comparator == "and" && !left_bool) {
                        return left_datum;
                    }
                    if (comparator == "or" && left_bool) {
                        return left_datum;
                    }
                } else {
                    throw std::runtime_error(
                        "join_expr_internal for conjunction left_datum was "
                        "null bool.");
                }
            } else {
                throw std::runtime_error(
                    "join_expr_internal for conjunction left_datum wasn't "
                    "bool.");
            }
        }
        arrow::Datum right_datum = children[1]->join_expr_internal(
            left_table, right_table, left_data, right_data, left_null_bitmap,
            right_null_bitmap, left_index, right_index);
        arrow::Datum ret =
            do_arrow_compute_binary(left_datum, right_datum, comparator);
        if (ret.is_scalar() && !ret.scalar()->is_valid) {
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
                           std::shared_ptr<arrow::DataType> _return_type)
        : PhysicalExpression(PhysicalExpressionType::CAST),
          return_type(_return_type) {
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
        auto left_as_scalar =
            std::dynamic_pointer_cast<ScalarExprResult>(left_res);
        if (left_as_scalar) {
            return std::make_shared<ScalarExprResult>(result);
        }
        return std::make_shared<ArrayExprResult>(result, "Cast");
    }

    virtual arrow::Datum join_expr_internal(
        array_info **left_table, array_info **right_table, void **left_data,
        void **right_data, void **left_null_bitmap, void **right_null_bitmap,
        int64_t left_index, int64_t right_index) {
        arrow::Datum left_datum = children[0]->join_expr_internal(
            left_table, right_table, left_data, right_data, left_null_bitmap,
            right_null_bitmap, left_index, right_index);
        return do_arrow_compute_cast(left_datum, return_type);
    }

   protected:
    std::shared_ptr<arrow::DataType> return_type;
};

/**
 * @brief Physical expression tree node type for unary of array.
 *
 */
class PhysicalUnaryExpression : public PhysicalExpression {
   public:
    PhysicalUnaryExpression(
        std::shared_ptr<PhysicalExpression> left, duckdb::ExpressionType etype,
        const std::shared_ptr<arrow::DataType> _result_type = nullptr)
        : PhysicalExpression(PhysicalExpressionType::UNARY),
          result_type(_result_type) {
        children.push_back(left);
        switch (etype) {
            case duckdb::ExpressionType::OPERATOR_NOT:
                comparator = "invert";
                break;
            case duckdb::ExpressionType::OPERATOR_IS_NOT_NULL:
                comparator = "is_not_null";
                break;
            case duckdb::ExpressionType::OPERATOR_IS_NULL:
                comparator = "is_null";
                break;
            case duckdb::ExpressionType::OPERATOR_IS_TRUE:
                comparator = "is_true";
                break;
            case duckdb::ExpressionType::OPERATOR_IS_NOT_TRUE:
                comparator = "is_not_true";
                break;
            case duckdb::ExpressionType::OPERATOR_IS_FALSE:
                comparator = "is_false";
                break;
            case duckdb::ExpressionType::OPERATOR_IS_NOT_FALSE:
                comparator = "is_not_false";
                break;
            case duckdb::ExpressionType::OPERATOR_NEG:
                comparator = "negate";
                break;
            default:
                throw std::runtime_error("Unhandled unary op expression type.");
        }
    }
    PhysicalUnaryExpression(
        std::shared_ptr<PhysicalExpression> left, std::string &opstr,
        const std::shared_ptr<arrow::DataType> _result_type = nullptr)
        : PhysicalExpression(PhysicalExpressionType::UNARY),
          result_type(_result_type) {
        children.push_back(left);
        if (opstr == "floor" || opstr == "ceil" || opstr == "abs" ||
            opstr == "sqrt" || opstr == "cbrt" || opstr == "exp" ||
            opstr == "ln" || opstr == "log10" || opstr == "log2" ||
            opstr == "round" || opstr == "trunc" || opstr == "sign") {
            comparator = opstr;
        } else {
            throw std::runtime_error("Unhandled unary expression opstr: " +
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

        arrow::compute::FunctionOptions *func_options;
        if (comparator == "round") {
            // Override the default Arrow round mode to
            // HALF_TOWARDS_INFINITY to match SQL semantics.
            // See
            // https://docs.bodo.ai/latest/api_docs/sql/functions/numeric/round/
            arrow::compute::RoundOptions opts(
                0, arrow::compute::RoundMode::HALF_TOWARDS_INFINITY);
            func_options = &opts;
        } else {
            func_options = nullptr;
        }

        std::shared_ptr<array_info> result;
        if (comparator == "cbrt") {
            // Arrow doesn't have a cube root function, so we raise to the power
            // 1/3 instead. However Arrow will return NaN for a negative base in
            // this case, so we have to first take the absolute value and then
            // restore the sign afterwards.
            arrow::Datum left_res_datum =
                ConvertExprResultToDatum(left_res, "cbrt input");

            arrow::Datum abs_left_res =
                do_arrow_compute_unary(left_res_datum, "abs");

            arrow::Datum exponent_datum(
                std::make_shared<arrow::FloatScalar>(1.0 / 3.0));
            arrow::Datum power_result = do_arrow_compute_binary(
                abs_left_res, exponent_datum, "power", func_options);

            arrow::Datum left_res_sign =
                do_arrow_compute_unary(left_res_datum, "sign");
            arrow::Datum corrected_power_result = do_arrow_compute_binary(
                power_result, left_res_sign, "multiply", nullptr, result_type);

            result = ConvertDatumToArrayInfo(corrected_power_result);
        } else {
            result = do_arrow_compute_unary(left_res, comparator, func_options,
                                            result_type);
        }

        auto left_as_scalar =
            std::dynamic_pointer_cast<ScalarExprResult>(left_res);
        if (left_as_scalar) {
            return std::make_shared<ScalarExprResult>(result);
        }

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
    const std::shared_ptr<arrow::DataType> result_type;
};

void EnsureModRegistered();
void EnsureSubstrRegistered();

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
        : PhysicalExpression(PhysicalExpressionType::BINARY),
          result_type(_result_type) {
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
        } else if (opstr == "%") {
            EnsureModRegistered();
            comparator = "bodo_mod";
        } else if (opstr == "floor" || opstr == "power") {
            comparator = opstr;
        } else if (opstr == "round") {
            comparator = "round_binary";
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

        arrow::compute::FunctionOptions *func_options;
        if (comparator == "round_binary") {
            // Override the default Arrow round mode to
            // HALF_TOWARDS_INFINITY to match SQL semantics.
            // See
            // https://docs.bodo.ai/latest/api_docs/sql/functions/numeric/round/
            arrow::compute::RoundBinaryOptions opts(
                arrow::compute::RoundMode::HALF_TOWARDS_INFINITY);
            func_options = &opts;
        } else {
            func_options = nullptr;
        }
        std::shared_ptr<array_info> result = do_arrow_compute_binary(
            left_res, right_res, comparator, func_options, result_type);

        auto left_as_scalar =
            std::dynamic_pointer_cast<ScalarExprResult>(left_res);
        auto right_as_scalar =
            std::dynamic_pointer_cast<ScalarExprResult>(right_res);
        if (left_as_scalar && right_as_scalar) {
            return std::make_shared<ScalarExprResult>(result);
        }
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
 * @brief Physical expression tree node for calendar-aware interval arithmetic
 * (year/month offsets) that Arrow's duration-based add cannot handle.
 * Uses DuckDB's Interval::Add() for month-end clamping.
 *
 */
class PhysicalCalendarIntervalExpression : public PhysicalExpression {
   public:
    PhysicalCalendarIntervalExpression(
        std::shared_ptr<PhysicalExpression> date_child,
        duckdb::interval_t interval, bool is_subtract,
        std::shared_ptr<arrow::DataType> result_type)
        : PhysicalExpression(PhysicalExpressionType::CALENDAR_INTERVAL),
          date_child(std::move(date_child)),
          calendar_interval(interval),
          is_subtract(is_subtract),
          result_type(std::move(result_type)) {}

    virtual ~PhysicalCalendarIntervalExpression() = default;

    virtual std::shared_ptr<ExprResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch);

    virtual arrow::Datum join_expr_internal(
        array_info **left_table, array_info **right_table, void **left_data,
        void **right_data, void **left_null_bitmap, void **right_null_bitmap,
        int64_t left_index, int64_t right_index);

   private:
    std::shared_ptr<PhysicalExpression> date_child;
    duckdb::interval_t calendar_interval;
    bool is_subtract;
    const std::shared_ptr<arrow::DataType> result_type;
};

/**
 * @brief Physical expression tree node type for case expressions.
 *
 */
class PhysicalCaseExpression : public PhysicalExpression {
   public:
    PhysicalCaseExpression(
        std::shared_ptr<PhysicalExpression> when_expr,
        std::shared_ptr<PhysicalExpression> then_expr,
        std::shared_ptr<PhysicalExpression> else_expr,
        const std::shared_ptr<arrow::DataType> _result_type = nullptr)
        : PhysicalExpression(PhysicalExpressionType::CASE),
          result_type(_result_type) {
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

        // Arrow uses NULL for the "else" case if it is not provided.
        // Bodo creates string arrays for null data type currently
        // which can cause errors in Arrow so this check is necessary.
        std::shared_ptr<ExprResult> else_res = nullptr;
        if (children[2]->GetExpressionType() !=
            PhysicalExpressionType::NULL_CONSTANT) {
            else_res = children[2]->ProcessBatch(input_batch);
        }

        auto result =
            do_arrow_compute_case(when_res, then_res, else_res, result_type);

        auto when_as_scalar =
            std::dynamic_pointer_cast<ScalarExprResult>(when_res);
        auto then_as_scalar =
            std::dynamic_pointer_cast<ScalarExprResult>(then_res);
        auto else_as_scalar =
            std::dynamic_pointer_cast<ScalarExprResult>(else_res);
        if (when_as_scalar && then_as_scalar &&
            (else_res == nullptr || else_as_scalar)) {
            return std::make_shared<ScalarExprResult>(result);
        }
        return std::make_shared<ArrayExprResult>(result, "Case");
    }

    virtual arrow::Datum join_expr_internal(
        array_info **left_table, array_info **right_table, void **left_data,
        void **right_data, void **left_null_bitmap, void **right_null_bitmap,
        int64_t left_index, int64_t right_index) {
        arrow::Datum when_datum = children[0]->join_expr_internal(
            left_table, right_table, left_data, right_data, left_null_bitmap,
            right_null_bitmap, left_index, right_index);
        arrow::Datum then_datum = children[1]->join_expr_internal(
            left_table, right_table, left_data, right_data, left_null_bitmap,
            right_null_bitmap, left_index, right_index);
        arrow::Datum else_datum = children[2]->join_expr_internal(
            left_table, right_table, left_data, right_data, left_null_bitmap,
            right_null_bitmap, left_index, right_index);

        arrow::Result<arrow::Datum> cmp_res = arrow::compute::CallFunction(
            "if_else", {when_datum, then_datum, else_datum});
        if (!cmp_res.ok()) [[unlikely]] {
            throw std::runtime_error(
                "do_arrow_compute_case if_else: Error in Arrow compute: " +
                cmp_res.status().message());
        }
        return cmp_res.ValueOrDie();
    }

   private:
    const std::shared_ptr<arrow::DataType> result_type;
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
        : PhysicalExpression(children, PhysicalExpressionType::UDF),
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
        std::vector<std::shared_ptr<array_info>> child_arrays;
        for (const auto &child : children) {
            arrow::Datum child_datum = child->join_expr_internal(
                left_table, right_table, left_data, right_data,
                left_null_bitmap, right_null_bitmap, left_index, right_index);
            std::shared_ptr<arrow::Array> child_arrow_array;
            if (child_datum.is_array()) {
                child_arrow_array = child_datum.make_array();
            } else if (child_datum.is_scalar()) {
                // Convert scalar to array of length 1 for UDF input.
                child_arrow_array = ScalarToArrowArray(child_datum.scalar(), 1);
            } else {
                throw std::runtime_error(
                    "Child datum is neither array nor scalar in "
                    "PhysicalUDFExpression::join_expr_internal");
            }
            auto child_array_info = arrow_array_to_bodo(
                child_arrow_array, bodo::BufferPool::DefaultPtr());
            child_arrays.push_back(std::move(child_array_info));
        }
        std::shared_ptr<table_info> udf_input =
            std::make_shared<table_info>(std::move(child_arrays));
        // Actually run the UDF.
        std::shared_ptr<table_info> udf_output;
        if (cfunc_ptr) {
            if (cfunc_ptr == (table_udf_t)1) {
                PyThreadState *save = PyEval_SaveThread();
                cfunc_ptr = compile_future.get();
                PyEval_RestoreThread(save);
            }
            time_pt start_init_time = start_timer();
            udf_output = runCfuncScalarFunction(udf_input, cfunc_ptr);
            this->metrics.udf_execution_time += end_timer(start_init_time);
        } else {
            auto [out_temp, cpp_to_py_time, udf_time, py_to_cpp_time] =
                runPythonScalarFunction(udf_input, result_type,
                                        scalar_func_data.args,
                                        scalar_func_data.has_state, init_state);
            udf_output = out_temp;
            // Update the metrics.
            this->metrics.cpp_to_py_time += cpp_to_py_time;
            this->metrics.udf_execution_time += udf_time;
            this->metrics.py_to_cpp_time += py_to_cpp_time;
        }
        std::shared_ptr<arrow::ChunkedArray> res_array =
            bodo_table_to_arrow(udf_output)->column(0);
        arrow::Result<std::shared_ptr<arrow::Scalar>> res_scalar =
            res_array->GetScalar(0);
        if (!res_scalar.ok()) {
            throw std::runtime_error(
                "Error getting scalar from UDF result array: " +
                res_scalar.status().message());
        }

        return res_scalar.ValueOrDie();
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
        : PhysicalExpression(children, PhysicalExpressionType::ARROW),
          scalar_func_data(_scalar_func_data),
          result_type(_result_type) {}

    /**
     * @brief How to process this expression tree node.
     *
     */
    std::shared_ptr<ExprResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch) override;

    void Finalize() override {
        for (const auto &child : children) {
            child->Finalize();
        }

        gen = nullptr;  // Reset RNG
        named_regexp = nullptr;
    }

    arrow::Datum join_expr_internal(array_info **left_table,
                                    array_info **right_table, void **left_data,
                                    void **right_data, void **left_null_bitmap,
                                    void **right_null_bitmap,
                                    int64_t left_index,
                                    int64_t right_index) override {
        arrow::Datum child_datum = children[0]->join_expr_internal(
            left_table, right_table, left_data, right_data, left_null_bitmap,
            right_null_bitmap, left_index, right_index);
        return do_arrow_compute(child_datum);
    }

    void ReportMetrics(std::vector<MetricBase> &metrics_out) override {
        metrics_out.push_back(
            TimerMetric("arrow_compute_time", metrics.arrow_compute_time));
    }

   protected:
    // PRNG for random_int64
    std::shared_ptr<std::mt19937_64> gen;
    // Named regex pattern for REGEXP_SUBSTR / REGEXP_INSTR
    std::shared_ptr<std::tuple<std::string, int>> named_regexp;

    BodoScalarFunctionData scalar_func_data;
    const std::shared_ptr<arrow::DataType> result_type;
    PhysicalArrowExpressionMetrics metrics;

    template <typename T>
    using compute_return_t =
        std::conditional_t<std::is_same_v<T, std::shared_ptr<ExprResult>>,
                           std::shared_ptr<array_info>, T>;

    std::tuple<std::string, int> convert_to_named_regexp(
        std::string pattern_str) {
        /* Return a tuple of the input regex pattern with its capturing
        groups converted to named groups, and the number of groups that were
        found in the pattern. */

        std::string named_pattern;
        // Number of groups found in regex pattern so far
        int num_groups = 0;

        // Convert all groups to _groupN format for extract_regex
        for (size_t i = 0; i < pattern_str.length(); i++) {
            // Handle escaped characters by reading the backslash and the
            // following character together
            if (pattern_str[i] == '\\' && i + 1 < pattern_str.length()) {
                named_pattern += pattern_str[i];
                named_pattern += pattern_str[i + 1];
                i++;
            } else if (pattern_str[i] == '(') {  // Start of group
                // Check if it's a named group of the form (?<name>...)
                // or (?P<name>...)
                if (i + 1 < pattern_str.length() && pattern_str[i + 1] == '?') {
                    // Offset by 1 if it is of the form (?P<name>...)
                    int p_offset = (i + 2 < pattern_str.length() &&
                                    pattern_str[i + 2] == 'P')
                                       ? 1
                                       : 0;
                    if (i + 2 + p_offset < pattern_str.length() &&
                        pattern_str[i + 2 + p_offset] == '<') {
                        // Rename existing name to _groupN
                        size_t close = pattern_str.find('>', i + 3 + p_offset);
                        if (close != std::string::npos) {
                            named_pattern += "(?<_group" +
                                             std::to_string(num_groups++) + ">";
                            i = close;  // Skip to after the >
                        } else {
                            named_pattern += pattern_str[i];
                        }
                    } else {
                        // Non-capturing or other special group, keep as-is
                        named_pattern += pattern_str[i];
                    }
                } else {
                    // Unnamed group: convert to named group
                    named_pattern +=
                        "(?<_group" + std::to_string(num_groups++) + ">";
                }
            } else {
                named_pattern += pattern_str[i];
            }
        }

        return std::tuple(named_pattern, num_groups);
    }

    template <typename T>
    arrow::Datum do_arrow_compute_regexp_substr(T res, std::string pattern_str,
                                                std::string regex_params_str,
                                                int64_t group_to_extract) {
        bool extract_submatches =
            regex_params_str.find('e') != std::string::npos;

        // Ensure all groups are named so that we can pass to extract_regex.
        // Do this once per operator.
        if (!named_regexp) {
            named_regexp = std::make_shared<std::tuple<std::string, int>>(
                convert_to_named_regexp(pattern_str));
        }

        std::string named_pattern = std::get<0>(*named_regexp);
        int num_groups = std::get<1>(*named_regexp);

        if (!extract_submatches || num_groups == 0) {
            // Wrap the whole pattern in a group
            named_pattern = "(?<_whole>" + named_pattern + ")";
            extract_submatches = false;
        }

        arrow::compute::ExtractRegexOptions opts(named_pattern);
        auto extract_regex_result =
            do_arrow_compute_unary(res, "extract_regex", &opts);

        // Convert to Arrow array and extract field
        std::shared_ptr<arrow::Array> extract_array;
        if constexpr (std::is_same_v<T, arrow::Datum>) {
            extract_array = extract_regex_result.make_array();
        } else {
            extract_array = prepare_arrow_compute(extract_regex_result);
        }

        std::shared_ptr<arrow::StructArray> struct_result =
            std::static_pointer_cast<arrow::StructArray>(extract_array);

        // Extract the requested field
        std::shared_ptr<arrow::Array> captured_field;
        if (extract_submatches && group_to_extract >= num_groups) {
            // Return null array if requested group is greater than number
            // of groups in regex
            captured_field =
                arrow::MakeArrayOfNull(arrow::utf8(), struct_result->length())
                    .ValueOrDie();
        } else {
            std::string field_name;
            if (!extract_submatches) {
                // No group extraction requested, return whole match
                field_name = "_whole";
            } else {
                // extract_submatches && group_to_extract < num_groups:
                // valid group number requested.
                field_name = "_group" + std::to_string(group_to_extract);
            }

            // NOTE: StructArray.GetFieldByName() exists, but does not propagate
            // the validity bitmap to the child fields. We need to retain the
            // nulls that are emitted when there is no match for the regexp, so
            // we use GetFlattenedField() instead.
            auto struct_type = std::static_pointer_cast<arrow::StructType>(
                struct_result->type());
            int field_index = struct_type->GetFieldIndex(field_name);
            auto captured_field_res =
                struct_result->GetFlattenedField(field_index);
            if (!captured_field_res.ok()) {
                throw std::runtime_error(
                    "do_arrow_compute_regexp_substr: Error getting flattened "
                    "field from struct array: " +
                    captured_field_res.status().message());
            }
            captured_field = captured_field_res.ValueOrDie();
        }

        return arrow::Datum(captured_field);
    }

    template <typename T>
    arrow::Datum do_arrow_compute_regexp_instr(T res, std::string pattern_str,
                                               bool get_start_index,
                                               std::string regex_params_str,
                                               int64_t group_to_extract) {
        bool extract_submatches =
            regex_params_str.find('e') != std::string::npos;

        // Ensure all groups are named so that we can pass to
        // extract_regex_span. Do this once per operator.
        if (!named_regexp) {
            named_regexp = std::make_shared<std::tuple<std::string, int>>(
                convert_to_named_regexp(pattern_str));
        }

        std::string named_pattern = std::get<0>(*named_regexp);
        int num_groups = std::get<1>(*named_regexp);

        if (!extract_submatches || num_groups == 0) {
            // Wrap the whole pattern in a group
            named_pattern = "(?<_whole>" + named_pattern + ")";
            extract_submatches = false;
        }

        // Intermediate results must be Datums only because Bodo doesn't support
        // the fixed_size_list type.
        arrow::Datum res_datum =
            ConvertExprResultToDatum(res, "regexp_instr string");

        arrow::compute::ExtractRegexSpanOptions opts(named_pattern);
        auto extract_regex_span_result =
            do_arrow_compute_unary(res_datum, "extract_regex_span", &opts);

        // Convert to Arrow StructArray so we can extract the
        // field corresponding to the requested group
        std::shared_ptr<arrow::StructArray> struct_result =
            std::static_pointer_cast<arrow::StructArray>(
                extract_regex_span_result.make_array());

        // Extract the requested field
        std::shared_ptr<arrow::Array> captured_field_span = nullptr;
        if (extract_submatches && group_to_extract < num_groups) {
            // Valid group number requested
            captured_field_span = struct_result->GetFieldByName(
                "_group" + std::to_string(group_to_extract));
        } else if (!extract_submatches) {
            // No group extraction requested, return whole match
            captured_field_span = struct_result->GetFieldByName("_whole");
        }

        arrow::Datum captured_field_index_datum;
        if (!captured_field_span) {
            // Return array of -1 if requested group is greater than number
            // of groups in regex
            arrow::Int64Scalar negative_one_scalar(-1);
            captured_field_index_datum =
                arrow::Datum(arrow::MakeArrayFromScalar(negative_one_scalar,
                                                        struct_result->length())
                                 .ValueOrDie());
        } else {
            // The values of the each StructArray field are two-element
            // fixed_size_lists. The first element of the FixedSizeList is the
            // start index of the substring matched by the group, and the second
            // element is the length of that substring.

            // Get zero-based start indices
            auto first_element_index = std::make_shared<arrow::Int64Scalar>(0);
            arrow::Datum start_indices = do_arrow_compute_binary(
                arrow::Datum(captured_field_span),
                arrow::Datum(first_element_index), "list_element");

            if (!get_start_index) {
                // If get_start_index is False, we should return the index of
                // the first character after the substring matched by the group.
                // So we need to add the substring length to the start index.

                // Get lengths
                auto second_element_index =
                    std::make_shared<arrow::Int64Scalar>(1);
                arrow::Datum lengths = do_arrow_compute_binary(
                    arrow::Datum(captured_field_span),
                    arrow::Datum(second_element_index), "list_element");

                // Add lengths of to start indices
                captured_field_index_datum =
                    do_arrow_compute_binary(start_indices, lengths, "add");
            } else {
                captured_field_index_datum = start_indices;
            }

            // The validity bitmap of the parent StructArray is not propagated
            // automatically to the child fields (see
            // https://github.com/apache/arrow/issues/41833), so we have to
            // check it manually and return -1 if not valid.
            auto negative_one_scalar = std::make_shared<arrow::Int64Scalar>(-1);
            arrow::Datum valid_mask =
                do_arrow_compute_unary(extract_regex_span_result, "is_valid");
            auto nulled_index_res = arrow::compute::CallFunction(
                "if_else",
                {valid_mask, captured_field_index_datum, negative_one_scalar});
            if (!nulled_index_res.ok()) [[unlikely]] {
                throw std::runtime_error(
                    "do_arrow_compute_regexp_instr: Error in Arrow compute "
                    "(if_else): " +
                    nulled_index_res.status().message());
            }
            captured_field_index_datum = nulled_index_res.ValueOrDie();
        }

        return captured_field_index_datum;
    }

    template <typename T>
    arrow::Datum do_arrow_compute_dow_num(T res) {
        // We strip off the leading spaces and only look at the first two
        // characters of the input string in accordance with Snowflake (e.g.
        // NEXT_DAY/PREVIOUS_DAY)

        // Create Arrow array containing the days of the week. The order (index)
        // determines the result DoW number.
        arrow::StringBuilder builder;
        arrow::Status status =
            builder.AppendValues({"mo", "tu", "we", "th", "fr", "sa", "su"});
        if (!status.ok()) {
            throw std::runtime_error(
                "do_arrow_compute_dow_num: Failed to append values to "
                "StringBuilder");
        }
        std::shared_ptr<arrow::Array> dow_array = builder.Finish().ValueOrDie();

        arrow::Datum res_datum =
            ConvertExprResultToDatum(res, "day_of_week_num input");

        // Normalize string to two lowercase characters representing the day of
        // the week
        arrow::Datum trimmed_dow_string =
            do_arrow_compute_unary(res_datum, "utf8_ltrim_whitespace");
        arrow::compute::SliceOptions slice_opts(0, 2, 1);
        arrow::Datum sliced_dow_string = do_arrow_compute_unary(
            trimmed_dow_string, "utf8_slice_codeunits", &slice_opts);
        arrow::Datum lowered_dow_string =
            do_arrow_compute_unary(sliced_dow_string, "utf8_lower");

        // Get index of string into DoW array, which equals the DoW number
        arrow::compute::SetLookupOptions set_lookup_opts(dow_array);
        arrow::Datum dow_num = do_arrow_compute_unary(
            lowered_dow_string, "index_in", &set_lookup_opts);

        return dow_num;
    }

    template <typename T>
    arrow::Datum do_arrow_compute_random_int64(T res) {
        // Get dummy input as an Arrow array.
        // We need the result to match the length of this array.
        std::shared_ptr<arrow::Array> res_array =
            ConvertExprResultToDatum(res, "random_int64 dummy array")
                .make_array();

        // Only create PRNG once so we keep state across batches
        // and don't start from the same position for each batch.
        if (!gen) {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);

            assert_py_args_is_tuple(scalar_func_data.args,
                                    scalar_func_data.arrow_func_name.c_str());
            size_t num_args = PyTuple_Size(scalar_func_data.args);

            if (num_args == 1) {
                // Seed was explicitly provided, so use it
                auto [seed] = get_py_args_as_types(
                    scalar_func_data.args,
                    scalar_func_data.arrow_func_name.c_str(),
                    get_py_object_as_int64);

                if (rank == 0) {
                    // Create psuedo-random number generator.
                    // For rank 0 we use the input seed directly
                    gen = std::make_shared<std::mt19937_64>(
                        std::mt19937_64(seed));
                } else {
                    // seed_seq applies a transformation on the master seed and
                    // the rank number to scramble and break linear correlation
                    // between the seeds used by each rank.
                    // Because of this, the generated sequence will be different
                    // depending on the number of workers used, but it should
                    // still be deterministic.
                    // The Dynamic Creator algorithm might be theoretically
                    // better here, but seed_seq is probably sufficient in
                    // practice.

                    // Note that we have to split the 64-bit input seed because
                    // seed_seq only accepts 32-bit integers
                    std::seed_seq rank_seed{
                        static_cast<uint32_t>(seed >> 32),
                        static_cast<uint32_t>(seed & 0xFFFFFFFF),
                        static_cast<uint32_t>(rank)};

                    // Create PRNG with a (hopefully) independent seed
                    gen = std::make_shared<std::mt19937_64>(
                        std::mt19937_64(rank_seed));
                }
            } else if (num_args == 0) {
                // Generate a seed with 96-bits of entropy from system's
                // random_device and the current system time. time(NULL) is
                // mainly a backup in case std::random_device falls back
                // to a deterministic implementation.
                // We also integrate the rank number in the seed calculation
                // to ensure that two ranks do not get the same seed by chance.
                std::random_device rd;
                std::seed_seq seed{rd(), static_cast<uint32_t>(time(NULL)),
                                   rd(), static_cast<uint32_t>(rank)};
                // Create psuedo-random number generator
                gen = std::make_shared<std::mt19937_64>(std::mt19937_64(seed));
            } else {
                throw std::runtime_error(
                    "random_int64 only accepts 0 or 1 arguments");
            }
        }

        // Full-range uniform int64 distribution
        std::uniform_int_distribution<int64_t> int64_dist(0x8000000000000000,
                                                          0x7FFFFFFFFFFFFFFF);

        // Compute array of random integers based on the length of the dummy
        // input
        arrow::Int64Builder builder;
        for (int i = 0; i < res_array->length(); i++) {
            arrow::Status status = builder.Append(int64_dist(*gen));
            if (!status.ok()) {
                throw std::runtime_error(
                    "do_arrow_compute (random_int64): Failed to append "
                    "value to "
                    "Int64Builder");
            }
        }
        std::shared_ptr<arrow::Array> random_int64_array =
            builder.Finish().ValueOrDie();

        return arrow::Datum(random_int64_array);
    }

    template <typename T>
    compute_return_t<T> do_arrow_compute(T res) {
        compute_return_t<T> result;

        if (scalar_func_data.arrow_func_name == "if_else") {
            auto [then_datum, else_datum] = get_py_args_as_types(
                scalar_func_data.args, scalar_func_data.arrow_func_name.c_str(),
                get_scalar_py_object_as_datum, get_scalar_py_object_as_datum);
            arrow::Datum when_datum = ConvertExprResultToDatum(
                res, scalar_func_data.arrow_func_name + " when");

            arrow::Result<arrow::Datum> if_else_res =
                arrow::compute::CallFunction(
                    "if_else", {when_datum, then_datum, else_datum});
            if (!if_else_res.ok()) [[unlikely]] {
                throw std::runtime_error(
                    "do_arrow_compute if_else: Error in Arrow compute: " +
                    if_else_res.status().message());
            }

            if constexpr (std::is_same_v<T, arrow::Datum>) {
                result = if_else_res.ValueOrDie();
            } else {
                result = ConvertDatumToArrayInfo(if_else_res.ValueOrDie());
            }
        } else if (scalar_func_data.arrow_func_name == "rand") {
            // Get dummy input as an Arrow array.
            // We need the result to match the length of this array.
            std::shared_ptr<arrow::Array> res_array =
                ConvertExprResultToDatum(res, "rand input").make_array();

            // Retrieve the function object for random()
            static auto *registry = arrow::compute::GetFunctionRegistry();
            static std::shared_ptr<arrow::compute::Function> random_func;
            static std::once_flag once_flag_;
            std::call_once(once_flag_, [&]() {
                auto lookup_status =
                    registry->GetFunction("random").Value(&random_func);
                if (!lookup_status.ok()) {
                    throw std::runtime_error(
                        "Failed to get the 'random' function from the Arrow "
                        "registry.");
                }
            });

            // Use a system-generated seed
            arrow::compute::RandomOptions options =
                arrow::compute::RandomOptions::FromSystemRandom();

            // Since random() takes no arguments, we need to manually specify
            // the desired output length through ExecBatch. CallFunction() has
            // no overload for this.
            std::vector<arrow::Datum> empty_args;
            arrow::compute::ExecBatch batch(empty_args, res_array->length());

            // Compute the array of random values
            arrow::Result<arrow::Datum> random_result =
                random_func->Execute(batch, &options, nullptr);
            if (!random_result.ok()) {
                throw std::runtime_error("Error in Arrow compute (random): " +
                                         random_result.status().message());
            }

            if constexpr (std::is_same_v<T, arrow::Datum>) {
                result = random_result.ValueOrDie();
            } else {
                result = ConvertDatumToArrayInfo(random_result.ValueOrDie());
            }
        } else if (scalar_func_data.arrow_func_name == "random_int64") {
            // Not a real Arrow compute function; we use this to implement
            // Snowflake SQL RANDOM().
            arrow::Datum random_int64_datum =
                do_arrow_compute_random_int64(res);

            if constexpr (std::is_same_v<T, arrow::Datum>) {
                result = random_int64_datum;
            } else {
                result = ConvertDatumToArrayInfo(random_int64_datum);
            }
        } else if (scalar_func_data.arrow_func_name == "date") {
            // The Arrow compute equivalent of Series.dt.date() is
            // year_month_day, which returns a struct. To match the output dtype
            // of Pandas, we Cast to Date32 instead.
            result = do_arrow_compute_cast(res, arrow::date32());
        } else if (scalar_func_data.arrow_func_name == "day_of_week") {
            assert_py_args_is_tuple(scalar_func_data.args,
                                    scalar_func_data.arrow_func_name.c_str());
            size_t num_args = PyTuple_Size(scalar_func_data.args);
            if (num_args == 2) {
                auto [count_from_zero, week_start] = get_py_args_as_types(
                    scalar_func_data.args,
                    scalar_func_data.arrow_func_name.c_str(),
                    get_py_object_as_bool, get_py_object_as_int64);

                arrow::compute::DayOfWeekOptions opts(count_from_zero,
                                                      week_start);
                result = do_arrow_compute_unary(res, "day_of_week", &opts);
            } else {
                // Only support 0 or 2 optional parameters for now
                result = do_arrow_compute_unary(res, "day_of_week");
            }
        } else if (scalar_func_data.arrow_func_name == "day_of_week_num") {
            // day_of_week_num is a made up function representing the
            // number representing a day of week string.
            // Monday = 0, ..., Sunday = 6
            arrow::Datum dow_num_datum = do_arrow_compute_dow_num(res);

            // Convert DoW number and assign to result based on input type
            if constexpr (std::is_same_v<T, arrow::Datum>) {
                result = dow_num_datum;
            } else {
                result = ConvertDatumToArrayInfo(dow_num_datum);
            }
        } else if (scalar_func_data.arrow_func_name ==
                       "match_substring_regex" ||
                   scalar_func_data.arrow_func_name ==
                       "match_substring_regex_first" ||
                   scalar_func_data.arrow_func_name == "match_substring" ||
                   scalar_func_data.arrow_func_name == "starts_with" ||
                   scalar_func_data.arrow_func_name == "ends_with" ||
                   scalar_func_data.arrow_func_name == "find_substring") {
            assert_py_args_is_tuple(scalar_func_data.args,
                                    scalar_func_data.arrow_func_name.c_str());
            size_t num_args = PyTuple_Size(scalar_func_data.args);
            const char *c_str;
            bool ignore_case = false;
            if (num_args == 1) {
                // Only string was passed
                c_str = get_py_single_arg_as_cstr(
                    scalar_func_data.args,
                    scalar_func_data.arrow_func_name.c_str());
            } else {
                // string and ignore_case passed
                std::tie(c_str, ignore_case) = get_py_args_as_types(
                    scalar_func_data.args,
                    scalar_func_data.arrow_func_name.c_str(),
                    get_py_object_as_cstr, get_py_object_as_bool);
            }

            std::string func_name = scalar_func_data.arrow_func_name;
            std::string pattern(c_str);
            if (func_name == "match_substring_regex_first") {
                // match_substring_regex in Arrow matches anywhere in the string
                // but Series.str.match() matches from the start. Add ^ to the
                // pattern to match from the start same as Pandas:
                // https://github.com/pandas-dev/pandas/blob/366ccdfcd8ed1e5543bfb6d4ee0c9bc519898670/pandas/core/arrays/_arrow_string_mixins.py#L378
                func_name = "match_substring_regex";
                pattern = "^(" + pattern + ")";
            }

            arrow::compute::MatchSubstringOptions opts(pattern, ignore_case);
            result = do_arrow_compute_unary(res, func_name, &opts);
        } else if (scalar_func_data.arrow_func_name == "round") {
            auto [digits, round_mode] =
                get_py_round_args(scalar_func_data.args);

            arrow::compute::RoundOptions opts(digits, round_mode);
            result = do_arrow_compute_unary(
                res, scalar_func_data.arrow_func_name, &opts);
        } else if (scalar_func_data.arrow_func_name == "is_null") {
            // Set nan_is_null option to match Pandas isna behavior
            arrow::compute::NullOptions opts(true);
            result = do_arrow_compute_unary(
                res, scalar_func_data.arrow_func_name, &opts);
        } else if (scalar_func_data.arrow_func_name == "is_not_null") {
            // Set nan_is_null option to match Pandas isna behavior
            arrow::compute::NullOptions opts(true);
            result = do_arrow_compute_unary(
                res, scalar_func_data.arrow_func_name, &opts);
        } else if (scalar_func_data.arrow_func_name == "utf8_slice_codeunits") {
            arrow::Type::type res_type = GetArrowTypeOfRes(res);
            std::string func_name = (res_type == arrow::Type::BINARY ||
                                     res_type == arrow::Type::LARGE_BINARY)
                                        ? "binary_slice"
                                        : "utf8_slice_codeunits";

            auto [start, stop, step] = get_py_slice_args(scalar_func_data.args);

            arrow::compute::SliceOptions opts(start, stop, step);
            result = do_arrow_compute_unary(res, func_name, &opts);
        } else if (scalar_func_data.arrow_func_name == "regexp_substr") {
            auto [pattern, regex_params, group_to_extract] =
                get_py_args_as_types(scalar_func_data.args,
                                     scalar_func_data.arrow_func_name.c_str(),
                                     get_py_object_as_cstr,
                                     get_py_object_as_cstr,
                                     get_py_object_as_int64);
            std::string pattern_str(pattern);
            std::string regex_params_str(regex_params);

            arrow::Datum captured_field_datum = do_arrow_compute_regexp_substr(
                res, pattern_str, regex_params_str, group_to_extract);

            // Convert field and assign to result based on input type
            if constexpr (std::is_same_v<T, arrow::Datum>) {
                result = captured_field_datum;
            } else {
                result = ConvertDatumToArrayInfo(captured_field_datum);
            }
        } else if (scalar_func_data.arrow_func_name == "regexp_instr") {
            auto [pattern, get_start_index, regex_params, group_to_extract] =
                get_py_args_as_types(
                    scalar_func_data.args,
                    scalar_func_data.arrow_func_name.c_str(),
                    get_py_object_as_cstr, get_py_object_as_bool,
                    get_py_object_as_cstr, get_py_object_as_int64);
            std::string pattern_str(pattern);
            std::string regex_params_str(regex_params);

            arrow::Datum index_datum = do_arrow_compute_regexp_instr(
                res, pattern_str, get_start_index, regex_params_str,
                group_to_extract);

            // Assign to result based on input type
            if constexpr (std::is_same_v<T, arrow::Datum>) {
                result = index_datum;
            } else {
                result = ConvertDatumToArrayInfo(index_datum);
            }
        } else if (scalar_func_data.arrow_func_name == "utf8_trim" ||
                   scalar_func_data.arrow_func_name == "utf8_ltrim" ||
                   scalar_func_data.arrow_func_name == "utf8_rtrim" ||
                   scalar_func_data.arrow_func_name == "ascii_trim" ||
                   scalar_func_data.arrow_func_name == "ascii_ltrim" ||
                   scalar_func_data.arrow_func_name == "ascii_rtrim") {
            const char *c_str = get_py_single_arg_as_cstr(
                scalar_func_data.args,
                scalar_func_data.arrow_func_name.c_str());

            arrow::compute::TrimOptions opts(c_str);
            result = do_arrow_compute_unary(
                res, scalar_func_data.arrow_func_name, &opts);
        } else if (scalar_func_data.arrow_func_name == "utf8_lpad" ||
                   scalar_func_data.arrow_func_name == "utf8_rpad") {
            int64_t width;
            const char *padding;

            assert_py_args_is_tuple(scalar_func_data.args,
                                    scalar_func_data.arrow_func_name.c_str());
            if (PyTuple_Size(scalar_func_data.args) > 1) {
                std::tie(width, padding) = get_py_args_as_types(
                    scalar_func_data.args,
                    scalar_func_data.arrow_func_name.c_str(),
                    get_py_object_as_int64, get_py_object_as_cstr);
            } else {
                std::tie(width) = get_py_args_as_types(
                    scalar_func_data.args,
                    scalar_func_data.arrow_func_name.c_str(),
                    get_py_object_as_int64);
                padding = " ";
            }

            std::string padding_str(padding);

            arrow::compute::PadOptions opts(width, padding_str);
            result = do_arrow_compute_unary(
                res, scalar_func_data.arrow_func_name, &opts);
        } else if (scalar_func_data.arrow_func_name == "binary_repeat") {
            auto [num_repeats] = get_py_args_as_types(
                scalar_func_data.args, scalar_func_data.arrow_func_name.c_str(),
                get_py_object_as_int64);
            arrow::Datum num_repeats_datum(
                std::make_shared<arrow::Int64Scalar>(num_repeats));
            result = do_arrow_compute_binary(res, num_repeats_datum,
                                             "binary_repeat");
        } else if (scalar_func_data.arrow_func_name == "replace_substring" ||
                   scalar_func_data.arrow_func_name ==
                       "replace_substring_regex") {
            const char *pattern;
            const char *replacement;
            int64_t max_replacements = -1;

            assert_py_args_is_tuple(scalar_func_data.args,
                                    scalar_func_data.arrow_func_name.c_str());

            if (PyTuple_Size(scalar_func_data.args) > 2) {
                std::tie(pattern, replacement, max_replacements) =
                    get_py_args_as_types(
                        scalar_func_data.args,
                        scalar_func_data.arrow_func_name.c_str(),
                        get_py_object_as_cstr, get_py_object_as_cstr,
                        get_py_object_as_int64);
            } else {
                std::tie(pattern, replacement) = get_py_args_as_types(
                    scalar_func_data.args,
                    scalar_func_data.arrow_func_name.c_str(),
                    get_py_object_as_cstr, get_py_object_as_cstr);
            }

            std::string pattern_str(pattern);
            std::string replacement_str(replacement);

            arrow::compute::ReplaceSubstringOptions opts(pattern, replacement,
                                                         max_replacements);
            result = do_arrow_compute_unary(
                res, scalar_func_data.arrow_func_name, &opts);
        } else if (scalar_func_data.arrow_func_name == "utf8_replace_slice") {
            arrow::Type::type res_type = GetArrowTypeOfRes(res);
            std::string func_name = (res_type == arrow::Type::BINARY ||
                                     res_type == arrow::Type::LARGE_BINARY)
                                        ? "binary_replace_slice"
                                        : "utf8_replace_slice";

            auto [start, stop, replacement] = get_py_args_as_types(
                scalar_func_data.args, func_name.c_str(),
                get_py_object_as_int64, get_py_object_as_int64,
                get_py_object_as_cstr);

            std::string replacement_str(replacement);

            arrow::compute::ReplaceSliceOptions opts(start, stop,
                                                     replacement_str);
            result = do_arrow_compute_unary(res, func_name, &opts);
        } else if (scalar_func_data.arrow_func_name == "is_in") {
            std::shared_ptr<arrow::Array> values_array =
                get_py_isin_arg_as_arrow_array(scalar_func_data.args);
            arrow::compute::SetLookupOptions opts(values_array);
            result = do_arrow_compute_unary(res, "is_in", &opts);
        } else if (scalar_func_data.arrow_func_name == "assume_timezone") {
            const char *c_str = get_py_single_arg_as_cstr(
                scalar_func_data.args,
                scalar_func_data.arrow_func_name.c_str());
            arrow::compute::AssumeTimezoneOptions opts(c_str);
            result = do_arrow_compute_unary(res, "assume_timezone", &opts);
        } else if (scalar_func_data.arrow_func_name == "strftime") {
            const char *fmt_str = get_py_single_arg_as_cstr(
                scalar_func_data.args,
                scalar_func_data.arrow_func_name.c_str());
            arrow::compute::StrftimeOptions opts(fmt_str);
            result = do_arrow_compute_unary(res, "strftime", &opts);
        } else if (scalar_func_data.arrow_func_name == "strptime") {
            auto [fmt_str, unit_cstr] = get_py_args_as_types(
                scalar_func_data.args, scalar_func_data.arrow_func_name.c_str(),
                get_py_object_as_cstr, get_py_object_as_cstr);
            std::string unit_str(unit_cstr);
            arrow::TimeUnit::type time_unit;
            if (unit_str == "s") {
                time_unit = arrow::TimeUnit::SECOND;
            } else if (unit_str == "ms") {
                time_unit = arrow::TimeUnit::MILLI;
            } else if (unit_str == "us") {
                time_unit = arrow::TimeUnit::MICRO;
            } else if (unit_str == "ns") {
                time_unit = arrow::TimeUnit::NANO;
            } else {
                throw std::invalid_argument(
                    "strptime: Invalid time unit string: " + unit_str);
            }
            arrow::compute::StrptimeOptions opts(fmt_str, time_unit, false);
            result = do_arrow_compute_unary(res, "strptime", &opts);
        } else if (scalar_func_data.arrow_func_name == "floor_temporal" ||
                   scalar_func_data.arrow_func_name == "ceil_temporal" ||
                   scalar_func_data.arrow_func_name == "round_temporal") {
            // Args: (multiple, unit_string).
            // multiple is the first PyLong (e.g. 1),
            // unit_string is the second PyUnicode (e.g. "day").
            PyObject *args = scalar_func_data.args;
            int64_t multiple = PyLong_AsLongLong(PyTuple_GetItem(args, 0));
            PyObject *unit_py = PyTuple_GetItem(args, 1);
            const char *unit_cstr = PyUnicode_AsUTF8(unit_py);
            arrow::compute::CalendarUnit unit = getArrowCalendarUnit(unit_cstr);
            arrow::compute::RoundTemporalOptions opts(multiple, unit);
            result = do_arrow_compute_unary(
                res, scalar_func_data.arrow_func_name, &opts);
        } else {
            std::string func_name = scalar_func_data.arrow_func_name;
            arrow::Type::type res_type = GetArrowTypeOfRes(res);
            if (res_type == arrow::Type::BINARY ||
                res_type == arrow::Type::LARGE_BINARY) {
                if (func_name == "utf8_length") {
                    func_name = "binary_length";
                } else if (func_name == "utf8_reverse") {
                    func_name = "binary_reverse";
                }
            }

            result = do_arrow_compute_unary(res, func_name);
        }
        return result;
    }
};
