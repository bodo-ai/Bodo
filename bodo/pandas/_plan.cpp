#include "_plan.h"
#include <arrow/python/pyarrow.h>
#include <fmt/format.h>
#include <cstddef>
#include <utility>

#include <arrow/filesystem/filesystem.h>
#include <arrow/python/api.h>
#include "../io/arrow_compat.h"
#include "_bodo_scan_function.h"
#include "_bodo_write_function.h"
#include "_executor.h"
#include "duckdb/catalog/catalog_entry/scalar_function_catalog_entry.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/types/timestamp.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_copy_to_file.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_limit.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_sample.hpp"
#include "duckdb/transaction/duck_transaction_manager.hpp"
#include "physical/project.h"

// if status of arrow::Result is not ok, form an err msg and raise a
// runtime_error with it
#undef CHECK_ARROW
#define CHECK_ARROW(expr, msg)                                          \
    if (!(expr.ok())) {                                                 \
        std::string err_msg = std::string(msg) + " " + expr.ToString(); \
        throw std::runtime_error(err_msg);                              \
    }

// if status of arrow::Result is not ok, form an err msg and raise a
// runtime_error with it. If it is ok, get value using ValueOrDie
// and assign it to lhs using std::move
#undef CHECK_ARROW_AND_ASSIGN
#define CHECK_ARROW_AND_ASSIGN(res, msg, lhs) \
    CHECK_ARROW(res.status(), msg)            \
    lhs = std::move(res).ValueOrDie();

/**
 * @brief Convert a std::unique_ptr to the duckdb equivalent.
 *
 */
template <class T>
duckdb::unique_ptr<T> to_duckdb(std::unique_ptr<T> &val) {
    return duckdb::unique_ptr<T>(val.release());
}

duckdb::unique_ptr<duckdb::LogicalOperator> optimize_plan(
    std::unique_ptr<duckdb::LogicalOperator> plan) {
    duckdb::shared_ptr<duckdb::Optimizer> optimizer = get_duckdb_optimizer();

    // Convert std::unique_ptr to duckdb::unique_ptr
    // Input is using std since Cython supports it
    auto in_plan = to_duckdb(plan);

    duckdb::unique_ptr<duckdb::LogicalOperator> out_plan =
        optimizer->Optimize(std::move(in_plan));
    return out_plan;
}

duckdb::unique_ptr<duckdb::Expression> make_const_null(PyObject *out_schema_py,
                                                       int64_t field_idx) {
    std::shared_ptr<arrow::Schema> arrow_schema = unwrap_schema(out_schema_py);
    const std::shared_ptr<arrow::Field> &field = arrow_schema->field(field_idx);
    auto [_, out_type] = arrow_field_to_duckdb(field);
    // This is how duckdb makes a NULL value of a specific type.
    // You just pass the duckdb type to the Value constructor.
    return duckdb::make_uniq<duckdb::BoundConstantExpression>(
        duckdb::Value(out_type));
}

duckdb::unique_ptr<duckdb::Expression> make_const_int_expr(int64_t val) {
    return duckdb::make_uniq<duckdb::BoundConstantExpression>(
        duckdb::Value(val));
}

duckdb::unique_ptr<duckdb::Expression> make_const_double_expr(double val) {
    return duckdb::make_uniq<duckdb::BoundConstantExpression>(
        duckdb::Value(val));
}

duckdb::unique_ptr<duckdb::Expression> make_const_bool_expr(bool val) {
    return duckdb::make_uniq<duckdb::BoundConstantExpression>(
        duckdb::Value(val));
}

duckdb::unique_ptr<duckdb::Expression> make_const_timestamp_ns_expr(
    int64_t val) {
    return duckdb::make_uniq<duckdb::BoundConstantExpression>(
        duckdb::Value::TIMESTAMPNS(duckdb::timestamp_ns_t(val)));
}

duckdb::unique_ptr<duckdb::Expression> make_const_string_expr(
    const std::string &val) {
    return duckdb::make_uniq<duckdb::BoundConstantExpression>(
        duckdb::Value(val));
}

duckdb::unique_ptr<duckdb::Expression> make_col_ref_expr_internal(
    std::unique_ptr<duckdb::LogicalOperator> &source,
    std::shared_ptr<arrow::Field> field, int col_idx) {
    auto [_, ctype] = arrow_field_to_duckdb(field);

    std::vector<duckdb::ColumnBinding> source_cols =
        source->GetColumnBindings();
    assert((size_t)col_idx < source_cols.size());

    return duckdb::make_uniq<duckdb::BoundColumnRefExpression>(
        ctype, source_cols[col_idx]);
}

duckdb::unique_ptr<duckdb::Expression> make_col_ref_expr(
    std::unique_ptr<duckdb::LogicalOperator> &source, PyObject *field_py,
    int col_idx) {
    auto field_res = arrow::py::unwrap_field(field_py);
    std::shared_ptr<arrow::Field> field;
    CHECK_ARROW_AND_ASSIGN(field_res,
                           "make_col_ref_expr: unable to unwrap field", field);
    return make_col_ref_expr_internal(source, field, col_idx);
}

duckdb::unique_ptr<duckdb::Expression> make_agg_expr(
    std::unique_ptr<duckdb::LogicalOperator> &source, PyObject *out_schema_py,
    std::string function_name, std::vector<int> input_column_indices,
    bool dropna) {
    auto out_schema_res = arrow::py::unwrap_schema(out_schema_py);
    std::shared_ptr<arrow::Schema> out_schema;
    CHECK_ARROW_AND_ASSIGN(
        out_schema_res, "make_agg_expr: unable to unwrap schema", out_schema);

    // Get DuckDB output type
    auto field = out_schema->field(0);
    auto [_, out_type] = arrow_field_to_duckdb(field);

    // Get arguments and their types for the aggregate function.
    source->ResolveOperatorTypes();
    std::vector<duckdb::ColumnBinding> source_cols =
        source->GetColumnBindings();
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> children;
    duckdb::vector<duckdb::LogicalType> arg_types;
    for (int col_idx : input_column_indices) {
        if (col_idx < 0 || static_cast<size_t>(col_idx) >= source_cols.size()) {
            throw std::runtime_error(
                fmt::format("make_agg_expr: Column index {} out of bounds for "
                            "source columns",
                            col_idx));
        }
        duckdb::LogicalType col_type = source->types[col_idx];
        children.push_back(
            std::move(duckdb::make_uniq<duckdb::BoundColumnRefExpression>(
                col_type, source_cols[col_idx])));
        arg_types.push_back(col_type);
    }

    duckdb::AggregateFunction function(
        function_name, arg_types, out_type, nullptr, nullptr, nullptr, nullptr,
        nullptr, duckdb::FunctionNullHandling::DEFAULT_NULL_HANDLING);

    // The name parameter in AggregateFunction is ignored when determining
    // whether two AggregateFunctions are equal, adding function_name to
    // BodoAggFunctionData ensures two different functions applied to the same
    // column are not optimized out.
    auto bind_info = duckdb::make_uniq<BodoAggFunctionData>(
        dropna, function_name, out_schema);

    return duckdb::make_uniq<duckdb::BoundAggregateExpression>(
        function, std::move(children), nullptr, std::move(bind_info),
        duckdb::AggregateType::NON_DISTINCT);
}

/**
 * @brief Change the type of a constant to match the type of the other side
 *        of a binary op expr.
 *
 * params series - the non-constant part of the binary op expr
 * params constant - the constant part of the binary op expr
 * returns a BinaryConstantExpression where the internal value has been changed
 *         to match the type of the other side of the binary op
 */
duckdb::unique_ptr<duckdb::Expression> matchType(
    duckdb::unique_ptr<duckdb::Expression> &series,
    duckdb::unique_ptr<duckdb::Expression> &constant) {
    // Cast to constant to BoundConstantExpression.
    duckdb::unique_ptr<duckdb::BoundConstantExpression> bce_constant =
        dynamic_cast_unique_ptr<duckdb::BoundConstantExpression>(
            std::move(constant));

    // Get the type to convert the constant to.
    duckdb::LogicalType series_type = series->return_type;
    // Change the value to the given type.
    // Will throw an exception if such a conversion is not possible.
    bce_constant->value = bce_constant->value.DefaultCastAs(series_type);
    // Change the expression's return type to match.
    bce_constant->return_type = series_type;
    return bce_constant;
}

std::unique_ptr<duckdb::Expression> make_comparison_expr(
    std::unique_ptr<duckdb::Expression> &lhs,
    std::unique_ptr<duckdb::Expression> &rhs, duckdb::ExpressionType etype) {
    // Convert std::unique_ptr to duckdb::unique_ptr.
    auto lhs_duck = to_duckdb(lhs);
    auto rhs_duck = to_duckdb(rhs);

    // If the left and right side of a binary op expression don't have
    // matching types then filter pushdown will be skipped.  Here we force
    // the constant to have the type as the other side of the binary op.
    if (lhs_duck->GetExpressionClass() ==
        duckdb::ExpressionClass::BOUND_CONSTANT) {
        lhs_duck = matchType(rhs_duck, lhs_duck);
    } else if (rhs_duck->GetExpressionClass() ==
               duckdb::ExpressionClass::BOUND_CONSTANT) {
        rhs_duck = matchType(lhs_duck, rhs_duck);
    }
    return duckdb::make_uniq<duckdb::BoundComparisonExpression>(
        etype, std::move(lhs_duck), std::move(rhs_duck));
}

std::unique_ptr<duckdb::Expression> make_arithop_expr(
    std::unique_ptr<duckdb::Expression> &lhs,
    std::unique_ptr<duckdb::Expression> &rhs, std::string opstr,
    PyObject *out_schema_py) {
    // Convert std::unique_ptr to duckdb::unique_ptr.
    auto lhs_duck = to_duckdb(lhs);
    auto rhs_duck = to_duckdb(rhs);
    std::shared_ptr<arrow::Schema> out_schema = unwrap_schema(out_schema_py);

    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> children;
    children.emplace_back(std::move(lhs_duck));
    children.emplace_back(std::move(rhs_duck));

    duckdb::ErrorData error;
    duckdb::QueryErrorContext error_context;

    duckdb::shared_ptr<duckdb::ClientContext> client_context =
        get_duckdb_context();
    client_context->transaction.BeginTransaction();
    duckdb::EntryLookupInfo function_lookup(
        duckdb::CatalogType::SCALAR_FUNCTION_ENTRY, opstr, error_context);
    duckdb::shared_ptr<duckdb::Binder> binder = get_duckdb_binder();
    duckdb::optional_ptr<duckdb::CatalogEntry> entry = binder->GetCatalogEntry(
        "system", "", function_lookup, duckdb::OnEntryNotFound::RETURN_NULL);
    if (!entry) {
        throw std::runtime_error("make_arithop_expr GetCatalogEntry failed");
    }
    duckdb::ScalarFunctionCatalogEntry &func =
        entry->Cast<duckdb::ScalarFunctionCatalogEntry>();

    duckdb::FunctionBinder function_binder(*binder);
    duckdb::unique_ptr<duckdb::Expression> result =
        function_binder.BindScalarFunction(
            func, std::move(children), error,
            true,  // function is an operator
            duckdb::optional_ptr<duckdb::Binder>(*binder));
    if (!result) {
        throw std::runtime_error("make_arithop_expr BindScalarFunction failed");
    }
    if (result->GetExpressionType() != duckdb::ExpressionType::BOUND_FUNCTION) {
        throw std::runtime_error(
            "make_arithop_expr BindScalarFunction did not return a "
            "BOUND_FUNCTION");
    }

    auto &bound_func_expr = result->Cast<duckdb::BoundFunctionExpression>();
    bound_func_expr.bind_info =
        duckdb::make_uniq<BodoPythonScalarFunctionData>(out_schema);

    client_context->transaction.ClearTransaction();
    return result;
}

std::unique_ptr<duckdb::Expression> make_unaryop_expr(
    std::unique_ptr<duckdb::Expression> &source, std::string opstr) {
    // Convert std::unique_ptr to duckdb::unique_ptr.
    auto lhs_duck = to_duckdb(source);
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> children;
    children.emplace_back(std::move(lhs_duck));

    duckdb::ErrorData error;
    duckdb::QueryErrorContext error_context;

    duckdb::shared_ptr<duckdb::ClientContext> client_context =
        get_duckdb_context();
    client_context->transaction.BeginTransaction();
    duckdb::EntryLookupInfo function_lookup(
        duckdb::CatalogType::SCALAR_FUNCTION_ENTRY, opstr, error_context);
    duckdb::shared_ptr<duckdb::Binder> binder = get_duckdb_binder();
    duckdb::optional_ptr<duckdb::CatalogEntry> entry =
        binder->GetCatalogEntry("system", "main", function_lookup,
                                duckdb::OnEntryNotFound::RETURN_NULL);
    if (!entry) {
        throw std::runtime_error("make_unaryop_expr GetCatalogEntry failed");
    }
    duckdb::ScalarFunctionCatalogEntry &func =
        entry->Cast<duckdb::ScalarFunctionCatalogEntry>();

    duckdb::FunctionBinder function_binder(*binder);
    duckdb::unique_ptr<duckdb::Expression> result =
        function_binder.BindScalarFunction(
            func, std::move(children), error,
            false,  // function is an operator
            duckdb::optional_ptr<duckdb::Binder>(*binder));
    if (!result) {
        throw std::runtime_error("make_unaryop_expr BindScalarFunction failed");
    }
    if (result->GetExpressionType() != duckdb::ExpressionType::BOUND_FUNCTION) {
        throw std::runtime_error(
            "make_unaryop_expr BindScalarFunction did not return a "
            "BOUND_FUNCTION");
    }
    client_context->transaction.ClearTransaction();
    return result;
}

duckdb::unique_ptr<duckdb::Expression> make_conjunction_expr(
    std::unique_ptr<duckdb::Expression> &lhs,
    std::unique_ptr<duckdb::Expression> &rhs, duckdb::ExpressionType etype) {
    // Convert std::unique_ptr to duckdb::unique_ptr.
    auto lhs_duck = to_duckdb(lhs);
    auto rhs_duck = to_duckdb(rhs);

    return duckdb::make_uniq<duckdb::BoundConjunctionExpression>(
        etype, std::move(lhs_duck), std::move(rhs_duck));
}

duckdb::unique_ptr<duckdb::Expression> make_unary_expr(
    std::unique_ptr<duckdb::Expression> &lhs, duckdb::ExpressionType etype) {
    // Convert std::unique_ptr to duckdb::unique_ptr.
    auto lhs_duck = to_duckdb(lhs);

    switch (etype) {
        case duckdb::ExpressionType::OPERATOR_NOT: {
            auto ret = duckdb::make_uniq<duckdb::BoundOperatorExpression>(
                etype, duckdb::LogicalType(duckdb::LogicalTypeId::BOOLEAN));
            ret->children.push_back(std::move(lhs_duck));
            return ret;
        } break;
        default:
            throw std::runtime_error("make_unary_expr unsupported etype " +
                                     std::to_string(static_cast<int>(etype)));
    }
}

duckdb::unique_ptr<duckdb::LogicalFilter> make_filter(
    std::unique_ptr<duckdb::LogicalOperator> &source,
    std::unique_ptr<duckdb::Expression> &filter_expr) {
    // Convert std::unique_ptr to duckdb::unique_ptr.
    auto source_duck = to_duckdb(source);
    auto filter_expr_duck = to_duckdb(filter_expr);
    auto logical_filter =
        duckdb::make_uniq<duckdb::LogicalFilter>(std::move(filter_expr_duck));

    logical_filter->children.push_back(std::move(source_duck));
    return logical_filter;
}

duckdb::unique_ptr<duckdb::LogicalSample> make_sample(
    std::unique_ptr<duckdb::LogicalOperator> &source, int n) {
    // Convert std::unique_ptr to duckdb::unique_ptr.
    auto source_duck = to_duckdb(source);
    duckdb::unique_ptr<duckdb::SampleOptions> sampleOptions =
        duckdb::make_uniq<duckdb::SampleOptions>();
    sampleOptions->sample_size = duckdb::Value(n);
    sampleOptions->is_percentage = false;
    sampleOptions->method = duckdb::SampleMethod::SYSTEM_SAMPLE;
    sampleOptions->repeatable = true;  // Not sure if this is correct.
    auto logical_sample = duckdb::make_uniq<duckdb::LogicalSample>(
        std::move(sampleOptions), std::move(source_duck));

    return logical_sample;
}

duckdb::unique_ptr<duckdb::LogicalLimit> make_limit(
    std::unique_ptr<duckdb::LogicalOperator> &source, int n) {
    // Convert std::unique_ptr to duckdb::unique_ptr.
    auto source_duck = to_duckdb(source);
    auto logical_limit = duckdb::make_uniq<duckdb::LogicalLimit>(
        duckdb::BoundLimitNode::ConstantValue(n),
        duckdb::BoundLimitNode::ConstantValue(0));

    logical_limit->children.push_back(std::move(source_duck));
    return logical_limit;
}

duckdb::unique_ptr<duckdb::LogicalProjection> make_projection(
    std::unique_ptr<duckdb::LogicalOperator> &source,
    std::vector<std::unique_ptr<duckdb::Expression>> &expr_vec,
    PyObject *out_schema_py) {
    // Convert std::unique_ptr to duckdb::unique_ptr.
    auto source_duck = to_duckdb(source);

    auto binder = get_duckdb_binder();
    auto table_idx = binder.get()->GenerateTableIndex();

    std::vector<duckdb::unique_ptr<duckdb::Expression>> projection_expressions;
    for (auto &expr : expr_vec) {
        // Convert std::unique_ptr to duckdb::unique_ptr.
        auto expr_duck = to_duckdb(expr);
        projection_expressions.push_back(std::move(expr_duck));
    }

    // Create projection node.
    duckdb::unique_ptr<duckdb::LogicalProjection> proj =
        duckdb::make_uniq<duckdb::LogicalProjection>(
            table_idx, std::move(projection_expressions));

    // Add the source of the projection.
    proj->children.push_back(std::move(source_duck));

    return proj;
}

duckdb::unique_ptr<duckdb::LogicalOrder> make_order(
    std::unique_ptr<duckdb::LogicalOperator> &source, std::vector<bool> &asc,
    std::vector<bool> &na_position, std::vector<int> &cols,
    PyObject *schema_py) {
    auto schema_res = arrow::py::unwrap_schema(schema_py);
    std::shared_ptr<arrow::Schema> schema;
    CHECK_ARROW_AND_ASSIGN(schema_res, "make_order: unable to unwrap schema",
                           schema);

    // Convert std::unique_ptr to duckdb::unique_ptr.
    auto source_duck = to_duckdb(source);
    duckdb::vector<duckdb::BoundOrderByNode> col_orders;
    for (size_t i = 0; i < asc.size(); ++i) {
        col_orders.emplace_back(duckdb::BoundOrderByNode(
            asc[i] ? duckdb::OrderType::ASCENDING
                   : duckdb::OrderType::DESCENDING,
            na_position[i] ? duckdb::OrderByNullType::NULLS_FIRST
                           : duckdb::OrderByNullType::NULLS_LAST,
            make_col_ref_expr_internal(source_duck, schema->field(i),
                                       cols[i])));
    }

    // Create projection node.
    duckdb::unique_ptr<duckdb::LogicalOrder> order =
        duckdb::make_uniq<duckdb::LogicalOrder>(std::move(col_orders));

    // Add the source of the order.
    order->children.push_back(std::move(source_duck));

    return order;
}

duckdb::unique_ptr<duckdb::LogicalAggregate> make_aggregate(
    std::unique_ptr<duckdb::LogicalOperator> &source,
    std::vector<int> &key_indices,
    std::vector<std::unique_ptr<duckdb::Expression>> &expr_vec,
    PyObject *out_schema_py) {
    // Convert std::unique_ptr to duckdb::unique_ptr.
    auto source_duck = to_duckdb(source);
    std::vector<duckdb::ColumnBinding> source_cols =
        source_duck->GetColumnBindings();

    source_duck->ResolveOperatorTypes();

    std::vector<duckdb::unique_ptr<duckdb::Expression>> aggregate_expressions;
    for (auto &expr : expr_vec) {
        // Convert std::unique_ptr to duckdb::unique_ptr.
        auto expr_duck = to_duckdb(expr);
        aggregate_expressions.push_back(std::move(expr_duck));
    }

    duckdb::shared_ptr<duckdb::Binder> binder = get_duckdb_binder();

    // Create aggregate node.
    duckdb::unique_ptr<duckdb::LogicalAggregate> aggr =
        duckdb::make_uniq<duckdb::LogicalAggregate>(
            binder->GenerateTableIndex(), binder->GenerateTableIndex(),
            std::move(aggregate_expressions));

    std::vector<duckdb::unique_ptr<duckdb::Expression>> group_exprs;
    for (int key_idx : key_indices) {
        if (key_idx < 0 || static_cast<size_t>(key_idx) >= source_cols.size()) {
            throw std::runtime_error(
                fmt::format("make_aggregate: Key index {} out of bounds for "
                            "source columns",
                            key_idx));
        }
        duckdb::LogicalType col_type = source_duck->types[key_idx];
        group_exprs.push_back(
            duckdb::make_uniq<duckdb::BoundColumnRefExpression>(
                col_type, source_cols[key_idx]));
    }

    aggr->groups = std::move(group_exprs);

    // Add the source to be aggregated on.
    aggr->children.push_back(std::move(source_duck));

    return aggr;
}

std::vector<int> get_projection_pushed_down_columns(
    std::unique_ptr<duckdb::LogicalOperator> &proj) {
    if (proj->children.size() != 1) {
        throw std::runtime_error(
            "Only one child operator expected in LogicalProjection");
    }
    duckdb::LogicalGet &get_plan =
        proj->children[0]->Cast<duckdb::LogicalGet>();

    std::vector<int> selected_columns;
    for (auto &ci : get_plan.GetColumnIds()) {
        selected_columns.push_back(ci.GetPrimaryIndex());
    }
    return selected_columns;
}

/**
 * @brief Dummy function to pass to DuckDB for UDFs. DuckDB runs some functions
 * during optimization for constant folding, but we avoid it by throwing an
 * exception.
 *
 */
static void RunFunction(duckdb::DataChunk &args, duckdb::ExpressionState &state,
                        duckdb::Vector &result) {
    throw std::runtime_error("Cannot run Bodo UDFs during optimization.");
}

duckdb::unique_ptr<duckdb::Expression> make_python_scalar_func_expr(
    std::unique_ptr<duckdb::LogicalOperator> &source, PyObject *out_schema_py,
    PyObject *args, const std::vector<int> &selected_columns, bool is_cfunc) {
    // Get output data type (UDF output is a single column)
    std::shared_ptr<arrow::Schema> out_schema = unwrap_schema(out_schema_py);
    auto [_, out_types] = arrow_schema_to_duckdb(out_schema);
    // Maybe not be exactly 1 due to index column.
    assert(out_types.size() > 0);
    duckdb::LogicalType out_type = out_types[0];

    // Necessary before accessing source->types attribute
    source->ResolveOperatorTypes();

    // Create ScalarFunction for UDF
    duckdb::ScalarFunction scalar_function = duckdb::ScalarFunction(
        "bodo_udf", source->types, out_type, RunFunction);
    duckdb::unique_ptr<duckdb::FunctionData> bind_data1 =
        duckdb::make_uniq<BodoPythonScalarFunctionData>(args, out_schema,
                                                        is_cfunc);

    std::vector<duckdb::ColumnBinding> source_cols =
        source->GetColumnBindings();

    // Add UDF input expressions for selected columns
    std::vector<duckdb::unique_ptr<duckdb::Expression>> udf_in_exprs;
    for (int col_idx : selected_columns) {
        auto expr = duckdb::make_uniq<duckdb::BoundColumnRefExpression>(
            source->types[col_idx], source_cols[col_idx]);
        udf_in_exprs.emplace_back(std::move(expr));
    }

    // Create UDF expression
    duckdb::unique_ptr<duckdb::BoundFunctionExpression> scalar_expr =
        make_uniq<duckdb::BoundFunctionExpression>(out_type, scalar_function,
                                                   std::move(udf_in_exprs),
                                                   std::move(bind_data1));

    return scalar_expr;
}

duckdb::unique_ptr<duckdb::LogicalComparisonJoin> make_comparison_join(
    std::unique_ptr<duckdb::LogicalOperator> &lhs,
    std::unique_ptr<duckdb::LogicalOperator> &rhs, duckdb::JoinType join_type,
    std::vector<std::pair<int, int>> &cond_vec) {
    // Convert std::unique_ptr to duckdb::unique_ptr.
    auto lhs_duck = to_duckdb(lhs);
    auto rhs_duck = to_duckdb(rhs);
    // Create join node.
    auto comp_join =
        duckdb::make_uniq<duckdb::LogicalComparisonJoin>(join_type);
    // Create join condition.
    duckdb::LogicalType cbtype(duckdb::LogicalTypeId::INTEGER);
    for (std::pair<int, int> cond_pair : cond_vec) {
        duckdb::JoinCondition cond;
        cond.comparison = duckdb::ExpressionType::COMPARE_EQUAL;
        cond.left = duckdb::make_uniq<duckdb::BoundColumnRefExpression>(
            cbtype, lhs_duck->GetColumnBindings()[cond_pair.first]);
        cond.right = duckdb::make_uniq<duckdb::BoundColumnRefExpression>(
            cbtype, rhs_duck->GetColumnBindings()[cond_pair.second]);
        // Add the join condition to the join node.
        comp_join->conditions.push_back(std::move(cond));
    }
    // Add the sources to be joined.
    comp_join->children.push_back(std::move(lhs_duck));
    comp_join->children.push_back(std::move(rhs_duck));

    return comp_join;
}

duckdb::unique_ptr<duckdb::LogicalSetOperation> make_set_operation(
    std::unique_ptr<duckdb::LogicalOperator> &lhs,
    std::unique_ptr<duckdb::LogicalOperator> &rhs, const std::string &setop,
    int64_t num_cols) {
    // Convert std::unique_ptr to duckdb::unique_ptr.
    auto lhs_duck = to_duckdb(lhs);
    auto rhs_duck = to_duckdb(rhs);
    auto binder = get_duckdb_binder();
    auto table_idx = binder.get()->GenerateTableIndex();
    bool setop_all = false;

    duckdb::LogicalOperatorType optype;
    if (setop == "union" || setop == "union all") {
        optype = duckdb::LogicalOperatorType::LOGICAL_UNION;
        setop_all = (setop == "union all");
    } else {
        throw std::runtime_error("make_set_operation unsupported type " +
                                 setop);
    }
    auto set_operation = duckdb::make_uniq<duckdb::LogicalSetOperation>(
        table_idx, num_cols, std::move(lhs_duck), std::move(rhs_duck), optype,
        setop_all);
    return set_operation;
}

std::pair<int64_t, PyObject *> execute_plan(
    std::unique_ptr<duckdb::LogicalOperator> plan, PyObject *out_schema_py) {
    std::shared_ptr<arrow::Schema> out_schema = unwrap_schema(out_schema_py);
    Executor executor(std::move(plan), out_schema);
    std::variant<std::shared_ptr<table_info>, PyObject *> output =
        executor.ExecutePipelines();

    // Iceberg write returns a PyObject* with file information
    if (std::holds_alternative<PyObject *>(output)) {
        PyObject *file_infos = std::get<PyObject *>(output);
        return {0, file_infos};
    }

    std::shared_ptr<table_info> output_table = std::get<0>(output);

    // Parquet write doesn't return data
    if (output_table == nullptr) {
        return {0, nullptr};
    }

    PyObject *pyarrow_schema =
        arrow::py::wrap_schema(output_table->schema()->ToArrowSchema());

    return {reinterpret_cast<int64_t>(new table_info(*output_table)),
            pyarrow_schema};
}

duckdb::unique_ptr<duckdb::LogicalGet> make_parquet_get_node(
    PyObject *parquet_path, PyObject *pyarrow_schema, PyObject *storage_options,
    int64_t num_rows) {
    duckdb::shared_ptr<duckdb::Binder> binder = get_duckdb_binder();
    std::shared_ptr<arrow::Schema> arrow_schema = unwrap_schema(pyarrow_schema);

    BodoParquetScanFunction table_function =
        BodoParquetScanFunction(arrow_schema);
    duckdb::unique_ptr<duckdb::FunctionData> bind_data1 =
        duckdb::make_uniq<BodoParquetScanFunctionData>(
            parquet_path, pyarrow_schema, storage_options);

    // Convert Arrow schema to DuckDB
    auto [return_names, return_types] = arrow_schema_to_duckdb(arrow_schema);

    duckdb::virtual_column_map_t virtual_columns;

    duckdb::unique_ptr<duckdb::LogicalGet> out_get =
        duckdb::make_uniq<duckdb::LogicalGet>(
            binder->GenerateTableIndex(), table_function, std::move(bind_data1),
            return_types, return_names, virtual_columns);
    out_get->SetEstimatedCardinality(num_rows);

    // Column ids need to be added separately.
    // DuckDB column id initialization example:
    // https://github.com/duckdb/duckdb/blob/d29a92f371179170688b4df394478f389bf7d1a6/src/catalog/catalog_entry/table_catalog_entry.cpp#L252
    for (size_t i = 0; i < return_names.size(); i++) {
        out_get->AddColumnId(i);
    }

    return out_get;
}

duckdb::unique_ptr<duckdb::LogicalCopyToFile> make_parquet_write_node(
    std::unique_ptr<duckdb::LogicalOperator> &source, PyObject *pyarrow_schema,
    std::string path, std::string compression, std::string bucket_region,
    int64_t row_group_size) {
    auto source_duck = to_duckdb(source);
    std::shared_ptr<arrow::Schema> arrow_schema = unwrap_schema(pyarrow_schema);

    duckdb::CopyFunction copy_function =
        duckdb::CopyFunction("bodo_parquet_write");
    duckdb::unique_ptr<duckdb::FunctionData> bind_data =
        duckdb::make_uniq<ParquetWriteFunctionData>(
            path, arrow_schema, compression, bucket_region, row_group_size);

    duckdb::unique_ptr<duckdb::LogicalCopyToFile> copy_node =
        duckdb::make_uniq<duckdb::LogicalCopyToFile>(
            copy_function, std::move(bind_data),
            duckdb::make_uniq<duckdb::CopyInfo>());

    copy_node->return_type = duckdb::CopyFunctionReturnType::CHANGED_ROWS;
    copy_node->AddChild(std::move(source_duck));

    return copy_node;
}

duckdb::unique_ptr<duckdb::LogicalCopyToFile> make_iceberg_write_node(
    std::unique_ptr<duckdb::LogicalOperator> &source, PyObject *pyarrow_schema,
    std::string table_loc, std::string bucket_region, int64_t max_pq_chunksize,
    std::string compression, PyObject *partition_tuples, PyObject *sort_tuples,
    std::string iceberg_schema_str, PyObject *output_pa_schema,
    PyObject *pyfs) {
    auto source_duck = to_duckdb(source);
    std::shared_ptr<arrow::Schema> arrow_schema = unwrap_schema(pyarrow_schema);

    if (arrow::py::import_pyarrow_wrappers()) {
        throw std::runtime_error("Importing pyarrow_wrappers failed!");
    }

    std::shared_ptr<arrow::Schema> iceberg_schema;
    CHECK_ARROW_AND_ASSIGN(arrow::py::unwrap_schema(output_pa_schema),
                           "Iceberg Schema Couldn't Unwrap from Python",
                           iceberg_schema);

    std::shared_ptr<arrow::fs::FileSystem> fs;
    CHECK_ARROW_AND_ASSIGN(
        arrow::py::unwrap_filesystem(pyfs),
        "Error during Iceberg write: Failed to unwrap Arrow filesystem", fs);

    duckdb::CopyFunction copy_function =
        duckdb::CopyFunction("bodo_iceberg_write");
    duckdb::unique_ptr<duckdb::FunctionData> bind_data =
        duckdb::make_uniq<IcebergWriteFunctionData>(
            arrow_schema, table_loc, bucket_region, max_pq_chunksize,
            compression, partition_tuples, sort_tuples, iceberg_schema_str,
            iceberg_schema, fs);

    duckdb::unique_ptr<duckdb::LogicalCopyToFile> copy_node =
        duckdb::make_uniq<duckdb::LogicalCopyToFile>(
            copy_function, std::move(bind_data),
            duckdb::make_uniq<duckdb::CopyInfo>());

    copy_node->return_type = duckdb::CopyFunctionReturnType::CHANGED_ROWS;
    copy_node->AddChild(std::move(source_duck));

    return copy_node;
}

duckdb::unique_ptr<duckdb::LogicalGet> make_dataframe_get_seq_node(
    PyObject *df, PyObject *pyarrow_schema, int64_t num_rows) {
    // See DuckDB Pandas scan code:
    // https://github.com/duckdb/duckdb/blob/d29a92f371179170688b4df394478f389bf7d1a6/tools/pythonpkg/src/include/duckdb_python/pandas/pandas_scan.hpp#L19
    // https://github.com/duckdb/duckdb/blob/d29a92f371179170688b4df394478f389bf7d1a6/tools/pythonpkg/src/include/duckdb_python/pandas/pandas_bind.hpp#L19
    // https://github.com/duckdb/duckdb/blob/d29a92f371179170688b4df394478f389bf7d1a6/tools/pythonpkg/src/pandas/scan.cpp#L185

    std::shared_ptr<arrow::Schema> arrow_schema = unwrap_schema(pyarrow_schema);

    duckdb::shared_ptr<duckdb::Binder> binder = get_duckdb_binder();

    BodoDataFrameScanFunction table_function =
        BodoDataFrameScanFunction(arrow_schema);
    duckdb::unique_ptr<duckdb::FunctionData> bind_data1 =
        duckdb::make_uniq<BodoDataFrameSeqScanFunctionData>(df, arrow_schema);

    // Convert Arrow schema to DuckDB
    auto [return_names, return_types] = arrow_schema_to_duckdb(arrow_schema);

    duckdb::virtual_column_map_t virtual_columns;

    auto out_get = duckdb::make_uniq<duckdb::LogicalGet>(
        binder->GenerateTableIndex(), table_function, std::move(bind_data1),
        return_types, return_names, virtual_columns);
    out_get->SetEstimatedCardinality(num_rows);

    // Column ids need to be added separately.
    // DuckDB column id initialization example:
    // https://github.com/duckdb/duckdb/blob/d29a92f371179170688b4df394478f389bf7d1a6/src/catalog/catalog_entry/table_catalog_entry.cpp#L252
    for (size_t i = 0; i < return_names.size(); i++) {
        out_get->AddColumnId(i);
    }

    return out_get;
}

duckdb::unique_ptr<duckdb::LogicalGet> make_dataframe_get_parallel_node(
    std::string result_id, PyObject *pyarrow_schema, int64_t num_rows) {
    duckdb::shared_ptr<duckdb::Binder> binder = get_duckdb_binder();
    std::shared_ptr<arrow::Schema> arrow_schema = unwrap_schema(pyarrow_schema);

    BodoDataFrameScanFunction table_function =
        BodoDataFrameScanFunction(arrow_schema);
    duckdb::unique_ptr<duckdb::FunctionData> bind_data1 =
        duckdb::make_uniq<BodoDataFrameParallelScanFunctionData>(result_id,
                                                                 arrow_schema);

    // Convert Arrow schema to DuckDB
    auto [return_names, return_types] = arrow_schema_to_duckdb(arrow_schema);

    auto out_get = duckdb::make_uniq<duckdb::LogicalGet>(
        binder->GenerateTableIndex(), table_function, std::move(bind_data1),
        return_types, return_names);
    out_get->SetEstimatedCardinality(num_rows);

    // Column ids need to be added separately.
    // DuckDB column id initialization example:
    // https://github.com/duckdb/duckdb/blob/d29a92f371179170688b4df394478f389bf7d1a6/src/catalog/catalog_entry/table_catalog_entry.cpp#L252
    for (size_t i = 0; i < return_names.size(); i++) {
        out_get->AddColumnId(i);
    }

    return out_get;
}

duckdb::unique_ptr<duckdb::LogicalGet> make_iceberg_get_node(
    PyObject *pyarrow_schema, std::string table_name,
    PyObject *pyiceberg_catalog, PyObject *iceberg_filter,
    PyObject *iceberg_schema, int64_t snapshot_id,
    uint64_t table_len_estimate) {
    duckdb::shared_ptr<duckdb::Binder> binder = get_duckdb_binder();

    // Convert Arrow schema to DuckDB
    std::shared_ptr<arrow::Schema> arrow_schema = unwrap_schema(pyarrow_schema);
    auto [return_names, return_types] = arrow_schema_to_duckdb(arrow_schema);

    BodoIcebergScanFunction table_function =
        BodoIcebergScanFunction(arrow_schema);
    duckdb::unique_ptr<duckdb::FunctionData> bind_data1 =
        duckdb::make_uniq<BodoIcebergScanFunctionData>(
            arrow_schema, pyiceberg_catalog, table_name, iceberg_filter,
            iceberg_schema, snapshot_id);

    duckdb::virtual_column_map_t virtual_columns;

    duckdb::unique_ptr<duckdb::LogicalGet> out_get =
        duckdb::make_uniq<duckdb::LogicalGet>(
            binder->GenerateTableIndex(), table_function, std::move(bind_data1),
            return_types, return_names, virtual_columns);

    out_get->SetEstimatedCardinality(table_len_estimate);

    // Column ids need to be added separately.
    // DuckDB column id initialization example:
    // https:  //
    // github.com/duckdb/duckdb/blob/d29a92f371179170688b4df394478f389bf7d1a6/src/catalog/catalog_entry/table_catalog_entry.cpp#L252
    for (size_t i = 0; i < return_names.size(); i++) {
        out_get->AddColumnId(i);
    }

    return out_get;
}

void registerFloor(duckdb::shared_ptr<duckdb::DuckDB> db) {
    duckdb::LogicalType double_type(duckdb::LogicalType::DOUBLE);
    duckdb::LogicalType float_type(duckdb::LogicalType::FLOAT);
    duckdb::vector<duckdb::LogicalType> double_arguments = {double_type};
    duckdb::vector<duckdb::LogicalType> float_arguments = {float_type};
    duckdb::ScalarFunction floor_fun_double("floor", double_arguments,
                                            double_type, nullptr);
    duckdb::ScalarFunction floor_fun_float("floor", float_arguments, float_type,
                                           nullptr);
    duckdb::ScalarFunctionSet floor_set("floor");
    floor_set.AddFunction(floor_fun_double);
    floor_set.AddFunction(floor_fun_float);
    duckdb::CreateScalarFunctionInfo floor_info(floor_set);
    auto &system_catalog = duckdb::Catalog::GetSystemCatalog(*(db->instance));
    auto data =
        duckdb::CatalogTransaction::GetSystemTransaction(*(db->instance));
    system_catalog.CreateFunction(data, floor_info);
}

duckdb::shared_ptr<duckdb::DuckDB> get_duckdb() {
    static duckdb::shared_ptr<duckdb::DuckDB> db =
        duckdb::make_shared_ptr<duckdb::DuckDB>(nullptr);
    static bool floor_registered = []() {
        registerFloor(db);
        return true;
    }();
    // Prevent unused variable error.
    (void)floor_registered;
    return db;
}

duckdb::shared_ptr<duckdb::ClientContext> get_duckdb_context() {
    duckdb::shared_ptr<duckdb::DuckDB> db = get_duckdb();
    static duckdb::shared_ptr<duckdb::ClientContext> context =
        duckdb::make_shared_ptr<duckdb::ClientContext>(db->instance);
    return context;
}

duckdb::shared_ptr<duckdb::Binder> get_duckdb_binder() {
    duckdb::shared_ptr<duckdb::ClientContext> cc = get_duckdb_context();
    static duckdb::shared_ptr<duckdb::Binder> binder =
        duckdb::Binder::CreateBinder(*cc);
    return binder;
}

duckdb::shared_ptr<duckdb::Optimizer> get_duckdb_optimizer() {
    duckdb::shared_ptr<duckdb::ClientContext> cc = get_duckdb_context();
    duckdb::shared_ptr<duckdb::Binder> binder = get_duckdb_binder();
    static duckdb::shared_ptr<duckdb::Optimizer> optimizer =
        duckdb::make_shared_ptr<duckdb::Optimizer>(*binder, *cc);
    return optimizer;
}

std::pair<duckdb::vector<duckdb::string>, duckdb::vector<duckdb::LogicalType>>
arrow_schema_to_duckdb(const std::shared_ptr<arrow::Schema> &arrow_schema) {
    // See Arrow type handling in DuckDB for possible cases:
    // https://github.com/duckdb/duckdb/blob/d29a92f371179170688b4df394478f389bf7d1a6/src/function/table/arrow/arrow_duck_schema.cpp#L59
    // https://github.com/duckdb/duckdb/blob/d29a92f371179170688b4df394478f389bf7d1a6/src/common/adbc/nanoarrow/schema.cpp#L73
    // Arrow types:
    // https://github.com/apache/arrow/blob/5e9fce493f21098d616f08034bc233fcc529b3ad/cpp/src/arrow/type_fwd.h#L322

    duckdb::vector<duckdb::string> return_names;
    duckdb::vector<duckdb::LogicalType> logical_types;

    for (int i = 0; i < arrow_schema->num_fields(); i++) {
        const std::shared_ptr<arrow::Field> &field = arrow_schema->field(i);
        auto [return_name, duckdb_type] = arrow_field_to_duckdb(field);

        return_names.emplace_back(field->name());
        logical_types.push_back(duckdb_type);
    }

    return {return_names, logical_types};
}

std::pair<duckdb::string, duckdb::LogicalType> arrow_field_to_duckdb(
    const std::shared_ptr<arrow::Field> &field) {
    // Convert Arrow type to DuckDB LogicalType
    // TODO: handle all types
    duckdb::LogicalType duckdb_type;
    const std::shared_ptr<arrow::DataType> &arrow_type = field->type();
    switch (arrow_type->id()) {
        case arrow::Type::NA: {
            duckdb_type = duckdb::LogicalType::SQLNULL;
            break;
        }
        case arrow::Type::STRING:
        case arrow::Type::LARGE_STRING: {
            duckdb_type = duckdb::LogicalType::VARCHAR;
            break;
        }
        case arrow::Type::BINARY:
        case arrow::Type::LARGE_BINARY: {
            duckdb_type = duckdb::LogicalType::BLOB;
            break;
        }
        case arrow::Type::UINT8: {
            duckdb_type = duckdb::LogicalType::UTINYINT;
            break;
        }
        case arrow::Type::INT8: {
            duckdb_type = duckdb::LogicalType::TINYINT;
            break;
        }
        case arrow::Type::UINT16: {
            duckdb_type = duckdb::LogicalType::USMALLINT;
            break;
        }
        case arrow::Type::INT16: {
            duckdb_type = duckdb::LogicalType::SMALLINT;
            break;
        }
        case arrow::Type::UINT32: {
            duckdb_type = duckdb::LogicalType::UINTEGER;
            break;
        }
        case arrow::Type::INT32: {
            duckdb_type = duckdb::LogicalType::INTEGER;
            break;
        }
        case arrow::Type::UINT64: {
            duckdb_type = duckdb::LogicalType::UBIGINT;
            break;
        }
        case arrow::Type::INT64: {
            duckdb_type = duckdb::LogicalType::BIGINT;
            break;
        }
        case arrow::Type::FLOAT: {
            duckdb_type = duckdb::LogicalType::FLOAT;
            break;
        }
        case arrow::Type::DOUBLE: {
            duckdb_type = duckdb::LogicalType::DOUBLE;
            break;
        }
        case arrow::Type::BOOL: {
            duckdb_type = duckdb::LogicalType::BOOLEAN;
            break;
        }
        case arrow::Type::DATE32: {
            duckdb_type = duckdb::LogicalType::DATE;
            break;
        }
        case arrow::Type::DURATION: {
            duckdb_type = duckdb::LogicalType::INTERVAL;
            break;
        }
        case arrow::Type::TIMESTAMP: {
            auto timestamp_type =
                std::static_pointer_cast<arrow::TimestampType>(arrow_type);
            arrow::TimeUnit::type unit = timestamp_type->unit();
            std::string tz = timestamp_type->timezone();
            if (tz == "") {
                switch (unit) {
                    case arrow::TimeUnit::NANO:
                        duckdb_type = duckdb::LogicalType::TIMESTAMP_NS;
                        break;
                    // TODO: Support these types in Bodo
                    case arrow::TimeUnit::MICRO:
                        // microseconds
                        duckdb_type = duckdb::LogicalType::TIMESTAMP;
                        break;
                    case arrow::TimeUnit::MILLI:
                        duckdb_type = duckdb::LogicalType::TIMESTAMP_MS;
                        break;
                    case arrow::TimeUnit::SECOND:
                        duckdb_type = duckdb::LogicalType::TIMESTAMP_S;
                        break;
                }
            } else {
                // TODO: Do we need to check units here?
                // Technically this is supposed to be in microseconds like
                // TIMESTAMP
                duckdb_type = duckdb::LogicalType::TIMESTAMP_TZ;
            }
            break;
        }
        case arrow::Type::DECIMAL128: {
            auto decimal_type =
                std::static_pointer_cast<arrow::DecimalType>(arrow_type);
            int32_t precision = decimal_type->precision();
            int32_t scale = decimal_type->scale();
            duckdb_type = duckdb::LogicalType::DECIMAL(precision, scale);
            break;
        }
        case arrow::Type::LIST: {
            auto list_type =
                std::static_pointer_cast<arrow::ListType>(arrow_type);
            auto [name, child_type] =
                arrow_field_to_duckdb(list_type->value_field());
            duckdb_type = duckdb::LogicalType::LIST(child_type);
            break;
        }
        case arrow::Type::LARGE_LIST: {
            auto list_type =
                std::static_pointer_cast<arrow::LargeListType>(arrow_type);
            auto [name, child_type] =
                arrow_field_to_duckdb(list_type->value_field());
            duckdb_type = duckdb::LogicalType::LIST(child_type);
            break;
        }
        case arrow::Type::STRUCT: {
            duckdb::child_list_t<duckdb::LogicalType> children;
            for (std::shared_ptr<arrow::Field> field : arrow_type->fields()) {
                auto [field_name, duckdb_type] = arrow_field_to_duckdb(field);
                children.push_back({field_name, duckdb_type});
            }
            duckdb_type = duckdb::LogicalType::STRUCT(children);
            break;
        }
        case arrow::Type::MAP: {
            auto map_type =
                std::static_pointer_cast<arrow::MapType>(arrow_type);
            auto [key_name, duckdb_key_type] =
                arrow_field_to_duckdb(map_type->key_field());
            auto [item_name, duckdb_value_type] =
                arrow_field_to_duckdb(map_type->item_field());
            duckdb_type =
                duckdb::LogicalType::MAP(duckdb_key_type, duckdb_value_type);
            break;
        }
        case arrow::Type::DICTIONARY: {
            auto dict_type =
                std::static_pointer_cast<arrow::DictionaryType>(arrow_type);
            std::shared_ptr<arrow::Field> value_field =
                arrow::field("name", dict_type->value_type());
            auto [field_name, inner_type] = arrow_field_to_duckdb(value_field);
            duckdb_type = inner_type;
            break;
        }
        case arrow::Type::TIME64: {
            auto time64_type =
                std::static_pointer_cast<arrow::Time64Type>(arrow_type);
            switch (time64_type->unit()) {
                case arrow::TimeUnit::MICRO:
                    duckdb_type = duckdb::LogicalType::TIME;
                    break;
                case arrow::TimeUnit::NANO:
                    duckdb_type = duckdb::LogicalType::TIME;
                    break;
                default:
                    throw std::runtime_error("Unsupported Time64 unit");
            }
            break;
        }
        default:
            throw std::runtime_error(
                "Unsupported Arrow type: " + arrow_type->ToString() +
                ". Please extend the arrow_schema_to_duckdb function to handle "
                "this type.");
    }
    return {field->name(), duckdb_type};
}

std::string plan_to_string(std::unique_ptr<duckdb::LogicalOperator> &plan,
                           bool graphviz_format) {
    return plan->ToString(graphviz_format ? duckdb::ExplainFormat::GRAPHVIZ
                                          : duckdb::ExplainFormat::TEXT);
}

int planCountNodes(std::unique_ptr<duckdb::LogicalOperator> &op) {
    int ret = 1;  // count yourself
    for (auto &child : op->children) {
        ret += planCountNodes(child);
    }
    return ret;
}

int64_t pyarrow_to_cpp_table(PyObject *pyarrow_table) {
    // Unwrap Arrow table from Python object
    std::shared_ptr<arrow::Table> table =
        arrow::py::unwrap_table(pyarrow_table).ValueOrDie();
    std::shared_ptr<table_info> out_table = arrow_table_to_bodo(table, nullptr);
    return reinterpret_cast<int64_t>(new table_info(*out_table));
}

PyObject *cpp_table_to_pyarrow(int64_t cpp_table) {
    std::shared_ptr<table_info> table =
        std::shared_ptr<table_info>(reinterpret_cast<table_info *>(cpp_table));
    std::shared_ptr<arrow::Table> arrow_table = bodo_table_to_arrow(table);
    return arrow::py::wrap_table(arrow_table);
}

#undef CHECK_ARROW
#undef CHECK_ARROW_AND_ASSIGN
