#include "_bodo_plan.h"
#include <utility>
#include "duckdb.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"

template <class T>
duckdb::unique_ptr<T> to_duckdb(std::unique_ptr<T> &val) {
    duckdb::unique_ptr<T> ret = duckdb::unique_ptr<T>(val.release());
    return ret;
}

duckdb::unique_ptr<duckdb::LogicalOperator> optimize_plan(
    std::unique_ptr<duckdb::LogicalOperator> plan) {
    duckdb::Optimizer &optimizer = get_duckdb_optimizer();

    // Convert std::unique_ptr to duckdb::unique_ptr
    // Input is using std since Cython supports it
    auto in_plan = to_duckdb(plan);

    duckdb::unique_ptr<duckdb::LogicalOperator> optimized_plan =
        optimizer.Optimize(std::move(in_plan));

    return optimized_plan;
    // return
    // std::unique_ptr<duckdb::LogicalOperator>(optimized_plan.release());
}

duckdb::unique_ptr<duckdb::Expression> make_const_int_expr(int val) {
    return duckdb::make_uniq<duckdb::BoundConstantExpression>(
        duckdb::Value(val));
}

duckdb::unique_ptr<duckdb::Expression> make_col_ref_expr(
    duckdb::LogicalTypeId ctype, int col_idx) {
    auto binder = get_duckdb_binder();
    auto table_idx = binder.get()->GenerateTableIndex();
    return duckdb::make_uniq<duckdb::BoundColumnRefExpression>(
        ctype, duckdb::ColumnBinding(table_idx, col_idx));
}

duckdb::unique_ptr<duckdb::Expression> make_binop_expr(
    std::unique_ptr<duckdb::Expression> &lhs,
    std::unique_ptr<duckdb::Expression> &rhs, duckdb::ExpressionType etype) {
    // Convert std::unique_ptr to duckdb::unique_ptr.
    auto lhs_duck = to_duckdb(lhs);
    auto rhs_duck = to_duckdb(rhs);
    auto filter_expression =
        duckdb::make_uniq<duckdb::BoundComparisonExpression>(
            etype, std::move(lhs_duck), std::move(rhs_duck));
    return filter_expression;
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

duckdb::unique_ptr<duckdb::LogicalProjection> make_projection(
    std::unique_ptr<duckdb::LogicalOperator> &source,
    std::vector<int> &select_vec,
    std::vector<duckdb::LogicalTypeId> &type_vec) {
    // Convert std::unique_ptr to duckdb::unique_ptr.
    auto source_duck = to_duckdb(source);
    auto binder = get_duckdb_binder();
    auto table_idx = binder.get()->GenerateTableIndex();

    assert(select_vec.size() == type_vec.size());
    std::vector<duckdb::unique_ptr<duckdb::Expression>> projection_expressions;
    for (size_t i = 0; i < select_vec.size(); ++i) {
        auto selection = select_vec[i];
        auto stype = type_vec[i];
        auto expr = duckdb::make_uniq<duckdb::BoundColumnRefExpression>(
            stype, duckdb::ColumnBinding(table_idx, selection));
        projection_expressions.push_back(std::move(expr));
    }

    // Create join node.
    duckdb::unique_ptr<duckdb::LogicalProjection> proj =
        duckdb::make_uniq<duckdb::LogicalProjection>(
            table_idx, std::move(projection_expressions));

    // Add the sources to be joined.
    proj->children.push_back(std::move(source_duck));

    return proj;
}

duckdb::unique_ptr<duckdb::LogicalComparisonJoin> make_comparison_join(
    std::unique_ptr<duckdb::LogicalOperator> &lhs,
    std::unique_ptr<duckdb::LogicalOperator> &rhs, duckdb::JoinType join_type,
    std::vector<std::pair<int, int>> &cond_vec) {
    // Convert std::unique_ptr to duckdb::unique_ptr.
    auto lhs_duck = to_duckdb(lhs);
    auto rhs_duck = to_duckdb(rhs);
    // Create join node.
    duckdb::unique_ptr<duckdb::LogicalComparisonJoin> comp_join =
        duckdb::make_uniq<duckdb::LogicalComparisonJoin>(join_type);
    // Create join condition.
    duckdb::LogicalType cbtype(duckdb::LogicalTypeId::INTEGER);
    for (std::pair<int, int> cond_pair : cond_vec) {
        duckdb::JoinCondition cond;
        cond.comparison = duckdb::ExpressionType::COMPARE_EQUAL;
        cond.left = duckdb::make_uniq<duckdb::BoundColumnRefExpression>(
            cbtype, duckdb::ColumnBinding(0, cond_pair.first));
        cond.right = duckdb::make_uniq<duckdb::BoundColumnRefExpression>(
            cbtype, duckdb::ColumnBinding(0, cond_pair.second));
        // Add the join condition to the join node.
        comp_join->conditions.push_back(std::move(cond));
    }
    // Add the sources to be joined.
    comp_join->children.push_back(std::move(lhs_duck));
    comp_join->children.push_back(std::move(rhs_duck));

    return comp_join;
}

duckdb::unique_ptr<duckdb::LogicalGet> make_parquet_get_node(
    std::string parquet_path) {
    duckdb::shared_ptr<duckdb::Binder> binder = get_duckdb_binder();

    BodoParquetScanFunction table_function = BodoParquetScanFunction();
    duckdb::unique_ptr<duckdb::FunctionData> bind_data1 =
        duckdb::make_uniq<BodoParquetScanFunctionData>(parquet_path);

    // TODO: replace dummy values with actual schema data
    duckdb::vector<duckdb::LogicalType> return_types;
    return_types.push_back(duckdb::LogicalType::INTEGER);
    duckdb::vector<duckdb::string> return_names;
    return_names.push_back("A");
    duckdb::virtual_column_map_t virtual_columns;

    return duckdb::make_uniq<duckdb::LogicalGet>(
        binder->GenerateTableIndex(), table_function, std::move(bind_data1),
        return_types, return_names, virtual_columns);
}

duckdb::ClientContext &get_duckdb_context() {
    static duckdb::DuckDB db(nullptr);
    static duckdb::ClientContext context(db.instance);
    return context;
}

duckdb::shared_ptr<duckdb::Binder> get_duckdb_binder() {
    static duckdb::shared_ptr<duckdb::Binder> binder =
        duckdb::Binder::CreateBinder(get_duckdb_context());
    return binder;
}

duckdb::Optimizer &get_duckdb_optimizer() {
    static duckdb::shared_ptr<duckdb::Binder> binder = get_duckdb_binder();
    static duckdb::Optimizer optimizer(*binder, get_duckdb_context());
    return optimizer;
}
