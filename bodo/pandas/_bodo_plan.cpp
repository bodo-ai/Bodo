#include "_bodo_plan.h"
#include <utility>
#include "duckdb.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_get.hpp"

std::unique_ptr<duckdb::LogicalOperator> optimize_plan(
    std::unique_ptr<duckdb::LogicalOperator> plan) {
    duckdb::Optimizer& optimizer = get_duckdb_optimizer();

    // Convert std::unique_ptr to duckdb::unique_ptr
    // Input is using std since Cython supports it
    duckdb::unique_ptr<duckdb::LogicalOperator> in_plan =
        duckdb::unique_ptr<duckdb::LogicalOperator>(plan.release());

    duckdb::unique_ptr<duckdb::LogicalOperator> optimized_plan =
        optimizer.Optimize(std::move(in_plan));

    return std::unique_ptr<duckdb::LogicalOperator>(optimized_plan.release());
}

duckdb::unique_ptr<duckdb::LogicalComparisonJoin> make_comparison_join(
    std::unique_ptr<duckdb::LogicalOperator>& lhs,
    std::unique_ptr<duckdb::LogicalOperator>& rhs, duckdb::JoinType join_type,
    std::vector<std::pair<int, int>>& cond_vec) {
    // Convert std::unique_ptr to duckdb::unique_ptr.
    duckdb::unique_ptr<duckdb::LogicalOperator> lhs_duck =
        static_cast<duckdb::unique_ptr<duckdb::LogicalOperator>&&>(
            std::move(lhs));
    duckdb::unique_ptr<duckdb::LogicalOperator> rhs_duck =
        static_cast<duckdb::unique_ptr<duckdb::LogicalOperator>&&>(
            std::move(rhs));
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

duckdb::ClientContext& get_duckdb_context() {
    static duckdb::DuckDB db(nullptr);
    static duckdb::ClientContext context(db.instance);
    return context;
}

duckdb::shared_ptr<duckdb::Binder> get_duckdb_binder() {
    static duckdb::shared_ptr<duckdb::Binder> binder =
        duckdb::Binder::CreateBinder(get_duckdb_context());
    return binder;
}

duckdb::Optimizer& get_duckdb_optimizer() {
    static duckdb::shared_ptr<duckdb::Binder> binder = get_duckdb_binder();
    static duckdb::Optimizer optimizer(*binder, get_duckdb_context());
    return optimizer;
}
