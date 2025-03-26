#include "_bodo_plan.h"
#include <utility>
#include "duckdb.hpp"
#include "duckdb/common/unique_ptr.hpp"
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
