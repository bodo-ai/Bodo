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
    std::string parquet_path, PyObject* pyarrow_schema) {
    duckdb::shared_ptr<duckdb::Binder> binder = get_duckdb_binder();

    BodoParquetScanFunction table_function = BodoParquetScanFunction();
    duckdb::unique_ptr<duckdb::FunctionData> bind_data1 =
        duckdb::make_uniq<BodoParquetScanFunctionData>(parquet_path);

    // Convert Arrow schema to DuckDB
    std::shared_ptr<arrow::Schema> arrow_schema = unwrap_schema(pyarrow_schema);
    auto [return_names, return_types] = arrow_schema_to_duckdb(arrow_schema);

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

std::pair<duckdb::vector<duckdb::string>, duckdb::vector<duckdb::LogicalType>>
arrow_schema_to_duckdb(std::shared_ptr<arrow::Schema> arrow_schema) {
    // See Arrow type handling in DuckDB for possible cases:
    // https://github.com/duckdb/duckdb/blob/d29a92f371179170688b4df394478f389bf7d1a6/src/function/table/arrow/arrow_duck_schema.cpp#L59
    // https://github.com/duckdb/duckdb/blob/d29a92f371179170688b4df394478f389bf7d1a6/src/common/adbc/nanoarrow/schema.cpp#L73
    // Arrow types:
    // https://github.com/apache/arrow/blob/5e9fce493f21098d616f08034bc233fcc529b3ad/cpp/src/arrow/type_fwd.h#L322

    duckdb::vector<duckdb::string> return_names;
    duckdb::vector<duckdb::LogicalType> logical_types;

    for (int i = 0; i < arrow_schema->num_fields(); i++) {
        const std::shared_ptr<arrow::Field>& field = arrow_schema->field(i);
        return_names.emplace_back(field->name());
        const std::shared_ptr<arrow::DataType>& arrow_type = field->type();

        // Convert Arrow type to DuckDB LogicalType
        // TODO: handle all types
        duckdb::LogicalType duckdb_type;
        if ((arrow_type->id() == arrow::Type::STRING) ||
            (arrow_type->id() == arrow::Type::LARGE_STRING)) {
            duckdb_type = duckdb::LogicalType::VARCHAR;
        } else if (arrow_type->id() == arrow::Type::INT32) {
            duckdb_type = duckdb::LogicalType::INTEGER;
        } else if (arrow_type->id() == arrow::Type::INT64) {
            duckdb_type = duckdb::LogicalType::BIGINT;
        } else if (arrow_type->id() == arrow::Type::FLOAT) {
            duckdb_type = duckdb::LogicalType::FLOAT;
        } else if (arrow_type->id() == arrow::Type::DOUBLE) {
            duckdb_type = duckdb::LogicalType::DOUBLE;
        } else if (arrow_type->id() == arrow::Type::BOOL) {
            duckdb_type = duckdb::LogicalType::BOOLEAN;
        } else {
            throw std::runtime_error(
                "Unsupported Arrow type: " + arrow_type->ToString() +
                ". Please extend the arrow_schema_to_duckdb function to handle "
                "this type.");
        }

        logical_types.push_back(duckdb_type);
    }

    return {return_names, logical_types};
}
