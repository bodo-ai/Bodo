#include "read_iceberg.h"

PhysicalReadIceberg::PhysicalReadIceberg(
    std::shared_ptr<arrow::Schema> arrow_schema,
    std::vector<int> &selected_columns, duckdb::TableFilterSet &filter_exprs,
    duckdb::unique_ptr<duckdb::BoundLimitNode> &limit_val)
    : arrow_schema(std::move(arrow_schema)), {
    std::vector<bool> is_nullable(selected_columns.size(), true);
}

std::pair<std::shared_ptr<table_info>, OperatorResult>
PhysicalReadIceberg::ProduceBatch() {
    return std::make_pair(nullptr, OperatorResult::FINISHED);
}

const std::shared_ptr<bodo::Schema> PhysicalReadIceberg::getOutputSchema() {
    return bodo::Schema::FromArrowSchema(this->arrow_schema);
}
