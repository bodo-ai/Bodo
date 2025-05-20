#include "read_iceberg.h"

PhysicalReadIceberg::PhysicalReadIceberg(
    std::vector<int> &selected_columns, duckdb::TableFilterSet &filter_exprs,
    duckdb::unique_ptr<duckdb::BoundLimitNode> &limit_val) {}

std::pair<std::shared_ptr<table_info>, OperatorResult>
PhysicalReadIceberg::ProduceBatch() {
    return std::make_pair(nullptr, OperatorResult::FINISHED);
}

const std::shared_ptr<bodo::Schema> PhysicalReadIceberg::getOutputSchema() {
    return bodo::Schema::FromArrowSchema(this->arrow_schema);
}
