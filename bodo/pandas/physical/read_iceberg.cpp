#include "read_iceberg.h"
#include <arrow/util/key_value_metadata.h>

PhysicalReadIceberg::PhysicalReadIceberg(
    std::shared_ptr<arrow::Schema> arrow_schema,
    std::vector<int> &selected_columns, duckdb::TableFilterSet &filter_exprs,
    duckdb::unique_ptr<duckdb::BoundLimitNode> &limit_val)
    : arrow_schema(std::move(arrow_schema)),
      internal_reader(this->_create_internal_reader()),
      out_metadata(std::make_unique<TableMetadata>(
          this->arrow_schema->metadata()->keys(),
          this->arrow_schema->metadata()->values())),
      out_column_names(this->_create_out_column_names(selected_columns,
                                                      this->arrow_schema)) {
    // TODO: Handle filter expressions and limit value
}

std::pair<std::shared_ptr<table_info>, OperatorResult>
PhysicalReadIceberg::ProduceBatch() {
    return std::make_pair(nullptr, OperatorResult::FINISHED);
}

const std::shared_ptr<bodo::Schema> PhysicalReadIceberg::getOutputSchema() {
    return bodo::Schema::FromArrowSchema(this->arrow_schema);
}

std::vector<std::string> PhysicalReadIceberg::_create_out_column_names(
    const std::vector<int> &selected_columns,
    const std::shared_ptr<arrow::Schema> schema) {
    std::vector<std::string> out_column_names;
    for (int i : selected_columns) {
        if (!(i >= 0 && i < schema->num_fields())) {
            throw std::runtime_error(
                "PhysicalReadParquet(): invalid column index " +
                std::to_string(i) + " for schema with " +
                std::to_string(schema->num_fields()) + " fields");
        }
        out_column_names.emplace_back(schema->field(i)->name());
    }
    return out_column_names;
}

std::unique_ptr<IcebergParquetReader>
PhysicalReadIceberg::_create_internal_reader() {
    return nullptr;
}
