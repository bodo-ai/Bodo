#include "read_iceberg.h"
#include <arrow/util/key_value_metadata.h>
#include "physical/operator.h"

PhysicalReadIceberg::PhysicalReadIceberg(
    PyObject *catalog, const std::string table_id, PyObject *iceberg_filter,
    std::shared_ptr<arrow::Schema> arrow_schema,
    std::vector<int> &selected_columns, duckdb::TableFilterSet &filter_exprs,
    duckdb::unique_ptr<duckdb::BoundLimitNode> &limit_val)
    : arrow_schema(std::move(arrow_schema)),
      out_arrow_schema(
          this->create_out_arrow_schema(this->arrow_schema, selected_columns)),
      internal_reader(this->create_internal_reader(
          catalog, table_id, iceberg_filter, this->arrow_schema,
          selected_columns, limit_val)),
      out_metadata(std::make_shared<TableMetadata>(
          this->arrow_schema->metadata()->keys(),
          this->arrow_schema->metadata()->values())),
      out_column_names(this->create_out_column_names(selected_columns,
                                                     this->arrow_schema)) {}

std::pair<std::shared_ptr<table_info>, OperatorResult>
PhysicalReadIceberg::ProduceBatch() {
    uint64_t total_rows;
    bool is_last;

    table_info *batch = internal_reader->read_batch(is_last, total_rows, true);
    auto result =
        is_last ? OperatorResult::FINISHED : OperatorResult::HAVE_MORE_OUTPUT;

    batch->column_names = out_column_names;
    batch->metadata = out_metadata;
    return std::make_pair(std::shared_ptr<table_info>(batch), result);
}

const std::shared_ptr<bodo::Schema> PhysicalReadIceberg::getOutputSchema() {
    return bodo::Schema::FromArrowSchema(this->out_arrow_schema);
}

std::vector<std::string> PhysicalReadIceberg::create_out_column_names(
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
PhysicalReadIceberg::create_internal_reader(
    PyObject *catalog, const std::string table_id, PyObject *iceberg_filter,
    std::shared_ptr<arrow::Schema> arrow_schema,
    std::vector<int> &selected_columns,
    duckdb::unique_ptr<duckdb::BoundLimitNode> &limit_val) {
    // Pipeline buffers assume everything is nullable
    std::vector<bool> is_nullable(selected_columns.size(), true);

    int64_t total_rows_to_read = -1;  // Default to read everything.
    if (limit_val) {
        // If the limit option is present...
        if (limit_val->Type() != duckdb::LimitNodeType::CONSTANT_VALUE) {
            throw std::runtime_error(
                "PhysicalReadParquet unsupported limit type");
        }
        // Limit the rows to read to the limit value.
        total_rows_to_read = limit_val->GetConstantValue();
    }
    // We're borrowing a reference to the catalog object, so we need to
    // increment the reference count since the reader steals it.
    Py_INCREF(catalog);
    auto reader = std::make_unique<IcebergParquetReader>(
        catalog, table_id.c_str(), true, total_rows_to_read, iceberg_filter, "",
        Py_None, selected_columns, is_nullable,
        arrow::py::wrap_schema(arrow_schema), get_streaming_batch_size(), -1,
        -1);
    // TODO: Figure out cols to dict encode
    reader->init_iceberg_reader({}, false);
    return reader;
}

std::shared_ptr<arrow::Schema> PhysicalReadIceberg::create_out_arrow_schema(
    std::shared_ptr<arrow::Schema> arrow_schema,
    const std::vector<int> &selected_columns) {
    // Create a new schema with only the selected columns.
    std::vector<std::shared_ptr<arrow::Field>> fields;
    fields.reserve(selected_columns.size());
    for (int i : selected_columns) {
        if (!(i >= 0 && i < arrow_schema->num_fields())) {
            throw std::runtime_error(
                "PhysicalReadParquet(): invalid column index " +
                std::to_string(i) + " for schema with " +
                std::to_string(arrow_schema->num_fields()) + " fields");
        }
        fields.push_back(arrow_schema->field(i));
    }
    return arrow::schema(fields, arrow_schema->metadata());
}
