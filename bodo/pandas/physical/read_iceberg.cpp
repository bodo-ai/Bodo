#include "read_iceberg.h"
#include <arrow/util/key_value_metadata.h>
#include "../../libs/_utils.h"
#include "../_util.h"
#include "physical/operator.h"

PhysicalReadIceberg::PhysicalReadIceberg(
    PyObject *catalog, const std::string table_id, PyObject *iceberg_filter,
    PyObject *iceberg_schema, const std::shared_ptr<arrow::Schema> arrow_schema,
    const int64_t snapshot_id, const std::vector<int> &selected_columns,
    duckdb::TableFilterSet &filter_exprs,
    duckdb::unique_ptr<duckdb::BoundLimitNode> &limit_val,
    JoinFilterColStats join_filter_col_stats)
    : catalog(catalog),
      table_id(table_id),
      iceberg_filter(iceberg_filter),
      iceberg_schema(iceberg_schema),
      snapshot_id(snapshot_id),
      filter_exprs(filter_exprs.Copy()),
      arrow_schema(std::move(arrow_schema)),
      selected_columns(selected_columns),
      out_arrow_schema(
          this->create_out_arrow_schema(this->arrow_schema, selected_columns)),
      join_filter_col_stats(std::move(join_filter_col_stats)),
      out_metadata(std::make_shared<bodo::TableMetadata>(
          this->arrow_schema->metadata()->keys(),
          this->arrow_schema->metadata()->values())),
      out_column_names(
          this->create_out_column_names(selected_columns, this->arrow_schema)) {
    Py_INCREF(this->catalog);
    Py_INCREF(this->iceberg_filter);
    Py_INCREF(this->iceberg_schema);

    // Determine total rows to read based on limit_val
    if (limit_val) {
        // If the limit option is present...
        if (limit_val->Type() != duckdb::LimitNodeType::CONSTANT_VALUE) {
            throw std::runtime_error(
                "PhysicalReadIceberg unsupported limit type");
        }
        // Limit the rows to read to the limit value.
        this->total_rows_to_read = limit_val->GetConstantValue();
    }
}

std::pair<std::shared_ptr<table_info>, OperatorResult>
PhysicalReadIceberg::ProduceBatch() {
    if (!this->internal_reader) {
        time_pt start_init = start_timer();

        this->internal_reader = this->create_internal_reader();

        this->metrics.init_time += end_timer(start_init);
    }

    uint64_t total_rows;
    bool is_last;
    time_pt start_produce = start_timer();

    table_info *batch = internal_reader->read_batch(is_last, total_rows, true);
    auto result =
        is_last ? OperatorResult::FINISHED : OperatorResult::HAVE_MORE_OUTPUT;

    batch->column_names = out_column_names;
    batch->metadata = out_metadata;
    this->metrics.produce_time += end_timer(start_produce);
    this->metrics.rows_read += total_rows;
    return std::make_pair(std::shared_ptr<table_info>(batch), result);
}

void PhysicalReadIceberg::FinalizeSource() {
    std::vector<MetricBase> metrics;
    this->internal_reader->ReportReadStageMetrics(metrics);

    QueryProfileCollector::Default().RegisterOperatorStageMetrics(
        QueryProfileCollector::MakeOperatorStageID(this->op_id, 1),
        std::move(metrics));

    QueryProfileCollector::Default().SubmitOperatorName(getOpId(), ToString());
    QueryProfileCollector::Default().SubmitOperatorStageTime(
        QueryProfileCollector::MakeOperatorStageID(getOpId(), 0),
        this->metrics.init_time);
    QueryProfileCollector::Default().SubmitOperatorStageTime(
        QueryProfileCollector::MakeOperatorStageID(getOpId(), 1),
        this->metrics.produce_time);
    QueryProfileCollector::Default().SubmitOperatorStageRowCounts(
        QueryProfileCollector::MakeOperatorStageID(getOpId(), 1),
        this->metrics.rows_read);
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
                "PhysicalReadIceberg(): invalid column index " +
                std::to_string(i) + " for schema with " +
                std::to_string(schema->num_fields()) + " fields");
        }
        out_column_names.emplace_back(schema->field(i)->name());
    }
    return out_column_names;
}

std::unique_ptr<IcebergParquetReader>
PhysicalReadIceberg::create_internal_reader() {
    // Pipeline buffers assume everything is nullable
    std::vector<bool> is_nullable(selected_columns.size(), true);

    // Insert join filter min/max stats into the duckdb table filters
    this->filter_exprs = join_filter_col_stats.insert_filters(
        std::move(this->filter_exprs), this->selected_columns);

    // We need to & the iceberg_filter with converted duckdb table filters
    // to apply the filters at the file level.
    PyObjectPtr duckdb_iceberg_filter =
        duckdbFilterSetToPyicebergFilter(*filter_exprs, arrow_schema);

    // Perform the python & to combine the filters
    // IcebergParquetReader takes ownership, so don't decref
    PyObject *py_iceberg_filter_and_duckdb_filter = PyObject_CallMethod(
        duckdb_iceberg_filter, "__and__", "O", iceberg_filter);
    if (!py_iceberg_filter_and_duckdb_filter) {
        throw std::runtime_error(
            "failed to combine iceberg filter with duckdb table filters");
    }

    // We need to convert the combined pyiceberg iceberg filter to an Arrow
    // filter format string so it can be applied at the row level in addition to
    // the file level.
    //  import bodo.io.iceberg.common
    PyObjectPtr iceberg_common_mod =
        PyImport_ImportModule("bodo.io.iceberg.common");
    if (PyErr_Occurred()) {
        throw std::runtime_error(
            "failed to import bodo.io.iceberg.common module");
    }

    PyObjectPtr convert_func = PyObject_GetAttrString(
        iceberg_common_mod,
        "pyiceberg_filter_to_pyarrow_format_str_and_scalars");
    if (!convert_func || !PyCallable_Check(convert_func)) {
        throw std::runtime_error(
            "failed to get convert_iceberg_filter_to_arrow function from "
            "bodo.io.iceberg.common module");
    }
    // call python function to convert the combined pyiceberg filter
    // bodo.io.iceberg.common.convert_iceberg_filter_to_arrow(iceberg_filter,iceberg_schema,
    // true)
    PyObjectPtr filter_f_str_and_scalars = PyObject_CallFunctionObjArgs(
        convert_func, py_iceberg_filter_and_duckdb_filter, iceberg_schema,
        Py_True, nullptr);
    if (!filter_f_str_and_scalars) {
        throw std::runtime_error(
            "failed to convert pyiceberg filter to arrow filter format string");
    }

    // The result is a tuple of (iceberg_filter_f_str, filter_scalars)
    // we need to unpack it.
    PyObject *iceberg_filter_f_str =
        PyTuple_GetItem(filter_f_str_and_scalars, 0);
    if (!iceberg_filter_f_str) {
        throw std::runtime_error(
            "failed to get iceberg filter format string from tuple");
    }
    PyObject *filter_scalars = PyTuple_GetItem(filter_f_str_and_scalars, 1);
    if (!filter_scalars) {
        throw std::runtime_error("failed to get filter scalars from tuple");
    }
    // Convert the PyObject filter format string to a std::string
    std::string iceberg_filter_str;
    if (iceberg_filter_f_str != Py_None) {
        iceberg_filter_str = PyUnicode_AsUTF8(iceberg_filter_f_str);
        if (PyErr_Occurred()) {
            throw std::runtime_error(
                "failed to convert iceberg filter format string to "
                "std::string");
        }
    } else {
        iceberg_filter_str = "";
    }

    // Incref filter scalars so we can pass it to the reader.
    // The reader will decref it when it's done.
    if (filter_scalars != Py_None) {
        Py_INCREF(filter_scalars);
    }

    // We're borrowing a reference to the catalog object, so we need to
    // increment the reference count since the reader steals it.
    Py_INCREF(catalog);
    auto reader = std::make_unique<IcebergParquetReader>(
        catalog, table_id.c_str(), true, total_rows_to_read,
        py_iceberg_filter_and_duckdb_filter, iceberg_filter_str, filter_scalars,
        selected_columns, is_nullable, arrow::py::wrap_schema(arrow_schema),
        get_streaming_batch_size(), this->getOpId(), snapshot_id);
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
                "PhysicalReadIceberg(): invalid column index " +
                std::to_string(i) + " for schema with " +
                std::to_string(arrow_schema->num_fields()) + " fields");
        }
        fields.push_back(arrow_schema->field(i));
    }
    return arrow::schema(fields, arrow_schema->metadata());
}
