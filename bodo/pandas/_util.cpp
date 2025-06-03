#include "_util.h"
#include <arrow/python/pyarrow.h>

std::variant<int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t,
             uint64_t, bool, std::string, float, double, arrow::TimestampScalar>
extractValue(const duckdb::Value& value) {
    duckdb::LogicalTypeId type = value.type().id();
    switch (type) {
        case duckdb::LogicalTypeId::TINYINT:
            return value.GetValue<int8_t>();
        case duckdb::LogicalTypeId::SMALLINT:
            return value.GetValue<int16_t>();
        case duckdb::LogicalTypeId::INTEGER:
            return value.GetValue<int32_t>();
        case duckdb::LogicalTypeId::BIGINT:
            return value.GetValue<int64_t>();
        case duckdb::LogicalTypeId::UTINYINT:
            return value.GetValue<uint8_t>();
        case duckdb::LogicalTypeId::USMALLINT:
            return value.GetValue<uint16_t>();
        case duckdb::LogicalTypeId::UINTEGER:
            return value.GetValue<uint32_t>();
        case duckdb::LogicalTypeId::UBIGINT:
            return value.GetValue<uint64_t>();
        case duckdb::LogicalTypeId::FLOAT:
            return value.GetValue<float>();
        case duckdb::LogicalTypeId::DOUBLE:
            return value.GetValue<double>();
        case duckdb::LogicalTypeId::BOOLEAN:
            return value.GetValue<bool>();
        case duckdb::LogicalTypeId::VARCHAR:
            return value.GetValue<std::string>();
        case duckdb::LogicalTypeId::TIMESTAMP_NS: {
            // Define a timestamp type with nanosecond precision
            auto timestamp_type = arrow::timestamp(arrow::TimeUnit::NANO);
            duckdb::timestamp_ns_t extracted =
                value.GetValue<duckdb::timestamp_ns_t>();
            // Create a TimestampScalar with nanosecond value
            return arrow::TimestampScalar(extracted.value, timestamp_type);
        } break;
        default:
            throw std::runtime_error("extractValue unhandled type." +
                                     std::to_string(static_cast<int>(type)));
    }
}

std::string schemaColumnNamesToString(
    const std::shared_ptr<arrow::Schema> arrow_schema) {
    std::string ret = "";
    for (int i = 0; i < arrow_schema->num_fields(); i++) {
        ret += arrow_schema->field(i)->name();
        if (i != arrow_schema->num_fields() - 1) {
            ret += ", ";
        }
    }
    return ret;
}

void initInputColumnMapping(std::vector<int64_t>& col_inds,
                            std::vector<uint64_t>& keys, uint64_t ncols) {
    for (uint64_t i : keys) {
        col_inds.push_back(i);
    }
    for (uint64_t i = 0; i < ncols; i++) {
        if (std::find(keys.begin(), keys.end(), i) != keys.end()) {
            continue;
        }
        col_inds.push_back(i);
    }
}

std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t> getColRefMap(
    std::vector<duckdb::ColumnBinding> source_cols) {
    std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t> col_ref_map;
    for (size_t i = 0; i < source_cols.size(); i++) {
        duckdb::ColumnBinding& col = source_cols[i];
        col_ref_map[{col.table_index, col.column_index}] = i;
    }
    return col_ref_map;
}

std::shared_ptr<arrow::DataType> duckdbTypeToArrow(
    const duckdb::LogicalType& type) {
    switch (type.id()) {
        case duckdb::LogicalTypeId::TINYINT:
            return arrow::int8();
        case duckdb::LogicalTypeId::SMALLINT:
            return arrow::int16();
        case duckdb::LogicalTypeId::INTEGER:
            return arrow::int32();
        case duckdb::LogicalTypeId::BIGINT:
            return arrow::int64();
        case duckdb::LogicalTypeId::UTINYINT:
            return arrow::uint8();
        case duckdb::LogicalTypeId::USMALLINT:
            return arrow::uint16();
        case duckdb::LogicalTypeId::UINTEGER:
            return arrow::uint32();
        case duckdb::LogicalTypeId::UBIGINT:
            return arrow::uint64();
        case duckdb::LogicalTypeId::FLOAT:
            return arrow::float32();
        case duckdb::LogicalTypeId::DOUBLE:
            return arrow::float64();
        case duckdb::LogicalTypeId::BOOLEAN:
            return arrow::boolean();
        case duckdb::LogicalTypeId::VARCHAR:
            return arrow::large_utf8();
        default:
            throw std::runtime_error(
                "duckdbTypeToArrow unsupported LogicalType conversion " +
                std::to_string(static_cast<int>(type.id())));
    }
}

std::shared_ptr<table_info> runPythonScalarFunction(
    std::shared_ptr<table_info> input_batch,
    const std::shared_ptr<arrow::DataType>& result_type, PyObject* args) {
    // Call bodo.pandas.utils.run_apply_udf() to run the UDF

    // Import the bodo.pandas.utils module
    PyObject* bodo_module = PyImport_ImportModule("bodo.pandas.utils");
    if (!bodo_module) {
        PyErr_Print();
        throw std::runtime_error("Failed to import bodo.pandas.utils module");
    }

    // Call the run_apply_udf() with the table_info pointer, Arrow schema
    // and UDF function
    PyObject* pyarrow_schema =
        arrow::py::wrap_schema(input_batch->schema()->ToArrowSchema());
    PyObject* result_type_py(arrow::py::wrap_data_type(result_type));
    PyObject* result = PyObject_CallMethod(
        bodo_module, "run_func_on_table", "LOOO",
        reinterpret_cast<int64_t>(new table_info(*input_batch)), pyarrow_schema,
        result_type_py, args);
    if (!result) {
        PyErr_Print();
        Py_DECREF(bodo_module);
        throw std::runtime_error("Error calling run_apply_udf");
    }

    // Result should be a pointer to a C++ table_info
    if (!PyLong_Check(result)) {
        Py_DECREF(result);
        Py_DECREF(bodo_module);
        throw std::runtime_error("Expected an integer from run_apply_udf");
    }

    int64_t table_info_ptr = PyLong_AsLongLong(result);

    std::shared_ptr<table_info> out_batch(
        reinterpret_cast<table_info*>(table_info_ptr));

    Py_DECREF(bodo_module);
    Py_DECREF(result);

    return out_batch;
}
