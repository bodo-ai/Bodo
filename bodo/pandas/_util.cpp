#include "_util.h"

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
            return arrow::utf8();
        default:
            throw std::runtime_error(
                "duckdbTypeToArrow unsupported LogicalType conversion " +
                std::to_string(static_cast<int>(type.id())));
    }
}
