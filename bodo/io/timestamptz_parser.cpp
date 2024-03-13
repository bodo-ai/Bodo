#include "timestamptz_parser.h"

#include <arrow/builder.h>
#include <arrow/type.h>
#include <iostream>

#include "../libs/_bodo_common.h"

int Parser::consume_char(char c) {
    if (pos >= ts_str.size()) {
        return -1;
    }

    if (ts_str[pos] == c) {
        pos++;
        return 0;
    }

    return -1;
}

int Parser::parse_digit() {
    if (pos >= ts_str.size()) {
        return -1;
    }

    if (isdigit(ts_str[pos])) {
        int c = ts_str[pos] - '0';
        pos++;
        return c;
    }

    return -1;
}

int Parser::parse_year() {
    // Parse YYYY
    int year = 0;
    for (int i = 0; i < 4; i++) {
        int digit = parse_digit();
        if (digit == -1) {
            return -1;
        }
        year = year * 10 + digit;
    }

    return year;
}

int Parser::parse_2_digit() {
    int digit = parse_digit();
    if (digit == -1) {
        return -1;
    }

    int digit2 = parse_digit();
    if (digit2 == -1) {
        return -1;
    }

    return digit * 10 + digit2;
}

int Parser::parse_month() {
    // Parse MM
    int month = parse_2_digit();
    if (month < 1 || month > 12) {
        return -1;
    }

    return month;
}

int Parser::parse_day() {
    // Parse DD
    int day = parse_2_digit();
    if (day < 1 || day > 31) {
        return -1;
    }
    return day;
}

int Parser::parse_hour() {
    // Parse HH
    int hour = parse_2_digit();
    if (hour < 0 || hour > 23) {
        return -1;
    }
    return hour;
}

int Parser::parse_minute_second() {
    // Parse MM or SS
    int minute = parse_2_digit();
    if (minute < 0 || minute > 59) {
        return -1;
    }
    return minute;
}

int Parser::parse_nanoseconds() {
    // Parse SSSSSSSSS
    int nanoseconds = 0;
    for (int i = 0; i < 9; i++) {
        int digit = parse_digit();
        if (digit == -1) {
            return -1;
        }
        nanoseconds = nanoseconds * 10 + digit;
    }

    return nanoseconds;
}

// TODO(aneesh): The error checking might not be needed in practice - we could
// optimize it out unless it's a debug build.
#define CHECK(expr...)                                                       \
    ({                                                                       \
        auto v = expr;                                                       \
        if (v == -1) {                                                       \
            std::string msg = "Invalid timestamp: Failed executing: " #expr; \
            msg += "\n\tat position " + std::to_string(pos);                 \
            msg += "\n\tin string: " + std::string(ts_str);                  \
            throw std::runtime_error(msg);                                   \
        }                                                                    \
        v;                                                                   \
    })

#define CHECK_CHAR(c) CHECK(consume_char(c))

std::pair<int64_t, int16_t> Parser::parse_timestamptz() {
    // The only supported formats are:
    // "YYYY-MM-DD HH:MM:SS.S... [+-]TZH:TZM"
    // "YYYY-MM-DD HH:MM:SS.S... 'Z'" (for the 0 offset)
    CHECK_CHAR('"');

    // Parse the timestamp
    auto year = CHECK(parse_year());
    CHECK_CHAR('-');
    auto month = CHECK(parse_month());
    CHECK_CHAR('-');
    auto day = CHECK(parse_day());
    CHECK_CHAR(' ');
    auto hour = CHECK(parse_hour());
    CHECK_CHAR(':');
    auto minute = CHECK(parse_minute_second());
    CHECK_CHAR(':');
    auto second = CHECK(parse_minute_second());
    auto nanoseconds = 0;
    if (consume_char('.') == 0) {
        nanoseconds = CHECK(parse_nanoseconds());
        if (nanoseconds == -1) {
            throw std::runtime_error("Invalid nanoseconds");
        }
    }
    CHECK_CHAR(' ');

    // Parse the offset
    // If the offset is 'Z', then it's 0
    int16_t offset_sign = 0;
    int16_t offset_h = 0;
    int16_t offset_m = 0;
    if (pos >= ts_str.size()) {
        throw std::runtime_error("Invalid offset");
    }
    if (ts_str[pos] == 'Z') {
        pos++;
    } else {
        if (consume_char('-') == 0) {
            offset_sign = -1;
        } else if (consume_char('+') == 0) {
            offset_sign = 1;
        } else {
            throw std::runtime_error("Invalid offset");
        }

        offset_h = CHECK(parse_2_digit());
        CHECK_CHAR(':');
        offset_m = CHECK(parse_2_digit());
    }
    CHECK_CHAR('"');
    assert(pos == ts_str.size());
    int16_t offset = offset_sign * (offset_h * 60 + offset_m);

    // Convert the timestamp to UTC by subtracting the offset from the
    // timestamp, and normalize
    struct tm t = {0};
    t.tm_year = year - 1900;
    t.tm_mon = month - 1;
    t.tm_mday = day;
    // This is where we need to subtract the offset - we want to store the UTC
    // time, not the local time.
    t.tm_hour = hour - offset_sign * offset_h;
    t.tm_min = minute - offset_sign * offset_m;
    t.tm_sec = second;

    // Convert calendar time to epoch time. Note that timegm doesn't depend on
    // the local timezone, and interprets the input as UTC.
    time_t epoch_time = timegm(&t);
    // Convert epoch_time to nanoseconds
    int64_t ts = epoch_time * 1000000000 + nanoseconds;

    return std::make_pair(ts, offset);
}

#undef CHECK_CHAR
#undef CHECK

/**
 * @brief Parses a TimestampTZ string into a Struct Array
 *
 * @param struct_builder ArrayBuilder for keys to insert output into
 * @param ttz_str The input TimestampTZ string
 */
void parse_to_struct(arrow::StructBuilder& struct_builder,
                     std::string_view ttz_str) {
    auto ts_builder = std::static_pointer_cast<arrow::Int64Builder>(
        struct_builder.child_builder(0));
    auto offset_builder = std::static_pointer_cast<arrow::Int16Builder>(
        struct_builder.child_builder(1));

    Parser parser = {ttz_str};
    auto [ts, offset] = parser.parse_timestamptz();

    if (!ts_builder->Append(ts).ok()) {
        throw std::runtime_error("Failure while appending timestamp");
    }
    if (!offset_builder->Append(offset).ok()) {
        throw std::runtime_error("Failure while appending offset");
    }
}

std::shared_ptr<arrow::Array> string_to_timestamptz_arr(
    std::shared_ptr<arrow::StringArray> in_arr,
    std::shared_ptr<arrow::DataType> in_type) {
    // Cast in_type to ExtensionType
    auto ext_type = std::dynamic_pointer_cast<arrow::ExtensionType>(in_type);
    auto struct_type = ext_type->storage_type();

    std::shared_ptr<arrow::ArrayBuilder> builder =
        arrow::MakeBuilder(struct_type, bodo::BufferPool::DefaultPtr())
            .ValueOrDie();
    auto struct_builder =
        std::static_pointer_cast<arrow::StructBuilder>(builder);

    // Iterate over StringArray, Parse TimestampTZ, Validate, and Insert
    for (int64_t i = 0; i < in_arr->length(); i++) {
        if (in_arr->IsNull(i)) {
            auto status = struct_builder->AppendNull();
            continue;
        }

        auto ttz_str = in_arr->GetView(i);
        auto status = struct_builder->Append();
        if (!status.ok()) {
            throw std::runtime_error("Failure while appending map: " +
                                     status.ToString());
        }

        parse_to_struct(*struct_builder, ttz_str);
    }

    auto struct_arr = struct_builder->Finish().ValueOrDie();
    return arrow::ExtensionType::WrapArray(ext_type, struct_arr);
}
