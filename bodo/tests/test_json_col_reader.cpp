#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include <arrow/api.h>
#include <arrow/type.h>

#include "../io/json_col_parser.h"
#include "../libs/_memory.h"
#include "./test.hpp"

///@brief Construct an Arrow StringArray from a Vector of Optional Strings
static std::shared_ptr<arrow::StringArray> const_str_arr(
    std::vector<std::optional<std::string>> values) {
    auto builder = arrow::StringBuilder();
    // TODO: Replace with AppendValues and debug issue
    for (auto &value : values) {
        if (value.has_value()) {
            bodo::tests::check(builder.Append(value.value()).ok());
        } else {
            bodo::tests::check(builder.AppendNull().ok());
        }
    }

    return std::static_pointer_cast<arrow::StringArray>(
        builder.Finish().ValueOrDie());
}

/// @brief Helper Function to Insert Inner Vectors into a ListArray
template <class ValueBuilder, typename Value = ValueBuilder::value_type>
void construct_list_arr(
    arrow::LargeListBuilder &list_builder,
    std::shared_ptr<ValueBuilder> &value_builder,
    std::vector<std::optional<std::vector<std::optional<Value>>>> values) {
    for (auto &value : values) {
        if (value.has_value()) {
            bodo::tests::check(list_builder.Append().ok());
            for (auto &inner_value : value.value()) {
                if (inner_value.has_value()) {
                    bodo::tests::check(
                        value_builder->Append(inner_value.value()).ok());
                } else {
                    bodo::tests::check(value_builder->AppendNull().ok());
                }
            }
        } else {
            bodo::tests::check(list_builder.AppendNull().ok());
        }
    }
}

/// @brief Convert a Chrono Date to a Int32 Representing Days Since Epoch
int32_t date32(std::chrono::year_month_day ymd) {
    using namespace std::chrono;
    auto ymd_sys = sys_days(ymd);
    auto ymd_epoch = ymd_sys.time_since_epoch();
    auto days = duration_cast<std::chrono::days>(ymd_epoch);
    return days.count();
}

/// @brief Convert a Timestamp w/out Timezone String to an Int64
/// Representing Nanoseconds Since Epoch
int64_t ts(const std::string &tstr) {
    auto scalar = std::static_pointer_cast<arrow::TimestampScalar>(
        arrow::Scalar::Parse(arrow::timestamp(arrow::TimeUnit::NANO), tstr)
            .ValueOrDie());

    return scalar->value;
}

static bodo::tests::suite tests([] {
    bodo::tests::test("test_jsont_example", [] {
        const char *inbuf =
            "{ "
            "\"\\\"fo\\\"o\": \"Foo\","  // "\"fo\"o": "Foo"
            // Unlike JSON, Snowflake unicode is not escaped
            "\"1\" :  \"\\u2192\","
            "\"n\":1234,"
            "\"x\"  :  \t 12.34,"
            "\"overflow\"  :  \t 9999999999999999999999999999999999,"
            // Unlike JSON, Snowflake doesn't need escaped /
            "\"b\\/a\\/r\": ["
            "null,"
            "true,"
            "false,"
            "{"
            "\"x\":12.3"
            "},"
            "\n123,"
            "\"456\","
            "\"a\\\"b\\\"\","
            "\"a\\u0000b\","
            "\"a\\bb\","
            "\"a\\fb\","
            "\"a\\nb\","
            "\"a\\rb\","
            "\"a\\tb\","
            "\"\","
            "\"   \""
            "]"
            "}";

        using namespace std::string_literals;

        Tokenizer tokenizer;
        tokenizer.reset(inbuf);
        std::string out;

        bodo::tests::check(tokenizer.next() == Token::ObjectStart);

        bodo::tests::check(tokenizer.next() == Token::FieldName);
        tokenizer.stringValue(out);
        bodo::tests::check(out == "\"fo\"o");
        bodo::tests::check(tokenizer.next() == Token::String);
        tokenizer.stringValue(out);
        bodo::tests::check(out == "Foo");

        bodo::tests::check(tokenizer.next() == Token::FieldName);
        tokenizer.stringValue(out);
        bodo::tests::check(out == "1");
        bodo::tests::check(tokenizer.next() == Token::String);
        bodo::tests::check(tokenizer.value() == "\\u2192");
        bodo::tests::check_exception(
            [&] { tokenizer.stringValue(out); },
            "Found Unescaped Unicode In Snowflake JSON String: \\u2192");

        bodo::tests::check(tokenizer.next() == Token::FieldName);
        tokenizer.stringValue(out);
        bodo::tests::check(out == "n");
        bodo::tests::check(tokenizer.next() == Token::Integer);
        bodo::tests::check(tokenizer.intValue() == 1234);

        bodo::tests::check(tokenizer.next() == Token::FieldName);
        tokenizer.stringValue(out);
        bodo::tests::check(out == "x");
        bodo::tests::check(tokenizer.next() == Token::Float);
        bodo::tests::check(tokenizer.floatValue() == 12.34);

        bodo::tests::check(tokenizer.next() == Token::FieldName);
        tokenizer.stringValue(out);
        bodo::tests::check(out == "overflow");
        bodo::tests::check(tokenizer.next() == Token::Integer);
        bodo::tests::check(tokenizer.value() ==
                           "9999999999999999999999999999999999");
        bodo::tests::check_exception(
            [&] { tokenizer.intValue(); },
            "Failed to Parse Expected Integer in JSON: ");

        bodo::tests::check(tokenizer.next() == Token::FieldName);
        bodo::tests::check(tokenizer.value() == "b\\/a\\/r");
        bodo::tests::check_exception(
            [&] { tokenizer.stringValue(out); },
            "Invalid Escape Sequence In String: b\\/a\\/r");

        bodo::tests::check(tokenizer.next() == Token::ArrayStart);
        bodo::tests::check(tokenizer.next() == Token::Null);
        bodo::tests::check(tokenizer.next() == Token::True);
        bodo::tests::check(tokenizer.next() == Token::False);

        bodo::tests::check(tokenizer.next() == Token::ObjectStart);
        bodo::tests::check(tokenizer.next() == Token::FieldName);
        tokenizer.stringValue(out);
        bodo::tests::check(out == "x");
        bodo::tests::check(tokenizer.next() == Token::Float);
        bodo::tests::check(tokenizer.floatValue() == 12.3);
        bodo::tests::check(tokenizer.next() == Token::ObjectEnd);

        bodo::tests::check(tokenizer.next() == Token::Integer);
        bodo::tests::check(tokenizer.intValue() == 123);
        bodo::tests::check(tokenizer.next() == Token::String);
        tokenizer.stringValue(out);
        bodo::tests::check(out == "456");
        bodo::tests::check(tokenizer.next() == Token::String);
        tokenizer.stringValue(out);
        bodo::tests::check(out == "a\"b\"");
        bodo::tests::check(tokenizer.next() == Token::String);
        tokenizer.stringValue(out);
        bodo::tests::check(out == "a\0b"s);
        bodo::tests::check(tokenizer.next() == Token::String);
        tokenizer.stringValue(out);
        bodo::tests::check(out == "a\bb");
        bodo::tests::check(tokenizer.next() == Token::String);
        tokenizer.stringValue(out);
        bodo::tests::check(out == "a\fb");
        bodo::tests::check(tokenizer.next() == Token::String);
        tokenizer.stringValue(out);
        bodo::tests::check(out == "a\nb");
        bodo::tests::check(tokenizer.next() == Token::String);
        tokenizer.stringValue(out);
        bodo::tests::check(out == "a\rb");
        bodo::tests::check(tokenizer.next() == Token::String);
        tokenizer.stringValue(out);
        bodo::tests::check(out == "a\tb");
        bodo::tests::check(tokenizer.next() == Token::String);
        tokenizer.stringValue(out);
        bodo::tests::check(out == "");
        bodo::tests::check(tokenizer.next() == Token::String);
        tokenizer.stringValue(out);
        bodo::tests::check(out == "   ");

        bodo::tests::check(tokenizer.next() == Token::ArrayEnd);
        bodo::tests::check(tokenizer.next() == Token::ObjectEnd);
    });

    bodo::tests::test("test_unexpected_comma", [] {
        Tokenizer tokenizer;

        tokenizer.reset("[,1]");
        bodo::tests::check(tokenizer.next() == Token::ArrayStart);
        bodo::tests::check(tokenizer.next() == Token::Error);
        bodo::tests::check(tokenizer.error() == ErrorCode::UnexpectedComma);

        tokenizer.reset("[1,,]");
        bodo::tests::check(tokenizer.next() == Token::ArrayStart);
        bodo::tests::check(tokenizer.next() == Token::Integer);
        bodo::tests::check(tokenizer.next() == Token::Error);
        bodo::tests::check(tokenizer.error() == ErrorCode::UnexpectedComma);

        tokenizer.reset("{\"a\" : , 1}");
        bodo::tests::check(tokenizer.next() == Token::ObjectStart);
        bodo::tests::check(tokenizer.next() == Token::FieldName);
        bodo::tests::check(tokenizer.next() == Token::Error);
        bodo::tests::check(tokenizer.error() == ErrorCode::UnexpectedComma);

        tokenizer.reset("[1,]");
        bodo::tests::check(tokenizer.next() == Token::ArrayStart);
        bodo::tests::check(tokenizer.next() == Token::Integer);
        bodo::tests::check(tokenizer.next() == Token::Error);
        bodo::tests::check(tokenizer.error() ==
                           ErrorCode::UnexpectedTrailingComma);

        tokenizer.reset("{\"a\": \n\t{\"b\": 10,}}");
        bodo::tests::check(tokenizer.next() == Token::ObjectStart);
        bodo::tests::check(tokenizer.next() == Token::FieldName);
        bodo::tests::check(tokenizer.next() == Token::ObjectStart);
        bodo::tests::check(tokenizer.next() == Token::FieldName);
        bodo::tests::check(tokenizer.next() == Token::Integer);
        bodo::tests::check(tokenizer.next() == Token::Error);
        bodo::tests::check(tokenizer.error() ==
                           ErrorCode::UnexpectedTrailingComma);
    });

    bodo::tests::test("test_malformed_number_literal", [] {
        Tokenizer tokenizer;

        tokenizer.reset("+ ");
        bodo::tests::check(tokenizer.next() == Token::Error);
        bodo::tests::check(tokenizer.error() ==
                           ErrorCode::MalformedNumberLiteral);

        tokenizer.reset("-0+1");
        bodo::tests::check(tokenizer.next() == Token::Error);
        bodo::tests::check(tokenizer.error() ==
                           ErrorCode::MalformedNumberLiteral);

        tokenizer.reset("10e4");
        bodo::tests::check(tokenizer.next() == Token::Error);
        bodo::tests::check(tokenizer.error() ==
                           ErrorCode::MalformedNumberLiteral);

        tokenizer.reset(".11");
        bodo::tests::check(tokenizer.next() == Token::Error);
        bodo::tests::check(tokenizer.error() == ErrorCode::InvalidByte);
    });

    bodo::tests::test("test_premature_end", [] {
        Tokenizer tokenizer;

        tokenizer.reset("[");
        bodo::tests::check(tokenizer.next() == Token::ArrayStart);
        bodo::tests::check(tokenizer.next() == Token::End);

        tokenizer.reset("{\"a\":");
        bodo::tests::check(tokenizer.next() == Token::ObjectStart);
        bodo::tests::check(tokenizer.next() == Token::FieldName);
        bodo::tests::check(tokenizer.next() == Token::End);

        tokenizer.reset("undefin");
        bodo::tests::check(tokenizer.next() == Token::Error);
        bodo::tests::check(tokenizer.error() == ErrorCode::PrematureEndOfInput);

        // Unterminated String
        tokenizer.reset("\"test");
        bodo::tests::check(tokenizer.next() == Token::Error);
        bodo::tests::check(tokenizer.error() == ErrorCode::UnterminatedString);
    });

    bodo::tests::test("test_invalid_syntax", [] {
        Tokenizer tokenizer;

        // Invalid Byte from Unknown Keyword
        tokenizer.reset("[inf, -inf]");
        bodo::tests::check(tokenizer.next() == Token::ArrayStart);
        bodo::tests::check(tokenizer.next() == Token::Error);
        bodo::tests::check(tokenizer.error() == ErrorCode::InvalidByte);

        // Unable to Determine String Type
        tokenizer.reset("[\"tet\" -10]");
        bodo::tests::check(tokenizer.next() == Token::ArrayStart);
        bodo::tests::check(tokenizer.next() == Token::Error);
        bodo::tests::check(tokenizer.error() == ErrorCode::SyntaxError);
    });

    bodo::tests::test("test_empty_array", [] {
        auto arr = const_str_arr({
            // Null Outer Array
            std::nullopt,
            // Empty Array,
            "[]",
            // With Spacing
            "[\n\t\n]",
            // Only Null Inside
            "[undefined]",
        });
        auto list_arr =
            string_to_list_arr(arr, arrow::large_list(arrow::null()));

        auto value_builder = std::make_shared<arrow::NullBuilder>();
        auto list_builder = arrow::LargeListBuilder(
            bodo::BufferPool::DefaultPtr(),
            std::static_pointer_cast<arrow::ArrayBuilder>(value_builder));

        // Null Outer Array
        bodo::tests::check(list_builder.AppendNull().ok());
        // Empty Outer Array
        bodo::tests::check(list_builder.Append().ok());
        // Empty with Spacing
        bodo::tests::check(list_builder.Append().ok());
        // Only Null Inside
        bodo::tests::check(list_builder.Append().ok());
        bodo::tests::check(value_builder->AppendNull().ok());

        bodo::tests::check(
            list_arr->Equals(list_builder.Finish().ValueOrDie()));
    });

    bodo::tests::test("test_mixed_array_invalid", [] {
        // Tests that Array Doesn't Contain Multiple Inner Datatypes
        auto arr = const_str_arr({"[true, 10, \"hello\"]"});
        bodo::tests::check_exception(
            [&] {
                string_to_list_arr(arr, arrow::large_list(arrow::boolean()));
            },
            "Found an unexpected integer value");
    });

    bodo::tests::test("test_bool_array", [] {
        auto out_type = arrow::large_list(arrow::boolean());

        auto arr = const_str_arr({// Null Outer Array
                                  std::nullopt,
                                  // Simple Test Cases
                                  "[true]", "[false]", "[true, false]",
                                  // Only Null Inside
                                  "[undefined]",
                                  // With Spacing
                                  "[\n\ttrue,\n\tfalse,\n\tundefined\n]"});
        auto list_arr = string_to_list_arr(arr, out_type);

        auto value_builder = std::make_shared<arrow::BooleanBuilder>();
        auto list_builder = arrow::LargeListBuilder(
            bodo::BufferPool::DefaultPtr(),
            std::static_pointer_cast<arrow::ArrayBuilder>(value_builder),
            out_type);

        using vec = std::vector<std::optional<bool>>;
        construct_list_arr<arrow::BooleanBuilder>(
            list_builder, value_builder,
            {std::nullopt, vec{true}, vec{false}, vec{true, false},
             vec{std::nullopt}, vec{true, false, std::nullopt}});

        auto exp_list_arr = list_builder.Finish().ValueOrDie();
        bodo::tests::check(list_arr->Equals(exp_list_arr));
    });

    bodo::tests::test("test_string_array", [] {
        auto out_type = arrow::large_list(arrow::large_utf8());

        auto arr = const_str_arr({
            // Null Outer Array
            std::nullopt,
            // Simple Test Case
            "[\"id1\"]",
            // Multiple Elements
            "[\"id10\", \"id11\", \"id5\", \"id20\"]",
            // With Spacing and Nulls Inside
            "[\n\t\"why\",\n\t\"does\",\n\tundefined,\n\t\"snowflake\","
            "\n\t\"use\"\n]",
            // Only Null
            "[undefined]",
            // Multiline Text
            R"(["\r\n        test stuff \\t \n   \b   \f  for all\n        "])",
            // Unicode
            "[\"\041\", \"\x21\", \"⛄\", \"\u26c4\", \"❄é\"]",
            // Escape Characters
            R"(["'", "\"", "\t\n", "\\"])",
            "[\"test \\u0000 zero\"]",
            // String that represents other types
            "[\"true\", \"10\", \"2023-10-20\", \"hello\"]",
        });
        auto list_arr = string_to_list_arr(arr, out_type);

        auto value_builder = std::make_shared<arrow::LargeStringBuilder>();
        auto list_builder = arrow::LargeListBuilder(
            bodo::BufferPool::DefaultPtr(),
            std::static_pointer_cast<arrow::ArrayBuilder>(value_builder),
            out_type);

        using namespace std::string_literals;
        using vec = std::vector<std::optional<std::string>>;
        construct_list_arr<arrow::LargeStringBuilder, std::string>(
            list_builder, value_builder,
            {
                std::nullopt,
                vec{"id1"},
                vec{"id10", "id11", "id5", "id20"},
                vec{"why", "does", std::nullopt, "snowflake", "use"},
                vec{std::nullopt},
                vec{"\r\n        test stuff \\t \n   \b   \f  for all\n       "
                    " "},
                vec{"\041", "\x21", "⛄", "\u26c4", "❄é"},
                vec{"'", "\"", "\t\n", "\\"},
                // Shenanigans to get a \0 in the string
                vec{"test \0 zero"s},
                vec{"true", "10", "2023-10-20", "hello"},
            });

        auto exp_list_arr = list_builder.Finish().ValueOrDie();
        bodo::tests::check(list_arr->Equals(exp_list_arr));
    });

    bodo::tests::test("test_string_array_invalid_unescaped_unicode", [] {
        auto out_type = arrow::large_list(arrow::large_utf8());
        auto arr = const_str_arr({"[\"\\u26c4\"]"});
        bodo::tests::check_exception(
            [&] { string_to_list_arr(arr, out_type); },
            "Found Unescaped Unicode In Snowflake JSON String");
    });

    bodo::tests::test("test_int_array", [] {
        auto arr = const_str_arr({
            // Null Outer Array
            std::nullopt,
            // Positives and Negatives
            // Note: Snowflake does not include + in integers (like +1) but its
            // valid JSON
            "[+1, -1]",
            // With spacing and nulls inside
            "[\n\t1,\n\t2,\n\t3,\n\t4,\n\t5\n\t,undefined\n]",
            // Only null
            "[undefined]",
        });

        auto list_arr =
            string_to_list_arr(arr, arrow::large_list(arrow::int64()));

        auto value_builder = std::make_shared<arrow::Int64Builder>();
        auto list_builder = arrow::LargeListBuilder(
            bodo::BufferPool::DefaultPtr(),
            std::static_pointer_cast<arrow::ArrayBuilder>(value_builder));

        using vec = std::vector<std::optional<int64_t>>;
        construct_list_arr<arrow::Int64Builder>(
            list_builder, value_builder,
            {
                std::nullopt,
                vec{1, -1},
                vec{1, 2, 3, 4, 5, std::nullopt},
                vec{std::nullopt},
            });

        auto exp_list_arr = list_builder.Finish().ValueOrDie();
        bodo::tests::check(list_arr->Equals(exp_list_arr));
    });

    bodo::tests::test("test_int_array_overflow", [] {
        auto arr = const_str_arr({
            // Can't Fit in Int64: INT64_MAX + 1
            "[9223372036854775808]",
        });
        bodo::tests::check_exception(
            [&] { string_to_list_arr(arr, arrow::large_list(arrow::int64())); },
            "Failed to Parse Expected Integer in JSON: 9223372036854775808");

        arr = const_str_arr({
            // Similar for Negative: INT64_MIN - 1
            "[-9223372036854775809]",
        });
        bodo::tests::check_exception(
            [&] { string_to_list_arr(arr, arrow::large_list(arrow::int64())); },
            "Failed to Parse Expected Integer in JSON: -9223372036854775809");
    });

    bodo::tests::test("test_float_array", [] {
        // Note double is unique in that its used when Double and Int is
        // Available
        auto arr = const_str_arr({
            // Null Outer Array
            std::nullopt,
            // Simple Test Case (Integer, Float, Scientific Notation)
            // Again, haven't seen Snowflake use exponential notation but its
            // valid JSON
            "[1, -1, 1.0, -1.0, 5.7, -2.4, 1.e2, -1.2E-2]",
            // With Spacing and Nulls Inside
            "[\n\t1.0,\n\t2.0,\n\t3.0,\n\t4.0,\n\t5.0\n\t,undefined\n]",
            // Overflow Integers (INT64_MAX + 1 and INT64_MIN - 1)
            "[9223372036854775808, -9223372036854775809.0]",
            // Over precision of 38 (Snowflake's max precision)
            "[1.123456789012345678901234567890123456789]",
            // Special Floating-Point Cases
            "[Infinity, -Infinity]",
            // Only Null
            "[undefined]",
        });
        auto list_arr =
            string_to_list_arr(arr, arrow::large_list(arrow::float64()));

        auto value_builder = std::make_shared<arrow::DoubleBuilder>();
        auto list_builder = arrow::LargeListBuilder(
            bodo::BufferPool::DefaultPtr(),
            std::static_pointer_cast<arrow::ArrayBuilder>(value_builder));

        using vec = std::vector<std::optional<double>>;
        construct_list_arr<arrow::DoubleBuilder>(
            list_builder, value_builder,
            {
                std::nullopt,
                vec{1.0, -1.0, 1.0, -1.0, 5.7, -2.4, 100.0, -0.012},
                vec{1.0, 2.0, 3.0, 4.0, 5.0, std::nullopt},
                vec{9223372036854775808.0, -9223372036854775809.0},
                vec{1.123456789012345678901234567890123456789},
                vec{std::numeric_limits<double>::infinity(),
                    -std::numeric_limits<double>::infinity()},
                vec{std::nullopt},
            });

        auto exp_list_arr = list_builder.Finish().ValueOrDie();
        bodo::tests::check(list_arr->Equals(exp_list_arr));
    });

    bodo::tests::test("test_float_array_nan", [] {
        auto arr = const_str_arr({"[NaN]"});
        auto list_arr =
            string_to_list_arr(arr, arrow::large_list(arrow::float64()));

        bodo::tests::check(list_arr->IsValid(0));
        auto list_inner = std::static_pointer_cast<arrow::LargeListScalar>(
                              list_arr->GetScalar(0).ValueOrDie())
                              ->value;
        bodo::tests::check(list_inner->IsValid(0));
        auto list_content = std::static_pointer_cast<arrow::DoubleScalar>(
                                list_inner->GetScalar(0).ValueOrDie())
                                ->value;
        bodo::tests::check(std::isnan(list_content));
    });

    bodo::tests::test("test_date_array", [] {
        auto arr = const_str_arr({
            // Outer Null Array
            std::nullopt,
            // Simple Test Case Including Min
            "[\"2023-10-24\", \"1970-01-01\", \"2024-01-01\"]",
            // With Spacing and Nulls Inside
            "[\n\t\"2008-10-09\",\n\tundefined\n]",
            // Only Null
            "[undefined]",
        });
        auto list_arr =
            string_to_list_arr(arr, arrow::large_list(arrow::date32()));

        auto value_builder = std::make_shared<arrow::Date32Builder>();
        auto list_builder = arrow::LargeListBuilder(
            bodo::BufferPool::DefaultPtr(),
            std::static_pointer_cast<arrow::ArrayBuilder>(value_builder));

        using namespace std::literals;
        using namespace std::chrono;

        using vec = std::vector<std::optional<int32_t>>;
        construct_list_arr<arrow::Date32Builder>(
            list_builder, value_builder,
            {
                std::nullopt,
                vec{date32({2023y, October, 24d}), date32({1970y, January, 1d}),
                    date32({2024y, January, 1d})},
                vec{date32({2008y, October, 9d}), std::nullopt},
                vec{std::nullopt},
            });

        auto exp_list_arr = list_builder.Finish().ValueOrDie();
        bodo::tests::check(list_arr->Equals(exp_list_arr));
    });

    bodo::tests::test("test_timestamp_ntz_array", [] {
        auto value_type = arrow::timestamp(arrow::TimeUnit::NANO);

        auto arr = const_str_arr({
            // Outer Null Array
            std::nullopt,
            // Simple Test Case Including Min
            "[\"2023-10-24 12:34:56.789\", \"1970-01-01 00:00:00.000\", "
            "\"2023-10-04 00:51:45.758\"]",
            // With Spacing and Nulls Inside
            "[\n\t\"2008-10-09 05:10:32.142\",\n\tundefined\n]",
            // Different Precisions
            "[\"2023-10-24 12:34:56\", \"2023-10-24 12:34:56.789123\", "
            "\"2023-10-24 12:34:56.789123456\"]",
            // Only Null
            "[undefined]",
        });
        auto list_arr = string_to_list_arr(arr, arrow::large_list(value_type));

        auto value_builder = std::make_shared<arrow::TimestampBuilder>(
            value_type, bodo::BufferPool::DefaultPtr());
        auto list_builder = arrow::LargeListBuilder(
            bodo::BufferPool::DefaultPtr(),
            std::static_pointer_cast<arrow::ArrayBuilder>(value_builder));

        using vec = std::vector<std::optional<int64_t>>;
        construct_list_arr<arrow::TimestampBuilder>(
            list_builder, value_builder,
            {
                std::nullopt,
                vec{ts("2023-10-24 12:34:56.789"),
                    ts("1970-01-01 00:00:00.000"),
                    ts("2023-10-04 00:51:45.758")},
                vec{ts("2008-10-09 05:10:32.142"), std::nullopt},
                vec{ts("2023-10-24 12:34:56"), ts("2023-10-24 12:34:56.789123"),
                    ts("2023-10-24 12:34:56.789123456")},
                vec{std::nullopt},
            });

        auto exp_list_arr = list_builder.Finish().ValueOrDie();
        bodo::tests::check(list_arr->Equals(exp_list_arr));
    });

    // ------------------------------ Map Tests ------------------------------
    // Note we will assume that the map code is
    // similar to array and mainly test differences, especially around the
    // fields

    bodo::tests::test("test_empty_map", [] {
        auto in_type = arrow::map(arrow::large_utf8(), arrow::null());
        auto arr = const_str_arr({
            // Null Outer Map
            std::nullopt,
            // Empty Map,
            "{}",
            // With Spacing
            "{\n\t\n}",
            // Only Null Inside
            "{\"null\": null}",
        });
        auto map_arr = string_to_map_arr(arr, in_type);

        auto key_builder = std::make_shared<arrow::LargeStringBuilder>();
        auto value_builder = std::make_shared<arrow::NullBuilder>();
        auto map_builder =
            arrow::MapBuilder(bodo::BufferPool::DefaultPtr(), key_builder,
                              value_builder, in_type);

        // Null Outer Array
        bodo::tests::check(map_builder.AppendNull().ok());
        // Empty Outer Array
        bodo::tests::check(map_builder.Append().ok());
        // Empty with Spacing
        bodo::tests::check(map_builder.Append().ok());
        // Only Null Inside
        bodo::tests::check(map_builder.Append().ok());
        bodo::tests::check(key_builder->Append("null").ok());
        bodo::tests::check(value_builder->AppendNull().ok());

        auto exp_map_arr = map_builder.Finish().ValueOrDie();
        bodo::tests::check(map_arr->Equals(exp_map_arr));
    });

    bodo::tests::test("test_mixed_map_invalid", [] {
        // Tests that Map Doesn't Contain Multiple Inner Datatypes
        auto arr1 = const_str_arr(
            {"{\"test1\"  : true, \"test10\": 10, \"test24\": \"hello\"}"});
        bodo::tests::check_exception(
            [&] {
                string_to_map_arr(
                    arr1, arrow::map(arrow::large_utf8(), arrow::boolean()));
            },
            "Found an unexpected integer value");
    });

    bodo::tests::test("test_invalid_map_key", [] {
        // Tests that Map Doesn't Contain a non-string Key
        // Should be invalid in Snowflake
        auto arr1 = const_str_arr({"{10: true}"});
        bodo::tests::check_exception(
            [&] {
                string_to_map_arr(
                    arr1, arrow::map(arrow::large_utf8(), arrow::boolean()));
            },
            "Found an integer when parsing a map column row, but expected a "
            "field name");
    });

    bodo::tests::test("test_utf8_string_keys_map", [] {
        auto out_type = arrow::map(arrow::large_utf8(), arrow::large_utf8());

        auto arr = const_str_arr({
            // Simple Test Case
            "{\"id10\": \"id11\", \"id5\": \"id20\"}",
            // Unicode
            "{\"\041\": \"\", \"\x21\": \"\", \"\u26c4\": \"\", \"❄é\": \"\"}",
            // Escape Characters
            R"({"\"": "", "\t\n": "", "\\": ""})",
        });
        auto map_arr = string_to_map_arr(arr, out_type);

        auto key_builder = std::make_shared<arrow::LargeStringBuilder>();
        auto value_builder = std::make_shared<arrow::LargeStringBuilder>();
        auto map_builder =
            arrow::MapBuilder(bodo::BufferPool::DefaultPtr(), key_builder,
                              value_builder, out_type);

        bodo::tests::check(map_builder.Append().ok());
        bodo::tests::check(key_builder->Append("id10").ok());
        bodo::tests::check(value_builder->Append("id11").ok());
        bodo::tests::check(key_builder->Append("id5").ok());
        bodo::tests::check(value_builder->Append("id20").ok());

        bodo::tests::check(map_builder.Append().ok());
        bodo::tests::check(key_builder->Append("\041").ok());
        bodo::tests::check(value_builder->Append("").ok());
        bodo::tests::check(key_builder->Append("\x21").ok());
        bodo::tests::check(value_builder->Append("").ok());
        bodo::tests::check(key_builder->Append("\u26c4").ok());
        bodo::tests::check(value_builder->Append("").ok());
        bodo::tests::check(key_builder->Append("❄é").ok());
        bodo::tests::check(value_builder->Append("").ok());

        bodo::tests::check(map_builder.Append().ok());
        bodo::tests::check(key_builder->Append("\"").ok());
        bodo::tests::check(value_builder->Append("").ok());
        bodo::tests::check(key_builder->Append("\t\n").ok());
        bodo::tests::check(value_builder->Append("").ok());
        bodo::tests::check(key_builder->Append("\\").ok());
        bodo::tests::check(value_builder->Append("").ok());

        auto exp_map_arr = map_builder.Finish().ValueOrDie();
        bodo::tests::check(map_arr->Equals(exp_map_arr));
    });
});
