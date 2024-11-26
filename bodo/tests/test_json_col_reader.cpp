#include <chrono>
#include <cmath>
#include <ctime>
#include <optional>
#include <string>

#include <arrow/api.h>
#include <arrow/array/builder_base.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_nested.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/scalar.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>

#include "../io/json_col_parser.h"
#include "../libs/_memory.h"
#include "./test.hpp"

using namespace std::literals;
using namespace std::chrono;

template <typename V>
using ordered_map = std::vector<std::pair<std::string, std::optional<V>>>;

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

template <typename T>
void insert_elem(arrow::ArrayBuilder *child_builder, T value) {
    auto type_id = typeid(T).name();
    throw std::runtime_error("Unsupported Type: " + std::string(type_id));
}

template <typename V>
void insert_elem(arrow::ArrayBuilder *child_builder,
                 std::vector<std::pair<std::string, std::optional<V>>> value);

template <typename... Tp>
void insert_elem(arrow::ArrayBuilder *child_builder,
                 std::tuple<std::optional<Tp>...> value);

template <typename V>
void insert_elem(arrow::ArrayBuilder *child_builder,
                 std::vector<std::optional<V>> value) {
    auto builder = static_cast<arrow::LargeListBuilder *>(child_builder);
    bodo::tests::check(builder->Append().ok());
    auto value_builder = builder->value_builder();

    for (auto &inner_value : value) {
        if (inner_value.has_value()) {
            insert_elem(value_builder, inner_value.value());
        } else {
            bodo::tests::check(value_builder->AppendNull().ok());
        }
    }
}

template <typename V>
void insert_elem(arrow::ArrayBuilder *child_builder, ordered_map<V> value) {
    auto builder = static_cast<arrow::MapBuilder *>(child_builder);
    auto key_builder =
        static_cast<arrow::LargeStringBuilder *>(builder->key_builder());
    auto value_builder = builder->item_builder();

    bodo::tests::check(builder->Append().ok());
    for (auto &elem : value) {
        std::cout << "key: " << elem.first << std::endl;
        bodo::tests::check(key_builder->Append(elem.first).ok());
        if (elem.second.has_value()) {
            insert_elem(value_builder, elem.second.value());
        } else {
            bodo::tests::check(value_builder->AppendNull().ok());
        }
    }
}

template <typename T>
void insert_struct_field(arrow::ArrayBuilder *child_builder,
                         std::optional<T> value) {
    if (value.has_value()) {
        insert_elem(child_builder, value.value());
    } else {
        bodo::tests::check(child_builder->AppendNull().ok());
    }
}

template <typename... Tp>
void insert_elem(arrow::ArrayBuilder *child_builder,
                 std::tuple<std::optional<Tp>...> value) {
    auto struct_builder = static_cast<arrow::StructBuilder *>(child_builder);
    bodo::tests::check(struct_builder->Append().ok());
    std::apply(
        [&struct_builder](auto &&...args) {
            std::size_t n{0};
            (...,
             (insert_struct_field(struct_builder->field_builder(n++), args)));
        },
        value);
}

template <>
void insert_elem(arrow::ArrayBuilder *child_builder,
                 std::shared_ptr<arrow::TimestampScalar> value) {
    auto builder = static_cast<arrow::TimestampBuilder *>(child_builder);
    bodo::tests::check(builder->Append(value->value).ok());
}

#define _INSERT_ELEM(BUILDER, VALUE)                                    \
    template <>                                                         \
    void insert_elem(arrow::ArrayBuilder *child_builder, VALUE value) { \
        auto builder = static_cast<BUILDER *>(child_builder);           \
        bodo::tests::check(builder->Append(value).ok());                \
    }

_INSERT_ELEM(arrow::Int64Builder, int64_t)
_INSERT_ELEM(arrow::DoubleBuilder, double)
_INSERT_ELEM(arrow::BooleanBuilder, bool)
_INSERT_ELEM(arrow::LargeStringBuilder, std::string)
_INSERT_ELEM(arrow::Date32Builder, int32_t)
// TODO: Avoid redefinition with int64_t
// _INSERT_ELEM(arrow::TimestampBuilder, int64_t);

#undef _INSERT_ELEM

/// @brief Helper Function to Insert Inner Vectors into a ListArray
template <typename V>
void construct_list_arr(
    arrow::LargeListBuilder &list_builder,
    std::vector<std::optional<std::vector<std::optional<V>>>> values) {
    for (auto &value : values) {
        if (value.has_value()) {
            std::vector<std::optional<V>> inner = value.value();
            insert_elem(&list_builder, inner);
        } else {
            bodo::tests::check(list_builder.AppendNull().ok());
        }
    }
}

/// @brief Helper Function to Insert Inner Vectors into a ListArray
template <typename V>
void construct_map_arr(arrow::MapBuilder &map_builder,
                       std::vector<std::optional<ordered_map<V>>> values) {
    for (auto &value : values) {
        if (value.has_value()) {
            ordered_map<V> inner = value.value();
            insert_elem(&map_builder, inner);
        } else {
            bodo::tests::check(map_builder.AppendNull().ok());
        }
    }
}

template <typename... Tp>
void construct_struct_arr(
    std::shared_ptr<arrow::StructBuilder> &struct_builder,
    std::vector<std::optional<std::tuple<std::optional<Tp>...>>> values) {
    for (auto &value : values) {
        if (value.has_value()) {
            bodo::tests::check(struct_builder->Append().ok());
            std::tuple<std::optional<Tp>...> inner = value.value();
            std::apply(
                [&struct_builder](auto &&...args) {
                    std::size_t n{0};
                    (..., (insert_struct_field(
                              struct_builder->field_builder(n++), args)));
                },
                inner);
        } else {
            bodo::tests::check(struct_builder->AppendNull().ok());
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
std::shared_ptr<arrow::TimestampScalar> ts(const std::string &tstr) {
    auto scalar = std::static_pointer_cast<arrow::TimestampScalar>(
        arrow::Scalar::Parse(arrow::timestamp(arrow::TimeUnit::NANO), tstr)
            .ValueOrDie());

    return scalar;
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

    // ----------------------------- Array Tests -----------------------------

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
        construct_list_arr<bool>(
            list_builder,
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
            R"(["id10", "id11", "id5", "id20"])",
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
            R"(["test \u0000 zero"])",
            // String that represents other types
            R"(["true", "10", "2023-10-20", "hello"])",
        });
        auto list_arr = string_to_list_arr(arr, out_type);

        auto value_builder = std::make_shared<arrow::LargeStringBuilder>();
        auto list_builder = arrow::LargeListBuilder(
            bodo::BufferPool::DefaultPtr(),
            std::static_pointer_cast<arrow::ArrayBuilder>(value_builder),
            out_type);

        using namespace std::string_literals;
        using vec = std::vector<std::optional<std::string>>;
        construct_list_arr<std::string>(
            list_builder,
            {
                std::nullopt,
                vec{"id1"},
                vec{"id10", "id11", "id5", "id20"},
                vec{"why", "does", std::nullopt, "snowflake", "use"},
                vec{std::nullopt},
                vec{"\r\n        test stuff \\t \n   \b   \f  for all\n       "
                    " "},
                vec{R"(!)", R"(!)", "⛄", "\u26c4", "❄é"},
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
        auto arr = const_str_arr({R"(["\u26c4"])"});
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
        construct_list_arr<int64_t>(list_builder,
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
        construct_list_arr<double>(
            list_builder,
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
            R"(["2023-10-24", "1970-01-01", "2024-01-01"])",
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

        using vec = std::vector<std::optional<int32_t>>;
        construct_list_arr<int32_t>(
            list_builder,
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

        using vec =
            std::vector<std::optional<std::shared_ptr<arrow::TimestampScalar>>>;
        construct_list_arr<std::shared_ptr<arrow::TimestampScalar>>(
            list_builder,
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

    bodo::tests::test("test_nested_array_in_array", [] {
        auto arr = const_str_arr({
            // Outer Null Array
            std::nullopt,
            // Only Null
            "[undefined]",
            // Empty Lists
            "[[], undefined, [undefined]]",
            // Simple Test Case
            R"(
[
    [
        null
    ],
    [
        1,
        -1,
        1.0,
        -1.0,
        5.7,
        -2.4
    ],
    [
        1.e2,
        -1.2E-2
    ],
    [],
    undefined
]
            )",
        });

        auto in_type = arrow::large_list(arrow::large_list(arrow::float64()));
        auto list_arr = string_to_list_arr(arr, in_type);

        auto list_builder = std::static_pointer_cast<arrow::LargeListBuilder>(
            std::shared_ptr(arrow::MakeBuilder(in_type).ValueOrDie()));

        using in = std::vector<std::optional<double>>;
        using vec = std::vector<std::optional<in>>;
        construct_list_arr<in>(
            *list_builder,
            {
                std::nullopt,
                vec{std::nullopt},
                vec{in{}, std::nullopt, in{std::nullopt}},
                vec{in{std::nullopt}, in{1.0, -1.0, 1.0, -1.0, 5.7, -2.4},
                    in{100.0, -0.012}, in{}, std::nullopt},
            });

        auto exp_list_arr = list_builder->Finish().ValueOrDie();
        bodo::tests::check(list_arr->Equals(exp_list_arr));
    });

    bodo::tests::test("test_nested_map_in_array", [] {
        auto arr = const_str_arr({
            // Outer Null Array
            std::nullopt,
            // Only Null
            "[undefined]",
            // Empty Maps
            "[{}, undefined, {}]",
            // Simple Test Case
            R"(
[
    {
        "grade": undefined
    },
    {
        "owner": 20,
        "sound": null
    },
    {
        "test": 50111,
        "once": -17,
        "give": 467
    }
]
        )",
        });

        auto in_type =
            arrow::large_list(arrow::map(arrow::large_utf8(), arrow::int64()));
        auto list_arr = string_to_list_arr(arr, in_type);
        auto list_builder = std::static_pointer_cast<arrow::LargeListBuilder>(
            std::shared_ptr(arrow::MakeBuilder(in_type).ValueOrDie()));

        using in = ordered_map<int64_t>;
        using vec = std::vector<std::optional<in>>;
        construct_list_arr<in>(
            *list_builder,
            {
                std::nullopt,
                vec{std::nullopt},
                vec{in{}, std::nullopt, in{}},
                vec{
                    in{{"grade", std::nullopt}},
                    in{{"owner", 20}, {"sound", std::nullopt}},
                    in{{"test", 50111}, {"once", -17}, {"give", 467}},
                },
            });

        auto exp_list_arr = list_builder->Finish().ValueOrDie();
        bodo::tests::check(list_arr->Equals(exp_list_arr));
    });

    bodo::tests::test("test_nested_struct_in_array", [] {
        auto arr = const_str_arr({
            // Outer Null Array
            std::nullopt,
            // Only Null
            "[undefined]",
            // Empty Struct, All Null Fields
            R"([{}, undefined, {"a": null, "b": null, "c": null, "d": null}])",
            // Simple Test Case
            R"(
[
    {
        "a": "2023-10-24",
        "b": 105,
        "c": [true, false, undefined],
        "d": {
            "test": 50111,
            "once": undefined,
            "give": 467
        }
    },
    {
        "a": undefined,
        "b": -1467,
        "c": [],
        "d": {
            "alpha": 1.578e4,
            "beta": -200.e-2,
            "omega": Infinity
        }
    }
]
        )",
        });

        auto value_type = arrow::struct_(
            {arrow::field("a", arrow::date32()),
             arrow::field("b", arrow::int64()),
             arrow::field("c", arrow::large_list(arrow::boolean())),
             arrow::field("d",
                          arrow::map(arrow::large_utf8(), arrow::float64()))});
        auto list_arr = string_to_list_arr(arr, arrow::large_list(value_type));

        auto struct_builder = std::static_pointer_cast<arrow::StructBuilder>(
            std::shared_ptr(arrow::MakeBuilder(value_type).ValueOrDie()));
        auto list_builder = arrow::LargeListBuilder(
            bodo::BufferPool::DefaultPtr(), struct_builder);

        using invec = std::vector<std::optional<bool>>;
        using inmap = ordered_map<double>;
        using in = std::tuple<std::optional<int32_t>, std::optional<int64_t>,
                              std::optional<invec>, std::optional<inmap>>;
        using vec = std::vector<std::optional<in>>;
        construct_list_arr<in>(
            list_builder,
            {std::nullopt, vec{std::nullopt},
             vec{
                 in{std::nullopt, std::nullopt, std::nullopt, std::nullopt},
                 std::nullopt,
                 in{std::nullopt, std::nullopt, std::nullopt, std::nullopt},
             },
             vec{
                 in{date32({2023y, October, 24d}), 105,
                    invec{true, false, std::nullopt},
                    inmap{{"test", 50111},
                          {"once", std::nullopt},
                          {"give", 467}}},
                 in{std::nullopt, -1467, invec{},
                    inmap{{"alpha", 1.578e4},
                          {"beta", -200e-2},
                          {"omega", std::numeric_limits<double>::infinity()}}},
             }});

        auto exp_list_arr = list_builder.Finish().ValueOrDie();
        bodo::tests::check(list_arr->Equals(exp_list_arr));
    });

    // ------------------------------ Map Tests ------------------------------
    // Note we will assume that the map code is similar to array and mainly
    // test differences, especially around the fields

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
            {R"({"test1"  : true, "test10": 10, "test24": "hello"})"});
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
            R"({"id10": "id11", "id5": "id20"})",
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

        using map = ordered_map<std::string>;
        construct_map_arr<std::string>(
            map_builder,
            {
                // Simple Test Case
                map{{"id10", "id11"}, {"id5", "id20"}},
                // Unicode
                map{{R"(!)", ""}, {R"(!)", ""}, {"\u26c4", ""}, {"❄é", ""}},
                // Escape Characters
                map{{"\"", ""}, {"\t\n", ""}, {"\\", ""}},
            });

        auto exp_map_arr = map_builder.Finish().ValueOrDie();
        bodo::tests::check(map_arr->Equals(exp_map_arr));
    });

    bodo::tests::test("test_nested_array_in_map", [] {
        auto arr = const_str_arr({
            // Outer Null Map
            std::nullopt,
            // Only Null
            "{\"null\": undefined}",
            // Empty Lists
            R"({"a": [], "b": undefined, "c": [undefined]})",
            // Simple Test Case
            R"(
{
    "b": [
        1,
        -Infinity,
        1.0
    ],
    "c": [
        1.e2,
        -1.2E-2
    ]
}
            )",
        });

        auto in_type = arrow::map(arrow::large_utf8(),
                                  arrow::large_list(arrow::float64()));
        auto map_arr = string_to_map_arr(arr, in_type);

        auto map_builder = std::static_pointer_cast<arrow::MapBuilder>(
            std::shared_ptr(arrow::MakeBuilder(in_type).ValueOrDie()));

        using in = std::vector<std::optional<double>>;
        using map = ordered_map<in>;
        construct_map_arr<in>(
            *map_builder,
            {
                std::nullopt,
                map{{"null", std::nullopt}},
                map{{"a", in{}}, {"b", std::nullopt}, {"c", in{std::nullopt}}},
                map{
                    {"b",
                     in{1.0, -std::numeric_limits<double>::infinity(), 1.0}},
                    {"c", in{100.0, -0.012}},
                },
            });

        auto exp_map_arr = map_builder->Finish().ValueOrDie();
        bodo::tests::check(map_arr->Equals(exp_map_arr));
    });

    bodo::tests::test("test_nested_map_in_map", [] {
        auto arr = const_str_arr({// Outer Null Map
                                  std::nullopt,
                                  // Only Null
                                  "{\"null\": undefined}",
                                  // Empty Maps
                                  R"({"a": {}, "b": undefined, "c": {}})",
                                  // Simple Test Case
                                  R"(
{
    "b": {
        "test": "2021-05-11",
        "once": undefined,
        "give": "1970-01-01"
    },
    "c": {
        "alpha": "2023-11-11",
        "beta": "2000-01-01",
        "omega": null
    }
}
            )"});

        auto in_type =
            arrow::map(arrow::large_utf8(),
                       arrow::map(arrow::large_utf8(), arrow::date32()));
        auto map_arr = string_to_map_arr(arr, in_type);

        auto map_builder = std::static_pointer_cast<arrow::MapBuilder>(
            std::shared_ptr(arrow::MakeBuilder(in_type).ValueOrDie()));

        using in = ordered_map<int32_t>;
        using map = ordered_map<in>;
        construct_map_arr<in>(
            *map_builder,
            {
                std::nullopt,
                map{{"null", std::nullopt}},
                map{{"a", in{}}, {"b", std::nullopt}, {"c", in{}}},
                map{
                    {"b", in{{"test", date32({2021y, May, 11d})},
                             {"once", std::nullopt},
                             {"give", date32({1970y, January, 1d})}}},
                    {"c", in{{"alpha", date32({2023y, November, 11d})},
                             {"beta", date32({2000y, January, 1d})},
                             {"omega", std::nullopt}}},
                },
            });

        auto exp_map_arr = map_builder->Finish().ValueOrDie();
        bodo::tests::check(map_arr->Equals(exp_map_arr));
    });

    bodo::tests::test("test_nested_struct_in_map", [] {
        auto arr = const_str_arr(
            {// Outer Null Map
             std::nullopt,
             // Only Null
             "{\"null\": undefined}",
             // Empty Struct, All Null Fields
             R"({"1": {}, "2": undefined, "3": {"ts": null, "value": null, "domain": {}}})",
             // Simple Test Case
             R"(
{
    "bodo": {
        "ts": "2023-10-24 12:34:56.789",
        "value": 100000000,
        "bonus": [true, false, undefined],
        "domain": {
            "test": 50111,
            "once": undefined,
            "give": 467
        }
    },
    "snowflake": {
        "ts": undefined,
        "value": -1467,
        "bonus": []
    }
}
            )"});

        auto value_type = arrow::struct_(
            {arrow::field("ts", arrow::timestamp(arrow::TimeUnit::NANO)),
             arrow::field("value", arrow::int64()),
             arrow::field("bonus", arrow::large_list(arrow::boolean())),
             arrow::field("domain",
                          arrow::map(arrow::large_utf8(), arrow::int64()))});
        auto in_type = arrow::map(arrow::large_utf8(), value_type);
        auto map_arr = string_to_map_arr(arr, in_type);

        auto builder = std::static_pointer_cast<arrow::MapBuilder>(
            std::shared_ptr(arrow::MakeBuilder(in_type).ValueOrDie()));

        using invec = std::vector<std::optional<bool>>;
        using inmap = ordered_map<int64_t>;
        using in =
            std::tuple<std::optional<std::shared_ptr<arrow::TimestampScalar>>,
                       std::optional<int64_t>, std::optional<invec>,
                       std::optional<inmap>>;
        using map = ordered_map<in>;
        construct_map_arr<in>(
            *builder,
            {
                std::nullopt,
                map{{"null", std::nullopt}},
                map{{"1", in{std::nullopt, std::nullopt, std::nullopt,
                             std::nullopt}},
                    {"2", std::nullopt},
                    {"3",
                     in{std::nullopt, std::nullopt, std::nullopt, inmap{}}}},
                map{
                    {"bodo", in{ts("2023-10-24 12:34:56.789"), 100000000,
                                invec{true, false, std::nullopt},
                                inmap{{"test", 50111},
                                      {"once", std::nullopt},
                                      {"give", 467}}}},
                    {"snowflake",
                     in{
                         std::nullopt,
                         -1467,
                         invec{},
                         std::nullopt,
                     }},
                },
            });

        auto exp_map_arr = builder->Finish().ValueOrDie();
        bodo::tests::check(map_arr->Equals(exp_map_arr));
    });

    // ----------------------------- Struct Tests -----------------------------
    // Note we will assume that the map code is similar to array and mainly
    // test differences, especially around the fields

    bodo::tests::test("test_empty_struct", [] {
        auto in_type = std::make_shared<arrow::StructType>(
            std::vector<std::shared_ptr<arrow::Field>>{
                arrow::field("null", arrow::null())});
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
        auto struct_arr = string_to_struct_arr(arr, in_type);

        auto null_builder = std::make_shared<arrow::NullBuilder>();
        auto struct_builder = arrow::StructBuilder(
            in_type, bodo::BufferPool::DefaultPtr(), {null_builder});

        // Null Outer Array
        bodo::tests::check(struct_builder.AppendNull().ok());
        // Empty Outer Array
        bodo::tests::check(struct_builder.Append().ok());
        bodo::tests::check(null_builder->AppendNull().ok());
        // Empty with Spacing
        bodo::tests::check(struct_builder.Append().ok());
        bodo::tests::check(null_builder->AppendNull().ok());
        // Only Null Inside
        bodo::tests::check(struct_builder.Append().ok());
        bodo::tests::check(null_builder->AppendNull().ok());

        auto exp_struct_arr = struct_builder.Finish().ValueOrDie();
        bodo::tests::check(struct_arr->Equals(exp_struct_arr));
    });

    bodo::tests::test("test_extra_keys", [] {
        auto in_type = std::make_shared<arrow::StructType>(
            std::vector<std::shared_ptr<arrow::Field>>{
                arrow::field("base", arrow::int64())});
        auto arr = const_str_arr({
            // Normal Row
            "{\"base\": 25}",
            // Empty Map,
            R"({"base": 10, "extra": null})",
        });
        bodo::tests::check_exception(
            [&] { string_to_struct_arr(arr, in_type); },
            "Found an unexpected field name");
    });

    bodo::tests::test("test_different_order_struct", [] {
        auto in_type = std::make_shared<arrow::StructType>(
            std::vector<std::shared_ptr<arrow::Field>>{
                arrow::field("f1", arrow::int64()),
                arrow::field("f2", arrow::boolean()),
                arrow::field("f3", arrow::large_utf8())});

        auto arr = const_str_arr({
            // Normal
            R"({"f1": 25, "f2": true, "f3": "hello"})",
            // Rotate
            R"({"f2": true,  "f3": "middle",  "f1": 10})",
            // Flip All
            R"({"f3": "bye", "f2": false, "f1": -5})",
        });
        auto struct_arr = string_to_struct_arr(arr, in_type);

        auto struct_builder = std::static_pointer_cast<arrow::StructBuilder>(
            std::shared_ptr(arrow::MakeBuilder(in_type).ValueOrDie()));

        construct_struct_arr<int64_t, bool, std::string>(
            struct_builder, {
                                // Normal
                                std::tuple{25, true, "hello"},
                                // Rotate
                                std::tuple{10, true, "middle"},
                                // Flip All
                                std::tuple{-5, false, "bye"},
                            });

        auto exp_struct_arr = struct_builder->Finish().ValueOrDie();
        bodo::tests::check(struct_arr->Equals(exp_struct_arr));
    });

    bodo::tests::test("test_missing_keys", [] {
        auto in_type = std::make_shared<arrow::StructType>(
            std::vector<std::shared_ptr<arrow::Field>>{
                arrow::field("f1", arrow::int64()),
                arrow::field("f2", arrow::boolean()),
                arrow::field("f3", arrow::large_utf8())});

        auto arr = const_str_arr({
            // Missing All
            "{}",
            // Missing f2
            R"({"f1": 10, "f3": "middle"})",
            // Missing f3 and f1
            "{\"f2\": false}",
        });
        auto struct_arr = string_to_struct_arr(arr, in_type);

        auto struct_builder = std::static_pointer_cast<arrow::StructBuilder>(
            std::shared_ptr(arrow::MakeBuilder(in_type).ValueOrDie()));

        construct_struct_arr<int64_t, bool, std::string>(
            struct_builder,
            {
                // Missing All
                std::tuple{std::nullopt, std::nullopt, std::nullopt},
                // Missing f2
                std::tuple{10, std::nullopt, "middle"},
                // Missing f3 and f1
                std::tuple{std::nullopt, false, std::nullopt},
            });

        auto exp_struct_arr = struct_builder->Finish().ValueOrDie();
        bodo::tests::check(struct_arr->Equals(exp_struct_arr));
    });

    bodo::tests::test("test_nested_in_struct", [] {
        auto arr =
            const_str_arr({// Outer Null Struct
                           std::nullopt,
                           // Empty Lists
                           R"({"array": [undefined], "map": {}, "struct": {}})",
                           // Simple Test Case
                           R"(
{
    "array": [
        1.0,
        -1.0,
        5.7
    ],
    "map": {
        "une": "uno",
        "deux": "dos",
        "trois": "tres"
    },
    "struct": {
        "ts": "2023-10-24 12:34:56.789",
        "value": {"date": "2023-10-22", "diff": null, "name": "bodo"},
        "bonus": [true, false, undefined],
        "domain": {
            "base1": 50111,
            "extra1": undefined,
            "base10": 467
        }
    }
}
            )"});

        auto value_type = arrow::struct_(
            {arrow::field("ts", arrow::timestamp(arrow::TimeUnit::NANO)),
             arrow::field(
                 "value",
                 arrow::struct_({arrow::field("date", arrow::date32()),
                                 arrow::field("diff", arrow::int64()),
                                 arrow::field("name", arrow::large_utf8())})),
             arrow::field("bonus", arrow::large_list(arrow::boolean())),
             arrow::field("domain",
                          arrow::map(arrow::large_utf8(), arrow::int64()))});
        auto in_type = std::make_shared<arrow::StructType>(
            std::vector<std::shared_ptr<arrow::Field>>{
                arrow::field("array", arrow::large_list(arrow::float64())),
                arrow::field("map", arrow::map(arrow::large_utf8(),
                                               arrow::large_utf8())),
                arrow::field("struct", value_type)});
        auto struct_arr = string_to_struct_arr(arr, in_type);

        auto struct_builder = std::static_pointer_cast<arrow::StructBuilder>(
            std::shared_ptr(arrow::MakeBuilder(in_type).ValueOrDie()));

        using invec = std::vector<std::optional<double>>;
        using inmap = ordered_map<std::string>;
        using ininvec = std::vector<std::optional<bool>>;
        using ininmap = ordered_map<int64_t>;
        using ininstruct =
            std::tuple<std::optional<int32_t>, std::optional<int64_t>,
                       std::optional<std::string>>;
        using instruct =
            std::tuple<std::optional<std::shared_ptr<arrow::TimestampScalar>>,
                       std::optional<ininstruct>, std::optional<ininvec>,
                       std::optional<ininmap>>;
        using in = std::tuple<std::optional<invec>, std::optional<inmap>,
                              std::optional<instruct>>;
        construct_struct_arr<invec, inmap, instruct>(
            struct_builder,
            {
                std::nullopt,
                in{invec{std::nullopt}, inmap{},
                   instruct{
                       std::nullopt,
                       std::nullopt,
                       std::nullopt,
                       std::nullopt,
                   }},
                in{invec{1.0, -1.0, 5.7},
                   inmap{{"une", "uno"}, {"deux", "dos"}, {"trois", "tres"}},
                   instruct{ts("2023-10-24 12:34:56.789"),
                            ininstruct{date32({2023y, October, 22d}),
                                       std::nullopt, "bodo"},
                            ininvec{true, false, std::nullopt},
                            ininmap{{"base1", 50111},
                                    {"extra1", std::nullopt},
                                    {"base10", 467}}}},
            });

        auto exp_struct_arr = struct_builder->Finish().ValueOrDie();
        bodo::tests::check(struct_arr->Equals(exp_struct_arr));
    });
});
