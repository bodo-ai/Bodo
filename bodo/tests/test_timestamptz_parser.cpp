#include "../io/timestamptz_parser.h"
#include "./test.hpp"

static bodo::tests::suite tests([] {
#define TEST_FAILS(str, func)          \
    {                                  \
        std::string_view ts_str = str; \
        Parser parser{ts_str};         \
        int v = parser.func();         \
        bodo::tests::check(v == -1);   \
    }

#define TEST_PASSES(str, func, result, expected_pos)    \
    {                                                   \
        std::string_view ts_str = str;                  \
        Parser parser{ts_str};                          \
        int v = parser.func();                          \
        bodo::tests::check(v == result);                \
        bodo::tests::check(parser.pos == expected_pos); \
    }
    bodo::tests::test("test_consume_char", [] {
        std::string_view ts_str = "a";
        Parser parser{.ts_str = ts_str};
        int v = parser.consume_char('a');
        bodo::tests::check(v == 0);
        bodo::tests::check(parser.pos == 1);

        v = parser.consume_char('a');
        bodo::tests::check(v == -1);
        bodo::tests::check(parser.pos == 1);
    });

    bodo::tests::test("test_parse_year", [] {
        TEST_PASSES("2021", parse_year, 2021, 4);
        TEST_PASSES("2021-01-01 00:00:00 +00:00", parse_year, 2021, 4);

        TEST_FAILS("YYYY", parse_year);
        TEST_FAILS("202", parse_year);
        TEST_FAILS("", parse_year);
    });

    bodo::tests::test("test_parse_month", [] {
        TEST_PASSES("01", parse_month, 1, 2);
        TEST_PASSES("12", parse_month, 12, 2);

        TEST_FAILS("00", parse_month);
        TEST_FAILS("13", parse_month);
        TEST_FAILS("1", parse_month);
        TEST_FAILS("XX", parse_month);
        TEST_FAILS("", parse_month);
    });

    bodo::tests::test("test_parse_day", [] {
        TEST_PASSES("01", parse_day, 1, 2);
        TEST_PASSES("31", parse_day, 31, 2);

        TEST_FAILS("00", parse_day);
        TEST_FAILS("32", parse_day);
        TEST_FAILS("1", parse_day);
        TEST_FAILS("XX", parse_day);
        TEST_FAILS("", parse_day);
    });

    bodo::tests::test("test_parse_hour", [] {
        TEST_PASSES("00", parse_hour, 0, 2);
        TEST_PASSES("23", parse_hour, 23, 2);

        TEST_FAILS("24", parse_hour);
        TEST_FAILS("1", parse_hour);
        TEST_FAILS("XX", parse_hour);
        TEST_FAILS("", parse_hour);
    });

    bodo::tests::test("test_parse_minute_second", [] {
        TEST_PASSES("00", parse_minute_second, 0, 2);
        TEST_PASSES("59", parse_minute_second, 59, 2);

        TEST_FAILS("60", parse_minute_second);
        TEST_FAILS("1", parse_minute_second);
        TEST_FAILS("XX", parse_minute_second);
        TEST_FAILS("", parse_minute_second);
    });

    bodo::tests::test("test_parse_nanoseconds", [] {
        TEST_PASSES("123456789", parse_nanoseconds, 123456789, 9);
        TEST_PASSES("000000001", parse_nanoseconds, 1, 9);
        TEST_PASSES("000000000", parse_nanoseconds, 0, 9);

        TEST_FAILS("0", parse_nanoseconds);
        TEST_FAILS("X", parse_nanoseconds);
        TEST_FAILS("", parse_nanoseconds);
    });

    bodo::tests::test("test_parse_ttz", [] {
        {
            std::string_view ts_str =
                "\"2020-01-02 03:04:05.000000000 +06:07\"";
            Parser parser{.ts_str = ts_str};
            auto [ts, tz] = parser.parse_timestamptz();
            bodo::tests::check(tz == 367);
            bodo::tests::check(ts == 1577912225000000000);
        }

        {
            std::string_view ts_str =
                "\"2020-01-02 03:04:05.123456789 -06:07\"";
            Parser parser{.ts_str = ts_str};
            auto [ts, tz] = parser.parse_timestamptz();
            bodo::tests::check(tz == -367);
            bodo::tests::check(ts == 1577956265123456789);
        }
    });

    bodo::tests::test("test_parse_ttz_0_offset", [] {
        std::string_view ts_str = "\"2020-01-02 03:04:05.000000000 Z\"";
        Parser parser{.ts_str = ts_str};
        auto [ts, tz] = parser.parse_timestamptz();
        bodo::tests::check(tz == 0);
        bodo::tests::check(ts == 1577934245000000000);
    });

    bodo::tests::test("test_parse_ttz_0_offset_with_ns", [] {
        {
            std::string_view ts_str = "\"2020-01-02 03:04:05.123456789 Z\"";
            Parser parser{.ts_str = ts_str};
            auto [ts, tz] = parser.parse_timestamptz();
            bodo::tests::check(tz == 0);
            bodo::tests::check(ts == 1577934245123456789);
        }

        {
            std::string_view ts_str = "\"2020-01-02 03:04:05.100000000 Z\"";
            Parser parser{.ts_str = ts_str};
            auto [ts, tz] = parser.parse_timestamptz();
            bodo::tests::check(tz == 0);
            bodo::tests::check(ts == 1577934245100000000);
        }
    });
#undef TEST_FAILS
#undef TEST_PASSES
});
