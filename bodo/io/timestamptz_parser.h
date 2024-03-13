#pragma once

#include <arrow/array.h>

/**
 * A simple parser for the timestamptz string format
 */
struct Parser {
    // The string to parse
    std::string_view ts_str;
    // The current position in the string
    size_t pos = 0;

    /**
     * Expects that the character at pos is c, and consumes it by advancing pos.
     * Returns 0 if the character is c, -1 otherwise. In the case of failure,
     * pos is not advanced.
     */
    int consume_char(char c);
    /**
     * Expects that the character at pos is a digit, and consumes it by
     * advancing pos. Returns the digit if it is a digit, -1 otherwise.
     */
    int parse_digit();
    /**
     * Expects to parse 2 digits.
     * Returns the parsed 2-digit number if it is a digit, -1 otherwise.
     */
    int parse_2_digit();

    /**
     * Expects to parse a year in the format YYYY.
     * Returns the parsed year if the string is valid, -1 otherwise.
     */
    int parse_year();
    /**
     * Expects to parse a month in the format MM.
     * Returns the parsed month if the string is valid, -1 otherwise.
     */
    int parse_month();
    /**
     * Expects to parse a day in the format DD.
     * Returns the parsed day if the string is valid, -1 otherwise.
     */
    int parse_day();
    /**
     * Expects to parse a hour in the format HH.
     * Returns the parsed hour if the string is valid, -1 otherwise.
     */
    int parse_hour();
    /**
     * Expects to parse a minute in the format MM.
     * Returns the parsed minute if the string is valid, -1 otherwise.
     */
    int parse_minute_second();
    /**
     * Expects to parse a second in the format SS.
     * Returns the parsed second if the string is valid, -1 otherwise.
     */
    int parse_nanoseconds();
    std::pair<int64_t, int16_t> parse_timestamptz();
};

/**
 * @brief Parses a String Array containing string encoded arrays into a
 * TimestampTZArray. The only supported formats are:
 *   "YYYY-MM-DD HH:MM:SS.S... [+-]TZH:TZM"
 *   "YYYY-MM-DD HH:MM:SS.S... 'Z'" (for the 0 offset)
 * @param in_arr The String Array containing the timestamptz strings
 * @return A TimestampTZArray containing the parsed timestamps
 */
std::shared_ptr<arrow::Array> string_to_timestamptz_arr(
    std::shared_ptr<arrow::StringArray> in_arr,
    std::shared_ptr<arrow::DataType> in_type);
