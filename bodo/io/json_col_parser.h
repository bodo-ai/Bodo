#pragma once

#include <arrow/array.h>
#include <arrow/table.h>
#include <arrow/type.h>

// ------------------------------- JSON PARSER -------------------------------

/// @brief JSON Tokens for Custom Parser (including Custom Array Pieces)
typedef enum {
    End = 0,      // Input ended
    ObjectStart,  // {
    ObjectEnd,    // }
    ArrayStart,   // [
    ArrayEnd,     // ]
    True,         // True
    False,        // False
    Null,         // Null
    Integer,      // Number value without a fraction part
    Float,        // Number value with a fraction part
    String,       // String value
    FieldName,    // Field name
    Error,        // An error occurred (see `error()` for details)
    NaN,          // Special Character for Floating-Point NaN
    Infinity,     // Special Character for Floating-Point inf
    NegInfinity,  // Special Character for Floating-Point -inf
    _Comma,
} Token;

/// @brief Error Codes while Parsing JSON
typedef enum {
    UnspecifiedError = 0,
    UnexpectedComma,
    UnexpectedTrailingComma,
    InvalidByte,
    PrematureEndOfInput,
    MalformedNumberLiteral,
    UnterminatedString,
    SyntaxError,
} ErrorCode;

/// @brief Reads a sequence of bytes representing JSON data and produces tokens
/// and values while doing so Based off of
/// https://github.com/rsms/jsont/tree/master
struct Tokenizer {
    /// @brief Original Source Text
    std::string_view source;
    /// @brief Number of Bytes Currently Parsed from Source
    uint64_t offset = 0;
    /// @brief Last Token Parsed from Source
    Token curr_token = End;

    /// @brief Start of String in Source Representing Current Tokens Value
    uint64_t value_start = 0;
    /// @brief Length of String in Source
    uint64_t value_len = 0;
    /// @brief Does the value contain an unescape string (basically a /)
    bool value_has_unescape = false;
    /// @brief Separate Place to Store Value If Conversion is Needed
    std::string value_str;

    /// @brief Current Error
    ErrorCode error_code = UnspecifiedError;

    Tokenizer() = default;
    ~Tokenizer() = default;

    /**
     * @brief Reset the tokenizer, making it possible to reuse this parser so to
     * avoid unnecessary memory allocation and deallocation.
     *
     * @param input The new input to parse
     */
    inline void reset(std::string_view input) {
        this->source = input;
        this->offset = 0;
        this->curr_token = End;

        this->value_start = 0;
        this->value_len = 0;

        this->error_code = UnspecifiedError;
    }

    /// @brief Read next token from input
    Token next();

    /// @brief Access current token
    inline Token current() const { return curr_token; }

    /// @brief Get a string_view slice of the value of the current token
    inline std::string_view value() const {
        assert(curr_token == NaN || curr_token == Infinity ||
               curr_token == NegInfinity || curr_token == String ||
               curr_token == FieldName || curr_token == Integer ||
               curr_token == Float);
        return source.substr(value_start, value_len);
    }

    /// @brief Returns the current value as a double-precision floating-point
    /// number
    double floatValue() const;

    /// @brief Returns the current value as a signed 64-bit integer
    int64_t intValue();

    /**
     * @brief Returns the current value as a string (undoing escaped characters
     * if necessary) Converted Characters are based on:
     * https://docs.snowflake.com/en/sql-reference/data-types-text#escape-sequences-in-single-quoted-string-constants
     * Not all of these remain escaped in the output though:
     * - \" Yes but \' No
     * - \\ Yes
     * - \0 Yes, gets converted to \u0000
     * - Whitespace Yes: \n \t \b \f \r
     * - Normal Octal (\ooo), Hex (\xhh), and Unicode (\uhhhh) seem to
     *   be properly unescaped
     * @param[out] out The output string to write to
     */
    void stringValue(std::string& out);

    /// @brief Returns the error code of the last error
    inline ErrorCode error() const {
        assert(curr_token == Error);
        return error_code;
    }

    /// @brief Returns true if tokenizer has reached the end of input
    size_t endOfInput() const;

    /// @brief Helper function to set the current token and return it
    inline Token setToken(Token t);

    /// @brief Helper function to set the error code and return the Error token
    inline Token setError(ErrorCode error);

    /**
     * @brief Helper Function to Tokenize a Fixed-Length Known Atomic Word
     *
     * @param str The word as a constant string
     * @param len The length of the word. TODO: Replace with compile-time len
     * function
     * @param token The token to return if the word is found
     * @return token on successful parse, Error on failure
     */
    inline Token readAtom(const char* str, size_t len, Token token);

    /// @brief Skip any whitespace characters until next non-whitespace
    /// character
    void skipWhitespace();
};

/**
 * @brief Parses a String Array containing JSON-encoded arrays into a ListArray
 * @param in_arr The String Array containing the JSON array values
 * @param in_type The type of the output ListArray
 * @return A ListArray containing the parsed JSON arrays
 */
std::shared_ptr<arrow::Array> string_to_list_arr(
    std::shared_ptr<arrow::StringArray> in_arr,
    std::shared_ptr<arrow::DataType> in_type);

/**
 * @brief Parses a String Array containing JSON-encoded arrays into a MapArray
 * @param in_arr The String Array containing the JSON array values
 * @param in_type The type of the output MapArray
 * @return A MapArray containing the parsed JSON arrays
 */
std::shared_ptr<arrow::Array> string_to_map_arr(
    std::shared_ptr<arrow::StringArray> in_arr,
    std::shared_ptr<arrow::DataType> in_type);

/**
 * @brief Parses a String Array containing JSON-encoded arrays into a
 * StructArray
 * @param in_arr The String Array containing the JSON array values
 * @param in_type The type of the output StructArray
 * @return A StructArray containing the parsed JSON arrays
 */
std::shared_ptr<arrow::Array> string_to_struct_arr(
    std::shared_ptr<arrow::StringArray> in_arr,
    std::shared_ptr<arrow::DataType> in_type);
