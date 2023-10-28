#include "json_col_parser.h"

#include <charconv>
#include <string>

#include <arrow/builder.h>
#include <arrow/type.h>

#include "../libs/_bodo_common.h"

// ------------------------------- JSON PARSER -------------------------------

const char* TokenStr[] = {
    "End",   "ObjectStart", "ObjectEnd", "ArrayStart",  "ArrayEnd", "True",
    "False", "Null",        "Integer",   "Float",       "String",   "FieldName",
    "Error", "NaN",         "Infinity",  "NegInfinity", "_Comma",
};

const char* ErrorCodeStr[] = {
    "Unspecified Error",         "Unexpected Comma",
    "Unexpected Trailing Comma", "Invalid Byte",
    "Premature End of Input",    "Malformed Number Literal",
    "Unterminated String",       "Syntax Error",
};

Token Tokenizer::next() {
    skipWhitespace();
    if (endOfInput()) {
        return setToken(End);
    }
    uint8_t b = source[offset++];

    // Repeated Consecutive Commas
    while (b == ',') {
        switch (curr_token) {
            case ObjectStart:
            case ArrayStart:
            case FieldName:
            case _Comma:
                return setError(ErrorCode::UnexpectedComma);
            default:
                break;
        }
        curr_token = _Comma;
        skipWhitespace();
        b = source[offset++];
    }

    switch (b) {
        case '{':
            return setToken(ObjectStart);
        case '}': {
            if (curr_token == _Comma) {
                return setError(ErrorCode::UnexpectedTrailingComma);
            }
            return setToken(ObjectEnd);
        }

        case '[':
            return setToken(ArrayStart);
        case ']': {
            if (curr_token == _Comma) {
                return setError(ErrorCode::UnexpectedTrailingComma);
            }
            return setToken(ArrayEnd);
        }

        case 'n':
            return readAtom("ull", 3, Null);
        case 't':
            return readAtom("rue", 3, True);
        case 'f':
            return readAtom("alse", 4, False);
        case 'u':
            return readAtom("ndefined", 8, Null);
        case 'N':
            return readAtom("aN", 2, NaN);
        case 'I':
            return readAtom("nfinity", 7, Infinity);

        case 0:
            return setError(ErrorCode::InvalidByte);

        // When we read a value, we don't produce a token until we either
        // reach end of input, a colon (then the value is a field name), a
        // comma, or an array or object terminator.
        case '"': {
            value_start = offset;
            value_has_unescape = false;

            while (!endOfInput() && source[offset] != '"') {
                b = source[offset++];
                assert(offset < source.length());

                // The old code would parse UTF-8 sequences, but lets do so
                // in a separate function rather than nesting inside parser
                if (b == 0) {
                    return setError(InvalidByte);
                } else if (b == '\\') {
                    value_has_unescape = true;
                    if (source[offset] == '"' || source[offset] == '\\') {
                        // Case 1: Skip escaped double quote
                        // Case 2 (like "\\") second slash can confuse this
                        // check, so explicitly skip
                        offset++;
                    }
                }
            }

            if (source[offset] != '"') {
                return setError(UnterminatedString);
            }
            // Ex: "test". Want value_start = 1, value_len = 4
            // At this point, offset will be 5
            value_len = offset - value_start;
            offset++;

            // Is this a field name?
            skipWhitespace();
            b = source[offset++];
            switch (b) {
                case ':':
                    return setToken(FieldName);
                case ',':
                    break;
                case ']':
                case '}': {
                    --offset;  // rewind
                    break;
                }
                case 0:
                    return setError(InvalidByte);
                default: {
                    // Expected a comma or a colon
                    return setError(SyntaxError);
                }
            }

            return setToken(String);
        }

        // We are reading a number
        case '+':
        case '-':
        case '0' ... '9': {
            value_start = offset - 1;
            Token token = Integer;

            // Check if we have -Infinity
            if (source[offset] == 'I') {
                return readAtom("Infinity", 8, NegInfinity);
            }

            while (!endOfInput()) {
                b = source[offset];
                switch (b) {
                    case '0' ... '9':
                        break;
                    case '.':
                        token = Float;
                        break;
                    case 'E':
                    case 'e':
                    case '-':
                    case '+': {
                        if (token != Float) {
                            return setError(MalformedNumberLiteral);
                        }
                        break;
                    }
                    default: {
                        // Just a + or - symbol is invalid
                        if ((offset - value_start == 1) &&
                            (source[value_start] == '-' ||
                             source[value_start] == '+')) {
                            return setError(MalformedNumberLiteral);
                        }

                        value_len = offset - value_start;
                        return setToken(token);
                    }
                }
                offset++;
            }

            return setToken(End);
        }

        default: {
            return setError(InvalidByte);
        }
    }

    return setToken(End);
}

double Tokenizer::floatValue() const {
    assert(curr_token == NaN || curr_token == Infinity ||
           curr_token == NegInfinity || curr_token == Float ||
           curr_token == Integer);

    switch (curr_token) {
        case NaN:
            return std::nan("");
        case Infinity:
            return std::numeric_limits<double>::infinity();
        case NegInfinity:
            return -std::numeric_limits<double>::infinity();
        default:
            break;
    }

    double res;

#ifdef __cpp_lib_to_chars
    // GCC supports direct conversion of string_view -> double

    auto [_, ec] =
        std::from_chars(source.data() + value_start,
                        source.data() + value_start + value_len, res);
    if (ec != std::errc()) {
        throw std::runtime_error("Failed to Parse Expected Float in JSON: " +
                                 std::string(value()));
    }
#else
    // Clang requires converting string_view -> string -> double
    // Ideally, we don't want to ever do the extra allocation
    // TODO: Find an alternative to std::stod. Check Boost
    res = std::stod(std::string(source.substr(value_start, value_len)));
#endif

    return res;
}

int64_t Tokenizer::intValue() {
    assert(curr_token == Integer);

    // If number starts with +, skip
    int plus_offset = source[value_start] == '+' ? 1 : 0;

    int64_t res;
    auto [_, ec] = std::from_chars(
        source.data() + value_start + plus_offset,
        source.data() + value_start + plus_offset + value_len, res);
    if (ec != std::errc()) {
        throw std::runtime_error("Failed to Parse Expected Integer in JSON: " +
                                 std::string(value()));
    }

    return res;
}

void Tokenizer::stringValue(std::string& out) {
    std::string_view in = value();
    out.clear();
    out.reserve(in.length());

    for (size_t i = 0; i < in.length(); i++) {
        if (in[i] != '\\') {
            out.push_back(in[i]);
            continue;
        }

        i++;
        switch (in[i]) {
            case '"':
            case '\\':
                out.push_back(in[i]);
                break;
            case 'n':
                out.push_back('\n');
                break;
            case 't':
                out.push_back('\t');
                break;
            case 'b':
                out.push_back('\b');
                break;
            case 'f':
                out.push_back('\f');
                break;
            case 'r':
                out.push_back('\r');
                break;
            case 'u': {
                if (i + 4 >= in.length()) {
                    throw std::runtime_error(
                        "Incomplete Unicode Escape Sequence In String: " +
                        std::string(in));
                } else if (in.substr(i + 1, 4) != "0000") {
                    throw std::runtime_error(
                        "Found Unescaped Unicode In Snowflake JSON String: " +
                        std::string(in));
                }

                out.push_back('\0');
                i += 4;
                break;
            }
            default:
                throw std::runtime_error("Invalid Escape Sequence In String: " +
                                         std::string(in));
        }
    }
}

size_t Tokenizer::endOfInput() const { return offset >= source.length(); }

inline Token Tokenizer::setToken(Token t) {
    curr_token = t;
    return curr_token;
}

inline Token Tokenizer::setError(ErrorCode error) {
    this->error_code = error;
    curr_token = Error;
    return curr_token;
}

inline Token Tokenizer::readAtom(const char* str, size_t len, Token token) {
    if (source.length() - offset < len) {
        return setError(ErrorCode::PrematureEndOfInput);
    } else if (!source.substr(offset).starts_with(str)) {
        return setError(ErrorCode::InvalidByte);
    } else {
        offset += len;
        return setToken(token);
    }
}

void Tokenizer::skipWhitespace() {
    while (!endOfInput()) {
        uint8_t b = source[offset++];
        switch (b) {
            case ' ':
            case '\t':
            case '\r':
            case '\n':  // IETF RFC4627
                // ignore whitespace and let the outer "while" do its thing
                break;
            default: {
                --offset;  // Rewind
                return;
            }
        }
    }
}

// --------------------------- JSON to Array Reader ---------------------------

/**
 * @brief Consume output from a tokenizer for an input JSON and
 * insert into the ArrayBuilder for the values of a single element in a
 * ListArray
 *
 * @param value_builder ArrayBuilder to insert output into
 * @param tokenizer Tokenizer to read from. Should be at the start of an array
 * @param value_type
 */
void parse_to_list(std::shared_ptr<arrow::ArrayBuilder> value_builder,
                   Tokenizer& tokenizer,
                   const std::shared_ptr<arrow::DataType>& value_type) {
    assert(tokenizer.current() == Token::ArrayStart);
    auto value_id = value_type->id();

    while (true) {
        Token token = tokenizer.next();
        switch (token) {
            // Hit the end of the main array we're currently parsing
            case Token::ArrayEnd:
                return;
            case Token::_Comma:
                continue;
            case Token::End: {
                throw std::runtime_error(
                    "Unexpected end of input when parsing a list column "
                    "row: " +
                    std::string(tokenizer.source));
                break;
            }
            case Token::Null: {
                auto status = value_builder->AppendNull();
                if (!status.ok()) {
                    throw std::runtime_error(
                        "Failure while parsing NULL from a list element:\n" +
                        std::string(tokenizer.source) + "\nWith error:\n" +
                        status.ToString());
                }
                break;
            }
            case Token::True:
            case Token::False: {
                if (value_id != arrow::Type::BOOL) {
                    throw std::runtime_error(
                        "Found an unexpected boolean value when parsing a "
                        "list[" +
                        value_type->ToString() + "] element:\n" +
                        std::string(tokenizer.source));
                }
                auto bool_builder =
                    std::static_pointer_cast<arrow::BooleanBuilder>(
                        value_builder);

                auto status = bool_builder->Append(token == Token::True);
                if (!status.ok()) {
                    throw std::runtime_error(
                        "Failure while parsing a boolean from a list "
                        "element:\n" +
                        std::string(tokenizer.source) + "\nWith error:\n" +
                        status.ToString());
                }
                break;
            }
            case Token::NaN:
            case Token::Infinity:
            case Token::NegInfinity:
            case Token::Float: {
                if (value_id != arrow::Type::DOUBLE) {
                    throw std::runtime_error(
                        "Found an unexpected float value when parsing a list[" +
                        value_type->ToString() + "] element:\n" +
                        std::string(tokenizer.source));
                }
                auto double_builder =
                    std::static_pointer_cast<arrow::DoubleBuilder>(
                        value_builder);

                auto status = double_builder->Append(tokenizer.floatValue());
                if (!status.ok()) {
                    throw std::runtime_error(
                        "Failure while parsing a float from a list element:\n" +
                        std::string(tokenizer.source) + "\nWith error:\n" +
                        status.ToString());
                }
                break;
            }
            case Token::Integer: {
                if (value_id == arrow::Type::INT64) {
                    auto int_builder =
                        std::static_pointer_cast<arrow::Int64Builder>(
                            value_builder);
                    auto status = int_builder->Append(tokenizer.intValue());
                    if (!status.ok()) {
                        throw std::runtime_error(
                            "Failure while parsing an integer from a list[int] "
                            "element:\n" +
                            std::string(tokenizer.source) + "\nWith error:\n" +
                            status.ToString());
                    }
                } else if (value_id == arrow::Type::DOUBLE) {
                    auto double_builder =
                        std::static_pointer_cast<arrow::DoubleBuilder>(
                            value_builder);
                    auto status =
                        double_builder->Append(tokenizer.floatValue());
                    if (!status.ok()) {
                        throw std::runtime_error(
                            "Failure while parsing a integer from a "
                            "list[float] element:\n" +
                            std::string(tokenizer.source) + "\nWith error:\n" +
                            status.ToString());
                    }
                } else {
                    throw std::runtime_error(
                        "Found an unexpected integer value when parsing a "
                        "list[" +
                        value_type->ToString() + "] element:\n" +
                        std::string(tokenizer.source));
                }
                break;
            }
            case Token::String: {
                if (value_id == arrow::Type::DATE32) {
                    auto date_builder =
                        std::static_pointer_cast<arrow::Date32Builder>(
                            value_builder);
                    auto parsed =
                        arrow::Scalar::Parse(value_type, tokenizer.value())
                            .ValueOrDie();
                    auto status = date_builder->AppendScalar(*parsed);
                    if (!status.ok()) {
                        throw std::runtime_error(
                            "Failure while parsing a date from a list "
                            "element:\n" +
                            std::string(tokenizer.source) + "\nWith error:\n" +
                            status.ToString());
                    }
                    break;
                } else if (value_id == arrow::Type::TIMESTAMP) {
                    auto timestamp_builder =
                        std::static_pointer_cast<arrow::TimestampBuilder>(
                            value_builder);
                    auto parsed =
                        arrow::Scalar::Parse(value_type, tokenizer.value())
                            .ValueOrDie();
                    auto status = timestamp_builder->AppendScalar(*parsed);
                    if (!status.ok()) {
                        throw std::runtime_error(
                            "Failure while parsing a timestamp from a list "
                            "element:\n" +
                            std::string(tokenizer.source) + "\nWith error:\n" +
                            status.ToString());
                    }
                    break;
                } else if (value_id == arrow::Type::LARGE_STRING ||
                           value_id == arrow::Type::STRING) {
                    auto string_builder =
                        std::static_pointer_cast<arrow::LargeStringBuilder>(
                            value_builder);

                    arrow::Status status;
                    if (tokenizer.value_has_unescape) {
                        tokenizer.stringValue(tokenizer.value_str);
                        status = string_builder->Append(tokenizer.value_str);
                    } else {
                        status = string_builder->Append(tokenizer.value());
                    }

                    if (!status.ok()) {
                        throw std::runtime_error(
                            "Failure while parsing a string from a list "
                            "element:\n" +
                            std::string(tokenizer.source) + "\nWith error:\n" +
                            status.ToString());
                    }
                } else {
                    throw std::runtime_error(
                        "Found an unexpected string value when parsing a "
                        "list[" +
                        value_type->ToString() + "] element:\n" +
                        std::string(tokenizer.source));
                }
                break;
            }
            // TODO: Handle Nested Arrays
            case Token::ArrayStart: {
                throw std::runtime_error(
                    "Found a nested array when parsing a ListArray column");
                break;
            }
            // TODO: Handle Nested Objects
            case Token::ObjectStart: {
                throw std::runtime_error(
                    "Found a nested object when parsing a ListArray column");
                break;
            }
            case Token::Error: {
                std::string error_msg(ErrorCodeStr[tokenizer.error()]);
                throw std::runtime_error(
                    "Found an error when parsing a ListArray column: " +
                    error_msg + "\n\t" + std::string(tokenizer.source) +
                    " at offset " + std::to_string(tokenizer.offset));
                break;
            }
            default: {
                throw std::runtime_error(
                    "Found an invalid token when parsing a ListArray "
                    "column:\n\t" +
                    std::string(tokenizer.source));
                break;
            }
        }
    }
}

std::shared_ptr<arrow::Array> string_to_list_arr(
    std::shared_ptr<arrow::StringArray> in_arr,
    std::shared_ptr<arrow::DataType> in_type) {
    // Extract List Types from DataType
    // Shouldn't be a list type if constructed in Bodo
    assert(in_type->id() == arrow::Type::LARGE_LIST);

    auto list_type = std::static_pointer_cast<arrow::LargeListType>(in_type);
    auto value_type = list_type->value_type();

    std::shared_ptr<arrow::ArrayBuilder> value_builder =
        arrow::MakeBuilder(value_type, bodo::BufferPool::DefaultPtr())
            .ValueOrDie();
    auto list_builder = std::make_unique<arrow::LargeListBuilder>(
        bodo::BufferPool::DefaultPtr(), value_builder, in_type);

    Tokenizer tokenizer;

    // Iterate over StringArray, Parse JSON, Validate, and Insert
    // TODO: Is there an Iterator over StringArray contents?
    for (int64_t i = 0; i < in_arr->length(); i++) {
        if (in_arr->IsNull(i)) {
            auto status = list_builder->AppendNull();
            continue;
        }

        auto json_str = in_arr->GetView(i);
        auto status = list_builder->Append();
        if (!status.ok()) {
            throw std::runtime_error("Failure while appending list: " +
                                     status.ToString());
        }

        tokenizer.reset(json_str);
        [[maybe_unused]] auto start_token = tokenizer.next();
        assert(start_token == Token::ArrayStart);
        parse_to_list(value_builder, tokenizer, value_type);
    }

    return list_builder->Finish().ValueOrDie();
}
