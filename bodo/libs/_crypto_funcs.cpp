/**
 * @file _crypto_funcs.cpp
 * @author Pintao Zou (pintao@bodo.ai)
 * @brief Implementation for SHA2 and other crypto SQL functions.
 *
 *
 */
#include "_crypto_funcs.h"
#include "_base64.h"
#include "_bodo_common.h"

#include <memory>
#include <stdexcept>
#include <vector>

#include <Python.h>
#include <openssl/evp.h>
#include <openssl/sha.h>

// Invocation of the OpenSSL MD5 implementation as described in the following
// links: https://github.com/yhirose/cpp-httplib/issues/1030
// https://github.com/yhirose/cpp-httplib/pull/1241/files
void run_md5(char *in_str, int64_t in_len, char *output) {
    auto context = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>(
        EVP_MD_CTX_new(), EVP_MD_CTX_free);
    uint32_t md5_length = 0;
    uint8_t md5_str[EVP_MAX_MD_SIZE];
    EVP_DigestInit_ex(context.get(), EVP_md5(), nullptr);
    EVP_DigestUpdate(context.get(), in_str, in_len);
    EVP_DigestFinal_ex(context.get(), md5_str, &md5_length);
    for (uint32_t i = 0; i < md5_length; i++) {
        std::snprintf(output + (i * 2), 4, "%02x", md5_str[i]);
    }
}

void run_crypto_function(char *in_str, int64_t in_len, crypto_function func,
                         char *output) {
    switch (func) {
        case crypto_function::md5:
            run_md5(in_str, in_len, output);
            break;
        case crypto_function::sha1:
            // TODO: Support SHA1
            break;
        case crypto_function::sha224:
            unsigned char sha224_str[SHA224_DIGEST_LENGTH];
            SHA224((const unsigned char *)in_str, in_len, sha224_str);
            /*
             https://docs.snowflake.com/en/sql-reference/functions/sha2
             Snowflake SHA2 function requires the output be hex encoded
            */
            for (int i = 0; i < SHA224_DIGEST_LENGTH; i++) {
                std::snprintf(output + (i * 2), 4, "%02x", sha224_str[i]);
            }
            break;
        case crypto_function::sha256:
            unsigned char sha256_str[SHA256_DIGEST_LENGTH];
            SHA256((const unsigned char *)in_str, in_len, sha256_str);
            for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
                std::snprintf(output + (i * 2), 4, "%02x", sha256_str[i]);
            }
            break;
        case crypto_function::sha384:
            unsigned char sha384_str[SHA384_DIGEST_LENGTH];
            SHA384((const unsigned char *)in_str, in_len, sha384_str);
            for (int i = 0; i < SHA384_DIGEST_LENGTH; i++) {
                std::snprintf(output + (i * 2), 4, "%02x", sha384_str[i]);
            }
            break;
        case crypto_function::sha512:
            unsigned char sha512_str[SHA512_DIGEST_LENGTH];
            SHA512((const unsigned char *)in_str, in_len, sha512_str);
            for (int i = 0; i < SHA512_DIGEST_LENGTH; i++) {
                std::snprintf(output + (i * 2), 4, "%02x", sha512_str[i]);
            }
            break;
        default:
            throw std::runtime_error("Invalid crypto function type.");
    }
}

/** Implementation of Snowflake function BASE64_ENCODE
 *
 *  Wikipedia:
 *      https://en.wikipedia.org/wiki/Base64
 *
 *  Snowflake:
 *      https://docs.snowflake.com/en/sql-reference/functions/base64_encode
 */

void run_base64_encode(char *in_str, int in_len, int out_len, int max_line_len,
                       char *char_62_str, char *char_63_str, char *char_pad_str,
                       char *out_str) {
    char char_62 = char_62_str[0];
    char char_63 = char_63_str[0];
    char char_pad = char_pad_str[0];

    // Use the base64 library to do the regular encoding scheme, writing into
    // a temporary character buffer
    int encode_length = Base64encode_len(in_len);
    std::vector<char> temp(encode_length);
    Base64encode(temp.data(), in_str, in_len);

    // Copy over the characters from the temporary buffer into the output
    // buffer, adding newline characters and replacing the 62/63/pad characters
    // as needed.
    int write_offset = 0;
    for (int read_offset = 0; read_offset < encode_length - 1; read_offset++) {
        char c = temp[read_offset];
        switch (c) {
            case '+': {
                out_str[write_offset] = char_62;
                break;
            }
            case '/': {
                out_str[write_offset] = char_63;
                break;
            }
            case '=': {
                out_str[write_offset] = char_pad;
                break;
            }
            default: {
                out_str[write_offset] = c;
            }
        }
        write_offset++;
        if (max_line_len > 0 &&
            ((read_offset % max_line_len) == (max_line_len - 1)) &&
            (read_offset != (encode_length - 2))) {
            out_str[write_offset] = '\n';
            write_offset++;
        }
    }
}

/** Implementation of Snowflake function BASE64_DECODE_STRING
 *
 *  Wikipedia:
 *      https://en.wikipedia.org/wiki/Base64
 *
 *  Snowflake:
 *      https://docs.snowflake.com/en/sql-reference/functions/base64_decode_string
 */

bool run_base64_decode_string(char *in_str, int in_len, char *char_62_str,
                              char *char_63_str, char *char_pad_str,
                              char *out_str) {
    char char_62 = char_62_str[0];
    char char_63 = char_63_str[0];
    char char_pad = char_pad_str[0];

    // Return failure if the length is not a multiple of 4
    if ((in_len % 4) != 0) {
        return false;
    }

    // Convert the characters from the input string to the original encoding
    // setup, writing the result into a temporary buffer, plus an extra null
    // terminator.
    std::vector<char> temp_in(in_len + 1);
    for (int read_offset = 0; read_offset < in_len; read_offset++) {
        char c = in_str[read_offset];
        if (c == char_62) {
            // Replace occurrences of the index 62 character with the default:
            // '+'
            temp_in[read_offset] = '+';
        } else if (c == char_63) {
            // Replace occurrences of the index 63 character with the default:
            // '/'
            temp_in[read_offset] = '/';
        } else if (c == char_pad) {
            // Replace occurrences of the padding character with the default:
            // '='
            temp_in[read_offset] = '=';
        } else if (!('a' <= c && c <= 'z') && !('A' <= c && c <= 'Z') &&
                   !('0' <= c && c <= '9')) {
            // Return failure if the character is not one of the remaining legal
            // characters
            return false;
        } else {
            // For any other character, copy it over directly
            temp_in[read_offset] = c;
        }
    }
    temp_in[in_len] = '\x00';

    // Use the base64 library to do the decoding encoding scheme, writing into
    // a temporary output buffer which has room for an extra null terminator.
    int out_len = (in_len >> 2) * 3;
    std::vector<char> temp_out(out_len + 1);
    int decoded_length = Base64decode(temp_out.data(), temp_in.data());

    // Copy the result to the output buffer, without the extra null terminator.
    for (int write_offset = 0; write_offset < decoded_length; write_offset++) {
        out_str[write_offset] = temp_out[write_offset];
    }

    return true;
}

// Initialize encryption/encoding functions for usage with python
PyMODINIT_FUNC PyInit_crypto_funcs(void) {
    PyObject *m;
    MOD_DEF(m, "crypto_funcs", "No docs", nullptr);
    if (m == nullptr) {
        return nullptr;
    }

    // No need to init bodo_common since just using the SetAttrStringFromVoidPtr
    // macro in this module

    SetAttrStringFromVoidPtr(m, run_crypto_function);
    SetAttrStringFromVoidPtr(m, run_base64_encode);
    SetAttrStringFromVoidPtr(m, run_base64_decode_string);

    return m;
}
