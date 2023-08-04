/**
 * @file _crypto_funcs.cpp
 * @author Pintao Zou (pintao@bodo.ai)
 * @brief Implementation for SHA2 and other crypto SQL functions.
 *
 * @copyright Copyright (C) 2023 Bodo Inc. All rights reserved.
 *
 */
#include "_bodo_common.h"
#include "_crypto_funcs.h"
#include <openssl/md5.h>
#include <openssl/sha.h>
#include <Python.h>
#include <stdexcept>

void run_crypto_function(char *in_str, int64_t in_len, crypto_function func, char *output) {
    switch (func) {
        case crypto_function::md5:
            unsigned char md5_str[MD5_DIGEST_LENGTH];
            MD5((const unsigned char *)in_str, in_len, md5_str);
            for (int i = 0; i < MD5_DIGEST_LENGTH; i++) {
                std::snprintf(output + (i * 2), 4, "%02x", md5_str[i]);
            }
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

// Initialize run_crypto_function function for usage with python
PyMODINIT_FUNC PyInit_crypto_funcs(void) {
    PyObject *m;
    MOD_DEF(m, "crypto_funcs", "No docs", NULL);
    if (m == NULL) {
        return NULL;
    }

    bodo_common_init();

    SetAttrStringFromVoidPtr(m, run_crypto_function);

    return m;
}
