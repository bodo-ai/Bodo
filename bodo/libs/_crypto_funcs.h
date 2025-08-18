#pragma once

#include <cstdint>

enum crypto_function {
    md5 = 0,
    sha1 = 1,
    sha224 = 224,
    sha256 = 256,
    sha384 = 384,
    sha512 = 512
};

/**
 * @brief Computes the MD5, SHA1 or SHA2 encryption results
 *
 * @param[in] char *in_str input string pointer
 * @param[in] int64_t in_len input length
 * @param[in] crypto_function func which encryption function to be used
 * @param[out] char *output output string pointer
 */
void run_crypto_function(char *in_str, int64_t in_len, crypto_function func,
                         char *output);
