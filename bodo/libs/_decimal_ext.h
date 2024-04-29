#pragma once

#include <string>

#define DECIMAL128_MAX_PRECISION 38

std::string int128_decimal_to_std_string(__int128 const& value,
                                         int const& scale);

double decimal_to_double(__int128 const& val, uint8_t scale = 18);
