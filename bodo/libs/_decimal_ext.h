#ifndef INCLUDE_DECIMAL_EXT_H
#define INCLUDE_DECIMAL_EXT_H

std::string int128_decimal_to_std_string(__int128 const& value,
                                         int const& scale);

double decimal_to_double(__int128 const& val);

#endif
