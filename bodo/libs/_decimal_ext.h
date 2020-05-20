#ifndef INCLUDE_DECIMAL_EXT_H
#define INCLUDE_DECIMAL_EXT_H


std::string decimal_value_cpp_to_std_string(decimal_value_cpp const& value, int const& scale);

bool operator<(decimal_value_cpp const& left, decimal_value_cpp const& right);

double decimal_to_double(decimal_value_cpp const& val);

#endif
