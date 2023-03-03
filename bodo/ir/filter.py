"""
This file contains dictionaries mapping SQL functions to 
corresponding Arrow compute functions. 

We assume that all BodoSQL kernels are named as 
the exact SQL function it provides functionality for.
In other words, the keys of the dictionaries are both the name 
of the BodoSQL kernel and the name of the SQL function name. 

Dictionaries are separated by category
(string functions, datetime functions, etc.) and 
number of arguments.
"""

# SQL functions that only operate over a single input column
string_funcs_no_arg_map = {
    "lower": "utf8_lower",
    "upper": "utf8_upper",
    "length": "utf8_length",
    "reverse": "utf8_reverse",
    "ltrim": "utf8_ltrim_whitespace",
    "rtrim": "utf8_rtrim_whitespace",
    "trim": "utf8_trim_whitespace",
    "initcap": "utf8_capitalize",
    "reverse": "utf8_reverse",
}


# SQL functions that take in 1 additional argument
string_funcs_one_arg_map = {
    "lpad": "utf8_lpad",
    "rpad": "utf8_rpad",
    "ltrim": "utf8_rtrim",
    "rtrim": "utf8_rtrim",
    "trim": "utf8_trim",
    "split": "split_pattern",
    "startswith": "starts_with",
    "endswith": "ends_with",
    "contains": "match_substring",
    "coalesce": "coalesce",
    "case_insensitive_startswith": "starts_with",
    "case_insensitive_endswith": "ends_with",
    "case_insensitive_contains": "match_substring",
}


supported_compute_funcs = list(
    string_funcs_one_arg_map.keys() | string_funcs_no_arg_map.keys()
)
