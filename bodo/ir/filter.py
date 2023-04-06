"""
This file contains dictionaries mapping BodoSQL kernel name to 
corresponding Arrow compute functions.

Dictionaries are separated by category 
(string functions, datetime functions, etc.) and 
number of arguments.

The keys are the name of the BodoSQL kernel.
The values are tuple pairs of the
    - Snowflake SQL Function Name
    - PyArrow Compute function 
"""
from typing import Dict, Tuple, Union

from numba.core import ir


def use_same_name_sql_func_arrow_func(x: Dict[str, str]):
    return {k: (v, v) for k, v in x.items()}


def use_same_name_kernel_sql_func(x: Dict[str, str]):
    return {k: (k, v) for k, v in x.items()}


date_funcs_no_arg_map: Dict[str, Tuple[str, str]] = {
    "get_hour": ("HOUR", "hour"),
    "get_minute": ("MINUTE", "minute"),
    "get_second": ("SECOND", "second"),
    "get_year": ("YEAR", "year"),
    # TODO: YEAROFWEEK seems to map to get_year, but I think thats wrong (no account for weeks that start in previous year)
    "yearofweekiso": ("YEAROFWEEKISO", "iso_year"),
    "dayofmonth": ("DAY", "day"),
    "dayofweek": ("DAYOFWEEK", "day_of_week"),
    "dayofweekiso": ("DAYOFWEEKISO", "day_of_week"),
    "dayofyear": ("DAYOFYEAR", "day_of_year"),
    # TODO: Why are there 2 different ones?
    "get_week": ("WEEK", "week"),
    "get_weekofyear": ("WEEKOFYEAR", "week"),
    # TODO: WEEKISO seems to map to get_weekofyear, but I think thats wrong (non ISO version)
    "get_month": ("MONTH", "month"),
    "get_quarter": ("QUARTER", "quarter"),
}

numeric_funcs_no_arg_map = use_same_name_sql_func_arrow_func(
    {
        "abs": "abs",
        "sign": "sign",
        "ceil": "ceil",
        "floor": "floor",
    }
)

numeric_funcs_one_arg_map = use_same_name_sql_func_arrow_func(
    {
        "mod": "mod",
        "round": "round",
        "truncate": "trunc",
        "trunc": "trunc",
    }
)

# SQL functions that only operate over a single input column
string_funcs_no_arg_map = use_same_name_kernel_sql_func(
    {
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
)

# SQL functions that take in 1 additional argument
string_funcs_one_arg_map = use_same_name_kernel_sql_func(
    {
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
)


# TODO: Replace with | operation once upgrading to Python 3.9
supported_funcs_no_arg_map = {
    **string_funcs_no_arg_map,
    **numeric_funcs_no_arg_map,
    **date_funcs_no_arg_map,
}

supported_compute_funcs = list(
    string_funcs_one_arg_map.keys()
    | string_funcs_no_arg_map.keys()
    | numeric_funcs_no_arg_map.keys()
    | numeric_funcs_one_arg_map.keys()
    | date_funcs_no_arg_map.keys()
)


# Typing Aliases
Filter = Tuple[Union[str, "Filter"], str, Union[ir.Var, str]]
