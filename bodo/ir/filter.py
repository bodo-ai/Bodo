"""
This file contains dictionaries mapping BodoSQL kernel name to 
corresponding SQL functions. This file also contains
supported_arrow_funcs_map, which is a dictionary that maps
BodoSQL kernel name to an equivalent PyArrow compute function.

Dictionaries are separated by category 
(string functions, datetime functions, etc.) and 
number of arguments.

The keys are the name of the BodoSQL kernel.
"""
from typing import Tuple, Union

from numba.core import ir

string_funcs_no_arg_map = {
    "lower": "LOWER",
    "upper": "UPPER",
    "length": "LENGTH",
    "reverse": "REVERSE",
}

numeric_funcs_no_arg_map = {
    "abs": "ABS",
    "sign": "SIGN",
    "ceil": "CEIL",
    "floor": "FLOOR",
}

date_funcs_no_arg_map = {
    "get_hour": "HOUR",
    "get_minute": "MINUTE",
    "get_second": "SECOND",
    # TODO (srilman): YEAROFWEEK seems to map to get_year, but I think thats wrong (no account for weeks that start in previous year)
    "get_year": "YEAR",
    "yearofweekiso": "YEAROFWEEKISO",
    "dayofmonth": "DAY",
    "dayofweek": "DAYOFWEEK",
    "dayofweekiso": "DAYOFWEEKISO",
    "dayofyear": "DAYOFYEAR",
    # TODO (srilman): Why are there 2 different ones?
    "get_week": "WEEK",
    "get_weekofyear": "WEEKOFYEAR",
    # TODO (srilman): WEEKISO seems to map to get_weekofyear, but I think thats wrong (non ISO version)
    "get_month": "MONTH",
    "get_quarter": "QUARTER",
}

string_funcs_map = {
    "ltrim": "LTRIM",
    "rtrim": "RTRIM",
    "lpad": "LPAD",
    "rpad": "RPAD",
    "trim": "TRIM",
    "split": "SPLIT_PART",
    "contains": "CONTAINS",
    "coalesce": "COALESCE",
    "repeat": "REPEAT",
    "translate": "TRANSLATE",
    "strtok": "STRTOK",
    "split": "SPLIT_PART",
    "initcap": "INITCAP",
    "concat_ws": "CONCAT",
    "left": "LEFT",
    "right": "RIGHT",
    "position": "POSITION",
    "replace": "REPLACE",
    "substring": "SUBSTRING",
    "charindex": "POSITION",
    "editdistance_no_max": "EDITDISTANCE",
    "editdistance_with_max": "EDITDISTANCE",
    "regexp_substr": "REGEXP_SUBSTR",
    "regexp_instr": "REGEXP_INSTR",
    "regexp_replace": "REGEXP_REPLACE",
    "regexp_count": "REGEXP_COUNT",
}

numeric_funcs_map = {
    "mod": "MOD",
    "round": "ROUND",
    "trunc": "TRUNC",
    "truncate": "TRUNCATE",
}

cond_funcs_map = {
    "least": "LEAST",
    "greatest": "GREATEST",
}

# TODO (srilman): Replace with | operation once upgrading to Python 3.9
supported_funcs_no_arg_map = {
    **string_funcs_no_arg_map,
    **numeric_funcs_no_arg_map,
    **date_funcs_no_arg_map,
}

supported_funcs_map = {
    **supported_funcs_no_arg_map,
    **numeric_funcs_map,
    **string_funcs_map,
    **cond_funcs_map,
}

supported_arrow_funcs_map = {
    "lower": "utf8_lower",
    "upper": "utf8_upper",
    "length": "utf8_length",
    "reverse": "utf8_reverse",
    "startswith": "starts_with",
    "endswith": "ends_with",
    "contains": "match_substring",
    "coalesce": "coalesce",
    "case_insensitive_startswith": "starts_with",
    "case_insensitive_endswith": "ends_with",
    "case_insensitive_contains": "match_substring",
    "initcap": "utf8_capitalize",
}
# Typing Aliases
Filter = Tuple[Union[str, "Filter"], str, Union[ir.Var, str]]
