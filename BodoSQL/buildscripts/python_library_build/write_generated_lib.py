# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
    File that handles writing the generated library/test cases to their respective files
"""
import os

import buildscripts.python_library_build.generate_libraries
import buildscripts.python_library_build.generate_library_tests

# If these are changed the gitignore will also need to be updated
# If the file is moved, these will also need to be updated
REPO_ROOT = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir
)
# TODO: put the generated library/testing path in a global variable somewhere that is commonly accessible
GENERATED_LIB_FILE_PATH = os.path.join(REPO_ROOT, "bodosql", "libs", "generated_lib.py")
GENERATED_LIB_TESTCASES_PATH = os.path.join(
    REPO_ROOT, "bodosql", "tests", "test_generated_lib_testcases.py"
)


def write_generated_lib(lib_string, file_path=GENERATED_LIB_FILE_PATH):
    """Writes the library string to the specified file"""
    # w option will overwrite the file if it already exists
    lib_file = open(file_path, "w")
    lib_file.write(lib_string)
    lib_file.close()


def write_generated_lib_testcases(
    testcases_string,
    file_path=GENERATED_LIB_TESTCASES_PATH,
    library_file_path=None,
    module_name=None,
):
    """Writes the generated library testcases to the specified file location

    If library file path and module name are BOTH specified, it will import the file found at library_file_path
    as a module with the suplied module_name. Otherwise, the arguments have no effect
    """
    # w option will overwrite the file if it already exists
    lib_file = open(file_path, "w")

    # if both module name and library_file_path != None, we need to prepend an import of the module with the specified module name
    if module_name != None and library_file_path != None:
        # Code for importing a module from an arbitrary file location is modified from here:
        # https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
        import_text = f"""
import importlib.util
spec = importlib.util.spec_from_file_location("__temporary_module_name_", \"{library_file_path}\")
{module_name} = importlib.util.module_from_spec(spec)
spec.loader.exec_module({module_name})
"""
        testcases_string = import_text + testcases_string

    lib_file.write(testcases_string)
    lib_file.close()


def generate_standard_python_fn_call(fn_name):
    """given the name of the function call, returns a lambda function that
    returns the correctly formatted function call when suplied a list of argument"""

    def impl(args_list):
        args_list = ", ".join(args_list)
        return f"{fn_name}({args_list})"

    return impl


def generate_standard_method_call(method_name):
    """given the name of the method call, returns a lambda function that
    returns the correctly formated method call when suplied a list of arguments"""

    def impl(args_list):
        method_args_string = ", ".join(args_list[1:])
        return f"{args_list[0]}.{method_name}({method_args_string})"

    return impl


def generate_atribute_reference(atribute_name):
    """given the name of the atribute, returns a lambda function that returns
    the correctly formated atribute reference when suplied a list of arguments"""
    return lambda x: f"{x[0]}.{atribute_name}"


def generate_and_write_library():
    """generates the function library, and writes it to the expected location.
    This function should be called within setup.py"""

    library_fns_info = (
        [
            ("not", generate_standard_python_fn_call("not"), 1),
            ("addition", generate_standard_python_fn_call("operator.add"), 2),
            ("subtraction", generate_standard_python_fn_call("operator.sub"), 2),
            ("multiplication", generate_standard_python_fn_call("operator.mul"), 2),
            ("true_division", generate_standard_python_fn_call("np.true_divide"), 2),
            ("modulo", generate_standard_python_fn_call("np.mod"), 2),
            ("power", generate_standard_python_fn_call("operator.pow"), 2),
            ("equal", generate_standard_python_fn_call("operator.eq"), 2),
            ("not_equal", generate_standard_python_fn_call("operator.ne"), 2),
            ("less_than", generate_standard_python_fn_call("operator.lt"), 2),
            ("less_than_or_equal", generate_standard_python_fn_call("operator.le"), 2),
            ("greater_than", generate_standard_python_fn_call("operator.gt"), 2),
            (
                "greater_than_or_equal",
                generate_standard_python_fn_call("operator.ge"),
                2,
            ),
            # library wrappers
            (
                "sql_to_python",
                generate_standard_python_fn_call("bodosql.libs.regex.sql_to_python"),
                1,
            ),
            # string Fn's
            ("strip", generate_standard_method_call("strip"), 2),
            ("lstrip", generate_standard_method_call("lstrip"), 2),
            ("rstrip", generate_standard_method_call("rstrip"), 2),
            ("len", generate_standard_python_fn_call("len"), 1),
            ("upper", generate_standard_method_call("upper"), 1),
            ("lower", generate_standard_method_call("lower"), 1),
            ("replace", generate_standard_method_call("replace"), 3),
            # ("spaces", generate_standard_python_fn_call("bodo.libs.bodosql_string_array_kernels.space_util"), 1),
            # ("reverse", generate_standard_python_fn_call("bodo.libs.bodosql_string_array_kernels.reverse_util"), 1),
            # stuff for like
            ("in", (lambda args_list: f"({args_list[0]} in {args_list[1]})"), 2),
            (
                "re_match",
                (lambda args_list: f"bool(re.match({args_list[0]}, {args_list[1]}))"),
                2,
            ),
            # stuff for case insensitive like
            (
                "in_nocase",
                (
                    lambda args_list: f"({args_list[0]}.lower() in {args_list[1]}.lower())"
                ),
                2,
            ),
            (
                "re_match_nocase",
                (
                    lambda args_list: f"bool(re.match({args_list[0]}, {args_list[1]}, flags=re.I))"
                ),
                2,
            ),
            # DATETIME fns
            ("timestamp_dayfloor", lambda x: f"{x[0]}.floor(freq='D')", 1),
            ("strftime", generate_standard_method_call("strftime"), 2),
            (
                "pd_to_datetime_with_format",
                generate_standard_python_fn_call(
                    "bodosql.libs.sql_operators.pd_to_datetime_with_format"
                ),
                2,
            ),
            (
                "pd_Timestamp_single_value",
                generate_standard_python_fn_call("pd.Timestamp"),
                1,
            ),
            (
                "pd_Timestamp_single_value_with_second_unit",
                lambda args: f"pd.Timestamp({args[0]}, unit='s')",
                1,
            ),
            (
                "pd_Timestamp_single_value_with_day_unit",
                lambda args: f"pd.Timestamp({args[0]}, unit='D')",
                1,
            ),
            (
                "pd_Timestamp_single_value_with_year_unit",
                lambda args: f"pd.Timestamp({args[0]}, unit='Y')",
                1,
            ),
            ("pd_timedelta_days", generate_atribute_reference("days"), 1),
            (
                "pd_timedelta_total_seconds",
                generate_standard_method_call("total_seconds"),
                1,
            ),
            ("yearofweek", generate_atribute_reference("isocalendar()[0]"), 1),
        ]
        + [
            (x, generate_atribute_reference(x), 1)
            for x in [
                "weekofyear",
                "dayofyear",
                "nanosecond",
                "microsecond",
                "millisecond",
                "second",
                "minute",
                "hour",
                "day",
                "month",
                "quarter",
                "year",
            ]
        ]
        + [
            (
                "dayofweek",
                generate_standard_python_fn_call("bodosql.libs.sql_operators.sql_dow"),
                1,
            ),
            # Scalar Conversion functions
            ("scalar_conv_bool", generate_standard_python_fn_call("np.bool_"), 1),
            ("scalar_conv_int8", generate_standard_python_fn_call("np.int8"), 1),
            ("scalar_conv_int16", generate_standard_python_fn_call("np.int16"), 1),
            ("scalar_conv_int32", generate_standard_python_fn_call("np.int32"), 1),
            ("scalar_conv_int64", generate_standard_python_fn_call("np.int64"), 1),
            ("scalar_conv_str", generate_standard_python_fn_call("str"), 1),
            (
                "scalar_conv_binary",
                generate_standard_python_fn_call(
                    "bodosql.libs.binary_functions.cast_binary"
                ),
                1,
            ),
            ("scalar_conv_float32", generate_standard_python_fn_call("np.float32"), 1),
            ("scalar_conv_float64", generate_standard_python_fn_call("np.float64"), 1),
            ("scalar_conv_str", generate_standard_python_fn_call("str"), 1),
        ]
    )

    library_fn_strings = []

    for (fn_name, fn_expr, num_args) in library_fns_info:
        library_fn_strings.append(
            buildscripts.python_library_build.generate_libraries.generate_library_fn_string(
                fn_name, fn_expr, num_args
            )
        )

    header_string = """
# Copyright (C) 2022 Bodo Inc. All rights reserved.
\"\"\" There are a large number of operators that need a wrapper that returns null if any of the input arguments are null,
and otherwise return the result of the original function. This file is an automatically generated file, that contains
these library functions.
DO NOT MANUALLY CHANGE THIS FILE!
\"\"\"
import bodosql
import bodo
import operator
import numpy as np
import pandas as pd
import re
from numba import generated_jit
"""
    library_string = "\n".join([header_string] + library_fn_strings)
    write_generated_lib(library_string)


def nested_str(L):
    return [str(x) for x in L]


def library_fn_from_name(fn_name):
    lib_string_path = "bodosql.libs.generated_lib."
    return (
        lib_string_path
        + buildscripts.python_library_build.generate_libraries.bodosql_library_fn_name(
            fn_name
        )
    )
