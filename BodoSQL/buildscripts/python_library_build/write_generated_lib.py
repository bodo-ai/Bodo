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


def write_generated_lib(lib_string, file_path=GENERATED_LIB_FILE_PATH):
    """Writes the library string to the specified file"""
    # w option will overwrite the file if it already exists
    lib_file = open(file_path, "w")
    lib_file.write(lib_string)
    lib_file.close()


def generate_standard_python_fn_call(fn_name):
    """given the name of the function call, returns a lambda function that
    returns the correctly formatted function call when supplied a list of argument"""

    def impl(args_list):
        args_list = ", ".join(args_list)
        return f"{fn_name}({args_list})"

    return impl


def generate_standard_method_call(method_name):
    """given the name of the method call, returns a lambda function that
    returns the correctly formatted method call when supplied a list of arguments"""

    def impl(args_list):
        method_args_string = ", ".join(args_list[1:])
        return f"{args_list[0]}.{method_name}({method_args_string})"

    return impl


def generate_attribute_reference(attribute_name):
    """given the name of the attribute, returns a lambda function that returns
    the correctly formatted attribute reference when supplied a list of arguments"""
    return lambda x: f"{x[0]}.{attribute_name}"


def generate_and_write_library():
    """generates the function library, and writes it to the expected location.
    This function should be called within setup.py"""

    library_fns_info = [
        # DATETIME fns
        ("strftime", generate_standard_method_call("strftime"), 2),
        (
            "pd_to_datetime_with_format",
            generate_standard_python_fn_call(
                "bodosql.libs.sql_operators.pd_to_datetime_with_format"
            ),
            2,
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
        (
            "scalar_conv_pd_to_datetime",
            generate_standard_python_fn_call("pd.to_datetime"),
            1,
        ),
        (
            "scalar_conv_pd_to_date",
            generate_standard_python_fn_call("bodosql.libs.sql_operators.pd_to_date"),
            1,
        ),
    ]

    library_fn_strings = []

    for fn_name, fn_expr, num_args in library_fns_info:
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
import numba
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
