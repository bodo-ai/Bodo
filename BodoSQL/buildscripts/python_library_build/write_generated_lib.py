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
    REPO_ROOT, "bodosql", "tests", "generated_lib_testcases.py"
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
            # stuff for like
            ("in", (lambda args_list: f"({args_list[0]} in {args_list[1]})"), 2),
            (
                "re_match",
                (lambda args_list: f"bool(re.match({args_list[0]}, {args_list[1]}))"),
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
                "microsecond",
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
            ("scalar_conv_float32", generate_standard_python_fn_call("np.float32"), 1),
            ("scalar_conv_float64", generate_standard_python_fn_call("np.float64"), 1),
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


def generate_and_write_library_tests():
    """generates the function library tests, and writes them to the expected location.
    This function should be called within setup.py"""

    library_tests_info = [
        (
            "not",
            library_fn_from_name("not"),
            [[True], [False]],
            [False, True],
            False,
        ),
        (
            "addition",
            library_fn_from_name("addition"),
            [[1, 2], [3, -7], [0, 0.0], [-9.23, 12.898]],
            [3, -4, 0, 3.668],
            True,
        ),
        (
            "concatination",
            library_fn_from_name("addition"),
            [["'hello'", "' world'"], ["[1,2]", "[3,4]"]],
            ["'hello world'", [1, 2, 3, 4]],
            False,
        ),
        (
            "subtraction",
            library_fn_from_name("subtraction"),
            [[1, 2], [3, -7], [0, 0.0], [-9.23, 12.898]],
            [-1, 10, 0, -22.128],
            True,
        ),
        (
            "multiplication",
            library_fn_from_name("multiplication"),
            [[1, 2], [3, -12], [0.0, 100.0], [-1.123, 2.765]],
            [2, -36, 0, -3.105095],
            True,
        ),
        (
            "true_division",
            library_fn_from_name("true_division"),
            [[1, 0.0], [0, 4], [0.123, 0.7]],
            ["np.inf", 0, 0.17571428571],
            True,
        ),
        # 0 mod 0 is 0 in mysql, and np.mod returns 0 in this case by default
        (
            "modulo",
            library_fn_from_name("modulo"),
            [[0, 0], [1, 4], [13.23, 2.6], [5, 3]],
            [0, 0, 0.23, 2],
            False,
        ),
        (
            "power",
            library_fn_from_name("power"),
            [[1, 2], [2, 4], [0.876, 1.8]],
            [1, 2**4, 0.876**1.8],
            True,
        ),
        (
            "equal",
            library_fn_from_name("equal"),
            [[1, 2], [True, True], ["'hello world'", "'hello'"]],
            [False, True, False],
            False,
        ),
        (
            "not_equal",
            library_fn_from_name("not_equal"),
            [[1, 2], [True, True], ["'hello world'", "'hello'"]],
            [True, False, True],
            False,
        ),
        (
            "less_than",
            library_fn_from_name("less_than"),
            [[1, 2], [True, False], ["'hi'", "'hi'"]],
            [True, False, False],
            False,
        ),
        (
            "less_than_or_equal",
            library_fn_from_name("less_than_or_equal"),
            [[1, 2], [True, False], ["'hi'", "'hi'"]],
            [True, False, True],
            False,
        ),
        (
            "greater_than",
            library_fn_from_name("greater_than"),
            [[1, 2], [True, False], ["'hi'", "'hi'"]],
            [False, True, False],
            False,
        ),
        (
            "greater_than_or_equal",
            library_fn_from_name("greater_than_or_equal"),
            [[1, 2], [True, False], ["'hi'", "'hi'"]],
            [False, True, True],
            False,
        ),
        # string Fn's
        (
            "strip",
            library_fn_from_name("strip"),
            [["'   hello'", "' '"], ["'hello   '", "' '"], ["'   hello   '", "' '"]],
            ["'hello'"] * 3,
            False,
        ),
        (
            "lstrip",
            library_fn_from_name("lstrip"),
            [["'   hello'", "' '"], ["'hello   '", "' '"], ["'   hello   '", "' '"]],
            ["'hello'", "'hello   '", "'hello   '"],
            False,
        ),
        (
            "rstrip",
            library_fn_from_name("rstrip"),
            [["'   hello'", "' '"], ["'hello   '", "' '"], ["'   hello   '", "' '"]],
            ["'   hello'", "'hello'", "'   hello'"],
            False,
        ),
        (
            "len",
            library_fn_from_name("len"),
            [
                ["'hello'"],
                ["[1,2,3]"],
            ],
            ["5", "3"],
            False,
        ),
        (
            "upper",
            library_fn_from_name("upper"),
            [["'HELLO'"], ["'HeLlO'"], ["'hello'"]],
            ["'HELLO'"] * 3,
            False,
        ),
        (
            "lower",
            library_fn_from_name("lower"),
            [["'HELLO'"], ["'HeLlO'"], ["'hello'"]],
            ["'hello'"] * 3,
            False,
        ),
        (
            "replace",
            library_fn_from_name("replace"),
            [["'hello world'", "' '", "'_'"], ["'hello world'", "'o'", "'0'"]],
            ["'hello_world'", "'hell0 w0rld'"],
            False,
        ),
        (
            "spaces",
            library_fn_from_name("spaces"),
            [[12], [0], [-10], [2]],
            [f"'{12 * ' '}'", "''", "''", "'  '"],
            False,
        ),
        (
            "reverse",
            library_fn_from_name("reverse"),
            [["'hello'"], ["[1,2,3]"], ["''"]],
            ["'olleh'", "[3,2,1]", "''"],
            False,
        ),
        # stuff for like
        (
            "in",
            library_fn_from_name("in"),
            [["1", "[1,2,3,4]"], ["'e'", "'hello'"]],
            [True, True],
            False,
        ),
        # seems to be some sort of
        # an issue using floor with the frequency argument on timestamps, see BE-1022
        # (
        #     "timestamp_dayfloor",
        #     library_fn_from_name("timestamp_dayfloor"),
        #     [
        #         ["pd.Timestamp('1680-02-13 11:24:02')"],
        #         ["pd.Timestamp('1680-07-14 15:44:04.498582')"],
        #     ],
        #     ["pd.Timestamp('1680-02-13')", "pd.Timestamp('1680-07-14')"],
        #     False,
        # ),
        (
            "strftime",
            library_fn_from_name("strftime"),
            [
                ["pd.Timestamp('2021-07-14 15:44:04.498582')", "'%Y-%m-%d'"],
                [
                    "pd.Timestamp('2021-07-14 15:44:04.498582')",
                    "'%d, %b, %y. Time: %H:%M'",
                ],
            ],
            ["'2021-07-14'", "'14, Jul, 21. Time: 15:44'"],
            False,
        ),
        (
            "pd_to_datetime_with_format",
            library_fn_from_name("pd_to_datetime_with_format"),
            [
                ["'2021-07-14'", "'%Y-%m-%d'"],
                ["'14, Jul, 21. Time: 15:44'", "'%d, %b, %y. Time: %H:%M'"],
            ],
            ["pd.Timestamp('2021-07-14')", "pd.Timestamp('2021-07-14 15:44:00')"],
            False,
        ),
        (
            "pd_Timestamp_single_value",
            library_fn_from_name("pd_Timestamp_single_value"),
            [[1000], ["'2021-07-14 15:44:04.498582'"]],
            ["pd.Timestamp(1000)", "pd.Timestamp('2021-07-14 15:44:04.498582')"],
            False,
        ),
        (
            "pd_Timestamp_single_value_with_second_unit",
            library_fn_from_name("pd_Timestamp_single_value_with_second_unit"),
            [[100], [12], [14]],
            [
                "pd.Timestamp(100, unit='s')",
                "pd.Timestamp(12, unit='s')",
                "pd.Timestamp(14, unit='s')",
            ],
            False,
        ),
        (
            "pd_Timestamp_single_value_with_day_unit",
            library_fn_from_name("pd_Timestamp_single_value_with_day_unit"),
            [[100], [12], [14]],
            [
                "pd.Timestamp(100, unit='D')",
                "pd.Timestamp(12, unit='D')",
                "pd.Timestamp(14, unit='D')",
            ],
            False,
        ),
        (
            "pd_Timestamp_single_value_with_year_unit",
            library_fn_from_name("pd_Timestamp_single_value_with_year_unit"),
            [[100], [12], [14]],
            [
                "pd.Timestamp(100, unit='Y')",
                "pd.Timestamp(12, unit='Y')",
                "pd.Timestamp(14, unit='Y')",
            ],
            False,
        ),
        (
            "pd_Timestamp_y_m_d",
            library_fn_from_name("pd_Timestamp_y_m_d"),
            [[2020, 2, 12], [2000, 9, 13], [2018, 10, 1]],
            [
                "pd.Timestamp(year=2020, month=2, day=12)",
                "pd.Timestamp(year=2000, month=9, day=13)",
                "pd.Timestamp(year=2018, month=10, day=1)",
            ],
            False,
        ),
        (
            "pd_timedelta_days",
            library_fn_from_name("pd_timedelta_days"),
            [
                ["pd.Timedelta(100)"],
                ["pd.Timedelta(100, unit='D')"],
                ["pd.Timedelta(3, unit='W')"],
            ],
            [0, 100, 21],
            False,
        ),
        (
            "pd_timedelta_total_seconds",
            library_fn_from_name("pd_timedelta_total_seconds"),
            [
                ["pd.Timedelta(100)"],
                ["pd.Timedelta(100, unit='s')"],
                ["pd.Timedelta(1, unit='D')"],
            ],
            [0, 100, 86400],
            False,
        ),
        # Extract Fn's
        (
            "dayofweek",
            library_fn_from_name("dayofweek"),
            [
                ["pd.Timestamp('2021-07-14 15:44:04.498582')"],
                ["pd.Timestamp('2010-01-01')"],
                ["pd.Timestamp('2010-01-03')"],
                ["pd.Timestamp('2010-01-04')"],
            ],
            [3, 5, 7, 1],
            False,
        ),
        (
            "dayofyear",
            library_fn_from_name("dayofyear"),
            [
                ["pd.Timestamp('2021-07-14 15:44:04.498582')"],
                ["pd.Timestamp('2010-01-01')"],
            ],
            [195, 1],
            False,
        ),
        (
            "weekofyear",
            library_fn_from_name("weekofyear"),
            [
                ["pd.Timestamp('2021-07-14 15:44:04.498582')"],
                ["pd.Timestamp('2010-01-01')"],
            ],
            [29, 1],
            False,
        ),
        (
            "microsecond",
            library_fn_from_name("microsecond"),
            [
                ["pd.Timestamp('2021-07-14 15:44:04.498582')"],
                ["pd.Timestamp('2010-01-01')"],
            ],
            [498582, 0],
            False,
        ),
        (
            "second",
            library_fn_from_name("second"),
            [
                ["pd.Timestamp('2021-07-14 15:44:04.498582')"],
                ["pd.Timestamp('2010-01-01')"],
            ],
            [4, 0],
            False,
        ),
        (
            "minute",
            library_fn_from_name("minute"),
            [
                ["pd.Timestamp('2021-07-14 15:44:04.498582')"],
                ["pd.Timestamp('2010-01-01')"],
            ],
            [44, 0],
            False,
        ),
        (
            "hour",
            library_fn_from_name("hour"),
            [
                ["pd.Timestamp('2021-07-14 15:44:04.498582')"],
                ["pd.Timestamp('2010-01-01')"],
            ],
            [15, 0],
            False,
        ),
        (
            "day",
            library_fn_from_name("day"),
            [
                ["pd.Timestamp('2021-07-14 15:44:04.498582')"],
                ["pd.Timestamp('2010-01-01')"],
            ],
            [14, 1],
            False,
        ),
        (
            "month",
            library_fn_from_name("month"),
            [
                ["pd.Timestamp('2021-07-14 15:44:04.498582')"],
                ["pd.Timestamp('2010-01-01')"],
            ],
            [7, 1],
            False,
        ),
        (
            "quarter",
            library_fn_from_name("quarter"),
            [
                ["pd.Timestamp('2021-07-14 15:44:04.498582')"],
                ["pd.Timestamp('2010-01-01')"],
            ],
            [3, 1],
            False,
        ),
        (
            "year",
            library_fn_from_name("year"),
            [
                ["pd.Timestamp('2021-07-14 15:44:04.498582')"],
                ["pd.Timestamp('2010-01-01')"],
            ],
            [2021, 2010],
            False,
        ),
    ] + [
        (
            "sql_to_python",
            library_fn_from_name("sql_to_python"),
            [["'%%dfwfwe'"], ["'dfwfwe%%'"], ["'%%dfwfwe%%'"]],
            ["'dfwfwe$'", "'^dfwfwe'", "'dfwfwe'"],
            False,
        ),
    ]

    library_tests = []
    for (
        fn_name,
        fn_expr,
        arg_vals,
        expected_outputs,
        use_np_isclose,
    ) in library_tests_info:
        (
            defalt_tests,
            none_tests,
            opt_tests,
        ) = buildscripts.python_library_build.generate_library_tests.generate_fn_test_strings(
            fn_expr,
            fn_name,
            # I decided to call str on the inputs, so that writing/reading the test cases was faster/easier
            list(map(nested_str, arg_vals)),
            list(map(str, expected_outputs)),
            use_np_isclose,
        )
        library_tests.append("\n".join(defalt_tests + none_tests + opt_tests))

    header_string = """
# Copyright (C) 2022 Bodo Inc. All rights reserved.
\"\"\" There are a large number of operators that need a wrapper that returns null if any of the input arguments are null,
and otherwise return the result of the original function. This file is an automatically generated file, that tests
these library functions.
DO NOT MANUALLY CHANGE THIS FILE!
\"\"\"
import bodosql
import bodo
import pytest
import numpy as np
import pandas as pd
import re
from numba import generated_jit

"""
    testfile_string = "\n".join([header_string] + library_tests)
    write_generated_lib_testcases(testfile_string)
