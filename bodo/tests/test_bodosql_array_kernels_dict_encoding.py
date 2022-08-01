# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Test Bodo's array kernel utilities for BodoSQL with dictionary encoding
"""


import pandas as pd
import pyarrow as pa
import pytest

import bodo
from bodo.libs.bodosql_array_kernels import *
from bodo.tests.utils import SeriesOptTestPipeline, check_func, dist_IR_contains


@pytest.mark.slow
@pytest.mark.parametrize(
    "args",
    [
        pytest.param(("lpad", (20, "_")), id="lpad"),
        pytest.param(("rpad", (15, "🐍")), id="rpad"),
        pytest.param(("left", (5,)), id="left"),
        pytest.param(("right", (10,)), id="right"),
        pytest.param(("repeat", (3,)), id="repeat"),
        pytest.param(("reverse", ()), id="reverse"),
        pytest.param(("substring", (5, 10)), id="substring"),
        pytest.param(("substring_index", (" ", 1)), id="substring_index"),
    ],
)
def test_dict_other_string_kernels(args):
    def impl1(arg0, arg1, arg2):
        return bodo.libs.bodosql_array_kernels.lpad(arg0, arg1, arg2).str.capitalize()

    def impl2(arg0, arg1, arg2):
        return bodo.libs.bodosql_array_kernels.rpad(arg0, arg1, arg2).str.capitalize()

    def impl3(arg0, arg1):
        return bodo.libs.bodosql_array_kernels.left(arg0, arg1).str.capitalize()

    def impl4(arg0, arg1):
        return bodo.libs.bodosql_array_kernels.right(arg0, arg1).str.capitalize()

    def impl5(arg0, arg1):
        return bodo.libs.bodosql_array_kernels.repeat(arg0, arg1).str.capitalize()

    def impl6(arg0):
        return bodo.libs.bodosql_array_kernels.reverse(arg0).str.capitalize()

    def impl7(arg0, arg1, arg2):
        return bodo.libs.bodosql_array_kernels.substring(
            arg0, arg1, arg2
        ).str.capitalize()

    def impl8(arg0, arg1, arg2):
        return bodo.libs.bodosql_array_kernels.substring_index(
            arg0, arg1, arg2
        ).str.capitalize()

    # Simulates the relevent function on a single row (these are not quite
    # accurate, but work for simple inputs like the ones in the parametrization)
    def scalar_fn(func, *args):
        if any([(pd.isna(arg) or str(arg) == "None") for arg in args]):
            return None
        args = list(args)
        args[0] = str(args[0])
        if func == "lpad":
            s = args[0][: args[1]]
            return (args[2] * (args[1] - len(s)) + s).capitalize()
        elif func == "rpad":
            s = args[0][: args[1]]
            return (s + args[2] * (args[1] - len(s))).capitalize()
        elif func == "left":
            return args[0][: args[1]].capitalize()
        elif func == "right":
            return args[0][-args[1] :].capitalize()
        elif func == "repeat":
            return (args[0] * args[1]).capitalize()
        elif func == "reverse":
            return args[0][::-1].capitalize()
        elif func == "substring":
            return args[0][args[1] - 1 : args[1] + args[2] - 1].capitalize()
        elif func == "substring_index":
            return args[1].join(args[0].split(args[1])[: args[2]]).capitalize()

    dictionary = pa.array(
        [
            "alpha beta",
            "soup is very very",
            None,
            "alpha beta gamma",
            None,
            "alpha beta",
        ]
        * 2,
        type=pa.dictionary(pa.int32(), pa.string()),
    )

    func, args = args
    answer = vectorized_sol((func, dictionary, *args), scalar_fn, None)

    impl = {
        "lpad": impl1,
        "rpad": impl2,
        "left": impl3,
        "right": impl4,
        "repeat": impl5,
        "reverse": impl6,
        "substring": impl7,
        "substring_index": impl8,
    }[func]
    check_func(
        impl,
        (dictionary, *args),
        py_output=answer,
        check_dtype=False,
        reset_index=True,
        additional_compiler_arguments={"pipeline_class": SeriesOptTestPipeline},
    )
    # Make sure IR has the optimized function
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl)
    bodo_func(dictionary, *args)
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, "str_capitalize")


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pa.array(
                    [
                        "alpha beta",
                        "soup is very very",
                        None,
                        "alpha beta gamma",
                        None,
                        "alpha beta",
                    ]
                    * 2,
                    type=pa.dictionary(pa.int32(), pa.string()),
                ),
                " ",
                "🐍",
                True,
            ),
            id="dict_scalar_scalar",
        ),
        pytest.param(
            (
                "alphabet soup is so very very delicious!",
                " ",
                pa.array(
                    ["_", "bBb", " c ", "_", None, "_", " c ", None] * 2,
                    type=pa.dictionary(pa.int32(), pa.string()),
                ),
                True,
            ),
            id="scalar_scalar_dict",
        ),
        pytest.param(
            (
                "alphabet soup is so very very delicious!",
                pa.array(
                    ["_", " ", " ", "_", None, "$", "$$$", None] * 2,
                    type=pa.dictionary(pa.int32(), pa.string()),
                ),
                None,
                True,
            ),
            id="scalar_dict_null",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(
                    pa.array(
                        [
                            "alpha beta",
                            "soup is very very",
                            None,
                            "alpha beta gamma",
                            None,
                            "alpha beta",
                        ]
                        * 2,
                        type=pa.dictionary(pa.int32(), pa.string()),
                    )
                ),
                " ",
                pd.Series(["_", "$", "***"] * 4),
                False,
            ),
            id="dict_scalar_vector",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_dict_replace(args):
    arr, to_replace, replace_with, output_encoded = args

    def impl(arr, to_replace, replace_with):
        return bodo.libs.bodosql_array_kernels.replace(
            arr, to_replace, replace_with
        ).str.capitalize()

    # Simulates REPLACE on a single row
    def replace_scalar_fn(elem, to_replace, replace_with):
        if (
            pd.isna(elem)
            or pd.isna(to_replace)
            or pd.isna(replace_with)
            or "None" in [str(elem), str(to_replace), str(replace_with)]
        ):
            return None
        elif to_replace == "":
            return str(elem).capitalize()
        else:
            return str(elem).replace(str(to_replace), str(replace_with)).capitalize()

    replace_answer = vectorized_sol(
        (arr, to_replace, replace_with), replace_scalar_fn, None
    )
    check_func(
        impl,
        (arr, to_replace, replace_with),
        py_output=replace_answer,
        check_dtype=False,
        reset_index=True,
        additional_compiler_arguments={"pipeline_class": SeriesOptTestPipeline},
    )
    # Make sure IR has the optimized function if it is supposed to
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl)
    bodo_func(arr, to_replace, replace_with)
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert (dist_IR_contains(f_ir, "str_capitalize")) == output_encoded


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                (
                    pa.array(
                        [
                            "alpha beta",
                            "soup is very very",
                            None,
                            "alpha beta gamma",
                            None,
                            "alpha beta",
                        ]
                        * 2,
                        type=pa.dictionary(pa.int32(), pa.string()),
                    ),
                    " ",
                ),
                False,
            ),
            id="dict_scalar",
        ),
        pytest.param(
            (
                (
                    pa.array(
                        ["_", "bBb", "c", "_", None, "_", " c ", None] * 2,
                        type=pa.dictionary(pa.int32(), pa.string()),
                    ),
                    pd.Series(["a", "b", "c", "d"] * 4),
                ),
                False,
            ),
            id="dict_vector",
        ),
    ],
)
def test_dict_coalesce(args):
    def impl(x, y):
        return bodo.libs.bodosql_array_kernels.coalesce((x, y)).str.capitalize()

    # Simulates COALESCE on a single row
    def coalesce_scalar_fn(*args):
        for arg in args:
            if not pd.isna(arg) and str(arg) != "None":
                return str(arg).capitalize()

    A, output_encoded = args

    coalesce_answer = vectorized_sol(A, coalesce_scalar_fn, None)
    check_func(
        impl,
        A,
        py_output=coalesce_answer,
        check_dtype=False,
        reset_index=True,
        additional_compiler_arguments={"pipeline_class": SeriesOptTestPipeline},
    )
    # Make sure IR has the optimized function if it is supposed to
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl)
    bodo_func(*A)
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert (dist_IR_contains(f_ir, "str_capitalize")) == output_encoded


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                (
                    pa.array(
                        [
                            "alpha beta",
                            "soup is very very",
                            None,
                            "alpha beta gamma",
                            None,
                            "alpha beta",
                        ]
                        * 2,
                        type=pa.dictionary(pa.int32(), pa.string()),
                    ),
                    " ",
                ),
                True,
            ),
            id="dict_scalar",
        ),
        pytest.param(
            (
                (
                    pa.array(
                        ["_", "bBb", "c", "_", None, "_", " c ", None] * 2,
                        type=pa.dictionary(pa.int32(), pa.string()),
                    ),
                    pd.Series(["a", "b", "c", "d"] * 4),
                ),
                False,
            ),
            id="dict_vector",
        ),
    ],
)
def test_dict_nullif(args):
    def impl(x, y):
        return bodo.libs.bodosql_array_kernels.nullif(x, y).str.capitalize()

    # Simulates NULLIF on a single row
    def nullif_scalar_fn(x, y):
        if not pd.isna(x) and str(x) != "None" and str(x) != str(y):
            return str(x).capitalize()
        else:
            return None

    A, output_encoded = args

    nullif_answer = vectorized_sol(A, nullif_scalar_fn, None)
    check_func(
        impl,
        A,
        py_output=nullif_answer,
        check_dtype=False,
        reset_index=True,
        additional_compiler_arguments={"pipeline_class": SeriesOptTestPipeline},
    )
    # Make sure IR has the optimized function if it is supposed to
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl)
    bodo_func(*A)
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert (dist_IR_contains(f_ir, "str_capitalize")) == output_encoded
