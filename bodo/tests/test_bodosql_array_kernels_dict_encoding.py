# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Test Bodo's array kernel utilities for BodoSQL with dictionary encoding
"""


import pandas as pd
import pyarrow as pa
import pytest

import bodo
from bodo.libs.bodosql_array_kernels import *
from bodo.tests.utils import SeriesOptTestPipeline, check_func, dist_IR_contains


def verify_dictionary_optimization(func, args, dict_func, output_encoded):
    """Verifies whether or not the output to a function with certain arguments
    is dictionary encoded by looking for an occurrence of a certain funciton
    that only operates on dictionary encoded arrays.

    Args:
        func (function): the function being tested
        args (any tuple): the arguments to the function
        dict_func (string): the string function being used to detect whether
        the output is dictionary encoded
        output_encoded (boolean): whether the output should be dictionary encoded

    """
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(func)
    bodo_func(*args)
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, dict_func) == output_encoded


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
    verify_dictionary_optimization(impl, (dictionary, *args), "str_capitalize", True)


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                "editdistance_no_max",
                (
                    pa.array(
                        [
                            "wonderlust",
                            "wonder",
                            None,
                            "terrible",
                            None,
                            "wendigo",
                        ]
                        * 2,
                        type=pa.dictionary(pa.int32(), pa.string()),
                    ),
                    "wonderful",
                ),
                pd.Series([3, 3, None, 8, None, 6] * 2, dtype=pd.Int32Dtype()),
            ),
            id="editdistance_no_max",
        ),
        pytest.param(
            (
                "editdistance_with_max",
                (
                    pa.array(
                        [
                            "wonderlust",
                            "wonder",
                            None,
                            "terrible",
                            None,
                            "wendigo",
                        ]
                        * 2,
                        type=pa.dictionary(pa.int32(), pa.string()),
                    ),
                    "wonderful",
                    5,
                ),
                pd.Series([3, 3, None, 5, None, 5] * 2, dtype=pd.Int32Dtype()),
            ),
            id="editdistance_with_max",
        ),
        pytest.param(
            (
                "instr",
                (
                    pa.array(
                        [
                            "alphabet soup is delicious",
                            "yay",
                            None,
                            " a b c ",
                            None,
                            "alphabet soup is delicious",
                        ]
                        * 2,
                        type=pa.dictionary(pa.int32(), pa.string()),
                    ),
                    " ",
                ),
                pd.Series([9, 0, None, 1, None, 9] * 2, dtype=pd.Int32Dtype()),
            ),
            id="instr",
        ),
        pytest.param(
            (
                "strcmp",
                (
                    pa.array(
                        [
                            "alpha",
                            "beta",
                            None,
                            "alphabet",
                            None,
                            "alphabets",
                        ]
                        * 2,
                        type=pa.dictionary(pa.int32(), pa.string()),
                    ),
                    "alphabet",
                ),
                pd.Series([-1, 1, None, 0, None, 1] * 2, dtype=pd.Int32Dtype()),
            ),
            id="strcmp",
        ),
        pytest.param(
            (
                "ord_ascii",
                (
                    pa.array(
                        [
                            "abc",
                            "DEF",
                            None,
                            "a",
                            None,
                            "!@#$%",
                        ]
                        * 2,
                        type=pa.dictionary(pa.int32(), pa.string()),
                    ),
                ),
                pd.Series([97, 68, None, 97, None, 33] * 2, dtype=pd.Int32Dtype()),
            ),
            id="ord_ascii",
        ),
    ],
)
def test_dict_str2int(args):
    def impl1(s, t):
        return bodo.libs.bodosql_array_kernels.editdistance_no_max(s, t)

    def impl2(s, t, n):
        return bodo.libs.bodosql_array_kernels.editdistance_with_max(s, t, n)

    def impl3(s, t):
        return bodo.libs.bodosql_array_kernels.instr(s, t)

    def impl4(s, t):
        return bodo.libs.bodosql_array_kernels.strcmp(s, t)

    def impl5(s):
        return bodo.libs.bodosql_array_kernels.ord_ascii(s)

    func, args, answer = args
    impl = {
        "editdistance_no_max": impl1,
        "editdistance_with_max": impl2,
        "instr": impl3,
        "strcmp": impl4,
        "ord_ascii": impl5,
    }[func]
    check_func(
        impl,
        args,
        py_output=answer,
        check_dtype=False,
        reset_index=True,
        additional_compiler_arguments={"pipeline_class": SeriesOptTestPipeline},
    )


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
    verify_dictionary_optimization(
        impl, (arr, to_replace, replace_with), "str_capitalize", output_encoded
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                (
                    pa.array(
                        [
                            "alpha beta gamma delta",
                            "alpha beta gamma",
                            None,
                            "alphabet soup is very delicious",
                            None,
                            "alpha beta gamma delta",
                        ]
                        * 2,
                        type=pa.dictionary(pa.int32(), pa.string()),
                    ),
                    " ",
                    -2,
                ),
                pd.Series(["Gamma", "Beta", None, "Very", None, "Gamma"] * 2),
                True,
            ),
            id="dict_scalar_scalar",
        ),
        pytest.param(
            (
                (
                    pa.array(
                        [
                            "alpha beta gamma delta",
                            "alpha beta gamma",
                            None,
                            "alphabet soup is very delicious",
                            None,
                            "alpha beta gamma delta",
                        ]
                        * 2,
                        type=pa.dictionary(pa.int32(), pa.string()),
                    ),
                    "a",
                    pd.Series([4, 4, 4, 4, 2, 2, 2, 2, 3, 3, 3, 3]),
                ),
                pd.Series(
                    [
                        " g",
                        " g",
                        None,
                        "",
                        None,
                        "Lph",
                        "Lph",
                        "Lph",
                        None,
                        "Bet soup is very delicious",
                        None,
                        " bet",
                    ]
                ),
                False,
            ),
            id="dict_scalar_vector",
        ),
    ],
)
def test_dict_split_part(args):
    def impl(source, delim, part):
        return bodo.libs.bodosql_array_kernels.split_part(
            source, delim, part
        ).str.capitalize()

    args, answer, output_encoded = args
    check_func(
        impl,
        args,
        py_output=answer,
        check_dtype=False,
        reset_index=True,
        additional_compiler_arguments={"pipeline_class": SeriesOptTestPipeline},
    )
    verify_dictionary_optimization(impl, args, "str_capitalize", output_encoded)


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                (
                    pa.array(
                        [
                            "25..115..22.13",
                            "168.....227....48.212.",
                            "25,\n115,\n22,\n13,\n",
                            "200  107   58 89",
                            "200.107..58......89",
                            None,
                        ]
                        * 2,
                        type=pa.dictionary(pa.int32(), pa.string()),
                    ),
                    " \n\t,.-",
                    2,
                ),
                pd.Series(["115", "227", "115", "107", "107", None] * 2),
                True,
            ),
            id="dict_scalar_scalar",
        ),
        pytest.param(
            (
                (
                    pa.array(
                        [
                            "25..115..22.13",
                            "168.....227....48.212.",
                            "25,\n115,\n22,\n13,\n",
                            "200  107   58 89",
                            "200.107..58......89",
                            None,
                        ]
                        * 2,
                        type=pa.dictionary(pa.int32(), pa.string()),
                    ),
                    " \n\t,.-",
                    pd.Series([2] * 12),
                ),
                pd.Series(["115", "227", "115", "107", "107", None] * 2),
                False,
            ),
            id="dict_scalar_vector",
        ),
    ],
)
def test_dict_strtok(args):
    def impl(source, delim, part):
        return bodo.libs.bodosql_array_kernels.strtok(
            source, delim, part
        ).str.capitalize()

    args, answer, output_encoded = args
    check_func(
        impl,
        args,
        py_output=answer,
        check_dtype=False,
        reset_index=True,
        additional_compiler_arguments={"pipeline_class": SeriesOptTestPipeline},
    )
    verify_dictionary_optimization(impl, args, "str_capitalize", output_encoded)


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
    verify_dictionary_optimization(impl, A, "str_capitalize", output_encoded)


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
    verify_dictionary_optimization(impl, A, "str_capitalize", output_encoded)


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                (
                    pa.array(
                        ["E", "I", None, "E", None, "I", "O"] * 2,
                        type=pa.dictionary(pa.int32(), pa.string()),
                    ),
                    "E",
                    "e",
                    "O",
                    "fudge",
                ),
                True,
            ),
            id="dict_scalar_no_nulls_no_default",
        ),
        pytest.param(
            (
                (
                    pa.array(
                        ["E", "I", None, "E", None, "I", "O"] * 2,
                        type=pa.dictionary(pa.int32(), pa.string()),
                    ),
                    "E",
                    "e",
                    None,
                    "uh oh",
                ),
                False,
            ),
            id="dict_scalar_with_null_input",
        ),
        pytest.param(
            (
                (
                    pa.array(
                        ["E", "I", None, "E", None, "I", "O"] * 2,
                        type=pa.dictionary(pa.int32(), pa.string()),
                    ),
                    "E",
                    None,
                    "I",
                    "i",
                ),
                True,
            ),
            id="dict_scalar_with_null_output",
        ),
        pytest.param(
            (
                (
                    pa.array(
                        ["E", "I", None, "E", None, "I", "O"] * 2,
                        type=pa.dictionary(pa.int32(), pa.string()),
                    ),
                    "E",
                    "e",
                    "O",
                    "o",
                    "!!!",
                ),
                False,
            ),
            id="dict_scalar_with_default",
        ),
        pytest.param(
            (
                (
                    pa.array(
                        ["E", "I", None, "E", None, "I", "O"] * 2,
                        type=pa.dictionary(pa.int32(), pa.string()),
                    ),
                    pd.Series(["E", "I"] * 7),
                    "A",
                    "O",
                    "B",
                ),
                False,
            ),
            id="dict_vector",
        ),
    ],
)
def test_dict_decode(args):
    def impl5(A, B, C, D, E):
        return bodo.libs.bodosql_array_kernels.decode((A, B, C, D, E)).str.capitalize()

    def impl6(A, B, C, D, E, F):
        return bodo.libs.bodosql_array_kernels.decode(
            (A, B, C, D, E, F)
        ).str.capitalize()

    # Simulates DECODE on a single row
    def decode_scalar_fn(*args):
        for i in range(1, len(args) - 1, 2):
            if (str(args[0]) == "None" and pd.isna(args[i])) or (
                str(args[0]) != "None"
                and not pd.isna(args[i])
                and str(args[0]) == str(args[i])
            ):
                if args[i + 1] == None:
                    return None
                return str(args[i + 1]).capitalize()
        if len(args) % 2 == 0:
            if args[-1] == None:
                return None
            return str(args[-1]).capitalize()

    A, output_encoded = args

    decode_answer = vectorized_sol(A, decode_scalar_fn, None)
    impl = impl5 if len(A) == 5 else impl6
    check_func(
        impl,
        A,
        py_output=decode_answer,
        check_dtype=False,
        reset_index=True,
        additional_compiler_arguments={"pipeline_class": SeriesOptTestPipeline},
    )
    verify_dictionary_optimization(impl, A, "str_capitalize", output_encoded)


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                (
                    pa.array(
                        [
                            "15-112 15-122 15-150",
                            "15-210 15-213 15-251\n15-281",
                            "15-312 15-330",
                            "15-451",
                            None,
                        ]
                        * 2,
                        type=pa.dictionary(pa.int32(), pa.string()),
                    ),
                    " \n-",
                    ",,",
                ),
                pd.Series(
                    [
                        "15112,15122,15150",
                        "15210,15213,15251,15281",
                        "15312,15330",
                        "15451",
                        None,
                    ]
                    * 2
                ),
                True,
            ),
            id="dict_scalar_scalar",
        ),
        pytest.param(
            (
                (
                    pa.array(
                        [
                            "15-112 15-122 15-150",
                            "15-210 15-213 15-251\n15-281",
                            "15-312 15-330",
                            "15-451",
                            None,
                        ]
                        * 2,
                        type=pa.dictionary(pa.int32(), pa.string()),
                    ),
                    " \n-",
                    pd.Series([",,"] * 12),
                ),
                pd.Series(
                    [
                        "15112,15122,15150",
                        "15210,15213,15251,15281",
                        "15312,15330",
                        "15451",
                        None,
                    ]
                    * 2
                ),
                False,
            ),
            id="dict_scalar_vector",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_dict_translate(args):
    def impl(arr, source, target):
        return bodo.libs.bodosql_array_kernels.translate(
            arr, source, target
        ).str.capitalize()

    args, answer, output_encoded = args

    check_func(
        impl,
        args,
        py_output=answer,
        check_dtype=False,
        reset_index=True,
        additional_compiler_arguments={"pipeline_class": SeriesOptTestPipeline},
    )
    verify_dictionary_optimization(impl, args, "str_capitalize", output_encoded)


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                (
                    pa.array(
                        [
                            "alpha beta gamma delta",
                            "ALPHABET SOUP IS DELICIOUS",
                            "epsilon,theta.iota;pi\tsigma",
                            "The cheese will be served when I want it served. And I want it served now.",
                            None,
                        ]
                        * 2,
                        type=pa.dictionary(pa.int32(), pa.string()),
                    ),
                    " ",
                ),
                pd.Series(
                    [
                        "Alpha Beta Gamma Delta",
                        "Alphabet Soup Is Delicious",
                        "Epsilon,theta.iota;pi\tsigma",
                        "The Cheese Will Be Served When I Want It Served. And I Want It Served Now.",
                        None,
                    ]
                    * 2
                ),
                True,
            ),
            id="dict_scalar",
        ),
        pytest.param(
            (
                (
                    pa.array(
                        [
                            "alpha beta gamma delta",
                            "ALPHABET SOUP IS DELICIOUS",
                            "epsilon,theta.iota;pi\tsigma",
                            "The cheese will be served when I want it served. And I want it served now.",
                            None,
                        ]
                        * 2,
                        type=pa.dictionary(pa.int32(), pa.string()),
                    ),
                    pd.Series([" "] * 12),
                ),
                pd.Series(
                    [
                        "Alpha Beta Gamma Delta",
                        "Alphabet Soup Is Delicious",
                        "Epsilon,theta.iota;pi\tsigma",
                        "The Cheese Will Be Served When I Want It Served. And I Want It Served Now.",
                        None,
                    ]
                    * 2
                ),
                False,
            ),
            id="dict_vector",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_dict_initcap(args):
    def impl(arr, delim):
        return bodo.libs.bodosql_array_kernels.initcap(arr, delim).str.strip()

    args, answer, output_encoded = args

    check_func(
        impl,
        args,
        py_output=answer,
        check_dtype=False,
        reset_index=True,
        additional_compiler_arguments={"pipeline_class": SeriesOptTestPipeline},
    )

    verify_dictionary_optimization(impl, args, "str_strip", output_encoded)
