# Copyright (C) 2023 Bodo Inc. All rights reserved.

import string

import numpy as np
import pandas as pd
import pytest

import bodosql
from bodo.tests.utils import check_func
from bodosql.kernels.listagg import bodosql_listagg, bodosql_listagg_seq


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.DataFrame(
                    {
                        "B": np.arange(9),
                        "A": ["½⅓¼⅕⅙⅐⅛⅑ ⅔⅖ ¾⅗ ⅘ ⅚⅝ ⅞", "₩", None] * 3,
                    }
                ),
                "_._",
            ),
            id="simple_not_utf8",
        ),
        pytest.param(
            (
                pd.DataFrame(
                    {
                        "B": np.arange(9),
                        "A": ["A", "B", "C"] * 3,
                    }
                ),
                ", ",
            ),
            id="simple_utf8",
        ),
    ],
)
def test_listagg_seq_simple(args, memory_leak_check):
    """
    Simple unit test for the internal listagg_seq function.

    This function should not be called directly,
    but it is called by the dataframe method,
    which handles re-ordering the input columns of the dataframe
    """

    input_df, sep = args

    expected_output = sep.join(input_df["A"].dropna())

    def impl(df):
        return bodosql_listagg_seq(df, (True,), (True,), sep)

    check_func(impl, (input_df,), py_output=expected_output, only_seq=True)


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.DataFrame(
                    {
                        "B": np.arange(9),
                        "agg_col": ["½⅓¼⅕⅙⅐⅛⅑ ⅔⅖ ¾⅗ ⅘ ⅚⅝ ⅞", "₩", None] * 3,
                    }
                ),
                ("B",),
                (True,),
                ("last",),
                "_._",
                "_._".join(["½⅓¼⅕⅙⅐⅛⅑ ⅔⅖ ¾⅗ ⅘ ⅚⅝ ⅞", "₩"] * 3),
            ),
            id="simple_not_utf8",
        ),
        pytest.param(
            (
                pd.DataFrame(
                    {
                        "A": [None] * 2 + [1] + [None] * 7,
                        "B": [None, 1, None, 1, None, 1, None, 1, None, 1],
                        "C": [97, 1, 98, 1, 100, 1, 99, 1, 97, 1],
                        "D": [97, 1, 1, 1, 1, 1, 1, 1, None, 1],
                        "E": [97, -98, 1, -99, 1, -98, 1, -100, 1, None],
                        "agg_col": list(string.ascii_uppercase[:10]),
                    }
                ),
                ("A", "B", "C", "D", "E"),
                (True, False, True, True, False),
                ("last", "last", "first", "last", "last"),
                "-",
                "-".join(["C", "B", "F", "D", "H", "J", "A", "I", "G", "E"]),
            ),
            id="listagg_multilevel_order",
        ),
    ],
)
def test_bodosql_listagg(args, memory_leak_check):
    (
        input_df,
        order_cols,
        window_ascending,
        window_na_position,
        sep,
        expected_output,
    ) = args

    def impl(df):
        return bodosql.kernels.listagg.bodosql_listagg(
            df, "agg_col", order_cols, window_ascending, window_na_position, sep
        )

    check_func(
        impl,
        (input_df,),
        py_output=expected_output,
    )


def test_listagg_no_order(memory_leak_check):
    input_df = pd.DataFrame({"agg_col": ["foo"] * 10})
    order_cols = window_ascending = window_na_position = ()
    sep = ", "
    expected_out = sep.join(["foo"] * 10)

    def impl(df):
        return bodosql_listagg(
            df, "agg_col", order_cols, window_ascending, window_na_position, sep
        )

    check_func(
        impl,
        (input_df,),
        py_output=expected_out,
    )
