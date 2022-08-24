# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Tests of series.map and dataframe.apply used for parity
with pyspark.sql.functions that operation on strings as
column elements.

Test names refer to the names of the spark function they map to.
"""

import numpy as np
import pandas as pd
import pytest

from bodo.tests.utils import check_func, gen_nonascii_list


@pytest.fixture(
    params=[
        pd.DataFrame(
            {
                "A": [
                    "hello",
                    "world",
                ]
                * 40,
                "B": [
                    "weird",
                    "world",
                ]
                * 40,
                "C": [
                    "krusty",
                    "klown",
                ]
                * 40,
                "D": [1.2, 5.3231] * 40,
            }
        ),
        pd.DataFrame(
            {
                "A": gen_nonascii_list(10) * 8,
                "B": gen_nonascii_list(5) * 16,
                "C": gen_nonascii_list(20) * 4,
                "D": [1.2, 5.3231] * 40,
            }
        ),
    ]
)
def dataframe_val(request):
    return request.param


@pytest.fixture(
    params=[
        pd.DataFrame(
            {
                "A": [b"a", b"ZZZZZZZ"] * 40,
                "B": [bytes(10), b"0123123123"] * 40,
                "C": [
                    bytes(4),
                    b"1231",
                ]
                * 40,
                "D": [1.2, 5.3231] * 40,
            }
        ),
    ]
)
def binary_dataframe_val(request):
    return request.param


@pytest.mark.slow
def test_bin(memory_leak_check):
    def test_impl(df):
        return df.A.map(lambda x: "{0:b}".format(x))

    df = pd.DataFrame(
        {
            "A": [-1, 47, 5251, -14124214, 53532, 5325, 1001, 0] * 10,
            "B": [
                "weird",
                "world",
            ]
            * 40,
        }
    )
    check_func(test_impl, (df,))


@pytest.mark.slow
def test_concat_strings(dataframe_val, memory_leak_check):
    def test_impl(df):
        return df[["A", "B", "C"]].apply(lambda x: "".join(x), axis=1)

    check_func(test_impl, (dataframe_val,))


@pytest.mark.slow
def test_concat_binary(binary_dataframe_val, memory_leak_check):
    def test_impl(df):
        return df[["A", "B", "C"]].apply(lambda x: b"".join(x), axis=1)

    check_func(test_impl, (binary_dataframe_val,))


@pytest.mark.slow
def test_concat_ws(dataframe_val, memory_leak_check):
    def test_impl(df, sep):
        return df[["A", "B", "C"]].apply(lambda x, sep: sep.join(x), sep=sep, axis=1)

    sep = "..."
    check_func(test_impl, (dataframe_val, sep))


@pytest.mark.slow
def test_concat_ws_binary(binary_dataframe_val, memory_leak_check):
    def test_impl(df, sep):
        return df[["A", "B", "C"]].apply(lambda x, sep: sep.join(x), sep=sep, axis=1)

    sep = b"..."
    check_func(test_impl, (binary_dataframe_val, sep))


@pytest.mark.slow
def test_conv(memory_leak_check):
    def test_impl(df, old_base, new_base):
        base_map = {2: "{0:b}", 8: "{0:o}", 10: "{0:d}", 16: "{0:x}"}
        new_format = base_map[new_base]
        df.A.apply(
            lambda x, old_base, new_format: new_format.format(int(x, old_base)),
            old_base=old_base,
            new_format=new_format,
        )

    df = pd.DataFrame(
        {
            "A": [
                "001110",
                "111111",
            ]
            * 40,
        }
    )
    check_func(test_impl, (df, 2, 10))
    check_func(test_impl, (df, 2, 16))
    df = pd.DataFrame(
        {
            "A": [
                "03fc1a10",
                "eeff23",
            ]
            * 40,
        }
    )
    check_func(test_impl, (df, 16, 8))
    check_func(test_impl, (df, 16, 2))
    df = pd.DataFrame(
        {
            "A": [
                "09483141210",
                "21",
            ]
            * 40,
        }
    )
    check_func(test_impl, (df, 10, 8))
    check_func(test_impl, (df, 10, 16))
    df = pd.DataFrame(
        {
            "A": [
                "73424",
                "10242",
            ]
            * 40,
        }
    )
    check_func(test_impl, (df, 8, 2))
    check_func(test_impl, (df, 8, 8))


@pytest.mark.slow
def test_format_number(dataframe_val, memory_leak_check):
    def test_impl(df, num_decimals):
        return df.D.apply(
            lambda x, d: ("{:,." + str(d) + "f}").format(np.round(x, d)), d=num_decimals
        )

    check_func(test_impl, (dataframe_val, 1))
    check_func(test_impl, (dataframe_val, 3))


@pytest.mark.slow
def test_format_string(dataframe_val, memory_leak_check):
    def test_impl(df, format_str):
        return df.A.apply(
            lambda x, format_str: format_str.format(x), format_str=format_str
        )

    check_func(test_impl, (dataframe_val, "Welcome to the {} time."))


@pytest.mark.slow
def test_translate(dataframe_val, memory_leak_check):
    def test_impl(df, to_replace, values):
        return df.A.str.split("").apply(
            lambda x, to_replace, values: "".join(
                pd.Series(x).replace(to_replace, values).tolist()
            ),
            to_replace=to_replace,
            values=values,
        )

    to_replace = ["a", "o", "l"]
    values = ["o", "z", "q"]
    check_func(test_impl, (dataframe_val, to_replace, values))
