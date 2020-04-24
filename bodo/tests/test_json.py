# Copyright (C) 2019 Bodo Inc. All rights reserved.
import pytest
import pandas as pd
import numpy as np
import bodo
from bodo.tests.utils import check_func, _get_dist_arg, _test_equal_guard, reduce_sum


def test_json_read_df(datapath):
    """
    test read_json reads a dataframe containing mutliple columns
    from a single file, a directory containg a single json file,
    and a directory containg multiple json files
    """
    fname_file = datapath("example.json")
    fname_dir_single = datapath("example_single.json")
    fname_dir_multi = datapath("example_multi.json")

    def test_impl_file(fname):
        return pd.read_json(fname, orient="records", lines=True)

    def test_impl_dir(fname):
        return pd.read_json(
            fname,
            orient="records",
            lines=True,
            dtype={
                "one": np.float32,
                "two": str,
                "three": "bool",
                "four": np.float32,
                "five": str,
            },
        )

    py_out = test_impl_file(fname_file)
    check_func(test_impl_file, (fname_file,), py_output=py_out)
    check_func(test_impl_dir, (fname_dir_single,), py_output=py_out)
    check_func(test_impl_dir, (fname_dir_multi,), py_output=py_out)


def test_json_read_int_nulls(datapath):
    """
    test read_json reads a dataframe containing nullable int column
    from a single file, a directory containg a single json file,
    and a directory containg multiple json files
    """
    fname_file = datapath("int_nulls.json")
    fname_dir_single = datapath("int_nulls_single.json")
    fname_dir_multi = datapath("int_nulls_multi.json")

    def test_impl(fname):
        # dtype specified here & using float instad of int
        # because of pandas bug
        # return pd.read_json(fname, orient='records', lines=True)
        return pd.read_json(
            fname, orient="records", lines=True, dtype={"A": np.float32}
        )

    py_out = test_impl(fname_file)
    check_func(test_impl, (fname_file,), py_output=py_out)
    check_func(test_impl, (fname_dir_single,), py_output=py_out)
    check_func(test_impl, (fname_dir_multi,), py_output=py_out)


def test_json_read_str_arr(datapath):
    """
    test read_json reads a dataframe containing str column
    from a single file, a directory containg a single json file,
    and a directory containg multiple json files
    """
    fname_file = datapath("str_arr.json")
    # Because spark and pandas writes null entries in json file differently
    # which causes different null values(None vs. nan) when pandas read them,
    # we pass this spark output file for pandas to read and use it as py_output
    fname_dir_file = datapath(
        "str_arr_single.json/part-00000-a0ff525c-31ec-499a-bde3-fe2a95cfbf8e-c000.json"
    )
    fname_dir_single = datapath("str_arr_single.json")
    fname_dir_multi = datapath("str_arr_parts.json")

    def test_impl_file(fname):
        return pd.read_json(
            fname, orient="records", lines=True, dtype={"A": str, "B": str}
        )

    def test_impl_dir(fname):
        return pd.read_json(
            fname, orient="records", lines=True, dtype={"A": str, "B": str}
        )

    py_out = test_impl_file(fname_file)
    check_func(test_impl_file, (fname_file,), py_output=py_out)
    py_out = test_impl_file(fname_dir_file)
    check_func(test_impl_dir, (fname_dir_single,), py_output=py_out)
    check_func(test_impl_dir, (fname_dir_multi,), py_output=py_out)


def test_json_read_multiline_object(datapath):
    """
    test read_json where json object is multi-lined
    from a single file 
    TODO: read a directory
    """
    fname = datapath("multiline_obj.json")

    def test_impl():
        return pd.read_json(
            fname,
            orient="records",
            lines=False,
            dtype={
                "RecordNumber": np.int,
                "Zipcode": np.int,
                "ZipCodeType": str,
                "City": str,
                "State": str,
            },
        )

    check_func(
        test_impl, (),
    )
