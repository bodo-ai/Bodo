# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Tests I/O for CSV, Parquet, HDF5, etc.
"""
import datetime
import io
import os
import random
import shutil
import subprocess
import unittest
from decimal import Decimal

import h5py
import numba
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import pytz
from numba.core.ir_utils import build_definitions, find_callname

import bodo
from bodo.pandas_compat import pandas_version
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import (
    DeadcodeTestPipeline,
    DistTestPipeline,
    SeriesOptTestPipeline,
    _check_for_io_reader_filters,
    _get_dist_arg,
    _test_equal_guard,
    check_func,
    count_array_REPs,
    count_parfor_REPs,
    gen_random_arrow_array_struct_int,
    gen_random_arrow_list_list_double,
    gen_random_arrow_list_list_int,
    gen_random_arrow_struct_struct,
    get_rank,
    get_start_end,
    reduce_sum,
)
from bodo.utils.testing import ensure_clean, ensure_clean_dir
from bodo.utils.typing import BodoError, BodoWarning
from bodo.utils.utils import is_call_assign

kde_file = os.path.join("bodo", "tests", "data", "kde.parquet")


def compress_file(fname, dummy_extension=""):
    assert not os.path.isdir(fname)
    if bodo.get_rank() == 0:
        subprocess.run(["gzip", "-k", "-f", fname])
        subprocess.run(["bzip2", "-k", "-f", fname])
        if dummy_extension != "":
            os.rename(fname + ".gz", fname + ".gz" + dummy_extension)
            os.rename(fname + ".bz2", fname + ".bz2" + dummy_extension)
    bodo.barrier()
    return [fname + ".gz" + dummy_extension, fname + ".bz2" + dummy_extension]


def remove_files(file_names):
    if bodo.get_rank() == 0:
        for fname in file_names:
            os.remove(fname)
    bodo.barrier()


def compress_dir(dir_name):
    if bodo.get_rank() == 0:
        for fname in [
            f
            for f in os.listdir(dir_name)
            if f.endswith(".csv") and os.path.getsize(dir_name + "/" + f) > 0
        ]:
            subprocess.run(["gzip", "-f", fname], cwd=dir_name)
    bodo.barrier()


def uncompress_dir(dir_name):
    if bodo.get_rank() == 0:
        for fname in [f for f in os.listdir(dir_name) if f.endswith(".gz")]:
            subprocess.run(["gunzip", fname], cwd=dir_name)
    bodo.barrier()


@pytest.fixture(
    params=[
        pd.DataFrame(
            {"A": [4, 6, 7, 1, 3], "B": [11, 12, 13, 14, 15], "C": [9, 7, 5, 3, 1]},
            index=pd.RangeIndex(start=1, stop=15, step=3, name="RI"),
        ),
        pd.DataFrame(
            {"A": [4, 6, 7, 1, 3], "B": [11, 12, 13, 14, 15], "C": [9, 7, 5, 3, 1]},
            index=pd.RangeIndex(start=1, stop=15, step=3, name=None),
        ),
        pd.DataFrame(
            {"A": [4, 6, 7, 1, 3], "B": [11, 12, 13, 14, 15], "C": [9, 7, 5, 3, 1]},
            index=pd.RangeIndex(start=1, stop=15, step=3),
        ),
        pd.DataFrame(
            {"A": [4, 6, 7, 1, 3], "B": [11, 12, 13, 14, 15], "C": [9, 7, 5, 3, 1]},
            index=None,
        ),
        pd.DataFrame(
            {"A": [4, 6, 7, 1, 3], "B": [11, 12, 13, 14, 15], "C": [9, 7, 5, 3, 1]},
            index=[-1, -2, -3, -4, -5],
        ),
    ]
)
def test_RangeIndex_input(request, memory_leak_check):
    return request.param


@pytest.mark.parametrize("pq_write_idx", [True, None, False])
def test_pq_RangeIndex(test_RangeIndex_input, pq_write_idx, memory_leak_check):
    def impl():
        df = pd.read_parquet("test.pq")
        return df

    try:
        if bodo.libs.distributed_api.get_rank() == 0:
            test_RangeIndex_input.to_parquet("test.pq", index=pq_write_idx)
        bodo.barrier()
        check_func(impl, (), only_seq=True, reset_index=True)
        bodo.barrier()
    finally:
        if bodo.libs.distributed_api.get_rank() == 0:
            os.remove("test.pq")


@pytest.mark.parametrize("index_name", [None, "HELLO"])
@pytest.mark.parametrize("pq_write_idx", [True, None, False])
def test_pq_select_column(
    test_RangeIndex_input, index_name, pq_write_idx, memory_leak_check
):
    def impl():
        df = pd.read_parquet("test.pq", columns=["A", "C"])
        return df

    try:
        if bodo.libs.distributed_api.get_rank() == 0:
            test_RangeIndex_input.index.name = index_name
            test_RangeIndex_input.to_parquet("test.pq", index=pq_write_idx)
        bodo.barrier()
        check_func(impl, (), only_seq=True, reset_index=True)
        bodo.barrier()
    finally:
        if bodo.libs.distributed_api.get_rank() == 0:
            os.remove("test.pq")


# TODO: Implement Delta Lake in a way similar to Iceberg:
# https://bodo.atlassian.net/browse/BE-3048
@pytest.mark.skip
def test_read_parquet_from_deltalake(memory_leak_check):
    def impl():
        return pd.read_parquet("bodo/tests/data/example_deltalake")

    py_output = pd.DataFrame({"value": [1, 1, 2, 3, 2, 3]})
    check_func(impl, (), py_output=py_output, check_dtype=False)


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(
            {"A": [4, 6, 7, 1, 3]}, index=pd.RangeIndex(start=1, stop=15, step=3)
        ),
        pd.DataFrame(
            {"A": [4, 6, 7, 1, 3]},
            index=pd.RangeIndex(start=0, stop=5, step=1, name=None),
        ),
        pd.DataFrame({"A": [4, 6, 7, 1, 3]}, index=["X", "Y", "Z", "M", "N"]),
        pd.DataFrame({"A": [4, 6, 7, 1, 3]}, index=[-1, -2, -3, -4, -5]),
        pd.DataFrame(
            {"A": [4, 6, 7, 1, 3]},
            index=pd.date_range(start="1/1/2022", end="1/05/2022"),
        ),
        # TODO: Open Issue: Missing Support for Storing Interval Type
        # TODO: Add Test for PeriodIndex when Pandas Serializes to Parquet Correctly
        # TODO: Add Test for CategoricalIndex after Including Metadata
        # TODO: Add Test for TimedeltaIndex when PyArrow Adds Support
    ],
)
@pytest.mark.parametrize("index_name", [None, "HELLO"])
def test_pq_write_metadata(df, index_name, memory_leak_check):
    import pyarrow.parquet as pq

    def impl_index_false(df, path):
        df.to_parquet(path, index=False)

    def impl_index_none(df, path):
        df.to_parquet(path, index=None)

    def impl_index_true(df, path):
        df.to_parquet(path, index=True)

    df.index.name = index_name
    try:
        if bodo.libs.distributed_api.get_size() == 1:
            for func in [impl_index_false, impl_index_none, impl_index_true]:
                func(df, "pandas_metadatatest.pq")
                pandas_table = pq.read_table("pandas_metadatatest.pq")

                bodo_func = bodo.jit(func)
                bodo_func(df, "bodo_metadatatest.pq")
                bodo_table = pq.read_table("bodo_metadatatest.pq")

                assert bodo_table.schema.pandas_metadata.get(
                    "index_columns"
                ) == pandas_table.schema.pandas_metadata.get("index_columns")

                # Also make sure result of reading parquet file is same as that of pandas
                pd.testing.assert_frame_equal(
                    pd.read_parquet("bodo_metadatatest.pq"),
                    pd.read_parquet("pandas_metadatatest.pq"),
                )

        elif bodo.libs.distributed_api.get_size() > 1:
            for mode in ["1d-distributed", "1d-distributed-varlength"]:
                for func in [impl_index_false, impl_index_none, impl_index_true]:
                    if mode == "1d-distributed":
                        bodo_func = bodo.jit(func, all_args_distributed_block=True)
                    else:
                        bodo_func = bodo.jit(func, all_args_distributed_varlength=True)

                    bodo_func(_get_dist_arg(df, False), "bodo_metadatatest.pq")
                    bodo.barrier()
                    bodo_table = pq.read_table("bodo_metadatatest.pq")
                    if func is impl_index_false:
                        assert [] == bodo_table.schema.pandas_metadata.get(
                            "index_columns"
                        )
                    else:
                        if df.index.name is None:
                            assert (
                                "__index_level_0__"
                                in bodo_table.schema.pandas_metadata.get(
                                    "index_columns"
                                )
                            )
                        else:
                            assert (
                                df.index.name
                                in bodo_table.schema.pandas_metadata.get(
                                    "index_columns"
                                )
                            )
                    bodo.barrier()
    finally:
        if bodo.libs.distributed_api.get_size() == 1:
            os.remove("bodo_metadatatest.pq")
            os.remove("pandas_metadatatest.pq")
        else:
            shutil.rmtree("bodo_metadatatest.pq", ignore_errors=True)


def test_pq_pandas_date(datapath, memory_leak_check):
    fname = datapath("pandas_dt.pq")

    def impl():
        df = pd.read_parquet(fname)
        return pd.DataFrame({"DT64": df.DT64, "col2": df.DATE})

    check_func(impl, (), only_seq=True)


def test_pq_spark_date(datapath, memory_leak_check):
    fname = datapath("sdf_dt.pq")

    def impl():
        df = pd.read_parquet(fname)
        return pd.DataFrame({"DT64": df.DT64, "col2": df.DATE})

    check_func(impl, (), only_seq=True)


def test_pq_index(datapath, memory_leak_check):
    def test_impl(fname):
        return pd.read_parquet(fname)

    # passing function name as value to test value-based dispatch
    fname = datapath("index_test1.pq")
    check_func(test_impl, (fname,), only_seq=True, check_dtype=False)

    # string index
    fname = datapath("index_test2.pq")

    def test_impl2():
        return pd.read_parquet(fname)

    check_func(test_impl2, (), only_seq=True, check_dtype=False)


def test_pq_nullable_int_single(datapath, memory_leak_check):
    # single piece parquet
    fname = datapath("int_nulls_single.pq")

    def test_impl():
        return pd.read_parquet(fname)

    check_func(test_impl, (), check_dtype=False)


def test_pq_nullable_int_multi(datapath, memory_leak_check):
    # multi piece parquet
    fname = datapath("int_nulls_multi.pq")

    def test_impl():
        return pd.read_parquet(fname)

    check_func(test_impl, (), check_dtype=False)


def test_pq_bool_with_nulls(datapath, memory_leak_check):
    fname = datapath("bool_nulls.pq")

    def test_impl():
        return pd.read_parquet(fname)

    check_func(test_impl, ())


def test_pq_schema(datapath, memory_leak_check):
    fname = datapath("example.parquet")

    def impl(f):
        df = pd.read_parquet(f)
        return df

    bodo_func = bodo.jit(
        distributed=False,
        locals={
            "df": {
                "one": bodo.float64[:],
                "two": bodo.string_array_type,
                "three": bodo.bool_[:],
                "four": bodo.float64[:],
                "five": bodo.string_array_type,
            }
        },
    )(impl)
    pd.testing.assert_frame_equal(bodo_func(fname), impl(fname), check_dtype=False)


def test_pq_list_str(datapath, memory_leak_check):
    def test_impl(fname):
        return pd.read_parquet(fname)

    check_func(test_impl, (datapath("list_str_arr.pq"),))
    check_func(test_impl, (datapath("list_str_parts.pq"),))


# TODO [BE-1424]: Add memory_leak_check when bugs are resolved.
def test_pq_arrow_array_random():
    def test_impl(fname):
        return pd.read_parquet(fname)

    def gen_random_arrow_array_struct_single_int(span, n):
        e_list = []
        for _ in range(n):
            valA = random.randint(0, span)
            e_ent = {"A": valA}
            e_list.append(e_ent)
        return e_list

    random.seed(5)
    n = 20
    # One single entry {"A": 1} pass
    df_work1 = pd.DataFrame({"X": gen_random_arrow_array_struct_single_int(10, n)})

    # Two degree of recursion and missing values. It passes.
    df_work2 = pd.DataFrame({"X": gen_random_arrow_list_list_double(2, 0.1, n)})

    # Two degrees of recursion and integers. It passes.
    df_work3 = pd.DataFrame({"X": gen_random_arrow_list_list_int(2, 0.1, n)})

    # One degree of recursion. Calls another code path!
    df_work4 = pd.DataFrame({"X": gen_random_arrow_list_list_double(1, 0.1, n)})

    # One degree of freedom. Converting to a arrow array
    df_work5 = pd.DataFrame({"X": gen_random_arrow_list_list_int(1, 0.1, n)})

    # Two entries in the rows is failing {"A":1, "B":3}.
    # We treat this by calling the function several times.
    df_work6 = pd.DataFrame({"X": gen_random_arrow_array_struct_int(10, n)})

    # recursive struct construction
    df_work7 = pd.DataFrame({"X": gen_random_arrow_struct_struct(10, n)})

    # Missing in pyarrow and arrow-cpp 1.0 when reading parquet files:
    # E   pyarrow.lib.ArrowInvalid: Mix of struct and list types not yet supported
    # It also does not work in pandas
    # df_bug = pd.DataFrame({"X": gen_random_arrow_array_struct_list_int(10, n)})
    def process_df(df):
        fname = "test_pq_nested_tmp.pq"
        with ensure_clean(fname):
            if bodo.get_rank() == 0:
                df.to_parquet(fname)
            bodo.barrier()
            check_func(test_impl, (fname,), check_dtype=False)

    for df in [df_work1, df_work2, df_work3, df_work4, df_work5, df_work6, df_work7]:
        process_df(df)


# TODO [BE-1424]: Add memory_leak_check when bugs are resolved.
def test_pq_array_item(datapath):
    # TODO: [BE-581] Handle cases where the number of processes are
    # greater than the number of rows for nested arrays and other types.
    def test_impl(fname):
        return pd.read_parquet(fname)

    check_func(test_impl, (datapath("list_int.pq"),))

    a = np.array([[2.0, -3.2], [2.2, 1.3], None, [4.1, 5.2, 6.3], [], [1.1, 1.2]])
    b = np.array([[1, 3], [2], None, [4, 5, 6], [], [1, 1]])
    # for list of bools there are some things missing like (un)boxing
    # c = np.array([[True, False], None, None, [True, True, True], [False, False], []])
    df = pd.DataFrame({"A": a, "B": b})
    with ensure_clean("test_pq_list_item.pq"):
        if bodo.get_rank() == 0:
            df.to_parquet("test_pq_list_item.pq")
        bodo.barrier()
        check_func(test_impl, ("test_pq_list_item.pq",))

    a = np.array(
        [[[2.0], [-3.2]], [[2.2, 1.3]], None, [[4.1, 5.2], [6.3]], [], [[1.1, 1.2]]]
    )
    b = np.array([[[1], [3]], [[2]], None, [[4, 5, 6]], [], [[1, 1]]])
    df = pd.DataFrame({"A": a, "B": b})
    with ensure_clean("test_pq_list_item.pq"):
        if bodo.get_rank() == 0:
            df.to_parquet("test_pq_list_item.pq")
        bodo.barrier()
        check_func(test_impl, ("test_pq_list_item.pq",))


def test_pq_categorical_read(memory_leak_check):
    """test reading categorical data from Parquet files"""

    def impl():
        df = pd.read_parquet("test_cat.pq")
        return df

    try:
        df = pd.DataFrame(
            {"A": pd.Categorical(["A", "B", "AB", "A", None, "B"] * 4 + [None, "C"])}
        )
        if bodo.libs.distributed_api.get_rank() == 0:
            df.to_parquet("test_cat.pq", row_group_size=4)
        bodo.barrier()
        check_func(impl, ())
        bodo.barrier()
    finally:
        if bodo.libs.distributed_api.get_rank() == 0:
            os.remove("test_cat.pq")


@pytest.mark.slow
def test_pq_unsupported_types(datapath, memory_leak_check):
    """test unsupported data types in unselected columns"""

    def test_impl(fname):
        return pd.read_parquet(fname, columns=["B"])

    # FIXME I think we do suport everything in nested_struct_example.pq
    check_func(test_impl, (datapath("nested_struct_example.pq"),))


def test_pq_decimal(datapath, memory_leak_check):
    def test_impl(fname):
        return pd.read_parquet(fname)

    check_func(test_impl, (datapath("decimal1.pq"),))


def test_pq_date32(datapath, memory_leak_check):
    """Test reading date32 values into datetime.date array"""

    def test_impl(fname):
        return pd.read_parquet(fname)

    check_func(test_impl, (datapath("date32_1.pq"),))


def test_pq_processes_greater_than_string_rows(datapath):
    f = lambda filename: pd.read_parquet(filename)
    check_func(f, [datapath("small_strings.pq")])


@pytest.mark.slow
def test_csv_infer_type_error(datapath):
    ints = [0] * 1000
    strings = ["a"] * 1000
    df = pd.DataFrame({"A": ints + strings})
    filepath = datapath("test_csv_infer_type_error.csv", check_exists=False)
    with ensure_clean(filepath):
        if bodo.get_rank() == 0:
            df.to_csv(filepath, index=False)
        bodo.barrier()
        message = r"pd.read_csv\(\): Bodo could not infer dtypes correctly."
        with pytest.raises(TypeError, match=message):
            bodo.jit(lambda: pd.read_csv(filepath), all_returns_distributed=True)()
        with pytest.raises(TypeError, match=message):
            bodo.jit(lambda: pd.read_csv(filepath), distributed=False)()


def test_bodo_upcast(datapath):
    """
    Tests the _bodo_upcast_to_float64 custom argument to read csv. This ensures that a file
    which would otherwise cause typing issues will upcast to a shared type when possible
    """
    # Create strings of ints and floats
    ints = ["0"] * 1000
    floats = ["1.1"] * 1000
    df = pd.DataFrame({"A": ints + floats})
    filepath = datapath("test_mixed_int_float.csv", check_exists=False)
    with ensure_clean(filepath):
        if bodo.get_rank() == 0:
            df.to_csv(filepath, index=False)
        bodo.barrier()
        message = r"pd.read_csv\(\): Bodo could not infer dtypes correctly."
        with pytest.raises(TypeError, match=message):
            bodo.jit(lambda: pd.read_csv(filepath), all_returns_distributed=True)()
        bodo.jit(
            lambda: pd.read_csv(filepath, _bodo_upcast_to_float64=True),
            distributed=False,
        )()


def test_to_csv_none_arg0(memory_leak_check):
    """checks that passing None as the filepath argument is properly supported"""

    def impl(df):
        return df.to_csv(path_or_buf=None)

    def impl2(df):
        return df.to_csv()

    def impl3(df):
        return df.to_csv(None)

    df = pd.DataFrame({"A": np.arange(100)})

    check_func(impl, (df,), only_seq=True)
    check_func(impl2, (df,), only_seq=True)
    check_func(impl3, (df,), only_seq=True)

    # for the distributed cases, the output for each rank is the same as calling
    # df.to_csv(None) on the distributed dataframe
    py_out_1d = _get_dist_arg(df).to_csv(None)
    check_func(impl, (df,), only_1D=True, py_output=py_out_1d)
    check_func(impl2, (df,), only_1D=True, py_output=py_out_1d)
    check_func(impl3, (df,), only_1D=True, py_output=py_out_1d)

    py_out_1d_vars = _get_dist_arg(df, var_length=True).to_csv(None)
    check_func(impl, (df,), only_1DVar=True, py_output=py_out_1d_vars)
    check_func(impl2, (df,), only_1DVar=True, py_output=py_out_1d_vars)
    check_func(impl3, (df,), only_1DVar=True, py_output=py_out_1d_vars)


def test_to_csv_filepath_as_kwd_arg(memory_leak_check):
    """checks that passing the filepath as a keyword argument is properly supported"""

    def impl(df, f_name):
        return df.to_csv(path_or_buf=f_name)

    df = pd.DataFrame({"A": np.arange(10), "B": np.arange(10), "C": np.arange(10)})
    check_CSV_write(impl, df)


def test_basic_paralel_write(memory_leak_check):
    """does a basic test of to_csv with no arguments"""

    def impl(df, f_name):
        return df.to_csv(f_name)

    df = pd.DataFrame({"A": np.arange(10), "B": np.arange(10), "C": np.arange(10)})
    check_CSV_write(impl, df)


def test_to_csv_sep_kwd_arg(memory_leak_check):
    """tests the sep keyword argument to to_csv."""

    def impl_none(df):
        return df.to_csv(None, sep="-")

    def impl(df, f_name):
        return df.to_csv(f_name, sep=":")

    df = pd.DataFrame({"A": np.arange(100)})

    check_to_csv_string_output(df, impl_none)
    check_CSV_write(impl, df)


def test_to_csv_na_rep_kwd_arg(memory_leak_check):
    """tests the na_rep keyword argument to to_csv."""

    def impl_none(df):
        return df.to_csv(None, na_rep="NA_VALUE")

    def impl(df, f_name):
        return df.to_csv(f_name, na_rep="-1")

    df = pd.DataFrame({"A": np.arange(10)}, dtype="Int64")
    df["A"][0] = None

    check_to_csv_string_output(df, impl_none)
    check_CSV_write(impl, df)


def test_to_csv_float_format_kwd_arg(memory_leak_check):
    """tests the float_format keyword argument to to_csv."""

    def impl_none(df):
        return df.to_csv(None, float_format="%.3f")

    def impl(df, f_name):
        return df.to_csv(f_name, float_format="%.3f")

    # This should generate floats with a large number of decimal places
    df = pd.DataFrame({"A": [x / 7 for x in range(10)]})
    check_to_csv_string_output(df, impl_none)
    check_CSV_write(impl, df)


def test_to_csv_columns_kwd_arg(memory_leak_check):
    """tests the columns keyword argument to to_csv. List input is not currently supported, see BE-1505"""

    def impl_none(df):
        return df.to_csv(None, columns=("A", "C"))

    def impl(df, f_name):
        return df.to_csv(f_name, columns=("A", "C"))

    def impl_list(df, f_name):
        return df.to_csv(f_name, columns=["A", "C"])

    df = pd.DataFrame({"A": np.arange(10), "B": np.arange(10), "C": np.arange(10)})
    check_to_csv_string_output(df, impl_none)
    check_CSV_write(impl, df)
    check_CSV_write(impl_list, df)


# Header argument tested in test_csv_header_write_read


def test_to_csv_index_kwd_arg(memory_leak_check):
    """tests the index keyword argument to to_csv"""

    def impl_none(df):
        return df.to_csv(None, index=False)

    def impl(df, f_name):
        return df.to_csv(f_name, index=False)

    df = pd.DataFrame({"A": np.arange(10), "B": np.arange(10), "C": np.arange(10)})

    check_to_csv_string_output(df, impl_none)
    check_CSV_write(impl, df)


def test_to_csv_index_label_kwd_arg(memory_leak_check):
    """tests the index_label keyword argument to to_csv"""

    def impl_none(df):
        return df.to_csv(None, index_label=False)

    def impl(df, f_name):
        return df.to_csv(f_name, index_label=False)

    def impl_none2(df):
        return df.to_csv(None, index_label="LABEL_")

    def impl2(df, f_name):
        return df.to_csv(f_name, index_label="LABEL_")

    df = pd.DataFrame({"A": np.arange(10), "B": np.arange(10), "C": np.arange(10)})
    check_to_csv_string_output(df, impl_none)
    check_to_csv_string_output(df, impl_none2)
    check_CSV_write(impl, df)
    check_CSV_write(impl2, df)


def test_to_csv_quoting_kwd_arg(memory_leak_check):
    """tests the quoting keyword argument to to_csv"""
    import csv

    def impl_none(df):
        return df.to_csv(None, quoting=csv.QUOTE_ALL)

    def impl(df, f_name):
        return df.to_csv(f_name, quoting=csv.QUOTE_ALL)

    df = pd.DataFrame({"A": np.arange(10), "B": np.arange(10), "C": np.arange(10)})
    check_to_csv_string_output(df, impl_none)
    check_CSV_write(
        impl,
        df,
    )


def test_to_csv_quotechar_kwd_arg(memory_leak_check):
    """tests the quotechar keyword argument to to_csv"""
    import csv

    def impl_none(df):
        return df.to_csv(None, quoting=csv.QUOTE_ALL, quotechar="Q")

    def impl(df, f_name):
        return df.to_csv(f_name, quoting=csv.QUOTE_ALL, quotechar="X")

    def read_impl(f_name):
        return pd.read_csv(f_name, quotechar="X")

    df = pd.DataFrame({"A": np.arange(10), "B": np.arange(10), "C": np.arange(10)})
    check_to_csv_string_output(df, impl_none)
    check_CSV_write(
        impl,
        df,
        read_impl=read_impl,
    )


def test_to_csv_line_terminator_kwd_arg(memory_leak_check):
    """tests the line_terminator keyword argument to to_csv"""

    def impl_none(df):
        return df.to_csv(None, line_terminator="__LINE_TERMINATION__")

    def impl(df, f_name):
        return df.to_csv(f_name, line_terminator="\t")

    def read_impl(f_name):
        return pd.read_csv(f_name, lineterminator="\t")

    df = pd.DataFrame({"A": np.arange(10), "B": np.arange(10), "C": np.arange(10)})
    check_to_csv_string_output(df, impl_none)
    check_CSV_write(
        impl,
        df,
        read_impl=read_impl,
    )


def test_to_csv_chunksize_kwd_arg(memory_leak_check):
    """tests the chunksize keyword argument to to_csv"""

    def impl_none(df):
        return df.to_csv(None, chunksize=7)

    def impl(df, f_name):
        return df.to_csv(f_name, chunksize=7)

    df = pd.DataFrame({"A": np.arange(100), "B": np.arange(100), "C": np.arange(100)})
    check_to_csv_string_output(df, impl_none)
    check_CSV_write(
        impl,
        df,
    )


@pytest.mark.slow
def test_to_csv_date_format_kwd_arg(memory_leak_check):
    """tests the date_format keyword argument to to_csv."""

    def impl_none(df):
        return df.to_csv(None, date_format="%a, %b, %Y, %Z, %x")

    def impl(df, f_name):
        return df.to_csv(f_name, date_format="%W, %z, %f, %S, %x")

    df = pd.DataFrame(
        {"A": pd.date_range(start="1998-04-24", end="2000-04-29", periods=100)}
    )
    check_to_csv_string_output(df, impl_none)
    check_CSV_write(
        impl,
        df,
    )


def test_to_csv_doublequote_escapechar_kwd_args(memory_leak_check):
    """tests the doublequote and escapechar keyword argument to to_csv.
    Doublequote and escapechar need to be tested together, as escapechar is
    the char used to escape when not double quoting.
    """

    def impl_none(df):
        return df.to_csv(
            None, doublequote=False, quotechar="a", sep="-", escapechar="E"
        )

    def impl(df, f_name):
        return df.to_csv(
            f_name, doublequote=False, quotechar="a", sep="-", escapechar="E"
        )

    def read_impl(f_name):
        return pd.read_csv(
            f_name, doublequote=False, quotechar="a", sep="-", escapechar="E"
        )

    df = pd.DataFrame({"A": ["a - a - a - a"] * 10})
    check_to_csv_string_output(df, impl_none)
    check_CSV_write(impl, df, read_impl=read_impl)


@pytest.mark.slow
def test_to_csv_decimal_kwd_arg(memory_leak_check):
    """tests the decimal keyword argument to to_csv"""

    def impl_none(df):
        return df.to_csv(None, decimal="_")

    def impl(df, f_name):
        return df.to_csv(f_name, decimal="_")

    def read_impl(f_name):
        return pd.read_csv(f_name, decimal="_")

    # This should generate floats with a large number of decimal places
    df = pd.DataFrame({"A": [x / 7 for x in range(10)]})
    check_to_csv_string_output(df, impl_none)
    check_CSV_write(impl, df, read_impl=read_impl)


@pytest.mark.slow
def test_read_csv_bad_dtype_column(datapath, memory_leak_check):
    """Checks calling read_csv() with columns in the dtype that
    aren't in the DataFrame. This raises a warning so the code
    should still execute."""

    fname = datapath("csv_data_infer1.csv")

    def test_impl(fname):
        dtype = {"B": "float64", "I_AM_A_MISSING_COLUMN": pd.Int32Dtype()}
        return pd.read_csv(fname, dtype=dtype)

    # Set check_dtype=False for nullable differences
    check_func(test_impl, (fname,), check_dtype=False)


def test_read_csv_nonascii(datapath, memory_leak_check):
    """Checks calling read_csv() with non-ascii data"""

    fname = datapath("csv_data_nonascii1.csv")

    def test_impl(fname):
        return pd.read_csv(fname)

    check_func(test_impl, (fname,))


@pytest.mark.slow
def test_csv_remove_col0_used_for_len(datapath, memory_leak_check):
    """read_csv() handling code uses the first column for creating RangeIndex of the
    output dataframe. In cases where the first column array is dead, it should be
    replaced by an alternative live array. This test makes sure this replacement happens
    properly.
    """
    fname = datapath("csv_data1.csv")
    fname_gzipped = fname + ".gz"

    def impl():
        df = pd.read_csv(fname, names=["A", "B", "C", "D"], compression=None)
        return df.C

    def impl2():
        df = pd.read_csv(fname_gzipped, names=["A", "B", "C", "D"], compression="gzip")
        return df.C

    check_func(impl, (), only_seq=True)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit()(impl)()
        check_logger_msg(stream, "Columns loaded ['C']")

    if bodo.get_rank() == 0:
        subprocess.run(["gzip", "-k", "-f", fname])
    bodo.barrier()
    with ensure_clean(fname_gzipped):
        check_func(impl2, (), only_seq=True)
        with set_logging_stream(logger, 1):
            bodo.jit()(impl2)()
            check_logger_msg(stream, "Columns loaded ['C']")


@pytest.mark.slow
def test_csv_remove_col_keep_date(datapath, memory_leak_check):
    """Test parse_date position matches usecols after removing unused column
    See [BE-2561]"""
    fname = datapath("csv_data_date1.csv")

    def test_impl(fname):
        df = pd.read_csv(
            fname,
            names=["A", "B", "C", "D"],
            dtype={"A": int, "B": float, "C": str, "D": int},
            parse_dates=["C"],
        )
        return df["C"]

    check_func(test_impl, (fname,))


@pytest.mark.slow
def test_csv_usecols_parse_dates(datapath, memory_leak_check):
    """Test usecols with parse_date See [BE-2544]"""
    fname = datapath("csv_data_date1.csv")

    def test_impl(fname):
        df = pd.read_csv(
            fname,
            names=["A", "B", "C", "D"],
            dtype={"A": int, "B": float, "C": str, "D": int},
            parse_dates=["C"],
            usecols=["A", "C", "D"],
        )
        return df

    check_func(test_impl, (fname,))


@pytest.mark.slow
def test_csv_usecols_names_args(datapath, memory_leak_check):
    """Test usecols and names argument together"""
    fname = datapath("example.csv")

    # subset for both names and usecols
    def impl1(fname):
        df = pd.read_csv(fname, names=["A", "B"], usecols=[0, 2])
        return df

    check_func(impl1, (fname,))

    # all column names and subset for usecols
    def impl2(fname):
        df = pd.read_csv(fname, names=["A", "B", "C", "D", "E"], usecols=[0, 2, 3])
        return df

    check_func(impl2, (fname,))

    # colnames > usecols but usecols has duplicates
    def impl3(fname):
        df = pd.read_csv(fname, names=["A", "B", "C"], usecols=[0, 2, 1, 0, 1])
        return df

    check_func(impl3, (fname,))

    # few names + dtypes=None + usecols=None
    def impl4(fname):
        df = pd.read_csv(fname, names=["A", "B", "C"])
        return df

    # Ignore index check See [BE-2596]
    check_func(impl4, (fname,), reset_index=True)

    # few names + dtypes + usecols=None
    def impl5(fname):
        df = pd.read_csv(
            fname, names=["A", "B", "C"], dtypes={"A": "float", "B": "str", "C": "bool"}
        )
        return df

    check_func(impl4, (fname,), reset_index=True)

    # colnames > usecols
    def impl6(fname):
        df = pd.read_csv(fname, names=["A", "B", "C", "D", "E"], usecols=[0])
        return df

    check_func(impl6, (fname,))


@pytest.mark.slow
def test_h5_remove_dead(datapath, memory_leak_check):
    """make sure dead hdf5 read calls are removed properly"""
    fname = datapath("lr.hdf5")

    def impl():
        f = h5py.File(fname, "r")
        X = f["points"][:, :]
        f.close()

    bodo_func = numba.njit(pipeline_class=DeadcodeTestPipeline, parallel=True)(impl)
    bodo_func()
    fir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    fir._definitions = build_definitions(fir.blocks)
    for stmt in fir.blocks[0].body:
        assert not (
            is_call_assign(stmt)
            and find_callname(fir, stmt.value) == ("h5read", "bodo.io.h5_api")
        )


@pytest.mark.slow
def test_read_parquet_list_files(datapath, memory_leak_check):
    """test read_parquet passing a list of files"""

    def test_impl():
        return pd.read_parquet(
            ["bodo/tests/data/example.parquet", "bodo/tests/data/example2.parquet"]
        )

    def test_impl2(fpaths):
        return pd.read_parquet(fpaths)

    py_output_part1 = pd.read_parquet(datapath("example.parquet"))
    py_output_part2 = pd.read_parquet(datapath("example2.parquet"))
    py_output = pd.concat([py_output_part1, py_output_part2])
    check_func(test_impl, (), py_output=py_output)
    fpaths = [datapath("example.parquet"), datapath("example2.parquet")]
    check_func(test_impl2, (fpaths,), py_output=py_output)


def test_read_parquet_nonascii(datapath, memory_leak_check):
    """Checks calling read_parquet() with non-ascii data"""

    fname = datapath("parquet_data_nonascii1.parquet")

    def test_impl(fname):
        return pd.read_parquet(fname)

    check_func(test_impl, (fname,))


@pytest.mark.slow
def test_pq_cache_print(datapath, capsys, memory_leak_check):
    """Make sure FilenameType behaves like a regular value and not a literal when loaded
    from cache. This allows the file name value to be set correctly and not baked in.
    """

    @bodo.jit(cache=True)
    def f(fname):
        bodo.parallel_print(fname)
        return pd.read_parquet(fname)

    fname1 = datapath("example.parquet")
    fname2 = datapath("example2.parquet")
    f(fname1)
    f(fname2)
    captured = capsys.readouterr()
    assert "example2.parquet" in captured.out


def clean_pq_files(mode, pandas_pq_path, bodo_pq_path):
    if bodo.get_rank() == 0:
        try:
            os.remove(pandas_pq_path)
        except FileNotFoundError:
            pass
    if mode == "sequential":
        # in sequential mode each process has written to a different file
        if os.path.exists(bodo_pq_path):
            os.remove(bodo_pq_path)
    elif bodo.get_rank() == 0:
        # in parallel mode, the path is a directory containing multiple
        # parquet files (one per process)
        shutil.rmtree(bodo_pq_path, ignore_errors=True)


@pytest.mark.slow
def test_read_write_parquet(memory_leak_check):
    def write(df, filename):
        df.to_parquet(filename)

    def read():
        return pd.read_parquet("_test_io___.pq")

    def pandas_write(df, filename):
        # pandas/pyarrow throws this error when writing datetime64[ns]
        # to parquet 1.x:
        # pyarrow.lib.ArrowInvalid: Casting from timestamp[ns] to timestamp[ms] would lose data: xxx
        # So we have to truncate to ms precision. pandas 1.1.0 apparently got
        # rid of allow_truncated_timestamps option of to_parquet, so we do this
        # manually
        if hasattr(df, "_datetime_col"):
            df[df._datetime_col] = df[df._datetime_col].dt.floor("ms")
        df.to_parquet(filename)

    def gen_dataframe(num_elements, write_index):
        df = pd.DataFrame()
        cur_col = 0
        for dtype in [
            "int8",
            "uint8",
            "int16",
            "uint16",
            "int32",
            "uint32",
            "int64",
            "uint64",
            "float32",
            "float64",
            "bool",
            "String",
            "Binary",
            "Int8",
            "UInt8",
            "Int16",
            "UInt16",
            "Int32",
            "UInt32",
            "Int64",
            "UInt64",
            "Decimal",
            "Date",
            "Datetime",
            "nested_arrow0",
            "nested_arrow1",
            "nested_arrow2",
        ]:
            col_name = "col_" + str(cur_col)
            if dtype == "String":
                # missing values every 5 elements
                data = [str(x) * 3 if x % 5 != 0 else None for x in range(num_elements)]
                df[col_name] = data
            elif dtype == "Binary":
                # missing values every 5 elements
                data = [
                    str(x).encode() * 3 if x % 5 != 0 else None
                    for x in range(num_elements)
                ]
                df[col_name] = data
            elif dtype == "bool":
                data = [True if x % 2 == 0 else False for x in range(num_elements)]
                df[col_name] = np.array(data, dtype="bool")
            elif dtype.startswith("Int") or dtype.startswith("UInt"):
                # missing values every 5 elements
                data = [x if x % 5 != 0 else np.nan for x in range(num_elements)]
                df[col_name] = pd.Series(data, dtype=dtype)
            elif dtype == "Decimal":
                assert num_elements % 8 == 0
                data = np.array(
                    [
                        Decimal("1.6"),
                        None,
                        Decimal("-0.222"),
                        Decimal("1111.316"),
                        Decimal("1234.00046"),
                        Decimal("5.1"),
                        Decimal("-11131.0056"),
                        Decimal("0.0"),
                    ]
                    * (num_elements // 8)
                )
                df[col_name] = pd.Series(data, dtype=object)
            elif dtype == "Date":
                dates = pd.Series(
                    pd.date_range(
                        start="1998-04-24", end="1998-04-29", periods=num_elements
                    )
                )
                df[col_name] = dates.dt.date
            elif dtype == "Datetime":
                dates = pd.Series(
                    pd.date_range(
                        start="1998-04-24", end="1998-04-29", periods=num_elements
                    )
                )
                if num_elements >= 20:
                    # set some elements to NaT
                    dates[4] = None
                    dates[17] = None
                df[col_name] = dates
                df._datetime_col = col_name
            elif dtype == "nested_arrow0":
                # Disabling Nones because currently they very easily induce
                # typing errors during unboxing for nested lists.
                # _infer_ndarray_obj_dtype in boxing.py needs to be made more robust.
                # TODO: include Nones
                df[col_name] = pd.Series(
                    gen_random_arrow_list_list_int(2, 0, num_elements)
                )
            elif dtype == "nested_arrow1":
                df[col_name] = pd.Series(
                    gen_random_arrow_array_struct_int(10, num_elements)
                )
            elif dtype == "nested_arrow2":
                # TODO: Include following types when they are supported in PYARROW:
                # We cannot read this dataframe in bodo. Fails at unboxing.
                # df_bug1 = pd.DataFrame({"X": gen_random_arrow_list_list_double(2, -0.1, n)})
                # This dataframe can be written by the code. However we cannot read
                # due to a limitation in pyarrow
                # df_bug2 = pd.DataFrame({"X": gen_random_arrow_array_struct_list_int(10, n)})
                df[col_name] = pd.Series(
                    gen_random_arrow_struct_struct(10, num_elements)
                )
            else:
                df[col_name] = np.arange(num_elements, dtype=dtype)
            cur_col += 1
        if write_index == "string":
            # set a string index
            max_zeros = len(str(num_elements - 1))
            df.index = [
                ("0" * (max_zeros - len(str(val)))) + str(val)
                for val in range(num_elements)
            ]
        elif write_index == "numeric":
            # set a numeric index (not range)
            df.index = [v**2 for v in range(num_elements)]
        return df

    n_pes = bodo.get_size()
    NUM_ELEMS = 80  # length of each column in generated dataset

    random.seed(5)
    for write_index in [None, "string", "numeric"]:
        for mode in ["sequential", "1d-distributed", "1d-distributed-varlength"]:
            df = gen_dataframe(NUM_ELEMS, write_index)

            pandas_pq_filename = "test_io___pandas.pq"
            if mode == "sequential":
                bodo_pq_filename = str(bodo.get_rank()) + "_test_io___bodo.pq"
            else:
                # in parallel mode, each process writes its piece to a separate
                # file in the same directory
                bodo_pq_filename = "test_io___bodo_pq_write_dir"

            try:
                # write the same dataset with pandas and bodo
                if bodo.get_rank() == 0:
                    pandas_write(df, pandas_pq_filename)
                if mode == "sequential":
                    bodo_write = bodo.jit(write)
                    bodo_write(df, bodo_pq_filename)
                elif mode == "1d-distributed":
                    bodo_write = bodo.jit(write, all_args_distributed_block=True)
                    bodo_write(_get_dist_arg(df, False), bodo_pq_filename)
                elif mode == "1d-distributed-varlength":
                    bodo_write = bodo.jit(write, all_args_distributed_varlength=True)
                    bodo_write(_get_dist_arg(df, False, True), bodo_pq_filename)
                bodo.barrier()
                # read both files with pandas
                df1 = pd.read_parquet(pandas_pq_filename)
                df2 = pd.read_parquet(bodo_pq_filename)

                # to test equality, we have to coerce datetime columns to ms
                # because pandas writes to parquet as datetime64[ms]
                df[df._datetime_col] = df[df._datetime_col].astype("datetime64[ms]")
                # need to coerce column from bodo-generated parquet to ms (note
                # that the column has us precision because Arrow cpp converts
                # nanoseconds to microseconds when writing to parquet version 1)
                df2[df._datetime_col] = df2[df._datetime_col].astype("datetime64[ms]")

                # read dataframes must be same as original except for dtypes
                passed = _test_equal_guard(
                    df, df1, sort_output=False, check_names=True, check_dtype=False
                )
                n_passed = reduce_sum(passed)
                assert n_passed == n_pes
                passed = _test_equal_guard(
                    df, df2, sort_output=False, check_names=True, check_dtype=False
                )
                n_passed = reduce_sum(passed)
                assert n_passed == n_pes
                # both read dataframes should be equal in everything
                passed = _test_equal_guard(
                    df1, df2, sort_output=False, check_names=True, check_dtype=True
                )
                n_passed = reduce_sum(passed)
                assert n_passed == n_pes
            finally:
                # cleanup
                clean_pq_files(mode, pandas_pq_filename, bodo_pq_filename)
                bodo.barrier()

    for write_index in [None, "string", "numeric"]:
        # test that nothing breaks when BODO_MIN_IO_THREADS and
        # BODO_MAX_IO_THREADS are set
        os.environ["BODO_MIN_IO_THREADS"] = "2"
        os.environ["BODO_MAX_IO_THREADS"] = "2"
        try:
            df = gen_dataframe(NUM_ELEMS, write_index)
            # to test equality, we have to coerce datetime columns to ms
            # because pandas writes to parquet as datetime64[ms]
            df[df._datetime_col] = df[df._datetime_col].astype("datetime64[ms]")
            if bodo.get_rank() == 0:
                df.to_parquet("_test_io___.pq")
            bodo.barrier()
            check_func(read, (), sort_output=False, check_names=True, check_dtype=False)
        finally:
            clean_pq_files("none", "_test_io___.pq", "_test_io___.pq")
            del os.environ["BODO_MIN_IO_THREADS"]
            del os.environ["BODO_MAX_IO_THREADS"]

    def error_check2(df):
        df.to_parquet("out.parquet", compression="wrong")

    def error_check3(df):
        df.to_parquet("out.parquet", index=3)

    df = pd.DataFrame({"A": range(5)})

    with pytest.raises(BodoError, match="Unsupported compression"):
        bodo.jit(error_check2)(df)

    with pytest.raises(BodoError, match="index must be a constant bool or None"):
        bodo.jit(error_check3)(df)


@pytest.mark.slow
def test_partition_cols(memory_leak_check):

    TEST_DIR = "test_part_tmp"

    @bodo.jit(distributed=["df"])
    def impl(df, part_cols):
        df.to_parquet(TEST_DIR, partition_cols=part_cols)

    datetime_series = pd.Series(
        pd.date_range(start="2/1/2021", end="2/8/2021", periods=8)
    )
    date_series_str = datetime_series.astype(str)
    date_series = datetime_series.dt.date
    df = pd.DataFrame(
        {
            "A": [0, 0, 0, 0, 1, 1, 1, 1],
            "B": ["AA", "AA", "B", "B", "AA", "AA", "B", "B"],
            "C": [True, True, False, False, True, True, False, False],
            "D": pd.Categorical(date_series_str),
            "E": date_series,
            "F": datetime_series,
            # TODO test following F column as partition column
            # for some reason, Bodo throws
            # "bodo.utils.typing.BodoError: Cannot convert dtype DatetimeDateType() to index type"
            # in _get_dist_arg with this:
            # "F": pd.Categorical(date_series),
            "G": range(8),
        }
    )

    import pyarrow.parquet as pq

    try:
        to_test = [["A"], ["A", "B"], ["A", "B", "C"], ["D"], ["E"], ["D", "B"]]
        for part_cols in to_test:
            df_in = df.copy()
            if "D" not in part_cols:
                # TODO. can't write categorical to parquet currently because of metadata issue
                del df_in["D"]
            bodo.barrier()
            impl(_get_dist_arg(df_in, False), part_cols)
            bodo.barrier()
            if bodo.get_rank() == 0:
                # verify partitions are there
                ds = pq.ParquetDataset(TEST_DIR)
                assert [
                    ds.partitions.levels[i].name
                    for i in range(len(ds.partitions.levels))
                ] == part_cols
                # read bodo output with pandas
                df_test = pd.read_parquet(TEST_DIR)
                # pandas reads the partition columns as categorical, but they
                # are not categorical in input dataframe, so we do some dtype
                # conversions to be able to compare the dataframes
                for part_col in part_cols:
                    if part_col == "E":
                        # convert categorical back to date
                        df_test[part_col] = (
                            df_test[part_col].astype("datetime64").dt.date
                        )
                    elif part_col == "C":
                        # convert the bool input column to categorical of strings
                        df_in[part_col] = (
                            df_in[part_col].astype(str).astype(df_test[part_col].dtype)
                        )
                    else:
                        # convert the categorical to same input dtype
                        df_test[part_col] = df_test[part_col].astype(
                            df_in[part_col].dtype
                        )
                # use check_like=True because the order of columns has changed
                # (partition columns appear at the end after reading)
                pd.testing.assert_frame_equal(df_test, df_in, check_like=True)
                shutil.rmtree(TEST_DIR)
            bodo.barrier()
    finally:
        if bodo.get_rank() == 0:
            shutil.rmtree(TEST_DIR, ignore_errors=True)
        bodo.barrier()


def test_read_parquet_glob(datapath, memory_leak_check):
    def test_impl(filename):
        df = pd.read_parquet(filename)
        return df

    filename = datapath("int_nulls_multi.pq")
    pyout = pd.read_parquet(filename)
    # add glob patterns (only for Bodo, pandas doesn't support it)
    glob_pattern_1 = filename + "/part*.parquet"
    check_func(test_impl, (glob_pattern_1,), py_output=pyout, check_dtype=False)
    glob_pattern_2 = filename + "/part*-3af07a60-*ab59*.parquet"
    check_func(test_impl, (glob_pattern_2,), py_output=pyout, check_dtype=False)


def test_read_parquet_list_of_globs(datapath, memory_leak_check):
    """test reading when passing a list of globstrings"""

    def test_impl(filename):
        df = pd.read_parquet(filename)
        return df

    globstrings = [
        "bodo/tests/data/test_partitioned.pq/A=2/part-0000[1-2]-*.parquet",
        "bodo/tests/data/test_partitioned.pq/A=7/part-0000[5-7]-bfd81e52-9210-4ee9-84a0-5ee2ab5e6345.c000.snappy.parquet",
    ]

    import glob

    # construct expected pandas output manually (pandas doesn't support list of files)
    files = []
    for globstring in globstrings:
        files += sorted(glob.glob(globstring))
    # Arrow ParquetDatasetV2 adds the partition column to the dataset
    # when passing it a list of files that are in partitioned directories.
    # So we need to add the partition column to pandas output
    chunks = []
    import re

    regexp = re.compile("\/A=(\d+)\/")
    for f in files:
        df = pd.read_parquet(f)
        df["A"] = int(regexp.search(f).group(1))
        chunks.append(df)
    pyout = pd.concat(chunks).reset_index(drop=True)
    pyout["A"] = pyout["A"].astype("category")

    check_func(test_impl, (globstrings,), py_output=pyout, check_dtype=False)


@pytest.mark.slow
def test_read_dask_parquet(datapath, memory_leak_check):
    def test_impl(filename):
        df = pd.read_parquet(filename)
        return df

    filename = datapath("dask_data.parquet")
    check_func(test_impl, (filename,))


@pytest.mark.slow
def test_read_partitions(memory_leak_check):
    """test reading and filtering partitioned parquet data"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        if bodo.get_rank() == 0:
            table = pa.table(
                {
                    "a": range(10),
                    "b": np.random.randn(10),
                    "c": [1, 2] * 5,
                    "part": ["a"] * 5 + ["b"] * 5,
                }
            )
            pq.write_to_dataset(table, "pq_data", partition_cols=["part"])
        bodo.barrier()

        def impl(path):
            return pd.read_parquet(path)

        def impl2(path, val):
            df = pd.read_parquet(path)
            return df[df["part"] == val]

        # make sure filtering doesn't happen if df is used before filtering
        def impl3(path, val):
            df = pd.read_parquet(path)
            n = len(df)
            n2 = len(df[df["part"] == val])
            return n, n2

        # make sure filtering doesn't happen if df is used after filtering
        def impl4(path, val):
            df = pd.read_parquet(path)
            n2 = len(df[df["part"] == val])
            return len(df), n2

        # make sure filtering happens if df name is reused
        def impl5(path, val):
            df = pd.read_parquet(path)
            n = len(df[df["part"] == val])
            df = pd.DataFrame({"A": np.arange(11)})
            n += df.A.sum()
            return n

        # make sure filtering works when there are no matching files
        def impl6(path):
            df = pd.read_parquet(path)
            return df[df["part"] == "z"]

        # TODO(ehsan): make sure filtering happens if df name is reused in control flow
        # def impl7(path, val):
        #     df = pd.read_parquet(path)
        #     n = len(df[df["part"] == val])
        #     if val == "b":
        #         df = pd.DataFrame({"A": np.arange(11)})
        #         n += df.A.sum()
        #     return n

        bodo.parquet_validate_schema = False
        # check_dtype=False because Arrow 8 doesn't write pandas metadata
        # and therefore Bodo reads int64 as Int64, not maching pandas
        check_func(impl, ("pq_data",), check_dtype=False)
        bodo.parquet_validate_schema = True
        check_func(impl2, ("pq_data", "a"), check_dtype=False)
        # make sure the ParquetReader node has filters parameter set
        bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl2)
        bodo_func("pq_data", "a")
        _check_for_io_reader_filters(bodo_func, bodo.ir.parquet_ext.ParquetReader)
        check_func(impl3, ("pq_data", "a"), check_dtype=False)
        check_func(impl4, ("pq_data", "a"), check_dtype=False)
        check_func(impl5, ("pq_data", "a"), check_dtype=False)
        check_func(impl6, ("pq_data",), check_dtype=False)
        bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl5)
        bodo_func("pq_data", "a")
        _check_for_io_reader_filters(bodo_func, bodo.ir.parquet_ext.ParquetReader)
        bodo.barrier()
    finally:
        bodo.parquet_validate_schema = True
        if bodo.get_rank() == 0:
            shutil.rmtree("pq_data", ignore_errors=True)


@pytest.mark.slow
def test_read_partitions2(memory_leak_check):
    """test reading and filtering partitioned parquet data in more complex cases"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        if bodo.get_rank() == 0:
            table = pa.table(
                {
                    "a": range(10),
                    "b": np.random.randn(10),
                    "c": [1, 2, 3, 4, 5] * 2,
                    "part": ["a"] * 5 + ["b"] * 5,
                }
            )
            pq.write_to_dataset(table, "pq_data", partition_cols=["c"])
        bodo.barrier()

        def impl1(path, val):
            df = pd.read_parquet(path)
            return df[(df["c"].astype(np.int32) > val) | (df["c"] == 2)]

        # TODO(ehsan): match Index
        check_func(impl1, ("pq_data", 3), reset_index=True)
        # make sure the ParquetReader node has filters parameter set
        bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl1)
        bodo_func("pq_data", 3)
        _check_for_io_reader_filters(bodo_func, bodo.ir.parquet_ext.ParquetReader)
        bodo.barrier()
    finally:
        if bodo.get_rank() == 0:
            shutil.rmtree("pq_data", ignore_errors=True)


def test_read_partitions_predicate_dead_column(memory_leak_check):
    """test reading and filtering predicate + partition columns
    doesn't load the columns if they are unused."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        if bodo.get_rank() == 0:
            table = pa.table(
                {
                    "a": range(10),
                    "b": np.random.randn(10),
                    "c": [1, 2, 3, 4, 5] * 2,
                    "part": ["a"] * 5 + ["b"] * 5,
                }
            )
            pq.write_to_dataset(table, "pq_data", partition_cols=["c"])
        bodo.barrier()

        def impl1(path):
            df = pd.read_parquet(path)
            return df[(df["a"] > 5) & (df["c"] == 2)].b

        # TODO(ehsan): match Index
        check_func(impl1, ("pq_data",), reset_index=True)
        # make sure the ParquetReader node has filters parameter set
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl1)
            bodo_func("pq_data")
            _check_for_io_reader_filters(bodo_func, bodo.ir.parquet_ext.ParquetReader)
            check_logger_msg(stream, "Columns loaded ['b']")
        bodo.barrier()
    finally:
        if bodo.get_rank() == 0:
            shutil.rmtree("pq_data", ignore_errors=True)


@pytest.mark.slow
def test_read_partitions_cat_ordering(memory_leak_check):
    """test reading and filtering multi-level partitioned parquet data with many
    directories, to make sure order of partition values in categorical dtype
    of partition columns is consistent (same at compile time and runtime)
    and matches pandas"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        if bodo.get_rank() == 0:
            table = pa.table(
                {
                    "a": range(20),
                    "b": [1, 2, 3, 4] * 5,
                    "c": [1, 2, 3, 4, 5] * 4,
                    "part": ["a"] * 10 + ["b"] * 10,
                }
            )
            pq.write_to_dataset(table, "pq_data", partition_cols=["b", "c", "part"])
        bodo.barrier()

        def impl1(path):
            df = pd.read_parquet(path)
            return df

        def impl2(path):
            df = pd.read_parquet(path)
            return df[(df["c"] != 3) | (df["part"] == "a")]

        check_func(impl1, ("pq_data",), check_dtype=False)
        # TODO(ehsan): match Index
        check_func(impl2, ("pq_data",), reset_index=True, check_dtype=False)
        # make sure the ParquetReader node has filters parameter set
        bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl2)
        bodo_func(
            "pq_data",
        )
        _check_for_io_reader_filters(bodo_func, bodo.ir.parquet_ext.ParquetReader)
        bodo.barrier()
    finally:
        if bodo.get_rank() == 0:
            shutil.rmtree("pq_data", ignore_errors=True)


@pytest.mark.slow
def test_read_partitions_two_level(memory_leak_check):
    """test reading and filtering partitioned parquet data for two levels partitions"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        if bodo.get_rank() == 0:
            table = pa.table(
                {
                    "a": range(10),
                    "b": np.random.randn(10),
                    "c": [1, 2, 3, 4, 5] * 2,
                    "part": ["a"] * 5 + ["b"] * 5,
                }
            )
            pq.write_to_dataset(table, "pq_data", partition_cols=["c", "part"])
        bodo.barrier()

        def impl1(path, val):
            df = pd.read_parquet(path)
            return df[(df["c"].astype(np.int32) > val) | (df["part"] == "a")]

        # TODO(ehsan): match Index
        check_func(impl1, ("pq_data", 3), reset_index=True, check_dtype=False)
        # make sure the ParquetReader node has filters parameter set
        bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl1)
        bodo_func("pq_data", 3)
        _check_for_io_reader_filters(bodo_func, bodo.ir.parquet_ext.ParquetReader)
        bodo.barrier()
    finally:
        if bodo.get_rank() == 0:
            shutil.rmtree("pq_data", ignore_errors=True)


@pytest.mark.slow
def test_read_partitions_string_int(memory_leak_check):
    """test reading from a file where the partition column could have
    a mix of strings and ints
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        if bodo.get_rank() == 0:
            table = pa.table(
                {
                    "a": range(10),
                    "b": np.random.randn(10),
                    "c": ["abc", "2", "adf", "4", "5"] * 2,
                    "part": ["a"] * 5 + ["b"] * 5,
                }
            )
            pq.write_to_dataset(table, "pq_data", partition_cols=["c"])
        bodo.barrier()

        def impl1(path):
            df = pd.read_parquet(path)
            return df[(df["c"] == "abc") | (df["c"] == "4")]

        # TODO(ehsan): match Index
        check_func(impl1, ("pq_data",), reset_index=True)
        # make sure the ParquetReader node has filters parameter set
        bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl1)
        bodo_func("pq_data")
        _check_for_io_reader_filters(bodo_func, bodo.ir.parquet_ext.ParquetReader)
        bodo.barrier()
    finally:
        if bodo.get_rank() == 0:
            shutil.rmtree("pq_data", ignore_errors=True)


@pytest.mark.slow
def test_read_partitions_implicit_and_simple(memory_leak_check):
    """test reading and filtering partitioned parquet data with multiple levels
    of partitions and an implicit and"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        if bodo.get_rank() == 0:
            table = pa.table(
                {
                    "a": range(10),
                    "b": np.random.randn(10),
                    "c": [1, 2, 3, 4, 5] * 2,
                    "part": ["a"] * 5 + ["b"] * 5,
                }
            )
            pq.write_to_dataset(table, "pq_data", partition_cols=["c", "part"])
        bodo.barrier()

        def impl1(path, val):
            df = pd.read_parquet(path)
            df = df[df["part"] == "a"]
            df = df[df["c"] == val]
            return df

        def impl2(path, val):
            df = pd.read_parquet(path)
            df = df[df["part"] == "a"]
            # This function call should prevent lowering the second filter.
            sum1 = df["a"].sum()
            df = df[df["c"] == val]
            sum2 = df["a"].sum()
            return sum1 + sum2

        # TODO: match Index
        check_func(impl1, ("pq_data", 3), reset_index=True, check_dtype=False)
        # make sure the ParquetReader node has filters parameter set
        bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl1)
        bodo_func("pq_data", 3)
        _check_for_io_reader_filters(bodo_func, bodo.ir.parquet_ext.ParquetReader)
        check_func(impl1, ("pq_data", 2), reset_index=True, check_dtype=False)
    finally:
        if bodo.get_rank() == 0:
            shutil.rmtree("pq_data", ignore_errors=True)


@pytest.mark.slow
def test_read_partitions_implicit_and_detailed(memory_leak_check):
    """test reading and filtering partitioned parquet data with multiple levels
    of partitions and a complex implicit and"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        if bodo.get_rank() == 0:
            table = pa.table(
                {
                    "a": range(20),
                    "b": ["a", "b", "c", "d"] * 5,
                    "c": [1, 2, 3, 4, 5] * 4,
                    "part": ["a"] * 10 + ["b"] * 10,
                }
            )
            pq.write_to_dataset(table, "pq_data", partition_cols=["b", "c", "part"])
        bodo.barrier()

        def impl1(path, val):
            df = pd.read_parquet(path)
            df = df[(df["part"] == "a") | ((df["b"] != "d") & (df["c"] != 4))]
            df = df[((df["b"] == "a") & (df["part"] == "b")) | (df["c"] == val)]
            return df

        # TODO: match Index
        check_func(impl1, ("pq_data", 3), reset_index=True, check_dtype=False)
        # make sure the ParquetReader node has filters parameter set
        bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl1)
        bodo_func("pq_data", 3)
        _check_for_io_reader_filters(bodo_func, bodo.ir.parquet_ext.ParquetReader)
        bodo.barrier()
    finally:
        if bodo.get_rank() == 0:
            shutil.rmtree("pq_data", ignore_errors=True)


@pytest.mark.slow
def test_read_partitions_datetime(memory_leak_check):
    """test reading and filtering partitioned parquet data for datetime data"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        if bodo.get_rank() == 0:
            table = pa.table(
                {
                    "a": range(10),
                    "b": np.random.randn(10),
                    "c": [1, 2, 3, 4, 5] * 2,
                    "part": [
                        "2017-01-02",
                        "2018-01-02",
                        "2019-01-02",
                        "2017-01-02",
                        "2021-01-02",
                    ]
                    * 2,
                }
            )
            pq.write_to_dataset(table, "pq_data", partition_cols=["part"])
        bodo.barrier()

        def impl1(path, s_d, e_d):
            df = pd.read_parquet(path, columns=["c", "part", "a"])
            return df[
                (pd.to_datetime(df["part"]) >= pd.to_datetime(s_d))
                & (pd.to_datetime(df["part"]) <= pd.to_datetime(e_d))
            ]

        # With arrow8 we output a nullable integer
        check_func(
            impl1,
            ("pq_data", "2018-01-02", "2019-10-02"),
            reset_index=True,
            check_dtype=False,
        )
        # make sure the ParquetReader node has filters parameter set
        bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl1)
        bodo_func("pq_data", "2018-01-02", "2019-10-02")
        _check_for_io_reader_filters(bodo_func, bodo.ir.parquet_ext.ParquetReader)
        bodo.barrier()
    finally:
        if bodo.get_rank() == 0:
            shutil.rmtree("pq_data", ignore_errors=True)


@pytest.mark.slow
def test_read_partitions_to_datetime_format(memory_leak_check):
    """test that we don't incorrectly perform filter pushdown when to_datetime includes
    a format string."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        if bodo.get_rank() == 0:
            table = pa.table(
                {
                    "a": range(10),
                    "b": np.random.randn(10),
                    "c": [1, 2, 3, 4, 5] * 2,
                    "part": [
                        "2017-01-02",
                        "2018-01-02",
                        "2019-01-02",
                        "2017-01-02",
                        "2021-01-02",
                    ]
                    * 2,
                }
            )
            pq.write_to_dataset(table, "pq_data", partition_cols=["part"])
        bodo.barrier()

        def impl1(path, s_d, e_d):
            df = pd.read_parquet(path, columns=["c", "part", "a"])
            return df[
                (pd.to_datetime(df["part"], format="%Y-%d-%m") >= pd.to_datetime(s_d))
                & (pd.to_datetime(df["part"], format="%Y-%d-%m") <= pd.to_datetime(e_d))
            ]

        check_func(
            impl1,
            ("pq_data", "2018-01-31", "2018-02-28"),
            reset_index=True,
            check_dtype=False,
        )
        # make sure the ParquetReader node doesn't have filters parameter set
        bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl1)
        bodo_func("pq_data", "2018-01-02", "2019-10-02")
        try:
            _check_for_io_reader_filters(bodo_func, bodo.ir.parquet_ext.ParquetReader)
            # If we reach failed we have incorrectly performed filter pushdown
            passed = 0
        except AssertionError:
            passed = 1
        n_pes = bodo.get_size()
        n_passed = bodo.tests.utils.reduce_sum(passed)
        assert n_passed == n_pes, "Filter pushdown detected on at least 1 rank"
        bodo.barrier()
    finally:
        if bodo.get_rank() == 0:
            shutil.rmtree("pq_data", ignore_errors=True)


@pytest.mark.slow
def test_read_predicates_pushdown_pandas_metadata(memory_leak_check):
    """test that predicate pushdown executes when there is Pandas range metadata."""

    try:
        if bodo.get_rank() == 0:
            df = pd.DataFrame({"A": [0, 1, 2] * 10})
            df.to_parquet("pq_data")
        bodo.barrier()

        def impl(path):
            df = pd.read_parquet(path)
            df = df[(df["A"] != 2)]
            return df

        # test for predicate pushdown removing all rows
        def impl2(path):
            df = pd.read_parquet(path)
            df = df[(df["A"] == 100)]
            return df

        # TODO: Fix index
        check_func(impl, ("pq_data",), reset_index=True)
        check_func(impl2, ("pq_data",), reset_index=True)
        # make sure the ParquetReader node has filters parameter set
        bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl)
        bodo_func(
            "pq_data",
        )
        _check_for_io_reader_filters(bodo_func, bodo.ir.parquet_ext.ParquetReader)
        bodo.barrier()
    finally:
        if bodo.get_rank() == 0:
            os.remove("pq_data")


@pytest.mark.slow
def test_read_predicates_isnull(memory_leak_check):
    """test that predicate pushdown with isnull in the binops."""

    try:
        if bodo.get_rank() == 0:
            df = pd.DataFrame(
                {
                    "A": pd.Series([0, 1, 2, None] * 10, dtype="Int64"),
                    "B": [1, 2] * 20,
                }
            )
            df.to_parquet("pq_data")
        bodo.barrier()

        def impl(path):
            df = pd.read_parquet("pq_data")
            df = df[(df["B"] == 2) & (df["A"].notna())]
            return df

        # TODO: Fix index
        check_func(impl, ("pq_data",), reset_index=True)
        # make sure the ParquetReader node has filters parameter set
        bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl)
        bodo_func(
            "pq_data",
        )
        _check_for_io_reader_filters(bodo_func, bodo.ir.parquet_ext.ParquetReader)
        bodo.barrier()
    finally:
        if bodo.get_rank() == 0:
            os.remove("pq_data")


def test_read_predicates_timestamp_date(memory_leak_check):
    """Test predicate pushdown where a date column is filtered
    by a timestamp."""
    filepath = "pq_data"
    if bodo.get_rank() == 0:
        df = pd.DataFrame(
            {
                "A": [
                    datetime.date(2021, 2, 1),
                    datetime.date(2021, 3, 1),
                    datetime.date(2021, 4, 1),
                    datetime.date(2021, 5, 1),
                    datetime.date(2021, 6, 1),
                ]
                * 10,
                "B": np.arange(50),
            }
        )
        df.to_parquet(filepath)
    bodo.barrier()
    with ensure_clean(filepath):

        def impl(path):
            df = pd.read_parquet(filepath)
            df = df[df.A > pd.Timestamp(2021, 3, 30)]
            return df

        # TODO: Fix index
        check_func(impl, (filepath,), reset_index=True)
        # make sure the ParquetReader node has filters parameter set
        bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl)
        bodo_func(filepath)
        _check_for_io_reader_filters(bodo_func, bodo.ir.parquet_ext.ParquetReader)
        bodo.barrier()


def test_read_predicates_isnull_alone(memory_leak_check):
    """test that predicate pushdown with isnull as the sole filter."""

    try:
        if bodo.get_rank() == 0:
            df = pd.DataFrame(
                {
                    "A": pd.Series([0, 1, 2, None] * 10, dtype="Int64"),
                    "B": [1, 2] * 20,
                }
            )
            df.to_parquet("pq_data")
        bodo.barrier()

        def impl(path):
            df = pd.read_parquet("pq_data")
            df = df[df["A"].notna()]
            return df

        # TODO: Fix index
        check_func(impl, ("pq_data",), reset_index=True)
        # make sure the ParquetReader node has filters parameter set
        bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl)
        bodo_func(
            "pq_data",
        )
        _check_for_io_reader_filters(bodo_func, bodo.ir.parquet_ext.ParquetReader)
        bodo.barrier()
    finally:
        if bodo.get_rank() == 0:
            os.remove("pq_data")


@pytest.mark.slow
def test_read_predicates_isin(memory_leak_check):
    """test that predicate pushdown with isin"""

    try:
        if bodo.get_rank() == 0:
            df = pd.DataFrame(
                {
                    "A": pd.Series([0, 1, 2, None] * 10, dtype="Int64"),
                    "B": [1, 2] * 20,
                    "C": ["A", "B", "C", "D", "E"] * 8,
                }
            )
            df.to_parquet("pq_data")
        bodo.barrier()

        def impl1(path):
            df = pd.read_parquet("pq_data")
            df = df[df.A.isin([1, 2])]
            return df.B

        def impl2(path):
            df = pd.read_parquet("pq_data")
            df = df[df.A.isin({1, 2})]
            return df.B

        def impl3(path):
            df = pd.read_parquet("pq_data")
            df = df[df["A"].isin([1, 2]) & df["C"].isin(["B"])]
            return df.B

        def impl4(path):
            df = pd.read_parquet("pq_data")
            df = df[df["A"].isin({1, 2}) | df["C"].isin(["B"])]
            return df.B

        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            # TODO [BE-2351]: Fix index
            check_func(impl1, ("pq_data",), reset_index=True)
            # Check filter pushdown succeeded
            check_logger_msg(stream, "Filter pushdown successfully performed")
            # Check the columns were pruned
            check_logger_msg(stream, "Columns loaded ['B']")
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            # TODO [BE-2351]: Fix index
            check_func(impl2, ("pq_data",), reset_index=True)
            # Check filter pushdown succeeded
            check_logger_msg(stream, "Filter pushdown successfully performed")
            # Check the columns were pruned
            check_logger_msg(stream, "Columns loaded ['B']")
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            # TODO [BE-2351]: Fix index
            check_func(impl3, ("pq_data",), reset_index=True)
            # Check filter pushdown succeeded
            check_logger_msg(stream, "Filter pushdown successfully performed")
            # Check the columns were pruned
            check_logger_msg(stream, "Columns loaded ['B']")
        bodo.barrier()
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            # TODO [BE-2351]: Fix index
            check_func(impl4, ("pq_data",), reset_index=True)
            # Check filter pushdown succeeded
            check_logger_msg(stream, "Filter pushdown successfully performed")
            # Check the columns were pruned
            check_logger_msg(stream, "Columns loaded ['B']")
    finally:
        if bodo.get_rank() == 0:
            os.remove("pq_data")


@pytest.mark.slow
def test_read_partitions_isin(memory_leak_check):
    """test that partition pushdown with isin"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        if bodo.get_rank() == 0:
            table = pa.table(
                {
                    "A": [0, 1, 2, 3] * 10,
                    "B": [1, 2] * 20,
                    "C": ["A", "B", "C", "D", "E"] * 8,
                }
            )
            pq.write_to_dataset(table, "pq_data", partition_cols=["A"])
        bodo.barrier()

        def impl1(path):
            df = pd.read_parquet("pq_data")
            df = df[df.A.isin([1, 2])]
            return df.B

        def impl2(path):
            df = pd.read_parquet("pq_data")
            df = df[df["A"].isin([1, 2]) & df["C"].isin(["B"])]
            return df.B

        def impl3(path):
            df = pd.read_parquet("pq_data")
            df = df[df["A"].isin([1, 2]) | df["C"].isin(["B"])]
            return df.B

        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            # TODO: Fix index
            check_func(impl1, ("pq_data",), reset_index=True, check_dtype=False)
            # Check filter pushdown succeeded
            check_logger_msg(stream, "Filter pushdown successfully performed")
            # Check the columns were pruned
            check_logger_msg(stream, "Columns loaded ['B']")
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            # TODO: Fix index
            check_func(impl2, ("pq_data",), reset_index=True, check_dtype=False)
            # Check filter pushdown succeeded
            check_logger_msg(stream, "Filter pushdown successfully performed")
            # Check the columns were pruned
            check_logger_msg(stream, "Columns loaded ['B']")
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            # TODO: Fix index
            check_func(impl3, ("pq_data",), reset_index=True, check_dtype=False)
            # Check filter pushdown succeeded
            check_logger_msg(stream, "Filter pushdown successfully performed")
            # Check the columns were pruned
            check_logger_msg(stream, "Columns loaded ['B']")
        bodo.barrier()
    finally:
        if bodo.get_rank() == 0:
            shutil.rmtree("pq_data", ignore_errors=True)


@pytest.mark.slow
def test_read_predicates_and_or(memory_leak_check):
    """test that predicate pushdown with and/or in the expression."""

    try:
        if bodo.get_rank() == 0:
            df = pd.DataFrame(
                {
                    "A": [0, 1, 2, 3] * 10,
                    "B": [1, 2, 3, 4, 5] * 8,
                }
            )
            df.to_parquet("pq_data")
        bodo.barrier()

        def impl(path):
            df = pd.read_parquet("pq_data")
            df = df[
                (((df["A"] == 2) | (df["B"] == 1)) & (df["B"] != 4))
                & (((df["A"] == 3) | (df["B"] == 5)) & (df["B"] != 2))
            ]
            return df

        # TODO: Fix index
        check_func(impl, ("pq_data",), reset_index=True)
        # make sure the ParquetReader node has filters parameter set
        bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl)
        bodo_func(
            "pq_data",
        )
        _check_for_io_reader_filters(bodo_func, bodo.ir.parquet_ext.ParquetReader)
        bodo.barrier()
    finally:
        if bodo.get_rank() == 0:
            os.remove("pq_data")


@pytest.mark.slow
def test_read_partitions_large(memory_leak_check):
    """
    test reading and filtering partitioned parquet data with large number of partitions
    """

    try:
        if bodo.get_rank() == 0:
            # The number of dates can't exceed 1024 because that is the default
            # max_partitions of Arrow when writing parquet. XXX how to pass
            # max_partitions option to Arrow from df.to_parquet?
            I = pd.date_range("2018-01-03", "2020-10-05")
            df = pd.DataFrame(
                {"A": np.repeat(I.values, 100), "B": np.arange(100 * len(I))}
            )
            df.to_parquet("pq_data", partition_cols="A")
        bodo.barrier()

        def impl1(path, s_d, e_d):
            df = pd.read_parquet(path, columns=["A", "B"])
            return df[
                (pd.to_datetime(df["A"].astype(str)) >= pd.to_datetime(s_d))
                & (pd.to_datetime(df["A"].astype(str)) <= pd.to_datetime(e_d))
            ]

        check_func(impl1, ("pq_data", "2018-01-02", "2019-10-02"), reset_index=True)
        # make sure the ParquetReader node has filters parameter set
        bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl1)
        bodo_func("pq_data", "2018-01-02", "2019-10-02")
        _check_for_io_reader_filters(bodo_func, bodo.ir.parquet_ext.ParquetReader)
        bodo.barrier()
    finally:
        if bodo.get_rank() == 0:
            shutil.rmtree("pq_data", ignore_errors=True)


@pytest.mark.skip(
    reason="""This test is waiting on support for pushing filters past column filters.
See https://bodo.atlassian.net/browse/BE-1522"""
)
def test_filter_pushdown_past_column_filters():
    """Tests that filters can be pushed past loc's that exclusivly filter the dataframe's columns."""

    dict = {}
    arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for i in ["A", "B", "C", "D"]:
        dict[i] = arr

    df = pd.DataFrame(dict)

    stream = io.StringIO()
    logger = create_string_io_logger(stream)

    # TODO: We currnetly clear the logger stream before each call to check_func
    # via calling stream.seek/truncate. We should investigate if this actually
    # fully clears the logger, or if additional work is needed.
    with set_logging_stream(logger, 1):
        try:
            if bodo.get_rank() == 0:
                df.to_parquet("pq_data", partition_cols="A")

            def impl1():
                df = pd.read_parquet("pq_data")
                df = df[["A", "C"]]
                df = df[df["C"] > 5]
                return df

            def impl2():
                df = pd.read_parquet("pq_data")
                df = df.loc[:, ["B", "A", "D"]]
                df = df[df["D"] < 4]
                return df

            def impl3():
                df = pd.read_parquet("pq_data")
                df = df.loc[:, ["C", "B", "A", "D"]]
                df = df.loc[:, ["B", "A", "D"]]
                df = df.loc[:, ["B", "D"]]
                df = df.loc[:, ["D"]]
                df = df[df["D"] < 4]
                return df

            check_func
            check_func(impl1, ())
            check_logger_msg(stream, f"TODO")
            stream.truncate(0)
            stream.seek(0)
            check_func(impl2, ())
            check_logger_msg(stream, f"TODO")
            stream.truncate(0)
            stream.seek(0)
            check_func(impl3, ())
            check_logger_msg(stream, f"TODO")
            stream.truncate(0)
            stream.seek(0)
            bodo.barrier()
        finally:
            if bodo.get_rank() == 0:
                shutil.rmtree("pq_data", ignore_errors=True)

    def impl1():
        df = pd.DataFrame(
            {"A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "B": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        )
        df = df[["A", "C"]]
        return df

    def impl2():
        df = pd.DataFrame(
            {"A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "B": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        )
        df = df.loc[df["A"] > 1, :]
        return df


@pytest.mark.slow
def test_read_pq_head_only(datapath, memory_leak_check):
    """
    test reading only shape and/or head from Parquet file if possible
    (limit pushdown)
    """

    # read both shape and head()
    def impl1(path):
        df = pd.read_parquet(path)
        return df.shape, df.head(4)

    # shape only
    def impl2(path):
        df = pd.read_parquet(path)
        return len(df)

    # head only
    def impl3(path):
        df = pd.read_parquet(path)
        return df.head()

    # head and shape without table format
    def impl4(path):
        df = pd.read_parquet(path).loc[:, ["A"]]
        return df.shape, df.head(4)

    # large file
    fname = datapath("int_nulls_multi.pq")
    check_func(impl1, (fname,), check_dtype=False)
    check_func(impl2, (fname,), check_dtype=False)
    check_func(impl4, (fname,), check_dtype=False)
    check_func(impl3, (fname,), check_dtype=False)
    # small file with Index data
    check_func(impl1, (datapath("index_test2.pq"),), check_dtype=False)
    # make sure head-only read is recognized correctly
    bodo_func = bodo.jit(pipeline_class=DistTestPipeline)(impl1)
    bodo_func(fname)
    _check_for_pq_read_head_only(bodo_func)
    bodo_func = bodo.jit(pipeline_class=DistTestPipeline)(impl2)
    bodo_func(fname)
    _check_for_pq_read_head_only(bodo_func, has_read=False)
    bodo_func = bodo.jit(pipeline_class=DistTestPipeline)(impl3)
    bodo_func(fname)
    _check_for_pq_read_head_only(bodo_func)
    bodo_func = bodo.jit(pipeline_class=DistTestPipeline)(impl4)
    bodo_func(fname)
    _check_for_pq_read_head_only(bodo_func)
    # partitioned data
    try:
        if bodo.get_rank() == 0:
            # The number of dates can't exceed 1024 because that is the default
            # max_partitions of Arrow when writing parquet. XXX how to pass
            # max_partitions option to Arrow from df.to_parquet?
            I = pd.date_range("2018-01-03", "2020-10-05")
            df = pd.DataFrame(
                {"A": np.repeat(I.values, 100), "B": np.arange(100 * len(I))}
            )
            df.to_parquet("pq_data", partition_cols="A")
        bodo.barrier()

        check_func(impl1, ("pq_data",), check_dtype=False)
        bodo_func = bodo.jit(pipeline_class=DistTestPipeline)(impl1)
        bodo_func("pq_data")
        _check_for_pq_read_head_only(bodo_func)
        bodo.barrier()
    finally:
        if bodo.get_rank() == 0:
            shutil.rmtree("pq_data", ignore_errors=True)


def _check_for_pq_read_head_only(bodo_func, has_read=True):
    """make sure head-only parquet read optimization is recognized"""
    fir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert hasattr(fir, "meta_head_only_info")
    assert not has_read or fir.meta_head_only_info[0] is not None


@pytest.mark.slow
def test_write_parquet_empty_chunks(memory_leak_check):
    """Here we check that our to_parquet output in distributed mode
    (directory of parquet files) can be read by pandas even when some
    processes have empty chunks"""

    def f(n, write_filename):
        df = pd.DataFrame({"A": np.arange(n)})
        df.to_parquet(write_filename)

    write_filename = "test__empty_chunks.pq"
    n = 1  # make dataframe of length 1 so that rest of processes have empty chunk
    try:
        bodo.jit(f)(n, write_filename)
        bodo.barrier()
        if bodo.get_rank() == 0:
            df = pd.read_parquet(write_filename)
    finally:
        if bodo.get_rank() == 0:
            shutil.rmtree(write_filename)


@pytest.mark.slow
def test_write_parquet_decimal(datapath, memory_leak_check):
    """Here we check that we can write the data read from decimal1.pq directory
    (has columns that use a precision and scale different from our default).
    See test_write_parquet above for main parquet write decimal test"""

    def write(read_path, write_filename):
        df = pd.read_parquet(read_path)
        df.to_parquet(write_filename)

    write_filename = "test__write_decimal1.pq"
    try:
        bodo.jit(write)(datapath("decimal1.pq"), write_filename)
        bodo.barrier()
        if bodo.get_rank() == 0:
            df1 = pd.read_parquet(datapath("decimal1.pq"))
            df2 = pd.read_parquet(write_filename)
            pd.testing.assert_frame_equal(df1, df2)
    finally:
        if bodo.get_rank() == 0:
            shutil.rmtree(write_filename)


def test_write_parquet_params(memory_leak_check):
    def write1(df, filename):
        df.to_parquet(compression="snappy", path=filename)

    def write2(df, filename):
        df.to_parquet(path=filename, index=None, compression="gzip")

    def write3(df, filename):
        df.to_parquet(path=filename, index=True, compression="brotli")

    def write4(df, filename):
        df.to_parquet(path=filename, index=False, compression=None)

    S1 = ["¡Y tú quién te crees?", "🐍⚡", "大处着眼，小处着手。"] * 4
    S2 = ["abc¡Y tú quién te crees?", "dd2🐍⚡", "22 大处着眼，小处着手。"] * 4
    df = pd.DataFrame({"A": S1, "B": S2})
    # set a numeric index (not range)
    df.index = [v**2 for v in range(len(df))]

    for mode in ["sequential", "1d-distributed"]:
        pd_fname = "test_io___pandas.pq"
        if mode == "sequential":
            bodo_fname = str(bodo.get_rank()) + "_test_io___bodo.pq"
        else:
            # in parallel mode, each process writes its piece to a separate
            # file in the same directory
            bodo_fname = "test_io___bodo_pq_write_dir"
        for func in [write1, write2, write3, write4]:
            try:
                if mode == "sequential":
                    bodo_func = bodo.jit(func)
                    data = df
                elif mode == "1d-distributed":
                    bodo_func = bodo.jit(func, all_args_distributed_block=True)
                    data = _get_dist_arg(df, False)
                if bodo.get_rank() == 0:
                    func(df, pd_fname)  # write with pandas
                bodo.barrier()
                bodo_func(data, bodo_fname)
                bodo.barrier()
                df_a = pd.read_parquet(pd_fname)
                df_b = pd.read_parquet(bodo_fname)
                pd.testing.assert_frame_equal(df_a, df_b)
                bodo.barrier()
            finally:
                # cleanup
                clean_pq_files(mode, pd_fname, bodo_fname)
                bodo.barrier()


def test_write_parquet_dict(memory_leak_check):
    """
    Test to_parquet when dictionary arrays are used
    in a DataFrame.
    """

    import pyarrow as pa
    import pyarrow.parquet as pq

    @bodo.jit(distributed=["arr1", "arr2"])
    def impl(arr1, arr2):
        df = pd.DataFrame(
            {
                "A": arr1,
                "B": arr2,
            }
        )
        df.to_parquet("arr_dict_test.pq", index=False)

    arr1 = pa.array(
        ["abc", "b", None, "abc", None, "b", "cde"] * 4,
        type=pa.dictionary(pa.int32(), pa.string()),
    )
    arr2 = pa.array(
        ["gh", "b", "gh", "eg", None, "b", "eg"] * 4,
        type=pa.dictionary(pa.int32(), pa.string()),
    )
    arr1 = _get_dist_arg(arr1, False)
    arr2 = _get_dist_arg(arr2, False)
    impl(arr1, arr2)
    passed = 1
    bodo.barrier()
    if bodo.get_rank() == 0:
        try:
            # Check the output.
            result = pd.read_parquet("arr_dict_test.pq")
            py_output = pd.DataFrame(
                {
                    "A": ["abc", "b", None, "abc", None, "b", "cde"] * 4,
                    "B": ["gh", "b", "gh", "eg", None, "b", "eg"] * 4,
                }
            )
            passed = _test_equal_guard(
                result,
                py_output,
            )

            # Check the schema to ensure its stored as string
            bodo_table = pq.read_table("arr_dict_test.pq")
            schema = bodo_table.schema
            expected_dtype = pa.string()
            for c in py_output.columns:
                assert (
                    schema.field(c).type == expected_dtype
                ), f"Field '{c}' has an incorrect type"
        except Exception:
            passed = 0
        finally:
            shutil.rmtree("arr_dict_test.pq")
    n_passed = reduce_sum(passed)
    assert (
        n_passed == bodo.get_size()
    ), "to_parquet output doesn't match expected pandas output"


def test_write_parquet_dict_table(memory_leak_check):
    """
    Test to_parquet when dictionary arrays are used
    in a DataFrame containing a table representation.

    To do this consistently we load heavily compressed data
    from parquet.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if bodo.get_rank() == 0:
        df = pd.DataFrame(
            {
                "A": ["a" * 100, "b" * 100, None, "c" * 100, "a" * 100] * 1000,
                "B": ["feefw" * 50, "bf3" * 500, None, "32c" * 20, "a"] * 1000,
            }
        )
        df.to_parquet("dummy_source.pq", index=False)
    bodo.barrier()

    @bodo.jit
    def impl():
        df = pd.read_parquet("dummy_source.pq")
        df.to_parquet("arr_dict_test.pq", index=False)

    impl()
    bodo.barrier()
    passed = 1
    if bodo.get_rank() == 0:
        try:
            # Check the output.
            result = pd.read_parquet("arr_dict_test.pq")
            py_output = pd.read_parquet("dummy_source.pq")
            passed = _test_equal_guard(
                result,
                py_output,
            )

            # Check the schema to ensure its stored as string
            bodo_table = pq.read_table("arr_dict_test.pq")
            schema = bodo_table.schema
            expected_dtype = pa.string()
            for c in py_output.columns:
                assert (
                    schema.field(c).type == expected_dtype
                ), f"Field '{c}' has an incorrect type"
        except Exception:
            passed = 0
        finally:
            shutil.rmtree("arr_dict_test.pq")
            os.remove("dummy_source.pq")
    n_passed = reduce_sum(passed)
    assert (
        n_passed == bodo.get_size()
    ), "to_parquet output doesn't match expected pandas output"


def test_write_parquet_row_group_size(memory_leak_check):
    """Test df.to_parquet(..., row_group_size=n)"""
    if bodo.get_rank() == 0:

        # We don't need to test the distributed case, because in the distributed
        # case each rank writes its own data to a separate file. row_group_size
        # is passed to Arrow WriteTable in the same way regardless

        @bodo.jit(replicated=["df"])
        def impl(df, output_filename, n):
            df.to_parquet(output_filename, row_group_size=n)

        output_filename = "bodo_temp.pq"
        try:
            df = pd.DataFrame({"A": range(93)})
            impl(df, output_filename, 20)
            m = pq.ParquetFile(output_filename).metadata
            assert [m.row_group(i).num_rows for i in range(m.num_row_groups)] == [
                20,
                20,
                20,
                20,
                13,
            ]
        finally:
            os.remove(output_filename)
    bodo.barrier()


def test_write_parquet_no_empty_files(memory_leak_check):
    """Test that when a rank has no data, it doesn't write a file"""
    # The test is most useful when run with multiple ranks
    # but should pass on a single rank too.

    @bodo.jit(distributed=["df"])
    def impl(df, out_name):
        df.to_parquet(out_name)

    if bodo.get_rank() == 0:
        df = pd.DataFrame({"A": [1], "B": [1]})
    else:
        df = pd.DataFrame({"A": [], "B": []})

    output_filename = "1row.pq"
    with ensure_clean_dir(output_filename):
        impl(df, output_filename)
        bodo.barrier()
        # Only rank 0 should've written a file
        assert len(os.listdir(output_filename)) == 1


def test_write_parquet_file_prefix(memory_leak_check):
    """Test to_parquet distributed case when file prefix is provided"""

    @bodo.jit(distributed=["df"])
    def impl(df, out_name):
        df.to_parquet(out_name, _bodo_file_prefix="test-")

    if bodo.get_rank() == 0:
        df = pd.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"]})
    else:
        df = pd.DataFrame({"A": [4, 5, 6], "B": ["d", "e", "f"]})

    output_filename = "file_prefix_test.pq"
    with ensure_clean_dir(output_filename):
        impl(df, output_filename)
        bodo.barrier()
        files = os.listdir(output_filename)
        assert all(file.startswith("test-") for file in files)


def test_csv_bool1(datapath, memory_leak_check):
    """Test boolean data in CSV files.
    Also test extra separator at the end of the file
    which requires index_col=False.
    """

    def test_impl(fname):
        dtype = {"A": "int", "B": "bool", "C": "float"}
        return pd.read_csv(
            fname, names=list(dtype.keys()), dtype=dtype, index_col=False
        )

    # passing file name as argument to exercise value-based dispatch
    fname = datapath("csv_data_bool1.csv")
    check_func(test_impl, (fname,))

    compressed_names = compress_file(fname)
    try:
        for fname in compressed_names:
            check_func(test_impl, (fname,))
    finally:
        remove_files(compressed_names)


def test_csv_int_na1(datapath, memory_leak_check):
    fname = datapath("csv_data_int_na1.csv")

    def test_impl(fname):
        dtype = {"A": "int", "B": "Int32"}
        return pd.read_csv(
            fname, names=list(dtype.keys()), dtype=dtype, compression="infer"
        )

    check_func(test_impl, (fname,))

    compressed_names = compress_file(fname)
    try:
        for f in compressed_names:
            check_func(test_impl, (f,))
    finally:
        remove_files(compressed_names)

    # test reading csv file with non ".csv" extension
    new_fname = fname[:-4] + ".custom"  # change .csv to .custom
    if bodo.get_rank() == 0:
        os.rename(fname, new_fname)
    bodo.barrier()
    try:
        check_func(test_impl, (new_fname,))
    finally:
        if bodo.get_rank() == 0 and os.path.exists(new_fname):
            os.rename(new_fname, fname)
        bodo.barrier()


def test_csv_int_na2(datapath, memory_leak_check):
    fname = datapath("csv_data_int_na1.csv")

    def test_impl(fname, compression):
        dtype = {"A": "int", "B": pd.Int32Dtype()}
        return pd.read_csv(
            fname, names=list(dtype.keys()), dtype=dtype, compression=compression
        )

    check_func(test_impl, (fname, "infer"))

    compressed_names = compress_file(fname, dummy_extension=".dummy")
    try:
        check_func(test_impl, (compressed_names[0], "gzip"))
        check_func(test_impl, (compressed_names[1], "bz2"))
    finally:
        remove_files(compressed_names)


def test_csv_bool_na(datapath, memory_leak_check):
    fname = datapath("bool_nulls.csv")

    def test_impl(fname):
        # TODO: support column 1 which is bool with NAs when possible with
        # Pandas dtypes
        # see Pandas GH20591
        dtype = {"ind": "int32", "B": "bool"}
        return pd.read_csv(fname, names=list(dtype.keys()), dtype=dtype, usecols=[0, 2])

    check_func(test_impl, (fname,))

    compressed_names = compress_file(fname)
    try:
        for fname in compressed_names:
            check_func(test_impl, (fname,))
    finally:
        remove_files(compressed_names)


def test_csv_fname_comp(datapath, memory_leak_check):
    """Test CSV read with filename computed across Bodo functions"""

    @bodo.jit
    def test_impl(data_folder):
        return load_func(data_folder)

    @bodo.jit
    def load_func(data_folder):
        fname = data_folder + "/csv_data1.csv"
        return pd.read_csv(fname, header=None)

    data_folder = os.path.join("bodo", "tests", "data")
    # should not raise exception
    test_impl(data_folder)


def test_write_csv_parallel_unicode(memory_leak_check):
    def test_impl(df, fname):
        df.to_csv(fname)

    bodo_func = bodo.jit(all_args_distributed_block=True)(test_impl)
    S1 = ["¡Y tú quién te crees?", "🐍⚡", "大处着眼，小处着手。"] * 2
    S2 = ["abc¡Y tú quién te crees?", "dd2🐍⚡", "22 大处着眼，小处着手。"] * 2
    df = pd.DataFrame({"A": S1, "B": S2})
    hp_fname = "test_write_csv1_bodo_par_unicode.csv"
    pd_fname = "test_write_csv1_pd_par_unicode.csv"
    with ensure_clean(pd_fname), ensure_clean(hp_fname):
        start, end = get_start_end(len(df))
        bdf = df.iloc[start:end]
        bodo_func(bdf, hp_fname)
        bodo.barrier()
        if get_rank() == 0:
            test_impl(df, pd_fname)
            pd.testing.assert_frame_equal(pd.read_csv(hp_fname), pd.read_csv(pd_fname))


@pytest.mark.smoke
def test_h5_read_seq(datapath, memory_leak_check):
    def test_impl(fname):
        f = h5py.File(fname, "r")
        X = f["points"][:]
        f.close()
        return X

    # passing function name as value to test value-based dispatch
    fname = datapath("lr.hdf5")
    check_func(test_impl, (fname,), only_seq=True)


def test_h5_read_const_infer_seq(datapath, memory_leak_check):
    fname = datapath("")

    def test_impl():
        p = fname + "lr"
        f = h5py.File(p + ".hdf5", "r")
        s = "po"
        X = f[s + "ints"][:]
        f.close()
        return X

    check_func(test_impl, (), only_seq=True)


def test_h5_read_parallel(datapath, memory_leak_check):
    fname = datapath("lr.hdf5")

    def test_impl():
        f = h5py.File(fname, "r")
        X = f["points"][:]
        Y = f["responses"][:]
        f.close()
        return X.sum() + Y.sum()

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_almost_equal(bodo_func(), test_impl(), decimal=2)
    assert count_array_REPs() == 0
    assert count_parfor_REPs() == 0


@pytest.mark.skip(
    "H5py bug breaks boolean arrays, https://github.com/h5py/h5py/issues/1847"
)
def test_h5_filter(datapath, memory_leak_check):
    fname = datapath("h5_test_filter.h5")

    def test_impl():
        f = h5py.File(fname, "r")
        b = np.arange(11) % 3 == 0
        X = f["test"][b, :, :, :]
        f.close()
        return X

    bodo_func = bodo.jit(distributed=["X"])(test_impl)
    n = 4  # len(test_impl())
    start, end = get_start_end(n)
    np.testing.assert_allclose(bodo_func(), test_impl()[start:end])


def test_h5_slice1(datapath, memory_leak_check):
    fname = datapath("h5_test_filter.h5")

    def test_impl():
        f = h5py.File(fname, "r")
        X = f["test"][:, 1:, :, :]
        f.close()
        return X

    bodo_func = bodo.jit(distributed=["X"])(test_impl)
    n = 11  # len(test_impl())
    start, end = get_start_end(n)
    np.testing.assert_allclose(bodo_func(), test_impl()[start:end])


def test_h5_slice2(datapath, memory_leak_check):
    fname = datapath("lr.hdf5")

    def test_impl():
        f = h5py.File(fname, "r")
        X = f["points"][:, 1]
        f.close()
        return X

    bodo_func = bodo.jit(distributed=["X"])(test_impl)
    n = 101  # len(test_impl())
    start, end = get_start_end(n)
    np.testing.assert_allclose(bodo_func(), test_impl()[start:end])


def test_h5_read_group(datapath, memory_leak_check):
    fname = datapath("test_group_read.hdf5")

    def test_impl():
        f = h5py.File(fname, "r")
        g1 = f["G"]
        X = g1["data"][:]
        f.close()
        return X.sum()

    bodo_func = bodo.jit(test_impl)
    assert bodo_func() == test_impl()


def test_h5_file_keys(datapath, memory_leak_check):
    fname = datapath("test_group_read.hdf5")

    def test_impl():
        f = h5py.File(fname, "r")
        s = 0
        for gname in f.keys():
            X = f[gname]["data"][:]
            s += X.sum()
        f.close()
        return s

    bodo_func = bodo.jit(test_impl, h5_types={"X": bodo.int64[:]})
    assert bodo_func() == test_impl()
    # test using locals for typing
    bodo_func = bodo.jit(test_impl, locals={"X": bodo.int64[:]})
    assert bodo_func() == test_impl()


def test_h5_group_keys(datapath, memory_leak_check):
    fname = datapath("test_group_read.hdf5")

    def test_impl():
        f = h5py.File(fname, "r")
        g1 = f["G"]
        s = 0
        for dname in g1.keys():
            X = g1[dname][:]
            s += X.sum()
        f.close()
        return s

    bodo_func = bodo.jit(test_impl, h5_types={"X": bodo.int64[:]})
    assert bodo_func() == test_impl()


@pytest.mark.smoke
def test_h5_write(memory_leak_check):
    # run only on 1 processor
    if bodo.get_size() != 1:
        return

    def test_impl(A, fname):
        f = h5py.File(fname, "w")
        dset1 = f.create_dataset("A", A.shape, "f8")
        dset1[:] = A
        f.close()

    fname = "test_w.hdf5"
    n = 11
    A = np.arange(n).astype(np.float64)
    with ensure_clean(fname):
        bodo.jit(
            test_impl, returns_maybe_distributed=False, args_maybe_distributed=False
        )(A, fname)
        f = h5py.File(fname, "r")
        A2 = f["A"][:]
        f.close()
        np.testing.assert_array_equal(A, A2)


def test_h5_group_write(memory_leak_check):
    # run only on 1 processor
    if bodo.get_size() != 1:
        return

    def test_impl(A, fname):
        f = h5py.File(fname, "w")
        g1 = f.create_group("AA")
        g2 = g1.create_group("BB")
        dset1 = g2.create_dataset("A", A.shape, "f8")
        dset1[:] = A
        f.close()

    fname = "test_w.hdf5"
    n = 11
    A = np.arange(n).astype(np.float64)
    with ensure_clean(fname):
        bodo.jit(test_impl)(A, fname)
        f = h5py.File(fname, "r")
        A2 = f["AA"]["BB"]["A"][:]
        f.close()
        np.testing.assert_array_equal(A, A2)


@pytest.mark.smoke
def test_np_io1(datapath, memory_leak_check):
    fname = datapath("np_file1.dat")

    def test_impl():
        A = np.fromfile(fname, np.float64)
        return A

    bodo_func = bodo.jit(test_impl, returns_maybe_distributed=False)
    np.testing.assert_almost_equal(bodo_func(), test_impl())


def test_np_io2(datapath, memory_leak_check):
    fname = datapath("np_file1.dat")
    # parallel version
    def test_impl():
        A = np.fromfile(fname, np.float64)
        return A.sum()

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_almost_equal(bodo_func(), test_impl())
    assert count_array_REPs() == 0
    assert count_parfor_REPs() == 0


def test_np_io3(memory_leak_check):
    def test_impl(A):
        if get_rank() == 0:
            A.tofile("np_file_3.dat")

    bodo_func = bodo.jit(test_impl)
    n = 111
    np.random.seed(0)
    A = np.random.ranf(n)
    with ensure_clean("np_file_3.dat"):
        bodo_func(A)
        if get_rank() == 0:
            B = np.fromfile("np_file_3.dat", np.float64)
            np.testing.assert_almost_equal(A, B)


def test_np_io4(memory_leak_check):
    # parallel version
    def test_impl(n):
        A = np.arange(n)
        A.tofile("np_file_4.dat")

    bodo_func = bodo.jit(test_impl)
    n1 = 111000
    n2 = 111
    A1 = np.arange(n1)
    A2 = np.arange(n2)
    with ensure_clean("np_file_4.dat"):
        bodo_func(n1)
        B1 = np.fromfile("np_file_4.dat", np.int64)
        np.testing.assert_almost_equal(A1, B1)

        bodo.barrier()
        bodo_func(n2)
        B2 = np.fromfile("np_file_4.dat", np.int64)
        np.testing.assert_almost_equal(A2, B2)


def test_np_io5(datapath, memory_leak_check):
    # Test count optional argument small
    fname = datapath("np_file1.dat")

    def test_impl():
        A = np.fromfile(fname, np.float64, count=10)
        return A

    check_func(test_impl, (), only_seq=True)


def test_np_io6(datapath, memory_leak_check):
    # Test count optional argument huge
    fname = datapath("np_file1.dat")

    def test_impl():
        A = np.fromfile(fname, np.float64, count=100000000)
        return A

    check_func(test_impl, (), only_seq=True)


def test_np_io7(datapath, memory_leak_check):
    # Test offset optional argument
    fname = datapath("np_file1.dat")

    def test_impl():
        A = np.fromfile(fname, np.float64, offset=10)
        return A

    check_func(test_impl, (), only_seq=True)


def test_np_io8(datapath, memory_leak_check):
    # Test offset and count optional arguments
    fname = datapath("np_file1.dat")

    def test_impl():
        A = np.fromfile(fname, np.float64, offset=10, count=10)
        return A

    check_func(test_impl, (), only_seq=True)


def test_np_io9(datapath, memory_leak_check):
    # Test count optional argument small, parallel
    fname = datapath("np_file1.dat")

    def test_impl():
        A = np.fromfile(fname, np.float64, count=10)
        return A.sum()

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_almost_equal(bodo_func(), test_impl())


def test_np_io10(datapath, memory_leak_check):
    # Test count optional argument huge, parallel
    fname = datapath("np_file1.dat")

    def test_impl():
        A = np.fromfile(fname, np.float64, count=100000000)
        return A.sum()

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_almost_equal(bodo_func(), test_impl())


def test_np_io11(datapath, memory_leak_check):
    # Test offset optional argument, parallel
    fname = datapath("np_file1.dat")

    def test_impl():
        A = np.fromfile(fname, np.float64, offset=10)
        print(A.shape, A)
        return A.sum()

    bodo_func = bodo.jit(test_impl)
    v1 = bodo_func()
    v2 = test_impl()
    np.testing.assert_almost_equal(v1, v2)


def test_np_io12(datapath, memory_leak_check):
    # Test offset and count optional arguments, parallel
    fname = datapath("np_file1.dat")

    def test_impl():
        A = np.fromfile(fname, np.float64, offset=10, count=10)
        return A.sum()

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_almost_equal(bodo_func(), test_impl())


@pytest.mark.slow
def test_np_io13(datapath, memory_leak_check):
    # Test fromfile with all keyword arguments
    fname = datapath("np_file1.dat")

    def test_impl():
        A = np.fromfile(file=fname, dtype=np.float64, count=10, sep="", offset=10)
        return A.sum()

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_almost_equal(bodo_func(), test_impl())


@pytest.mark.slow
def test_np_io14(datapath, memory_leak_check):
    # Test fromfile with all positional arguments
    fname = datapath("np_file1.dat")

    def test_impl():
        A = np.fromfile(fname, np.float64, 10, "", 10)
        return A.sum()

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_almost_equal(bodo_func(), test_impl())


def test_csv_double_box(datapath, memory_leak_check):
    """Make sure boxing the output of read_csv() twice doesn't cause crashes
    See dataframe boxing function for extra incref of native arrays.
    """
    fname = datapath("csv_data1.csv")

    def test_impl():
        df = pd.read_csv(fname, header=None)
        print(df)
        return df

    bodo_func = bodo.jit(test_impl)
    print(bodo_func())


def test_csv_header_none(datapath, memory_leak_check):
    """Test header=None in read_csv() when column names are not provided, so numbers
    should be assigned as column names.
    """
    fname = datapath("csv_data1.csv")

    def test_impl():
        return pd.read_csv(fname, header=None)

    bodo_func = bodo.jit(returns_maybe_distributed=False)(test_impl)
    b_df = bodo_func()
    p_df = test_impl()
    # convert column names from integer to string since Bodo only supports string names
    p_df.columns = [str(c) for c in p_df.columns]
    pd.testing.assert_frame_equal(b_df, p_df, check_dtype=False)


def test_csv_sep_arg(datapath, memory_leak_check):
    """Test passing 'sep' argument as JIT argument in read_csv()"""
    fname = datapath("csv_data2.csv")

    def test_impl(fname, sep):
        return pd.read_csv(fname, sep=sep)

    check_func(
        test_impl,
        (
            fname,
            "|",
        ),
        check_dtype=False,
    )
    # testing reading whole lines with sep="\n"
    # This is no long supported in pandas 1.4
    if pandas_version == (1, 3):
        check_func(
            test_impl,
            (
                fname,
                "\n",
            ),
            check_dtype=False,
        )
    else:
        assert pandas_version == (1, 4), "Check if this test is still valid"
        with pytest.raises(
            BodoError, match=r".*Specified \\n as separator or delimiter.*"
        ):
            bodo.jit(test_impl)(fname, "\n")

    compressed_names = compress_file(fname)
    try:
        for fname in compressed_names:
            check_func(
                test_impl,
                (
                    fname,
                    "|",
                ),
                check_dtype=False,
            )
    finally:
        remove_files(compressed_names)


def test_csv_int_none(datapath, memory_leak_check):
    """Make sure int columns that have nulls later in the data (not seen by our 100 row
    type inference step) are handled properly
    """

    def test_impl(fname):
        df = pd.read_csv(fname)
        return df

    fname = datapath("csv_data_int_none.csv")
    check_func(test_impl, (fname,), check_dtype=False)


def test_csv_dtype_col_ind(datapath, memory_leak_check):
    """Make sure integer column index works for referring to columns in 'dtype'"""

    def test_impl(fname):
        df = pd.read_csv(fname, dtype={3: str})
        return df

    fname = datapath("csv_data_infer1.csv")
    check_func(test_impl, (fname,))


def test_csv_usecols_names(datapath, memory_leak_check):
    """test passing column names in usecols"""

    def test_impl(fname):
        df = pd.read_csv(fname, usecols=["A", "B"])
        return df

    fname = datapath("csv_data_infer1.csv")
    check_func(test_impl, (fname,), check_dtype=False)


def test_csv_sep_whitespace(datapath, memory_leak_check):
    """test that using all whitespace"""

    def test_impl(fname):
        df = pd.read_csv(fname, sep=r"\s+")
        return df

    fname = datapath("csv_data_infer1.csv")
    check_func(test_impl, (fname,), check_dtype=False)


def test_csv_spark_header(datapath, memory_leak_check):
    """Test reading Spark csv outputs containing header & infer dtypes"""
    fname1 = datapath("example_single.csv")
    fname2 = datapath("example_multi.csv")

    def test_impl(fname):
        return pd.read_csv(fname)

    py_output = pd.read_csv(datapath("example.csv"))
    check_func(test_impl, (fname1,), py_output=py_output)
    check_func(test_impl, (fname2,), py_output=py_output)

    for fname in (fname1, fname2):
        compress_dir(fname)
        try:
            check_func(test_impl, (fname,), py_output=py_output)
        finally:
            uncompress_dir(fname)

    # test reading a directory of csv files not ending in ".csv" extension
    dirname = fname2
    if bodo.get_rank() == 0:
        # rename all .csv files in directory to .custom
        for f in os.listdir(dirname):
            if f.endswith(".csv"):
                newfname = f[:-4] + ".custom"
                os.rename(os.path.join(dirname, f), os.path.join(dirname, newfname))
    bodo.barrier()
    try:
        check_func(test_impl, (fname2,), py_output=py_output)
    finally:
        if bodo.get_rank() == 0:
            for f in os.listdir(fname2):
                if f.endswith(".custom"):
                    newfname = f[:-7] + ".csv"
                    os.rename(os.path.join(dirname, f), os.path.join(dirname, newfname))
        bodo.barrier()


def test_csv_header_write_read(datapath, memory_leak_check):
    """Test writing and reading csv outputs containing headers"""

    df = pd.read_csv(datapath("example.csv"))
    pd_fname = "pd_csv_header_test.csv"

    def write_impl(df, fname):
        df.to_csv(fname)

    def read_impl(fname):
        return pd.read_csv(fname)

    if bodo.get_rank() == 0:
        write_impl(df, pd_fname)

    bodo.barrier()
    pd_res = read_impl(pd_fname)

    bodo_seq_write = bodo.jit(write_impl)
    bodo_1D_write = bodo.jit(all_args_distributed_block=True)(write_impl)
    bodo_1D_var_write = bodo.jit(all_args_distributed_varlength=True)(write_impl)
    arg_seq = (bodo_seq_write, df, "bodo_csv_header_test_seq.csv")
    arg_1D = (bodo_1D_write, _get_dist_arg(df, False), "bodo_csv_header_test_1D.csv")
    arg_1D_var = (
        bodo_1D_var_write,
        _get_dist_arg(df, False, True),
        "bodo_csv_header_test_1D_var.csv",
    )
    args = [arg_seq, arg_1D, arg_1D_var]
    for (func, df_arg, fname_arg) in args:
        with ensure_clean(fname_arg):
            func(df_arg, fname_arg)
            check_func(read_impl, (fname_arg,), py_output=pd_res)

    bodo.barrier()
    if bodo.get_rank() == 0:
        os.remove(pd_fname)


cat_csv_dtypes = {
    "C1": pd.Int64Dtype(),
    "C2": pd.CategoricalDtype(["A", "B", "C"]),
    "C3": str,
}


def test_csv_cat1(datapath, memory_leak_check):
    fname = datapath("csv_data_cat1.csv")

    def test_impl(fname):
        ct_dtype = pd.CategoricalDtype(["A", "B", "C"])
        dtypes = {"C1": np.dtype("int32"), "C2": ct_dtype, "C3": str}
        df = pd.read_csv(fname, names=["C1", "C2", "C3"], dtype=dtypes)
        return df

    def test_impl2(fname):
        df = pd.read_csv(fname, names=["C1", "C2", "C3"], dtype=cat_csv_dtypes)
        return df

    check_func(test_impl, (fname,))
    check_func(test_impl2, (fname,))

    compressed_names = compress_file(fname)
    try:
        for fname in compressed_names:
            check_func(test_impl, (fname,))
    finally:
        remove_files(compressed_names)


def test_csv_date_col_name(datapath, memory_leak_check):
    """Test the use of column names in "parse_dates" of read_csv"""
    fname = datapath("csv_data_date1.csv")

    def test_impl(fname):
        return pd.read_csv(
            fname,
            names=["A", "B", "C", "D"],
            dtype={"A": int, "B": float, "C": str, "D": int},
            parse_dates=["C"],
        )

    check_func(test_impl, (fname,))

    compressed_names = compress_file(fname)
    try:
        for fname in compressed_names:
            check_func(test_impl, (fname,))
    finally:
        remove_files(compressed_names)


def test_csv_read_only_datetime1(datapath, memory_leak_check):
    """Test the use of reading dataframe containing
    single datetime like column
    """
    fname = datapath("csv_data_only_date1.csv")

    def test_impl(fname):
        return pd.read_csv(
            fname,
            names=["A"],
            dtype={"A": str},
            parse_dates=["A"],
        )

    check_func(test_impl, (fname,))

    compressed_names = compress_file(fname)
    try:
        for fname in compressed_names:
            check_func(test_impl, (fname,))
    finally:
        remove_files(compressed_names)


def test_csv_read_only_datetime2(datapath, memory_leak_check):
    """Test the use of reading dataframe containing
    only datetime-like columns
    """
    fname = datapath("csv_data_only_date2.csv")

    def test_impl(fname):
        return pd.read_csv(
            fname,
            names=["A", "B"],
            dtype={"A": str, "B": str},
            parse_dates=[0, 1],
        )

    check_func(test_impl, (fname,))

    compressed_names = compress_file(fname)
    try:
        for fname in compressed_names:
            check_func(test_impl, (fname,))
    finally:
        remove_files(compressed_names)


def test_csv_dir_int_nulls_single(datapath, memory_leak_check):
    """
    Test read_csv reading dataframe containing int column with nulls
    from a directory(Spark output) containing single csv file.
    """
    fname = datapath("int_nulls_single.csv")

    def test_impl(fname):
        return pd.read_csv(fname, names=["A"], dtype={"A": "Int32"}, header=None)

    py_output = pd.read_csv(
        datapath("int_nulls.csv"), names=["A"], dtype={"A": "Int32"}, header=None
    )

    check_func(test_impl, (fname,), py_output=py_output)

    compress_dir(fname)
    try:
        check_func(test_impl, (fname,), py_output=py_output)
    finally:
        uncompress_dir(fname)


def test_csv_dir_int_nulls_header_single(datapath, memory_leak_check):
    """
    Test read_csv reading dataframe containing int column with nulls
    from a directory(Spark output) containing single csv file with header.
    """
    fname = datapath("int_nulls_header_single.csv")

    def test_impl(fname):
        return pd.read_csv(fname)

    # index_col = 0 because int_nulls.csv has index written
    # names=["A"] because int_nulls.csv does not have header
    py_output = pd.read_csv(datapath("int_nulls.csv"), index_col=0, names=["A"])

    check_func(test_impl, (fname,), py_output=py_output)

    compress_dir(fname)
    try:
        check_func(test_impl, (fname,), py_output=py_output)
    finally:
        uncompress_dir(fname)


def test_csv_dir_int_nulls_multi(datapath, memory_leak_check):
    """
    Test read_csv reading dataframe containing int column with nulls
    from a directory(Spark output) containing multiple csv file.
    """
    fname = datapath("int_nulls_multi.csv")

    def test_impl(fname):
        return pd.read_csv(
            fname,
            names=["A"],
            dtype={"A": "Int32"},
        )

    py_output = pd.read_csv(
        datapath("int_nulls.csv"),
        names=["A"],
        dtype={"A": "Int32"},
    )

    check_func(test_impl, (fname,), py_output=py_output)

    compress_dir(fname)
    try:
        check_func(test_impl, (fname,), py_output=py_output)
    finally:
        uncompress_dir(fname)


def test_csv_dir_int_nulls_header_multi(datapath, memory_leak_check):
    """
    Test read_csv reading dataframe containing int column with nulls
    from a directory(Spark output) containing multiple csv files with header,
    wtih infer dtypes.
    """
    fname = datapath("int_nulls_header_multi.csv")

    def test_impl(fname):
        return pd.read_csv(fname)

    # index_col = 0 because int_nulls.csv has index written
    # names=["A"] because int_nulls.csv does not have header
    py_output = pd.read_csv(datapath("int_nulls.csv"), index_col=0, names=["A"])

    check_func(test_impl, (fname,), py_output=py_output)

    compress_dir(fname)
    try:
        check_func(test_impl, (fname,), py_output=py_output)
    finally:
        uncompress_dir(fname)


def test_csv_dir_str_arr_single(datapath, memory_leak_check):
    """
    Test read_csv reading dataframe containing str column with nulls
    from a directory(Spark output) containing single csv file.
    """
    fname = datapath("str_arr_single.csv")

    def test_impl(fname):
        return pd.read_csv(
            fname,
            names=["A", "B"],
            dtype={"A": str, "B": str},
        ).fillna("")

    py_output = pd.read_csv(
        datapath("str_arr.csv"),
        names=["A", "B"],
        dtype={"A": str, "B": str},
    ).fillna("")

    check_func(test_impl, (fname,), py_output=py_output)

    compress_dir(fname)
    try:
        check_func(test_impl, (fname,), py_output=py_output)
    finally:
        uncompress_dir(fname)


def test_csv_dir_str_arr_multi(datapath, memory_leak_check):
    """
    Test read_csv reading dataframe containing str column with nulls
    from a directory(Spark output) containing multiple csv file.
    """
    fname = datapath("str_arr_parts.csv")

    def test_impl(fname):
        return pd.read_csv(
            fname,
            names=["A", "B"],
            dtype={"A": str, "B": str},
        ).fillna("")

    py_output = pd.read_csv(
        datapath("str_arr.csv"),
        names=["A", "B"],
        dtype={"A": str, "B": str},
    ).fillna("")

    check_func(test_impl, (fname,), py_output=py_output)

    compress_dir(fname)
    try:
        check_func(test_impl, (fname,), py_output=py_output)
    finally:
        uncompress_dir(fname)


@pytest.mark.smoke
def test_excel1(datapath, memory_leak_check):
    """Test pd.read_excel()"""

    def test_impl1(fname):
        return pd.read_excel(fname, parse_dates=[2])

    def test_impl2(fname):
        dtype = {
            "A": int,
            "B": float,
            "C": np.dtype("datetime64[ns]"),
            "D": str,
            "E": np.bool_,
        }
        return pd.read_excel(
            fname,
            sheet_name="Sheet1",
            parse_dates=["C"],
            dtype=dtype,
            names=list(dtype.keys()),
        )

    def test_impl3(fname):
        return pd.read_excel(fname, parse_dates=[2], comment="#")

    def test_impl4(fname, sheet):
        return pd.read_excel(fname, sheet, parse_dates=[2])

    def test_impl5(fname):
        dtype = {
            "A": int,
            "B": float,
            "C": np.dtype("datetime64[ns]"),
            "D": str,
            "E": np.bool_,
        }
        return pd.read_excel(
            fname,
            sheet_name="Sheet1",
            parse_dates=["C"],
            dtype=dtype,
        )

    # passing file name as argument to exercise value-based dispatch
    fname = datapath("data.xlsx")
    check_func(test_impl1, (fname,), is_out_distributed=False)
    check_func(test_impl2, (fname,), is_out_distributed=False)
    fname = datapath("data_comment.xlsx")
    assert pandas_version in (
        (1, 3),
        (1, 4),
    ), "`name` na-filtering issue for 1.4, check if it's fixed in later versions"
    if pandas_version == (1, 3):
        check_func(test_impl3, (fname,), is_out_distributed=False)
    fname = datapath("data.xlsx")
    check_func(test_impl4, (fname, "Sheet1"), is_out_distributed=False)
    with pytest.raises(BodoError, match="both 'dtype' and 'names' should be provided"):
        bodo.jit(test_impl5)(fname)


@pytest.mark.parametrize(
    "fname",
    [
        "all_null_col_eg1.pq",
        "all_null_col_eg2.pq",
    ],
)
def test_read_parquet_all_null_col(fname, memory_leak_check, datapath):
    """test that columns with all nulls can be read successfully"""

    fname_ = datapath(fname)

    def test_impl(_fname_):
        df = pd.read_parquet(_fname_)
        return df

    py_output = pd.read_parquet(fname_)

    check_func(test_impl, (fname_,), py_output=py_output)


@pytest.mark.parametrize(
    "col_subset",
    [
        ["A", "A2", "C"],
        ["C", "B", "A2"],
        ["B"],
    ],
)
def test_read_parquet_all_null_col_subsets(col_subset, memory_leak_check, datapath):
    """test that columns with all nulls can be read successfully"""
    fname = datapath("all_null_col_eg2.pq")

    def test_impl(fname):
        df = pd.read_parquet(fname, columns=col_subset)
        return df

    py_output = pd.read_parquet(fname, columns=col_subset)

    check_func(test_impl, (fname,), py_output=py_output)


def test_read_parquet_input_file_name_col(datapath, memory_leak_check):
    """test basic input_col_name functionality for read_parquet"""
    fname = datapath("decimal1.pq")

    import pyspark.sql.functions as F
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()

    def test_impl(fname):
        df = pd.read_parquet(fname, _bodo_input_file_name_col="fname")
        # pyspark adds this prefix for local files, so we're adding it
        # here for comparison
        # XXX Should we do this by default?
        df.fname = df.fname.apply(lambda x: f"file://{x}")
        return df

    py_output = (
        spark.read.format("parquet")
        .load(fname)
        .withColumn("fname", F.input_file_name())
    ).toPandas()

    check_func(
        test_impl,
        (fname,),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_output,
    )


def test_read_parquet_input_file_name_col_with_partitions(datapath, memory_leak_check):
    """
    test input_col_name functionality for read_parquet
    In particular, this tests that it works as expected when
    the input dataset has partitions
    """

    fname = datapath("test_partitioned.pq")

    import pyspark.sql.functions as F
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()

    def test_impl(fname):
        df = pd.read_parquet(fname, _bodo_input_file_name_col="fname")
        # pyspark adds this prefix for local files, so we're adding it
        # here for comparison
        # XXX Should we do this by default?
        df.fname = df.fname.apply(lambda x: f"file://{x}")
        return df

    py_output = (
        spark.read.format("parquet")
        .load(fname)
        .withColumn("fname", F.input_file_name())
    ).toPandas()
    # Spark reads it as int32 by default, so to make it comparable
    # we convert it to categorical
    py_output.A = py_output.A.astype("category")

    check_func(
        test_impl,
        (fname,),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_output,
    )


def test_read_parquet_input_file_name_col_with_index(datapath, memory_leak_check):
    """
    test input_col_name functionality for read_parquet
    In particular we check that it works fine with files containing
    index columns (e.g. written by pandas)
    """
    fname = datapath("example.parquet")

    def test_impl(fname):
        df = pd.read_parquet(fname, _bodo_input_file_name_col="fname")
        return df

    # Unlike the other tests, we're only checking for a specific code path,
    # so we don't need to check against PySpark directly.
    py_output = pd.read_parquet(fname)
    py_output["fname"] = fname

    check_func(
        test_impl,
        (fname,),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_output,
    )


def test_read_parquet_input_file_name_col_pruned_out(datapath, memory_leak_check):
    """
    test input_col_name functionality for read_parquet
    In particular we check that it works fine when the input_file_name
    column is pruned out.
    This test should also trigger the memory_leak_check if the pruning
    doesn't work as expected
    """
    fname = datapath("example.parquet")

    def test_impl(fname):
        df = pd.read_parquet(fname, _bodo_input_file_name_col="fname")
        df = df[["one", "two", "three"]]
        return df

    # Check that columns were pruned using verbose logging
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(test_impl)(fname)
        check_logger_msg(stream, "Columns loaded ['one', 'two', 'three']")

    # Check that output is correct
    # Unlike the other tests, we're only checking for a specific optimization,
    # so we don't need to check against PySpark directly.
    py_output = pd.read_parquet(fname)
    py_output = py_output[["one", "two", "three"]]

    check_func(
        test_impl,
        (fname,),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_output,
    )


def test_read_parquet_only_input_file_name_col(datapath, memory_leak_check):
    """
    test input_col_name functionality for read_parquet
    In particular test that it works as expected when only the filename
    column is used (the rest are pruned).
    """
    fname = datapath("decimal1.pq")

    import pyspark.sql.functions as F
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()

    def test_impl(fname):
        df = pd.read_parquet(fname, _bodo_input_file_name_col="fname")
        # pyspark adds this prefix for local files, so we're adding it
        # here for comparison
        df.fname = df.fname.apply(lambda x: f"file://{x}")
        return df[["fname"]]

    # Check that columns were pruned using verbose logging
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(test_impl)(fname)
        check_logger_msg(stream, "Columns loaded ['fname']")

    # Check that output is correct
    py_output = (
        spark.read.format("parquet")
        .load(fname)
        .withColumn("fname", F.input_file_name())
    ).toPandas()
    py_output = py_output[["fname"]]

    check_func(
        test_impl,
        (fname,),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_output,
    )


def test_read_parquet_bodo_read_as_dict(memory_leak_check):
    """
    Test _bodo_read_as_dict functionality for read_parquet.
    """
    fname = "encoding_bodo_read_as_dict_test.pq"

    if bodo.get_rank() == 0:
        # Write to parquet on rank 0
        df = pd.DataFrame(
            {
                # A should be dictionary encoded
                "A": ["awerwe", "awerwev24v2", "3r2r32rfc3", "ERr32r23rrrrrr"] * 250,
                # B should not be dictionary encoded
                "B": [str(i) for i in range(1000)],
                # C should be dictionary encoded
                "C": ["r32r23r32r32r23"] * 1000,
                # D is non-string column, so shouldn't be encoded even if specified
                "D": np.arange(1000),
            }
        )
        df.to_parquet(fname, index=False)
    bodo.barrier()

    @bodo.jit
    def test_impl1(fname):
        return pd.read_parquet(fname, _bodo_read_as_dict=["A"])

    @bodo.jit
    def test_impl2(fname):
        return pd.read_parquet(fname, _bodo_read_as_dict=["B"])

    @bodo.jit
    def test_impl3(fname):
        return pd.read_parquet(fname, _bodo_read_as_dict=["C"])

    @bodo.jit
    def test_impl4(fname):
        return pd.read_parquet(fname, _bodo_read_as_dict=["A", "C"])

    @bodo.jit
    def test_impl5(fname):
        return pd.read_parquet(fname, _bodo_read_as_dict=["B", "C"])

    @bodo.jit
    def test_impl6(fname):
        return pd.read_parquet(fname, _bodo_read_as_dict=["A", "B"])

    @bodo.jit
    def test_impl7(fname):
        return pd.read_parquet(fname, _bodo_read_as_dict=["A", "B", "C"])

    # 'D' shouldn't be read as dictionary encoded since it's not a string column

    @bodo.jit
    def test_impl8(fname):
        return pd.read_parquet(fname, _bodo_read_as_dict=["A", "B", "C", "D"])

    @bodo.jit
    def test_impl9(fname):
        return pd.read_parquet(fname, _bodo_read_as_dict=["D"])

    @bodo.jit
    def test_impl10(fname):
        return pd.read_parquet(fname, _bodo_read_as_dict=["A", "D"])

    try:
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            test_impl1(fname)
            check_logger_msg(stream, "Columns ['A', 'C'] using dictionary encoding")

        with set_logging_stream(logger, 1):
            test_impl2(fname)
            check_logger_msg(
                stream, "Columns ['A', 'B', 'C'] using dictionary encoding"
            )

        with set_logging_stream(logger, 1):
            test_impl3(fname)
            check_logger_msg(stream, "Columns ['A', 'C'] using dictionary encoding")

        with set_logging_stream(logger, 1):
            test_impl4(fname)
            check_logger_msg(stream, "Columns ['A', 'C'] using dictionary encoding")

        with set_logging_stream(logger, 1):
            test_impl5(fname)
            check_logger_msg(
                stream, "Columns ['A', 'B', 'C'] using dictionary encoding"
            )

        with set_logging_stream(logger, 1):
            test_impl6(fname)
            check_logger_msg(
                stream, "Columns ['A', 'B', 'C'] using dictionary encoding"
            )

        with set_logging_stream(logger, 1):
            test_impl7(fname)
            check_logger_msg(
                stream, "Columns ['A', 'B', 'C'] using dictionary encoding"
            )

        with set_logging_stream(logger, 1):
            test_impl8(fname)
            check_logger_msg(
                stream, "Columns ['A', 'B', 'C'] using dictionary encoding"
            )

        with set_logging_stream(logger, 1):
            test_impl9(fname)
            check_logger_msg(stream, "Columns ['A', 'C'] using dictionary encoding")

        with set_logging_stream(logger, 1):
            test_impl10(fname)
            check_logger_msg(stream, "Columns ['A', 'C'] using dictionary encoding")

        if bodo.get_rank() == 0:  # warning is thrown only on rank 0
            with pytest.warns(
                BodoWarning,
                match="The following columns are not of datatype string and hence cannot be read with dictionary encoding: {'D'}",
            ):
                test_impl8(fname)
        else:
            test_impl8(fname)

        if bodo.get_rank() == 0:  # warning is thrown only on rank 0
            with pytest.warns(
                BodoWarning,
                match="The following columns are not of datatype string and hence cannot be read with dictionary encoding: {'D'}",
            ):
                test_impl9(fname)
        else:
            test_impl9(fname)

        if bodo.get_rank() == 0:  # warning is thrown only on rank 0
            with pytest.warns(
                BodoWarning,
                match="The following columns are not of datatype string and hence cannot be read with dictionary encoding: {'D'}",
            ):
                test_impl10(fname)
        else:
            test_impl10(fname)

    finally:
        if bodo.get_rank() == 0:
            os.remove(fname)


def test_read_parquet_large_string_array(memory_leak_check):
    """
    Test that we can read `pa.large_string` arrays.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Use both large and regular, just to confirm that they both work
    table = pa.table(
        [
            pa.array(["A", "B", "C", "D"] * 25, type=pa.large_string()),
            pa.array(["lorem", "ipsum"] * 50, type=pa.string()),
            pa.array((["A"] * 10) + (["b"] * 90), type=pa.large_string()),
        ],
        names=["A", "B", "C"],
    )

    fname = "large_str_eg.pq"
    with ensure_clean(fname):
        if bodo.get_rank() == 0:
            pq.write_table(table, fname)
        bodo.barrier()

        def impl(path):
            df = pd.read_parquet(path)
            return df

        check_func(impl, (fname,))


def test_csv_dtype_unicode(memory_leak_check):
    """
    Tests read_csv using dtype="unicode"
    """

    def test_impl(fname):
        return pd.read_csv(fname, names=["A", "B", "C", "D"], dtype="unicode")

    fname = os.path.join("bodo", "tests", "data", "csv_data1.csv")
    check_func(test_impl, (fname,))


def _check_filenotfound(fname, func):
    with pytest.raises(BodoError) as excinfo:
        bodo.jit(func)(fname)
    err_track = excinfo.getrepr(style="native")
    assert "Pseudo-exception" not in str(err_track)


def test_file_not_found(memory_leak_check):
    """Test removal Pseduo-exception with FileNotFoundError"""

    def test_csv(fname):
        df = pd.read_csv(fname)
        return df.C

    def test_pq(fname):
        df = pd.read_parquet(fname)
        return len(df)

    def test_json(fname):
        df = pd.read_json(fname)
        return df.C

    _check_filenotfound("nofile.csv", test_csv)
    _check_filenotfound("nofile.pq", test_pq)
    _check_filenotfound("nofile.json", test_json)
    _check_filenotfound("s3://bodo-test/csv_data_date_not_found.csv", test_csv)
    _check_filenotfound(
        "gcs://anaconda-public-data/nyc-taxi/nyc.parquet/art.0.parquet", test_pq
    )


@pytest.mark.slow
def test_csv_relative_path(datapath, memory_leak_check):
    """Test pd.read_csv with relative path"""

    # File
    filename = os.path.join(".", "bodo", "tests", "data", "example.csv")

    def impl1():
        return pd.read_csv(filename)

    check_func(impl1, (), check_dtype=False)

    # Folder
    foldername = os.path.join(".", "bodo", "tests", "data", "example_multi.csv")
    py_output = pd.read_csv(datapath("example.csv"))

    def impl1():
        return pd.read_csv(foldername)

    check_func(impl1, (), check_dtype=False, py_output=py_output)


@pytest.mark.slow
def test_csv_nrows(memory_leak_check):
    """Test pd.read_csv with nrows argument"""
    fname = os.path.join("bodo", "tests", "data", "example.csv")

    def impl1():
        return pd.read_csv(fname, nrows=2)

    check_func(impl1, (), check_dtype=False)

    # Test nrows as variable.
    nrows = 6

    def impl2():
        return pd.read_csv(fname, nrows=nrows)

    check_func(impl2, (), check_dtype=False)

    # Test nrows + skiprows
    def impl3():
        return pd.read_csv(fname, skiprows=2, nrows=4)

    check_func(impl3, (), check_dtype=False)

    # Test with names provided
    fname = os.path.join("bodo", "tests", "data", "csv_data1.csv")

    def impl4():
        df = pd.read_csv(
            fname,
            names=["A", "B", "C", "D"],
            dtype={"A": int, "B": float, "C": float, "D": int},
            nrows=2,
            skiprows=1,
        )
        return df.B.values

    check_func(impl4, (), check_dtype=False)


@pytest.mark.slow
def test_csv_skiprows_var(memory_leak_check):
    """Test pd.read_csv with skiprows argument as variable"""
    fname = os.path.join("bodo", "tests", "data", "example.csv")

    skip = 3

    def impl_skip():
        ans = pd.read_csv(fname, skiprows=skip)
        return ans

    check_func(impl_skip, (), check_dtype=False)

    # Test skiprows using loop index variable
    # set dtype=str since Bodo sets uniform type for the whole column
    # while Pandas sets type per chunk, resulting in miexed type output
    def impl_loop():
        nrows = 3
        colnames = ["A", "B", "C", "D", "E"]
        df_all = []
        for i in range(4):
            df = pd.read_csv(
                fname, skiprows=(i * nrows), nrows=nrows, names=colnames, dtype=str
            )
            df_all.append(df)
        return pd.concat(df_all)

    check_func(impl_loop, (), check_dtype=False, sort_output=True)


@pytest.mark.slow
def test_csv_chunksize_forloop(datapath, memory_leak_check):
    """
    Check the pd.read_csv iterator using chunksize and a for loop.
    """
    fname = datapath("example.csv")

    # This test checks array analysis generated shape nodes for this iterator
    # structure also, since .max() produces a parfor. Array analysis
    # modifies the IR output for the iterator. Here is an example:
    # $20for_iter.1 = iternext(value=$18get_iter.7) ['$18get_iter.7', '$20for_iter.1']
    # $val.156 = pair_first(value=$20for_iter.1) ['$20for_iter.1', '$val.156']
    # $20for_iter.2_shape.149 = getattr(value=$val.156, attr=shape) ['$20for_iter.2_shape.149', '$val.156']
    # $20for_iter.2_size0.150 = static_getitem(value=$20for_iter.2_shape.149, index=0, index_var=None, fn=<built-in function getitem>) ['$20for_iter.2_shape.149', '$20for_iter.2_size0.150']
    # $20for_iter.3 = pair_second(value=$20for_iter.1) ['$20for_iter.1', '$20for_iter.3']
    # branch $20for_iter.3, 22, 234            ['$20for_iter.3']
    #
    # This means that even once iterations are finished, the IR attempts to check
    # the shape value of the DataFrame which is a null value (potential segfault
    # if not handled properly).
    #
    # We handle this situation in our implementation of len(df), which is what gets produced
    # when shape is called. Here we check if the meminfo pointer is null and if so return
    # a garbage value (0) rather than looking at the data. We can do this because we assume
    # Numba 0 initializes structs with no provided value. Here are the relevant source code lines.
    #
    # https://github.com/numba/numba/blob/3131959c98e567d74ab6db402230cfea6ceecafe/numba/core/imputils.py#L335
    # https://github.com/numba/numba/blob/3131959c98e567d74ab6db402230cfea6ceecafe/numba/core/base.py#L987
    # https://github.com/numba/numba/blob/3131959c98e567d74ab6db402230cfea6ceecafe/numba/core/cgutils.py#L56
    # https://github.com/numba/numba/blob/3131959c98e567d74ab6db402230cfea6ceecafe/numba/core/cgutils.py#L130
    # Note the zfill=True in the last link should show that it is 0 initialized.

    def impl1(fname):
        total = 0.0
        for val in pd.read_csv(fname, chunksize=3):
            result = val["four"].max()
            total += result
        return total

    def impl2(fname):
        total = 0.0
        for val in pd.read_csv(fname, chunksize=100):
            # Single chunk
            result = val["four"].max()
            total += result
        return total

    def impl3(fname):
        total = 0.0
        for val in pd.read_csv(fname, chunksize=1):
            # Empty data on all ranks but 0
            result = val["four"].max()
            total += result
        return total

    check_func(impl1, (fname,))
    check_func(impl2, (fname,))
    check_func(impl3, (fname,))


@pytest.mark.slow
def test_csv_chunksize_index_col(datapath, memory_leak_check):
    """
    Check the pd.read_csv iterator using chunksize and an index_col.
    """
    fname = datapath("example.csv")

    def impl(fname):
        total = 0.0
        for val in pd.read_csv(fname, chunksize=3, index_col="four"):
            total += val.index[-1]
        return total

    check_func(impl, (fname,))


@pytest.mark.slow
def test_csv_chunksize_enumerate(datapath, memory_leak_check):
    """
    Check the pd.read_csv iterator using chunksize and enumerate.
    """
    fname = datapath("example.csv")

    def impl1(fname):
        total = 0.0
        count_total = 0
        for i, val in enumerate(pd.read_csv(fname, chunksize=3)):
            result = val["four"].max()
            total += result
            count_total += i
        return (count_total, total)

    def impl2(fname):
        total = 0.0
        count_total = 0
        for i, val in enumerate(pd.read_csv(fname, chunksize=100)):
            # Single chunk
            result = val["four"].max()
            total += result
            count_total += i
        return (count_total, total)

    def impl3(fname):
        total = 0.0
        count_total = 0
        for i, val in enumerate(pd.read_csv(fname, chunksize=1)):
            # Empty data on all ranks but 0
            result = val["four"].max()
            total += result
            count_total += i
        return (count_total, total)

    check_func(impl1, (fname,))
    check_func(impl2, (fname,))
    check_func(impl3, (fname,))


@pytest.mark.slow
def test_csv_chunksize_forloop_append(datapath, memory_leak_check):
    """
    Check returning a dataframe with the pd.read_csv iterator using
    chunksize and a for loop.
    """

    fname = datapath("example.csv")

    def impl1(fname):
        df_list = []
        for val in pd.read_csv(fname, chunksize=3):
            df_list.append(val)
        return pd.concat(df_list)

    def impl2(fname):
        df_list = []
        for val in pd.read_csv(fname, chunksize=100):
            # Single chunk
            df_list.append(val)
        return pd.concat(df_list)

    def impl3(fname):
        df_list = []
        for val in pd.read_csv(fname, chunksize=1):
            # Empty data on all ranks but 0
            df_list.append(val)
        return pd.concat(df_list)

    # Check only sequential implementation because previous tests check
    # situations where the csv read is parallel. Returning a sequential
    # output should enable checking read_csv with parallel=False
    check_func(impl1, (fname,), is_out_distributed=False, dist_test=False)
    check_func(impl2, (fname,), is_out_distributed=False, dist_test=False)
    check_func(impl3, (fname,), is_out_distributed=False, dist_test=False)


@pytest.mark.slow
def test_csv_chunksize_enumerate_append(datapath, memory_leak_check):
    """
    Check returning a dataframe with the pd.read_csv iterator using
    chunksize and enumerate.
    """

    fname = datapath("example.csv")

    def impl1(fname):
        df_list = []
        count_total = 0
        for i, val in enumerate(pd.read_csv(fname, chunksize=3)):
            df_list.append(val)
            count_total += i
        return (count_total, pd.concat(df_list))

    def impl2(fname):
        df_list = []
        count_total = 0
        for i, val in enumerate(pd.read_csv(fname, chunksize=100)):
            # Single chunk
            df_list.append(val)
            count_total += i
        return (count_total, pd.concat(df_list))

    def impl3(fname):
        df_list = []
        count_total = 0
        for i, val in enumerate(pd.read_csv(fname, chunksize=1)):
            # Empty data on all ranks but 0
            df_list.append(val)
            count_total += i
        return (count_total, pd.concat(df_list))

    # Check only sequential implementation because previous tests check
    # situations where the csv read is parallel. Returning a sequential
    # output should enable checking read_csv with parallel=False
    check_func(impl1, (fname,), is_out_distributed=False, dist_test=False)
    check_func(impl2, (fname,), is_out_distributed=False, dist_test=False)
    check_func(impl3, (fname,), is_out_distributed=False, dist_test=False)


@pytest.mark.slow
def test_csv_chunksize_forloop_nested(datapath, memory_leak_check):
    """
    Check multiple calls to the pd.read_csv iterator using
    chunksize and a for loop.
    """

    fname = datapath("example.csv")

    def impl1(fname, n):
        total = 0.0
        for _ in range(n):
            for val in pd.read_csv(fname, chunksize=3):
                result = val["four"].max()
                total += result
        return total

    def impl2(fname, n):
        total = 0.0
        for _ in range(n):
            for val in pd.read_csv(fname, chunksize=100):
                # Single chunk
                result = val["four"].max()
                total += result
        return total

    def impl3(fname, n):
        total = 0.0
        for _ in range(n):
            for val in pd.read_csv(fname, chunksize=1):
                # Empty data on all ranks but 0
                result = val["four"].max()
                total += result
        return total

    check_func(impl1, (fname, 5))
    check_func(impl2, (fname, 5))
    check_func(impl3, (fname, 5))


@pytest.mark.slow
def test_csv_chunksize_enumerate_nested(datapath, memory_leak_check):
    """
    Check multiple calls to the pd.read_csv iterator using
    chunksize and enumerate.
    """

    fname = datapath("example.csv")

    def impl1(fname, n):
        total = 0.0
        count_total = 0
        for _ in range(n):
            for i, val in enumerate(pd.read_csv(fname, chunksize=3)):
                result = val["four"].max()
                total += result
                count_total += i
        return (count_total, total)

    def impl2(fname, n):
        total = 0.0
        count_total = 0
        for _ in range(n):
            for i, val in enumerate(pd.read_csv(fname, chunksize=100)):
                # Single chunk
                result = val["four"].max()
                total += result
                count_total += i
        return (count_total, total)

    def impl3(fname, n):
        total = 0.0
        count_total = 0
        for _ in range(n):
            for i, val in enumerate(pd.read_csv(fname, chunksize=1)):
                # Empty data on all ranks but 0
                result = val["four"].max()
                total += result
                count_total += i
        return (count_total, total)

    check_func(impl1, (fname, 5))
    check_func(impl2, (fname, 5))
    check_func(impl3, (fname, 5))


@pytest.mark.slow
def test_csv_chunksize_skiprows(datapath, memory_leak_check):
    """
    Check multiple calls to the pd.read_csv iterator using
    chunksize with skiprows
    """

    fname = datapath("example.csv")

    # skiprows and multiple chunks
    def impl1(fname, n):
        total = 0.0
        count_total = 0
        for _ in range(n):
            for i, val in enumerate(pd.read_csv(fname, chunksize=3, skiprows=5)):
                result = val.iloc[:, 0].max()
                total += result
                count_total += i
        return (count_total, total)

    # skiprows and one chunk
    def impl2(fname, n):
        total = 0.0
        count_total = 0
        for _ in range(n):
            for i, val in enumerate(pd.read_csv(fname, chunksize=100, skiprows=5)):
                # Single chunk
                result = val.iloc[:, 0].max()
                total += result
                count_total += i
        return (count_total, total)

    # skiprows all rows but one with one chunk on one rank only
    def impl3(fname, n):
        total = 0.0
        count_total = 0
        for _ in range(n):
            for i, val in enumerate(pd.read_csv(fname, chunksize=1, skiprows=13)):
                # Empty data on all ranks but 0
                result = val.iloc[:, 0].max()
                total += result
                count_total += i
        return (count_total, total)

    check_func(impl1, (fname, 5))
    check_func(impl2, (fname, 5))
    check_func(impl3, (fname, 5))


@pytest.mark.slow
def test_csv_chunksize_nrows(datapath, memory_leak_check):
    """
    Check multiple calls to the pd.read_csv iterator using
    chunksize with nrows
    """

    fname = datapath("example.csv")

    # nrows and multiple chunks
    def impl1(fname, n):
        total = 0.0
        count_total = 0
        for _ in range(n):
            for i, val in enumerate(pd.read_csv(fname, chunksize=3, nrows=7)):
                result = val.iloc[:, 0].max()
                total += result
                count_total += i
        return (count_total, total)

    # nrows and one chunk
    def impl2(fname, n):
        total = 0.0
        count_total = 0
        for _ in range(n):
            for i, val in enumerate(pd.read_csv(fname, chunksize=100, nrows=5)):
                # Single chunk
                result = val.iloc[:, 0].max()
                total += result
                count_total += i
        return (count_total, total)

    # nrows and chunk on one rank only
    def impl3(fname, n):
        total = 0.0
        count_total = 0
        for _ in range(n):
            for i, val in enumerate(pd.read_csv(fname, chunksize=1, nrows=3)):
                # Empty data on all ranks but 0
                result = val.iloc[:, 0].max()
                total += result
                count_total += i
        return (count_total, total)

    check_func(impl1, (fname, 5))
    check_func(impl2, (fname, 5))
    check_func(impl3, (fname, 5))


@pytest.mark.slow
def test_csv_chunksize_nrows_skiprows(datapath, memory_leak_check):
    """
    Check the pd.read_csv iterator using chunksize with nrows/skiprows
    """
    fname = datapath("example.csv")

    # nrows, skiprows and multiple chunks
    def impl1(fname):
        total = 0.0
        count_total = 0
        for i, val in enumerate(pd.read_csv(fname, chunksize=3, nrows=10, skiprows=5)):
            result = val.iloc[:, 0].max()
            total += result
            count_total += i
        return (count_total, total)

    # nrows, skiprows and one chunk
    def impl2(fname):
        total = 0.0
        count_total = 0
        for i, val in enumerate(pd.read_csv(fname, chunksize=100, skiprows=4, nrows=5)):
            # Single chunk
            result = val.iloc[:, 0].max()
            total += result
            count_total += i
        return (count_total, total)

    # nrows, skiprows and all data on one rank only
    def impl3(fname):
        total = 0.0
        count_total = 0
        for i, val in enumerate(pd.read_csv(fname, chunksize=1, nrows=10, skiprows=5)):
            # Empty data on all ranks but 0
            result = val.iloc[:, 0].max()
            total += result
            count_total += i
        return (count_total, total)

    check_func(impl1, (fname,))
    check_func(impl2, (fname,))
    check_func(impl3, (fname,))


@pytest.mark.slow
def test_csv_skiprows_list_low_memory(datapath, memory_leak_check):
    """
    Test pd.read_csv skiprows as list for low_memory path
    """
    fname = datapath("example.csv")

    # skiprows list
    # (include 1st row and case where rank's start_pos is inside a skipped row)
    def impl1(fname):
        return pd.read_csv(fname, low_memory=True, skiprows=[1, 5, 7, 12])

    check_func(impl1, (fname,))

    # skiprows > available rows
    def impl2(fname):
        return pd.read_csv(fname, low_memory=True, skiprows=[17, 20])

    check_func(impl2, (fname,))

    # skiprows list unordered and duplicated
    def impl3(fname):
        return pd.read_csv(fname, low_memory=True, skiprows=[13, 4, 12, 4, 13])

    check_func(impl3, (fname,))

    # nrows + skiprows list (list has values in and out of nrows range)
    def impl4(fname):
        return pd.read_csv(fname, low_memory=True, nrows=10, skiprows=[4, 7, 12])

    check_func(impl4, (fname,))

    # skiprows list with 0
    def impl5(fname):
        return pd.read_csv(fname, low_memory=True, skiprows=[0, 2, 3, 4, 5, 6])

    check_func(impl5, (fname,))

    # contiguous list
    def impl6(fname):
        return pd.read_csv(fname, low_memory=True, skiprows=[6, 7, 8])

    check_func(impl6, (fname,))

    # header=None
    # Note: Bodo's index is object while pandas is int64
    # check_dtype=False does not work index values are '0' vs. 0
    py_output = pd.read_csv(
        fname, names=["0", "1", "2", "3", "4"], skiprows=[0, 2, 3, 7, 5, 9]
    )

    def impl7(fname):
        ans = pd.read_csv(fname, header=None, skiprows=[0, 2, 3, 7, 5, 9])
        return ans

    check_func(impl7, (fname,), py_output=py_output)


@pytest.mark.slow
def test_csv_skiprows_list(datapath, memory_leak_check):
    """
    Test pd.read_csv skiprows as list default case (i.e. low_memory=False)
    """
    fname = datapath("example.csv")

    # skiprows list
    # (include 1st row and case where rank's start_pos is inside a skipped row)
    def impl1(fname):
        return pd.read_csv(fname, low_memory=True, skiprows=[1, 5, 7, 12])

    check_func(impl1, (fname,))

    # skiprows > available rows
    def impl2(fname):
        return pd.read_csv(fname, skiprows=[17, 20])

    check_func(impl2, (fname,))

    # nrows + skiprows list (list has values in and out of nrows range)
    def impl3(fname):
        return pd.read_csv(fname, nrows=10, skiprows=[4, 7, 12])

    check_func(impl3, (fname,))

    # skiprows list with 0
    def impl4(fname):
        return pd.read_csv(fname, skiprows=[0, 2, 3, 4, 5, 6])

    check_func(impl4, (fname,))

    # header=None
    # Note: Bodo's index is object while pandas is int64
    # check_dtype=False does not work index values are '0' vs. 0
    py_output = pd.read_csv(
        fname, names=["0", "1", "2", "3", "4"], skiprows=[0, 2, 3, 7, 5, 9]
    )

    def impl5(fname):
        ans = pd.read_csv(fname, header=None, skiprows=[0, 2, 3, 7, 5, 9])
        return ans

    check_func(impl5, (fname,), py_output=py_output)


@pytest.mark.slow
def test_csv_skiprows_list_chunksize(datapath, memory_leak_check):
    """
    Test pd.read_csv skiprows as list with chunksize
    """
    fname = datapath("example.csv")
    # list + chunksize (each rank gets a chunk)
    # (include case where rank's chunk start_pos and end_pos is inside a skipped row)
    def impl1(fname):
        total = 0.0
        for val in pd.read_csv(fname, chunksize=5, skiprows=[4, 5, 7, 13]):
            result = val.iloc[:, 0].max()
            total += result
        return total

    check_func(impl1, (fname,))

    # list + chunksize (one chunk)
    def impl2(fname):
        total = 0.0
        for val in pd.read_csv(fname, chunksize=15, skiprows=[9, 4, 2, 10]):
            result = val.iloc[:, 0].max()
            total += result
        return total

    check_func(impl2, (fname,))

    # list + chunksize (only rank 0 gets the chunk)
    def impl3(fname):
        total = 0.0
        for val in pd.read_csv(fname, chunksize=1, skiprows=[4, 7]):
            result = val.iloc[:, 0].max()
            total += result
        return total

    check_func(impl3, (fname,))

    # list + nrows + chunksize (each rank gets a chunk)
    def impl4(fname):
        total = 0.0
        for val in pd.read_csv(fname, chunksize=3, nrows=12, skiprows=[2, 5, 8, 14]):
            result = val.iloc[:, 0].max()
            total += result
        return total

    check_func(impl4, (fname,))

    # list + nrows + chunksize (one chunk)
    def impl5(fname):
        total = 0.0
        for val in pd.read_csv(fname, chunksize=15, nrows=12, skiprows=[1, 4, 7, 13]):
            result = val.iloc[:, 0].max()
            total += result
        return total

    check_func(impl5, (fname,))

    # list + nrows + chunksize (only rank 0 gets the chunk)
    def impl6(fname):
        total = 0.0
        for val in pd.read_csv(fname, chunksize=1, nrows=6, skiprows=[14, 4, 9, 13]):
            result = val.iloc[:, 0].max()
            total += result
        return total

    check_func(impl6, (fname,))


@pytest.mark.slow
def test_csv_np_gt_rows(datapath, memory_leak_check):
    """Test when number of rows < number of ranks (np) with
    read_csv(). "small_data.csv" has one row.
    Running with np>1 will not fail
    """
    fname = datapath("small_data.csv")

    def impl1():
        return pd.read_csv(fname)

    check_func(impl1, (), check_dtype=False)

    # Test usecols
    def impl2():
        return pd.read_csv(fname, usecols=["A", "C"])

    check_func(impl2, (), check_dtype=False)


@pytest.mark.slow
def test_csv_escapechar(datapath, memory_leak_check):
    """Test pd.read_csv with escapechar argument"""

    fname = datapath("escapechar_data.csv")

    def impl():
        return pd.read_csv(fname, escapechar="\\")

    check_func(impl, (), check_dtype=False)


@pytest.mark.slow
def test_csv_unsupported_arg_match(memory_leak_check):
    """
    Test read_csv(): Test that passing an unsupported arg that matches
    the default doesn't raise an error
    """
    fname = os.path.join("bodo", "tests", "data", "example.csv")

    def impl():
        # squeeze is provided but not supported. It matches the default
        # so it should still work.
        return pd.read_csv(
            fname,
            ",",
            None,
            "infer",
            ["A", "B", "C", "D", "E"],
            None,
            None,
            False,
        )

    check_func(impl, (), check_dtype=False)


@pytest.mark.slow
def test_csv_unsupported_kwarg_match(memory_leak_check):
    """
    Test read_csv(): Test that passing an unsupported kwarg that matches
    the default doesn't raise an error
    """
    fname = os.path.join("bodo", "tests", "data", "example.csv")

    def impl():
        # comment is provided but not supported. It matches the default
        # so it should still work.
        return pd.read_csv(fname, comment=None)

    check_func(impl, (), check_dtype=False)


@pytest.mark.slow
def test_read_csv_sample_nrows(datapath, memory_leak_check):
    """Test read_csv with sample_nrows argument

    Args:
        datapath (fixture function): get path to test file
        memory_leak_check (fixture function): check memory leak in the test.

    """
    fname = datapath("large_data.csv")

    def impl1():
        return pd.read_csv(fname, sample_nrows=120)

    py_df = pd.read_csv(fname)
    check_func(impl1, (), py_output=py_df)


@pytest.mark.slow
def test_read_json_sample_nrows(datapath, memory_leak_check):
    """Test read_json with sample_nrows argument

    Args:
        datapath (fixture function): get path to test file
        memory_leak_check (fixture function): check memory leak in the test.

    """
    fname = datapath("large_data.json")

    def impl1():
        return pd.read_json(fname, sample_nrows=120)

    py_df = pd.read_json(fname, lines=True, orient="records")
    check_func(impl1, (), py_output=py_df)


@pytest.mark.slow
class TestIO(unittest.TestCase):
    def test_h5_write_parallel(self):
        fname = "lr_w.hdf5"

        def test_impl(N, D):
            points = np.ones((N, D))
            responses = np.arange(N) + 1.0
            f = h5py.File(fname, "w")
            dset1 = f.create_dataset("points", (N, D), dtype="f8")
            dset1[:] = points
            dset2 = f.create_dataset("responses", (N,), dtype="f8")
            dset2[:] = responses
            f.close()

        N = 101
        D = 10
        bodo_func = bodo.jit(test_impl)
        with ensure_clean(fname):
            bodo_func(N, D)
            f = h5py.File("lr_w.hdf5", "r")
            X = f["points"][:]
            Y = f["responses"][:]
            f.close()
            np.testing.assert_almost_equal(X, np.ones((N, D)))
            np.testing.assert_almost_equal(Y, np.arange(N) + 1.0)

    def test_h5_write_group(self):
        def test_impl(n, fname):
            arr = np.arange(n)
            n = len(arr)
            f = h5py.File(fname, "w")
            g1 = f.create_group("G")
            dset1 = g1.create_dataset("data", (n,), dtype="i8")
            dset1[:] = arr
            f.close()

        n = 101
        arr = np.arange(n)
        fname = "test_group.hdf5"
        bodo_func = bodo.jit(test_impl)
        with ensure_clean(fname):
            bodo_func(n, fname)
            f = h5py.File(fname, "r")
            X = f["G"]["data"][:]
            f.close()
            np.testing.assert_almost_equal(X, arr)

    def test_pq_read(self):
        fname = os.path.join("bodo", "tests", "data", "kde.parquet")

        def test_impl():
            df = pd.read_parquet(fname)
            X = df["points"]
            return X.sum()

        bodo_func = bodo.jit(test_impl)
        np.testing.assert_almost_equal(bodo_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_pq_read_global_str1(self):
        def test_impl():
            df = pd.read_parquet(kde_file)
            X = df["points"]
            return X.sum()

        bodo_func = bodo.jit(test_impl)
        np.testing.assert_almost_equal(bodo_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_pq_read_freevar_str1(self):
        kde_file2 = os.path.join("bodo", "tests", "data", "kde.parquet")

        def test_impl():
            df = pd.read_parquet(kde_file2)
            X = df["points"]
            return X.sum()

        bodo_func = bodo.jit(test_impl)
        np.testing.assert_almost_equal(bodo_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_pd_read_parquet(self):
        fname = os.path.join("bodo", "tests", "data", "kde.parquet")

        def test_impl():
            df = pd.read_parquet(fname)
            X = df["points"]
            return X.sum()

        bodo_func = bodo.jit(test_impl)
        np.testing.assert_almost_equal(bodo_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_pq_str(self):
        fname = os.path.join("bodo", "tests", "data", "example.parquet")

        def test_impl():
            df = pd.read_parquet(fname)
            A = df.two.values == "foo"
            return A.sum()

        bodo_func = bodo.jit(test_impl)
        np.testing.assert_almost_equal(bodo_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_pq_columns(self):
        fname = os.path.join("bodo", "tests", "data", "example.parquet")

        def test_impl():
            return pd.read_parquet(fname, columns=["three", "five"])

        check_func(test_impl, (), only_seq=True, check_dtype=False)

    def test_pq_str_with_nan_seq(self):
        fname = os.path.join("bodo", "tests", "data", "example.parquet")

        def test_impl():
            df = pd.read_parquet(fname)
            A = df.five.values == "foo"
            return A.sum()

        check_func(test_impl, (), dist_test=False)

    def test_pq_str_with_nan_par(self):
        fname = os.path.join("bodo", "tests", "data", "example.parquet")

        def test_impl():
            df = pd.read_parquet(fname)
            A = df.five.values == "foo"
            return A.sum()

        bodo_func = bodo.jit(test_impl)
        np.testing.assert_almost_equal(bodo_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_pq_str_with_nan_par_multigroup(self):
        fname = os.path.join("bodo", "tests", "data", "example2.parquet")

        def test_impl():
            df = pd.read_parquet(fname)
            A = df.five.values == "foo"
            return A.sum()

        bodo_func = bodo.jit(test_impl)
        np.testing.assert_almost_equal(bodo_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_pq_bool(self):
        fname = os.path.join("bodo", "tests", "data", "example.parquet")

        def test_impl():
            df = pd.read_parquet(fname)
            return df.three.sum()

        bodo_func = bodo.jit(test_impl)
        np.testing.assert_almost_equal(bodo_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_pq_nan(self):
        fname = os.path.join("bodo", "tests", "data", "example.parquet")

        def test_impl():
            df = pd.read_parquet(fname)
            return df.one.sum()

        bodo_func = bodo.jit(test_impl)
        np.testing.assert_almost_equal(bodo_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_pq_float_no_nan(self):
        fname = os.path.join("bodo", "tests", "data", "example.parquet")

        def test_impl():
            df = pd.read_parquet(fname)
            return df.four.sum()

        bodo_func = bodo.jit(test_impl)
        np.testing.assert_almost_equal(bodo_func(), test_impl())
        self.assertEqual(count_array_REPs(), 0)
        self.assertEqual(count_parfor_REPs(), 0)

    def test_csv1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data1.csv")

        def test_impl():
            return pd.read_csv(
                fname,
                names=["A", "B", "C", "D"],
                dtype={"A": int, "B": float, "C": float, "D": int},
            )

        check_func(test_impl, (), only_seq=True)

    def test_csv_keys1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data1.csv")

        def test_impl():
            dtype = {"A": int, "B": float, "C": float, "D": int}
            return pd.read_csv(fname, names=list(dtype.keys()), dtype=dtype)

        check_func(test_impl, (), only_seq=True)

    def test_csv_const_dtype1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data1.csv")

        def test_impl():
            dtype = {"A": "int", "B": "float64", "C": "float", "D": "int64"}
            return pd.read_csv(fname, names=list(dtype.keys()), dtype=dtype)

        check_func(test_impl, (), only_seq=True)

    def test_csv_infer1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data_infer1.csv")

        def test_impl():
            return pd.read_csv(fname)

        check_func(test_impl, (), only_seq=True, check_dtype=False)

    def test_csv_infer_parallel1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data_infer1.csv")

        def test_impl():
            df = pd.read_csv(fname)
            return df.A.sum(), df.B.sum(), df.C.sum(), df.D.sum()

        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(), test_impl())

    def test_csv_infer_str1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data_cat1.csv")

        def test_impl():
            df = pd.read_csv(fname)
            return df

        check_func(test_impl, (), only_seq=True, check_dtype=False)

    def test_csv_skip1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data1.csv")

        def test_impl():
            return pd.read_csv(
                fname,
                names=["A", "B", "C", "D"],
                dtype={"A": int, "B": float, "C": float, "D": int},
                skiprows=2,
            )

        check_func(test_impl, (), only_seq=True)

    def test_csv_infer_skip1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data_infer1.csv")

        def test_impl():
            return pd.read_csv(fname, skiprows=2)

        check_func(test_impl, (), only_seq=True, check_dtype=False)

    def test_csv_infer_skip_parallel1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data_infer1.csv")

        def test_impl():
            df = pd.read_csv(fname, skiprows=2, names=["A", "B", "C", "D"])
            return df.A.sum(), df.B.sum(), df.C.sum(), df.D.sum()

        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(), test_impl())

    def test_csv_rm_dead1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data1.csv")

        def test_impl():
            df = pd.read_csv(
                fname,
                names=["A", "B", "C", "D"],
                dtype={"A": int, "B": float, "C": float, "D": int},
            )
            return df.B.values

        check_func(test_impl, (), only_seq=True)

    def test_csv_date1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data_date1.csv")

        def test_impl():
            return pd.read_csv(
                fname,
                names=["A", "B", "C", "D"],
                dtype={"A": int, "B": float, "C": str, "D": int},
                parse_dates=[2],
            )

        check_func(test_impl, (), only_seq=True)

    def test_csv_str1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data_date1.csv")

        def test_impl():
            return pd.read_csv(
                fname,
                names=["A", "B", "C", "D"],
                dtype={"A": int, "B": float, "C": str, "D": int},
            )

        check_func(test_impl, (), only_seq=True, check_dtype=False)

    def test_csv_index_name1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data_date1.csv")

        def test_impl():
            return pd.read_csv(
                fname,
                names=["A", "B", "C", "D"],
                dtype={"A": int, "B": float, "C": str, "D": int},
                index_col="A",
            )

        check_func(test_impl, (), only_seq=True, check_dtype=False)

    def test_csv_index_ind0(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data_date1.csv")

        def test_impl():
            return pd.read_csv(
                fname,
                names=["A", "B", "C", "D"],
                dtype={"A": int, "B": float, "C": str, "D": int},
                index_col=0,
            )

        check_func(test_impl, (), only_seq=True, check_dtype=False)

    def test_csv_index_ind1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data_date1.csv")

        def test_impl():
            return pd.read_csv(
                fname,
                names=["A", "B", "C", "D"],
                dtype={"A": int, "B": float, "C": str, "D": int},
                index_col=1,
            )

        check_func(test_impl, (), only_seq=True, check_dtype=False)

    def test_csv_parallel1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data1.csv")

        def test_impl():
            df = pd.read_csv(
                fname,
                names=["A", "B", "C", "D"],
                dtype={"A": int, "B": float, "C": float, "D": int},
            )
            return (df.A.sum(), df.B.sum(), df.C.sum(), df.D.sum())

        bodo_func = bodo.jit(test_impl)
        self.assertEqual(bodo_func(), test_impl())

    def test_csv_str_parallel1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data_date1.csv")

        def test_impl():
            df = pd.read_csv(
                fname,
                names=["A", "B", "C", "D"],
                dtype={"A": int, "B": float, "C": str, "D": int},
            )
            return (df.A.sum(), df.B.sum(), (df.C == "1966-11-13").sum(), df.D.sum())

        bodo_func = bodo.jit(distributed=["df"])(test_impl)
        self.assertEqual(bodo_func(), test_impl())

    def test_csv_usecols1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data1.csv")

        def test_impl():
            return pd.read_csv(fname, names=["C"], dtype={"C": float}, usecols=[2])

        check_func(test_impl, (), only_seq=True)

    def test_csv_usecols2(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data1.csv")

        def test_impl():
            return pd.read_csv(fname, names=["B", "C"], usecols=[1, 2])

        check_func(test_impl, (), only_seq=True)

    def test_csv_usecols3(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data2.csv")

        def test_impl():
            return pd.read_csv(fname, sep="|", names=["B", "C"], usecols=[1, 2])

        check_func(test_impl, (), only_seq=True)

    def test_csv_cat2(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data_cat1.csv")

        def test_impl():
            ct_dtype = pd.CategoricalDtype(["A", "B", "C", "D"])
            df = pd.read_csv(
                fname,
                names=["C1", "C2", "C3"],
                dtype={"C1": int, "C2": ct_dtype, "C3": str},
            )
            return df

        check_func(test_impl, (), only_seq=True, check_dtype=False)

    def test_csv_single_dtype1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data_dtype1.csv")

        def test_impl():
            df = pd.read_csv(fname, names=["C1", "C2"], dtype=np.float64)
            return df

        check_func(test_impl, (), only_seq=True)

    def write_csv(self, data_structure):
        # only run on a single processor
        if bodo.get_size() != 1:
            return

        def test_impl(data_structure, fname):
            data_structure.to_csv(fname)

        bodo_func = bodo.jit(test_impl)
        n = 111
        hp_fname = "test_write_csv1_bodo.csv"
        pd_fname = "test_write_csv1_pd.csv"
        with ensure_clean(pd_fname), ensure_clean(hp_fname):
            bodo_func(data_structure, hp_fname)
            test_impl(data_structure, pd_fname)
            pd.testing.assert_frame_equal(pd.read_csv(hp_fname), pd.read_csv(pd_fname))

    def test_write_dataframe_csv1(self):
        n = 111
        df = pd.DataFrame({"A": np.arange(n)}, index=np.arange(n) * 2)
        self.write_csv(df)

    def test_write_series_csv1(self):
        n = 111
        series = pd.Series(data=np.arange(n), dtype=np.float64, name="bodo")
        self.write_csv(series)

    def test_series_invalid_path_or_buf(self):
        n = 111
        series = pd.Series(data=np.arange(n), dtype=np.float64, name="bodo")

        def test_impl(data_structure, fname):
            data_structure.to_csv(fname)

        bodo_func = bodo.jit(test_impl)
        with pytest.raises(
            BodoError,
            match="argument should be None or string",
        ):
            bodo_func(series, 1)

    def write_csv_parallel(self, test_impl):
        bodo_func = bodo.jit(test_impl)
        n = 111
        hp_fname = "test_write_csv1_bodo_par.csv"
        pd_fname = "test_write_csv1_pd_par.csv"
        with ensure_clean(pd_fname), ensure_clean(hp_fname):
            bodo_func(n, hp_fname)
            self.assertEqual(count_array_REPs(), 0)
            self.assertEqual(count_parfor_REPs(), 0)
            if get_rank() == 0:
                test_impl(n, pd_fname)
                pd.testing.assert_frame_equal(
                    pd.read_csv(hp_fname), pd.read_csv(pd_fname)
                )

    def test_write_dataframe_csv_parallel1(self):
        def test_impl(n, fname):
            df = pd.DataFrame({"A": np.arange(n)})
            df.to_csv(fname)

        self.write_csv_parallel(test_impl)

    def test_write_series_csv_parallel1(self):
        def test_impl(n, fname):
            series = pd.Series(data=np.arange(n), dtype=np.float64, name="bodo")
            series.to_csv(fname)

        self.write_csv_parallel(test_impl)

    def test_write_csv_parallel2(self):
        # 1D_Var case
        def test_impl(n, fname):
            df = pd.DataFrame({"A": np.arange(n)})
            df = df[df.A % 2 == 1]
            df.to_csv(fname, index=False)

        bodo_func = bodo.jit(test_impl)
        n = 111
        hp_fname = "test_write_csv1_bodo_par.csv"
        pd_fname = "test_write_csv1_pd_par.csv"
        with ensure_clean(pd_fname), ensure_clean(hp_fname):
            bodo_func(n, hp_fname)
            self.assertEqual(count_array_REPs(), 0)
            self.assertEqual(count_parfor_REPs(), 0)
            if get_rank() == 0:
                test_impl(n, pd_fname)
                pd.testing.assert_frame_equal(
                    pd.read_csv(hp_fname), pd.read_csv(pd_fname)
                )


@pytest.mark.slow
def test_read_parquet_read_sanitize_colnames(memory_leak_check):
    """tests that parquet read works when reading a dataframe with column names
    that must be sanitized when generating the func_text"""

    def read_impl(path):
        return pd.read_parquet(path)

    check_func(read_impl, ("bodo/tests/data/sanitization_test.pq",))


def check_CSV_write(
    write_impl,
    df,
    pandas_filename="pandas_out.csv",
    bodo_filename="bodo_out.csv",
    read_impl=None,
):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    # as of right now, only testing on posix, since it doesn't write to distributed file system
    # TODO: make this work for non posix
    if os.name != "posix":
        return
    if read_impl == None:
        read_impl = pd.read_csv

    n_pes = bodo.get_size()

    try:
        # only do the python CSV write on rank 0,
        # so the different ranks don't clobber each other.
        if bodo.get_rank() == 0:
            write_impl(df, pandas_filename)

        # copied from pq read/write section
        for mode in ["sequential", "1d-distributed", "1d-distributed-varlength"]:
            try:
                try:
                    if mode == "sequential":
                        bodo_write = bodo.jit(write_impl)
                        bodo_write(df, bodo_filename)
                    elif mode == "1d-distributed":
                        bodo_write = bodo.jit(
                            write_impl, all_args_distributed_block=True
                        )
                        bodo_write(_get_dist_arg(df, False), bodo_filename)
                    elif mode == "1d-distributed-varlength":
                        bodo_write = bodo.jit(
                            write_impl, all_args_distributed_varlength=True
                        )
                        bodo_write(_get_dist_arg(df, False, True), bodo_filename)
                    errors = comm.allgather(None)
                except Exception as e:
                    # In the case that one rank raises an exception, make sure that all the
                    # ranks raise an error, so we don't hang in the barrier.
                    comm.allgather(e)
                    raise

                for e in errors:
                    if isinstance(e, Exception):
                        raise e

                # wait until each rank has finished writing
                bodo.barrier()

                # read both files with pandas
                df1 = read_impl(pandas_filename)
                df2 = read_impl(bodo_filename)
                # read dataframes must be same as original except for dtypes
                passed = _test_equal_guard(
                    df1, df2, sort_output=False, check_names=True, check_dtype=False
                )
                n_passed = reduce_sum(passed)
                assert n_passed == n_pes
            finally:
                # cleanup the bodo file
                # TODO: update this if we use this test on non POSIX
                if bodo.get_rank() == 0:
                    try:
                        os.remove(bodo_filename)
                    except FileNotFoundError:
                        pass
    finally:
        # cleanup the pandas file
        if bodo.get_rank() == 0:
            try:
                os.remove(pandas_filename)
            except FileNotFoundError:
                pass


def check_to_csv_string_output(df, impl):
    """helper function that insures that output of to_csv is correct when returning a string from JIT code"""
    check_func(impl, (df,), only_seq=True)

    # check for distributed case, the output for each rank is the same as calling
    # df.to_csv(None) on the distributed dataframe
    py_out_1d = impl(_get_dist_arg(df))
    check_func(impl, (df,), only_1D=True, py_output=py_out_1d)

    py_out_1d_vars = impl(_get_dist_arg(df, var_length=True))
    check_func(impl, (df,), only_1DVar=True, py_output=py_out_1d_vars)


@pytest.mark.slow
def test_csv_non_constant_filepath_error(datapath):

    f1 = datapath("csv_data_cat1.csv")

    @bodo.jit
    def impl():
        for filepath in [f1]:
            df = pd.read_csv(
                filepath,
            )
        return df

    @bodo.jit
    def impl2():
        for filepath in [f1]:
            df = pd.read_csv(
                filepath,
                names=["A", "B", "C"],
            )
        return df

    @bodo.jit
    def impl3():
        for filepath in [f1]:
            df = pd.read_csv(
                filepath,
                dtype={
                    "A": int,
                    "B": str,
                    "C": str,
                },
            )
        return df

    def impl4():
        for filepath in [f1]:
            df = pd.read_csv(
                filepath,
                names=["A", "B", "C"],
                dtype={"A": int, "B": str, "C": str},
            )
        return df

    msg = (
        r".*pd.read_csv\(\) requires explicit type annotation using the 'names' "
        r"and 'dtype' arguments if the filename is not constant. For more information, "
        r"see: https://docs.bodo.ai/latest/file_io/#io_workflow."
    )

    with pytest.raises(BodoError, match=msg):
        bodo.jit(lambda: impl())()
    with pytest.raises(BodoError, match=msg):
        bodo.jit(lambda: impl2())()
    with pytest.raises(BodoError, match=msg):
        bodo.jit(lambda: impl3())()
    check_func(impl4, ())


@pytest.mark.slow
def test_json_non_constant_filepath_error(datapath):

    f1 = datapath("example.json")

    @bodo.jit
    def impl():
        for filepath in [f1]:
            df = pd.read_json(filepath, orient="records", lines=True)
        return df

    @bodo.jit
    def impl2():
        for filepath in [f1]:
            df = pd.read_json(
                filepath,
                orient="records",
                lines=True,
                dtype={
                    "one": float,
                    "two": str,
                    "three": np.bool_,
                    "four": float,
                    "five": str,
                },
            )
        return df

    @bodo.jit(
        locals={
            "df": {
                "one": bodo.float64[:],
                "two": bodo.string_array_type,
                "three": bodo.boolean_array,
                "four": bodo.float64[:],
                "five": bodo.string_array_type,
            }
        }
    )
    def impl3():
        for filepath in [f1]:
            df = pd.read_json(
                filepath,
                orient="records",
                lines=True,
            )
        return df

    msg = (
        r".*pd.read_json\(\) requires the filename to be a compile time constant. "
        r"For more information, see: https://docs.bodo.ai/latest/file_io/#json-section."
    )

    with pytest.raises(BodoError, match=msg):
        bodo.jit(lambda: impl())()
    with pytest.raises(BodoError, match=msg):
        bodo.jit(lambda: impl2())()
    with pytest.raises(BodoError, match=msg):
        bodo.jit(lambda: impl3())()


@pytest.mark.slow
def test_excel_non_constant_filepath_error(datapath):

    f1 = datapath("data.xlsx")

    @bodo.jit
    def impl():
        for filepath in [f1]:
            df = pd.read_excel(
                filepath,
            )
        return df

    @bodo.jit
    def impl2():
        for filepath in [f1]:
            df = pd.read_excel(
                filepath,
                names=["A", "B", "C", "D", "E"],
            )
        return df

    @bodo.jit
    def impl3():
        for filepath in [f1]:
            df = pd.read_excel(
                filepath,
                dtype={"A": int, "B": float, "C": str, "D": str, "E": np.bool_},
            )
        return df

    def impl4():
        for filepath in [f1]:
            df = pd.read_excel(
                filepath,
                names=["A", "B", "C", "D", "E"],
                dtype={"A": int, "B": float, "C": str, "D": str, "E": np.bool_},
            )
        return df

    msg1 = (
        r".*pd.read_excel\(\) requires explicit type annotation using the 'names' "
        r"and 'dtype' arguments if the filename is not constant. For more information, "
        r"see: https://docs.bodo.ai/latest/file_io/#io_workflow"
    )
    msg2 = r".*pd.read_excel\(\): both 'dtype' and 'names' should be provided if either is provided.*"

    with pytest.raises(BodoError, match=msg1):
        bodo.jit(lambda: impl())()
    with pytest.raises(BodoError, match=msg2):
        bodo.jit(lambda: impl2())()
    with pytest.raises(BodoError, match=msg2):
        bodo.jit(lambda: impl3())()
    # TODO [BE-1420]: Support distributed read_excel
    check_func(impl4, (), only_seq=True)


@pytest.mark.slow
def test_pq_non_constant_filepath_error(datapath):
    f1 = datapath("example.parquet")

    @bodo.jit
    def impl():
        for filepath in [f1]:
            pd.read_parquet(filepath)

    @bodo.jit(
        locals={
            "df": {
                "one": bodo.float64[:],
                "two": bodo.string_array_type,
                "three": bodo.boolean_array,
                "four": bodo.float64[:],
                "five": bodo.string_array_type,
            }
        }
    )
    def impl2():
        for filepath in [f1]:
            df = pd.read_parquet(filepath)
        return df

    msg = (
        r".*Parquet schema not available. Either path argument should be constant "
        r"for Bodo to look at the file at compile time or schema should be provided. "
        r"For more information, see: https://docs.bodo.ai/latest/file_io/#parquet-section."
    )

    with pytest.raises(BodoError, match=msg):
        bodo.jit(lambda: impl())()
    bodo.jit(lambda: impl2())()


def test_unify_null_column(memory_leak_check):
    """
    Tests reading from parquet with a null column in the first
    file unifies properly.
    """
    if bodo.get_rank() == 0:
        os.mkdir("temp_parquet_test")
        df1 = pd.DataFrame({"A": np.arange(10), "B": [None] * 10})
        df1.to_parquet("temp_parquet_test/f1.pq")
        df2 = pd.DataFrame({"A": np.arange(10, 16), "B": [None, "A"] * 3})
        df2.to_parquet("temp_parquet_test/f2.pq")
    bodo.barrier()
    try:

        def impl():
            return pd.read_parquet("temp_parquet_test")

        # Pandas doesn't seem to be able to unify data.
        # TODO: Open a Pandas issue?
        py_output = pd.DataFrame(
            {"A": np.arange(16), "B": ([None] * 10) + ([None, "A"] * 3)}
        )

        check_func(impl, (), py_output=py_output)
    finally:
        bodo.barrier()
        if bodo.get_rank() == 0:
            shutil.rmtree("temp_parquet_test")


def test_pd_datetime_arr_load_from_arrow(memory_leak_check):
    """
    Tests loading and returning an array with timezone information
    from Arrow.
    """
    if bodo.get_rank() == 0:
        df = pd.DataFrame(
            {
                "A": pd.date_range(
                    "2018-04-09", periods=50, freq="2D1H", tz="America/Los_Angeles"
                ),
                "B": pd.date_range("2018-04-09", periods=50, freq="2D1H"),
                "C": pd.date_range("2018-04-09", periods=50, freq="2D1H", tz="Poland"),
            }
        )
        # Create a pq ex
        df.to_parquet("test_tz.pq", index=False)

    def test_impl1():
        """
        Read parquet that should succeed
        because there are no tz columns.
        """
        df = pd.read_parquet("test_tz.pq")
        return df

    def test_impl2():
        """
        Read parquet that should succeed
        because there are no tz columns.
        """
        df = pd.read_parquet("test_tz.pq", columns=["B", "C"])
        return df

    bodo.barrier()
    try:
        check_func(test_impl1, (), only_seq=True)
        check_func(test_impl2, (), only_seq=True)
    finally:
        bodo.barrier()
        if bodo.get_rank() == 0:
            os.remove("test_tz.pq")


def test_tz_to_parquet(memory_leak_check):
    """
    Tests loading and returning an array with timezone information
    from Arrow. This tests both `to_parquet` and `array_to_info` support.
    """
    py_output = pd.DataFrame(
        {
            "A": pd.date_range(
                "2018-04-09", periods=50, freq="2D1H", tz="America/Los_Angeles"
            ),
            "B": pd.date_range("2018-04-09", periods=50, freq="2D1H"),
            "C": pd.date_range("2018-04-09", periods=50, freq="2D1H", tz="Poland"),
            "D": pd.date_range(
                "2018-04-09", periods=50, freq="2D1H", tz=pytz.FixedOffset(240)
            ),
        }
    )

    @bodo.jit(distributed=["df"])
    def impl(df, write_filename):
        df.to_parquet(write_filename, index=False)

    output_filename = "bodo_temp.pq"
    df = _get_dist_arg(py_output, True)
    impl(df, output_filename)
    bodo.barrier()
    # Read the data on rank 0 and compare
    passed = 1
    if bodo.get_rank() == 0:
        try:
            result = pd.read_parquet(output_filename)
            passed = _test_equal_guard(result, py_output)
            # Check the metadata. We want to verify that columns A and C
            # have the correct pandas type, numpy types, and metadata because
            # this is the first type that adds metadata.
            bodo_table = pq.read_table(output_filename)
            metadata = bodo_table.schema.pandas_metadata
            columns_info = metadata["columns"]
            tz_columns = ("A", "C")
            for col_name in tz_columns:
                col_index = result.columns.get_loc(col_name)
                col_metadata = columns_info[col_index]
                assert (
                    col_metadata["pandas_type"] == "datetimetz"
                ), f"incorrect pandas_type metadata for column {col_name}"
                assert (
                    col_metadata["numpy_type"] == "datetime64[ns]"
                ), f"incorrect numpy_type metadata for column {col_name}"
                metadata_field = col_metadata["metadata"]
                assert isinstance(
                    metadata_field, dict
                ), f"incorrect metadata field for column {col_name}"
                fields = [(k, v) for k, v in metadata_field.items()]
                assert fields == [
                    ("timezone", result.dtypes[col_index].tz.zone)
                ], f"incorrect metadata field for column {col_name}"
        except Exception:
            passed = 0
        finally:
            shutil.rmtree(output_filename)
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), "Output doesn't match Pandas data"


if __name__ == "__main__":
    unittest.main()
