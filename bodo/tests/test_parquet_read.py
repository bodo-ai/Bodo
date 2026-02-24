import datetime
import glob
import io
import os
import random
import re
import shutil
import string
import sys

import numba  # noqa TID253
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import bodo
from bodo import BodoWarning
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import (
    _check_for_io_reader_filters,
    _get_dist_arg,
    _test_equal,
    cast_dt64_to_ns,
    check_func,
    count_array_REPs,
    count_parfor_REPs,
    gen_random_arrow_array_struct_int,
    gen_random_arrow_list_list_double,
    gen_random_arrow_list_list_int,
    gen_random_arrow_struct_struct,
    pytest_mark_not_df_lib,
    temp_env_override,
)
from bodo.utils.testing import ensure_clean, ensure_clean2

pytestmark = [pytest.mark.parquet, pytest.mark.df_lib]


# ---------------------------- Test Different DataTypes ---------------------------- #
@pytest.mark.parametrize(
    "fname",
    [
        pytest.param("int_nulls_single.pq", id="int_single"),  # single piece
        pytest.param("int_nulls_multi.pq", id="int_multi"),  # multi piece
    ],
)
def test_pq_nullable(fname, datapath, memory_leak_check):
    fname = datapath(fname)

    def test_impl():
        return pd.read_parquet(fname, dtype_backend="pyarrow")

    check_func(test_impl, (), check_dtype=False)


@pytest.mark.parametrize(
    "fname",
    [
        pytest.param("bool_nulls.pq", id="bool_with_nulls"),
        pytest.param("list_str_arr.pq", id="list_str_arr"),
        pytest.param("list_str_parts.pq", id="list_str_parts"),
        pytest.param("decimal1.pq", id="decimal"),
        pytest.param("date32_1.pq", id="date32"),
        pytest.param(
            "small_strings.pq",
            id="processes_greater_than_string_rows",
        ),
        pytest.param("parquet_data_nonascii1.parquet", id="nonascii"),
        pytest.param("nullable_float.pq", id="nullable_float"),
        pytest.param("datetime64ns_1.pq", id="datetime64ns"),
        pytest.param("datetime64ns_tz_1.pq", id="datetime64ns_tz"),
        pytest.param("struct_1.pq", id="struct"),
        pytest.param("map_1.pq", id="map"),
        pytest.param("dictionary_string_1.pq", id="dictionary_string"),
    ],
)
def test_pq_read_types(fname, datapath, memory_leak_check):
    def test_impl(fname):
        return pd.read_parquet(fname, dtype_backend="pyarrow")

    check_func(test_impl, (datapath(fname),))


@pytest.mark.skipif(
    bodo.test_dataframe_library_enabled,
    reason="[BSE-4772] Creating Pandas DataFrame from BodoDataFrame columns.",
)
@pytest.mark.parametrize(
    "fname",
    [
        pytest.param("pandas_dt.pq", id="pandas_date"),
        pytest.param("sdf_dt.pq", id="spark_date"),
    ],
)
def test_pq_read_date(fname, datapath, memory_leak_check):
    fpath = datapath(fname)

    def impl():
        df = pd.read_parquet(fpath, dtype_backend="pyarrow")
        return pd.DataFrame({"DT64": df.DT64, "col2": df.DATE})

    df = pd.read_parquet(fpath, dtype_backend="pyarrow")
    output = cast_dt64_to_ns(pd.DataFrame({"DT64": df.DT64, "col2": df.DATE}))
    check_func(impl, (), only_seq=True, py_output=output)


@pytest.mark.slow
def test_read_partitions_datetime(memory_leak_check):
    """Test reading and filtering partitioned parquet data for datetime data"""
    with ensure_clean2("pq_data"):
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
            df = pd.read_parquet(
                path, columns=["c", "part", "a"], dtype_backend="pyarrow"
            )
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

        if not bodo.test_dataframe_library_enabled:
            from bodo.tests.utils_jit import SeriesOptTestPipeline

            # make sure the ParquetReader node has filters parameter set
            bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl1)
            bodo_func("pq_data", "2018-01-02", "2019-10-02")
            _check_for_io_reader_filters(bodo_func, bodo.ir.parquet_ext.ParquetReader)


def test_pd_datetime_arr_load_from_arrow(memory_leak_check):
    """
    Tests loading and returning an array with timezone information
    from Arrow.
    """
    if bodo.get_rank() == 0:
        df = pd.DataFrame(
            {
                "A": pd.date_range(
                    "2018-04-09", periods=50, freq="2D1h", tz="America/Los_Angeles"
                ),
                "B": pd.date_range("2018-04-09", periods=50, freq="2D1h"),
                "C": pd.date_range("2018-04-09", periods=50, freq="2D1h", tz="Poland"),
                "D": pd.date_range(
                    "2018-04-09", periods=50, freq="5001ns"
                ),  # Force timestamp64 as NS
            }
        )
        # Create a pq ex
        df.to_parquet("test_tz.pq", index=False, version="2.6")

    def test_impl1():
        """
        Read parquet that should succeed
        because there are no tz columns.
        """
        df = pd.read_parquet("test_tz.pq", dtype_backend="pyarrow")
        return df

    def test_impl2():
        """
        Read parquet that should succeed
        because there are no tz columns.
        """
        df = pd.read_parquet("test_tz.pq", columns=["B", "C"], dtype_backend="pyarrow")
        return df

    bodo.barrier()

    with ensure_clean2("test_tz.pq"):
        check_func(test_impl1, (), only_seq=True)
        check_func(test_impl2, (), only_seq=True)


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
        df = pd.read_parquet(_fname_, dtype_backend="pyarrow")
        return df

    py_output = pd.read_parquet(fname_, dtype_backend="pyarrow")

    check_func(test_impl, (fname_,), py_output=py_output)


def test_read_parquet_large_string_array(memory_leak_check):
    """
    Test that we can read `pa.large_string` arrays.
    """

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
            df = pd.read_parquet(path, dtype_backend="pyarrow")
            return df

        check_func(impl, (fname,))


@pytest.mark.weekly
@pytest.mark.jit_dependency
def test_read_parquet_vv_large_string_array(memory_leak_check):
    """
    Test that when reading a parquet dataset with total string length
    (and offsets) greater than Int32, the concatenate functionality
    works as expected. In particular, this is to protect against
    regressions to https://bodo.atlassian.net/browse/BSE-308.
    """

    def generate_random_string(length):
        # Get all the ASCII letters in lowercase
        letters = string.ascii_lowercase
        # Randomly choose characters from letters for the given length of the string.
        # We tile 1M at a time to speed up creation. In our use case
        # this should still have sufficient randomness.
        tile_size = 1_000_000
        random_string = "".join(
            f"{random.choice(letters)}" * tile_size for i in range(length // tile_size)
        )
        return random_string

    table = pa.table(
        [
            pa.array(
                [
                    # Generate a very large string.
                    # This can safely be different on every rank.
                    # This is just under INT32.MAX. Having 2 of
                    # these is enough to trigger the error.
                    generate_random_string(2_147_400_000),
                ],
                type=pa.large_string(),
            ),
        ],
        names=["A"],
    )

    import os

    fname = "vv_large_str_eg.pq"
    with ensure_clean2(fname):
        if bodo.get_rank() == 0:
            # Create folder
            os.mkdir(fname, mode=0o777)
        bodo.barrier()
        # Write 2 files on each rank (minimum required to reproduce the issue).
        # Writing a single file with >INT32.MAX bytes was causing some issues in write_table,
        # so instead we can write the same value twice in separate files. During the read,
        # this will read them as "one" dataset and hence allow us to reproduce
        # the issue.
        for i in range(2):
            pq.write_table(table, os.path.join(fname, f"part{bodo.get_rank()}-{i}.pq"))
        bodo.barrier()

        if bodo.test_dataframe_library_enabled:
            import bodo.pandas as bd

            def impl(path):
                df = bd.read_parquet(path)
                return df

            impl(fname)
        else:

            def impl(path):
                df = pd.read_parquet(path, dtype_backend="pyarrow")
                return df

            # Verify that it runs without error
            bodo.jit(impl)(fname)


# TODO [BE-1424]: Add memory_leak_check when bugs are resolved.
@pytest.mark.jit_dependency
def test_pq_arrow_array_random():
    def test_impl(fname):
        return pd.read_parquet(fname, dtype_backend="pyarrow")

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
            # Using Bodo to write since Pandas as of 2.0.3 doesn't read/write
            # Arrow arrays properly
            if bodo.get_rank() == 0:

                @bodo.jit(spawn=False, distributed=False)
                def write_file(df):
                    df.to_parquet(fname)

                write_file(df)
            bodo.barrier()
            check_func(
                test_impl,
                (fname,),
                check_dtype=False,
                py_output=pd.read_parquet(fname, dtype_backend="pyarrow"),
            )

    for df in [df_work1, df_work2, df_work3, df_work4, df_work5, df_work6, df_work7]:
        process_df(df)


def test_pq_categorical_read(memory_leak_check):
    """test reading categorical data from Parquet files"""

    def impl():
        df = pd.read_parquet("test_cat.pq", dtype_backend="pyarrow")
        return df

    try:
        df = pd.DataFrame(
            {"A": pd.Categorical(["A", "B", "AB", "A", None, "B"] * 4 + [None, "C"])}
        )
        if bodo.get_rank() == 0:
            df.to_parquet("test_cat.pq", row_group_size=4)
        bodo.barrier()
        check_func(impl, ())
        bodo.barrier()
    finally:
        if bodo.get_rank() == 0:
            os.remove("test_cat.pq")


# TODO [BE-1424]: Add memory_leak_check when bugs are resolved.
def test_pq_array_item(datapath):
    # TODO: [BE-581] Handle cases where the number of processes are
    # greater than the number of rows for nested arrays and other types.
    def test_impl(fname):
        return pd.read_parquet(fname, dtype_backend="pyarrow")

    check_func(test_impl, (datapath("list_int.pq"),))

    a = np.array(
        [[2.0, -3.2], [2.2, 1.3], None, [4.1, 5.2, 6.3], [], [1.1, 1.2]], object
    )
    b = np.array([[1, 3], [2], None, [4, 5, 6], [], [1, 1]], object)
    # for list of bools there are some things missing like (un)boxing
    # c = np.array([[True, False], None, None, [True, True, True], [False, False], []])
    df = pd.DataFrame({"A": a, "B": b})
    with ensure_clean("test_pq_list_item.pq"):
        if bodo.get_rank() == 0:
            df.to_parquet("test_pq_list_item.pq")
        bodo.barrier()
        check_func(test_impl, ("test_pq_list_item.pq",))

    a = np.array(
        [[[2.0], [-3.2]], [[2.2, 1.3]], None, [[4.1, 5.2], [6.3]], [], [[1.1, 1.2]]],
        object,
    )
    b = np.array([[[1], [3]], [[2]], None, [[4, 5, 6]], [], [[1, 1]]], object)
    df = pd.DataFrame({"A": a, "B": b})
    with ensure_clean("test_pq_list_item.pq"):
        if bodo.get_rank() == 0:
            df.to_parquet("test_pq_list_item.pq")
        bodo.barrier()
        check_func(test_impl, ("test_pq_list_item.pq",))


@pytest.mark.slow
def test_pq_unsupported_types(datapath, memory_leak_check):
    """test unsupported data types in unselected columns"""

    def test_impl(fname):
        return pd.read_parquet(fname, columns=["B"], dtype_backend="pyarrow")

    # FIXME I think we do suport everything in nested_struct_example.pq
    check_func(test_impl, (datapath("nested_struct_example.pq"),))


# ---------------------------- Test Read Indexes ---------------------------- #
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
        df = pd.read_parquet("test.pq", dtype_backend="pyarrow")
        return df

    try:
        if bodo.get_rank() == 0:
            test_RangeIndex_input.to_parquet("test.pq", index=pq_write_idx)
        bodo.barrier()
        check_func(impl, (), only_seq=True, reset_index=True)
        bodo.barrier()
    finally:
        if bodo.get_rank() == 0:
            os.remove("test.pq")
        bodo.barrier()


@pytest.mark.parametrize("index_name", [None, "HELLO"])
@pytest.mark.parametrize("pq_write_idx", [True, None, False])
def test_pq_select_column(
    test_RangeIndex_input, index_name, pq_write_idx, memory_leak_check
):
    def impl():
        df = pd.read_parquet("test.pq", columns=["A", "C"], dtype_backend="pyarrow")
        return df

    try:
        if bodo.get_rank() == 0:
            test_RangeIndex_input.index.name = index_name
            test_RangeIndex_input.to_parquet("test.pq", index=pq_write_idx)
        bodo.barrier()
        check_func(impl, (), only_seq=True, reset_index=True)
        bodo.barrier()
    finally:
        if bodo.get_rank() == 0:
            os.remove("test.pq")
        bodo.barrier()


def test_pq_index(datapath, memory_leak_check):
    def test_impl(fname):
        return pd.read_parquet(fname, dtype_backend="pyarrow")

    # passing function name as value to test value-based dispatch
    fname = datapath("index_test1.pq")
    check_func(test_impl, (fname,), only_seq=True, check_dtype=False)

    # string index
    fname = datapath("index_test2.pq")

    def test_impl2():
        return pd.read_parquet(fname, dtype_backend="pyarrow")

    check_func(test_impl2, (), only_seq=True, check_dtype=False)


def test_pq_multi_idx(memory_leak_check):
    """Remove this test when multi index is supported for read_parquet"""
    np.random.seed(0)

    def impl():
        return pd.read_parquet("multi_idx_parquet.pq", dtype_backend="pyarrow")

    try:
        if bodo.get_rank() == 0:
            arrays = [
                ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
                ["one", "two", "one", "two", "one", "two", "one", "two"],
                [1, 2, 2, 1] * 2,
            ]
            tuples = list(zip(*arrays))
            idx = pd.MultiIndex.from_tuples(tuples, names=["first", None, "count"])
            df = pd.DataFrame(np.random.randn(8, 2), index=idx, columns=["A", "B"])
            df.to_parquet("multi_idx_parquet.pq")
        bodo.barrier()

        check_func(impl, ())

    finally:
        if bodo.get_rank() == 0:
            os.remove("multi_idx_parquet.pq")


# ---------------------------- Test Read Partitions ---------------------------- #
@pytest.mark.slow
@pytest.mark.jit_dependency
def test_read_partitions(memory_leak_check):
    """test reading and filtering partitioned parquet data"""
    from bodo.tests.utils_jit import SeriesOptTestPipeline

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
            return pd.read_parquet(path, dtype_backend="pyarrow")

        def impl2(path, val):
            df = pd.read_parquet(path, dtype_backend="pyarrow")
            return df[df["part"] == val]

        # make sure filtering doesn't happen if df is used before filtering
        def impl3(path, val):
            df = pd.read_parquet(path, dtype_backend="pyarrow")
            n = len(df)
            n2 = len(df[df["part"] == val])
            return n, n2

        # make sure filtering doesn't happen if df is used after filtering
        def impl4(path, val):
            df = pd.read_parquet(path, dtype_backend="pyarrow")
            n2 = len(df[df["part"] == val])
            return len(df), n2

        # make sure filtering happens if df name is reused
        def impl5(path, val):
            df = pd.read_parquet(path, dtype_backend="pyarrow")
            n = len(df[df["part"] == val])
            df = pd.DataFrame({"A": np.arange(11)})
            n += df.A.sum()
            return n

        # make sure filtering works when there are no matching files
        def impl6(path):
            df = pd.read_parquet(path, dtype_backend="pyarrow")
            return df[df["part"] == "z"]

        # TODO(ehsan): make sure filtering happens if df name is reused in control flow
        # def impl7(path, val):
        #     df = pd.read_parquet(path, dtype_backend="pyarrow")
        #     n = len(df[df["part"] == val])
        #     if val == "b":
        #         df = pd.DataFrame({"A": np.arange(11)})
        #         n += df.A.sum()
        #     return n

        bodo.parquet_validate_schema = False  # type: ignore
        # check_dtype=False because Arrow 8 doesn't write pandas metadata
        # and therefore Bodo reads int64 as Int64, not maching pandas
        check_func(impl, ("pq_data",), check_dtype=False)
        bodo.parquet_validate_schema = True  # type: ignore
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
        bodo.parquet_validate_schema = True  # type: ignore
        if bodo.get_rank() == 0:
            shutil.rmtree("pq_data", ignore_errors=True)


@pytest.mark.slow
@pytest.mark.skipif(
    bodo.test_dataframe_library_enabled,
    reason="[BSE-4764] Dictionary columns with non-string value type not supported yet.",
)
def test_read_partitions2(memory_leak_check):
    """test reading and filtering partitioned parquet data in more complex cases"""
    from bodo.tests.utils_jit import SeriesOptTestPipeline

    with ensure_clean2("pq_data"):
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

        if not bodo.test_dataframe_library_enabled:
            # make sure the ParquetReader node has filters parameter set
            bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl1)
            bodo_func("pq_data", 3)
            _check_for_io_reader_filters(bodo_func, bodo.ir.parquet_ext.ParquetReader)


@pytest.mark.slow
@pytest.mark.skipif(
    bodo.test_dataframe_library_enabled,
    reason="[BSE-4764] Dictionary columns with non-string value type not supported yet.",
)
def test_read_partitions_implicit_and_detailed(memory_leak_check):
    """test reading and filtering partitioned parquet data with multiple levels
    of partitions and a complex implicit and"""
    from bodo.tests.utils_jit import SeriesOptTestPipeline

    with ensure_clean2("pq_data"):
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

        if not bodo.test_dataframe_library_enabled:
            # make sure the ParquetReader node has filters parameter set
            bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl1)
            bodo_func("pq_data", 3)
            _check_for_io_reader_filters(bodo_func, bodo.ir.parquet_ext.ParquetReader)


@pytest.mark.slow
@pytest.mark.skipif(
    bodo.test_dataframe_library_enabled,
    reason="[BSE-4764] Dictionary columns with non-string value type not supported yet.",
)
def test_read_partitions_implicit_and_simple(memory_leak_check):
    """test reading and filtering partitioned parquet data with multiple levels
    of partitions and an implicit and"""
    from bodo.tests.utils_jit import SeriesOptTestPipeline

    with ensure_clean2("pq_data"):
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
            df = pd.read_parquet(path, dtype_backend="pyarrow")
            df = df[df["part"] == "a"]
            # This function call should prevent lowering the second filter.
            sum1 = df["a"].sum()
            df = df[df["c"] == val]
            sum2 = df["a"].sum()
            return sum1 + sum2

        # TODO: match Index
        check_func(impl1, ("pq_data", 3), reset_index=True, check_dtype=False)

        if not bodo.test_dataframe_library_enabled:
            # make sure the ParquetReader node has filters parameter set
            bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl1)
            bodo_func("pq_data", 3)
            _check_for_io_reader_filters(bodo_func, bodo.ir.parquet_ext.ParquetReader)
            check_func(impl1, ("pq_data", 2), reset_index=True, check_dtype=False)


@pytest.mark.slow
@pytest.mark.skipif(
    bodo.test_dataframe_library_enabled,
    reason="[BSE-4764] Dictionary columns with non-string value type not supported yet.",
)
def test_read_partitions_string_int(memory_leak_check):
    """test reading from a file where the partition column could have
    a mix of strings and ints
    """
    from bodo.tests.utils_jit import SeriesOptTestPipeline

    with ensure_clean2("pq_data"):
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
            df = pd.read_parquet(path, dtype_backend="pyarrow")
            return df[(df["c"] == "abc") | (df["c"] == "4")]

        # TODO(ehsan): match Index
        check_func(impl1, ("pq_data",), reset_index=True)

        if not bodo.test_dataframe_library_enabled:
            # make sure the ParquetReader node has filters parameter set
            bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl1)
            bodo_func("pq_data")
            _check_for_io_reader_filters(bodo_func, bodo.ir.parquet_ext.ParquetReader)


@pytest.mark.slow
@pytest.mark.skipif(
    bodo.test_dataframe_library_enabled,
    reason="[BSE-4764] Dictionary columns with non-string value type not supported yet.",
)
def test_read_partitions_two_level(memory_leak_check):
    """test reading and filtering partitioned parquet data for two levels partitions"""
    from bodo.tests.utils_jit import SeriesOptTestPipeline

    with ensure_clean2("pq_data"):
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

        if not bodo.test_dataframe_library_enabled:
            # make sure the ParquetReader node has filters parameter set
            bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl1)
            bodo_func("pq_data", 3)
            _check_for_io_reader_filters(bodo_func, bodo.ir.parquet_ext.ParquetReader)


@pytest.mark.skipif(
    bodo.test_dataframe_library_enabled,
    reason="[BSE-4764] Dictionary columns with non-string value type not supported yet.",
)
def test_read_partitions_predicate_dead_column(memory_leak_check):
    """test reading and filtering predicate + partition columns
    doesn't load the columns if they are unused."""
    from bodo.tests.utils_jit import SeriesOptTestPipeline

    with ensure_clean2("pq_data"):
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
            df = pd.read_parquet(path, dtype_backend="pyarrow")
            return df[(df["a"] > 5) & (df["c"] == 2)].b

        # TODO(ehsan): match Index
        check_func(impl1, ("pq_data",), reset_index=True, check_dtype=False)

        if not bodo.test_dataframe_library_enabled:
            # make sure the ParquetReader node has filters parameter set
            stream = io.StringIO()
            logger = create_string_io_logger(stream)
            with set_logging_stream(logger, 1):
                bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl1)
                bodo_func("pq_data")
                _check_for_io_reader_filters(
                    bodo_func, bodo.ir.parquet_ext.ParquetReader
                )
                check_logger_msg(stream, "Columns loaded ['b']")


@pytest.mark.slow
@pytest.mark.skipif(
    bodo.test_dataframe_library_enabled,
    reason="[BSE-4764] Dictionary columns with non-string value type not supported yet.",
)
def test_read_partitions_cat_ordering(memory_leak_check):
    """test reading and filtering multi-level partitioned parquet data with many
    directories, to make sure order of partition values in categorical dtype
    of partition columns is consistent (same at compile time and runtime)
    and matches pandas"""
    from bodo.tests.utils_jit import SeriesOptTestPipeline

    with ensure_clean2("pq_data"):
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

        if not bodo.test_dataframe_library_enabled:
            # make sure the ParquetReader node has filters parameter set
            bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl2)
            bodo_func(
                "pq_data",
            )
            _check_for_io_reader_filters(bodo_func, bodo.ir.parquet_ext.ParquetReader)


@pytest.mark.slow
@pytest_mark_not_df_lib
@pytest.mark.parametrize("test_tz", [True, False])
def test_partition_cols(test_tz: bool, memory_leak_check):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    TEST_DIR = "test_part_tmp"

    @bodo.jit(distributed=["df"])
    def impl(df, part_cols):
        df.to_parquet(TEST_DIR, partition_cols=part_cols)

    @bodo.jit(distributed=["df"])
    def impl_tz(df, part_cols):
        df.to_parquet(TEST_DIR, partition_cols=part_cols, _bodo_timestamp_tz="CET")

    datetime_series = pd.Series(
        pd.date_range(start="2/1/2021", end="2/8/2021", periods=8)
    )
    datetime_series_tz = pd.Series(
        pd.date_range(start="2/1/2021", end="2/8/2021", periods=8, tz="CET")
    )
    timestamp_series = pd.Series([pd.Timestamp(f"200{i}-01-01") for i in range(8)])
    timestamp_series_tz = pd.Series(
        [pd.Timestamp(f"200{i}-01-01", tz="CET") for i in range(8)]
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
            "F": datetime_series_tz if test_tz else datetime_series,
            # TODO test following F column as partition column
            # for some reason, Bodo throws
            # "bodo.utils.typing.BodoError: Cannot convert dtype DatetimeDateType() to index type"
            # in _get_dist_arg with this:
            # "F": pd.Categorical(date_series),
            "G": range(8),
            "HH": timestamp_series_tz if test_tz else timestamp_series,
        }
    )

    try:
        to_test = [["A"], ["A", "B"], ["A", "B", "C"], ["D"], ["E"], ["D", "B"]]
        for part_cols in to_test:
            df_in = df.copy()
            if "D" not in part_cols:
                # TODO. can't write categorical to parquet currently because of metadata issue
                del df_in["D"]
            bodo.barrier()
            if test_tz:
                impl_tz(_get_dist_arg(df_in, False), part_cols)
            else:
                impl(_get_dist_arg(df_in, False), part_cols)
            bodo.barrier()

            err = None
            try:
                if bodo.get_rank() == 0:
                    # verify partitions are there
                    ds = pq.ParquetDataset(TEST_DIR)
                    assert ds.partitioning.schema.names == part_cols
                    # read bodo output with pandas
                    df_test = cast_dt64_to_ns(
                        pd.read_parquet(TEST_DIR, dtype_backend="pyarrow")
                    )
                    # pandas reads the partition columns as categorical, but they
                    # are not categorical in input dataframe, so we do some dtype
                    # conversions to be able to compare the dataframes
                    for part_col in part_cols:
                        if part_col == "E":
                            # convert categorical back to date
                            df_test[part_col] = (
                                df_test[part_col].astype("datetime64[ns]").dt.date
                            )
                        elif part_col == "C":
                            # convert the bool input column to categorical of strings
                            df_in[part_col] = (
                                df_in[part_col]
                                .astype(str)
                                .astype(df_test[part_col].dtype)
                            )
                        else:
                            # convert the categorical to same input dtype
                            df_test[part_col] = df_test[part_col].astype(
                                df_in[part_col].dtype
                            )
                    # use check_like=True because the order of columns has changed
                    # (partition columns appear at the end after reading)
                    pd.testing.assert_frame_equal(
                        df_test,
                        df_in,
                        check_like=True,
                        check_column_type=False,
                        check_dtype=False,
                    )
                    shutil.rmtree(TEST_DIR)
            except Exception as e:
                err = e
            err = comm.bcast(err)
            if isinstance(err, Exception):
                raise err
    finally:
        if bodo.get_rank() == 0:
            shutil.rmtree(TEST_DIR, ignore_errors=True)
        bodo.barrier()


@pytest.mark.slow
def test_read_partitions_to_datetime_format(memory_leak_check):
    """test that we don't incorrectly perform filter pushdown when to_datetime includes
    a format string."""

    with ensure_clean2("pq_data"):
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
            df = pd.read_parquet(
                path, columns=["c", "part", "a"], dtype_backend="pyarrow"
            )
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

        if not bodo.test_dataframe_library_enabled:
            from bodo.tests.utils_jit import SeriesOptTestPipeline, reduce_sum

            # make sure the ParquetReader node doesn't have filters parameter set
            bodo_func = bodo.jit(
                (
                    numba.types.literal("pq_data"),
                    numba.types.literal("2018-01-02"),
                    numba.types.literal("2019-10-02"),
                ),
                pipeline_class=SeriesOptTestPipeline,
            )(impl1)
            try:
                _check_for_io_reader_filters(
                    bodo_func, bodo.ir.parquet_ext.ParquetReader
                )
                # If we reach failed we have incorrectly performed filter pushdown
                passed = 0
            except AssertionError:
                passed = 1
            n_pes = bodo.get_size()
            n_passed = reduce_sum(passed)
            assert n_passed == n_pes, "Filter pushdown detected on at least 1 rank"


@pytest.mark.slow
def test_read_partitions_large(memory_leak_check):
    """
    test reading and filtering partitioned parquet data with large number of partitions
    """
    with ensure_clean2("pq_data"):
        if bodo.get_rank() == 0:
            # The number of dates can't exceed 1024 because that is the default
            # max_partitions of Arrow when writing parquet. XXX how to pass
            # max_partitions option to Arrow from df.to_parquet?
            I = pd.date_range("2018-01-03", "2020-10-05")
            df = pd.DataFrame(
                {"A": np.repeat(I.values, 100), "B": np.arange(100 * len(I))}
            )
            df.to_parquet("pq_data", partition_cols="A")  # type: ignore
        bodo.barrier()

        def impl1(path, s_d, e_d):
            df = pd.read_parquet(path, columns=["A", "B"], dtype_backend="pyarrow")
            return df[
                (pd.to_datetime(df["A"].astype(str)) >= pd.to_datetime(s_d))
                & (pd.to_datetime(df["A"].astype(str)) <= pd.to_datetime(e_d))
            ]

        check_func(impl1, ("pq_data", "2018-01-02", "2019-10-02"), reset_index=True)

        if not bodo.test_dataframe_library_enabled:
            from bodo.tests.utils_jit import SeriesOptTestPipeline

            # make sure the ParquetReader node has filters parameter set
            bodo_func = bodo.jit(
                (
                    numba.types.literal("pq_data"),
                    numba.types.literal("2018-01-02"),
                    numba.types.literal("2019-10-02"),
                ),
                pipeline_class=SeriesOptTestPipeline,
            )(impl1)
            _check_for_io_reader_filters(bodo_func, bodo.ir.parquet_ext.ParquetReader)


@pytest.mark.skipif(
    bodo.test_dataframe_library_enabled,
    reason="[BSE-4764] Unsupported column type dictionary<values=int32...>.",
)
def test_from_parquet_partition_bitsize(datapath):
    """Tests an issue with the bitsize of a partitioned dataframe"""
    import bodo.decorators  # isort:skip # noqa
    from bodo.hiframes.pd_series_ext import get_series_data

    path = datapath("test_partition_bitwidth.pq/")

    # For some reason, when the number of rows was small enough, the output was correct, despite the differing bitwidth.
    # However, checking the actual categories still exposes the error.
    def impl2(path):
        df = pd.read_parquet(path, dtype_backend="pyarrow")
        return (
            get_series_data(df["parent_wom"]).dtype.categories[0],
            get_series_data(df["parent_wom"]).dtype.categories[1],
            get_series_data(df["parent_wom"]).dtype.categories[2],
            get_series_data(df["parent_wom"]).dtype.categories[3],
        )

    check_func(impl2, (path,), py_output=(104, 105, 133, 134), check_dtype=False)


# ---------------------------- Test Read with Pushdown ---------------------------- #
@pytest.mark.slow
def test_read_predicates_pushdown_pandas_metadata(memory_leak_check):
    """test that predicate pushdown executes when there is Pandas range metadata."""
    with ensure_clean2("pq_data"):
        if bodo.get_rank() == 0:
            df = pd.DataFrame({"A": [0, 1, 2] * 10})
            df.to_parquet("pq_data")
        bodo.barrier()

        def impl(path):
            df = pd.read_parquet(path, dtype_backend="pyarrow")
            df = df[(df["A"] != 2)]
            return df

        # test for predicate pushdown removing all rows
        def impl2(path):
            df = pd.read_parquet(path, dtype_backend="pyarrow")
            df = df[(df["A"] == 100)]
            return df

        # TODO: Fix index
        check_func(impl, ("pq_data",), reset_index=True)
        check_func(impl2, ("pq_data",), reset_index=True)

        # TODO [BSE-4776]: Check filter conditions are set in DataFrame Lib.
        if not bodo.test_dataframe_library_enabled:
            from bodo.tests.utils_jit import SeriesOptTestPipeline

            # make sure the ParquetReader node has filters parameter set
            bodo_func = bodo.jit(
                (numba.types.literal("pq_data"),), pipeline_class=SeriesOptTestPipeline
            )(impl)
            _check_for_io_reader_filters(bodo_func, bodo.ir.parquet_ext.ParquetReader)


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
            df = pd.read_parquet("pq_data", dtype_backend="pyarrow")
            df = df[(df["B"] == 2) & (df["A"].notna())]
            return df

        # TODO: Fix index
        check_func(impl, ("pq_data",), reset_index=True)

        if not bodo.test_dataframe_library_enabled:
            # make sure the ParquetReader node has filters parameter set
            from bodo.tests.utils_jit import SeriesOptTestPipeline

            bodo_func = bodo.jit(
                (numba.types.literal("pq_data"),), pipeline_class=SeriesOptTestPipeline
            )(impl)
            _check_for_io_reader_filters(bodo_func, bodo.ir.parquet_ext.ParquetReader)
            bodo.barrier()
    finally:
        if bodo.get_rank() == 0:
            os.remove("pq_data")


@pytest.mark.skipif(
    bodo.test_dataframe_library_enabled,
    reason="[BSE-4768] datetime.date expr not supported yet.",
)
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
            df = pd.read_parquet(filepath, dtype_backend="pyarrow")
            df = df[df.A > datetime.date(2021, 3, 30)]
            return df

        # TODO: Fix index
        check_func(impl, (filepath,), reset_index=True)

        if not bodo.test_dataframe_library_enabled:
            from bodo.tests.utils_jit import SeriesOptTestPipeline

            # make sure the ParquetReader node has filters parameter set
            bodo_func = bodo.jit(
                (numba.types.literal(filepath),), pipeline_class=SeriesOptTestPipeline
            )(impl)
            _check_for_io_reader_filters(bodo_func, bodo.ir.parquet_ext.ParquetReader)
            bodo.barrier()


def test_read_predicates_isnull_alone(memory_leak_check):
    """test that predicate pushdown with isnull as the sole filter."""

    with ensure_clean2("pq_data"):
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
            df = pd.read_parquet(path, dtype_backend="pyarrow")
            df = df[df["A"].notna()]
            return df

        # TODO: Fix index
        check_func(impl, ("pq_data",), reset_index=True)

        if not bodo.test_dataframe_library_enabled:
            from bodo.tests.utils_jit import SeriesOptTestPipeline

            # make sure the ParquetReader node has filters parameter set
            bodo_func = bodo.jit(
                (numba.types.literal("pq_data"),), pipeline_class=SeriesOptTestPipeline
            )(impl)
            _check_for_io_reader_filters(bodo_func, bodo.ir.parquet_ext.ParquetReader)


@pytest.mark.slow
@pytest.mark.jit_dependency
def test_read_predicates_isin(memory_leak_check):
    """test that predicate pushdown with isin"""

    with ensure_clean2("pq_data"):
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
            df = pd.read_parquet("pq_data", dtype_backend="pyarrow")
            df = df[df.A.isin([1, 2])]
            return df.B

        def impl2(path):
            df = pd.read_parquet("pq_data", dtype_backend="pyarrow")
            df = df[df.A.isin({1, 2})]
            return df.B

        def impl3(path):
            df = pd.read_parquet("pq_data", dtype_backend="pyarrow")
            df = df[df["A"].isin([1, 2]) & df["C"].isin(["B"])]
            return df.B

        def impl4(path):
            df = pd.read_parquet("pq_data", dtype_backend="pyarrow")
            df = df[df["A"].isin({1, 2}) | df["C"].isin(["B"])]
            return df.B

        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            # TODO [BE-2351]: Fix index
            check_func(impl1, ("pq_data",), reset_index=True)

            if not bodo.test_dataframe_library_enabled:
                # Check filter pushdown succeeded
                check_logger_msg(stream, "Filter pushdown successfully performed")
                # Check the columns were pruned
                check_logger_msg(stream, "Columns loaded ['B']")
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            # TODO [BE-2351]: Fix index
            check_func(impl2, ("pq_data",), reset_index=True)

            if not bodo.test_dataframe_library_enabled:
                # Check filter pushdown succeeded
                check_logger_msg(stream, "Filter pushdown successfully performed")
                # Check the columns were pruned
                check_logger_msg(stream, "Columns loaded ['B']")
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            # TODO [BE-2351]: Fix index
            check_func(impl3, ("pq_data",), reset_index=True)

            if not bodo.test_dataframe_library_enabled:
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

            if not bodo.test_dataframe_library_enabled:
                # Check filter pushdown succeeded
                check_logger_msg(stream, "Filter pushdown successfully performed")
                # Check the columns were pruned
                check_logger_msg(stream, "Columns loaded ['B']")


# This test is slow on Windows
@pytest.mark.timeout(600)
@pytest.mark.slow
@pytest.mark.skipif(
    bodo.test_dataframe_library_enabled, reason="[BSE-4768] isin not supported yet."
)
def test_read_partitions_isin(memory_leak_check):
    """test that partition pushdown with isin"""

    with ensure_clean2("pq_data"):
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
            df = pd.read_parquet("pq_data", dtype_backend="pyarrow")
            df = df[df.A.isin([1, 2])]
            return df.B

        def impl2(path):
            df = pd.read_parquet("pq_data", dtype_backend="pyarrow")
            df = df[df["A"].isin([1, 2]) & df["C"].isin(["B"])]
            return df.B

        def impl3(path):
            df = pd.read_parquet("pq_data", dtype_backend="pyarrow")
            df = df[df["A"].isin([1, 2]) | df["C"].isin(["B"])]
            return df.B

        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            # TODO: Fix index
            check_func(impl1, ("pq_data",), reset_index=True, check_dtype=False)

            if not bodo.test_dataframe_library_enabled:
                # Check filter pushdown succeeded
                check_logger_msg(stream, "Filter pushdown successfully performed")
                # Check the columns were pruned
                check_logger_msg(stream, "Columns loaded ['B']")
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            # TODO: Fix index
            check_func(impl2, ("pq_data",), reset_index=True, check_dtype=False)

            if not bodo.test_dataframe_library_enabled:
                # Check filter pushdown succeeded
                check_logger_msg(stream, "Filter pushdown successfully performed")
                # Check the columns were pruned
                check_logger_msg(stream, "Columns loaded ['B']")
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            # TODO: Fix index
            check_func(impl3, ("pq_data",), reset_index=True, check_dtype=False)

            if not bodo.test_dataframe_library_enabled:
                # Check filter pushdown succeeded
                check_logger_msg(stream, "Filter pushdown successfully performed")
                # Check the columns were pruned
                check_logger_msg(stream, "Columns loaded ['B']")


@pytest.mark.slow
def test_read_predicates_and_or(memory_leak_check):
    """test that predicate pushdown with and/or in the expression."""

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
        df = pd.read_parquet("pq_data", dtype_backend="pyarrow")
        df = df[
            (((df["A"] == 2) | (df["B"] == 1)) & (df["B"] != 4))
            & (((df["A"] == 3) | (df["B"] == 5)) & (df["B"] != 2))
        ]
        return df

    with ensure_clean2("pq_data"):
        # TODO: Fix index
        check_func(impl, ("pq_data",), reset_index=True)

        if not bodo.test_dataframe_library_enabled:
            from bodo.tests.utils_jit import SeriesOptTestPipeline

            # make sure the ParquetReader node has filters parameter set
            bodo_func = bodo.jit(
                (numba.types.literal("pq_data"),), pipeline_class=SeriesOptTestPipeline
            )(impl)
            _check_for_io_reader_filters(bodo_func, bodo.ir.parquet_ext.ParquetReader)


@pytest.mark.slow
@pytest.mark.skipif(bodo.dataframe_library_enabled, reason="Invalid Pandas code.")
def test_bodosql_pushdown_codegen(datapath):
    """
    Make sure possible generated codes by BodoSQL work with filter pushdown.
    See [BE-3557]
    """
    from bodo.tests.utils_jit import SeriesOptTestPipeline

    def impl1(filename):
        df = pd.read_parquet(filename, dtype_backend="pyarrow")
        df = df[
            pd.Series(bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, 0)).values
            > 1
        ]
        return df["two"]

    filename = datapath("example.parquet")

    py_output = pd.Series(["baz", "foo", "bar", "baz", "foo"], name="two")
    check_func(impl1, (filename,), py_output=py_output, reset_index=True)
    bodo_func = bodo.jit(
        (numba.types.literal(filename),), pipeline_class=SeriesOptTestPipeline
    )(impl1)
    _check_for_io_reader_filters(bodo_func, bodo.ir.parquet_ext.ParquetReader)


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

    # TODO: We currently clear the logger stream before each call to check_func
    # via calling stream.seek/truncate. We should investigate if this actually
    # fully clears the logger, or if additional work is needed.
    with set_logging_stream(logger, 1):
        try:
            if bodo.get_rank() == 0:
                df.to_parquet("pq_data", partition_cols="A")  # type: ignore

            def impl1():
                df = pd.read_parquet("pq_data", dtype_backend="pyarrow")
                df = df[["A", "C"]]
                df = df[df["C"] > 5]
                return df

            def impl2():
                df = pd.read_parquet("pq_data", dtype_backend="pyarrow")
                df = df.loc[:, ["B", "A", "D"]]
                df = df[df["D"] < 4]
                return df

            def impl3():
                df = pd.read_parquet("pq_data", dtype_backend="pyarrow")
                df = df.loc[:, ["C", "B", "A", "D"]]
                df = df.loc[:, ["B", "A", "D"]]
                df = df.loc[:, ["B", "D"]]
                df = df.loc[:, ["D"]]
                df = df[df["D"] < 4]
                return df

            check_func(impl1, ())
            check_logger_msg(stream, "TODO")
            stream.truncate(0)
            stream.seek(0)
            check_func(impl2, ())
            check_logger_msg(stream, "TODO")
            stream.truncate(0)
            stream.seek(0)
            check_func(impl3, ())
            check_logger_msg(stream, "TODO")
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
@pytest.mark.skipif(
    bodo.test_dataframe_library_enabled,
    reason="[BSE-4764] Dictionary columns with non-string value type not supported yet.",
)
def test_read_pq_head_only(datapath, memory_leak_check):
    """
    test reading only shape and/or head from Parquet file if possible
    (limit pushdown)
    """
    from bodo.tests.utils_jit import DistTestPipeline

    # read both shape and head()
    def impl1(path):
        df = pd.read_parquet(path, dtype_backend="pyarrow")
        return df.shape, df.head(4)

    # shape only
    def impl2(path):
        df = pd.read_parquet(path, dtype_backend="pyarrow")
        return len(df)

    # head only
    def impl3(path):
        df = pd.read_parquet(path, dtype_backend="pyarrow")
        return df.head()

    # head and shape without table format
    def impl4(path):
        df = pd.read_parquet(path, dtype_backend="pyarrow").loc[:, ["A"]]
        return df.shape, df.head(4)

    # large file
    fname = datapath("int_nulls_multi.pq")
    check_func(impl1, (fname,), check_dtype=False)
    check_func(impl2, (fname,), check_dtype=False)
    check_func(impl4, (fname,), check_dtype=False)
    check_func(impl3, (fname,), check_dtype=False)
    # small file with Index data
    check_func(impl1, (datapath("index_test2.pq"),), check_dtype=False)

    if not bodo.test_dataframe_library_enabled:
        # make sure head-only read is recognized correctly
        bodo_func = bodo.jit(
            (numba.types.literal(fname),), pipeline_class=DistTestPipeline
        )(impl1)
        _check_for_pq_read_head_only(bodo_func)
        bodo_func = bodo.jit(
            (numba.types.literal(fname),), pipeline_class=DistTestPipeline
        )(impl2)
        _check_for_pq_read_head_only(bodo_func, has_read=False)
        bodo_func = bodo.jit(
            (numba.types.literal(fname),), pipeline_class=DistTestPipeline
        )(impl3)
        _check_for_pq_read_head_only(bodo_func)
        bodo_func = bodo.jit(
            (numba.types.literal(fname),), pipeline_class=DistTestPipeline
        )(impl4)
        _check_for_pq_read_head_only(bodo_func)
    # partitioned data
    with ensure_clean2("pq_data"):
        if bodo.get_rank() == 0:
            # The number of dates can't exceed 1024 because that is the default
            # max_partitions of Arrow when writing parquet. XXX how to pass
            # max_partitions option to Arrow from df.to_parquet?
            I = pd.date_range("2018-01-03", "2020-10-05")
            df = pd.DataFrame(
                {"A": np.repeat(I.values, 100), "B": np.arange(100 * len(I))}
            )
            df.to_parquet("pq_data", partition_cols="A")  # type: ignore
        bodo.barrier()

        check_func(impl1, ("pq_data",), check_dtype=False)

        if not bodo.test_dataframe_library_enabled:
            bodo_func = bodo.jit(
                (numba.types.literal("pq_data"),), pipeline_class=DistTestPipeline
            )(impl1)
            _check_for_pq_read_head_only(bodo_func)


@pytest.mark.jit_dependency
def test_limit_pushdown_multiple_tables(datapath, memory_leak_check):
    """
    test reading only shape and/or head from Parquet file if possible
    (limit pushdown) with multiple DataFrames in the same block.
    """

    def impl(path1, path2):
        df1 = pd.read_parquet(path1, dtype_backend="pyarrow")
        df2 = pd.read_parquet(path2, dtype_backend="pyarrow")
        return df1.head(4), df2.head(5)

    fname = datapath("int_nulls_multi.pq")
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        out = pd.read_parquet(fname, dtype_backend="pyarrow")
        check_func(
            impl,
            (fname, fname),
            check_dtype=False,
            py_output=(out.head(4), out.head(5)),
        )

        if not bodo.test_dataframe_library_enabled:
            check_logger_msg(stream, "Constant limit detected, reading at most 4 rows")
            check_logger_msg(stream, "Constant limit detected, reading at most 5 rows")


def _check_for_pq_read_head_only(bodo_func, has_read=True):
    """make sure head-only parquet read optimization is recognized"""
    fir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert hasattr(fir, "meta_head_only_info")
    assert not has_read or fir.meta_head_only_info[0] is not None


# ---------------------------- Test Different Paths ---------------------------- #
def test_read_pq_trailing_sep(datapath, memory_leak_check):
    folder_name = datapath("list_int.pq/")

    def impl():
        return pd.read_parquet(folder_name, dtype_backend="pyarrow")

    check_func(impl, ())


def test_read_parquet_glob(datapath, memory_leak_check):
    def test_impl(filename):
        df = pd.read_parquet(filename, dtype_backend="pyarrow")
        return df

    filename = datapath("int_nulls_multi.pq")
    pyout = pd.read_parquet(filename, dtype_backend="pyarrow")
    # add glob patterns (only for Bodo, pandas doesn't support it)
    glob_pattern_1 = filename + "/part*.parquet"
    check_func(test_impl, (glob_pattern_1,), py_output=pyout, check_dtype=False)
    glob_pattern_2 = filename + "/part*-3af07a60-*ab59*.parquet"
    check_func(test_impl, (glob_pattern_2,), py_output=pyout, check_dtype=False)


@pytest.mark.skipif(
    bodo.test_dataframe_library_enabled,
    reason="[BSE-4764] Unsupported column type dictionary<values=int32...>.",
)
def test_read_parquet_list_of_globs(memory_leak_check):
    """test reading when passing a list of globstrings"""

    def test_impl(filename):
        df = pd.read_parquet(filename, dtype_backend="pyarrow")
        return df

    globstrings = [
        "bodo/tests/data/test_partitioned.pq/A=2/part-0000[1-2]-*.parquet",
        "bodo/tests/data/test_partitioned.pq/A=7/part-0000[5-7]-bfd81e52-9210-4ee9-84a0-5ee2ab5e6345.c000.snappy.parquet",
    ]

    if sys.platform == "win32":
        globstrings = [p.replace("/", "\\") for p in globstrings]

    # construct expected pandas output manually (pandas doesn't support list of files)
    files = []
    for globstring in globstrings:
        files += sorted(glob.glob(globstring))
    # Arrow ParquetDatasetV2 adds the partition column to the dataset
    # when passing it a list of files that are in partitioned directories.
    # So we need to add the partition column to pandas output
    chunks = []

    regex_str = r"\/A=(\d+)\/" if sys.platform != "win32" else r"\\A=(\d+)\\"
    regexp = re.compile(regex_str)
    for f in files:
        df = pd.read_parquet(f, dtype_backend="pyarrow")
        df["A"] = int(regexp.search(f).group(1))
        chunks.append(df)
    pyout = pd.concat(chunks).reset_index(drop=True)
    pyout["A"] = pyout["A"].astype("category")

    check_func(test_impl, (globstrings,), py_output=pyout, check_dtype=False)


@pytest.mark.slow
def test_read_parquet_list_files(datapath, memory_leak_check):
    """test read_parquet passing a list of files"""

    def test_impl():
        return pd.read_parquet(
            ["bodo/tests/data/example.parquet", "bodo/tests/data/example2.parquet"],  # type: ignore
            dtype_backend="pyarrow",
        )

    def test_impl2(fpaths):
        return pd.read_parquet(fpaths, dtype_backend="pyarrow")

    py_output_part1 = pd.read_parquet(
        datapath("example.parquet"), dtype_backend="pyarrow"
    )
    py_output_part2 = pd.read_parquet(
        datapath("example2.parquet"), dtype_backend="pyarrow"
    )
    py_output = pd.concat([py_output_part1, py_output_part2])
    check_func(test_impl, (), py_output=py_output)
    fpaths = [datapath("example.parquet"), datapath("example2.parquet")]
    check_func(test_impl2, (fpaths,), py_output=py_output)


@pytest.mark.slow
@pytest_mark_not_df_lib
def test_pq_non_constant_filepath_error(datapath):
    from bodo.utils.typing import BodoError

    f1 = datapath("example.parquet")

    @bodo.jit
    def impl():
        for filepath in [f1]:
            pd.read_parquet(filepath, dtype_backend="pyarrow")

    @bodo.jit(
        locals={
            "df": {
                "one": bodo.types.float64[:],
                "two": bodo.types.string_array_type,
                "three": bodo.types.boolean_array_type,
                "four": bodo.types.float64[:],
                "five": bodo.types.string_array_type,
                "__index_level_0__": bodo.types.int64[:],
            }
        }
    )
    def impl2():
        for filepath in [f1]:
            df = pd.read_parquet(filepath, dtype_backend="pyarrow")
        return df  # type: ignore

    msg = (
        r".*Parquet schema not available. Either path argument should be constant "
        r"for Bodo to look at the file at compile time or schema should be provided. "
        r"For more information, see: https://docs.bodo.ai/latest/file_io/#parquet-section."
    )

    with pytest.raises(BodoError, match=msg):
        bodo.jit(lambda: impl())()
    bodo.jit(lambda: impl2())()


# TODO: Not sure why this fails with memory_leak_check. Seems like the
# exception returned from pq_read_py_entry prevents the code that follows from
# freeing something
@pytest_mark_not_df_lib
def test_read_parquet_invalid_path():
    """test error raise when parquet file path is invalid in C++ code."""
    from bodo.utils.typing import BodoError

    def test_impl():
        df = pd.read_parquet("I_dont_exist.pq", dtype_backend="pyarrow")
        return df

    with pytest.raises(BodoError, match="error from pyarrow: FileNotFoundError"):
        bodo.jit(locals={"df": {"A": bodo.types.int64[:]}})(test_impl)()


@pytest_mark_not_df_lib
def test_read_parquet_invalid_path_glob():
    from bodo.utils.typing import BodoError

    def test_impl():
        df = pd.read_parquet("I*dont*exist", dtype_backend="pyarrow")
        return df

    with pytest.raises(BodoError, match="No files found matching glob pattern"):
        bodo.jit(locals={"df": {"A": bodo.types.int64[:]}})(test_impl)()


@pytest_mark_not_df_lib
def test_read_parquet_invalid_list_of_files(datapath):
    from bodo.utils.typing import BodoError

    def test_impl(fnames):
        df = pd.read_parquet(fnames, dtype_backend="pyarrow")
        return df

    with pytest.raises(
        BodoError,
        match=re.escape(
            "Make sure the list/glob passed to read_parquet() only contains paths to files (no directories)"
        ),
    ):
        fnames = [datapath("decimal1.pq"), datapath("dask_data.parquet")]
        bodo.jit(test_impl)(fnames)

    with pytest.raises(
        BodoError,
        match=re.escape(
            "Make sure the list/glob passed to read_parquet() only contains paths to files (no directories)"
        ),
    ):
        fnames = []
        fnames.append(
            datapath("decimal1.pq")
            + "/part-00000-053de46f-b6f6-47bc-b732-fa60df446076-c000.snappy.parquet"
        )
        fnames.append(datapath("decimal1.pq"))
        bodo.jit(test_impl)(fnames)


@pytest_mark_not_df_lib
def test_read_parquet_invalid_path_const(memory_leak_check):
    """test error raise when parquet file path provided as constant but is invalid."""
    from bodo.utils.typing import BodoError

    def test_impl():
        return pd.read_parquet("I_dont_exist.pq", dtype_backend="pyarrow")

    with pytest.raises(BodoError, match="error from pyarrow: FileNotFoundError"):
        bodo.jit(test_impl)()


# ---------------------------- Test Dictionary Encoding ---------------------------- #
@pytest_mark_not_df_lib
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
                # CC2 should be dictionary encoded
                "CC2": ["r32r23r32r32r23"] * 1000,
                # D is non-string column, so shouldn't be encoded even if specified
                "D": np.arange(1000),
            }
        )
        df.to_parquet(fname, index=False)
    bodo.barrier()

    @bodo.jit
    def test_impl1(fname):
        return pd.read_parquet(fname, _bodo_read_as_dict=["A"], dtype_backend="pyarrow")

    @bodo.jit
    def test_impl2(fname):
        return pd.read_parquet(fname, _bodo_read_as_dict=["B"], dtype_backend="pyarrow")

    @bodo.jit
    def test_impl3(fname):
        return pd.read_parquet(
            fname, _bodo_read_as_dict=["CC2"], dtype_backend="pyarrow"
        )

    @bodo.jit
    def test_impl4(fname):
        return pd.read_parquet(
            fname, _bodo_read_as_dict=["A", "CC2"], dtype_backend="pyarrow"
        )

    @bodo.jit
    def test_impl5(fname):
        return pd.read_parquet(
            fname, _bodo_read_as_dict=["B", "CC2"], dtype_backend="pyarrow"
        )

    @bodo.jit
    def test_impl6(fname):
        return pd.read_parquet(
            fname, _bodo_read_as_dict=["A", "B"], dtype_backend="pyarrow"
        )

    @bodo.jit
    def test_impl7(fname):
        return pd.read_parquet(
            fname, _bodo_read_as_dict=["A", "B", "CC2"], dtype_backend="pyarrow"
        )

    # 'D' shouldn't be read as dictionary encoded since it's not a string column
    @bodo.jit
    def test_impl8(fname):
        return pd.read_parquet(
            fname, _bodo_read_as_dict=["A", "B", "CC2", "D"], dtype_backend="pyarrow"
        )

    @bodo.jit
    def test_impl9(fname):
        return pd.read_parquet(fname, _bodo_read_as_dict=["D"], dtype_backend="pyarrow")

    @bodo.jit
    def test_impl10(fname):
        return pd.read_parquet(
            fname, _bodo_read_as_dict=["A", "D"], dtype_backend="pyarrow"
        )

    try:
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            test_impl1(fname)
            check_logger_msg(stream, "Columns ['A', 'CC2'] using dictionary encoding")

        with set_logging_stream(logger, 1):
            test_impl2(fname)
            check_logger_msg(
                stream, "Columns ['A', 'B', 'CC2'] using dictionary encoding"
            )

        with set_logging_stream(logger, 1):
            test_impl3(fname)
            check_logger_msg(stream, "Columns ['A', 'CC2'] using dictionary encoding")

        with set_logging_stream(logger, 1):
            test_impl4(fname)
            check_logger_msg(stream, "Columns ['A', 'CC2'] using dictionary encoding")

        with set_logging_stream(logger, 1):
            test_impl5(fname)
            check_logger_msg(
                stream, "Columns ['A', 'B', 'CC2'] using dictionary encoding"
            )

        with set_logging_stream(logger, 1):
            test_impl6(fname)
            check_logger_msg(
                stream, "Columns ['A', 'B', 'CC2'] using dictionary encoding"
            )

        with set_logging_stream(logger, 1):
            test_impl7(fname)
            check_logger_msg(
                stream, "Columns ['A', 'B', 'CC2'] using dictionary encoding"
            )

        with set_logging_stream(logger, 1):
            test_impl8(fname)
            check_logger_msg(
                stream, "Columns ['A', 'B', 'CC2'] using dictionary encoding"
            )

        with set_logging_stream(logger, 1):
            test_impl9(fname)
            check_logger_msg(stream, "Columns ['A', 'CC2'] using dictionary encoding")

        with set_logging_stream(logger, 1):
            test_impl10(fname)
            check_logger_msg(stream, "Columns ['A', 'CC2'] using dictionary encoding")

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
        bodo.barrier()
        if bodo.get_rank() == 0:
            os.remove(fname)


def test_read_parquet_partitioned_read_as_dict(memory_leak_check):
    """
    Make sure dict-encoding determination works for partitioned Parquet datasets
    See https://bodo.atlassian.net/browse/BE-3679
    """
    fname = "pq_data_dict"

    with ensure_clean2(fname):
        if bodo.get_rank() == 0:
            table = pa.table(
                {
                    # A should be dictionary encoded
                    "A": ["awerwe", "awerwev24v2", "3r2r32rfc3", "ERr32r23rrrrrr"]
                    * 250,
                    # B should not be dictionary encoded
                    "B": [str(i) for i in range(1000)],
                    # CC2 should be dictionary encoded
                    "CC2": ["r32r23r32r32r23"] * 1000,
                    # D is non-string column, so shouldn't be encoded even if specified
                    "D": np.arange(1000),
                    # partitions
                    "part": ["A", "B", "C", "D"] * 250,
                }
            )
            pq.write_to_dataset(table, fname, partition_cols=["part"])
        bodo.barrier()

        def test_impl1(fname):
            return pd.read_parquet(fname, dtype_backend="pyarrow")

        check_func(test_impl1, (fname,))

        if not bodo.test_dataframe_library_enabled:
            stream = io.StringIO()
            logger = create_string_io_logger(stream)
            with set_logging_stream(logger, 1):
                bodo.jit(test_impl1)(fname)
                check_logger_msg(
                    stream, "Columns ['A', 'CC2'] using dictionary encoding"
                )


# ---------------------------- Test Additional Args ---------------------------- #
@pytest.mark.skipif(
    bodo.test_dataframe_library_enabled,
    reason="[BSE-4770] Support the columns argument for read_parquet.",
)
@pytest.mark.parametrize(
    "col_subset",
    [
        ["A", "A2", "C"],
        ["C", "B", "A2"],
        ["B"],
    ],
)
def test_read_parquet_all_null_col_subsets(
    col_subset: list[str], memory_leak_check, datapath
):
    """test that columns with all nulls can be read successfully"""
    fname = datapath("all_null_col_eg2.pq")

    def test_impl(fname):
        df = pd.read_parquet(fname, columns=col_subset, dtype_backend="pyarrow")
        return df

    py_output = pd.read_parquet(fname, columns=col_subset, dtype_backend="pyarrow")

    check_func(test_impl, (fname,), py_output=py_output)


@pytest.mark.skipif(
    bodo.test_dataframe_library_enabled,
    reason="[BSE-4770] _bodo_input_file_name_col not supported in DataFrame library.",
)
def test_read_parquet_input_file_name_col(datapath, memory_leak_check):
    """test basic input_col_name functionality for read_parquet"""
    fname = datapath("decimal1.pq")

    def test_impl(fname):
        df = pd.read_parquet(
            fname, _bodo_input_file_name_col="fname", dtype_backend="pyarrow"
        )
        # pyspark adds prefix `file://` for local files, but we follow PyArrow
        # XXX Should we do this by default?
        return df

    # PyArrow engine for Pandas supports reading the input file name via the
    # __filename column
    df: pd.DataFrame = pq.read_table(fname, columns=["A", "__filename"]).to_pandas()
    py_output = df.rename({"__filename": "fname"}, axis=1)
    py_output["A"] = py_output["A"].astype(pd.ArrowDtype(pa.decimal128(38, 18)))

    check_func(
        test_impl,
        (fname,),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_output,
    )


@pytest.mark.skipif(
    bodo.test_dataframe_library_enabled,
    reason="[BSE-4770] _bodo_input_file_name_col not supported in DataFrame library.",
)
def test_read_parquet_input_file_name_col_with_partitions(datapath, memory_leak_check):
    """
    test input_col_name functionality for read_parquet
    In particular, this tests that it works as expected when
    the input dataset has partitions
    """

    fname = datapath("test_partitioned.pq")

    def test_impl(fname):
        df = pd.read_parquet(
            fname, _bodo_input_file_name_col="fname", dtype_backend="pyarrow"
        )
        # pyspark adds prefix `file://` for local files, but we follow PyArrow
        # XXX Should we do this by default?
        return df

    # PyArrow engine for Pandas supports reading the input file name via the
    # __filename column
    df: pd.DataFrame = pq.read_table(
        fname, columns=["B", "A", "__filename"]
    ).to_pandas()
    py_output = df.rename({"__filename": "fname"}, axis=1)

    check_func(
        test_impl,
        (fname,),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_output,
    )


@pytest.mark.skipif(
    bodo.test_dataframe_library_enabled,
    reason="[BSE-4770] _bodo_input_file_name_col not supported in DataFrame library.",
)
def test_read_parquet_input_file_name_col_with_index(datapath, memory_leak_check):
    """
    test input_col_name functionality for read_parquet
    In particular we check that it works fine with files containing
    index columns (e.g. written by pandas)
    """
    fname = datapath("example.parquet")

    def test_impl(fname):
        df = pd.read_parquet(
            fname, _bodo_input_file_name_col="fname", dtype_backend="pyarrow"
        )
        return df

    # Unlike the other tests, we're only checking for a specific code path,
    # so we don't need to check against PySpark directly.
    py_output = pd.read_parquet(fname, dtype_backend="pyarrow")
    # PyArrow replaces "\" with "/" in file paths on Windows for some reason
    py_output["fname"] = fname.replace("\\", "/")

    check_func(
        test_impl,
        (fname,),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_output,
    )


@pytest.mark.skipif(
    bodo.test_dataframe_library_enabled,
    reason="[BSE-4770] _bodo_input_file_name_col not supported in DataFrame library.",
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
        df = pd.read_parquet(
            fname, _bodo_input_file_name_col="fname", dtype_backend="pyarrow"
        )
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
    py_output = pd.read_parquet(fname, dtype_backend="pyarrow")
    py_output = py_output[["one", "two", "three"]]

    check_func(
        test_impl,
        (fname,),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_output,
    )


@pytest.mark.skipif(
    bodo.test_dataframe_library_enabled,
    reason="[BSE-4770] _bodo_input_file_name_col not supported in DataFrame library.",
)
def test_read_parquet_only_input_file_name_col(datapath, memory_leak_check):
    """
    test input_col_name functionality for read_parquet
    In particular test that it works as expected when only the filename
    column is used (the rest are pruned).
    """
    fname = datapath("decimal1.pq")

    def test_impl(fname):
        df = pd.read_parquet(
            fname, _bodo_input_file_name_col="fname", dtype_backend="pyarrow"
        )
        # pyspark adds prefix `file://` for local files, but we follow PyArrow
        # XXX Should we do this by default?
        return df[["fname"]]

    # Check that columns were pruned using verbose logging
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(test_impl)(fname)
        check_logger_msg(stream, "Columns loaded ['fname']")

    # Check that output is correct
    # PyArrow engine for Pandas supports reading the input file name via the
    # __filename column
    df: pd.DataFrame = pq.read_table(fname, columns=["__filename"]).to_pandas()
    py_output = df.rename({"__filename": "fname"}, axis=1)

    check_func(
        test_impl,
        (fname,),
        sort_output=True,
        reset_index=True,
        check_dtype=False,
        py_output=py_output,
    )


@pytest_mark_not_df_lib
def test_read_parquet_unsupported_arg(memory_leak_check):
    """
    test that an error is raised when unsupported arg is passed.
    """
    from bodo.utils.typing import BodoError

    def test_impl():
        df = pd.read_parquet(
            "some_file.pq", invalid_arg="invalid", dtype_backend="pyarrow"
        )
        return df

    with pytest.raises(
        BodoError, match=r"read_parquet\(\) arguments {'invalid_arg'} not supported yet"
    ):
        bodo.jit(distributed=["df"])(test_impl)()


@pytest_mark_not_df_lib
def test_read_parquet_unsupported_storage_options_arg(memory_leak_check):
    """
    test that an error is raised when storage_options is passed for local FS
    """
    from bodo.utils.typing import BodoError

    def test_impl1():
        df = pd.read_parquet(
            "some_file.pq",
            storage_options={"invalid_arg": "invalid"},
            dtype_backend="pyarrow",
        )
        return df

    def test_impl2():
        df = pd.read_parquet(
            "some_file.pq",
            storage_options={"invalid_arg": "invalid", "anon": True},
            dtype_backend="pyarrow",
        )
        return df

    def test_impl3():
        df = pd.read_parquet(
            "some_file.pq", storage_options="invalid", dtype_backend="pyarrow"
        )  # type: ignore
        return df

    with pytest.raises(
        ValueError,
        match="ParquetReader: `storage_options` is not supported for protocol",
    ):
        bodo.jit(distributed=["df"])(test_impl1)()

    with pytest.raises(
        ValueError,
        match="ParquetReader: `storage_options` is not supported for protocol",
    ):
        bodo.jit(distributed=["df"])(test_impl2)()

    with pytest.raises(
        BodoError,
        match=re.escape(
            "read_parquet(): 'storage_options' must be a constant dictionary"
        ),
    ):
        bodo.jit(distributed=["df"])(test_impl3)()


@pytest_mark_not_df_lib
def test_read_parquet_non_bool_storage_options_anon(memory_leak_check):
    """
    test that an error is raised when non-boolean is passed in for anon in storage_options
    """

    def test_impl():
        df = pd.read_parquet(
            "some_file.pq", storage_options={"anon": "True"}, dtype_backend="pyarrow"
        )
        return df

    with pytest.raises(
        ValueError,
        match=re.escape(
            "ParquetReader: `storage_options` is not supported for protocol"
        ),
    ):
        bodo.jit(distributed=["df"])(test_impl)()


@pytest.mark.slow
@pytest.mark.skipif(
    bodo.dataframe_library_enabled,
    reason="[BSE-4773] Error calling get_parquet_dataset.",
)
def test_read_path_hive_partitions(datapath, memory_leak_check):
    filepath = datapath(os.path.join("hive-part-sample-pq", "data"))

    def test_impl():
        return pd.read_parquet(
            filepath, _bodo_use_hive=False, dtype_backend="pyarrow"
        ).count()

    exp_output = pd.Series(
        {
            "unit_serial_number": 200,
            "product_code": 200,
            "ingest_date": 0,
            "ingest_timestamp": 200,
            "fixture_id": 0,
            "units": 200,
            "effective_test_date": 200,
            "test_date": 200,
        }
    )

    check_func(
        test_impl,
        (),
        py_output=exp_output,
        is_out_distributed=False,
    )


@pytest.mark.slow
@pytest_mark_not_df_lib
def test_read_parquet_hive_partitions_type_clash(datapath):
    """
    Test error raise when parquet file path provided as constant but is invalid
    because the path uses the hive naming convention which leads to incorrectly
    inferred schema. In this case we use _bodo_use_hive=False to disable reading
    with the hive naming convention.
    """
    from bodo.utils.typing import BodoError

    filepath = datapath(os.path.join("hive-part-sample-pq", "data"))

    def test_impl():
        return pd.read_parquet(filepath, _bodo_use_hive=True, dtype_backend="pyarrow")

    with pytest.raises(
        BodoError,
        match="error from pyarrow: ArrowTypeError: Unable to merge: Field test_date has incompatible types:",
    ):
        bodo.jit(test_impl)()


# ---------------------------- Test Others ---------------------------- #
@pytest.mark.slow
def test_read_parquet_read_sanitize_colnames(datapath, memory_leak_check):
    """tests that parquet read works when reading a dataframe with column names
    that must be sanitized when generating the func_text"""

    def read_impl(path):
        return pd.read_parquet(path, dtype_backend="pyarrow")

    check_func(read_impl, (datapath("sanitization_test.pq"),))


def test_pq_columns(datapath):
    fname = datapath("example.parquet")

    def test_impl():
        return pd.read_parquet(
            fname, columns=["three", "five"], dtype_backend="pyarrow"
        )

    check_func(test_impl, (), only_seq=True, check_dtype=False)


def test_pq_str_with_nan_seq(datapath):
    fname = datapath("example.parquet")

    def test_impl():
        df = pd.read_parquet(fname, dtype_backend="pyarrow")
        A = df.five == "foo"
        return A.sum()

    check_func(test_impl, (), dist_test=False)


@pytest.mark.jit_dependency
def test_series_str_upper_lower_dce(datapath):
    """Tests Series.str.upper and Series.str.lower can be safely removed as dead code"""
    filename = datapath("example.parquet")

    def impl(filename):
        df = pd.read_parquet(filename, dtype_backend="pyarrow")
        df["two"] = df["two"].str.upper()
        df["five"] = df["five"].str.upper()
        return df.three

    check_func(impl, (filename,))

    if not bodo.dataframe_library_enabled:
        # Check that columns were pruned using verbose logging
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            bodo.jit(impl)(filename)
            check_logger_msg(stream, "Columns loaded ['three']")


@pytest_mark_not_df_lib
def test_pq_read(datapath):
    fname = datapath("kde.parquet")

    def test_impl():
        df = pd.read_parquet(fname, dtype_backend="pyarrow")
        X = df["points"]
        return X.sum()

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_almost_equal(bodo_func(), test_impl())
    assert count_array_REPs() == 0
    assert count_parfor_REPs() == 0


@pytest_mark_not_df_lib
def test_pq_read_global_str1(datapath):
    kde_file = datapath("kde.parquet")

    def test_impl():
        df = pd.read_parquet(kde_file, dtype_backend="pyarrow")
        X = df["points"]
        return X.sum()

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_almost_equal(bodo_func(), test_impl())
    assert count_array_REPs() == 0
    assert count_parfor_REPs() == 0


@pytest_mark_not_df_lib
def test_pq_read_freevar_str1(datapath):
    kde_file2 = datapath("kde.parquet")

    def test_impl():
        df = pd.read_parquet(kde_file2, dtype_backend="pyarrow")
        X = df["points"]
        return X.sum()

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_almost_equal(bodo_func(), test_impl())
    assert count_array_REPs() == 0
    assert count_parfor_REPs() == 0


@pytest_mark_not_df_lib
def test_pd_read_parquet(datapath):
    fname = datapath("kde.parquet")

    def test_impl():
        df = pd.read_parquet(fname, dtype_backend="pyarrow")
        X = df["points"]
        return X.sum()

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_almost_equal(bodo_func(), test_impl())
    assert count_array_REPs() == 0
    assert count_parfor_REPs() == 0


@pytest_mark_not_df_lib
def test_pq_str(datapath):
    fname = datapath("example.parquet")

    def test_impl():
        df = pd.read_parquet(fname, dtype_backend="pyarrow")
        A = df.two == "foo"
        return A.sum()

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_almost_equal(bodo_func(), test_impl())
    assert count_array_REPs() == 0
    assert count_parfor_REPs() == 0


@pytest_mark_not_df_lib
def test_pq_str_with_nan_par(datapath):
    fname = datapath("example.parquet")

    def test_impl():
        df = pd.read_parquet(fname, dtype_backend="pyarrow")
        A = df.five == "foo"
        return A.sum()

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_almost_equal(bodo_func(), test_impl())
    assert count_array_REPs() == 0
    assert count_parfor_REPs() == 0


@pytest_mark_not_df_lib
def test_pq_str_with_nan_par_multigroup(datapath):
    fname = datapath("example2.parquet")

    def test_impl():
        df = pd.read_parquet(fname, dtype_backend="pyarrow")
        A = df.five == "foo"
        return A.sum()

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_almost_equal(bodo_func(), test_impl())
    assert count_array_REPs() == 0
    assert count_parfor_REPs() == 0


@pytest_mark_not_df_lib
def test_pq_bool(datapath):
    fname = datapath("example.parquet")

    def test_impl():
        df = pd.read_parquet(fname, dtype_backend="pyarrow")
        return df.three.sum()

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_almost_equal(bodo_func(), test_impl())
    assert count_array_REPs() == 0
    assert count_parfor_REPs() == 0


@pytest_mark_not_df_lib
def test_pq_nan(datapath):
    fname = datapath("example.parquet")

    def test_impl():
        df = pd.read_parquet(fname, dtype_backend="pyarrow")
        return df.one.sum()

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_almost_equal(bodo_func(), test_impl())
    assert count_array_REPs() == 0
    assert count_parfor_REPs() == 0


@pytest_mark_not_df_lib
def test_pq_float_no_nan(datapath):
    fname = datapath("example.parquet")

    def test_impl():
        df = pd.read_parquet(fname, dtype_backend="pyarrow")
        return df.four.sum()

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_almost_equal(bodo_func(), test_impl())
    assert count_array_REPs() == 0
    assert count_parfor_REPs() == 0


@pytest.mark.slow
def test_read_dask_parquet(datapath, memory_leak_check):
    def test_impl(filename):
        df = pd.read_parquet(filename, dtype_backend="pyarrow")
        return df

    filename = datapath("dask_data.parquet")
    check_func(test_impl, (filename,))


@pytest_mark_not_df_lib
def test_pq_schema(datapath, memory_leak_check):
    fname = datapath("example.parquet")

    def impl(f):
        df = pd.read_parquet(f, dtype_backend="pyarrow")
        return df

    bodo_func = bodo.jit(
        distributed=False,
        locals={
            "df": {
                "one": bodo.types.float64[:],
                "two": bodo.types.string_array_type,
                "three": bodo.types.bool_[:],
                "four": bodo.types.float64[:],
                "five": bodo.types.string_array_type,
                "__index_level_0__": bodo.types.int64[:],
            }
        },
    )(impl)
    _test_equal(bodo_func(fname), impl(fname), check_dtype=False)


def test_unify_null_column(memory_leak_check):
    """
    Tests reading from parquet with a null column in the first
    file unifies properly.
    """
    with ensure_clean2("temp_parquet_test"):
        if bodo.get_rank() == 0:
            os.mkdir("temp_parquet_test")
            df1 = pd.DataFrame({"A": np.arange(10), "B": [None] * 10})
            df1.to_parquet("temp_parquet_test/f1.pq")
            df2 = pd.DataFrame({"A": np.arange(10, 16), "B": [None, "A"] * 3})
            df2.to_parquet("temp_parquet_test/f2.pq")
        bodo.barrier()

        def impl():
            return pd.read_parquet("temp_parquet_test", dtype_backend="pyarrow")

        # Pandas doesn't seem to be able to unify data.
        # TODO: Open a Pandas issue?
        py_output = pd.DataFrame(
            {"A": np.arange(16), "B": ([None] * 10) + ([None, "A"] * 3)}
        )

        check_func(impl, (), py_output=py_output)


@pytest.mark.slow
@pytest_mark_not_df_lib
def test_pq_cache_print(datapath, capsys, memory_leak_check):
    """Make sure FilenameType behaves like a regular value and not a literal when loaded
    from cache. This allows the file name value to be set correctly and not baked in.
    """

    @bodo.jit(cache=True)
    def f(fname):
        bodo.parallel_print(fname)
        return pd.read_parquet(fname, dtype_backend="pyarrow")

    fname1 = datapath("example.parquet")
    fname2 = datapath("example2.parquet")
    f(fname1)
    f(fname2)
    captured = capsys.readouterr()
    assert "example2.parquet" in captured.out


@pytest_mark_not_df_lib
def test_read_parquet_incorrect_s3_credentials(memory_leak_check):
    """test error raise when AWS credentials are incorrect for parquet
    file path passed by another bodo.jit function"""
    from bodo.utils.typing import BodoError

    filename = "s3://test-pq-2/item.pq"
    # Save default developer mode value
    default_mode = numba.core.config.DEVELOPER_MODE
    # Test as a user
    numba.core.config.DEVELOPER_MODE = 0

    with temp_env_override(
        {
            "AWS_ACCESS_KEY_ID": "bad_key_id",
            "AWS_SECRET_ACCESS_KEY": "bad_key",
            "AWS_SESSION_TOKEN": "bad_token",
        }
    ):

        @bodo.jit
        def read(filename):
            df = pd.read_parquet(filename, dtype_backend="pyarrow")
            return df

        # Test CallConstraint error
        def test_impl(filename):
            return read(filename)

        with pytest.raises(BodoError, match="No response body"):
            bodo.jit(test_impl)(filename)

        # Test ForceLiteralArg error
        def test_impl2(filename):
            df = pd.read_parquet(filename, dtype_backend="pyarrow")
            return df

        with pytest.raises(BodoError, match="No response body"):
            bodo.jit(test_impl2)(filename)

    # Reset developer mode
    numba.core.config.DEVELOPER_MODE = default_mode


@pytest_mark_not_df_lib
def test_pq_invalid_column_selection(datapath, memory_leak_check):
    """test error raise when selected column is not in file schema"""
    from bodo.utils.typing import BodoError

    def test_impl(fname):
        return pd.read_parquet(fname, columns=["C"], dtype_backend="pyarrow")

    with pytest.raises(BodoError, match="C not in Parquet file schema"):
        bodo.jit(test_impl)(datapath("nested_struct_example.pq"))


@pytest.mark.jit_dependency
def test_python_not_filter_pushdown(memory_leak_check):
    """Tests that the Pandas ~ operator works with filter pushdown."""
    with ensure_clean2("pq_data"):
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
            df = pd.read_parquet("pq_data", dtype_backend="pyarrow")
            df = df[~(df.B == 2)]
            return df["A"]

        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        # TODO [BSE-4776]: Check filter pushdown happens in DataFrame Lib.
        with set_logging_stream(logger, 1):
            # TODO: Fix index
            check_func(impl, ("pq_data",), reset_index=True)
            if not bodo.test_dataframe_library_enabled:
                check_logger_msg(
                    stream,
                    "Filter pushdown successfully performed. Moving filter step:",
                )
                check_logger_msg(stream, "Columns loaded ['A']")


@pytest.mark.jit_dependency
def test_filter_pushdown_string(datapath, memory_leak_check):
    def impl(path):
        df = pd.read_parquet(path, dtype_backend="pyarrow")
        val = "f"
        val += "Oo"
        val = val.lower()
        return df[df["two"] == val]

    path = datapath("parquet_data_nonascii1.parquet")
    stream = io.StringIO()
    logger = create_string_io_logger(stream)

    with set_logging_stream(logger, 2):
        check_func(impl, (path,), check_dtype=False, reset_index=True)

        if not bodo.test_dataframe_library_enabled:
            check_logger_msg(
                stream, "Filter pushdown successfully performed. Moving filter step:"
            )
            check_logger_msg(stream, "(ds.field('two') == ds.scalar(f0))")


@pytest.mark.skip("[BSE-3362] Filter pushdown with timestamp in Python not working")
@pytest.mark.jit_dependency
def test_filter_pushdown_timestamp(datapath, memory_leak_check):
    def impl(path):
        df = pd.read_parquet(path, dtype_backend="pyarrow")
        d = datetime.date(2015, 1, 1)
        d2 = d.replace(year=1992)
        return df[df["DT64"].dt.tz_convert(None) < pd.Timestamp(d2)]

    path = datapath("pandas_dt.pq")
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        check_func(impl, (path,), check_dtype=False, reset_index=True)
        check_logger_msg(
            stream, "Filter pushdown successfully performed. Moving filter step:"
        )
        check_logger_msg(stream, "(ds.field('DT64') < ds.scalar(f0))")


@pytest.mark.jit_dependency
def test_filter_pushdown_mutated_list(datapath, memory_leak_check):
    def impl(path):
        df = pd.read_parquet(path, dtype_backend="pyarrow")
        lst = [0]
        tup = (lst, 1)
        new_lst, val = tup
        new_lst.append(val)
        return df[df["A"].isin(new_lst)]

    path = datapath("int_nulls_single.pq")
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        check_func(impl, (path,), check_dtype=False, reset_index=True)
        if not bodo.test_dataframe_library_enabled:
            check_logger_msg(
                stream, "Filter pushdown successfully performed. Moving filter step:"
            )
            check_logger_msg(stream, "(ds.field('A').isin(f0))")


@pytest.mark.skipif(
    bodo.test_dataframe_library_enabled,
    reason="[BSE-4768] Enable filter pushdown tests.",
)
def test_filter_pushdown_tuple(datapath, memory_leak_check):
    def impl(path):
        df = pd.read_parquet(path, dtype_backend="pyarrow")
        x = 1 + 1
        y = x * 5
        tup = (3, 4)
        tup = (x, y)
        m = tup[0]
        n = tup[1]
        return df[df["A"].isin([m, n])]

    path = datapath("int_nulls_single.pq")
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        check_func(impl, (path,), check_dtype=False, reset_index=True)
        if not bodo.test_dataframe_library_enabled:
            check_logger_msg(
                stream, "Filter pushdown successfully performed. Moving filter step:"
            )
            check_logger_msg(stream, "(ds.field('A').isin(f0))")


@pytest.mark.jit_dependency
def test_filter_pushdown_tuple_function(datapath, memory_leak_check):
    @bodo.jit
    def comp(val):
        return val[0] + val[1][0]

    def impl(path):
        df = pd.read_parquet(path, dtype_backend="pyarrow")
        tup = (3 * 4, (7, "hello"))
        comp(tup)
        return df[~(df["A"] == tup[1][1])]

    path = datapath("small_strings.pq")
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        check_func(impl, (path,), check_dtype=False, reset_index=True)
        if not bodo.test_dataframe_library_enabled:
            check_logger_msg(
                stream, "Filter pushdown successfully performed. Moving filter step:"
            )
            check_logger_msg(stream, "~((ds.field('A') == ds.scalar(f0)))")


@pytest.mark.jit_dependency
def test_filter_pushdown_intermediate_comp_func(datapath, memory_leak_check):
    @bodo.jit
    def unused(x, test=None):
        return x

    @bodo.jit
    def used(x, test=None):
        return x - 1

    def impl(path):
        df = pd.read_parquet(path, dtype_backend="pyarrow")
        c = None
        unused(20, c)
        y = used(10, c)
        return df[df["A"] == y]

    path = datapath("int_nulls_single.pq")
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        check_func(impl, (path,), check_dtype=False, reset_index=True)
        if not bodo.test_dataframe_library_enabled:
            check_logger_msg(
                stream, "Filter pushdown successfully performed. Moving filter step:"
            )
            check_logger_msg(stream, "(ds.field('A') == ds.scalar(f0))")


@pytest.mark.skip("[BE-4498] Dictionaries are not passed-by-reference")
def test_filter_pushdown_dictionary(datapath, memory_leak_check):
    @bodo.jit
    def comp(d, v):
        d["test"] = v

    def impl(path):
        df = pd.read_parquet(path, dtype_backend="pyarrow")
        c = {"a": 1, "b": 2, "c": 3}
        comp(c, c["c"] + c["a"])
        return df[(df["A"] == c["test"] + c["a"]) | (df["A"] == c["c"])]

    path = datapath("int_nulls_single.pq")
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        check_func(impl, (path,), check_dtype=False, reset_index=True)
        check_logger_msg(
            stream, "Filter pushdown successfully performed. Moving filter step:"
        )
        check_logger_msg(
            stream,
            "(((ds.field('A') == ds.scalar(f0))) | ((ds.field('A') == ds.scalar(f1))))",
        )


@pytest.mark.skipif(bodo.test_dataframe_library_enabled, reason="BodoSQL codegen test.")
def test_batched_read_agg(datapath, memory_leak_check):
    """
    Test a simple use of batched Parquet reads by
    getting the max of a column
    """
    from bodo.io.arrow_reader import arrow_reader_del, read_arrow_next

    def impl(path):
        total_max = 0
        is_last_global = False
        reader = pd.read_parquet(
            path, _bodo_use_index=False, _bodo_chunksize=4096, dtype_backend="pyarrow"
        )  # type: ignore

        while not is_last_global:
            T1, is_last = read_arrow_next(reader, True)
            T2 = T1[pd.Series(bodo.hiframes.table.get_table_data(T1, 0)) > 10]
            # Perform more compute in between to see caching speedup
            local_max = pd.Series(bodo.hiframes.table.get_table_data(T2, 1)).max()
            total_max = max(total_max, local_max)

            is_last_global = bodo.libs.distributed_api.dist_reduce(
                is_last,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )

        arrow_reader_del(reader)
        return total_max

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(
            impl, (datapath("tpch-test_data/parquet/lineitem.pq"),), py_output=4000
        )
        check_logger_msg(stream, "Filter pushdown successfully performed")
        check_logger_msg(stream, "Columns loaded ['L_PARTKEY']")


@pytest.mark.skipif(bodo.test_dataframe_library_enabled, reason="BodoSQL codegen test.")
def test_batched_read_only_len(datapath, memory_leak_check):
    """
    Test shape pushdown with batched Snowflake reads
    """
    from bodo.io.arrow_reader import arrow_reader_del, read_arrow_next

    def impl(path):
        total_len = 0
        is_last_global = False
        reader = pd.read_parquet(
            path, _bodo_use_index=False, _bodo_chunksize=4096, dtype_backend="pyarrow"
        )  # type: ignore
        while not is_last_global:
            T1, is_last = read_arrow_next(reader, True)
            total_len += len(T1)

            is_last_global = bodo.libs.distributed_api.dist_reduce(
                is_last,
                np.int32(bodo.libs.distributed_api.Reduce_Type.Logical_And.value),
            )

        arrow_reader_del(reader)
        return total_len

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(
            impl,
            (datapath("tpch-test_data/parquet/lineitem.pq"),),
            py_output=120515,
        )
        check_logger_msg(stream, "Columns loaded []")
