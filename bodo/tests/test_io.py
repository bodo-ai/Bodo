# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Tests I/O for CSV, Parquet, HDF5, etc.
"""
import os
import random
import shutil
import subprocess
import sys
import unittest
from decimal import Decimal

import h5py
import numba
import numpy as np
import pandas as pd
import pytest
from numba.core.ir_utils import build_definitions, find_callname

import bodo
from bodo.tests.utils import (
    DeadcodeTestPipeline,
    SeriesOptTestPipeline,
    _get_dist_arg,
    _test_equal_guard,
    check_caching,
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
from bodo.utils.testing import ensure_clean
from bodo.utils.typing import BodoError
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
        bodo_func = bodo.jit(impl)
        # TODO: Parallel Test
        pd.testing.assert_frame_equal(bodo_func(), impl())
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
        bodo_func = bodo.jit(impl)
        # # TODO: Parallel Test
        pd.testing.assert_frame_equal(bodo_func(), impl())
        bodo.barrier()
    finally:
        if bodo.libs.distributed_api.get_rank() == 0:
            os.remove("test.pq")


def test_read_parquet_from_deltalake(memory_leak_check):
    def impl():
        return pd.read_parquet("bodo/tests/data/example_deltalake")

    py_output = pd.DataFrame({"value": [1, 1, 2, 3, 2, 3]})
    check_func(impl, (), py_output=py_output, check_dtype=False)


def test_pq_multiIdx_errcheck(memory_leak_check):
    """ Remove this test when multi index is supported for read_parquet """
    np.random.seed(0)

    def impl():
        df = pd.read_parquet("multi_idx_parquet.pq")

    try:
        if bodo.libs.distributed_api.get_rank() == 0:
            arrays = [
                ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
                ["one", "two", "one", "two", "one", "two", "one", "two"],
            ]
            tuples = list(zip(*arrays))
            idx = pd.MultiIndex.from_tuples(tuples, names=["first", "second"])
            df = pd.DataFrame(np.random.randn(8, 2), index=idx, columns=["A", "B"])
            df.to_parquet("multi_idx_parquet.pq")
        bodo.barrier()
        with pytest.raises(
            BodoError, match="read_parquet: MultiIndex not supported yet"
        ):
            bodo.jit(impl)()
        bodo.barrier()
    finally:
        if bodo.libs.distributed_api.get_rank() == 0:
            os.remove("multi_idx_parquet.pq")


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
    ],
)
@pytest.mark.parametrize("index_name", [None, "HELLO"])
# TODO: Add memory_leak_check when bugs are resolved.
def test_pq_write_metadata(df, index_name):
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

    bodo_func = bodo.jit(impl)
    pd.testing.assert_frame_equal(bodo_func(), impl())


@pytest.mark.skip("Needs datetime.date() support in parquet for latest arrow")
def test_pq_spark_date(datapath, memory_leak_check):
    fname = datapath("sdf_dt.pq")

    def impl():
        df = pd.read_parquet(fname)
        return pd.DataFrame({"DT64": df.DT64, "col2": df.DATE})

    bodo_func = bodo.jit(impl)
    pd.testing.assert_frame_equal(bodo_func(), impl())


# TODO: add memory_leak_check when tracking parquet allocs is possible
def test_pq_index(datapath):
    def test_impl(fname):
        return pd.read_parquet(fname)

    # passing function name as value to test value-based dispatch
    fname = datapath("index_test1.pq")
    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_frame_equal(bodo_func(fname), test_impl(fname), check_dtype=False)

    # string index
    fname = datapath("index_test2.pq")

    def test_impl2():
        return pd.read_parquet(fname)

    bodo_func = bodo.jit(test_impl2)
    pd.testing.assert_frame_equal(bodo_func(), test_impl2(), check_dtype=False)


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


# TODO: add memory_leak_check when tracking parquet allocs is possible
def test_pq_schema(datapath):
    fname = datapath("example.parquet")

    def impl(f):
        df = pd.read_parquet(f)
        return df

    bodo_func = bodo.jit(
        locals={
            "df": {
                "one": bodo.float64[:],
                "two": bodo.string_array_type,
                "three": bodo.bool_[:],
                "four": bodo.float64[:],
                "five": bodo.string_array_type,
            }
        }
    )(impl)
    pd.testing.assert_frame_equal(bodo_func(fname), impl(fname), check_dtype=False)


# TODO: Add memory_leak_check when bugs are resolved.
def test_pq_list_str(datapath):
    def test_impl(fname):
        return pd.read_parquet(fname)

    check_func(test_impl, (datapath("list_str_arr.pq"),))
    check_func(test_impl, (datapath("list_str_parts.pq"),))


# TODO: Add memory_leak_check when bugs are resolved.
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


# TODO: Add memory_leak_check when bugs are resolved.
def test_pq_array_item(datapath):
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


@pytest.mark.slow
def test_pq_unsupported_types(datapath, memory_leak_check):
    """test unsupported data types in unselected columns"""

    def test_impl(fname):
        return pd.read_parquet(fname, columns=["B"])

    check_func(test_impl, (datapath("nested_struct_example.pq"),))


def test_pq_invalid_column_selection(datapath, memory_leak_check):
    """test error raise when selected column is not in file schema"""

    def test_impl(fname):
        return pd.read_parquet(fname, columns=["C"])

    with pytest.raises(BodoError, match="C not in Parquet file schema"):
        bodo.jit(test_impl)(datapath("nested_struct_example.pq"))


def test_pq_decimal(datapath, memory_leak_check):
    def test_impl(fname):
        return pd.read_parquet(fname)

    check_func(test_impl, (datapath("decimal1.pq"),))


def test_pq_date32(datapath, memory_leak_check):
    """Test reading date32 values into datetime.date array"""

    def test_impl(fname):
        return pd.read_parquet(fname)

    check_func(test_impl, (datapath("date32_1.pq"),))


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

    bodo_func = numba.njit(pipeline_class=DeadcodeTestPipeline, parallel=True)(impl)
    pd.testing.assert_series_equal(bodo_func(), impl())
    fir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    read_csv_found = False
    # find CsvReader node and make sure it has only 1 column
    for stmt in fir.blocks[0].body:
        if isinstance(stmt, bodo.ir.csv_ext.CsvReader):
            read_csv_found = True
            assert len(stmt.df_colnames) == 1
            break
    assert read_csv_found

    if bodo.get_rank() == 0:
        subprocess.run(["gzip", "-k", "-f", fname])
    bodo.barrier()
    with ensure_clean(fname_gzipped):
        check_func(impl2, ())


@pytest.mark.smoke
def test_read_csv_cache(datapath, memory_leak_check):
    """
    test read_csv with cache=True
    """
    fname = datapath("csv_data1.csv")

    def impl():
        df = pd.read_csv(fname, names=["A", "B", "C", "D"], compression=None)
        return df.C

    py_out = impl()
    bodo_out1, bodo_out2 = check_caching(
        sys.modules[__name__], "test_read_csv_cache", impl, ()
    )
    pd.testing.assert_series_equal(py_out, bodo_out1)
    pd.testing.assert_series_equal(py_out, bodo_out2)


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


@pytest.mark.smoke
def test_read_parquet_cache(datapath, memory_leak_check):
    """
    test read_parquet with cache=True
    """

    def impl(fname):
        return pd.read_parquet(fname)

    fname = datapath("groupby3.pq")
    py_out = impl(fname)
    bodo_out1, bodo_out2 = check_caching(
        sys.modules[__name__], "test_read_parquet_cache", impl, (fname,)
    )
    pd.testing.assert_frame_equal(py_out, bodo_out1)
    pd.testing.assert_frame_equal(py_out, bodo_out2)


def test_read_parquet_cache_fname_arg(datapath, memory_leak_check):
    """
    test read_parquet with cache=True and passing different file name as
    argument to the Bodo function
    """

    def impl(fname):
        return pd.read_parquet(fname)

    fname1 = datapath("int_nulls_single.pq")
    fname2 = datapath("int_nulls_multi.pq")

    py_out = impl(fname1)
    py_out = py_out.astype({"A": "Int64"})
    bodo_out1, bodo_out2 = check_caching(
        sys.modules[__name__],
        "test_read_parquet_cache_fname_arg",
        impl,
        (fname1,),
        (fname2,),
    )
    pd.testing.assert_frame_equal(py_out, bodo_out1)
    pd.testing.assert_frame_equal(py_out, bodo_out2)


def test_read_csv_cache_fname_arg(datapath, memory_leak_check):
    """
    test read_csv with cache=True and passing different file name as
    argument to the Bodo function
    """

    def impl(fname):
        return pd.read_csv(fname)

    fname1 = datapath("example.csv")
    fname2 = datapath("example_multi.csv")  # directory of csv files

    py_out = impl(fname1)
    py_out = py_out.astype({"three": "boolean"})
    bodo_out1, bodo_out2 = check_caching(
        sys.modules[__name__],
        "test_read_csv_cache_fname_arg",
        impl,
        (fname1,),
        (fname2,),
    )
    pd.testing.assert_frame_equal(py_out, bodo_out1)
    pd.testing.assert_frame_equal(py_out, bodo_out2)


def test_read_json_cache_fname_arg(datapath, memory_leak_check):
    """
    test read_json with cache=True and passing different file name as
    argument to the Bodo function
    """

    def impl(fname):
        return pd.read_json(fname, orient="records", lines=True)

    fname1 = datapath("example.json")
    fname2 = datapath("example_single.json")  # directory with one json file

    py_out = impl(fname1)
    py_out = py_out.astype({"three": "boolean"})
    bodo_out1, bodo_out2 = check_caching(
        sys.modules[__name__],
        "test_read_json_cache_fname_arg",
        impl,
        (fname1,),
        (fname2,),
    )
    pd.testing.assert_frame_equal(py_out, bodo_out1)
    pd.testing.assert_frame_equal(py_out, bodo_out2)


# TODO: Add memory_leak_check when bugs are resolved.
def test_read_write_parquet():
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
            "Int8",
            "UInt8",
            "Int16",
            "UInt16",
            "Int32",
            # "UInt32",
            # pandas read_parquet has incorrect output with pyarrow 0.16.0 for UInt32
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
            df.index = [v ** 2 for v in range(num_elements)]
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

    def error_check2(df):
        df.to_parquet("out.parquet", compression="wrong")

    def error_check3(df):
        df.to_parquet("out.parquet", index=3)

    df = pd.DataFrame({"A": range(5)})

    with pytest.raises(BodoError, match="Unsupported compression"):
        bodo.jit(error_check2)(df)

    with pytest.raises(BodoError, match="index must be a constant bool or None"):
        bodo.jit(error_check3)(df)


def test_partition_cols():

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
            "B": ["A", "A", "B", "B", "A", "A", "B", "B"],
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

    import glob

    import pyarrow.parquet as pq

    try:
        to_test = [["A"], ["A", "B"], ["A", "B", "C"], ["D"], ["E"], ["D", "B"]]
        for part_cols in to_test:
            df_in = df.copy()
            if "D" not in part_cols:
                # TODO. can't write categorical to parquet currently because of metadata issue
                del df_in["D"]
            if bodo.get_rank() == 0:
                # use pandas to create all the required directories since we don't
                # create them yet
                df_in.to_parquet(TEST_DIR, partition_cols=part_cols)
                for f in glob.glob(TEST_DIR + "/**/*.parquet", recursive=True):
                    os.remove(f)
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


def test_read_dask_parquet(datapath):
    def test_impl(filename):
        df = pd.read_parquet(filename)
        return df

    filename = datapath("dask_data.parquet")
    check_func(test_impl, (filename,))


def test_read_partitions():
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

        check_func(impl, ("pq_data",))
        check_func(impl2, ("pq_data", "a"))
        # make sure the ParquetReader node has filters parameter set
        bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl2)
        bodo_func("pq_data", "a")
        _check_for_pq_reader_filters(bodo_func)
        check_func(impl3, ("pq_data", "a"))
        bodo.barrier()
    finally:
        if bodo.get_rank() == 0:
            shutil.rmtree("pq_data", ignore_errors=True)


def test_read_partitions2():
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
            return df[(df["c"].astype(int) > val) | (df["c"] == 2)]

        # TODO(ehsan): match Index
        check_func(impl1, ("pq_data", 3), reset_index=True)
        # make sure the ParquetReader node has filters parameter set
        bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl1)
        bodo_func("pq_data", 3)
        _check_for_pq_reader_filters(bodo_func)
        bodo.barrier()
    finally:
        if bodo.get_rank() == 0:
            shutil.rmtree("pq_data", ignore_errors=True)


def test_read_partitions_two_level():
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
            return df[(df["c"].astype(int) > val) | (df["part"] == "a")]

        # TODO(ehsan): match Index
        check_func(impl1, ("pq_data", 3), reset_index=True)
        # make sure the ParquetReader node has filters parameter set
        bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl1)
        bodo_func("pq_data", 3)
        _check_for_pq_reader_filters(bodo_func)
        bodo.barrier()
    finally:
        if bodo.get_rank() == 0:
            shutil.rmtree("pq_data", ignore_errors=True)


def test_read_partitions_datetime():
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

        check_func(impl1, ("pq_data", "2018-01-02", "2019-10-02"), reset_index=True)
        # make sure the ParquetReader node has filters parameter set
        bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl1)
        bodo_func("pq_data", "2018-01-02", "2019-10-02")
        _check_for_pq_reader_filters(bodo_func)
        bodo.barrier()
    finally:
        if bodo.get_rank() == 0:
            shutil.rmtree("pq_data", ignore_errors=True)


def test_read_partitions_large():
    """
    test reading and filtering partitioned parquet data with large number of partitions
    """

    try:
        if bodo.get_rank() == 0:
            I = pd.date_range("2018-01-03", "2020-12-05")
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
        _check_for_pq_reader_filters(bodo_func)
        bodo.barrier()
    finally:
        if bodo.get_rank() == 0:
            shutil.rmtree("pq_data", ignore_errors=True)


def _check_for_pq_reader_filters(bodo_func):
    """make sure ParquetReader node has filters set"""
    fir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    pq_read_found = False
    for stmt in fir.blocks[0].body:
        if isinstance(stmt, bodo.ir.parquet_ext.ParquetReader):
            assert stmt.filters is not None
            pq_read_found = True
            break
    assert pq_read_found


# TODO: Add memory_leak_check when bugs are resolved.
def test_write_parquet_empty_chunks():
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


# TODO: Add memory_leak_check when bugs are resolved.
def test_write_parquet_decimal(datapath):
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


# TODO: Add memory_leak_check when bugs are resolved.
def test_write_parquet_params():
    def write1(df, filename):
        df.to_parquet(compression="snappy", fname=filename)

    def write2(df, filename):
        df.to_parquet(fname=filename, index=None, compression="gzip")

    def write3(df, filename):
        df.to_parquet(fname=filename, index=True, compression="brotli")

    def write4(df, filename):
        df.to_parquet(fname=filename, index=False, compression=None)

    S1 = ["¡Y tú quién te crees?", "🐍⚡", "大处着眼，小处着手。"] * 4
    S2 = ["abc¡Y tú quién te crees?", "dd2🐍⚡", "22 大处着眼，小处着手。"] * 4
    df = pd.DataFrame({"A": S1, "B": S2})
    # set a numeric index (not range)
    df.index = [v ** 2 for v in range(len(df))]

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


def test_pseudo_exception(datapath, memory_leak_check):
    """Test removal of ForceLiteralArg message"""

    def test_csv(fname):
        df = pd.read_csv(fname)
        return df.clip()

    fname = datapath("csv_data1.csv")

    with pytest.raises(Exception) as excinfo:
        bodo.jit(test_csv)(fname)
    # Save full traceback as a string
    csv_track = excinfo.getrepr(style="native")
    assert "Pseudo-exception" not in str(csv_track)

    def test_pq(file_name):
        df = pd.read_parquet(file_name)
        return df.A.WRONG()

    fname = datapath("groupby3.pq")
    with pytest.raises(Exception) as excinfo:
        bodo.jit(test_pq)(fname)

    pq_track = excinfo.getrepr(style="native")
    assert "Pseudo-exception" not in str(pq_track)


@pytest.mark.smoke
def test_h5_read_seq(datapath, memory_leak_check):
    def test_impl(fname):
        f = h5py.File(fname, "r")
        X = f["points"][:]
        f.close()
        return X

    # passing function name as value to test value-based dispatch
    fname = datapath("lr.hdf5")
    bodo_func = bodo.jit(test_impl)
    np.testing.assert_allclose(bodo_func(fname), test_impl(fname))


def test_h5_read_const_infer_seq(datapath, memory_leak_check):
    fname = datapath("")

    def test_impl():
        p = fname + "lr"
        f = h5py.File(p + ".hdf5", "r")
        s = "po"
        X = f[s + "ints"][:]
        f.close()
        return X

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_allclose(bodo_func(), test_impl())


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


def test_h5_filter(datapath, memory_leak_check):
    fname = datapath("h5_test_filter.h5")

    def test_impl():
        f = h5py.File(fname, "r")
        b = np.arange(11) % 3 == 0
        X = f["test"][b, :, :, :]
        f.close()
        return X

    bodo_func = bodo.jit(locals={"X:return": "distributed"})(test_impl)
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

    bodo_func = bodo.jit(locals={"X:return": "distributed"})(test_impl)
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

    bodo_func = bodo.jit(locals={"X:return": "distributed"})(test_impl)
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
        bodo.jit(test_impl)(A, fname)
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

    bodo_func = bodo.jit(test_impl)
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

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_almost_equal(bodo_func(), test_impl())


def test_np_io6(datapath, memory_leak_check):
    # Test count optional argument huge
    fname = datapath("np_file1.dat")

    def test_impl():
        A = np.fromfile(fname, np.float64, count=100000000)
        return A

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_almost_equal(bodo_func(), test_impl())


def test_np_io7(datapath, memory_leak_check):
    # Test offset optional argument
    fname = datapath("np_file1.dat")

    def test_impl():
        A = np.fromfile(fname, np.float64, offset=10)
        return A

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_almost_equal(bodo_func(), test_impl())


def test_np_io8(datapath, memory_leak_check):
    # Test offset and count optional arguments
    fname = datapath("np_file1.dat")

    def test_impl():
        A = np.fromfile(fname, np.float64, offset=10, count=10)
        return A

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_almost_equal(bodo_func(), test_impl())


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


# TODO(Nick): Add a test for Parallel version with both offset and count.
@pytest.mark.slow
def test_np_io_sep_unsupported(datapath, memory_leak_check):
    fname = datapath("np_file1.dat")

    def test_impl():
        A = np.fromfile(fname, np.float64, sep=" ")
        return A.sum()

    bodo_func = bodo.jit(test_impl)
    with pytest.raises(
        BodoError, match=r"np.fromfile\(\): sep argument is not supported"
    ):
        # Test that we cannot swap the value of sep.
        bodo_func()


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

    bodo_func = bodo.jit(test_impl)
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
            dtype={"A": np.int, "B": np.float, "C": str, "D": np.int},
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
@pytest.mark.skip(
    reason="TODO: Fails in current Pandas as well. Unskip when we switch to Pandas 1.2.0"
)
def test_excel1(datapath, memory_leak_check):
    """Test pd.read_excel()"""

    def test_impl1(fname):
        return pd.read_excel(fname, parse_dates=[2])

    def test_impl2(fname):
        dtype = {
            "A": np.int,
            "B": np.float,
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
            "A": np.int,
            "B": np.float,
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
    check_func(test_impl3, (fname,), is_out_distributed=False)
    fname = datapath("data.xlsx")
    check_func(test_impl4, (fname, "Sheet1"), is_out_distributed=False)
    with pytest.raises(BodoError, match="both 'dtype' and 'names' should be provided"):
        bodo.jit(test_impl5)(fname)


def test_unsupported_error_checking(memory_leak_check):
    """make sure BodoError is raised for unsupported call"""
    # test an example I/O call
    def test_impl():
        return pd.read_spss("data.dat")

    with pytest.raises(BodoError, match="not supported yet"):
        bodo.jit(test_impl)()


# TODO: Add memory_leak_check when bugs are resolved.
def test_csv_invalid_path():
    """test error raise when CSV file path is invalid and the data types are provided
    explicitly (so that path check is done in C++ runtime).
    """

    def test_impl(fname):
        return pd.read_csv(fname, names=["A"], dtype={"A": np.int64})

    with pytest.raises(RuntimeError, match="invalid path"):
        bodo.jit(test_impl)("f.csv")


def test_csv_invalid_path_const(memory_leak_check):
    """test error raise when CSV file path provided as constant but is invalid."""

    def test_impl():
        return pd.read_csv("in_csv.csv")

    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        bodo.jit(test_impl)()


def test_csv_repeat_args(memory_leak_check):
    """
    test error raise when an untyped pass function provides an argument
    as both an arg and a kwarg
    """

    def test_impl():
        return pd.read_csv("csv_data1.csv", filepath_or_buffer="csv_data1.csv")

    with pytest.raises(
        BodoError,
        match=r"read_csv\(\) got multiple values for argument 'filepath_or_buffer'",
    ):
        bodo.jit(test_impl)()


def test_read_parquet_invalid_path(memory_leak_check):
    """test error raise when parquet file path is invalid in C++ code."""

    def test_impl():
        df = pd.read_parquet("I_dont_exist.pq")
        return df

    with pytest.raises(OSError, match="Passed non-file path"):
        bodo.jit(locals={"df": {"A": bodo.int64[:]}})(test_impl)()


def test_read_parquet_invalid_path_const(memory_leak_check):
    """test error raise when parquet file path provided as constant but is invalid."""

    def test_impl():
        return pd.read_parquet("I_dont_exist.pq")

    with pytest.raises(OSError, match="Passed non-file path"):
        bodo.jit(test_impl)()


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


def test_csv_dtype_unicode(memory_leak_check):
    """
    Tests read_csv using dtype="unicode"
    """

    def test_impl(fname):
        return pd.read_csv(fname, names=["A", "B", "C", "D"], dtype="unicode")

    fname = os.path.join("bodo", "tests", "data", "csv_data1.csv")
    check_func(test_impl, (fname,))


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

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(), test_impl(), check_dtype=False)

    def test_pq_str_with_nan_seq(self):
        fname = os.path.join("bodo", "tests", "data", "example.parquet")

        def test_impl():
            df = pd.read_parquet(fname)
            A = df.five.values == "foo"
            return A

        bodo_func = bodo.jit(test_impl)
        np.testing.assert_almost_equal(bodo_func(), test_impl())

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
                dtype={"A": np.int, "B": np.float, "C": np.float, "D": np.int},
            )

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(), test_impl())

    def test_csv_keys1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data1.csv")

        def test_impl():
            dtype = {"A": np.int, "B": np.float, "C": np.float, "D": np.int}
            return pd.read_csv(fname, names=list(dtype.keys()), dtype=dtype)

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(), test_impl())

    def test_csv_const_dtype1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data1.csv")

        def test_impl():
            dtype = {"A": "int", "B": "float64", "C": "float", "D": "int64"}
            return pd.read_csv(fname, names=list(dtype.keys()), dtype=dtype)

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(), test_impl())

    def test_csv_infer1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data_infer1.csv")

        def test_impl():
            return pd.read_csv(fname)

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(), test_impl(), check_dtype=False)

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

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(), test_impl(), check_dtype=False)

    def test_csv_skip1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data1.csv")

        def test_impl():
            return pd.read_csv(
                fname,
                names=["A", "B", "C", "D"],
                dtype={"A": np.int, "B": np.float, "C": np.float, "D": np.int},
                skiprows=2,
            )

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(), test_impl())

    def test_csv_infer_skip1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data_infer1.csv")

        def test_impl():
            return pd.read_csv(fname, skiprows=2)

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(), test_impl(), check_dtype=False)

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
                dtype={"A": np.int, "B": np.float, "C": np.float, "D": np.int},
            )
            return df.B.values

        bodo_func = bodo.jit(test_impl)
        np.testing.assert_array_equal(bodo_func(), test_impl())

    def test_csv_date1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data_date1.csv")

        def test_impl():
            return pd.read_csv(
                fname,
                names=["A", "B", "C", "D"],
                dtype={"A": np.int, "B": np.float, "C": str, "D": np.int},
                parse_dates=[2],
            )

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(), test_impl())

    def test_csv_str1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data_date1.csv")

        def test_impl():
            return pd.read_csv(
                fname,
                names=["A", "B", "C", "D"],
                dtype={"A": np.int, "B": np.float, "C": str, "D": np.int},
            )

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(), test_impl(), check_dtype=False)

    def test_csv_index_name1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data_date1.csv")

        def test_impl():
            return pd.read_csv(
                fname,
                names=["A", "B", "C", "D"],
                dtype={"A": np.int, "B": np.float, "C": str, "D": np.int},
                index_col="A",
            )

        bodo_func = bodo.jit(test_impl)
        pd_expected = test_impl()
        pd.testing.assert_frame_equal(bodo_func(), pd_expected, check_dtype=False)

    def test_csv_index_ind0(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data_date1.csv")

        def test_impl():
            return pd.read_csv(
                fname,
                names=["A", "B", "C", "D"],
                dtype={"A": np.int, "B": np.float, "C": str, "D": np.int},
                index_col=0,
            )

        bodo_func = bodo.jit(test_impl)
        pd_expected = test_impl()
        pd.testing.assert_frame_equal(bodo_func(), pd_expected, check_dtype=False)

    def test_csv_index_ind1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data_date1.csv")

        def test_impl():
            return pd.read_csv(
                fname,
                names=["A", "B", "C", "D"],
                dtype={"A": np.int, "B": np.float, "C": str, "D": np.int},
                index_col=1,
            )

        bodo_func = bodo.jit(test_impl)
        pd_expected = test_impl()
        pd.testing.assert_frame_equal(bodo_func(), pd_expected, check_dtype=False)

    def test_csv_parallel1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data1.csv")

        def test_impl():
            df = pd.read_csv(
                fname,
                names=["A", "B", "C", "D"],
                dtype={"A": np.int, "B": np.float, "C": np.float, "D": np.int},
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
                dtype={"A": np.int, "B": np.float, "C": str, "D": np.int},
            )
            return (df.A.sum(), df.B.sum(), (df.C == "1966-11-13").sum(), df.D.sum())

        bodo_func = bodo.jit(locals={"df:return": "distributed"})(test_impl)
        self.assertEqual(bodo_func(), test_impl())

    def test_csv_usecols1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data1.csv")

        def test_impl():
            return pd.read_csv(fname, names=["C"], dtype={"C": np.float}, usecols=[2])

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(), test_impl())

    def test_csv_usecols2(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data1.csv")

        def test_impl():
            return pd.read_csv(fname, names=["B", "C"], usecols=[1, 2])

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(), test_impl())

    def test_csv_usecols3(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data2.csv")

        def test_impl():
            return pd.read_csv(fname, sep="|", names=["B", "C"], usecols=[1, 2])

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(), test_impl())

    def test_csv_cat2(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data_cat1.csv")

        def test_impl():
            ct_dtype = pd.CategoricalDtype(["A", "B", "C", "D"])
            df = pd.read_csv(
                fname,
                names=["C1", "C2", "C3"],
                dtype={"C1": np.int, "C2": ct_dtype, "C3": str},
            )
            return df

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(), test_impl(), check_dtype=False)

    def test_csv_single_dtype1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data_dtype1.csv")

        def test_impl():
            df = pd.read_csv(fname, names=["C1", "C2"], dtype=np.float64)
            return df

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(), test_impl())

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


if __name__ == "__main__":
    unittest.main()
