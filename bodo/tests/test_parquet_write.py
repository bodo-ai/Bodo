from __future__ import annotations

import os
import random
import shutil
import traceback
from decimal import Decimal
from urllib.parse import urlparse

import numba  # noqa TID253
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import pytz
from mpi4py import MPI

import bodo
from bodo.tests.conftest import DataPath
from bodo.tests.utils import (
    _get_dist_arg,
    _test_equal_guard,
    cast_dt64_to_ns,
    check_func,
    gen_random_arrow_array_struct_int,
    gen_random_arrow_list_list_int,
    gen_random_arrow_struct_struct,
    pytest_mark_one_rank,
)
from bodo.utils.testing import ensure_clean2, ensure_clean_dir

pytestmark = pytest.mark.parquet


def check_write_func(
    fn,
    df: pd.DataFrame,
    folder_path: str,
    file_id: str,
    check_index: list[str] | None = None,
    pandas_fn=None,
):
    import bodo.decorators  # isort:skip # noqa
    from bodo.spawn.utils import run_rank0

    DISTRIBUTIONS = {
        "sequential": [lambda x, *args: x, [], {}],
        "1d-distributed": [
            _get_dist_arg,
            [False],
            {"all_args_distributed_block": True},
        ],
        "1d-distributed-varlength": [
            _get_dist_arg,
            [False, True],
            {"all_args_distributed_varlength": True},
        ],
    }

    bodo_file_path = os.path.join(folder_path, f"bodo_{file_id}.pq")
    pandas_file_path = os.path.join(folder_path, f"pandas_{file_id}.pq")
    pandas_fn = pandas_fn if pandas_fn else fn

    for dist_func, args, kwargs in DISTRIBUTIONS.values():
        with ensure_clean2(bodo_file_path), ensure_clean2(pandas_file_path):
            write_jit = bodo.jit(fn, **kwargs)
            write_jit(dist_func(df, *args), bodo_file_path)
            run_rank0(pandas_fn)(df, pandas_file_path)  # Pandas version
            bodo.barrier()

            # Use fsspec for all reads
            df_bodo = pd.read_parquet(
                bodo_file_path, storage_options={}, dtype_backend="pyarrow"
            )
            df_pandas = pd.read_parquet(
                pandas_file_path, storage_options={}, dtype_backend="pyarrow"
            )
            df_pandas_test = pd.read_parquet(
                bodo_file_path, storage_options={}, dtype_backend="pyarrow"
            )
            pd.testing.assert_frame_equal(
                df_bodo,
                df_pandas,
                check_column_type=True,
                check_dtype=False,
                check_index_type=False,
            )
            pd.testing.assert_frame_equal(
                df_bodo,
                df_pandas_test,
                check_column_type=True,
                check_dtype=False,
                check_index_type=False,
            )

            if check_index is not None:
                pandas_meta = pq.read_schema(pandas_file_path)
                bodo_fps = (
                    list(os.walk(bodo_file_path))
                    if os.path.isdir(bodo_file_path)
                    else [("", [], [bodo_file_path])]
                )
                for prefix, _, file_names in bodo_fps:
                    for file in file_names:
                        bodo_meta = pq.read_schema(os.path.join(prefix, file))
                        index_cols = bodo_meta.pandas_metadata["index_columns"]
                        assert set(check_index).issubset(index_cols)
                        assert (
                            index_cols == pandas_meta.pandas_metadata["index_columns"]
                        )


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(  # nested strings
            {
                "A": [
                    ["a", None, "cde"],
                    None,
                    ["random"],
                    ["pad"],
                    ["shrd", "lu"],
                    ["je", "op", "ardy"],
                    ["double", "je", "op", "ardy"],
                    None,
                ]
            }
        ),
        pd.DataFrame(  # nested arrays
            {
                "A": [
                    [[1], [2], None],
                    [[3], None, [4]],
                ]
                * 10
            }
        ),
        pd.DataFrame(  # struct arrays
            {
                "A": [
                    {
                        "X": "D",
                        "Y": [4.0, 6.0],
                        "Z": [[1], None],
                        "W": {"A": 1, "B": ""},
                    },
                    {
                        "X": "VFD",
                        "Y": [1.2],
                        "Z": [[], [3, 1]],
                        "W": {"A": 1, "B": "AA"},
                    },
                ]
                * 10
            }
        ),
        # TODO BSE-1317
        # pd.DataFrame( # map arrays
        #        {
        #            "a": np.arange(2),
        #             "b": [{'1': 1.4, '2': 3.1}, {'3':7.0, '4':8.0}],
        #        }
        # )
    ],
    ids=(
        "nested_strings",
        "nested_arrays",
        "struct_arrays",
        # TODO BSE-1317
        # "map_arrays"
    ),
)
def test_semi_structured_data(df, tmp_path):
    comm = MPI.COMM_WORLD
    tmp_path = comm.bcast(tmp_path if bodo.get_rank() == 0 else None, root=0)
    check_write_func(
        lambda df, path: df.to_parquet(path),
        df,
        str(tmp_path),
        "semi_structured_data",
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
                    cast_dt64_to_ns(
                        pd.read_parquet("bodo_metadatatest.pq", dtype_backend="pyarrow")
                    ),
                    cast_dt64_to_ns(
                        pd.read_parquet(
                            "pandas_metadatatest.pq", dtype_backend="pyarrow"
                        )
                    ),
                    check_column_type=False,
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
            try:
                os.remove("bodo_metadatatest.pq")
                os.remove("pandas_metadatatest.pq")
            except FileNotFoundError:
                pass
        else:
            shutil.rmtree("bodo_metadatatest.pq", ignore_errors=True)


def gen_dataframe(num_elements, write_index):
    """Generates a dataframe for Parquet read/write test below"""
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
        "Float32",
        "Float64",
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
            df[col_name] = pd.array(data, pd.ArrowDtype(pa.large_string()))
        elif dtype == "Binary":
            # missing values every 5 elements
            data = [
                str(x).encode() * 3 if x % 5 != 0 else None for x in range(num_elements)
            ]
            df[col_name] = pd.array(data, pd.ArrowDtype(pa.large_binary()))
        elif dtype == "bool":
            data = [True if x % 2 == 0 else False for x in range(num_elements)]
            df[col_name] = np.array(data, dtype="bool")
        elif dtype.startswith("Int") or dtype.startswith("UInt"):
            # missing values every 5 elements
            data = [x if x % 5 != 0 else np.nan for x in range(num_elements)]
            df[col_name] = pd.Series(data, dtype=dtype)
        elif dtype.startswith("Float"):
            # missing values every 5 elements
            data = [x if x % 5 != 0 else None for x in range(num_elements)]
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
            df[col_name] = pd.Series(
                data, dtype=pd.ArrowDtype(pa.decimal128(precision=38, scale=18))
            )
        elif dtype == "Date":
            dates = pd.Series(
                pd.date_range(
                    start="1998-04-24", end="1998-04-29", periods=num_elements
                )
            )
            df[col_name] = pd.array(dates.dt.date, pd.ArrowDtype(pa.date32()))
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
            df[col_name] = dates.astype("datetime64[ns]")
            df._datetime_col = col_name
        elif dtype == "nested_arrow0":
            # Disabling Nones because currently they very easily induce
            # typing errors during unboxing for nested lists.
            # _infer_ndarray_obj_dtype in boxing.py needs to be made more robust.
            # TODO: include Nones
            df[col_name] = pd.Series(gen_random_arrow_list_list_int(2, 0, num_elements))
        elif dtype == "nested_arrow1":
            # NOTE: converting to object since Pandas as of 2.0.3 fails with Arrow types
            df[col_name] = pd.Series(
                gen_random_arrow_array_struct_int(10, num_elements), dtype=object
            )
        elif dtype == "nested_arrow2":
            # TODO: Include following types when they are supported in PYARROW:
            # We cannot read this dataframe in bodo. Fails at unboxing.
            # df_bug1 = pd.DataFrame({"X": gen_random_arrow_list_list_double(2, -0.1, n)})
            # This dataframe can be written by the code. However we cannot read
            # due to a limitation in pyarrow
            # df_bug2 = pd.DataFrame({"X": gen_random_arrow_array_struct_list_int(10, n)})
            # NOTE: converting to object since Pandas as of 2.0.3 fails with Arrow types
            df[col_name] = pd.Series(
                gen_random_arrow_struct_struct(10, num_elements), dtype=object
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
        ]  # type: ignore
    elif write_index == "numeric":
        # set a numeric index (not range)
        df.index = [v**2 for v in range(num_elements)]  # type: ignore
    return df


@pytest.mark.slow
def test_read_write_parquet(memory_leak_check):
    from bodo.tests.utils_jit import reduce_sum
    from bodo.utils.typing import BodoError

    def write(df, filename):
        df.to_parquet(filename)

    def read():
        return pd.read_parquet("_test_io___.pq", dtype_backend="pyarrow")

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
                    bodo_write = bodo.jit(write, distributed=False)
                    bodo_write(df, bodo_pq_filename)
                elif mode == "1d-distributed":
                    bodo_write = bodo.jit(write, all_args_distributed_block=True)
                    bodo_write(_get_dist_arg(df, False), bodo_pq_filename)
                elif mode == "1d-distributed-varlength":
                    bodo_write = bodo.jit(write, all_args_distributed_varlength=True)
                    bodo_write(_get_dist_arg(df, False, True), bodo_pq_filename)
                bodo.barrier()
                # read both files with pandas
                df1 = cast_dt64_to_ns(
                    pd.read_parquet(pandas_pq_filename, dtype_backend="pyarrow")
                )
                df2 = cast_dt64_to_ns(
                    pd.read_parquet(bodo_pq_filename, dtype_backend="pyarrow")
                )

                # to test equality, we have to coerce datetime columns to ms
                # because pandas writes to parquet as datetime64[ms]
                df[df._datetime_col] = df[df._datetime_col].dt.floor("ms")
                # need to coerce column from bodo-generated parquet to ms (note
                # that the column has us precision because Arrow cpp converts
                # nanoseconds to microseconds when writing to parquet version 1)
                df2[df._datetime_col] = df2[df._datetime_col].dt.floor("ms")

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
                    df1, df2, sort_output=False, check_names=True, check_dtype=False
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
            df[df._datetime_col] = df[df._datetime_col].dt.floor("ms")
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
def test_write_parquet_empty_chunks(memory_leak_check):
    """Here we check that our to_parquet output in distributed mode
    (directory of parquet files) can be read by pandas even when some
    processes have empty chunks"""

    def f(n, write_filename):
        df = pd.DataFrame({"A": np.arange(n, dtype=np.int64)})
        df.to_parquet(write_filename)

    write_filename = "test__empty_chunks.pq"
    n = 1  # make dataframe of length 1 so that rest of processes have empty chunk
    try:
        bodo.jit(f)(n, write_filename)
        bodo.barrier()
        if bodo.get_rank() == 0:
            df = pd.read_parquet(write_filename, dtype_backend="pyarrow")
            pd.testing.assert_frame_equal(
                df,
                pd.DataFrame({"A": np.arange(n, dtype=np.int64)}),
                check_column_type=False,
                check_index_type=False,
                check_dtype=False,
            )
    finally:
        if bodo.get_rank() == 0:
            shutil.rmtree(write_filename)


@pytest.mark.slow
def test_write_parquet_decimal(datapath: DataPath, memory_leak_check):
    """Here we check that we can write the data read from decimal1.pq directory
    (has columns that use a precision and scale different from our default).
    See test_write_parquet above for main parquet write decimal test"""
    fname = datapath("decimal1.pq")

    def write(read_path, write_filename):
        df = pd.read_parquet(read_path, dtype_backend="pyarrow")
        df.to_parquet(write_filename)

    write_filename = "test__write_decimal1.pq"
    try:
        bodo.jit(write)(fname, write_filename)
        bodo.barrier()
        if bodo.get_rank() == 0:
            df1 = pd.read_parquet(fname, dtype_backend="pyarrow")
            df2 = pd.read_parquet(write_filename, dtype_backend="pyarrow")
            pd.testing.assert_frame_equal(
                df1,
                df2,
                check_column_type=False,
                check_index_type=False,
                check_dtype=False,
            )
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
    df.index = [v**2 for v in range(len(df))]  # type: ignore

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
                df_a = pd.read_parquet(pd_fname, dtype_backend="pyarrow")
                df_b = pd.read_parquet(bodo_fname, dtype_backend="pyarrow")
                pd.testing.assert_frame_equal(df_a, df_b, check_column_type=False)
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
    from bodo.tests.utils_jit import reduce_sum

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
            result = pd.read_parquet("arr_dict_test.pq", dtype_backend="pyarrow")
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
                assert schema.field(c).type == expected_dtype, (
                    f"Field '{c}' has an incorrect type"
                )
        except Exception:
            passed = 0
        finally:
            shutil.rmtree("arr_dict_test.pq")
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), (
        "to_parquet output doesn't match expected pandas output"
    )


def test_write_parquet_dict_table(memory_leak_check):
    """
    Test to_parquet when dictionary arrays are used
    in a DataFrame containing a table representation.

    To do this consistently we load heavily compressed data
    from parquet.
    """
    from bodo.tests.utils_jit import reduce_sum

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
        df = pd.read_parquet("dummy_source.pq", dtype_backend="pyarrow")
        df.to_parquet("arr_dict_test.pq", index=False)

    impl()
    bodo.barrier()
    passed = 1
    if bodo.get_rank() == 0:
        try:
            # Check the output.
            result = pd.read_parquet("arr_dict_test.pq", dtype_backend="pyarrow")
            py_output = pd.read_parquet("dummy_source.pq", dtype_backend="pyarrow")
            passed = _test_equal_guard(
                result,
                py_output,
            )

            # Check the schema to ensure its stored as string
            bodo_table = pq.read_table("arr_dict_test.pq")
            schema = bodo_table.schema
            expected_dtype = pa.string()
            for c in py_output.columns:
                assert schema.field(c).type == expected_dtype, (
                    f"Field '{c}' has an incorrect type"
                )
        except Exception:
            passed = 0
        finally:
            shutil.rmtree("arr_dict_test.pq")
            os.remove("dummy_source.pq")
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), (
        "to_parquet output doesn't match expected pandas output"
    )


@pytest_mark_one_rank
def test_write_parquet_row_group_size(memory_leak_check):
    """Test df.to_parquet(..., row_group_size=n)"""
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


def test_tz_to_parquet(memory_leak_check):
    """
    Tests loading and returning an array with timezone information
    from Arrow. This tests both `to_parquet` and `array_to_info` support.
    """
    from bodo.tests.utils_jit import reduce_sum

    py_output = pd.DataFrame(
        {
            "A": pd.date_range(
                "2018-04-09", periods=50, freq="2D1h", tz="America/Los_Angeles"
            ),
            "B": pd.date_range("2018-04-09", periods=50, freq="2D1h"),
            "C": pd.date_range("2018-04-09", periods=50, freq="2D1h", tz="Poland"),
            "D": pd.date_range(
                "2018-04-09", periods=50, freq="2D1h", tz=pytz.FixedOffset(240)
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
            result = pd.read_parquet(output_filename, dtype_backend="pyarrow")
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
                assert col_metadata["pandas_type"] == "datetimetz", (
                    f"incorrect pandas_type metadata for column {col_name}"
                )
                assert col_metadata["numpy_type"] == "datetime64[ns]", (
                    f"incorrect numpy_type metadata for column {col_name}"
                )
                metadata_field = col_metadata["metadata"]
                assert isinstance(metadata_field, dict), (
                    f"incorrect metadata field for column {col_name}"
                )
                fields = list(metadata_field.items())
                assert fields == [
                    ("timezone", result.dtypes.iloc[col_index].pyarrow_dtype.tz)
                ], f"incorrect metadata field for column {col_name}"
        except Exception:
            passed = 0
        finally:
            shutil.rmtree(output_filename)
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), "Output doesn't match Pandas data"


def test_to_pq_nulls_in_dict(memory_leak_check):
    """
    Test that parquet write works even when there are nulls in
    the dictionary of a dictionary encoded array. Arrow doesn't
    support this, so we drop the NAs from all dictionaries ourselves
    before the write step.
    See [BE-4331](https://bodo.atlassian.net/browse/BE-4331)
    for more context.
    We also explicitly test the table-format case since the codegen
    for it is slightly different.
    """
    from bodo.tests.utils_jit import reduce_sum

    S = pa.DictionaryArray.from_arrays(
        np.array([0, 2, 1, 0, 1, 3, 0, 1, 3, 3, 2, 0], dtype=np.int32),
        pd.Series(["B", None, "A", None]),
    )
    A = np.arange(12)

    def test_output(func, S, A, part_cols, py_output):
        with ensure_clean2("test_to_pq_nulls_in_dict.pq"):
            func(_get_dist_arg(S), _get_dist_arg(A), part_cols)
            bodo.barrier()
            passed = 1
            if bodo.get_rank() == 0:
                try:
                    df = pd.read_parquet(
                        "test_to_pq_nulls_in_dict.pq", dtype_backend="pyarrow"
                    )
                    if part_cols:
                        for col in part_cols:
                            df[col] = df[col].astype(py_output[col].dtype)
                            if col == "S":
                                df[col] = df[col].where(df[col] != "null", None)
                        # Need to re-arrange the column order since partition columns are put at the end after reading
                        df = df[list(py_output.columns)]
                    passed = _test_equal_guard(
                        df, py_output, sort_output=True, reset_index=True
                    )
                except Exception as e:
                    print("".join(traceback.format_exception(None, e, e.__traceback__)))
                    passed = 0
            n_passed = reduce_sum(passed)
            assert n_passed == bodo.get_size()

    @bodo.jit(distributed=["S", "A"])
    def impl(S, A, part_cols):
        S = pd.Series(S)
        df = pd.DataFrame({"S": S, "A": A})
        df.to_parquet("test_to_pq_nulls_in_dict.pq", partition_cols=part_cols)
        return df

    @bodo.jit(distributed=["S", "A"])
    def impl_table_format(S, A, part_cols):
        S = pd.Series(S)
        df = pd.DataFrame({"S": S, "A": A})
        # Force dataframe to use table format
        df = bodo.hiframes.pd_dataframe_ext._tuple_to_table_format_decoded(df)
        df.to_parquet("test_to_pq_nulls_in_dict.pq", partition_cols=part_cols)
        return df

    # Regular write
    py_output = pd.DataFrame({"S": S, "A": A})
    test_output(impl, S, A, None, py_output)
    test_output(impl_table_format, S, A, None, py_output)

    # Test that partitioned write works as well

    # Partition on the dict-encoded column
    test_output(impl, S, A, ["S"], py_output)
    test_output(impl_table_format, S, A, ["S"], py_output)

    # Partition on the non-dict-encoded column
    test_output(impl, S, A, ["A"], py_output)
    test_output(impl_table_format, S, A, ["A"], py_output)

    # Use the same dict-encoded array for both columns and
    # partition on one of it (should catch any refcount bugs)
    py_output = pd.DataFrame({"S": S, "A": S})
    test_output(impl, S, S, ["S"], py_output)
    test_output(impl_table_format, S, S, ["S"], py_output)


def test_streaming_parquet_write(memory_leak_check):
    """Test streaming Parquet write"""
    from bodo.io.stream_parquet_write import (
        parquet_writer_append_table,
        parquet_writer_init,
    )
    from bodo.tests.utils_jit import reduce_sum

    df = gen_dataframe(80, None)

    col_meta = bodo.utils.typing.ColNamesMetaType(tuple(df.columns))
    batch_size = 11

    @bodo.jit(distributed=["df"])
    def impl(df, path):
        writer = parquet_writer_init(-1, path, "snappy", -1, "part-", "")
        all_is_last = False
        iter_val = 0
        T1 = bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df)
        while not all_is_last:
            table = bodo.hiframes.table.table_local_filter(
                T1,
                slice(
                    (iter_val * batch_size),
                    ((iter_val + 1) * batch_size),
                ),
            )
            is_last = (iter_val * batch_size) >= len(df)
            all_is_last = parquet_writer_append_table(
                writer, table, col_meta, is_last, iter_val
            )
            iter_val += 1

    write_filename = "test_stream_pq_write.pq"
    orig_chunk_size = bodo.io.stream_parquet_write.PARQUET_WRITE_CHUNK_SIZE
    try:
        # set chunk size to a small number to make sure multiple iterations write files
        bodo.io.stream_parquet_write.PARQUET_WRITE_CHUNK_SIZE = 300
        impl(_get_dist_arg(df, False), write_filename)
        bodo.barrier()
        df2 = cast_dt64_to_ns(pd.read_parquet(write_filename, dtype_backend="pyarrow"))
        # to test equality, we have to coerce datetime columns to ms
        # because pandas writes to parquet as datetime64[ms]
        df[df._datetime_col] = df[df._datetime_col].dt.floor("ms")
        # need to coerce column from bodo-generated parquet to ms (note
        # that the column has us precision because Arrow cpp converts
        # nanoseconds to microseconds when writing to parquet version 1)
        df2[df._datetime_col] = df2[df._datetime_col].dt.floor("ms")

        # Sort data since streaming writes chunks out of order.
        # Not using sort_output since it fails with nested columns (first column is enough).
        df2 = df2.sort_values(by="col_0")
        passed = _test_equal_guard(
            df,
            df2,
            sort_output=False,
            check_names=True,
            check_dtype=False,
            reset_index=True,
        )
        n_passed = reduce_sum(passed)
        assert n_passed == bodo.get_size()
    finally:
        bodo.io.stream_parquet_write.PARQUET_WRITE_CHUNK_SIZE = orig_chunk_size
        if bodo.get_rank() == 0:
            shutil.rmtree(write_filename)
        bodo.barrier()


def test_streaming_parquet_write_rep(memory_leak_check):
    """Test streaming Parquet write with replicated input"""
    from bodo.io.stream_parquet_write import (
        parquet_writer_append_table,
        parquet_writer_init,
    )
    from bodo.tests.utils_jit import reduce_sum

    df = gen_dataframe(80, None)

    col_meta = bodo.utils.typing.ColNamesMetaType(tuple(df.columns))
    batch_size = 11

    @bodo.jit(distributed=False)
    def impl(df, path):
        writer = parquet_writer_init(-1, path, "snappy", -1, "part-", "")
        all_is_last = False
        iter_val = 0
        T1 = bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data(df)
        while not all_is_last:
            table = bodo.hiframes.table.table_local_filter(
                T1,
                slice(
                    (iter_val * batch_size),
                    ((iter_val + 1) * batch_size),
                ),
            )
            is_last = (iter_val * batch_size) >= len(df)
            all_is_last = parquet_writer_append_table(
                writer, table, col_meta, is_last, iter_val
            )
            iter_val += 1

    write_filename = "test_stream_pq_write.pq"
    orig_chunk_size = bodo.io.stream_parquet_write.PARQUET_WRITE_CHUNK_SIZE
    try:
        # set chunk size to a small number to make sure multiple iterations write files
        bodo.io.stream_parquet_write.PARQUET_WRITE_CHUNK_SIZE = 300
        impl(df, write_filename)
        bodo.barrier()
        df2 = cast_dt64_to_ns(pd.read_parquet(write_filename, dtype_backend="pyarrow"))
        # to test equality, we have to coerce datetime columns to ms
        # because pandas writes to parquet as datetime64[ms]
        df[df._datetime_col] = df[df._datetime_col].dt.floor("ms")
        # need to coerce column from bodo-generated parquet to ms (note
        # that the column has us precision because Arrow cpp converts
        # nanoseconds to microseconds when writing to parquet version 1)
        df2[df._datetime_col] = df2[df._datetime_col].dt.floor("ms")

        # Sort data since streaming writes chunks out of order.
        # Not using sort_output since it fails with nested columns (first column is enough).
        df2 = df2.sort_values(by="col_0")
        passed = _test_equal_guard(
            df,
            df2,
            sort_output=False,
            check_names=True,
            check_dtype=False,
            reset_index=True,
        )
        n_passed = reduce_sum(passed)
        assert n_passed == bodo.get_size()
    finally:
        bodo.io.stream_parquet_write.PARQUET_WRITE_CHUNK_SIZE = orig_chunk_size
        if bodo.get_rank() == 0:
            shutil.rmtree(write_filename)
        bodo.barrier()


def test_to_pq_multiIdx(tmp_path, memory_leak_check):
    """Test to_parquet with MultiIndexType"""
    np.random.seed(0)
    comm = MPI.COMM_WORLD
    tmp_path = comm.bcast(tmp_path if bodo.get_rank() == 0 else None, root=0)

    arrays = [
        ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
        ["one", "two", "one", "two", "one", "two", "one", "two"],
        [1, 2, 2, 1] * 2,
    ]
    tuples = list(zip(*arrays))
    idx = pd.MultiIndex.from_tuples(tuples, names=["first", "second", "third"])
    df = pd.DataFrame(np.random.randn(8, 2), index=idx, columns=["A", "B"])

    check_write_func(
        lambda df, fname: df.to_parquet(fname),
        df,
        str(tmp_path),
        "multi_idx_parquet",
        check_index=["first", "second", "third"],
    )


def test_to_pq_multiIdx_no_name(tmp_path, memory_leak_check):
    """Test to_parquet with MultiIndexType with no name at 1 level"""
    np.random.seed(0)
    comm = MPI.COMM_WORLD
    tmp_path = comm.bcast(tmp_path if bodo.get_rank() == 0 else None, root=0)

    arrays = [
        ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
        ["one", "two", "one", "two", "one", "two", "one", "two"],
        [1, 2, 2, 1] * 2,
    ]
    tuples = list(zip(*arrays))
    idx = pd.MultiIndex.from_tuples(tuples, names=[None, None, "nums"])
    df = pd.DataFrame(np.random.randn(8, 2), index=idx, columns=["A", "B"])

    check_write_func(
        lambda df, fname: df.to_parquet(fname),
        df,
        str(tmp_path),
        "multi_idx_parquet_no_name",
        check_index=["__index_level_0__", "__index_level_1__", "nums"],
    )


@pytest.mark.parametrize("is_short_path", [True, False])
def test_write_to_azure(tmp_abfs_path: str, is_short_path: bool) -> None:
    if is_short_path:
        # Long Path: abfs://<container>@<account>.dfs.windows.net/<path>/
        # Short Path: abfs://<container>/<path>/
        res = urlparse(tmp_abfs_path)
        path = f"abfs://{res.username}{res.path}"
    else:
        path = tmp_abfs_path

    print(path)

    df = pd.DataFrame({"A": [1, 2]})
    check_write_func(
        lambda df, fname: df.to_parquet(fname),
        df,
        path,
        "write_to_azure",
        # This triggers Pandas to use fsspec to write, cause using PyArrow causes issues
        pandas_fn=lambda df, fname: df.to_parquet(fname, storage_options={}),
    )


# ---------------------------- Test Error Checking ---------------------------- #
@pytest.mark.slow
def test_to_parquet_missing_arg(memory_leak_check):
    """test error raise when user didn't provide all required arguments
    in Bodo supported method"""

    # Save default developer mode value
    default_mode = numba.core.config.DEVELOPER_MODE
    # Test as a user
    numba.core.config.DEVELOPER_MODE = 0

    def impl1():
        df = pd.DataFrame({"A": np.arange(10)})
        df.to_parquet()

    with pytest.raises(
        numba.core.errors.TypingError, match="missing a required argument"
    ):
        bodo.jit(impl1)()

    # Reset developer mode
    numba.core.config.DEVELOPER_MODE = default_mode


@pytest.mark.slow
def test_to_parquet_engine():
    from bodo.utils.typing import BodoError

    msg = r".*DataFrame.to_parquet\(\): only pyarrow engine supported.*"

    @bodo.jit
    def impl(df):
        df.to_parquet("test.pq", engine="fastparquet")

    df = pd.DataFrame({"A": np.arange(10)})
    with pytest.raises(BodoError, match=msg):
        bodo.jit(lambda f: impl(df))(df)


@pytest.mark.slow
def test_to_parquet_row_group_size():
    from bodo.utils.typing import BodoError

    msg = r".*to_parquet\(\): row_group_size must be integer.*"

    @bodo.jit
    def impl(df):
        df.to_parquet("test.pq", row_group_size="3")

    df = pd.DataFrame({"A": np.arange(10)})
    with pytest.raises(BodoError, match=msg):
        bodo.jit(lambda f: impl(df))(df)
