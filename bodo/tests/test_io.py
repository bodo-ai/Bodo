# Copyright (C) 2019 Bodo Inc. All rights reserved.
import unittest
import pytest
import os
import shutil
import pandas as pd
import numpy as np
import h5py
import numba
import bodo
from bodo.utils.typing import BodoError
from bodo.utils.testing import ensure_clean
from bodo.tests.utils import (
    count_array_REPs,
    count_parfor_REPs,
    count_parfor_OneDs,
    count_array_OneDs,
    dist_IR_contains,
    get_rank,
    get_start_end,
    check_func,
    _get_dist_arg,
    reduce_sum,
    _test_equal_guard,
    DeadcodeTestPipeline,
)
from bodo.utils.utils import is_call_assign
from numba.ir_utils import find_callname, build_definitions
from decimal import Decimal


kde_file = os.path.join("bodo", "tests", "data", "kde.parquet")


def test_pq_pandas_date(datapath):
    fname = datapath("pandas_dt.pq")

    def impl():
        df = pd.read_parquet(fname)
        return pd.DataFrame({"DT64": df.DT64, "col2": df.DATE})

    bodo_func = bodo.jit(impl)
    pd.testing.assert_frame_equal(bodo_func(), impl())


@pytest.mark.skip("Needs datetime.date() support in parquet for latest arrow")
def test_pq_spark_date(datapath):
    fname = datapath("sdf_dt.pq")

    def impl():
        df = pd.read_parquet(fname)
        return pd.DataFrame({"DT64": df.DT64, "col2": df.DATE})

    bodo_func = bodo.jit(impl)
    pd.testing.assert_frame_equal(bodo_func(), impl())


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


def test_pq_nullable_int_single(datapath):
    # single piece parquet
    fname = datapath("int_nulls_single.pq")

    def test_impl():
        return pd.read_parquet(fname)

    try:
        bodo.io.parquet_pio.use_nullable_int_arr = True
        check_func(test_impl, (), check_dtype=False)
    finally:
        bodo.io.parquet_pio.use_nullable_int_arr = False


def test_pq_nullable_int_multi(datapath):
    # multi piece parquet
    fname = datapath("int_nulls_multi.pq")

    def test_impl():
        return pd.read_parquet(fname)

    try:
        bodo.io.parquet_pio.use_nullable_int_arr = True
        check_func(test_impl, (), check_dtype=False)
    finally:
        bodo.io.parquet_pio.use_nullable_int_arr = False


def test_pq_bool_with_nulls(datapath):
    fname = datapath("bool_nulls.pq")

    def test_impl():
        return pd.read_parquet(fname)

    check_func(test_impl, ())


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


def test_pq_list_str(datapath):
    def test_impl(fname):
        return pd.read_parquet(fname)

    check_func(test_impl, (datapath("list_str_arr.pq"),))
    check_func(test_impl, (datapath("list_str_parts.pq"),))


def test_pq_unsupported_types(datapath):
    """test unsupported data types in unselected columns
    """

    def test_impl(fname):
        return pd.read_parquet(fname, columns=["B"])

    check_func(test_impl, (datapath("nested_struct_example.pq"),))


def test_pq_invalid_column_selection(datapath):
    """test error raise when selected column is not in file schema
    """

    def test_impl(fname):
        return pd.read_parquet(fname, columns=["C"])

    with pytest.raises(BodoError, match="C not in Parquet file schema"):
        bodo.jit(test_impl)(datapath("nested_struct_example.pq"))


def test_pq_decimal(datapath):
    def test_impl(fname):
        return pd.read_parquet(fname)

    check_func(test_impl, (datapath("decimal1.pq"),))


def test_pq_date32(datapath):
    """Test reading date32 values into datetime.date array
    """
    def test_impl(fname):
        return pd.read_parquet(fname)

    check_func(test_impl, (datapath("date32_1.pq"),))


def test_csv_remove_col0_used_for_len(datapath):
    """read_csv() handling code uses the first column for creating RangeIndex of the
    output dataframe. In cases where the first column array is dead, it should be
    replaced by an alternative live array. This test makes sure this replacement happens
    properly.
    """
    fname = datapath("csv_data1.csv")

    def impl():
        df = pd.read_csv(fname, names=["A", "B", "C", "D"])
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


def test_h5_remove_dead(datapath):
    """make sure dead hdf5 read calls are removed properly
    """
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


def test_write_parquet():
    def write(df, filename):
        df.to_parquet(filename)

    def pandas_write(df, filename):
        # pandas/pyarrow throws this error when writing datetime64[ns]:
        # pyarrow.lib.ArrowInvalid: Casting from timestamp[ns] to timestamp[ms] would lose data: xxx
        # unless allow_truncated_timestamps=True.
        # NOTE: it will write with ms precision
        df.to_parquet(filename, allow_truncated_timestamps=True)

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
                dates = pd.Series(pd.date_range(start="1998-04-24", end="1998-04-29", periods=num_elements))
                df[col_name] = dates.dt.date
            elif dtype == "Datetime":
                dates = pd.Series(pd.date_range(start="1998-04-24", end="1998-04-29", periods=num_elements))
                df[col_name] = dates
                df._datetime_col = col_name
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

    # workaround for pandas/pyarrow writing nullable Int64 issue:
    # https://issues.apache.org/jira/browse/ARROW-5379
    import pyarrow

    pd.arrays.IntegerArray.__arrow_array__ = lambda self, type: pyarrow.array(
        self._data, mask=self._mask, type=type
    )

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
                    bodo_write = bodo.jit(write, all_args_distributed=True)
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

    def error_check1(df):
        df.to_parquet("out.parquet", partition_cols=["col1"])

    def error_check2(df):
        df.to_parquet("out.parquet", compression="wrong")

    def error_check3(df):
        df.to_parquet("out.parquet", index=3)

    df = pd.DataFrame({"A": range(5)})

    with pytest.raises(
        BodoError, match="Bodo does not currently support partition_cols option"
    ):
        bodo.jit(error_check1)(df)

    with pytest.raises(BodoError, match="Unsupported compression"):
        bodo.jit(error_check2)(df)

    with pytest.raises(BodoError, match="index must be a constant bool or None"):
        bodo.jit(error_check3)(df)


def test_write_parquet_decimal(datapath):
    """ Here we check that we can write the data read from decimal1.pq directory
        (has columns that use a precision and scale different from our default).
        See test_write_parquet above for main parquet write decimal test """

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
                    bodo_func = bodo.jit(func, all_args_distributed=True)
                    data = _get_dist_arg(df, False)
                if bodo.get_rank() == 0:
                    func(df, pd_fname)  # write with pandas
                bodo.barrier()
                bodo_func(data, bodo_fname)
                df_a = pd.read_parquet(pd_fname)
                df_b = pd.read_parquet(bodo_fname)
                pd.testing.assert_frame_equal(df_a, df_b)
                bodo.barrier()
            finally:
                # cleanup
                clean_pq_files(mode, pd_fname, bodo_fname)
                bodo.barrier()


def test_csv_bool1(datapath):
    """Test boolean data in CSV files.
    Also test extra separator at the end of the file
    which requires index_col=False.
    """

    def test_impl(fname):
        dtype = {"A": "int", "B": "bool", "C": "float"}
        return pd.read_csv(fname, names=dtype.keys(), dtype=dtype, index_col=False)

    # passing file name as argument to exercise value-based dispatch
    fname = datapath("csv_data_bool1.csv")
    check_func(test_impl, (fname,))


def test_csv_int_na1(datapath):
    fname = datapath("csv_data_int_na1.csv")

    def test_impl():
        dtype = {"A": "int", "B": "Int32"}
        return pd.read_csv(fname, names=dtype.keys(), dtype=dtype)

    check_func(test_impl, ())


def test_csv_int_na2(datapath):
    fname = datapath("csv_data_int_na1.csv")

    def test_impl():
        dtype = {"A": "int", "B": pd.Int32Dtype()}
        return pd.read_csv(fname, names=dtype.keys(), dtype=dtype)

    check_func(test_impl, ())


def test_csv_bool_na(datapath):
    fname = datapath("bool_nulls.csv")

    def test_impl():
        # TODO: support column 1 which is bool with NAs when possible with
        # Pandas dtypes
        # see Pandas GH20591
        dtype = {"ind": "int32", "B": "bool"}
        return pd.read_csv(fname, names=dtype.keys(), dtype=dtype, usecols=[0, 2])

    check_func(test_impl, ())


def test_csv_fname_comp(datapath):
    """Test CSV read with filename computed across Bodo functions
    """

    @bodo.jit
    def test_impl(data_folder):
        return load_func(data_folder)

    @bodo.jit
    def load_func(data_folder):
        fname = data_folder + "/csv_data1.csv"
        return pd.read_csv(fname)

    data_folder = os.path.join("bodo", "tests", "data")
    # should not raise exception
    test_impl(data_folder)


def test_write_csv_parallel_unicode():
    def test_impl(df, fname):
        df.to_csv(fname)

    bodo_func = bodo.jit(all_args_distributed=True)(test_impl)
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


def test_h5_read_seq(datapath):
    def test_impl(fname):
        f = h5py.File(fname, "r")
        X = f["points"][:]
        f.close()
        return X

    # passing function name as value to test value-based dispatch
    fname = datapath("lr.hdf5")
    bodo_func = bodo.jit(test_impl)
    np.testing.assert_allclose(bodo_func(fname), test_impl(fname))


def test_h5_read_const_infer_seq(datapath):
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


def test_h5_read_parallel(datapath):
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


def test_h5_filter(datapath):
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


def test_h5_slice1(datapath):
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


def test_h5_slice2(datapath):
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


def test_h5_read_group(datapath):
    fname = datapath("test_group_read.hdf5")

    def test_impl():
        f = h5py.File(fname, "r")
        g1 = f["G"]
        X = g1["data"][:]
        f.close()
        return X.sum()

    bodo_func = bodo.jit(test_impl)
    assert bodo_func() == test_impl()


def test_h5_file_keys(datapath):
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


def test_h5_group_keys(datapath):
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


def test_h5_write():
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


def test_h5_group_write():
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


def test_np_io1(datapath):
    fname = datapath("np_file1.dat")

    def test_impl():
        A = np.fromfile(fname, np.float64)
        return A

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_almost_equal(bodo_func(), test_impl())


def test_np_io2(datapath):
    fname = datapath("np_file1.dat")
    # parallel version
    def test_impl():
        A = np.fromfile(fname, np.float64)
        return A.sum()

    bodo_func = bodo.jit(test_impl)
    np.testing.assert_almost_equal(bodo_func(), test_impl())
    assert count_array_REPs() == 0
    assert count_parfor_REPs() == 0


def test_np_io3():
    def test_impl(A):
        if get_rank() == 0:
            A.tofile("np_file_3.dat")

    bodo_func = bodo.jit(test_impl)
    n = 111
    A = np.random.ranf(n)
    with ensure_clean("np_file_3.dat"):
        bodo_func(A)
        if get_rank() == 0:
            B = np.fromfile("np_file_3.dat", np.float64)
            np.testing.assert_almost_equal(A, B)


def test_np_io4():
    # parallel version
    def test_impl(n):
        A = np.arange(n)
        A.tofile("np_file_3.dat")

    bodo_func = bodo.jit(test_impl)
    n = 111
    A = np.arange(n)
    with ensure_clean("np_file_3.dat"):
        bodo_func(n)
        B = np.fromfile("np_file_3.dat", np.int64)
        np.testing.assert_almost_equal(A, B)


def test_csv_double_box(datapath):
    """Make sure boxing the output of read_csv() twice doesn't cause crashes
    See dataframe boxing function for extra incref of native arrays.
    """
    fname = datapath("csv_data1.csv")

    def test_impl():
        df = pd.read_csv(fname)
        print(df)
        return df

    bodo_func = bodo.jit(test_impl)
    print(bodo_func())


def test_csv_header_none(datapath):
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
    pd.testing.assert_frame_equal(b_df, p_df)


def test_csv_cat1(datapath):
    fname = datapath("csv_data_cat1.csv")

    def test_impl():
        ct_dtype = pd.CategoricalDtype(["A", "B", "C"])
        dtypes = {"C1": np.int, "C2": ct_dtype, "C3": str}
        df = pd.read_csv(fname, names=["C1", "C2", "C3"], dtype=dtypes)
        return df

    check_func(test_impl, ())


def test_csv_date_col_name(datapath):
    """Test the use of column names in "parse_dates" of read_csv
    """
    fname = datapath("csv_data_date1.csv")

    def test_impl():
        return pd.read_csv(
            fname,
            names=["A", "B", "C", "D"],
            dtype={"A": np.int, "B": np.float, "C": str, "D": np.int},
            parse_dates=["C"],
        )

    check_func(test_impl, ())


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
            return pd.read_csv(fname, names=dtype.keys(), dtype=dtype)

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(), test_impl())

    def test_csv_const_dtype1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data1.csv")

        def test_impl():
            dtype = {"A": "int", "B": "float64", "C": "float", "D": "int64"}
            return pd.read_csv(fname, names=dtype.keys(), dtype=dtype)

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(), test_impl())

    def test_csv_infer1(self):
        fname = os.path.join("bodo", "tests", "data", "csv_data_infer1.csv")

        def test_impl():
            return pd.read_csv(fname)

        bodo_func = bodo.jit(test_impl)
        pd.testing.assert_frame_equal(bodo_func(), test_impl())

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
        pd.testing.assert_frame_equal(bodo_func(), test_impl())

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
        pd_expected.index.name = None  # TODO: handle index name
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
        pd_expected.index.name = None  # TODO: handle index name
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

    def test_write_csv1(self):
        # only run on single processor
        if bodo.get_size() != 1:
            return

        def test_impl(df, fname):
            df.to_csv(fname)

        bodo_func = bodo.jit(test_impl)
        n = 111
        df = pd.DataFrame({"A": np.arange(n)}, index=np.arange(n) * 2)
        hp_fname = "test_write_csv1_bodo.csv"
        pd_fname = "test_write_csv1_pd.csv"
        with ensure_clean(pd_fname), ensure_clean(hp_fname):
            bodo_func(df, hp_fname)
            test_impl(df, pd_fname)
            pd.testing.assert_frame_equal(pd.read_csv(hp_fname), pd.read_csv(pd_fname))

    def test_write_csv_parallel1(self):
        def test_impl(n, fname):
            df = pd.DataFrame({"A": np.arange(n)})
            df.to_csv(fname)

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
