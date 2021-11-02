import datetime

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.hiframes.boxing import (
    _dtype_from_type_enum_list,
    _infer_series_dtype,
    dtype_to_array_type,
)
from bodo.tests.dataframe_common import df_value  # noqa
from bodo.tests.series_common import series_val  # noqa
from bodo.tests.utils import reduce_sum


@pytest.mark.slow
def test_metadata_typemaps():
    """checks that _one_to_one_enum_to_type_map, and _one_to_one_type_to_enum_map are reflections of each other"""
    from bodo.hiframes.boxing import (
        _one_to_one_enum_to_type_map,
        _one_to_one_type_to_enum_map,
    )

    for key, val in _one_to_one_type_to_enum_map.items():
        assert _one_to_one_enum_to_type_map[val] == key

    for key, val in _one_to_one_enum_to_type_map.items():
        assert _one_to_one_type_to_enum_map[val] == key


@pytest.mark.parametrize(
    "typ_val",
    [
        1,
        b"jakhsgdfusdlj",
        {"A": 1, "B": 2, "C": 1},
        -13,
        "hello world",
        pd.Timestamp("2021-10-01"),
        np.arange(10),
        pd.array([1, 8, 4, 10, 3] * 2, dtype="Int32"),
        pd.array([True, False, True, pd.NA, False]),
        np.array(
            [b"32234", b"342432", b"g43b2", b"4t242t", b" weew"] * 2, dtype=object
        ),
        pd.Int64Index([10, 12]),
        pd.Float64Index([10.1, 12.1]),
        pd.UInt64Index([10, 12]),
        pd.Index(["A", "B"] * 4),
        pytest.param(
            pd.RangeIndex(10),
            marks=pytest.mark.skip("Works, but the output is slightly different"),
        ),
        pytest.param(
            pd.date_range(start="2018-04-24", end="2018-04-27", periods=3, name="A"),
            marks=pytest.mark.skip("Works, but the output is slightly different"),
        ),
        pytest.param(
            pd.timedelta_range(start="1D", end="3D", name="A"),
            marks=pytest.mark.skip("Works, but the output is slightly different"),
        ),
        pd.CategoricalIndex(["A", "B", "A", "C", "B"]),
        pd.PeriodIndex(year=[2015, 2016, 2018], month=[1, 2, 3], freq="M"),
        pytest.param(pd.Index([b"hkjl", bytes(2), b""] * 3), id="binary_case"),
        np.array(
            [
                [datetime.date(2018, 1, 24), datetime.date(1983, 1, 3)],
                [datetime.date(1966, 4, 27), datetime.date(1999, 12, 7)],
                None,
                [datetime.date(1966, 4, 27), datetime.date(2004, 7, 8)],
                [],
                [datetime.date(2020, 11, 17)],
            ]
            * 2,
            dtype=object,
        ),
        datetime.date(2020, 11, 17),
    ],
)
def test_dtype_converter(typ_val):
    """
    tests _dtype_to_type_enum_list and _dtype_from_type_enum_list works for some non series types.

    This function unit tests specific subtypes which need may need to be reduced/expanded
    in the process of converting series/array types. For example, in the process of converting categorical
    arrays, we need to need to be able to convert the various index types. Therefore, we test converting
    those index types here.

    The handled non-series types are as follows:
        All the index types (as of Oct 2021)
        All the array types (as of Oct 2021)
        Scalar integers, strings, and bytes (needed for some named index conversions)
        Scalar None (needed for some types can be initialized with none values)
        Scalar boolean, datetime/timedelta types (no specific reason, but it was trivial to support them)
    """
    from numba.core import types

    from bodo.hiframes.boxing import (
        _dtype_from_type_enum_list,
        _dtype_to_type_enum_list,
    )

    typ_from_converter = _dtype_from_type_enum_list(
        _dtype_to_type_enum_list(types.unliteral(bodo.typeof(typ_val)))
    )

    assert typ_from_converter == types.unliteral(
        bodo.typeof(typ_val)
    ) or typ_from_converter == bodo.typeof(typ_val)


def check_series_typing_metadata(orig_series, output_series):
    """Helper function that returns True if the series has Bodo metadata such that the dtype infered
    from the metadata will be the same as the infered dtype of the original series
    (original series must contain >= 1 non null element)

    """
    return (
        hasattr(output_series, "_bodo_meta")
        and "type_metadata" in output_series._bodo_meta
        and _dtype_from_type_enum_list(output_series._bodo_meta["type_metadata"])
        == _infer_series_dtype(orig_series)
    )


def check_dataframe_typing_metadata(orig_df, output_df):
    """Helper function return True if the dataframe has Bodo metadata such that the dtypes infered
    from the metadata will be the same as the infered dtypes of the original dataframe
    (original dataframe must contain 1 >= columns, each containing >= 1 non null element)
    """
    if not (
        hasattr(output_df, "_bodo_meta")
        and "type_metadata" in output_df._bodo_meta
        and len(output_df._bodo_meta["type_metadata"]) == len(orig_df.columns)
    ):
        return False

    for i in range(len(orig_df.columns)):
        cur_type_enum_list = output_df._bodo_meta["type_metadata"][i]
        cur_series = orig_df.iloc[:, i]
        if not (
            _dtype_from_type_enum_list(cur_type_enum_list)
            == dtype_to_array_type(_infer_series_dtype(cur_series))
        ):
            return False

    return True


def test_series_typing_metadata(series_val, memory_leak_check):
    """tests that the typing metadata is properly set when returning an series from Bodo"""

    @bodo.jit
    def impl(S):
        return S.iloc[:0]

    retval = impl(series_val)
    assert check_series_typing_metadata(series_val, retval)


def test_dataframe_typing_metadata(df_value, memory_leak_check):
    """tests that the typing metadata is properly set when returning a DataFrame from Bodo"""

    @bodo.jit
    def impl(df):
        return df.iloc[:0, :]

    retval = impl(df_value)
    assert check_dataframe_typing_metadata(df_value, retval)


@pytest.mark.parametrize(
    "struct_series_val",
    [
        pd.Series(
            [
                {"A": 1, "B": 2, "C": 3},
                {"A": 4, "B": -2, "C": 31},
                {"A": 231, "B": 92, "C": 1},
            ]
        ),
        pd.Series(
            [
                {
                    "adsfal": "hello",
                    "admafd": None,
                    "OIUHJ": {"A": 1, "B": 2, "C": 3},
                },
                {
                    "adsfal": "otherstr",
                    "admafd": None,
                    "OIUHJ": {"A": 1, "B": 2, "C": 3},
                },
                {
                    "adsfal": "otherstr2",
                    "admafd": None,
                    "OIUHJ": {"A": 1, "B": 2, "C": 3},
                },
            ]
        ),
    ],
)
def test_series_typing_metadata_struct(struct_series_val, memory_leak_check):
    """tests that the typing metadata is properly set when passing a struct series from Bodo"""

    @bodo.jit
    def impl(S):
        return S

    @bodo.jit
    def impl2(S):
        return pd.DataFrame({"A": S})

    retval_series = impl(struct_series_val)
    retval_df = impl2(struct_series_val)
    assert check_series_typing_metadata(struct_series_val, retval_series)
    assert check_dataframe_typing_metadata(
        pd.DataFrame({"A": struct_series_val}), retval_df
    )


# manually adding distributed flags, just to insure no future changes
# cause the output to not be dist


@bodo.jit(distributed=["out"])
def int_gen_dist_df():
    out = pd.DataFrame({"A": pd.Series(np.arange(1))})
    return out


@bodo.jit(distributed=["df"])
def int_use_dist_df(df):
    return df["A"].sum()


@bodo.jit(distributed=["df"])
def str_gen_dist_df():
    df = pd.DataFrame({"A": pd.Series(np.arange(1))})
    df["A"] = df["A"].astype(str)
    return df


@bodo.jit(distributed=["df"])
def str_use_dist_df(df):
    df["B"] = df["A"] + "A"
    return df


@bodo.jit(distributed=["df"])
def bytes_gen_dist_df():
    df = pd.DataFrame({"A": pd.Series(np.arange(1))})
    df["A"] = df["A"].apply(lambda x: b"foo")
    return df


@bodo.jit(distributed=["df"])
def bytes_use_dist_df(df):
    df["B"] = df["A"].apply(lambda x: x.hex())
    return df


@bodo.jit(distributed=["df"])
def struct_gen_dist_df():
    df = pd.DataFrame({"A": pd.Series(np.arange(1))})
    df["A"] = df["A"].apply(lambda x: {"key1": 10, "key2": -10, "key3": 200})
    return df


# TODO: fix this, BE-1479
# @bodo.jit(distributed=["df"])
@bodo.jit
def struct_use_dist_df(df):
    df["B"] = df["A"].apply(lambda x: x["key1"])
    return df


@pytest.mark.parametrize(
    "gen_func, use_func",
    [
        pytest.param(int_gen_dist_df, int_use_dist_df, id="int"),
        pytest.param(str_gen_dist_df, str_use_dist_df, id="str"),
        pytest.param(bytes_gen_dist_df, bytes_use_dist_df, id="bytes"),
        pytest.param(struct_gen_dist_df, struct_use_dist_df, id="struct"),
    ],
)
def test_df_return_metadata(gen_func, use_func):
    """
    Tests that Bodo can properly use the returned metadata across functions. "gen_func" is a Bodo function that returns
    a dataframe that only contains data on rank 0. "use_func" is a Bodo function that uses the distributed data, in
    such a way that an error would be thrown if the dataframe's column's type was infered as the default (string).
    """

    out = gen_func()

    # sanity check to insure that gen_func does actually produce a distributed output

    passed = 1
    if bodo.get_rank() != 0 and len(out["A"]) != 0:
        passed = 0
    elif bodo.get_rank() == 0 and len(out["A"]) != 1:
        passed = 0

    fail_msg = f"gen_func failed to return a dataframe with only data on rank0\nDF on rank {bodo.get_rank()}:\n{str(out)}"
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), fail_msg

    use_func(out)

    # sanity check that the supplied use_func would throw an error, if the input was stripped of typing information && the dtype was object:
    if out["A"].dtype == object:
        is_str = (
            _dtype_from_type_enum_list(out._bodo_meta["type_metadata"][0])
            == bodo.string_array_type
        )

        # if the dtype is str, it would be infered corectly, as the default is str
        if not is_str:
            out._bodo_meta = None
            passed = 1
            if bodo.get_rank() != 0:
                passed = 0
                try:
                    use_func(out)
                except:
                    passed = 1
            elif bodo.get_rank() == 0:
                pass

            fail_msg = (
                f"use_func failed to throw an error when metadata was not present"
            )
            n_passed = reduce_sum(passed)
            assert n_passed == bodo.get_size(), fail_msg


# manually adding distributed flags, just to insure no future improvements
# cause the output to not be dist
@bodo.jit(distributed=["out"])
def int_gen_dist_series():
    out = pd.Series(np.arange(1))
    return out


@bodo.jit(distributed=["S"])
def int_use_dist_series(S):
    return S.sum()


@bodo.jit(distributed=["out"])
def str_gen_dist_series():
    out = pd.Series(np.arange(1)).apply(lambda x: str(x))
    return out


@bodo.jit(distributed=["df"])
def str_use_dist_series(S):
    return S + "A"


@bodo.jit(distributed=["out"])
def bytes_gen_dist_series():
    out = pd.Series(np.arange(1)).apply(lambda x: b"hello!")
    return out


# TODO: fix this, BE-1479
# @bodo.jit(distributed=["S"])
@bodo.jit
def bytes_use_dist_series(S):
    return S.apply(lambda x: x.hex())


@bodo.jit(distributed=["out"])
def struct_gen_dist_series():
    out = pd.Series(np.arange(1)).apply(
        lambda x: {"key1": 10, "key2": -10, "key3": 200}
    )
    return out


# TODO: Variable 'S' has distributed flag in function 'struct_use_dist_series', but it's not possible to distribute it.
# It's definitely distributed, I did a sanity check on it.
# fix this, BE-1479
# @bodo.jit(distributed=["S"])
@bodo.jit
def struct_use_dist_series(S):
    return S.apply(lambda x: x["key1"])


@pytest.mark.parametrize(
    "gen_func, use_func",
    [
        pytest.param(int_gen_dist_series, int_use_dist_series, id="int"),
        pytest.param(str_gen_dist_series, str_use_dist_series, id="str"),
        pytest.param(bytes_gen_dist_series, bytes_use_dist_series, id="bytes"),
        pytest.param(struct_gen_dist_series, struct_use_dist_series, id="struct"),
    ],
)
def test_series_return_metadata(gen_func, use_func):
    """
    Tests that Bodo can properly use the returned metadata across functions. "gen_func" is a Bodo function that returns
    a series that only contains data on rank 0. "use_func" is a Bodo function that uses the distributed data, in
    such a way that an error would be thrown if the series dtype was infered as the default (string).
    """

    out = gen_func()

    # sanity check to insure that gen_func does actually produce a distributed output,
    # with only data on rank0
    passed = 1
    if bodo.get_rank() != 0 and len(out) != 0:
        passed = 0
    if bodo.get_rank() == 0 and len(out) != 1:
        passed = 0

    n_passed = reduce_sum(passed)

    fail_msg = f"gen_func failed to return a series with only data on rank0\nSeries on rank {bodo.get_rank()}:\n{str(out)}"
    assert n_passed == bodo.get_size(), fail_msg
    use_func(out)

    # sanity check that the supplied use_func would throw an error, if the input was stripped of typing information && the dtype was object:
    if out.dtype == object:
        is_str = (
            _dtype_from_type_enum_list(out._bodo_meta["type_metadata"])
            == bodo.string_type
        )

        # if the dtype is str, it would be infered corectly, as the default is str
        if not is_str:
            out._bodo_meta = None
            passed = 1
            if bodo.get_rank() != 0:
                passed = 0
                try:
                    use_func(out)
                except:
                    passed = 1
            elif bodo.get_rank() == 0:
                pass

            fail_msg = (
                f"use_func failed to throw an error when metadata was not present"
            )
            n_passed = reduce_sum(passed)
            assert n_passed == bodo.get_size(), fail_msg
