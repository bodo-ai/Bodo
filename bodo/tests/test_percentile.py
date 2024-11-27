import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import check_func, pytest_slow_unless_codegen

# Skip unless any library or BodoSQL codegen or files were changed
pytestmark = pytest_slow_unless_codegen


@pytest.fixture(
    params=[
        pytest.param((7, "sorted"), id="small-sorted"),
        pytest.param((7, "random_a"), id="small-random_a", marks=pytest.mark.slow),
        pytest.param((7, "random_b"), id="small-random_b", marks=pytest.mark.slow),
        pytest.param((10000, "sorted"), id="large-sorted", marks=pytest.mark.slow),
        pytest.param((10000, "random_a"), id="large-random_a"),
        pytest.param((10000, "random_b"), id="large-random_b", marks=pytest.mark.slow),
    ]
)
def percentile_data(request):
    """
    Produces the data used for PERCENTILE_DISC/PERCENTILE_CONT tests with various
    sizes and orderings.

    The various sizes are used to stress test the parallel tools and to enable
    different selection/interpolation edge cases based on the relationship between
    the percentile being sought and the total number of elements.

    The different orderings are used to ensure the quality of the parallel
    implementations when different subsets of the data are distribtued
    across the ranks.
    """
    size, ordering = request.param
    arr = np.arange(size, dtype=np.int64)
    if ordering != "sorted":
        rng = np.random.default_rng(42 if ordering == "random_a" else 75)
        arr = rng.permutation(arr)
    S = pd.Series(
        [None if i % 1023 == 5 else i**2 for i in range(size)], dtype=pd.Int64Dtype()
    )
    return S.iloc[arr].values


@pytest.fixture
def test_percentiles():
    """
    Produces the 20 percentile values that are to be used when calculating
    PERCENTILE_DISC/PERCENTILE_CONT on the datasets from percentile_data.
    """
    return np.array(
        [
            0.0,
            0.007123,
            0.01,
            0.058066,
            0.15,
            0.2,
            0.3081275,
            0.3456,
            0.4,
            0.499,
            0.5,
            0.501,
            0.654,
            0.75,
            0.8989,
            0.9,
            0.995,
            0.99987654,
            0.999999999,
            1.0,
        ],
        dtype=np.float64,
    )


@pytest.fixture(
    params=[
        pytest.param("Int32", id="int32"),
        pytest.param("UInt64", id="uint64", marks=pytest.mark.slow),
        pytest.param("Float32", id="float32", marks=pytest.mark.slow),
        pytest.param("Float64", id="float64", marks=pytest.mark.slow),
    ]
)
def data_type(request):
    """
    The dtypes that the data from percentile_data is to be coerced to in
    order to test various type edge cases.
    """
    return request.param


def get_percentile_cont_answers(size):
    """
    Obtains the result of calling PERCENTILE_CONT on percentile_data
    (on either size option) for the 20 percentile values specified in
    test_percentiles.
    """
    if size == 7:
        return np.array(
            [
                0.0,
                0.035615,
                0.05,
                0.29033,
                0.75,
                1.0,
                2.6219125,
                3.184,
                4.0,
                6.475,
                6.5,
                6.525,
                10.89,
                14.25,
                25.89,
                26.0,
                35.5,
                35.987654,
                35.9999999,
                36.0,
            ],
            dtype=np.float64,
        )
    else:
        return np.array(
            [
                0.0,
                5205.988815,
                10178.89,
                337585.741661,
                2251050.35,
                3999200.2,
                9498018.937392501,
                11945307.5392,
                15996800.4,
                24895220.269,
                24995000.5,
                25094980.291,
                42769062.674,
                56246250.25,
                80786159.2217,
                80983800.9,
                98983695.44500001,
                99955340.32740971,
                99980000.80025,
                99980001.0,
            ],
            dtype=np.float64,
        )


def get_percentile_disc_answers(size):
    """
    Obtains the result of calling PERCENTILE_CONT on percentile_data
    (on either size option) for the 20 percentile values specified in
    test_percentiles.
    """
    if size == 7:
        return np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                4.0,
                4.0,
                4.0,
                4.0,
                9.0,
                9.0,
                16.0,
                36.0,
                36.0,
                36.0,
                36.0,
                36.0,
                36.0,
            ],
            dtype=np.float64,
        )
    else:
        return np.array(
            [
                0.0,
                5184.0,
                10000.0,
                337561.0,
                2250000.0,
                3996001.0,
                9498724.0,
                11943936.0,
                15992001.0,
                24900100.0,
                24990001.0,
                25090081.0,
                42771600.0,
                56250000.0,
                80802121.0,
                80982001.0,
                99002500.0,
                99960004.0,
                99980001.0,
                99980001.0,
            ],
            dtype=np.float64,
        )


def test_percentile_cont(
    percentile_data, test_percentiles, data_type, memory_leak_check
):
    """
    Tests that calling PERCENTILE_CONT on percentile_data casted to the dtype
    data_type with the 20 percentile values specified in test_percentiles corectly
    produces the expected answers.
    """
    percentile_data = percentile_data.astype(data_type)

    def impl(data):
        out = np.empty(len(test_percentiles), dtype=np.float64)
        for i in range(len(test_percentiles)):
            out[i] = bodo.libs.array_kernels.percentile_cont(data, test_percentiles[i])
        return out

    answer = get_percentile_cont_answers(len(percentile_data))
    check_func(
        impl,
        (percentile_data,),
        py_output=answer,
        is_out_distributed=False,
    )


def test_percentile_disc(
    percentile_data, test_percentiles, data_type, memory_leak_check
):
    """
    Tests that calling PERCENTILE_DISC on percentile_data casted to the dtype
    data_type with the 20 percentile values specified in test_percentiles corectly
    produces the expected answers.
    """
    percentile_data = percentile_data.astype(data_type)

    def impl(data):
        out = np.empty(len(test_percentiles), dtype=np.float64)
        for i in range(len(test_percentiles)):
            out[i] = bodo.libs.array_kernels.percentile_disc(data, test_percentiles[i])
        return out

    answer = get_percentile_disc_answers(len(percentile_data))
    check_func(
        impl,
        (percentile_data,),
        py_output=answer,
        is_out_distributed=False,
    )


def test_percentile_all_null(memory_leak_check):
    """
    Verifies that calling PERCENTILE_CONT or PERCENTILE_DISC
    on all-null data returns None.
    """

    def impl(data):
        return (
            bodo.libs.array_kernels.percentile_cont(data, 0.5),
            bodo.libs.array_kernels.percentile_disc(data, 0.3),
        )

    null_data = pd.array([None] * 1000, dtype=pd.Int32Dtype())
    check_func(
        impl,
        (null_data,),
        py_output=(None, None),
        is_out_distributed=False,
    )


def test_percentile_with_groupby(memory_leak_check):
    """
    Tests PERCENTILE_DISC/PERCENTILE_CONT inside of a groupby
    """

    def impl(df):
        return df.groupby(["key"], as_index=False, dropna=False, _is_bodosql=True).agg(
            res0=bodo.utils.utils.ExtendedNamedAgg(
                column="data", aggfunc="percentile_cont", additional_args=("p0",)
            ),
            res1=bodo.utils.utils.ExtendedNamedAgg(
                column="data", aggfunc="percentile_disc", additional_args=("p1",)
            ),
        )

    df = pd.DataFrame(
        {
            "key": list("AAABBBCCCDDD"),
            "data": [
                None,
                None,
                None,
                1.0,
                2.0,
                4.0,
                0.0,
                10.0,
                None,
                None,
                None,
                100.0,
            ],
            "p0": [0.25] * 12,
            "p1": [0.5] * 12,
        }
    )

    answer = pd.DataFrame(
        {
            "key": list("ABCD"),
            "res0": [None, 1.5, 2.5, 100.0],
            "res1": [None, 2.0, 0.0, 100.0],
        }
    )
    check_func(
        impl,
        (df,),
        py_output=answer,
        sort_output=True,
        reset_index=True,
    )
