# Copyright (C) 2022 Bodo Inc. All rights reserved.
import datetime

import pytest

import bodo
import numba
import numpy as np
import pandas as pd
from bodo.tests.utils import check_func


@pytest.mark.parametrize(
    "to_array_input, dtype, answer",
    [
        pytest.param(1, bodo.IntegerArrayType(bodo.int32), pd.array([1]), id="scalar_integer"),
        pytest.param(
            pd.Series([-253.123, None, 534.958, -4.37, 0.9305] * 4),
            bodo.FloatingArrayType(bodo.float64),
            pd.Series(
                [
                    pd.array([-253.123]),
                    None,
                    pd.array([534.958]),
                    pd.array([-4.37]),
                    pd.array([0.9305]),
                ] * 4
            ),
            id="vector_float",
        ),
        pytest.param(None, bodo.null_array_type, None, id="scalar_null"),
        pytest.param(
            pd.Series(["asfdav", "1423", "!@#$", None, "0.9305"] * 4),
            bodo.string_array_type,
            pd.Series(
                [
                    pd.array(["asfdav"]),
                    pd.array(["1423"]),
                    pd.array(["!@#$"]),
                    None,
                    pd.array(["0.9305"]),
                ] * 4
            ),
            id="vector_string",
        ),
        pytest.param(
            bodo.Time(18, 32, 59),
            bodo.TimeArrayType(9),
            pd.array([bodo.Time(18, 32, 59)]),
            id="scalar_time",
        ),
        pytest.param(
            pd.Series(
                [
                    datetime.date(2016, 3, 3),
                    datetime.date(2012, 6, 18),
                    datetime.date(1997, 1, 14),
                    None,
                    datetime.date(2025, 1, 28),
                ] * 4
            ),
            bodo.datetime_date_array_type,
            pd.Series(
                [
                    pd.array([datetime.date(2016, 3, 3)]),
                    pd.array([datetime.date(2012, 6, 18)]),
                    pd.array([datetime.date(1997, 1, 14)]),
                    None,
                    pd.array([datetime.date(2025, 1, 28)]),
                ] * 4
            ),
            id="vector_date",
        ),
        pytest.param(
            pd.Timestamp("2021-12-08"),
            numba.core.types.Array(bodo.datetime64ns, 1, 'C'),
            np.array([pd.Timestamp("2021-12-08")], dtype='datetime64[ns]'),
            id="scalar_timestamp",
        ),
    ]
)
def test_to_array(to_array_input, dtype, answer, memory_leak_check):
    is_scalar = False
    def impl(to_array_input, dtype):
        return pd.Series(
            bodo.libs.bodosql_array_kernels.to_array(to_array_input, dtype)
        )

    if not isinstance(to_array_input, pd.Series):
        is_scalar = True
        impl = lambda to_array_input, dtype: bodo.libs.bodosql_array_kernels.to_array(
            to_array_input, dtype
        )

    check_func(
        impl,
        (to_array_input, dtype,),
        py_output=answer,
        check_dtype=False,
        is_out_distributed=not is_scalar,
    )
