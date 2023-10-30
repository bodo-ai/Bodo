# Copyright (C) 2022 Bodo Inc. All rights reserved.
import datetime

import numba
import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import check_func


@pytest.mark.parametrize(
    "to_array_input, dtype, answer",
    [
        pytest.param(
            1, bodo.IntegerArrayType(bodo.int32), pd.array([1]), id="scalar_integer"
        ),
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
                ]
                * 4
            ),
            id="vector_float",
        ),
        pytest.param(None, bodo.null_array_type, None, id="scalar_null"),
        pytest.param(
            pd.Series(["asfdav", "1423", "!@#$", None, "0.9305"] * 4),
            bodo.string_array_type,
            pd.Series(
                [
                    pd.array(["asfdav"], dtype="string[pyarrow]"),
                    pd.array(["1423"], dtype="string[pyarrow]"),
                    pd.array(["!@#$"], dtype="string[pyarrow]"),
                    None,
                    pd.array(["0.9305"], dtype="string[pyarrow]"),
                ]
                * 4
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
                ]
                * 4
            ),
            bodo.datetime_date_array_type,
            pd.Series(
                [
                    pd.array([datetime.date(2016, 3, 3)]),
                    pd.array([datetime.date(2012, 6, 18)]),
                    pd.array([datetime.date(1997, 1, 14)]),
                    None,
                    pd.array([datetime.date(2025, 1, 28)]),
                ]
                * 4
            ),
            id="vector_date",
        ),
        pytest.param(
            pd.Timestamp("2021-12-08"),
            numba.core.types.Array(bodo.datetime64ns, 1, "C"),
            np.array([pd.Timestamp("2021-12-08")], dtype="datetime64[ns]"),
            id="scalar_timestamp",
        ),
    ],
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
        (
            to_array_input,
            dtype,
        ),
        py_output=answer,
        check_dtype=False,
        is_out_distributed=not is_scalar,
    )


@pytest.mark.parametrize(
    "array, separator, answer",
    [
        pytest.param(
            np.array([4234, 401, -820]),
            "+",
            "4234+401+-820",
            id="int_scalar",
        ),
        pytest.param(
            pd.Series(
                [
                    [-253.123, None, 534.958, -4.37, 0.9305],
                    [19.9, -235.104, 437.0, -0.2952],
                    [1312.2423, None],
                    None,
                ]
                * 4
            ),
            "-",
            pd.Series(
                [
                    "-253.123000--534.958000--4.370000-0.930500",
                    "19.900000--235.104000-437.000000--0.295200",
                    "1312.242300-",
                    None,
                ]
                * 4
            ),
            id="float_vector",
        ),
        pytest.param(
            np.array([False, True, None, True]),
            "&",
            "False&True&&True",
            id="bool_scalar",
        ),
        pytest.param(
            pd.Series(
                [
                    ["-253.123", "534.958", None, "-4.37", "0.9305"],
                    None,
                    ["oneword"],
                    ["g0q0ejdif", "ewkf", "%%@", ",..;", "BLSDF"],
                ]
                * 4
            ),
            "",
            pd.Series(
                [
                    "-253.123534.958-4.370.9305",
                    None,
                    "oneword",
                    "g0q0ejdifewkf%%@,..;BLSDF",
                ]
                * 4
            ),
            id="string_vector",
        ),
        pytest.param(
            np.array(
                [
                    datetime.date(1932, 10, 5),
                    datetime.date(2012, 7, 23),
                    datetime.date(1999, 3, 15),
                    datetime.date(2022, 12, 29),
                ],
            ),
            "_",
            "1932-10-05_2012-07-23_1999-03-15_2022-12-29",
            id="date_scalar",
        ),
    ],
)
def test_array_to_string(array, separator, answer, memory_leak_check):
    distributed = True

    def impl(array, separator):
        return pd.Series(
            bodo.libs.bodosql_array_kernels.array_to_string(array, separator)
        )

    if not isinstance(array, pd.Series):
        distributed = False
        impl = lambda array, separator: bodo.libs.bodosql_array_kernels.array_to_string(
            array, separator
        )

    check_func(
        impl,
        (
            array,
            separator,
        ),
        py_output=answer,
        distributed=distributed,
        is_out_distributed=distributed,
    )


@pytest.mark.parametrize(
    "array, answer",
    [
        pytest.param(
            pd.Series([[[1, 2, 3]], None, [[1], [2, 3]]] * 4),
            pd.Series([1, None, 2] * 4),
            id="null_int_nested",
        ),
        pytest.param(
            pd.Series([["abc", "bce"], ["bce"], ["def", "xyz", "abc"], [], None] * 4),
            pd.Series([2, 1, 3, 0, None] * 4),
            id="null_string_nested",
        ),
        pytest.param(
            pd.Series(
                [[[1, 2, 3], [None]], None, [], [[1, 2, 3, 4, 5, 6], None, [7, 8, 9]]]
                * 4
            ),
            pd.Series([2, None, 0, 3] * 4),
            id="null_nested_nested_array",
        ),
        pytest.param(
            pd.Series(
                [
                    [{"W": [1], "Y": "abc"}],
                    None,
                    [{"W": [1, 2, 3], "Y": "xyz"}, {"W": [], "Y": "123"}],
                ]
                * 4
            ),
            pd.Series([1, None, 2] * 4),
            id="null_nested_nested_struct",
        ),
    ],
)
def test_array_size_array(array, answer, memory_leak_check):
    def impl(array):
        return pd.Series(bodo.libs.bodosql_array_kernels.array_size(array, False))

    check_func(impl, (array,), py_output=answer, check_dtype=False)


@pytest.mark.parametrize(
    "array,answer",
    [
        pytest.param(pd.Series(["A", "BC", None]), 3, id="null_string"),
        pytest.param(pd.Series([1, 2, None, 3]), 4, id="null_int"),
        pytest.param(None, None, id="null"),
    ],
)
def test_array_size_scalar(array, answer, memory_leak_check):
    def impl(array):
        return bodo.libs.bodosql_array_kernels.array_size(array, True)

    check_func(
        impl, (array,), py_output=answer, distributed=False, is_out_distributed=False
    )
