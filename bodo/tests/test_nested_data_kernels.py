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
    "arg0, arg1, is_scalar_0, is_scalar_1, use_map_arrays, answer",
    [
        pytest.param(
            pd.Series(
                pd.Series([[None], [1], [1, 2, 3], None, [None, 3]]).repeat(5).values
            ),
            pd.Series([[], [1], [5, 3, 0], None, [None, 4]] * 5),
            False,
            False,
            False,
            pd.Series(
                [False, False, False, None, True]
                + [False, True, False, None, False]
                + [False, True, True, None, False]
                + [None] * 5
                + [False, False, True, None, True]
            ),
            id="int_arrays-vector",
        ),
        pytest.param(
            pd.Series(
                pd.Series([[None], [""], ["", "A", "BC"], None, [None, "BC"]])
                .repeat(5)
                .values
            ),
            pd.Series([[], [""], ["GHIJ", "BC", "KLMNO"], None, [None, "DEF"]] * 5),
            False,
            False,
            False,
            pd.Series(
                [False, False, False, None, True]
                + [False, True, False, None, False]
                + [False, True, True, None, False]
                + [None] * 5
                + [False, False, True, None, True]
            ),
            id="string_arrays-vector",
        ),
        pytest.param(
            pd.Series(
                pd.Series([[[1]], None, [[2], [None]], [], [[1], [None], [0]]])
                .repeat(5)
                .values
            ),
            pd.Series([[[1]], None, [[None], [2]], [], [[None], [1], [0]]] * 5),
            False,
            False,
            False,
            pd.Series(
                [True, None, False, False, True]
                + [None] * 5
                + [False, None, True, False, True]
                + [False, None, False, False, False]
                + [True, None, True, False, True]
            ),
            id="nested_int_arrays-vector",
        ),
        pytest.param(
            pd.Series(
                [
                    [{"A": 0, "B": [1]}],
                    [None, {"A": 0, "B": [1, 0]}],
                    [{"A": 0, "B": [0, 1]}],
                    None,
                    [{"A": 0, "B": [1]}, {"A": 1, "B": [0, 1]}],
                ]
            )
            .repeat(5)
            .values,
            pd.Series(
                [
                    [{"A": 0, "B": [1]}],
                    [None, {"A": 0, "B": [1, 0]}],
                    [{"A": 0, "B": [0, 1]}],
                    None,
                    [{"A": 0, "B": [1]}, None, {"A": 1, "B": [0, 1]}],
                ]
                * 5
            ).values,
            False,
            False,
            False,
            pd.Series(
                [True, False, False, None, True]
                + [False, True, False, None, True]
                + [False, False, True, None, False]
                + [None] * 5
                + [True, False, False, None, True]
            ),
            id="nested_struct_arrays-vector",
        ),
        pytest.param(
            pd.Series(
                pd.Series(
                    [
                        [{"A": 0, "B": 1}, {"A": 0, "B": 1, "C": 2}],
                        [{}, {"A": 0}],
                        [
                            {"A": 1, "B": 0},
                        ],
                        None,
                        [{}, {"B": 1, "A": 0}],
                    ]
                )
                .repeat(5)
                .values
            ),
            pd.Series(
                [
                    [{"A": 0, "B": 1}, {"A": 0, "B": 1, "C": 2}],
                    [{}, {"A": 0}],
                    [
                        {"A": 1, "B": 0},
                    ],
                    None,
                    [{}, {"B": 1, "A": 0}],
                ]
                * 5
            ),
            False,
            False,
            True,
            pd.Series(
                [True, False, False, None, True]
                + [False, True, False, None, True]
                + [False, False, True, None, False]
                + [None] * 5
                + [True, True, False, None, True]
            ),
            id="nested_map_arrays-vector",
            marks=pytest.mark.skip(
                reason="[BSE-1829] TODO: support copy_data on map arrays"
            ),
        ),
        pytest.param(
            np.array([1, 2, 4, 8, 16]),
            np.array([None, 3, 9, 27]),
            True,
            True,
            False,
            False,
            id="int_arrays-scalar_no_match",
        ),
        pytest.param(
            np.array([1, 2, 4, 8, 16]),
            np.array([None, 3, 9, 8, 27]),
            True,
            True,
            False,
            True,
            id="int_arrays-scalar_match",
        ),
        pytest.param(
            np.array([1, 2, 4, None, 16]),
            np.array([None, 3, 9, 27]),
            True,
            True,
            False,
            True,
            id="int_arrays-scalar_null_match",
        ),
        pytest.param(
            np.array([1, 2, 4, None, 16]),
            pd.Series(
                [
                    [0],
                    [1],
                    [2, 3],
                    [4, 5],
                    [6, 7, 8],
                    [9, None, 11],
                    [12, 13, 14, 15],
                    [16],
                ]
            ),
            True,
            False,
            False,
            pd.Series([False, True, True, True, False, True, False, True]),
            id="int_arrays-scalar_vector",
        ),
    ],
)
def test_arrays_overlap(
    arg0, arg1, is_scalar_0, is_scalar_1, use_map_arrays, answer, memory_leak_check
):
    both_scalar = is_scalar_0 and is_scalar_1
    either_scalar = is_scalar_0 or is_scalar_1
    if both_scalar:

        def impl(arg0, arg1):
            return bodo.libs.bodosql_array_kernels.arrays_overlap(
                arg0, arg1, is_scalar_0, is_scalar_1
            )

    else:

        def impl(arg0, arg1):
            return pd.Series(
                bodo.libs.bodosql_array_kernels.arrays_overlap(
                    arg0, arg1, is_scalar_0, is_scalar_1
                )
            )

    check_func(
        impl,
        (arg0, arg1),
        py_output=answer,
        distributed=not either_scalar,
        is_out_distributed=not either_scalar,
        dist_test=not either_scalar,
        only_seq=either_scalar,
        use_map_arrays=use_map_arrays,
    )


@pytest.mark.parametrize(
    "elem, container, elem_is_scalar, container_is_scalar, use_map_arrays, answer",
    [
        pytest.param(
            pd.Series([1, 2, None] * 2, dtype=pd.Int32Dtype()).values,
            pd.Series([[3, 1, 4, None, 2, 1]] * 3 + [[2, None, 2]] * 3).values,
            False,
            False,
            False,
            pd.Series([1, 4, 3, None, 0, 1], dtype=pd.Int32Dtype()),
            id="int-vector_vector",
        ),
        pytest.param(
            0,
            pd.Series(
                [
                    [],
                    [0],
                    [1, 0, 1],
                    None,
                    list(range(10, -11, -1)),
                    [None],
                    [2, 3, None, 1, 0, 4],
                ]
            ).values,
            True,
            False,
            False,
            pd.Series([None, 0, 1, None, 10, None, 4], dtype=pd.Int32Dtype()),
            id="int-scalar_vector",
        ),
        pytest.param(
            None,
            pd.Series(
                [
                    [None, 0, None],
                    [None],
                    [],
                    list(range(20)) + [None],
                    None,
                    [2, 3, None, 1, 0, 4],
                ]
            ).values,
            True,
            False,
            False,
            pd.Series([0, 0, None, 20, None, 2], dtype=pd.Int32Dtype()),
            id="int-null_vector",
        ),
        pytest.param(
            16,
            np.array([0, 1, 4, 9, 16, 25]),
            True,
            True,
            False,
            4,
            id="int-scalars",
        ),
        pytest.param(
            "foo",
            pd.Series(
                [
                    ["oof", "foo"],
                    [""] * 3,
                    [None] * 5,
                    ["Foo", None, "foo", None] * 2,
                    None,
                    ["foo"] * 8,
                    None,
                    ["f", "fo", "fooo", "foo", "foooo"],
                    ["FOO", "fOo", "fo", "oo"],
                ]
            ).values,
            True,
            False,
            False,
            pd.Series(
                [1, None, None, 2, None, 0, None, 3, None], dtype=pd.Int32Dtype()
            ),
            id="string-scalar_vector",
        ),
        pytest.param(
            np.array([1, 2, 3]),
            pd.Series(
                [
                    [],
                    [[], [1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
                    [[1, 0, 1], []],
                    [[3, 2, 1], [1, 3, 2], [2, 1, 3], [3, 1, 2], [2, 3, 1], [1, 2, 3]],
                    None,
                    [[1, 2, 3], None],
                ]
            ),
            True,
            False,
            False,
            pd.Series([None, 3, None, 5, None, 0], dtype=pd.Int32Dtype()),
            id="int_array-scalar_vector",
        ),
        pytest.param(
            np.array([1, 2, 3]),
            pd.Series([[], [1], [1, 2], [1, 2, 3], [1, 2, 3, 4]]).values,
            True,
            True,
            False,
            3,
            id="int_array-scalar_scalar",
        ),
        pytest.param(
            pd.Series(
                [
                    [],
                    [1],
                    [1, 2],
                    [1, 2, 3],
                    [1, 2, 3, 4],
                    [1, 2, 3, 4, 5],
                ]
            ).values,
            pd.Series([[], [1], [1, 2], [1, 2, 3], [1, 2, 3, 4]]).values,
            False,
            True,
            False,
            pd.Series([0, 1, 2, 3, 4, None], dtype=pd.Int32Dtype()),
            id="int_array-vector_scalar",
        ),
        pytest.param(
            pd.Series(
                [
                    [],
                    [1],
                    [1, 2],
                    [1, 2, 3],
                    [1, 2, 3, 4],
                    [1, 2, 3, 4, 5],
                ]
            ).values,
            pd.Series(
                [
                    [],
                    [[0, 1]],
                    [[], [1], [1, 2], [1, 2, 3]],
                    [[None], [1, 2, 3], [1, 2, 3]],
                    [[1, 2, 3, 4], [1, 2, 3], [1, 2], [1]],
                    [[1, 2, 3, 4], [2, 3, 4, 5]],
                ]
            ).values,
            False,
            False,
            False,
            pd.Series([None, None, 2, 1, 0, None], dtype=pd.Int32Dtype()),
            id="int_array-vector_vector",
        ),
        pytest.param(
            np.array([["A"]]),
            pd.Series(
                [
                    [[]],
                    [[["A"]]],
                    [[["A", "B"], ["A"]], [["B"]], []],
                    [[["A", "B"]], [["A"], ["A", "C"]], [["A"]]],
                    [],
                ]
            ).values,
            True,
            False,
            False,
            pd.Series([None, 0, None, 2, None], dtype=pd.Int32Dtype()),
            id="string_array_array-scalar_vector",
        ),
        pytest.param(
            pd.Series([{"A": 0, "B": 1}] * 3 + [{"A": 1, "B": 0}] * 3),
            pd.Series(
                [
                    [{"A": 0, "B": 0}, {"A": 1, "B": 1}],
                    [{"A": 0, "B": 1}, None, {"A": 1, "B": 0}, None],
                    [
                        {"A": 2, "B": 1},
                        {"A": 1, "B": 1},
                        None,
                        None,
                        None,
                        {"A": 0, "B": 0},
                    ],
                ]
                * 2
            ),
            False,
            False,
            False,
            pd.Series([None, 0, None, None, 2, None], dtype=pd.Int32Dtype()),
            id="struct-vector_vector",
        ),
        pytest.param(
            {"A": 1, "B": 0},
            pd.Series(
                [
                    [{"A": 0, "B": 0}, {"A": 1, "B": 1}, {"A": 0, "B": 0}],
                    [{"A": 0, "B": 1}, {"A": 1, "B": 0}, {"A": 1, "B": 0}],
                    [{"A": 0, "B": 1}, {"A": 1, "B": 1}, {"A": 0, "B": 0}],
                ]
                * 5
            ).values,
            True,
            False,
            False,
            pd.Series([None, 1, None] * 5, dtype=pd.Int32Dtype()),
            id="struct-scalar_vector",
            marks=pytest.mark.skip(
                reason="[BSE-1781] TODO: fix array_construct and array_position when inputs are mix of struct arrays and scalars"
            ),
        ),
        pytest.param(
            pd.Series([{"hex": "660c21", "name": "pomegranate"}] * 8),
            pd.Series(
                [
                    {"name": "pomegranate"},
                    {},
                    {"hex": "660c21"},
                    {"hex": "660c21", "name": "red"},
                    {"hex": "#660c21", "name": "pomegranate"},
                    {"hex": "660c21", "name": "pomegranate"},
                    None,
                    {"hex": "660c21", "name": "pomegranate"},
                ],
            ).values,
            False,
            True,
            True,
            pd.Series([5] * 8, dtype=pd.Int32Dtype()),
            id="map-vector_scalar",
        ),
        pytest.param(
            {"hex": "660c21", "name": "pomegranate"},
            pd.Series(
                [
                    {"name": "pomegranate"},
                    {},
                    {"hex": "660c21"},
                    {"hex": "660c21", "name": "red"},
                    {"hex": "#660c21", "name": "pomegranate"},
                    {"hex": "660c21", "name": "pomegranate"},
                    None,
                    {"hex": "660c21", "name": "pomegranate"},
                ],
            ).values,
            True,
            True,
            True,
            5,
            id="map-scalar_scalar",
        ),
    ],
)
def test_array_position(
    elem,
    container,
    elem_is_scalar,
    container_is_scalar,
    use_map_arrays,
    answer,
    memory_leak_check,
):
    # [BSE-1840] properly address distributed handling of array scalars
    all_scalar = elem_is_scalar and container_is_scalar
    any_scalar = elem_is_scalar or container_is_scalar
    if all_scalar:

        def impl(elem, container):
            return bodo.libs.bodosql_array_kernels.array_position(
                elem, container, elem_is_scalar, container_is_scalar
            )

    else:

        def impl(elem, container):
            return pd.Series(
                bodo.libs.bodosql_array_kernels.array_position(
                    elem, container, elem_is_scalar, container_is_scalar
                )
            )

    check_func(
        impl,
        (elem, container),
        py_output=answer,
        distributed=not all_scalar,
        is_out_distributed=not all_scalar,
        dist_test=not any_scalar,
        only_seq=any_scalar,
        use_map_arrays=use_map_arrays,
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
