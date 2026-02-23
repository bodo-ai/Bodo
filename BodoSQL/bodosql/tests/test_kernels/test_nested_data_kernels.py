import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bodo
import bodosql
from bodo.tests.utils import check_func
from bodo.utils.typing import BodoError
from bodo.utils.utils import is_array_typ


@pytest.mark.parametrize(
    "arr, to_remove, arr_is_scalar, to_remove_is_scalar, answer",
    [
        pytest.param(
            pd.Series(
                [
                    [1, 1, 2, 1, 1, 2, 3, 2, 1],
                    [0, 1, 4, 9, 16, 25],
                    None,
                    [2, None, 3, None, 5],
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.int32())),
            )
            .repeat(4)
            .values,
            pd.Series(
                [
                    [3, 2, 0, 1, 0, 1],
                    [0, 2, None, 4, 6, 8],
                    [2, 3, 5, 9, 17],
                    None,
                ]
                * 4,
                dtype=pd.ArrowDtype(pa.large_list(pa.int32())),
            ).values,
            False,
            False,
            pd.Series(
                [
                    [1, 1, 2, 2, 1],
                    [1, 1, 1, 1, 2, 3, 2, 1],
                    [1, 1, 1, 1, 2, 2, 1],
                    None,
                    [4, 9, 16, 25],
                    [1, 9, 16, 25],
                    [0, 1, 4, 16, 25],
                    None,
                    None,
                    None,
                    None,
                    None,
                    [None, None, 5],
                    [3, None, 5],
                    [None, None],
                    None,
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.int32())),
            ),
            id="int_array-vector-vector",
        ),
        pytest.param(
            pd.Series(
                [
                    list("ABCDEFGHIJABCDEFGHIJ"),
                    None,
                    list("Alphabet Soup Is Delicious"),
                    "ALPHABET SOUP IS DELICIOUS".split(),
                    None,
                    ["A", None, "E", "I", None, "OU", None, "Y"],
                    list("EIEIO"),
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.string())),
            ).values,
            np.array(["A", "E", "I", "O", "U", "Y", None]),
            False,
            True,
            pd.Series(
                [
                    list("BCDFGHJABCDEFGHIJ"),
                    None,
                    list("lphabet Soup s Delicious"),
                    "ALPHABET SOUP IS DELICIOUS".split(),
                    None,
                    [None, "OU", None],
                    list("EI"),
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.string())),
            ),
            id="string_array-vector-scalar",
        ),
        pytest.param(
            pd.array(
                [[i, i**2, None][i % 3] for i in range(20)], dtype=pd.Int32Dtype()
            ),
            pd.array([i**2 for i in range(100)], dtype=pd.Int32Dtype()),
            True,
            True,
            pd.array(
                [None, 3, None, 6, None, None, 12, None, 15, None, 18],
                dtype=pd.Int32Dtype(),
            ),
            id="int_array-scalar-scalar",
        ),
        pytest.param(
            pd.Series(
                [[1], [], None, [1, 2], [1, 2, 3]],
                dtype=pd.ArrowDtype(pa.large_list(pa.int32())),
            ).values,
            pd.Series(
                [
                    [[1], [2], [3], None, [4], [None]],
                    [[], [1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
                    [[1, 2], [2, 3], None, [2, 1], [3, 2], [1, 3], None, [3, 1]],
                    [[1], [1, 2], [1], [1, 2, 3], [1]],
                    [],
                    [[4], []],
                    None,
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.large_list(pa.int32()))),
            ).values,
            True,
            False,
            pd.Series(
                [
                    [[], [1, 2], [1, 2, 3]],
                    [None],
                    [[1], [], [1, 2, 3]],
                    [[], None],
                    [[1], [], None, [1, 2], [1, 2, 3]],
                    [[1], None, [1, 2], [1, 2, 3]],
                    None,
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.large_list(pa.int32()))),
            ),
            id="int_array_array-scalar-vector",
        ),
        pytest.param(
            pd.Series(
                [
                    [
                        {"A": 0, "B": "foo"},
                        None,
                        {"A": 1, "B": "bar"},
                        None,
                        {"A": 2, "B": "fizz"},
                    ],
                    [{"A": 3, "B": "buzz"}, {"A": 4, "B": "foo"}],
                    [{"A": 5, "B": "bar"}],
                    [
                        {"A": 6, "B": "fizz"},
                        None,
                        {"A": 7, "B": "buzz"},
                        None,
                        {"A": 8, "B": "foo"},
                    ],
                    [{"A": 9, "B": "bar"}, {"A": 0, "B": "fizz"}],
                    [],
                    [{"A": 1, "B": "buzz"}],
                ]
                * 2,
                dtype=pd.ArrowDtype(
                    pa.large_list(
                        pa.struct(
                            [
                                pa.field("A", pa.int32()),
                                pa.field("B", pa.string()),
                            ]
                        )
                    )
                ),
            ).values,
            pd.Series(
                [
                    [{"A": 0, "B": "foo"}],
                    [],
                    [{"A": 5, "B": "foo"}, None],
                    [None, {"A": 7, "B": "buzz"}],
                    [
                        {"A": 3, "B": "bar"},
                        None,
                        {"A": 0, "B": "bar"},
                        {"A": 9, "B": "fizz"},
                    ],
                    [{"A": 0, "B": "foo"}, {"A": 1, "B": "bar"}],
                    [{"A": 1, "B": "fizz"}, {"A": 0, "B": "buzz"}],
                ]
                * 2,
                dtype=pd.ArrowDtype(
                    pa.large_list(
                        pa.struct(
                            [
                                pa.field("A", pa.int32()),
                                pa.field("B", pa.string()),
                            ]
                        )
                    )
                ),
            ).values,
            False,
            False,
            pd.Series(
                [
                    [None, {"A": 1, "B": "bar"}, None, {"A": 2, "B": "fizz"}],
                    [{"A": 3, "B": "buzz"}, {"A": 4, "B": "foo"}],
                    [{"A": 5, "B": "bar"}],
                    [{"A": 6, "B": "fizz"}, None, {"A": 8, "B": "foo"}],
                    [{"A": 9, "B": "bar"}, {"A": 0, "B": "fizz"}],
                    [],
                    [{"A": 1, "B": "buzz"}],
                ]
                * 2,
                dtype=pd.ArrowDtype(
                    pa.large_list(
                        pa.struct(
                            [
                                pa.field("A", pa.int32()),
                                pa.field("B", pa.string()),
                            ]
                        )
                    )
                ),
            ),
            id="struct_array-vector-vector",
        ),
        pytest.param(
            pd.Series(
                [
                    [{"A": [0], "B": None, "C": [1, 2]}, {"D": [3]}],
                    [],
                    [{"E": [4], "F": [5, 6]}, {"A": [7]}, None, {"B": [8]}],
                    [{"D": [3]}, None, None, {"D": [3]}, None],
                    [{"D": [3], "A": [7]}, None, {"D": [3, None]}],
                    None,
                    [{"A": [0, 1], "D": [3]}, {}],
                ]
                * 2,
                dtype=pd.ArrowDtype(
                    pa.large_list(pa.map_(pa.large_string(), pa.large_list(pa.int8())))
                ),
            ).values,
            pd.Series(
                [[{"A": [7]}, {"D": [3]}, None]] * 14,
                dtype=pd.ArrowDtype(
                    pa.large_list(pa.map_(pa.large_string(), pa.large_list(pa.int8())))
                ),
            ).values,
            False,
            False,
            pd.Series(
                [
                    [{"A": [0], "B": None, "C": [1, 2]}],
                    [],
                    [{"E": [4], "F": [5, 6]}, {"A": [7]}, None, {"B": [8]}],
                    [{"D": [3]}],
                    [{"D": [3], "A": [7]}, {"D": [3, None]}],
                    None,
                    [{"A": [0, 1], "D": [3]}, {}],
                ]
                * 2,
                dtype=pd.ArrowDtype(
                    pa.large_list(pa.map_(pa.large_string(), pa.large_list(pa.int8())))
                ),
            ),
            id="map_array-vector-vector",
            marks=pytest.mark.skip(
                reason="[BSE-2126] fix value error with writing to column of arrays of map arrays of integer arrays"
            ),
        ),
    ],
)
def test_array_except(
    arr,
    to_remove,
    arr_is_scalar,
    to_remove_is_scalar,
    answer,
    memory_leak_check,
):
    either_scalar = arr_is_scalar or to_remove_is_scalar
    both_scalar = arr_is_scalar and to_remove_is_scalar
    if not both_scalar:

        def impl(arr, to_remove):
            return pd.Series(
                bodosql.kernels.array_except(
                    arr, to_remove, arr_is_scalar, to_remove_is_scalar
                )
            )

    else:

        def impl(arr, to_remove):
            return bodosql.kernels.array_except(
                arr, to_remove, arr_is_scalar, to_remove_is_scalar
            )

    check_func(
        impl,
        (arr, to_remove),
        py_output=answer,
        check_dtype=False,
        check_names=False,
        distributed=not either_scalar,
        is_out_distributed=not either_scalar,
        dist_test=not either_scalar,
        only_seq=either_scalar,
        convert_columns_to_pandas=True,
    )


@pytest.mark.parametrize("flag0", [True, False])
@pytest.mark.parametrize("flag1", [True, False])
@pytest.mark.slow
def test_option_array_except(flag0, flag1, memory_leak_check):
    def impl(arr, to_remove, flag0, flag1):
        arg0 = None if flag0 else arr
        arg1 = None if flag1 else to_remove
        return bodosql.kernels.array_except(
            arg0, arg1, is_scalar_0=True, is_scalar_1=True
        )

    arr = pd.array(
        [1, None, 1, 2, None, 1, 2, 3, None, 1, 2, 3, 4], dtype=pd.Int32Dtype()
    )
    to_remove = pd.array([1, None, 2, None, 3], dtype=pd.Int32Dtype())
    answer = (
        None
        if flag0 or flag1
        else pd.array([1, 1, 2, None, 1, 2, 3, 4], dtype=pd.Int32Dtype())
    )

    check_func(
        impl,
        (arr, to_remove, flag0, flag1),
        py_output=answer,
        check_dtype=False,
        check_names=False,
        distributed=False,
        is_out_distributed=False,
        dist_test=False,
        only_seq=True,
    )


@pytest.mark.parametrize(
    "arr_0, arr_1, arr_is_scalar, to_remove_is_scalar, answer",
    [
        pytest.param(
            pd.Series(
                [
                    [1, 1, 2, 1, 1, 2, 3, 2, 1],
                    [0, 1, 4, 9, 16, 25],
                    None,
                    [2, None, 3, None, 5],
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.int32())),
            )
            .repeat(4)
            .values,
            pd.Series(
                [
                    [3, 2, 0, 1, 0, 1],
                    [0, 2, None, 4, 6, 8],
                    [2, 3, 5, 9, 17],
                    None,
                ]
                * 4,
                dtype=pd.ArrowDtype(pa.large_list(pa.int32())),
            ).values,
            False,
            False,
            pd.Series(
                [
                    [1, 1, 2, 3],
                    [2],
                    [2, 3],
                    None,
                    [0, 1],
                    [0, 4],
                    [9],
                    None,
                    None,
                    None,
                    None,
                    None,
                    [2, 3],
                    [2, None],
                    [2, 3, 5],
                    None,
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.int32())),
            ),
            id="int_array-vector-vector",
        ),
        pytest.param(
            pd.Series(
                [
                    list("ABCDEFGHIJABCDEFGHIJ"),
                    None,
                    list("Alphabet Soup Is Delicious"),
                    "ALPHABET SOUP IS DELICIOUS".split(),
                    None,
                    ["A", None, "E", "I", None, "OU", None, "Y"],
                    list("EIEIO"),
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.string())),
            ).values,
            np.array(["A", "E", "I", "O", "U", "Y", None]),
            False,
            True,
            pd.Series(
                [
                    list("AEI"),
                    None,
                    list("AI"),
                    [],
                    None,
                    ["A", None, "E", "I", "Y"],
                    list("EIO"),
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.string())),
            ),
            id="string_array-vector-scalar",
        ),
        pytest.param(
            pd.array(
                [[i, i**2, None][i % 3] for i in range(20)], dtype=pd.Int32Dtype()
            ),
            pd.array([i**2 for i in range(100)], dtype=pd.Int32Dtype()),
            True,
            True,
            pd.array(
                [0, 1, 16, 49, 9, 100, 169, 256, 361],
                dtype=pd.Int32Dtype(),
            ),
            id="int_array-scalar-scalar",
        ),
        pytest.param(
            pd.Series(
                [[1], [], None, [1, 2], [1, 2, 3]],
                dtype=pd.ArrowDtype(pa.large_list(pa.int32())),
            ).values,
            pd.Series(
                [
                    [[1], [2], [3], None, [4], [None]],
                    [[], [1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
                    [[1, 2], [2, 3], None, [2, 1], [3, 2], [1, 3], None, [3, 1]],
                    [[1], [1, 2], [1], [1, 2, 3], [1]],
                    [],
                    [[4], [], [1, 2, 3]],
                    None,
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.large_list(pa.int32()))),
            ).values,
            True,
            False,
            pd.Series(
                [
                    [[1], None],
                    [[1], [], [1, 2], [1, 2, 3]],
                    [None, [1, 2]],
                    [[1], [1, 2], [1, 2, 3]],
                    [],
                    [[], [1, 2, 3]],
                    None,
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.large_list(pa.int32()))),
            ),
            id="int_array_array-scalar-vector",
        ),
        pytest.param(
            pd.Series(
                [[{"A": -1, "B": ""}, {"A": 3, "B": "xxx"}] * 3] * 40
                + [
                    [
                        {"A": 0, "B": "foo"},
                        None,
                        {"A": 1, "B": "bar"},
                        None,
                        {"A": 2, "B": "fizz"},
                    ],
                    [{"A": 3, "B": "buzz"}, {"A": 4, "B": "foo"}],
                    [{"A": 5, "B": "bar"}, {"A": 5, "B": "foo"}, {"A": 5, "B": "foo"}],
                    [
                        {"A": 6, "B": "fizz"},
                        {"A": 7, "B": "buzz"},
                        None,
                        None,
                        {"A": 8, "B": "foo"},
                    ],
                    [
                        {"A": 9, "B": "bar"},
                        {"A": 9, "B": "fizz"},
                        {"A": 0, "B": "fizz"},
                        {"A": 9, "B": "fizz"},
                        {"A": 0, "B": "bar"},
                    ],
                    [],
                    [{"A": 1, "B": "buzz"}],
                ],
                dtype=pd.ArrowDtype(
                    pa.large_list(
                        pa.struct(
                            [
                                pa.field("A", pa.int32()),
                                pa.field("B", pa.string()),
                            ]
                        )
                    )
                ),
            ).values,
            pd.Series(
                [[{"A": -1, "B": ""}, {"A": -1, "B": ""}, {"A": 3, "B": "xxx"}]] * 40
                + [
                    [{"A": 0, "B": "foo"}],
                    [],
                    [{"A": 5, "B": "foo"}, None],
                    [None, {"A": 7, "B": "buzz"}],
                    [
                        {"A": 3, "B": "bar"},
                        None,
                        {"A": 0, "B": "bar"},
                        {"A": 9, "B": "fizz"},
                    ],
                    [{"A": 0, "B": "foo"}, {"A": 1, "B": "bar"}],
                    [{"A": 1, "B": "fizz"}, {"A": 1, "B": "buzz"}],
                ],
                dtype=pd.ArrowDtype(
                    pa.large_list(
                        pa.struct(
                            [
                                pa.field("A", pa.int32()),
                                pa.field("B", pa.string()),
                            ]
                        )
                    )
                ),
            ).values,
            False,
            False,
            pd.Series(
                [[{"A": -1, "B": ""}, {"A": 3, "B": "xxx"}, {"A": -1, "B": ""}]] * 40
                + [
                    [{"A": 0, "B": "foo"}],
                    [],
                    [{"A": 5, "B": "foo"}],
                    [{"A": 7, "B": "buzz"}, None],
                    [{"A": 9, "B": "fizz"}, {"A": 0, "B": "bar"}],
                    [],
                    [{"A": 1, "B": "buzz"}],
                ],
                dtype=pd.ArrowDtype(
                    pa.large_list(
                        pa.struct(
                            [
                                pa.field("A", pa.int32()),
                                pa.field("B", pa.string()),
                            ]
                        )
                    )
                ),
            ),
            id="struct_array-vector-vector",
        ),
        pytest.param(
            pd.Series(
                [[{"A": [0], "B": None, "C": [1, 2]}, {"D": [3]}]] * 40
                + [
                    [],
                    [
                        {"E": [4], "F": [5, 6]},
                        {"A": [7]},
                        None,
                        {"B": [8]},
                        {"A": [7]},
                        {"A": [7]},
                    ],
                    [{"D": [3]}, None, None, {"D": [3]}, None, {}],
                    [{}, {"D": [3], "A": [7]}, {}, None, {"D": [3, None]}],
                    None,
                    [{"A": [0, 1], "D": [3]}],
                ],
                dtype=pd.ArrowDtype(
                    pa.large_list(pa.map_(pa.large_string(), pa.large_list(pa.int8())))
                ),
            ).values,
            pd.Series(
                [[{"A": [7]}, {}, {"D": [3]}, None, None, {"A": [7]}]] * 46,
                dtype=pd.ArrowDtype(
                    pa.large_list(pa.map_(pa.large_string(), pa.large_list(pa.int8())))
                ),
            ).values,
            False,
            False,
            pd.Series(
                [[{"D": [3]}]] * 40
                + [
                    [],
                    [{"A": [7]}, None, {"A": [7]}],
                    [{"D": [3]}, None, None, {}],
                    [{}, None],
                    None,
                    [],
                ],
                dtype=pd.ArrowDtype(
                    pa.large_list(pa.map_(pa.large_string(), pa.large_list(pa.int8())))
                ),
            ),
            id="map_array-vector-vector",
        ),
    ],
)
def test_array_intersection(
    arr_0,
    arr_1,
    arr_is_scalar,
    to_remove_is_scalar,
    answer,
    memory_leak_check,
):
    either_scalar = arr_is_scalar or to_remove_is_scalar
    both_scalar = arr_is_scalar and to_remove_is_scalar
    if not both_scalar:

        def impl(arr_0, arr_1):
            return pd.DataFrame(
                {
                    "answer": pd.Series(
                        bodosql.kernels.array_intersection(
                            arr_0, arr_1, arr_is_scalar, to_remove_is_scalar
                        )
                    )
                }
            )

        answer = pd.DataFrame({"answer": answer})

    else:

        def impl(arr_0, arr_1):
            return bodosql.kernels.array_intersection(
                arr_0, arr_1, arr_is_scalar, to_remove_is_scalar
            )

    check_func(
        impl,
        (arr_0, arr_1),
        py_output=answer,
        check_dtype=False,
        check_names=False,
        distributed=not either_scalar,
        is_out_distributed=not either_scalar,
        dist_test=not either_scalar,
        only_seq=either_scalar,
        convert_columns_to_pandas=True,
    )


@pytest.mark.parametrize("flag0", [True, False])
@pytest.mark.parametrize("flag1", [True, False])
@pytest.mark.slow
def test_option_array_intersection(flag0, flag1, memory_leak_check):
    def impl(arr, to_remove, flag0, flag1):
        arg0 = None if flag0 else arr
        arg1 = None if flag1 else to_remove
        return bodosql.kernels.array_intersection(
            arg0, arg1, is_scalar_0=True, is_scalar_1=True
        )

    arr = pd.array([1, 1, 1, 2, 2, 3, None, None, None, 4, 4, 5], dtype=pd.Int32Dtype())
    to_remove = pd.array([1, None, 2, None, 3], dtype=pd.Int32Dtype())
    answer = (
        None
        if flag0 or flag1
        else pd.array([1, 2, 3, None, None], dtype=pd.Int32Dtype())
    )

    check_func(
        impl,
        (arr, to_remove, flag0, flag1),
        py_output=answer,
        check_dtype=False,
        check_names=False,
        distributed=False,
        is_out_distributed=False,
        dist_test=False,
        only_seq=True,
    )


@pytest.mark.parametrize(
    "arr_0, arr_1, is_scalar_0, is_scalar_1, answer",
    [
        pytest.param(
            pd.Series(
                [[1, None]] * 40
                + [
                    [],
                    None,
                    [2],
                    [None],
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.int32())),
            ).values,
            pd.Series(
                [[2, None]] * 40
                + [
                    [3, 4],
                    [5],
                    None,
                    [],
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.int32())),
            ).values,
            False,
            False,
            pd.Series(
                [[1, None, 2, None]] * 40 + [[3, 4], None, None, [None]],
                dtype=pd.ArrowDtype(pa.large_list(pa.int32())),
            ),
            id="int_array-vector-vector",
        ),
        pytest.param(
            pd.Series(
                [["B"]] * 40 + [["AB", "CDE"], [], None, [None, "Q", "RST"]],
                dtype=pd.ArrowDtype(pa.large_list(pa.string())),
            ).values,
            np.array(["A", None]),
            False,
            True,
            pd.Series(
                [["B", "A", None]] * 40
                + [
                    ["AB", "CDE", "A", None],
                    ["A", None],
                    None,
                    [None, "Q", "RST", "A", None],
                ]
            ),
            id="string_array-vector-scalar",
        ),
        pytest.param(
            pd.array([0, 1, 2], dtype=pd.Int32Dtype()),
            pd.array([3, None, 4], dtype=pd.Int32Dtype()),
            True,
            True,
            pd.array([0, 1, 2, 3, None, 4], dtype=pd.Int32Dtype()),
            id="int_array-scalar-scalar",
        ),
        pytest.param(
            pd.Series(
                [[1], [None], [2, 3]], dtype=pd.ArrowDtype(pa.large_list(pa.int32()))
            ).values,
            pd.Series(
                [[[4]]] * 40 + [[], None, [[5], [], [6]], [None, [7, 8, 9]]],
                dtype=pd.ArrowDtype(pa.large_list(pa.large_list(pa.int32()))),
            ).values,
            True,
            False,
            pd.Series(
                [[[1], [None], [2, 3], [4]]] * 40
                + [
                    [[1], [None], [2, 3]],
                    None,
                    [[1], [None], [2, 3], [5], [], [6]],
                    [[1], [None], [2, 3], None, [7, 8, 9]],
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.large_list(pa.int32()))),
            ),
            id="int_array_array-scalar-vector",
        ),
        pytest.param(
            pd.Series(
                [[{"A": 0, "B": "B1"}, {"A": 1, "B": "C4"}]] * 40
                + [
                    [],
                    None,
                    [None],
                    [{"A": 5, "B": "A1"}, None, None],
                ],
                dtype=pd.ArrowDtype(
                    pa.large_list(
                        pa.struct(
                            [
                                pa.field("A", pa.int32()),
                                pa.field("B", pa.string()),
                            ]
                        )
                    )
                ),
            ).values,
            pd.Series(
                [[{"A": 2, "B": "A8"}, {"A": 3, "B": "H4"}]] * 40
                + [
                    [None, {"A": 6, "B": "D6"}, None],
                    [{"A": 7, "B": "E1"}, None],
                    [],
                    None,
                ],
                dtype=pd.ArrowDtype(
                    pa.large_list(
                        pa.struct(
                            [
                                pa.field("A", pa.int32()),
                                pa.field("B", pa.string()),
                            ]
                        )
                    )
                ),
            ).values,
            False,
            False,
            pd.Series(
                [
                    [
                        {"A": 0, "B": "B1"},
                        {"A": 1, "B": "C4"},
                        {"A": 2, "B": "A8"},
                        {"A": 3, "B": "H4"},
                    ]
                ]
                * 40
                + [[None, {"A": 6, "B": "D6"}, None], None, [None], None],
                dtype=pd.ArrowDtype(
                    pa.large_list(
                        pa.struct(
                            [
                                pa.field("A", pa.int32()),
                                pa.field("B", pa.string()),
                            ]
                        )
                    )
                ),
            ),
            id="struct_array-vector-vector",
        ),
        pytest.param(
            pd.Series(
                [[{"RS": [6, 7, None, 9]}]] * 40
                + [
                    None,
                    [],
                    [None, {"A": None, "B": [None]}],
                ],
                dtype=pd.ArrowDtype(
                    pa.large_list(pa.map_(pa.large_string(), pa.large_list(pa.int8())))
                ),
            ).values,
            pd.Series(
                [[{"A": [7]}, {"D": [3]}, None, None]] * 43,
                dtype=pd.ArrowDtype(
                    pa.large_list(pa.map_(pa.large_string(), pa.large_list(pa.int8())))
                ),
            ).values,
            False,
            False,
            pd.Series(
                [[{"RS": [6, 7, None, 9]}, {"A": [7]}, {"D": [3]}, None, None]] * 40
                + [
                    None,
                    [{"RS": [6, 7, None, 9]}],
                    [{"RS": [6, 7, None, 9]}, None, {"A": None, "B": [None]}],
                ],
                dtype=pd.ArrowDtype(
                    pa.large_list(pa.map_(pa.large_string(), pa.large_list(pa.int8())))
                ),
            ),
            id="map_array-vector-vector",
            marks=pytest.mark.skip(
                reason="[BSE-2126] fix value error with writing to column of arrays of map arrays of integer arrays"
            ),
        ),
    ],
)
def test_array_cat(
    arr_0,
    arr_1,
    is_scalar_0,
    is_scalar_1,
    answer,
    memory_leak_check,
):
    either_scalar = is_scalar_0 or is_scalar_1
    both_scalar = is_scalar_0 and is_scalar_1
    if not both_scalar:

        def impl(arr_0, arr_1):
            return pd.Series(
                bodosql.kernels.array_cat(arr_0, arr_1, is_scalar_0, is_scalar_1)
            )

    else:

        def impl(arr_0, arr_1):
            return bodosql.kernels.array_cat(arr_0, arr_1, is_scalar_0, is_scalar_1)

    check_func(
        impl,
        (arr_0, arr_1),
        py_output=answer,
        check_dtype=False,
        check_names=False,
        distributed=not either_scalar,
        is_out_distributed=not either_scalar,
        dist_test=not either_scalar,
        only_seq=either_scalar,
        convert_columns_to_pandas=True,
    )


@pytest.mark.parametrize("flag0", [True, False])
@pytest.mark.parametrize("flag1", [True, False])
@pytest.mark.slow
def test_option_array_cat(flag0, flag1, memory_leak_check):
    def impl(arr, to_remove, flag0, flag1):
        arg0 = None if flag0 else arr
        arg1 = None if flag1 else to_remove
        return bodosql.kernels.array_cat(arg0, arg1, is_scalar_0=True, is_scalar_1=True)

    arr = pd.array([1, 2, None, 1, 2, 3], dtype=pd.Int32Dtype())
    to_remove = pd.array([4, None], dtype=pd.Int32Dtype())
    answer = (
        None
        if flag0 or flag1
        else pd.array([1, 2, None, 1, 2, 3, 4, None], dtype=pd.Int32Dtype())
    )

    check_func(
        impl,
        (arr, to_remove, flag0, flag1),
        py_output=answer,
        check_dtype=False,
        check_names=False,
        distributed=False,
        is_out_distributed=False,
        dist_test=False,
        only_seq=True,
    )


@pytest.mark.parametrize(
    "arr, pos, arr_is_scalar, expected",
    [
        pytest.param(
            pd.Series(
                [
                    [0, 1, 4, 9, 16, 3],
                    None,
                    [2, None, 3, None, 5],
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.int32())),
            )
            .repeat(5)
            .values,
            pd.array([0, 10, None, -3, -10] * 3, dtype=pd.Int32Dtype()),
            False,
            pd.array(
                [
                    [1, 4, 9, 16, 3],
                    [0, 1, 4, 9, 16, 3],
                    None,
                    [0, 1, 4, 16, 3],
                    [0, 1, 4, 9, 16, 3],
                    None,
                    None,
                    None,
                    None,
                    None,
                    [None, 3, None, 5],
                    [2, None, 3, None, 5],
                    None,
                    [2, None, None, 5],
                    [2, None, 3, None, 5],
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.int32())),
            ),
            id="int_array-vector-vector",
        ),
        pytest.param(
            pd.array(
                [
                    list("ABCDEFGHIJABCDEFGHIJ"),
                    list("Alphabet Soup Is Delicious"),
                    "ALPHABET SOUP IS DELICIOUS".split(),
                    None,
                    ["A", None, "E", "AI", None, "OU", None],
                    list("EIEIO"),
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.string())),
            ),
            2,
            False,
            pd.array(
                [
                    list("ABDEFGHIJABCDEFGHIJ"),
                    list("Alhabet Soup Is Delicious"),
                    "ALPHABET SOUP DELICIOUS".split(),
                    None,
                    ["A", None, "AI", None, "OU", None],
                    list("EIIO"),
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.string())),
            ),
            id="string_array-vector-scalar",
        ),
        pytest.param(
            pd.array(
                [[1], [], None, [1, 2], [1, 2, 3]],
                pd.ArrowDtype(pa.large_list(pa.int32())),
            ),
            pd.array([0, 1, 2, 3, 10, None], dtype=pd.Int32Dtype()),
            True,
            pd.array(
                [
                    [[], None, [1, 2], [1, 2, 3]],
                    [[1], None, [1, 2], [1, 2, 3]],
                    [[1], [], [1, 2], [1, 2, 3]],
                    [[1], [], None, [1, 2, 3]],
                    [[1], [], None, [1, 2], [1, 2, 3]],
                    None,
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.large_list(pa.int32()))),
            ),
            id="int_array_array-scalar-vector",
        ),
        pytest.param(
            pd.array(
                [
                    [
                        {"A": 0, "B": "foo"},
                        None,
                        {"A": 1, "B": "bar"},
                        None,
                        {"A": 2, "B": "fizz"},
                    ],
                    [{"A": 3, "B": "buzz"}, {"A": 4, "B": "foo"}],
                    [{"A": 5, "B": "bar"}],
                    [
                        {"A": 6, "B": "fizz"},
                        None,
                        {"A": 7, "B": "buzz"},
                        None,
                        {"A": 8, "B": "foo"},
                    ],
                    [{"A": 9, "B": "bar"}, {"A": 0, "B": "fizz"}],
                    [],
                ]
                * 2,
                dtype=pd.ArrowDtype(
                    pa.large_list(
                        pa.struct(
                            [pa.field("A", pa.int32()), pa.field("B", pa.string())]
                        )
                    )
                ),
            ),
            pd.array(
                [
                    0,
                    1,
                    2,
                    None,
                    -2,
                    0,
                ]
                * 2,
                dtype=pd.Int32Dtype(),
            ),
            False,
            pd.array(
                [
                    [
                        None,
                        {"A": 1, "B": "bar"},
                        None,
                        {"A": 2, "B": "fizz"},
                    ],
                    [{"A": 3, "B": "buzz"}],
                    [{"A": 5, "B": "bar"}],
                    None,
                    [{"A": 0, "B": "fizz"}],
                    [],
                ]
                * 2,
                dtype=pd.ArrowDtype(
                    pa.large_list(
                        pa.struct(
                            [pa.field("A", pa.int32()), pa.field("B", pa.string())]
                        )
                    )
                ),
            ),
            id="struct_array-vector-vector",
        ),
        pytest.param(
            pd.array(
                [
                    [{"A": [0], "B": None, "C": [1, 2]}, {"D": [3]}],
                    [],
                    [{"E": [4], "F": [5, 6]}, {"A": [7]}, None, {"B": [8]}],
                    [{"D": [3]}, None, {}, {"D": [3]}, None],
                    [{"D": [3], "A": [7]}, None, {"D": [3, None]}],
                    None,
                    [{"A": [0, 1], "D": [3]}, {}],
                ],
                dtype=pd.ArrowDtype(
                    pa.large_list(pa.map_(pa.string(), pa.large_list(pa.int32())))
                ),
            ),
            pd.array([0, 1, 20, None, -3, -1, -10], dtype=pd.Int32Dtype()),
            False,
            pd.array(
                [
                    [{"D": [3]}],
                    [],
                    [{"E": [4], "F": [5, 6]}, {"A": [7]}, None, {"B": [8]}],
                    None,
                    [None, {"D": [3, None]}],
                    None,
                    [{"A": [0, 1], "D": [3]}, {}],
                ],
                dtype=pd.ArrowDtype(
                    pa.large_list(pa.map_(pa.string(), pa.large_list(pa.int32())))
                ),
            ),
            id="map_array-vector-vector",
        ),
    ],
)
def test_array_remove_at(
    arr,
    pos,
    arr_is_scalar,
    expected,
    memory_leak_check,
):
    check_func(
        lambda arr, pos: bodosql.kernels.array_remove_at(arr, pos, arr_is_scalar),
        (arr, pos),
        py_output=expected,
        check_dtype=False,
        distributed=not arr_is_scalar and is_array_typ(pos),
        is_out_distributed=not arr_is_scalar and is_array_typ(pos),
        dist_test=not arr_is_scalar and is_array_typ(pos),
    )


@pytest.mark.parametrize("flag0", [True, False])
@pytest.mark.parametrize("flag1", [True, False])
@pytest.mark.parametrize("flag2", [True, False])
@pytest.mark.slow
def test_option_array_slice(flag0, flag1, flag2, memory_leak_check):
    def impl(A, B, C, flag0, flag1, flag2):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        arg2 = C if flag2 else None
        return bodosql.kernels.array_slice(arg0, arg1, arg2, True)

    check_func(
        impl,
        (pd.array([0, 1, 4, 9], pd.Int32Dtype()), 1, 3, flag0, flag1, flag2),
        py_output=pd.array([1, 4], pd.Int32Dtype())
        if flag0 and flag1 and flag2
        else None,
        distributed=False,
        is_out_distributed=False,
        dist_test=False,
    )


@pytest.mark.parametrize(
    "arr, to_remove, arr_is_scalar, to_remove_is_scalar, expected",
    [
        pytest.param(
            pd.Series(
                [
                    [1, 1, 2, 1, 1, 2, 3, 2, 1, 4],
                    [0, 1, 4, 9, 16, 3],
                    None,
                    [2, None, 3, None, 5],
                    [1, 1, 1, 1],
                ]
            )
            .repeat(4)
            .values,
            pd.Series([1, 2, 3, None] * 5),
            False,
            False,
            pd.Series(
                [
                    [2, 2, 3, 2, 4],
                    [1, 1, 1, 1, 3, 1, 4],
                    [1, 1, 2, 1, 1, 2, 2, 1, 4],
                    None,
                    [0, 4, 9, 16, 3],
                    [0, 1, 4, 9, 16, 3],
                    [0, 1, 4, 9, 16],
                    None,
                    None,
                    None,
                    None,
                    None,
                    [2, None, 3, None, 5],
                    [None, 3, None, 5],
                    [2, None, None, 5],
                    None,
                    [],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    None,
                ]
            ),
            id="int_array-vector-vector",
        ),
        pytest.param(
            pd.array([[i, None][i % 2] for i in range(10)], pd.Int32Dtype()),
            2,
            True,
            True,
            pd.array([0, None, None] + [[i, None][i % 2] for i in range(4, 10)]),
            id="int_array-scalar-scalar",
        ),
        pytest.param(
            pd.array(
                [
                    list("ABCDEFGHIJABCDEFGHIJ"),
                    list("Alphabet Soup Is Delicious"),
                    "ALPHABET SOUP IS DELICIOUS".split(),
                    None,
                    ["A", None, "E", "AI", None, "OU", None],
                    list("EIEIO"),
                ],
                pd.ArrowDtype(pa.large_list(pa.string())),
            ),
            "A",
            False,
            True,
            pd.Series(
                [
                    list("BCDEFGHIJBCDEFGHIJ"),
                    list("lphabet Soup Is Delicious"),
                    "ALPHABET SOUP IS DELICIOUS".split(),
                    None,
                    [None, "E", "AI", None, "OU", None],
                    list("EIEIO"),
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.string())),
            ),
            id="string_array-vector-scalar",
        ),
        pytest.param(
            pd.array(
                [[1], [], None, [1, 2], [1, 2, 3]],
                pd.ArrowDtype(pa.large_list(pa.int32())),
            ),
            pd.Series([[1], [], None, [1, 2], [1, 2, 3], [2]]),
            True,
            False,
            pd.Series(
                [
                    [[], None, [1, 2], [1, 2, 3]],
                    [[1], None, [1, 2], [1, 2, 3]],
                    None,
                    [[1], [], None, [1, 2, 3]],
                    [[1], [], None, [1, 2]],
                    [[1], [], None, [1, 2], [1, 2, 3]],
                ]
            ),
            id="int_array_array-scalar-vector",
        ),
        pytest.param(
            pd.array(
                [
                    [
                        {"A": 0, "B": "foo"},
                        None,
                        {"A": 1, "B": "bar"},
                        None,
                        {"A": 2, "B": "fizz"},
                    ],
                    [{"A": 3, "B": "buzz"}, {"A": 4, "B": "foo"}],
                    [{"A": 5, "B": "bar"}],
                    [
                        {"A": 6, "B": "fizz"},
                        None,
                        {"A": 7, "B": "buzz"},
                        None,
                        {"A": 8, "B": "foo"},
                    ],
                    [{"A": 9, "B": "bar"}, {"A": 0, "B": "fizz"}],
                    [],
                ]
                * 2,
                dtype=pd.ArrowDtype(
                    pa.large_list(
                        pa.struct(
                            [pa.field("A", pa.int32()), pa.field("B", pa.string())]
                        )
                    )
                ),
            ),
            pd.array(
                [
                    {"A": 0, "B": "foo"},
                    {"A": 5, "B": "buzz"},
                    {"A": 5, "B": "bar"},
                    None,
                    {"A": 9, "B": "buzz"},
                    {"A": 9, "B": "bar"},
                ]
                * 2,
                dtype=pd.ArrowDtype(
                    pa.struct([pa.field("A", pa.int32()), pa.field("B", pa.string())])
                ),
            ),
            False,
            False,
            pd.Series(
                [
                    [
                        None,
                        {"A": 1, "B": "bar"},
                        None,
                        {"A": 2, "B": "fizz"},
                    ],
                    [{"A": 3, "B": "buzz"}, {"A": 4, "B": "foo"}],
                    [],
                    None,
                    [{"A": 9, "B": "bar"}, {"A": 0, "B": "fizz"}],
                    [],
                ]
                * 2,
                dtype=pd.ArrowDtype(
                    pa.large_list(
                        pa.struct(
                            [pa.field("A", pa.int32()), pa.field("B", pa.string())]
                        )
                    )
                ),
            ),
            id="struct_array-vector-vector",
        ),
        pytest.param(
            pd.array(
                [
                    [{"A": [0], "B": None, "C": [1, 2]}, {"D": [3]}],
                    [],
                    [{"E": [4], "F": [5, 6]}, {"A": [7]}, None, {"B": [8]}],
                    [{"D": [3]}, None, {}, {"D": [3]}, None],
                    [{"D": [3], "A": [7]}, None, {"D": [3, None]}],
                    None,
                    [{"A": [0, 1], "D": [3]}, {}],
                ]
                * 2,
                dtype=pd.ArrowDtype(
                    pa.large_list(pa.map_(pa.string(), pa.large_list(pa.int32())))
                ),
            ),
            pd.array(
                [
                    {"A": [0], "B": None, "C": [1, 2]},
                    {"A": [0], "B": None, "C": [1, 2]},
                    None,
                    {"D": [3]},
                    {"D": None, "A": [7]},
                    None,
                    {},
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.large_list(pa.int32()))),
            ),
            False,
            False,
            pd.array(
                [
                    [{"D": [3]}],
                    [],
                    None,
                    [None, {}, None],
                    [{"D": [3], "A": [7]}, None, {"D": [3, None]}],
                    None,
                    [{"A": [0, 1], "D": [3]}],
                ]
                * 2,
                dtype=pd.ArrowDtype(
                    pa.large_list(pa.map_(pa.string(), pa.large_list(pa.int32())))
                ),
            ),
            id="map_array-vector-vector",
            marks=pytest.mark.skip(
                reason="TODO: Support DictType for semi_safe_equals"
            ),
        ),
    ],
)
def test_array_remove(
    arr,
    to_remove,
    arr_is_scalar,
    to_remove_is_scalar,
    expected,
    memory_leak_check,
):
    both_scalar = arr_is_scalar and to_remove_is_scalar
    no_scalar = not arr_is_scalar and not to_remove_is_scalar
    if both_scalar:

        def impl(arr, to_remove):
            return bodosql.kernels.array_remove(
                arr, to_remove, arr_is_scalar, to_remove_is_scalar
            )

    else:

        def impl(arr, to_remove):
            return pd.Series(
                bodosql.kernels.array_remove(
                    arr, to_remove, arr_is_scalar, to_remove_is_scalar
                )
            )

    check_func(
        impl,
        (arr, to_remove),
        py_output=expected,
        check_dtype=False,
        distributed=no_scalar,
        is_out_distributed=no_scalar,
        dist_test=no_scalar,
    )


@pytest.mark.parametrize(
    "arr, is_scalar, expected",
    [
        pytest.param(1, True, pd.array([1]), id="scalar_integer"),
        pytest.param(
            pd.Series([1, 2, None, 3, 4, None, 5], dtype=pd.Int64Dtype()),
            False,
            pd.Series(
                [[1], [2], None, [3], [4], None, [5]],
                dtype=pd.ArrowDtype(pa.large_list(pa.int32())),
            ),
            id="vector_integer",
        ),
        pytest.param(
            pd.Series([-253.123, None, 534.958, -4.37, 0.9305] * 4),
            False,
            pd.Series(
                [
                    pd.array([-253.123]),
                    None,
                    pd.array([534.958]),
                    pd.array([-4.37]),
                    pd.array([0.9305]),
                ]
                * 4,
                dtype=pd.ArrowDtype(pa.large_list(pa.float64())),
            ),
            id="vector_float",
        ),
        pytest.param(
            pd.Series(["asfdav", "1423", "!@#$", None, "0.9305"] * 4),
            False,
            pd.Series(
                [
                    pd.array(["asfdav"], dtype="string[pyarrow]"),
                    pd.array(["1423"], dtype="string[pyarrow]"),
                    pd.array(["!@#$"], dtype="string[pyarrow]"),
                    None,
                    pd.array(["0.9305"], dtype="string[pyarrow]"),
                ]
                * 4,
                dtype=pd.ArrowDtype(pa.large_list(pa.string())),
            ),
            id="vector_string",
        ),
        pytest.param(
            bodo.types.Time(18, 32, 59),
            True,
            pd.array([bodo.types.Time(18, 32, 59)]),
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
            False,
            pd.Series(
                [
                    pd.array([datetime.date(2016, 3, 3)]),
                    pd.array([datetime.date(2012, 6, 18)]),
                    pd.array([datetime.date(1997, 1, 14)]),
                    None,
                    pd.array([datetime.date(2025, 1, 28)]),
                ]
                * 4,
                dtype=pd.ArrowDtype(pa.large_list(pa.date32())),
            ),
            id="vector_date",
        ),
        pytest.param(
            pd.Timestamp("2021-12-08"),
            True,
            np.array([pd.Timestamp("2021-12-08")], dtype="datetime64[ns]"),
            id="scalar_timestamp",
        ),
        pytest.param(
            pd.array([5, None, 1, 2, 3, 4], pd.Int64Dtype()),
            True,
            pd.array([5, None, 1, 2, 3, 4], pd.Int64Dtype()),
            id="scalar_int_array",
        ),
        pytest.param(
            pd.Series(
                [[1], [2, 3], [4, None], [None], None] * 2,
                dtype=pd.ArrowDtype(pa.large_list(pa.int32())),
            ),
            False,
            pd.Series(
                [[1], [2, 3], [4, None], [None], None] * 2,
                dtype=pd.ArrowDtype(pa.large_list(pa.int32())),
            ),
            id="vector_int_array",
        ),
        pytest.param(
            pd.Series(
                [
                    {"A": 0, "B": [1]},
                    None,
                    {"A": 0, "B": [1, 0]},
                    {"A": 0, "B": [0, 1]},
                ]
                * 2,
                dtype=pd.ArrowDtype(
                    pa.struct(
                        [
                            pa.field("A", pa.int32()),
                            pa.field("B", pa.large_list(pa.int32())),
                        ]
                    )
                ),
            ),
            False,
            pd.Series(
                [
                    [{"A": 0, "B": [1]}],
                    None,
                    [{"A": 0, "B": [1, 0]}],
                    [{"A": 0, "B": [0, 1]}],
                ]
                * 2,
                dtype=pd.ArrowDtype(
                    pa.large_list(
                        pa.struct(
                            [
                                pa.field("A", pa.int32()),
                                pa.field("B", pa.large_list(pa.int32())),
                            ]
                        )
                    )
                ),
            ),
            id="vector_struct",
        ),
        pytest.param(
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
                dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.string())),
            ),
            False,
            pd.Series(
                [
                    [{"name": "pomegranate"}],
                    [{}],
                    [{"hex": "660c21"}],
                    [{"hex": "660c21", "name": "red"}],
                    [{"hex": "#660c21", "name": "pomegranate"}],
                    [{"hex": "660c21", "name": "pomegranate"}],
                    None,
                    [{"hex": "660c21", "name": "pomegranate"}],
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.map_(pa.string(), pa.string()))),
            ),
            id="vector_map",
        ),
    ],
)
def test_to_array(arr, is_scalar, expected, memory_leak_check):
    if is_scalar:

        def impl(arr):
            return bodosql.kernels.to_array(arr, is_scalar)

    else:

        def impl(arr):
            return pd.Series(bodosql.kernels.to_array(arr, is_scalar))

    check_func(
        impl,
        (arr,),
        py_output=expected,
        check_dtype=False,
        distributed=not is_scalar,
        is_out_distributed=not is_scalar,
        dist_test=not is_scalar,
    )


@pytest.mark.parametrize(
    "arg0, arg1, is_scalar_0, is_scalar_1, answer",
    [
        pytest.param(
            pd.Series(
                pd.Series(
                    [[None], [1], [1, 2, 3], None, [None, 3]],
                    dtype=pd.ArrowDtype(pa.large_list(pa.int64())),
                )
                .repeat(5)
                .values
            ),
            pd.Series(
                [[], [1], [5, 3, 0], None, [None, 4]] * 5,
                dtype=pd.ArrowDtype(pa.large_list(pa.int16())),
            ),
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
                pd.Series(
                    [[None], [""], ["", "A", "BC"], None, [None, "BC"]],
                    dtype=pd.ArrowDtype(pa.large_list(pa.string())),
                )
                .repeat(5)
                .values
            ),
            pd.Series(
                [[], [""], ["GHIJ", "BC", "KLMNO"], None, [None, "DEF"]] * 5,
                dtype=pd.ArrowDtype(pa.large_list(pa.string())),
            ),
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
                pd.Series(
                    [[[1]], None, [[2], [None]], [], [[1], [None], [0]]],
                    dtype=pd.ArrowDtype(pa.large_list(pa.large_list(pa.int8()))),
                )
                .repeat(5)
                .values
            ),
            pd.Series(
                [[[1]], None, [[None], [2]], [], [[None], [1], [0]]] * 5,
                dtype=pd.ArrowDtype(pa.large_list(pa.large_list(pa.int8()))),
            ),
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
                ],
                dtype=pd.ArrowDtype(
                    pa.large_list(
                        pa.struct(
                            [
                                pa.field("A", pa.int32()),
                                pa.field("B", pa.large_list(pa.int32())),
                            ]
                        )
                    )
                ),
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
                * 5,
                dtype=pd.ArrowDtype(
                    pa.large_list(
                        pa.struct(
                            [
                                pa.field("A", pa.int32()),
                                pa.field("B", pa.large_list(pa.int32())),
                            ]
                        )
                    )
                ),
            ).values,
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
                [
                    [{"A": 0, "B": 1}, {"A": 0, "B": 1, "C": 2}],
                    [{}, {"A": 0}],
                    [
                        {"A": 1, "B": 0},
                    ],
                    None,
                    [{}, {"B": 1, "A": 0}],
                ],
                dtype=pd.ArrowDtype(
                    pa.large_list(pa.map_(pa.large_string(), pa.int16()))
                ),
            )
            .repeat(5)
            .values,
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
                * 5,
                dtype=pd.ArrowDtype(
                    pa.large_list(pa.map_(pa.large_string(), pa.int16()))
                ),
            ).values,
            False,
            False,
            pd.Series(
                [True, False, False, None, True]
                + [False, True, False, None, True]
                + [False, False, True, None, False]
                + [None] * 5
                + [True, True, False, None, True]
            ),
            id="nested_map_arrays-vector",
        ),
        pytest.param(
            np.array([1, 2, 4, 8, 16]),
            np.array([None, 3, 9, 27]),
            True,
            True,
            False,
            id="int_arrays-scalar_no_match",
        ),
        pytest.param(
            np.array([1, 2, 4, 8, 16]),
            np.array([None, 3, 9, 8, 27]),
            True,
            True,
            True,
            id="int_arrays-scalar_match",
        ),
        pytest.param(
            np.array([1, 2, 4, None, 16]),
            np.array([None, 3, 9, 27]),
            True,
            True,
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
            pd.Series([False, True, True, True, False, True, False, True]),
            id="int_arrays-scalar_vector",
        ),
    ],
)
def test_arrays_overlap(
    arg0, arg1, is_scalar_0, is_scalar_1, answer, memory_leak_check
):
    both_scalar = is_scalar_0 and is_scalar_1
    either_scalar = is_scalar_0 or is_scalar_1
    if both_scalar:

        def impl(arg0, arg1):
            return bodosql.kernels.arrays_overlap(arg0, arg1, is_scalar_0, is_scalar_1)

    else:

        def impl(arg0, arg1):
            return pd.Series(
                bodosql.kernels.arrays_overlap(arg0, arg1, is_scalar_0, is_scalar_1)
            )

    check_func(
        impl,
        (arg0, arg1),
        py_output=answer,
        distributed=not either_scalar,
        is_out_distributed=not either_scalar,
        dist_test=not either_scalar,
        only_seq=either_scalar,
    )


@pytest.mark.parametrize(
    "arr, is_scalar, expected",
    [
        pytest.param(
            pd.array([None, 1, 2, None, 3, None, None], pd.Int32Dtype()),
            True,
            pd.array([1, 2, 3], pd.Int32Dtype()),
            id="int_arrays-scalar",
        ),
        pytest.param(
            pd.Series(
                [
                    [None, 1, 2, None, 3, None, None],
                    [1, None, None, 3],
                    [None, None, None],
                    [],
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.large_list(pa.int32())),
            ),
            False,
            pd.Series([[1, 2, 3], [1, 3], [], [], None] * 2),
            id="int_arrays-vector",
        ),
        pytest.param(
            pd.Series(
                [
                    [None, "ABC", "", None, "DEF", None, None],
                    ["gnls", None, None, "hello"],
                    [None, None, None],
                    [],
                    ["", ""],
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.large_list(pa.string())),
            ),
            False,
            pd.Series(
                [["ABC", "", "DEF"], ["gnls", "hello"], [], [], ["", ""], None] * 2,
                dtype=pd.ArrowDtype(pa.large_list(pa.string())),
            ),
            id="string_arrays-vector",
        ),
        pytest.param(
            pd.Series(
                [
                    [None, [1], [2, 3], None, [None], None, None],
                    [[4, 6, None], None, None, []],
                    [None, None, None],
                    [],
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.large_list(pa.large_list(pa.int32()))),
            ),
            False,
            pd.Series(
                [[[1], [2, 3], [None]], [[4, 6, None], []], [], [], None] * 2,
                dtype=pd.ArrowDtype(pa.large_list(pa.large_list(pa.int32()))),
            ),
            id="nested_int_arrays-vector",
        ),
        pytest.param(
            pd.Series(
                [
                    [{"A": 0, "B": [1]}],
                    [None, {"A": 0, "B": [1, 0]}],
                    [None],
                    [],
                    [{"A": 0, "B": [0, 1]}, None],
                    None,
                    [{"A": 0, "B": [1]}, None, {"A": 1, "B": [0, 1]}],
                ],
                dtype=pd.ArrowDtype(
                    pa.large_list(
                        pa.struct(
                            [
                                pa.field("A", pa.int32()),
                                pa.field("B", pa.large_list(pa.int32())),
                            ]
                        )
                    )
                ),
            ),
            False,
            pd.Series(
                [
                    [{"A": 0, "B": [1]}],
                    [{"A": 0, "B": [1, 0]}],
                    [],
                    [],
                    [{"A": 0, "B": [0, 1]}],
                    None,
                    [{"A": 0, "B": [1]}, {"A": 1, "B": [0, 1]}],
                ],
                dtype=pd.ArrowDtype(
                    pa.large_list(
                        pa.struct(
                            [
                                pa.field("A", pa.int32()),
                                pa.field("B", pa.large_list(pa.int32())),
                            ]
                        )
                    )
                ),
            ),
            id="nested_struct_arrays-vector",
        ),
        pytest.param(
            pd.Series(
                [
                    [{"A": [0], "B": [1]}],
                    [None, {"A": [0], "C": [1, 0]}],
                    [None],
                    [{"E": [0], "": [0, 1], "H": [None]}, None],
                    [],
                    None,
                    [{"A": [4, 5], "K": [1]}, None, {"ok": [1, None], "B": [0, 1]}],
                ],
                dtype=pd.ArrowDtype(
                    pa.large_list(pa.map_(pa.string(), pa.large_list(pa.int32())))
                ),
            ),
            False,
            pd.Series(
                [
                    [{"A": [0], "B": [1]}],
                    [{"A": [0], "C": [1, 0]}],
                    [],
                    [{"E": [0], "": [0, 1], "H": [None]}],
                    [],
                    None,
                    [{"A": [4, 5], "K": [1]}, {"ok": [1, None], "B": [0, 1]}],
                ],
                dtype=pd.ArrowDtype(
                    pa.large_list(pa.map_(pa.string(), pa.large_list(pa.int32())))
                ),
            ),
            id="map_array-vector",
            marks=pytest.mark.skip(
                reason="[BSE-2126] fix value error with writing to column of arrays of map arrays of integer arrays"
            ),
        ),
    ],
)
def test_array_compact(arr, is_scalar, expected, memory_leak_check):
    if is_scalar:

        def impl(arr):
            return bodosql.kernels.array_compact(arr, True)

    else:

        def impl(arr):
            return pd.Series(bodosql.kernels.array_compact(arr, False))

    check_func(
        impl,
        (arr,),
        py_output=expected,
        check_dtype=False,
        distributed=not is_scalar,
        is_out_distributed=not is_scalar,
        dist_test=not is_scalar,
    )


@pytest.mark.parametrize(
    "elem, container, elem_is_scalar, container_is_scalar, answer",
    [
        pytest.param(
            pd.Series([1, 2, None] * 2, dtype=pd.Int32Dtype()).values,
            pd.Series(
                [[3, 1, 4, None, 2, 1]] * 3 + [[2, None, 2]] * 3,
                dtype=pd.ArrowDtype(pa.large_list(pa.int32())),
            ).values,
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
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.int32())),
            ).values,
            True,
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
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.int32())),
            ).values,
            True,
            False,
            pd.Series([0, 0, None, 20, None, 2], dtype=pd.Int32Dtype()),
            id="int-null_vector",
        ),
        pytest.param(
            16,
            np.array([0, 1, 4, 9, 16, 25]),
            True,
            True,
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
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.string())),
            ).values,
            True,
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
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.large_list(pa.int32()))),
            ),
            True,
            False,
            pd.Series([None, 3, None, 5, None, 0], dtype=pd.Int32Dtype()),
            id="int_array-scalar_vector",
        ),
        pytest.param(
            np.array([1, 2, 3]),
            pd.Series(
                [[], [1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
                dtype=pd.ArrowDtype(pa.large_list(pa.int32())),
            ).values,
            True,
            True,
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
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.int32())),
            ).values,
            pd.Series(
                [[], [1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
                dtype=pd.ArrowDtype(pa.large_list(pa.int32())),
            ).values,
            False,
            True,
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
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.int32())),
            ).values,
            pd.Series(
                [
                    [],
                    [[0, 1]],
                    [[], [1], [1, 2], [1, 2, 3]],
                    [[None], [1, 2, 3], [1, 2, 3]],
                    [[1, 2, 3, 4], [1, 2, 3], [1, 2], [1]],
                    [[1, 2, 3, 4], [2, 3, 4, 5]],
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.large_list(pa.int32()))),
            ).values,
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
                ],
                dtype=pd.ArrowDtype(
                    pa.large_list(pa.large_list(pa.large_list(pa.string())))
                ),
            ).values,
            True,
            False,
            pd.Series([None, 0, None, 2, None], dtype=pd.Int32Dtype()),
            id="string_array_array-scalar_vector",
        ),
        pytest.param(
            pd.Series(
                [{"A": 0, "B": 1}] * 3 + [{"A": 1, "B": 0}] * 3,
                dtype=pd.ArrowDtype(
                    pa.struct([pa.field("A", pa.int32()), pa.field("B", pa.int32())])
                ),
            ),
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
                * 2,
                dtype=pd.ArrowDtype(
                    pa.large_list(
                        pa.struct(
                            [pa.field("A", pa.int32()), pa.field("B", pa.int32())]
                        )
                    )
                ),
            ),
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
                * 5,
                dtype=pd.ArrowDtype(
                    pa.large_list(
                        pa.struct(
                            [pa.field("A", pa.int32()), pa.field("B", pa.int32())]
                        )
                    )
                ),
            ).values,
            True,
            False,
            pd.Series([None, 1, None] * 5, dtype=pd.Int32Dtype()),
            id="struct-scalar_vector",
            marks=pytest.mark.skip(
                reason="[BSE-1781] TODO: fix array_construct and array_position when inputs are mix of struct arrays and scalars"
            ),
        ),
        pytest.param(
            pd.Series(
                [{"hex": "660c21", "name": "pomegranate"}] * 8,
                dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.string())),
            ),
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
                dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.string())),
            ).values,
            False,
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
                dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.string())),
            ).values,
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
    answer,
    memory_leak_check,
):
    # [BSE-1840] properly address distributed handling of array scalars
    all_scalar = elem_is_scalar and container_is_scalar
    any_scalar = elem_is_scalar or container_is_scalar
    if all_scalar:

        def impl(elem, container):
            return bodosql.kernels.array_position(
                elem, container, elem_is_scalar, container_is_scalar
            )

    else:

        def impl(elem, container):
            return pd.Series(
                bodosql.kernels.array_position(
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
    )


@pytest.mark.parametrize(
    "arr, from_, to, is_scalar, expected",
    [
        pytest.param(
            pd.array([1, 2, 3, None, 4, 5, None], pd.Int32Dtype()),
            1,
            5,
            True,
            pd.array([2, 3, None, 4], pd.Int32Dtype()),
            id="int-scalar-scalar-scalar",
        ),
        pytest.param(
            pd.array([1, 2, 3, None, 4, 5, None], pd.Int32Dtype()),
            pd.array([-10, -10, -12, 12, 10, -3], pd.Int32Dtype()),
            pd.array([-6, -12, -10, 10, 12, 10], pd.Int32Dtype()),
            True,
            pd.array(
                [[1], [], [], [], [], [4, 5, None]],
                pd.ArrowDtype(pa.large_list(pa.int32())),
            ),
            id="int_out_of_bound-scalar-vector-vector",
        ),
        pytest.param(
            pd.array(["foo", "bar", None, "ok", "a"], "string[pyarrow]"),
            1,
            4,
            True,
            pd.array(["bar", None, "ok"], "string[pyarrow]"),
            id="string-scalar-scalar-scalar",
        ),
        pytest.param(
            pd.Series([[1, 2], [3, None], None, [4], [None], [5, 6, 7]]).values,
            1,
            6,
            True,
            pd.Series([[3, None], None, [4], [None], [5, 6, 7]]).values,
            id="int_array-scalar-scalar-scalar",
        ),
        pytest.param(
            pd.Series([[[1, 2], [3, None], None, [4], [None], [5, 6, 7]]] * 16),
            pd.Series(
                [0, 1, 2, 3, 4, 5, 6] + [-5, 2, -4] + [-10, 8, -1, -20] + [2, None]
            ),
            pd.Series([6] * 7 + [3, -2, -2] + [-11, 9, 9, 2] + [None, 3]),
            False,
            pd.Series(
                [
                    [[1, 2], [3, None], None, [4], [None], [5, 6, 7]],
                    [[3, None], None, [4], [None], [5, 6, 7]],
                    [None, [4], [None], [5, 6, 7]],
                    [[4], [None], [5, 6, 7]],
                    [[None], [5, 6, 7]],
                    [[5, 6, 7]],
                    [],
                    [[3, None], None],
                    [None, [4]],
                    [None, [4]],
                    [],
                    [],
                    [[5, 6, 7]],
                    [[1, 2], [3, None]],
                    None,
                    None,
                ]
            ),
            id="int_array-vector-vector-vector",
        ),
        pytest.param(
            pd.Series(
                [
                    {"A": 0, "B": [0, 1]},
                    {"A": 1, "B": [1, 2]},
                    None,
                    {"A": 2, "B": [2, 3]},
                ]
            ).values,
            pd.Series([0, 1, 2, 3, 4]),
            4,
            True,
            pd.Series(
                [
                    [
                        {"A": 0, "B": [0, 1]},
                        {"A": 1, "B": [1, 2]},
                        None,
                        {"A": 2, "B": [2, 3]},
                    ],
                    [{"A": 1, "B": [1, 2]}, None, {"A": 2, "B": [2, 3]}],
                    [None, {"A": 2, "B": [2, 3]}],
                    [{"A": 2, "B": [2, 3]}],
                    [],
                ]
            ),
            id="struct-vector-vector-scalar",
        ),
        pytest.param(
            pd.Series(
                [
                    {"name": "pomegranate"},
                    {"hex": "660c21", "name": "pomegranate"},
                    {},
                    {"hex": "660c21"},
                    {"hex": "660c21", "name": "red"},
                    {"hex": "#660c21", "read": "pomegranate"},
                    None,
                    {"hex": "660c21", "name": "pomegranate"},
                ],
                dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.string())),
            ).values,
            0,
            pd.Series([1, 3, 5, 7, 9, 11, None]),
            True,
            pd.Series(
                [
                    [
                        {"name": "pomegranate"},
                    ],
                    [
                        {"name": "pomegranate"},
                        {"hex": "660c21", "name": "pomegranate"},
                        {},
                    ],
                    [
                        {"name": "pomegranate"},
                        {"hex": "660c21", "name": "pomegranate"},
                        {},
                        {"hex": "660c21"},
                        {"hex": "660c21", "name": "red"},
                    ],
                    [
                        {"name": "pomegranate"},
                        {"hex": "660c21", "name": "pomegranate"},
                        {},
                        {"hex": "660c21"},
                        {"hex": "660c21", "name": "red"},
                        {"hex": "#660c21", "read": "pomegranate"},
                        None,
                    ],
                    [
                        {"name": "pomegranate"},
                        {"hex": "660c21", "name": "pomegranate"},
                        {},
                        {"hex": "660c21"},
                        {"hex": "660c21", "name": "red"},
                        {"hex": "#660c21", "read": "pomegranate"},
                        None,
                        {"hex": "660c21", "name": "pomegranate"},
                    ],
                    [
                        {"name": "pomegranate"},
                        {"hex": "660c21", "name": "pomegranate"},
                        {},
                        {"hex": "660c21"},
                        {"hex": "660c21", "name": "red"},
                        {"hex": "#660c21", "read": "pomegranate"},
                        None,
                        {"hex": "660c21", "name": "pomegranate"},
                    ],
                    None,
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.map_(pa.string(), pa.string()))),
            ),
            id="map-vector-scalar-vector",
            marks=pytest.mark.skip(
                reason="[BSE-2126] fix value error with writing to column of arrays of map arrays"
            ),
        ),
    ],
)
def test_array_slice(arr, from_, to, is_scalar, expected, memory_leak_check):
    all_scalar = (
        is_scalar and not isinstance(from_, pd.Series) and not isinstance(to, pd.Series)
    )
    no_scalar = (
        not is_scalar and isinstance(from_, pd.Series) and isinstance(to, pd.Series)
    )
    if all_scalar:

        def impl(arr, from_, to):
            return bodosql.kernels.array_slice(arr, from_, to, is_scalar)

    else:

        def impl(arr, from_, to):
            return pd.Series(bodosql.kernels.array_slice(arr, from_, to, is_scalar))

    check_func(
        impl,
        (arr, from_, to),
        py_output=expected,
        distributed=no_scalar,
        is_out_distributed=no_scalar,
        dist_test=no_scalar,
    )


@pytest.mark.parametrize(
    "elem, container, elem_is_scalar, container_is_scalar, expected",
    [
        pytest.param(
            16,
            np.array([0, 1, 4, 9, 16, 25]),
            True,
            True,
            True,
            id="int-scalars",
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
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.int32())),
            ).values,
            True,
            False,
            pd.Series(
                [False, True, True, None, True, False, True], dtype=pd.BooleanDtype()
            ),
            id="int-scalar_vector",
        ),
        pytest.param(
            pd.Series([1, 2, None] * 2, dtype=pd.Int32Dtype()).values,
            pd.Series(
                [[3, 1, 4, None, 2, 1]] * 3 + [[2, 4]] * 3,
                dtype=pd.ArrowDtype(pa.large_list(pa.int32())),
            ).values,
            False,
            False,
            pd.Series([True, True, True, False, True, False]),
            id="int-vector_vector",
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
            pd.Series([True, True, False, True, None, True], dtype=pd.BooleanDtype()),
            id="int-null_vector",
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
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.string())),
            ).values,
            True,
            False,
            pd.Series(
                [True, False, False, True, None, True, None, True, False],
                dtype=pd.BooleanDtype(),
            ),
            id="string-scalar_vector",
        ),
        pytest.param(
            np.array([1, 2, 3]),
            pd.Series(
                [[], [1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
                dtype=pd.ArrowDtype(pa.large_list(pa.int32())),
            ).values,
            True,
            True,
            True,
            id="int_array-scalar_scalar",
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
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.large_list(pa.int32()))),
            ),
            True,
            False,
            pd.Series([False, True, False, True, None, True], dtype=pd.BooleanDtype()),
            id="int_array-scalar_vector",
        ),
        pytest.param(
            pd.Series(
                [[], [1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5], None],
                dtype=pd.ArrowDtype(pa.large_list(pa.int32())),
            ).values,
            pd.Series(
                [[], [1], [1, 2], [1, 2, 3], [1, 2, 3, 4]],
                dtype=pd.ArrowDtype(pa.large_list(pa.int32())),
            ).values,
            False,
            True,
            pd.Series(
                [True, True, True, True, True, False, False], dtype=pd.BooleanDtype()
            ),
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
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.int32())),
            ).values,
            pd.Series(
                [
                    [],
                    [[0, 1]],
                    [[], [1], [1, 2], [1, 2, 3]],
                    [[None], [1, 2, 3], [1, 2, 3]],
                    [[1, 2, 3, 4], [1, 2, 3], [1, 2], [1]],
                    [[1, 2, 3, 4], [2, 3, 4, 5]],
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.large_list(pa.int32()))),
            ).values,
            False,
            False,
            pd.Series([False, False, True, True, True, False], dtype=pd.BooleanDtype()),
            id="int_array-vector_vector",
        ),
        pytest.param(
            np.array([["A"]]),
            pd.Series(
                [
                    [[]],
                    [[["A"]]],
                    [[["A", "B"], ["A"]], [["B"]], []],
                    None,
                    [[["A", "B"]], [["A"], ["A", "C"]], [["A"]]],
                    [],
                ],
                dtype=pd.ArrowDtype(
                    pa.large_list(pa.large_list(pa.large_list(pa.string())))
                ),
            ).values,
            True,
            False,
            pd.Series([False, True, False, None, True, False], dtype=pd.BooleanDtype()),
            id="string_array_array-scalar_vector",
        ),
        pytest.param(
            pd.Series(
                ([{"A": 0, "B": 1}] * 3 + [{"A": 1, "B": 0}] * 3 + [None] * 3) * 2,
                dtype=pd.ArrowDtype(
                    pa.struct([pa.field("A", pa.int32()), pa.field("B", pa.int32())])
                ),
            ),
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
                * 6,
                dtype=pd.ArrowDtype(
                    pa.large_list(
                        pa.struct(
                            [pa.field("A", pa.int32()), pa.field("B", pa.int32())]
                        )
                    )
                ),
            ),
            False,
            False,
            pd.Series(
                [False, True, False, False, True, False, False, True, True] * 2,
                dtype=pd.BooleanDtype(),
            ),
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
                * 5,
                dtype=pd.ArrowDtype(
                    pa.large_list(
                        pa.struct(
                            [pa.field("A", pa.int32()), pa.field("B", pa.int32())]
                        )
                    )
                ),
            ).values,
            True,
            False,
            pd.Series([False, True, False] * 5, dtype=pd.BooleanDtype()),
            id="struct-scalar_vector",
            marks=pytest.mark.skip(
                reason="[BSE-1781] TODO: fix array_contains when inputs are mix of struct arrays and scalars"
            ),
        ),
        pytest.param(
            pd.Series(
                [
                    {"hex": "660c21", "name": "pomegranate"},
                    {},
                    None,
                    {"hex": "660c21", "name": "ok"},
                ]
                * 3,
                dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.string())),
            ),
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
                dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.string())),
            ).values,
            False,
            True,
            pd.Series([True, True, True, False] * 3, dtype=pd.BooleanDtype()),
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
                dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.string())),
            ).values,
            True,
            True,
            True,
            id="map-scalar_scalar",
        ),
    ],
)
def test_array_contains(
    elem,
    container,
    elem_is_scalar,
    container_is_scalar,
    expected,
    memory_leak_check,
):
    all_scalar = elem_is_scalar and container_is_scalar
    any_scalar = elem_is_scalar or container_is_scalar
    if all_scalar:

        def impl(elem, container):
            return bodosql.kernels.array_contains(
                elem, container, elem_is_scalar, container_is_scalar
            )

    else:

        def impl(elem, container):
            return pd.Series(
                bodosql.kernels.array_contains(
                    elem, container, elem_is_scalar, container_is_scalar
                )
            )

    check_func(
        impl,
        (elem, container),
        py_output=expected,
        distributed=not all_scalar,
        is_out_distributed=not all_scalar,
        dist_test=not any_scalar,
    )


@pytest.mark.parametrize("flag0", [True, False])
@pytest.mark.parametrize("flag1", [True, False])
@pytest.mark.slow
def test_option_array_contains(flag0, flag1, memory_leak_check):
    def impl(A, B, flag0, flag1):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        return bodosql.kernels.array_contains(arg0, arg1, True, True)

    A, B = 1, pd.array([0, 1, 4, 9])
    check_func(
        impl,
        (A, B, flag0, flag1),
        py_output=1 if flag0 and flag1 else None,
        only_seq=True,
    )


@pytest.mark.parametrize("flag0", [True, False])
@pytest.mark.parametrize("flag1", [True, False])
@pytest.mark.slow
def test_option_arrays_overlap(flag0, flag1, memory_leak_check):
    def impl(array_0, array_1, flag0, flag1):
        arg0 = None if flag0 else array_0
        arg1 = None if flag1 else array_1
        return bodosql.kernels.arrays_overlap(
            arg0, arg1, is_scalar_0=True, is_scalar_1=True
        )

    array_0 = pd.array([1, 2, None], pd.Int32Dtype())
    array_1 = pd.array([None, 3], pd.Int32Dtype())
    expected = None if flag0 or flag1 else True

    check_func(
        impl,
        (array_0, array_1, flag0, flag1),
        py_output=expected,
        check_dtype=False,
        distributed=False,
        is_out_distributed=False,
        dist_test=False,
    )


@pytest.mark.parametrize("flag0", [True, False])
@pytest.mark.parametrize("flag1", [True, False])
@pytest.mark.slow
def test_option_array_position(flag0, flag1, memory_leak_check):
    def impl(elem, container, flag0, flag1):
        arg0 = None if flag0 else elem
        arg1 = None if flag1 else container
        return bodosql.kernels.array_position(
            arg0, arg1, is_scalar_0=True, is_scalar_1=True
        )

    elem = 1
    container = pd.array([1, 2, None], pd.Int32Dtype())
    expected = None if flag1 else (2 if flag0 else 0)

    check_func(
        impl,
        (elem, container, flag0, flag1),
        py_output=expected,
        check_dtype=False,
        distributed=False,
        is_out_distributed=False,
        dist_test=False,
    )


@pytest.mark.parametrize("flag", [True, False])
@pytest.mark.slow
def test_option_array_size(flag, memory_leak_check):
    def impl(A, flag):
        arg0 = A if flag else None
        return bodosql.kernels.array_size(arg0, True)

    check_func(
        impl,
        (pd.array([0, 1, 4, 9], pd.Int32Dtype()), flag),
        py_output=4 if flag else None,
        distributed=False,
        is_out_distributed=False,
        dist_test=False,
    )


@pytest.mark.parametrize("flag0", [True, False])
@pytest.mark.parametrize("flag1", [True, False])
@pytest.mark.slow
def test_option_array_remove(flag0, flag1, memory_leak_check):
    def impl(A, B, flag0, flag1):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        return bodosql.kernels.array_remove(arg0, arg1, True, True)

    A, B = pd.array([0, 1, 4, 9], pd.Int32Dtype()), 0
    check_func(
        impl,
        (A, B, flag0, flag1),
        py_output=pd.array([1, 4, 9], pd.Int32Dtype()) if flag0 and flag1 else None,
        distributed=False,
        is_out_distributed=False,
        dist_test=False,
    )


@pytest.mark.parametrize(
    "array, separator, is_scalar_arr, answer",
    [
        pytest.param(
            np.array([4234, 401, -820]),
            "+",
            True,
            "4234+401+-820",
            id="int-scalar-scalar",
        ),
        pytest.param(
            np.array([4234, 401, -820]),
            pd.Series(["+", " - ", None] * 2),
            True,
            pd.Series(["4234+401+-820", "4234 - 401 - -820", None] * 2),
            marks=pytest.mark.slow,
            id="int-scalar-vector",
        ),
        pytest.param(
            pd.Series([[4234, 401, -820], [4234], [], None] * 2),
            "+",
            False,
            pd.Series(["4234+401+-820", "4234", "", None] * 2),
            marks=pytest.mark.slow,
            id="int-vector-scalar",
        ),
        pytest.param(
            pd.Series(
                [[4234, 401, -820], [4234, 401, -820], [4234]] * 5
                + [[], [], None, None]
            ),
            pd.Series(["+", "", " - "] * 5 + [" ", None, " + ", None]),
            False,
            pd.Series(
                ["4234+401+-820", "4234401-820", "4234"] * 5 + ["", None, None, None]
            ),
            marks=pytest.mark.slow,
            id="int-vector-vector",
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
            False,
            pd.Series(
                [
                    "-253.123000--534.958000--4.370000-0.930500",
                    "19.900000--235.104000-437.000000--0.295200",
                    "1312.242300-",
                    None,
                ]
                * 4
            ),
            id="float-vector-scalar",
        ),
        pytest.param(
            np.array([False, True, None, True]),
            "&",
            True,
            "false&true&&true",
            id="bool-scalar-scalar",
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
            False,
            pd.Series(
                [
                    "-253.123534.958-4.370.9305",
                    None,
                    "oneword",
                    "g0q0ejdifewkf%%@,..;BLSDF",
                ]
                * 4
            ),
            id="string-vector-scalar",
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
            True,
            "1932-10-05_2012-07-23_1999-03-15_2022-12-29",
            id="date-scalar-scalar",
        ),
        pytest.param(
            pd.Series([[1, 2], [3, None], None]).values,
            "_",
            True,
            "[[1, 2], [3, None], None]",
            id="int_array-scalar-scalar",
            marks=pytest.mark.skip(
                reason="TODO: Make TO_VARCHAR support semi-structured data types"
            ),
        ),
    ],
)
def test_array_to_string(array, separator, is_scalar_arr, answer, memory_leak_check):
    both_scalar = is_scalar_arr and not isinstance(separator, pd.Series)
    no_scalar = not is_scalar_arr and isinstance(separator, pd.Series)
    if both_scalar:

        def impl(array, separator):
            return bodosql.kernels.array_to_string(array, separator, is_scalar_arr)

    else:

        def impl(array, separator):
            return pd.Series(
                bodosql.kernels.array_to_string(array, separator, is_scalar_arr)
            )

    check_func(
        impl,
        (array, separator),
        py_output=answer,
        distributed=no_scalar,
        is_out_distributed=no_scalar,
        dist_test=no_scalar,
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
        return pd.Series(bodosql.kernels.array_size(array, False))

    check_func(impl, (array,), py_output=answer, check_dtype=False)


@pytest.mark.parametrize(
    "array,answer",
    [
        pytest.param(pd.array(["A", "BC", None]), 3, id="null_string"),
        pytest.param(pd.array([1, 2, None, 3]), 4, id="null_int"),
        pytest.param(None, None, id="null"),
    ],
)
def test_array_size_scalar(array, answer, memory_leak_check):
    def impl(array):
        return bodosql.kernels.array_size(array, True)

    check_func(
        impl, (array,), py_output=answer, distributed=False, is_out_distributed=False
    )


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(
            pd.array(
                [{"A": [i, None], "B": None} for i in range(5)],
                dtype=pd.ArrowDtype(
                    pa.struct(
                        [
                            pa.field("A", pa.large_list(pa.int64())),
                            pa.field("B", pa.string()),
                        ]
                    )
                ),
            ),
            id="struct",
        ),
        pytest.param(
            pd.array(
                [{"A": i, "B": i + 1, "C": i + 2} for i in range(5)],
                dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int32())),
            ),
            id="map",
        ),
        pytest.param(pd.array([None] * 5, dtype=pd.ArrowDtype(pa.null())), id="null"),
    ],
)
def test_to_object_valid(data, memory_leak_check):
    def impl_vector(data):
        return pd.DataFrame({"res": bodosql.kernels.to_object(data)})

    def impl_scalar(data):
        return pd.DataFrame({"res": [bodosql.kernels.to_object(data[0])]})

    def impl_optional(data, flag):
        arg = None if flag else data[0]
        return pd.DataFrame({"res": [bodosql.kernels.to_object(arg)]})

    vector_res = pd.DataFrame({"res": pd.array(data, data.dtype)})
    scalar_res = pd.DataFrame({"res": pd.array([data[0]], data.dtype)})
    null_res = pd.DataFrame({"res": pd.array([None], data.dtype)})
    check_func(
        impl_vector,
        (data,),
        py_output=vector_res,
        convert_columns_to_pandas=True,
        check_dtype=False,
    )
    check_func(
        impl_scalar,
        (data,),
        py_output=scalar_res,
        only_seq=True,
        convert_columns_to_pandas=True,
        check_dtype=False,
    )
    check_func(
        impl_optional,
        (data, False),
        py_output=scalar_res,
        only_seq=True,
        convert_columns_to_pandas=True,
        check_dtype=False,
    )
    check_func(
        impl_optional,
        (data, True),
        py_output=null_res,
        only_seq=True,
        convert_columns_to_pandas=True,
        check_dtype=False,
    )


@pytest.mark.skipif(
    bodo.tests.utils.test_spawn_mode_enabled,
    reason="only_seq=True disables spawn testing so pytest.raises fails",
)
@pytest.mark.parametrize(
    "data",
    [
        pytest.param(pd.array([1, 2, 3, 4, 5]), id="integer"),
        pytest.param(pd.array([2.5, 3.0, -16.13, 2.71828, 3.14]), id="float"),
        pytest.param(pd.array(["A", None, None, None, "E"]), id="string"),
        pytest.param(
            pd.array(
                [[datetime.date(2023, 7, 4), None] for _ in range(5)],
                dtype=pd.ArrowDtype(pa.large_list(pa.date32())),
            ),
            id="array_date",
        ),
    ],
)
def test_to_object_invalid(data):
    def impl_vector(data):
        return bodosql.kernels.to_object(data)

    def impl_scalar(data):
        return bodosql.kernels.to_object(data[0])

    with pytest.raises(BodoError):
        check_func(impl_vector, (data,), py_output="dummy_output")

    with pytest.raises(BodoError):
        check_func(impl_scalar, (data,), py_output="dummy_output", only_seq=True)
