# Copyright (C) 2023 Bodo Inc. All rights reserved.
import pandas as pd
import pyarrow as pa
import pytest

import bodosql
from bodo.tests.utils import check_func


@pytest.mark.parametrize(
    "vector",
    [
        pytest.param(True, id="vector"),
        pytest.param(False, id="scalar"),
    ],
)
@pytest.mark.parametrize(
    "data, answer",
    [
        pytest.param(
            pd.Series(["a", "b", "c"]),
            pd.Series([False] * 3),
            id="string",
        ),
        pytest.param(
            pd.Series(list(range(3))),
            pd.Series([False] * 3),
            id="int",
        ),
        pytest.param(
            pd.Series([[0], [1], [2]], dtype=pd.ArrowDtype(pa.large_list(pa.int32()))),
            pd.Series([True] * 3),
            id="list",
        ),
        pytest.param(
            pd.Series(
                [{"a": 0}] * 3, dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int32()))
            ),
            pd.Series([False] * 3),
            id="map",
        ),
        pytest.param(
            pd.Series(
                [{"a": 0}] * 3,
                dtype=pd.ArrowDtype(pa.struct([pa.field("a", pa.int32())])),
            ),
            pd.Series([False] * 3),
            id="struct",
        ),
    ],
)
def test_is_array(vector, data, answer, memory_leak_check):
    """Test is_array bodo kernel"""

    if not vector and isinstance(data[0], list):
        # This test would fail because passing in data[0] would just result in
        # function running on the value within the array - it's tested through
        # SQL instead.
        return

    def impl_vector(data):
        return pd.Series(bodosql.kernels.is_array(data))

    def impl_scalar(data):
        return bodosql.kernels.is_array(data[0])

    impl = impl_vector if vector else impl_scalar
    answer = answer if vector else answer[0]

    check_func(
        impl,
        (data,),
        py_output=answer,
        distributed=vector,
    )


@pytest.mark.parametrize(
    "data, answer",
    [
        pytest.param(
            pd.Series(["a", "b", "c"]),
            pd.Series([False] * 3),
            id="string",
        ),
        pytest.param(
            pd.Series(list(range(3))),
            pd.Series([False] * 3),
            id="int",
        ),
        pytest.param(
            pd.Series([[0], [1], [2]]),
            pd.Series([False] * 3),
            id="list",
        ),
        pytest.param(
            pd.Series(
                [{"a": 0}] * 3, dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int32()))
            ),
            pd.Series([True] * 3),
            id="map",
        ),
        pytest.param(
            pd.Series(
                [{"a": 0}] * 3,
                dtype=pd.ArrowDtype(pa.struct([pa.field("a", pa.int32())])),
            ),
            pd.Series([True] * 3),
            id="struct",
        ),
    ],
)
def test_is_object(data, answer, memory_leak_check):
    """Test is_object bodo kernel"""

    def impl(data):
        return pd.Series(bodosql.kernels.is_object(data))

    check_func(
        impl,
        (data,),
        py_output=answer,
    )
