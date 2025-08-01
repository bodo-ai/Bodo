import pandas as pd
import pytest
from test_end_to_end import index_val  # noqa
from test_series_generator import _generate_series_test

import bodo.pandas as bd
from bodo.pandas.plan import assert_executed_plan_count
from bodo.tests.utils import _test_equal


@pytest.mark.parametrize("use_index1", [True, False])
@pytest.mark.parametrize("use_index2", [True, False])
def test_series_isin(index_val, use_index1, use_index2):
    S1 = pd.Series([1, 2, 3, 4, 5] * 3)
    S2 = pd.Series([2, 8, 8, 1, 3, 11] * 4)
    if use_index1:
        S1.index = index_val[: len(S1)]

    if use_index2:
        S2.index = index_val[: len(S2)]

    py_out = S2.isin(S1)

    with assert_executed_plan_count(0):
        bdf1 = bd.from_pandas(pd.DataFrame({"A": S1}))
        bdf2 = bd.from_pandas(pd.DataFrame({"B": S2}))

        bd_out = bdf2["B"].isin(bdf1["A"])

    bd_out.name = None  # Drop name to match pandas output
    _test_equal(
        bd_out.copy(),
        py_out,
        check_pandas_types=False,
        sort_output=True,
        # We don't track RangeIndex in output
        reset_index=isinstance(S2.index, pd.RangeIndex),
    )


def _install_series_direct_tests():
    """Installs tests for direct Series.<method> methods."""
    for method_name, arg_sets in test_map_arg_direct.items():
        test = _generate_series_test(method_name, df, arg_sets)
        globals()[f"test_dir_{method_name}"] = test


test_map_arg_direct = {
    "clip": [
        ((), {}),
        ((1,), {}),
        ((1, 3), {}),
        ((), {"lower": 0.1}),
        ((), {"upper": 2}),
        ((), {"lower": 1.0, "upper": 4}),
    ],
    "replace": [
        ((1, 999), {}),
        (({2: 20, 3: 30},), {}),
        (([1, 2], [10, 20]), {}),
    ],
    "round": [
        ((), {}),
        ((1,), {}),
        ((0,), {}),
        ((-1,), {}),
    ],
    "isin": [
        (([1, 2],), {}),
        (([999],), {}),
    ],
    "notnull": [((), {})],
    "isnull": [((), {})],
    "abs": [((), {})],
}

df = pd.DataFrame(
    {
        "A": pd.array([1.445, 2.12, None, -4.133], dtype="Float64"),
        "B": pd.array([1, 2, 3, 4], dtype="Int64"),
    }
)

_install_series_direct_tests()
