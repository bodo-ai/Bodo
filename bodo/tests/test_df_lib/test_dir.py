import pandas as pd
import pytest

import bodo.pandas as bd
from bodo.tests.utils import _test_equal


def gen_direct_series_test(name, arg_sets):
    """
    Generates a parameterized test case for direct Series.<name> method.
    """

    @pytest.mark.parametrize(
        "args, kwargs",
        arg_sets,
        ids=[
            f"{name}-args{args}-kwargs{sorted(kwargs.items())}"
            for args, kwargs in arg_sets
        ],
    )
    def test_func(args, kwargs):
        df = pd.DataFrame(
            {
                "A": pd.array([1.445, 2.12, None, -4.133], dtype="Float64"),
                "B": pd.array([1, 2, 3, 4], dtype="Int64"),
            }
        )
        bdf = bd.from_pandas(df)

        keys = ["A", "B"]

        for key in keys:
            pd_obj = getattr(df, key)
            bodo_obj = getattr(bdf, key)

            pd_func = getattr(pd_obj, name)
            bodo_func = getattr(bodo_obj, name)

            pd_error, bodo_error = (False, None), (False, None)

            try:
                out_pd = pd_func(*args, **kwargs)
            except Exception as e:
                pd_error = (True, e)
            try:
                out_bodo = bodo_func(*args, **kwargs)
                assert out_bodo.is_lazy_plan()
                out_bodo.execute_plan()
            except Exception as e:
                bodo_error = (True, e)

            assert pd_error[0] == bodo_error[0]
            if pd_error[0]:
                return

            _test_equal(out_bodo, out_pd, check_pandas_types=False)

    return test_func


test_map_arg_direct = {
    "clip": [
        ((), {}),
        ((1,), {}),
        ((1, 3), {}),
        ((), {"lower": 0}),
        ((), {"upper": 2}),
        ((), {"lower": 1, "upper": 4}),
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


def _install_series_direct_tests():
    """Installs tests for direct Series.<method> methods."""
    for method_name, arg_sets in test_map_arg_direct.items():
        test = gen_direct_series_test(method_name, arg_sets)
        globals()[f"test_dir_{method_name}"] = test


_install_series_direct_tests()
