import pandas as pd
from test_series_generator import _generate_series_test


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
