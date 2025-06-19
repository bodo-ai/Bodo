import pandas as pd
from test_series_generator import _generate_series_accessor_test, _generate_series_test

from bodo.pandas.series import dt_accessors, dt_methods

timedelta_methods = ("total_seconds",)


def _install_series_dt_tests():
    """Install Series.dt tests."""
    # Tests Series.dt accessors
    for accessor_pair in dt_accessors:
        for accessor_name in accessor_pair[0]:
            test = _generate_series_accessor_test(accessor_name, df, "dt")
            globals()[f"test_{accessor_name}"] = test

    # Tests Series.dt methods
    for method_pair in dt_methods:
        for method_name in method_pair[0]:
            test = _generate_series_test(
                method_name,
                timedelta_df if method_name in timedelta_methods else df,
                test_map_arg[method_name],
                accessor="dt",
            )
            globals()[f"test_{method_name}"] = test


# Maps method name to test case for pytest param
# More rigorous testing NEEDED
test_map_arg = {
    "normalize": [
        ((), {}),
    ],
    "month_name": [
        ((), {}),
        ((), {"locale": "en_US.UTF-8"}),
        ((), {"locale": "en_US.utf-8"}),
        ((), {"locale": "fr_FR.UTF-8"}),
        ((), {"locale": "pt_BR.UTF-8"}),
    ],
    "day_name": [
        ((), {}),
        ((), {"locale": "en_US.UTF-8"}),
        ((), {"locale": "en_US.utf-8"}),
        ((), {"locale": "fr_FR.UTF-8"}),
        ((), {"locale": "pt_BR.UTF-8"}),
    ],
    "floor": [
        (("2h"), {}),
        (("h"), {}),
        (("D"), {}),
    ],
    "ceil": [
        (("2h"), {}),
        (("h"), {}),
        (("D"), {}),
    ],
    "total_seconds": [((), {})],
    # TODO [BSE-4880]: %S prints up to nanoseconds by default, fix this to make the cases below work.
    # "strftime": [
    #     (("%Y-%m-%D %H:%M",), {}),
    #     (("%Y-%m-%D %H:%M:%S",), {}),
    #     (("date is: %S",), {}),
    # ],
}

date_m = pd.Series(pd.date_range("20130101 09:10:12", periods=10, freq="MS"))
date_s = pd.Series(pd.date_range("20221201 09:10:12", periods=10, freq="s"))
date_y = pd.Series(pd.date_range("19990303 09:10:12", periods=10, freq="YE"))
date_none = pd.Series([pd.NaT] * 10, dtype="datetime64[ns]")

df = pd.DataFrame({"A": date_m, "B": date_s, "C": date_y, "D": date_none})
timedelta_df = pd.DataFrame(
    {
        "A": pd.to_timedelta(range(10), unit="s"),
        "B": pd.to_timedelta(range(10, 20), unit="s"),
    }
)


_install_series_dt_tests()
