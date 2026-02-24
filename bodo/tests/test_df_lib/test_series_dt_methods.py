import pandas as pd

from bodo.pandas.series import dt_accessors, dt_methods
from bodo.tests.test_df_lib.series_test_generator import (
    generate_series_accessor_test,
    generate_series_test,
)

timedelta_methods = (
    "total_seconds",
    "days",
    "seconds",
    "microseconds",
    "nanoseconds",
    "components",
)


def _install_series_dt_tests():
    """Install Series.dt tests."""

    def install_dt_accessor_test(accessor_name):
        test = generate_series_accessor_test(
            accessor_name,
            timedelta_df if accessor_name in timedelta_methods else df,
            "dt",
        )
        globals()[f"test_{accessor_name}"] = test

    def install_dt_method_test(method_name):
        test = generate_series_test(
            method_name,
            timedelta_df if method_name in timedelta_methods else df,
            test_map_arg[method_name],
            accessor="dt",
        )
        globals()[f"test_{method_name}"] = test

    # Tests Series.dt accessors
    for accessor_pair in dt_accessors:
        for accessor_name in accessor_pair[0]:
            install_dt_accessor_test(accessor_name)
    for accessor_name in untracked_accessors:
        install_dt_accessor_test(accessor_name)

    # Tests Series.dt methods
    for method_pair in dt_methods:
        for method_name in method_pair[0]:
            install_dt_method_test(method_name)
    for method_name in untracked_methods:
        install_dt_method_test(method_name)


# Accessors that are not auto-generated
untracked_accessors = ("components",)
# Methods that are not auto-generated
untracked_methods = ("isocalendar", "tz_localize")

# Maps method name to test case for pytest param
# More rigorous testing NEEDED
test_map_arg = {
    "normalize": [
        ((), {}),
    ],
    "month_name": [
        ((), {}),
        # NOTE: Comments out locale tests due to locale issues in CI
        # ((), {"locale": "en_US.UTF-8"}),
        # ((), {"locale": "en_US.utf-8"}),
        # ((), {"locale": "fr_FR.UTF-8"}),
        # ((), {"locale": "pt_BR.UTF-8"}),
    ],
    "day_name": [
        ((), {}),
        # NOTE: Comments out locale tests due to locale issues in CI
        # ((), {"locale": "en_US.UTF-8"}),
        # ((), {"locale": "en_US.utf-8"}),
        # ((), {"locale": "fr_FR.UTF-8"}),
        # ((), {"locale": "pt_BR.UTF-8"}),
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
    "isocalendar": [((), {})],
    "round": [(("h"), {})],
    "tz_localize": [((), {"CET": "CET", "ambiguous": "NaT", "nonexistent": "NaT"})],
}

date_tz = pd.Series(
    pd.date_range(
        "20130101 00:00:00", periods=10, freq="h", tz="America/New_York"
    ).astype("datetime64[ns, America/New_York]")
)
date_m = pd.Series(
    pd.date_range("20130101 09:10:12", periods=10, freq="MS").astype("datetime64[ns]")
)
date_s = pd.Series(
    pd.date_range("20221201 09:10:12", periods=10, freq="s").astype("datetime64[ns]")
)
date_y = pd.Series(
    pd.date_range("19990303 09:10:12", periods=10, freq="YE").astype("datetime64[ns]")
)
date_none = pd.Series([pd.NaT] * 10, dtype="datetime64[ns]")

df = pd.DataFrame({"A": date_m, "B": date_s, "C": date_y, "D": date_none, "E": date_tz})
timedelta_df = pd.DataFrame(
    {
        "A": pd.to_timedelta(range(10), unit="s").astype("timedelta64[ns]"),
        "B": pd.to_timedelta(range(10, 20), unit="s").astype("timedelta64[ns]"),
    }
)


_install_series_dt_tests()
