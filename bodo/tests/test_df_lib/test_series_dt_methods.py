import pandas as pd
import pytest

import bodo.pandas as bd
from bodo.pandas.series import dt_accessors, dt_methods
from bodo.tests.utils import _test_equal


def gen_dt_accessor_test(name):
    """
    Generates a test case for Series.dt.<name> accessor.
    """

    def test_func():
        date_m = pd.Series(pd.date_range("20130101 09:10:12", periods=10, freq="MS"))
        date_s = pd.Series(pd.date_range("20221201 09:10:12", periods=10, freq="s"))
        date_y = pd.Series(pd.date_range("19990303 09:10:12", periods=10, freq="YE"))

        df = pd.DataFrame({"A": date_m, "B": date_s, "C": date_y})
        bdf = bd.from_pandas(df)

        keys = ["A", "B", "C"]

        for key in keys:
            a = getattr(df, key)
            b = getattr(bdf, key)
            pd_accessor = getattr(a.dt, name)
            bodo_accessor = getattr(b.dt, name)
            assert bodo_accessor.is_lazy_plan()
            _test_equal(pd_accessor, bodo_accessor, check_pandas_types=False)

    return test_func


def gen_dt_method_test(name, arg_sets):
    """
    Generates a test case for Series.dt.<name> method.
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
        date_m = pd.Series(pd.date_range("20130101 09:10:12", periods=10, freq="MS"))
        date_s = pd.Series(pd.date_range("20221201 09:10:12", periods=10, freq="s"))
        date_y = pd.Series(pd.date_range("19990303 09:10:12", periods=10, freq="YE"))
        date_none = pd.Series([pd.NaT] * 10, dtype="datetime64[ns]")

        df = pd.DataFrame({"A": date_m, "B": date_s, "C": date_y, "D": date_none})
        bdf = bd.from_pandas(df)

        keys = ["A", "B", "C", "D"]

        for key in keys:
            pd_obj = getattr(df, key)
            bodo_obj = getattr(bdf, key)

            pd_func = getattr(pd_obj.dt, name)
            bodo_func = getattr(bodo_obj.dt, name)

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

            # Precise types of Exceptions might differ
            assert pd_error[0] == bodo_error[0]
            if pd_error[0]:
                return

            try:
                _test_equal(out_bodo, out_pd, check_pandas_types=False)
            except AssertionError as e:
                """
                Exception case handler: currently month_name and day_name with locale arg
                returns outputs that trivially differ from Pandas. 
                In this case, we print out the outputs Bodo vs. Pandas for manual inspection. 
                """
                if name in ["month_name", "day_name"]:  # Exception list
                    print(
                        f"Outputs may or may not differ, manually compare: \nPandas:\n{out_pd}\nBodo:\n{out_bodo}"
                    )
                    print("Terminating loop...\n")
                    break
                else:
                    raise AssertionError(e)

    return test_func


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
    # TODO [BSE-4880]: %S prints up to nanoseconds by default, fix this to make the cases below work.
    # "strftime": [
    #     (("%Y-%m-%D %H:%M",), {}),
    #     (("%Y-%m-%D %H:%M:%S",), {}),
    #     (("date is: %S",), {}),
    # ],
}


def _install_series_dt_tests():
    """Install Series.dt tests."""
    # Tests Series.dt accessors
    for accessor_pair in dt_accessors:
        for accessor_name in accessor_pair[0]:
            test = gen_dt_accessor_test(accessor_name)
            globals()[f"test_dt_{accessor_name}"] = test

    # Tests Series.dt methods
    for method_pair in dt_methods:
        for method_name in method_pair[0]:
            test = gen_dt_method_test(method_name, test_map_arg[method_name])
            globals()[f"test_dt_{method_name}"] = test


_install_series_dt_tests()
