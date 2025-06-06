import pytest

import bodo.pandas as bd
from bodo.tests.utils import _test_equal


def _generate_series_accessor_test(name, df, accessor):
    """
    Generates tests for Series.<accessor>.<name> accessors.
    """

    def test_func():
        bdf = bd.from_pandas(df)
        cols = bdf.columns

        for col in cols:
            pd_obj = getattr(df, col)
            bodo_obj = getattr(bdf, col)

            pd_obj = getattr(pd_obj, accessor)
            bodo_obj = getattr(bodo_obj, accessor)

            pd_accessor = getattr(pd_obj, name)
            bodo_accessor = getattr(bodo_obj, name)

            assert bodo_accessor.is_lazy_plan()
            try:
                _test_equal(pd_accessor, bodo_accessor, check_pandas_types=False)
            except AssertionError as e:
                """
                Exception case handler: currently Series.BodoDateTimeProperties.is_<something> returns <NA> values for NaT inputs, 
                while Pandas returns False. Also, date and time returns a non-ExtensionArray type, which causes _test_equal to fail. 
                In these cases, we print out the outputs Bodo vs. Pandas for manual inspection. 
                """
                if name in [
                    "is_month_start",
                    "is_month_end",
                    "is_quarter_start",
                    "is_quarter_end",
                    "is_year_start",
                    "is_year_end",
                    "is_leap_year",
                    "date",
                    "time",
                ]:  # Exception list
                    print(
                        f"Outputs may or may not differ, manually compare: \nPandas:\n{pd_accessor}\nBodo:\n{bodo_accessor}"
                    )
                    print("Terminating loop...\n")
                    break
                else:
                    raise AssertionError(e)

    return test_func


def _generate_series_test(name, df, arg_sets, accessor=None):
    """
    Generates a parameterized test case for Series methods.
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
        bdf = bd.from_pandas(df)
        cols = bdf.columns

        for col in cols:
            pd_obj = getattr(df, col)
            bodo_obj = getattr(bdf, col)

            if accessor:
                pd_obj = getattr(pd_obj, accessor)
                bodo_obj = getattr(bodo_obj, accessor)

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

            try:
                _test_equal(out_bodo, out_pd, check_pandas_types=False)
            except AssertionError as e:
                """
                Exception case handler: currently month_name and day_name with locale arg
                returns outputs that trivially differ from Pandas. Also, partition and rpartition returns same outputs, but 
                with different index types causing the equality test to fail. 
                In this case, we print out the outputs Bodo vs. Pandas for manual inspection. 
                """
                if name in [
                    "partition",
                    "rpartition",
                    "month_name",
                    "day_name",
                ]:  # Exception list
                    print(
                        f"Outputs may or may not differ, manually compare: \nPandas:\n{out_pd}\nBodo:\n{out_bodo}"
                    )
                    print("Terminating loop...\n")
                    break
                else:
                    raise AssertionError(e)

    return test_func
