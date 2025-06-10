import pandas as pd
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
            """
            Exception case handler: some methods return values that trivially differ from Pandas. 
            In these cases, we compare the Bodo output with the reference outputs in the expected_results list. 
            """
            if name in expected_results:
                expected_result = lookup_result(name, (), {}, col)
                if expected_result is not None:
                    _test_equal(
                        bodo_accessor,
                        expected_result,
                        check_pandas_types=False,
                        check_names=False,
                    )
                    continue
            _test_equal(pd_accessor, bodo_accessor, check_pandas_types=False)

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

            pd_error, bodo_error = False, False

            try:
                out_pd = pd_func(*args, **kwargs)
            except Exception:
                pd_error = True
            try:
                out_bodo = bodo_func(*args, **kwargs)
                assert out_bodo.is_lazy_plan()
                out_bodo.execute_plan()
            except Exception:
                bodo_error = True

            # Pandas and Bodo should behave the same way
            assert pd_error == bodo_error
            if pd_error:
                continue

            """
            Exception case handler: some methods return values that trivially differ from Pandas. 
            In these cases, we compare the Bodo output with the reference outputs in the expected_results list. 
            """
            if name in expected_results:
                expected_result = lookup_result(name, args, kwargs, col)
                if expected_result is not None:
                    _test_equal(
                        out_bodo,
                        expected_result,
                        check_pandas_types=False,
                        check_names=False,
                    )
                    continue
            _test_equal(out_bodo, out_pd, check_pandas_types=False)

    return test_func


"""
Below are expected results, which we compare our out_bodo results against
for cases where out_bodo and out_pd have trivial differences. 
"""
partition_res = pd.DataFrame(
    {
        "0": ["Apple", "Banana", None, None, "App", "B", " E", "Do"],
        "1": ["", "", None, None, "-", "-", "-", "-"],
        "2": ["", "", None, None, "le", "anan-a", "xc i-ted ", "g"],
    }
)
rpartition_res = pd.DataFrame(
    {
        "0": ["", "", None, None, "App", "B-anan", " E-xc i", "Do"],
        "1": ["", "", None, None, "-", "-", "-", "-"],
        "2": ["Apple", "Banana", None, None, "le", "a", "ted ", "g"],
    }
)

month_name_res_fr_A = pd.Series(
    [
        "janvier",
        "février",
        "mars",
        "avril",
        "mai",
        "juin",
        "juillet",
        "août",
        "septembre",
        "octobre",
    ]
)
month_name_res_fr_BC = pd.Series(
    [
        "décembre",
        "décembre",
        "décembre",
        "décembre",
        "décembre",
        "décembre",
        "décembre",
        "décembre",
        "décembre",
        "décembre",
    ]
)

day_name_res_pt_A = pd.Series(
    [
        "Terça Feira",
        "Sexta Feira",
        "Sexta Feira",
        "Segunda Feira",
        "Quarta Feira",
        "Sábado",
        "Segunda Feira",
        "Quinta Feira",
        "Domingo",
        "Terça Feira",
    ]
)
day_name_res_pt_B = pd.Series(
    [
        "Quinta Feira",
        "Quinta Feira",
        "Quinta Feira",
        "Quinta Feira",
        "Quinta Feira",
        "Quinta Feira",
        "Quinta Feira",
        "Quinta Feira",
        "Quinta Feira",
        "Quinta Feira",
    ]
)
day_name_res_pt_C = pd.Series(
    [
        "Sexta Feira",
        "Domingo",
        "Segunda Feira",
        "Terça Feira",
        "Quarta Feira",
        "Sexta Feira",
        "Sábado",
        "Domingo",
        "Segunda Feira",
        "Quarta Feira",
    ]
)

null_array = pd.Series(
    [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA]
)


expected_results = {
    "partition": [
        ((), {"sep": "-", "expand": True}, "A", partition_res),
        ((), {"sep": "-"}, "A", partition_res),
    ],
    "rpartition": [
        ((), {"sep": "-", "expand": True}, "A", rpartition_res),
        ((), {"sep": "-"}, "A", rpartition_res),
    ],
    "month_name": [
        ((), {"locale": "fr_FR.UTF-8"}, "A", month_name_res_fr_A),
        ((), {"locale": "fr_FR.UTF-8"}, "B", month_name_res_fr_BC),
        ((), {"locale": "fr_FR.UTF-8"}, "C", month_name_res_fr_BC),
    ],
    "day_name": [
        ((), {"locale": "pt_BR.UTF-8"}, "A", day_name_res_pt_A),
        ((), {"locale": "pt_BR.UTF-8"}, "B", day_name_res_pt_B),
        ((), {"locale": "pt_BR.UTF-8"}, "C", day_name_res_pt_C),
    ],
    "is_month_start": [
        ((), {}, "D", null_array),
    ],
    "is_month_end": [
        ((), {}, "D", null_array),
    ],
    "is_quarter_start": [
        ((), {}, "D", null_array),
    ],
    "is_quarter_end": [
        ((), {}, "D", null_array),
    ],
    "is_year_start": [
        ((), {}, "D", null_array),
    ],
    "is_year_end": [
        ((), {}, "D", null_array),
    ],
    "is_leap_year": [
        ((), {}, "D", null_array),
    ],
    "date": [
        ((), {}, "D", null_array),
    ],
    "time": [
        ((), {}, "D", null_array),
    ],
}


def lookup_result(name, args, kwargs, col):
    """Look up expected result using method name, args, kwargs, and col name."""
    for expected_args, expected_kwargs, expected_col, result in expected_results.get(
        name, []
    ):
        if args == expected_args and kwargs == expected_kwargs and col == expected_col:
            return result
    return None
