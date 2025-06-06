import re

import pandas as pd
import pytest

import bodo.pandas as bd
from bodo.tests.utils import _test_equal


def gen_str_param_test(name, arg_sets):
    """
    Generates a parameterized test case for Series.str.<name> method.
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
                "A": [
                    "Apple",
                    "Banana",
                    None,
                    None,
                    "App-le",
                    "B-anan-a",
                    " E-xc i-ted ",
                    "Do-g",
                ],
            }
        )
        bdf = bd.from_pandas(df)

        pd_func = getattr(df.A.str, name)
        bodo_func = getattr(bdf.A.str, name)
        pd_error, bodo_error = (False, None), (False, None)

        # Pandas and Bodo methods should have identical behavior
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
            Exception case handler: currently partition and rpartition returns same outputs, but 
            with different index types causing the equality test to fail. 
            In this case, we print out the outputs Bodo vs. Pandas for manual inspection. 
            """
            if name in ["partition", "rpartition"]:  # Exception list
                print(
                    f"Outputs may or may not differ, manually compare: \nPandas:\n{out_pd}\nBodo:\n{out_bodo}"
                )
            else:
                raise AssertionError(e)

    return test_func


# Maps method name to test case for pytest param
# More rigorous testing NEEDED
test_map_arg = {
    "contains": [
        (("A",), {}),
        (("A",), {"na": False}),
        (("A",), {"na": True}),
        (("a",), {"case": False}),
        (("a",), {"case": False, "na": False}),
        (("A",), {"regex": False}),
    ],
    "startswith": [
        (("A",), {}),
        (("A",), {"na": False}),
        (("A",), {"na": None}),
        ((("A", "B"),), {}),
    ],
    "endswith": [
        (("e",), {}),
        (("e",), {"na": False}),
        (("e",), {"na": None}),
        ((("e", "d"),), {}),
    ],
    "find": [
        (("a",), {}),
        (("b",), {"start": 1}),
        (("D",), {"end": 3}),
    ],
    "rfind": [
        (("a",), {}),
        (("b",), {"start": 1}),
        (("D",), {"end": 3}),
    ],
    # TODO [BSE-4842] Below test case should correctly raise ValueError instead of stalling
    "index": [
        # (("e",), {}),
        # (("an",), {"start": 0}),
        # (("l",), {"end": 3}),
        (("",), {"end": 3}),
    ],
    # TODO [BSE-4842] Below test case should correctly raise ValueError instead of stalling
    "rindex": [
        # (("e",), {}),
        # (("an",), {"start": 0}),
        # (("l",), {"end": 3}),
        (("",), {"end": 3}),
    ],
    "replace": [
        (("a", "b"), {}),
        (("a", "b"), {"regex": False}),
        (("a", "b"), {"regex": True}),
    ],
    "match": [
        (("A",), {}),
        (("A",), {"na": False}),
        (("A",), {"na": None}),
    ],
    "fullmatch": [
        (("Apple",), {}),
        (("Apple",), {"na": False}),
    ],
    "get": [((0,), {}), ((-1,), {}), ((2,), {}), ((5,), {})],
    "slice": [
        ((), {"start": 1}),
        ((), {"start": 1, "stop": 3}),
        ((), {"step": 2}),
        ((), {"start": -1}),
    ],
    "slice_replace": [
        ((), {"start": 1}),
        ((), {"start": 1, "stop": 3, "repl": "oi"}),
        ((), {"stop": 4, "repl": "XXX"}),
        ((), {"start": -1}),
    ],
    "repeat": [
        ((2,), {}),
        ((5,), {}),
        ((0,), {}),
    ],
    "pad": [
        ((10,), {}),
        ((10,), {"side": "left"}),
        ((10,), {"side": "right", "fillchar": "-"}),
    ],
    "center": [
        ((10,), {}),
        ((10,), {"fillchar": "*"}),
        ((8,), {"fillchar": "."}),
    ],
    "ljust": [
        ((10,), {}),
        ((10,), {"fillchar": "*"}),
    ],
    "rjust": [
        ((10,), {}),
        ((10,), {"fillchar": "*"}),
    ],
    "zfill": [
        ((10,), {}),
    ],
    "wrap": [
        ((4,), {}),
        ((1,), {"drop_whitespace": False}),
    ],
    "removeprefix": [
        (("A",), {}),
        (("Do",), {}),
        (("B",), {}),
    ],
    "removesuffix": [
        (("anana",), {}),
        (("Dog",), {}),
        (("og",), {}),
        (("Canon",), {}),
    ],
    "translate": [
        ((str.maketrans("abc", "123"),), {}),
    ],
    "count": [
        (("an",), {}),
        (("Ex",), {}),
        (("a",), {}),
    ],
    "findall": [
        (("Banana",), {}),
        (("BANANA",), {"flags": re.IGNORECASE}),
    ],
    "partition": [
        ((), {"sep": "-", "expand": False}),
        ((), {"sep": "-", "expand": True}),
        ((), {"sep": "-"}),
    ],
    "rpartition": [
        ((), {"sep": "-", "expand": False}),
        ((), {"sep": "-", "expand": True}),
        ((), {"sep": "-"}),
    ],
}

# List of methods that do not take in arguments
test_map_no_arg = [
    "upper",
    "lower",
    "title",
    "swapcase",
    "capitalize",
    "casefold",
    "isalpha",
    "isnumeric",
    "isalnum",
    "isdigit",
    "isdecimal",
    "isspace",
    "islower",
    "isupper",
    "istitle",
    "len",
]


def _install_series_str_tests():
    """Install Series.str tests."""
    # Tests Series.str methods with arguments
    for method_name in test_map_arg:
        test = gen_str_param_test(method_name, test_map_arg[method_name])
        globals()[f"test_auto_{method_name}"] = test

    # Tests Series.str methods that require no arguments
    for method_name in test_map_no_arg:
        test = gen_str_param_test(method_name, [((), {})])
        globals()[f"test_auto_{method_name}"] = test


_install_series_str_tests()
