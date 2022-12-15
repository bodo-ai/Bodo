# Copyright (C) 2022 Bodo Inc. All rights reserved.


import os
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bodo
from bodo.tests.utils import (
    SeriesOptTestPipeline,
    _test_equal_guard,
    check_func,
    dist_IR_contains,
    reduce_sum,
)


@pytest.fixture(
    params=[
        pytest.param(
            pd.arrays.ArrowStringArray(
                pa.array(
                    ["abc", "b", None, "abc", None, "b", "cde"],
                    type=pa.dictionary(pa.int32(), pa.string()),
                ).cast(pa.dictionary(pa.int32(), pa.large_string()))
            )
        ),
    ]
)
def dict_arr_value(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param(
            pa.array(
                ["  abc", "b ", None, " bbCD ", "\n \tbbCD\t \n"] * 10,
                type=pa.dictionary(pa.int32(), pa.string()),
            )
        ),
    ]
)
def test_strip_dict_arr_value(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param(
            pa.array(
                [
                    "ABCDD,OSAJD",
                    "a1b2d314f,sdf234",
                    "22!@#,$@#$AB",
                    None,
                    "A,C,V,B,B",
                    "AA",
                    "",
                ]
                * 2,
                type=pa.dictionary(pa.int32(), pa.string()),
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pa.array(
                [
                    "¿abc¡Y tú, quién te crees?",
                    "ÕÕÕú¡úú,úũ¿ééé",
                    "россия очень, холодная страна",
                    None,
                    "مرحبا, العالم ، هذا هو بودو",
                    "Γειά σου ,Κόσμε",
                    "Español es agra,dable escuchar",
                ]
                * 2,
                type=pa.dictionary(pa.int32(), pa.string()),
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pa.array(
                [
                    "아1, 오늘 저녁은 뭐먹지",
                    "나,는 유,니,코,드 테스팅 중",
                    None,
                    "こんにち,は世界",
                    "大处着眼，小处着手。",
                    "오늘도 피츠버그의 날씨는 매우, 구림",
                    "한국,가,고싶다ㅠ",
                ]
                * 2,
                type=pa.dictionary(pa.int32(), pa.string()),
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pa.array(
                [
                    "😀🐍,⚡😅😂",
                    "🌶🍔,🏈💔💑💕",
                    "𠁆𠁪,𠀓𠄩𠆶",
                    None,
                    "🏈,💔,𠄩,😅",
                    "🠂,🠋🢇🄐,🞧",
                    "🢇🄐,🏈𠆶💑😅",
                ]
                * 2,
                type=pa.dictionary(pa.int32(), pa.string()),
            ),
            marks=pytest.mark.slow,
        ),
        pa.array(
            [
                "A",
                " bbCD",
                " mCDm",
                "C,ABB, D",
                "B,B,CC",
                "ABBD",
                "ABCDD,OSAJD",
                "a1b2d314f,sdf234",
                "C,ABB,D",
                "¿abc¡Y tú, quién te cre\t\tes?",
                "오늘도 피츠버그의 날씨는 매\t우, 구림",
                None,
                "🏈,💔,𠄩,😅",
                "大处着眼，小处着手。",
                "🠂,🠋🢇🄐,🞧",
                "россия очень, холодная страна",
                "",
                " ",
            ],
            type=pa.dictionary(pa.int32(), pa.string()),
        ),
    ]
)
def test_unicode_dict_str_arr(request):
    return request.param


@pytest.mark.slow
def test_unbox(dict_arr_value, memory_leak_check):
    # just unbox
    def impl(arr_arg):
        return True

    assert bodo.typeof(dict_arr_value) == bodo.dict_str_arr_type
    check_func(impl, (dict_arr_value,))

    # unbox and box
    def impl2(arr_arg):
        return arr_arg

    check_func(impl2, (dict_arr_value,))


@pytest.mark.slow
def test_len(dict_arr_value, memory_leak_check):
    def test_impl(A):
        return len(A)

    check_func(test_impl, (dict_arr_value,))


@pytest.mark.slow
def test_shape(dict_arr_value, memory_leak_check):
    def test_impl(A):
        return A.shape

    # PyArrow doesn't support shape
    assert bodo.jit(test_impl)(dict_arr_value) == (len(dict_arr_value),)


@pytest.mark.slow
def test_dtype(dict_arr_value, memory_leak_check):
    def test_impl(A):
        return A.dtype

    # PyArrow doesn't support dtype
    assert bodo.jit(test_impl)(dict_arr_value) == pd.StringDtype()


@pytest.mark.slow
def test_ndim(dict_arr_value, memory_leak_check):
    def test_impl(A):
        return A.ndim

    # PyArrow doesn't support ndim
    assert bodo.jit(test_impl)(dict_arr_value) == 1


@pytest.mark.slow
def test_copy(dict_arr_value, memory_leak_check):
    def test_impl(A):
        return A.copy()

    pd.testing.assert_extension_array_equal(
        bodo.jit(test_impl)(dict_arr_value), dict_arr_value
    )


def test_all_null_pa_bug(memory_leak_check):
    """Test for a bug in Arrow that fails in to_numpy() when all values are null"""

    def test_impl(A):
        df = pd.DataFrame({"A": A})
        print(df)
        return df

    A = pd.arrays.ArrowStringArray(
        pa.array(
            [None, None],
            type=pa.dictionary(pa.int32(), pa.string()),
        ).cast(pa.dictionary(pa.int32(), pa.large_string()))
    )

    check_func(test_impl, (A,), only_seq=True)
    A = pd.arrays.ArrowStringArray(
        pa.array(
            [],
            type=pa.dictionary(pa.int32(), pa.string()),
        ).cast(pa.dictionary(pa.int32(), pa.large_string()))
    )

    check_func(test_impl, (A,), only_seq=True)


@pytest.mark.slow
def test_repeat(dict_arr_value, memory_leak_check):
    def test_impl(A, n):
        return pd.Series(A).repeat(n)

    check_func(test_impl, (dict_arr_value, 3))
    check_func(test_impl, (dict_arr_value, np.arange(len(dict_arr_value))))


def test_cmp_opt(dict_arr_value, memory_leak_check):
    """test optimizaton of comparison operators (eq, ne) for dict array"""

    def impl1(A, val):
        return A == val

    def impl2(A, val):
        return val == A

    def impl3(A, val):
        return A != val

    def impl4(A, val):
        return val != A

    # convert to Pandas array since PyArrow doesn't support cmp operators
    pd_arr = pd.array(dict_arr_value, "string")

    for val in ("abc", "defg"):
        check_func(impl1, (dict_arr_value, val), py_output=(pd_arr == val))
        check_func(impl2, (dict_arr_value, val), py_output=(val == pd_arr))
        check_func(impl3, (dict_arr_value, val), py_output=(pd_arr != val))
        check_func(impl4, (dict_arr_value, val), py_output=(val != pd_arr))

    # make sure IR has the optimized functions
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl1)
    bodo_func(dict_arr_value, "abc")
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, "dict_arr_eq")
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl2)
    bodo_func("abc", dict_arr_value)
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, "dict_arr_eq")
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl3)
    bodo_func(dict_arr_value, "abc")
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, "dict_arr_ne")
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl4)
    bodo_func("abc", dict_arr_value)
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, "dict_arr_ne")


@pytest.mark.slow
def test_int_convert_opt(memory_leak_check):
    """test optimizaton of integer conversion for dict array"""

    def impl(A):
        return pd.Series(A).astype("Int32")

    data = ["14", None, "-3", "11", "-155", None]
    A = pa.array(data, type=pa.dictionary(pa.int32(), pa.string()))
    check_func(
        impl, (A,), py_output=pd.Series(pd.array(data, "string")).astype("Int32")
    )
    # make sure IR has the optimized function
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl)
    bodo_func(A)
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, "convert_dict_arr_to_int")


def test_str_to_dict_astype(memory_leak_check):
    """Test .astype() Casting from String to Dict Array"""

    def impl(S):
        return S.astype(bodo.dict_str_arr_type)

    data = ["a", "b", "a", "c", "a", "b", "c"] * 20
    py_out = pd.Series(
        pd.arrays.ArrowStringArray(
            pa.array(data, type=pa.dictionary(pa.int32(), pa.string())).cast(
                pa.dictionary(pa.int32(), pa.large_string())
            )
        )
    )

    check_func(
        impl,
        (pd.Series(data, dtype="string"),),
        py_output=py_out,
        use_dict_encoded_strings=False,
    )


@pytest.mark.slow
def test_gatherv_rm(dict_arr_value, memory_leak_check):
    """make sure gatherv() is removed in non-distributed pipeline without errors"""

    @bodo.jit(distributed=False)
    def impl(A):
        return pd.Series(A).unique()

    res = impl(dict_arr_value)
    pd.testing.assert_series_equal(
        pd.Series(pd.Series(dict_arr_value).unique()), pd.Series(res)
    )


def test_scalar_to_arr(memory_leak_check):
    """tests that appending a scalar to a DataFrame creates
    a dictionary encoded array."""

    def impl(arr):
        return pd.DataFrame({"A": arr, "b": "my string to copy"})

    arr = np.arange(100)
    check_func(impl, (arr,))
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl)
    bodo_func(arr)
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, "init_dict_arr")


def test_str_cat_opt(memory_leak_check):
    """test optimizaton of Series.str.cat() for dict array"""

    def impl1(S, A, B):
        S = pd.Series(S)
        df = pd.DataFrame({"A": A, "B": B})
        return S.str.cat(df, sep=", ")

    data1 = ["AB", None, "CDE", "ABBB", "ABB", "AC"]
    data2 = ["123", "312", "091", "345", None, "AC"]
    data3 = ["UAW", "13", None, "hb3 g", "h56", "AC"]
    A = pa.array(data1, type=pa.dictionary(pa.int32(), pa.string()))
    B = pa.array(data2, type=pa.dictionary(pa.int32(), pa.string()))
    S = pa.array(data3, type=pa.dictionary(pa.int32(), pa.string()))

    py_output = pd.Series(data3).str.cat(
        pd.DataFrame({"A": data1, "B": data2}), sep=", "
    )
    check_func(impl1, (S, A, B), py_output=py_output)
    # make sure IR has the optimized function
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl1)
    bodo_func(S, A, B)
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, "cat_dict_str")


def test_to_numeric(memory_leak_check):
    """test optimized pd.to_numeric() for dict-encoded string arrays"""

    def impl(A):
        return pd.to_numeric(pd.Series(A), errors="coerce", downcast="float")

    data = ["1.4", "2.3333", None, "1.22", "555.1"] * 2
    A = pa.array(data, type=pa.dictionary(pa.int32(), pa.string()))

    check_func(impl, (A,), py_output=pd.to_numeric(data))
    # make sure IR has the optimized function
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl)
    bodo_func(A)
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, "dict_arr_to_numeric")


def test_str_replace(memory_leak_check):
    """test optimizaton of Series.str.replace() for dict array"""

    def impl1(A):
        return pd.Series(A).str.replace("AB*", "EE", regex=True)

    def impl2(A):
        return pd.Series(A).str.replace("피츠*", "뉴욕의", regex=True)

    def impl3(A):
        return pd.Series(A).str.replace("AB", "EE", regex=False)

    data1 = ["AB", None, "ABCD", "CDE", None, "ABBB", "ABB", "AC"]
    data2 = ["피츠", None, "피츠뉴욕의", "뉴욕의", None, "뉴욕의뉴욕의", "피츠츠츠", "츠"]
    A1 = pa.array(data1, type=pa.dictionary(pa.int32(), pa.string()))
    A2 = pa.array(data2, type=pa.dictionary(pa.int32(), pa.string()))

    check_func(
        impl1, (A1,), py_output=pd.Series(data1).str.replace("AB*", "EE", regex=True)
    )
    check_func(
        impl2, (A2,), py_output=pd.Series(data2).str.replace("피츠*", "뉴욕의", regex=True)
    )
    check_func(
        impl3, (A1,), py_output=pd.Series(data1).str.replace("AB", "EE", regex=False)
    )
    # make sure IR has the optimized function
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl1)
    bodo_func(A1)
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, "str_replace")


def test_str_startswith(test_unicode_dict_str_arr, memory_leak_check):
    """
    test optimization of Series.str.startswith() for dict array
    """

    def impl1(A):
        return pd.Series(A).str.startswith("AB")

    def impl2(A):
        return pd.Series(A).str.startswith("테스")

    check_func(
        impl1,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.startswith("AB"),
    )
    check_func(
        impl2,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.startswith("테스"),
    )

    # make sure IR has the optimized function
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl1)
    bodo_func(test_unicode_dict_str_arr)
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, "str_startswith")


def test_str_endswith(test_unicode_dict_str_arr, memory_leak_check):
    """
    test optimization of Series.str.endswith() for dict array
    """

    def impl1(A):
        return pd.Series(A).str.endswith("AB")

    def impl2(A):
        return pd.Series(A).str.endswith("테스팅")

    check_func(
        impl1,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.endswith("AB"),
    )
    check_func(
        impl2,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.endswith("테스"),
    )

    # make sure IR has the optimized function
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl1)
    bodo_func(test_unicode_dict_str_arr)
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, "str_endswith")


def test_str_simple_str2str_methods(test_unicode_dict_str_arr, memory_leak_check):
    """
    test optimization of Series.str.capitalize/upper/lower/swapcase/title for dict array
    """

    def impl1(A):
        return pd.Series(A).str.capitalize()

    def impl2(A):
        return pd.Series(A).str.lower()

    def impl3(A):
        return pd.Series(A).str.upper()

    def impl4(A):
        return pd.Series(A).str.title()

    def impl5(A):
        return pd.Series(A).str.swapcase()

    check_func(
        impl1,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.capitalize(),
    )
    check_func(
        impl2,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.lower(),
    )
    check_func(
        impl3,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.upper(),
    )
    check_func(
        impl4,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.title(),
    )
    check_func(
        impl5,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.swapcase(),
    )

    # make sure IR has the optimized function
    impls = [impl1, impl2, impl3, impl4, impl5]
    func_names = ["capitalize", "lower", "upper", "title", "swapcase"]
    for i in range(len(impls)):
        bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impls[i])
        bodo_func(test_unicode_dict_str_arr)
        f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
        assert dist_IR_contains(f_ir, f"str_{func_names[i]}")


def test_str_pad(test_unicode_dict_str_arr, memory_leak_check):
    """
    test optimization of Series.str.center/ljust/rjust/zfill for dict array
    """

    def impl1(A):
        return pd.Series(A).str.center(5, "*")

    def impl2(A):
        return pd.Series(A).str.rjust(1, "d")

    def impl3(A):
        return pd.Series(A).str.ljust(1, "a")

    def impl4(A):
        return pd.Series(A).str.zfill(1)

    def impl5(A):
        return pd.Series(A).str.pad(1, "left", "🍔")

    def impl6(A):
        return pd.Series(A).str.pad(1, "right", "🍔")

    def impl7(A):
        return pd.Series(A).str.pad(1, "both", "🍔")

    def impl8(A):
        return pd.Series(A).str.center(5)

    check_func(
        impl1,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.center(5, "*"),
    )
    check_func(
        impl2,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.rjust(1, "d"),
    )
    check_func(
        impl3,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.ljust(1, "a"),
    )
    check_func(
        impl4,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.zfill(1),
    )
    check_func(
        impl5,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.pad(1, "left", "🍔"),
    )
    check_func(
        impl6,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.pad(1, "right", "🍔"),
    )
    check_func(
        impl7,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.pad(1, "both", "🍔"),
    )
    check_func(
        impl8,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.center(5),
    )

    # make sure IR has the optimized function
    impl_names = [
        (impl1, "center"),
        (impl2, "rjust"),
        (impl3, "ljust"),
        (impl4, "zfill"),
        (impl5, "rjust"),
        (impl6, "ljust"),
        (impl7, "center"),
        (impl8, "center"),
    ]
    for i in range(len(impl_names)):
        bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl_names[i][0])
        bodo_func(test_unicode_dict_str_arr)
        f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
        assert dist_IR_contains(f_ir, f"str_{impl_names[i][1]}")


def test_str_slice(test_unicode_dict_str_arr, memory_leak_check):
    """
    test optimization of Series.str.slice for dict array
    """

    def impl1(A):
        return pd.Series(A).str.slice(step=2)

    def impl2(A):
        return pd.Series(A).str.slice(start=1)

    def impl3(A):
        return pd.Series(A).str.slice(stop=3)

    def impl4(A):
        return pd.Series(A).str.slice(2, 1, 3)

    def impl5(A):
        return pd.Series(A).str.slice(1, 3, 2)

    check_func(
        impl1,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.slice(step=2),
    )
    check_func(
        impl2,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.slice(start=1),
    )
    check_func(
        impl3,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.slice(stop=3),
    )
    check_func(
        impl4,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.slice(2, 1, 3),
    )
    check_func(
        impl5,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.slice(1, 3, 2),
    )

    # make sure IR has the optimized function
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl1)
    bodo_func(test_unicode_dict_str_arr)
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, "str_slice")


@pytest.mark.parametrize(
    "S, sub, start, end",
    [
        (
            pa.array(
                ["alpha", "beta", "alphabet", "patatasbravas", None, "houseofcards"]
                * 5,
                type=pa.dictionary(pa.int32(), pa.string()),
            ),
            "a",
            0,
            10,
        ),
        (
            pa.array(
                ["alpha", "beta", "alphabet", "patatasbravas", "emeralds"],
                type=pa.dictionary(pa.int32(), pa.string()),
            ),
            "a",
            2,
            6,
        ),
        (
            pa.array(
                ["bagel", None, "gelatin", "gelato", "angelfish", "evangelist"],
                type=pa.dictionary(pa.int32(), pa.string()),
            ),
            "gel",
            0,
            10,
        ),
    ],
)
@pytest.mark.parametrize("method", ["index", "rindex"])
def test_str_index_rindex(S, sub, start, end, method, memory_leak_check):
    """Test optimization of pd.Series.str.index/rindex"""
    func_dict = {
        "index": pd.Series(S).str.index,
        "rindex": pd.Series(S).str.rindex,
    }
    func_text = (
        "def test_impl1(S, sub):\n"
        f"    return pd.Series(S).str.{method}(sub)\n"
        "def test_impl2(S, sub, start):\n"
        f"    return pd.Series(S).str.{method}(sub, start=start)\n"
        "def test_impl3(S, sub, end):\n"
        f"    return pd.Series(S).str.{method}(sub, end=end)\n"
        "def test_impl4(S, sub, start, end):\n"
        f"    return pd.Series(S).str.{method}(sub, start, end)\n"
    )
    local_vars = {}
    exec(func_text, {"pd": pd}, local_vars)
    test_impl1 = local_vars["test_impl1"]
    test_impl2 = local_vars["test_impl2"]
    test_impl3 = local_vars["test_impl3"]
    test_impl4 = local_vars["test_impl4"]
    check_func(
        test_impl1, (S, sub), py_output=func_dict[method](sub), check_dtype=False
    )
    check_func(
        test_impl2,
        (S, sub, start),
        py_output=func_dict[method](sub, start=start),
        check_dtype=False,
    )
    check_func(
        test_impl3,
        (S, sub, end),
        py_output=func_dict[method](sub, end=end),
        check_dtype=False,
    )
    check_func(
        test_impl4,
        (S, sub, start, end),
        py_output=func_dict[method](sub, start, end),
        check_dtype=False,
    )
    # make sure IR has the optimized function
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(test_impl1)
    bodo_func(S, sub)
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, f"str_{method}")


@pytest.mark.parametrize("method", ["index", "rindex"])
def test_str_index_rindex_not_found(test_unicode_dict_str_arr, method):
    """Test error handling for Series.str.index/rindex"""
    func_text = (
        "def test_impl1(A):\n"
        f"    return pd.Series(A).str.{method}('123')\n"
        "def test_impl2(A, start):\n"
        f"    return pd.Series(A).str.{method}('123', start=start)\n"
        "def test_impl3(A, end):\n"
        f"    return pd.Series(A).str.{method}('123', end=end)\n"
        "def test_impl4(A, start, end):\n"
        f"    return pd.Series(A).str.{method}('123', start, end)\n"
    )
    local_vars = {}
    exec(func_text, {"pd": pd}, local_vars)
    test_impl1 = local_vars["test_impl1"]
    test_impl2 = local_vars["test_impl2"]
    test_impl3 = local_vars["test_impl3"]
    test_impl4 = local_vars["test_impl4"]
    with pytest.raises(ValueError, match="substring not found"):
        bodo.jit(test_impl1)(test_unicode_dict_str_arr)
    with pytest.raises(ValueError, match="substring not found"):
        bodo.jit(test_impl2)(test_unicode_dict_str_arr, start=2)
    with pytest.raises(ValueError, match="substring not found"):
        bodo.jit(test_impl3)(test_unicode_dict_str_arr, end=10)
    with pytest.raises(ValueError, match="substring not found"):
        bodo.jit(test_impl4)(test_unicode_dict_str_arr, start=2, end=10)


@pytest.mark.parametrize("method", ["find", "rfind"])
def test_str_find(test_unicode_dict_str_arr, memory_leak_check, method):
    """
    test optimization of Series.str.find/rfind
    """
    func_dict = {
        "find": pd.Series(test_unicode_dict_str_arr).str.find,
        "rfind": pd.Series(test_unicode_dict_str_arr).str.rfind,
    }
    func_text = (
        "def impl1(A):\n"
        f"    return pd.Series(A).str.{method}('AB')\n"
        "def impl2(A):\n"
        f"    return pd.Series(A).str.{method}('🍔')\n"
        "def impl3(A):\n"
        f"    return pd.Series(A).str.{method}('*', start=3)\n"
        "def impl4(A):\n"
        f"    return pd.Series(A).str.{method}('着', end=5)\n"
        "def impl5(A):\n"
        f"    return pd.Series(A).str.{method}('ん', start=2, end=8)\n"
    )
    loc_vars = {}
    global_vars = {"pd": pd}
    exec(func_text, global_vars, loc_vars)
    impl1 = loc_vars["impl1"]
    impl2 = loc_vars["impl2"]
    impl3 = loc_vars["impl3"]
    impl4 = loc_vars["impl4"]
    impl5 = loc_vars["impl5"]

    check_func(
        impl1,
        (test_unicode_dict_str_arr,),
        py_output=func_dict[method]("AB"),
        check_dtype=False,
    )
    check_func(
        impl2,
        (test_unicode_dict_str_arr,),
        py_output=func_dict[method]("🍔"),
        check_dtype=False,
    )
    check_func(
        impl3,
        (test_unicode_dict_str_arr,),
        py_output=func_dict[method]("*", start=3),
        check_dtype=False,
    )
    check_func(
        impl4,
        (test_unicode_dict_str_arr,),
        py_output=func_dict[method]("着", end=5),
        check_dtype=False,
    )
    check_func(
        impl5,
        (test_unicode_dict_str_arr,),
        py_output=func_dict[method]("ん", start=2, end=8),
        check_dtype=False,
    )

    # make sure IR has the optimized function
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl1)
    bodo_func(test_unicode_dict_str_arr)
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, f"str_{method}")


@pytest.mark.parametrize("method", ["lstrip", "rstrip", "strip"])
def test_str_strip(test_strip_dict_arr_value, memory_leak_check, method):
    """
    test optimization of Series.str.strip/lstrip/rstip for dict array
    """
    func_dict = {
        "lstrip": pd.Series(test_strip_dict_arr_value).str.lstrip,
        "rstrip": pd.Series(test_strip_dict_arr_value).str.rstrip,
        "strip": pd.Series(test_strip_dict_arr_value).str.strip,
    }
    func_text = (
        "def impl1(A):\n"
        f"    return pd.Series(A).str.{method}(' ')\n"
        "def impl2(A):\n"
        f"    return pd.Series(A).str.{method}('\\n')\n"
        "def impl3(A):\n"
        f"    return pd.Series(A).str.{method}()\n"
    )
    loc_vars = {}
    global_vars = {"pd": pd}
    exec(func_text, global_vars, loc_vars)
    impl1 = loc_vars["impl1"]
    impl2 = loc_vars["impl2"]
    impl3 = loc_vars["impl3"]

    check_func(
        impl1,
        (test_strip_dict_arr_value,),
        py_output=func_dict[method](" "),
    )
    check_func(
        impl2,
        (test_strip_dict_arr_value,),
        py_output=func_dict[method]("\n"),
    )
    check_func(
        impl3,
        (test_strip_dict_arr_value,),
        py_output=func_dict[method](),
    )

    # make sure IR has the optimized function
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl1)
    bodo_func(test_strip_dict_arr_value)
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, f"str_{method}")


def test_str_get(test_unicode_dict_str_arr):
    """
    test optimization of Series.str.get for dict array
    """

    def impl1(A):
        return pd.Series(A).str.get(1)

    check_func(
        impl1,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.get(1),
    )
    # make sure IR has the optimized function
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl1)
    bodo_func(test_unicode_dict_str_arr)
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, "str_get")


def test_str_repeat(test_unicode_dict_str_arr):
    """
    test optimization of Series.str.repeat for dict array
    """

    def impl1(A):
        return pd.Series(A).str.repeat(3)

    check_func(
        impl1,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.repeat(3),
    )

    # make sure IR has the optimized function
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl1)
    bodo_func(test_unicode_dict_str_arr)
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, "str_repeat_int")


@pytest.mark.parametrize("case", [True, False])
def test_str_contains_regex(memory_leak_check, test_unicode_dict_str_arr, case):
    """
    test optimization of Series.str.contains(regex=True) for dict array
    """

    def impl1(A):
        return pd.Series(A).str.contains("AB*", regex=True, case=case)

    def impl2(A):
        return pd.Series(A).str.contains("피츠버*", regex=True, case=case)

    def impl3(A):
        return pd.Series(A).str.contains("ab*", regex=True, case=case)

    check_func(
        impl1,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.contains(
            "AB*", regex=True, case=case
        ),
    )
    check_func(
        impl2,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.contains(
            "피츠버*", regex=True, case=case
        ),
    )
    check_func(
        impl3,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.contains(
            "ab*", regex=True, case=case
        ),
    )

    # Test flags (and hence `str_series_contains_regex`)
    import re

    flag = re.M.value

    def impl4(A):
        return pd.Series(A).str.contains(r"ab*", regex=True, case=case, flags=flag)

    check_func(
        impl4,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.contains(
            r"ab*", regex=True, case=case, flags=flag
        ),
    )

    # make sure IR has the optimized function
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl4)
    bodo_func(test_unicode_dict_str_arr)
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, "str_series_contains_regex")


@pytest.mark.parametrize("case", [True, False])
def test_str_contains_noregex(memory_leak_check, test_unicode_dict_str_arr, case):
    """
    test optimization of Series.str.contains(regex=False) for dict array
    """

    def impl1(A):
        return pd.Series(A).str.contains("AB", regex=False, case=case)

    def impl2(A):
        return pd.Series(A).str.contains("피츠버", regex=False, case=case)

    def impl3(A):
        return pd.Series(A).str.contains("ab", regex=False, case=case)

    check_func(
        impl1,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.contains(
            "AB", regex=False, case=case
        ),
    )
    check_func(
        impl2,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.contains(
            "피츠버", regex=False, case=case
        ),
    )
    check_func(
        impl3,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.contains(
            "ab", regex=False, case=case
        ),
    )

    # make sure IR has the optimized function
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl1)
    bodo_func(test_unicode_dict_str_arr)
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, "str_contains_non_regex")


@pytest.mark.parametrize("case", [True, False])
def test_str_match(memory_leak_check, test_unicode_dict_str_arr, case):
    """
    test optimization of Series.str.contains(regex=False) for dict array
    """

    def impl1(A):
        return pd.Series(A).str.match("AB", case=case)

    def impl2(A):
        return pd.Series(A).str.match("피츠버", case=case)

    def impl3(A):
        return pd.Series(A).str.match("ab", case=case)

    def impl4(A):
        return pd.Series(A).str.match("AB*", case=case)

    def impl5(A):
        return pd.Series(A).str.match("피츠버*", case=case)

    def impl6(A):
        return pd.Series(A).str.match("ab*", case=case)

    check_func(
        impl1,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.match("AB", case=case),
    )
    check_func(
        impl2,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.match("피츠버", case=case),
    )
    check_func(
        impl3,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.match("ab", case=case),
    )
    check_func(
        impl4,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.match("AB*", case=case),
    )
    check_func(
        impl5,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.match("피츠버*", case=case),
    )
    check_func(
        impl6,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.match("ab*", case=case),
    )
    # Test flags (and hence `str_match` in dict_arr_ext.py)
    import re

    flag = re.M.value

    def impl7(A):
        return pd.Series(A).str.match(r"ab*", case=case, flags=flag)

    check_func(
        impl7,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.match(
            r"ab*", case=case, flags=flag
        ),
    )

    # make sure IR has the optimized function
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl7)
    bodo_func(test_unicode_dict_str_arr)
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, "str_match")


def test_sort_values(memory_leak_check):
    """test that sort_values works for dict array"""

    def impl(A, col):
        return A.sort_values(col)

    dataA = ["ABC", "def", "abc", "ABC", "DE", "def", "SG"]
    A = pa.array(dataA, type=pa.dictionary(pa.int32(), pa.string()))
    DF = pd.DataFrame({"A": A})
    check_func(
        impl,
        (DF, "A"),
        py_output=pd.DataFrame({"A": dataA}).sort_values("A"),
    )

    dataA[4] = None
    A = pa.array(dataA, type=pa.dictionary(pa.int32(), pa.string()))
    DF = pd.DataFrame({"A": A})
    check_func(
        impl,
        (DF, "A"),
        py_output=pd.DataFrame({"A": dataA}).sort_values("A"),
    )

    dataB = ["ABC", "DEF", "abc", "ABC", "DE", "re", "DEF"]
    DF = pd.DataFrame({"A": A, "B": pd.Series(dataB)})
    check_func(
        impl,
        (DF, ["A", "B"]),
        py_output=pd.DataFrame({"A": dataA, "B": dataB}).sort_values(["A", "B"]),
    )


def test_dict_array_unify(dict_arr_value):
    """Tests that unifying dict arrays works as expected."""
    # TODO: Add memory leak check, casting bug

    @bodo.jit
    def impl(A):
        # This condition is False at runtime, so unifying
        # the arrays will require casting the series.
        if len(A) > 30:
            A = pd.Series(A).sort_values().values
        return A

    bodo_out = impl(dict_arr_value)
    pd.testing.assert_extension_array_equal(bodo_out, dict_arr_value)


def test_str_len(test_unicode_dict_str_arr, memory_leak_check):
    """
    test optimization of Series.str.len for dict array
    """

    def impl1(A):
        return pd.Series(A).str.len()

    check_func(
        impl1,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.len(),
        check_dtype=False,
    )
    # make sure IR has the optimized function
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl1)
    bodo_func(test_unicode_dict_str_arr)
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, "str_len")


def test_str_count(test_unicode_dict_str_arr, memory_leak_check):
    """
    test optimization of Series.str.capitalize/upper/lower/swapcase/title for dict array
    """
    flag = re.IGNORECASE.value

    def impl1(A):
        return pd.Series(A).str.count("A")

    def impl2(A):
        return pd.Series(A).str.count("피츠")

    def impl3(A):
        return pd.Series(A).str.count("A", flag)

    def impl4(A):
        return pd.Series(A).str.count("피츠", flag)

    check_func(
        impl1,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.count("A"),
        check_dtype=False,
    )
    check_func(
        impl2,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.count("피츠"),
        check_dtype=False,
    )
    check_func(
        impl3,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.count("A", flag),
        check_dtype=False,
    )
    check_func(
        impl4,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.count("피츠", flag),
        check_dtype=False,
    )
    # make sure IR has the optimized function
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl1)
    bodo_func(test_unicode_dict_str_arr)
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, "str_count")


@pytest.mark.parametrize("method", bodo.hiframes.pd_series_ext.str2bool_methods)
def test_str_str2bool_methods(test_unicode_dict_str_arr, memory_leak_check, method):
    """
    test optimization of Series.str.isalnum/isalpha/isdigit/isspae/islower/
    isupper/istitle/isnumeric/isdecimal for dict array
    """
    func_text = (
        "def impl1(A):\n"
        f"    return pd.Series(A).str.{method}()\n"
        f"py_output=pd.Series(test_unicode_dict_str_arr).str.{method}()"
    )

    loc_vars = {}
    global_vars = {"pd": pd, "test_unicode_dict_str_arr": test_unicode_dict_str_arr}
    exec(func_text, global_vars, loc_vars)
    impl1 = loc_vars["impl1"]
    py_output = loc_vars["py_output"]

    check_func(
        impl1,
        (test_unicode_dict_str_arr,),
        py_output=py_output,
    )

    # make sure IR has the optimized function
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl1)
    bodo_func(test_unicode_dict_str_arr)
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, f"str_{method}")


def test_str_extract(memory_leak_check, test_unicode_dict_str_arr):
    """
    tests optimization for Series.str.extract for dictionary arrays
    """

    def impl1(A):
        return pd.Series(A).str.extract(r"(?P<BBB>[abd])(?P<C>\d+)")

    def impl2(A):
        return pd.Series(A).str.extract(r"(?P<BBB>[아])(?P<C>\d+)")

    def impl3(A):
        return pd.Series(A).str.extract(r"(P<DD>[着])")

    def impl4(A):
        return pd.Series(A).str.extract(r"(P<DD>[着])", expand=False)

    def impl5(A):
        return pd.Series(A).str.extract(r"(?P<BBB>[아])(?P<C>\d+)", expand=False)

    # when regex group has no name, Series name should be used
    def impl6(A):
        return pd.Series(A).str.extract(r"([abd]+)\d+", expand=False)

    check_func(
        impl1,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.extract(
            r"(?P<BBB>[abd])(?P<C>\d+)"
        ),
    )
    check_func(
        impl2,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.extract(
            r"(?P<BBB>[아])(?P<C>\d+)"
        ),
    )
    check_func(
        impl3,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.extract(r"(P<DD>[着])"),
    )
    check_func(
        impl4,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.extract(
            r"(P<DD>[着])", expand=False
        ),
    )
    check_func(
        impl5,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.extract(
            r"(?P<BBB>[아])(?P<C>\d+)", expand=False
        ),
    )
    check_func(
        impl6,
        (test_unicode_dict_str_arr,),
        py_output=pd.Series(test_unicode_dict_str_arr).str.extract(
            r"([abd]+)\d+", expand=False
        ),
    )
    # make sure IR has the optimized function
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl1)
    bodo_func(test_unicode_dict_str_arr)
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, "str_extract")


def test_str_extractall(memory_leak_check):
    """test optimizaton of Series.str.extractall() for dict array"""
    # non-string index, single group
    def impl1(A):
        return pd.Series(A, name="AA").str.extractall(r"(?P<BBB>[abd]+)\d+")

    # non-string index, multiple groups
    def impl2(A):
        return pd.Series(A).str.extractall(r"([чен]+)\d+([ст]+)\d+")

    # string index, single group
    def impl3(A, I):
        return pd.Series(data=A, index=I).str.extractall(r"(?P<BBB>[abd]+)\d+")

    # string index, multiple groups
    def impl4(A, I):
        return pd.Series(data=A, index=I).str.extractall(r"([чен]+)\d+([ст]+)\d+")

    S1 = pd.Series(
        ["a1b1", "b1", np.nan, "a2", "c2", "ddd", "dd4d1", "d22c2"],
        [4, 3, 5, 1, 0, 2, 6, 11],
        name="AA",
    )
    S2 = pd.Series(
        ["чьь1т33", "ьнн2с222", "странаст2", np.nan, "ьнне33ст3"] * 2,
        ["е3", "не3", "н2с2", "AA", "C"] * 2,
    )
    A1 = pa.array(S1, type=pa.dictionary(pa.int32(), pa.string()))
    A2 = pa.array(S2, type=pa.dictionary(pa.int32(), pa.string()))

    I1 = pd.Index(["a", "b", "e", "好", "e2", "yun", "c", "dd"])
    I2 = pd.Index(["е3", "не3", "н2с2", "AA", "C"] * 2)

    check_func(
        impl1,
        (A1,),
        py_output=pd.Series(A1, name="AA").str.extractall(r"(?P<BBB>[abd]+)\d+"),
    )
    check_func(
        impl2,
        (A2,),
        py_output=pd.Series(A2).str.extractall(r"([чен]+)\d+([ст]+)\d+"),
    )

    check_func(
        impl3,
        (A1, I1),
        py_output=pd.Series(
            data=A1, index=["a", "b", "e", "好", "e2", "yun", "c", "dd"]
        ).str.extractall(r"(?P<BBB>[abd]+)\d+"),
    )
    check_func(
        impl4,
        (A2, I2),
        py_output=pd.Series(
            data=A2, index=["е3", "не3", "н2с2", "AA", "C"] * 2
        ).str.extractall(r"([чен]+)\d+([ст]+)\d+"),
    )
    # make sure IR has the optimized function

    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl1)
    bodo_func(A1)
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, "str_extractall")

    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl2)
    bodo_func(A2)
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, "str_extractall_multi")

    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl3)
    bodo_func(A1, I1)
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, "str_extractall")

    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl4)
    bodo_func(A2, I2)
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, "str_extractall_multi")


def test_concat(memory_leak_check):
    """test pd.concat() for dict arrays"""
    A1 = pa.array(
        ["abc", "b", None, "abc", None, "b", "cde"],
        type=pa.dictionary(pa.int32(), pa.string()),
    )

    A2 = pa.array(
        ["  abc", "b ", None, " bbCD ", "\n \tbbCD\t \n", "abc", ""] * 10,
        type=pa.dictionary(pa.int32(), pa.string()),
    )

    S1 = pd.Series(["aefw", "ER", None, "Few"] * 15)

    def impl1(A1, A2):
        return pd.concat((pd.Series(A1), pd.Series(A2)), ignore_index=True)

    def impl2(A1, S1):
        return pd.concat((S1, pd.Series(A1)), ignore_index=True)

    def impl3(file1, file2):
        df1 = pd.read_parquet(file1, _bodo_read_as_dict=["A", "B"])
        df2 = pd.read_parquet(file2, _bodo_read_as_dict=["A", "B"])
        return pd.concat([df1, df2], ignore_index=True)

    check_func(
        impl1,
        (A1, A2),
        py_output=pd.concat((pd.Series(A1), pd.Series(A2)), ignore_index=True),
        sort_output=True,
        reset_index=True,
    )

    check_func(
        impl2,
        (A1, S1),
        py_output=pd.concat((S1, pd.Series(A1)), ignore_index=True),
        sort_output=True,
        reset_index=True,
    )

    df1 = pd.DataFrame(
        {"A": ["aefw", "ER", None, "Few"] * 25, "B": ["#$@4", "!1"] * 50}
    )
    df2 = pd.DataFrame(
        {
            "A": ["  abc", "b ", None, " bbCD ", "\n \tbbCD\t \n", "abc", ""] * 5,
            "B": ["abc", "b", None, "abc", None, "b", "cde"] * 5,
        }
    )
    temp_file_1 = "temp_file_1.pq"
    temp_file_2 = "temp_file_2.pq"
    if bodo.get_rank() == 0:
        # Send to a file to create table source
        df1.to_parquet(temp_file_1, index=False)
        df2.to_parquet(temp_file_2, index=False)
    bodo.barrier()
    # use dictionary-encoded arrays for strings
    try:
        check_func(
            impl3,
            (temp_file_1, temp_file_2),
            py_output=pd.concat([df1, df2], ignore_index=True),
            sort_output=True,
            reset_index=True,
            use_dict_encoded_strings=True,
        )
    finally:
        if bodo.get_rank() == 0:
            os.remove(temp_file_1)
            os.remove(temp_file_2)


@pytest.mark.parametrize(
    "A_enc",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "B_enc",
    [
        True,
        pytest.param(False, marks=pytest.mark.slow),
    ],
)
@pytest.mark.parametrize(
    "C_enc",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "D_enc",
    [
        True,
        pytest.param(False, marks=pytest.mark.slow),
    ],
)
def test_gatherv_dict_enc_and_normal_str_array_table(
    A_enc, B_enc, C_enc, D_enc, memory_leak_check
):
    """
    Tests that we can properly handle tables contianing both string and dict encoded string arrays,
    in gatherv. Currently, this is an operation which will decode the dict encoded string arrays,
    but this may be changed in the future: https://bodo.atlassian.net/browse/BE-3681
    """

    func_text = "def make_df_impl(A, B, C, D):\n"
    if not A_enc:
        func_text += "  A = decode_if_dict_array(A)\n"
    if not B_enc:
        func_text += "  B = decode_if_dict_array(B)\n"
    if not C_enc:
        func_text += "  C = decode_if_dict_array(C)\n"
    if not D_enc:
        func_text += "  D = decode_if_dict_array(D)\n"

    func_text += "  df = pd.DataFrame({'A': A, 'B': B, 'C': C, 'D': D })\n"
    func_text += "  scattered_df = bodo.scatterv(df)\n"
    func_text += "  return scattered_df\n"

    loc_vars = {}
    exec(
        func_text,
        {
            "bodo": bodo,
            "pd": pd,
            "decode_if_dict_array": bodo.utils.typing.decode_if_dict_array,
        },
        loc_vars,
    )
    make_df_impl = bodo.jit(loc_vars["make_df_impl"])
    tmp_range = pd.Series(np.arange(10).astype(str))
    orig_use_dict_str_type = bodo.hiframes.boxing._use_dict_str_type
    orig_TABLE_FORMAT_THRESHOLD = bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD
    try:
        bodo.hiframes.boxing._use_dict_str_type = True
        bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD = 0
        (A, B, C, D) = (
            "A_" + tmp_range,
            "B_" + tmp_range,
            "C_" + tmp_range,
            "D_" + tmp_range,
        )
        expected_output = pd.DataFrame({"A": A, "B": B, "C": C, "D": D})
        table_format_df_scattered = make_df_impl(A, B, C, D)
        bodo.hiframes.boxing._use_dict_str_type = False
        # Gatherv will cause the dict encoded
        bodo_output = bodo.gatherv(table_format_df_scattered)
        passed = 1
        if bodo.get_rank() == 0:
            passed = _test_equal_guard(
                bodo_output,
                expected_output,
            )
        n_passed = reduce_sum(passed)
        assert n_passed == bodo.get_size()

    finally:
        bodo.hiframes.boxing._use_dict_str_type = orig_use_dict_str_type
        bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD = orig_TABLE_FORMAT_THRESHOLD
