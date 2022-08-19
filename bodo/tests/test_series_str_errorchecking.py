import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.utils.typing import BodoError


@pytest.fixture(params=[pd.Series(["New_York", "Lisbon", "Tokyo", "Paris", "Munich"])])
def test_sr(request, memory_leak_check):
    return request.param


@pytest.fixture(params=[pd.Series(["New_York", "Lisbon", "Tokyo", "Paris", "Munich"])])
def test_sr_no_memory_leak_check(request):
    return request.param


# ------------------------------ center() ------------------------------ #
def test_center_fillchar_nonchar(test_sr, memory_leak_check):
    """
    tests error for center with the argument 'fillchar' being non-char type
    """

    def impl(test_sr):
        return test_sr.str.center(width=13, fillchar=1)

    def impl2(test_sr):
        return test_sr.str.center(width=13, fillchar="**")

    with pytest.raises(BodoError, match="fillchar must be a character, not"):
        bodo.jit(impl)(test_sr)
    with pytest.raises(BodoError, match="fillchar must be a character, not"):
        bodo.jit(impl2)(test_sr)


def test_center_width_noint(test_sr, memory_leak_check):
    """
    tests error for center with the argument 'width' being non-integer type
    """

    def impl(test_sr):
        return test_sr.str.center(width="1", fillchar="*")

    with pytest.raises(BodoError, match="expected an int object"):
        bodo.jit(impl)(test_sr)


# ------------------------------ ljust() ------------------------------ #
def test_ljust_fillchar_nonchar(test_sr, memory_leak_check):
    """
    tests error for ljust with the argument 'fillchar' being non-char type
    """

    def impl(test_sr):
        return test_sr.str.ljust(width=13, fillchar=1)

    def impl2(test_sr):
        return test_sr.str.ljust(width=13, fillchar="**")

    with pytest.raises(BodoError, match="fillchar must be a character, not"):
        bodo.jit(impl)(test_sr)
    with pytest.raises(BodoError, match="fillchar must be a character, not"):
        bodo.jit(impl2)(test_sr)


def test_ljust_width_noint(test_sr, memory_leak_check):
    """
    tests error for ljust with the argument 'width' being non-integer type
    """

    def impl(test_sr):
        return test_sr.str.ljust(width="1", fillchar="*")

    with pytest.raises(BodoError, match="expected an int object"):
        bodo.jit(impl)(test_sr)


# ------------------------------ rjust() ------------------------------ #
def test_rjust_fillchar_nonchar(test_sr, memory_leak_check):
    """
    tests error for rjust with the argument 'fillchar' being non-char type
    """

    def impl(test_sr):
        return test_sr.str.rjust(width=13, fillchar=1)

    def impl2(test_sr):
        return test_sr.str.rjust(width=13, fillchar="**")

    with pytest.raises(BodoError, match="fillchar must be a character, not"):
        bodo.jit(impl)(test_sr)
    with pytest.raises(BodoError, match="fillchar must be a character, not"):
        bodo.jit(impl2)(test_sr)


def test_rjust_width_noint(test_sr, memory_leak_check):
    """
    tests error for rjust with the argument 'width' being non-integer type
    """

    def impl(test_sr):
        return test_sr.str.rjust(width="1", fillchar="*")

    with pytest.raises(BodoError, match="expected an int object"):
        bodo.jit(impl)(test_sr)


# ------------------------------ zfill() ------------------------------ #
def test_zfill_width_noint(test_sr, memory_leak_check):
    """
    tests error for zfill with the argument 'width' being non-integer type
    """

    def impl(test_sr):
        return test_sr.str.zfill(width="1")

    with pytest.raises(BodoError, match="expected an int object"):
        bodo.jit(impl)(test_sr)


# ------------------------------ pad() ------------------------------ #
def test_pad_fillchar_nonchar(test_sr, memory_leak_check):
    """
    tests error for pad with the argument 'fillchar' being non-char type
    """

    def impl(test_sr):
        return test_sr.str.pad(width=13, fillchar=1)

    def impl2(test_sr):
        return test_sr.str.pad(width=13, fillchar="**")

    with pytest.raises(BodoError, match="fillchar must be a character, not"):
        bodo.jit(impl)(test_sr)
    with pytest.raises(BodoError, match="fillchar must be a character, not"):
        bodo.jit(impl2)(test_sr)


def test_pad_width_noint(test_sr, memory_leak_check):
    """
    tests error for pad with the argument 'width' being non-integer type
    """

    def impl(test_sr):
        return test_sr.str.pad(width="1", fillchar="*")

    with pytest.raises(BodoError, match="expected an int object"):
        bodo.jit(impl)(test_sr)


def test_pad_side_invalid(test_sr, memory_leak_check):
    """
    tests error for pad with the argument 'side' being invalid
    """

    def impl(test_sr):
        return test_sr.str.pad(width=13, side="123", fillchar="*")

    def impl2(test_sr):
        return test_sr.str.pad(width=13, side=123, fillchar="*")

    with pytest.raises(BodoError, match="Invalid Side"):
        bodo.jit(impl)(test_sr)
    with pytest.raises(BodoError, match="Invalid Side"):
        bodo.jit(impl2)(test_sr)


# ------------------------------ find() ------------------------------ #
def test_find_sub(test_sr, memory_leak_check):
    """
    tests error for find with the argument 'sub' being non-str type
    """

    def impl(test_sr):
        return test_sr.str.find(123)

    def impl2(test_sr):
        return test_sr.str.find(None)

    with pytest.raises(BodoError, match="expected a string object, not"):
        bodo.jit(impl)(test_sr)
    with pytest.raises(BodoError, match="expected a string object, not"):
        bodo.jit(impl2)(test_sr)


# ------------------------------ rfind() ------------------------------ #
def test_rfind_sub(test_sr, memory_leak_check):
    """
    tests error for rfind with the argument 'sub' being non-str type
    """

    def impl(test_sr):
        return test_sr.str.rfind(123)

    def impl2(test_sr):
        return test_sr.str.rfind(None)

    with pytest.raises(BodoError, match="expected a string object, not"):
        bodo.jit(impl)(test_sr)
    with pytest.raises(BodoError, match="expected a string object, not"):
        bodo.jit(impl2)(test_sr)


def test_rfind_start_end(test_sr, memory_leak_check):
    """
    tests error for rfind with the argument start/end NOT being integer or None type
    """

    def impl(test_sr):
        return test_sr.str.rfind("123", "x", 5)

    def impl2(test_sr):
        return test_sr.str.rfind("123", 5, "x")

    def impl3(test_sr):
        return test_sr.str.rfind("123", "x", "x")

    with pytest.raises(BodoError, match="expected an int object, not"):
        bodo.jit(impl)(test_sr)
    with pytest.raises(BodoError, match="expected an int object, not"):
        bodo.jit(impl2)(test_sr)
    with pytest.raises(BodoError, match="expected an int object, not"):
        bodo.jit(impl3)(test_sr)


# ------------------------------ index/rindex() ------------------------------ #
@pytest.mark.parametrize("method", ["index", "rindex"])
def test_index_rindex_sub(test_sr, method, memory_leak_check):
    """
    tests error for index/rindex with the argument 'sub' being non-str type
    """
    func_text = (
        "def impl1(test_sr):\n"
        f"  return test_sr.str.{method}(123)\n"
        "def impl2(test_sr):\n"
        f"  return test_sr.str.{method}(None)\n"
    )
    local_vars = {}
    exec(func_text, {}, local_vars)
    impl1 = local_vars["impl1"]
    impl2 = local_vars["impl2"]
    with pytest.raises(BodoError, match="expected a string object, not"):
        bodo.jit(impl1)(test_sr)
    with pytest.raises(BodoError, match="expected a string object, not"):
        bodo.jit(impl2)(test_sr)


@pytest.mark.parametrize("method", ["index", "rindex"])
def test_index_rindex_start_end(test_sr, method, memory_leak_check):
    """
    tests error for index/rindex with the argument start/end NOT being integer or None type
    """
    func_text = (
        "def impl(test_sr):\n"
        f"    return test_sr.str.{method}('123', 'x', 5)\n"
        "def impl2(test_sr):\n"
        f"    return test_sr.str.{method}('123', 5, 'x')\n"
        "def impl3(test_sr):\n"
        f"    return test_sr.str.{method}('123', 'x', 'x')\n"
    )
    local_vars = {}
    exec(func_text, {}, local_vars)
    impl = local_vars["impl"]
    impl2 = local_vars["impl2"]
    impl3 = local_vars["impl3"]
    with pytest.raises(BodoError, match="expected an int object, not"):
        bodo.jit(impl)(test_sr)
    with pytest.raises(BodoError, match="expected an int object, not"):
        bodo.jit(impl2)(test_sr)
    with pytest.raises(BodoError, match="expected an int object, not"):
        bodo.jit(impl3)(test_sr)


@pytest.mark.parametrize("method", ["index", "rindex"])
def test_index_rindex_not_found(test_sr_no_memory_leak_check, method):
    """
    tests error for index/rindex with the substring not found
    """
    func_text = (
        "def impl1(test_sr):\n"
        f"    return test_sr.str.{method}('123')\n"
        "def impl2(test_sr):\n"
        f"    return test_sr.str.{method}('123', 1, 4)\n"
        "def impl3(test_sr):\n"
        f"    return test_sr.str.{method}('123', end=2)\n"
        "def impl4(test_sr):\n"
        f"    return test_sr.str.{method}('123', start=7)\n"
    )
    local_vars = {}
    exec(func_text, {}, local_vars)
    impl1 = local_vars["impl1"]
    impl2 = local_vars["impl2"]
    impl3 = local_vars["impl3"]
    impl4 = local_vars["impl4"]
    with pytest.raises(ValueError, match="substring not found"):
        bodo.jit(impl1)(test_sr_no_memory_leak_check)
    with pytest.raises(ValueError, match="substring not found"):
        bodo.jit(impl2)(test_sr_no_memory_leak_check)
    with pytest.raises(ValueError, match="substring not found"):
        bodo.jit(impl3)(test_sr_no_memory_leak_check)
    with pytest.raises(ValueError, match="substring not found"):
        bodo.jit(impl4)(test_sr_no_memory_leak_check)


# ------------------------------ get() ------------------------------ #
@pytest.mark.parametrize(
    "input",
    [
        pd.Series([1, 2, 3]),
        # pd.Series([(1, 2, 3), (3, 4, 5)])  # TODO: support unboxing Series of tuples
    ],
)
def test_get_input(input, memory_leak_check):
    """
    tests error for get with the input series not being ListStringArrayType or
    StringArrayType
    """

    def impl(input):
        return input.str.get(1)

    with pytest.raises(
        BodoError, match="input should be a series of string/binary or arrays"
    ):
        bodo.jit(impl)(input)


@pytest.mark.parametrize("input", [pd.Series([["a", "b", "c"], ["aa", "bb", "cc"]])])
def test_get_i(input, memory_leak_check):
    """
    tests error for get with the argument i not being int type
    """

    def impl(input):
        return input.str.get("2")

    with pytest.raises(BodoError, match="expected an int object, not"):
        bodo.jit(impl)(input)


# ------------------------------ getitem ------------------------------ #


@pytest.mark.parametrize("ind", ["3", 1.2])
def test_getitem_ind(ind, memory_leak_check):
    """
    tests error for getitem for index inputs other than slice and int
    """

    def impl(S, ind):
        return S.str[ind]

    with pytest.raises(BodoError, match="index input to Series.str"):
        bodo.jit(impl)(pd.Series(["AA", "BB"]), ind)


# ------------------------------ split() ------------------------------ #
@pytest.fixture(
    params=[
        pd.Series(
            [
                "this is a regular sentence",
                "https://docs.python.org/3/tutorial/index.html",
                np.nan,
            ]
        )
    ]
)
def test_sr_split(request, memory_leak_check):
    return request.param


def test_split_args(test_sr_split, memory_leak_check):
    """
    tests error for split arguments that are not supported
    """

    def impl(test_sr_split):
        return test_sr_split.str.split(expand=True)

    with pytest.raises(BodoError, match="is not supported"):
        bodo.jit(impl)(test_sr_split)


def test_split_pat(test_sr_split, memory_leak_check):
    """
    tests error for split argument pat being non-str type
    """

    def impl(test_sr_split):
        return test_sr_split.str.split(pat=123)

    with pytest.raises(BodoError, match="expected a string object"):
        bodo.jit(impl)(test_sr_split)


# ------------------------------ replace() ------------------------------ #
def test_replace_args(test_sr, memory_leak_check):
    """
    tests error for replace arguments that are not supported
    """

    def impl(test_sr):
        return test_sr.str.replace("New", "hi", n=2)

    def impl2(test_sr):
        return test_sr.str.replace("New", "hi", case=True)

    with pytest.raises(BodoError, match="is not supported"):
        bodo.jit(impl)(test_sr)
    with pytest.raises(BodoError, match="is not supported"):
        bodo.jit(impl2)(test_sr)


def test_replace_pat(test_sr, memory_leak_check):
    """
    tests error for replace argument pat/repl/flags being incorrect type
    """

    def impl(test_sr):
        return test_sr.str.replace(pat=123, repl="asdf", flags=1)

    def impl2(test_sr):
        return test_sr.str.replace(pat="asdf", repl=123, flags=1)

    def impl3(test_sr):
        return test_sr.str.replace(pat="asdf", repl="asdf", flags="x")

    with pytest.raises(BodoError, match="expected a string object"):
        bodo.jit(impl)(test_sr)
    with pytest.raises(BodoError, match="expected a string object"):
        bodo.jit(impl2)(test_sr)
    with pytest.raises(BodoError, match="expected an int object"):
        bodo.jit(impl3)(test_sr)


# ------------------------------ contains() ------------------------------ #
def test_contains_args(test_sr, memory_leak_check):
    """
    tests error for contains arguments that are not supported
    """

    def impl(test_sr):
        return test_sr.str.contains("New", na=np.nan)

    with pytest.raises(BodoError, match="is not supported"):
        bodo.jit(impl)(test_sr)


def test_contains_flags(test_sr, memory_leak_check):
    """
    tests error for contains argument flags being incorrect type
    """

    def impl(test_sr):
        return test_sr.str.contains("New", flags="x")

    with pytest.raises(BodoError, match="expected an int object"):
        bodo.jit(impl)(test_sr)


def test_contains_regex(test_sr, memory_leak_check):
    """
    tests error for contains argument regex being incorrect type
    """

    def impl(test_sr):
        return test_sr.str.contains("New", regex="x")

    with pytest.raises(
        BodoError, match="'regex' argument should be a constant boolean"
    ):
        bodo.jit(impl)(test_sr)


def test_contains_case(test_sr, memory_leak_check):
    """
    tests error for contains argument case being incorrect type
    """

    def impl(test_sr):
        return test_sr.str.contains("New", case="x")

    with pytest.raises(BodoError, match="'case' argument should be a constant boolean"):
        bodo.jit(impl)(test_sr)


# ------------------------------ match() ------------------------------ #
def test_match_args(test_sr, memory_leak_check):
    """
    tests error for match arguments that are not supported
    """

    def impl(test_sr):
        return test_sr.str.match("New", na=np.nan)

    with pytest.raises(BodoError, match="is not supported"):
        bodo.jit(impl)(test_sr)


def test_match_flags(test_sr, memory_leak_check):
    """
    tests error for match argument flags being incorrect type
    """

    def impl(test_sr):
        return test_sr.str.match("New", flags="x")

    with pytest.raises(BodoError, match="expected an int object"):
        bodo.jit(impl)(test_sr)


def test_match_case(test_sr, memory_leak_check):
    """
    tests error for match argument case being incorrect type
    """

    def impl(test_sr):
        return test_sr.str.match("New", case="x")

    with pytest.raises(BodoError, match="'case' argument should be a constant boolean"):
        bodo.jit(impl)(test_sr)


# ------------------------------ count() ------------------------------ #
def test_count_args(test_sr, memory_leak_check):
    """
    tests error for count argument pat, flags being incorrect type
    """

    def impl(test_sr):
        return test_sr.str.count(pat=123)

    def impl2(test_sr):
        return test_sr.str.count(pat="sdf", flags="x")

    with pytest.raises(BodoError, match="expected a string object"):
        bodo.jit(impl)(test_sr)
    with pytest.raises(BodoError, match="expected an int object"):
        bodo.jit(impl2)(test_sr)


# ------------------------------ startswith() ------------------------------ #
def test_startswith_args(test_sr, memory_leak_check):
    """
    tests error for startswith argument that is not supported
    """

    def impl(test_sr):
        return test_sr.str.startswith("New", na=1)

    with pytest.raises(BodoError, match="is not supported"):
        bodo.jit(impl)(test_sr)


def test_startswith_pat(test_sr, memory_leak_check):
    """
    tests error for startswith argument pat being non-str type
    """

    def impl(test_sr):
        return test_sr.str.startswith(123)

    with pytest.raises(BodoError, match="expected a string object"):
        bodo.jit(impl)(test_sr)


# ------------------------------ endswith() ------------------------------ #
def test_endswith_args(test_sr, memory_leak_check):
    """
    tests error for endswith argument that is not supported
    """

    def impl(test_sr):
        return test_sr.str.endswith("New", na=1)

    with pytest.raises(BodoError, match="is not supported"):
        bodo.jit(impl)(test_sr)


def test_endswith_pat(test_sr, memory_leak_check):
    """
    tests error for endswith argument pat being non-str type
    """

    def impl(test_sr):
        return test_sr.str.endswith(123)

    with pytest.raises(BodoError, match="expected a string object"):
        bodo.jit(impl)(test_sr)


# ------------------------------ slice() ------------------------------ #
def test_slice_args(test_sr, memory_leak_check):
    """
    tests error for slice arguments being non-int type
    """

    def impl(test_sr):
        return test_sr.str.slice("x", 1, 2)

    def impl2(test_sr):
        return test_sr.str.slice(1, "x", 2)

    def impl3(test_sr):
        return test_sr.str.slice(1, 2, "x")

    with pytest.raises(BodoError, match="expected an int object, not"):
        bodo.jit(impl)(test_sr)
    with pytest.raises(BodoError, match="expected an int object, not"):
        bodo.jit(impl2)(test_sr)
    with pytest.raises(BodoError, match="expected an int object, not"):
        bodo.jit(impl3)(test_sr)


# ------------------------------ extract() ------------------------------ #
def test_extract_args(test_sr, memory_leak_check):
    """
    tests error for extract arguments being non-int type
    """

    def impl(test_sr, pat, flags, expand):
        return test_sr.str.extract(pat, 1, True)

    pat = "x"
    flags = 1
    expand = False
    with pytest.raises(BodoError, match="contains no capture groups"):
        bodo.jit(impl)(test_sr, pat, flags, expand)


# ------------------------------ join() ------------------------------ #
@pytest.mark.parametrize("input", [pd.Series([1, 2, 3])])
def test_join_input(input, memory_leak_check):
    """
    tests error for join with the input series not being ListStringArrayType or
    StringArrayType
    """

    def impl(input):
        return input.str.join("-")

    with pytest.raises(
        BodoError, match="input should be a series of string/binary or arrays"
    ):
        bodo.jit(impl)(input)


@pytest.mark.parametrize(
    "input", [pd.Series([["aaa", "bbb", "ccc"], ["ddd", "eee", "fff"]])]
)
def test_join_sep(input, memory_leak_check):
    """
    tests error for join's argument 'sep' being non-str type
    """

    def impl(input):
        return input.str.join(1)

    with pytest.raises(BodoError, match="expected a string object"):
        bodo.jit(impl)(input)


def test_center_errorcheck(memory_leak_check):
    def f(S):
        return S.str.center(3)

    S = pd.Series(pd.date_range(start="1/1/2018", periods=5, freq="3ms"))
    with pytest.raises(BodoError, match="input should be a series of string"):
        bodo.jit(f)(S)


@pytest.mark.slow
def test_unsupported_str_method(memory_leak_check):
    """Raise Bodo error for unsupported str methods"""

    def test_impl():
        return pd.Series([" 123", "abc  "]).str.cat(sep=" ")

    with pytest.raises(BodoError, match="must be a DataFrame"):
        bodo.jit(test_impl)()
