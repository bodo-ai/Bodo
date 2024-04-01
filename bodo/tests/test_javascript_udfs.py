import datetime

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.libs.bodosql_array_kernels import (
    create_javascript_udf,
    delete_javascript_udf,
    execute_javascript_udf,
)
from bodo.libs.bool_arr_ext import BooleanArrayType
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.tests.conftest import (
    pytest_mark_javascript,
)
from bodo.tests.utils import check_func
from bodo.utils.typing import MetaType

pytestmark = pytest_mark_javascript


def test_javascript_udf_no_args_return_int(memory_leak_check):
    """
    Test a simple UDF without arguments that returns an integer value.
    """
    body = MetaType("return 2 + 1")
    args = MetaType(tuple())
    ret_type = IntegerArrayType(bodo.int32)

    def f():
        f = create_javascript_udf(body, args, ret_type)
        out_arr = execute_javascript_udf(f, tuple())
        delete_javascript_udf(f)
        return out_arr

    expected_output = 3
    check_func(f, tuple(), py_output=expected_output)


def test_javascript_interleaved_execution(memory_leak_check):
    """
    Test interleaved execution of two UDFs.
    """
    body_a = MetaType("return 2 + 1")
    body_b = MetaType("return 2 + 2")
    args = MetaType(tuple())
    ret_type = IntegerArrayType(bodo.int32)

    def f():
        a = create_javascript_udf(body_a, args, ret_type)
        b = create_javascript_udf(body_b, args, ret_type)
        out_1 = execute_javascript_udf(a, tuple())
        out_2 = execute_javascript_udf(b, tuple())
        out_3 = execute_javascript_udf(a, tuple())
        out_4 = execute_javascript_udf(b, tuple())
        delete_javascript_udf(a)
        delete_javascript_udf(b)
        return out_1, out_2, out_3, out_4

    expected_output = (3, 4, 3, 4)
    check_func(f, tuple(), py_output=expected_output)


def test_javascript_udf_single_arg_return_int(memory_leak_check):
    """
    Test a simple UDF with a single argument that returns an integer value.
    """
    body = MetaType("return A + 1")
    args = MetaType(("A",))
    ret_type = IntegerArrayType(bodo.int32)

    def f(arr):
        f = create_javascript_udf(body, args, ret_type)
        out_arr = execute_javascript_udf(f, (arr,))
        delete_javascript_udf(f)
        return out_arr

    arr = np.arange(0, 5)
    expected_output = np.arange(1, 6)
    check_func(f, (arr,), py_output=expected_output)


@pytest.mark.parametrize(
    "inputs, answer",
    [
        pytest.param(
            (
                pd.array([None, -20, 88, None, 132], dtype=pd.Int32Dtype()),
                pd.array([42, 99, -105, None, -85], dtype=pd.Int32Dtype()),
            ),
            pd.array([None, 101, 137, None, 157], dtype=pd.Int32Dtype()),
            id="vector-vector",
        ),
        pytest.param(
            (
                0,
                pd.array([None, 20, 88, None, -132], dtype=pd.Int32Dtype()),
            ),
            pd.array([None, 20, 88, None, 132], dtype=pd.Int32Dtype()),
            id="scalar-vector",
        ),
        pytest.param(
            (
                pd.array([None, -20, 88, None, 132], dtype=pd.Int32Dtype()),
                0,
            ),
            pd.array([None, 20, 88, None, 132], dtype=pd.Int32Dtype()),
            id="vector-scalar",
        ),
        pytest.param(
            (
                pd.array([None, 20, -88, None, 132], dtype=pd.Int32Dtype()),
                None,
            ),
            pd.array([None] * 5, dtype=pd.Int32Dtype()),
            id="vector-null",
        ),
        pytest.param((-231, 160), 281, id="scalar-scalar"),
        pytest.param((None, -6), None, id="null-scalar"),
        pytest.param((None, None), None, id="null-null"),
    ],
)
def test_javascript_udf_multiple_args_return_int(inputs, answer, memory_leak_check):
    """
    Test a simple UDF with multiple integer argument that returns an integer value.
    """
    body = MetaType("return (A == null || B == null) ? null : Math.sqrt(A * A + B * B)")
    args = MetaType(("A", "B"))
    ret_type = IntegerArrayType(bodo.int32)

    def f(arr0, arr1):
        f = create_javascript_udf(body, args, ret_type)
        out_arr = execute_javascript_udf(f, (arr0, arr1))
        delete_javascript_udf(f)
        return out_arr

    check_func(f, inputs, py_output=answer)


@pytest.mark.parametrize(
    "flags, answer",
    [
        pytest.param((True, True), 42, id="value-value"),
        pytest.param((True, False), None, id="value-null"),
        pytest.param((False, True), None, id="null-value"),
        pytest.param((False, False), None, id="null-null"),
    ],
)
def test_javascript_udf_optional_args_return_int(flags, answer, memory_leak_check):
    """
    Test a simple UDF with multiple integer argument that returns an integer value.
    """
    body = MetaType("return (x == null || y == null) ? null : x * y")
    args = MetaType(("x", "y"))
    ret_type = IntegerArrayType(bodo.int32)

    def f(flag0, flag1):
        arg0 = 6 if flag0 else None
        arg1 = 7 if flag1 else None
        f = create_javascript_udf(body, args, ret_type)
        out_arr = execute_javascript_udf(f, (arg0, arg1))
        delete_javascript_udf(f)
        return out_arr

    check_func(f, flags, py_output=answer)


@pytest.mark.parametrize(
    "arr, answer",
    [
        pytest.param(
            (
                pd.array(
                    [
                        "the lazy fox the fox fox",
                        None,
                        "alphabet soup is delicious",
                        "aa aa aa b,b,b,b a",
                        "",
                    ]
                ),
            ),
            pd.array([4, None, 9, 7, 0], dtype=pd.Int32Dtype()),
            id="vector",
        ),
        pytest.param(("a big dog jumped over a fence",), 6, id="scalar"),
        pytest.param((None,), None, id="null"),
    ],
)
def test_javascript_udf_string_args_return_int(arr, answer, memory_leak_check):
    """
    Test a UDF with multiple strings argument that returns an integer value.
    """
    # Function: find the length of the longest word in the string.
    body = MetaType(
        """
    if (sentence == null) return null;
    var longest = 0;
    for(word of sentence.split(' ')) {
        if (word.length > longest) longest = word.length;
    }
    return longest;"""
    )
    args = MetaType(("sentence",))
    ret_type = IntegerArrayType(bodo.int32)

    def f(arr):
        f = create_javascript_udf(body, args, ret_type)
        out_arr = execute_javascript_udf(f, (arr,))
        delete_javascript_udf(f)
        return out_arr

    check_func(f, arr, py_output=answer)


@pytest.mark.parametrize(
    "arr, answer",
    [
        pytest.param(
            (
                pd.array(
                    [
                        0,
                        None,
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                    ]
                ),
            ),
            pd.array(["0", None, "1", "3", "6", "2", "7", "13", "20"]),
            id="vector",
        ),
        pytest.param((8,), "12", id="scalar"),
        pytest.param((None,), None, id="null"),
    ],
)
def test_javascript_udf_complex_function(arr, answer, memory_leak_check):
    """
    Test a UDF that takes in a number and returns the corresponding value
    of the recaman sequence as a string.
    """
    body = MetaType(
        """
    if (x == null) return null;
    let sequence = [0]
    let idx = 1
    while (sequence.length <= x) {
        let minus = sequence[idx-1] - idx;
        let plus = sequence[idx-1] + idx;
        if (minus < 0 || sequence.indexOf(minus) != -1) {
            sequence.push(plus);
        } else {
            sequence.push(minus);
        }
        idx++;
    }
    return sequence[idx-1].toString()"""
    )
    args = MetaType(("x",))
    ret_type = bodo.string_array_type

    def f(arr):
        f = create_javascript_udf(body, args, ret_type)
        out_arr = execute_javascript_udf(f, (arr,))
        delete_javascript_udf(f)
        return out_arr

    check_func(f, arr, py_output=answer)


def test_javascript_invalid_body(memory_leak_check):
    """
    Test a UDF with an invalid body and check if an exception is raised.
    """
    body = MetaType("return 2 + '")
    args = MetaType(tuple())
    ret_type = IntegerArrayType(bodo.int32)

    @bodo.jit
    def f():
        f = create_javascript_udf(body, args, ret_type)
        out = execute_javascript_udf(f, tuple())
        delete_javascript_udf(f)
        return out

    with pytest.raises(Exception, match="1: SyntaxError: Invalid or unexpected token"):
        f()


def test_javascript_throws_exception(memory_leak_check):
    """
    Test a UDF that throws an exception and check if the exception is raised.
    """
    body = MetaType("throw 'error_string'")
    args = MetaType(tuple())
    ret_type = IntegerArrayType(bodo.int32)

    @bodo.jit
    def f():
        f = create_javascript_udf(body, args, ret_type)
        out = execute_javascript_udf(f, tuple())
        delete_javascript_udf(f)
        return out

    with pytest.raises(Exception, match="1: error_string"):
        f()


def test_javascript_unicode_in_body(memory_leak_check):
    body = MetaType("return 'hëllo'")
    args = MetaType(tuple())
    ret_type = bodo.string_array_type

    def f():
        f = create_javascript_udf(body, args, ret_type)
        out_arr = execute_javascript_udf(f, tuple())
        delete_javascript_udf(f)
        return out_arr

    expected_output = "hëllo"
    check_func(f, tuple(), py_output=expected_output)


@pytest.mark.parametrize(
    "body_text, ret_type, expected_output",
    [
        pytest.param(
            "return 255",
            IntegerArrayType(bodo.uint8),
            255,
            id="uint8",
        ),
        pytest.param(
            "return 2 ** 16 - 1",
            IntegerArrayType(bodo.uint16),
            2**16 - 1,
            id="uint16",
        ),
        pytest.param(
            "return 2 ** 32 -1",
            IntegerArrayType(bodo.uint32),
            2**32 - 1,
            id="uint32",
        ),
        pytest.param(
            "return 2 ** 64 - 1",
            IntegerArrayType(bodo.uint64),
            2**64 - 1,
            id="uint64",
        ),
        pytest.param(
            "return 2 ** 8 / 2 - 1",
            IntegerArrayType(bodo.int8),
            127,
            id="int8",
        ),
        pytest.param(
            "return 2 ** 16 / 2 - 1",
            IntegerArrayType(bodo.int16),
            2**16 / 2 - 1,
            id="int16",
        ),
        pytest.param(
            "return 2 ** 32 / 2 - 1",
            IntegerArrayType(bodo.int32),
            2**32 / 2 - 1,
            id="int32",
        ),
        pytest.param(
            "return 2 ** 64 / 2 - 1",
            IntegerArrayType(bodo.int64),
            2**64 / 2 - 1,
            id="int64",
        ),
        pytest.param(
            "return 2 + 1.1",
            IntegerArrayType(bodo.float32),
            3.1,
            id="float32",
        ),
        pytest.param(
            # Bigger than a float32 can represent
            "return 4 * 10**40",
            IntegerArrayType(bodo.float64),
            float(4 * 10**40),
            id="float64",
        ),
        pytest.param(
            "return 2 + 1",
            BooleanArrayType(),
            True,
            id="bool",
        ),
        pytest.param(
            "return 'hello'",
            bodo.string_array_type,
            "hello",
            id="string",
        ),
        pytest.param(
            "return new Date('2021-01-01')",
            bodo.datetime_date_array_type,
            datetime.date(2021, 1, 1),
            id="date",
        ),
        pytest.param(
            "return new Uint8Array([1, 2, 3])",
            bodo.binary_array_type,
            b"\x01\x02\x03",
            id="binary",
        ),
    ],
)
def test_javascript_return(body_text, ret_type, expected_output, memory_leak_check):
    """
    Test a UDF that returns a value for all supported types.
    """
    body = MetaType(body_text)
    args = MetaType(tuple())

    def f():
        f = create_javascript_udf(body, args, ret_type)
        out_arr = execute_javascript_udf(f, tuple())
        delete_javascript_udf(f)
        return out_arr

    check_func(f, tuple(), py_output=expected_output)
