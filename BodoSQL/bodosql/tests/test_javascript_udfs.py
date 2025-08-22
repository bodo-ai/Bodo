"""
Definitions of UDFs used in this test in our Bodo Snowflake account under the SYSADMIN user
in TEST_DB.PUBLIC:

create or replace function JAVASCRIPT_ADD_ONE(n DOUBLE)
returns DOUBLE
LANGUAGE JAVASCRIPT
as
$$
  return N+1
$$

create or replace function JAVASCRIPT_CONCAT(s VARCHAR, t VARCHAR)
returns VARCHAR
LANGUAGE JAVASCRIPT
as
$$
  return S + T
$$

create or replace function JAVASCRIPT_ADD_ONE_WRAPPER(n DOUBLE)
returns DOUBLE
LANGUAGE SQL
as
$$
  SELECT JAVASCRIPT_ADD_ONE(n)
$$

create or replace function JAVASCRIPT_ADD_ONE_WRAPPER_WRAPPER(n DOUBLE)
returns DOUBLE
LANGUAGE SQL
as
$$
  SELECT JAVASCRIPT_ADD_ONE_WRAPPER(n)
$$

create or replace function test_regex_udf(A varchar) RETURNS DOUBLE LANGUAGE JAVASCRIPT AS
$$
try {
return parseInt(A.match(/(\\d+).*?(\\d+).*?(\\d+)/)[2]);
} catch (e) {
    return null;
}
$$
"""

import datetime

import numpy as np
import pandas as pd
import pytest

import bodo
import bodosql
from bodo.libs.bool_arr_ext import BooleanArrayType
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.tests.conftest import (  # noqa
    enable_numba_alloc_stats,
    memory_leak_check,
)
from bodo.tests.utils import check_func
from bodo.utils.typing import MetaType
from bodosql.kernels import (
    create_javascript_udf,
    delete_javascript_udf,
    execute_javascript_udf,
)
from bodosql.tests.conftest import pytest_mark_javascript
from bodosql.tests.test_types.snowflake_catalog_common import (
    test_db_snowflake_catalog,  # noqa
)

pytestmark = pytest_mark_javascript


def test_javascript_udf_no_args_return_int(memory_leak_check):
    """
    Test a simple UDF without arguments that returns an integer value.
    """
    body = MetaType("return 2 + 1")
    args = MetaType(())
    ret_type = IntegerArrayType(bodo.types.int32)

    def f():
        f = create_javascript_udf(body, args, ret_type)
        out_arr = execute_javascript_udf(f, ())
        delete_javascript_udf(f)
        return out_arr

    expected_output = 3
    check_func(f, (), py_output=expected_output)


def test_javascript_interleaved_execution(memory_leak_check):
    """
    Test interleaved execution of two UDFs.
    """
    body_a = MetaType("return 2 + 1")
    body_b = MetaType("return 2 + 2")
    args = MetaType(())
    ret_type = IntegerArrayType(bodo.types.int32)

    def f():
        a = create_javascript_udf(body_a, args, ret_type)
        b = create_javascript_udf(body_b, args, ret_type)
        out_1 = execute_javascript_udf(a, ())
        out_2 = execute_javascript_udf(b, ())
        out_3 = execute_javascript_udf(a, ())
        out_4 = execute_javascript_udf(b, ())
        delete_javascript_udf(a)
        delete_javascript_udf(b)
        return out_1, out_2, out_3, out_4

    expected_output = (3, 4, 3, 4)
    check_func(f, (), py_output=expected_output)


def test_javascript_udf_single_arg_return_int(memory_leak_check):
    """
    Test a simple UDF with a single argument that returns an integer value.
    """
    body = MetaType("return A + 1")
    args = MetaType(("A",))
    ret_type = IntegerArrayType(bodo.types.int32)

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
    ret_type = IntegerArrayType(bodo.types.int32)

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
    ret_type = IntegerArrayType(bodo.types.int32)

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
    ret_type = IntegerArrayType(bodo.types.int32)

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


def test_javascript_udf_regex(test_db_snowflake_catalog, memory_leak_check):
    """
    Test a JavaScript UDF that contains regex syntax in the definition.
    The function looks for a sequence of 3 numbers (can be multiple
    integers) with any characters in between, and returns the middle
    number as an integer.
    """

    def impl(bc, query):
        return bc.sql(query)

    query = "select TEST_REGEX_UDF(A) as OUTPUT from LOCAL_TABLE"
    bc = bodosql.BodoSQLContext(
        {
            "LOCAL_TABLE": pd.DataFrame(
                {
                    "A": [
                        "I count 3 dogs, 17 chickens, and 103 goldfish",
                        "1 2 3 4 5 6 7 8 9",
                        "3*64=192",
                        "123456789",
                        "one two three four five",
                        "12.34.56.78",
                    ]
                }
            )
        },
        catalog=test_db_snowflake_catalog,
    )
    check_func(
        impl,
        (bc, query),
        py_output=pd.DataFrame(
            {"OUTPUT": pd.array([17, 2, 64, 8, None, 34], dtype=pd.Int32Dtype())}
        ),
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )


def test_javascript_invalid_body(memory_leak_check):
    """
    Test a UDF with an invalid body and check if an exception is raised.
    """
    body = MetaType("return 2 + '")
    args = MetaType(())
    ret_type = IntegerArrayType(bodo.types.int32)

    @bodo.jit
    def f():
        f = create_javascript_udf(body, args, ret_type)
        out = execute_javascript_udf(f, ())
        delete_javascript_udf(f)
        return out

    with pytest.raises(Exception, match="1: SyntaxError: Invalid or unexpected token"):
        f()


def test_javascript_throws_exception(memory_leak_check):
    """
    Test a UDF that throws an exception and check if the exception is raised.
    """
    body = MetaType("throw 'error_string'")
    args = MetaType(())
    ret_type = IntegerArrayType(bodo.types.int32)

    @bodo.jit
    def f():
        f = create_javascript_udf(body, args, ret_type)
        out = execute_javascript_udf(f, ())
        delete_javascript_udf(f)
        return out

    with pytest.raises(Exception, match="1: error_string"):
        f()


def test_javascript_unicode_in_body(memory_leak_check):
    body = MetaType("return 'hëllo'")
    args = MetaType(())
    ret_type = bodo.string_array_type

    def f():
        f = create_javascript_udf(body, args, ret_type)
        out_arr = execute_javascript_udf(f, ())
        delete_javascript_udf(f)
        return out_arr

    expected_output = "hëllo"
    check_func(f, (), py_output=expected_output)


@pytest.fixture
def calculate_upc():
    """Fixture for setting up theCALCULATE_UPC UDF.
    Returns a tuple of the function body string, the argument
    names tuple, and the return type."""
    body = """
/*
** Calculate valid UPC from barcode string.
**
** Given a text string, assume it's a numeric barcode and calculate and
** return a formatted and padded UPC.  The default output length is 12
** digits (UPC12) but this can be overridden to create a 13 digit EAN13
** for a 14 digit GTIN14.  If has_check is TRUE, never add a check digit.
** If FALSE, always add a check digit.  If None (default) only add if we
** think it's needed.  Default action is to mask Type2 UPCs.  To turn off
** the masking, set the mask parameter to FALSE.
**
** Inputs:   barcode     VARCHAR
**           length      INTEGER (Default 12)
**           has_check   BOOLEAN (Default NULL)
**           mask_type2  BOOLEAN (Default TRUE)
**
** Outputs:  upc         TEXT
**
** Language: javascript
*/
if ( P_LENGTH < 12 )
{
    P_LENGTH = 12;
}
else if ( P_LENGTH > 14 )
{
    P_LENGTH = 14;
}
else
{
    P_LENGTH = parseInt(P_LENGTH);
}
function calculate_check_digit(p_barcode) {
    let barcode = undefined;
    let check_digit = 0;
    let odd_pos = true;
    // If barcode is a Type2 UPC and has been masked, check digit calculation
    // will fail (it has been masked).  In this case, simply return 0.
    if ( (p_barcode.length == 12 && p_barcode.slice(0, 1) ==   '2') ||
         (p_barcode.length == 13 && p_barcode.slice(0, 2) ==  '02') ||
         (p_barcode.length == 14 && p_barcode.slice(0, 3) == '002') )
    {
        if ( p_barcode.slice(-6) == '000000' )
        {
            return 0;
        }
    }
    try
    {
        // Pull out only digits and left trim zeros
        barcode = p_barcode.match(/(\\d+)/)[0].replace(/^0+/,"");
    }
    catch(err)
    {
        return undefined;
    }
    barcode.split("").reverse().forEach(function(char) {
        if ( odd_pos == true )
        {
            check_digit += parseInt(char)*3;
        }
        else
        {
            check_digit += parseInt(char);
        }
        odd_pos = !(odd_pos) // alternate
    });
    check_digit = check_digit % 10;
    check_digit = 10 - check_digit;
    check_digit = check_digit % 10;
    return check_digit;
};
function convert_upce_to_upca(p_upce_value) {
    let upce_value = undefined;
    let middle_digits = undefined;
    let mfrnum = undefined;
    let itemnum = undefined;
    let newmsg = undefined;
    try
    {
        // Pull out only digits and left trim zeros
        upce_value = p_upce_value.match(/(\\d+)/)[0].replace(/^0+/,"");
    }
    catch(err)
    {
        return undefined;
    }
    
    if ( upce_value.length == 6 )
    {
        // Assume we're getting just middle 6 digits
        middle_digits = upce_value;
    }
    else if ( upce_value.length == 7 )
    {
        let check_digit = calculate_check_digit(upce_value.slice(0, 6));
        
        if ( check_digit != upce_value.slice(-1) )
        {
            // Truncate last digit: it's just check digit
            middle_digits = upce_value.slice(0, 6);
        }
        else
        {
            // Not a check digit; assume it's a number system
            middle_digits = upce_value.slice(1, 7);
        }
    }
    else if ( upce_value.length == 8 )
    {
        // Truncate first and last digit, assume first digit is number
        // system digit last digit is check digit
        middle_digits = upce_value.slice(1, 7);
    }
    else
    {
        return undefined;
    }
    let [dig1, dig2, dig3, dig4, dig5, dig6] = middle_digits.split("");
    if ( ["0", "1", "2"].includes(dig6) == true )
    {
        mfrnum = dig1 + dig2 + dig6 + "00";
        itemnum = "00" + dig3 + dig4 + dig5;
    }
    else if ( dig6 == "3" )
    {
        mfrnum = dig1 + dig2 + dig3 + "00";
        itemnum = "000" + dig4 + dig5;
    }
    else if ( dig6 == "4" )
    {
        mfrnum = dig1 + dig2 + dig3 + dig4 + "0";
        itemnum = "0000" + dig5;
    }
    else
    {
        mfrnum = dig1 + dig2 + dig3 + dig4 + dig5;
        itemnum = "0000" + dig6;
    }
    
    newmsg = "0" + mfrnum + itemnum;
    // Calculate check digit, they are the same for both UPCA and UPCE
    let check_digit = calculate_check_digit(newmsg);
    return newmsg + check_digit;
};
let barcode = undefined;
let upc = undefined;
let is_masked = false;
try
{
    // Pull out only digits and left trim zeros
    barcode = P_BARCODE.match(/(\\d+)/)[0].replace(/^0+/,"");
}
catch(err)
{
    return undefined;
}
if ( (barcode.length == 12 && barcode.slice(0, 1) ==   '2') ||
     (barcode.length == 13 && barcode.slice(0, 2) ==  '02') ||
     (barcode.length == 14 && barcode.slice(0, 3) == '002') )
{
    if ( barcode.slice(-6) == '000000' )
    {
        is_masked = true;
    }
}
if ( barcode.length >= 15 )
{
    // Lengths greater than 14
    return undefined;
}
else if ( barcode.length <= 3 )
{
    // Length = 1, 2, 3
    return undefined;
}
else if ( barcode.length == 4 && (barcode.slice(0, 1) == '3' || barcode.slice(0, 1) == '4') )
{
    // PLU: 3xxx, 4xxx
    return barcode;
}
else if ( barcode.length == 5 && (barcode.slice(0, 2) == '93' || barcode.slice(0, 2) == '94') )
{
    // PLU: 93xxx, 94xxx
    return barcode;
}
else if ( barcode.length == 5 && (barcode.slice(0, 1) == '3' || barcode.slice(0, 1) == '4') && calculate_check_digit(barcode.slice(0, -1)) == barcode.slice(-1) )
{
    // PLU: 3xxx, 4xxx (with checkdigit: remove!)
    return barcode.slice(0, 4);
}
else if ( barcode.length == 6 && (barcode.slice(0, 2) == '93' || barcode.slice(0, 2) == '94') && calculate_check_digit(barcode.slice(0, -1)) == barcode.slice(-1) )
{
    // PLU: 93xxx, 94xxx (with checkdigit: remove!)
    return barcode.slice(0, 5);
}
else if ( barcode.length <= 5 )
{
    // Length = 4, 5: bad PLU
    return undefined;
}
else
{
    // Length = 6 -> 14
    /*
    ** REMOVE ** if ( [6, 7].includes(barcode.length) )
    ** REMOVE ** {
    ** REMOVE **     barcode = convert_upce_to_upca(barcode);
    ** REMOVE ** }
    */
    let test_check_digit = calculate_check_digit(barcode.slice(0, -1));
    let check_digit = calculate_check_digit(barcode);
    if ( P_HAS_CHECK == true || is_masked == true)
    {
        upc = barcode;
    }
    else if ( P_HAS_CHECK == false)
    {
        upc = barcode + check_digit;
    }
    else
    {
        if ( barcode.slice(-1) != test_check_digit )
        {
            upc = barcode + check_digit;
        }
        else
        {
            upc = barcode;
        }
    }
}
upc = upc.padStart(P_LENGTH, '0');
// Correct for type-2 barcodes
// E3 default policy is to mask the last digits for type2 UPCs
if ( P_MASK_TYPE2 == true )
{
    if ( (upc.length == 12 && upc.slice(0, 1) ==   '2') ||
         (upc.length == 13 && upc.slice(0, 2) ==  '02') ||
         (upc.length == 14 && upc.slice(0, 3) == '002') )
    {
        upc = upc.slice(0, -6) + '000000';
    }
}
return upc;
"""
    args = ("P_BARCODE", "P_LENGTH", "P_HAS_CHECK", "P_MASK_TYPE2")
    return body, args, bodo.string_array_type


def test_javascript_udf_calculate_upc(calculate_upc, memory_leak_check):
    """
    Tests running theCALCULATE_UPC UDF. Answers derived
    from sample calculations with the actualUDF.
    """
    body, args, ret_type = calculate_upc
    body = MetaType(body)
    args = MetaType(args)

    barcodes = pd.array(
        [
            "0004529904438",
            "0065281081996",
            "0001650054118",
            None,
            "0002742633055",
        ]
    )
    answer = pd.array(
        [
            "045299044380",
            "652810819961",
            "016500541189",
            None,
            "027426330559",
        ]
    )

    def f(barcodes):
        f = create_javascript_udf(body, args, ret_type)
        out_arr = execute_javascript_udf(f, (barcodes, 12, None, True))
        delete_javascript_udf(f)
        return out_arr

    check_func(f, (barcodes,), py_output=answer)


@pytest.mark.parametrize(
    "body_text, ret_type, expected_output",
    [
        pytest.param(
            "return 255",
            IntegerArrayType(bodo.types.uint8),
            255,
            id="uint8",
        ),
        pytest.param(
            "return 2 ** 16 - 1",
            IntegerArrayType(bodo.types.uint16),
            2**16 - 1,
            id="uint16",
        ),
        pytest.param(
            "return 2 ** 32 -1",
            IntegerArrayType(bodo.types.uint32),
            2**32 - 1,
            id="uint32",
        ),
        # This is bigint syntax in javascript, all numerics are double by default which can't represent this value
        pytest.param(
            "return 2n ** 64n - 1n",
            IntegerArrayType(bodo.types.uint64),
            2**64 - 1,
            id="uint64",
        ),
        pytest.param(
            "return 2 ** 8 / 2 - 1",
            IntegerArrayType(bodo.types.int8),
            127,
            id="int8",
        ),
        pytest.param(
            "return 2 ** 16 / 2 - 1",
            IntegerArrayType(bodo.types.int16),
            2**16 / 2 - 1,
            id="int16",
        ),
        pytest.param(
            "return 2 ** 32 / 2 - 1",
            IntegerArrayType(bodo.types.int32),
            2**32 / 2 - 1,
            id="int32",
        ),
        # This is bigint syntax in javascript, all numerics are double by default which can't represent this value
        pytest.param(
            "return 2n ** 64n  / 2n - 1n",
            IntegerArrayType(bodo.types.int64),
            2**64 / 2 - 1,
            id="int64",
        ),
        pytest.param(
            "return 2 + 1.1",
            IntegerArrayType(bodo.types.float32),
            3.1,
            id="float32",
        ),
        pytest.param(
            # Bigger than a float32 can represent
            "return 4 * 10**40",
            IntegerArrayType(bodo.types.float64),
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
    args = MetaType(())

    def f():
        f = create_javascript_udf(body, args, ret_type)
        out_arr = execute_javascript_udf(f, ())
        delete_javascript_udf(f)
        return out_arr

    check_func(f, (), py_output=expected_output)


def test_javascript(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that a UDF that is written in JavaScript can be inlined.

    JAVASCRIPT_ADD_ONE is manually defined inside TEST_DB.PUBLIC.
    It takes one argument and returns it plus one.
    """

    def impl(bc, query):
        return bc.sql(query)

    bc = bodosql.BodoSQLContext(
        {"LOCAL_TABLE": pd.DataFrame({"A": [1, 2, 3, 4, None]})},
        catalog=test_db_snowflake_catalog,
    )

    query = "select JAVASCRIPT_ADD_ONE(A) as OUTPUT from LOCAL_TABLE\n"
    check_func(
        impl,
        (bc, query),
        py_output=pd.DataFrame({"OUTPUT": [2, 3, 4, 5, 1]}),
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )


def test_javascript_multiple_args(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that a UDF that is written in JavaScript can be inlined.

    JAVASCRIPT_CONCAT is manually defined inside TEST_DB.PUBLIC.
    It takes two arguments and calls + on them and returns VARCHAR.
    """

    def impl(bc, query):
        return bc.sql(query)

    bc = bodosql.BodoSQLContext(
        {"LOCAL_TABLE": pd.DataFrame({"A": [1] * 10, "B": ["abc"] * 10})},
        catalog=test_db_snowflake_catalog,
    )

    query = "select JAVASCRIPT_CONCAT(A, B) as OUTPUT from LOCAL_TABLE\n"
    check_func(
        impl,
        (bc, query),
        py_output=pd.DataFrame({"OUTPUT": ["1abc"] * 10}),
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )


def test_nested_javascript_inline(test_db_snowflake_catalog, memory_leak_check):
    """
    Test to insure that nested javascript functions work as expected.
    """

    def impl(bc, query):
        return bc.sql(query)

    bc = bodosql.BodoSQLContext(
        {"LOCAL_TABLE": pd.DataFrame({"A": [1] * 10})},
        catalog=test_db_snowflake_catalog,
    )

    query = "select JAVASCRIPT_ADD_ONE_WRAPPER(A) as OUTPUT FROM LOCAL_TABLE\n"

    check_func(
        impl,
        (bc, query),
        py_output=pd.DataFrame({"OUTPUT": [2] * 10}),
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )


def test_javascript_in_case(test_db_snowflake_catalog, memory_leak_check):
    """
    Test that a UDF that is written in JavaScript can be inlined when used in a case statement
    """

    def impl(bc, query):
        return bc.sql(query)

    bc = bodosql.BodoSQLContext(
        {
            "LOCAL_TABLE": pd.DataFrame(
                {"A": pd.Series([None, 1, 2, 3, 4], dtype="Int64")}
            )
        },
        catalog=test_db_snowflake_catalog,
    )
    query = "select case when A is not null then JAVASCRIPT_ADD_ONE(A) else 0 end as OUTPUT from LOCAL_TABLE"

    check_func(
        impl,
        (bc, query),
        py_output=pd.DataFrame({"OUTPUT": [0, 2, 3, 4, 5]}),
        check_dtype=False,
        sort_output=True,
        reset_index=True,
    )
