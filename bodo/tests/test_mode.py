import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bodo
from bodo.tests.utils import (
    check_func,
    nullable_float_arr_maker,
    pytest_slow_unless_codegen,
)

# Skip unless any library or BodoSQL codegen or files were changed
pytestmark = pytest_slow_unless_codegen


def make_integer_mode_test_params(name, dtype, nullable, is_slow):
    """
    Generates a pytest param for test_mode with integer arguments.
    The parameter contains a mapping of each unique key to all of
    the values, in that group, a mapping of each key to the most
    frequent element in that group, the dtype to use (if any)
    when placing the elements in a Series, and whether to use
    dictionary encoding (the last is always False for this function).

    Args:
        name [string]: The name of this argument parameterization.
        (e.g. "int8_nullable" or "uint64_numpy")
        dtype [dtype]: Which integer dtype to use.
        nullable [boolean]: If true produces a nullable array with some
        null terms, otherwise produces a numpy array where the null terms
        are replaced with an integer.
        is_slow [boolean]: Is this argument parameterization a slow test?
    """
    null_opt = None if nullable else 127
    data_dict = {
        "A": [0, 0, 0, 1, 1],
        "B": [0, 0, 1, 1, 1] * 12,
        "C": [49, 61, 61, 72, null_opt],
        "D": [2, 17, 17, null_opt, null_opt, null_opt],
        "E": [null_opt] * 40,
        "F": [12] * 7,
        "G": list(range(100)) + [42],
        "H": [100, 100, 101, 101, 102, 102, 102],
        "I": [null_opt] * 40 + [64],
    }
    answer_dict = {
        "A": 0,
        "B": 1,
        "C": 61,
        "D": 17 if nullable else null_opt,
        "E": null_opt,
        "F": 12,
        "G": 42,
        "H": 102,
        "I": 64 if nullable else null_opt,
    }
    slow_mark_opt = pytest.mark.slow if is_slow else ()
    return pytest.param(
        data_dict, answer_dict, dtype, False, id=name, marks=slow_mark_opt
    )


def make_date_time_mode_test_params(name, format_str, is_slow):
    """
    Generates a pytest param for test_mode with date/time/etc. arguments.
    The parameter contains a mapping of each unique key to all of
    the values, in that group, a mapping of each key to the most
    frequent element in that group, the dtype to use (if any)
    when placing the elements in a Series, and whether to use
    dictionary encoding (the last is always False for this function).

    Args:
        name [string]: The name of this argument parameterization.
        (e.g. "date" or "timestamp_ltz")
        format_str [str]: A string with a format injection site {}
        designed so that if an integer were to be injected and the
        string were evaluated, it would produce a value of the desired
        type based on the input integer.
        is_slow [boolean]: Is this argument parameterization a slow test?
    """

    def val_from_format(x):
        return eval(
            format_str.format(x), {"bodo": bodo, "datetime": datetime, "pd": pd}
        )

    # Creating lambdas to avoid evaluating "bodo.types.Time" outside of test
    # where JIT is not imported.
    def create_data_dict():
        import bodo.decorators  # noqa

        return {
            13: [val_from_format(500)] * 2 + [val_from_format(5000)] * 3,
            10: [val_from_format((i**8) % (10**4 - 1)) for i in range(500)],
            -1: [None] * 4 + [val_from_format(0)] + [val_from_format(7**6)] * 3,
            50: [None] * 4,
            42: [val_from_format(2**i) for i in range(16)] + [val_from_format(2**10)],
            1024: [val_from_format(76543)],
            256: [val_from_format(2**i) for i in [10, 10, 11, 11, 11, 12]],
        }

    def create_answer_dict():
        import bodo.decorators  # noqa

        return {
            13: val_from_format(5000),
            10: val_from_format(6561),
            -1: val_from_format(7**6),
            50: None,
            42: val_from_format(2**10),
            1024: val_from_format(76543),
            256: val_from_format(2**11),
        }

    slow_mark_opt = pytest.mark.slow if is_slow else ()
    dtype = pd.ArrowDtype(pa.time64("ns")) if name == "time" else None
    return pytest.param(
        create_data_dict, create_answer_dict, dtype, False, id=name, marks=slow_mark_opt
    )


def make_bool_mode_test_params(name, dtype, nullable, is_slow):
    """
    Generates a pytest param for test_mode with boolean arguments.
    The parameter contains a mapping of each unique key to all of
    the values, in that group, a mapping of each key to the most
    frequent element in that group, the dtype to use (if any)
    when placing the elements in a Series, and whether to use
    dictionary encoding (the last is always False for this function).

    Args:
        name [string]: The name of this argument parameterization.
        (e.g. "bool_nullable" or "bool_numpy")
        dtype [dtype]: Which integer dtype to use.
        nullable [boolean]: If true produces a nullable array with some
        null terms, otherwise produces a numpy array where the null terms
        are replaced with False.
        is_slow [boolean]: Is this argument parameterization a slow test?
    """
    null_opt = None if nullable else False
    data_dict = {
        1: [False] * 3 + [True] * 2,
        2: [False, True, True] * 10,
        3: [False, True, True, True, null_opt],
        4: [False, False, True, null_opt, null_opt, null_opt],
        5: [null_opt] * 40,
        6: [True] * 50,
        7: [False] * 100,
    }
    answer_dict = {1: False, 2: True, 3: True, 4: False, 5: null_opt, 6: True, 7: False}
    slow_mark_opt = pytest.mark.slow if is_slow else ()
    return pytest.param(
        data_dict, answer_dict, dtype, False, id=name, marks=slow_mark_opt
    )


def make_float_mode_test_params(name, dtype, format_str, is_slow):
    """
    Generates a pytest param for test_mode with float/decimal. arguments.
    The parameter contains a mapping of each unique key to all of
    the values, in that group, a mapping of each key to the most
    frequent element in that group, the dtype to use (if any)
    when placing the elements in a Series, and whether to use
    dictionary encoding (the last is always False for this function).

    Note: this function does not produce float arrays with both
    NaN and NULL

    Args:
        name [string]: The name of this argument parameterization.
        (e.g. "float32" or "decimal")
        dtype [dtype]: Which dtype to use, if any).
        format_str [str]: A string with a format injection site {}
        designed so that if an integer/float were to be injected and the
        string were evaluated, it would produce a value of the desired
        type based on the input number.
        is_slow [boolean]: Is this argument parameterization a slow test?
    """
    val_from_format = (
        lambda x: None
        if x is None
        else eval(format_str.format(x), {"np": np, "Decimal": Decimal})
    )
    data_dict = {
        "1": [val_from_format(i) for i in [0.5, 0.5, -12345, -12345, -12345]],
        "2": [val_from_format(i) for i in ([-1.0, 0, 0, 2.5, 2.5, 2.5, 5, 1024])],
        "3": [val_from_format(i) for i in [None] * 7 + [0] * 3],
        "4": [val_from_format(i) for i in [None, None, None, 42.25, 42.25, 43.0]],
        "5": [None] * 5,
    }
    answer_dict = {
        "1": val_from_format(-12345),
        "2": val_from_format(2.5),
        "3": val_from_format(0.0),
        "4": val_from_format(42.25),
        "5": None,
    }
    slow_mark_opt = pytest.mark.slow if is_slow else ()
    return pytest.param(
        data_dict, answer_dict, dtype, False, id=name, marks=slow_mark_opt
    )


def make_string_mode_test_params(name, alphabet, is_dict, is_slow):
    """
    Generates a pytest param for test_mode with string/binary arguments.
    The parameter contains a mapping of each unique key to all of
    the values, in that group, a mapping of each key to the most
    frequent element in that group, the dtype to use (if any)
    when placing the elements in a Series, and whether to use
    dictionary encoding.

    Args:
        name [string]: The name of this argument parameterization.
        (e.g. "string_ascii" or "string_nonascii_dict")
        alphabet [string]: A sequence of 26 unique characters
        (strings of binary) where substrings of the sequence
        are used to build the values in the groups.
        is_dict [boolean]: Should dictionary encoding be used?
        is_slow [boolean]: Is this argument parameterization a slow test?
    """
    data_dict = {
        "1": [alphabet[:3]] * 3 + [alphabet[-3:]] * 2,
        "2": [alphabet[-i:] for i in range(1, 20)]
        + [alphabet[i : i + 2] for i in range(25)]
        + [alphabet],
        "3": [None] * 5 + [alphabet] * 3 + [alphabet[:13]] * 2,
        "4": [alphabet[:0]] * 2 + [alphabet[-3:]],
        "5": [None] * 1,
        "6": [alphabet[i : i + 4] for i in range(23)] + [alphabet[6:10]],
    }
    answer_dict = {
        "1": alphabet[:3],
        "2": alphabet[-2:],
        "3": alphabet,
        "4": alphabet[:0],
        "5": None,
        "6": alphabet[6:10],
    }
    slow_mark_opt = pytest.mark.slow if is_slow else ()
    return pytest.param(
        data_dict, answer_dict, None, is_dict, id=name, marks=slow_mark_opt
    )


@pytest.mark.parametrize(
    "data_dict, answer_dict, dtype, is_dict",
    [
        make_integer_mode_test_params("int8_nullable", pd.Int8Dtype(), True, False),
        make_integer_mode_test_params("int16_nullable", pd.Int16Dtype(), True, False),
        make_integer_mode_test_params("int32_nullable", pd.Int32Dtype(), True, False),
        make_integer_mode_test_params("int64_nullable", pd.Int64Dtype(), True, False),
        make_integer_mode_test_params("uint8_nullable", pd.UInt8Dtype(), True, False),
        make_integer_mode_test_params("uint16_nullable", pd.UInt16Dtype(), True, False),
        make_integer_mode_test_params("uint32_nullable", pd.UInt32Dtype(), True, False),
        make_integer_mode_test_params("uint64_nullable", pd.UInt64Dtype(), True, False),
        make_integer_mode_test_params("int8_numpy", np.int8, False, False),
        make_integer_mode_test_params("int16_numpy", np.int16, False, False),
        make_integer_mode_test_params("int32_numpy", np.int32, False, False),
        make_integer_mode_test_params("int64_numpy", np.int64, False, False),
        make_integer_mode_test_params("uint8_numpy", np.uint8, False, False),
        make_integer_mode_test_params("uint16_numpy", np.uint16, False, False),
        make_integer_mode_test_params("uint32_numpy", np.uint32, False, False),
        make_integer_mode_test_params("uint64_numpy", np.uint64, False, False),
        make_date_time_mode_test_params(
            "timestamp_naive",
            "pd.Timestamp('1999-12-31') + pd.Timedelta(hours={})",
            False,
        ),
        make_date_time_mode_test_params(
            "timestamp_ltz",
            "pd.Timestamp('1999-12-31', tz='US/Pacific') + pd.Timedelta(hours={})",
            False,
        ),
        make_date_time_mode_test_params("timedelta", "pd.Timedelta(hours={})", False),
        make_date_time_mode_test_params(
            "date", "datetime.date.fromordinal(710000+{})", False
        ),
        make_date_time_mode_test_params(
            "time", "bodo.types.Time(microsecond={}**2)", False
        ),
        make_bool_mode_test_params("bool_nullable", pd.BooleanDtype(), True, False),
        make_bool_mode_test_params("bool_numpy", None, False, False),
        make_string_mode_test_params(
            "string_ascii", "#Alphab3tSOUPI5DeLic1ous42", False, False
        ),
        make_string_mode_test_params(
            "string_ascii_dict", "#Alphab3tSOUPI5DeLic1ous42", True, False
        ),
        make_string_mode_test_params(
            "string_non_ascii", "♬♫♯.🐍💚📈 㗨⅐⅛⅑€฿∞¿§÷yåY◊∑›º±", False, False
        ),
        make_string_mode_test_params(
            "string_non_ascii_dict", "♬♫♯.🐍💚📈 㗨⅐⅛⅑€฿∞¿§÷yåY◊∑›º±", True, False
        ),
        make_string_mode_test_params(
            "binary", b"abcdefghijKLMNOPQRST012345", False, False
        ),
        make_float_mode_test_params("decimal", None, "Decimal({})", False),
        make_float_mode_test_params(
            "float32_no_nan", pd.Float32Dtype(), "np.float32({})", False
        ),
        make_float_mode_test_params(
            "float64_no_nan", pd.Float64Dtype(), "np.float64({})", False
        ),
    ],
)
def test_mode(data_dict, answer_dict, dtype, is_dict, memory_leak_check):
    """Tests calling a groupby with mode, which is used by BodoSQL and not part
    of regular pandas, on all possible datatypes. For convenience, no ties
    are present. The rows of the DataFrame are shuffled randomly at the start.

    Args:

        data_dict [Dict[any, List[any]]]: Mapping of each groupby key
        to a list of values from which the mode is being sought.
        answer_dict [Dict[any, any]]: Mapping of each group key to
        the mode of the corresponding group.
        dtype [dtype]: What dtype should be used for the values.
        id_dict [boolean]: Should dictionary encoding be used.

    For example, if we have the following arguments:
        data_dict = {"A": [1, 5, 4, 5], "B": [1, 1, 2, 1]}
        answer_dict = {"A": 5, "B": 1}
        dtype = np.int8
        is_dict = False

    Then we would have the following input DataFrame (with the rows
    shuffled into any order):
        key     data
        A       1
        A       5
        A       4
        A       5
        B       1
        B       1
        B       2
        B       1

    And the following expected output:
        key     data
        A       5
        B       1
    """
    # Create data_dict and answer_dict if they are lazy parameters
    # Avoids importing the compiler at collection time.
    data_dict = data_dict() if callable(data_dict) else data_dict
    answer_dict = answer_dict() if callable(answer_dict) else answer_dict

    def impl(df):
        # Note we choose all of these flag + code format because
        # these are the generated SQL flags
        return df.groupby(["key"], as_index=False, dropna=False).agg(
            result=pd.NamedAgg(column="data", aggfunc="mode")
        )

    # Place the values of the data dict into a list of keys and
    # a corresponding list of data entries
    keys = []
    data = []
    for key in data_dict:
        for val in data_dict[key]:
            keys.append(key)
            data.append(val)

    # Randomly shuffle the order of the rows
    keys_reordered = []
    data_reordered = []
    rng = np.random.default_rng(42)
    ordering = np.arange(len(keys))
    ordering = rng.permutation(ordering)
    for i in ordering:
        keys_reordered.append(keys[i])
        data_reordered.append(data[i])

    # Place the input data into a DataFrame, and if need be convert the
    # data to dicitonary encoding
    if is_dict:
        data_reordered = pa.array(
            data_reordered, type=pa.dictionary(pa.int32(), pa.string())
        )
    df = pd.DataFrame(
        {
            "key": keys_reordered,
            "data": pd.Series(data_reordered, dtype=dtype),
        }
    )

    # For each distinct key, create a (key, mode) entry in
    # the output table
    group_keys = []
    group_modes = []
    for key in answer_dict:
        group_keys.append(key)
        group_modes.append(answer_dict[key])
    expected_output = pd.DataFrame(
        {
            "key": group_keys,
            "result": pd.Series(group_modes, dtype=dtype),
        }
    )

    check_func(
        impl,
        (df,),
        sort_output=True,
        reset_index=True,
        py_output=expected_output,
        check_names=False,
        use_dict_encoded_strings=is_dict,
    )


def test_mode_nullable_floats(memory_leak_check):
    """Tests calling a groupby with mode, which is used by BodoSQL and not part
    of regular pandas, on all possible datatypes. For convenience, no ties
    are present. The rows of the DataFrame are shuffled randomly at the start.

    This version of the function tests nullable float types specifically, since
    they require special handling.
    """

    def impl(df):
        # Note we choose all of these flag + code format because
        # these are the generated SQL flags
        return df.groupby(["key"], as_index=False, dropna=False).agg(
            result=pd.NamedAgg(column="data", aggfunc="mode")
        )

    data_dict = {
        "A": [0.0, 1.0, 2.0, 1.0, 1.0, 0.0, 0.0, 2.0, 0.0],
        "B": [0.0, None, 1.0, None, 2.0, None, 1.0, None],
        "C": [None] * 5,
        "D": [-1.5] * 3 + [np.nan] * 2,
        "E": [42.75] * 2 + [np.nan] * 3,
        "F": [None] * 5 + [10.0] * 2 + [np.nan] * 3,
        "G": [None] * 5 + [-64.0] * 2 + [np.nan] * 1,
    }
    answer_dict = {
        "A": 0.0,
        "B": 1.0,
        "C": None,
        "D": -1.5,
        "E": np.nan,
        "F": np.nan,
        "G": -64.0,
    }

    # Convert data_dict into a list of (key, value) pairs
    # and then shuffle the ordering
    pairs = []
    for key in data_dict:
        for val in data_dict[key]:
            pairs.append((key, val))
    rng = np.random.default_rng(42)
    pairs = rng.permutation(np.array(pairs))

    # Place the keys and values into seperate lists and then
    # convert the values into a nullable float array
    keys = []
    data = []
    nulls = [-1]
    nans = [-1]
    for i, (key, val) in enumerate(pairs):
        keys.append(key)
        if val is None:
            data.append(0.0)
            nulls.append(i)
        elif val is np.nan:
            data.append(0.0)
            nans.append(i)
        else:
            data.append(val)
    data = nullable_float_arr_maker(data, nulls, nans)
    df = pd.DataFrame({"key": keys, "data": data})

    # Convert answer into a list of (key, value) pairs
    # then convert the values into a nullable float array
    answer_keys = []
    answer_data = []
    answer_nulls = [-1]
    answer_nans = [-1]
    for i, key in enumerate(answer_dict):
        val = answer_dict[key]
        answer_keys.append(key)
        if val is None:
            answer_data.append(0.0)
            answer_nulls.append(i)
        elif val is np.nan:
            answer_data.append(0.0)
            answer_nans.append(i)
        else:
            answer_data.append(val)
    answer_data = nullable_float_arr_maker(answer_data, answer_nulls, answer_nans)
    expected_output = pd.DataFrame({"key": answer_keys, "result": answer_data})

    check_func(
        impl,
        (df,),
        sort_output=True,
        reset_index=True,
        py_output=expected_output,
        check_names=False,
    )
