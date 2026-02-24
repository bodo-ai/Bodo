import datetime
import hashlib
import os
import string
import sys

import numpy as np
import pandas as pd
import py4j
import pyarrow as pa
import pyspark
import pytest
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

import bodo
import bodo.utils.allocation_tracking
import bodosql

# TODO[BSE-5181]: remove compiler import when not needed
import bodosql.compiler  # isort:skip # noqa
from bodo.tests.conftest import (  # noqa
    iceberg_database,
    memory_leak_check,
)
from bodo.tests.iceberg_database_helpers.utils import get_spark
from bodo.tests.utils import gen_nonascii_list

# Patch to avoid PySpark's Py4j exception handler in testing.
# See:
# https://github.com/apache/spark/blob/add49b3c115f34ab8e693f7e67579292afface4c/python/pyspark/sql/session.py#L67
# https://github.com/apache/spark/blob/add49b3c115f34ab8e693f7e67579292afface4c/python/pyspark/sql/context.py#L45
# https://github.com/apache/spark/blob/add49b3c115f34ab8e693f7e67579292afface4c/python/pyspark/errors/exceptions/captured.py#L244
# Revert the change
py4j.java_gateway.get_return_value = py4j.protocol.get_return_value
# Remove the exception handler
pyspark.errors.exceptions.captured.install_exception_handler = lambda: None
# Remove the imports of individual functions. These make a copy.
pyspark.sql.context.install_exception_handler = lambda: None
pyspark.sql.session.install_exception_handler = lambda: None


# Fix Issue on Azure CI where the driver defaults to a different Python version
# See: https://stackoverflow.com/a/65010346/14810655
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Disable broadcast join as the default.
# To test broadcast join either remove or increase this.
os.environ["BODO_BCAST_JOIN_THRESHOLD"] = "0"


def pytest_addoption(parser):
    """Used with caching tests, stores the --is_cached flag into the pytestconfig"""
    parser.addoption("--is_cached", type=str, action="store", default="n")


@pytest.fixture()
def is_cached(pytestconfig):
    """Fixture used with caching tests, returns the value passed to pytest
    with --is_cached, or 'n' if not passed."""
    return pytestconfig.getoption("is_cached")


# similar to Pandas
@pytest.fixture(scope="session")
def datapath():
    """Get the path to a test data file.

    Parameters
    ----------
    path : str
        Path to the file, relative to ``bodosql/tests/data``

    Returns
    -------
    path : path including ``bodosql/tests/data``.

    Raises
    ------
    ValueError
        If the path doesn't exist.
    """
    BASE_PATH = os.path.join(os.path.dirname(__file__), "data")

    def deco(*args, check_exists=True):
        path = os.path.join(BASE_PATH, *args)
        if check_exists and not os.path.exists(path):
            msg = "Could not find file {}."
            raise ValueError(msg.format(path))
        return path

    return deco


@pytest.fixture(scope="session", autouse=True)
def enable_numba_alloc_stats():
    """Enable Numba's allocation stat collection for memory_leak_check"""
    from numba.core.runtime import _nrt_python

    _nrt_python.memsys_enable_stats()


@pytest.fixture(scope="module")
def spark_info():
    spark = get_spark()
    yield spark


@pytest.fixture
def basic_df():
    df = pd.DataFrame(
        {"A": [1, 2, 3] * 4, "B": [4, 5, 6, 7] * 3, "C": [7, 8, 9, 10, 11, 12] * 2}
    )
    return {"TABLE1": df}


@pytest.fixture
def zeros_df():
    """
    DataFrame containing zero entries. This is used
    to check issues with dividing by zero
    """
    df = pd.DataFrame({"A": np.arange(12), "B": [0, 1, -2, 1] * 3})
    return {"TABLE1": df}


@pytest.fixture(
    params=[
        pytest.param(np.int8, marks=pytest.mark.slow),
        pytest.param(
            np.uint8,
            marks=[
                pytest.mark.slow,
                pytest.mark.skipif(
                    sys.platform == "win32",
                    reason="Spark doesn't support unsigned int on Windows.",
                ),
            ],
        ),
        pytest.param(np.int16, marks=pytest.mark.slow),
        pytest.param(
            np.uint16,
            marks=[
                pytest.mark.slow,
                pytest.mark.skipif(
                    sys.platform == "win32",
                    reason="Spark doesn't support unsigned int on Windows.",
                ),
            ],
        ),
        pytest.param(np.int32, marks=pytest.mark.slow),
        pytest.param(
            np.uint32,
            marks=[
                pytest.mark.slow,
                pytest.mark.skipif(
                    sys.platform == "win32",
                    reason="Spark doesn't support unsigned int on Windows.",
                ),
            ],
        ),
        np.int64,
        pytest.param(
            np.uint64,
            marks=[
                pytest.mark.slow,
                pytest.mark.skipif(
                    sys.platform == "win32",
                    reason="Spark doesn't support unsigned int on Windows.",
                ),
            ],
        ),
        pytest.param(np.float32, marks=pytest.mark.slow),
        np.float64,
    ]
)
def bodosql_numeric_types(request):
    """
    Fixture for DataFrames with numeric BodoSQL types:
        - int8
        - uint8
        - int16
        - uint16
        - int32
        - uint32
        - int64
        - uint64
        - float32
        - float64

    For each data table, it provides a dictionary mapping table1 -> DataFrame.
    All DataFrames have the same column names so the queries can be applied to
    each table.
    """
    dtype = request.param
    int_data = {"A": [1, 2, 3] * 4, "B": [4, 5, 6] * 4, "C": [7, 8, 9] * 4}
    return {"TABLE1": pd.DataFrame(data=int_data, dtype=dtype)}


@pytest.fixture(
    params=[
        {
            "TABLE1": pd.DataFrame(
                {
                    "A": [4, 1, 2, 3] * 4,
                    "B": pd.array([1.0, 2.0, 3.0, 4.0] * 4, "Float64"),
                    "C": ["bird", "dog", "flamingo", "cat"] * 4,
                }
            ),
            "TABLE2": pd.DataFrame(
                {
                    "A": [3, 1, 2, 4] * 4,
                    "B": pd.array([1.0, 2.0, 4.0, 3.0] * 4, "Float64"),
                    "D": pd.Series(
                        [
                            pd.Timestamp(2021, 5, 19),
                            pd.Timestamp(1999, 12, 31),
                            pd.Timestamp(2020, 10, 11),
                            pd.Timestamp(2025, 1, 1),
                        ]
                        * 4,
                        dtype="datetime64[ns]",
                    ),
                }
            ),
            "TABLE3": pd.DataFrame({"Y": [1, 2, 3, 4, 5, 6] * 2}),
        },
        {
            "TABLE1": pd.DataFrame(
                {
                    "A": pd.array([1, 1, 3, 3, 5, 7] * 2, dtype="Int64"),
                    "B": pd.array([1.0, 2.0, 4.0, 3.0] * 3, dtype="Int64"),
                    "C": ["T1_1", "T1_2", "T1_3", "T1_4", "T1_5", "T1_6"] * 2,
                }
            ),
            "TABLE2": pd.DataFrame(
                {
                    "A": pd.array([2, 4, 6, 6, 1, 1] * 2, dtype="Int64"),
                    "B": pd.array([1.0, 2.0, 4.0, 3.0] * 3, dtype="Int64"),
                    "D": ["T2_1", "T2_2", "T2_3", "T2_4", "T2_5", "T2_6"] * 2,
                }
            ),
            "TABLE3": pd.DataFrame(
                {"Y": pd.array([1, 2, 3, 4, 5, 6] * 2, dtype="Int64")}
            ),
        },
        pytest.param(
            {
                "TABLE1": pd.DataFrame(
                    {
                        "A": np.array([b"abc", b"c", None, b"ccdefg"] * 3, object),
                        "B": np.array(
                            [bytes(32), b"abcde", b"ihohi04324", None] * 3, object
                        ),
                        "C": np.array(
                            [None, b"poiu", b"fewfqqqqq", b"3f3"] * 3, object
                        ),
                    }
                ),
                "TABLE2": pd.DataFrame(
                    {
                        "A": np.array([b"cew", b"abc", b"r2r", None] * 3, object),
                        "B": np.array(
                            [bytes(12), b"abcde", b"ihohi04324", None] * 3, object
                        ),
                        "D": np.array([b"r32r2", b"poiu", b"3r32", b"3f3"] * 3, object),
                    }
                ),
                "TABLE3": pd.DataFrame(
                    {"Y": [b"abc", b"c", b"cew", b"ce2r", b"r2r", None] * 2}
                ),
            },
            id="join_binary_keys",
        ),
        pytest.param(
            {
                "TABLE1": pd.DataFrame(
                    {
                        "A": pd.Series([5, None, 1, 0, None, 7] * 2, dtype="Int64"),
                        "B": pd.Series([1, 2, None, 3] * 3, dtype="Int64"),
                        "C": ["T1_1", "T1_2", "T1_3", "T1_4", "T1_5", "T1_6"] * 2,
                    }
                ),
                "TABLE2": pd.DataFrame(
                    {
                        "A": pd.Series([2, 5, 6, 6, None, 1] * 2, dtype="Int64"),
                        "B": pd.Series([None, 2, 4, 3] * 3, dtype="Int64"),
                        "D": ["T2_1", "T2_2", "T2_3", "T2_4", "T2_5", "T2_6"] * 2,
                    }
                ),
                "TABLE3": pd.DataFrame({"Y": [1, 2, 3, 4, 5, 6] * 2}),
            },
            marks=pytest.mark.slow,
        ),
        pytest.param(
            {
                "TABLE1": pd.DataFrame(
                    {
                        "A": (gen_nonascii_list(2) + ["A", "B"]) * 3,
                        "B": gen_nonascii_list(3) * 4,
                        "C": gen_nonascii_list(6)[3:] * 4,
                    }
                ),
                "TABLE2": pd.DataFrame(
                    {
                        "A": ["a", "b", "c", None, "e", "aa"] * 2,
                        "B": gen_nonascii_list(6)[3:] * 4,
                        "D": gen_nonascii_list(9)[6:] * 4,
                    }
                ),
                "TABLE3": pd.DataFrame({"Y": ["Z", "Y", "X", "W", "V", "U"] * 2}),
            },
            marks=pytest.mark.slow,
        ),
    ]
)
def join_dataframes(request):
    """
    Fixture with similar DataFrames to use for join queries.

    table1 has columns A, B, C
    table2 has columns A, B, D
    """
    return request.param


@pytest.fixture(
    params=[
        {
            "TABLE1": pd.DataFrame({"A": [1, 2, 3], "D": [4, 5, 6]}),
            "TABLE2": pd.DataFrame({"B": [1, 2, 3], "C": [4, 5, 6]}),
        },
    ]
)
def simple_join_fixture(request):
    """a very simple context with no overlapping column names
    mostly used for testing join optimizations
    """
    return request.param


@pytest.fixture(
    params=[
        pytest.param(
            {
                "TABLE1": pd.DataFrame(
                    data={
                        "E": [2**63, 2**62, 2**61] * 4,
                        "F": [2**60, 2**59, 2**58] * 4,
                    },
                    dtype=np.uint64,
                )
            },
            marks=pytest.mark.slow,
        ),
        pytest.param(
            {
                "TABLE1": pd.DataFrame(
                    data={
                        "E": [(-2) ** 61, (-2) ** 60, (-2) ** 59] * 4,
                        "F": [(-2) ** 58, (-2) ** 57, (-2) ** 56] * 4,
                    },
                    dtype=np.int64,
                )
            },
            marks=pytest.mark.slow,
        ),
        pytest.param(
            {
                "TABLE1": pd.DataFrame(
                    data={
                        "E": [2**31, 2**30, 2**29] * 4,
                        "F": [2**28, 2**27, 2**26] * 4,
                    },
                    dtype=np.uint32,
                )
            },
            marks=pytest.mark.slow,
        ),
        pytest.param(
            {
                "TABLE1": pd.DataFrame(
                    data={
                        "E": [(-2) ** 30, (-2) ** 29, (-2) ** 28] * 4,
                        "F": [(-2) ** 27, (-2) ** 26, (-2) ** 25] * 4,
                    },
                    dtype=np.int32,
                )
            },
            marks=pytest.mark.slow,
        ),
    ]
)
def bodosql_large_numeric_types(request):
    """
    Fixture used to check very large values for numeric types.
    """
    return request.param


@pytest.fixture(
    params=[
        {
            "TABLE1": pd.DataFrame(
                {
                    "A": [
                        np.datetime64("2011-01-01"),
                        np.datetime64("1971-02-02"),
                        np.datetime64("2021-03-03"),
                        np.datetime64("2004-12-07"),
                    ]
                    * 3,
                    "B": [
                        np.datetime64("2007-01-01T03:30"),
                        np.datetime64("nat"),
                        np.datetime64("2020-12-01T13:56:03.172"),
                    ]
                    * 4,
                    "C": pd.Series(
                        [
                            pd.Timestamp(2021, 11, 21),
                            pd.Timestamp(2022, 1, 12),
                            pd.Timestamp(2021, 3, 3),
                        ]
                        * 4,
                        dtype="datetime64[ns]",
                    ),
                }
            ),
        },
    ]
)
def bodosql_datetime_types(request):
    return request.param


@pytest.fixture(
    params=[
        {
            "TABLE1": pd.DataFrame(
                {
                    "A": [
                        bodo.types.Time(19, 53, 6, 15),
                        bodo.types.Time(14, 28, 57),
                        bodo.types.Time(8, 2, 5, 0, 1),
                        None,
                    ]
                    * 3,
                    "B": [
                        bodo.types.Time(5, 13, 29),
                        None,
                        bodo.types.Time(22, 7, 16),
                    ]
                    * 4,
                    "C": [
                        None,
                        bodo.types.Time(13, 37, 45),
                        bodo.types.Time(1, 47, 59, 290, 574),
                    ]
                    * 4,
                }
            ),
        },
    ]
)
def bodosql_time_types(request):
    """
    Fixture used to test bodo.types.Time type.
    """
    return request.param


@pytest.fixture(
    params=[
        {
            "TABLE1": pd.DataFrame(
                {
                    "A": pd.Series(
                        [
                            pd.NaT,
                            pd.Timestamp("2020-01-16 22:06:10.378782"),
                            pd.Timestamp("2000-01-21 02:23:16.009049"),
                            pd.Timestamp("2010-01-08 12:10:20.097528"),
                        ],
                        dtype="datetime64[ns]",
                    )
                }
            ),
            "TABLE2": pd.DataFrame(
                {
                    "B": pd.Series(
                        [
                            pd.Timestamp("2013-01-16 05:25:32.145547"),
                            pd.Timestamp("2019-01-17 01:17:56.740445"),
                            pd.NaT,
                            pd.Timestamp("2015-01-29 06:35:09.810264"),
                        ],
                        dtype="datetime64[ns]",
                    )
                }
            ),
        },
    ]
)
def bodosql_datetime_types_small(request):
    return request.param


@pytest.fixture(
    params=[
        {
            "TABLE1": pd.DataFrame(
                {
                    "A": [
                        datetime.date(2011, 4, 24),
                        datetime.date(2020, 6, 7),
                        datetime.date(2022, 1, 1),
                        None,
                    ]
                    * 3,
                    "B": [
                        datetime.date(2021, 11, 2),
                        datetime.date(2022, 11, 21),
                        None,
                    ]
                    * 4,
                    "C": [
                        datetime.date(2021, 11, 21),
                        None,
                        datetime.date(2021, 3, 3),
                    ]
                    * 4,
                }
            ),
        },
    ]
)
def bodosql_date_types(request):
    # TODO: Use this fixture more when we have a proper date type
    return request.param


@pytest.fixture(
    params=[
        {
            "TABLE1": pd.DataFrame(
                {
                    "A": pd.Series(
                        [
                            np.timedelta64(100, "h"),
                            np.timedelta64(9, "h"),
                            np.timedelta64(8, "W"),
                        ]
                        * 4,
                        dtype="timedelta64[ns]",
                    ),
                    "B": pd.Series(
                        [
                            np.timedelta64("nat"),
                            np.timedelta64(6, "h"),
                            np.timedelta64(5, "m"),
                        ]
                        * 4,
                        dtype="timedelta64[ns]",
                    ),
                    "C": pd.Series(
                        [
                            np.timedelta64(4, "s"),
                            np.timedelta64(3, "ms"),
                            np.timedelta64(2000000, "us"),
                        ]
                        * 4,
                        dtype="timedelta64[ns]",
                    ),
                }
            )
        },
    ]
)
def bodosql_interval_types(request):
    return request.param


@pytest.fixture(
    params=[
        {
            "TABLE1": pd.DataFrame(
                {
                    "A": [True, False, False, None] * 3,
                    "B": [False, None, True, False] * 3,
                    "C": [False, False, False, False] * 3,
                }
            )
        }
    ]
)
def bodosql_boolean_types(request):
    return request.param


@pytest.fixture(
    params=[
        "2011-01-01",
        pytest.param("1971-02-02", marks=pytest.mark.slow),
        "2021-03-03",
        pytest.param("2004-12-07", marks=pytest.mark.slow),
        "2007-01-01 03:30:00",
    ]
)
def timestamp_literal_strings(request):
    """
    Fixture containing timestamp literal strings for use in testing
    """
    return request.param


@pytest.fixture(
    params=[
        {
            "TABLE1": pd.DataFrame(
                {
                    "A": ["hElLo", "GoOdByE", "SpEaK", "WeIrD"] * 3,
                    "B": ["hello", "world", "how"] * 4,
                    "C": ["ARE", "YOU", "TODAY"] * 4,
                }
            )
        },
        pytest.param(
            {
                "TABLE1": pd.DataFrame(
                    {
                        "A": ["HELLO", "HeLlLllo", "heyO", "HI"] * 3,
                        "B": ["hi", "HaPpY", "how"] * 4,
                        "C": ["Ello", "O", "HowEver"] * 4,
                    }
                )
            },
            marks=pytest.mark.slow,
        ),
        pytest.param(
            {
                "TABLE1": pd.DataFrame(
                    {
                        "A": (gen_nonascii_list(2) + ["A", "B"]) * 3,
                        "B": gen_nonascii_list(3) * 4,
                        "C": gen_nonascii_list(3) * 4,
                    }
                )
            },
            marks=pytest.mark.slow,
        ),
    ]
)
def bodosql_string_types(request):
    return request.param


@pytest.fixture(
    params=[
        {
            "TABLE1": pd.DataFrame(
                {
                    "A": ["hElLo GoOdByE", "GoOdByE SpEaK", "SpEaK", "hElLo WeIrD"] * 3,
                    "B": ["hello world how", "how how how how world", "how"] * 4,
                    "C": ["ARE YOU TODAY", "YOU", "TODAY"] * 4,
                }
            )
        },
        pytest.param(
            {
                "TABLE1": pd.DataFrame(
                    {
                        "A": ["John Ddoe", "Joe Doe", "John_down", "Joe down"] * 3,
                        "B": ["Tome Doe", "Tim down"] * 6,
                        "C": ["Joe Doe", "John_down", "Tome Doe"] * 4,
                    }
                )
            },
            marks=pytest.mark.slow,
        ),
    ]
)
def bodosql_multiple_string_types(request):
    return request.param


@pytest.fixture(
    params=[
        {
            "TABLE1": pd.DataFrame(
                {
                    "A": ["0", "1", "24", "27"] * 3,
                    "B": ["43", "121", "43"] * 4,
                    "C": ["44", "14", "42"] * 4,
                }
            )
        },
    ]
)
def bodosql_integers_string_types(request):
    """Fixture for strings where all values can be cast to integers"""
    return request.param


@pytest.fixture(
    params=[
        {
            "TABLE1": pd.DataFrame(
                {
                    "A": ["2011-01-01", "1971-02-02", "2021-03-03", "2004-12-07"] * 3,
                    "B": ["2021-11-09", "2019-08-25", "2017-05-04"] * 4,
                    "C": ["2010-09-14", "2021-01-12", "2022-07-13"] * 4,
                }
            )
        },
    ]
)
def bodosql_date_string_types(request):
    """Fixture for strings where all values can be cast to dates"""
    return request.param


@pytest.fixture(
    params=[
        pytest.param("Int8", marks=pytest.mark.slow),
        pytest.param("UInt8", marks=pytest.mark.slow),
        pytest.param("Int16", marks=pytest.mark.slow),
        pytest.param("UInt16", marks=pytest.mark.slow),
        pytest.param("Int32", marks=pytest.mark.slow),
        pytest.param("UInt32", marks=pytest.mark.slow),
        "Int64",
        pytest.param("UInt64", marks=pytest.mark.slow),
    ]
)
def bodosql_nullable_numeric_types(request):
    """
    Fixture for DataFrames nullable numeric BodoSQL types:
        - Int8
        - UInt8
        - Int16
        - UInt16
        - Int32
        - UInt32
        - Int64
        - UInt64

    For each data table, it provides a dictionary mapping table1 -> DataFrame.
    All DataFrames have the same column names so the queries can be applied to
    each table.
    """
    dtype = request.param
    int_data = {
        "A": [1, 2, 3, None] * 3,
        "B": [4, None, 5, 6] * 3,
        "C": [7, 8, None, 9] * 3,
    }
    return {"TABLE1": pd.DataFrame(data=int_data, dtype=dtype)}


@pytest.fixture(
    params=[
        pytest.param(
            pd.DataFrame(
                {
                    "A": pd.Series([1, 2, None, 10000, None] * 3, dtype="Int64"),
                    "B": pd.Series([-43241, None, None, None, 523] * 3, dtype="Int64"),
                    "C": pd.Series([None, None, None, -234325, 0] * 3, dtype="Int64"),
                }
            ),
            id="Integer",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": pd.Series(
                        [1.312, 2.1, np.nan, 10000.0, np.nan] * 3, dtype="Float64"
                    ),
                    "B": pd.Series(
                        [-432.41, np.nan, np.nan, np.nan, 52.3] * 3, dtype="Float64"
                    ),
                    "C": pd.Series(
                        [np.nan, np.nan, np.nan, -234325.0, 0.0] * 3, dtype="Float64"
                    ),
                }
            ),
            id="float",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": pd.Series([True, True, None, False, None] * 3),
                    "B": pd.Series([True, None, None, None, True] * 3),
                    "C": pd.Series([None, None, None, False, False] * 3),
                }
            ),
            id="boolean",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": pd.Series(
                        ["bvrewg", "word", None, "32r9hẞ{} ß#rk230-k320-rk23", None] * 3
                    ),
                    "B": pd.Series(["V", None, None, None, "38442bhbedwẞ ß"] * 3),
                    "C": pd.Series([None, None, None, "erwrewẞ ß", ""] * 3),
                }
            ),
            id="string",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": pd.Series(
                        [b"bvrewg", b"word", None, b"32r9h {}#rk230-k320-rk23", None]
                        * 3
                    ),
                    "B": pd.Series([b"V", None, None, None, b"38442bhbedw "] * 3),
                    "C": pd.Series([None, None, None, b"erwrew ", b""] * 3),
                }
            ),
            id="binary",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": pd.Series(
                        [
                            datetime.date(2022, 1, 1),
                            datetime.date(2022, 3, 15),
                            None,
                            datetime.date(2019, 3, 15),
                            None,
                        ]
                        * 3
                    ),
                    "B": pd.Series(
                        [
                            datetime.date(2010, 1, 11),
                            None,
                            None,
                            None,
                            datetime.date(2018, 12, 25),
                        ]
                        * 3
                    ),
                    "C": pd.Series(
                        [
                            None,
                            None,
                            None,
                            datetime.date(2028, 2, 25),
                            datetime.date(2017, 11, 21),
                        ]
                        * 3
                    ),
                }
            ),
            id="date",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": pd.Series(
                        [
                            pd.Timestamp(2021, 11, 21),
                            pd.Timestamp(2021, 3, 3),
                            np.datetime64("NaT"),
                            np.datetime64("2007-01-01T03:30"),
                            np.datetime64("NaT"),
                        ]
                        * 3,
                        dtype="datetime64[ns]",
                    ),
                    "B": pd.Series(
                        [
                            pd.Timestamp(2022, 1, 12),
                            np.datetime64("NaT"),
                            np.datetime64("NaT"),
                            np.datetime64("NaT"),
                            np.datetime64("2020-11-11T13:21:03.172"),
                        ]
                        * 3,
                        dtype="datetime64[ns]",
                    ),
                    "C": pd.Series(
                        [
                            np.datetime64("NaT"),
                            np.datetime64("NaT"),
                            np.datetime64("NaT"),
                            np.datetime64("2020-12-01T13:56:03.172"),
                            np.datetime64("2020-02-11"),
                        ]
                        * 3,
                        dtype="datetime64[ns]",
                    ),
                }
            ),
            id="timestamp",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": pd.Series(
                        [
                            np.timedelta64(2000000, "us"),
                            np.timedelta64(3, "ms"),
                            np.timedelta64("NaT"),
                            np.timedelta64(4, "s"),
                            np.timedelta64("NaT"),
                        ]
                        * 3,
                        dtype="timedelta64[ns]",
                    ),
                    "B": pd.Series(
                        [
                            np.timedelta64(100, "h"),
                            np.timedelta64("NaT"),
                            np.timedelta64("NaT"),
                            np.timedelta64("NaT"),
                            np.timedelta64(5, "m"),
                        ]
                        * 3,
                        dtype="timedelta64[ns]",
                    ),
                    "C": pd.Series(
                        [
                            np.timedelta64("NaT"),
                            np.timedelta64("NaT"),
                            np.timedelta64("NaT"),
                            np.timedelta64(8, "W"),
                            np.timedelta64(6, "h"),
                        ]
                        * 3,
                        dtype="timedelta64[ns]",
                    ),
                }
            ),
            id="timedelta",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": pd.Series(
                        [
                            bodo.types.Time(23, 55, 55, precision=0),
                            bodo.types.Time(0, 0, 0, precision=0),
                            None,
                            bodo.types.Time(23, 59, 59, precision=0),
                            None,
                        ]
                        * 3
                    ),
                    "B": pd.Series(
                        [
                            bodo.types.Time(9, 9, 9, precision=0),
                            None,
                            None,
                            None,
                            bodo.types.Time(14, 11, 4, precision=0),
                        ]
                        * 3
                    ),
                    "C": pd.Series(
                        [
                            None,
                            None,
                            None,
                            bodo.types.Time(12, 55, 56, precision=0),
                            bodo.types.Time(1, 34, 51, precision=0),
                        ]
                        * 3
                    ),
                }
            ),
            id="time",
            marks=pytest.mark.skip(
                "[BE-3649] Time type needs to forward precision to SQL."
            ),
        ),
    ]
)
def major_types_nullable(request):
    """
    Fixture that contains an entry for every major type class (e.g. integer
    but not each bitwidth) with null values. These columns are always named
    A, B, and C. In addition every DataFrame has an identical boolean column
    that can be used as the condition in case statements named COND_COL. They also
    have a distinct column for sorting called ORDERBY_COl.
    """
    df = request.param.copy()
    assert not (len(df) % 3), (
        "Appending a boolean column requires the DataFrame to be divisible by 3"
    )
    df["COND_COL"] = pd.Series([True, False, None] * (len(df) // 3))
    df["ORDERBY_COl"] = np.arange(len(df))
    return {"TABLE1": df}


@pytest.fixture(
    params=[
        pytest.param(
            pd.DataFrame(
                {
                    "A": np.array([b"abc", b"c", None, b"ccdefg"] * 3, object),
                    "B": np.array(
                        [bytes(32), b"abcde", b"ihohi04324", None] * 3, object
                    ),
                    "C": np.array([None, b"poiu", b"fewfqqqqq", b"3f3"] * 3, object),
                }
            )
        ),
    ]
)
def bodosql_binary_types(request):
    """
    Fixture for dataframe of binary array types in BodoSQL.
    Binary arrays are nullable.
    """
    return {"TABLE1": request.param}


@pytest.fixture(
    params=[
        "=",
        "<>",
        pytest.param("!=", marks=pytest.mark.slow),
        pytest.param("<=", marks=pytest.mark.slow),
        pytest.param("<", marks=pytest.mark.slow),
        pytest.param(">=", marks=pytest.mark.slow),
        pytest.param(">", marks=pytest.mark.slow),
        pytest.param("<=>", marks=pytest.mark.slow),
    ]
)
def comparison_ops(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param(
            "BIT_AND",
            marks=pytest.mark.skip(
                "Bitwise aggregations not currently supported, see [BE-919]"
            ),
        ),
        pytest.param(
            "BIT_OR",
            marks=pytest.mark.skip(
                "Bitwise aggregations not currently supported, see [BE-919]"
            ),
        ),
        pytest.param(
            "BIT_XOR",
            marks=pytest.mark.skip(
                "Bitwise aggregations not currently supported, see [BE-919]"
            ),
        ),
        "AVG",
        "COUNT",
        "MAX",
        "MIN",
        "STDDEV",
        pytest.param("STDDEV_SAMP", marks=pytest.mark.slow),
        "SUM",
        "VARIANCE",
        pytest.param("VAR_SAMP", marks=pytest.mark.slow),
        pytest.param("VARIANCE_SAMP", marks=pytest.mark.slow),
        "STDDEV_POP",
        "VAR_POP",
        pytest.param("VARIANCE_POP", marks=pytest.mark.slow),
    ]
)
def numeric_agg_builtin_funcs(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param("TINYINT", marks=pytest.mark.slow),
        pytest.param("SMALLINT", marks=pytest.mark.slow),
        "INTEGER",
        pytest.param("BIGINT", marks=pytest.mark.slow),
        "FLOAT",
        pytest.param("DOUBLE", marks=pytest.mark.slow),
        pytest.param("DECIMAL", marks=pytest.mark.slow),
    ]
)
def sql_numeric_typestrings(request):
    """
    Collection of numeric sql types
    """
    return request.param


@pytest.fixture(params=["-", "+", "*", "/"])
def arith_ops(request):
    """fixture that returns arithmetic operators"""
    return request.param


@pytest.fixture(params=["-", "+"])
def datetime_arith_ops(request):
    """fixture that returns datetime valid arithmetic operators"""
    return request.param


@pytest.fixture(params=["DATE", "TIMESTAMP"])
def sql_datetime_typestrings(request):
    """
    Collection of numeric sql types
    """
    return request.param


@pytest.fixture(
    params=[
        0,
        pytest.param(1, marks=pytest.mark.slow),
        pytest.param(-1, marks=pytest.mark.slow),
        pytest.param(0.0, marks=pytest.mark.slow),
        pytest.param(1.0, marks=pytest.mark.slow),
        -1.0,
        pytest.param(2, marks=pytest.mark.slow),
        0.001,
        pytest.param(4, marks=pytest.mark.slow),
        -7,
        pytest.param(11, marks=pytest.mark.slow),
    ]
)
def numeric_values(request):
    """
    Collection of numeric values used for testing, all the integer values should fit within a byte
    """
    return request.param


@pytest.fixture
def local_tz():
    """Returns the local timezone as a string"""

    # BodoSQL uses UTC without the Snowflake catalog and doesn't have an API to
    # set it locally.
    return "UTC"


@pytest.fixture
def tz_aware_df():
    # Transition to Daylight Savings
    # "1D2h37min48s" --> 1 day, 2 hours, 37 minutes, 48 seconds
    to_dst_series = pd.date_range(
        start="11/3/2021", freq="1D2h37min48s", periods=30, tz="US/Pacific", unit="ns"
    ).to_series()

    # Transition back from Daylight Savings
    from_dst_series = pd.date_range(
        start="03/1/2022", freq="0D12h30min1s", periods=60, tz="US/Pacific", unit="ns"
    ).to_series()

    # February is weird with leap years
    feb_leap_year_series = pd.date_range(
        start="02/20/2020", freq="1D0h30min0s", periods=20, tz="US/Pacific", unit="ns"
    ).to_series()

    second_quarter_series = pd.date_range(
        start="05/01/2015", freq="2D0h1min59s", periods=20, tz="US/Pacific", unit="ns"
    ).to_series()

    third_quarter_series = pd.date_range(
        start="08/17/2000", freq="10D1h1min10s", periods=20, tz="US/Pacific", unit="ns"
    ).to_series()

    df = pd.DataFrame(
        {
            "A": pd.concat(
                [
                    to_dst_series,
                    from_dst_series,
                    feb_leap_year_series,
                    second_quarter_series,
                    third_quarter_series,
                ]
            )
        }
    )

    return {"TABLE1": df}


@pytest.fixture
def trim_df():
    df = pd.DataFrame(
        {
            "A": pd.Series(["   ABC   ", "A  BC  ", "  AB  C", "        ", ""] * 4),
            "B": pd.Series(
                ["+**ABC+**", "***+ABC+***", None, "**++*ABC++", "+++***+++"] * 4,
            ),
            "C": pd.Series(
                ["asdafzcvdf", "dsasdaadsd", None, "xcvxcbxasd", "dakjhkjhsd"] * 4,
            ),
        }
    )
    return {"TABLE1": df}


@pytest.fixture(scope="module")
def tpch_data(datapath):
    """
    Fixture with TPCH data for BodoSQL Contexts.
    """
    return tpch_data_helper(datapath)


def tpch_data_helper(datapath):
    (
        customer_df,
        orders_df,
        lineitem_df,
        nation_df,
        region_df,
        supplier_df,
        part_df,
        partsupp_df,
    ) = load_tpch_data(datapath("tpch-test-data/parquet"))
    dataframe_dict = {
        "CUSTOMER": customer_df,
        "ORDERS": orders_df,
        "LINEITEM": lineitem_df,
        "NATION": nation_df,
        "REGION": region_df,
        "SUPPLIER": supplier_df,
        "PART": part_df,
        "PARTSUPP": partsupp_df,
    }
    return dataframe_dict


@bodo.jit(returns_maybe_distributed=False, cache=True)
def load_tpch_data(dir_name):
    """Load the necessary TPCH DataFrames given a root directory.
    We use bodo.jit so we can read easily from a directory.

    If rows is not None, only fetches that many rows from each table"""
    customer_df = pd.read_parquet(dir_name + "/customer.pq/", dtype_backend="pyarrow")
    orders_df = pd.read_parquet(dir_name + "/orders.pq/", dtype_backend="pyarrow")
    lineitem_df = pd.read_parquet(dir_name + "/lineitem.pq/", dtype_backend="pyarrow")
    nation_df = pd.read_parquet(dir_name + "/nation.pq/", dtype_backend="pyarrow")
    region_df = pd.read_parquet(dir_name + "/region.pq/", dtype_backend="pyarrow")
    supplier_df = pd.read_parquet(dir_name + "/supplier.pq/", dtype_backend="pyarrow")
    part_df = pd.read_parquet(dir_name + "/part.pq/", dtype_backend="pyarrow")
    partsupp_df = pd.read_parquet(dir_name + "/partsupp.pq/", dtype_backend="pyarrow")
    return (
        customer_df,
        orders_df,
        lineitem_df,
        nation_df,
        region_df,
        supplier_df,
        part_df,
        partsupp_df,
    )


@pytest.fixture(scope="module")
def tpcxbb_data(datapath):
    """
    Fixture with TPCxBB data for BodoSQL Contexts.
    """
    # TODO: Update when we know how to generate the full
    # TPCxBB dataset.
    (
        store_sales_df,
        item_df,
        customer_df,
        customer_address_df,
        customer_demographics_df,
        date_dim_df,
        product_reviews_df,
        store_df,
        web_clickstreams_df,
        web_sales_df,
        household_demographics_df,
        inventory_df,
        item_marketprices_df,
        promotion_df,
        store_returns_df,
        time_dim_df,
        warehouse_df,
        web_page_df,
        web_returns_df,
    ) = load_tpcxbb_data(datapath("tpcxbb-test-data"))
    # Some end dates are all null values, so its unclear its type.
    # Convert to string.
    item_df["I_REC_END_DATE"] = item_df["I_REC_END_DATE"].astype(pd.StringDtype())
    store_df["S_REC_END_DATE"] = store_df["S_REC_END_DATE"].astype(pd.StringDtype())
    web_page_df["WP_REC_END_DATE"] = web_page_df["WP_REC_END_DATE"].astype(
        pd.StringDtype()
    )
    time_dim_df["T_MEAL_TIME"] = time_dim_df["T_MEAL_TIME"].astype(pd.StringDtype())
    dataframe_dict = {
        "STORE_SALES": store_sales_df,
        "ITEM": item_df,
        "CUSTOMER": customer_df,
        "CUSTOMER_ADDRESS": customer_address_df,
        "CUSTOMER_DEMOGRAPHICS": customer_demographics_df,
        "DATE_DIM": date_dim_df,
        "PRODUCT_REVIEWS": product_reviews_df,
        "STORE": store_df,
        "WEB_CLICKSTREAMS": web_clickstreams_df,
        "WEB_SALES": web_sales_df,
        "HOUSEHOLD_DEMOGRAPHICS": household_demographics_df,
        "INVENTORY": inventory_df,
        "ITEM_MARKETPRICES": item_marketprices_df,
        "PROMOTION": promotion_df,
        "STORE_RETURNS": store_returns_df,
        "TIME_DIM": time_dim_df,
        "WAREHOUSE": warehouse_df,
        "WEB_PAGE": web_page_df,
        "WEB_RETURNS": web_returns_df,
    }
    pyspark_schemas = {
        "store_sales": StructType(
            [
                StructField("ss_sold_date_sk", IntegerType(), True),
                StructField("ss_sold_time_sk", IntegerType(), True),
                StructField("ss_item_sk", IntegerType(), True),
                StructField("ss_customer_sk", IntegerType(), True),
                StructField("ss_cdemo_sk", IntegerType(), True),
                StructField("ss_hdemo_sk", IntegerType(), True),
                StructField("ss_addr_sk", IntegerType(), True),
                StructField("ss_store_sk", IntegerType(), True),
                StructField("ss_promo_sk", IntegerType(), True),
                StructField("ss_ticket_number", IntegerType(), True),
                StructField("ss_quantity", IntegerType(), True),
                StructField("ss_wholesale_cost", DoubleType(), False),
                StructField("ss_list_price", DoubleType(), False),
                StructField("ss_sales_price", DoubleType(), False),
                StructField("ss_ext_discount_amt", DoubleType(), False),
                StructField("ss_ext_sales_price", DoubleType(), False),
                StructField("ss_ext_wholesale_cost", DoubleType(), False),
                StructField("ss_ext_list_price", DoubleType(), False),
                StructField("ss_ext_tax", DoubleType(), False),
                StructField("ss_coupon_amt", DoubleType(), False),
                StructField("ss_net_paid", DoubleType(), False),
                StructField("ss_net_paid_inc_tax", DoubleType(), False),
                StructField("ss_net_profit", DoubleType(), False),
            ]
        ),
        "item": StructType(
            [
                StructField("i_item_sk", IntegerType(), True),
                StructField("i_item_id", StringType(), True),
                StructField("i_rec_start_date", StringType(), True),
                StructField("i_rec_end_date", StringType(), True),
                StructField("i_item_desc", StringType(), True),
                StructField("i_current_price", DoubleType(), False),
                StructField("i_wholesale_cost", DoubleType(), False),
                StructField("i_brand_id", IntegerType(), True),
                StructField("i_brand", StringType(), True),
                StructField("i_class_id", IntegerType(), True),
                StructField("i_class", StringType(), True),
                StructField("i_category_id", IntegerType(), True),
                StructField("i_category", StringType(), True),
                StructField("i_manufact_id", IntegerType(), True),
                StructField("i_manufact", StringType(), True),
                StructField("i_size", StringType(), True),
                StructField("i_formulation", StringType(), True),
                StructField("i_color", StringType(), True),
                StructField("i_units", StringType(), True),
                StructField("i_container", StringType(), True),
                StructField("i_manager_id", IntegerType(), True),
                StructField("i_product_name", StringType(), True),
            ]
        ),
        # Taken from https://github.com/rapidsai/gpu-bdb/tree/main/gpu_bdb/spark_table_schemas
        "customer": StructType(
            [
                StructField("c_customer_sk", IntegerType(), True),
                StructField("c_customer_id", StringType(), True),
                StructField("c_current_cdemo_sk", IntegerType(), True),
                StructField("c_current_hdemo_sk", IntegerType(), True),
                StructField("c_current_addr_sk", IntegerType(), True),
                StructField("c_first_shipto_date_sk", IntegerType(), True),
                StructField("c_first_sales_date_sk", IntegerType(), True),
                StructField("c_salutation", StringType(), True),
                StructField("c_first_name", StringType(), True),
                StructField("c_last_name", StringType(), True),
                StructField("c_preferred_cust_flag", StringType(), True),
                StructField("c_birth_day", IntegerType(), True),
                StructField("c_birth_month", IntegerType(), True),
                StructField("c_birth_year", IntegerType(), True),
                StructField("c_birth_country", StringType(), True),
                StructField("c_login", StringType(), True),
                StructField("c_email_address", StringType(), True),
                StructField("c_last_review_date", StringType(), True),
            ]
        ),
        "customer_address": StructType(
            [
                StructField("ca_address_sk", IntegerType(), True),
                StructField("ca_address_id", StringType(), True),
                StructField("ca_street_number", StringType(), True),
                StructField("ca_street_name", StringType(), True),
                StructField("ca_street_type", StringType(), True),
                StructField("ca_suite_number", StringType(), True),
                StructField("ca_city", StringType(), True),
                StructField("ca_county", StringType(), True),
                StructField("ca_state", StringType(), True),
                StructField("ca_zip", StringType(), True),
                StructField("ca_country", StringType(), True),
                StructField("ca_gmt_offset", DoubleType(), False),
                StructField("ca_location_type", StringType(), True),
            ]
        ),
        "customer_demographics": StructType(
            [
                StructField("cd_demo_sk", IntegerType(), True),
                StructField("cd_gender", StringType(), True),
                StructField("cd_marital_status", StringType(), True),
                StructField("cd_education_status", StringType(), True),
                StructField("cd_purchase_estimate", IntegerType(), True),
                StructField("cd_credit_rating", StringType(), True),
                StructField("cd_dep_count", IntegerType(), True),
                StructField("cd_dep_employed_count", IntegerType(), True),
                StructField("cd_dep_college_count", IntegerType(), True),
            ]
        ),
        "date_dim": StructType(
            [
                StructField("d_date_sk", IntegerType(), True),
                StructField("d_date_id", StringType(), True),
                StructField("d_date", StringType(), True),
                StructField("d_month_seq", IntegerType(), True),
                StructField("d_week_seq", IntegerType(), True),
                StructField("d_quarter_seq", IntegerType(), True),
                StructField("d_year", IntegerType(), True),
                StructField("d_dow", IntegerType(), True),
                StructField("d_moy", IntegerType(), True),
                StructField("d_dom", IntegerType(), True),
                StructField("d_qoy", IntegerType(), True),
                StructField("d_fy_year", IntegerType(), True),
                StructField("d_fy_quarter_seq", IntegerType(), True),
                StructField("d_fy_week_seq", IntegerType(), True),
                StructField("d_day_name", StringType(), True),
                StructField("d_quarter_name", StringType(), True),
                StructField("d_holiday", StringType(), True),
                StructField("d_weekend", StringType(), True),
                StructField("d_following_holiday", StringType(), True),
                StructField("d_first_dom", IntegerType(), True),
                StructField("d_last_dom", IntegerType(), True),
                StructField("d_same_day_ly", IntegerType(), True),
                StructField("d_same_day_lq", IntegerType(), True),
                StructField("d_current_day", StringType(), True),
                StructField("d_current_week", StringType(), True),
                StructField("d_current_month", StringType(), True),
                StructField("d_current_quarter", StringType(), True),
                StructField("d_current_year", StringType(), True),
            ]
        ),
        "product_reviews": StructType(
            [
                StructField("pr_review_sk", IntegerType(), True),
                StructField("pr_review_date", StringType(), True),
                StructField("pr_review_time", StringType(), True),
                StructField("pr_review_rating", IntegerType(), True),
                StructField("pr_item_sk", IntegerType(), True),
                StructField("pr_user_sk", IntegerType(), True),
                StructField("pr_order_sk", IntegerType(), True),
                StructField("pr_review_content", StringType(), True),
            ]
        ),
        "store": StructType(
            [
                StructField("s_store_sk", IntegerType(), True),
                StructField("s_store_id", StringType(), True),
                StructField("s_rec_start_date", StringType(), True),
                StructField("s_rec_end_date", StringType(), True),
                StructField("s_closed_date_sk", IntegerType(), True),
                StructField("s_store_name", StringType(), True),
                StructField("s_number_employees", IntegerType(), True),
                StructField("s_floor_space", IntegerType(), True),
                StructField("s_hours", StringType(), True),
                StructField("s_manager", StringType(), True),
                StructField("s_market_id", IntegerType(), True),
                StructField("s_geography_class", StringType(), True),
                StructField("s_market_desc", StringType(), True),
                StructField("s_market_manager", StringType(), True),
                StructField("s_division_id", IntegerType(), True),
                StructField("s_division_name", StringType(), True),
                StructField("s_company_id", IntegerType(), True),
                StructField("s_company_name", StringType(), True),
                StructField("s_street_number", StringType(), True),
                StructField("s_street_name", StringType(), True),
                StructField("s_street_type", StringType(), True),
                StructField("s_suite_number", StringType(), True),
                StructField("s_city", StringType(), True),
                StructField("s_county", StringType(), True),
                StructField("s_state", StringType(), True),
                StructField("s_zip", StringType(), True),
                StructField("s_country", StringType(), True),
                StructField("s_gmt_offset", DoubleType(), False),
                StructField("s_tax_precentage", DoubleType(), False),
            ]
        ),
        "web_clickstreams": StructType(
            [
                StructField("wcs_click_date_sk", IntegerType(), True),
                StructField("wcs_click_time_sk", IntegerType(), True),
                StructField("wcs_sales_sk", IntegerType(), True),
                StructField("wcs_item_sk", IntegerType(), True),
                StructField("wcs_web_page_sk", IntegerType(), True),
                StructField("wcs_user_sk", IntegerType(), True),
            ]
        ),
        "web_sales": StructType(
            [
                StructField("ws_sold_date_sk", IntegerType(), True),
                StructField("ws_sold_time_sk", IntegerType(), True),
                StructField("ws_ship_date_sk", IntegerType(), True),
                StructField("ws_item_sk", IntegerType(), True),
                StructField("ws_bill_customer_sk", IntegerType(), True),
                StructField("ws_bill_cdemo_sk", IntegerType(), True),
                StructField("ws_bill_hdemo_sk", IntegerType(), True),
                StructField("ws_bill_addr_sk", IntegerType(), True),
                StructField("ws_ship_customer_sk", IntegerType(), True),
                StructField("ws_ship_cdemo_sk", IntegerType(), True),
                StructField("ws_ship_hdemo_sk", IntegerType(), True),
                StructField("ws_ship_addr_sk", IntegerType(), True),
                StructField("ws_web_page_sk", IntegerType(), True),
                StructField("ws_web_site_sk", IntegerType(), True),
                StructField("ws_ship_mode_sk", IntegerType(), True),
                StructField("ws_warehouse_sk", IntegerType(), True),
                StructField("ws_promo_sk", IntegerType(), True),
                StructField("ws_order_number", IntegerType(), True),
                StructField("ws_quantity", IntegerType(), True),
                StructField("ws_wholesale_cost", DoubleType(), False),
                StructField("ws_list_price", DoubleType(), False),
                StructField("ws_sales_price", DoubleType(), False),
                StructField("ws_ext_discount_amt", DoubleType(), False),
                StructField("ws_ext_sales_price", DoubleType(), False),
                StructField("ws_ext_wholesale_cost", DoubleType(), False),
                StructField("ws_ext_list_price", DoubleType(), False),
                StructField("ws_ext_tax", DoubleType(), False),
                StructField("ws_coupon_amt", DoubleType(), False),
                StructField("ws_ext_ship_cost", DoubleType(), False),
                StructField("ws_net_paid", DoubleType(), False),
                StructField("ws_net_paid_inc_tax", DoubleType(), False),
                StructField("ws_net_paid_inc_ship", DoubleType(), False),
                StructField("ws_net_paid_inc_ship_tax", DoubleType(), False),
                StructField("ws_net_profit", DoubleType(), False),
            ]
        ),
        "household_demographics": StructType(
            [
                StructField("hd_demo_sk", IntegerType(), True),
                StructField("hd_income_band_sk", IntegerType(), True),
                StructField("hd_buy_potential", StringType(), True),
                StructField("hd_dep_count", IntegerType(), True),
                StructField("hd_vehicle_count", IntegerType(), True),
            ]
        ),
        "inventory": StructType(
            [
                StructField("inv_date_sk", IntegerType(), True),
                StructField("inv_item_sk", IntegerType(), True),
                StructField("inv_warehouse_sk", IntegerType(), True),
                StructField("inv_quantity_on_hand", IntegerType(), True),
            ]
        ),
        "item_marketprices": StructType(
            [
                StructField("imp_sk", IntegerType(), True),
                StructField("imp_item_sk", IntegerType(), True),
                StructField("imp_competitor", StringType(), True),
                StructField("imp_competitor_price", DoubleType(), True),
                StructField("imp_start_date", IntegerType(), True),
                StructField("imp_end_date", IntegerType(), True),
            ]
        ),
        "promotion": StructType(
            [
                StructField("p_promo_sk", IntegerType(), True),
                StructField("p_promo_id", StringType(), True),
                StructField("p_start_date_sk", IntegerType(), True),
                StructField("p_end_date_sk", IntegerType(), True),
                StructField("p_item_sk", IntegerType(), True),
                StructField("p_cost", DoubleType(), False),
                StructField("p_response_target", StringType(), True),
                StructField("p_promo_name", StringType(), True),
                StructField("p_channel_dmail", StringType(), True),
                StructField("p_channel_email", StringType(), True),
                StructField("p_channel_catalog", StringType(), True),
                StructField("p_channel_tv", StringType(), True),
                StructField("p_channel_radio", StringType(), True),
                StructField("p_channel_press", StringType(), True),
                StructField("p_channel_event", StringType(), True),
                StructField("p_channel_demo", StringType(), True),
                StructField("p_channel_details", StringType(), True),
                StructField("p_purpose", StringType(), True),
                StructField("p_discount_active", StringType(), True),
            ]
        ),
        "store_returns": StructType(
            [
                StructField("sr_returned_date_sk", IntegerType(), True),
                StructField("sr_return_time_sk", IntegerType(), True),
                StructField("sr_item_sk", IntegerType(), True),
                StructField("sr_customer_sk", IntegerType(), True),
                StructField("sr_cdemo_sk", IntegerType(), True),
                StructField("sr_hdemo_sk", IntegerType(), True),
                StructField("sr_addr_sk", IntegerType(), True),
                StructField("sr_store_sk", IntegerType(), True),
                StructField("sr_reason_sk", IntegerType(), True),
                StructField("sr_ticket_number", IntegerType(), True),
                StructField("sr_return_quantity", IntegerType(), True),
                StructField("sr_return_amt", DoubleType(), False),
                StructField("sr_return_tax", DoubleType(), False),
                StructField("sr_return_amt_inc_tax", DoubleType(), False),
                StructField("sr_fee", DoubleType(), False),
                StructField("sr_return_ship_cost", DoubleType(), False),
                StructField("sr_refunded_cash", DoubleType(), False),
                StructField("sr_reversed_charge", DoubleType(), False),
                StructField("sr_store_credit", DoubleType(), False),
                StructField("sr_net_loss", DoubleType(), False),
            ]
        ),
        "time_dim": StructType(
            [
                StructField("t_time_sk", IntegerType(), True),
                StructField("t_time_id", StringType(), True),
                StructField("t_time", IntegerType(), True),
                StructField("t_hour", IntegerType(), True),
                StructField("t_minute", IntegerType(), True),
                StructField("t_second", IntegerType(), True),
                StructField("t_am_pm", StringType(), True),
                StructField("t_shift", StringType(), True),
                StructField("t_sub_shift", StringType(), True),
                StructField("t_meal_time", StringType(), True),
            ]
        ),
        "warehouse": StructType(
            [
                StructField("w_warehouse_sk", IntegerType(), True),
                StructField("w_warehouse_id", StringType(), True),
                StructField("w_warehouse_name", StringType(), True),
                StructField("w_warehouse_sq_ft", IntegerType(), True),
                StructField("w_street_number", StringType(), True),
                StructField("w_street_name", StringType(), True),
                StructField("w_street_type", StringType(), True),
                StructField("w_suite_number", StringType(), True),
                StructField("w_city", StringType(), True),
                StructField("w_county", StringType(), True),
                StructField("w_state", StringType(), True),
                StructField("w_zip", StringType(), True),
                StructField("w_country", StringType(), True),
                StructField("w_gmt_offset", DoubleType(), False),
            ]
        ),
        "web_page": StructType(
            [
                StructField("wp_web_page_sk", IntegerType(), True),
                StructField("wp_web_page_id", StringType(), True),
                StructField("wp_rec_start_date", StringType(), True),
                StructField("wp_rec_end_date", StringType(), True),
                StructField("wp_creation_date_sk", IntegerType(), True),
                StructField("wp_access_date_sk", IntegerType(), True),
                StructField("wp_autogen_flag", StringType(), True),
                StructField("wp_customer_sk", IntegerType(), True),
                StructField("wp_url", StringType(), True),
                StructField("wp_type", StringType(), True),
                StructField("wp_char_count", IntegerType(), True),
                StructField("wp_link_count", IntegerType(), True),
                StructField("wp_image_count", IntegerType(), True),
                StructField("wp_max_ad_count", IntegerType(), True),
            ]
        ),
        "web_returns": StructType(
            [
                StructField("wr_returned_date_sk", IntegerType(), True),
                StructField("wr_returned_time_sk", IntegerType(), True),
                StructField("wr_item_sk", IntegerType(), True),
                StructField("wr_refunded_customer_sk", IntegerType(), True),
                StructField("wr_refunded_cdemo_sk", IntegerType(), True),
                StructField("wr_refunded_hdemo_sk", IntegerType(), True),
                StructField("wr_refunded_addr_sk", IntegerType(), True),
                StructField("wr_returning_customer_sk", IntegerType(), True),
                StructField("wr_returning_cdemo_sk", IntegerType(), True),
                StructField("wr_returning_hdemo_sk", IntegerType(), True),
                StructField("wr_returning_addr_sk", IntegerType(), True),
                StructField("wr_web_page_sk", IntegerType(), True),
                StructField("wr_reason_sk", IntegerType(), True),
                StructField("wr_order_number", IntegerType(), True),
                StructField("wr_return_quantity", IntegerType(), True),
                StructField("wr_return_amt", DoubleType(), False),
                StructField("wr_return_tax", DoubleType(), False),
                StructField("wr_return_amt_inc_tax", DoubleType(), False),
                StructField("wr_fee", DoubleType(), False),
                StructField("wr_return_ship_cost", DoubleType(), False),
                StructField("wr_refunded_cash", DoubleType(), False),
                StructField("wr_reversed_charge", DoubleType(), False),
                StructField("wr_account_credit", DoubleType(), False),
                StructField("wr_net_loss", DoubleType(), False),
            ]
        ),
    }
    return (dataframe_dict, pyspark_schemas)


@bodo.jit(returns_maybe_distributed=False, cache=True)
def load_tpcxbb_data(dir_name):
    """Load the necessary TPCxBB dataframes given a root directory.
    We use bodo.jit so we can read easily from a directory."""
    # TODO: Update all the data frames selected so every query returns
    # a non-empty DataFrame
    store_sales_df = pd.read_parquet(
        dir_name + "/store_sales", dtype_backend="pyarrow"
    ).head(1000)
    # we need the entire item df, so we don't get empty queries
    item_df = pd.read_parquet(dir_name + "/item", dtype_backend="pyarrow")
    customer_df = pd.read_parquet(dir_name + "/customer", dtype_backend="pyarrow").head(
        1000
    )
    customer_address_df = pd.read_parquet(
        dir_name + "/customer_address", dtype_backend="pyarrow"
    ).head(1000)
    customer_demographics_df = pd.read_parquet(
        dir_name + "/customer_demographics", dtype_backend="pyarrow"
    ).head(1000)
    date_dim_df = pd.read_parquet(dir_name + "/date_dim", dtype_backend="pyarrow").head(
        1000
    )
    product_reviews_df = pd.read_parquet(
        dir_name + "/product_reviews", dtype_backend="pyarrow"
    ).head(1000)
    store_df = pd.read_parquet(dir_name + "/store", dtype_backend="pyarrow").head(1000)
    web_clickstreams_df = pd.read_parquet(
        dir_name + "/web_clickstreams", dtype_backend="pyarrow"
    ).head(1000)
    web_sales_df = pd.read_parquet(
        dir_name + "/web_sales", dtype_backend="pyarrow"
    ).head(1000)
    household_demographics_df = pd.read_parquet(
        dir_name + "/household_demographics", dtype_backend="pyarrow"
    ).head(1000)
    inventory_df = pd.read_parquet(
        dir_name + "/inventory", dtype_backend="pyarrow"
    ).head(1000)
    item_marketprices_df = pd.read_parquet(
        dir_name + "/item_marketprices", dtype_backend="pyarrow"
    ).head(1000)
    promotion_df = pd.read_parquet(
        dir_name + "/promotion", dtype_backend="pyarrow"
    ).head(1000)
    store_returns_df = pd.read_parquet(
        dir_name + "/store_returns", dtype_backend="pyarrow"
    ).head(1000)
    time_dim_df = pd.read_parquet(dir_name + "/time_dim", dtype_backend="pyarrow").head(
        1000
    )
    # the warehouse df and web_page_df currently only contains 3 and 4 values respectivley,
    # which causes issues with distributed tests.
    # For right now, since neither of these dataframes are actually being used in any of the
    # queries, I'm just concatinating it to itself 3 times to make it large enough so
    # we don't get distribution errors.
    warehouse_df = pd.read_parquet(dir_name + "/warehouse", dtype_backend="pyarrow")
    warehouse_df = pd.concat(
        [warehouse_df, warehouse_df, warehouse_df], ignore_index=True
    )
    web_page_df = pd.read_parquet(dir_name + "/web_page", dtype_backend="pyarrow")
    web_page_df = pd.concat([web_page_df, web_page_df, web_page_df], ignore_index=True)
    web_returns_df = pd.read_parquet(
        dir_name + "/web_returns", dtype_backend="pyarrow"
    ).head(1000)
    return (
        store_sales_df,
        item_df,
        customer_df,
        customer_address_df,
        customer_demographics_df,
        date_dim_df,
        product_reviews_df,
        store_df,
        web_clickstreams_df,
        web_sales_df,
        household_demographics_df,
        inventory_df,
        item_marketprices_df,
        promotion_df,
        store_returns_df,
        time_dim_df,
        warehouse_df,
        web_page_df,
        web_returns_df,
    )


def pytest_collection_modifyitems(items):
    """
    called after collection has been performed.
    Marks the test to run as single_mode.
    Also Marks the tests with marker "bodosql_<x>of4".
    """
    azure_1p_markers = [
        pytest.mark.bodosql_1of12,
        pytest.mark.bodosql_2of12,
        pytest.mark.bodosql_3of12,
        pytest.mark.bodosql_4of12,
        pytest.mark.bodosql_5of12,
        pytest.mark.bodosql_6of12,
        pytest.mark.bodosql_7of12,
        pytest.mark.bodosql_8of12,
        pytest.mark.bodosql_9of12,
        pytest.mark.bodosql_10of12,
        pytest.mark.bodosql_11of12,
        pytest.mark.bodosql_12of12,
    ]
    azure_2p_markers = [
        pytest.mark.bodosql_1of22,
        pytest.mark.bodosql_2of22,
        pytest.mark.bodosql_3of22,
        pytest.mark.bodosql_4of22,
        pytest.mark.bodosql_5of22,
        pytest.mark.bodosql_6of22,
        pytest.mark.bodosql_7of22,
        pytest.mark.bodosql_8of22,
        pytest.mark.bodosql_9of22,
        pytest.mark.bodosql_10of22,
        pytest.mark.bodosql_11of22,
        pytest.mark.bodosql_12of22,
        pytest.mark.bodosql_13of22,
        pytest.mark.bodosql_14of22,
        pytest.mark.bodosql_15of22,
        pytest.mark.bodosql_16of22,
        pytest.mark.bodosql_17of22,
        pytest.mark.bodosql_18of22,
        pytest.mark.bodosql_19of22,
        pytest.mark.bodosql_20of22,
        pytest.mark.bodosql_21of22,
        pytest.mark.bodosql_22of22,
    ]
    azure_3p_markers = [
        pytest.mark.bodosql_1of7,
        pytest.mark.bodosql_2of7,
        pytest.mark.bodosql_3of7,
        pytest.mark.bodosql_4of7,
        pytest.mark.bodosql_5of7,
        pytest.mark.bodosql_6of7,
        pytest.mark.bodosql_7of7,
    ]
    # to run the tests from the given test file. In this case, we add the
    # "single_mod" mark to the tests belonging to that module. This envvar is
    # set in runtests.py, which also adds the "-m single_mod" to the pytest
    # command (thus ensuring that only those tests run)
    module_to_run = os.environ.get("BODO_TEST_PYTEST_MOD", None)
    if module_to_run is not None:
        for item in items:
            if module_to_run == item.module.__name__.split(".")[-1] + ".py":
                item.add_marker(pytest.mark.single_mod)

    for item in items:
        hash_ = bodo.tests.conftest.get_last_byte_of_test_hash(item)
        # Divide the tests evenly so larger tests like TPCH
        # don't end up entirely in 1 group
        azure_1p_marker = azure_1p_markers[hash_ % len(azure_1p_markers)]
        azure_2p_marker = azure_2p_markers[hash_ % len(azure_2p_markers)]
        azure_3p_marker = azure_3p_markers[hash_ % len(azure_3p_markers)]
        item.add_marker(azure_1p_marker)
        item.add_marker(azure_2p_marker)
        item.add_marker(azure_3p_marker)


def group_from_hash(testname, num_groups):
    """
    Hash function to randomly distribute tests not found in the log.
    Keeps all s3 tests together in group 0.
    """
    if "test_s3.py" in testname:
        return "0"
    # TODO(Nick): Replace with a cheaper function.
    # Python's builtin hash fails on mpiexec -n 2 because
    # it has randomness. Instead we use a cryptographic hash,
    # but we don't need that level of support.
    hash_val = hashlib.sha1(testname.encode("utf-8")).hexdigest()
    # Hash val is a hex-string
    int_hash = int(hash_val, base=16) % num_groups
    return str(int_hash)


@pytest.fixture
def timeadd_dataframe():
    """Returns a table wkth two columns: one of various time values,
    and the other is various amounts of each unit that can be added
    by a query."""
    time_args_list = [
        (0, 0, 0, 0),
        (1, 1, 1, 1),
        (2, 4, 8, 16),
        None,
        (14, 52, 48, 20736),
        (16, 25, 37, 28561),
        (18, 1, 44, 38416),
    ]
    return {
        "TABLE1": pd.DataFrame(
            {
                "T": [
                    None
                    if t is None
                    else bodo.types.Time(
                        hour=t[0], minute=t[1], second=t[2], nanosecond=t[3]
                    )
                    for t in time_args_list
                ],
                "N": [-50, 7, -22, 13, -42, -17, 122],
            }
        )
    }


@pytest.fixture(
    params=[
        "HOUR",
        pytest.param("MINUTE", marks=pytest.mark.slow),
        "SECOND",
        "MILLISECOND",
        pytest.param("MICROSECOND", marks=pytest.mark.slow),
        "NANOSECOND",
    ]
)
def timeadd_arguments(request, timeadd_dataframe):
    """For each time unit, returns the answer column created by using timeadd
    on the columns created in timeadd_dataframe with the specified unit"""
    time_args_lists = {
        "HOUR": [
            (22, 0, 0, 0),
            (8, 1, 1, 1),
            (4, 4, 8, 16),
            None,
            (20, 52, 48, 20736),
            (23, 25, 37, 28561),
            (20, 1, 44, 38416),
        ],
        "MINUTE": [
            (23, 10, 0, 0),
            (1, 8, 1, 1),
            (1, 42, 8, 16),
            None,
            (14, 10, 48, 20736),
            (16, 8, 37, 28561),
            (20, 3, 44, 38416),
        ],
        "SECOND": [
            (23, 59, 10, 0),
            (1, 1, 8, 1),
            (2, 3, 46, 16),
            None,
            (14, 52, 6, 20736),
            (16, 25, 20, 28561),
            (18, 3, 46, 38416),
        ],
        "MILLISECOND": [
            (23, 59, 59, 950000000),
            (1, 1, 1, 7000001),
            (2, 4, 7, 978000016),
            None,
            (14, 52, 47, 958020736),
            (16, 25, 36, 983028561),
            (18, 1, 44, 122038416),
        ],
        "MICROSECOND": [
            (23, 59, 59, 999950000),
            (1, 1, 1, 7001),
            (2, 4, 7, 999978016),
            None,
            (14, 52, 47, 999978736),
            (16, 25, 37, 11561),
            (18, 1, 44, 160416),
        ],
        "NANOSECOND": [
            (23, 59, 59, 999999950),
            (1, 1, 1, 8),
            (2, 4, 7, 999999994),
            None,
            (14, 52, 48, 20694),
            (16, 25, 37, 28544),
            (18, 1, 44, 38538),
        ],
    }
    answer = pd.DataFrame(
        {
            0: timeadd_dataframe["TABLE1"]["T"],
            1: [
                None
                if t is None
                else bodo.types.Time(
                    hour=t[0], minute=t[1], second=t[2], nanosecond=t[3]
                )
                for t in time_args_lists[request.param]
            ],
        }
    )
    return request.param, answer


@pytest.fixture()
def listagg_data():
    """When doing listagg without any grouping, the order is completely random.
    Therefore, to avoid non-deterministic expected output,
    we include several columns for which the
    value is always the same per group.
    """
    return {
        "TABLE1": pd.DataFrame(
            {
                "KEY_COL": [1] * 6 + [2] * 6 + [3] * 6,
                "GROUP_CONSTANT_STR_COL": ["a"] * 6 + ["œ"] * 6 + ["e"] * 6,
                "GROUP_CONSTANT_STR_COL2": ["į"] * 6 + ["ë"] * 6 + ["₠"] * 6,
                "NON_CONSTANT_STR_COL": list(string.ascii_uppercase[:6]) * 3,
                "NON_CONSTANT_STR_COL_WITH_NULLS": np.array(
                    [
                        None,
                        None,
                        "hi",
                        "hello",
                        None,
                        "world",
                    ]
                    * 3
                ),
                "ALL_NULL_COLUMN": pd.array(
                    [None] * 18, dtype=pd.ArrowDtype(pa.string())
                ),
                "ORDER_COL_1": [None, 1, None, 2, None, 3] * 3,
                "ORDER_COL_2": [1, None, 2, None, 3, None] * 3,
                "ORDER_COL_3": np.arange(18),
                # Group 1 will pass the HAVING check, group 2 will not
                # Group 3 will pass
                "HAVING_LEN_STR": [""] * 6 + ["aaaa"] * 6 + ["a"] * 6,
            }
        )
    }


@pytest.fixture
def glue_catalog():
    """
    Returns a glue catalog object
    """

    warehouse = "s3://icebergglue-ci"
    return bodosql.GlueCatalog(warehouse=warehouse)


@pytest.fixture
def s3_tables_catalog():
    """
    Returns a s3 tables catalog object
    """

    warehouse = "arn:aws:s3tables:us-east-2:427443013497:bucket/unittest-bucket"
    return bodosql.S3TablesCatalog(warehouse=warehouse)


@pytest.fixture(
    params=[
        pytest.param(
            pd.array(
                [
                    "1",
                    "1.55",
                    "1.56",
                    "10.56",
                    "1000.5",
                    None,
                    None,
                    "10004.1",
                    "-11.41",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            )
        ),
    ]
)
def precision_scale_decimal_array(request):
    return request.param


pytest_mark_javascript = pytest.mark.skipif(
    not bodosql.kernels.javascript_udf_array_kernels.javascript_udf_enabled,
    reason="JavaScript UDFs are not enabled",
)
