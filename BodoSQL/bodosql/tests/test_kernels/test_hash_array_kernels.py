"""Test Bodo's array kernel utilities for BodoSQL hashing functions"""

import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from numba.core.utils import PYVERSION

import bodo
import bodosql
from bodo.tests.utils import check_func, pytest_slow_unless_codegen

# Skip unless any library or BodoSQL codegen or files were changed
pytestmark = pytest_slow_unless_codegen


@pytest.mark.skip("TODO[BSE-4112]: fix flaky test")
@pytest.mark.parametrize(
    "args, scalars, distinct",
    [
        pytest.param(
            (
                pd.Series(
                    [
                        None if (i % 10000) == 77 else (i**3) % 2**20
                        for i in range(2**12)
                    ],
                    dtype=pd.Int32Dtype(),
                ),
            ),
            (False,),
            3987,
            id="int32",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        None if "0" in str(i) else str(int(str(i)[:3]) ** 5)
                        for i in range(2**12)
                    ]
                ),
                pd.Series(
                    [[True, False, None][i % 3] for i in range(2**12)],
                    dtype=pd.BooleanDtype(),
                ),
            ),
            (False, False),
            (661 if PYVERSION in ((3, 11), (3, 13), (3, 14)) else 1308),
            id="string-bool",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        None
                        if "9" in str(i)
                        else bytes(
                            str(int(str(i**2)[:3]) ** 5) * len(str(i**2)),
                            encoding="utf-8",
                        )
                        for i in range(2**12)
                    ]
                ),
            ),
            (False,),
            (745 if PYVERSION in ((3, 11), (3, 13), (3, 14)) else 1479),
            id="binary",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        None if i % 10 == 0 else bodo.types.Time(minute=i**2)
                        for i in range(2**12)
                    ]
                ),
                pd.Series([np.round(np.tan(i), 0) + 0.5 for i in range(2**12)]),
                pd.Series(
                    [
                        datetime.date.fromordinal(736695 + (i % 60) ** 2)
                        for i in range(2**12)
                    ]
                ),
            ),
            (False, False, False),
            2359,
            id="time-float-date",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        pd.Timestamp("2020") + pd.Timedelta(days=(i % 1000))
                        for i in range(2**12)
                    ],
                    dtype="datetime64[ns]",
                ),
                pd.Series(
                    [
                        pd.Timestamp("2020", tz="US/Pacific")
                        + pd.Timedelta(hours=(2 ** (i % 18)))
                        for i in range(2**12)
                    ],
                    dtype="datetime64[ns, US/Pacific]",
                ),
                None,
            ),
            (False, False, True),
            4096,
            id="naive-timezone-null",
        ),
        pytest.param(
            (
                42,
                pd.Series([str(i**2)[-3:] for i in range(2**12)]),
                "foo",
                None,
                pd.Series([int(i**0.5) for i in range(2**12)], dtype=pd.Int32Dtype()),
                pd.Series(
                    [[b"foo", b"bar", b"fizz", b"buzz"][i % 4] for i in range(2**12)]
                ),
                datetime.date(2020, 7, 3),
            ),
            (True, False, True, True, False, False, True),
            (1873 if PYVERSION in ((3, 11), (3, 13), (3, 14)) else 3564),
            id="mixed",
        ),
        pytest.param(
            (
                pd.array(
                    [
                        [],
                        [1],
                        [1, 2],
                        [2, 1, None],
                        [1, None],
                        [2, 1],
                        [None, 1],
                        [1, 1],
                        [0, 1, 2, None, 1],
                        None,
                    ]
                    * 1_000,
                    dtype=pd.ArrowDtype(pa.large_list(pa.int8())),
                ),
                pd.array(
                    [
                        {
                            "id": (i // 1000) % 100,
                            "tag": None
                            if ((i // 1000) % 10) == 7
                            else str((i // 1000) % 10),
                        }
                        for i in range(10_000)
                    ],
                    dtype=pd.ArrowDtype(
                        pa.struct(
                            [pa.field("id", pa.int32()), pa.field("tag", pa.string())]
                        )
                    ),
                ),
                pd.array(
                    [
                        [
                            {},
                            {"A": 0},
                            {"A": 0, "B": 1},
                            {"A": 1, "B": 1},
                            {"A": 0, "B": 0},
                            {"A": 1, "B": 0},
                            {"A": 0, "B": None},
                            {"A": None, "B": None},
                            {"A": 1, "B": 0, "C": 2},
                            None,
                        ][(i // 10) % 10]
                        for i in range(10_000)
                    ],
                    dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int32())),
                ),
            ),
            (False, False, False),
            1000,
            id="semi_structured",
        ),
    ],
)
def test_sql_hash_qualities(args, distinct, scalars, memory_leak_check):
    """
    Tests the quality of the sql HASH kernel by verifying that the number of
    distinct hashes matches the number of distinct inputs and that the hashes
    are arbitrarily distributed across the domain of int64. It checks this by
    verifying that each of of the 64 bits is set 50% of the time, and each pair
    of bits has both bits set 25% of the time. If this is true, it means that
    each bit is effectively an independent random bernouli variable with p=0.5.

    All of these requirements are reasonable expectations for an array of inputs
    that is not trivially small yet not big enough that there is a reasonable
    chance of a hash collision in the domain of int64.

    Note: the various types are tested together so that these tests can explore
    the variadic nature of the HASH function.
    """

    n_args = len(args)
    args_str = ", ".join([f"A{i}" for i in range(n_args)])
    params_str = args_str + ", n"
    if n_args == 1:
        args_str += ","

    # Returns a series where the first 64 entries are the ratio of times where
    # that bit was set to one, and the remainder each correspond to the ratio
    # of times that a specific combination of bits are both set. If the hash
    # function is working correctly, the first 64 values should be approximately
    # 0.5 and the rest should be approximately 0.25. Also returns the number
    # of distinct hash values. Uses abs() so that the highest order bit does
    # not mess with the sign of the ratios.
    test_impl = f"def impl({params_str}):\n"
    test_impl += f"  H = pd.Series(bodosql.kernels.sql_hash(({args_str}), scalars))\n"
    test_impl += "  distinct_hashes = pd.Series(H.unique())\n"
    test_impl += "  masks = []\n"
    test_impl += "  L = []\n"
    test_impl += "  for i in range(n):\n"
    test_impl += "    mask = pd.Series((distinct_hashes.values & (1 << i)) >> i)\n"
    test_impl += "    masks.append(mask)\n"
    test_impl += "    L.append(abs(mask.mean()))\n"
    test_impl += "  for i in range(n):\n"
    test_impl += "    for j in range(i+1, n):\n"
    test_impl += "      mask_both = pd.Series(masks[i] & masks[j])\n"
    test_impl += "      L.append(abs(mask_both.mean()))\n"
    test_impl += "  return len(distinct_hashes), pd.Series(L)\n"
    impl_vars = {}
    exec(
        test_impl,
        {"bodosql": bodosql, "pd": pd, "scalars": bodo.utils.typing.MetaType(scalars)},
        impl_vars,
    )
    impl = impl_vars["impl"]
    # Pass loop lengths as argument to avoid slow compilation time due to extensive
    # loop unrolling
    n = 64

    # 2016 = number of combinations of i & j
    expected_bits = pd.Series([0.5] * n + [0.25] * 2016)

    check_func(
        impl,
        args + (n,),
        py_output=(distinct, expected_bits),
        check_dtype=False,
        is_out_distributed=False,
        atol=0.1,
    )


@pytest.mark.parametrize(
    "args, expected_hash",
    [
        pytest.param((10, None, True), -2243023243364725697, id="int-null-bool"),
        pytest.param((np.float32(3.1415),), -9176520268571280804, id="float"),
        pytest.param((np.int8(100),), 2279747396317938166, id="int8"),
        pytest.param((np.uint8(100),), 2279747396317938166, id="uint8"),
        pytest.param((np.int32(100),), 2279747396317938166, id="int32"),
        pytest.param(
            (datetime.date(1999, 12, 31), bodo.types.Time(12, 30, 0)),
            5066071753504198015,
            id="date-time",
        ),
        pytest.param(
            (
                pd.Timestamp("2023-7-4 6:00:00"),
                pd.Timestamp("2020-4-1", tz="US/Pacific"),
            ),
            -8057843889034702324,
            id="timestamps_baseline",
        ),
        pytest.param(
            (
                pd.Timestamp("2023-7-4 8:00:00", tz="Europe/Berlin"),
                pd.Timestamp("2020-4-1 3:00:00", tz="US/Eastern"),
            ),
            -8057843889034702324,
            id="timestamps_equivalent",
        ),
        # NOTE: Python 3.11 changes calculation semantics somehow even though our
        # implementation hasn't changed (seems more accurate than 3.10 based on
        # experiments)
        pytest.param(
            ("theta",),
            (
                -850204814874656711
                if PYVERSION in ((3, 11), (3, 13), (3, 14))
                else 7137812097207502893
            ),
            id="string",
        ),
        pytest.param(
            (b"theta",),
            (
                1852596461571890431
                if PYVERSION in ((3, 11), (3, 13), (3, 14))
                else -4192600820579827718
            ),
            id="binary",
        ),
        pytest.param(
            (Decimal("20.31"),),
            2905488202071118327
            if PYVERSION in ((3, 13), (3, 14))
            else -7665313548287772755,
            id="decimal",
        ),
    ],
)
def test_sql_hash_determinism(args, expected_hash, memory_leak_check):
    """
    Verifies that the sql_hash kernel returns the same outputs for the same
    combination of inputs every time, including for equivalent values
    of different types"
    """
    scalars = bodo.utils.typing.MetaType((True,) * len(args))
    n_args = len(args)
    args_str = ", ".join([f"A{i}" for i in range(n_args)])
    params_str = args_str
    if n_args == 1:
        args_str += ","

    test_impl = f"def impl({params_str}):\n"
    test_impl += f"  return bodosql.kernels.sql_hash(({args_str}), scalars)"
    impl_vars = {}
    exec(test_impl, {"bodosql": bodosql, "scalars": scalars}, impl_vars)
    impl = impl_vars["impl"]
    check_func(impl, args, py_output=expected_hash)
