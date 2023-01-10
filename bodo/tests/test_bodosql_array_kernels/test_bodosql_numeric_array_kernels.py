# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Test Bodo's array kernel utilities for BodoSQL numeric functions
"""

import math

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.libs.bodosql_array_kernels import *
from bodo.tests.utils import check_func


@pytest.fixture(
    params=[
        pytest.param(
            (
                pd.Series(
                    [0, 1, -1, 255, 42, 2147483647, None], dtype=pd.Int32Dtype()
                ).repeat(7),
                pd.Series(
                    [0, 1, -1, 255, 42, 2147483647, None] * 7, dtype=pd.Int32Dtype()
                ),
            ),
            id="vector_vector_int32_int32",
        ),
        pytest.param(
            (
                pd.Series(
                    [0, 1, 2, 3, 42, 64, 127, 128, 255, None], dtype=pd.UInt8Dtype()
                ).repeat(10),
                pd.Series(
                    [0, 1, -1, 127, 128, -127, -128, 255, 2147483647, None] * 10,
                    dtype=pd.Int32Dtype(),
                ),
            ),
            id="vector_vector_uint8_int32",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(
                    [0, 1, -1, 2**63 - 1, 2**16 - 1, None], dtype=pd.Int64Dtype()
                ).repeat(7),
                pd.Series(
                    [0, 1, 13, 43690, 2**16 - 1, None] * 7, dtype=pd.UInt16Dtype()
                ),
            ),
            id="vector_vector_int64_uint16",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series([0, 1, 127, 128, 255, None], dtype=pd.UInt8Dtype()).repeat(6),
                pd.Series([0, 1, 127, 128, 255, None] * 6, dtype=pd.UInt8Dtype()),
            ),
            id="vector_vector_uint8_uint8",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        5566427618757447113,
                        11811387436593211185,
                        9116740843431364080,
                        10956942410480843441,
                        2266820874918711799,
                    ],
                    dtype=pd.UInt64Dtype(),
                ).repeat(5),
                pd.Series(
                    [
                        8468723804801797255,
                        5301527685162013423,
                        16398923257674642784,
                        11948855153700719831,
                        16424487987459669129,
                    ]
                    * 5,
                    dtype=pd.UInt64Dtype(),
                ),
            ),
            id="vector_vector_uint64_uint64",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series([0, 1, 42, 2**64 - 1, None], dtype=pd.UInt64Dtype()),
                np.int64(-42),
            ),
            id="vector_scalar_uint64_int64",
        ),
        pytest.param(
            (pd.Series([0, 1, 42, 2**64 - 1, None], dtype=pd.UInt64Dtype()), None),
            id="vector_null",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (np.uint16(2**16 - 1), np.int32(2**16 - 1)),
            id="scalar_scalar_uint16_int32",
        ),
        pytest.param(
            (np.uint16(2**16 - 1), None), id="scalar_null", marks=pytest.mark.slow
        ),
        pytest.param((None, None), id="null_null", marks=pytest.mark.slow),
    ]
)
def test_bitwise_number_number(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param(
            (
                pd.Series(
                    [0, 1, -1, 255, 42, 2147483647, 16, 64, None], dtype=pd.Int32Dtype()
                ).repeat(9),
                pd.Series(
                    [0, 1, 2, 3, 10, 29, 30, 31, None] * 9, dtype=pd.Int32Dtype()
                ),
            ),
            id="vector_vector_int32",
        ),
        pytest.param(
            (
                pd.Series(
                    [0, 1, 2, 8, 255, 127, 128, 42, None], dtype=pd.UInt8Dtype()
                ).repeat(9),
                pd.Series([0, 1, 2, 3, 4, 5, 6, 7, None] * 9, dtype=pd.Int32Dtype()),
            ),
            id="vector_vector_uint8",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(
                    [0, 1, -1, -42, -127, 127, -128, 42, None],
                    dtype=pd.Int8Dtype(),
                ).repeat(9),
                pd.Series([0, 1, 2, 3, 4, 5, 6, 7, None] * 9, dtype=pd.Int32Dtype()),
            ),
            id="vector_vector_int8",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        0,
                        1,
                        2,
                        100,
                        2**33,
                        2**63 - 1,
                        -(2**63),
                        10**15,
                        -(10**15),
                        None,
                    ],
                    dtype=pd.Int64Dtype(),
                ).repeat(10),
                pd.Series(
                    [0, 1, 2, 4, 32, 63, 62, 61, 20, None] * 10, dtype=pd.Int32Dtype()
                ),
            ),
            id="vector_vector_int64",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(
                    [0, 1, 2, 3, 2**16 - 1, 2**15 - 1, 12345, None],
                    dtype=pd.UInt16Dtype(),
                ),
                8,
            ),
            id="vector_scalar_uint16",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (np.uint64(2**64 - 2**62 - 1), np.uint64(60)),
            id="scalar_scalar_uint64",
        ),
        pytest.param((None, np.uint64(42)), id="null_scalar", marks=pytest.mark.slow),
        pytest.param((None, None), id="null_null", marks=pytest.mark.slow),
    ]
)
def test_bitwise_number_bits(request):
    return request.param


def test_bitand(test_bitwise_number_number):
    def impl(A, B):
        return pd.Series(bodo.libs.bodosql_array_kernels.bitand(A, B))

    # avoid Series conversion for scalar output
    if not isinstance(test_bitwise_number_number[0], pd.Series) and not isinstance(
        test_bitwise_number_number[1], pd.Series
    ):
        impl = lambda A, B: bodo.libs.bodosql_array_kernels.bitand(A, B)

    # Simulates BITAND on a single row
    def bitand_scalar_fn(A, B):
        if pd.isna(A) or pd.isna(B):
            return None
        else:
            if {type(A), type(B)} == {np.uint64, np.int64}:
                A, B = np.int64(A), np.int64(B)
            return A & B

    bitand_answer = vectorized_sol(test_bitwise_number_number, bitand_scalar_fn, None)
    check_func(
        impl,
        test_bitwise_number_number,
        py_output=bitand_answer,
        check_dtype=False,
        reset_index=True,
        sort_output=False,
    )


def test_bitshiftleft(test_bitwise_number_bits):
    def impl(A, B):
        return pd.Series(bodo.libs.bodosql_array_kernels.bitshiftleft(A, B))

    # avoid Series conversion for scalar output
    if not isinstance(test_bitwise_number_bits[0], pd.Series):
        impl = lambda A, B: bodo.libs.bodosql_array_kernels.bitshiftleft(A, B)

    # Simulates BITSHIFTLEFT on a single row
    def bitshiftleft_scalar_fn(A, B):
        if pd.isna(A) or pd.isna(B):
            return None
        else:
            if not isinstance(A, (np.uint8, np.uint16, np.uint32, np.uint64)):
                return np.int64(A) << B
            else:
                return np.uint64(A) << np.uint8(B)

    bitshiftleft_answer = vectorized_sol(
        test_bitwise_number_bits, bitshiftleft_scalar_fn, pd.Int64Dtype()
    )
    check_func(
        impl,
        test_bitwise_number_bits,
        py_output=bitshiftleft_answer,
        check_dtype=False,
        reset_index=True,
        sort_output=False,
    )


def test_bitnot(test_bitwise_number_number):
    def impl(A):
        return pd.Series(bodo.libs.bodosql_array_kernels.bitnot(A))

    # avoid Series conversion for scalar values
    impl_scalar = lambda A: bodo.libs.bodosql_array_kernels.bitnot(A)

    # Simulates BITNOT on a single row
    def bitnot_scalar_fn(A):
        if pd.isna(A):
            return None
        else:
            return ~A

    bitnot_answer_0 = vectorized_sol(
        (test_bitwise_number_number[0],), bitnot_scalar_fn, None
    )
    bitnot_answer_1 = vectorized_sol(
        (test_bitwise_number_number[1],), bitnot_scalar_fn, None
    )

    check_func(
        impl if isinstance(test_bitwise_number_number[0], pd.Series) else impl_scalar,
        (test_bitwise_number_number[0],),
        py_output=bitnot_answer_0,
        check_dtype=False,
        reset_index=True,
        sort_output=False,
    )
    check_func(
        impl if isinstance(test_bitwise_number_number[1], pd.Series) else impl_scalar,
        (test_bitwise_number_number[1],),
        py_output=bitnot_answer_1,
        check_dtype=False,
        reset_index=True,
        sort_output=False,
    )


def test_bitor(test_bitwise_number_number):
    def impl(A, B):
        return pd.Series(bodo.libs.bodosql_array_kernels.bitor(A, B))

    # avoid Series conversion for scalar values
    if not isinstance(test_bitwise_number_number[0], pd.Series) and not isinstance(
        test_bitwise_number_number[1], pd.Series
    ):
        impl = lambda A, B: bodo.libs.bodosql_array_kernels.bitor(A, B)

    # Simulates BITOR on a single row
    def bitor_scalar_fn(A, B):
        if pd.isna(A) or pd.isna(B):
            return None
        else:
            if {type(A), type(B)} == {np.uint64, np.int64}:
                A, B = np.int64(A), np.int64(B)
            return A | B

    bitor_answer = vectorized_sol(test_bitwise_number_number, bitor_scalar_fn, None)
    check_func(
        impl,
        test_bitwise_number_number,
        py_output=bitor_answer,
        check_dtype=False,
        reset_index=True,
        sort_output=False,
    )


def test_bitshiftright(test_bitwise_number_bits):
    def impl(A, B):
        return pd.Series(bodo.libs.bodosql_array_kernels.bitshiftright(A, B))

    # avoid Series conversion for scalar output
    if not isinstance(test_bitwise_number_bits[0], pd.Series):
        impl = lambda A, B: bodo.libs.bodosql_array_kernels.bitshiftright(A, B)

    # Simulates BITSHIFTRIGHT on a single row
    def bitshiftright_scalar_fn(A, B):
        if pd.isna(A) or pd.isna(B):
            return None
        else:
            return A >> B

    bitshiftright_answer = vectorized_sol(
        test_bitwise_number_bits, bitshiftright_scalar_fn, pd.Int64Dtype()
    )
    check_func(
        impl,
        test_bitwise_number_bits,
        py_output=bitshiftright_answer,
        check_dtype=False,
        reset_index=True,
        sort_output=False,
    )


def test_bitxor(test_bitwise_number_number):
    def impl(A, B):
        return pd.Series(bodo.libs.bodosql_array_kernels.bitxor(A, B))

    # avoid Series conversion for scalar values
    if not isinstance(test_bitwise_number_number[0], pd.Series) and not isinstance(
        test_bitwise_number_number[1], pd.Series
    ):
        impl = lambda A, B: bodo.libs.bodosql_array_kernels.bitxor(A, B)

    # Simulates BITAND on a single row
    def bitxor_scalar_fn(A, B):
        if pd.isna(A) or pd.isna(B):
            return None
        else:
            if {type(A), type(B)} == {np.uint64, np.int64}:
                A, B = np.int64(A), np.int64(B)
            return A ^ B

    bitxor_answer = vectorized_sol(test_bitwise_number_number, bitxor_scalar_fn, None)
    check_func(
        impl,
        test_bitwise_number_number,
        py_output=bitxor_answer,
        check_dtype=False,
        reset_index=True,
        sort_output=False,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Series(pd.array(["10", "11", "12", "13", "14", "15"])),
                pd.Series(pd.array([10, 10, 10, 16, 16, 16])),
                pd.Series(pd.array([2, 10, 16, 2, 10, 16])),
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                "11111",
                pd.Series(
                    pd.array(
                        [2, 2, 2, 2, 8, 8, 8, 8, 10, 10, 10, 10, 16, 16, 16, 16, 10, 10]
                    )
                ),
                pd.Series(
                    pd.array(
                        [2, 8, 10, 16, 2, 8, 10, 16, 2, 8, 10, 16, 2, 8, 10, 16, 17, -1]
                    )
                ),
            ),
            id="scalar_vector_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(pd.array(["2", "4", None, "8", "16", "32", "64", None])),
                pd.Series(pd.array([3, None, None, None, 16, 7, 36, 3])),
                10,
            ),
            id="vector_vector_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "FGHIJ",
                pd.Series(pd.array([20, 21, 22, 23, 24, 25])),
                10,
            ),
            id="scalar_vector_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            ("ff", 16, 2),
            id="all_scalar",
        ),
    ],
)
def test_conv(args):
    def impl(arr, old_base, new_base):
        return bodo.libs.bodosql_array_kernels.conv(arr, old_base, new_base)

    # Simulates CONV on a single row
    def conv_scalar_fn(elem, old_base, new_base):
        if (
            pd.isna(elem)
            or pd.isna(old_base)
            or pd.isna(new_base)
            or old_base <= 1
            or new_base not in [2, 8, 10, 16]
        ):
            return None
        else:
            old = int(elem, base=old_base)
            if new_base == 2:
                return "{:b}".format(old)
            if new_base == 8:
                return "{:o}".format(old)
            if new_base == 10:
                return "{:d}".format(old)
            if new_base == 16:
                return "{:x}".format(old)
            return None

    conv_answer = vectorized_sol(args, conv_scalar_fn, pd.StringDtype())
    if any([isinstance(args[i], pd.Series) for i in range(len(args))]):
        conv_answer = conv_answer.values

    check_func(
        impl,
        args,
        py_output=conv_answer,
        check_dtype=False,
    )


def test_getbit(test_bitwise_number_bits):
    def impl(A, B):
        return pd.Series(bodo.libs.bodosql_array_kernels.getbit(A, B))

    # avoid Series conversion for scalar output
    if not isinstance(test_bitwise_number_bits[0], pd.Series):
        impl = lambda A, B: bodo.libs.bodosql_array_kernels.getbit(A, B)

    # Simulates BITOR on a single row
    def getbit_scalar_fn(A, B):
        if pd.isna(A) or pd.isna(B):
            return None
        else:
            return (A >> B) & type(A)(1)

    getbit_answer = vectorized_sol(
        test_bitwise_number_bits, getbit_scalar_fn, pd.UInt8Dtype()
    )
    check_func(
        impl,
        test_bitwise_number_bits,
        py_output=getbit_answer,
        check_dtype=False,
        reset_index=True,
        sort_output=False,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Series(
                    [
                        -78.85726945311177,
                        -156.93375710814954,
                        -19.04115625541181,
                        152.50829951977403,
                        -74.45179663009215,
                        79.7887184211461,
                        -138.35469002785266,
                        67.04585260584803,
                        -176.64446359700187,
                        35.80103907658909,
                    ]
                ),
                pd.Series(
                    [
                        -20.084286862031966,
                        10.54265714664346,
                        -151.46059389329457,
                        46.01738493171913,
                        -10.672486935277629,
                        -13.376068085178595,
                        4.691735421976833,
                        -21.386902158059634,
                        39.163128201592635,
                        -31.934136047289407,
                    ]
                ),
                pd.Series(
                    [
                        -72.87052711351764,
                        -149.9021469512544,
                        -150.25995724259116,
                        164.9038532551427,
                        114.20487610943262,
                        -143.57171764060908,
                        -161.5398050194357,
                        77.64509120521262,
                        26.97530674361341,
                        31.567191669054036,
                    ]
                ),
                pd.Series(
                    [
                        -75.6061111037667,
                        -133.39597066391605,
                        -73.22893509637717,
                        -133.28934531314374,
                        55.532533813250694,
                        -47.65240562710813,
                        18.592614860798093,
                        -12.592941464617144,
                        -74.84722965705978,
                        -155.0151916730945,
                    ]
                ),
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        -78.85726945311177,
                        np.nan,
                        -19.04115625541181,
                        152.50829951977403,
                        -74.45179663009215,
                        79.7887184211461,
                        -138.35469002785266,
                        67.04585260584803,
                        -176.64446359700187,
                        35.80103907658909,
                    ]
                ),
                pd.Series(
                    [
                        -20.084286862031966,
                        10.54265714664346,
                        -151.46059389329457,
                        46.01738493171913,
                        -10.672486935277629,
                        -13.376068085178595,
                        None,
                        -21.386902158059634,
                        39.163128201592635,
                        -31.934136047289407,
                    ]
                ),
                26.97530674361341,
                -74.84722965705978,
            ),
            id="vector_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (40.7127, -74.0059, 34.0500, -118.2500),
            id="all_scalar_no_null",
        ),
        pytest.param(
            (40.7127, -74.0059, None, -118.2500),
            id="all_scalar_with_null",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_haversine(args):
    def impl(lat1, lon1, lat2, lon2):
        return pd.Series(
            bodo.libs.bodosql_array_kernels.haversine(lat1, lon1, lat2, lon2)
        )

    # avoid Series conversion for scalar output
    if all(not isinstance(arg, pd.Series) for arg in args):
        impl = lambda lat1, lon1, lat2, lon2: bodo.libs.bodosql_array_kernels.haversine(
            lat1, lon1, lat2, lon2
        )

    def haversine_scalar_fn(lat1, lon1, lat2, lon2):
        if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
            return None
        lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.square(np.sin(dlat / 2))
        a += np.cos(lat1) * np.cos(lat2) * np.square(np.sin(dlon / 2))
        d = 2 * 6371 * np.arcsin(np.sqrt(a))
        return d

    py_output = vectorized_sol(args, haversine_scalar_fn, np.float64)
    check_func(
        impl,
        args,
        py_output=py_output,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Series([1.0, 2.0, 3.0, 4.0, 8.0]),
                pd.Series([6.0, 0.0, 2.0, 0.0, 0.0]),
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                pd.Series([1.1, None, 3.6, 10.0, 16.0, 17.3, 101.0]),
                0.0,
            ),
            id="vector_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (1.0, 0.0),
            id="all_scalar_no_null",
        ),
        pytest.param((None, 5.6), id="all_scalar_with_null", marks=pytest.mark.slow),
    ],
)
def test_div0(args):
    def impl(a, b):
        return pd.Series(bodo.libs.bodosql_array_kernels.div0(a, b))

    # avoid Series conversion for scalar output
    if all(not isinstance(arg, pd.Series) for arg in args):
        impl = lambda a, b: bodo.libs.bodosql_array_kernels.div0(a, b)

    def div0_scalar_fn(a, b):
        if pd.isna(a) or pd.isna(b):
            return None
        elif b != 0:
            return a / b
        else:
            return 0

    a, b = args
    expected_output = vectorized_sol(args, div0_scalar_fn, np.float64)

    check_func(
        impl,
        (
            a,
            b,
        ),
        py_output=expected_output,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Series([1.0, 2.0, 3.0, 4.0, 8.0]),
                pd.Series([6.0, 2.0, 2.0, 10.5, 2.0]),
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                pd.Series([1.1, None, 3.6, 10.0, 16.0, 17.3, 101.0]),
                2.0,
            ),
            id="vector_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (64.0, 4.0),
            id="all_scalar_no_null",
        ),
        pytest.param((None, 5.6), id="all_scalar_with_null", marks=pytest.mark.slow),
    ],
)
def test_log(args):
    def impl(arr, base):
        return pd.Series(bodo.libs.bodosql_array_kernels.log(arr, base))

    # avoid Series conversion for scalar output
    if all(not isinstance(arg, pd.Series) for arg in args):
        impl = lambda arr, base: bodo.libs.bodosql_array_kernels.log(arr, base)

    # Simulates LOG on a single row
    def log_scalar_fn(elem, base):
        if pd.isna(elem) or pd.isna(base):
            return None
        else:
            return np.log(elem) / np.log(base)

    log_answer = vectorized_sol(args, log_scalar_fn, np.float64)
    check_func(
        impl,
        args,
        py_output=log_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "numbers",
    [
        pytest.param(
            pd.Series([1, 0, 2345678, -910, None], dtype=pd.Int64Dtype()),
            id="vector_int",
        ),
        pytest.param(
            pd.Series(pd.array([0, 1, 32, 127, -126, 125], dtype=pd.Int8Dtype())),
            id="vector_int8",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(
                pd.array(
                    [0, 1, 32, 127, 128, 129, 251, 252, 253, 254, 255],
                    dtype=pd.UInt8Dtype(),
                )
            ),
            id="vector_uint8",
        ),
        pytest.param(
            pd.Series(
                pd.array([0, 1, 100, 1000, 32767, 32768, 65535], dtype=pd.UInt16Dtype())
            ),
            id="vector_uint16",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(
                pd.array(
                    [0, 100, 32767, 32768, 65535, 4294967295], dtype=pd.UInt32Dtype()
                )
            ),
            id="vector_uint32",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(
                pd.array(
                    [
                        0,
                        100,
                        32767,
                        4294967295,
                        9223372036854775806,
                        9223372036854775807,
                    ],
                    dtype=pd.UInt64Dtype(),
                )
            ),
            id="vector_uint64",
        ),
        pytest.param(
            pd.Series([-1.0, 0.0, -123.456, 4096.1, None], dtype=np.float64),
            id="vector_float",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            42,
            id="scalar_int",
        ),
        pytest.param(-12.345, id="scalar_float", marks=pytest.mark.slow),
    ],
)
def test_negate(numbers):
    def impl(arr):
        return pd.Series(bodo.libs.bodosql_array_kernels.negate(arr))

    # avoid Series conversion for scalar output
    if not isinstance(numbers, pd.Series):
        impl = lambda arr: bodo.libs.bodosql_array_kernels.negate(arr)

    # Simulates -X on a single row
    def negate_scalar_fn(elem):
        if pd.isna(elem):
            return None
        else:
            return -elem

    if (
        isinstance(numbers, pd.Series)
        and not isinstance(numbers.dtype, np.dtype)
        and numbers.dtype
        in (pd.UInt8Dtype(), pd.UInt16Dtype(), pd.UInt32Dtype(), pd.UInt64Dtype())
    ):
        dtype = {
            pd.UInt8Dtype(): pd.Int16Dtype(),
            pd.UInt16Dtype(): pd.Int32Dtype(),
            pd.UInt32Dtype(): pd.Int64Dtype(),
            pd.UInt64Dtype(): pd.Int64Dtype(),
        }[numbers.dtype]
        negate_answer = vectorized_sol(
            (pd.Series(pd.array(list(numbers), dtype=dtype)),), negate_scalar_fn, dtype
        )
    else:
        negate_answer = vectorized_sol((numbers,), negate_scalar_fn, None)

    check_func(
        impl,
        (numbers,),
        py_output=negate_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Series([-0.5, -1, 2, 3, np.nan, 5, 6, 7, 8, 9, 11]),
                pd.Series([0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10]),
                pd.Series([1, 10, 11, 12, 13, 14, 15, np.nan, 17, 18, 19, 20]),
                pd.Series([20, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                pd.Series([-1, 2, 3, np.nan, 5, 6, 7, 8, 9, 11]),
                0,
                10,
                5,
            ),
            id="vectors_scalars",
        ),
        pytest.param((7, 0, 10, 3), id="all_scalar_no_null"),
        pytest.param((7, None, 10, 3), id="all_scalar_with_null"),
    ],
)
def test_width_bucket(args):
    def impl(arr, min_val, max_val, num_buckets):
        return pd.Series(
            bodo.libs.bodosql_array_kernels.width_bucket(
                arr, min_val, max_val, num_buckets
            )
        )

    # avoid Series conversion for scalar output
    if all(not isinstance(arg, pd.Series) for arg in args):
        impl = lambda arr, min_val, max_val, num_buckets: bodo.libs.bodosql_array_kernels.width_bucket(
            arr, min_val, max_val, num_buckets
        )

    def wb_scalar_fn(arr, mnv, mxv, nb):
        if pd.isna(arr) or pd.isna(mnv) or pd.isna(mxv) or pd.isna(nb):
            return None
        return min(max(-1.0, math.floor((arr - mnv) / ((mxv - mnv) / nb))), nb) + 1.0

    py_output = vectorized_sol(args, wb_scalar_fn, pd.Int32Dtype())
    check_func(
        impl,
        args,
        py_output=py_output,
        check_dtype=False,
        dist_test=False,
        is_out_distributed=False,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param((2, 3, 2, 2), id="min_over_max"),
        pytest.param((2, 1, 3, -1), id="neg_num_buckets"),
    ],
)
def test_width_bucket_error(args):
    @bodo.jit
    def impl(arr, min_val, max_val, num_buckets):
        return bodo.libs.bodosql_array_kernels.width_bucket(
            arr, min_val, max_val, num_buckets
        )

    if args[-1] < 0:
        with pytest.raises(ValueError, match="num_buckets must be a positive integer"):
            impl(*args)
    else:
        with pytest.raises(ValueError, match="min_val must be less than max_val"):
            impl(*args)


@pytest.mark.slow
def test_bitwise_option():
    def impl(A, B, flag0, flag1):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        return (
            bodo.libs.bodosql_array_kernels.bitand(arg0, arg1),
            bodo.libs.bodosql_array_kernels.bitor(arg0, arg1),
            bodo.libs.bodosql_array_kernels.bitxor(arg0, arg1),
            bodo.libs.bodosql_array_kernels.bitnot(arg0),
            bodo.libs.bodosql_array_kernels.bitnot(arg1),
            bodo.libs.bodosql_array_kernels.bitshiftleft(arg0, arg1),
            bodo.libs.bodosql_array_kernels.bitshiftright(arg0, arg1),
            bodo.libs.bodosql_array_kernels.getbit(arg0, arg1),
        )

    for flag0 in [True, False]:
        for flag1 in [True, False]:
            A0 = 2 if flag0 and flag1 else None
            A1 = 43 if flag0 and flag1 else None
            A2 = 41 if flag0 and flag1 else None
            A3 = 4294967253 if flag0 else None
            A4 = -4 if flag1 else None
            A5 = 336 if flag0 and flag1 else None
            A6 = 5 if flag0 and flag1 else None
            A7 = 1 if flag0 and flag1 else None
            answer = (A0, A1, A2, A3, A4, A5, A6, A7)
            check_func(
                impl, (np.uint32(42), np.int16(3), flag0, flag1), py_output=answer
            )


@pytest.mark.slow
def test_conv_option():
    def impl(A, B, C, flag0, flag1, flag2):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        arg2 = C if flag2 else None
        return bodo.libs.bodosql_array_kernels.conv(arg0, arg1, arg2)

    for flag0 in [True, False]:
        for flag1 in [True, False]:
            for flag2 in [True, False]:
                answer = "101010" if flag0 and flag1 and flag2 else None
                check_func(impl, ("42", 10, 2, flag0, flag1, flag2), py_output=answer)


@pytest.mark.slow
def test_haversine_option():
    def impl(a, b, c, d, flag0, flag1, flag2, flag3):
        arg0 = a if flag0 else None
        arg1 = b if flag1 else None
        arg2 = c if flag2 else None
        arg3 = d if flag3 else None
        return bodo.libs.bodosql_array_kernels.haversine(arg0, arg1, arg2, arg3)

    for flag0 in [True, False]:
        for flag1 in [True, False]:
            for flag2 in [True, False]:
                for flag3 in [True, False]:
                    answer = (
                        3936.385096389 if flag0 and flag1 and flag2 and flag3 else None
                    )
                    check_func(
                        impl,
                        (
                            40.7127,
                            -74.0059,
                            34.0500,
                            -118.2500,
                            flag0,
                            flag1,
                            flag2,
                            flag3,
                        ),
                        py_output=answer,
                    )


def test_div0_option():
    def impl(a, b, flag0, flag1):
        arg0 = a if flag0 else None
        arg1 = b if flag1 else None
        return bodo.libs.bodosql_array_kernels.div0(arg0, arg1)

    for flag0 in [True, False]:
        for flag1 in [True, False]:
            answer = 0.0 if flag0 and flag1 else None
            check_func(impl, (8.0, 0.0, flag0, flag1), py_output=answer)


@pytest.mark.slow
def test_log_option():
    def impl(A, B, flag0, flag1):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        return bodo.libs.bodosql_array_kernels.log(arg0, arg1)

    for flag0 in [True, False]:
        for flag1 in [True, False]:
            answer = 3.0 if flag0 and flag1 else None
            check_func(impl, (8.0, 2.0, flag0, flag1), py_output=answer)


@pytest.mark.slow
def test_negate_option():
    def impl(A, flag0):
        arg = A if flag0 else None
        return bodo.libs.bodosql_array_kernels.negate(arg)

    for flag0 in [True, False]:
        answer = -42 if flag0 else None
        check_func(impl, (42, flag0), py_output=answer)


test_arrs = [
    pd.Series([0, 0.5, 1, -1, -0.5, 0.3212, -0.78]),
    pd.Series(
        [
            0,
            1,
            -1,
            10000,
            -100000,
            20,
            -139,
        ]
    ),
    pd.Series([0, 1, 2, 3, 4, 5, 20]),
    pd.Series([1, -1, 0.1, -0.1, 1234, np.nan, -4321, 3], dtype=np.float32),
    pd.Series([-1, 0, 1, None, 2], dtype=pd.Int32Dtype()),
    pd.Series([0, 2, 3, 5, 3], dtype=np.uint32),
]

single_arg_np_map = {
    "abs": "np.abs",
    "cbrt": "np.cbrt",
    "ceil": "np.ceil",
    "exp": "np.exp",
    "factorial": "(lambda x: np.math.factorial(np.int64(x)) if np.abs(np.int64(x)) == x else None)",
    "floor": "np.floor",
    "ln": "np.log",
    "log2": "np.log2",
    "log10": "np.log10",
    "sign": "np.sign",
    "sqrt": "np.sqrt",
    "square": "np.square",
}
single_arg_np_list = list(single_arg_np_map.keys())
double_arg_np_map = {
    "mod": "(lambda a, b: np.fmod(a, b) if b != 0 else np.nan)",
    "power": "(lambda a, b: np.power(np.float64(a), b))",
    "round": "np.round",
    "trunc": "(lambda a, b: np.trunc(a * (10 ** b)) * (10 ** -b) if int(b) == b else np.nan)",
}
double_arg_np_list = list(double_arg_np_map.keys())


@pytest.mark.parametrize("arr", test_arrs)
@pytest.mark.parametrize("func", single_arg_np_list)
def test_numeric_single_arg_funcs(arr, func):
    if func == "factorial":
        if np.any(np.abs(arr) > 20):
            return  # skip factorial for large values
    test_impl = "def impl(arr):\n"
    test_impl += f"  return pd.Series(bodo.libs.bodosql_array_kernels.{func}(arr))"
    impl_vars = {}
    exec(test_impl, {"bodo": bodo, "pd": pd}, impl_vars)

    # Simulates numeric on a single row
    scalar_impl = "def impl(elem):\n"
    scalar_impl += (
        f"  return {single_arg_np_map[func]}(elem) if not pd.isna(elem) else None"
    )
    scalar_vars = {}
    exec(scalar_impl, {"np": np, "pd": pd}, scalar_vars)
    numeric_answer = vectorized_sol((arr,), scalar_vars["impl"], np.float64)
    check_func(
        impl_vars["impl"],
        (arr,),
        py_output=numeric_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize("func", single_arg_np_list)
def test_numeric_single_arg_option(func):

    test_impl = "def impl(a, flag0):\n"
    test_impl += "  arg0 = a if flag0 else None\n"
    test_impl += f"  return bodo.libs.bodosql_array_kernels.{func}(arg0)"
    impl_vars = {}
    exec(test_impl, {"bodo": bodo}, impl_vars)

    for flag0 in [True, False]:
        answer = eval(f"{single_arg_np_map[func]}(0.75)") if flag0 else None
        check_func(impl_vars["impl"], (0.75, flag0), py_output=answer)


@pytest.mark.parametrize(
    "arr1",
    [
        pd.Series([0, 1, 2, 3, 4, 5, 6] * 2),
        pd.Series([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5] * 2),
        pd.Series([0, -1.5, -2, -3.5, -4.5, -5, -6.5] * 2),
        pd.Series([0, -1, -2, -3, -4, -5, -6] * 2, dtype=np.int32),
    ],
)
@pytest.mark.parametrize(
    "arr0",
    [
        pd.Series(
            [0, 1, 2, 3, 4, 5, 6, None, -1, -2, -3, -4, -5, -6], dtype=pd.Int32Dtype()
        ),
        pd.Series(
            [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 0, -1.5, -2, -3.5, -4.5, -5, -6.5],
            dtype=np.float32,
        ),
        pd.Series(
            [
                0.52343,
                1.3434325,
                2.3435,
                None,
                3.345,
                4,
                5.5764,
                6.982334235,
                120.5233,
                None,
                134.325,
                69.82334235,
                None,
                None,
            ]
        ),
    ],
)
@pytest.mark.parametrize("func", double_arg_np_list)
def test_numeric_double_arg_funcs(arr0, arr1, func):
    if len(arr0) != len(arr1):
        return
    if func == "round":
        if any(np.int64(arr1) != arr1):
            return
    test_impl = "def impl(arr0, arr1):\n"
    test_impl += (
        f"  return pd.Series(bodo.libs.bodosql_array_kernels.{func}(arr0, arr1))"
    )
    impl_vars = {}
    exec(test_impl, {"bodo": bodo, "pd": pd}, impl_vars)

    # Simulates numeric func on a single row
    # Mod with divisor of 0 returns None in Bodo, but vanilla np.mod or % will throw an error.
    scalar_impl = "def impl(elem0, elem1):\n"
    scalar_impl += f"    if pd.isna(elem0) or pd.isna(elem1):\n"
    scalar_impl += f"        return None\n"
    scalar_impl += f"    else:\n"
    scalar_impl += f"        return {double_arg_np_map[func]}(elem0, elem1)"
    scalar_vars = {}
    exec(scalar_impl, {"np": np, "pd": pd}, scalar_vars)

    numeric_func_answer = vectorized_sol((arr0, arr1), scalar_vars["impl"], np.float64)
    check_func(
        impl_vars["impl"],
        (arr0, arr1),
        py_output=numeric_func_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize("func", double_arg_np_list)
def test_numeric_double_arg_option(func):
    test_impl = "def impl(a, b, flag0, flag1):\n"
    test_impl += "  arg0 = a if flag0 else None\n"
    test_impl += "  arg1 = b if flag1 else None\n"
    test_impl += f"  return bodo.libs.bodosql_array_kernels.{func}(arg0, arg1)"
    impl_vars = {}
    exec(test_impl, {"bodo": bodo}, impl_vars)

    for flag0 in [True, False]:
        for flag1 in [True, False]:
            answer = (
                eval(f"{double_arg_np_map[func]}(4.555, 2)")
                if flag0 and flag1
                else None
            )
            check_func(impl_vars["impl"], (4.555, 2, flag0, flag1), py_output=answer)


@pytest.mark.slow
def test_width_bucket_option():
    def impl(a, b, c, d, flag0, flag1, flag2, flag3):
        arg0 = a if flag0 else None
        arg1 = b if flag1 else None
        arg2 = c if flag2 else None
        arg3 = d if flag3 else None
        return bodo.libs.bodosql_array_kernels.width_bucket(arg0, arg1, arg2, arg3)

    for flag0 in [True, False]:
        for flag1 in [True, False]:
            for flag2 in [True, False]:
                for flag3 in [True, False]:
                    check_func(
                        impl,
                        (
                            7,
                            0,
                            10,
                            3,
                            flag0,
                            flag1,
                            flag2,
                            flag3,
                        ),
                        py_output=3 if flag0 and flag1 and flag2 and flag3 else None,
                    )
