# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of SQL bitwise functions on BodoSQL
"""

import pandas as pd
import pytest
from bodosql.tests.utils import check_query


@pytest.fixture
def bitwise_df():
    return {
        "table1": pd.DataFrame(
            {
                "A": pd.Series(
                    [0, 1, 42, 2147483647, -2147483648, None], dtype=pd.Int32Dtype()
                ),
                "B": pd.Series(
                    [42, 1, 100, -2147483648, 2147483647, 16], dtype=pd.Int32Dtype()
                ),
                "C": pd.Series([0, 1, 127, 128, 255, 13], dtype=pd.UInt8Dtype()),
                "D": pd.Series([100, 42, 255, 128, 200, None], dtype=pd.UInt8Dtype()),
                "E": pd.Series([127, -128, -1, -127, 42, None], dtype=pd.Int8Dtype()),
                "F": pd.Series(
                    [18446744073709551615, 127, 42, 128, 13, None],
                    dtype=pd.UInt64Dtype(),
                ),
            }
        )
    }


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                "SELECT BITAND(A, B) FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [0, 1, 32, 0, 0, None],
                            dtype=pd.Int32Dtype(),
                        )
                    }
                ),
            ),
            id="all_vector_int32",
        ),
        pytest.param(
            (
                "SELECT BITAND(C, D) FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [0, 0, 127, 128, 200, None],
                            dtype=pd.Int32Dtype(),
                        )
                    }
                ),
            ),
            id="all_vector_uint8",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "SELECT BITAND(E, F) FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [127, 0, 42, 128, 8, None],
                            dtype=pd.Int64Dtype(),
                        )
                    }
                ),
            ),
            id="all_vector_mixed",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                # 6148914691236517205 = 0x5555555555555555
                "SELECT CASE WHEN F IS NULL THEN -1 ELSE BITAND(F, 6148914691236517205) END FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [6148914691236517205, 85, 0, 0, 5, -1],
                            dtype=pd.Int64Dtype(),
                        )
                    }
                ),
            ),
            id="vector_scalar_case",
        ),
    ],
)
def test_bitand(args, bitwise_df, spark_info, memory_leak_check):
    query, answer = args
    check_query(
        query,
        bitwise_df,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=answer,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                "SELECT BITOR(A, B) FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [42, 1, 110, -1, -1, None],
                            dtype=pd.Int32Dtype(),
                        )
                    }
                ),
            ),
            id="all_vector_int32",
        ),
        pytest.param(
            (
                "SELECT BITOR(C, D) FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [100, 43, 255, 128, 255, None],
                            dtype=pd.Int32Dtype(),
                        )
                    }
                ),
            ),
            id="all_vector_uint8",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "SELECT BITOR(E, F) FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [-1, -1, -1, -127, 47, None],
                            dtype=pd.Int64Dtype(),
                        )
                    }
                ),
            ),
            id="all_vector_mixed",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                # 6148914691236517205 = 0x5555555555555555
                "SELECT CASE WHEN F IS NULL THEN -1 ELSE BITOR(F, 6148914691236517205) END FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [
                                -1,
                                6148914691236517247,
                                6148914691236517247,
                                6148914691236517333,
                                6148914691236517213,
                                -1,
                            ],
                            dtype=pd.Int64Dtype(),
                        )
                    }
                ),
            ),
            id="vector_scalar_case",
        ),
    ],
)
def test_bitor(args, bitwise_df, spark_info, memory_leak_check):
    query, answer = args
    check_query(
        query,
        bitwise_df,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=answer,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                "SELECT BITXOR(A, B) FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [42, 0, 78, -1, -1, None],
                            dtype=pd.Int32Dtype(),
                        )
                    }
                ),
            ),
            id="all_vector_int32",
        ),
        pytest.param(
            (
                "SELECT BITXOR(C, D) FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [100, 43, 128, 0, 55, None],
                            dtype=pd.Int32Dtype(),
                        )
                    }
                ),
            ),
            id="all_vector_uint8",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "SELECT BITXOR(E, F) FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [-128, -1, -43, -255, 39, None],
                            dtype=pd.Int64Dtype(),
                        )
                    }
                ),
            ),
            id="all_vector_mixed",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                # 6148914691236517205 = 0x5555555555555555
                "SELECT CASE WHEN F IS NULL THEN -1 ELSE BITXOR(F, 6148914691236517205) END FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [
                                -6148914691236517206,
                                6148914691236517162,
                                6148914691236517247,
                                6148914691236517333,
                                6148914691236517208,
                                -1,
                            ],
                            dtype=pd.Int64Dtype(),
                        )
                    }
                ),
            ),
            id="vector_scalar_case",
        ),
    ],
)
def test_bitxor(args, bitwise_df, spark_info, memory_leak_check):
    query, answer = args
    check_query(
        query,
        bitwise_df,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=answer,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                "SELECT BITNOT(A) FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [-1, -2, -43, -2147483648, 2147483647, None],
                            dtype=pd.Int32Dtype(),
                        )
                    }
                ),
            ),
            id="vector_int32",
        ),
        pytest.param(
            (
                "SELECT BITNOT(C) FROM table1",
                pd.DataFrame({0: pd.Series([255, 254, 128, 127, 0, 242])}),
            ),
            id="vector_uint8",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "SELECT BITNOT(E) FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [-128, 127, 0, 126, -43, None],
                            dtype=pd.Int32Dtype(),
                        )
                    }
                ),
            ),
            id="vector_int8",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "SELECT CASE WHEN F IS NULL THEN 0 ELSE BITNOT(F) END FROM table1",
                pd.DataFrame(
                    {
                        # NOTE: Calcite makes the result signed int64 and doesn't really
                        # support unsigned int properly. See [BE-3419].
                        0: pd.Series(
                            [
                                -9223372036854775808,
                                -9223372036854775808,
                                -9223372036854775808,
                                -9223372036854775808,
                                0,
                                0,
                            ],
                        )
                    }
                ),
            ),
            id="case_uint64",
        ),
    ],
)
def test_bitnot(args, bitwise_df, spark_info, memory_leak_check):
    query, answer = args
    check_query(
        query,
        bitwise_df,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=answer,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                "SELECT BITSHIFTLEFT(A, 1) FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [0, 2, 84, 4294967294, -4294967296, None],
                            dtype=pd.Int64Dtype(),
                        )
                    }
                ),
            ),
            id="vector_scalar_int32",
        ),
        pytest.param(
            (
                "SELECT BITSHIFTLEFT(C, 4) FROM table1",
                pd.DataFrame({0: pd.Series([0, 16, 2032, 2048, 4080, 208])}),
            ),
            id="vector_scalar_uint8",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "SELECT CASE WHEN F IS NULL THEN 0 ELSE BITSHIFTLEFT(F, 7) END FROM table1",
                pd.DataFrame(
                    {
                        # NOTE: Calcite makes the result signed int64 and doesn't really
                        # support unsigned int properly. See [BE-3419].
                        0: pd.Series(
                            [-9223372036854775808, 0, 1664, 5376, 16256, 16384],
                        )
                    }
                ),
            ),
            id="vector_scalar_uint64_case",
        ),
    ],
)
def test_bitshiftleft(args, bitwise_df, spark_info, memory_leak_check):
    query, answer = args
    check_query(
        query,
        bitwise_df,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=answer,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                "SELECT BITSHIFTRIGHT(A, 1) FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [0, 0, 21, 1073741823, -1073741824, None],
                            dtype=pd.Int32Dtype(),
                        )
                    }
                ),
            ),
            id="vector_scalar_int32",
        ),
        pytest.param(
            (
                "SELECT BITSHIFTRIGHT(C, 4) FROM table1",
                pd.DataFrame({0: pd.Series([0, 0, 7, 8, 15, 0])}),
            ),
            id="vector_scalar_uint8",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "SELECT CASE WHEN F IS NULL THEN 0 ELSE BITSHIFTRIGHT(F, 7) END FROM table1",
                pd.DataFrame({0: pd.Series([144115188075855871, 0, 0, 1, 0, 0])}),
            ),
            id="vector_scalar_uint64_case",
        ),
    ],
)
def test_bitshiftright(args, bitwise_df, spark_info, memory_leak_check):
    query, answer = args
    check_query(
        query,
        bitwise_df,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=answer,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                "SELECT GETBIT(A, B % 32) FROM table1",
                pd.DataFrame(
                    {
                        0: pd.Series(
                            [0, 0, 0, 1, 1, None],
                            dtype=pd.UInt8Dtype(),
                        )
                    }
                ),
            ),
            id="vector_vector_int32",
        ),
        pytest.param(
            (
                "SELECT GETBIT(C, 0) FROM table1",
                pd.DataFrame({0: pd.Series([0, 1, 1, 0, 1, 1])}),
            ),
            id="vector_scalar_uint8",
        ),
        pytest.param(
            (
                "SELECT CASE WHEN F IS NULL THEN 2 ELSE GETBIT(F, 5) END FROM table1",
                pd.DataFrame({0: pd.Series([1, 1, 1, 0, 0, 2])}),
            ),
            id="vector_scalar_uint64_case",
        ),
    ],
)
def test_getbit(args, bitwise_df, spark_info, memory_leak_check):
    query, answer = args
    check_query(
        query,
        bitwise_df,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=answer,
    )
