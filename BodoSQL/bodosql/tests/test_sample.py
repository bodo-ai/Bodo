"""
Test file for tests related to the SAMPLE/TABLESAMPLE operator.
"""

import numpy as np
import pandas as pd
import pytest

import bodo
import bodosql
from bodo.utils.typing import BodoError
from bodosql.tests.utils import check_query


@pytest.fixture(
    params=[
        pytest.param("SAMPLE", id="sample_op"),
        pytest.param("TABLESAMPLE", id="tablesample", marks=pytest.mark.slow),
    ]
)
def sample_cmds(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param("", id="default"),
        pytest.param("BERNOULLI", id="bernoulli", marks=pytest.mark.slow),
        pytest.param("ROW", id="row", marks=pytest.mark.slow),
        pytest.param(
            "SYSTEM",
            id="system",
            marks=pytest.mark.skip("[BE-XXXX] Block sampling not supported"),
        ),
        pytest.param(
            "BLOCK",
            id="block",
            marks=pytest.mark.skip("[BE-XXXX] Block sampling not supported"),
        ),
    ]
)
def sample_methods(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param("SEED", id="seed"),
        pytest.param("REPEATABLE", id="repeatable", marks=pytest.mark.slow),
    ]
)
def sample_seeds(request):
    return request.param


@pytest.mark.parametrize(
    "df, py_output, frac",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "A": np.zeros((50,), dtype=np.int64),
                    "B": np.zeros((50,), dtype=np.int64),
                }
            ),
            pd.DataFrame(
                {
                    "A": np.zeros((25,), dtype=np.int64),
                    "B": np.zeros((25,), dtype=np.int64),
                }
            ),
            50,
            id="int50",
        ),
    ],
)
def test_sample_frac_numeric_cols(
    df, py_output, frac, sample_cmds, sample_methods, memory_leak_check
):
    """
    Test that SAMPLE (frac) works for numeric columns
    """
    ctx = {"TABLE1": df}
    query = f"SELECT A, B FROM table1 {sample_cmds} {sample_methods}({frac})"
    check_query(query, ctx, None, expected_output=py_output, check_names=False)


@pytest.mark.parametrize(
    "df, py_output, nrows",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "A": np.array(["A"] * 30, dtype=object),
                    "B": np.array(["AA"] * 30, dtype=object),
                }
            ),
            pd.DataFrame(
                {
                    "A": np.array(["A"] * 10, dtype=object),
                    "B": np.array(["AA"] * 10, dtype=object),
                }
            ),
            10,
            id="str33",
        ),
    ],
)
def test_sample_rows_string_cols(
    df, py_output, nrows, sample_cmds, sample_methods, memory_leak_check
):
    """
    Test that SAMPLE (rows ROWS) works for string columns
    """
    query = f"SELECT A, B FROM table1 {sample_cmds} {sample_methods}({nrows} ROWS)"

    if sample_methods in ("SYSTEM", "BLOCK"):
        # SYSTEM | BLOCK are not supported for fixed-size sampling:
        # https://docs.snowflake.com/en/sql-reference/constructs/sample
        def impl(df, query):
            bc = bodosql.BodoSQLContext({"TABLE1": df})
            return bc.sql(query)

        with pytest.raises(
            BodoError,
            match=r".*SYSTEM | BLOCK are not supported for fixed-size sampling",
        ):
            impl(df, query)

    else:
        ctx = {"TABLE1": df}
        check_query(query, ctx, None, expected_output=py_output, check_names=False)


@pytest.mark.parametrize(
    "frac, seed",
    [
        pytest.param(1, 0, id="frac01"),
        pytest.param(25, 2500, id="frac25"),
        pytest.param(99, 1234567890, id="frac99"),
    ],
)
def test_sample_frac_numeric_seed(
    frac, seed, sample_cmds, sample_seeds, memory_leak_check
):
    """
    Test that SAMPLE (frac) matches `df.sample` outputs from bodo.jit
    """
    query = f"SELECT A FROM table1 {sample_cmds} ({frac}) {sample_seeds} ({seed})"
    df = pd.DataFrame({"A": np.arange(5000, dtype=np.int64)})
    ctx = {"TABLE1": df}

    @bodo.jit
    def sample_impl(df):
        return df.sample(frac=frac / 100, random_state=seed)

    py_output = sample_impl(df)
    check_query(query, ctx, None, expected_output=py_output)
