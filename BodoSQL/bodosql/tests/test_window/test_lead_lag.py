import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bodo
from bodo.tests.utils import pytest_slow_unless_window, temp_config_override
from bodo.types import Time, TimestampTZ
from bodosql.tests.test_window.window_common import count_window_applies
from bodosql.tests.utils import check_query

# Skip unless any window-related files were changed
pytestmark = pytest_slow_unless_window


def test_lead_lag_mixed(spark_info, memory_leak_check):
    """Tests the window functions LEAD/LAG with a minimal set of combinations
    that cover the essential of behavior."""
    selects = [
        "LEAD(I) RESPECT NULLS OVER (PARTITION BY P ORDER BY O)",
        "LAG(S, 10) RESPECT NULLS OVER (PARTITION BY P ORDER BY O)",
        "LEAD(I, 5, 0) RESPECT NULLS OVER (PARTITION BY P ORDER BY O)",
        "LEAD(S, 3) IGNORE NULLS OVER (PARTITION BY P ORDER BY O)",
        "LAG(I, 7, -1) IGNORE NULLS OVER (PARTITION BY P ORDER BY O)",
        "LEAD(S, 1, 'hello') IGNORE NULLS OVER (PARTITION BY P ORDER BY O)",
    ]
    query = f"SELECT I, P, O, {', '.join(selects)} FROM table1"
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "I": pd.Series(
                    [None if (i**2) % 6 < 2 else i for i in range(500)],
                    dtype=pd.Int32Dtype(),
                ),
                "S": pd.Series(
                    [None if (i**3) % 17 > 10 else str(i) for i in range(500)]
                ),
                "P": pd.Series(["A", "B", "C", "D", "E"] * 100),
                "O": pd.Series([np.tan(i) for i in range(500)]),
            }
        )
    }
    pandas_code = check_query(
        query,
        ctx,
        spark_info,
        sort_output=True,
        check_dtype=False,
        check_names=False,
        return_codegen=True,
    )["pandas_code"]

    # Verify that fusion is working correctly so only one closure is produced
    count_window_applies(pandas_code, 1, ["LEAD", "LAG"])


@pytest.mark.skipif(
    bodo.tests.utils.test_spawn_mode_enabled,
    reason="capfd doesn't work for spawn",
)
@pytest.mark.parametrize(
    "func, shift_amt",
    [
        pytest.param("LEAD", 3, id="lead_3"),
        pytest.param("LAG", 3, id="lag_3"),
        pytest.param("LEAD", 57, id="lead_57", marks=pytest.mark.slow),
        pytest.param("LAG", 57, id="lag_57", marks=pytest.mark.slow),
        pytest.param("LEAD", 9999, id="lead_large_shift", marks=pytest.mark.slow),
        pytest.param("LAG", 9999, id="lag_large_shift", marks=pytest.mark.slow),
        pytest.param("LEAD", -2, id="lead_negative", marks=pytest.mark.slow),
    ],
)
def test_lead_lag_shift(func, shift_amt, spark_info, capfd):
    """
    Test different values for lead/lag with a pass through column,
    not keeping the input
    """
    from mpi4py import MPI

    from bodo.tests.utils import temp_env_override

    df = pd.DataFrame(
        {
            "C": np.arange(1000),
            "D": ["shoes", "pants", "shirt", "jacket", "tie"] * 200,
            "B": np.arange(1000),
            "A": ["A"] * 1 + ["B"] * 17 + ["C"] * 255 + ["D"] * 727,
        }
    )

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(df))
    df = df.iloc[perm, :]

    # default is NULL
    query = f"SELECT B, {func}(C, {shift_amt}) OVER(PARTITION BY A ORDER BY B ASC NULLS LAST), D FROM TABLE1"
    expected_log_message = "[DEBUG] GroupbyState::FinalizeBuild:"

    with temp_env_override(
        {
            "BODO_DEBUG_STREAM_GROUPBY_PARTITIONING": "1",
        }
    ):
        check_query(
            query,
            {"TABLE1": df},
            spark_info,
            check_names=False,
            sort_output=True,
            check_dtype=False,
        )

    comm = MPI.COMM_WORLD
    _, err = capfd.readouterr()
    assert_success = expected_log_message in err
    assert_success = comm.allreduce(assert_success, op=MPI.LAND)

    assert assert_success


@pytest.mark.parametrize(
    "use_default",
    [
        pytest.param(False, id="no_default"),
        pytest.param(True, id="use_default"),
    ],
)
@pytest.mark.parametrize(
    "input_arr, default, default_str",
    [
        pytest.param(
            pd.array(
                [datetime.date(2020, i % 12 + 1, i % 28 + 1) for i in range(1000)]
            ),
            datetime.date(2000, 1, 1),
            "'2000-01-01' :: DATE",
            id="date",
        ),
        pytest.param(
            pd.array(list(range(1000)), dtype=pd.Int32Dtype()), -1, -1, id="int"
        ),
        pytest.param(
            pd.array(
                [
                    pd.Timestamp(
                        f"200{i % 10}-{i % 12 + 1}-{i % 28 + 1} {(i + 6) % 24}:{(i + 16) % 60}:{(i + 3) % 60}"
                    )
                    for i in range(1000)
                ],
                dtype="datetime64[ns]",
            ),
            pd.Timestamp("2000-01-01 00:00:00"),
            "'2000-01-01 00:00:00' :: TIMESTAMP_NTZ",
            id="timestamp_ntz",
            marks=pytest.mark.skip("Fix for Pandas 3"),
        ),
        pytest.param(
            pd.array(
                [
                    TimestampTZ(
                        pd.Timestamp(
                            f"200{i % 10}-{i % 12 + 1}-{i % 28 + 1} {(i + 6) % 24}:{(i + 16) % 60}:{(i + 3) % 60}"
                        ),
                        i,
                    )
                    for i in range(1000)
                ]
            ),
            TimestampTZ(pd.Timestamp("2000-01-01 11:00:00"), 60),
            "'2000-01-01 12:00:00+0100'::TIMESTAMP WITH TIME ZONE",
            id="timestamptz",
        ),
        pytest.param(
            pd.array(
                [
                    pd.Timestamp(
                        f"200{i % 10}-{i % 12 + 1}-{i % 28 + 1} {(i + 6) % 24}:{(i + 16) % 60}:{(i + 3) % 60}",
                        tz="US/Pacific",
                    )
                    for i in range(1000)
                ],
                dtype="datetime64[ns, US/Pacific]",
            ),
            pd.Timestamp("2000-01-01 00:00:00", tz="US/Pacific"),
            "'2000-01-01 00:00:00' :: TIMESTAMP_LTZ",
            id="timestamp_ltz",
        ),
        pytest.param(
            pd.array(
                [Time(hour=i % 100, minute=i % 60, second=i % 60) for i in range(1000)]
            ),
            Time(hour=12, minute=0, second=0),
            "'12:00:00' :: Time(9)",
            id="time",
        ),
        pytest.param(
            pd.array([bytes(i) for i in range(1000)]),
            b"\xc0\xff\xee",
            "X'C0FFEE'",
            id="binary",
        ),
        pytest.param(
            pd.array([f"{i}{i + 1}{i + 3}" for i in range(1000)]),
            "hello",
            "'hello'",
            id="string",
        ),
        pytest.param(
            pd.array(
                [Decimal(str(f"{i + 1}{i + 2}{i + 3}")) for i in range(1000)],
                dtype=pd.ArrowDtype(pa.decimal128(38, 0)),
            ),
            Decimal("-1"),
            "-1::NUMBER(38,0)",
            id="decimal",
        ),
    ],
)
def test_lead_lag_defaults(input_arr, default, default_str, use_default):
    """
    Tests that lead/lag works with different literal types
    """

    shift_amt = 10

    if not use_default:
        default = None

    partition_col = ["A"] * 1 + ["B"] * 17 + ["C"] * 255 + ["D"] * 727
    order_col = np.arange(1000)

    in_df = pd.DataFrame({"A": partition_col, "B": order_col, "C": input_arr})

    output_col = pd.array(
        [
            (
                default
                if i + shift_amt >= len(partition_col)
                or partition_col[i] != partition_col[i + shift_amt]
                else input_arr[i + shift_amt]
            )
            for i in range(len(partition_col))
        ],
    )

    out_df = pd.DataFrame({"OUT": output_col})

    query = (
        f"SELECT LEAD(C,10, {default_str}) OVER (PARTITION BY A ORDER BY B) FROM TABLE1"
    )
    query_no_default = "SELECT LEAD(C,10) OVER (PARTITION BY A ORDER BY B) FROM TABLE1"

    with temp_config_override("bodo_use_decimal", True):
        check_query(
            query if use_default else query_no_default,
            {"TABLE1": in_df},
            None,
            expected_output=out_df,
            check_names=False,
            check_dtype=False,
            sort_output=True,
            session_tz="US/Pacific",
            enable_timestamp_tz=True,
        )


def test_lead_lag_multiple(spark_info, memory_leak_check):
    """Tests multiple lead/lag computations alongside other functions that can be done together."""
    window = "OVER (PARTITION BY P ORDER BY O)"
    query = f"SELECT T, I, LEAD(S, 12, '') {window}, ROW_NUMBER() {window}, LAG(I) {window}, LEAD(T, 1, 'foobar') {window} FROM TABLE1"
    n_rows = 8000
    df = pd.DataFrame(
        {
            "P": [int(np.tan(i)) for i in range(n_rows)],
            "O": [np.tan(i) for i in range(n_rows)],
            "I": range(n_rows),
            "S": [None if i % 7 == 1 else str(i)[2:] for i in range(n_rows)],
            "T": [None if (i % 6) == (i % 7) else hex(i % 500) for i in range(n_rows)],
        }
    )
    check_query(
        query,
        {"TABLE1": df},
        spark_info,
        check_names=False,
        check_dtype=False,
        only_jit_1DVar=True,
    )
