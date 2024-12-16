"""
Test correctness of SQL conditional functions on BodoSQL
"""

import pandas as pd
import pytest

import bodo
from bodo.tests.utils import temp_config_override
from bodosql.tests.utils import check_query


@pytest.mark.skipif(bodo.get_size() != 1, reason="skip on multiple ranks")
def test_spark_name_matching_invalid(memory_leak_check):
    """
    Tests a query that will fail to match even though the column name
    is used because we are not using the spark protocol.
    """
    query = "SELECT GAMMA, ALPHA FROM TABLE1"
    df = pd.DataFrame(
        {
            "Alpha": [1, 2, 3, 4, 5],
            "Beta": [10, 20, 30, 40, 50],
            "Gamma": [100, 200, 300, 400, 500],
        }
    )

    with pytest.raises(
        bodo.utils.typing.BodoError,
        match=r"Object 'TABLE1' not found within '__BODOLOCAL__'; did you mean 'table1'?",
    ):
        with temp_config_override("bodo_sql_style", "SNOWFLAKE"):
            check_query(
                query,
                {"table1": df},
                None,
                check_names=False,
                check_dtype=False,
                expected_output=df.loc[:, ["Gamma", "Alpha"]],
            )


def test_spark_name_matching_valid(memory_leak_check):
    """
    Same setup as test_spark_name_matching_invalid but with a correct
    match since we are using the spark protocol.
    """
    query = "SELECT GAMMA, ALPHA FROM TABLE1"
    df = pd.DataFrame(
        {
            "Alpha": [1, 2, 3, 4, 5],
            "Beta": [10, 20, 30, 40, 50],
            "Gamma": [100, 200, 300, 400, 500],
        }
    )
    with temp_config_override("bodo_sql_style", "SPARK"):
        check_query(
            query,
            {"table1": df},
            None,
            check_names=False,
            check_dtype=False,
            expected_output=df.loc[:, ["Gamma", "Alpha"]],
        )
