import numpy as np
import pandas as pd
import pytest

import bodosql
from bodo.tests.utils import temp_env_override
from bodo.utils.testing import ensure_clean2

pytestmark = pytest.mark.test_docs


def test_quickstart_local_sql():
    """Test example equivalent to docs/quick_start/quickstart_local_sql.md"""
    NUM_GROUPS = 30
    NUM_ROWS = 20_000_000

    df = pd.DataFrame({"A": np.arange(NUM_ROWS) % NUM_GROUPS, "B": np.arange(NUM_ROWS)})

    output_df_path = "my_data.pq"
    with ensure_clean2(output_df_path):
        df.to_parquet(output_df_path)

        with temp_env_override({"BODO_SPAWN_MODE": "1"}):
            bc = bodosql.BodoSQLContext(
                {"TABLE1": bodosql.TablePath(output_df_path, "parquet")}
            )

            bc.sql("SELECT SUM(A) as SUM_OF_COLUMN_A FROM TABLE1 WHERE B > 4")
