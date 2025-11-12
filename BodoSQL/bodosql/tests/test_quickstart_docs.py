import numpy as np
import pandas as pd
import pytest

from bodo.tests.utils import pytest_mark_one_rank, temp_env_override
from bodo.utils.testing import ensure_clean2

pytestmark = [pytest.mark.test_docs]


@pytest_mark_one_rank
def test_quickstart_local_sql():
    """Test example equivalent to docs/quick_start/quickstart_local_sql.md"""
    import bodosql.compiler  # isort:skip # noqa

    NUM_GROUPS = 30
    NUM_ROWS = 2_000

    df = pd.DataFrame({"A": np.arange(NUM_ROWS) % NUM_GROUPS, "B": np.arange(NUM_ROWS)})

    output_df_path = "my_data.pq"
    with ensure_clean2(output_df_path):
        df.to_parquet(output_df_path)

        with temp_env_override({"BODO_SPAWN_MODE": "1"}):
            bc = bodosql.BodoSQLContext(
                {"TABLE1": bodosql.TablePath(output_df_path, "parquet")}
            )

            out_df = bc.sql("SELECT SUM(A) as SUM_OF_COLUMN_A FROM TABLE1 WHERE B > 4")

            answer_df = pd.DataFrame({"SUM_OF_COLUMN_A": [28890]})
            pd.testing.assert_frame_equal(answer_df, out_df, check_dtype=False)
