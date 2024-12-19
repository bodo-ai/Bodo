import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import pytest_spawn_mode
from bodo.utils.testing import ensure_clean2

pytestmark = pytest_spawn_mode + [pytest.mark.test_docs]


def test_quickstart_local_python():
    """Runs example equivalent to code from top-level README.md
    and docs/quick_start/quickstart_local_python.md
    """
    # Generate sample data
    NUM_GROUPS = 30
    NUM_ROWS = 2_000

    df = pd.DataFrame({"A": np.arange(NUM_ROWS) % NUM_GROUPS, "B": np.arange(NUM_ROWS)})

    input_df_path = "my_data.pq"
    output_df_path = "out.pq"

    with ensure_clean2(input_df_path):
        df.to_parquet(input_df_path)

        def computation():
            df = pd.read_parquet(input_df_path)
            df2 = pd.DataFrame(
                {"A": df.apply(lambda r: 0 if r.A == 0 else (r.B // r.A), axis=1)}
            )
            df2.to_parquet(output_df_path)

        with ensure_clean2(output_df_path):
            bodo.jit(cache=True, spawn=True)(computation)()
            bodo_out = pd.read_parquet(output_df_path)

        with ensure_clean2(output_df_path):
            computation()
            pandas_out = pd.read_parquet(output_df_path)

        pd.testing.assert_frame_equal(bodo_out, pandas_out)


@pytest.mark.iceberg
def test_quickstart_local_iceberg():
    """Test that the example in docs/quick_start/quickstart_local_iceberg.md"""
    NUM_GROUPS = 30
    NUM_ROWS = 2_000

    db_name = "MY_DATABASE"
    input_df = pd.DataFrame(
        {"A": np.arange(NUM_ROWS) % NUM_GROUPS, "B": np.arange(NUM_ROWS)}
    )

    with ensure_clean2(db_name):

        @bodo.jit(spawn=True)
        def example_write_iceberg_table(df):
            df.to_sql(
                name="MY_TABLE",
                con=f"iceberg://{db_name}",
                schema="MY_SCHEMA",
                if_exists="replace",
            )

        example_write_iceberg_table(input_df)

        @bodo.jit(spawn=True)
        def example_read_iceberg():
            df = pd.read_sql_table(
                table_name="MY_TABLE", con=f"iceberg://{db_name}", schema="MY_SCHEMA"
            )
            return df

        out_df = example_read_iceberg()
        pd.testing.assert_frame_equal(out_df, input_df)
