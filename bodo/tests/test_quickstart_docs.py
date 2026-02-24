"""Tests code snippets from Top level readme/quickstart and dev guides run as expected.
TODO [BSE-4150]: Add proper caching check for spawn mode.
"""

import os

import numpy as np
import pandas as pd
import pytest
from numba.core.errors import TypingError  # noqa TID253

import bodo
from bodo.tests.utils import (
    _test_equal,
    check_func,
    pytest_mark_spawn_mode,
    temp_env_override,
)
from bodo.utils.testing import ensure_clean2

pytestmark = [pytest.mark.test_docs]


@pytest.mark.df_lib
@pytest.mark.skipif(
    os.getenv("BODO_ENABLE_TEST_DATAFRAME_LIBRARY", "0") == "0",
    reason="BODO_ENABLE_TEST_DATAFRAME_LIBRARY is not set, this is required for df_lib tests",
)
def test_quickstart_local_python_df():
    """Runs example equivalent to Bodo DF Library code from top-level README.md
    and docs/quick_start/quickstart_local_python.md and ensures
    that it is consistent with pandas.
    """
    # Generate sample data
    NUM_GROUPS = 30
    NUM_ROWS = 2_000
    output_path = "my_data.pq"

    df = pd.DataFrame({"A": np.arange(NUM_ROWS) % NUM_GROUPS, "B": np.arange(NUM_ROWS)})
    pandas_df = df.groupby("A", as_index=False)["B"].max()

    with ensure_clean2(output_path):
        pandas_df.to_parquet(output_path)
        pandas_out = pd.read_parquet(output_path, dtype_backend="pyarrow")

    def bodo_groupby_write():
        import bodo.pandas as pd

        df = pd.DataFrame(
            {"A": np.arange(NUM_ROWS) % NUM_GROUPS, "B": np.arange(NUM_ROWS)}
        )
        df2 = df.groupby("A", as_index=False)["B"].max()
        df2.to_parquet(output_path)

    with ensure_clean2(output_path):
        bodo_groupby_write()
        bodo_out = pd.read_parquet(output_path, dtype_backend="pyarrow")

    pandas_out = pandas_out.sort_values("A").reset_index(drop=True)
    bodo_out = bodo_out.sort_values("A").reset_index(drop=True)

    _test_equal(bodo_out, pandas_out, check_dtype=False)


@pytest_mark_spawn_mode
def test_quickstart_local_python_jit():
    """Runs example equivalent to Bodo jit code from top-level README.md
    and docs/quick_start/quickstart_local_python.md and ensures
    that it is consistent with pandas.
    """
    # Generate sample data
    NUM_GROUPS = 30
    NUM_ROWS = 2_000

    df = pd.DataFrame({"A": np.arange(NUM_ROWS) % NUM_GROUPS, "B": np.arange(NUM_ROWS)})

    output_df_path = "my_data.pq"

    def computation(df):
        return df.apply(lambda r: 0 if r.A == 0 else (r.B // r.A), axis=1)

    with ensure_clean2(output_df_path):
        S = bodo.jit(cache=True, spawn=True)(computation)(df)
        pd.DataFrame({"C": S}).to_parquet(output_df_path)
        bodo_out = pd.read_parquet(output_df_path, dtype_backend="pyarrow")

    with ensure_clean2(output_df_path):
        S = computation(df)
        pd.DataFrame({"C": S}).to_parquet(output_df_path)
        pandas_out = pd.read_parquet(output_df_path, dtype_backend="pyarrow")

    _test_equal(bodo_out, pandas_out)


@pytest.mark.iceberg
@pytest.mark.df_lib
@pytest.mark.skipif(
    os.getenv("BODO_ENABLE_TEST_DATAFRAME_LIBRARY", "0") == "0",
    reason="BODO_ENABLE_TEST_DATAFRAME_LIBRARY is not set, this is required for df_lib tests",
)
def test_quickstart_local_iceberg_df():
    """Test the Bodo DF Library example in docs/quick_start/quickstart_local_iceberg.md"""
    import bodo.pandas as pd

    NUM_GROUPS = 30
    NUM_ROWS = 2_000

    df = pd.DataFrame({"A": np.arange(NUM_ROWS) % NUM_GROUPS, "B": np.arange(NUM_ROWS)})
    df.to_iceberg("test_table", location="./iceberg_warehouse")

    out_df = pd.read_iceberg("test_table", location="./iceberg_warehouse")
    _test_equal(out_df, df)


@pytest.mark.iceberg
@pytest_mark_spawn_mode
def test_quickstart_local_iceberg_jit():
    """Test the Bodo jit example in docs/quick_start/quickstart_local_iceberg.md"""
    NUM_GROUPS = 30
    NUM_ROWS = 2_000

    db_name = "/tmp/MY_DATABASE"
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
        _test_equal(out_df, input_df)


@pytest.fixture(scope="module")
def devguide_df_path():
    """Writes to parquet file used by multiple examples in docs/quick_start/devguide.md"""
    df = pd.DataFrame(
        {
            "A": np.repeat(pd.date_range("2013-01-03", periods=1000, unit="ns"), 1),
            "B": np.arange(1_000),
        }
    )
    df.iloc[np.arange(10) * 3, 0] = pd.NA
    out_path = "pd_example.pq"
    df.to_parquet("pd_example.pq", row_group_size=100)

    yield out_path

    if os.path.exists(out_path):
        os.remove(out_path)


@pytest_mark_spawn_mode
def test_devguide_transform(devguide_df_path):
    """Test transform example from docs/quick_start/devguide.md and
    ensures behavior is consistent with pandas.
    """
    output_df_path = "output_df.pq"

    def data_transform(devguide_df_path):
        df = pd.read_parquet(devguide_df_path, dtype_backend="pyarrow")
        df["B"] = df.apply(
            lambda r: "NA" if pd.isna(r.A) else "P1" if r.A.month < 5 else "P2", axis=1
        )
        df["C"] = df.A.dt.month
        df.to_parquet(output_df_path)

    # BODO_NUM_WORKERS=1 python bodo_data_transform.py
    with temp_env_override({"BODO_NUM_WORKERS": "1"}):
        with ensure_clean2(output_df_path):
            bodo.jit(cache=True, spawn=True)(data_transform)(devguide_df_path)
            bodo_out = pd.read_parquet(output_df_path, dtype_backend="pyarrow")

    with ensure_clean2(output_df_path):
        data_transform(devguide_df_path)
        pandas_out = pd.read_parquet(output_df_path, dtype_backend="pyarrow")

    bodo_out["A"] = bodo_out["A"].astype("datetime64[ns]")
    _test_equal(bodo_out, pandas_out, check_dtype=False)


@pytest_mark_spawn_mode
def test_devguide_parallel1(devguide_df_path):
    def load_data_bodo(devguide_df_path):
        df = pd.read_parquet(devguide_df_path, dtype_backend="pyarrow")
        return df

    # BODO_NUM_WORKERS=2 python load_data.py
    with temp_env_override({"BODO_NUM_WORKERS": "2"}):
        bodo_out = bodo.jit(spawn=True)(load_data_bodo)(devguide_df_path)
        pandas_out = load_data_bodo(devguide_df_path)

    _test_equal(bodo_out, pandas_out)


@pytest_mark_spawn_mode
def test_devguide_parallel2(devguide_df_path):
    output_df_path = "output_df.pq"

    def data_groupby(devguide_df_path):
        df = pd.read_parquet(devguide_df_path, dtype_backend="pyarrow")
        df2 = df.groupby("A", as_index=False).sum()
        df2.to_parquet(output_df_path)

    # BODO_NUM_WORKERS=8 python data_groupby.py
    with temp_env_override({"BODO_NUM_WORKERS": "8"}):
        with ensure_clean2(output_df_path):
            bodo.jit(cache=True, spawn=True)(data_groupby)(devguide_df_path)
            bodo_out = pd.read_parquet(output_df_path, dtype_backend="pyarrow")

    with ensure_clean2(output_df_path):
        data_groupby(devguide_df_path)
        pandas_out = pd.read_parquet(output_df_path, dtype_backend="pyarrow")

    bodo_out = bodo_out.sort_values("A").reset_index(drop=True)
    pandas_out = pandas_out.sort_values("A").reset_index(drop=True)

    _test_equal(bodo_out, pandas_out, check_dtype=False)


@pytest_mark_spawn_mode
def test_devguide_type_error(devguide_df_path):
    from bodo.utils.typing import BodoError

    @bodo.jit(spawn=True)
    def groupby_keys(devguide_df_path, extra_keys):
        df = pd.read_parquet(devguide_df_path, dtype_backend="pyarrow")
        keys = [c for c in df.columns if c not in ["B", "C"]]
        if extra_keys:
            keys.append("B")
        df2 = df.groupby(keys).sum()
        print(df2)

    with pytest.raises(
        BodoError,
        match=r"groupby\(\): argument 'by' requires a constant value but variable 'keys' is updated inplace using 'append'",
    ):
        groupby_keys(devguide_df_path, False)


@pytest_mark_spawn_mode
def test_devguide_groupby_keys_append(devguide_df_path):
    @bodo.jit(distributed=False)
    def get_keys(df_columns, extra_keys):
        keys = [c for c in df_columns if c not in ["B", "C"]]
        if extra_keys:
            keys.append("B")
        return keys

    def groupby_keys(devguide_df_path, extra_keys):
        df = pd.read_parquet(devguide_df_path, dtype_backend="pyarrow")
        keys = get_keys(df.columns, extra_keys)
        df2 = df.groupby(keys).sum()
        return df2

    check_func(
        groupby_keys, args=(devguide_df_path, False), only_spawn=True, sort_output=True
    )


@pytest_mark_spawn_mode
def test_devguide_list_typing_error():
    @bodo.jit(spawn=True)
    def create_list():
        out = []
        out.append(0)
        out.append("A")
        out.append(1)
        out.append("B")
        return out

    with pytest.raises(TypingError):
        create_list()


@pytest_mark_spawn_mode
def test_devguide_tuple_typing():
    def create_list():
        out = []
        out.append((0, "A"))
        out.append((1, "B"))
        return out

    check_func(create_list, args=(), only_spawn=True)
