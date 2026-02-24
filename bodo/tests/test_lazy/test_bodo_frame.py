import os

import pandas as pd
import pytest

try:
    from pandas.core.internals.array_manager import ArrayManager
except ModuleNotFoundError:
    # Pandas >= 3 does not have an array_manager module (uses BlockManager/SinglBlockManager).
    class ArrayManager:
        pass


import bodo
from bodo.pandas.frame import BodoDataFrame
from bodo.pandas.managers import LazyBlockManager
from bodo.tests.iceberg_database_helpers.utils import create_iceberg_table, get_spark
from bodo.tests.test_lazy.utils import pandas_managers  # noqa
from bodo.tests.utils import (
    _gather_output,
    _test_equal,
    pytest_mark_one_rank,
    pytest_spawn_mode,
)
from bodo.utils.testing import ensure_clean2

pytestmark = pytest_spawn_mode


@pytest.fixture
def head_df():
    return pd.DataFrame(
        {
            "A0": pd.array([1, 2, 3, 4, 5], dtype="Int64"),
            "B5": pd.array(["a", "bc", "def", "ghij", "klmno"]),
        }
    )


@pytest.fixture
def collect_func():
    return lambda _: pd.DataFrame(
        {
            "A0": pd.array([1, 2, 3, 4, 5] * 8, dtype="Int64"),
            "B5": pd.array(["a", "bc", "def", "ghij", "klmno"] * 8),
        }
    )


@pytest.fixture
def del_func():
    return lambda _: None


def test_pandas_df_lazy_manager_metadata_data(
    pandas_managers, head_df, collect_func, del_func
):
    """
    Test metadata operations are accurate and performed without
    collecting data and data operations are accurate and collect data on
    pandas DataFrame using lazy managers.
    """
    lazy_manager, pandas_manager = pandas_managers
    assert isinstance(head_df._mgr, pandas_manager)
    head_am = head_df._mgr
    lam = lazy_manager(
        [],
        [],
        result_id="abc",
        nrows=40,
        head=head_am,
        collect_func=collect_func,
        del_func=del_func,
    )
    lam_df: pd.DataFrame = pd.DataFrame._from_mgr(lam, [])

    # We haven't fetched data to start
    assert lam_df._mgr._md_result_id is not None

    assert lam_df.shape == (40, 2)
    assert lam_df.dtypes.equals(
        pd.Series(["Int64", "string"], index=["A0", "B5"], dtype=object)
    )

    # With block managers head operations cause data pull when not using a BodoDataFrame
    if isinstance(pandas_manager, ArrayManager):
        assert head_df.equals(lam_df.head(5))
    # Make sure we haven't fetched data
    assert lam_df._mgr._md_result_id is not None

    agg_df = lam_df.groupby("A0").size()
    assert agg_df.equals(
        pd.Series(
            [8, 8, 8, 8, 8], index=pd.Index([1, 2, 3, 4, 5], dtype="Int64", name="A0")
        )
    )
    assert lam_df.describe().equals(
        pd.DataFrame(
            {"A0": [40.0, 3.0, 1.4322297480788657, 1, 2, 3, 4, 5]},
            index=pd.Index(
                ["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
                dtype="object",
            ),
            dtype="Float64",
        )
    )
    # Make sure we have fetched data
    assert lam_df._mgr._md_result_id is None


def test_pandas_df_lazy_manager_data_metadata(
    pandas_managers, head_df, collect_func, del_func
):
    """
    Test data operations are accurate and collect data and metadata operations
    are accurate after data collection on
    pandas DataFrames using lazy managers.
    """
    head_df = pd.DataFrame(
        {
            "A0": pd.array([1, 2, 3, 4, 5], dtype="Int64"),
            "B5": pd.array(["a", "bc", "def", "ghij", "klmno"]),
        }
    )
    lazy_manager, pandas_manager = pandas_managers
    assert isinstance(head_df._mgr, pandas_manager)
    head_am = head_df._mgr
    lam = lazy_manager(
        [],
        [],
        result_id="abc",
        nrows=40,
        head=head_am,
        collect_func=collect_func,
        del_func=del_func,
    )
    lam_df: pd.DataFrame = pd.DataFrame._from_mgr(lam, [])

    # We haven't fetched data to start
    assert lam_df._mgr._md_result_id is not None

    agg_df = lam_df.groupby("A0").size()
    assert agg_df.equals(
        pd.Series(
            [8, 8, 8, 8, 8], index=pd.Index([1, 2, 3, 4, 5], dtype="Int64", name="A0")
        )
    )

    assert head_df.equals(lam_df.head(5))
    assert lam_df.describe().equals(
        pd.DataFrame(
            {"A0": [40.0, 3.0, 1.4322297480788657, 1, 2, 3, 4, 5]},
            index=pd.Index(
                ["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
                dtype="object",
            ),
            dtype="Float64",
        )
    )
    # Make sure we have fetched data
    assert lam_df._mgr._md_result_id is None
    # Metadata still works after fetch
    assert lam_df.shape == (40, 2)
    assert lam_df.dtypes.equals(
        pd.Series(["Int64", "string"], index=["A0", "B5"], dtype=object)
    )


def test_bodo_df_lazy_managers_metadata_data(
    pandas_managers, head_df, collect_func, del_func
):
    """
    Test metadata operations are accurate and performed without
    collecting data and data operations are accurate and collect data on
    BodoDataFrames using lazy managers.
    """
    if pandas_managers[1] == ArrayManager:
        pytest.skip("TODO: fix ArrayManager DataFrames test")

    head_df = pd.DataFrame(
        {
            "A0": pd.array([1, 2, 3, 4, 5], dtype="Int64"),
            "B5": pd.array(["a", "bc", "def", "ghij", "klmno"]),
        }
    )
    lazy_manager, pandas_manager = pandas_managers
    assert isinstance(head_df._mgr, pandas_manager)
    head_am = head_df._mgr
    lam = lazy_manager(
        [],
        [],
        result_id="abc",
        nrows=40,
        head=head_am,
        collect_func=collect_func,
        del_func=del_func,
    )
    lam_df: BodoDataFrame = BodoDataFrame.from_lazy_mgr(lam, head_df)

    # We haven't fetched data to start
    assert lam_df._mgr._md_result_id is not None

    assert lam_df.shape == (40, 2)
    assert lam_df.dtypes.equals(
        pd.Series(["Int64", "string"], index=["A0", "B5"], dtype=object)
    )

    assert head_df.equals(lam_df.head(5))
    # Make sure we haven't fetched data
    assert lam_df._mgr._md_result_id is not None

    agg_df = lam_df.groupby("A0").size()
    assert agg_df.equals(
        pd.Series(
            [8, 8, 8, 8, 8], index=pd.Index([1, 2, 3, 4, 5], dtype="Int64", name="A0")
        )
    )

    expected = pd.DataFrame(
        {"A0": [40.0, 3.0, 1.4322297480788657, 1, 2, 3, 4, 5]},
        index=pd.Index(
            ["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
            dtype="object",
        ),
        dtype="Float64",
    )
    _test_equal(lam_df.describe(), expected)

    # Make sure we have fetched data
    assert lam_df._mgr._md_result_id is None


def test_bodo_df_lazy_managers_data_metadata(
    pandas_managers, head_df, collect_func, del_func
):
    """
    Test data operations are accurate and collect data and metadata operations
    are accurate after data collection on
    BodoDataFrames using lazy managers.
    """
    if pandas_managers[1] == ArrayManager:
        pytest.skip("TODO: fix ArrayManager DataFrames test")

    head_df = pd.DataFrame(
        {
            "A0": pd.array([1, 2, 3, 4, 5], dtype="Int64"),
            "B5": pd.array(["a", "bc", "def", "ghij", "klmno"]),
        }
    )
    lazy_manager, pandas_manager = pandas_managers
    assert isinstance(head_df._mgr, pandas_manager)
    head_am = head_df._mgr
    lam = lazy_manager(
        [],
        [],
        result_id="abc",
        nrows=40,
        head=head_am,
        collect_func=collect_func,
        del_func=del_func,
    )
    lam_df: BodoDataFrame = BodoDataFrame.from_lazy_mgr(lam, head_df)

    # We haven't fetched data to start
    assert lam_df._mgr._md_result_id is not None

    agg_df = lam_df.groupby("A0").size()
    assert agg_df.equals(
        pd.Series(
            [8, 8, 8, 8, 8], index=pd.Index([1, 2, 3, 4, 5], dtype="Int64", name="A0")
        )
    )

    assert head_df.equals(lam_df.head(5))
    expected = pd.DataFrame(
        {"A0": [40.0, 3.0, 1.4322297480788657, 1, 2, 3, 4, 5]},
        index=pd.Index(
            ["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
            dtype="object",
        ),
        dtype="Float64",
    )
    _test_equal(lam_df.describe(), expected)
    # Make sure we have fetched data
    assert lam_df._mgr._md_result_id is None
    # Metadata still works after fetch
    assert lam_df.shape == (40, 2)
    assert lam_df.dtypes.equals(
        pd.Series(["Int64", "string"], index=["A0", "B5"], dtype=object)
    )


def test_bodo_data_frame_pandas_manager(pandas_managers):
    """
    Test basic operations on a bodo series using a pandas manager.
    """

    _, pandas_manager = pandas_managers

    if pandas_manager == ArrayManager:
        pytest.skip("TODO: fix ArrayManager DataFrames test")

    base_df = pd.DataFrame(
        {
            "A0": pd.array([1, 2, 3, 4, 5] * 8, dtype="Int64"),
            "B5": pd.array(["a", "bc", "def", "ghij", "klmno"] * 8),
        }
    )

    df = BodoDataFrame._from_mgr(base_df._mgr, [])
    assert df.shape == (40, 2)
    assert df.dtypes.equals(
        pd.Series(["Int64", "string"], index=["A0", "B5"], dtype=object)
    )

    assert base_df.head(5).equals(df.head(5))

    agg_df = df.groupby("A0").size()
    assert agg_df.equals(
        pd.Series(
            [8, 8, 8, 8, 8], index=pd.Index([1, 2, 3, 4, 5], dtype="Int64", name="A0")
        )
    )

    expected = pd.DataFrame(
        {"A0": [40.0, 3.0, 1.4322297480788657, 1, 2, 3, 4, 5]},
        index=pd.Index(
            ["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
            dtype="object",
        ),
        dtype="Float64",
    )

    _test_equal(df.describe(), expected)


def test_del_func_called_if_not_collected(pandas_managers, head_df, collect_func):
    """Tests that the del function is called when the manager is deleted if the data hasn't been collected yet"""
    lazy_manager, pandas_manager = pandas_managers
    del_called = False

    def del_func(_):
        nonlocal del_called
        del_called = True

    lsa = lazy_manager(
        [],
        [],
        result_id="abc",
        nrows=40,
        head=head_df._mgr,
        collect_func=collect_func,
        del_func=del_func,
    )
    del lsa
    assert del_called


def test_del_func_not_called_if_collected(pandas_managers, head_df, collect_func):
    """Tests that the del function is not called when the manager is deleted if the data has been collected"""
    lazy_manager, pandas_manager = pandas_managers
    del_called = False

    def del_func(_):
        nonlocal del_called
        del_called = True

    lsa = lazy_manager(
        [],
        [],
        result_id="abc",
        nrows=40,
        head=head_df._mgr,
        collect_func=collect_func,
        del_func=del_func,
    )
    lsa._collect()
    del lsa
    assert not del_called


def test_len(pandas_managers, head_df, collect_func):
    """Tests that len() returns the right value and does not trigger data fetch"""
    lazy_manager, pandas_manager = pandas_managers

    lam = lazy_manager(
        [],
        [],
        result_id="abc",
        nrows=40,
        head=head_df._mgr,
        collect_func=collect_func,
        del_func=del_func,
    )

    if isinstance(lam, LazyBlockManager):
        pytest.skip(
            "Can't easily override methods for `LazyBlockManager` since it's implemented in Cython"
        )

    # len() does not trigger a data fetch
    assert len(lam) == 40
    lam_df: BodoDataFrame = BodoDataFrame.from_lazy_mgr(lam, head_df)
    assert lam_df._lazy

    # force collect
    lam._collect()
    assert not lam_df._lazy
    assert len(lam) == 40


def test_slice(pandas_managers, head_df, collect_func):
    """Tests that slicing returns the correct value and does not trigger data fetch unnecessarily"""
    lazy_manager, pandas_manager = pandas_managers

    lam = lazy_manager(
        [],
        [],
        result_id="abc",
        nrows=40,
        head=head_df._mgr,
        collect_func=collect_func,
        del_func=del_func,
    )

    if isinstance(lam, LazyBlockManager):
        pytest.skip(
            "Can't easily override methods for `LazyBlockManager` since it's implemented in Cython"
        )

    # slicing head does not trigger a data fetch
    lam_df: BodoDataFrame = BodoDataFrame.from_lazy_mgr(lam, head_df)
    lam_sliced_head_df = lam_df[1:3]
    assert lam_df._lazy
    pd.testing.assert_frame_equal(lam_sliced_head_df, head_df[1:3])

    # Slicing with negative indices (does not trigger a data fetch)
    lam = lazy_manager(
        [],
        [],
        result_id="abc",
        nrows=40,
        head=head_df._mgr,
        collect_func=collect_func,
        del_func=del_func,
    )
    lam_df: BodoDataFrame = BodoDataFrame.from_lazy_mgr(lam, head_df)
    lam_sliced_head_df = lam_df.iloc[-38:-37]
    assert lam_df._lazy
    pd.testing.assert_frame_equal(lam_sliced_head_df, head_df[2:3])

    lam = lazy_manager(
        [],
        [],
        result_id="abc",
        nrows=40,
        head=head_df._mgr,
        collect_func=collect_func,
        del_func=del_func,
    )
    lam_df: BodoDataFrame = BodoDataFrame.from_lazy_mgr(lam, head_df)

    # Trigger a fetch
    lam_sliced_head_df = lam_df.iloc[-3:]
    assert not lam_df._lazy
    pd.testing.assert_frame_equal(lam_sliced_head_df, collect_func(0)[-3:])


@pytest.mark.iceberg
@pytest_mark_one_rank
def test_sql(iceberg_database, iceberg_table_conn, collect_func):
    """Tests that to_sql() writes the frame correctly and does not trigger data fetch"""

    @bodo.jit(spawn=True)
    def _get_bodo_df(df):
        return df

    df = collect_func(0)
    bodo_df = _get_bodo_df(df)

    # create table in iceberg_database
    table_name = "TEST_TABLE_NAME"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, False)
    sql_schema = [("A0", "float", True), ("B5", "string", True)]
    spark = get_spark()
    create_iceberg_table(df, sql_schema, table_name, spark)

    bodo_df.to_sql(table_name, conn, db_schema, if_exists="replace")
    assert bodo_df._lazy

    bodo_out = bodo.jit()(lambda: pd.read_sql_table(table_name, conn, db_schema))()
    read_df = _gather_output(bodo_out)

    pd.testing.assert_frame_equal(
        read_df,
        df,
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        bodo_df,
        df,
        check_dtype=False,
    )


def test_csv(collect_func):
    """Tests that to_csv() writes the frame correctly and does not trigger data fetch"""

    @bodo.jit(spawn=True)
    def _get_bodo_df(df):
        return df

    df = collect_func(0)
    bodo_df = _get_bodo_df(df)
    fname = os.path.join("bodo", "tests", "data", "example")
    with ensure_clean2(fname):
        bodo_df.to_csv(fname, index=False)
        assert bodo_df._lazy
        read_df = pd.read_csv(fname, dtype_backend="pyarrow")

    pd.testing.assert_frame_equal(
        read_df,
        df,
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        bodo_df,
        df,
        check_dtype=False,
    )


def test_csv_str(collect_func):
    """Tests that to_csv() with string output works properly (returns all string data to spawner)"""

    @bodo.jit(spawn=True)
    def _get_bodo_df(df):
        return df

    df = collect_func(0)
    bodo_df = _get_bodo_df(df)
    assert bodo_df.to_csv(index=False) == df.to_csv(index=False)
    assert bodo_df._lazy


def test_csv_param(collect_func):
    """Tests that to_csv() raises an error on unsupported parameters"""

    @bodo.jit(spawn=True)
    def _get_bodo_df(df):
        return df

    df = collect_func(0)
    bodo_df = _get_bodo_df(df)
    fname = os.path.join("bodo", "tests", "data", "example")

    with ensure_clean2(fname):
        with pytest.raises(
            bodo.utils.typing.BodoError,
            match=r"DataFrame.to_csv\(\): 'path_or_buf' argument should be None or string",
        ):
            bodo_df.to_csv(1)
        with pytest.raises(
            bodo.utils.typing.BodoError,
            match=r"DataFrame.to_csv\(\): 'compression' argument supports only None, which is the default in JIT code.",
        ):
            bodo_df.to_csv(fname, compression="a")
        with pytest.raises(
            bodo.utils.typing.BodoError,
            match=r"BodoDataFrame.to_csv\(\): mode parameter only supports default value w",
        ):
            bodo_df.to_csv(fname, mode="r")


def test_json(collect_func):
    """Tests that to_json() writes the frame correctly and does not trigger data fetch"""

    @bodo.jit(spawn=True)
    def _get_bodo_df(df):
        return df

    @bodo.jit(spawn=True)
    def _read_bodo_df(fname):
        read_df = pd.read_json(
            path_or_buf=fname, orient="records", lines=True, dtype_backend="pyarrow"
        )
        return read_df

    df = collect_func(0)
    bodo_df = _get_bodo_df(df)
    fname = os.path.join("bodo", "tests", "data", "example")

    with ensure_clean2(fname):
        bodo_df.to_json(path_or_buf=fname, orient="records", lines=True)
        assert bodo_df._lazy
        read_df = _read_bodo_df(fname)

    pd.testing.assert_frame_equal(
        read_df,
        df,
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        bodo_df,
        df,
        check_dtype=False,
    )


def test_json_str(collect_func):
    """Tests that to_json() with string output works properly (returns all string data to spawner)"""

    @bodo.jit(spawn=True)
    def _get_bodo_df(df):
        return df

    df = collect_func(0)
    bodo_df = _get_bodo_df(df)
    assert bodo_df.to_json(orient="records", lines=True) == df.to_json(
        orient="records", lines=True
    )
    assert bodo_df._lazy


def test_map_partitions():
    """Test map_partitions on BodoDataFrame"""

    @bodo.jit(spawn=True)
    def _get_bodo_df(df):
        return df

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    bodo_df = _get_bodo_df(df)

    def f(df, a, b=3):
        return df + a + 2 * b

    out = bodo_df.map_partitions(f, 2, b=4)
    assert out._lazy
    pd.testing.assert_frame_equal(
        out,
        f(df, 2, b=4),
        check_dtype=False,
    )
