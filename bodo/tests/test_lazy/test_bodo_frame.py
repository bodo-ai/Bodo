import pandas as pd
from pandas.core.internals.array_manager import ArrayManager

from bodo.pandas.frame import BodoDataFrame
from bodo.tests.test_lazy.utils import pandas_managers  # noqa


def test_pandas_df_lazy_manager_metadata_data(pandas_managers):
    """
    Test metadata operations are accurate and performed without
    collecting data and data operations are accurate and collect data on
    pandas DataFrame using lazy managers.
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
    lam = lazy_manager([], [], result_id="abc", nrows=40, head=head_am)
    lam_df: pd.DataFrame = pd.DataFrame._from_mgr(lam, [])

    # We haven't fetched data to start
    assert lam_df._mgr._md_result_id is not None

    assert lam_df.shape == (40, 2)
    assert lam_df.dtypes.equals(
        pd.Series(["Int64", "string[python]"], index=["A0", "B5"])
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


def test_pandas_df_lazy_manager_data_metadata(pandas_managers):
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
    lam = lazy_manager([], [], result_id="abc", nrows=40, head=head_am)
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
        pd.Series(["Int64", "string[python]"], index=["A0", "B5"])
    )


def test_bodo_df_lazy_managers_metadata_data(pandas_managers):
    """
    Test metadata operations are accurate and performed without
    collecting data and data operations are accurate and collect data on
    BodoDataFrames using lazy managers.
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
    lam = lazy_manager([], [], result_id="abc", nrows=40, head=head_am)
    lam_df: BodoDataFrame = BodoDataFrame.from_lazy_mgr(lam, head_df)

    # We haven't fetched data to start
    assert lam_df._mgr._md_result_id is not None

    assert lam_df.shape == (40, 2)
    assert lam_df.dtypes.equals(
        pd.Series(["Int64", "string[python]"], index=["A0", "B5"])
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


def test_bodo_df_lazy_managers_data_metadata(pandas_managers):
    """
    Test data operations are accurate and collect data and metadata operations
    are accurate after data collection on
    BodoDataFrames using lazy managers.
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
    lam = lazy_manager([], [], result_id="abc", nrows=40, head=head_am)
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
        pd.Series(["Int64", "string[python]"], index=["A0", "B5"])
    )


def test_bodo_data_frame_pandas_manager(pandas_managers):
    """
    Test basic operations on a bodo series using a pandas manager.
    """
    _, pandas_manager = pandas_managers
    base_df = pd.DataFrame(
        {
            "A0": pd.array([1, 2, 3, 4, 5] * 8, dtype="Int64"),
            "B5": pd.array(["a", "bc", "def", "ghij", "klmno"] * 8),
        }
    )

    df = BodoDataFrame._from_mgr(base_df._mgr, [])
    assert df.shape == (40, 2)
    assert df.dtypes.equals(pd.Series(["Int64", "string[python]"], index=["A0", "B5"]))

    assert base_df.head(5).equals(df.head(5))

    agg_df = df.groupby("A0").size()
    assert agg_df.equals(
        pd.Series(
            [8, 8, 8, 8, 8], index=pd.Index([1, 2, 3, 4, 5], dtype="Int64", name="A0")
        )
    )
    assert df.describe().equals(
        pd.DataFrame(
            {"A0": [40.0, 3.0, 1.4322297480788657, 1, 2, 3, 4, 5]},
            index=pd.Index(
                ["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
                dtype="object",
            ),
            dtype="Float64",
        )
    )
