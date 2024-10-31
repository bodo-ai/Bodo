import pandas as pd
from pandas.core.internals import SingleBlockManager
from pandas.core.internals.array_manager import SingleArrayManager

from bodo.pandas.series import BodoSeries
from bodo.tests.test_lazy.utils import single_pandas_managers  # noqa


def test_pandas_series_lazy_manager_metadata_data(single_pandas_managers):
    """
    Test metadata operations are accurate and performed without
    collecting data and data operations are accurate and collect data on
    pandas series using lazy managers.
    """
    lazy_manager, pandas_manager = single_pandas_managers
    head_s = pd.Series(pd.array(["a", "bc", "def", "ghij", "klmno"]))
    assert isinstance(head_s._mgr, pandas_manager)
    head_sam: SingleArrayManager = head_s._mgr

    lsam = lazy_manager([], [], result_id="abc", nrows=40, head=head_sam)
    lam_s: pd.Series = pd.Series._from_mgr(lsam, [])
    lam_s._name = None
    # Metadata present to start
    assert lam_s._mgr._md_result_id is not None

    assert lam_s.dtype == "string"
    assert lam_s.dtypes == "string"
    if isinstance(pandas_manager, SingleBlockManager):
        # This leads to data collection in regular Series,
        # so we've fixed it at the BodoSeries level.
        assert lam_s.head(5).tolist() == ["a", "bc", "def", "ghij", "klmno"]
    # Metadata present after metadata access
    assert lam_s._mgr._md_result_id is not None
    # This leads to data collection in regular Series,
    # so we've fixed it at the BodoSeries level.
    assert lam_s.shape == (40,)
    assert lam_s[6] == "bc"
    # Fetched data after accesing data
    assert lam_s._mgr._md_result_id is None


def test_pandas_series_lazy_manager_data_metadata(single_pandas_managers):
    """
    Test data operations are accurate and collect data and metadata operations
    are accurate after data collection on
    pandas series using lazy managers.
    """
    lazy_manager, pandas_manager = single_pandas_managers
    head_s = pd.Series(pd.array(["a", "bc", "def", "ghij", "klmno"]))
    assert isinstance(head_s._mgr, pandas_manager)
    head_sam: SingleArrayManager = head_s._mgr

    lsam = lazy_manager([], [], result_id="abc", nrows=40, head=head_sam)
    lam_s: pd.Series = pd.Series._from_mgr(lsam, [])
    lam_s._name = None
    # Metadata present to start
    assert lam_s._mgr._md_result_id is not None
    # This leads to data collection in regular Series,
    # so we've fixed it at the BodoSeries level.
    assert lam_s.shape == (40,)
    assert lam_s[6] == "bc"
    # Fetched data after accesing data
    assert lam_s._mgr._md_result_id is None

    # Metadata access still works
    assert lam_s.dtype == "string"
    assert lam_s.dtypes == "string"
    assert lam_s.head(5).tolist() == ["a", "bc", "def", "ghij", "klmno"]


def test_bodo_series_lazy_manager_metadata_data(single_pandas_managers):
    """
    Test metadata operations are accurate and performed without
    collecting data and data operations are accurate and collect data on
    BodoSeries using lazy managers.
    """
    lazy_manager, pandas_manager = single_pandas_managers
    head_s = pd.Series(pd.array(["a", "bc", "def", "ghij", "klmno"]))
    assert isinstance(head_s._mgr, pandas_manager)
    head_sam: SingleArrayManager = head_s._mgr

    lsam = lazy_manager([], [], result_id="abc", nrows=40, head=head_sam)
    lam_s: BodoSeries = BodoSeries.from_lazy_mgr(lsam, head_s)
    lam_s._name = None
    # Metadata present to start
    assert lam_s._mgr._md_result_id is not None

    assert lam_s.dtype == "string"
    assert lam_s.dtypes == "string"
    assert lam_s.head(5).tolist() == ["a", "bc", "def", "ghij", "klmno"]
    assert lam_s.shape == (40,)
    # Metadata present after metadata access
    assert lam_s._mgr._md_result_id is not None
    assert lam_s[6] == "bc"
    # Fetched data after accesing data
    assert lam_s._mgr._md_result_id is None


def test_bodo_series_lazy_manager_data_metadata(single_pandas_managers):
    """
    Test data operations are accurate and collect data and metadata operations
    are accurate after data collection on
    bodo series using lazy managers.
    """
    lazy_manager, pandas_manager = single_pandas_managers
    head_s = pd.Series(pd.array(["a", "bc", "def", "ghij", "klmno"]))
    assert isinstance(head_s._mgr, pandas_manager)
    head_sam: SingleArrayManager = head_s._mgr

    lsam = lazy_manager([], [], result_id="abc", nrows=40, head=head_sam)
    lam_s: BodoSeries = BodoSeries.from_lazy_mgr(lsam, head_s)
    lam_s._name = None
    # Metadata present to start
    assert lam_s._mgr._md_result_id is not None
    assert lam_s[6] == "bc"
    # Fetched data after accesing data
    assert lam_s._mgr._md_result_id is None

    # Metadata access still works
    assert lam_s.dtype == "string"
    assert lam_s.dtypes == "string"
    assert lam_s.head(5).tolist() == ["a", "bc", "def", "ghij", "klmno"]
    assert lam_s.shape == (40,)


def test_bodo_series_pandas_manager(single_pandas_managers):
    """
    Test basic operations on a bodo series using a pandas manager.
    """
    _, pandas_manager = single_pandas_managers
    base_s = pd.Series(pd.array(["a", "bc", "def", "ghij", "klmno"] * 8))

    series = BodoSeries._from_mgr(base_s._mgr, [])
    assert series.shape == (40,)
    assert series.dtypes == "string"

    assert base_s.head(5).equals(series.head(5))
