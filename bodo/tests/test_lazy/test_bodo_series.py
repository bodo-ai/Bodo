import pandas as pd
import pytest
from pandas.core.internals import SingleBlockManager

from bodo.pandas.series import BodoSeries
from bodo.tests.test_lazy.utils import single_pandas_managers  # noqa
from bodo.tests.utils import _test_equal


@pytest.fixture
def head_s():
    return pd.Series(pd.array(["a", "bc", "def", "ghij", "klmno"]))


@pytest.fixture
def collect_func():
    return lambda _: pd.Series(pd.array(["a", "bc", "def", "ghij", "klmno"] * 8))


@pytest.fixture
def del_func():
    return lambda _: None


def test_pandas_series_lazy_manager_metadata_data(
    single_pandas_managers, head_s, collect_func, del_func
):
    """
    Test metadata operations are accurate and performed without
    collecting data and data operations are accurate and collect data on
    pandas series using lazy managers.
    """
    lazy_manager, pandas_manager = single_pandas_managers
    assert isinstance(head_s._mgr, pandas_manager)
    head_sam = head_s._mgr

    lsam = lazy_manager(
        [],
        [],
        result_id="abc",
        nrows=40,
        head=head_sam,
        collect_func=collect_func,
        del_func=del_func,
    )
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


def test_pandas_series_lazy_manager_data_metadata(
    single_pandas_managers, head_s, collect_func, del_func
):
    """
    Test data operations are accurate and collect data and metadata operations
    are accurate after data collection on
    pandas series using lazy managers.
    """
    lazy_manager, pandas_manager = single_pandas_managers
    assert isinstance(head_s._mgr, pandas_manager)
    head_sam = head_s._mgr

    lsam = lazy_manager(
        [],
        [],
        result_id="abc",
        nrows=40,
        head=head_sam,
        collect_func=collect_func,
        del_func=del_func,
    )
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


def test_bodo_series_lazy_manager_metadata_data(
    single_pandas_managers, head_s, collect_func, del_func
):
    """
    Test metadata operations are accurate and performed without
    collecting data and data operations are accurate and collect data on
    BodoSeries using lazy managers.
    """
    lazy_manager, pandas_manager = single_pandas_managers
    assert isinstance(head_s._mgr, pandas_manager)
    head_sam = head_s._mgr

    lsam = lazy_manager(
        [],
        [],
        result_id="abc",
        nrows=40,
        head=head_sam,
        collect_func=collect_func,
        del_func=del_func,
    )
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


def test_bodo_series_lazy_manager_data_metadata(
    single_pandas_managers, head_s, collect_func, del_func
):
    """
    Test data operations are accurate and collect data and metadata operations
    are accurate after data collection on
    bodo series using lazy managers.
    """
    lazy_manager, pandas_manager = single_pandas_managers
    assert isinstance(head_s._mgr, pandas_manager)
    head_sam = head_s._mgr

    lsam = lazy_manager(
        [],
        [],
        result_id="abc",
        nrows=40,
        head=head_sam,
        collect_func=collect_func,
        del_func=del_func,
    )
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


def test_bodo_series_del_func_called_if_not_collected(single_pandas_managers):
    """Tests that the del function is called when the manager is deleted if the data hasn't been collected yet"""
    lazy_manager, pandas_manager = single_pandas_managers
    del_called = False

    def del_func(_):
        nonlocal del_called
        del_called = True

    lsa = lazy_manager(
        [],
        [],
        result_id="abc",
        nrows=40,
        head=pd.Series(pd.array(["a", "bc", "def", "ghij", "klmno"] * 8))._mgr,
        collect_func=lambda _: None,
        del_func=del_func,
    )
    del lsa
    assert del_called


def test_bodo_series_del_func_not_called_if_collected(single_pandas_managers):
    """Tests that the del function is not called when the manager is deleted if the data has been collected"""
    lazy_manager, pandas_manager = single_pandas_managers
    del_called = False

    def del_func(_):
        nonlocal del_called
        del_called = True

    lsa = lazy_manager(
        [],
        [],
        result_id="abc",
        nrows=40,
        head=pd.Series(pd.array(["a", "bc", "def", "ghij", "klmno"] * 8))._mgr,
        collect_func=lambda _: pd.Series(
            pd.array(["a", "bc", "def", "ghij", "klmno"] * 8)
        ),
        del_func=del_func,
    )
    lsa._collect()
    del lsa
    assert not del_called


def test_len(single_pandas_managers, head_s, collect_func, del_func):
    """Tests that len() returns the right value and does not trigger data fetch"""
    lazy_manager, pandas_manager = single_pandas_managers

    lsam = lazy_manager(
        [],
        [],
        result_id="abc",
        nrows=40,
        head=head_s._mgr,
        collect_func=collect_func,
        del_func=del_func,
    )

    # len() does not trigger a data fetch
    assert len(lsam) == 40
    lam_s: BodoSeries = BodoSeries._from_mgr(lsam, [])
    assert lam_s._lazy

    # force collect
    lsam._collect()
    assert not lam_s._lazy
    assert len(lsam) == 40


@pytest.mark.skip("Fix Series slice tests")
def test_slice(single_pandas_managers, head_s, collect_func, del_func):
    """Tests that slicing returns the correct value and does not trigger data fetch unnecessarily"""
    lazy_manager, pandas_manager = single_pandas_managers

    lsam = lazy_manager(
        [],
        [],
        result_id="abc",
        nrows=40,
        head=head_s._mgr,
        collect_func=collect_func,
        del_func=del_func,
    )

    # slicing head does not trigger a data fetch
    lam_s: BodoSeries = BodoSeries._from_mgr(lsam, [])
    lam_sliced_head_s = lam_s[1:3]
    assert lam_s._lazy
    assert lam_sliced_head_s.tolist() == (head_s[1:3]).tolist()

    # slicing for data outside of head triggers a fetch
    lam_sliced_s = lam_s[10:30]
    assert not lam_s._lazy
    assert lam_sliced_s.tolist() == (collect_func(0)[10:30]).tolist()

    # Slicing with negative indices (does not trigger a data fetch)
    lsam = lazy_manager(
        [],
        [],
        result_id="abc",
        nrows=40,
        head=head_s._mgr,
        collect_func=collect_func,
        del_func=del_func,
    )
    lam_s: BodoSeries = BodoSeries._from_mgr(lsam, [])
    lam_sliced_head_s = lam_s[-38:-37]
    assert lam_s._lazy

    # Ignoring index for now since BodoDataFrames resets RangeIndex
    _test_equal(
        lam_sliced_head_s, head_s[2:3], check_pandas_types=False, reset_index=True
    )

    # Triggers a fetch
    lam_sliced_head_s = lam_s[-3:]
    assert not lam_s._lazy
    _test_equal(
        lam_sliced_head_s,
        collect_func(0)[-3:],
        check_pandas_types=False,
        reset_index=True,
    )
