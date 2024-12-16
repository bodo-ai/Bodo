import pandas as pd
import pytest

from bodo.pandas import LazyArrowExtensionArray

# TODO: Refactor once we actually get results from workers


@pytest.fixture
def lazy_int_extension_array():
    nrows = 40
    head_aea = pd.array(range(nrows), dtype="Int64[pyarrow]")[:5]
    laea = LazyArrowExtensionArray(
        None,
        nrows=nrows,
        result_id="abc",
        head=head_aea,
        collect_func=lambda _: pd.array(range(nrows), dtype="Int64[pyarrow]"),
        del_func=lambda _: None,
    )
    return laea


def test_metadata_then_data(lazy_int_extension_array: LazyArrowExtensionArray):
    """Tests that metadata operations work even if the array doesn't have the data yet and data operations work after metadata operations"""
    assert len(lazy_int_extension_array) == 40
    # The array should still be lazy after accessing metadata
    assert lazy_int_extension_array._md_result_id is not None

    assert lazy_int_extension_array[6] == 6
    # The array shouldn't be lazy anymore
    assert lazy_int_extension_array._md_result_id is None


def test_data_then_metadata(lazy_int_extension_array: LazyArrowExtensionArray):
    """Tests that data operations work and collect data and metadata operations work after data is collected"""
    # The array should be lazy to start
    assert lazy_int_extension_array._md_result_id is not None
    assert lazy_int_extension_array[6] == 6
    # The array shouldn't be lazy anymore
    assert lazy_int_extension_array._md_result_id is None
    # Metadata operations shuold still work
    assert len(lazy_int_extension_array) == 40


def test_del_func_called_if_not_collected():
    """Tests that the del function is called when the array is deleted if the data hasn't been collected yet"""
    # Can't use the fixture because it will leave a reference and the destructor won't be called
    nrows = 40
    del_called = False

    def del_func(_):
        nonlocal del_called
        del_called = True

    head_aea = pd.array(range(nrows), dtype="Int64[pyarrow]")[:5]
    lazy_int_extension_array = LazyArrowExtensionArray(
        None,
        nrows=nrows,
        result_id="abc",
        head=head_aea,
        collect_func=lambda _: pd.array(range(nrows), dtype="Int64[pyarrow]"),
        del_func=del_func,
    )
    lazy_int_extension_array._del_func = del_func
    del lazy_int_extension_array
    assert del_called


def test_del_func_not_called_if_collected(
    lazy_int_extension_array: LazyArrowExtensionArray,
):
    """Tests that the del function is not called when the array is deleted if the data has been collected"""
    # Can't use the fixture because it will leave a reference and the destructor won't be called
    nrows = 40
    head_aea = pd.array(range(nrows), dtype="Int64[pyarrow]")[:5]
    del_called = False

    def del_func(_):
        nonlocal del_called
        del_called = True

    lazy_int_extension_array = LazyArrowExtensionArray(
        None,
        nrows=nrows,
        result_id="abc",
        head=head_aea,
        collect_func=lambda _: pd.array(range(nrows), dtype="Int64[pyarrow]"),
        del_func=del_func,
    )
    lazy_int_extension_array._del_func = del_func
    # Load the data
    lazy_int_extension_array[6]
    del lazy_int_extension_array
    assert not del_called
