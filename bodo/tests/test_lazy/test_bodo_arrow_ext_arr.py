import pandas as pd
import pytest

from bodo.pandas import LazyArrowExtensionArray

# TODO: Refactor once we actually get results from workers


@pytest.fixture
def lazy_int_extension_array():
    nrows = 40
    head_aea = pd.array(range(nrows), dtype="Int64[pyarrow]")[:5]
    laea = LazyArrowExtensionArray(None, nrows=nrows, result_id="abc", head=head_aea)
    return laea


def test_metadata_then_data(lazy_int_extension_array: LazyArrowExtensionArray):
    assert len(lazy_int_extension_array) == 40
    # The array should still be lazy after accessing metadata
    assert lazy_int_extension_array._md_result_id is not None

    assert lazy_int_extension_array[6] == 1
    # The array shouldn't be lazy anymore
    assert lazy_int_extension_array._md_result_id is None


def test_data_then_metadata(lazy_int_extension_array: LazyArrowExtensionArray):
    # The array should be lazy to start
    assert lazy_int_extension_array._md_result_id is not None
    assert lazy_int_extension_array[6] == 1
    # The array shouldn't be lazy anymore
    assert lazy_int_extension_array._md_result_id is None
    # Metadata operations shuold still work
    assert len(lazy_int_extension_array) == 40
