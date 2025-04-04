import pytest

import bodo


@pytest.fixture(autouse=True)
def enable_df_lib():
    """Fixture to enable the DataFrame library for the test."""
    previous_state = bodo.dataframe_library_enabled
    bodo.dataframe_library_enabled = True
    yield
    bodo.dataframe_library_enabled = previous_state
