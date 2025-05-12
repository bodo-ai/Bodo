import os

import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        # Add marker to all tests in this directory
        if "test_df_lib" in item.nodeid or "test_df_lib" in os.getcwd():
            # skip if BODO_ENABLE_TEST_DATAFRAME_LIBRARY is not set
            if os.getenv("BODO_ENABLE_TEST_DATAFRAME_LIBRARY", "0") == "0":
                item.add_marker(
                    pytest.mark.skip(
                        reason="BODO_ENABLE_TEST_DATAFRAME_LIBRARY is not set, this is required for df_lib tests"
                    )
                )
            item.add_marker(pytest.mark.df_lib)
