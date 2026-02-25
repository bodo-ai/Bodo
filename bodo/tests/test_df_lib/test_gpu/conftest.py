import os

import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        # Add marker to all tests in this directory
        if "test_gpu" in item.nodeid or "test_gpu" in os.getcwd():
            # skip if BODO_GPU is not set
            if os.getenv("BODO_GPU", "0") == "0":
                item.add_marker(
                    pytest.mark.skip(
                        reason="BODO_GPU is not set, this is required for GPU tests"
                    )
                )
            item.add_marker(pytest.mark.gpu)
