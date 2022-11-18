# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Common fixtures used for timezone testing."""
import pytest


@pytest.fixture(params=["Poland", None])
def sample_tz(request):
    return request.param
