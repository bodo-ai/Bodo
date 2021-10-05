# Copyright (C) 2019 Bodo Inc. All rights reserved.

import pytest

from bodo.tests.utils import InputDist


@pytest.fixture(
    params=[
        # Caching doesn't currently check compiler flags, so for right now, we only run one distribution, OneDVar
        pytest.param(
            InputDist.REP,
            marks=pytest.mark.skip(
                "Caching doesn't currently check compiler flags, see BE-1342"
            ),
        ),
        pytest.param(
            InputDist.OneD,
            marks=pytest.mark.skip(
                "Caching doesn't currently check compiler flags, see BE-1342"
            ),
        ),
        pytest.param(InputDist.OneDVar),
    ]
)
def fn_distribution(request):
    return request.param


@pytest.fixture()
def is_cached(pytestconfig):
    """Fixture used with caching tests, returns true if pytest was called with --is_cached
    and false otherwise"""
    return pytestconfig.getoption("is_cached")
