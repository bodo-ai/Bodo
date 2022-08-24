# Copyright (C) 2022 Bodo Inc. All rights reserved.

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
