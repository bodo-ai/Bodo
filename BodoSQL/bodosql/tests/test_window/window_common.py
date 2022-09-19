import os

import pytest

# Helper environment variable to allow for testing locally, while avoiding
# memory issues on CI
testing_locally = os.environ.get("BODOSQL_TESTING_LOCALLY", False)


@pytest.fixture(
    params=[
        "RESPECT NULLS",
        pytest.param(
            "IGNORE NULLS",
            marks=pytest.mark.skip("https://bodo.atlassian.net/browse/BE-3583"),
        ),
        pytest.param("", id="empty_str", marks=pytest.mark.slow),
    ]
)
def null_respect_string(request):
    """Returns the null behavior string, for use with LEAD/LAG and First/Last/Nth value"""
    return request.param
