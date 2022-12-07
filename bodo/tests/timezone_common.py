# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Common fixtures used for timezone testing."""
import pytest


@pytest.fixture(params=["Poland", None])
def sample_tz(request):
    return request.param


# create a fixture that's representative of all timezones
@pytest.fixture(
    params=[
        "UTC",
        "US/Pacific",  # timezone behind UTC
        "Europe/Berlin",  # timezone ahead of UTC
        "Africa/Casablanca",  # timezone that's ahead of UTC only during DST
        "Asia/Kolkata",  # timezone that's offset by 30 minutes
        "Asia/Kathmandu",  # timezone that's offset by 45 minutes
        "Australia/Lord_Howe",  # timezone that's offset by 30 minutes only during DST
        "Pacific/Honolulu",  # timezone that has no DST,
        "Etc/GMT+8",  # timezone that has fixed offset from UTC as opposed to zone
    ]
)
def representative_tz(request):
    return request.param
