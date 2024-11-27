"""Common fixtures used for timezone testing."""

import pytest

TIMEZONES = [
    "UTC",
    pytest.param("US/Pacific", marks=pytest.mark.slow),  # timezone behind UTC
    pytest.param("Europe/Berlin", marks=pytest.mark.slow),  # timezone ahead of UTC
    "Africa/Casablanca",  # timezone that's ahead of UTC only during DST
    pytest.param(
        "Asia/Kolkata", marks=pytest.mark.slow
    ),  # timezone that's offset by 30 minutes
    # timezone that's offset by 45 minutes
    pytest.param("Asia/Kathmandu", marks=pytest.mark.slow),
    "Australia/Lord_Howe",  # timezone that's offset by 30 minutes only during DST
    "Pacific/Honolulu",  # timezone that has no DST,
    # timezone that has fixed offset from UTC as opposed to zone
    pytest.param("Etc/GMT+8", marks=pytest.mark.slow),
]


@pytest.fixture(params=["Poland", None])
def sample_tz(request):
    return request.param


@pytest.fixture(params=TIMEZONES + [None])
def representative_tz_or_none(request):
    return request.param


# create a fixture that's representative of all timezones
@pytest.fixture(params=TIMEZONES)
def representative_tz(request):
    return request.param
