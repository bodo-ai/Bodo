import pandas as pd
import pytest

from bodo.tests.utils import pytest_slow_unless_codegen
from bodosql.tests.utils import check_query

# Skip unless any codegen files were changed
pytestmark = pytest_slow_unless_codegen


def year_data(has_tz):
    subtract_from_strings = [
        "2016-1-1",
        None,
        "2018-6-25",
        "2020-3-20",
        "2022-4-15",
        "2027-10-5",
        "2024-5-10",
    ]
    subtract_strings = [
        "2015-1-31",
        "2025-6-20",
        "2018-2-25",
        "2025-12-31",
        "2022-11-25",
        None,
        "2014-2-10",
    ]
    answers = [1, None, 0, -5, 0, None, 10]
    return (
        subtract_from_strings,
        subtract_strings,
        answers,
        "Europe/Berlin" if has_tz else None,
    )


def quarter_data(has_tz):
    subtract_from_strings = [
        "2020-1-1",
        None,
        "2020-3-30 12:00:00",
        "2021-2-20",
        "2021-4-1",
        "2021-7-4",
        "2022-10-12 20:00:00",
        None,
        "2022-12-31",
    ]
    subtract_strings = [
        "2021-1-1 23:59:59.999999999",
        None,
        "2020-4-1",
        "2020-7-20",
        "2021-6-30 2:30:00",
        "2024-12-31",
        "2022-5-1",
        "2025-10-31",
        "2012-2-20 13:45:12.250901500",
    ]
    answers = [-4, None, -1, 2, 0, -13, 2, None, 43]
    return (
        subtract_from_strings,
        subtract_strings,
        answers,
        "Pacific/Honolulu" if has_tz else None,
    )


def month_data(has_tz):
    subtract_from_strings = [
        "2020-1-1",
        None,
        "2020-3-30 2:34:56.789",
        "2021-2-20",
        "2021-4-1",
        "2021-7-4",
        "2022-10-12",
        None,
        "2022-12-31",
    ]
    subtract_strings = [
        "2021-1-1",
        None,
        "2020-4-1",
        "2020-7-20",
        "2021-6-30 9:00:00",
        "2021-7-26",
        "2022-5-1",
        "2025-10-31",
        "2012-2-20 22:10:10.123456789",
    ]
    answers = [-12, None, -1, 7, -2, 0, 5, None, 130]
    return (
        subtract_from_strings,
        subtract_strings,
        answers,
        "Asia/Kolkata" if has_tz else None,
    )


def week_data(has_tz):
    subtract_from_strings = [
        # First section: years related to end-of-year transitions
        "2021-1-1 12:00:00",
        "2022-1-1",
        "2023-1-1 18:30:00",
        "2024-1-1",
        "2025-1-1",
        "2026-1-1 12:00:00",
        "2027-1-1",
        "2028-1-1",
        "2029-1-1 12:00:00",
        None,
        # Second section: jumps involving multiple years
        "2018-7-4 18:30:00",
        "2020-7-4",
        "2022-7-4",
        "2024-7-4 12:00:00",
        "2026-7-4",
        None,
        # Third section: finer granularity tests covering multiple scales of jumps
        "2021-10-15",
        "2021-11-15",
        "2021-12-15",
        "2022-1-15",
        "2022-2-15",
        "2022-3-15",
        "2022-4-15",
        "2022-5-15",
        "2022-6-15",
        "2022-7-15",
        "2022-8-15",
        "2022-9-15",
        "2022-10-15 12:00:00",
        "2022-11-15",
        "2022-12-15",
        None,
        # Fourth section: jumps across many years to cover myriad ISO week cases
        "1980-01-01",
        "1981-05-15",
        "1982-09-27",
        "1984-02-09",
        "1985-06-23",
        "1986-11-05",
        "1988-03-19",
        "1989-08-01",
        "1990-12-14",
        "1992-04-27",
        "1993-09-09",
        "1995-01-22",
        "1996-06-05",
        "1997-10-18",
        "1999-03-02",
        "2000-07-14",
        "2001-11-26",
        "2003-04-10",
        "2004-08-22",
        "2006-01-04",
        "2007-05-19",
        "2008-09-30",
        "2010-02-12",
        "2011-06-27",
        "2012-11-08",
        "2014-03-23",
        "2015-08-05",
        "2016-12-17",
        "2018-05-01",
        "2019-09-13",
        "2021-01-25",
        "2022-06-09 18:30:00",
        "2023-10-22 12:00:00",
        "2025-03-05",
        "2026-07-18",
        "2027-11-30",
        "2029-04-13",
        "2030-08-26",
        "2032-01-08",
        "2033-05-22",
        "1973-07-01",
        "1981-09-17",
        "1989-12-04",
        "1998-02-20",
        "2006-05-09",
        "2014-07-26",
        "2022-10-12 12:00:00",
        "2030-12-29",
    ]
    subtract_strings = [
        "2020-12-31 12:00:00",
        "2021-12-31",
        "2022-12-31",
        "2023-12-31",
        "2024-12-31",
        "2025-12-31",
        "2026-12-31 18:30:00",
        "2027-12-31 18:30:00",
        "2028-12-31 18:30:00",
        None,
        # Second section: jumps involving multiple years
        "2020-3-1",
        "2021-3-1",
        "2022-3-1",
        "2023-3-1",
        "2024-3-1",
        None,
        # Third section: finer granularity tests
        "2022-1-1",
        "2022-1-14",
        "2022-1-27",
        "2022-2-9",
        "2022-2-22",
        "2022-3-7",
        "2022-3-20",
        "2022-4-2",
        "2022-4-15",
        "2022-4-28",
        "2022-5-11",
        "2022-5-24",
        "2022-6-6",
        "2022-6-19",
        "2022-7-2",
        None,
        # Fourth section: jumps across many years to cover myriad ISO week cases
        "1990-01-01 18:30:00",
        "1990-10-28",
        "1991-08-24",
        "1992-06-19",
        "1993-04-15",
        "1994-02-09",
        "1994-12-06",
        "1995-10-02",
        "1996-07-28",
        "1997-05-24",
        "1998-03-20",
        "1999-01-14",
        "1999-11-10",
        "2000-09-05",
        "2001-07-02",
        "2002-04-28",
        "2003-02-22",
        "2003-12-19",
        "2004-10-14",
        "2005-08-10",
        "2006-06-06",
        "2007-04-02",
        "2008-01-27",
        "2008-11-22",
        "2009-09-18",
        "2010-07-15",
        "2011-05-11",
        "2012-03-06",
        "2012-12-31",
        "2013-10-27",
        "2014-08-23",
        "2015-06-19",
        "2016-04-14",
        "2017-02-08",
        "2017-12-05",
        "2018-10-01",
        "2019-07-28",
        "2020-05-23",
        "2021-03-19",
        "2022-01-13",
        "1970-02-01",
        "1971-03-08",
        "1972-04-11",
        "1973-05-16",
        "1974-06-20",
        "1975-07-25",
        "1976-08-28",
        "1977-10-02 18:30:00",
    ]
    # Note: values derived from Snowflake
    eoy_answers = [0, 0, 0, 1, 0, 0, 0, 0, 1]
    year_answers = [-86, -35, 18, 70, 122]
    daylight_answers = [-11, -8, -6, -4, -1, 1, 4, 6, 9, 11, 14, 16, 18, 22, 24]
    iso_answers = [
        -522,
        -493,
        -464,
        -436,
        -408,
        -379,
        -351,
        -322,
        -293,
        -264,
        -236,
        -208,
        -179,
        -151,
        -122,
        -93,
        -64,
        -36,
        -8,
        21,
        49,
        78,
        107,
        136,
        164,
        192,
        221,
        249,
        278,
        307,
        336,
        364,
        392,
        421,
        449,
        478,
        507,
        536,
        564,
        592,
        178,
        549,
        921,
        1292,
        1664,
        2035,
        2407,
        2778,
    ]
    answers = (
        eoy_answers
        + [None]
        + year_answers
        + [None]
        + daylight_answers
        + [None]
        + iso_answers
    )
    return (
        subtract_from_strings,
        subtract_strings,
        answers,
        "Asia/Kathmandu" if has_tz else None,
    )


def day_data(has_tz):
    subtract_from_strings = [
        "2005-01-01 6:30:00",
        "2007-09-28",
        "2010-06-24 7:30:00",
        "2013-03-20",
        "2015-12-15",
        "2018-09-10 10:45:00",
        "2021-06-06",
        "2024-03-02 23:59:00",
    ]
    subtract_strings = [
        "2010-01-01 23:59:00",
        "2011-05-16 1:00:00",
        "2012-09-27 7:30:00",
        "2014-02-09",
        "2015-06-24",
        "2016-11-05",
        "2018-03-20 23:59:00",
        "2019-08-02",
    ]
    answers = [-1826, -1326, -826, -326, 174, 674, 1174, 1674]
    return (
        subtract_from_strings,
        subtract_strings,
        answers,
        "Poland" if has_tz else None,
    )


def hour_data(has_tz):
    subtract_from_strings = [
        "2020-07-04",
        "2022-05-25",
        "2022-08-11 06:45:00.250612999",
        "2020-12-31",
        None,
        "2021-01-01",
    ]
    subtract_strings = [
        "2020-03-01",
        "2022-06-25",
        "2020-08-01",
        "2020-10-31 12:00:00",
        None,
        "2020-11-15",
    ]
    if has_tz:
        answers = [2999, -744, 17766, 1453, None, 1128]
    else:
        answers = [3000, -744, 17766, 1452, None, 1128]
    return (
        subtract_from_strings,
        subtract_strings,
        answers,
        "US/Pacific" if has_tz else None,
    )


def minute_data(has_tz):
    subtract_from_strings = [
        "2020-07-04",
        "2022-05-25",
        "2022-08-11 06:45:00.250612999",
        "2020-12-31",
        None,
        "2021-01-01",
    ]
    subtract_strings = [
        "2020-03-01",
        "2022-06-25",
        "2020-08-01",
        "2020-10-31 12:00:00",
        None,
        "2020-11-15",
    ]
    answers = [180000, -44640, 1066005, 87120, None, 67680]
    return (
        subtract_from_strings,
        subtract_strings,
        answers,
        "Asia/Taipei" if has_tz else None,
    )


def second_data(has_tz):
    subtract_from_strings = [
        "2020-07-04",
        "2022-05-25",
        "2022-08-11 06:45:00.250612999",
        "2020-12-31",
        None,
        "2021-01-01",
    ]
    subtract_strings = [
        "2020-03-01",
        "2022-06-25",
        "2020-08-01",
        "2020-10-1 12:00:00",
        None,
        "2020-11-15",
    ]
    if has_tz:
        answers = [10801800, -2678400, 63960300, 7817400, None, 4060800]
    else:
        answers = [10800000, -2678400, 63960300, 7819200, None, 4060800]
    return (
        subtract_from_strings,
        subtract_strings,
        answers,
        "Australia/Lord_Howe" if has_tz else None,
    )


def microsecond_data(has_tz):
    subtract_from_strings = [
        "2019-12-31 23:59:59.876543211",
        "2020-05-22 21:21:18.87654321",
        "2020-10-12 18:42:37.876543209",
        None,
        "2021-03-04 16:03:56.876543208",
    ]
    subtract_strings = [
        "2020-01-01",
        "2020-05-22 21:21:18.901234567",
        "2020-10-12 18:42:37.802469134",
        None,
        "2021-03-04 16:03:56.703703701",
    ]
    # MICROSECOND cannot handle values as large as the other time tests because
    # it is constrained by the Calcite type being INTEGER instead of BIGINT
    answers = [-123457, -24691, 74074, None, 172840]
    return (
        subtract_from_strings,
        subtract_strings,
        answers,
        "US/Arizona" if has_tz else None,
    )


def nanosecond_data(has_tz):
    subtract_from_strings = [
        "2020-07-04",
        "2022-05-25",
        "2022-08-11 06:45:00.250612999",
        "2020-12-31",
        None,
        "2021-01-01",
    ]
    subtract_strings = [
        "2020-03-01",
        "2022-06-25",
        "2020-08-01",
        "2020-10-1 12:00:00",
        None,
        "2020-11-15",
    ]
    if has_tz:
        answers = [
            10796400000000000,
            -2678400000000000,
            63960300250612999,
            7822800000000000,
            None,
            4060800000000000,
        ]
    else:
        answers = [
            10800000000000000,
            -2678400000000000,
            63960300250612999,
            7819200000000000,
            None,
            4060800000000000,
        ]
    return (
        subtract_from_strings,
        subtract_strings,
        answers,
        "US/Mountain" if has_tz else None,
    )


@pytest.fixture
def timestampdiff_data():
    """Returns a dictionary mapping each unit to a function that takes in a
       boolean for whether the input is tz-aware and following testing arugments
       for TIMESTAMPDIFF on each of the tested units:
    1. The string representation of the timestamps being subtracted from
    2. The string representation of the timestamps that are subtracted from #2
    3. The numerical differnence between #2 and #3 in the units specified
    4. The timezone to use"""

    fixture_dict = {
        "YEAR": year_data,
        "QUARTER": quarter_data,
        "MONTH": month_data,
        "WEEK": week_data,
        "DAY": day_data,
        "HOUR": hour_data,
        "MINUTE": minute_data,
        "SECOND": second_data,
        "MICROSECOND": microsecond_data,
        "NANOSECOND": nanosecond_data,
    }
    return fixture_dict


@pytest.mark.parametrize(
    "has_tz, has_case",
    [
        pytest.param(False, False, id="no_tz-no_case"),
        pytest.param(False, True, id="no_tz-with_case", marks=pytest.mark.slow),
        pytest.param(False, False, id="with_tz-no_case", marks=pytest.mark.slow),
        pytest.param(True, True, id="with_tz-with_case"),
    ],
)
@pytest.mark.parametrize(
    "unit",
    [
        pytest.param("YEAR", marks=pytest.mark.slow),
        "QUARTER",
        pytest.param("MONTH", marks=pytest.mark.slow),
        "WEEK",
        pytest.param("DAY", marks=pytest.mark.slow),
        "HOUR",
        pytest.param("MINUTE", marks=pytest.mark.slow),
        "SECOND",
        pytest.param("MICROSECOND", marks=pytest.mark.slow),
        "NANOSECOND",
    ],
)
def test_timestampdiff_cols(
    timestampdiff_data, unit, has_tz, has_case, memory_leak_check
):
    """Tests TIMESTAMPDIFF on various units, with and without case statements, and on
    both tz-aware and tz-naive data. Parametrizes on several arguments at once
    to manually add slow markers while also diversifying coverage."""
    subtract_from_strings, subtract_strings, answers, time_zone = timestampdiff_data[
        unit
    ](has_tz)
    if has_case:
        query = f"SELECT A, B, CASE WHEN C THEN NULL ELSE TIMESTAMPDIFF({unit}, A, B) END AS RES FROM table1"
    else:
        query = f"SELECT A, B, TIMESTAMPDIFF({unit}, A, B) AS RES FROM table1"
    table1 = pd.DataFrame(
        {
            "A": pd.Series(
                [
                    None if s is None else pd.Timestamp(s, tz=time_zone)
                    for s in subtract_strings
                ],
                dtype=f"datetime64[ns, {time_zone}]" if time_zone else "datetime64[ns]",
            ),
            "B": pd.Series(
                [
                    None if s is None else pd.Timestamp(s, tz=time_zone)
                    for s in subtract_from_strings
                ],
                dtype=f"datetime64[ns, {time_zone}]" if time_zone else "datetime64[ns]",
            ),
            "C": [i in (1, 10, 100) for i in range(len(subtract_strings))],
        }
    )
    expected_output = pd.DataFrame(
        {"A": table1.A, "B": table1.B, "RES": pd.Series(answers, dtype=pd.Int64Dtype())}
    )
    if has_case:
        expected_output["RES"] = expected_output["RES"].where(~table1.C, None)
    check_query(
        query,
        {"TABLE1": table1},
        None,
        check_dtype=False,
        expected_output=expected_output,
    )
