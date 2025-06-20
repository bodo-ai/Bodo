import re

import numpy as np
import pandas as pd
import pytest

import bodo.pandas as bd
from bodo.tests.utils import _test_equal


@pytest.mark.parametrize(
    "df, expr, expand, expected, flags",
    [
        # Simple unnamed groups, expand=False
        (
            pd.DataFrame({"A": ["a1", "b2", "c3"]}, index=["x", "y", "z"]),
            r"[ab](\d)",
            False,
            pd.Series(["1", "2", np.nan], index=["x", "y", "z"], name="0"),
            0,
        ),
        (
            pd.DataFrame({"A": ["a1", "b2", "c3"]}, index=["x", "y", "z"]),
            r"([ab])(\d)",
            False,
            pd.DataFrame(
                {"0": ["a", "b", np.nan], "1": ["1", "2", np.nan]},
                index=["x", "y", "z"],
            ),
            0,
        ),
        (
            pd.DataFrame({"A": ["a1", "b2", "c3"]}, index=["x", "y", "z"]),
            r"(?P<letter>[ab])(?P<digit>\d)",
            False,
            pd.DataFrame(
                {"letter": ["a", "b", np.nan], "digit": ["1", "2", np.nan]},
                index=["x", "y", "z"],
            ),
            0,
        ),
        # Multiple unnamed groups, expand=True
        (
            pd.DataFrame({"A": ["abc-123", "def-456", "ghi-789"]}),
            r"([a-z]+)-(\d+)",
            True,
            pd.DataFrame({"0": ["abc", "def", "ghi"], "1": ["123", "456", "789"]}),
            0,
        ),
        # Named groups, expand=True
        (
            pd.DataFrame({"A": ["abc-123", "def-456", "ghi-789"]}),
            r"(?P<letters>[a-z]+)-(?P<numbers>\d+)",
            True,
            pd.DataFrame(
                {"letters": ["abc", "def", "ghi"], "numbers": ["123", "456", "789"]}
            ),
            0,
        ),
        # Partial matches, expand=True
        (
            pd.DataFrame({"A": ["abc-123", "no-match", "ghi-789"]}),
            r"([a-z]+)-(\d+)",
            True,
            pd.DataFrame({"0": ["abc", np.nan, "ghi"], "1": ["123", np.nan, "789"]}),
            0,
        ),
        # No matches at all, expand=True
        (
            pd.DataFrame({"A": ["nope", "still nope"]}),
            r"(\d+)",
            True,
            pd.DataFrame({"0": [np.nan, np.nan]}),
            0,
        ),
        # expand=False and one group
        (pd.DataFrame({"A": ["abc-123", "def-456"]}), r"\d+", False, None, 0),
        # expand=False and multiple groups
        (
            pd.DataFrame({"A": ["abc-123", "def-456"]}),
            r"([a-z]+)-(\d+)",
            False,
            pd.DataFrame({"0": ["abc", "def"], "1": ["123", "456"]}, index=[0, 1]),
            0,
        ),
        # Strings with None and string dtype
        (
            pd.DataFrame({"A": ["foo123", None, "bar456", "789"]}),
            r"(\d+)",
            True,
            pd.DataFrame({"0": ["123", np.nan, "456", "789"]}),
            0,
        ),
        # Zero capture groups
        (pd.DataFrame({"A": ["abc123", "def456"]}), r"[a-z]+\d+", True, None, 0),
        # Case-insensitive match
        (
            pd.DataFrame({"A": ["abc-123", "DEF-456", "Ghi-789"]}),
            r"([a-z]+)-(\d+)",
            True,
            pd.DataFrame({"0": ["abc", "DEF", "Ghi"], "1": ["123", "456", "789"]}),
            re.IGNORECASE,
        ),
        # Dot matches newline with DOTALL
        (
            pd.DataFrame({"A": ["abc\n123", "def\n456"]}),
            r"([a-z]+).(\d+)",
            True,
            pd.DataFrame({"0": ["abc", "def"], "1": ["123", "456"]}),
            re.DOTALL,
        ),
        # Multiline mode (though no impact here, just for coverage)
        (
            pd.DataFrame({"A": ["abc-123", "def-456"]}),
            r"([a-z]+)-(\d+)",
            True,
            pd.DataFrame({"0": ["abc", "def"], "1": ["123", "456"]}),
            re.MULTILINE,
        ),
    ],
)
def test_str_extract_expr_with_flags(df, expr, expand, expected, flags):
    bdf = bd.from_pandas(df)
    try:
        out = bdf.A.str.extract(expr, expand=expand, flags=flags)
    except ValueError:
        assert expected is None
        return
    assert out.is_lazy_plan()
    out = out.execute_plan()
    if expected is not None:
        _test_equal(out, expected, check_pandas_types=False, check_names=False)
