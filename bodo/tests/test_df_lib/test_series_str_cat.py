import pandas as pd
import pyarrow as pa
import pytest

import bodo.pandas as bd
from bodo.pandas.plan import assert_executed_plan_count
from bodo.pandas.utils import BodoLibFallbackWarning
from bodo.tests.utils import _test_equal


@pytest.fixture
def base_df():
    df_noidx = pd.DataFrame(
        {
            "B": pd.array(
                [None, "A    ", "B ", "  C", "D", "E"],
                dtype=pd.ArrowDtype(pa.large_string()),
            ),
            "D": pd.array(
                [None, "1", "1", "2", "2", "3"], dtype=pd.ArrowDtype(pa.large_string())
            ),
            "A": pd.array([None, 1, 1, 2, 2, 3], dtype=pd.ArrowDtype(pa.int64())),
            "E": pd.array(
                [None, "  a", "b  ", "c", "d", "e"],
                dtype=pd.ArrowDtype(pa.large_string()),
            ),
            "C": pd.array(
                [None, " 1", "     1     ", "h", "2", "3"],
                dtype=pd.ArrowDtype(pa.large_string()),
            ),
        },
    )
    df_idx = pd.DataFrame(
        {
            "B": pd.array(
                [None, "A    ", "B ", "  C", "D", "E"],
                dtype=pd.ArrowDtype(pa.large_string()),
            ),
            "D": pd.array(
                [None, "1", "1", "2", "2", "3"], dtype=pd.ArrowDtype(pa.large_string())
            ),
            "A": pd.array([None, 1, 1, 2, 2, 3], dtype=pd.ArrowDtype(pa.int64())),
            "E": pd.array(
                [None, "  a", "b  ", "c", "d", "e"],
                dtype=pd.ArrowDtype(pa.large_string()),
            ),
            "C": pd.array(
                [None, " 1", "     1     ", "h", "2", "3"],
                dtype=pd.ArrowDtype(pa.large_string()),
            ),
        },
        index=["a", "b", "c", "d", "e", "f"],
    )
    return [df_noidx, df_idx]


@pytest.fixture
def fallback_df():
    return pd.DataFrame(
        {
            "A": ["x", "y", "z"],
            "B": ["1", "2", "3"],
        }
    )


@pytest.mark.parametrize(
    "lhs_expr, rhs_expr, kwargs",
    [
        # TODO: implement arith expr recursive check
        # pytest.param(
        #     lambda df: df["B"].str.lower().str.capitalize().str.strip(),
        #     lambda df: (df["A"] + df["A"]).map(str).str.upper(),
        #     {},
        #     id="arith_expr_rhs",
        # ),
        pytest.param(
            lambda df: df["C"],
            lambda df: df["C"].str.upper().str.strip(),
            {},
            id="trivial_case",
        ),
        pytest.param(
            lambda df: df["B"].str.lower().str.strip(),
            lambda df: df["C"].str.upper().str.lower().str.upper(),
            {},
            id="diff_str_chains",
        ),
        pytest.param(
            lambda df: df["E"].str.upper().str.lower(),
            lambda df: df["B"].str.lower().str.upper().str.strip(),
            {},
            id="reversible_chains",
        ),
        pytest.param(
            lambda df: df["E"].str.upper().str.strip(),
            lambda df: df["B"].str.lower().str.strip(),
            {},
            id="same_length",
        ),
        pytest.param(
            lambda df: df["E"],
            lambda df: df["B"].str.lower(),
            {},
            id="basic_colref_vs_func",
        ),
        pytest.param(
            lambda df: df["B"],
            lambda df: df["C"],
            {"na_rep": "<NA>"},
            id="na_rep_fill",
        ),
        pytest.param(
            lambda df: df["B"],
            lambda df: df["C"],
            {"na_rep": ""},
            id="na_rep_empty",
        ),
        pytest.param(
            lambda df: df["B"],
            lambda df: df["C"],
            {"na_rep": "?"},
            id="na_rep_custom",
        ),
    ],
)
def test_str_cat_exprs(base_df, lhs_expr, rhs_expr, kwargs):
    for pdf in base_df:
        with assert_executed_plan_count(0):
            bdf = bd.from_pandas(pdf)

            lhs_pd = lhs_expr(pdf)
            rhs_pd = rhs_expr(pdf)
            lhs_bd = lhs_expr(bdf)
            rhs_bd = rhs_expr(bdf)

            out_pd = lhs_pd.astype(object).str.cat(
                others=rhs_pd.astype(object), sep="-", **kwargs
            )
            out_bd = lhs_bd.str.cat(others=rhs_bd, sep="-", **kwargs)
        with assert_executed_plan_count(1):
            out_bd = out_bd.execute_plan()
        _test_equal(
            out_bd,
            out_pd.astype("str").astype(pd.ArrowDtype(pa.large_string())),
            check_pandas_types=False,
            check_names=False,
        )


def test_str_cat_fallback_no_others(fallback_df):
    """Should fallback when others is None (i.e., not passed)."""
    bdf = bd.from_pandas(fallback_df)
    with pytest.warns(BodoLibFallbackWarning):
        _ = bdf["A"].str.cat()


def test_str_cat_fallback_not_bodo_series(fallback_df):
    """Should fallback when others is not a BodoSeries."""
    bdf = bd.from_pandas(fallback_df)
    with pytest.warns(BodoLibFallbackWarning):
        _ = bdf["A"].str.cat(others=fallback_df["B"])


def test_assignment_str_cat_lazy_plan():
    with assert_executed_plan_count(0):
        pdf = pd.DataFrame(
            {
                "A": [None, "A", "B ", "  C", "D", "E"],
                "B": [None, "1", "1", "2", "2", "3"],
            }
        )
        bdf = bd.from_pandas(pdf)

        bdf["C"] = bdf.A.str.cat(others=bdf.B)
        pdf["C"] = pdf["A"].str.cat(others=pdf["B"])

    _test_equal(bdf.execute_plan(), pdf, check_pandas_types=False, check_names=False)
