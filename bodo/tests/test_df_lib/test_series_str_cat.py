import pandas as pd
import pytest

import bodo.pandas as bd
from bodo.tests.utils import _test_equal


@pytest.fixture
def base_df():
    df = pd.DataFrame(
        {
            "B": [None, "A    ", "B ", "  C", "D", "E"],
            "D": [None, "1", "1", "2", "2", "3"],
            "A": [None, 1, 1, 2, 2, 3],
            "E": [None, "  a", "b  ", "c", "d", "e"],
            "C": [None, " 1", "     1     ", "h", "2", "3"],
        }
    )
    return df


@pytest.mark.parametrize(
    "lhs_expr, rhs_expr",
    [
        pytest.param(
            lambda df: df["B"].str.lower().str.capitalize().str.strip(),
            lambda df: (df["A"] + df["A"]).map(str).str.upper(),
            id="arith_expr_rhs",
        ),
        pytest.param(
            lambda df: df["C"],
            lambda df: df["C"].str.upper().str.strip(),
            id="trivial_case",
        ),
        pytest.param(
            lambda df: df["B"].str.lower().str.strip(),
            lambda df: df["C"].str.upper().str.lower().str.upper(),
            id="diff_str_chains",
        ),
        pytest.param(
            lambda df: df["E"].str.upper().str.lower(),
            lambda df: df["B"].str.lower().str.upper().str.strip(),
            id="reversible_chains",
        ),
        pytest.param(
            lambda df: df["E"].str.upper().str.strip(),
            lambda df: df["B"].str.lower().str.strip(),
            id="same_length",
        ),
        pytest.param(
            lambda df: df["E"],
            lambda df: df["B"].str.lower(),
            id="basic_colref_vs_func",
        ),
    ],
)
def test_str_cat_exprs(base_df, lhs_expr, rhs_expr):
    pdf = base_df
    bdf = bd.from_pandas(pdf)

    lhs_pd = lhs_expr(pdf)
    rhs_pd = rhs_expr(pdf)
    lhs_bd = lhs_expr(bdf)
    rhs_bd = rhs_expr(bdf)

    out_pd = lhs_pd.str.cat(others=rhs_pd, sep="-")
    out_bd = lhs_bd.str.cat(others=rhs_bd, sep="-")
    out_bd = out_bd.execute_plan()

    _test_equal(out_bd, out_pd, check_pandas_types=False, check_names=False)


@pytest.mark.skip(
    reason="TODO: fix failure in this case: df['C'] = df.A.str.cat(others=df.B) "
)
def test_assignment_str_cat_lazy_plan():
    pdf = pd.DataFrame(
        {
            "A": [None, "A", "B ", "  C", "D", "E"],
            "B": [None, "1", "1", "2", "2", "3"],
        }
    )
    bdf = bd.from_pandas(pdf)

    bdf["C"] = bdf.A.str.cat(others=bdf.B)
    pdf["C"] = pdf["A"].str.cat(others=pdf["B"])

    assert bdf.is_lazy_plan()
    _test_equal(bdf.execute_plan(), pdf, check_pandas_types=False, check_names=False)
