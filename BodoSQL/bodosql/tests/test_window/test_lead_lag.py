import numpy as np
import pytest
from bodosql.tests.test_window.window_common import (  # noqa
    null_respect_string,
    testing_locally,
)
from bodosql.tests.utils import check_query


@pytest.fixture(params=["LEAD", "LAG"])
def lead_or_lag(request):
    return request.param


def gen_lead_lag_queries(
    window,
    fill_value,
    df_len,
    lead_or_lag_value,
    null_respect_string,
    use_fill_repr=True,
):
    """Helper function, that generates a string of lead/lag queries"""

    shiftval_name_list = [
        (0, "0"),
        (1, "1"),
        (-1, "negative_1"),
        (3, "3"),
        (df_len, f"{df_len}"),
        (-df_len, f"negative_{df_len}"),
        (df_len * 2, f"{df_len*2}"),
        (-df_len * 2, f"negative_{df_len*2}"),
    ]
    if fill_value is not None:
        fill_input = repr(fill_value) if use_fill_repr else fill_value
        lead_lag_queries = ", ".join(
            [
                f"{lead_or_lag_value}(A, {x}, {fill_input}) {null_respect_string} OVER {window} as {lead_or_lag_value}_{name}"
                for x, name in shiftval_name_list
            ]
        )
    else:
        lead_lag_queries = ", ".join(
            [
                f"{lead_or_lag_value}(A, {x}) {null_respect_string} OVER {window} as {lead_or_lag_value}_{name}"
                for x, name in shiftval_name_list
            ]
            + [
                f"{lead_or_lag_value}(A) {null_respect_string} OVER {window} as {lead_or_lag_value}_default_shift",
            ]
        )
    return lead_lag_queries


@pytest.mark.parametrize(
    "fill_value",
    [pytest.param(None, id="No_fill"), pytest.param(1, id="With_fill")],
)
def test_lead_lag_consts(
    bodosql_numeric_types,
    lead_or_lag,
    spark_info,
    fill_value,
    null_respect_string,
    memory_leak_check,
):
    """tests the lead and lag aggregation functions"""

    # remove once memory leak is resolved
    df_dtype = bodosql_numeric_types["table1"]["A"].dtype
    if not (
        testing_locally
        or np.issubdtype(df_dtype, np.float64)
        or np.issubdtype(df_dtype, np.int64)
    ):
        pytest.skip("Skipped due to memory leak")

    window = "(PARTITION BY B ORDER BY C)"

    lead_lag_queries = gen_lead_lag_queries(
        window,
        fill_value,
        len(bodosql_numeric_types["table1"]["A"]),
        lead_or_lag,
        null_respect_string,
    )

    query = f"select A, B, C, {lead_lag_queries} from table1 ORDER BY B, C"

    check_query(
        query,
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "fill_value",
    [
        pytest.param(None, id="No_fill"),
        pytest.param("TIMESTAMP '2022-02-18'", id="With_fill"),
    ],
)
def test_lead_lag_consts_datetime(
    bodosql_datetime_types,
    lead_or_lag,
    spark_info,
    fill_value,
    null_respect_string,
    memory_leak_check,
):
    """tests the lead and lag aggregation functions on datetime types"""

    window = "(PARTITION BY B ORDER BY C)"
    lead_lag_queries = gen_lead_lag_queries(
        window,
        fill_value,
        len(bodosql_datetime_types["table1"]["A"]),
        lead_or_lag,
        null_respect_string,
        use_fill_repr=False,
    )
    query = f"select A, B, C, {lead_lag_queries} from table1 ORDER BY B, C"

    check_query(
        query,
        bodosql_datetime_types,
        spark_info,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "fill_value",
    [pytest.param(None, id="No_fill"), pytest.param("foo", id="With_fill")],
)
def test_lead_lag_consts_string(
    bodosql_string_types,
    lead_or_lag,
    spark_info,
    fill_value,
    null_respect_string,
    memory_leak_check,
):
    """tests the lead and lag aggregation functions on datetime types"""

    window = "(PARTITION BY B ORDER BY C)"

    lead_lag_queries = gen_lead_lag_queries(
        window,
        fill_value,
        len(bodosql_string_types["table1"]["A"]),
        lead_or_lag,
        null_respect_string,
    )

    query = f"select A, B, C, {lead_lag_queries} from table1 ORDER BY B, C"

    check_query(
        query,
        bodosql_string_types,
        spark_info,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "fill_value",
    [
        pytest.param(None, id="No_fill"),
        pytest.param(
            "X'412412'",
            id="With_fill",
            marks=pytest.mark.skip("BE-3304, no support for binary literals"),
        ),
    ],
)
def test_lead_lag_consts_binary(
    bodosql_binary_types,
    lead_or_lag,
    spark_info,
    fill_value,
    null_respect_string,
    memory_leak_check,
):
    """tests the lead and lag aggregation functions on datetime types"""

    cols_that_need_bytearray_conv = [
        "A",
        "B",
        "C",
        f"{lead_or_lag}_0",
        f"{lead_or_lag}_1",
        f"{lead_or_lag}_negative_1",
        f"{lead_or_lag}_3",
        f"{lead_or_lag}_12",
        f"{lead_or_lag}_negative_12",
        f"{lead_or_lag}_24",
        f"{lead_or_lag}_negative_24",
    ]
    if fill_value is None:
        cols_that_need_bytearray_conv.append(f"{lead_or_lag}_default_shift")

    window = "(PARTITION BY B ORDER BY C)"
    lead_lag_queries = gen_lead_lag_queries(
        window,
        fill_value,
        len(bodosql_binary_types["table1"]["A"]),
        lead_or_lag,
        null_respect_string,
    )
    query = f"select A, B, C, {lead_lag_queries} from table1 ORDER BY B, C"

    check_query(
        query,
        bodosql_binary_types,
        spark_info,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
        convert_columns_bytearray=cols_that_need_bytearray_conv,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "fill_values",
    [
        pytest.param((None, None), id="No_fill"),
        pytest.param(("Interval 3 DAYS", "259200000000000"), id="With_fill"),
    ],
)
def test_lead_lag_consts_timedelta(
    bodosql_interval_types,
    lead_or_lag,
    spark_info,
    fill_values,
    null_respect_string,
    memory_leak_check,
):
    """tests the lead and lag aggregation functions on timedelta types"""
    fill_value, spark_fill_value = fill_values

    cols_that_need_timedelta_conv = [
        "A",
        "B",
        "C",
        f"{lead_or_lag}_0",
        f"{lead_or_lag}_1",
        f"{lead_or_lag}_3",
        f"{lead_or_lag}_negative_1",
        f"{lead_or_lag}_12",
        f"{lead_or_lag}_negative_12",
        f"{lead_or_lag}_24",
        f"{lead_or_lag}_negative_24",
    ]
    if fill_value is None:
        cols_that_need_timedelta_conv.append(f"{lead_or_lag}_default_shift")

    window = "(PARTITION BY B ORDER BY C)"
    lead_lag_queries = gen_lead_lag_queries(
        window,
        fill_value,
        len(bodosql_interval_types["table1"]["A"]),
        lead_or_lag,
        null_respect_string,
        use_fill_repr=False,
    )
    query = f"select A, B, C, {lead_lag_queries} from table1 ORDER BY B, C"
    if fill_value is not None:
        # Pyspark treats timedelta as a regular integer. Convert days to nanoseconds
        spark_query = query.replace(fill_value, spark_fill_value)
    else:
        spark_query = query

    check_query(
        query,
        bodosql_interval_types,
        spark_info,
        check_dtype=False,
        check_names=False,
        convert_columns_timedelta=cols_that_need_timedelta_conv,
        only_jit_1DVar=True,
        equivalent_spark_query=spark_query,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "fill_value",
    [pytest.param(None, id="No_fill"), pytest.param("TRUE", id="With_fill")],
)
def test_lead_lag_consts_boolean(
    bodosql_boolean_types,
    lead_or_lag,
    spark_info,
    fill_value,
    null_respect_string,
    memory_leak_check,
):
    """tests the lead and lag aggregation functions on boolean types"""

    window = "(PARTITION BY B ORDER BY C)"
    lead_lag_queries = gen_lead_lag_queries(
        window,
        fill_value,
        len(bodosql_boolean_types["table1"]["A"]),
        lead_or_lag,
        null_respect_string,
        use_fill_repr=False,
    )

    query = f"select A, B, C, {lead_lag_queries} from table1 ORDER BY B, C"

    check_query(
        query,
        bodosql_boolean_types,
        spark_info,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


# @pytest.mark.skip(
#     "specifying non constant arg1 for lead/lag is not supported in spark, but currently allowed in Calcite. Can revisit this later if needed for a customer"
# )
# def test_lead_lag_variable_len(basic_df, lead_or_lag, spark_info, memory_leak_check):
#     """tests the lead and lag aggregation functions"""

#     query = f"select A, B, C, {lead_or_lag}(A, B) OVER (PARTITION BY B ORDER BY C) AS LEAD_LAG_COL from table1 ORDER BY B, C"

#     cols_to_cast = [("LEAD_LAG_COL", "float64")]
#     check_query(
#         query,
#         basic_df,
#         spark_info,
#         check_dtype=False,
#         check_names=False,
#         spark_output_cols_to_cast=cols_to_cast,
#         only_jit_1DVar=True,
#     )


# @pytest.mark.skip(
#     """TODO: Spark requires frame bound to be literal, Calcite does not have this restriction.
#     I think that adding this capability should be fairly easy, should it be needed in the future"""
# )
# def test_windowed_agg_nonconstant_values(
#     basic_df,
#     numeric_agg_funcs_subset,
#     over_clause_bounds,
#     spark_info,
#     memory_leak_check,
# ):
#     """Tests windowed aggregations works when performing aggregations, sorting by, and bounding by non constant values"""

#     # doing an orderby and calculating extra rows in the query so it's easier to tell what the error is by visual comparison
#     query = f"select A, B, C, (A + B + C) as AGG_SUM, (C + B) as ORDER_SUM, {numeric_agg_funcs_subset}(A + B + C) OVER (PARTITION BY B ORDER BY (C+B) ASC ROWS BETWEEN A PRECEDING AND C FOLLOWING) as WINDOW_AGG FROM table1 ORDER BY B, C"

#     # spark windowed min/max on integers returns an integer col.
#     # pandas rolling min/max on integer series returns a float col
#     # (and the method that we currently use returns a float col)
#     cols_to_cast = [("WINDOW_AGG", "float64")]
#     check_query(
#         query,
#         basic_df,
#         spark_info,
#         sort_output=False,
#         check_dtype=False,
#         check_names=False,
#         spark_output_cols_to_cast=cols_to_cast,
#         only_jit_1DVar=True,
#     )


# Some problematic queries that will need to be dealt with eventually:
# "SELECT CASE WHEN A > 1 THEN A * SUM(D) OVER (ORDER BY A ROWS BETWEEN 1 PRECEDING and 1 FOLLOWING) ELSE -1 END from table1"
# "SELECT MAX(A) OVER (ORDER BY A ROWS BETWEEN A PRECEDING and 1 FOLLOWING) from table1"
# "SELECT MAX(A) OVER (ORDER BY A+D ROWS BETWEEN CURRENT ROW and 1 FOLLOWING) from table1"
# SELECT 1 + MAX(A) OVER (ORDER BY A ROWS BETWEEN CURRENT ROW and 1 FOLLOWING) from table1
