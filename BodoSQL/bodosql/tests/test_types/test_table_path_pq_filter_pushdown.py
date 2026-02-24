"""
Tests filter pushdown with a parquet TablePath.
"""

import io

import pandas as pd
import pytest

import bodo
import bodosql
from bodo.ir.parquet_ext import ParquetReader
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    check_logger_no_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import (
    _check_for_io_reader_filters,
    check_func,
)
from bodo.tests.utils_jit import DistTestPipeline, SeriesOptTestPipeline
from bodosql.tests.utils import check_num_parquet_readers

pytestmark = pytest.mark.parquet


@pytest.mark.skip(reason="[BSE-787] TODO: support categorical read cast on tables")
@pytest.mark.slow
def test_table_path_filter_pushdown(datapath, memory_leak_check):
    """
    Tests basic filter pushdown support.
    """

    def impl1(f1):
        bc = bodosql.BodoSQLContext(
            {
                "TABLE1": bodosql.TablePath(f1, "parquet"),
            }
        )
        return bc.sql("Select * from table1 where part = 'b'")

    def impl2(f1):
        bc = bodosql.BodoSQLContext(
            {
                "TABLE1": bodosql.TablePath(f1, "parquet"),
            }
        )
        return bc.sql("Select A from table1 where part = 'b'")

    def impl3(filename):
        bc = bodosql.BodoSQLContext(
            {
                "TABLE1": bodosql.TablePath(filename, "parquet"),
            }
        )
        return bc.sql("Select A + 1 from table1 where part = 'b'")

    def impl4(filename):
        bc = bodosql.BodoSQLContext(
            {
                "TABLE1": bodosql.TablePath(filename, "parquet"),
            }
        )
        return bc.sql("Select A + 1 from table1 where part = 'b' and part is not null")

    filename = datapath("sample-parquet-data/partitioned")

    # Compare entirely to Pandas output to simplify the process.
    # Load the data once and then filter for each query.
    py_output = pd.read_parquet(filename, dtype_backend="pyarrow")
    py_output["part"] = py_output["part"].astype(str)
    py_output = py_output[py_output["part"] == "b"].reset_index(drop=True)
    py_output1 = py_output
    bodo_funcs = check_func(
        impl1,
        (filename,),
        py_output=py_output1,
        reset_index=True,
        additional_compiler_arguments={"pipeline_class": SeriesOptTestPipeline},
    )
    # Make sure the ParquetReader node has filters parameter set and we have trimmed
    # any unused columns.
    _check_for_io_reader_filters(bodo_funcs["seq"], ParquetReader)
    # TODO: Check which columns were actually loaded.

    py_output2 = pd.DataFrame({"A": py_output["A"]})
    check_func(impl2, (filename,), py_output=py_output2, reset_index=True)
    # make sure the ParquetReader node has filters parameter set and we have trimmed
    # any unused columns.
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl2)
    bodo_func(filename)
    _check_for_io_reader_filters(bodo_func, ParquetReader)
    # TODO: Check which columns were actually loaded.

    # TODO: Update Name when the name changes
    py_output3 = pd.DataFrame({"EXPR$0": py_output["A"] + 1})
    # don't check dtype because the output should use nullable int64 to match snowflake
    check_func(
        impl3, (filename,), py_output=py_output3, check_dtype=False, reset_index=True
    )
    # make sure the ParquetReader node has filters parameter set and we have trimmed
    # any unused columns.
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl3)
    bodo_func(filename)
    _check_for_io_reader_filters(bodo_func, ParquetReader)
    # TODO: Check which columns were actually loaded.

    # TODO: Update Name when the name changes
    py_output4 = pd.DataFrame({"EXPR$0": py_output["A"] + 1})
    # don't check dtype because the output should use nullable int64 to match snowflake
    check_func(
        impl4, (filename,), py_output=py_output4, check_dtype=False, reset_index=True
    )
    # make sure the ParquetReader node has filters parameter set and we have trimmed
    # any unused columns.
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl4)
    bodo_func(filename)
    _check_for_io_reader_filters(bodo_func, ParquetReader)
    # TODO: Check which columns were actually loaded.


@pytest.mark.skip(
    "BSE-1239 - BodoSQL column pruning breaks filter pushdown for non-snowflake catalog tables"
)
@pytest.mark.slow
def test_like_filter_pushdown(datapath, memory_leak_check):
    """
    Tests that queries with like perform filter pushdown for all the
    cases with the optimized paths.
    """
    filename = datapath("sample-parquet-data/rphd_sample.pq")
    bc = bodosql.BodoSQLContext(
        {
            "TABLE1": bodosql.TablePath(filename, "parquet"),
        }
    )

    def impl1(bc):
        # Test like where its just equality
        return bc.sql(
            "Select uuid from table1 where uuid like 'ce4d3aa7-476b-4772-94b4-18224490c7a1'"
        )

    def impl2(bc):
        # Test like where its endswith
        return bc.sql("Select uuid from table1 where uuid like '%c'")

    def impl3(bc):
        # Test like where its startswith
        return bc.sql("Select uuid from table1 where uuid like '1%'")

    def impl4(bc):
        # Test like where its contains
        return bc.sql("Select uuid from table1 where uuid like '%5%'")

    def impl5(bc):
        # Test like where its always true
        return bc.sql("Select uuid from table1 where uuid like '%'")

    # Compare entirely to Pandas output to simplify the process.
    # Load the data once and then filter for each query.
    py_output = pd.read_parquet(filename, dtype_backend="pyarrow")[["uuid"]]

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        expected_output = py_output[
            py_output.uuid == "ce4d3aa7-476b-4772-94b4-18224490c7a1"
        ]
        check_func(
            impl1, (bc,), py_output=expected_output, reset_index=True, sort_output=True
        )

        check_logger_msg(
            stream, "Filter pushdown successfully performed. Moving filter step:"
        )
        check_logger_msg(stream, "Columns loaded ['uuid']")
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        expected_output = py_output[py_output.uuid.str.endswith("c")]
        check_func(
            impl2, (bc,), py_output=expected_output, reset_index=True, sort_output=True
        )

        check_logger_msg(
            stream, "Filter pushdown successfully performed. Moving filter step:"
        )
        check_logger_msg(stream, "Columns loaded ['uuid']")
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        expected_output = py_output[py_output.uuid.str.startswith("1")]
        check_func(
            impl3, (bc,), py_output=expected_output, reset_index=True, sort_output=True
        )

        check_logger_msg(
            stream, "Filter pushdown successfully performed. Moving filter step:"
        )
        check_logger_msg(stream, "Columns loaded ['uuid']")
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        expected_output = py_output[py_output.uuid.str.contains("5")]
        check_func(
            impl4, (bc,), py_output=expected_output, reset_index=True, sort_output=True
        )

        check_logger_msg(
            stream, "Filter pushdown successfully performed. Moving filter step:"
        )
        check_logger_msg(stream, "Columns loaded ['uuid']")

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        expected_output = py_output[["uuid"]]
        check_func(
            impl5, (bc,), py_output=expected_output, reset_index=True, sort_output=True
        )

        check_logger_msg(
            stream, "Filter pushdown successfully performed. Moving filter step:"
        )
        check_logger_msg(stream, "Columns loaded ['uuid']")


@pytest.mark.skip(
    "BSE-1239 - BodoSQL column pruning breaks filter pushdown for non-snowflake catalog tables"
)
@pytest.mark.slow
def test_ilike_filter_pushdown(datapath, memory_leak_check):
    """
    Tests that queries with ilike perform filter pushdown for all the
    cases with the optimized paths.
    """
    filename = datapath("sample-parquet-data/rphd_sample.pq")
    bc = bodosql.BodoSQLContext(
        {
            "TABLE1": bodosql.TablePath(filename, "parquet"),
        }
    )

    def impl1(bc):
        # Test ilike where its just equality
        return bc.sql(
            "Select uuid from table1 where uuid ilike 'ce4d3AA7-476b-4772-94b4-18224490c7a1'"
        )

    def impl2(bc):
        # Test ilike where its endswith
        return bc.sql("Select uuid from table1 where uuid ilike '%C'")

    def impl3(bc):
        # Test ilike where its startswith
        return bc.sql("Select uuid from table1 where uuid ilike 'A1%'")

    def impl4(bc):
        # Test ilike where its contains
        return bc.sql("Select uuid from table1 where uuid ilike '%6B%'")

    # Compare entirely to Pandas output to simplify the process.
    # Load the data once and then filter for each query.
    py_output = pd.read_parquet(filename, dtype_backend="pyarrow")[["uuid"]]

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        expected_output = py_output[
            py_output.uuid == "ce4d3aa7-476b-4772-94b4-18224490c7a1"
        ]
        check_func(
            impl1, (bc,), py_output=expected_output, reset_index=True, sort_output=True
        )

        check_logger_msg(
            stream, "Filter pushdown successfully performed. Moving filter step:"
        )
        check_logger_msg(stream, "Columns loaded ['uuid']")
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        expected_output = py_output[py_output.uuid.str.endswith("c")]
        check_func(
            impl2, (bc,), py_output=expected_output, reset_index=True, sort_output=True
        )

        check_logger_msg(
            stream, "Filter pushdown successfully performed. Moving filter step:"
        )
        check_logger_msg(stream, "Columns loaded ['uuid']")
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        expected_output = py_output[py_output.uuid.str.startswith("1")]
        check_func(
            impl3, (bc,), py_output=expected_output, reset_index=True, sort_output=True
        )

        check_logger_msg(
            stream, "Filter pushdown successfully performed. Moving filter step:"
        )
        check_logger_msg(stream, "Columns loaded ['uuid']")
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        expected_output = py_output[py_output.uuid.str.contains("6b")]
        check_func(
            impl4, (bc,), py_output=expected_output, reset_index=True, sort_output=True
        )

        check_logger_msg(
            stream, "Filter pushdown successfully performed. Moving filter step:"
        )
        check_logger_msg(stream, "Columns loaded ['uuid']")


def test_coalesce_filter_pushdown(datapath, memory_leak_check):
    """
    Test coalesce support in Parquet filter pushdown
    """
    filename = datapath("date_coalesce.pq")
    bc = bodosql.BodoSQLContext(
        {
            "TABLE1": bodosql.TablePath(filename, "parquet"),
        }
    )

    def impl(bc):
        return bc.sql(
            "select * from table1 where coalesce(A, current_date()) > '2015-01-01'"
        )

    df = pd.read_parquet(filename, dtype_backend="pyarrow")
    py_output = df[
        df.A.fillna(pd.Timestamp.now().date()) > pd.to_datetime("2015-01-01").date()
    ]

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(impl, (bc,), py_output=py_output, reset_index=True, sort_output=True)
        check_logger_msg(
            stream, "Filter pushdown successfully performed. Moving filter step:"
        )
        check_logger_msg(
            stream,
            "(pa.compute.coalesce(ds.field('A'), ds.scalar(f0)) > ds.scalar(f1))",
        )


@pytest.mark.parametrize(
    "func_args",
    [
        pytest.param(("LOWER", "lower", "utf8_lower"), id="lower"),
        pytest.param(("UPPER", "upper", "utf8_upper"), id="upper"),
        pytest.param(
            ("INITCAP", "capitalize", "utf8_capitalize"),
            id="capitalize",
            marks=pytest.mark.skip(
                "[BE-4445] Additional arg in filter pushdown not supported"
            ),
        ),
    ],
)
def test_case_conversion_filter_pushdown(func_args, datapath, memory_leak_check):
    """
    Test upper, lower, initcap support in Parquet filter pushdown
    """
    sql_func, pd_func_name, arrow_func_name = func_args
    filename = datapath("string_lower_upper.pq")
    test_str_val = getattr("macedonia", pd_func_name)()

    df = pd.read_parquet(filename, dtype_backend="pyarrow")
    py_output = df[getattr(df.A.str, pd_func_name)() == test_str_val]

    bc = bodosql.BodoSQLContext(
        {
            "TABLE1": bodosql.TablePath(filename, "parquet"),
        }
    )

    def impl(bc):
        return bc.sql(f"select * from table1 where {sql_func}(A) = '{test_str_val}'")

    stream = io.StringIO()
    logger = create_string_io_logger(stream)

    with set_logging_stream(logger, 1):
        check_func(impl, (bc,), py_output=py_output, reset_index=True, sort_output=True)
        check_logger_msg(
            stream, "Filter pushdown successfully performed. Moving filter step:"
        )
        check_logger_msg(
            stream,
            f"((pa.compute.{arrow_func_name}(ds.field('A'))) == ds.scalar(f0))",
        )


def test_coalesce_lower_filter_pushdown(datapath, memory_leak_check):
    """
    Test nested coalesce and lower support in Parquet filter pushdown
    """
    filename = datapath("string_lower_upper.pq")
    test_str_val = "macedonia"
    df = pd.read_parquet(filename, dtype_backend="pyarrow")
    py_output = df[df.A.str.lower().fillna(test_str_val) == test_str_val]

    bc = bodosql.BodoSQLContext(
        {
            "TABLE1": bodosql.TablePath(filename, "parquet"),
        }
    )

    def impl(bc):
        return bc.sql(
            f"select * from table1 where coalesce(lower(A), '{test_str_val}') = '{test_str_val}'"
        )

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(impl, (bc,), py_output=py_output, reset_index=True, sort_output=True)
        check_logger_msg(
            stream, "Filter pushdown successfully performed. Moving filter step:"
        )
        # Note this is simplified in the planner to:
        # lower(A) = val OR A IS NULL
        check_logger_msg(
            stream,
            "(((pa.compute.utf8_lower(ds.field('A'))) == ds.scalar(f0)) | (ds.field('A').is_null()))",
        )


def test_upper_coalesce_filter_pushdown(datapath, memory_leak_check):
    """
    Test nested upper and coalesce support in Parquet filter pushdown
    """
    filename = datapath("string_lower_upper.pq")
    test_str_val = "macedonia"
    df = pd.read_parquet(filename, dtype_backend="pyarrow")
    py_output = df[df.A.fillna(test_str_val).str.upper() == test_str_val.upper()]

    bc = bodosql.BodoSQLContext(
        {
            "TABLE1": bodosql.TablePath(filename, "parquet"),
        }
    )

    def impl(bc):
        return bc.sql(
            f"select * from table1 where upper(coalesce(A, '{test_str_val}')) = '{test_str_val.upper()}'"
        )

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(impl, (bc,), py_output=py_output, reset_index=True, sort_output=True)
        check_logger_msg(
            stream, "Filter pushdown successfully performed. Moving filter step:"
        )
        check_logger_msg(
            stream,
            "((pa.compute.utf8_upper(pa.compute.coalesce(ds.field('A'), ds.scalar(f0)))) == ds.scalar(f1))",
        )


@pytest.mark.skip(reason="[BSE-787] TODO: support categorical read cast on tables")
@pytest.mark.slow
def test_table_path_filter_pushdown_multi_table(datapath, memory_leak_check):
    """
    Tests basic filter with multiple tables.
    """

    def impl(f1):
        bc = bodosql.BodoSQLContext(
            {
                "TABLE1": bodosql.TablePath(f1, "parquet"),
                "TABLE2": bodosql.TablePath(f1, "parquet"),
            }
        )
        return bc.sql(
            "Select table1.A as a1, table1.B as b1, table2.A as a2, table2.B as b2 from table1, table2 where table1.part = 'b' and table2.part = 'a' and table1.c = table2.c"
        )

    filename = datapath("sample-parquet-data/partitioned")

    # Compare entirely to Pandas output to simplify the process.
    # Load the data once and then filter for each query.
    py_output = pd.read_parquet(filename, dtype_backend="pyarrow")
    py_output["part"] = py_output["part"].astype(str)
    py_output_part1 = py_output[py_output["part"] == "b"]
    py_output_part2 = py_output[py_output["part"] == "a"]
    py_output = py_output_part1.merge(py_output_part2, on="C")
    py_output = pd.DataFrame(
        {
            "a1": py_output["A_x"],
            "b1": py_output["A_x"],
            "a2": py_output["A_y"],
            "b2": py_output["A_y"],
        }
    )
    check_func(
        impl, (filename,), py_output=py_output, reset_index=True, sort_output=True
    )
    # make sure the ParquetReader node has filters parameter set and we have trimmed
    # any unused columns.
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl)
    bodo_func(filename)
    _check_for_io_reader_filters(bodo_func, ParquetReader)
    # TODO: Check which columns were actually loaded.
    # At this point BodoSQL is expected to load the table twice, once for each table.
    check_num_parquet_readers(bodo_func, 2)


@pytest.mark.skip(reason="[BSE-787] TODO: support categorical read cast on tables")
@pytest.mark.slow
@pytest.mark.skipif(
    bodo.bodosql_use_streaming_plan, reason="Streaming doesn't Support Reusing Table"
)
def test_table_path_no_filter_pushdown(datapath, memory_leak_check):
    """
    Tests when filter pushdown should be rejected because a table is reused.
    """

    def impl(f1):
        bc = bodosql.BodoSQLContext(
            {
                "TABLE1": bodosql.TablePath(f1, "parquet"),
            }
        )
        return bc.sql(
            "Select t1.A as a1 from table1 as t1 inner join table1 on table1.c = t1.c where t1.part = 'a'"
        )

    filename = datapath("sample-parquet-data/partitioned")

    # Compare entirely to Pandas output to simplify the process.
    # Load the data once and then filter for each query.
    py_output = pd.read_parquet(filename, dtype_backend="pyarrow")
    py_output["part"] = py_output["part"].astype(str)
    py_output_part1 = py_output[py_output["part"] == "a"]
    py_output = py_output_part1.merge(py_output, on="C")
    py_output = pd.DataFrame(
        {
            "A1": py_output["A_x"],
        }
    )
    check_func(
        impl, (filename,), py_output=py_output, reset_index=True, sort_output=True
    )
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl)
    bodo_func(filename)
    try:
        _check_for_io_reader_filters(bodo_func, ParquetReader)
        # If we reach this line the test is wrong.
        failed = True
    except AssertionError:
        # We expect an assertion error because filter pushdown should not have occurred
        failed = False

    assert not failed


@pytest.mark.skip(reason="[BSE-787] TODO: support categorical read cast on tables")
@pytest.mark.timeout(600)
@pytest.mark.slow
def test_col_pruning_and_filter_pushdown_implicit_casting(
    datapath,
    memory_leak_check,
):
    """
    Tests that filter pushdown is correctly applied in the case that we perform implicit casting of the
    input dataframe types (done in visitTableScan)
    """

    # This dataframe has 3 columns, A -> categorical datetime64,
    # B -> categorical strings, C -> date, D -> int E -> partition column of string
    # A, B and E will be implicitly by bodosql in visitTableScan
    # Note, that
    filename = datapath("sample-parquet-data/needs_implicit_typ_conversion.pq")

    # tests filters/column pruning works on partitions
    def impl_simple_no_join_filter_partition(f1):
        bc = bodosql.BodoSQLContext(
            {
                "TABLE1": bodosql.TablePath(f1, "parquet"),
            }
        )
        return bc.sql(
            "Select table1.A, table1.B, table1.C, table1.D from table1 where table1.E='a'"
        )

    # tests filters/column pruning works on non partitions
    def impl_simple_no_join_filter_non_partition(f1):
        bc = bodosql.BodoSQLContext(
            {
                "TABLE1": bodosql.TablePath(f1, "parquet"),
            }
        )
        return bc.sql(
            "Select table1.A, table1.B, table1.C, table1.E from table1 where table1.D > 1"
        )

    # Compare entirely to Pandas output to simplify the process.
    # Load the data once and then filter for each query.
    read_df = pd.read_parquet(filename, dtype_backend="pyarrow")
    # Cast the categorical and date dtypes to the bodosql dtypes
    read_df["B"] = read_df["B"].astype(str)
    read_df["C"] = read_df["C"].astype("datetime64[ns]")
    read_df["E"] = read_df["E"].astype(str)

    stream = io.StringIO()
    logger = create_string_io_logger(stream)

    with set_logging_stream(logger, 1):
        py_output = read_df.loc[read_df["E"] == "a", ["A", "B", "C", "D"]]
        # make sure the ParquetReader node has filters parameter set and we have trimmed
        # any unused columns.
        check_func(
            impl_simple_no_join_filter_partition,
            (filename,),
            py_output=py_output,
            reset_index=True,
            sort_output=True,
        )
        # Unfortunately, we don't get information on which filters were pushed, as BodoSQL is loaded as a func text
        # TODO: find an effective way to check which filters were pushed.
        check_logger_msg(
            stream, "Filter pushdown successfully performed. Moving filter step:"
        )
        check_logger_msg(stream, "Columns loaded ['B', 'A', 'C', 'D']")
        stream.truncate(0)
        stream.seek(0)

        py_output0 = read_df.loc[read_df["D"] > 1, ["A", "B", "C", "E"]]
        # make sure the ParquetReader node has filters parameter set and we have trimmed
        # any unused columns.
        check_func(
            impl_simple_no_join_filter_non_partition,
            (filename,),
            py_output=py_output0,
            reset_index=True,
            sort_output=True,
        )
        # Unfortunately, we don't get information on which filters were pushed, as BodoSQL is loaded as a func text
        # TODO: find an effective way to check which filters were pushed.
        check_logger_msg(
            stream, "Filter pushdown successfully performed. Moving filter step:"
        )
        check_logger_msg(stream, "Columns loaded ['B', 'A', 'C', 'E']")


@pytest.mark.skip(reason="[BSE-787] TODO: support categorical read cast on tables")
@pytest.mark.timeout(600)
@pytest.mark.slow
def test_col_pruning_and_filter_pushdown_implicit_casting_multi_table(
    datapath,
    memory_leak_check,
):
    """
    Tests that filter pushdown is correctly applied in the case that we perform implicit casting of the
    input dataframe types (done in visitTableScan)
    """

    # This dataframe has 3 columns, A -> categorical datetime64,
    # B -> categorical strings, C -> date, D -> int E -> partition column of string
    # A, B and E will be implicitly by bodosql in visitTableScan
    # Note, that
    filename = datapath("sample-parquet-data/needs_implicit_typ_conversion.pq")

    # tests filters/column pruning works when filtering on partitions, with a join
    def impl_should_load_B_C_D(f1, df):
        bc = bodosql.BodoSQLContext(
            {
                "TABLE1": df,
                "TABLE2": bodosql.TablePath(f1, "parquet"),
            }
        )
        return bc.sql(
            "Select table2.B, table2.C from table1 JOIN table2 ON table1.D = table2.D where table2.E='a'"
        )

    # tests column pruning works without a filter, with a join
    def impl_should_load_A_E(f1, df):
        bc = bodosql.BodoSQLContext(
            {
                "TABLE1": df,
                "TABLE2": bodosql.TablePath(f1, "parquet"),
            }
        )
        return bc.sql("Select table2.A from table1 JOIN table2 ON table1.E = table2.E")

    # tests filters/column pruning works when no columns are loaded from table2
    def impl_should_load_None(f1, df):
        bc = bodosql.BodoSQLContext(
            {
                "TABLE1": df,
                "TABLE2": bodosql.TablePath(f1, "parquet"),
            }
        )
        return bc.sql("Select table1.A from table1 where table1.B='c'")

    # Compare entirely to Pandas output to simplify the process.
    # Load the data once and then filter for each query.
    read_df = pd.read_parquet(filename, dtype_backend="pyarrow")
    # Cast the categorical and date dtypes to the bodosql dtypes
    read_df["B"] = read_df["B"].astype(str)
    read_df["C"] = read_df["C"].astype("datetime64[ns]")
    read_df["E"] = read_df["E"].astype(str)

    stream = io.StringIO()
    logger = create_string_io_logger(stream)

    with set_logging_stream(logger, 1):
        py_output1 = read_df.copy()
        py_output1 = py_output1.merge(py_output1, on="D")
        py_output1 = py_output1.loc[py_output1["E_y"] == "a", ["B_x", "C_y"]].rename(
            columns={"B_x": "B", "C_y": "C"}
        )
        # make sure the ParquetReader node has filters parameter set and we have trimmed
        # any unused columns.
        check_func(
            impl_should_load_B_C_D,
            (filename, read_df),
            py_output=py_output1,
            reset_index=True,
            sort_output=True,
        )
        # Unfortunately, we don't get information on which filters were pushed, as BodoSQL is loaded as a func text
        # TODO: find an effective way to check which filters were pushed.
        check_logger_msg(
            stream, "Filter pushdown successfully performed. Moving filter step:"
        )
        check_logger_msg(stream, "Columns loaded ['B', 'C', 'D']")
        stream.truncate(0)
        stream.seek(0)

        py_output2 = read_df.copy()
        py_output2 = py_output2.merge(py_output2, on="E")
        py_output2 = py_output2.loc[:, ["A_x"]].rename(columns={"A_x": "A"})

        check_func(
            impl_should_load_A_E,
            (filename, read_df),
            py_output=py_output2,
            reset_index=True,
            sort_output=True,
        )
        # We expect no filter pushdown
        check_logger_no_msg(
            stream, "Filter pushdown successfully performed. Moving filter step:"
        )
        check_logger_msg(stream, "Columns loaded ['A', 'E']")
        stream.truncate(0)
        stream.seek(0)

        py_output3 = read_df.copy()
        py_output3 = py_output3.loc[py_output3["B"] == "c", ["A"]]

        # make sure the ParquetReader node has filters parameter set and we have trimmed
        # any unused columns.
        check_func(
            impl_should_load_None,
            (filename, read_df),
            py_output=py_output3,
            reset_index=True,
            sort_output=True,
        )
        check_logger_no_msg(stream, "Columns loaded")


@pytest.mark.skip(reason="[BSE-787] TODO: support categorical read cast on tables")
@pytest.mark.slow
def test_table_path_col_pruning_simple(datapath, memory_leak_check):
    """
    Tests that column pruning is correctly applied in the case that we perform implicit casting of the
    input dataframe types (done in visitTableScan)
    """

    # This dataframe has 3 columns, A -> categorical datetime64,
    # B -> categorical strings, C -> Datetype, D -> int E -> partition column of string
    # A, B, and C will be implicitly cast by bodosql in visitTableScan
    filename = datapath("sample-parquet-data/needs_implicit_typ_conversion.pq")

    def impl_simple_only_A(f1):
        bc = bodosql.BodoSQLContext(
            {
                "TABLE1": bodosql.TablePath(f1, "parquet"),
            }
        )
        return bc.sql("Select table1.A from table1")

    def impl_simple_only_D(f1):
        bc = bodosql.BodoSQLContext(
            {
                "TABLE1": bodosql.TablePath(f1, "parquet"),
            }
        )
        return bc.sql("Select table1.D from table1")

    # Compare entirely to Pandas output to simplify the process.
    # Load the data once and then filter for each query.
    read_df = pd.read_parquet(filename, dtype_backend="pyarrow")
    # Cast the categorical and date dtypes to the bodosql dtypes
    read_df["B"] = read_df["B"].astype(str)
    read_df["C"] = read_df["C"].astype("datetime64[ns]")
    read_df["E"] = read_df["E"].astype(str)

    stream = io.StringIO()
    logger = create_string_io_logger(stream)

    with set_logging_stream(logger, 1):
        py_output1 = read_df.loc[:, ["A"]]
        # make sure the ParquetReader node has filters parameter set and we have trimmed
        # any unused columns.
        check_func(
            impl_simple_only_A,
            (filename,),
            py_output=py_output1,
            reset_index=True,
            sort_output=True,
        )
        # TODO: remove this filter column (E) from columns loaded
        # it currently is still loaded due to the fact that it is of type string
        check_logger_msg(stream, "Columns loaded ['A']")
        stream.truncate(0)
        stream.seek(0)

        py_output0 = read_df.loc[:, ["D"]]
        # make sure the ParquetReader node has filters parameter set and we have trimmed
        # any unused columns.
        check_func(
            impl_simple_only_D,
            (filename,),
            py_output=py_output0,
            reset_index=True,
            sort_output=True,
        )
        check_logger_msg(stream, "Columns loaded ['D']")


@pytest.mark.slow
@pytest.mark.skipif(
    bodo.bodosql_use_streaming_plan,
    reason="Parquet Streaming doesn't Support Limit Pushdown",
)
def test_table_path_limit_pushdown(datapath, memory_leak_check):
    """
    Test basic limit pushdown support.
    """

    # select columns
    def impl1(f1):
        bc = bodosql.BodoSQLContext(
            {
                "TABLE1": bodosql.TablePath(f1, "parquet"),
            }
        )
        return bc.sql("Select A, B from table1 limit 5")

    # all columns
    def impl2(f1):
        bc = bodosql.BodoSQLContext(
            {
                "TABLE1": bodosql.TablePath(f1, "parquet"),
            }
        )
        return bc.sql("Select * from table1 limit 5")

    # TODO[BE-3581]: support and test limit pushdown for partitioned Parquet datasets
    # filename = "BodoSQL/bodosql/tests/data/sample-parquet-data/partitioned"
    filename = datapath("sample-parquet-data/no_index.pq")

    py_output = pd.read_parquet(filename, dtype_backend="pyarrow")[["A", "B"]].head(5)
    check_func(
        impl1, (filename,), py_output=py_output, reset_index=True, check_dtype=False
    )

    # make sure limit pushdown worked
    bodo_func = bodo.jit(pipeline_class=DistTestPipeline)(impl1)
    bodo_func(filename)
    fir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert hasattr(fir, "meta_head_only_info")
    assert fir.meta_head_only_info[0] is not None

    py_output = pd.read_parquet(filename, dtype_backend="pyarrow").head(5)
    check_func(
        impl2, (filename,), py_output=py_output, reset_index=True, check_dtype=False
    )

    # make sure limit pushdown worked
    bodo_func = bodo.jit(pipeline_class=DistTestPipeline)(impl2)
    bodo_func(filename)
    fir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert hasattr(fir, "meta_head_only_info")
    assert fir.meta_head_only_info[0] is not None


@pytest.mark.skip(reason="[BSE-787] TODO: support categorical read cast on tables")
def test_named_param_filter_pushdown(datapath, memory_leak_check):
    """
    Test that using a Python variable as a filter variable via the named
    parameter supports filter pushdown.
    """
    filename = datapath("sample-parquet-data/needs_implicit_typ_conversion.pq")

    def impl(f1, val):
        bc = bodosql.BodoSQLContext(
            {
                "TABLE1": bodosql.TablePath(f1, "parquet"),
            }
        )
        return bc.sql(
            "Select table1.A, table1.B, table1.C, table1.D from table1 where table1.E=@pyval",
            {"pyval": val},
        )

    # Compare entirely to Pandas output to simplify the process.
    read_df = pd.read_parquet(filename, dtype_backend="pyarrow")
    # Cast the categorical and date dtypes to the bodosql dtypes
    read_df["B"] = read_df["B"].astype(str)
    read_df["C"] = read_df["C"].astype("datetime64[ns]")
    read_df["E"] = read_df["E"].astype(str)
    py_output = read_df.loc[read_df["E"] == "e", ["A", "B", "C", "D"]]

    check_func(
        impl,
        (filename, "e"),
        py_output=py_output,
        reset_index=True,
        sort_output=True,
    )

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(impl)
        bodo_func(filename, "e")
        check_logger_msg(
            stream, "Filter pushdown successfully performed. Moving filter step:"
        )
        check_logger_msg(stream, "Columns loaded ['B', 'A', 'C', 'D']")


@pytest.mark.slow
@pytest.mark.skipif(
    bodo.bodosql_use_streaming_plan,
    reason="Parquet Streaming doesn't Support Limit Pushdown",
)
def test_table_path_limit_pushdown_complex(datapath, memory_leak_check):
    """
    Tests that limit pushdown works with a possible complicated projection.
    """

    def impl(f1):
        bc = bodosql.BodoSQLContext(
            {
                "TABLE1": bodosql.TablePath(f1, "parquet"),
            }
        )
        return bc.sql(
            "Select A + 1 as A, B as newCol1, 'my_name' as newCol2 from table1 limit 5"
        )

    filename = datapath("sample-parquet-data/no_index.pq")
    py_output = pd.read_parquet(filename, dtype_backend="pyarrow")[["A", "B"]]
    py_output["A"] += 1
    py_output = py_output.rename(columns={"B": "newCol1"}, copy=False)
    py_output["newCol2"] = "my_name"
    py_output = py_output.head(5)
    check_func(
        impl, (filename,), py_output=py_output, reset_index=True, check_dtype=False
    )

    # make sure limit pushdown worked
    bodo_func = bodo.jit(pipeline_class=DistTestPipeline)(impl)
    bodo_func(filename)
    fir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert hasattr(fir, "meta_head_only_info")
    assert fir.meta_head_only_info[0] is not None


@pytest.mark.skip(
    "BSE-1239 - BodoSQL column pruning breaks filter pushdown for non-snowflake catalog tables"
)
@pytest.mark.slow
def test_boolean_logic_filter_pushdown(datapath, memory_leak_check):
    """
    Tests that filter pushdown works with different boolean logic expressions of conditionals

    Different boolean logic expressions as followed:
        - A AND B,
        - A AND (B OR C)
        - A OR B
        - A OR (B AND C)
        - (A AND B) OR C
        - (A AND B) OR (C AND D)
        - (A OR B) AND C
        - (A OR B) AND (C OR D)

    where A, B, C, D are conditional expressions
    """

    def impl(filename, query):
        bc = bodosql.BodoSQLContext(
            {
                "TABLE1": bodosql.TablePath(filename, "parquet"),
            }
        )
        return bc.sql(query)

    filename = datapath("tpch-test-data/parquet/lineitem.pq")
    read_df = pd.read_parquet(filename, dtype_backend="pyarrow")

    expr_a = "L_ORDERKEY > 10"
    expr_b = "L_LINENUMBER = 3"
    expr_c = "L_COMMENT != 'SHIP'"
    expr_d = "L_SHIPMODE = 'SHIP'"

    cond_a = read_df["L_ORDERKEY"] > 10
    cond_b = read_df["L_LINENUMBER"] == 3
    cond_c = read_df["L_COMMENT"] != "SHIP"
    cond_d = read_df["L_SHIPMODE"] == "SHIP"

    out_cols = ["L_ORDERKEY", "L_LINENUMBER"]
    select_string = f"Select {out_cols[0]}, {out_cols[1]} from table1 where"

    queries = [
        f"{select_string} {expr_a} AND {expr_b}",
        f"{select_string} {expr_a} AND ({expr_b} OR {expr_c})",
        f"{select_string} {expr_a} OR {expr_b}",
        f"{select_string} {expr_a} OR ({expr_b} AND {expr_c})",
        f"{select_string} ({expr_a} AND {expr_b}) OR {expr_c}",
        f"{select_string} ({expr_a} AND {expr_b}) OR ({expr_c} AND {expr_d})",
        f"{select_string} ({expr_a} OR {expr_b}) AND {expr_c}",
        f"{select_string} ({expr_a} OR {expr_b}) AND ({expr_c} OR {expr_d})",
    ]

    py_outputs = [
        read_df.loc[cond_a & cond_b][out_cols],
        read_df.loc[cond_a & (cond_b | cond_c)][out_cols],
        read_df.loc[cond_a | cond_b][out_cols],
        read_df.loc[cond_a | (cond_b & cond_c)][out_cols],
        read_df.loc[(cond_a & cond_b) | (cond_c)][out_cols],
        read_df.loc[(cond_a & cond_b) | (cond_c & cond_d)][out_cols],
        read_df.loc[(cond_a | cond_b) & (cond_c)][out_cols],
        read_df.loc[(cond_a | cond_b) & (cond_c | cond_d)][out_cols],
    ]

    for i, query in enumerate(queries):
        check_func(
            impl,
            (
                filename,
                query,
            ),
            py_output=py_outputs[i],
            reset_index=True,
            check_dtype=False,
        )

        stream = io.StringIO()
        logger = create_string_io_logger(stream)

        with set_logging_stream(logger, 1):
            bodo_func = bodo.jit(impl)
            bodo_func(filename, query)
            check_logger_msg(stream, "Filter pushdown successfully performed. ")


@pytest.mark.skip(reason="[BSE-787] TODO: support categorical read cast on tables")
def test_in_filter_pushdown(datapath):
    """
    Basic test for filter pushdown of the bodosql in kernel. Equivalent correctness/codegen
    checks can be found in BodoSQL/bodosql/tests/test_in.py
    """

    test_in_query = """ SELECT * FROM table1 where part in ('a', 'b', 'Z')"""
    filepath = datapath("sample-parquet-data/partitioned")
    bc = bodosql.BodoSQLContext(
        {
            "TABLE1": bodosql.TablePath(filepath, "parquet"),
        }
    )

    def impl(bc, test_in_query):
        return bc.sql(test_in_query)

    # Compare entirely to Pandas output to simplify the process.
    # Load the data once and then filter for each query.
    py_output = pd.read_parquet(filepath, dtype_backend="pyarrow")
    py_output["part"] = py_output["part"].astype(str)
    py_output = py_output[(py_output["part"] == "a") | (py_output["part"] == "b")]

    stream = io.StringIO()
    logger = create_string_io_logger(stream)

    with set_logging_stream(logger, 1):
        check_func(impl, (bc, test_in_query), py_output=py_output, reset_index=True)
        check_logger_msg(stream, "Filter pushdown successfully performed.")


@pytest.mark.skip(
    "BSE-1239 - BodoSQL column pruning breaks filter pushdown for non-snowflake catalog tables"
)
@pytest.mark.slow
def test_in_filter_pushdown_e2e(datapath):
    """
    end to end test for filter pushdown of the bodosql in kernel. The equivalent correctness/codegen
    check can be found in BodoSQL/bodosql/tests/test_in.py::test_in_e2e. This version of
    the query has been slightly modified, to make sure that the 'X in Y' clause
    is only filter that can be pushed
    """

    test_in_query = """
    SELECT
        r.engineering_prod_cd as engineering_prod_cd,
        r.unit_serial_id as unit_serial_id,
        r.unit_plant_cd as unit_plant_cd,
        r.unit_manufacturing_week_nr as unit_manufacturing_week_nr,
        r.unit_config_cd as unit_config_cd,
        r.unit_type_cd as unit_type_cd,
        r.module_part_id as module_part_id,
        r.module_part_desc as module_part_desc,
        r.module_serial_id as module_serial_id,
        r.module_plant_cd as module_plant_cd,
        r.module_manufacturing_week_nr as module_manufacturing_week_nr,
        r.module_config_cd as module_config_cd,
        r.lot_cd as lot_cd,
        r.level_2_tech_comment_txt as level_2_tech_comment_txt,
        r.dt_cd as dt_cd,
        r.factory_local_ts as factory_local_ts,
        r.module_scan_type_cd as module_scan_type_cd,
        r.etl_change_batch_sk as etl_change_batch_sk,
        r.unit_manufacturing_day_nr as unit_manufacturing_day_nr,
        r.active_ind as active_ind,
        r.info_desc as info_desc,
        r.categ_cd as categ_cd,
        r.uu_id as uu_id,
        r.last_modified_dt as last_modified_dt,
        r.last_modified_ts as last_modified_ts,
        r.site_id as site_id,
        r.version_id as version_id,
        r.sending_type_cd as sending_type_cd,
        r.filter_flag as filter_flag
    FROM PDCA_MODULE AS r JOIN (
    SELECT
        t.engineering_prod_cd,
        t.unit_serial_id,
        t.unit_plant_cd,
        t.unit_manufacturing_week_nr,
        t.unit_config_cd,
        t.unit_type_cd,
        t.module_part_id,
        t.module_part_desc,
        t.module_serial_id,
        t.module_plant_cd,
        t.module_manufacturing_week_nr,
        t.module_config_cd,
        t.lot_cd,
        t.level_2_tech_comment_txt,
        t.dt_cd,
        t.factory_local_ts,
        t.module_scan_type_cd,
        t.etl_change_batch_sk,
        t.unit_manufacturing_day_nr,
        t.active_ind,
        t.info_desc,
        t.categ_cd,
        t.uu_id,
        t.last_modified_dt,
        t.last_modified_ts,
        t.site_id,
        t.version_id,
        t.sending_type_cd,
        t.filter_flag
    FROM
        WRK_MODULE_FATP t
    ) AS s ON r.engineering_prod_cd = s.engineering_prod_cd
    AND r.unit_serial_id = s.unit_serial_id
    AND r.module_part_id = s.module_part_id
    AND r.module_serial_id = s.module_serial_id
    AND r.lot_cd = s.lot_cd
    AND r.dt_cd = s.dt_cd
    AND r.factory_local_ts = s.factory_local_ts
    AND r.info_desc = s.info_desc
    AND r.active_ind = s.active_ind
    AND r.engineering_prod_cd in ('X2017', 'J407', 'D1763CG', 'J181', 'X1441', 'X1891', 'D5XCGA', 'D54', 'D52', 'R965', 'X1863', 'X1462', 'B222', 'D22', 'B298', 'X1653', 'J171', 'X1650', 'D17CSA',
        'D351', 'D167', 'D16PAM', 'D221', 'J172', 'X1450', 'X1457', 'D49H', 'N188S', 'D17H', 'X2010B', 'N121S', 'B427A', 'B494', 'R661', 'N187B', 'X2061B', 'R761', 'X1866', 'D059', 'B635', 'D292', 'J311', 'D53A', 'X1814', 'X1458', 'X1914',
        'B837A', 'X1673', 'N158B', 'X1497', 'D64H', 'D11', 'B520A', 'X1864', 'X1416', 'X1657S', 'X1934', 'X2071', 'X2097', 'N157S', 'X1916', 'J375', 'R831', 'X1856', 'D53GH', 'X1930', 'X2010S', 'D21', 'D17PAM', 'D17A', 'D64A', 'X1931',
        'B332', 'N157', 'D32', 'X2316', 'R765', 'D166', 'D63A', 'J522', 'X1666', 'J517', 'D64CSA', 'D101', 'D17-DKF', 'D63', 'X1443', 'D43', 'X2061', 'B288', 'N158S', 'X1442', 'X1779', 'X1871', 'B390', 'X1888', 'N142S', 'A149', 'D53P',
        'B520', 'D211', 'B688', 'X1769', 'D293', 'X1887', 'N140S', 'N142B', 'R865', 'X1417', 'D63-DKF', 'X1806', 'N140B', 'J518', 'B519', 'J42B', 'X1658B', 'X1699', 'D79', 'R665', 'D16', 'D54H', 'X1483', 'X2070', 'N121B', 'D63CSA',
        'D64PAM', 'X2011B', 'X898', 'J523', 'X1940', 'X1680', 'R631', 'N104H', 'D16A', 'X1406', 'X1818B', 'D16H', 'X2012', 'D16CSA', 'X2138', 'B515', 'J182', 'B507', 'B389', 'X2061A', 'N104', 'J307', 'X1819B', 'X1819S', 'D42', 'J305',
        'X897', 'X1642', 'N187S', 'X936', 'B882', 'B508', 'D52A', 'J524', 'X2013', 'J310', 'N157B', 'X935', 'X2571', 'B937', 'B372', 'X1879', 'N188B', 'X2125', 'X1657B', 'D64', 'D53', 'D10', 'D280', 'D64-DKF', 'X2007', 'B235', 'X1861',
        'D63PAM', 'X1818S', 'X1862', 'D17', 'J71S')
    """

    filepath1 = datapath(
        "sample-parquet-data/apple_sample_data/PDCA_MODULE_source_n1000_dest_n10000_match_percent70_null_percent10.pq/"
    )
    filepath2 = datapath(
        "sample-parquet-data/apple_sample_data/WRK_MODULE_FATP_source_n1000_dest_n10000_match_percent70_null_percent10.pq/"
    )

    expected_output_path = datapath(
        "sample-parquet-data/apple_sample_data/expected_query_output.pq"
    )
    expected_output = pd.read_parquet(expected_output_path, dtype_backend="pyarrow")

    bc = bodosql.BodoSQLContext(
        {
            "PDCA_MODULE": bodosql.TablePath(filepath1, "parquet"),
            "WRK_MODULE_FATP": bodosql.TablePath(filepath2, "parquet"),
        }
    )

    def impl(bc, test_in_query):
        return bc.sql(test_in_query)

    assert "bodosql.kernels.is_in" in bc.convert_to_pandas(test_in_query)

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(
            impl,
            (bc, test_in_query),
            py_output=expected_output,
            check_names=True,
            check_dtype=False,
            only_1DVar=True,  # Only running 1d var just to save time, since this is a larger query
            reset_index=True,
            sort_output=True,
        )
        check_logger_msg(stream, "Filter pushdown successfully performed.")


@pytest.mark.skip(reason="[BSE-787] TODO: support categorical read cast on tables")
def test_not_in_filter_pushdown(datapath):
    """
    Basic test for filter pushdown of NOT IN.
    """

    test_in_query = """ SELECT * FROM table1 where part not in ('a', 'b', 'Z')"""
    filepath = datapath("sample-parquet-data/partitioned")
    bc = bodosql.BodoSQLContext(
        {
            "TABLE1": bodosql.TablePath(filepath, "parquet"),
        }
    )

    def impl(bc, test_in_query):
        return bc.sql(test_in_query)

    # Compare entirely to Pandas output to simplify the process.
    # Load the data once and then filter for each query.
    py_output = pd.read_parquet(filepath, dtype_backend="pyarrow")
    py_output["part"] = py_output["part"].astype(str)
    py_output = py_output[~py_output["part"].isin(["a", "b", "Z"])]

    stream = io.StringIO()
    logger = create_string_io_logger(stream)

    with set_logging_stream(logger, 1):
        check_func(impl, (bc, test_in_query), py_output=py_output, reset_index=True)
        check_logger_msg(stream, "Filter pushdown successfully performed.")


@pytest.mark.skip(
    "BSE-1239 - BodoSQL column pruning breaks filter pushdown for non-snowflake catalog tables"
)
def test_not_like_filter_pushdown(datapath, memory_leak_check):
    """
    Tests that queries with not like perform filter pushdown for all the
    cases with the optimized paths.
    """
    filename = datapath("sample-parquet-data/rphd_sample.pq")
    bc = bodosql.BodoSQLContext(
        {
            "TABLE1": bodosql.TablePath(filename, "parquet"),
        }
    )

    def impl(bc, query):
        return bc.sql(query)

    # Compare entirely to Pandas output to simplify the process.
    # Load the data once and then filter for each query.
    py_output = pd.read_parquet(filename, dtype_backend="pyarrow")[["uuid"]]

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        expected_output = py_output[
            ~(py_output.uuid == "ce4d3aa7-476b-4772-94b4-18224490c7a1")
        ]
        query1 = "Select uuid from table1 where uuid not like 'ce4d3aa7-476b-4772-94b4-18224490c7a1'"
        check_func(
            impl,
            (bc, query1),
            py_output=expected_output,
            reset_index=True,
            sort_output=True,
        )
        check_logger_msg(
            stream, "Filter pushdown successfully performed. Moving filter step:"
        )
        check_logger_msg(stream, "Columns loaded ['uuid']")
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        expected_output = py_output[~py_output.uuid.str.endswith("c")]
        query2 = "Select uuid from table1 where uuid not like '%c'"
        check_func(
            impl,
            (bc, query2),
            py_output=expected_output,
            reset_index=True,
            sort_output=True,
        )
        check_logger_msg(
            stream, "Filter pushdown successfully performed. Moving filter step:"
        )
        check_logger_msg(stream, "Columns loaded ['uuid']")
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        expected_output = py_output[~py_output.uuid.str.startswith("1")]
        query3 = "Select uuid from table1 where uuid not like '1%'"
        check_func(
            impl,
            (bc, query3),
            py_output=expected_output,
            reset_index=True,
            sort_output=True,
        )
        check_logger_msg(
            stream, "Filter pushdown successfully performed. Moving filter step:"
        )
        check_logger_msg(stream, "Columns loaded ['uuid']")
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        expected_output = py_output[~py_output.uuid.str.contains("5")]
        query4 = "Select uuid from table1 where uuid not like '%5%'"
        check_func(
            impl,
            (bc, query4),
            py_output=expected_output,
            reset_index=True,
            sort_output=True,
        )
        check_logger_msg(
            stream, "Filter pushdown successfully performed. Moving filter step:"
        )
        check_logger_msg(stream, "Columns loaded ['uuid']")


@pytest.mark.skip(
    "BSE-1239 - BodoSQL column pruning breaks filter pushdown for non-snowflake catalog tables"
)
def test_not_ilike_filter_pushdown(datapath, memory_leak_check):
    """
    Tests that queries with not ilike perform filter pushdown for all the
    cases with the optimized paths.
    """
    filename = datapath("sample-parquet-data/rphd_sample.pq")
    bc = bodosql.BodoSQLContext(
        {
            "TABLE1": bodosql.TablePath(filename, "parquet"),
        }
    )

    def impl(bc, query):
        return bc.sql(query)

    # Compare entirely to Pandas output to simplify the process.
    # Load the data once and then filter for each query.
    py_output = pd.read_parquet(filename, dtype_backend="pyarrow")[["uuid"]]

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        expected_output = py_output[
            ~(py_output.uuid == "ce4d3aa7-476b-4772-94b4-18224490c7a1")
        ]
        query1 = "Select uuid from table1 where uuid not ilike 'ce4d3AA7-476b-4772-94b4-18224490c7a1'"
        check_func(
            impl,
            (bc, query1),
            py_output=expected_output,
            reset_index=True,
            sort_output=True,
        )

        check_logger_msg(
            stream, "Filter pushdown successfully performed. Moving filter step:"
        )
        check_logger_msg(stream, "Columns loaded ['uuid']")
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        expected_output = py_output[~py_output.uuid.str.endswith("c")]
        query2 = "Select uuid from table1 where uuid not ilike '%C'"
        check_func(
            impl,
            (bc, query2),
            py_output=expected_output,
            reset_index=True,
            sort_output=True,
        )
        check_logger_msg(
            stream, "Filter pushdown successfully performed. Moving filter step:"
        )
        check_logger_msg(stream, "Columns loaded ['uuid']")
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        expected_output = py_output[~py_output.uuid.str.lower().str.startswith("a1")]
        query3 = "Select uuid from table1 where uuid not ilike 'A1%'"
        check_func(
            impl,
            (bc, query3),
            py_output=expected_output,
            reset_index=True,
            sort_output=True,
        )
        check_logger_msg(
            stream, "Filter pushdown successfully performed. Moving filter step:"
        )
        check_logger_msg(stream, "Columns loaded ['uuid']")
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        expected_output = py_output[~py_output.uuid.str.contains("6b")]
        query4 = "Select uuid from table1 where uuid not ilike '%6B%'"
        check_func(
            impl,
            (bc, query4),
            py_output=expected_output,
            reset_index=True,
            sort_output=True,
        )
        check_logger_msg(
            stream, "Filter pushdown successfully performed. Moving filter step:"
        )
        check_logger_msg(stream, "Columns loaded ['uuid']")


@pytest.mark.skip(
    "BSE-1239 - BodoSQL column pruning breaks filter pushdown for non-snowflake catalog tables"
)
@pytest.mark.slow
def test_multiple_loads_filter_pushdown(datapath, memory_leak_check):
    """
    Tests that multiple loads of the same table with different filters
    results in filter pushdown succeeding, and two different
    pd.read_parquet calls.
    """

    def impl(bc, query):
        return bc.sql(query)

    filename = datapath("tpch-test-data/parquet/lineitem.pq")
    read_df = pd.read_parquet(filename, dtype_backend="pyarrow")
    t1 = read_df[read_df.L_LINENUMBER == 3]
    t2 = read_df[read_df.L_SHIPMODE == "SHIP"]
    join_output = t1.merge(t2, on="L_ORDERKEY")
    expected_output = pd.DataFrame({"cnt": [len(join_output)]})

    test_query = """
        select count(*) as cnt from table1 t1
            INNER JOIN table1 t2
            ON t1.L_ORDERKEY = t2.L_ORDERKEY
        WHERE t1.L_LINENUMBER = 3 and t2.L_SHIPMODE = 'SHIP'"""

    bc = bodosql.BodoSQLContext(
        {
            "TABLE1": bodosql.TablePath(filename, "parquet"),
        }
    )
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(
            impl,
            (bc, test_query),
            py_output=expected_output,
            is_out_distributed=False,
        )
        check_logger_msg(stream, "Filter pushdown successfully performed.")

    generate_code = bc.convert_to_pandas(test_query)
    assert generate_code.count("pd.read_parquet") == 2, "Expected 2 read parquet calls"


@pytest.mark.skip(
    "BSE-1239 - BodoSQL column pruning breaks filter pushdown for non-snowflake catalog tables"
)
def test_length_filter_pushdown(datapath, memory_leak_check):
    """
    Tests that queries with all aliases of LENGTH work with
    filter pushdown.
    """

    def impl(bc, query):
        return bc.sql(query)

    filename = datapath("tpch-test-data/parquet/lineitem.pq")
    read_df = pd.read_parquet(filename, dtype_backend="pyarrow")
    expected_output = read_df[read_df.L_SHIPMODE.str.len() == 4][["L_LINENUMBER"]]
    bc = bodosql.BodoSQLContext(
        {
            "TABLE1": bodosql.TablePath(filename, "parquet"),
        }
    )

    for func_name in ("CHAR_LENGTH", "LENGTH", "LEN", "CHARACTER_LENGTH"):
        test_query = f"""
            select L_LINENUMBER from table1 t1
            WHERE {func_name}(t1.L_SHIPMODE) = 4
        """
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            check_func(
                impl,
                (bc, test_query),
                py_output=expected_output,
                is_out_distributed=False,
                reset_index=True,
                sort_output=True,
                # Pandas output is non-nullable
                check_dtype=False,
            )
            check_logger_msg(stream, "Filter pushdown successfully performed.")
            check_logger_msg(stream, "Columns loaded ['L_LINENUMBER']")


@pytest.mark.parametrize(
    "func_args",
    [
        pytest.param(
            ("LTRIM", "lstrip", "utf8_ltrim_whitespace"),
            id="LTRIM",
        ),
        pytest.param(
            ("RTRIM", "rstrip", "utf8_rtrim_whitespace"),
            id="RTRIM",
        ),
        pytest.param(
            ("TRIM", "strip", "utf8_trim_whitespace"),
            id="TRIM",
        ),
    ],
)
@pytest.mark.skip(
    "[BE-4445] Extra arg in compute func not supported for filter pushdown"
)
def test_trim_filter_pushdown(func_args, datapath, memory_leak_check):
    """
    Tests that queries with all variations of TRIM work with
    filter pushdown (no optional chars).
    """
    sql_func_name, pd_func_name, arrow_func_name = func_args

    def impl(bc, query):
        return bc.sql(query)

    filename = datapath("tpch-test-data/parquet/lineitem.pq")
    read_df = pd.read_parquet(filename, dtype_backend="pyarrow")

    shipmode_col = getattr(read_df.L_SHIPMODE.str, pd_func_name)()
    expected_output = read_df[shipmode_col == "SHIP"][["L_ORDERKEY"]]

    bc = bodosql.BodoSQLContext(
        {
            "TABLE1": bodosql.TablePath(filename, "parquet"),
        }
    )

    test_query = f"""
        select L_ORDERKEY from table1 t1
        WHERE {sql_func_name}(t1.L_SHIPMODE) = 'SHIP'
    """
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(
            impl,
            (bc, test_query),
            py_output=expected_output,
            is_out_distributed=False,
            reset_index=True,
            sort_output=True,
            # Pandas output is non-nullable
            check_dtype=False,
        )
        check_logger_msg(stream, "Filter pushdown successfully performed.")
        check_logger_msg(stream, "Columns loaded ['L_ORDERKEY']")
        check_logger_msg(
            stream,
            f"(((pa.compute.{arrow_func_name}(ds.field('L_SHIPMODE')) == ds.scalar(f1))))",
        )


@pytest.mark.skip(
    "BSE-1239 - BodoSQL column pruning breaks filter pushdown for non-snowflake catalog tables"
)
def test_reverse_filter_pushdown(datapath, memory_leak_check):
    """
    Test reverse support in Parquet filter pushdown
    """

    def impl(bc, query):
        return bc.sql(query)

    filename = datapath("tpch-test-data/parquet/lineitem.pq")
    read_df = pd.read_parquet(filename, dtype_backend="pyarrow")

    shipmode_col = read_df.L_SHIPMODE.apply(lambda x: x[::-1])
    expected_output = read_df[shipmode_col == "PIHS"][["L_ORDERKEY"]]

    bc = bodosql.BodoSQLContext(
        {
            "TABLE1": bodosql.TablePath(filename, "parquet"),
        }
    )

    test_query = """
        select L_ORDERKEY from table1 t1
        WHERE REVERSE(t1.L_SHIPMODE) = 'PIHS'
    """
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        check_func(
            impl,
            (bc, test_query),
            py_output=expected_output,
            is_out_distributed=False,
            reset_index=True,
            sort_output=True,
            # Pandas output is non-nullable
            check_dtype=False,
        )
        check_logger_msg(stream, "Filter pushdown successfully performed.")
        check_logger_msg(stream, "Columns loaded ['L_ORDERKEY']")
        check_logger_msg(
            stream,
            "(((pa.compute.utf8_reverse(ds.field('L_SHIPMODE')) == ds.scalar(f1))))",
        )
