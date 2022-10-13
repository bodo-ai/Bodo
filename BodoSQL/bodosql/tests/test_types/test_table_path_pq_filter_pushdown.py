# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Tests filter pushdown with a parquet TablePath.
"""
import io

import bodosql
import pandas as pd
import pytest
from bodosql.tests.utils import check_num_parquet_readers

import bodo
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    check_logger_no_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import (
    DistTestPipeline,
    SeriesOptTestPipeline,
    _check_for_io_reader_filters,
    check_func,
)


@pytest.mark.slow
def test_table_path_filter_pushdown(datapath, memory_leak_check):
    """
    Tests basic filter pushdown support.
    """

    def impl1(f1):
        bc = bodosql.BodoSQLContext(
            {
                "table1": bodosql.TablePath(f1, "parquet"),
            }
        )
        return bc.sql("Select * from table1 where part = 'b'")

    def impl2(f1):
        bc = bodosql.BodoSQLContext(
            {
                "table1": bodosql.TablePath(f1, "parquet"),
            }
        )
        return bc.sql("Select A from table1 where part = 'b'")

    def impl3(filename):
        bc = bodosql.BodoSQLContext(
            {
                "table1": bodosql.TablePath(filename, "parquet"),
            }
        )
        return bc.sql("Select A + 1 from table1 where part = 'b'")

    def impl4(filename):
        bc = bodosql.BodoSQLContext(
            {
                "table1": bodosql.TablePath(filename, "parquet"),
            }
        )
        return bc.sql("Select A + 1 from table1 where part = 'b' and part is not null")

    filename = datapath("sample-parquet-data/partitioned")

    # Compare entirely to Pandas output to simplify the process.
    # Load the data once and then filter for each query.
    py_output = pd.read_parquet(filename)
    py_output["part"] = py_output["part"].astype(str)
    py_output = py_output[py_output["part"] == "b"]

    py_output1 = py_output
    check_func(impl1, (filename,), py_output=py_output1, reset_index=True)
    # make sure the ParquetReader node has filters parameter set and we have trimmed
    # any unused columns.
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl1)
    bodo_func(filename)
    _check_for_io_reader_filters(bodo_func, bodo.ir.parquet_ext.ParquetReader)
    # TODO: Check which columns were actually loaded.
    py_output2 = pd.DataFrame({"A": py_output["A"]})
    check_func(impl2, (filename,), py_output=py_output2, reset_index=True)
    # make sure the ParquetReader node has filters parameter set and we have trimmed
    # any unused columns.
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl2)
    bodo_func(filename)
    _check_for_io_reader_filters(bodo_func, bodo.ir.parquet_ext.ParquetReader)
    # TODO: Check which columns were actually loaded.

    # TODO: Update Name when the name changes
    py_output3 = pd.DataFrame({"EXPR$0": py_output["A"] + 1})
    check_func(impl3, (filename,), py_output=py_output3, reset_index=True)
    # make sure the ParquetReader node has filters parameter set and we have trimmed
    # any unused columns.
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl3)
    bodo_func(filename)
    _check_for_io_reader_filters(bodo_func, bodo.ir.parquet_ext.ParquetReader)
    # TODO: Check which columns were actually loaded.

    # TODO: Update Name when the name changes
    py_output4 = pd.DataFrame({"EXPR$0": py_output["A"] + 1})
    check_func(impl4, (filename,), py_output=py_output4, reset_index=True)
    # make sure the ParquetReader node has filters parameter set and we have trimmed
    # any unused columns.
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl4)
    bodo_func(filename)
    _check_for_io_reader_filters(bodo_func, bodo.ir.parquet_ext.ParquetReader)
    # TODO: Check which columns were actually loaded.


@pytest.mark.slow
def test_table_path_filter_pushdown_multitable(datapath, memory_leak_check):
    """
    Tests basic filter with multiple tables.
    """

    def impl(f1):
        bc = bodosql.BodoSQLContext(
            {
                "table1": bodosql.TablePath(f1, "parquet"),
                "table2": bodosql.TablePath(f1, "parquet"),
            }
        )
        return bc.sql(
            "Select table1.A as a1, table1.B as b1, table2.A as a2, table2.B as b2 from table1, table2 where table1.part = 'b' and table2.part = 'a' and table1.c = table2.c"
        )

    filename = datapath("sample-parquet-data/partitioned")

    # Compare entirely to Pandas output to simplify the process.
    # Load the data once and then filter for each query.
    py_output = pd.read_parquet(filename)
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
    _check_for_io_reader_filters(bodo_func, bodo.ir.parquet_ext.ParquetReader)
    # TODO: Check which columns were actually loaded.
    # At this point BodoSQL is expected to load the table twice, once for each table.
    check_num_parquet_readers(bodo_func, 2)


@pytest.mark.slow
def test_table_path_no_filter_pushdown(datapath, memory_leak_check):
    """
    Tests when filter pushdown should be rejected because a table is reused.
    """

    def impl(f1):
        bc = bodosql.BodoSQLContext(
            {
                "table1": bodosql.TablePath(f1, "parquet"),
            }
        )
        return bc.sql(
            "Select t1.A as a1 from table1 as t1 inner join table1 on table1.c = t1.c where t1.part = 'a'"
        )

    filename = datapath("sample-parquet-data/partitioned")

    # Compare entirely to Pandas output to simplify the process.
    # Load the data once and then filter for each query.
    py_output = pd.read_parquet(filename)
    py_output["part"] = py_output["part"].astype(str)
    py_output_part1 = py_output[py_output["part"] == "a"]
    py_output = py_output_part1.merge(py_output, on="C")
    py_output = pd.DataFrame(
        {
            "a1": py_output["A_x"],
        }
    )
    check_func(
        impl, (filename,), py_output=py_output, reset_index=True, sort_output=True
    )
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl)
    bodo_func(filename)
    try:
        _check_for_io_reader_filters(bodo_func, bodo.ir.parquet_ext.ParquetReader)
        # If we reach this line the test is wrong.
        failed = True
    except AssertionError:
        # We expect an assertion error because filter pushdown should not have occurred
        failed = False

    assert not failed


@pytest.mark.slow
def test_table_path_col_pruning_and_filter_pushdown_implicite_casting(
    datapath,
    memory_leak_check,
):
    """
    Tests that filter pushdown is correctly applied in the case that we perform implicit casting of the
    input dataframe types (done in visitTableScan)
    """

    # This dataframe has 3 columns, A -> categorical datetime64,
    # B -> categorial strings, C -> Datetype, D -> int E -> partition column of string
    # A, B, and C will be implictly by bodosql in visitTableScan
    # Note, that
    filename = datapath("sample-parquet-data/needs_implicit_typ_conversion.pq")

    # tests filters/column pruning works on partitions
    def impl_simple_no_join_filter_partition(f1):
        bc = bodosql.BodoSQLContext(
            {
                "table1": bodosql.TablePath(f1, "parquet"),
            }
        )
        return bc.sql(
            "Select table1.A, table1.B, table1.C, table1.D from table1 where table1.E='a'"
        )

    # tests filters/column pruning works on non partitions
    def impl_simple_no_join_filter_non_partition(f1):
        bc = bodosql.BodoSQLContext(
            {
                "table1": bodosql.TablePath(f1, "parquet"),
            }
        )
        return bc.sql(
            "Select table1.A, table1.B, table1.C, table1.E from table1 where table1.D > 1"
        )

    # tests filters/column pruning works when filtering on partitions, with a join
    def impl_should_load_B_C_D(f1, df):
        bc = bodosql.BodoSQLContext(
            {
                "table1": df,
                "table2": bodosql.TablePath(f1, "parquet"),
            }
        )
        return bc.sql(
            "Select table2.B, table2.C from table1 JOIN table2 ON table1.D = table2.D where table2.E='a'"
        )

    # tests column pruning works without a filter, with a join
    def impl_should_load_A_E(f1, df):
        bc = bodosql.BodoSQLContext(
            {
                "table1": df,
                "table2": bodosql.TablePath(f1, "parquet"),
            }
        )
        return bc.sql("Select table2.A from table1 JOIN table2 ON table1.E = table2.E")

    # tests filters/column pruning works when no columns are loaded from table2
    def impl_should_load_None(f1, df):
        bc = bodosql.BodoSQLContext(
            {
                "table1": df,
                "table2": bodosql.TablePath(f1, "parquet"),
            }
        )
        return bc.sql("Select table1.A from table1 where table1.B='c'")

    # Compare entirely to Pandas output to simplify the process.
    # Load the data once and then filter for each query.
    read_df = pd.read_parquet(filename)
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
        # Unfortunatly, we don't get information on which filters were pushed, as BodoSQL is loaded as a functext
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
        # Unfortunatly, we don't get information on which filters were pushed, as BodoSQL is loaded as a functext
        # TODO: find an effective way to check which filters were pushed.
        check_logger_msg(
            stream, "Filter pushdown successfully performed. Moving filter step:"
        )
        check_logger_msg(stream, "Columns loaded ['B', 'A', 'C', 'E']")
        stream.truncate(0)
        stream.seek(0)

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
        # Unfortunatly, we don't get information on which filters were pushed, as BodoSQL is loaded as a functext
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


@pytest.mark.slow
def test_table_path_col_pruning_simple(datapath, memory_leak_check):
    """
    Tests that column pruning is correctly applied in the case that we perform implicit casting of the
    input dataframe types (done in visitTableScan)
    """

    # This dataframe has 3 columns, A -> categorical datetime64,
    # B -> categorial strings, C -> Datetype, D -> int E -> partition column of string
    # A, B, and C will be implictly cast by bodosql in visitTableScan
    filename = datapath("sample-parquet-data/needs_implicit_typ_conversion.pq")

    def impl_simple_only_A(f1):
        bc = bodosql.BodoSQLContext(
            {
                "table1": bodosql.TablePath(f1, "parquet"),
            }
        )
        return bc.sql("Select table1.A from table1")

    def impl_simple_only_D(f1):
        bc = bodosql.BodoSQLContext(
            {
                "table1": bodosql.TablePath(f1, "parquet"),
            }
        )
        return bc.sql("Select table1.D from table1")

    # Compare entirely to Pandas output to simplify the process.
    # Load the data once and then filter for each query.
    read_df = pd.read_parquet(filename)
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
def test_table_path_limit_pushdown(datapath, memory_leak_check):
    """
    Test basic limit pushdown support.
    """

    # select columns
    def impl1(f1):
        bc = bodosql.BodoSQLContext(
            {
                "table1": bodosql.TablePath(f1, "parquet"),
            }
        )
        return bc.sql("Select A, B from table1 limit 5")

    # all columns
    def impl2(f1):
        bc = bodosql.BodoSQLContext(
            {
                "table1": bodosql.TablePath(f1, "parquet"),
            }
        )
        return bc.sql("Select * from table1 limit 5")

    # TODO[BE-3581]: support and test limit pushdown for partitioned Parquet datasets
    # filename = "BodoSQL/bodosql/tests/data/sample-parquet-data/partitioned"
    filename = datapath("sample-parquet-data/no_index.pq")

    py_output = pd.read_parquet(filename)[["A", "B"]].head(5)
    check_func(
        impl1, (filename,), py_output=py_output, reset_index=True, check_dtype=False
    )

    # make sure limit pushdown worked
    bodo_func = bodo.jit(pipeline_class=DistTestPipeline)(impl1)
    bodo_func(filename)
    fir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert hasattr(fir, "meta_head_only_info")
    assert fir.meta_head_only_info[0] is not None

    py_output = pd.read_parquet(filename).head(5)
    check_func(
        impl2, (filename,), py_output=py_output, reset_index=True, check_dtype=False
    )

    # make sure limit pushdown worked
    bodo_func = bodo.jit(pipeline_class=DistTestPipeline)(impl2)
    bodo_func(filename)
    fir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert hasattr(fir, "meta_head_only_info")
    assert fir.meta_head_only_info[0] is not None


def test_named_param_filter_pushdown(memory_leak_check):
    """
    Test that using a Python variable as a filter variable via the named
    parameter supports filter pushdown.
    """
    filename = datapath("sample-parquet-data/needs_implicit_typ_conversion.pq")

    def impl(f1, val):
        bc = bodosql.BodoSQLContext(
            {
                "table1": bodosql.TablePath(f1, "parquet"),
            }
        )
        return bc.sql(
            "Select table1.A, table1.B, table1.C, table1.D from table1 where table1.E=@pyval",
            {"pyval": val},
        )

    # Compare entirely to Pandas output to simplify the process.
    read_df = pd.read_parquet(filename)
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
def test_table_path_limit_pushdown_complex(datapath, memory_leak_check):
    """
    Tests that limit pushdown works with a possible complicated projection.
    """

    def impl(f1):
        bc = bodosql.BodoSQLContext(
            {
                "table1": bodosql.TablePath(f1, "parquet"),
            }
        )
        return bc.sql(
            "Select A + 1 as A, B as newCol1, 'my_name' as newCol2 from table1 limit 5"
        )

    filename = datapath("sample-parquet-data/no_index.pq")
    py_output = pd.read_parquet(filename)[["A", "B"]]
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
                "table1": bodosql.TablePath(filename, "parquet"),
            }
        )
        return bc.sql(query)

    filename = datapath("tpch-test-data/parquet/lineitem.parquet")
    read_df = pd.read_parquet(filename)

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
