import glob
import io
import os
import struct
import traceback
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

# We need to import the connector first in order to apply our Py4J
# monkey-patch. Since PySpark uses Py4J, it will load in the functions
# we want to patch into memory; thus, without this import, those
# functions will be saved and used before we can change them
import bodo_iceberg_connector  # noqa
import mmh3
import pandas as pd
import pytest
import pytz
from mpi4py import MPI

import bodo
from bodo.tests.iceberg_database_helpers import spark_reader
from bodo.tests.iceberg_database_helpers.part_sort_table import (
    BASE_NAME as PART_SORT_TABLE_BASE_NAME,
)
from bodo.tests.iceberg_database_helpers.part_sort_table import (
    PARTITION_SPEC as PART_SORT_TABLE_PARTITION_SPEC,
)
from bodo.tests.iceberg_database_helpers.part_sort_table import (
    SORT_ORDER as PART_SORT_TABLE_SORT_ORDER,
)
from bodo.tests.iceberg_database_helpers.partition_tables import (
    PARTITION_MAP,
    part_table_name,
)
from bodo.tests.iceberg_database_helpers.simple_tables import (
    TABLE_MAP as SIMPLE_TABLES_MAP,
)
from bodo.tests.iceberg_database_helpers.sort_tables import (
    SORT_MAP,
    sort_table_name,
)
from bodo.tests.iceberg_database_helpers.utils import (
    DATABASE_NAME,
    PartitionField,
    SortField,
    create_iceberg_table,
    get_spark,
)
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import (
    DistTestPipeline,
    _gather_output,
    _get_dist_arg,
    _test_equal,
    _test_equal_guard,
    check_func,
    reduce_sum,
    sync_dtypes,
)
from bodo.utils.testing import ensure_clean2
from bodo.utils.typing import BodoError

pytestmark = pytest.mark.iceberg

WRITE_TABLES = [
    "bool_binary_table",
    "dt_tsz_table",
    "tz_aware_table",
    "dtype_list_table",
    "numeric_table",
    "string_table",
    "list_table",
    "struct_table",
    # TODO Needs investigation.
    pytest.param(
        "map_table",
        marks=pytest.mark.skip(
            reason="Results in runtime error that's consistent with to_parquet."
        ),
    ),
    pytest.param(
        "decimals_table",
        marks=pytest.mark.skip(
            reason="We don't suppport custom precisions and scale at the moment."
        ),
    ),
    pytest.param(
        "decimals_list_table",
        marks=pytest.mark.skip(
            reason="We don't suppport custom precisions and scale at the moment."
        ),
    ),
    "dict_encoded_string_table",
]


@pytest.fixture(params=WRITE_TABLES)
def simple_dataframe(request):
    return (
        request.param,
        f"simple_{request.param}",
        SIMPLE_TABLES_MAP[request.param][0],
    )


@pytest.mark.parametrize(
    "table_name",
    [
        # TODO: BE-2831 Reading maps from parquet not supported yet
        pytest.param(
            "simple_map_table",
            marks=pytest.mark.skip(reason="Need to support reading maps from parquet."),
        ),
        "simple_string_table",
        "partitions_dt_table",
        "simple_dt_tsz_table",
        "simple_decimals_table",
    ],
)
def test_simple_table_read(
    iceberg_database,
    iceberg_table_conn,
    table_name,
    # Add back after fixing https://bodo.atlassian.net/browse/BE-3606
    # memory_leak_check,
):
    """
    Test simple read operation on test tables
    """

    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        return df

    py_out, _, _ = spark_reader.read_iceberg_table(table_name, db_schema)
    check_func(
        impl,
        (table_name, conn, db_schema),
        py_output=py_out,
        sort_output=True,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "table_name",
    [
        # TODO: BE-2831 Reading maps from parquet not supported yet
        pytest.param(
            "simple_map_table",
            marks=pytest.mark.skip(reason="Need to support reading maps from parquet."),
        ),
        "simple_string_table",
        "partitions_dt_table",
        "simple_dt_tsz_table",
        "simple_decimals_table",
    ],
)
def test_read_zero_cols(iceberg_database, iceberg_table_conn, table_name):
    """
    Test that computing just a length in Iceberg loads 0 columns.
    """
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        return len(df)

    py_out, _, _ = spark_reader.read_iceberg_table(table_name, db_schema)
    check_func(
        impl,
        (table_name, conn, db_schema),
        py_output=len(py_out),
    )
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo_func = bodo.jit(impl)
        bodo_func(table_name, conn, db_schema)

        check_logger_msg(stream, "Columns loaded []")


def test_simple_tz_aware_table_read(
    iceberg_database,
    iceberg_table_conn,
    # Add back after fixing https://bodo.atlassian.net/browse/BE-3606
    # memory_leak_check,
):
    """
    Test simple read operation on simple_tz_aware_table.
    Needs to be separate since there's a type mismatch between
    original and what's read from Iceberg (written by Spark).
    When Spark reads it and converts it to Pandas, the datatype
    is:
    A    datetime64[ns]
    B    datetime64[ns]
    but when Bodo reads it, it's:
    A    datetime64[ns, UTC]
    B    datetime64[ns, UTC]
    which causes the mismatch.
    """

    table_name = "simple_tz_aware_table"
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        return df

    py_out, _, _ = spark_reader.read_iceberg_table(table_name, db_schema)
    check_func(
        impl,
        (table_name, conn, db_schema),
        py_output=py_out,
        sort_output=True,
        reset_index=True,
        check_dtype=False,
    )


def test_simple_numeric_table_read(
    iceberg_database,
    iceberg_table_conn,
    # Add back after fixing https://bodo.atlassian.net/browse/BE-3606
    # memory_leak_check,
):
    """
    Test simple read operation on test table simple_numeric_table
    with column pruning.
    """

    table_name = "simple_numeric_table"
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        return df

    py_out, _, _ = spark_reader.read_iceberg_table(table_name, db_schema)
    res: pd.DataFrame = bodo.jit()(impl)(table_name, conn, db_schema)
    py_out = sync_dtypes(py_out, res.dtypes.values.tolist())
    py_out["E"] = py_out["E"].astype("Int32")
    py_out["F"] = py_out["F"].astype("Int64")
    check_func(impl, (table_name, conn, db_schema), py_output=py_out)


@pytest.mark.slow
@pytest.mark.parametrize(
    "table_name", ["simple_list_table", "simple_decimals_list_table"]
)
def test_simple_list_table_read(
    iceberg_database,
    iceberg_table_conn,
    table_name,
    # Add back after fixing https://bodo.atlassian.net/browse/BE-3606
    # memory_leak_check,
):
    """
    Test reading simple_list_table which consists of columns of lists.
    Need to compare Bodo and PySpark results without sorting them.
    """
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        return df

    py_out, _, _ = spark_reader.read_iceberg_table(table_name, db_schema)

    check_func(
        impl,
        (table_name, conn, db_schema),
        py_output=py_out,
        reset_index=True,
        # No sorting because lists are not hashable
    )


def test_simple_bool_binary_table_read(
    iceberg_database,
    iceberg_table_conn,
    # Add back after fixing https://bodo.atlassian.net/browse/BE-3606
    # memory_leak_check,
):
    """
    Test reading simple_bool_binary_table which consists of boolean
    and binary typs (bytes). Needs special handling to compare
    with PySpark.
    """
    table_name = "simple_bool_binary_table"
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        return df

    py_out, _, _ = spark_reader.read_iceberg_table(table_name, db_schema)
    # Bodo outputs binary data as bytes while Spark does bytearray (which Bodo doesn't support),
    # so we convert Spark output.
    # This has been copied from BodoSQL. See `convert_spark_bytearray`
    # in `bodosql/tests/utils.py`.
    py_out[["C"]] = py_out[["C"]].apply(
        lambda x: [bytes(y) if isinstance(y, bytearray) else y for y in x],
        axis=1,
        result_type="expand",
    )
    check_func(
        impl,
        (table_name, conn, db_schema),
        py_output=py_out,
        sort_output=True,
        reset_index=True,
    )


def test_simple_struct_table_read(
    iceberg_database,
    iceberg_table_conn,
    # Add back after fixing https://bodo.atlassian.net/browse/BE-3606
    # memory_leak_check,
):
    """
    Test reading simple_struct_table which consists of columns of structs.
    Needs special handling since PySpark returns nested structs as tuples.
    """
    table_name = "simple_struct_table"
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        return df

    # Convert columns with nested structs from tuples to dictionaries with correct keys
    py_out, _, _ = spark_reader.read_iceberg_table(table_name, db_schema)
    py_out["A"] = py_out["A"].map(lambda x: {"a": x["a"], "b": x["b"]})
    py_out["B"] = py_out["B"].map(lambda x: {"a": x["a"], "b": x["b"], "c": x["c"]})

    check_func(
        impl,
        (table_name, conn, db_schema),
        py_output=py_out,
        reset_index=True,
    )


# Add memory_leak_check after fixing https://bodo.atlassian.net/browse/BE-3606
def test_column_pruning(iceberg_database, iceberg_table_conn):
    """
    Test simple read operation on test table simple_numeric_table
    with column pruning.
    """

    table_name = "simple_numeric_table"
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        df = df[["A", "D"]]
        return df

    py_out, _, _ = spark_reader.read_iceberg_table(table_name, db_schema)
    py_out = py_out[["A", "D"]]

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    res = None
    with set_logging_stream(logger, 1):
        res = bodo.jit()(impl)(table_name, conn, db_schema)
        check_logger_msg(stream, "Columns loaded ['A', 'D']")

    py_out = sync_dtypes(py_out, res.dtypes.values.tolist())
    check_func(impl, (table_name, conn, db_schema), py_output=py_out)


# Add memory_leak_check after fixing https://bodo.atlassian.net/browse/BE-3606
def test_no_files_after_filter_pushdown(iceberg_database, iceberg_table_conn):
    """
    Test the use case where Iceberg filters out all files
    based on the provided filters. We need to load an empty
    dataframe with the right schema in this case.
    """

    table_name = "filter_pushdown_test_table"
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        df = df[df["TY"].isna()]
        return df

    spark = get_spark()
    py_out = spark.sql(
        f"""
    select * from hadoop_prod.{db_schema}.{table_name}
    where TY IS NULL;
    """
    )
    py_out = py_out.toPandas()
    assert (
        py_out.shape[0] == 0
    ), f"Expected DataFrame to be empty, found {py_out.shape[0]} rows instead."

    check_func(impl, (table_name, conn, db_schema), py_output=py_out)


# Add memory_leak_check after fixing https://bodo.atlassian.net/browse/BE-3606
def test_filter_pushdown_partitions(iceberg_database, iceberg_table_conn):
    """
    Test that simple date based partitions can be read as expected.
    """

    table_name = "partitions_dt_table"
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        df = df[df["A"] <= date(2018, 12, 12)]  # type: ignore
        return df

    spark = get_spark()
    py_out = spark.sql(
        f"""
    select * from hadoop_prod.{db_schema}.{table_name}
    where A <= '2018-12-12';
    """
    )
    py_out = py_out.toPandas()

    check_func(
        impl,
        (table_name, conn, db_schema),
        py_output=py_out,
        sort_output=True,
        reset_index=True,
    )


def _check_for_sql_read_head_only(bodo_func, head_size):
    """Make sure head-only SQL read optimization is recognized"""
    fir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert hasattr(fir, "meta_head_only_info")
    assert fir.meta_head_only_info[0] == head_size


# Add memory_leak_check after fixing https://bodo.atlassian.net/browse/BE-3606
def test_limit_pushdown(iceberg_database, iceberg_table_conn):
    """Test that Limit Pushdown is successfully enabled"""
    table_name = "simple_string_table"
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl():
        df = pd.read_sql_table(table_name, conn, db_schema)
        return df.head(5)  # type: ignore

    spark = get_spark()
    py_out = spark.sql(f"select * from hadoop_prod.{db_schema}.{table_name} LIMIT 5;")
    py_out = py_out.toPandas()

    check_func(
        impl,
        (),
        py_output=py_out,
        sort_output=True,
        reset_index=True,
    )

    bodo_func = bodo.jit(pipeline_class=DistTestPipeline)(impl)
    bodo_func()
    _check_for_sql_read_head_only(bodo_func, 5)


def test_schema_evolution_detection(iceberg_database, iceberg_table_conn):
    """
    Test that we throw the right error when dataset has schema evolution,
    which we don't support yet. This test should be removed once
    we add support for it.
    """

    table_name = "filter_pushdown_test_table"
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        df = df[(df["TY"].notnull()) & (df["B"] > 10)]
        return df

    with pytest.raises(
        BodoError,
        match="Bodo currently doesn't support reading Iceberg tables with schema evolution.",
    ):
        bodo.jit(impl)(table_name, conn, db_schema)


@pytest.mark.skip("[BE-3212] Fix Java failures on CI")
def test_iceberg_invalid_table(iceberg_database, iceberg_table_conn):
    """Tests error raised when a nonexistent Iceberg table is provided."""

    table_name = "no_table"
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        return df["A"].sum()

    with pytest.raises(BodoError, match="No such Iceberg table found"):
        bodo.jit(impl)(table_name, conn, db_schema)


def test_iceberg_invalid_path(iceberg_database, iceberg_table_conn):
    """Tests error raised when invalid path is provided."""

    table_name = "filter_pushdown_test_table"
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)
    db_schema += "not"

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        return df["A"].sum()

    with pytest.raises(BodoError, match="No such Iceberg table found"):
        bodo.jit(impl)(table_name, conn, db_schema)


def test_write_existing_fail(
    iceberg_database,
    iceberg_table_conn,
    simple_dataframe,
):
    """Test that writing to an existing table when if_exists='fail' errors"""
    base_name, table_name, df = simple_dataframe
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)

    def impl(df, table_name, conn, db_schema):
        df.to_sql(table_name, conn, db_schema, if_exists="fail")

    orig_use_dict_str_type = bodo.hiframes.boxing._use_dict_str_type
    if base_name == "dict_encoded_string_table":
        bodo.hiframes.boxing._use_dict_str_type = True

    try:
        # TODO Uncomment after adding replicated write support.
        # err = None
        # if bodo.get_rank() == 0:
        #     try:
        #         with pytest.raises(BodoError, match="already exists"):
        #             bodo.jit(replicated=["df"])(impl)(df, table_name, conn, db_schema)
        #     except Exception as e:
        #         err = e
        # err = comm.bcast(err)
        # if isinstance(err, Exception):
        #     raise err

        with pytest.raises(ValueError, match="already exists"):
            bodo.jit(distributed=["df"])(impl)(
                _get_dist_arg(df), table_name, conn, db_schema
            )
    finally:
        bodo.hiframes.boxing._use_dict_str_type = orig_use_dict_str_type


@pytest.mark.parametrize("read_behavior", ["spark", "bodo"])
def test_basic_write_replace(
    iceberg_database,
    iceberg_table_conn,
    simple_dataframe,
    read_behavior,
    # Add memory_leak_check after fixing https://bodo.atlassian.net/browse/BE-3606
    # memory_leak_check,
):
    """Test basic Iceberg table replace on Spark table"""

    comm = MPI.COMM_WORLD
    base_name, table_name, df = simple_dataframe
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)

    def impl(df, table_name, conn, db_schema):
        df.to_sql(table_name, conn, db_schema, if_exists="replace")

    orig_use_dict_str_type = bodo.hiframes.boxing._use_dict_str_type
    if base_name == "dict_encoded_string_table":
        bodo.hiframes.boxing._use_dict_str_type = True

    try:
        # Write using Bodo
        bodo.jit(distributed=["df"])(impl)(
            _get_dist_arg(df), table_name, conn, db_schema
        )
    finally:
        bodo.hiframes.boxing._use_dict_str_type = orig_use_dict_str_type
    # Read using PySpark or Bodo, and then check that it's what's expected

    if table_name == "simple_struct_table" and read_behavior == "spark":
        # There's an issue where Spark is unable to read structs that we
        # write through Iceberg. It's able to read the parquet file
        # when using `spark.read.format("parquet").load(fname)`
        # and the Iceberg metadata that we write looks correct,
        # so it seems like a Spark issue, but needs further investigation.
        # We're able to read the table using Bodo though.
        # TODO Open issue
        return

    if read_behavior == "spark":
        py_out, _, _ = spark_reader.read_iceberg_table(table_name, db_schema)
        # Spark doesn't handle null timestamps properly. It converts them to
        # 0 (i.e. epoch) instead of NaTs like Pandas does. This modifies both
        # dataframes to match Spark.
        if base_name == "dt_tsz_table":
            df["B"] = df["B"].fillna(datetime(1970, 1, 1))
            py_out["B"] = py_out["B"].fillna(datetime(1970, 1, 1))
    else:
        assert (
            read_behavior == "bodo"
        ), "Read Behavior can only be either `spark` or `bodo`"
        py_out = bodo.jit()(lambda: pd.read_sql_table(table_name, conn, db_schema))()
        py_out = _gather_output(py_out)

    # Uncomment if we get Spark to be able to read this table (see comment above)
    # if table_name == "simple_struct_table":
    #     py_out["A"] = py_out["A"].map(lambda x: {"a": x["a"], "b": x["b"]})
    #     py_out["B"] = py_out["B"].map(lambda x: {"a": x["a"], "b": x["b"], "c": x["c"]})

    comm = MPI.COMM_WORLD
    passed = None
    if comm.Get_rank() == 0:
        passed = _test_equal_guard(df, py_out, sort_output=False, check_dtype=False)
    passed = comm.bcast(passed)
    assert passed == 1

    # TODO Uncomment after adding replicated write support.
    # Test replicated -- only run on rank0, and synchronize errors to avoid hangs
    # if behavior == "create":
    #     table_name = f"{table_name}_repl"
    #
    # err = None
    # # Start with 1, it'll become 0 on rank 0 if it fails
    # passed = 1
    # if bodo.get_rank() == 0:
    #     try:
    #         bodo.jit(replicated=["df"])(impl)(df, table_name, conn, db_schema)
    #         # Read using Pyspark, and then check that it's what's expected
    #         passed = _test_equal_guard(
    #             orig_df,
    #             py_out,
    #             sort_output=False,
    #             check_names=True,
    #             check_dtype=False,
    #         )
    #     except Exception as e:
    #         err = e
    # err = comm.bcast(err)
    # if isinstance(err, Exception):
    #     raise err
    # n_passed = reduce_sum(passed)
    # assert n_passed == n_pes)


@pytest.mark.parametrize("behavior", ["create", "append"])
@pytest.mark.parametrize("initial_write", ["bodo", "spark"])
def test_basic_write_new_append(
    iceberg_database,
    iceberg_table_conn,
    simple_dataframe,
    behavior,
    initial_write,
    # Add memory_leak_check after fixing https://bodo.atlassian.net/browse/BE-3606
    # memory_leak_check,
):
    """
    Test basic Iceberg table write + append on new table
    (append to table written by Bodo)
    """

    comm = MPI.COMM_WORLD
    n_pes = comm.Get_size()
    base_name, table_name, df = simple_dataframe

    if (
        table_name == "simple_list_table"
        and initial_write == "spark"
        and behavior == "append"
    ):
        pytest.skip(
            reason="During unboxing of Series with lists, we always assume int64 (vs int32) and float64 (vs float32), which doesn't match original schema written by Spark."
        )

    # We want to use completely new table for each test
    table_name += f"_{behavior}_{initial_write}"
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)

    def create_impl(df, table_name, conn, db_schema):
        df.to_sql(table_name, conn, db_schema, if_exists="append")

    orig_use_dict_str_type = bodo.hiframes.boxing._use_dict_str_type
    if base_name == "dict_encoded_string_table":
        bodo.hiframes.boxing._use_dict_str_type = True

    try:
        impl = bodo.jit(distributed=["df"])(create_impl)

        if initial_write == "bodo" or behavior == "create":
            # Write using Bodo
            impl(_get_dist_arg(df), table_name, conn, db_schema)
        elif initial_write == "spark" and behavior == "append":
            spark = get_spark()
            # Write using Spark on rank 0
            if bodo.get_rank() == 0:
                _, sql_schema = SIMPLE_TABLES_MAP[base_name]
                create_iceberg_table(df, sql_schema, table_name, spark)
            bodo.barrier()
        elif initial_write == "spark" and behavior == "create":
            # Nothing to test here.
            return
        else:
            raise ValueError(
                f"Got unsupported values: initial_write: {initial_write} and behavior: {behavior}."
            )

        if behavior == "append":
            # Append using Bodo
            impl(_get_dist_arg(df), table_name, conn, db_schema)
    finally:
        bodo.hiframes.boxing._use_dict_str_type = orig_use_dict_str_type

    expected_df = (
        pd.concat([df, df]).reset_index(drop=True) if behavior == "append" else df
    )

    # Read using Bodo and PySpark, and then check that it's what's expected
    bodo_out = bodo.jit()(lambda: pd.read_sql_table(table_name, conn, db_schema))()
    bodo_out = _gather_output(bodo_out)
    passed = None
    comm = MPI.COMM_WORLD
    if bodo.get_rank() == 0:
        passed = _test_equal_guard(
            expected_df,
            bodo_out,
            sort_output=False,
            check_dtype=False,
            reset_index=True,
        )

    passed = comm.bcast(passed)
    assert passed == 1

    if table_name.startswith("simple_struct_table"):
        # There's an issue where Spark is unable to read structs that we
        # write through Iceberg. It's able to read the parquet file
        # when using `spark.read.format("parquet").load(fname)`
        # and the Iceberg metadata that we write looks correct,
        # so it seems like a Spark issue, but needs further investigation.
        # We're able to read the table using Bodo though.
        # TODO Open issue
        return

    if initial_write == "spark" and behavior == "append":
        # We need to invalidate spark cache, because it doesn't realize
        # that the table has been modified.
        spark.sql("CLEAR CACHE;")
        spark.sql(f"REFRESH TABLE hadoop_prod.{DATABASE_NAME}.{table_name};")

    spark_passed = 1
    if bodo.get_rank() == 0:
        spark_out, _, _ = spark_reader.read_iceberg_table(table_name, db_schema)

        # Uncomment if we get Spark to be able to read this table (see comment above)
        # if table_name == "simple_struct_table":
        #     spark_out["A"] = spark_out["A"].map(lambda x: {"a": x["a"], "b": x["b"]})
        #     spark_out["B"] = spark_out["B"].map(
        #         lambda x: {"a": x["a"], "b": x["b"], "c": x["c"]}
        #     )

        # Spark doesn't handle null timestamps properly. It converts them to
        # 0 (i.e. epoch) instead of NaTs like Pandas does. This modifies both
        # dataframes to match Spark.
        if base_name == "dt_tsz_table":
            expected_df["B"] = expected_df["B"].fillna(datetime(1970, 1, 1))
            spark_out["B"] = spark_out["B"].fillna(datetime(1970, 1, 1))

        spark_passed = _test_equal_guard(
            expected_df,
            spark_out,
            sort_output=False,
            check_dtype=False,
        )
    spark_n = reduce_sum(spark_passed)
    assert spark_n == n_pes


def test_iceberg_write_error_checking(iceberg_database, iceberg_table_conn):
    """
    Tests for known errors thrown when writing an Iceberg table.
    """
    table_name = "simple_bool_binary_table"
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)
    df = SIMPLE_TABLES_MAP["bool_binary_table"][0]

    # Check that error is raised when schema is not provided
    def impl1(df, table_name, conn):
        df.to_sql(table_name, conn)

    with pytest.raises(
        ValueError,
        match="schema must be provided when writing to an Iceberg table",
    ):
        bodo.jit(distributed=["df"])(impl1)(df, table_name, conn)  # type: ignore

    # Check that error is raised when chunksize is provided
    def impl2(df, table_name, conn, db_schema):
        df.to_sql(table_name, conn, db_schema, chunksize=5)

    with pytest.raises(ValueError, match="chunksize not supported for Iceberg tables"):
        bodo.jit(distributed=["df"])(impl2)(df, table_name, conn, db_schema)  # type: ignore

    # TODO Remove after adding replicated write support
    # Check that error is raise when trying to write a replicated dataframe
    # (unsupported for now)
    def impl3(df, table_name, conn, db_schema):
        df.to_sql(table_name, conn, db_schema)

    with pytest.raises(
        AssertionError, match="Iceberg Write only supported for distributed dataframes"
    ):
        bodo.jit(replicated=["df"])(impl3)(df, table_name, conn, db_schema)  # type: ignore


# Add memory_leak_check after fixing https://bodo.atlassian.net/browse/BE-3606
def test_read_pq_write_iceberg(iceberg_database, iceberg_table_conn):
    """
    Some compilation errors can only be observed when running multiple steps.
    This is to test one such common use case, which is reading a table
    from a parquet file and writing it as an Iceberg table.
    This unit test was added as part of https://github.com/Bodo-inc/Bodo/pull/4145
    where an error for such use case was found.
    """

    # The exact table to use doesn't matter, so picking one at random.
    df = SIMPLE_TABLES_MAP["numeric_table"][0]
    fname = "test_read_pq_write_iceberg_ds.pq"

    # Give it a unique name so there's no conflicts.
    table_name = "test_read_pq_write_iceberg_table"
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)

    def impl(pq_fname, table_name, conn, db_schema):
        df = pd.read_parquet(pq_fname)
        df.to_sql(
            table_name,
            conn,
            db_schema,
            if_exists="replace",
            index=False,
        )

    with ensure_clean2(fname):
        if bodo.get_rank() == 0:
            df.to_parquet(fname)
        bodo.barrier()
        # We're just running to make sure that it executes,
        # not for correctness itself, since that is
        # already being tested by the other unit tests.
        bodo.jit(impl)(fname, table_name, conn, db_schema)


def truncate_impl(x: pd.Series, W: int):
    """
    Apply the Iceberg truncate transform on Pandas series x.

    Args:
        x (pd.Series): Array to transform
        W (int): width for the truncate operation.

    Raises:
        NotImplementedError: when dtype of x is not string or integer

    Returns:
        Truncated array
    """
    if x.dtype in ["str", "object", "string[python]"]:
        return x.str.slice(stop=W)
    elif x.dtype in ["int32", "Int32", "int64", "Int64"]:
        return x - (((x % W) + W) % W)
    else:
        raise NotImplementedError(f"truncate_impl not implemented for {x.dtype}")


def null_scalar_wrapper(inner):
    return lambda x, y: None if x == "null" else inner(x, y)


def truncate_scalar_impl(x, W: int):
    if isinstance(x, str):
        return x[:W]
    elif isinstance(x, int):
        return x - (((x % W) + W) % W)
    else:
        raise NotImplementedError(f"truncate_scalar_impl not implemented for {type(x)}")


def bucket_scalar_impl(x, y: int) -> Optional[int]:
    if x is None:
        return None
    if x is pd.NA:
        return x

    if isinstance(x, int):
        res = mmh3.hash(struct.pack("<q", x))
    elif isinstance(x, (datetime, pd.Timestamp)):
        if pd.isna(x):
            return None
        res = mmh3.hash(
            struct.pack(
                "<q",
                int(
                    round(
                        (x - datetime(1970, 1, 1, tzinfo=pytz.utc)).total_seconds()
                        * 1e6
                    )
                ),
            )
        )
    elif isinstance(x, date):
        # Based on https://iceberg.apache.org/spec/#appendix-b-32-bit-hash-requirements,
        # dates should be hashed as int32 (after computing number of days since epoch),
        # however Spark, Iceberg-Python and example in Iceberg Spec seems to use
        # int64.
        res = mmh3.hash(struct.pack("<q", (x - date(1970, 1, 1)).days))
    elif isinstance(x, str):
        res = mmh3.hash(x)
    else:
        # TODO: Include Date/Time, Decimal, and UUID Types
        raise NotImplementedError(f"bucket_scalar_impl not implemented for {type(x)}")

    return (res & 2147483647) % y


def identity_scalar_impl(x: str, y):
    if x == "true":
        return True
    elif x == "false":
        return False
    else:
        return x


def month_scalar_impl(x: str, y):
    if "-" in x:
        parsed = x.split("-")
        year = int(parsed[0])
        month = int(parsed[1])
        return (year - 1970) * 12 + month - 1
    else:
        return int(x)


SCALAR_TRANSFORM_FUNC = {
    "years": null_scalar_wrapper(
        lambda x, _: int(x) - 1970 if int(x) > 1000 else int(x)
    ),
    "months": month_scalar_impl,
    "days": lambda x, _: datetime.strptime(x, "%Y-%m-%d").date() - date(1970, 1, 1)
    if "-" in x
    else int(x),
    "hours": lambda x, _: (
        datetime.strptime(x, "%Y-%m-%d-%H") - datetime(1970, 1, 1)
    ).total_seconds()
    // 3600
    if "-" in x
    else int(x),
    "identity": null_scalar_wrapper(identity_scalar_impl),
    "truncate": truncate_scalar_impl,
    "bucket": lambda x, _: int(x),  # The scalar is already correct
}

ARRAY_TRANSFORM_FUNC = {
    "years": lambda df, _: df.apply(lambda x: None if pd.isna(x) else x.year - 1970),
    "months": lambda df, _: df.apply(
        lambda x: None if pd.isna(x) else (x.year - 1970) * 12 + x.month - 1
    ),
    "days": lambda df, _: df.apply(
        lambda x: None
        if pd.isna(x)
        else (
            (x.date() if isinstance(x, (datetime, pd.Timestamp)) else x)
            - date(1970, 1, 1)
        ).days
    ),
    "hours": lambda df, _: df.apply(
        lambda x: None
        if pd.isna(x)
        else (x.date() - date(1970, 1, 1)).days * 24 + x.hour
    ),
    "identity": lambda df, _: df,
    "truncate": truncate_impl,
    # Since the function can return pd.NA, cast to nullable integer array by default
    "bucket": lambda df, val: df.apply(lambda x: bucket_scalar_impl(x, val)).astype(
        "Int64"
    ),
}


@pytest.mark.parametrize("base_name,part_spec", PARTITION_MAP)
def test_write_partitioned(
    iceberg_database,
    iceberg_table_conn,
    base_name: str,
    part_spec: List[PartitionField],
    # Add memory_leak_check after fixing https://bodo.atlassian.net/browse/BE-3606
    # memory_leak_check,
):
    """
    Tests that appending to a table with a defined partition spec works
    as expected, i.e. the generated files are partitioned based on the
    partitioned spec and the transform values are as expected.
    We then also read the table back using Spark and Bodo and validate
    that the contents are as expected.
    """
    db_schema, warehouse_loc = iceberg_database
    table_name = part_table_name(base_name, part_spec)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=True)
    comm = MPI.COMM_WORLD

    @bodo.jit(distributed=["df"])
    def impl(df, table_name, conn, db_schema):
        df.to_sql(table_name, conn, db_schema, if_exists="append")

    orig_use_dict_str_type = bodo.hiframes.boxing._use_dict_str_type
    if base_name == "dict_encoded_string_table":
        bodo.hiframes.boxing._use_dict_str_type = True

    # TODO Add repl test when supported

    df = SIMPLE_TABLES_MAP[base_name][0]
    try:
        impl(_get_dist_arg(df), table_name, conn, db_schema)
    finally:
        bodo.hiframes.boxing._use_dict_str_type = orig_use_dict_str_type

    bodo.barrier()

    # Check that the files are correctly partitioned
    passed = 1
    err = "File partition validation failed. See error on rank 0."
    if bodo.get_rank() == 0:
        try:
            data_files: list[str] = glob.glob(
                os.path.join(warehouse_loc, db_schema, table_name, "data", "**.parquet")
            )
            for data_file in data_files:
                _test_file_part(data_file, part_spec)
        except Exception as e:
            err = "".join(traceback.format_exception(None, e, e.__traceback__))
            passed = 0
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), err

    # Read the table back using Spark and Bodo and validate that the
    # contents are as expected
    expected_df = pd.concat([df, df]).reset_index(drop=True)

    if base_name == "bool_binary_table":
        # [BE-3585] Bodo write binary columns as string when partitioned,
        # so validating by reading the table back would fail.
        return

    # Spark doesn't handle null timestamps properly. It converts them to
    # 0 (i.e. epoch) instead of NaTs like Pandas does. This modifies expected
    # df to match Spark.
    if base_name == "dt_tsz_table":
        expected_df["B"] = expected_df["B"].fillna(datetime(1970, 1, 1))

    # Validate Bodo read output:
    bodo_out = bodo.jit(distributed=["df"])(
        lambda: pd.read_sql_table(table_name, conn, db_schema)
    )()  # type: ignore

    # Spark can have inconsistent behavior when reading/writing null
    # timestamps, so we convert all NaTs to epoch for consistent
    # comparison
    if base_name == "dt_tsz_table":
        bodo_out["B"] = bodo_out["B"].fillna(datetime(1970, 1, 1))
    bodo_out = _gather_output(bodo_out)

    passed = None
    if bodo.get_rank() == 0:
        passed = _test_equal_guard(
            expected_df,
            bodo_out,
            sort_output=True,
            check_dtype=False,
            reset_index=True,
        )

    passed = comm.bcast(passed)
    assert passed == 1, "Bodo read output doesn't match expected output"

    # Validate Spark read output:
    # We need to invalidate spark cache, because it doesn't realize
    # that the table has been modified.
    spark = get_spark()
    spark.sql("CLEAR CACHE;")
    spark.sql(f"REFRESH TABLE hadoop_prod.{DATABASE_NAME}.{table_name};")

    passed = None
    if bodo.get_rank() == 0:
        spark_out, _, _ = spark_reader.read_iceberg_table(table_name, db_schema, spark)
        # Spark doesn't handle null timestamps consistently. It converts them to
        # 0 (i.e. epoch) instead of NaTs like Pandas does. This modifies the
        # dataframe to match Spark.
        if base_name == "dt_tsz_table":
            spark_out["B"] = spark_out["B"].fillna(datetime(1970, 1, 1))
        passed = _test_equal_guard(
            expected_df,
            spark_out,
            sort_output=True,
            check_dtype=False,
            reset_index=True,
        )
    passed = comm.bcast(passed)
    assert passed == 1, "Spark read output doesn't match expected output"


@pytest.fixture(params=SORT_MAP, ids=lambda x: sort_table_name(x[0], x[1]))
def sort_cases(request):
    base_name, sort_order = request.param
    return (base_name, sort_order, sort_table_name(base_name, sort_order))


def test_write_sorted(
    iceberg_database,
    iceberg_table_conn,
    sort_cases,
    # Add memory_leak_check after fixing https://bodo.atlassian.net/browse/BE-3606
    # memory_leak_check,
):
    """
    Test that we can append to tables with a defined sort-order.
    We append rows to the table and then verify that all files
    for the table are sorted as expected.
    We then also read the table back using Spark and Bodo and validate
    that the contents are as expected.
    """
    base_name, sort_order, table_name = sort_cases
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=True)
    comm = MPI.COMM_WORLD

    @bodo.jit(distributed=["df"])
    def impl(df, table_name, conn, db_schema):
        df.to_sql(table_name, conn, db_schema, if_exists="append")

    orig_use_dict_str_type = bodo.hiframes.boxing._use_dict_str_type
    if base_name == "dict_encoded_string_table":
        bodo.hiframes.boxing._use_dict_str_type = True

    df = SIMPLE_TABLES_MAP[base_name][0]
    try:
        impl(_get_dist_arg(df), table_name, conn, db_schema)
    finally:
        bodo.hiframes.boxing._use_dict_str_type = orig_use_dict_str_type
    bodo.barrier()

    # TODO Add repl test when supported

    # Validate that the files are sorted based on the sort order
    passed = 1
    err = "Sorted file validation failed. See error on rank 0"
    if bodo.get_rank() == 0:
        try:
            # Get List of Sorted Data Files
            data_files = glob.glob(
                os.path.join(warehouse_loc, db_schema, table_name, "data", "*.parquet")
            )
            assert all(os.path.isfile(file) for file in data_files)

            # Check Contents of Each Folder
            for data_file in data_files:
                _test_file_sorted(data_file, sort_order)
        except Exception as e:
            err = "".join(traceback.format_exception(None, e, e.__traceback__))
            passed = 0
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), err

    # Read the table back using Spark and Bodo and validate that the
    # contents are as expected
    expected_df = pd.concat([df, df]).reset_index(drop=True)
    # Spark can have inconsistent behavior when reading/writing null
    # timestamps, so we convert all NaTs to epoch for consistent
    # comparison
    if base_name == "dt_tsz_table":
        expected_df["B"] = expected_df["B"].fillna(datetime(1970, 1, 1))

    if base_name == "bool_binary_table":
        # [BE-3585] Bodo write binary columns as string when partitioned,
        # so validating by reading the table back would fail.
        return

    # Validate Bodo read output:
    bodo_out = bodo.jit(distributed=["df"])(
        lambda: pd.read_sql_table(table_name, conn, db_schema)
    )()  # type: ignore
    # Spark can have inconsistent behavior when reading/writing null
    # timestamps, so we convert all NaTs to epoch for consistent
    # comparison
    if base_name == "dt_tsz_table":
        bodo_out["B"] = bodo_out["B"].fillna(datetime(1970, 1, 1))
    bodo_out = _gather_output(bodo_out)

    passed = None
    if bodo.get_rank() == 0:
        passed = _test_equal_guard(
            expected_df,
            bodo_out,
            sort_output=True,
            check_dtype=False,
            reset_index=True,
        )

    passed = comm.bcast(passed)
    assert passed == 1, "Bodo read output doesn't match expected output"

    # Validate Spark read output:
    # We need to invalidate spark cache, because it doesn't realize
    # that the table has been modified.
    spark = get_spark()
    spark.sql("CLEAR CACHE;")
    spark.sql(f"REFRESH TABLE hadoop_prod.{DATABASE_NAME}.{table_name};")

    passed = None
    if bodo.get_rank() == 0:
        spark_out, _, _ = spark_reader.read_iceberg_table(table_name, db_schema, spark)
        # Spark doesn't handle null timestamps consistenctly. It sometimes converts them to
        # 0 (i.e. epoch) instead of NaTs like Pandas does. This modifies expected
        # df to match Spark.
        if base_name == "dt_tsz_table":
            spark_out["B"] = spark_out["B"].fillna(datetime(1970, 1, 1))
        passed = _test_equal_guard(
            expected_df,
            spark_out,
            sort_output=True,
            check_dtype=False,
            reset_index=True,
        )
    passed = comm.bcast(passed)
    assert passed == 1, "Spark read output doesn't match expected output"


@pytest.mark.parametrize("use_dict_encoding_boxing", [False, True])
def test_write_part_sort(
    iceberg_database,
    iceberg_table_conn,
    use_dict_encoding_boxing,
    # Add memory_leak_check after fixing https://bodo.atlassian.net/browse/BE-3606
    # memory_leak_check,
):
    """
    Append to a table with both a partition spec and a sort order,
    and verify that the append was done correctly, i.e. validate
    that each file is correctly sorted and partitioned.
    Then read the table using Spark and Bodo and validate that the
    output is as expected.
    """
    table_name = f"partsort_{PART_SORT_TABLE_BASE_NAME}"
    df, sql_schema = SIMPLE_TABLES_MAP[PART_SORT_TABLE_BASE_NAME]
    if use_dict_encoding_boxing:
        table_name += "_dict_encoding"
        spark = get_spark()
        if bodo.get_rank() == 0:
            create_iceberg_table(
                df,
                sql_schema,
                table_name,
                spark,
                PART_SORT_TABLE_PARTITION_SPEC,
                PART_SORT_TABLE_SORT_ORDER,
            )
        bodo.barrier()
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=True)

    @bodo.jit(distributed=["df"])
    def impl(df, table_name, conn, db_schema):
        df.to_sql(table_name, conn, db_schema, if_exists="append")

    orig_use_dict_str_type = bodo.hiframes.boxing._use_dict_str_type
    bodo.hiframes.boxing._use_dict_str_type = use_dict_encoding_boxing

    try:
        impl(_get_dist_arg(df), table_name, conn, db_schema)
    finally:
        bodo.hiframes.boxing._use_dict_str_type = orig_use_dict_str_type
    bodo.barrier()

    data_files = glob.glob(
        os.path.join(warehouse_loc, db_schema, table_name, "data", "**.parquet")
    )
    assert all(os.path.isfile(file) for file in data_files)

    passed = 1
    err = "Partition-Spec/Sort-Order validation of files failed. See error on rank 0"
    if bodo.get_rank() == 0:
        try:
            for data_file in data_files:
                _test_file_part(data_file, PART_SORT_TABLE_PARTITION_SPEC)
                _test_file_sorted(data_file, PART_SORT_TABLE_SORT_ORDER)
        except Exception as e:
            err = "".join(traceback.format_exception(None, e, e.__traceback__))
            passed = 0
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), err

    # Read the table back using Spark and Bodo and validate that the
    # contents are as expected
    expected_df = pd.concat([df, df]).reset_index(drop=True)

    # Validate Spark read output:
    # We need to invalidate spark cache, because it doesn't realize
    # that the table has been modified.
    spark = get_spark()
    spark.sql("CLEAR CACHE;")
    spark.sql(f"REFRESH TABLE hadoop_prod.{DATABASE_NAME}.{table_name};")

    passed = None
    comm = MPI.COMM_WORLD
    if bodo.get_rank() == 0:
        spark_out, _, _ = spark_reader.read_iceberg_table(table_name, db_schema, spark)
        passed = _test_equal_guard(
            expected_df,
            spark_out,
            sort_output=True,
            check_dtype=False,
            reset_index=True,
        )
    passed = comm.bcast(passed)
    assert passed == 1, "Spark read output doesn't match expected output"

    # Validate Bodo read output:
    bodo_out = bodo.jit(distributed=["df"])(
        lambda: pd.read_sql_table(table_name, conn, db_schema)
    )()  # type: ignore
    bodo_out = _gather_output(bodo_out)

    passed = None
    if bodo.get_rank() == 0:
        passed = _test_equal_guard(
            expected_df,
            bodo_out,
            sort_output=True,
            check_dtype=False,
            reset_index=True,
        )

    passed = comm.bcast(passed)
    assert passed == 1, "Bodo read output doesn't match expected output"


def _test_file_part(file_name: str, part_spec: List[PartitionField]):
    # Construct Expected Partition Values
    before = True
    part_folders = [
        p.name
        for p in Path(file_name).parents
        if (before := before and not str(p).endswith("data"))
    ]
    part_values = [folder.split("=")[1] for folder in part_folders]
    expected_vals = [
        SCALAR_TRANSFORM_FUNC[trans](val, tval)
        for val, (_, trans, tval) in zip(part_values, part_spec)
    ]

    # Check if data adheres to partitioning
    df = pd.read_parquet(file_name)

    for (col, trans, tval), expected_val in zip(part_spec, expected_vals):
        trans_col = ARRAY_TRANSFORM_FUNC[trans](df[col], tval)

        if expected_val is None:
            assert (
                trans_col.isnull()
            ).all(), "Partition value does not equal the result after applying the transformation"
        else:
            expected_col = pd.Series([expected_val]).astype(trans_col.dtype)[0]
            assert (
                trans_col == expected_col
            ).all(), "Partition value does not equal the result after applying the transformation"


def _test_file_sorted(file_name: str, sort_order: List[SortField]):
    df = pd.read_parquet(file_name, use_nullable_dtypes=True)

    # Compute Transformed Columns
    new_cols: List[pd.Series] = [
        ARRAY_TRANSFORM_FUNC[trans](df[col], val)
        for (col, trans, val, _, _) in sort_order
    ]
    idxs = [str(x) for x in range(len(sort_order))]
    df_vals = pd.DataFrame(dict(zip(idxs, new_cols)))

    # Check if it's sorted
    ascending = [s.asc for s in sort_order]
    na_position = ["first" if s.nulls_first else "last" for s in sort_order]

    @bodo.jit(distributed=False)
    def bodo_sort(df):
        res = df.sort_values(
            by=idxs,
            ascending=ascending,
            na_position=na_position,
            ignore_index=False,
        )
        return res.reset_index(drop=True)

    sorted_vals = bodo_sort(df_vals)
    _test_equal(df_vals, sorted_vals, check_dtype=False)


@pytest.mark.parametrize("use_dict_encoding_boxing", [False, True])
def test_write_part_sort_return_orig(
    iceberg_database,
    iceberg_table_conn,
    use_dict_encoding_boxing,
):
    """
    Test that when writing to a table with a defined partition-spec
    and sort order, performing the sort for the write doesn't modify
    the original dataframe.
    This tests that refcounts are handled correctly in the C++
    code. If there's an issue, this should segfault.
    """

    comm = MPI.COMM_WORLD
    table_name = f"test_write_sorted_return_orig_table"
    df, sql_schema = SIMPLE_TABLES_MAP[PART_SORT_TABLE_BASE_NAME]
    if use_dict_encoding_boxing:
        table_name += "_dict_encoding"

    if bodo.get_rank() == 0:
        spark = get_spark()
        create_iceberg_table(
            df,
            sql_schema,
            table_name,
            spark,
            PART_SORT_TABLE_PARTITION_SPEC,
            PART_SORT_TABLE_SORT_ORDER,
        )
    bodo.barrier()
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=True)

    @bodo.jit(distributed=["df"])
    def impl(df, table_name, conn, db_schema):
        df.to_sql(
            table_name,
            conn,
            db_schema,
            if_exists="append",
            index=False,
        )
        return df

    orig_use_dict_str_type = bodo.hiframes.boxing._use_dict_str_type
    bodo.hiframes.boxing._use_dict_str_type = use_dict_encoding_boxing

    try:
        out = impl(_get_dist_arg(df), table_name, conn, db_schema)
        out = _gather_output(out)
    finally:
        bodo.hiframes.boxing._use_dict_str_type = orig_use_dict_str_type
    bodo.barrier()

    passed = None
    if bodo.get_rank() == 0:
        passed = _test_equal_guard(
            df,
            out,
            # Do not sort since that defeats the purpose
            sort_output=False,
            check_dtype=False,
            reset_index=False,
        )

    passed = comm.bcast(passed)
    assert passed == 1, "Bodo function output doesn't match expected output"
