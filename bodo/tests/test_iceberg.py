import datetime
import io

# We need to import the connector first in order to apply our Py4J
# monkey-patch. Since PySpark uses Py4J, it will load in the functions
# we want to patch into memory; thus, without this import, those
# functions will be saved and used before we can change them
import bodo_iceberg_connector  # noqa
import numpy as np
import pandas as pd
import pytest
from mpi4py import MPI

import bodo
from bodo.tests.iceberg_database_helpers import spark_reader
from bodo.tests.iceberg_database_helpers.simple_tables import (
    TABLE_MAP as SIMPLE_TABLES_MAP,
)
from bodo.tests.iceberg_database_helpers.utils import get_spark
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import (
    _gather_output,
    _get_dist_arg,
    _test_equal_guard,
    check_func,
    reduce_sum,
    sync_dtypes,
)
from bodo.utils.typing import BodoError

pytestmark = pytest.mark.iceberg

WRITE_TABLES = [
    ("simple_bool_binary_table", SIMPLE_TABLES_MAP["simple_bool_binary_table"][0]),
    ("simple_dt_tsz_table", SIMPLE_TABLES_MAP["simple_dt_tsz_table"][0]),
    ("simple_tz_aware_table", SIMPLE_TABLES_MAP["simple_tz_aware_table"][0]),
    ("simple_dtype_list_table", SIMPLE_TABLES_MAP["simple_dtype_list_table"][0]),
    ("simple_numeric_table", SIMPLE_TABLES_MAP["simple_numeric_table"][0]),
    ("simple_string_table", SIMPLE_TABLES_MAP["simple_string_table"][0]),
    ("simple_list_table", SIMPLE_TABLES_MAP["simple_list_table"][0]),
    ("simple_struct_table", SIMPLE_TABLES_MAP["simple_struct_table"][0]),
    # TODO Needs investigation.
    pytest.param(
        ("simple_map_table", SIMPLE_TABLES_MAP["simple_map_table"][0]),
        marks=pytest.mark.skip(
            reason="Results in runtime error that's consistent with to_parquet."
        ),
    ),
    pytest.param(
        ("simple_decimals_table", SIMPLE_TABLES_MAP["simple_decimals_table"][0]),
        marks=pytest.mark.skip(
            reason="We don't suppport custom precisions and scale at the moment."
        ),
    ),
    pytest.param(
        (
            "simple_decimals_list_table",
            SIMPLE_TABLES_MAP["simple_decimals_list_table"][0],
        ),
        marks=pytest.mark.skip(
            reason="We don't suppport custom precisions and scale at the moment."
        ),
    ),
]


@pytest.fixture(
    params=WRITE_TABLES,
    ids=[x.values[0][0] if hasattr(x, "values") else x[0] for x in WRITE_TABLES],
)
def simple_dataframe(request):
    return request.param


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
def test_simple_table_read(iceberg_database, iceberg_table_conn, table_name):
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


def test_simple_tz_aware_table_read(iceberg_database, iceberg_table_conn):
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


def test_simple_numeric_table_read(iceberg_database, iceberg_table_conn):
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
    res = bodo.jit()(impl)(table_name, conn, db_schema)
    py_out = sync_dtypes(py_out, res.dtypes.values.tolist())
    check_func(impl, (table_name, conn, db_schema), py_output=py_out)


@pytest.mark.slow
@pytest.mark.parametrize(
    "table_name", ["simple_list_table", "simple_decimals_list_table"]
)
def test_simple_list_table_read(iceberg_database, iceberg_table_conn, table_name):
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


def test_simple_bool_binary_table_read(iceberg_database, iceberg_table_conn):
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


def test_simple_struct_table_read(iceberg_database, iceberg_table_conn):
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


def test_filter_pushdown_partitions(iceberg_database, iceberg_table_conn):
    """
    Test that simple date based partitions can be read as expected.
    """

    table_name = "partitions_dt_table"
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)

    def impl(table_name, conn, db_schema):
        df = pd.read_sql_table(table_name, conn, db_schema)
        df = df[df["A"] <= datetime.date(2018, 12, 12)]  # type: ignore
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
    table_name, df = simple_dataframe
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)

    def impl(df, table_name, conn, db_schema):
        df.to_sql(table_name, conn, db_schema, if_exists="fail")

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


# TODO Add unit test for appending to a table written by Spark


@pytest.mark.parametrize("read_behavior", ["spark", "bodo"])
def test_basic_write_replace(
    iceberg_database,
    iceberg_table_conn,
    simple_dataframe,
    read_behavior,
):
    """Test basic Iceberg table replace on Spark table"""

    comm = MPI.COMM_WORLD
    n_pes = comm.Get_size()
    table_name, df = simple_dataframe
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)

    def impl(df, table_name, conn, db_schema):
        df.to_sql(table_name, conn, db_schema, if_exists="replace")

    # Write using Bodo
    bodo.jit(distributed=["df"])(impl)(_get_dist_arg(df), table_name, conn, db_schema)
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
def test_basic_write_new_append(
    iceberg_database,
    iceberg_table_conn,
    simple_dataframe,
    behavior,
):
    """
    Test basic Iceberg table write + append on new table
    (append to table written by Bodo)
    """

    comm = MPI.COMM_WORLD
    n_pes = comm.Get_size()
    table_name, df = simple_dataframe
    # We want to use completely new table for each test
    table_name += f"_{behavior}"
    db_schema, warehouse_loc = iceberg_database
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)

    def create_impl(df, table_name, conn, db_schema):
        df.to_sql(table_name, conn, db_schema, if_exists="append")

    impl = bodo.jit(distributed=["df"])(create_impl)

    # Write using Bodo (this will create the table and add the rows)
    impl(_get_dist_arg(df), table_name, conn, db_schema)
    if behavior == "append":
        # Write again to do the actual append
        impl(_get_dist_arg(df), table_name, conn, db_schema)

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

    spark_out, _, _ = spark_reader.read_iceberg_table(table_name, db_schema)

    # Uncomment if we get Spark to be able to read this table (see comment above)
    # if table_name == "simple_struct_table":
    #     spark_out["A"] = spark_out["A"].map(lambda x: {"a": x["a"], "b": x["b"]})
    #     spark_out["B"] = spark_out["B"].map(
    #         lambda x: {"a": x["a"], "b": x["b"], "c": x["c"]}
    #     )

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

    df = pd.DataFrame(
        {
            "A": np.array([True, False, True, True] * 25, dtype=np.bool_),
            "B": np.array([False, None] * 50, dtype=np.bool_),
            "C": np.array([1, 1, 0, 1, 0] * 20).tobytes(),
        }
    )

    # Check that error is raised when schema is not provided
    def impl1(df, table_name, conn):
        df.to_sql(table_name, conn)

    with pytest.raises(
        ValueError,
        match="schema must be provided when writing to an Iceberg table",
    ):
        bodo.jit(distributed=["df"])(impl1)(df, table_name, conn)

    # Check that error is raised when chunksize is provided
    def impl2(df, table_name, conn, db_schema):
        df.to_sql(table_name, conn, db_schema, chunksize=5)

    with pytest.raises(ValueError, match="chunksize not supported for Iceberg tables"):
        bodo.jit(distributed=["df"])(impl2)(df, table_name, conn, db_schema)

    # TODO Remove after adding replicated write support
    # Check that error is raise when trying to write a replicated dataframe
    # (unsupported for now)
    def impl3(df, table_name, conn, db_schema):
        df.to_sql(table_name, conn, db_schema)

    with pytest.raises(
        AssertionError, match="Iceberg Write only supported for distributed dataframes"
    ):
        bodo.jit(replicated=["df"])(impl3)(df, table_name, conn, db_schema)
