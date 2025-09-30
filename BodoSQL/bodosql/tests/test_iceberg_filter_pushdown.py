"""
Basic E2E tests for each type of filter pushdown on Iceberg tables.
"""

import io
import string
from datetime import date

import numpy as np
import pandas as pd
import pytest

import bodo
import bodosql
from bodo.spawn.utils import run_rank0
from bodo.tests.iceberg_database_helpers.utils import (
    PartitionField,
    create_iceberg_table,
)
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import check_func

pytestmark = pytest.mark.iceberg


@pytest.mark.slow
def test_filter_pushdown_in(iceberg_database):
    """
    Test for handling a simple IN filter.
    """
    table_name = "filter_pushdown_int_table_3"
    input_df = pd.DataFrame(
        {
            "ID": np.arange(10),
            "INT_COL_FILE_FILTER": [0 for i in range(5)] + [1 for i in range(5)],
        }
    )

    @run_rank0
    def setup():
        create_iceberg_table(
            input_df,
            [
                ("ID", "int", False),
                ("INT_COL_FILE_FILTER", "int", False),
            ],
            table_name,
            par_spec=[PartitionField("INT_COL_FILE_FILTER", "identity", -1)],
        )

    setup()
    db_schema, _ = iceberg_database()
    # TODO: Fix the FileSystemCatalog so that it can take in a full connection string
    # and not just a hardcoded path.
    catalog = bodosql.FileSystemCatalog(".")
    bc1 = bodosql.BodoSQLContext(catalog=catalog)

    query = f'SELECT * FROM "{db_schema}"."{table_name}" WHERE INT_COL_FILE_FILTER in (1,3,5)'

    def impl(bc, q):
        return bc.sql(q)

    output_df = input_df[input_df.INT_COL_FILE_FILTER == 1]

    check_func(
        impl,
        (bc1, query),
        check_names=True,
        check_dtype=False,
        reset_index=True,
        py_output=output_df,
    )

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        bodo.jit(impl)(bc1, query)
        check_logger_msg(
            stream,
            "ds.field('{INT_COL_FILE_FILTER}').isin(f0))",
        )


@pytest.mark.slow
def test_filter_pushdown_in_int_as_array(iceberg_database):
    """
    Test for handling a simple IN filter with numerous int values in the IN clause.
    """
    table_name = "filter_pushdown_int_table_4"
    input_df = pd.DataFrame(
        {
            "ID": np.arange(10),
            "INT_COL_FILE_FILTER": [0 for i in range(5)] + [1 for i in range(5)],
        }
    )

    @run_rank0
    def setup():
        create_iceberg_table(
            input_df,
            [
                ("ID", "int", False),
                ("INT_COL_FILE_FILTER", "int", False),
            ],
            table_name,
            par_spec=[PartitionField("INT_COL_FILE_FILTER", "identity", -1)],
        )

    setup()
    db_schema, warehouse_loc = iceberg_database()
    # TODO: Fix the FileSystemCatalog so that it can take in a full connection string
    # and not just a hardcoded path.
    catalog = bodosql.FileSystemCatalog(".")
    bc1 = bodosql.BodoSQLContext(catalog=catalog)

    filter_str = ", ".join([str(i) for i in range(1, 100, 2)])
    query = f'SELECT * FROM "{db_schema}"."{table_name}" WHERE INT_COL_FILE_FILTER in ({filter_str})'

    def impl(bc, q):
        return bc.sql(q)

    output_df = input_df[input_df.INT_COL_FILE_FILTER == 1]

    check_func(
        impl,
        (bc1, query),
        check_names=True,
        check_dtype=False,
        reset_index=True,
        py_output=output_df,
    )


@pytest.mark.slow
def test_filter_pushdown_in_str_as_array(iceberg_database):
    """
    Test for handling a simple IN filter with numerous str values in the IN clause.
    """
    table_name = "filter_pushdown_str_table_4"
    input_df = pd.DataFrame(
        {
            "ID": np.arange(10),
            "STR_COL_FILE_FILTER": ["a" for i in range(5)] + ["z" for i in range(5)],
        }
    )

    @run_rank0
    def setup():
        create_iceberg_table(
            input_df,
            [
                ("ID", "int", False),
                ("STR_COL_FILE_FILTER", "string", False),
            ],
            table_name,
            par_spec=[PartitionField("STR_COL_FILE_FILTER", "identity", -1)],
        )

    setup()
    db_schema, warehouse_loc = iceberg_database()
    # TODO: Fix the FileSystemCatalog so that it can take in a full connection string
    # and not just a hardcoded path.
    catalog = bodosql.FileSystemCatalog(".")
    bc1 = bodosql.BodoSQLContext(catalog=catalog)

    filter_str = ", ".join([repr(c) for c in string.ascii_lowercase[:-1]])
    query = f'SELECT * FROM "{db_schema}"."{table_name}" WHERE STR_COL_FILE_FILTER in ({filter_str})'

    def impl(bc, q):
        return bc.sql(q)

    output_df = input_df[input_df.STR_COL_FILE_FILTER == "a"]

    check_func(
        impl,
        (bc1, query),
        check_names=True,
        check_dtype=False,
        reset_index=True,
        py_output=output_df,
    )


@pytest.mark.skip("https://bodo.atlassian.net/browse/BSE-3125")
@pytest.mark.slow
def test_filter_pushdown_in_large_amount_of_date_scalars(iceberg_database):
    """
    Test for an issue related to python's bytecode.
    Specifically, if the number of arguments supplied to bodo's make_op function is greater than ~30,
    the calling convention will be changed to use an array instead of individual arguments.
    """
    table_name = "filter_pushdown_date_table_0"

    input_df = pd.DataFrame(
        {
            "A": pd.Series(
                [
                    date(2018, 11, 12),
                    date(2019, 11, 12),
                    date(2018, 12, 12),
                    date(2017, 11, 16),
                    None,
                    date(2017, 11, 30),
                    date(2016, 2, 3),
                    date(2019, 11, 12),
                    date(2018, 12, 20),
                    date(2017, 12, 12),
                ]
                * 5
            ),
        }
    )

    @run_rank0
    def setup():
        create_iceberg_table(
            input_df,
            [
                ("A", "date", True),
            ],
            table_name,
        )

    setup()
    db_schema, _ = iceberg_database()
    # TODO: Fix the FileSystemCatalog so that it can take in a full connection string
    # and not just a hardcoded path.
    catalog = bodosql.FileSystemCatalog(".")
    bc1 = bodosql.BodoSQLContext(catalog=catalog)

    filter_str = ", ".join(
        [repr(str(date(2018, 8 + i // 20, (i % 20) + 1))) for i in range(1, 100)]
    )
    query = f'SELECT * FROM "{db_schema}"."{table_name}" WHERE A in ({filter_str})'

    def impl(bc, q):
        return bc.sql(q)

    output_df = input_df[input_df.A == date(2018, 11, 12)]

    check_func(
        impl,
        (bc1, query),
        check_names=True,
        check_dtype=False,
        reset_index=True,
        py_output=output_df,
    )
