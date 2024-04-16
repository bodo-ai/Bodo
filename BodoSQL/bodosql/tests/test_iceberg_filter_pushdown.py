# Copyright (C) 2024 Bodo Inc. All rights reserved.
"""
Basic E2E tests for each type of filter pushdown on Iceberg tables.
"""
import io

import numpy as np
import pandas as pd
import pytest

import bodo
import bodosql
from bodo.tests.iceberg_database_helpers.utils import (
    PartitionField,
    create_iceberg_table,
)
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import check_func, run_rank0

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
    db_schema, warehouse_loc = iceberg_database()
    # TODO: Fix the FileSystemCatalog so that it can take in a full connection string
    # and not just a hardcoded path.
    catalog = bodosql.FileSystemCatalog(".")
    bc1 = bodosql.BodoSQLContext(catalog=catalog)

    query = f'SELECT * FROM "{db_schema}"."{table_name}" WHERE INT_COL_FILE_FILTER in (1,3,5,7,9)'

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
        # IN statement should be expanded to an OR of multiple equality statements
        for i in range(5):
            check_logger_msg(
                stream,
                f"bic.FilterExpr('==', [bic.ColumnRef('INT_COL_FILE_FILTER'), bic.Scalar(f{i})])",
            )
