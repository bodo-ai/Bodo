# Copyright (C) 2024 Bodo Inc. All rights reserved.
"""
Test correctness of table functions that ping Snowflake for data.
"""

import datetime
import io
import os
import random
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from mpi4py import MPI
from numba.core import types
from numba.core.ir_utils import find_callname, guard

import bodo
import bodosql
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    check_logger_no_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import (
    SeriesOptTestPipeline,
    check_func,
    create_snowflake_table,
    drop_snowflake_table,
    gen_unique_table_id,
    get_snowflake_connection_string,
    pytest_mark_one_rank,
    pytest_snowflake,
)
from bodo.utils.typing import BodoError
from bodo.utils.utils import is_call_assign
from bodosql.tests.test_datetime_fns import compute_valid_times
from bodosql.tests.test_types.snowflake_catalog_common import (  # noqa
    snowflake_sample_data_conn_str,
    snowflake_sample_data_snowflake_catalog,
    test_db_snowflake_catalog,
)
from bodosql.tests.utils import check_query

pytestmark = pytest_snowflake
from decimal import Decimal

from bodo.tests.utils import check_func


def test_external_table_files(test_db_snowflake_catalog, datapath, memory_leak_check):
    def impl(bc, query):
        return bc.sql(query)

    query = 'SELECT * FROM TABLE(INFORMATION_SCHEMA.EXTERNAL_TABLE_FILES(TABLE_NAME=>\'"TEST_DB"."PUBLIC"."CITIES_EXTERNAL_TABLE"\'))'
    bc = bodosql.BodoSQLContext(catalog=test_db_snowflake_catalog)
    # Read the refsol and transform the non-string columns to the appropriate types
    answer = pd.read_csv(datapath("external_table_file_results.csv"))
    answer["REGISTERED_ON"] = answer["REGISTERED_ON"].apply(pd.Timestamp)
    answer["FILE_SIZE"] = answer["FILE_SIZE"].apply(Decimal)
    answer["LAST_MODIFIED"] = answer["LAST_MODIFIED"].apply(pd.Timestamp)
    check_func(
        impl,
        (bc, query),
        py_output=answer,
        check_names=False,
        check_dtype=False,
    )
