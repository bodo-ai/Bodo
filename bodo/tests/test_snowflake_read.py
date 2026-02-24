"""
Tests for reading from Snowflake using Python APIs
"""

import datetime
import io
import json
import os
import re
from decimal import Decimal
from typing import TYPE_CHECKING

import numba  # noqa TID253
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from mpi4py import MPI
from numba.core import types  # noqa TID253

import bodo
from bodo import BodoWarning
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    check_logger_no_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import (
    check_func,
    create_snowflake_table,
    enable_timestamptz,
    get_snowflake_connection_string,
    pytest_mark_one_rank,
    pytest_snowflake,
    temp_env_override,
)

if TYPE_CHECKING:  # pragma: no cover
    from pytest_mock import MockerFixture
    from snowflake.connector import SnowflakeConnection


pytestmark = pytest_snowflake


def test_snowflake_basic_read(memory_leak_check):
    def impl(query, conn):
        df = pd.read_sql(query, conn)
        return df

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    # need to sort the output to make sure pandas and Bodo get the same rows
    query = "SELECT * FROM LINEITEM ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY LIMIT 70"
    check_func(impl, (query, conn))


def test_sql_snowflake_distributed_false(memory_leak_check):
    """
    Basic test for is_independent flag in Snowflake I/O, which is used to handle
    independent I/O calls in @bodo.jit(distributed=False) cases
    """

    def impl(query, conn):
        df = pd.read_sql(query, conn)
        return df

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)

    query = "SELECT * FROM LINEITEM ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY LIMIT 70"
    bodo_df = bodo.jit(distributed=False)(impl)(query, conn)
    pandas_df = impl(query, conn)

    pd.testing.assert_frame_equal(
        bodo_df, pandas_df, check_dtype=False, check_column_type=False
    )


def test_sql_snowflake_independent(memory_leak_check):
    """
    Make sure all ranks execute independently in the
    @bodo.jit(distributed) case that has a Snowflake read call.

    By putting a barrier on rank 0 first, and then non-zero ranks
    afterwards, we ensure that all other ranks must complete before rank 0,
    which means that all ranks must execute independently.
    """
    # initialize global node_ranks before compiling to avoid hangs
    bodo.get_nodes_first_ranks()

    def impl(query, conn):
        df = pd.read_sql(query, conn)
        return df

    query = "SELECT * FROM LINEITEM ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY LIMIT 70"
    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)

    if bodo.get_rank() == 0:
        bodo.barrier()

    bodo_df = bodo.jit(distributed=False)(impl)(query, conn)

    if bodo.get_rank() != 0:
        bodo.barrier()

    pandas_df = impl(query, conn)
    pd.testing.assert_frame_equal(
        bodo_df, pandas_df, check_dtype=False, check_column_type=False
    )


def test_snowflake_performance_warning(memory_leak_check):
    """
    Test that we raise a warning if we detect that the platform is in a different
    region than the snowflake account.
    """

    @bodo.jit
    def impl(query, conn):
        df = pd.read_sql(query, conn)
        return df

    old_region_info = os.environ.get("BODO_PLATFORM_WORKSPACE_REGION", None)
    try:
        # The snowflake account is in us-east-1. However we use canada
        # to avoid a potential change breaking tests.
        os.environ["BODO_PLATFORM_WORKSPACE_REGION"] = "ca-central-1"

        db = "SNOWFLAKE_SAMPLE_DATA"
        schema = "TPCH_SF1"
        conn = get_snowflake_connection_string(db, schema)
        # need to sort the output to make sure pandas and Bodo get the same rows
        query = "SELECT * FROM LINEITEM LIMIT 10"
        # We only throw the warning on rank0
        if bodo.get_rank() == 0:
            with pytest.warns(
                BodoWarning,
                match="The Snowflake warehouse and Bodo platform are in different cloud regions",
            ):
                impl(query, conn)
        else:
            impl(query, conn)
    finally:
        # Restore the region info.
        if old_region_info is None:
            del os.environ["BODO_PLATFORM_WORKSPACE_REGION"]
        else:
            os.environ["BODO_PLATFORM_WORKSPACE_REGION"] = old_region_info


@pytest.fixture(scope="session")
def snowflake_conn():
    """
    Temporary Snowflake Connection for Basic Testing
    """
    import bodo.decorators  # isort:skip # noqa
    from bodo.io.snowflake import snowflake_connect

    db = "TEST_DB"
    schema = "PUBLIC"
    conn_str = get_snowflake_connection_string(db, schema)
    conn: SnowflakeConnection = snowflake_connect(conn_str)
    yield conn
    conn.close()


@pytest.fixture
def cursor(snowflake_conn: "SnowflakeConnection"):
    """
    Temporary Snowflake Connection for Basic Testing
    """
    cursor = snowflake_conn.cursor()
    yield cursor
    cursor.close()


@pytest_mark_one_rank
def test_decimal_metadata_handling(cursor):
    """
    Test that Bodo's Snowflake schema inference can
    determine the correct output size of various
    """
    import bodo.io.snowflake

    # Test an int, double, maybe decimal, and decimal col for typing
    pa_fields = bodo.io.snowflake.get_schema_from_metadata(
        cursor,
        "SELECT 10, 10.11, 1.1010101::decimal(30, 8), 12345678901234567890.0987654321",
        None,
        None,
        True,
        False,
        False,
    )[0]

    assert len(pa_fields) == 4
    int_col, double_col, maybe_dec_col, dec_col = pa_fields
    assert pa.types.is_int8(int_col.type)
    assert pa.types.is_float64(double_col.type)
    assert pa.types.is_float64(maybe_dec_col.type)
    assert (
        pa.types.is_decimal128(dec_col.type)
        and dec_col.type.precision == 30
        and dec_col.type.scale == 10
    )


@pytest_mark_one_rank
def test_array_metadata_handling(cursor):
    """
    Test that Array Columns are Properly Typed
    """
    import bodo.io.snowflake

    pa_fields = bodo.io.snowflake.get_schema_from_metadata(
        cursor,
        """
        select null as A
        union all
        select ARRAY_CONSTRUCT(CURRENT_DATE(), '1980-01-05'::date, null) as A
        """,
        None,
        None,
        True,
        False,
        False,
    )[0]

    assert len(pa_fields) == 1
    assert pa_fields[0].equals(pa.field("A", pa.large_list(pa.date32()), nullable=True))


@pytest_mark_one_rank
def test_array_metadata_handling_err(cursor):
    """
    Test that an error is raised when an array item column
    has multiple value types that are incompatible
    """
    import bodo.io.snowflake
    from bodo.utils.typing import BodoError

    with pytest.raises(
        BodoError,
        match=r"type list\[variant\]. We are unable to narrow the type further, because the `variant` content has items of types \['DATE', 'VARCHAR'\]",
    ):
        bodo.io.snowflake.get_schema_from_metadata(
            cursor,
            """
            select null as A
            union all
            select ARRAY_CONSTRUCT(CURRENT_DATE(), '1980-01-05', null) as A
            """,
            None,
            None,
            True,
            False,
            False,
        )


@pytest_mark_one_rank
def test_map_metadata_handling(cursor):
    """
    Test that Object Columns are Properly Typed in Map
    """
    import bodo.io.snowflake

    pa_fields = bodo.io.snowflake.get_schema_from_metadata(
        cursor,
        """
        select null as A
        union all
        select OBJECT_CONSTRUCT_KEEP_NULL('a', CURRENT_DATE(), 'b', '1980-01-05'::date, 'c', null) as A
        """,
        None,
        None,
        True,
        False,
        False,
    )[0]

    assert len(pa_fields) == 1
    assert pa_fields[0].equals(
        pa.field("A", pa.map_(pa.large_string(), pa.date32()), nullable=True)
    )


@pytest_mark_one_rank
def test_struct_metadata_handling(cursor):
    """
    Test that Object Columns are Typed in Struct when Map is Invalid
    Tests the additional scenarios
    - B: Nulls with other types
    - C: Only Nulls
    - D: Different numeric types together (int, decimal, float)
    - E: Field not in all rows
    """
    import bodo.io.snowflake

    pa_fields = bodo.io.snowflake.get_schema_from_metadata(
        cursor,
        """
        select null as A
        union all
        select OBJECT_CONSTRUCT_KEEP_NULL('a', '1980-01-05'::date, 'b', true, 'c', null, 'd', 10, 'e', 'test') as A
        union all
        select OBJECT_CONSTRUCT_KEEP_NULL('a', CURRENT_DATE(), 'b', null, 'c', null, 'd', 'nan'::double) as A
        """,
        None,
        None,
        True,
        False,
        False,
    )[0]

    assert len(pa_fields) == 1
    assert pa_fields[0].name == "A"
    assert pa_fields[0].nullable

    stype = pa_fields[0].type
    assert pa.types.is_struct(stype)
    for f in (
        pa.field("a", pa.date32(), nullable=True),
        pa.field("b", pa.bool_(), nullable=True),
        pa.field("c", pa.null(), nullable=True),
        pa.field("d", pa.float64(), nullable=True),
        pa.field("e", pa.large_string(), nullable=True),
    ):
        assert stype.field(stype.get_field_index(f.name)).equals(f)


@pytest_mark_one_rank
def test_struct_metadata_handling_err_multiple_types(cursor):
    """
    Test that an error is raised when a object column has a field
    with different types in different rows
    """
    import bodo.io.snowflake
    from bodo.utils.typing import BodoError

    with pytest.raises(
        BodoError, match=r"containing multiple types \['DATE', 'INTEGER'\]"
    ):
        bodo.io.snowflake.get_schema_from_metadata(
            cursor,
            """
            select OBJECT_CONSTRUCT_KEEP_NULL('a', 10) as A
            union all
            select OBJECT_CONSTRUCT_KEEP_NULL('a', CURRENT_DATE(), 'b', '1980-01-05', 'c', null) as A
            """,
            None,
            None,
            True,
            False,
            False,
        )


@pytest_mark_one_rank
def test_struct_metadata_handling_err_uncommon_field(cursor):
    """
    Test that an error is raised when a object column with heterogenous types
    is assumed to be a struct, but there is a one-off or very uncommon field
    implying otherwise.
    """
    import bodo.io.snowflake
    from bodo.utils.typing import BodoError

    with pytest.raises(BodoError, match=r"has a field d in < 0.5% of non-null rows"):
        bodo.io.snowflake.get_schema_from_metadata(
            cursor,
            """
            with thousand as (
                select
                    OBJECT_CONSTRUCT_KEEP_NULL('a', CURRENT_DATE(), 'b', '1980-01-05', 'c', null) as A
                from table(generator(ROWCOUNT=>1000))
            )

            select OBJECT_CONSTRUCT_KEEP_NULL('b', '10', 'd', 'bad') as A
            union all
            select A from thousand
            """,
            None,
            None,
            True,
            False,
            False,
        )


@pytest_mark_one_rank
def test_variant_metadata_handling(cursor):
    """
    Test that Variant Columns are typed correctly, with the following original types:
    V1: timestamp_ntz (column casted as variant)
    V2: number(38, 0) (column casted as variant)
    V3: float (column casted as variant)
    V4: boolean (column casted as variant)
    V5: map[str,float] (map column stored as variant in Snowflake)
    V6: bigint (field pushdown on struct column stored as variant in Snowflake)
    V7: array[struct[A:bigint,B:array[varchar]]] (parse json)
    """
    import bodo.io.snowflake

    pa_fields = bodo.io.snowflake.get_schema_from_metadata(
        cursor,
        """
        select
            ts_create::variant as V1,
            id::variant as V2,
            tusd::variant as V3,
            ifd::variant as V4,
            json_ffdubt as V5,
            json_rnks['b'] as V6,
            parse_json('[{"A": 42, "B": []}, {"A": -1, "B": ["C", null, "D"]}, {"A": null, "B": null}]') as V7
        from can_brod
        """,
        None,
        None,
        True,
        False,
        False,
    )[0]

    assert len(pa_fields) == 7
    assert [f.name for f in pa_fields] == [f"V{i + 1}" for i in range(7)]
    assert all(f.nullable for f in pa_fields)

    target_types = [
        pa.timestamp("ns"),
        pa.int64(),
        pa.float64(),
        pa.bool_(),
        pa.map_(pa.large_string(), pa.float64()),
        pa.int64(),
        pa.large_list(
            pa.struct(
                [
                    pa.field("A", pa.int64()),
                    pa.field("B", pa.large_list(pa.large_string())),
                ]
            )
        ),
    ]
    for i in range(6):
        assert pa_fields[i].type == target_types[i]


@pytest_mark_one_rank
def test_float_array_metadata_handling(cursor):
    """
    Test that Numeric Array Columns, that may contain Integers, Floats, and Decimals
    are properly typed to float64 arrays
    """
    import bodo.io.snowflake

    pa_fields = bodo.io.snowflake.get_schema_from_metadata(
        cursor,
        """
        select null as A
        union all
        select ARRAY_CONSTRUCT(10, 10.0, null) as A
        union all
        select ARRAY_CONSTRUCT(null, 12.4, -0.57) as A
        union all
        select ARRAY_CONSTRUCT(12345678901234567890) as A
        union all
        select ARRAY_CONSTRUCT(-1235, 0.01234567890123456789) as A
        """,
        None,
        None,
        True,
        False,
        False,
    )[0]

    assert len(pa_fields) == 1
    assert pa_fields[0].equals(
        pa.field("A", pa.large_list(pa.float64()), nullable=True)
    )


@pytest_mark_one_rank
def test_nested_in_array_metadata_handling(cursor):
    """
    Test that nested semi-structured data in Array Columns are properly typed
    """
    import bodo.io.snowflake

    pa_fields = bodo.io.snowflake.get_schema_from_metadata(
        cursor,
        "SELECT A, B, C FROM NESTED_ARRAY_TEST",
        None,
        None,
        True,
        False,
        False,
    )[0]

    assert len(pa_fields) == 3
    assert pa_fields[0].equals(
        pa.field("A", pa.large_list(pa.large_list(pa.float64())))
    )
    assert pa_fields[1].equals(
        pa.field("B", pa.large_list(pa.map_(pa.large_string(), pa.date32())))
    )
    # Order of Object Fields is Non-Deterministic
    assert pa_fields[2].name == "C"
    assert pa.types.is_large_list(pa_fields[2].type)
    inner_struct = pa_fields[2].type.value_type
    assert pa.types.is_struct(inner_struct)
    assert inner_struct.num_fields == 3
    for f in (
        pa.field("stat", pa.bool_()),
        pa.field("name", pa.large_string()),
        pa.field("cnt", pa.int64()),
    ):
        assert inner_struct.field(inner_struct.get_field_index(f.name)).equals(f)


@pytest_mark_one_rank
def test_array_in_array_metadata_handling_err(cursor):
    import bodo.io.snowflake
    from bodo.utils.typing import BodoError

    with pytest.raises(
        BodoError,
        match=r"type list\[list\[variant\]\]. We are unable to narrow the type further, because the `variant` content has items of types \['BOOLEAN', 'INTEGER'\]",
    ):
        bodo.io.snowflake.get_schema_from_metadata(
            cursor,
            """
            select ARRAY_CONSTRUCT(ARRAY_CONSTRUCT(10, 10.0), ARRAY_CONSTRUCT(true, false)) as A
            """,
            None,
            None,
            True,
            False,
            False,
        )


@pytest_mark_one_rank
def test_map_in_array_metadata_handling_err(cursor):
    import bodo.io.snowflake
    from bodo.utils.typing import BodoError

    with pytest.raises(
        BodoError,
        match=r"type list\[map\[str, list\[variant\]\]\]. We are unable to narrow the type further, because the `variant` content has items of types \['BOOLEAN', 'INTEGER'\]",
    ):
        bodo.io.snowflake.get_schema_from_metadata(
            cursor,
            """
            select ARRAY_CONSTRUCT(
                OBJECT_CONSTRUCT_KEEP_NULL('a', ARRAY_CONSTRUCT(10, 10.0, false))
            ) as main
            """,
            None,
            None,
            True,
            False,
            False,
        )


@pytest_mark_one_rank
def test_struct_in_array_metadata_handling_err(cursor):
    import bodo.io.snowflake
    from bodo.utils.typing import BodoError

    with pytest.raises(
        BodoError,
        match=r"type list\[struct\[... b: variant ...\]\]. We are unable to narrow the type further, because field b was found containing multiple types \['INTEGER', 'VARCHAR'\]",
    ):
        bodo.io.snowflake.get_schema_from_metadata(
            cursor,
            """
            select ARRAY_CONSTRUCT(
                OBJECT_CONSTRUCT_KEEP_NULL(
                    'a', ARRAY_CONSTRUCT(10, 10.0),
                    'b', 'test'
                ),
                OBJECT_CONSTRUCT_KEEP_NULL('a', null, 'b', 10)
            ) as main
            """,
            None,
            None,
            True,
            False,
            False,
        )


@pytest_mark_one_rank
def test_nested_in_map_metadata_handling(cursor):
    """
    Test that nested semi-structured data in Map Columns are properly typed
    """
    import bodo.io.snowflake

    pa_fields = bodo.io.snowflake.get_schema_from_metadata(
        cursor,
        "SELECT A, B, C FROM NESTED_MAP_TEST",
        None,
        None,
        True,
        False,
        False,
    )[0]

    assert len(pa_fields) == 3
    assert pa_fields[0].equals(
        pa.field("A", pa.map_(pa.large_string(), pa.large_list(pa.float64())))
    )
    assert pa_fields[1].equals(
        pa.field(
            "B", pa.map_(pa.large_string(), pa.map_(pa.large_string(), pa.date32()))
        )
    )
    # Order of Object Fields is Non-Deterministic
    assert pa_fields[2].name == "C"
    assert pa.types.is_map(pa_fields[2].type)
    assert pa.types.is_large_string(pa_fields[2].type.key_type)
    inner_struct = pa_fields[2].type.item_type
    assert pa.types.is_struct(inner_struct)
    assert inner_struct.num_fields == 3
    for f in (
        pa.field("stat", pa.bool_()),
        pa.field("name", pa.large_string()),
        pa.field("cnt", pa.int64()),
    ):
        assert inner_struct.field(inner_struct.get_field_index(f.name)).equals(f)


@pytest_mark_one_rank
def test_array_in_map_metadata_handling_err(cursor):
    import bodo.io.snowflake
    from bodo.utils.typing import BodoError

    with pytest.raises(
        BodoError,
        match=r"type map\[str, list\[variant\]\]. We are unable to narrow the type further, because the `variant` content has items of types \['BOOLEAN', 'INTEGER'\]",
    ):
        bodo.io.snowflake.get_schema_from_metadata(
            cursor,
            """
            select OBJECT_CONSTRUCT_KEEP_NULL(
                'a', ARRAY_CONSTRUCT(10, 10.0),
                'b', ARRAY_CONSTRUCT(true, false)
            ) as A
            """,
            None,
            None,
            True,
            False,
            False,
        )


@pytest_mark_one_rank
def test_struct_in_map_metadata_handling_err(cursor):
    import bodo.io.snowflake
    from bodo.utils.typing import BodoError

    with pytest.raises(
        BodoError,
        match=r"type map\[str, struct\[... name: variant ...\]\]. We are unable to narrow the type further, because field name was found containing multiple types \['ARRAY', 'VARCHAR'\]",
    ):
        bodo.io.snowflake.get_schema_from_metadata(
            cursor,
            """
            select OBJECT_CONSTRUCT_KEEP_NULL(
                'bodo', OBJECT_CONSTRUCT_KEEP_NULL(
                    'name', ARRAY_CONSTRUCT(10, 10.0),
                    'owner', 'test'
                ),
                'data', OBJECT_CONSTRUCT_KEEP_NULL('name', 'data', 'owner', 'tester')
            ) as main
            """,
            None,
            None,
            True,
            False,
            False,
        )


@pytest_mark_one_rank
def test_nested_in_struct_metadata_handling(cursor):
    """
    Test that nested semi-structured data in Struct Columns are properly typed.
    Also tests multiple levels of nesting
    """
    import bodo.io.snowflake

    pa_fields = bodo.io.snowflake.get_schema_from_metadata(
        cursor,
        "SELECT info FROM NESTED_STRUCT_TEST",
        None,
        None,
        True,
        False,
        False,
    )[0]

    assert len(pa_fields) == 1
    f = pa_fields[0]
    assert f.name == "INFO"
    assert f.nullable

    stype = pa_fields[0].type
    assert pa.types.is_struct(stype)
    assert stype.num_fields == 5
    for f in (
        pa.field("group", pa.large_string()),
        pa.field("updated", pa.timestamp("ns")),
        pa.field("values", pa.large_list(pa.float64())),
        pa.field("ids", pa.map_(pa.large_string(), pa.int64())),
    ):
        assert stype.field(stype.get_field_index(f.name)).equals(f)

    inner_sfield = stype.field(stype.get_field_index("created"))
    assert inner_sfield.nullable
    assert pa.types.is_struct(inner_sfield.type)
    assert inner_sfield.type.num_fields == 3
    for f in (
        pa.field("creator", pa.large_string()),
        pa.field("at", pa.date32()),
        pa.field("atnew", pa.large_list(pa.int64())),
    ):
        assert inner_sfield.type.field(
            inner_sfield.type.get_field_index(f.name)
        ).equals(f)


@pytest.fixture
def bodo_schema(request):
    """Lazily create Bodo schemas to avoid importing JIT at collection time."""
    import bodo.decorators  # noqa

    val = request.param

    # TODO: Use numba_from_pyarrow to simplify parameterization
    if val == "bodo_int64":
        return bodo.types.DataFrameType(
            data=(
                types.Array(types.int64, 1, "C"),
                types.Array(types.int64, 1, "C"),
                types.Array(types.int64, 1, "C"),
            ),
            columns=("l_orderkey", "l_partkey", "l_suppkey"),
        )

    elif val == "bodo_int32":
        return bodo.types.DataFrameType(
            data=(
                types.Array(types.int32, 1, "C"),
                types.Array(types.int32, 1, "C"),
                types.Array(types.int32, 1, "C"),
            ),
            columns=("l_orderkey", "l_partkey", "l_suppkey"),
        )

    elif val == "bodo_decimal":
        return bodo.types.DataFrameType(
            data=(
                bodo.types.DecimalArrayType(38, 0),
                bodo.types.DecimalArrayType(38, 0),
                bodo.types.DecimalArrayType(18, 0),
            ),
            columns=("l_orderkey", "l_partkey", "l_suppkey"),
        )

    elif val == "bodo_decimal_int":
        return bodo.types.DataFrameType(
            data=(
                bodo.types.DecimalArrayType(38, 0),
                types.Array(types.int32, 1, "C"),
                types.Array(types.int32, 1, "C"),
            ),
            columns=("l_orderkey", "l_partkey", "l_suppkey"),
        )


@pytest.mark.parametrize(
    "bodo_schema,pa_schema",
    [
        # All larger int
        (
            "bodo_int64",
            pa.schema(
                [
                    pa.field("L_ORDERKEY", pa.int64(), nullable=False),
                    pa.field("L_PARTKEY", pa.int64(), nullable=False),
                    pa.field("L_SUPPKEY", pa.int64(), nullable=False),
                ]
            ),
        ),
        # Last column is larger int
        (
            "bodo_int32",
            pa.schema(
                [
                    pa.field("L_ORDERKEY", pa.int32(), nullable=False),
                    pa.field("L_PARTKEY", pa.int32(), nullable=False),
                    pa.field("L_SUPPKEY", pa.int32(), nullable=False),
                ]
            ),
        ),
    ],
    indirect=["bodo_schema"],
)
def test_snowflake_runtime_upcasting_int_to_int(
    mocker: "MockerFixture",
    bodo_schema,
    pa_schema,
    memory_leak_check,
):
    """
    Test that Bodo can handles a scenario where the compile-time
    schema uses larger types than the runtime data for integers.
    """
    # Mock the compile-time schema info
    # Original (and runtime data):
    #    L_ORDERKEY: int32 not null
    #    L_PARTKEY: int32 not null
    #    L_SUPPKEY: int16 not null
    mocker.patch(
        "bodo.io.snowflake.get_schema",
        return_value=(
            bodo_schema,
            {"l_orderkey", "l_partkey", "l_suppkey"},
            [],
            [],
            pa_schema,
            None,
            None,
        ),
    )

    def impl(query, conn):
        df = pd.read_sql(query, conn)
        return df

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    # need to sort the output to make sure pandas and Bodo get the same rows
    query = "SELECT L_ORDERKEY, L_PARTKEY, L_SUPPKEY FROM LINEITEM ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY LIMIT 70"
    check_func(impl, (query, conn), check_dtype=False)


# TODO: Use numba_from_pyarrow to simplify parameterization
@pytest.mark.parametrize(
    "bodo_schema,pa_schema",
    [
        # All are larger decimal
        (
            "bodo_decimal",
            pa.schema(
                [
                    pa.field("L_ORDERKEY", pa.decimal128(38, 0), nullable=False),
                    pa.field("L_PARTKEY", pa.decimal128(38, 0), nullable=False),
                    pa.field("L_SUPPKEY", pa.decimal128(18, 0), nullable=False),
                ]
            ),
        ),
        # First column is larger decimal
        (
            "bodo_decimal_int",
            pa.schema(
                [
                    pa.field("L_ORDERKEY", pa.decimal128(38, 0), nullable=False),
                    pa.field("L_PARTKEY", pa.int32(), nullable=False),
                    pa.field("L_SUPPKEY", pa.int32(), nullable=False),
                ]
            ),
        ),
    ],
    indirect=["bodo_schema"],
)
def test_snowflake_runtime_upcasting_int_to_decimal(
    mocker: "MockerFixture",
    bodo_schema,
    pa_schema,
    memory_leak_check,
):
    """
    Test that Bodo can handles a scenario where the compile-time
    schema uses larger types than the runtime data for integers.
    The larger type is a decimal type
    """
    from bodo.spawn.utils import run_rank0

    # Mock the compile-time schema info
    # Original (and runtime data):
    #    L_ORDERKEY: int32 not null
    #    L_PARTKEY: int32 not null
    #    L_SUPPKEY: int16 not null
    mocker.patch(
        "bodo.io.snowflake.get_schema",
        return_value=(
            bodo_schema,
            {"l_orderkey", "l_partkey", "l_suppkey"},
            [],
            [],
            pa_schema,
            None,
            None,
        ),
    )

    def impl(query, conn):
        df = pd.read_sql(query, conn)
        return df

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    # need to sort the output to make sure pandas and Bodo get the same rows
    query = "SELECT L_ORDERKEY, L_PARTKEY, L_SUPPKEY FROM LINEITEM ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY LIMIT 70"

    @run_rank0
    def read_to_decimal():
        df = pd.read_sql(query, conn)
        for field in pa_schema:
            if pa.types.is_decimal128(field.type):
                df[field.name.lower()] = df[field.name.lower()].apply(
                    lambda x: Decimal(x)
                )
        return df

    df = read_to_decimal()

    check_func(
        impl,
        (query, conn),
        py_output=df,
        sort_output=True,
        reset_index=True,
        check_dtype=False,
    )


def test_snowflake_runtime_upcasting_timestamp(memory_leak_check):
    """
    Test that Bodo can handles a scenario where the compile-time
    schema uses larger types than the runtime data for time and
    timestamp data
    """
    # Bodo seems to not correctly get time from Snowflake
    # so ignoring in test
    #    TIME_SEC TIME(0)           time("s")
    #    TIME_MILLI TIME(3)         time("ms")
    #    TIME_MICRO TIME(6)         time("us")
    #    TIME_NANO TIME(9)          time("ns")
    # Bodo will already upcast timestamp to nanoseconds internally
    #    LTZ_SEC TIMESTAMP_LTZ(0)   timestamptz
    #    LTZ_MILLI TIMESTAMP_LTZ(3) timestamptz
    #    LTZ_MICRO TIMESTAMP_LTZ(6) timestamptz
    #    LTZ_NANO TIMESTAMP_LTZ(9)  timestamptz
    #    NTZ_SEC TIMESTAMP_NTZ(0)   timestamp
    #    NTZ_MILLI TIMESTAMP_NTZ(3) timestamp
    #    NTZ_MICRO TIMESTAMP_NTZ(6) timestamp
    #    NTZ_NANO TIMESTAMP_NTZ(9)  timestamp
    # Bodo doesn't support TIMESTAMP_TZ
    #    TZ_SEC TIMESTAMP_TZ(0)     timestamptz
    #    TZ_MILLI TIMESTAMP_TZ(3)   timestamptz
    #    TZ_MICRO TIMESTAMP_TZ(6)   timestamptz
    #    TZ_NANO TIMESTAMP_TZ(9)    timestamptz

    def impl(query, conn):
        df = pd.read_sql(query, conn)
        return df

    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema)
    query = "SELECT LTZ_SEC, LTZ_MILLI, LTZ_MICRO, LTZ_NANO, NTZ_SEC, NTZ_MILLI, NTZ_MICRO, NTZ_NANO FROM TIMESTAMP_UNIT_TEST ORDER BY TIME_SEC"
    check_func(impl, (query, conn), check_dtype=False)


def test_snowflake_runtime_downcasting_ints_fail(mocker: "MockerFixture"):
    """
    Check that Bodo throws an error when the runtime schema
    is larger than the compile-time schema
    """
    mocker.patch(
        "bodo.io.snowflake.get_schema",
        return_value=(
            bodo.types.DataFrameType(
                data=(
                    types.Array(types.int8, 1, "C"),
                    types.Array(types.int8, 1, "C"),
                    types.Array(types.int8, 1, "C"),
                ),
                columns=("l_orderkey", "l_partkey", "l_suppkey"),
            ),
            {"l_orderkey", "l_partkey", "l_suppkey"},
            [],
            [],
            pa.schema(
                [
                    pa.field("L_ORDERKEY", pa.int8(), nullable=False),
                    pa.field("L_PARTKEY", pa.int8(), nullable=False),
                    pa.field("L_SUPPKEY", pa.int8(), nullable=False),
                ]
            ),
            None,
            None,
        ),
    )

    def impl(query, conn):
        df = pd.read_sql(query, conn)
        return df

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    query = "SELECT L_ORDERKEY, L_PARTKEY, L_SUPPKEY FROM LINEITEM ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY LIMIT 70"

    with pytest.raises(RuntimeError, match="Invalid Downcast from int32 to int8"):
        bodo.jit(impl)(query, conn)


def test_snowflake_runtime_downcasting_timestamp_fail(mocker: "MockerFixture"):
    """
    Check that Bodo throws an error when the runtime schema
    is larger than the compile-time schema in timestamps
    """
    mocker.patch(
        "bodo.io.snowflake.get_schema",
        return_value=(
            bodo.types.DataFrameType(
                data=(
                    bodo.types.datetime_date_array_type,
                    types.Array(bodo.types.datetime64ns, 1, "C"),
                    bodo.types.DatetimeArrayType("UTC"),
                ),
                columns=("date_col", "tz_naive_col", "tz_aware_col"),
            ),
            {"date_col", "tz_naive_col", "tz_aware_col"},
            [],
            [],
            pa.schema(
                [
                    pa.field("DATE_COL", pa.date32(), nullable=True),
                    pa.field("TZ_NAIVE_COL", pa.timestamp("ms"), nullable=True),
                    pa.field(
                        "TZ_AWARE_COL", pa.timestamp("us", tz="UTC"), nullable=True
                    ),
                ]
            ),
            None,
            None,
        ),
    )

    def impl(query, conn):
        df = pd.read_sql(query, conn)
        return df

    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema)
    # need to sort the output to make sure pandas and Bodo get the same rows
    query = "SELECT * FROM TIMESTAMP_FILTER_TEST ORDER BY DATE_COL"
    with pytest.raises(
        RuntimeError,
        match=re.escape("Invalid Downcast from timestamp[ns] to timestamp[ms]"),
    ):
        bodo.jit(impl)(query, conn)


def test_snowflake_runtime_downcasting_decimal(mocker: "MockerFixture"):
    """
    Check that Bodo normally throws an error when downcasting
    decimal, but will downcast to double with a flag
    """
    mocker.patch(
        "bodo.io.snowflake.get_schema",
        return_value=(
            bodo.types.DataFrameType(
                data=(
                    types.Array(bodo.types.float64, 1, "C"),
                    types.Array(bodo.types.float64, 1, "C"),
                ),
                columns=("h", "i"),
            ),
            {"h", "i"},
            [],
            [],
            pa.schema(
                [
                    pa.field("H", pa.float64(), nullable=True),
                    pa.field("I", pa.float64(), nullable=True),
                ]
            ),
            None,
            None,
        ),
    )

    def impl(query, conn, downcast):
        df = pd.read_sql(query, conn, _bodo_downcast_decimal_to_double=downcast)  # type: ignore
        return df

    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema)
    # need to sort the output to make sure pandas and Bodo get the same rows
    query = "SELECT H, I FROM DECIMAL_TEST"
    with pytest.raises(RuntimeError) as exc_info:
        bodo.jit(impl)(query, conn, False)
    assert exc_info.value.args[0] in [
        "Invalid Downcast from decimal128(38, 18) to double",
        "See other ranks for runtime errors",
    ]

    check_func(
        impl, (query, conn, True), py_output=pd.read_sql(query, conn), check_dtype=False
    )


def test_read_string_array_col(memory_leak_check):
    """
    Basic test of reading an array of strings column from Snowflake
    Pandas / PyArrow can't seem to compare output with pd.NA or np.nan
    so this test currently doesn't include nulls inside of arrays

    TODO: Try nulls in array post PyArrow or Python upgrade
    """

    def impl(query, conn):
        df = pd.read_sql(query, conn)
        return df

    conn = get_snowflake_connection_string("TEST_DB", "PUBLIC")
    query = r"""
    SELECT * FROM (
        select null as A
        union all
        select ARRAY_CONSTRUCT('why', 'does', 'snowflake', 'use') as A
        union all

        select ARRAY_CONSTRUCT($$
        test multiline \t
        string with junk
        $$) as A

        union all
        select ARRAY_CONSTRUCT('\041', '\x21', '\u26c4', '\z', '\b', '\f', '/') as A
        union all
        select ARRAY_CONSTRUCT('test \0 zero') as A
        union all
        select ARRAY_CONSTRUCT('\'', '"', '\"', '\t\n', '\\') as A
        union all
        select ARRAY_CONSTRUCT('true', '10', '2023-10-20', 'hello') as A
    )
    ORDER BY A
    """
    py_output = pd.DataFrame(
        {
            "a": [
                ["\n        test multiline \\t\n        string with junk\n        "],
                ["\041", "\x21", "\u26c4", "z", "\b", "\f", "/"],
                ["'", '"', '"', "\t\n", "\\"],
                ["test \0 zero"],
                ["true", "10", "2023-10-20", "hello"],
                ["why", "does", "snowflake", "use"],
                None,
            ]
        }
    )

    check_func(impl, (query, conn), py_output=py_output)


def test_read_numeric_array_col(memory_leak_check):
    """
    Basic test of reading an array of floats column from Snowflake
    """

    def impl(query, conn):
        df = pd.read_sql(query, conn)
        return df

    conn = get_snowflake_connection_string("TEST_DB", "PUBLIC")
    query = """
    SELECT * FROM (
        select null as A
        union all
        select ARRAY_CONSTRUCT(10, 10.0, null) as A
        union all
        select ARRAY_CONSTRUCT(null, 12.4, -0.57, '-inf'::float, 'inf'::float, 'nan'::float) as A
        union all
        select ARRAY_CONSTRUCT(12345678901234567890) as A
        union all
        select ARRAY_CONSTRUCT(-1235, 0.01234567890123456789) as A
    )
    ORDER BY A
    """
    py_output = pd.DataFrame(
        {
            "a": [
                [np.nan, 12.4, -0.57, -np.inf, np.inf, np.nan],
                [-1235.0, 0.01234567890123456789],
                [10.0, 10.0, np.nan],
                [12345678901234567890.0],
                None,
            ]
        }
    )

    check_func(impl, (query, conn), py_output=py_output)


def test_read_map_col(memory_leak_check):
    """
    Basic test of reading an map of floats column from Snowflake
    """

    def impl(query, conn):
        df = pd.read_sql(query, conn)
        return df

    conn = get_snowflake_connection_string("TEST_DB", "PUBLIC")
    query = """
    SELECT * FROM (
        select null as A
        union all
        select OBJECT_CONSTRUCT_KEEP_NULL('int', 10, 'whole_dec', 10.0, 'null', null) as A
        union all
        select OBJECT_CONSTRUCT_KEEP_NULL('null2', null, 'float', 12.4, 'neg_float', -0.57) as A
        union all
        select OBJECT_CONSTRUCT_KEEP_NULL('\\u2912', '-inf'::float, 'inf\\"ity', 'inf'::float, '/\\\\/\\\\', 'nan'::float) as A
        union all
        select OBJECT_CONSTRUCT_KEEP_NULL('int20', 12345678901234567890, 'null3', null) as A
        union all
        select OBJECT_CONSTRUCT_KEEP_NULL('neg_int', -1235, 'dec', 0.01234567890123456789) as A
    )
    ORDER BY A
    """
    py_output = pd.DataFrame(
        {
            "a": pd.Series(
                [
                    {"int20": 12345678901234567890.0, "null3": None},
                    {"int": 10.0, "null": None, "whole_dec": 10.0},
                    {"float": 12.4, "neg_float": -0.57, "null2": None},
                    {"dec": 0.01234567890123456789, "neg_int": -1235.0},
                    {
                        "/\\/\\": np.nan,
                        'inf"ity': np.inf,
                        "\u2912": -np.inf,
                    },
                    None,
                ],
                dtype=pd.ArrowDtype(pa.map_(pa.large_string(), pa.float64())),
            )
        }
    )

    check_func(
        impl,
        (query, conn),
        py_output=py_output,
        use_map_arrays=True,
    )


def test_read_struct_col(memory_leak_check):
    """
    Basic test of reading an struct column from Snowflake
    """

    def impl(query, conn):
        df = pd.read_sql(query, conn)
        return df

    conn = get_snowflake_connection_string("TEST_DB", "PUBLIC")
    query = """
    SELECT * FROM (
        select 1 as idx, null as A
        union all
        select 2 as idx, OBJECT_CONSTRUCT_KEEP_NULL(
            'a', null, 'b', null, 'c', null, 'd', null, 'e', null
        ) as A
        union all
        select 3 as idx, OBJECT_CONSTRUCT_KEEP_NULL(
            'a', null, 'b', 10, 'c', null, 'd', true, 'e', '2023-11-01'::date
        ) as A
        union all
        select 4 as idx, OBJECT_CONSTRUCT_KEEP_NULL(
            'a', 'test', 'b', -0.853, 'c', null, 'd', false, 'e', '1980-10-01'::date
        ) as A
        union all
        select 5 as idx, OBJECT_CONSTRUCT_KEEP_NULL(
            'a', 'once', 'b', 'nan'::float, 'c', null, 'd', null, 'e', null
        ) as A
        union all
        select 6 as idx, OBJECT_CONSTRUCT_KEEP_NULL(
            'a', 'none', 'b', 1635::float, 'c', null, 'd', null, 'e', '1970-01-01'::date
        ) as A
    )
    ORDER BY idx
    """
    py_output = pd.DataFrame(
        {
            "idx": [1, 2, 3, 4, 5, 6],
            "a": pd.array(
                [
                    None,
                    {"a": None, "b": None, "c": None, "d": None, "e": None},
                    {
                        "a": None,
                        "b": 10.0,
                        "c": None,
                        "d": True,
                        "e": datetime.date(2023, 11, 1),
                    },
                    {
                        "a": "test",
                        "b": -0.853,
                        "c": None,
                        "d": False,
                        "e": datetime.date(1980, 10, 1),
                    },
                    {"a": "once", "b": np.nan, "c": None, "d": None, "e": None},
                    {
                        "a": "none",
                        "b": 1635.0,
                        "c": None,
                        "d": None,
                        "e": datetime.date(1970, 1, 1),
                    },
                ],
                dtype=pd.ArrowDtype(
                    pa.struct(
                        [
                            pa.field("a", pa.large_string()),
                            pa.field("b", pa.float64()),
                            pa.field("c", pa.null()),
                            pa.field("d", pa.bool_()),
                            pa.field("e", pa.date32()),
                        ]
                    )
                ),
            ),
        }
    )

    check_func(
        impl,
        (query, conn),
        py_output=py_output,
    )


def test_read_variant_col(memory_leak_check):
    """
    Basic test of reading a variant column from Snowflake
    """

    def impl(query, conn):
        df = pd.read_sql(query, conn)
        return df

    conn = get_snowflake_connection_string("TEST_DB", "PUBLIC")
    query = """
    SELECT i, try_parse_json(s) as v FROM (
        select 1 as i, '{"A": []}' as s
        union all
        select 2 as i, '{"B": ["C"]}' as s
        union all
        select 3 as i, '{}' as s
        union all
        select 4 as i, '{"D": null, "E": ["F", "G"]}' as s
        union all
        select 5 as i, '{"H": [null]}' as s
    )
    ORDER BY i
    """
    py_output = pd.DataFrame(
        {
            "i": [1, 2, 3, 4, 5],
            "v": pd.array(
                [
                    {"A": []},
                    {"B": ["C"]},
                    {},
                    {"D": None, "E": ["F", "G"]},
                    {"H": [None]},
                ],
                dtype=pd.ArrowDtype(
                    pa.map_(pa.string(), pa.large_list(pa.large_string()))
                ),
            ),
        }
    )

    check_func(impl, (query, conn), py_output=py_output, check_dtype=False)


def test_read_null_variant_col_correctness(memory_leak_check):
    """
    Test reading a variant column from Snowflake that is all-null.
    """

    def impl(query, conn):
        df = pd.read_sql(query, conn)
        return df

    conn = get_snowflake_connection_string("TEST_DB", "PUBLIC")
    query = """
    SELECT to_variant(null) as V
    FROM table(generator(rowcount=>500))
    """
    py_output = pd.DataFrame(
        {"v": pd.array([None] * 500, dtype=pd.ArrowDtype(pa.string()))}
    )

    check_func(impl, (query, conn), py_output=py_output, check_dtype=False)


@pytest_mark_one_rank
def test_read_null_variant_col_warning(memory_leak_check):
    """
    Verify that the example from test_read_null_variant_col causes a warning.
    """

    @bodo.jit
    def impl(query, conn):
        df = pd.read_sql(query, conn)
        return df

    conn = get_snowflake_connection_string("TEST_DB", "PUBLIC")
    query = """
    SELECT to_variant(null) as V
    FROM table(generator(rowcount=>500))
    """
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        with pytest.warns(
            BodoWarning,
            match=r"The column V is typed as a null array since the source is a variant column with no non-null entries.",
        ):
            impl(query, conn)


def test_read_nested_in_array_col(memory_leak_check):
    """
    Basic test to read nested semi-structured data in Array Columns
    """

    def impl(query, conn):
        df = pd.read_sql(query, conn)
        return df

    conn = get_snowflake_connection_string("TEST_DB", "PUBLIC")

    py_output = pd.DataFrame(
        {
            "a": pd.Series(
                [
                    [[None], [12.4, -0.57]],
                    [[10.0, 10.0], None],
                    None,
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.large_list(pa.float64()))),
            ),
        }
    )
    queryA = "SELECT A FROM NESTED_ARRAY_TEST ORDER BY A"
    check_func(impl, (queryA, conn), py_output=py_output)

    py_output = pd.DataFrame(
        {
            "b": pd.Series(
                [
                    [
                        {
                            "a": datetime.date(2023, 11, 12),
                            "b": datetime.date(1980, 1, 5),
                            "c": None,
                        },
                        {"ten": datetime.date(2023, 11, 11), "ton": None},
                    ],
                    [None, {"m": datetime.date(2023, 11, 11), "mm": None}],
                    None,
                ],
                dtype=pd.ArrowDtype(
                    pa.large_list(pa.map_(pa.large_string(), pa.date32()))
                ),
            ),
        }
    )
    queryB = "SELECT B FROM NESTED_ARRAY_TEST ORDER BY A"
    check_func(impl, (queryB, conn), py_output=py_output, use_map_arrays=True)


def test_read_nested_struct_in_array_col(memory_leak_check):
    def impl(query, conn):
        df = pd.read_sql(query, conn)
        return df

    py_output = pd.DataFrame(
        {
            "c": pd.array(
                [
                    [
                        {"name": "dos", "stat": None, "cnt": None},
                        {"name": "tres", "stat": False, "cnt": -2},
                    ],
                    [None, {"name": "uno", "stat": False, "cnt": None}],
                    [],
                ],
                dtype=pd.ArrowDtype(
                    pa.large_list(
                        pa.struct(
                            [
                                pa.field("stat", pa.bool_()),
                                pa.field("name", pa.string()),
                                pa.field("cnt", pa.int64()),
                            ]
                        )
                    )
                ),
            )
        }
    )
    queryC = "SELECT C FROM NESTED_ARRAY_TEST ORDER BY A"
    conn = get_snowflake_connection_string("TEST_DB", "PUBLIC")
    check_func(
        impl,
        (queryC, conn),
        py_output=py_output,
        convert_columns_to_pandas=False,
        # Both check_dtype=False and convert_columns_to_pandas=False are necessary
        # because Snowflake can output fields in struct in any
        # order, and structs with different fields in different order are different
        # TODO: Make this more robust by either:
        # 1. Sorting the fields in the struct
        # 2. Using a different comparison method
        # 3. Casting the struct to the expected format
        check_dtype=False,
    )


def test_read_nested_in_map_col(memory_leak_check):
    """
    Basic test to read nested semi-structured data in Map Columns
    """

    def impl(query, conn):
        df = pd.read_sql(query, conn)
        return df

    conn = get_snowflake_connection_string("TEST_DB", "PUBLIC")

    py_output = pd.DataFrame(
        {
            "a": pd.Series(
                [
                    {},
                    {"bodo": [10.0, 10.0], "databricks": None},
                    {"a": [np.nan], "b": [12.4, -0.57]},
                ],
                dtype=pd.ArrowDtype(
                    pa.map_(pa.large_string(), pa.large_list(pa.float64()))
                ),
            ),
        }
    )
    queryA = "SELECT A FROM NESTED_MAP_TEST ORDER BY A"
    check_func(impl, (queryA, conn), py_output=py_output, use_map_arrays=True)

    py_output = pd.DataFrame(
        {
            "b": pd.Series(
                [
                    None,
                    {"bodo": {"m": datetime.date(2023, 11, 11), "mm": None}},
                    {
                        "bodo": {
                            "a": datetime.date(2023, 11, 12),
                            "b": datetime.date(1980, 1, 5),
                            "c": None,
                        },
                        "google": {"ten": datetime.date(2023, 11, 11), "ton": None},
                    },
                ],
                dtype=pd.ArrowDtype(
                    pa.map_(pa.large_string(), pa.map_(pa.large_string(), pa.date32()))
                ),
            ),
        }
    )
    queryB = "SELECT B FROM NESTED_MAP_TEST ORDER BY A"
    check_func(impl, (queryB, conn), py_output=py_output, use_map_arrays=True)

    py_output = pd.DataFrame(
        {
            "c": pd.Series(
                [
                    {},
                    {
                        "bodo": None,
                        "snowflake": {"stat": False, "name": "uno", "cnt": None},
                    },
                    {
                        "hive": {"stat": False, "name": "tres", "cnt": -2},
                        "teradata": {"stat": None, "name": "dos", "cnt": None},
                    },
                ],
                dtype=pd.ArrowDtype(
                    pa.map_(
                        pa.string(),
                        pa.struct(
                            [
                                pa.field("name", pa.string()),
                                pa.field("stat", pa.bool_()),
                                pa.field("cnt", pa.int64()),
                            ]
                        ),
                    )
                ),
            ),
        }
    )
    queryC = "SELECT C FROM NESTED_MAP_TEST ORDER BY A"
    check_func(
        impl,
        (queryC, conn),
        py_output=py_output,
        convert_columns_to_pandas=True,
        # Necessary because Snowflake can output fields in struct in any
        # order, and structs with different fields in different order are different
        # TODO: Make this more robust by either:
        # 1. Sorting the fields in the struct
        # 2. Using a different comparison method
        # 3. Casting the struct to the expected format
        check_dtype=False,
    )


def test_read_nested_in_struct_col(memory_leak_check):
    """
    Basic test to read nested semi-structured data in Struct Columns
    TODO: Test the whole column at once, instead of non-map columns only
    """

    def impl(query, conn):
        df = pd.read_sql(query, conn)
        return df

    conn = get_snowflake_connection_string("TEST_DB", "PUBLIC")

    py_output = pd.DataFrame(
        {
            "i": pd.array(
                [
                    {
                        "group": None,
                        "values": None,
                        "created": None,
                    },
                    {
                        "group": None,
                        "values": None,
                        "created": {"creator": "mark", "at": None, "atnew": None},
                    },
                    {
                        "group": "dirt",
                        "values": [-1.15e3, -164.0, 100056.0],
                        "created": {
                            "creator": None,
                            "at": None,
                            "atnew": [2010, 10, 10],
                        },
                    },
                    {
                        "group": "gravel",
                        "values": [10.0, 10.1],
                        "created": {
                            "creator": None,
                            "at": datetime.date(1990, 5, 5),
                            "atnew": None,
                        },
                    },
                ],
                dtype=pd.ArrowDtype(
                    pa.struct(
                        [
                            pa.field("group", pa.large_string()),
                            pa.field("values", pa.large_list(pa.float64())),
                            pa.field(
                                "created",
                                pa.struct(
                                    [
                                        pa.field("creator", pa.large_string()),
                                        pa.field("at", pa.date32()),
                                        pa.field("atnew", pa.large_list(pa.int64())),
                                    ]
                                ),
                            ),
                        ]
                    )
                ),
            ),
        }
    )

    # TODO: Add 'updated' for Sentinal NaNs and Struct Unboxing
    query = "SELECT OBJECT_PICK(INFO, 'group', 'values', 'created') as I FROM NESTED_STRUCT_TEST ORDER BY I"
    check_func(
        impl,
        (query, conn),
        py_output=py_output,
        # Necessary because Snowflake can output fields in struct in any
        # order, and structs with different fields in different order are different
        # TODO: Make this more robust by either:
        # 1. Sorting the fields in the struct
        # 2. Using a different comparison method
        # 3. Casting the struct to the expected format
        check_dtype=False,
    )


def test_snowflake_bodo_read_as_dict_no_table(memory_leak_check):
    """
    Test reading string columns as dictionary-encoded from Snowflake
    """
    import bodo.io.snowflake

    @bodo.jit
    def test_impl0(query, conn):
        return pd.read_sql(query, conn)

    @bodo.jit
    def test_impl1(query, conn):
        return pd.read_sql(query, conn, _bodo_read_as_dict=["l_shipmode"])  # type: ignore

    @bodo.jit
    def test_impl2(query, conn):
        return pd.read_sql(query, conn, _bodo_read_as_dict=["l_shipinstruct"])  # type: ignore

    @bodo.jit
    def test_impl3(query, conn):
        return pd.read_sql(query, conn, _bodo_read_as_dict=["l_comment"])  # type: ignore

    @bodo.jit
    def test_impl4(query, conn):
        return pd.read_sql(
            query, conn, _bodo_read_as_dict=["l_shipmode", "l_shipinstruct"]
        )  # type: ignore

    @bodo.jit
    def test_impl5(query, conn):
        return pd.read_sql(
            query, conn, _bodo_read_as_dict=["l_comment", "l_shipinstruct"]
        )  # type: ignore

    @bodo.jit
    def test_impl6(query, conn):
        return pd.read_sql(query, conn, _bodo_read_as_dict=["l_comment", "l_shipmode"])  # type: ignore

    @bodo.jit
    def test_impl7(query, conn):
        return pd.read_sql(
            query,
            conn,
            _bodo_read_as_dict=["l_shipmode", "l_comment", "l_shipinstruct"],
        )  # type: ignore

    # 'l_suppkey' shouldn't be read as dictionary encoded since it's not a string column

    @bodo.jit
    def test_impl8(query, conn):
        return pd.read_sql(
            query,
            conn,
            _bodo_read_as_dict=[
                "l_shipmode",
                "l_comment",
                "l_shipinstruct",
                "l_suppkey",
            ],
        )  # type: ignore

    @bodo.jit
    def test_impl9(query, conn):
        return pd.read_sql(query, conn, _bodo_read_as_dict=["l_suppkey"])  # type: ignore

    @bodo.jit
    def test_impl10(query, conn):
        return pd.read_sql(query, conn, _bodo_read_as_dict=["l_suppkey", "l_comment"])  # type: ignore

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    # need to sort the output to make sure pandas and Bodo get the same rows
    # l_shipmode, l_shipinstruct should be dictionary encoded by default
    # l_comment could be specified by the user to be dictionary encoded
    # l_suppkey is not of type string and could not be dictionary encoded
    query = "SELECT l_shipmode, l_shipinstruct, l_comment, l_suppkey FROM LINEITEM ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY LIMIT 3000"

    # Avoid small table encoding
    prev_small_table_threshold = bodo.io.snowflake.SF_SMALL_TABLE_THRESHOLD
    bodo.io.snowflake.SF_SMALL_TABLE_THRESHOLD = 0
    try:
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            test_impl0(query, conn)
            check_logger_msg(
                stream,
                "Columns ['l_shipmode', 'l_shipinstruct'] using dictionary encoding",
            )

        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            test_impl1(query, conn)
            check_logger_msg(
                stream,
                "Columns ['l_shipmode', 'l_shipinstruct'] using dictionary encoding",
            )

        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            test_impl2(query, conn)
            check_logger_msg(
                stream,
                "Columns ['l_shipmode', 'l_shipinstruct'] using dictionary encoding",
            )

        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            test_impl3(query, conn)
            check_logger_msg(
                stream,
                "Columns ['l_shipmode', 'l_shipinstruct', 'l_comment'] using dictionary encoding",
            )

        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            test_impl4(query, conn)
            check_logger_msg(
                stream,
                "Columns ['l_shipmode', 'l_shipinstruct'] using dictionary encoding",
            )

        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            test_impl5(query, conn)
            check_logger_msg(
                stream,
                "Columns ['l_shipmode', 'l_shipinstruct', 'l_comment'] using dictionary encoding",
            )

        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            test_impl6(query, conn)
            check_logger_msg(
                stream,
                "Columns ['l_shipmode', 'l_shipinstruct', 'l_comment'] using dictionary encoding",
            )

        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            test_impl7(query, conn)
            check_logger_msg(
                stream,
                "Columns ['l_shipmode', 'l_shipinstruct', 'l_comment'] using dictionary encoding",
            )

        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            if bodo.get_rank() == 0:  # warning is thrown only on rank 0
                with pytest.warns(
                    BodoWarning,
                    match="The following columns are not of datatype string and hence cannot be read with dictionary encoding: {'l_suppkey'}",
                ):
                    test_impl8(query, conn)
            else:
                test_impl8(query, conn)
            # we combine the two tests because otherwise caching would cause problems for logger.stream.
            check_logger_msg(
                stream,
                "Columns ['l_shipmode', 'l_shipinstruct', 'l_comment'] using dictionary encoding",
            )

        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            if bodo.get_rank() == 0:
                with pytest.warns(
                    BodoWarning,
                    match="The following columns are not of datatype string and hence cannot be read with dictionary encoding: {'l_suppkey'}",
                ):
                    test_impl9(query, conn)
            else:
                test_impl9(query, conn)
            check_logger_msg(
                stream,
                "Columns ['l_shipmode', 'l_shipinstruct'] using dictionary encoding",
            )

        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            if bodo.get_rank() == 0:  # warning is thrown only on rank 0
                with pytest.warns(
                    BodoWarning,
                    match="The following columns are not of datatype string and hence cannot be read with dictionary encoding: {'l_suppkey'}",
                ):
                    test_impl10(query, conn)
            else:
                test_impl10(query, conn)
            check_logger_msg(
                stream,
                "Columns ['l_shipmode', 'l_shipinstruct', 'l_comment'] using dictionary encoding",
            )
    finally:
        bodo.io.snowflake.SF_SMALL_TABLE_THRESHOLD = prev_small_table_threshold


@pytest.mark.parametrize("enable_dict_encoding", [True, False])
def test_snowflake_dict_encoding_enabled(enable_dict_encoding, memory_leak_check):
    """
    Test that SF_READ_AUTO_DICT_ENCODE_ENABLED works as expected.
    """
    import bodo.io.snowflake

    # need to sort the output to make sure pandas and Bodo get the same rows
    # l_shipmode, l_shipinstruct should be dictionary encoded based on Snowflake
    # probe query.
    # l_comment could be specified by the user to be dictionary encoded
    # l_suppkey is not of type string and could not be dictionary encoded
    query = "SELECT l_shipmode, l_shipinstruct, l_comment, l_suppkey FROM LINEITEM ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY LIMIT 3000"
    stream = io.StringIO()
    logger = create_string_io_logger(stream)

    @bodo.jit
    def test_impl(query, conn):
        return pd.read_sql(query, conn)

    @bodo.jit
    def test_impl_with_forced_dict_encode(query, conn):
        return pd.read_sql(query, conn, _bodo_read_as_dict=["l_comment"])  # type: ignore

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)

    orig_SF_READ_AUTO_DICT_ENCODE_ENABLED = (
        bodo.io.snowflake.SF_READ_AUTO_DICT_ENCODE_ENABLED
    )
    bodo.io.snowflake.SF_READ_AUTO_DICT_ENCODE_ENABLED = enable_dict_encoding
    # Avoid small table encoding when enable_dict_encoding
    prev_small_table_threshold = bodo.io.snowflake.SF_SMALL_TABLE_THRESHOLD
    bodo.io.snowflake.SF_SMALL_TABLE_THRESHOLD = 0

    try:
        if enable_dict_encoding:
            # check that dictionary encoding works
            with set_logging_stream(logger, 1):
                test_impl(query, conn)
                check_logger_msg(
                    stream,
                    "Columns ['l_shipmode', 'l_shipinstruct'] using dictionary encoding",
                )

            # Verify that _bodo_read_as_dict still works as expected
            with set_logging_stream(logger, 1):
                test_impl_with_forced_dict_encode(query, conn)
                check_logger_msg(
                    stream,
                    "Columns ['l_shipmode', 'l_shipinstruct', 'l_comment'] using dictionary encoding",
                )
        else:
            # check that dictionary encoding is disabled
            with set_logging_stream(logger, 1):
                test_impl(query, conn)
                check_logger_no_msg(
                    stream,
                    "Columns ['l_shipmode', 'l_shipinstruct'] using dictionary encoding",
                )

            # Verify that _bodo_read_as_dict still works as expected
            with set_logging_stream(logger, 1):
                test_impl_with_forced_dict_encode(query, conn)
                check_logger_msg(
                    stream,
                    "Columns ['l_comment'] using dictionary encoding",
                )

    finally:
        bodo.io.snowflake.SF_READ_AUTO_DICT_ENCODE_ENABLED = (
            orig_SF_READ_AUTO_DICT_ENCODE_ENABLED
        )
        bodo.io.snowflake.SF_SMALL_TABLE_THRESHOLD = prev_small_table_threshold


@pytest.mark.skip(
    "TODO(BSE-5302) Ensure dictionary encoding in LINEITEM_100_VIEW lineitem"
)
def test_snowflake_bodo_read_as_dict(memory_leak_check):
    """
    Test Snowflake system sampling for dictionary-encoding detection
    """

    @bodo.jit
    def impl(conn):
        df2 = pd.read_sql("LINEITEM", conn, _bodo_is_table_input=True)
        df1 = df2.loc[
            :,
            [
                "l_shipmode",
            ],
        ].head(100)
        return df1

    @bodo.jit
    def impl2(conn):
        df2 = pd.read_sql(
            "LINEITEM_100_VIEW",
            conn,
            _bodo_is_table_input=True,
        )
        df1 = df2.loc[
            :,
            [
                "l_shipmode",
            ],
        ].head(100)
        return df1

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1000"
    conn = get_snowflake_connection_string(db, schema)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)

    with set_logging_stream(logger, 2):
        impl(conn)
        check_logger_msg(
            stream, "Using Snowflake system sampling for dictionary-encoding detection"
        )
        check_logger_msg(stream, "Columns ['l_shipmode'] using dictionary encoding")

    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        impl2(conn)
        check_logger_no_msg(
            stream, "Using Snowflake system sampling for dictionary-encoding detection"
        )
        check_logger_msg(stream, "Columns ['l_shipmode'] using dictionary encoding")


def test_snowflake_nonascii(memory_leak_check):
    def impl(query, conn):
        df = pd.read_sql(query, conn)
        return df

    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema)
    # need to sort the output to make sure pandas and Bodo get the same rows
    query = "SELECT * FROM NONASCII_T1"
    check_func(impl, (query, conn), reset_index=True, sort_output=True)


def test_snowflake_single_column(memory_leak_check):
    """
    Test that loading using a single column from snowflake has a correct result
    that reduces the number of columns that need loading.
    """

    def impl(query, conn):
        df = pd.read_sql(query, conn)
        return df["l_suppkey"]

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    # need to sort the output to make sure pandas and Bodo get the same rows
    query = "SELECT * FROM LINEITEM ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY LIMIT 70"
    # Pandas will load Int64 instead of the Int16 we can get from snowflake.
    check_func(impl, (query, conn), check_dtype=False)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl)(query, conn)
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['l_suppkey']")


def test_snowflake_use_index(memory_leak_check):
    """
    Tests loading using index_col with pd.read_sql from snowflake
    has a correct result and only loads the columns that need loading.
    """

    def impl(query, conn):
        df = pd.read_sql(query, conn, index_col="l_partkey")
        # Returns l_suppkey and the index
        return df["l_suppkey"]

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    # need to sort the output to make sure pandas and Bodo get the same rows
    query = "SELECT * FROM LINEITEM ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY LIMIT 70"
    # Pandas will load Int64 instead of the Int16 we can get from snowflake.
    check_func(impl, (query, conn), check_dtype=False)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl)(query, conn)
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['l_suppkey', 'l_partkey']")


# TODO: Re-add this test once [BE-2758] is resolved
@pytest.mark.skip(reason="Outdated index returned by pandas")
def test_snowflake_use_index_dead_table(memory_leak_check):
    """
    Tests loading using index_col with pd.read_sql from snowflake
    where all columns are dead.
    """

    def impl(query, conn):
        df = pd.read_sql(query, conn, index_col="l_partkey")
        # Returns just the index
        return df.index

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    # need to sort the output to make sure pandas and Bodo get the same rows
    query = "SELECT * FROM LINEITEM ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY LIMIT 70"
    # Pandas will load Int64 instead of the Int16 we can get from snowflake.
    check_func(impl, (query, conn), check_dtype=False)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl)(query, conn)
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['l_partkey']")


def test_snowflake_no_index_dead_table(memory_leak_check):
    """
    Tests loading with pd.read_sql from snowflake
    where all columns are dead. This should load
    0 columns.
    """

    def impl(query, conn):
        df = pd.read_sql(query, conn)
        # Returns just the index
        return df.index

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    # need to sort the output to make sure pandas and Bodo get the same rows
    query = "SELECT * FROM LINEITEM ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY LIMIT 70"
    # Pandas will load Int64 instead of the Int16 we can get from snowflake.
    check_func(impl, (query, conn), check_dtype=False)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl)(query, conn)
        # Check the columns were pruned. l_orderkey is determined
        # by testing and we just need to confirm it loads a single column.
        check_logger_msg(stream, "Columns loaded []")


def test_snowflake_use_index_dead_index(memory_leak_check):
    """
    Tests loading using index_col with pd.read_sql from snowflake
    where the index is dead.
    """

    def impl(query, conn):
        df = pd.read_sql(query, conn, index_col="l_partkey")
        # Returns just l_suppkey array
        return df["l_suppkey"].values

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    # need to sort the output to make sure pandas and Bodo get the same rows
    query = "SELECT * FROM LINEITEM ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY LIMIT 70"
    # Pandas will load Int64 instead of the Int16 we can get from snowflake.
    check_func(impl, (query, conn), check_dtype=False)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl)(query, conn)
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['l_suppkey']")


def test_snowflake_groupby(memory_leak_check):
    """
    Test that using a sql function without an alias doesn't cause issues with
    dead column elimination.
    """

    def impl(query, conn):
        df = pd.read_sql(query, conn)
        # TODO: Pandas loads count(*) as COUNT(*) but we can't detect this difference
        # and load it as count(*)
        df.columns = [x.lower() for x in df.columns]  # type: ignore
        return df

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    # need to sort the output to make sure pandas and Bodo get the same rows
    query = 'SELECT L_ORDERKEY, count(*), min(L_PARTKEY) as min_key, max("L_ORDERKEY") FROM LINEITEM GROUP BY L_ORDERKEY ORDER BY L_ORDERKEY LIMIT 70'
    # Pandas will load Int64 instead of the Int16 we can get from snowflake.
    check_func(impl, (query, conn), check_dtype=False)


def test_snowflake_filter_pushdown(memory_leak_check):
    """
    Test that filter pushdown works properly with a variety of data types.
    """

    def impl_integer(query, conn, int_val):
        df = pd.read_sql(query, conn)
        df = df[(df["l_orderkey"] > 10) & (int_val >= df["l_linenumber"])]
        return df["l_suppkey"]

    def impl_string(query, conn, str_val):
        df = pd.read_sql(query, conn)
        df = df[(df["l_linestatus"] != str_val) | (df["l_shipmode"] == "FOB")]
        return df["l_suppkey"]

    def impl_date(query, conn, date_val):
        df = pd.read_sql(query, conn)
        df = df[date_val > df["l_shipdate"]]
        return df["l_suppkey"]

    def impl_mixed(query, conn, int_val, str_val, date_val):
        """
        Test a query with mixed parameter types.
        """
        df = pd.read_sql(query, conn)
        df = df[
            ((df["l_linenumber"] <= int_val) | (date_val > df["l_shipdate"]))
            | (df["l_linestatus"] != str_val)
        ]
        return df["l_suppkey"]

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    # need to sort the output to make sure pandas and Bodo get the same rows
    query = "SELECT * FROM LINEITEM ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY LIMIT 70"

    # Pandas will load Int64 instead of the Int16 we can get from snowflake.
    # Reset index because Pandas applies the filter later

    int_val = 2
    check_func(
        impl_integer, (query, conn, int_val), check_dtype=False, reset_index=True
    )
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl_integer)(query, conn, int_val)
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['l_suppkey']")
        # Check for filter pushdown
        check_logger_msg(stream, "Filter pushdown successfully performed")

    str_val = "O"
    check_func(impl_string, (query, conn, str_val), check_dtype=False, reset_index=True)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl_string)(query, conn, str_val)
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['l_suppkey']")
        # Check for filter pushdown
        check_logger_msg(stream, "Filter pushdown successfully performed")

    date_val = datetime.date(1996, 4, 12)
    check_func(impl_date, (query, conn, date_val), check_dtype=False, reset_index=True)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl_date)(query, conn, date_val)
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['l_suppkey']")
        # Check for filter pushdown
        check_logger_msg(stream, "Filter pushdown successfully performed")

    check_func(
        impl_mixed,
        (query, conn, int_val, str_val, date_val),
        check_dtype=False,
        reset_index=True,
    )
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl_mixed)(query, conn, int_val, str_val, date_val)
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['l_suppkey']")
        # Check for filter pushdown
        check_logger_msg(stream, "Filter pushdown successfully performed")


def test_ts_col_date_scalar_filter_pushdown(memory_leak_check):
    """
    Test filter pushdown when reading from a timestamp column and using a date
    filter.

    The table used for this was created directly in snowflake and has the schema

    Table: TIMESTAMP_FILTER_TEST
    Columns:
        date_col DATE
        tz_naive_col TIMESTAMP_NTZ
        tz_aware_col TIMESTAMP_TZ
        ltz_aware_col TIMESTAMP_TZ
    """
    # TODO(BSE-XXXX): Add support for TimestampTZ in this test and read tz_aware_col
    comm = MPI.COMM_WORLD

    def impl_tz_naive(query, conn, date_val):
        df = pd.read_sql(query, conn)
        df = df[df["tz_naive_col"] > date_val]
        return df["date_col"]

    def impl_tz_aware(query, conn, date_val):
        df = pd.read_sql(query, conn)
        df = df[df["ltz_aware_col"] > date_val]
        return df["date_col"]

    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema)
    date_val = datetime.date(2022, 4, 7)
    query = "select * from TIMESTAMP_FILTER_TEST"
    # Pandas doesn't support the date + timestamp comparison so we must load directly from snowflake
    py_output1 = None
    py_output2 = None
    if bodo.get_rank() == 0:
        py_output1 = pd.read_sql(
            "select date_col from TIMESTAMP_FILTER_TEST where tz_naive_col > date '2022-04-07'",
            conn,
        )["date_col"]
        py_output2 = pd.read_sql(
            "select date_col from TIMESTAMP_FILTER_TEST where ltz_aware_col > date '2022-04-07'",
            conn,
        )["date_col"]
    py_output1, py_output2 = comm.bcast((py_output1, py_output2))
    check_func(
        impl_tz_naive,
        (query, conn, date_val),
        sort_output=True,
        reset_index=True,
        py_output=py_output1,
    )
    check_func(
        impl_tz_aware,
        (query, conn, date_val),
        sort_output=True,
        reset_index=True,
        py_output=py_output2,
    )
    # Test for filter pushdown.
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl_tz_naive)(query, conn, date_val)
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['date_col']")
        # Check for filter pushdown
        check_logger_msg(stream, "Filter pushdown successfully performed")

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl_tz_aware)(query, conn, date_val)
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['date_col']")
        # Check for filter pushdown
        check_logger_msg(stream, "Filter pushdown successfully performed")


@pytest.mark.tz_aware
def test_tz_aware_filter_pushdown(memory_leak_check):
    """
    Test that filter pushdown works with a tz aware timestamp +
    a tz aware column.

    The table used for this was created directly in snowflake and has the schema

    Table: TIMESTAMP_FILTER_TEST
    Columns:
        date_col DATE
        tz_naive_col TIMESTAMP_NTZ
        tz_aware_col TIMESTAMP_TZ
        ltz_aware_col TIMESTAMP_TZ
    """
    # TODO(BSE-XXXX): Add support for TimestampTZ in this test and read tz_aware_col
    comm = MPI.COMM_WORLD

    def impl(query, conn, ts_value):
        df = pd.read_sql(query, conn)
        df = df[df["ltz_aware_col"] > ts_value]
        return df["date_col"]

    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema)
    query = "select * from TIMESTAMP_FILTER_TEST"
    # TZ_AWARE is in LA time
    ts_value = pd.Timestamp("2022-04-06 19:00:00", tz="America/Los_Angeles")
    py_output = None
    if bodo.get_rank() == 0:
        py_output = pd.read_sql(
            "select date_col from TIMESTAMP_FILTER_TEST where ltz_aware_col > '2022-04-06 19:00:00'::TIMESTAMP_LTZ",
            conn,
        )["date_col"]
    py_output = comm.bcast(py_output)
    check_func(
        impl,
        (query, conn, ts_value),
        sort_output=True,
        reset_index=True,
        py_output=py_output,
    )
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl)(query, conn, ts_value)
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['date_col']")
        # Check for filter pushdown
        check_logger_msg(stream, "Filter pushdown successfully performed")


def test_date_col_ts_scalar_filter_pushdown(memory_leak_check):
    """
    Test that filter pushdown works with a date column and a
    timezone scalar, both with and without a timezone.

    The table used for this was created directly in snowflake and has the schema

    Table: TIMESTAMP_FILTER_TEST
    Columns:
        date_col DATE
        tz_naive_col TIMESTAMP_NTZ
        tz_aware_col TIMESTAMP_TZ
        ltz_aware_col TIMESTAMP_LTZ
    """
    comm = MPI.COMM_WORLD

    def impl(query, conn, ts_value):
        df = pd.read_sql(query, conn)
        df = df[df["date_col"] > ts_value]
        return df["tz_naive_col"]

    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema)
    query = "select * from TIMESTAMP_FILTER_TEST"
    # TZ_AWARE is in LA time
    ts_value = pd.Timestamp("2022-04-06 19:00:00", tz="America/Los_Angeles")
    py_output = None
    if bodo.get_rank() == 0:
        py_output = pd.read_sql(
            "select tz_naive_col from TIMESTAMP_FILTER_TEST where date_col > '2022-04-06 19:00:00'::TIMESTAMP_LTZ",
            conn,
        )["tz_naive_col"]
    py_output = comm.bcast(py_output)
    check_func(
        impl,
        (query, conn, ts_value),
        sort_output=True,
        reset_index=True,
        py_output=py_output,
    )
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl)(query, conn, ts_value)
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['tz_naive_col']")
        # Check for filter pushdown
        check_logger_msg(stream, "Filter pushdown successfully performed")

    # Test with tz-naive
    ts_value = pd.Timestamp("2022-04-06 19:00:00")
    py_output = None
    if bodo.get_rank() == 0:
        py_output = pd.read_sql(
            "select tz_naive_col from TIMESTAMP_FILTER_TEST where date_col > '2022-04-06 19:00:00'::TIMESTAMP_NTZ",
            conn,
        )["tz_naive_col"]
    py_output = comm.bcast(py_output)
    check_func(
        impl,
        (query, conn, ts_value),
        sort_output=True,
        reset_index=True,
        py_output=py_output,
    )
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl)(query, conn, ts_value)
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['tz_naive_col']")
        # Check for filter pushdown
        check_logger_msg(stream, "Filter pushdown successfully performed")


def test_snowflake_na_pushdown(memory_leak_check):
    """
    Test that filter pushdown with isna/notna/isnull/notnull works in snowflake.
    """

    def impl_or_isna(query, conn):
        T = pd.read_sql(query, conn, _bodo_read_as_table=True)
        T = T[
            (pd.Series(bodo.hiframes.table.get_table_data(T, 0)) > 10)
            | (pd.Series(bodo.hiframes.table.get_table_data(T, 3)).isna())
        ]
        return pd.Series(bodo.hiframes.table.get_table_data(T, 2))

    def impl_and_notna(query, conn):
        T = pd.read_sql(query, conn, _bodo_read_as_table=True)
        T = T[
            (pd.Series(bodo.hiframes.table.get_table_data(T, 0)) > 10)
            & (pd.Series(bodo.hiframes.table.get_table_data(T, 3)).notna())
        ]
        return pd.Series(bodo.hiframes.table.get_table_data(T, 2))

    def impl_or_isnull(query, conn):
        T = pd.read_sql(query, conn, _bodo_read_as_table=True)
        T = T[
            (pd.Series(bodo.hiframes.table.get_table_data(T, 0)) > 10)
            | (pd.Series(bodo.hiframes.table.get_table_data(T, 3)).isnull())
        ]
        return pd.Series(bodo.hiframes.table.get_table_data(T, 2))

    def impl_and_notnull(query, conn):
        T = pd.read_sql(query, conn, _bodo_read_as_table=True)
        T = T[
            (pd.Series(bodo.hiframes.table.get_table_data(T, 0)) > 10)
            & (pd.Series(bodo.hiframes.table.get_table_data(T, 3)).notnull())
        ]
        return pd.Series(bodo.hiframes.table.get_table_data(T, 2))

    def impl_just_nona(query, conn):
        T = pd.read_sql(query, conn, _bodo_read_as_table=True)
        T = T[(pd.Series(bodo.hiframes.table.get_table_data(T, 3)).notna())]
        return pd.Series(bodo.hiframes.table.get_table_data(T, 2))

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    # need to sort the output to make sure pandas and Bodo get the same rows
    query = "SELECT * FROM LINEITEM ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY LIMIT 70"

    na_output = pd.Series(
        [
            7744,
            4117,
            6666,
            7721,
            8320,
            441,
            3919,
            5532,
            8855,
            9983,
            871,
            1923,
            4577,
            2951,
            3266,
            7684,
            4940,
            8433,
            4457,
            9768,
            5405,
            5133,
            1807,
            874,
            9821,
            3093,
            9530,
            5350,
            6878,
            4137,
            5952,
            3889,
            4705,
            8830,
            7630,
            3490,
            5198,
            9143,
            8126,
            7515,
            6118,
            824,
            9569,
            7484,
            5267,
        ]
    )
    check_func(
        impl_or_isna,
        (query, conn),
        py_output=na_output,
        check_dtype=False,
        reset_index=True,
    )
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl_or_isna)(query, conn)
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['l_suppkey']")
        # Check for filter pushdown
        check_logger_msg(stream, "Filter pushdown successfully performed")

    check_func(
        impl_and_notna,
        (query, conn),
        py_output=na_output,
        check_dtype=False,
        reset_index=True,
    )
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl_and_notna)(query, conn)
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['l_suppkey']")
        # Check for filter pushdown
        check_logger_msg(stream, "Filter pushdown successfully performed")

    check_func(
        impl_or_isnull,
        (query, conn),
        py_output=na_output,
        check_dtype=False,
        reset_index=True,
    )
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl_or_isnull)(query, conn)
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['l_suppkey']")
        # Check for filter pushdown
        check_logger_msg(stream, "Filter pushdown successfully performed")

    check_func(
        impl_and_notnull,
        (query, conn),
        py_output=na_output,
        check_dtype=False,
        reset_index=True,
    )
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl_and_notnull)(query, conn)
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['l_suppkey']")
        # Check for filter pushdown
        check_logger_msg(stream, "Filter pushdown successfully performed")

    only_na_output = pd.Series(
        [
            4633,
            638,
            1534,
            3701,
            7311,
            7706,
            1191,
            1798,
            6540,
            1883,
            9662,
            3474,
            650,
            5560,
            35,
            8571,
            3928,
            2150,
            1759,
            9799,
            7758,
            9440,
            2269,
            3074,
            9607,
            7744,
            4117,
            6666,
            7721,
            8320,
            441,
            3919,
            5532,
            8855,
            9983,
            871,
            1923,
            4577,
            2951,
            3266,
            7684,
            4940,
            8433,
            4457,
            9768,
            5405,
            5133,
            1807,
            874,
            9821,
            3093,
            9530,
            5350,
            6878,
            4137,
            5952,
            3889,
            4705,
            8830,
            7630,
            3490,
            5198,
            9143,
            8126,
            7515,
            6118,
            824,
            9569,
            7484,
            5267,
        ]
    )
    check_func(
        impl_just_nona,
        (query, conn),
        py_output=only_na_output,
        check_dtype=False,
        reset_index=True,
    )
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl_just_nona)(query, conn)
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['l_suppkey']")
        # Check for filter pushdown
        check_logger_msg(stream, "Filter pushdown successfully performed")


def test_snowflake_isin_pushdown(memory_leak_check):
    """
    Test that filter pushdown with isna/notna/isnull/notnull works in snowflake.
    """
    import bodo.io.snowflake

    def impl_isin(query, conn, isin_list):
        df = pd.read_sql(query, conn)
        df = df[df["l_orderkey"].isin(isin_list)]
        return df["l_suppkey"]

    def impl_isin_or(query, conn, isin_list):
        df = pd.read_sql(query, conn)
        df = df[(df["l_shipmode"] == "FOB") | df["l_orderkey"].isin(isin_list)]
        return df["l_suppkey"]

    def impl_isin_and(query, conn, isin_list):
        df = pd.read_sql(query, conn)
        df = df[(df["l_shipmode"] == "FOB") & df["l_orderkey"].isin(isin_list)]
        return df["l_suppkey"]

    isin_list = [32, 35]
    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    # need to sort the output to make sure pandas and Bodo get the same rows
    query = "SELECT * FROM LINEITEM ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY LIMIT 70"

    check_func(
        impl_isin,
        (query, conn, isin_list),
        check_dtype=False,
        reset_index=True,
        use_dict_encoded_strings=False,
    )
    # TODO: BE-3404: Support `pandas.Series.isin` for dictionary-encoded arrays
    prev_criterion = bodo.io.snowflake.SF_READ_DICT_ENCODE_CRITERION
    prev_small_table_threshold = bodo.io.snowflake.SF_SMALL_TABLE_THRESHOLD
    bodo.io.snowflake.SF_READ_DICT_ENCODE_CRITERION = -1
    bodo.io.snowflake.SF_SMALL_TABLE_THRESHOLD = 0
    try:
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            bodo.jit(impl_isin)(query, conn, isin_list)
            # Check the columns were pruned
            check_logger_msg(stream, "Columns loaded ['l_suppkey']")
            # Check for filter pushdown
            check_logger_msg(stream, "Filter pushdown successfully performed")

        check_func(
            impl_isin_or, (query, conn, isin_list), check_dtype=False, reset_index=True
        )
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            bodo.jit(impl_isin_or)(query, conn, isin_list)
            # Check the columns were pruned
            check_logger_msg(stream, "Columns loaded ['l_suppkey']")
            # Check for filter pushdown
            check_logger_msg(stream, "Filter pushdown successfully performed")

        check_func(
            impl_isin_and, (query, conn, isin_list), check_dtype=False, reset_index=True
        )
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            bodo.jit(impl_isin_and)(query, conn, isin_list)
            # Check the columns were pruned
            check_logger_msg(stream, "Columns loaded ['l_suppkey']")
            # Check for filter pushdown
            check_logger_msg(stream, "Filter pushdown successfully performed")
    finally:
        bodo.io.snowflake.SF_READ_DICT_ENCODE_CRITERION = prev_criterion
        bodo.io.snowflake.SF_SMALL_TABLE_THRESHOLD = prev_small_table_threshold


def test_snowflake_startswith_endswith_pushdown(memory_leak_check):
    """
    Test that filter pushdown with startswith/endswith works in snowflake.
    """

    def impl_startswith(query, conn, starts_val):
        df = pd.read_sql(query, conn)
        df = df[df["l_shipmode"].str.startswith(starts_val)]
        return df["l_suppkey"]

    def impl_startswith_or(query, conn, starts_val):
        df = pd.read_sql(query, conn)
        df = df[df["l_shipmode"].str.startswith(starts_val) | (df["l_orderkey"] == 32)]
        return df["l_suppkey"]

    def impl_startswith_and(query, conn, starts_val):
        df = pd.read_sql(query, conn)
        df = df[df["l_shipmode"].str.startswith(starts_val) & (df["l_orderkey"] == 32)]
        return df["l_suppkey"]

    def impl_endswith(query, conn, ends_val):
        df = pd.read_sql(query, conn)
        df = df[df["l_shipmode"].str.endswith(ends_val)]
        return df["l_suppkey"]

    def impl_endswith_or(query, conn, ends_val):
        df = pd.read_sql(query, conn)
        df = df[df["l_shipmode"].str.endswith(ends_val) | (df["l_orderkey"] == 32)]
        return df["l_suppkey"]

    def impl_endswith_and(query, conn, ends_val):
        df = pd.read_sql(query, conn)
        df = df[df["l_shipmode"].str.endswith(ends_val) & (df["l_orderkey"] == 32)]
        return df["l_suppkey"]

    starts_val = "AIR"
    ends_val = "AIL"
    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    # need to sort the output to make sure pandas and Bodo get the same rows
    query = "SELECT * FROM LINEITEM ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY LIMIT 70"

    for func in (impl_startswith, impl_startswith_or, impl_startswith_and):
        check_func(func, (query, conn, starts_val), check_dtype=False, reset_index=True)
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            bodo.jit(func)(query, conn, starts_val)
            # Check the columns were pruned
            check_logger_msg(stream, "Columns loaded ['l_suppkey']")
            # Check for filter pushdown
            check_logger_msg(stream, "Filter pushdown successfully performed")

    for func in (impl_endswith, impl_endswith_or, impl_endswith_and):
        check_func(func, (query, conn, ends_val), check_dtype=False, reset_index=True)
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            bodo.jit(func)(query, conn, ends_val)
            # Check the columns were pruned
            check_logger_msg(stream, "Columns loaded ['l_suppkey']")
            # Check for filter pushdown
            check_logger_msg(stream, "Filter pushdown successfully performed")


def test_snowflake_json_url(memory_leak_check):
    """
    Check running a snowflake query with a dictionary for connection parameters
    """

    def impl(query, conn):
        df = pd.read_sql(query, conn)
        return df

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    connection_params = {
        "warehouse": "DEMO_WH",
        "session_parameters": json.dumps({"JSON_INDENT": 0}),
        "paramstyle": "pyformat",
        "insecure_mode": True,
    }
    conn = get_snowflake_connection_string(db, schema, connection_params)
    # session_parameters bug exists in sqlalchemy/snowflake connector
    del connection_params["session_parameters"]
    pandas_conn = get_snowflake_connection_string(db, schema, connection_params)
    query = "SELECT * FROM LINEITEM ORDER BY L_ORDERKEY, L_PARTKEY, L_SUPPKEY LIMIT 70"
    check_func(impl, (query, conn), py_output=impl(query, pandas_conn))


@pytest.mark.skip(
    reason="[BSE-2911] This test is reading TimestampTZ columns - we need to specify py_output to compare output"
)
def test_snowflake_timezones(memory_leak_check):
    """
    Tests trying to read Arrow timestamp columns with
    timezones using Bodo + Snowflake succeeds.

    The table used for this was created directly in snowflake and has the schema

    Table: TZ_TEST
    Columns:
        A TIMESTAMP_TZ
        B NUMBER
        C TIMESTAMP_LTZ
    """
    with enable_timestamptz():

        def test_impl1(query, conn_str):
            """
            read_sql that should succeed
            and filters out tz columns.
            """
            df = pd.read_sql(query, conn_str)
            return df.b

        def test_impl2(query, conn_str):
            """
            Read parquet loading a single tz column.
            """
            df = pd.read_sql(query, conn_str)
            return df.a

        def test_impl3(query, conn_str):
            """
            Read parquet loading tz columns.
            """
            df = pd.read_sql(query, conn_str)
            return df

        db = "TEST_DB"
        schema = "PUBLIC"
        conn = get_snowflake_connection_string(db, schema)

        # Note that we can't just write select *, since we normally rely on
        # BodoSQL to convert TIMESTAMP_TZ columns to VARIANT.
        full_query = "select TO_VARIANT(A) as A, B, C from tz_test"
        partial_query = "select B, C from tz_test"
        # Loading just the non-tz columns should suceed.
        check_func(test_impl1, (full_query, conn), check_dtype=False)
        check_func(test_impl1, (partial_query, conn), check_dtype=False)
        check_func(test_impl2, (full_query, conn), check_dtype=False)
        check_func(test_impl3, (full_query, conn), check_dtype=False)
        check_func(test_impl3, (partial_query, conn), check_dtype=False)


def test_snowflake_empty_typing(memory_leak_check):
    """
    Tests support for read_sql when typing a query returns an empty DataFrame.
    """

    def test_impl(query, conn):
        return pd.read_sql(query, conn)

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    query = "SELECT L_ORDERKEY FROM LINEITEM WHERE L_ORDERKEY IN (10, 11)"
    check_func(test_impl, (query, conn))


def test_snowflake_empty_filter(memory_leak_check):
    """
    Tests support for read_sql when a query returns an empty DataFrame via filter pushdown.
    """

    def test_impl(query, conn):
        df = pd.read_sql(query, conn)
        df = df[df["l_orderkey"] == 10]
        return df

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    query = "SELECT L_ORDERKEY FROM LINEITEM"
    check_func(test_impl, (query, conn), check_dtype=False)


def test_snowflake_dead_node(memory_leak_check):
    """
    Tests when read_sql should be eliminated from the code.
    """

    def test_impl(query, conn):
        # This query should be optimized out.
        pd.read_sql(query, conn)
        return 1

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    query = "SELECT L_ORDERKEY FROM LINEITEM"
    check_func(test_impl, (query, conn))

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(test_impl)(query, conn)
        # Check that no Columns loaded message occurred,
        # so the whole node was deleted.
        check_logger_no_msg(stream, "Columns loaded")


def test_snowflake_zero_cols(memory_leak_check):
    """Tests when read_sql's table should load 0 columns."""

    def test_impl(query, conn):
        df = pd.read_sql(query, conn)
        return len(df)

    def test_impl_index(query, conn):
        # Test only loading an index
        df = pd.read_sql(query, conn, index_col="l_orderkey")
        return len(df), df.index.min()

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    # We only load 1 column because Pandas is loading all of the data.
    query = "SELECT L_ORDERKEY FROM LINEITEM"

    check_func(test_impl, (query, conn), only_seq=True)

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(test_impl)(query, conn)
        # Check that no columns were loaded.
        check_logger_msg(stream, "Columns loaded []")

    check_func(test_impl_index, (query, conn))

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit((numba.types.literal(query), numba.types.literal(conn)))(
            test_impl_index
        )
        # Check that we only load the index.
        check_logger_msg(stream, "Columns loaded ['l_orderkey']")


@pytest.mark.slow
def test_read_sql_error_snowflake(memory_leak_check):
    """This test for incorrect credentials and SQL sentence with snowflake"""

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)

    def test_impl_sql_err(conn):
        sql_request = "select * from invalid"
        frame = pd.read_sql(sql_request, conn)
        return frame

    with pytest.raises(RuntimeError, match="Error executing query"):
        bodo.jit(test_impl_sql_err)(conn)

    def test_impl_credentials_err(conn):
        sql_request = "select * from LINEITEM LIMIT 10"
        frame = pd.read_sql(sql_request, conn)
        return frame

    account = "bodopartner.us-east-1"
    warehouse = "DEMO_WH"
    conn = f"snowflake://unknown:wrong@{account}/{db}/{schema}?warehouse={warehouse}"
    with pytest.raises(RuntimeError, match="Error executing query"):
        bodo.jit(test_impl_credentials_err)(conn)


def test_dict_encoded_small_table(memory_leak_check):
    """Tests that reading small table, even with unique values
    enables dictionary encoding.
    """
    import bodo.io.snowflake

    def impl(query, conn_str):
        return pd.read_sql(query, conn_str)

    str_list = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    assert len(str_list) <= bodo.io.snowflake.SF_SMALL_TABLE_THRESHOLD, (
        "test_dict_encoded_small_table requires a small table input"
    )
    new_df = pd.DataFrame({"a": str_list})
    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema)
    with create_snowflake_table(
        new_df, "bodosql_catalog_write_test1", db, schema
    ) as table_name:
        query = f"select * from {table_name}"
        stream = io.StringIO()
        logger = create_string_io_logger(stream)
        with set_logging_stream(logger, 1):
            check_func(impl, (query, conn), py_output=new_df)
            check_logger_msg(stream, "Columns ['a'] using dictionary encoding")


def test_snowflake_filter_pushdown_edgecase(memory_leak_check):
    """
    Test that filter pushdown works for an edge case


    """

    @bodo.jit(inline="never")
    def no_op(df):
        return df

    def impl_should_not_do_filter_pushdown(conn):
        df7 = pd.read_sql(
            "DATES_FILTERPUSHDOWN_TEST_TABLE", conn, _bodo_is_table_input=True
        )  # type: ignore
        df5 = df7.loc[
            :,
            [
                "full_date",
            ],
        ]
        df5 = df5.rename(
            columns={
                "full_date": "FULL_DATE",
            },
            copy=False,
        )
        df16 = df7[
            (pd.notna(bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df7, 1)))
        ]
        # Add a "use" of df5
        no_op(df5)
        return df16

    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema)

    check_func(
        impl_should_not_do_filter_pushdown,
        (conn,),
        check_dtype=False,
        reset_index=True,
        py_output=pd.DataFrame(
            {
                "full_date": [pd.to_datetime("2023-05-05").date()],
                "other_column": [2.0],
            }
        ),
    )

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl_should_not_do_filter_pushdown)(conn)
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['full_date', 'other_column']")
        # Check for filter pushdown
        check_logger_no_msg(stream, "Filter pushdown successfully performed")


@pytest.mark.skip(
    reason="This test should do filter pushdown, but it's currently failing. See: https://bodo.atlassian.net/browse/BSE-336"
)
def test_snowflake_filter_pushdown_edgecase_2(memory_leak_check):
    """
    Test that filter pushdown works for a specific edge case

    Originally, this would throw a compile time error in the filter pushdown code.
    """

    def impl_should_do_filter_pushdown(conn):
        df7 = pd.read_sql(
            "DATES_FILTERPUSHDOWN_TEST_TABLE", conn, _bodo_is_table_input=True
        )  # type: ignore
        df5 = df7.loc[
            :,
            [
                "full_date",
            ],
        ]
        df5 = df5.rename(
            columns={
                "full_date": "FULL_DATE",
            },
            copy=False,
        )
        df16 = df7[
            (pd.notna(bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df7, 1)))
        ]
        return df16

    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema)

    check_func(
        impl_should_do_filter_pushdown,
        (conn,),
        check_dtype=False,
        reset_index=True,
        py_output=pd.DataFrame(
            {
                "full_date": [pd.to_datetime("2023-05-05").date()],
                "other_column": [2.0],
            }
        ),
    )

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 1):
        bodo.jit(impl_should_do_filter_pushdown)(conn)
        # Check the columns were pruned
        check_logger_msg(stream, "Columns loaded ['full_date', 'other_column']")
        # Check for filter pushdown
        check_logger_msg(stream, "Filter pushdown successfully performed")


def test_bodo_read_sql_bodo_orig_table_name_arg(memory_leak_check):
    """
    Test that bodo.read_sql works with the _bodo_orig_table_name argument.
    """
    import bodo.decorators  # isort:skip # noqa
    from bodo.libs.dict_arr_ext import is_dict_encoded

    # Two tables KEATON_TESTING_TABLE_STRING_ALL_UNIQUE, which contains one string
    # column with entirely unique values, and KEATON_TESTING_TABLE_STRING_ALL_DUPLICATE,
    # which contains one string column which is just the same value repeated.
    def impl1(conn):
        df1 = pd.read_sql(
            "SELECT * FROM KEATON_TESTING_TABLE_STRING_ALL_UNIQUE",
            conn,
        )
        # Check that the string columns are NOT dict encoded
        is_dict1_encoded = is_dict_encoded(df1["my_col"])
        return is_dict1_encoded

    def impl2(conn):
        df2 = pd.read_sql(
            "SELECT * FROM KEATON_TESTING_TABLE_STRING_ALL_UNIQUE",
            conn,
            # Note: BodoSQL always provides SCHEMA.TABLENAME as the _bodo_orig_table_name.
            _bodo_orig_table_name='"TEST_DB"."PUBLIC"."KEATON_TESTING_TABLE_STRING_ALL_DUPLICATE"',
            # Note: BodoSQL always provides _bodo_orig_table_indices if it provides
            # _bodo_orig_table_name.
            _bodo_orig_table_indices=(0,),
        )
        # Check that the string columns are dict encoded
        is_dict2_encoded = is_dict_encoded(df2["my_col"])

        return is_dict2_encoded

    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema)

    # Expect the read of the entirely unique table to NOT be dict encoded
    check_func(impl1, (conn,), py_output=False, check_dtype=False, reset_index=True)
    # Expect the read of the entirely duplicate table to be dict encoded
    check_func(impl2, (conn,), py_output=True, check_dtype=False, reset_index=True)


def test_logged_queryid_read(memory_leak_check):
    """Test query id is printed in read step when verbose logging is set to 2"""

    def impl(query, conn):
        df = pd.read_sql(query, conn)
        return df

    db = "SNOWFLAKE_SAMPLE_DATA"
    schema = "TPCH_SF1"
    conn = get_snowflake_connection_string(db, schema)
    query = "SELECT L_ORDERKEY FROM LINEITEM LIMIT 10"

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        bodo.jit(impl)(query, conn)
        check_logger_msg(stream, "Snowflake Query Submission (Read)")


@pytest_mark_one_rank
def test_disable_result_cache_session_param(memory_leak_check):
    """
    Test that our snowflake connection sets USE_CACHED_RESULT = False
    when the BODO_DISABLE_SF_RESULT_CACHE env var is set to 1.
    """
    import bodo.io.snowflake

    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema)
    df = pd.read_sql("show parameters like 'USE_CACHED_RESULT'", conn)
    old_value = df["value"].iloc[0]

    def get_use_cached_result_val():
        connection = bodo.io.snowflake.snowflake_connect(conn)
        cur = connection.cursor()
        cur.execute("show parameters like 'USE_CACHED_RESULT'")
        rows = cur.fetchall()
        result = rows[0][1]
        return result

    try:
        # Sanity check: Default value is the same as old_value
        result = get_use_cached_result_val()
        assert result.lower() == str(old_value).lower(), (
            f"USE_CACHED_RESULT is not the expected default ({str(old_value).lower()})"
        )

        # Now set the user level parameter to True.
        pd.read_sql("alter user set USE_CACHED_RESULT=True", conn)

        # Verify that it's now set to True by default:
        result = get_use_cached_result_val()
        assert result.lower() == "true", "'USE_CACHED_RESULT' is not set to true!"

        # Now verify that setting the BODO_DISABLE_SF_RESULT_CACHE env var sets
        # it to false.
        with temp_env_override({"BODO_DISABLE_SF_RESULT_CACHE": "1"}):
            result = get_use_cached_result_val()
            assert result.lower() == "false", (
                "'USE_CACHED_RESULT' is not set to false by snowflake_connect()"
            )
    finally:
        pd.read_sql(f"alter user set USE_CACHED_RESULT={old_value}", conn)


@pytest_mark_one_rank
def test_snowflake_read_empty_non_nullable_variant(memory_leak_check):
    """Make sure "A null type field may not be non-nullable" of Arrow is not thrown
    when reading an empty non-nullable VARIANT column.
    See https://bodo.atlassian.net/browse/BSE-2918?focusedCommentId=29750
    """

    def impl(query, conn):
        df = pd.read_sql(query, conn)
        return df

    db = "TEST_DB"
    schema = "PUBLIC"
    conn = get_snowflake_connection_string(db, schema)
    # Table created with:
    # create or replace TRANSIENT TABLE TEST_DB.PUBLIC.VARIANT_TABLE3 (
    #  A VARIANT not null
    # )
    query = "SELECT * FROM VARIANT_TABLE3"
    check_func(impl, (query, conn), only_seq=True)
