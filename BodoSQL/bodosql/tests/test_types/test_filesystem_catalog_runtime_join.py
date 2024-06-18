import io

import pandas as pd
import pytest
from mpi4py import MPI

import bodosql
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import (
    check_func,
)

comm = MPI.COMM_WORLD


def test_simple_join(memory_leak_check, iceberg_database):
    """
    Test data and file pruning runtime join filters are generated correctly when reading from filesystem catalog
    """
    table_names = [
        "ADVERSARIAL_SCHEMA_EVOLUTION_TABLE",
        "part_NUMERIC_TABLE_A_bucket_50",
    ]

    def impl(bc, query):
        return bc.sql(query)

    db_schema, warehouse_loc = iceberg_database(table_names)

    # ADVERSARIAL_SCHEMA_EVOLUTION_TABLE.G only contains 1, so we should only read SIMPLE_NUMERIC_TABLE.A where A=1
    # We limit the rows to 10 to force ADVERSARIAL_SCHEMA_EVOLUTION_TABLE to be the build side
    # We use a partitioned table to test file pruning
    query = f'SELECT "{table_names[1]}".A FROM (SELECT * FROM {table_names[0]} LIMIT 10) EVOLVED JOIN "{table_names[1]}" ON EVOLVED.G = "{table_names[1]}".A'

    catalog = bodosql.FileSystemCatalog(warehouse_loc, default_schema=f'"{db_schema}"')
    bc = bodosql.BodoSQLContext(catalog=catalog)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    py_output = pd.DataFrame({"A": [1] * 100})
    with set_logging_stream(logger, 2):
        check_func(
            impl,
            (bc, query),
            py_output=py_output,
            only_1DVar=True,
            sort_output=True,
            reset_index=True,
        )

        check_logger_msg(
            stream,
            "Runtime join filter expression: ((ds.field('{A}') >= 1) & (ds.field('{A}') <= 1))",
        )
        check_logger_msg(stream, "Total number of files is 5. Reading 1 files:")


@pytest.mark.parametrize("join_same_col", [True, False])
def test_multiple_filter_join(memory_leak_check, iceberg_database, join_same_col):
    """
    Test data runtime join filters are generated correctly when reading from filesystem catalog with multiple filters on the same reader
    """
    table_names = [
        "ADVERSARIAL_SCHEMA_EVOLUTION_TABLE",
        "part_NUMERIC_TABLE_A_bucket_50",
        "SIMPLE_NUMERIC_TABLE",
    ]

    def impl(bc, query):
        return bc.sql(query)

    db_schema, warehouse_loc = iceberg_database(table_names)
    second_key = "A" if join_same_col else "G"
    log_msg = (
        "Runtime join filter expression: ((ds.field('{A}') >= 1) & (ds.field('{A}') <= 5) & (ds.field('{A}') >= 2) & (ds.field('{A}') <= 2))"
        if join_same_col
        else "Runtime join filter expression: ((ds.field('{A}') >= 2) & (ds.field('{A}') <= 2) & (ds.field('{G}') >= 1) & (ds.field('{G}') <= 5))"
    )
    query = f'SELECT EVOLVED.A FROM (SELECT * FROM {table_names[0]}) EVOLVED, (SELECT * FROM "{table_names[1]}" LIMIT 10) PARTITIONED, (SELECT * FROM "{table_names[2]}" LIMIT 10) SIMPLE WHERE EVOLVED.A = PARTITIONED.A AND EVOLVED.{second_key} = SIMPLE.A'

    catalog = bodosql.FileSystemCatalog(warehouse_loc, default_schema=f'"{db_schema}"')
    bc = bodosql.BodoSQLContext(catalog=catalog)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    py_output = pd.DataFrame({"A": [2] * 200})
    with set_logging_stream(logger, 2):
        check_func(
            impl,
            (bc, query),
            py_output=py_output,
            only_1DVar=True,
            sort_output=True,
            reset_index=True,
        )
        print(stream.getvalue())

        check_logger_msg(
            stream,
            log_msg,
        )


def test_rtjf_schema_evolved(memory_leak_check, iceberg_database):
    """
    Test data runtime join filters are generated correctly when reading a schema evolved table from filesystem catalog
    """
    table_names = [
        "ADVERSARIAL_SCHEMA_EVOLUTION_TABLE",
        "part_NUMERIC_TABLE_A_bucket_50",
    ]

    def impl(bc, query):
        return bc.sql(query)

    db_schema, warehouse_loc = iceberg_database(table_names)

    # ADVERSARIAL_SCHEMA_EVOLUTION_TABLE.C is a renamed column
    query = f'SELECT "{table_names[1]}".A FROM {table_names[0]} EVOLVED JOIN "{table_names[1]}" ON EVOLVED.C = "{table_names[1]}".A'

    catalog = bodosql.FileSystemCatalog(warehouse_loc, default_schema=f'"{db_schema}"')
    bc = bodosql.BodoSQLContext(catalog=catalog)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    py_output = pd.DataFrame({"A": [1, 2, 3, 4, 5] * 1100})
    with set_logging_stream(logger, 2):
        check_func(
            impl,
            (bc, query),
            py_output=py_output,
            only_1DVar=True,
            sort_output=True,
            reset_index=True,
        )

        check_logger_msg(
            stream,
            "Runtime join filter expression: ((ds.field('{C}') >= 1) & (ds.field('{C}') <= 5))",
        )


def test_dict_keys(memory_leak_check, iceberg_database):
    """
    Variant of test_simple_join with dictionary encoded columns as keys.
    """
    table_names = [
        "ENGLISH_DICTIONARY_TABLE",
        "SHAKESPEARE_TABLE",
    ]

    def impl(bc, query):
        return bc.sql(query)

    db_schema, warehouse_loc = iceberg_database(table_names)

    # Join the shakespeare and english_dictionary tables to get the definitions
    # of all words spoken by mark antony. This should produce a runtime join
    # filter on english_dictionary where word is between "A" and "Autolycus".

    query = """
    WITH T1 AS (
        SELECT Initcap(SPLIT_PART(player, ' ', 0)) as PLAYER 
        FROM SHAKESPEARE_TABLE WHERE STARTSWITH(player, 'A') OR EDITDISTANCE(player, '') < 2
    )
    SELECT player, definition
    FROM T1, ENGLISH_DICTIONARY_TABLE T2
    WHERE player = word
    QUALIFY ROW_NUMBER() OVER (PARTITION BY player ORDER BY definition) = 1
    """

    catalog = bodosql.FileSystemCatalog(warehouse_loc, default_schema=f'"{db_schema}"')
    bc = bodosql.BodoSQLContext(catalog=catalog)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    py_output = pd.DataFrame(
        {
            "PLAYER": ["All", "Archbishop"],
            "DEFINITION": [
                '"Although; albeit."',
                '"A chief bishop; a church dignitary of the first class (often called a metropolitan or primate) who superintends the conduct of the suffragan bishops in his province  and also exercises episcopal authority in his own diocese."',
            ],
        }
    )

    with set_logging_stream(logger, 2):
        check_func(
            impl,
            (bc, query),
            py_output=py_output,
            only_1DVar=True,
            sort_output=True,
            reset_index=True,
        )
        check_logger_msg(
            stream,
            "Runtime join filter expression: ((ds.field('{WORD}') >= 'A') & (ds.field('{WORD}') <= 'Autolycus'))",
        )
        check_logger_msg(stream, "Total number of files is 16. Reading 1 files:")
