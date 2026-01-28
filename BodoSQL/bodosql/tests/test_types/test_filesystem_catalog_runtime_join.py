import datetime
import io

import numpy as np
import pandas as pd
import pytest
from mpi4py import MPI

import bodosql
from bodo.spawn.utils import run_rank0
from bodo.tests.iceberg_database_helpers.utils import (
    create_iceberg_table,
    get_spark,
)
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import check_func, temp_env_override

comm = MPI.COMM_WORLD

pytestmark = pytest.mark.iceberg


@pytest.mark.parametrize(
    "allow_low_ndv_filter",
    [
        pytest.param(True, id="low_ndv"),
        pytest.param(False, id="min_max"),
    ],
)
def test_simple_join(iceberg_database, allow_low_ndv_filter, memory_leak_check):
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
    if allow_low_ndv_filter:
        limit = 20
        log_msg = "Runtime join filter expression: ((ds.field('{A}').isin([1])))"
    else:
        limit = 0
        log_msg = "Runtime join filter expression: ((ds.field('{A}') >= 1) & (ds.field('{A}') <= 1))"
    with temp_env_override({"BODO_JOIN_UNIQUE_VALUES_LIMIT": str(limit)}):
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
                log_msg,
            )
            check_logger_msg(stream, "Total number of files is 5. Reading 1 files:")


@pytest.mark.parametrize(
    "allow_low_ndv_filter",
    [
        pytest.param(True, id="low_ndv"),
        pytest.param(False, id="min_max"),
    ],
)
@pytest.mark.parametrize("join_same_col", [True, False])
def test_multiple_filter_join(
    iceberg_database, allow_low_ndv_filter, join_same_col, memory_leak_check
):
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
    if allow_low_ndv_filter:
        limit = 20
        log_msg = (
            "Runtime join filter expression: ((ds.field('{A}').isin([1, 2, 3, 4, 5])) & (ds.field('{A}').isin([2])))"
            if join_same_col
            else "Runtime join filter expression: ((ds.field('{A}').isin([2])) & (ds.field('{G}').isin([1, 2, 3, 4, 5])))"
        )
    else:
        limit = 0
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
    with temp_env_override({"BODO_JOIN_UNIQUE_VALUES_LIMIT": str(limit)}):
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
                log_msg,
            )


@temp_env_override({"BODO_JOIN_UNIQUE_VALUES_LIMIT": "0"})
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


@temp_env_override({"BODO_JOIN_UNIQUE_VALUES_LIMIT": "0"})
def test_string_keys(memory_leak_check, iceberg_database):
    """
    Variant of test_simple_join with string columns as keys.
    """
    table_names = [
        "ENGLISH_DICTIONARY_TABLE",
        "SHAKESPEARE_TABLE",
    ]

    def impl(bc, query):
        return bc.sql(query)

    db_schema, warehouse_loc = iceberg_database(table_names)

    # Join the shakespeare and english_dictionary tables to get
    # the definition of every word with at least 6 characters.
    # This should produce a runtime join filter on english_dictionary
    # where word is between "Aaron" and "Zounds".

    query = """
    WITH shake_words AS (
        SELECT lat.value as word
        FROM SHAKESPEARE_TABLE,
        LATERAL FLATTEN(STRTOK_TO_ARRAY(PLAYERLINE, ' ')) lat
        WHERE LENGTH(lat.value) >= 6
    )
    SELECT sw.word, edt.definition
    FROM shake_words sw, ENGLISH_DICTIONARY_TABLE edt
    WHERE sw.word = edt.word
    QUALIFY ROW_NUMBER() OVER (PARTITION BY sw.word ORDER BY edt.definition) = 1
    """

    catalog = bodosql.FileSystemCatalog(warehouse_loc, default_schema=f'"{db_schema}"')
    bc = bodosql.BodoSQLContext(catalog=catalog)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    py_output = pd.DataFrame(
        {
            "WORD": ["Return", "Charge", "Strike", "Spring", "Draught", "Square"],
            "DEFINITION": [
                '"A day in bank. See Return day  below."',
                '"A bearing. See Bearing  n. 8."',
                '"A bushel; four pecks."',
                '"A crack or fissure in a mast or yard  running obliquely or transversely."',
                '"A current of air moving through an inclosed place  as through a room or up a chimney."',
                '"A body of troops formed in a square  esp. one formed to resist a charge of cavalry; a squadron."',
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
        ##### Disabled RTJF on non-dictionary-encoded string columns #####
        # check_logger_msg(
        #     stream,
        #     "Runtime join filter expression: ((ds.field('{WORD}') >= 'Aarons') & (ds.field('{WORD}') <= 'Zounds'))",
        # )


@pytest.mark.parametrize(
    "allow_low_ndv_filter",
    [
        pytest.param(True, id="low_ndv", marks=pytest.mark.skip("[BSE-3538]")),
        pytest.param(False, id="min_max"),
    ],
)
def test_dict_keys(allow_low_ndv_filter, iceberg_database, memory_leak_check):
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

    if allow_low_ndv_filter:
        limit = 20
        log_msg = ""  # TODO: add the runtime join expression once we have dict support for low-ndv filter
    else:
        limit = 0
        log_msg = "Runtime join filter expression: ((ds.field('{WORD}') >= 'A') & (ds.field('{WORD}') <= 'Autolycus'))"

    with temp_env_override({"BODO_JOIN_UNIQUE_VALUES_LIMIT": str(limit)}):
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
                log_msg,
            )
            check_logger_msg(stream, "Total number of files is 16. Reading 1 files:")


@pytest.fixture
def rtjf_test_tables():
    return {
        "FILTERKEYS1": (
            {"KEYS1": np.array([1, 2, 2, 4, 5] * 1, dtype=np.int32)},
            (("KEYS1", "int", True),),
        ),
        "FILTERKEYS2": (
            {"KEYS2": np.array([1, 3, 3, 4, 5] * 1, dtype=np.int64)},
            (("KEYS2", "long", True),),
        ),
        "FILTERKEYS3": (
            {"KEYS3": np.array([1, 3, 3, 3, 5] * 1, dtype=np.int32)},
            (("KEYS3", "int", True),),
        ),
        "INT_TABLE1": (
            {
                "A1": np.array([1, 2, 3, 4, 5] * 2, dtype=np.int32),
                "B1": np.array([9, 6, 7, 8, 5] * 2, dtype=np.int32),
                "C1": np.array([5, 10, 11, 12, 13] * 2, dtype=np.int32),
            },
            [
                ("A1", "int", True),
                ("B1", "int", True),
                ("C1", "int", True),
            ],
        ),
        "INT_TABLE2": (
            {
                "A2": np.array([1, 3, 5, 7, 9] * 2, dtype=np.int32),
                "B2": np.array([5, 7, 13, 9, 15] * 2, dtype=np.int32),
            },
            [
                ("A2", "int", True),
                ("B2", "int", True),
            ],
        ),
        "INTERESTING_TABLE1": (
            {
                "STRCOL1": pd.Series(
                    ["pastas", "pizza", "tacos", "adidas", "nike"] * 2, dtype="string"
                ),
                "INTCOL1": pd.Series([21, 22, 23, 24, 25] * 2, dtype="int32"),
                "LONGINTCOL1": pd.Series(
                    [9000000000000000000, 222, 23, -1, 5] * 2, dtype="int64"
                ),
            },
            [
                ("STRCOL1", "string", False),
                ("INTCOL1", "int", True),
                ("LONGINTCOL1", "long", True),
            ],
        ),
        "INTERESTING_TABLE2": (
            {
                "STRCOL2": pd.Series(
                    ["pastas", "pizza", "tacos", "adida", "puma"] * 2, dtype="string"
                ),
                "INTCOL2": pd.Series([21, 22, 25, 26, 25] * 2, dtype="int32"),
                "LONGINTCOL2": pd.Series(
                    [9000000000000000001, 222, 23, -1, 5] * 2, dtype="int64"
                ),
            },
            [
                ("STRCOL2", "string", False),
                ("INTCOL2", "int", True),
                ("LONGINTCOL2", "long", True),
            ],
        ),
    }


@temp_env_override({"BODO_JOIN_UNIQUE_VALUES_LIMIT": "0"})
@pytest.mark.parametrize(
    "query, expected_out",
    [
        pytest.param(
            """
            SELECT STRCOL2, COUNT(*) as NMATCH FROM
                INTERESTING_TABLE2
                JOIN
                (SELECT * FROM INTERESTING_TABLE1)
            ON STRCOL2 = STRCOL1
            GROUP BY 1
            """,
            pd.DataFrame({"STRCOL2": ["pastas", "pizza", "tacos"], "NMATCH": [4] * 3}),
            id="string_single_key",
        ),
        pytest.param(
            """
            SELECT A1 FROM
                (SELECT * FROM
                    (SELECT * FROM
                        (SELECT * FROM
                                FILTERKEYS1
                                JOIN
                                (SELECT * FROM INT_TABLE1)
                            ON A1 = KEYS1
                    ) JOIN FILTERKEYS2
                    ON A1 = KEYS2
                ) JOIN FILTERKEYS3
            ON A1 = KEYS3
            )
            """,
            pd.DataFrame({"A1": [1, 5] * 2}),
            id="multiple_join_same_key",
        ),
        pytest.param(
            """
            SELECT *
            FROM
            (SELECT *
                FROM
                (
                    SELECT *
                        FROM
                            (SELECT * FROM
                                INT_TABLE2
                                JOIN
                                (SELECT * FROM INT_TABLE1)
                            ON A2 = A1
                            )
                            JOIN INT_TABLE2 as t0
                            ON t0.B2 = C1
                )
                JOIN INT_TABLE2 as t1
                ON t1.A2 = B1
            )""",
            pd.DataFrame(
                {
                    "A2": np.array([1, 5] * 16, dtype=np.int32),
                    "B2": np.array([5, 13] * 16, dtype=np.int32),
                    "A1": np.array([1, 5] * 16, dtype=np.int32),
                    "B1": np.array([9, 5] * 16, dtype=np.int32),
                    "C1": np.array([5, 13] * 16, dtype=np.int32),
                    "A20": np.array([1, 5] * 16, dtype=np.int32),
                    "B20": np.array([5, 13] * 16, dtype=np.int32),
                    "A21": np.array([9, 5] * 16, dtype=np.int32),
                    "B21": np.array([15, 13] * 16, dtype=np.int32),
                }
            ),
            id="multiple_join_multiple_keys",
        ),
    ],
)
def test_merged_rtjf(
    memory_leak_check, iceberg_database, query, expected_out, rtjf_test_tables
):
    def impl(bc, query):
        return bc.sql(query)

    table_names = list(rtjf_test_tables.keys())
    db_schema, warehouse_loc = iceberg_database(table_names)

    @run_rank0
    def setup():
        spark = get_spark()
        for table_name, (rows, sql_schema) in rtjf_test_tables.items():
            create_iceberg_table(pd.DataFrame(rows), sql_schema, table_name, spark)

    setup()

    catalog = bodosql.FileSystemCatalog(warehouse_loc, default_schema=f'"{db_schema}"')
    bc = bodosql.BodoSQLContext(catalog=catalog)

    stream = io.StringIO()
    logger = create_string_io_logger(stream)

    with set_logging_stream(logger, 2):
        check_func(
            impl,
            (bc, query),
            py_output=expected_out,
            only_1DVar=True,
            sort_output=True,
            reset_index=True,
            use_dict_encoded_strings=True,
        )


@pytest.mark.parametrize(
    "allow_low_ndv_filter",
    [
        pytest.param(True, id="low_ndv"),
        pytest.param(False, id="min_max"),
    ],
)
def test_date_keys(allow_low_ndv_filter, iceberg_database, memory_leak_check):
    """
    Variant of test_simple_join with date columns as keys.
    """
    table_names = [
        "MOCK_NEWS_TABLE",
        "MOCK_HOLIDAY_TABLE",
    ]

    def impl(bc, query):
        return bc.sql(query)

    db_schema, warehouse_loc = iceberg_database(table_names)

    # Join the news & holiday tables on the date. This should produce a runtime
    # join filter on the news table where the date is between 2021-02-04 and 2021-12-31

    query = """
    SELECT news.day as DAY, news.event as EVENT, holiday.name as NAME
    FROM MOCK_NEWS_TABLE news, MOCK_HOLIDAY_TABLE holiday
    WHERE news.DAY = holiday.DAY
    """

    catalog = bodosql.FileSystemCatalog(warehouse_loc, default_schema=f'"{db_schema}"')
    bc = bodosql.BodoSQLContext(catalog=catalog)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    py_output = pd.DataFrame(
        {
            "DAY": [
                datetime.date(2021, 2, 14),
                datetime.date(2021, 7, 4),
                datetime.date(2021, 12, 25),
                datetime.date(2021, 10, 31),
                datetime.date(2021, 6, 19),
                datetime.date(2021, 9, 6),
                datetime.date(2021, 5, 31),
                datetime.date(2021, 11, 25),
            ],
            "EVENT": [
                "P10",
                "V50",
                "\\24",
                "Z69",
                "U35",
                "X14",
                "T16",
                "[94",
            ],
            "NAME": [
                "VALENTINE'S DAY",
                "INDEPENDENCE DAY",
                "CHRISTMAS",
                "HALLOWEEN",
                "JUNETEENTH",
                "LABOR DAY",
                "MEMORIAL DAY",
                "THANKSGIVING",
            ],
        }
    )

    if allow_low_ndv_filter:
        limit = 20
        log_msg = "Runtime join filter expression: ((ds.field('{DAY}').isin([pa.scalar(18672, pa.date32()), pa.scalar(18778, pa.date32()), pa.scalar(18797, pa.date32()), pa.scalar(18812, pa.date32()), pa.scalar(18876, pa.date32()), pa.scalar(18931, pa.date32()), pa.scalar(18956, pa.date32()), pa.scalar(18986, pa.date32())])))"
    else:
        limit = 0
        log_msg = "Runtime join filter expression: ((ds.field('{DAY}') >= pa.scalar(18672, pa.date32())) & (ds.field('{DAY}') <= pa.scalar(18986, pa.date32())))"

    with temp_env_override({"BODO_JOIN_UNIQUE_VALUES_LIMIT": str(limit)}):
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
                log_msg,
            )
            check_logger_msg(stream, "Total number of files is 3. Reading 1 files:")


@pytest.mark.parametrize(
    "allow_low_ndv_filter",
    [
        pytest.param(True, id="low_ndv"),
        pytest.param(False, id="min_max"),
    ],
)
def test_float_keys(allow_low_ndv_filter, iceberg_database, memory_leak_check):
    """
    Variant of test_simple_join with float columns as keys.
    """
    table_names = [
        "BANK_ACCOUNTS_TABLE",
        "SUSPICIOUS_SUMS_TABLE",
    ]

    def impl(bc, query):
        return bc.sql(query)

    db_schema, warehouse_loc = iceberg_database(table_names)

    # Join the bank accounts & suspicious sums tables on the date. This should produce a
    # runtime join filter on the accounts table where the balance is between 0.0 and 488000.09

    query = """
    SELECT acct.ACCTNMBR as ACCTNMBR, susp.FLAG as FLAG
    FROM BANK_ACCOUNTS_TABLE acct, SUSPICIOUS_SUMS_TABLE susp
    WHERE acct.BALANCE = susp.BALANCE
    """

    catalog = bodosql.FileSystemCatalog(warehouse_loc, default_schema=f'"{db_schema}"')
    bc = bodosql.BodoSQLContext(catalog=catalog)
    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    py_output = pd.DataFrame(
        {
            "ACCTNMBR": [
                10000,
                19100,
                28200,
                37300,
                46400,
                55500,
                20325,
                31525,
                53925,
                21750,
                24550,
                26950,
                29750,
                35750,
                38550,
                40950,
                43750,
                58150,
                29920,
                39920,
                47420,
                57420,
            ],
            "FLAG": [
                "EMPTY",
                "EMPTY",
                "EMPTY",
                "EMPTY",
                "EMPTY",
                "EMPTY",
                "SHELL",
                "SHELL",
                "SHELL",
                "HACK",
                "HACK",
                "HACK",
                "HACK",
                "HACK",
                "HACK",
                "HACK",
                "HACK",
                "HACK",
                "CAYMAN",
                "CAYMAN",
                "CAYMAN",
                "CAYMAN",
            ],
        }
    )

    if allow_low_ndv_filter:
        limit = 20
        log_msg = "Runtime join filter expression: ((ds.field('{BALANCE}').isin([0.0, 375000.81, 488000.09, 78125.42])))"
    else:
        limit = 0
        log_msg = "Runtime join filter expression: ((ds.field('{BALANCE}') >= 0.0) & (ds.field('{BALANCE}') <= 488000.09))"

    with temp_env_override({"BODO_JOIN_UNIQUE_VALUES_LIMIT": str(limit)}):
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
                log_msg,
            )


def test_interval_join_rtjf(memory_leak_check, iceberg_database):
    """
    Adds a test for Runtime Join Filter support for joins created with
    interval syntax.
    """
    table_names = ["BANK_ACCOUNTS_TABLE"]

    def impl(bc, query):
        return bc.sql(query)

    db_schema, warehouse_loc = iceberg_database(table_names)
    catalog = bodosql.FileSystemCatalog(warehouse_loc, default_schema=f'"{db_schema}"')

    df = pd.DataFrame(
        {
            "KEY1": [10001, 10002, 10003, 10005, 10007],
            "KEY2": [10004, 10004, 10006, 10008, 10010],
        }
    )

    bc = bodosql.BodoSQLContext({"SMALL_TABLE": df}, catalog=catalog)

    query = """SELECT B.ACCTNMBR FROM BANK_ACCOUNTS_TABLE B inner join SMALL_TABLE S on B.ACCTNMBR >= S.KEY1 AND B.ACCTNMBR < S.KEY2"""
    answer = pd.DataFrame(
        {
            "ACCTNMBR": [10001]
            + [10002] * 2
            + [10003] * 3
            + [10004]
            + [10005] * 2
            + [10006]
            + [10007] * 2
            + [10008, 10009],
        }
    )

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        check_func(
            impl,
            (bc, query),
            py_output=answer,
            only_1DVar=True,
            sort_output=True,
            reset_index=True,
        )
        # Verify that the correct bounds were added to the data requested
        # from Iceberg.
        check_logger_msg(
            stream,
            "Runtime join filter expression: ((ds.field('{ACCTNMBR}') < 10010) & (ds.field('{ACCTNMBR}') >= 10001))",
        )


def test_interval_join_rtjf_multi_column(memory_leak_check, iceberg_database):
    """
    Adds a test for Runtime Join Filter support for joins created with
    interval syntax and additional columns on the build side to ensure a remapping.
    """
    table_names = ["BANK_ACCOUNTS_TABLE"]

    def impl(bc, query):
        return bc.sql(query)

    db_schema, warehouse_loc = iceberg_database(table_names)
    catalog = bodosql.FileSystemCatalog(warehouse_loc, default_schema=f'"{db_schema}"')

    df = pd.DataFrame(
        {
            "NAME1": ["A"] * 5,
            "KEY1": [10001, 10002, 10003, 10005, 10007],
            "NAME2": ["B"] * 5,
            "KEY2": [10004, 10004, 10006, 10008, 10010],
        }
    )

    bc = bodosql.BodoSQLContext({"SMALL_TABLE": df}, catalog=catalog)

    query = """SELECT B.ACCTNMBR, S.NAME1, S.NAME2 FROM BANK_ACCOUNTS_TABLE B inner join SMALL_TABLE S on B.ACCTNMBR >= S.KEY1 AND B.ACCTNMBR < S.KEY2"""
    answer = pd.DataFrame(
        {
            "ACCTNMBR": [10001]
            + [10002] * 2
            + [10003] * 3
            + [10004]
            + [10005] * 2
            + [10006]
            + [10007] * 2
            + [10008, 10009],
            "NAME1": ["A"] * 14,
            "NAME2": ["B"] * 14,
        }
    )

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        check_func(
            impl,
            (bc, query),
            py_output=answer,
            only_1DVar=True,
            sort_output=True,
            reset_index=True,
        )
        # Verify that the correct bounds were added to the data requested
        # from Iceberg.
        check_logger_msg(
            stream,
            "Runtime join filter expression: ((ds.field('{ACCTNMBR}') < 10010) & (ds.field('{ACCTNMBR}') >= 10001))",
        )


def test_in_prune_files_extreme(memory_leak_check, iceberg_database):
    """Test that IN can prune all files based on NDV join filters. This
    tests against a mixed optimization where IN won't run any pruning on files
    after there are > 200 set entries.
    """
    table_names = ["BANK_ACCOUNTS_TABLE"]

    def impl(bc, query):
        return bc.sql(query)

    db_schema, warehouse_loc = iceberg_database(table_names)
    catalog = bodosql.FileSystemCatalog(warehouse_loc, default_schema=f'"{db_schema}"')

    df = pd.DataFrame(
        {
            "KEY": pd.array([-i for i in range(1, 201)], dtype="Int64"),
        }
    )

    bc = bodosql.BodoSQLContext({"SMALL_TABLE": df}, catalog=catalog)

    query = """SELECT B.ACCTNMBR FROM BANK_ACCOUNTS_TABLE B inner join SMALL_TABLE S on B.ACCTNMBR = S.KEY"""
    answer = pd.DataFrame(
        {
            "ACCTNMBR": pd.array([], dtype="Int64"),
        }
    )

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with temp_env_override({"BODO_JOIN_UNIQUE_VALUES_LIMIT": "300"}):
        with set_logging_stream(logger, 2):
            check_func(
                impl,
                (bc, query),
                py_output=answer,
                only_1DVar=True,
                sort_output=True,
                reset_index=True,
            )
            # Verify that no files were loaded.
            check_logger_msg(stream, "Reading 0 files:")


def test_empty_build_rtjf(memory_leak_check, iceberg_database):
    """
    Adds a test for Runtime Join Filter support that prunes the whole
    probe due to an empty build table..
    """
    table_names = ["BANK_ACCOUNTS_TABLE"]

    def impl(bc, query):
        return bc.sql(query)

    db_schema, warehouse_loc = iceberg_database(table_names)
    catalog = bodosql.FileSystemCatalog(warehouse_loc, default_schema=f'"{db_schema}"')

    df = pd.DataFrame({"KEY": pd.array([], dtype="Int64")})

    bc = bodosql.BodoSQLContext({"SMALL_TABLE": df}, catalog=catalog)

    query = """SELECT B.ACCTNMBR FROM BANK_ACCOUNTS_TABLE B inner join SMALL_TABLE S on B.ACCTNMBR = S.KEY"""
    answer = pd.DataFrame({"ACCTNMBR": pd.array([], dtype="Int64")})

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        check_func(
            impl,
            (bc, query),
            py_output=answer,
            only_1DVar=True,
            sort_output=True,
            reset_index=True,
        )
        # Verify that Iceberg doesn't load any files.
        check_logger_msg(stream, "Reading 0 files:")
