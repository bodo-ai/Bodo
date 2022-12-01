# Copyright (C) 2022 Bodo Inc. All rights reserved.


import bodosql
import pandas as pd
import pytest
from numba.core import types

import bodo
from bodo.tests.iceberg_database_helpers.utils import (
    create_iceberg_table,
    get_spark,
)
from bodo.tests.utils import check_func, find_funcname_in_annotation_ir


def verify_dict_encoding_in_columns(func_name, args, impl, expected_columns):
    """
    Verify that the given function has dict encoding in the given columns.
    """
    sig = types.void(*args)
    cr = impl.get_compile_result(sig)
    annotation = cr.type_annotation
    _, arg_names = find_funcname_in_annotation_ir(annotation, func_name)

    df_argname = arg_names[3]

    df_type = annotation.typemap[df_argname]

    dict_encoded_column_names = expected_columns.copy()

    for column_name, column_type in zip(df_type.columns, df_type.data):
        if column_name in dict_encoded_column_names:
            assert column_type == bodo.dict_str_arr_type
            dict_encoded_column_names.remove(column_name)

    assert len(dict_encoded_column_names) == 0


@pytest.mark.skip(reason="Waiting for merge into support")
def test_merge_into_dict_encoding_merge(
    iceberg_database, iceberg_table_conn, memory_leak_check
):
    """
    Test that merge into dict encoding works when delta and dest table
    have the same dict when a merge is performed.
    """

    target_table = pd.DataFrame(
        {
            "a": ["ä", "b", "c", "d", "ë", "f", "g", "h", "ï", "j"] * 100,
            "b": [str(i) for i in range(1000)],
        }
    )

    source = pd.DataFrame(
        {
            "a": ["h", "ï", "j", "k", "l", "m", "n", "ö", "p", "q"] * 100,
            "b": [str(i + 500) for i in range(1000)],
        }
    )
    expected = pd.DataFrame(
        {
            "a": ["ä", "b", "c", "d", "ë", "f", "g", "h", "ï", "j"] * 100
            + ["k", "l", "m", "n", "ö", "p", "q"] * 100,
            "b": [str(i) for i in range(1000)]
            + [str(i + 500) for i in range(300, 1000)],
        }
    )

    # open connection and create initial table
    spark = get_spark()
    db_schema, warehouse_loc = iceberg_database
    table_name = "merge_into_dict_encoding"
    sql_schema = [("a", "string", False), ("b", "string", False)]
    if bodo.get_rank() == 0:
        create_iceberg_table(
            target_table,
            sql_schema,
            table_name,
            spark,
        )
    bodo.barrier()
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)
    bc = bodosql.BodoSQLContext(
        {
            "target_table": bodosql.TablePath(
                table_name, "sql", conn_str=conn, db_schema=db_schema
            ),
            "source": source,
        }
    )
    bc.add_or_replace_view(table_name, target_table)

    def impl(bc, table_name, conn, db_schema):
        bc.sql(
            "MERGE INTO target_table AS t USING source AS s "
            "ON t.id = s.id "
            "WHEN NOT MATCHED THEN "
            "  INSERT (id, data) VALUES (s.id, s.data)"
        )

        return pd.read_sql_table(table_name, conn, db_schema)

    check_func(
        impl, (bc, table_name, conn, db_schema), only_1DVar=True, py_output=expected
    )

    args = (bodo.typeof(target_table),)

    func_name = ("impl", "bodosql.tests.test_merge_into_dict_encoding")

    dict_encoded_column_names = {"a"}

    verify_dict_encoding_in_columns(func_name, args, impl, dict_encoded_column_names)


@pytest.mark.skip(reason="Waiting for merge into support")
def test_merge_into_dict_encoding_no_merge(
    iceberg_database, iceberg_table_conn, memory_leak_check
):
    """
    Test that merge into dict encoding works when the contents of the dest
    table is the same when no merge is performed.
    """

    target_table = pd.DataFrame(
        {
            "a": ["ä", "b", "c", "d", "ë", "f", "g", "h", "ï", "j"] * 100,
            "b": [str(i) for i in range(1000)],
        }
    )

    source = target_table.copy()
    expected = target_table.copy()

    # open connection and create initial table
    spark = get_spark()
    db_schema, warehouse_loc = iceberg_database
    table_name = "merge_into_dict_encoding"
    sql_schema = [("a", "string", False), ("b", "string", False)]
    if bodo.get_rank() == 0:
        create_iceberg_table(
            target_table,
            sql_schema,
            table_name,
            spark,
        )
    bodo.barrier()
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)
    bc = bodosql.BodoSQLContext(
        {
            "target_table": bodosql.TablePath(
                table_name, "sql", conn_str=conn, db_schema=db_schema
            ),
            "source": source,
        }
    )
    bc.add_or_replace_view(table_name, target_table)

    def impl(bc, table_name, conn, db_schema):
        bc.sql(
            "MERGE INTO target_table AS t USING source AS s "
            "ON t.id = s.id "
            "WHEN NOT MATCHED THEN "
            "  INSERT (id, data) VALUES (s.id, s.data)"
        )

        return pd.read_sql_table(table_name, conn, db_schema)

    check_func(
        impl, (bc, table_name, conn, db_schema), only_1DVar=True, py_output=expected
    )

    args = (bodo.typeof(target_table),)

    func_name = ("impl", "bodosql.tests.test_merge_into_dict_encoding")

    dict_encoded_column_names = {"a"}

    verify_dict_encoding_in_columns(func_name, args, impl, dict_encoded_column_names)
