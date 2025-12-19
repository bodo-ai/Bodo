import pandas as pd
import pyarrow as pa
import pytest
from numba.core import types

import bodo
import bodosql
from bodo.tests.conftest import iceberg_database, iceberg_table_conn  # noqa
from bodo.tests.iceberg_database_helpers.utils import create_iceberg_table
from bodo.tests.utils import check_func, find_funcname_in_annotation_ir

pytestmark = [
    pytest.mark.iceberg,
    pytest.mark.skip(
        reason="[BSE-4569] MERGE INTO with PyIceberg is not supported yet"
    ),
]


def verify_dict_encoding_in_columns(
    func_name, args, impl, expected_columns: list[str] | None
):
    """
    Verify that the given function has dict encoding in the given columns.
    """
    sig = types.void(*args)
    cr = impl.get_compile_result(sig)
    annotation = cr.type_annotation
    _, arg_names = find_funcname_in_annotation_ir(annotation, func_name)

    df_argname = arg_names[3]

    df_type = annotation.typemap[df_argname]

    dict_encoded_column_names = (
        [] if expected_columns is None else expected_columns
    ).copy()

    for column_name, column_type in zip(df_type.columns, df_type.data):
        if column_name in dict_encoded_column_names:
            assert column_type == bodo.types.dict_str_arr_type, (
                f"Column {column_name} is not dictionary encoded"
            )
            dict_encoded_column_names.remove(column_name)

    assert len(dict_encoded_column_names) == 0


@pytest.fixture(params=[False, True])
def source_table(request):
    A = ["h", "ï", "j", "k", "l", "m", "n", "ö", "p", "q"] * 100
    B = [str(i + 500) for i in range(1000)]

    if request.param:
        A = pd.arrays.ArrowStringArray(  # type: ignore
            pa.array(
                A,
                type=pa.dictionary(pa.int32(), pa.string()),
            ).cast(pa.dictionary(pa.int32(), pa.large_string()))
        )

        B = pd.arrays.ArrowStringArray(  # type: ignore
            pa.array(
                B,
                type=pa.dictionary(pa.int32(), pa.string()),
            ).cast(pa.dictionary(pa.int32(), pa.large_string()))
        )

    return pd.DataFrame({"A": A, "B": B})


@pytest.fixture(params=[None, ["A", "B"]])
def bodosql_dict_context(iceberg_database, iceberg_table_conn, source_table, request):
    target_table = pd.DataFrame(
        {
            "A": ["ä", "b", "c", "d", "ë", "f", "g", "h", "ï", "j"] * 100,
            "B": [str(i) for i in range(1000)],
        }
    )

    db_schema, warehouse_loc = iceberg_database()
    calling_func_name: str = request.node.name
    table_name = (
        calling_func_name.replace("[", "_").replace("-", "_").replace("]", "").lower()
    )
    sql_schema = [("A", "string", False), ("B", "string", False)]
    if bodo.get_rank() == 0:
        create_iceberg_table(
            target_table,
            sql_schema,
            table_name,
        )

    bodo.barrier()
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)
    bc = bodosql.BodoSQLContext(
        {
            "TARGET_TABLE": bodosql.TablePath(
                table_name,
                "sql",
                conn_str=conn,
                db_schema=db_schema,
                bodo_read_as_dict=request.param,
            ),
            "SOURCE": source_table,
        }
    )
    bc.add_or_replace_view(table_name, target_table)
    return bc, request.param


@pytest.mark.timeout(700)
@pytest.mark.slow
def test_merge_into_dict_encoding_inserts(
    bodosql_dict_context,
):
    """
    Test that MERGE INTO dict encoding works when delta and dest table
    have the same dict when a merge is performed, and dictionary
    encoding is maintained with INSERT operations.
    """
    # TODO: Add memory_leak_check
    bc, dict_col_names = bodosql_dict_context

    expected = pd.DataFrame(
        {
            "A": ["ä", "b", "c", "d", "ë", "f", "g", "h", "ï", "j"] * 100
            + ["h", "ï", "j", "k", "l", "m", "n", "ö", "p", "q"] * 50,
            "B": [str(i) for i in range(1000)]
            + [str(i + 500) for i in range(500, 1000)],
        }
    )

    def impl(bc):
        bc.sql(
            "MERGE INTO target_table AS t USING source AS s "
            "ON t.b = s.b "
            "WHEN NOT MATCHED THEN "
            "  INSERT (a, b) VALUES (s.a, s.b)"
        )

        return bc.sql("SELECT * FROM target_table")

    check_func(
        impl,
        (bc,),
        only_1DVar=True,
        py_output=expected,
        sort_output=True,
        reset_index=True,
    )

    verify_dict_encoding_in_columns(
        ("iceberg_merge_cow_py", "bodo.io.iceberg.merge_into"),
        (bodo.typeof(bc),),
        bodo.jit(parallel=True)(impl),
        dict_col_names,
    )


@pytest.mark.slow
def test_merge_into_dict_encoding_updates(
    bodosql_dict_context,
):
    """
    Test that MERGE INTO dict encoding works when delta and dest table
    have the same dict when a merge is performed, and dictionary
    encoding is maintained with DELETE operations.
    """
    # TODO: Add memory_leak_check
    bc, dict_col_names = bodosql_dict_context

    expected = pd.DataFrame(
        {
            "A": ["ä", "b", "c", "d", "ë", "f", "g", "h", "ï", "j"] * 50
            + ["h", "ï", "j", "k", "l", "m", "n", "ö", "p", "q"] * 50,
            "B": [str(i) for i in range(1000)],
        }
    )

    def impl(bc):
        bc.sql(
            "MERGE INTO target_table AS t USING source AS s "
            "ON t.b = s.b "
            "WHEN MATCHED THEN "
            "  UPDATE SET a = s.a"
        )

        return bc.sql("SELECT * FROM target_table")

    check_func(
        impl,
        (bc,),
        only_1DVar=True,
        py_output=expected,
        sort_output=True,
        reset_index=True,
    )

    verify_dict_encoding_in_columns(
        ("iceberg_merge_cow_py", "bodo.io.iceberg.merge_into"),
        (bodo.typeof(bc),),
        bodo.jit(parallel=True)(impl),
        dict_col_names,
    )


@pytest.mark.slow
def test_merge_into_dict_encoding_deletes(bodosql_dict_context):
    """
    Test that MERGE INTO dict encoding works when delta and dest table
    have the same dict when a merge is performed, and dictionary
    encoding is maintained with DELETE operations.
    """
    # TODO: Add memory_leak_check
    bc, dict_col_names = bodosql_dict_context

    expected = pd.DataFrame(
        {
            "A": ["ä", "b", "c", "d", "ë", "f", "g", "h", "ï", "j"] * 50,
            "B": [str(i) for i in range(500)],
        }
    )

    def impl(bc):
        bc.sql(
            "MERGE INTO target_table AS t USING source AS s "
            "ON t.b = s.b "
            "WHEN MATCHED THEN "
            "  DELETE"
        )

        return bc.sql("SELECT * FROM target_table")

    check_func(
        impl,
        (bc,),
        only_1DVar=True,
        py_output=expected,
        sort_output=True,
        reset_index=True,
    )

    verify_dict_encoding_in_columns(
        ("iceberg_merge_cow_py", "bodo.io.iceberg.merge_into"),
        (bodo.typeof(bc),),
        bodo.jit(parallel=True)(impl),
        dict_col_names,
    )


@pytest.mark.timeout(700)
@pytest.mark.slow
def test_merge_into_dict_encoding_all(bodosql_dict_context):
    """
    Test that MERGE INTO dict encoding works when delta and dest table
    have the same dict when a merge is performed, and dictionary
    encoding is maintained with all operations combined.
    """
    # TODO: Add memory_leak_check
    bc, dict_col_names = bodosql_dict_context

    expected = pd.DataFrame(
        {
            "A": ["ä", "b", "c", "d", "ë", "f", "g", "h", "ï", "j"] * 50
            + ["h", "ï", "j", "k", "l", "m", "n", "ö", "p", "q"] * 75,
            "B": [str(i) for i in range(750)] + [str(i + 1000) for i in range(500)],
        }
    )

    def impl(bc):
        bc.sql(
            "MERGE INTO target_table AS t USING source AS s "
            "ON t.b = s.b "
            "WHEN MATCHED AND t.b < 750 THEN "
            "  UPDATE SET a = s.a "
            "WHEN MATCHED THEN "
            "  DELETE "
            "WHEN NOT MATCHED THEN "
            "  INSERT (a, b) VALUES (s.a, s.b)"
        )

        return bc.sql("SELECT * FROM target_table")

    check_func(
        impl,
        (bc,),
        only_1DVar=True,
        py_output=expected,
        sort_output=True,
        reset_index=True,
    )

    verify_dict_encoding_in_columns(
        ("iceberg_merge_cow_py", "bodo.io.iceberg.merge_into"),
        (bodo.typeof(bc),),
        bodo.jit(parallel=True)(impl),
        dict_col_names,
    )


@pytest.mark.timeout(600)
@pytest.mark.slow
def test_merge_into_dict_encoding_no_merge(iceberg_database, iceberg_table_conn):
    """
    Test that merge into dict encoding works when the contents of the dest
    table is the same when no merge is performed.
    """
    # TODO: Add memory_leak_check

    target_table = pd.DataFrame(
        {
            "A": ["ä", "b", "c", "d", "ë", "f", "g", "h", "ï", "j"] * 100,
            "B": [str(i) for i in range(1000)],
        }
    )

    source = target_table.copy()
    expected = target_table.copy()

    # open connection and create initial table
    db_schema, warehouse_loc = iceberg_database()
    table_name = "MERGE_INTO_DICT_ENCODING2"
    sql_schema = [("A", "string", False), ("B", "string", False)]
    if bodo.get_rank() == 0:
        create_iceberg_table(
            target_table,
            sql_schema,
            table_name,
        )
    bodo.barrier()
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc)
    bc = bodosql.BodoSQLContext(
        {
            "TARGET_TABLE": bodosql.TablePath(
                table_name, "sql", conn_str=conn, db_schema=db_schema
            ),
            "SOURCE": source,
        }
    )
    bc.add_or_replace_view(table_name, target_table)

    def impl(bc):
        bc.sql(
            "MERGE INTO target_table AS t USING source AS s "
            "ON t.b = s.b "
            "WHEN NOT MATCHED THEN "
            "  INSERT (a, b) VALUES (s.a, s.b)"
        )

        return bc.sql("Select * from target_table")

    check_func(
        impl,
        (bc,),
        only_1DVar=True,
        py_output=expected,
        sort_output=True,
        reset_index=True,
    )

    args = (bodo.typeof(bc),)
    func_name = ("iceberg_merge_cow_py", "bodo.io.iceberg.merge_into")
    dict_encoded_column_names = ["A"]

    # TODO: Fix dict encoding?
    verify_dict_encoding_in_columns(
        func_name, args, bodo.jit(parallel=True)(impl), dict_encoded_column_names
    )
