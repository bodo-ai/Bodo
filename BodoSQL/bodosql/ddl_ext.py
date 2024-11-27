"""
BodoSQL runtime interface for DDL operations.
Eventually all DDL operations are actually executed by the planner
in context.py, but this module provides the JIT interface for
DDL operations.
"""

from numba.extending import overload

import bodo
from bodo.hiframes.table import Table
from bodo.transforms.distributed_analysis import Distribution
from bodo.utils.typing import MetaType, unwrap_typeref
from bodosql.context import BodoSQLContext


def execute_ddl(
    bodo_sql_context: BodoSQLContext,
    query: str,
    column_types: MetaType,
) -> Table:
    pass


@overload(execute_ddl)
def overload_execute_ddl(bodo_sql_context, query, column_types):
    """
    Generic implementation of execute ddl. This handles the typing
    information for the output of the function and then calls into objmode.

    Args:
        bodo_sql_context (BodoSQLContext): The BodoSQLContext to call into the planner.
        query (str): The DDL query to execute.
        columnTypes (MetaType): The column types for the output DataFrame.

    Returns:
        bodo.TableType: A Table based upon column_names and column_types.
    """
    column_types = unwrap_typeref(column_types).meta
    # Note: A DDL operation always returns a replicated DataFrame.
    output_type = bodo.TableType(
        column_types,
        dist=Distribution.REP,
    )

    def impl(bodo_sql_context, query, column_types):
        with bodo.no_warning_objmode(output=output_type):
            output = execute_ddl_objmode(bodo_sql_context, query)
        return output

    return impl


def execute_ddl_objmode(bodo_sql_context: BodoSQLContext, query: str) -> Table:
    """
    Execute a DDL query in object mode.

    Args:
        bodo_sql_context (BodoSQLContext): The BodoSQLContext to call into the planner.
        query (str): The DDL query to execute.

    Returns:
        pd.DataFrame: The result of the DDL query.
    """
    # Execute ddl returns a DataFrame. We need to convert it to a Table
    # to be able to return it from the function.
    df = bodo_sql_context.execute_ddl(query)
    arrs = []
    for i in range(len(df.columns)):
        arr = df.iloc[:, i].array
        arrs.append(arr)
    return Table(arrs, dist=Distribution.REP)
