"""
Base class implementations for supporting streaming operators.
"""

from __future__ import annotations

import numba
import numpy as np
from numba.core import types

from bodo.hiframes.table import TableType
from bodo.utils.utils import numba_to_c_array_types, numba_to_c_types


class StreamingStateType(types.Type):
    """
    Base class for any streaming state type. This should not be
    used directly. This will become more comprehensive to represent
    a true abstract class over time, but for now its just used to hold
    duplicate code.
    """

    def __init__(self, name: str):
        super().__init__(name=name)

    @staticmethod
    def _derive_c_types(arr_types: list[types.ArrayCompatible]) -> np.ndarray:
        """Generate the CType Enum types for each array in the
        C++ build table via the indices.

        Args:
            arr_types (List[types.ArrayCompatible]): The array types to use.

        Returns:
            List(int): List with the integer values of each CTypeEnum value.
        """
        return numba_to_c_types(arr_types)

    @staticmethod
    def _derive_c_array_types(arr_types: list[types.ArrayCompatible]) -> np.ndarray:
        """Generate the CArrayTypeEnum Enum types for each array in the
        C++ build table via the indices.

        Args:
            arr_types (List[types.ArrayCompatible]): The array types to use.

        Returns:
            List(int): List with the integer values of each CTypeEnum value.
        """
        return numba_to_c_array_types(arr_types)

    @staticmethod
    def ensure_known_inputs(
        fn_name: str, table_types: tuple[TableType | type[types.unknown], ...]
    ):
        """
        Ensure no input in table_types is
        types.unknown. This will raise a NumbaError to skip
        unnecessary compilation if that invariant is violated.
        Args:
            fn_name (str): The name of the function for clarifying error messages.
            table_types (TableType | types.unknown): The table types to check.
        Raises:
            numba.NumbaError: A error if any input is still unknown.
        """
        for table_type in table_types:
            if table_type == types.unknown:
                raise numba.NumbaError(
                    f"{fn_name}: unknown input table type in streaming state"
                )


def is_streaming_build_funcs(func):
    """
    Check if a function is a streaming build function (join/groupby/IO etc.).
    Imports are in this function to make them lazy (since not necessary in DataFrame
    library).
    """

    from bodo.io.iceberg.stream_iceberg_write import iceberg_writer_append_table
    from bodo.io.snowflake_write import snowflake_writer_append_table
    from bodo.io.stream_parquet_write import parquet_writer_append_table
    from bodo.libs.streaming.groupby import (
        groupby_build_consume_batch,
        groupby_grouping_sets_build_consume_batch,
    )
    from bodo.libs.streaming.join import (
        join_build_consume_batch,
        join_probe_consume_batch,
    )
    from bodo.libs.streaming.sort import sort_build_consume_batch
    from bodo.libs.streaming.union import union_consume_batch
    from bodo.libs.streaming.window import window_build_consume_batch
    from bodo.libs.table_builder import table_builder_append

    streaming_build_funcs = (
        groupby_build_consume_batch,
        groupby_grouping_sets_build_consume_batch,
        join_build_consume_batch,
        join_probe_consume_batch,
        window_build_consume_batch,
        union_consume_batch,
        table_builder_append,
        sort_build_consume_batch,
        snowflake_writer_append_table,
        iceberg_writer_append_table,
        parquet_writer_append_table,
    )

    return func in streaming_build_funcs
