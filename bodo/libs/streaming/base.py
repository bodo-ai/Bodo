"""
Base class implementations for supporting streaming operators.
"""

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
