# Copyright (C) 2024 Bodo Inc. All rights reserved.
"""
Base class implementations for supporting streaming operators.
"""
import typing as pt

import numpy as np
from numba.core import types

from bodo.utils.utils import numba_to_c_array_types, numba_to_c_types


class StreamingStateType(types.Type):
    """
    Base class for any streaming state type. This should not be
    used directly. This will become more comprehensive to represent
    a true abstract class over time, but for now its just used to hold
    duplicate code.
    """

    def __init__(self, name: str):
        super(StreamingStateType, self).__init__(name=name)

    @staticmethod
    def _derive_c_types(arr_types: pt.List[types.ArrayCompatible]) -> np.ndarray:
        """Generate the CType Enum types for each array in the
        C++ build table via the indices.

        Args:
            arr_types (List[types.ArrayCompatible]): The array types to use.

        Returns:
            List(int): List with the integer values of each CTypeEnum value.
        """
        return numba_to_c_types(arr_types)

    @staticmethod
    def _derive_c_array_types(arr_types: pt.List[types.ArrayCompatible]) -> np.ndarray:
        """Generate the CArrayTypeEnum Enum types for each array in the
        C++ build table via the indices.

        Args:
            arr_types (List[types.ArrayCompatible]): The array types to use.

        Returns:
            List(int): List with the integer values of each CTypeEnum value.
        """
        return numba_to_c_array_types(arr_types)
