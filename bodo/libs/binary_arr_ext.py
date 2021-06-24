# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Array implementation for binary (bytes) objects, which are usually immutable.
It is equivalent to string array, except that it stores a 'bytes' object for each
element instead of 'str'.
"""

from numba.core import types

bytes_type = types.Bytes(types.uint8, 1, "C", readonly=True)


# type for ndarray with bytes object values
class BinaryArrayType(types.ArrayCompatible):
    def __init__(self):
        super(BinaryArrayType, self).__init__(name="BinaryArrayType()")

    @property
    def as_array(self):
        return types.Array(types.undefined, 1, "C")

    @property
    def dtype(self):
        return bytes_type

    def copy(self):
        return BinaryArrayType()


binary_array_type = BinaryArrayType()
