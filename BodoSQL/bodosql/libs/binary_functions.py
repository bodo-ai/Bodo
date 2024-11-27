"""
Library of BodoSQL functions used for binary data.
"""

from numba import generated_jit
from numba.core import types

import bodo
from bodo.utils.typing import BodoError


@generated_jit(nopython=True)
def cast_binary(val):
    unliteral_val = types.unliteral(val)
    if unliteral_val == bodo.bytes_type:
        # Binary data doesn't require any changes
        return lambda val: val  # pragma: no cover
    elif unliteral_val == bodo.string_type:
        # String data requires an encode.
        return lambda val: val.encode("utf-8")  # pragma: no cover
    else:
        raise BodoError(f"Unsupported cast from {unliteral_val} to bytes")
