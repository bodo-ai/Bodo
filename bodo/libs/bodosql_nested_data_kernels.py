# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Implements BodoSQL array kernels related to ARRAY utilities
"""

import bodo
import numba
import numpy as np
import pandas as pd
import types

from bodo.libs.bodosql_array_kernel_utils import *
from bodo.utils.typing import unwrap_typeref


@numba.generated_jit(nopython=True)
def to_array(arr, dtype):
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.to_array", ["arr"], 0
        )

    def impl(arr, dtype):  # pragma: no cover
        return to_array_util(arr, dtype)

    return impl


@numba.generated_jit(nopython=True)
def to_array_util(arr, dtype):
    arg_names = ["arr", "dtype"]
    arg_types = [arr, dtype]
    propagate_null = [True, False]
    arr_dtype = unwrap_typeref(dtype)
    out_dtype = bodo.libs.array_item_arr_ext.ArrayItemArrayType(arr_dtype)
    scalar_text = "res[i] = bodo.utils.conversion.coerce_scalar_to_array(arg0, 1, arg1, False)"
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
    )
