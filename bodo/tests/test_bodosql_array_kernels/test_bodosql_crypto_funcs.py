# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""
Implements tests for the BodoSQL kernels for SHA2 and other crypto functions.
"""
# -*- coding: utf-8 -*-

import hashlib
import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.libs.bodosql_array_kernel_utils import vectorized_sol
from bodo.tests.utils import check_func


@pytest.mark.parametrize(
    "msg, digest_size",
    [
        pytest.param(
            "nqlkgr23rwe",
            224,
            id="224_scalar",
        ),
        pytest.param(
            pd.Series(
                [
                    "132e8w7nudoifqe",
                    "'-=:>@#!!@$@#%Q!",
                    None,
                    "3oiurjqwdkKLNDW",
                ] * 4,
            ),
            256,
            id="256_vector",
        ),
        pytest.param(
            pd.Series(
                [
                    " pipcm$%YTYJF:",
                    None,
                    "pn98oiy7htngiAWE@W#E",
                    "OIJSD8964aflkaw",
                ] * 4,
            ),
            384,
            id="384_vector",
        ),
        pytest.param(
            "?><!@$@#%^$%^",
            512,
            id="512_scalar",
        ),
        pytest.param(
            b";'saflraed",
            224,
            id="224_binary_scalar",
        ),
        pytest.param(
            pd.Series(
                [
                    b"opimeaeda",
                    None,
                    b"'-=:>@lkadjlsx#%Q!",
                    b"+_)()_o8j:OJ",
                ] * 4,
            ),
            256,
            id="256_binary_vector",
        ),
    ]
)
def test_sha2_kernel(msg, digest_size, memory_leak_check):
    """Test sha2 bodo kernel"""
    is_out_distributed = True

    def impl(msg, digest_size):
        return pd.Series(
            bodo.libs.bodosql_array_kernels.sha2(msg, digest_size)
        )

    if not isinstance(msg, pd.Series):
        is_out_distributed = False
        # Setting is_output_distributed = False is necessary for scalar test with np > 1
        impl = lambda msg, digest_size: bodo.libs.bodosql_array_kernels.sha2(msg, digest_size)

    def scalar_fn(elem, size):
        if pd.isna(elem):
            return None
        if size == 224:
            func = hashlib.sha224
        elif size == 256:
            func = hashlib.sha256
        elif size == 384:
            func = hashlib.sha384
        else:
            func = hashlib.sha512
        if isinstance(elem, str):
            return func(elem.encode('utf-8')).hexdigest()
        else:  # bytes
            return func(elem).hexdigest()

    answer = vectorized_sol((msg, digest_size,), scalar_fn, None)

    check_func(
        impl,
        (msg, digest_size,),
        py_output=answer,
        is_out_distributed=is_out_distributed,
    )
