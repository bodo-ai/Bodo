# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
    Library of BodoSQL functions used to emulate MYSQL's ntile function
"""
import numpy as np

import bodo


# pythonization of sparks ntile algorithm found at:
# https://github.com/apache/spark/blob/375ca9467870000298a529b357c24ed6c8681892/sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/windowExpressions.scala#L782
# As literal a translation from the scala version as I can manage
# Slightly more readable/faster version
@bodo.jit(distributed=False)
def ntile_helper(df_len, num_bins):
    """
    this should only be used from groupby apply, hence distributed = False
    """
    remainder = df_len % num_bins
    num_items_in_larger_bin = np.int64(np.ceil(df_len / num_bins))
    num_items_in_smaller_bin = np.int64(np.floor(df_len / num_bins))
    out = np.empty(df_len, dtype=np.int64)
    end_idx = 0
    for bin_idx in range(1, min(num_bins, df_len) + 1):
        if bin_idx <= remainder:
            new_end_idx = end_idx + num_items_in_larger_bin
        else:
            new_end_idx = end_idx + num_items_in_smaller_bin
        out[end_idx:new_end_idx] = bin_idx
        end_idx = new_end_idx
    return out
